"""Prefetch Video-MME videos + QA metadata for pipeline testing.

Video-MME (`lmms-lab/Video-MME`) tiene il parquet di QA in
`videomme/test-00000-of-00001.parquet` (2700 righe, 4-MCQ con `duration`
short/medium/long) e i mp4 spalmati su 20 zip `videos_chunked_01.zip` …
`videos_chunked_20.zip` (~101 GB tot). Dentro ciascun zip i video stanno
in `data/<videoID>.mp4` — convenzione confermata da `lmms-eval`
(`videomme_doc_to_visual`).

Lo script scarica il parquet, decide quali `videoID` servono, poi itera
sui zip in ordine estraendo *solo* i mp4 target. Si ferma quando tutti i
target sono soddisfatti, evitando di tirare giù 101 GB per smoke test.

Output (self-contained, stesso schema degli altri prefetch):

    out_dir/
        metadata.jsonl     # una riga JSON per sample; `video` è path
                           #   relativo a out_dir.
        videos/
            data/
                <videoID>.mp4
                ...
        subtitle/          # solo se `subtitles=true`
            <videoID>.srt
            ...

Run on the login node (compute nodes typically lack internet).

Due modalità di selezione dei chunk:

- `chunks=null` (default): scarica i chunk in ordine 1..20 finché tutti i
  `videoID` dei `rows[:n]` sono soddisfatti. OK per full run; pessimo per
  smoke test perché la mappa videoID→chunk è opaca (i primi N row del
  parquet possono richiedere chunk arbitrari).
- `chunks=[i, j, ...]`: scarica solo quei chunk, **filtra** il parquet ai
  videoID realmente presenti dentro, poi tronca a `n`. Smoke test
  deterministico: con `chunks=[1] n=10` scarichi sempre ~5 GB e produci
  un `metadata.jsonl` con 10 sample.

Overrides (stessa sintassi `key=value` di `main.py`, vedi `utils/config.py`):
    uv run python fetch/prefetch_videomme.py n=20
    uv run python fetch/prefetch_videomme.py chunks=[1] n=10
    uv run python fetch/prefetch_videomme.py subtitles=true
    uv run python fetch/prefetch_videomme.py keep_zips=true
    uv run python fetch/prefetch_videomme.py --print-config

    # ~20 video diversi per lunghezza (fino a 7 short + 7 medium + 6 long,
    # via override CLI di `videos_per_duration`), ristretti ai chunk 7 e
    # 14 — quelli a cavallo dei confini short|medium e medium|long nel
    # parquet (video #300/#301 e #600/#601), quindi i più probabili a
    # contenere tutte e 3 le fasce in ~10GB invece di scandire i 20 chunk
    # (~101GB) alla ricerca di video sparsi ovunque. Ignora `duration`/`n`.
    uv run python fetch/prefetch_videomme.py chunks=[7,14] videos_per_duration=7
"""
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

# When invoked as `python fetch/prefetch_videomme.py`, only `fetch/` is on
# sys.path — add its parent (repo root) so the absolute import
# `from utils.obs import ...` resolves regardless of invocation mode
# (script, `python -m fetch.prefetch_videomme`, or installed entry point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weave
import yaml

from utils.config import Cfg, load_flat_config
from utils.obs import init_observability

CONF_PATH = Path(__file__).resolve().parent.parent / "conf" / "fetch" / "prefetch_videomme.yaml"

logger = logging.getLogger(__name__)


NUM_VIDEO_CHUNKS = 20
# Estensioni accettate dentro ai zip. In pratica i chunk Video-MME
# contengono solo .mp4 lowercase, ma teniamo .MP4 / .mkv come fallback
# difensivo (allineato a `lmms-eval`). NB: `_extract_targets_from_zip`
# usa setdefault sull'ordine di `infolist()` del zip, non sull'ordine di
# questa tupla — se mai un zip contenesse più estensioni per lo stesso
# stem, vincerebbe quella incontrata per prima nel namelist.
VIDEO_EXTS = (".mp4", ".MP4", ".mkv")


def _chunk_name(idx: int) -> str:
    """`videos_chunked_07.zip` per idx=7. I chunk vanno 01..20."""
    return f"videos_chunked_{idx:02d}.zip"


def _list_videoids_in_zip(zip_path: Path) -> set[str]:
    """Ritorna lo stem dei file video dentro a `zip_path` senza estrarli.

    Usato in modalità `chunks=[...]` per pre-filtrare i rows del parquet
    ai videoID realmente presenti nei chunk richiesti, prima di applicare
    `n` truncation.
    """
    ids: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.suffix in VIDEO_EXTS:
                ids.add(p.stem)
    return ids


def _extract_targets_from_zip(
    zip_path: Path,
    target_ids: set[str],
    videos_dir: Path,
) -> dict[str, str]:
    """Estrae da `zip_path` solo i video con stem ∈ target_ids.

    Mantiene il path interno del zip (es. `data/<id>.mp4`), così
    `videos/data/<id>.mp4` matcha la convenzione lmms-eval. Ritorna un
    dict `{videoID: path_relativo_a_videos_dir}` con la **vera**
    estensione estratta (`.mp4` / `.MP4` / `.mkv`) — il chiamante usa
    questo path per scrivere `video` in metadata.jsonl, così evitiamo
    di hardcodare `.mp4` quando il file ha case/extension diversi.
    """
    extracted: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        # Indicizza per stem così evitiamo O(target × namelist) e
        # gestiamo .MP4/.mkv senza assumere case.
        members_by_stem: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.suffix in VIDEO_EXTS:
                members_by_stem.setdefault(p.stem, info)

        for vid in target_ids:
            info = members_by_stem.get(vid)
            if info is None:
                continue
            zf.extract(info, videos_dir)
            extracted[vid] = info.filename
    return extracted


def _select_videos_per_duration(rows: list[dict], k: int) -> list[dict]:
    """Tiene tutte le righe dei primi `k` videoID distinti per ciascuna
    fascia `duration` (short/medium/long), nell'ordine in cui compaiono
    in `rows`. A differenza di un plain `rows[:n]`, garantisce diversità
    tra le 3 fasce invece di dipendere da come sono ordinate nel parquet
    (VideoMME le tiene in 3 blocchi contigui: prima tutte le short, poi
    le medium, poi le long — un `n` piatto pesca solo dalla prima fascia
    che incontra)."""
    quota = {"short": k, "medium": k, "long": k}
    accepted: dict[str, set[str]] = {"short": set(), "medium": set(), "long": set()}
    out = []
    for r in rows:
        d = r.get("duration")
        if d not in quota:
            continue
        vid = r["videoID"]
        if vid in accepted[d] or len(accepted[d]) < quota[d]:
            accepted[d].add(vid)
            out.append(r)
    short_of = {d: quota[d] - len(accepted[d]) for d in quota if len(accepted[d]) < quota[d]}
    if short_of:
        logger.warning("videos_per_duration=%d: fasce sotto quota (pool esaurito): %s", k, short_of)
    return out


def _fetch_video_chunks(
    hf_repo: str,
    target_ids: set[str],
    out: Path,
    keep_zips: bool,
) -> dict[str, str]:
    """Scarica chunk in ordine ed estrae i target. Si ferma appena tutti
    i target sono soddisfatti. Ritorna un dict `{videoID: path_relativo
    a `<out>/videos/`}` (incluso prefisso `data/` e estensione reale).
    """
    from huggingface_hub import hf_hub_download

    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = out / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # Quali sono già su disco? Permette re-run idempotenti senza
    # ri-scaricare nulla. `rglob` recurse così funziona anche se un
    # vecchio run aveva layout diverso (es. file flat senza `data/`).
    found: dict[str, str] = {}
    for ext in VIDEO_EXTS:
        for p in videos_dir.rglob(f"*{ext}"):
            if p.stem in target_ids:
                found.setdefault(p.stem, str(p.relative_to(videos_dir)))
    remaining = target_ids - set(found.keys())
    if not remaining:
        logger.info("tutti i %d videoID target sono già estratti, skip download", len(target_ids))
        return found
    logger.info(
        "già su disco: %d/%d; da fetchare via chunk: %d",
        len(found), len(target_ids), len(remaining),
    )

    for i in range(1, NUM_VIDEO_CHUNKS + 1):
        if not remaining:
            break
        zip_name = _chunk_name(i)
        logger.info(
            "[chunk %d/%d] scarico %s (target rimanenti: %d)",
            i, NUM_VIDEO_CHUNKS, zip_name, len(remaining),
        )
        # `local_dir=zips_dir` mette il file sotto `<out>/_zips/<zip_name>`
        # così possiamo cancellarlo a fine estrazione senza toccare la
        # cache HF condivisa.
        local_zip = Path(hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=zip_name,
            local_dir=str(zips_dir),
        ))

        extracted = _extract_targets_from_zip(local_zip, remaining, videos_dir)
        found.update(extracted)
        remaining -= set(extracted.keys())
        logger.info(
            "[chunk %d/%d] estratti %d video, rimanenti %d",
            i, NUM_VIDEO_CHUNKS, len(extracted), len(remaining),
        )

        if not keep_zips:
            local_zip.unlink(missing_ok=True)

    if not keep_zips:
        # Rimuove la dir solo se vuota; se l'utente ha keep_zips a metà
        # run, lasciamo intatto.
        try:
            zips_dir.rmdir()
        except OSError:
            pass

    return found


def _fetch_subtitles(
    hf_repo: str,
    target_ids: set[str],
    out: Path,
    keep_zips: bool,
) -> set[str]:
    """Scarica `subtitle.zip` ed estrae solo i .srt dei target.

    Ritorna i videoID con sottotitolo disponibile (sottoinsieme).
    Cancella `subtitle.zip` dopo l'estrazione se `keep_zips=False`,
    analogo a `_fetch_video_chunks`.
    """
    from huggingface_hub import hf_hub_download

    sub_dir = out / "subtitle"
    sub_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = out / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    logger.info("scarico subtitle.zip da %s", hf_repo)
    local_zip = Path(hf_hub_download(
        repo_id=hf_repo,
        repo_type="dataset",
        filename="subtitle.zip",
        local_dir=str(zips_dir),
    ))

    found: set[str] = set()
    with zipfile.ZipFile(local_zip) as zf:
        members_by_stem: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.suffix.lower() == ".srt":
                members_by_stem.setdefault(p.stem, info)
        for vid in target_ids:
            info = members_by_stem.get(vid)
            if info is None:
                continue
            # Layout piatto: `<out>/subtitle/<videoID>.srt`. Se lo zip ha
            # path nested, prendiamo i bytes e li riscriviamo flat.
            data = zf.read(info)
            (sub_dir / f"{vid}.srt").write_bytes(data)
            found.add(vid)

    if not keep_zips:
        local_zip.unlink(missing_ok=True)
        try:
            zips_dir.rmdir()
        except OSError:
            pass

    return found


@weave.op()
def prefetch(
    hf_repo: str,
    config: str,
    n: int | None,
    out_dir: str,
    duration: str | None,
    subtitles: bool,
    keep_zips: bool,
    chunks: list[int] | None,
    videos_per_duration: int | None = None,
) -> dict:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out: Path = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("carico metadata: %s [%s] split=test", hf_repo, config)
    # Parquet piccolo (~400 KB, 2700 righe) → non serve streaming.
    ds = load_dataset(hf_repo, config, split="test")
    rows: list[dict] = [dict(r) for r in ds]
    logger.info("metadata: %d righe, columns=%s", len(rows), list(rows[0].keys()))

    if videos_per_duration is not None and duration and duration != "all":
        logger.warning(
            "videos_per_duration=%d è impostato: ignoro duration=%r "
            "(la selezione bilanciata pesca comunque da tutte e 3 le fasce)",
            videos_per_duration, duration,
        )
    elif duration and duration != "all":
        rows = [r for r in rows if r.get("duration") == duration]
        logger.info("filtro duration=%r → %d righe", duration, len(rows))

    # `chunks` mode: pre-scarica i chunk indicati e filtra rows ai
    # videoID disponibili in essi. Garantisce smoke test deterministico
    # (esattamente N chunk scaricati, sempre produce metadata.jsonl).
    prescanned_chunks: list[Path] = []
    if chunks is not None:
        zips_dir = out / "_zips"
        zips_dir.mkdir(parents=True, exist_ok=True)
        available_ids: set[str] = set()
        for i in chunks:
            zip_name = _chunk_name(i)
            logger.info("[chunk %d] scarico %s per scansione contenuto", i, zip_name)
            local_zip = Path(hf_hub_download(
                repo_id=hf_repo,
                repo_type="dataset",
                filename=zip_name,
                local_dir=str(zips_dir),
            ))
            prescanned_chunks.append(local_zip)
            ids_here = _list_videoids_in_zip(local_zip)
            available_ids |= ids_here
            logger.info("[chunk %d] %d videoID dentro", i, len(ids_here))

        before = len(rows)
        rows = [r for r in rows if r["videoID"] in available_ids]
        logger.info(
            "filtro chunks=%s: %d → %d righe (videoID matching)",
            chunks, before, len(rows),
        )

    if videos_per_duration is not None:
        rows = _select_videos_per_duration(rows, videos_per_duration)
        logger.info(
            "selezione bilanciata: fino a %d video per fascia (short/medium/long) → %d righe",
            videos_per_duration, len(rows),
        )
    elif n is not None:
        rows = rows[:n]
        logger.info("trunca a n=%d righe", n)

    if not rows:
        raise RuntimeError("Nessuna riga selezionata dopo filtri.")

    # Una riga = una domanda; più domande condividono lo stesso video.
    target_ids: set[str] = {r["videoID"] for r in rows}
    logger.info("videoID unici da fetchare: %d (su %d domande)", len(target_ids), len(rows))

    if chunks is not None:
        # Estrai dai chunk già scaricati nel pre-scan.
        videos_dir = out / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        found_videos: dict[str, str] = {}
        for local_zip in prescanned_chunks:
            extracted = _extract_targets_from_zip(local_zip, target_ids, videos_dir)
            found_videos.update(extracted)
            logger.info("[%s] estratti %d video", local_zip.name, len(extracted))
            if not keep_zips:
                local_zip.unlink(missing_ok=True)
        if not keep_zips:
            try:
                (out / "_zips").rmdir()
            except OSError:
                pass
    else:
        found_videos = _fetch_video_chunks(hf_repo, target_ids, out, keep_zips)
    missing_videos = target_ids - set(found_videos)
    if missing_videos:
        logger.warning(
            "%d videoID non trovati in nessun chunk: %s",
            len(missing_videos), sorted(missing_videos)[:5],
        )

    found_subs: set[str] = set()
    if subtitles:
        found_subs = _fetch_subtitles(hf_repo, target_ids, out, keep_zips)
        logger.info("sottotitoli trovati: %d/%d", len(found_subs), len(target_ids))

    metadata: list[dict] = []
    dropped = 0
    for r in rows:
        vid = r["videoID"]
        rel_video = found_videos.get(vid)
        if rel_video is None:
            dropped += 1
            continue
        # `rel_video` ha già il prefisso interno del zip (es. `data/`)
        # e la vera estensione (`.mp4` / `.MP4` / `.mkv`).
        r["video"] = str(Path("videos") / rel_video)
        if subtitles:
            r["subtitle"] = (
                str(Path("subtitle") / f"{vid}.srt") if vid in found_subs else None
            )
        metadata.append(r)
    if dropped:
        logger.warning("scartate %d domande prive di video", dropped)

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for sample in metadata:
            f.write(json.dumps(sample) + "\n")

    logger.info("Wrote %d sample + metadata.jsonl in %s", len(metadata), out)
    return {
        "subset_size": len(metadata),
        "num_videos": len(found_videos),
        "out_dir": str(out),
        "metadata_path": str(metadata_path),
    }


def run(cfg: Cfg) -> None:
    # `datasets` / `huggingface_hub` leggono HF_HOME al primo import, va
    # settato prima dell'import lazy dentro `prefetch`.
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    chunks = None if cfg.chunks is None else [int(i) for i in cfg.chunks]
    prefetch(
        cfg.hf_repo,
        cfg.config,
        cfg.n,
        cfg.out_dir,
        cfg.duration,
        bool(cfg.subtitles),
        bool(cfg.keep_zips),
        chunks,
        cfg.videos_per_duration,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)

    print_config = "--print-config" in argv
    if print_config:
        argv.remove("--print-config")

    cfg = load_flat_config(CONF_PATH, argv)
    if print_config:
        print(yaml.safe_dump(dict(cfg), sort_keys=False))
        return
    run(cfg)


if __name__ == "__main__":
    main()
