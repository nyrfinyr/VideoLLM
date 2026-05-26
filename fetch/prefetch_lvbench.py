"""Prefetch LVBench videos + QA metadata for pipeline testing.

A differenza della versione precedente (che scaricava i video da YouTube
via `yt-dlp` partendo dai metadata di `zai-org/LVBench`), questo script
tira giù tutto da `lmms-lab/LVBench`, che ospita un mirror self-contained:

    data/train-00000-of-00001.parquet   — 1549 domande (4-MCQ), schema
                                           flat (una riga per QA):
                                           {video_path, uid, question,
                                            question_type, answer,
                                            time_reference, type, key}
    video_chunks/videos_chunk_001.zip … videos_chunk_014.zip
                                         — ~61 GB tot; ogni zip contiene
                                           gli mp4 flat, nominati esatta-
                                           mente come il campo `video_path`
                                           (es. `Cm73ma6Ibcs.mp4`).

Vantaggio rispetto a YouTube: niente rate-limit / blocco anti-bot, niente
mortalità irreversibile (video rimossi), download riproducibile. I 103
video unici sono grandi (fino a ~2h) → lo script scarica i chunk solo
quanto basta a coprire i video selezionati (vedi `n` / `chunks`).

Le opzioni MCQ sono già embedded nel testo di `question` (formato
"<stem>\n(A) ...\n(B) ...\n(C) ...\n(D) ..."); il loader `evals.lvbench`
le riparserà.

Output (self-contained, stesso schema della versione YouTube, così
`evals.lvbench.LVBench.loader` resta invariato):

    out_dir/
        metadata.jsonl     # una riga per QA:
                           #   {id, key, type, video, uid, question,
                           #    answer, question_type, time_reference}
                           #   `id` = "<key>/<uid>", `video` path
                           #   relativo a out_dir. Le righe relative a
                           #   video non estratti (assenti nei chunk
                           #   scaricati) sono scartate.
        dropped.jsonl      # una riga JSON per ogni video selezionato ma
                           #   non estratto, con
                           #   `{video_path, key, type, num_qa}`.
                           #   Sovrascritto a ogni run. Con `lmms-lab`
                           #   dovrebbe restare vuoto in full run; non
                           #   vuoto solo se si restringe con `chunks=[...]`
                           #   o per integrità incompleta del repo.
        videos/
            <video_path>           # mp4 flat, nome == campo `video_path`.

Note operative:
- Run on the login node (compute nodes typically lack internet).
- Idempotente: rilanciandolo salta i video già estratti su disco e
  scarica solo i chunk ancora necessari.
- I chunk sono grandi (~3-5 GB ciascuno): per default vengono cancellati
  dopo l'estrazione (`keep_zips=false`).

Due modalità di selezione dei chunk:

- `chunks=null` (default): scarica i chunk in ordine 001..014 finché tutti
  i video selezionati sono estratti. OK per full run; pessimo per smoke
  test perché la mappa video→chunk è opaca (i primi N video del parquet
  possono richiedere chunk arbitrari).
- `chunks=[i, j, ...]`: scarica solo quei chunk, **filtra** il parquet ai
  `video_path` realmente presenti dentro, poi tronca a `n`. Smoke test
  deterministico: con `chunks=[1] n=3` scarichi sempre un solo chunk e
  produci un `metadata.jsonl` con le QA dei primi 3 video presenti.

Hydra overrides:
    uv run python fetch/prefetch_lvbench.py n=3
    uv run python fetch/prefetch_lvbench.py chunks=[1] n=3
    uv run python fetch/prefetch_lvbench.py keep_zips=true
    uv run python fetch/prefetch_lvbench.py --cfg job
"""
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

# When invoked as `python fetch/prefetch_lvbench.py`, only `fetch/` is on
# sys.path — add its parent (repo root) so the absolute import
# `from utils.obs import ...` resolves regardless of invocation mode
# (script, `python -m fetch.prefetch_lvbench`, or installed entry point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import weave
from omegaconf import DictConfig

from utils.obs import init_observability

logger = logging.getLogger(__name__)


NUM_VIDEO_CHUNKS = 14
# Estensioni accettate dentro ai zip. In pratica i chunk LVBench
# contengono solo .mp4 lowercase (il campo `video_path` è sempre
# `<id>.mp4`), ma teniamo .MP4 / .mkv / .webm come fallback difensivo.
VIDEO_EXTS = (".mp4", ".MP4", ".mkv", ".webm")


def _chunk_name(idx: int) -> str:
    """`video_chunks/videos_chunk_007.zip` per idx=7. I chunk vanno 001..014."""
    return f"video_chunks/videos_chunk_{idx:03d}.zip"


def _list_videos_in_zip(zip_path: Path) -> set[str]:
    """Ritorna i nomi-file (basename) dei video dentro a `zip_path`.

    Usato in modalità `chunks=[...]` per pre-filtrare i rows del parquet
    ai `video_path` realmente presenti nei chunk richiesti, prima di
    applicare `n` truncation.
    """
    names: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.suffix in VIDEO_EXTS:
                names.add(p.name)
    return names


def _extract_targets_from_zip(
    zip_path: Path,
    target_names: set[str],
    videos_dir: Path,
) -> dict[str, str]:
    """Estrae da `zip_path` solo i video il cui basename ∈ target_names.

    I file vengono scritti **flat** sotto `videos_dir/<video_path>`,
    coerentemente col campo `video_path` del parquet (che non ha
    sottocartelle). Ritorna `{video_path: video_path}` per i video
    estratti — il valore è il path relativo a `videos_dir`, identico al
    nome perché il layout è piatto.
    """
    extracted: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        # Indicizza per basename: gestiamo zip flat o con prefisso-dir
        # senza assumere il layout interno.
        members_by_name: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.suffix in VIDEO_EXTS:
                members_by_name.setdefault(p.name, info)

        for name in target_names:
            info = members_by_name.get(name)
            if info is None:
                continue
            # Riscrive flat: legge i bytes e li deposita in
            # `videos_dir/<name>`, ignorando eventuale prefisso-dir interno.
            with zf.open(info) as src:
                (videos_dir / name).write_bytes(src.read())
            extracted[name] = name
    return extracted


def _fetch_video_chunks(
    hf_repo: str,
    target_names: set[str],
    out: Path,
    keep_zips: bool,
) -> dict[str, str]:
    """Scarica chunk in ordine ed estrae i target. Si ferma appena tutti
    i target sono estratti. Ritorna `{video_path: path_relativo a
    `<out>/videos/`}`.
    """
    from huggingface_hub import hf_hub_download

    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = out / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # Quali sono già su disco? Permette re-run idempotenti senza
    # ri-scaricare nulla. Layout flat: `videos/<video_path>`.
    found: dict[str, str] = {}
    for name in target_names:
        if (videos_dir / name).exists():
            found[name] = name
    remaining = target_names - set(found.keys())
    if not remaining:
        logger.info("tutti i %d video target sono già estratti, skip download", len(target_names))
        return found
    logger.info(
        "già su disco: %d/%d; da fetchare via chunk: %d",
        len(found), len(target_names), len(remaining),
    )

    for i in range(1, NUM_VIDEO_CHUNKS + 1):
        if not remaining:
            break
        zip_name = _chunk_name(i)
        logger.info(
            "[chunk %d/%d] scarico %s (target rimanenti: %d)",
            i, NUM_VIDEO_CHUNKS, zip_name, len(remaining),
        )
        # `local_dir=zips_dir` mette il file sotto
        # `<out>/_zips/video_chunks/<zip_name>` così possiamo cancellarlo a
        # fine estrazione senza toccare la cache HF condivisa.
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
        # Rimuove le dir solo se vuote; se l'utente ha keep_zips a metà
        # run, lasciamo intatto.
        for d in (zips_dir / "video_chunks", zips_dir):
            try:
                d.rmdir()
            except OSError:
                pass

    return found


@weave.op()
def prefetch(
    hf_repo: str,
    config: str,
    split: str,
    n: int | None,
    out_dir: str,
    keep_zips: bool,
    chunks: list[int] | None,
) -> dict:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out: Path = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("carico metadata: %s [%s] split=%s", hf_repo, config, split)
    # Parquet piccolo (~170 KB, 1549 righe) → non serve streaming.
    ds = load_dataset(hf_repo, config, split=split)
    rows: list[dict] = [dict(r) for r in ds]
    logger.info("metadata: %d righe (QA), columns=%s", len(rows), list(rows[0].keys()))

    # `chunks` mode: pre-scarica i chunk indicati e filtra i rows ai
    # `video_path` disponibili in essi. Garantisce smoke test
    # deterministico (esattamente N chunk scaricati).
    prescanned_chunks: list[Path] = []
    if chunks is not None:
        zips_dir = out / "_zips"
        zips_dir.mkdir(parents=True, exist_ok=True)
        available_names: set[str] = set()
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
            names_here = _list_videos_in_zip(local_zip)
            available_names |= names_here
            logger.info("[chunk %d] %d video dentro", i, len(names_here))

        before = len(rows)
        rows = [r for r in rows if r["video_path"] in available_names]
        logger.info(
            "filtro chunks=%s: %d → %d righe (video_path matching)",
            chunks, before, len(rows),
        )

    # `n` limita il numero di *video* (non di QA): tiene tutte le domande
    # dei primi N `video_path` distinti incontrati nel parquet. È il knob
    # che controlla il volume di download (i video sono grandi, ~600 MB).
    if n is not None:
        keep_videos: list[str] = []
        seen: set[str] = set()
        for r in rows:
            vp = r["video_path"]
            if vp not in seen:
                seen.add(vp)
                keep_videos.append(vp)
            if len(keep_videos) >= n:
                break
        keep_set = set(keep_videos)
        before = len(rows)
        rows = [r for r in rows if r["video_path"] in keep_set]
        logger.info("trunca a n=%d video: %d → %d righe (QA)", n, before, len(rows))

    if not rows:
        raise RuntimeError("Nessuna riga selezionata dopo filtri.")

    # Una riga = una domanda; più domande condividono lo stesso video.
    target_names: set[str] = {r["video_path"] for r in rows}
    logger.info("video unici da fetchare: %d (su %d domande)", len(target_names), len(rows))

    if chunks is not None:
        # Estrai dai chunk già scaricati nel pre-scan.
        videos_dir = out / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        found_videos: dict[str, str] = {}
        for local_zip in prescanned_chunks:
            extracted = _extract_targets_from_zip(local_zip, target_names, videos_dir)
            found_videos.update(extracted)
            logger.info("[%s] estratti %d video", local_zip.name, len(extracted))
            if not keep_zips:
                local_zip.unlink(missing_ok=True)
        if not keep_zips:
            for d in ((out / "_zips" / "video_chunks"), (out / "_zips")):
                try:
                    d.rmdir()
                except OSError:
                    pass
    else:
        found_videos = _fetch_video_chunks(hf_repo, target_names, out, keep_zips)

    missing_videos = target_names - set(found_videos)
    if missing_videos:
        logger.warning(
            "%d video non trovati in nessun chunk: %s",
            len(missing_videos), sorted(missing_videos)[:5],
        )

    # `dropped.jsonl`: video selezionati ma non estratti (con `lmms-lab`
    # normalmente vuoto in full run; non vuoto solo restringendo `chunks`).
    # Sovrascritto a ogni run. Una riga per video mancante.
    qa_per_video: dict[str, int] = {}
    type_per_video: dict[str, str | None] = {}
    for r in rows:
        vp = r["video_path"]
        qa_per_video[vp] = qa_per_video.get(vp, 0) + 1
        type_per_video.setdefault(vp, r.get("type"))
    dropped_path = out / "dropped.jsonl"
    with dropped_path.open("w") as f:
        for vp in sorted(missing_videos):
            f.write(json.dumps({
                "video_path": vp,
                "key": Path(vp).stem,
                "type": type_per_video.get(vp),
                "num_qa": qa_per_video.get(vp, 0),
            }) + "\n")

    metadata: list[dict] = []
    dropped = 0
    for r in rows:
        vp = r["video_path"]
        rel_video = found_videos.get(vp)
        if rel_video is None:
            dropped += 1
            continue
        key = r["key"]
        uid = r["uid"]
        # `question_type` arriva come list[str] da `datasets`; lo forziamo
        # a lista per robustezza (json non serializza ndarray).
        qtype = r.get("question_type")
        qtype = list(qtype) if qtype is not None else []
        metadata.append({
            "id": f"{key}/{uid}",
            "key": key,
            "type": r.get("type"),
            "video": str(Path("videos") / rel_video),
            "uid": uid,
            "question": r["question"],
            "answer": r["answer"],
            "question_type": qtype,
            "time_reference": r.get("time_reference"),
        })
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
        "missing_videos": len(missing_videos),
        "out_dir": str(out),
        "metadata_path": str(metadata_path),
        "dropped_path": str(dropped_path),
    }


def run(cfg: DictConfig) -> None:
    # `datasets` / `huggingface_hub` leggono HF_HOME al primo import, va
    # settato prima dell'import lazy dentro `prefetch`.
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    chunks = None if cfg.chunks is None else [int(i) for i in cfg.chunks]
    prefetch(
        cfg.hf_repo,
        cfg.config,
        cfg.split,
        cfg.n,
        cfg.out_dir,
        bool(cfg.keep_zips),
        chunks,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="fetch/prefetch_lvbench")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
