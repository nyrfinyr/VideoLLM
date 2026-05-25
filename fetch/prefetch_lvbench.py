"""Prefetch LVBench videos + QA metadata for pipeline testing.

LVBench (`zai-org/LVBench`) hosta solo `video_info.meta.jsonl` su HF —
i video (~500, fino a 2h ciascuno) vivono su YouTube e vanno scaricati
via `yt-dlp` usando il campo `key` (YouTube video ID).

Schema sorgente per riga del jsonl:

    {"key": "<youtube_id>",
     "type": "<categoria_video>",
     "qa": [{"uid", "question", "answer", "question_type", "time_reference"}, ...]}

Le opzioni MCQ sono già embedded nel testo di `question` (formato
"...\n(A) ...\n(B) ..."); il loader `evals.lvbench` le riparserà.

Output (self-contained, stesso schema degli altri prefetch):

    out_dir/
        metadata.jsonl     # una riga per QA; `video` path relativo a
                           #   out_dir. Le righe relative a video che
                           #   yt-dlp non è riuscito a recuperare
                           #   (privati/rimossi/geo-bloccati) sono
                           #   scartate.
        dropped.jsonl      # una riga JSON per ogni video saltato, con
                           #   `{key, type, num_qa, error}`. Sovrascritto
                           #   a ogni run: re-eseguendo lo script, i video
                           #   recuperati escono dalla lista. Usalo per
                           #   documentare il sample size effettivo e per
                           #   ritentare con cookies o più tardi.
        videos/
            <youtube_id>.<ext>     # mp4/webm/mkv in base al format
                                   #   selector di yt-dlp.

Note operative:
- Run on the login node (compute nodes typically lack internet).
- YouTube blocca aggressivamente i download anonimi: in caso di
  "Sign in to confirm you're not a bot" passa cookies via
  `cookies_from_browser=firefox` (o `cookiefile=/path/to/cookies.txt`).
- Lo script è idempotente: rilanciandolo (a) salta i video già scaricati,
  (b) ritenta quelli che prima erano falliti. Per un dataset completo
  conviene rilanciare 2-3 volte a distanza di qualche ora per assorbire
  rate-limit / network transient.
- Mortalità irreversibile attesa su LVBench: i video YouTube vengono
  rimossi nel tempo (account terminati, video privati, ecc.). Non esiste
  un mirror ufficiale dei video. `dropped.jsonl` è la fonte di verità
  per il sample size effettivo da riportare nei risultati.

Hydra overrides:
    uv run python fetch/prefetch_lvbench.py n=3
    uv run python fetch/prefetch_lvbench.py format='best[height<=360][ext=mp4]'
    uv run python fetch/prefetch_lvbench.py cookies_from_browser=firefox
    uv run python fetch/prefetch_lvbench.py --cfg job
"""
import json
import logging
import os
import sys
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


# Estensioni che yt-dlp può produrre. mp4 è il default per `best[ext=mp4]`;
# webm/mkv possono comparire con format selectors più liberali o quando un
# postprocessor effettua un merge audio/video.
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".m4v")


def _existing_video(videos_dir: Path, key: str) -> Path | None:
    """Ritorna il file già scaricato per `key`, o None se assente."""
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{key}{ext}"
        if p.exists():
            return p
    # Fallback: yt-dlp può salvare con suffissi inattesi (es. .mp4.part
    # se interrotto, o estensioni non in lista). Glob largo ma filtrato.
    for p in videos_dir.glob(f"{key}.*"):
        if p.is_file() and not p.name.endswith(".part"):
            return p
    return None


def _download_one(
    key: str,
    videos_dir: Path,
    format_selector: str,
    cookiefile: str | None,
    cookies_from_browser: str | None,
) -> tuple[Path | None, str | None]:
    """yt-dlp wrapper idempotente.

    Ritorna `(path, None)` se ok, `(None, error_msg)` se fallisce.
    `error_msg` è la stringa di errore raw (utile per `dropped.jsonl`).
    """
    import yt_dlp

    existing = _existing_video(videos_dir, key)
    if existing is not None:
        return existing, None

    url = f"https://www.youtube.com/watch?v={key}"
    ydl_opts: dict = {
        "outtmpl": str(videos_dir / "%(id)s.%(ext)s"),
        "format": format_selector,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    if cookies_from_browser:
        # yt-dlp accetta una tupla `(browser, profile, keyring, container)`;
        # nel 99% dei casi basta il browser.
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        logger.warning("yt-dlp DownloadError per %s: %s", key, msg)
        return None, msg
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.warning("errore inatteso scaricando %s: %s", key, msg)
        return None, msg

    local = _existing_video(videos_dir, key)
    if local is None:
        # yt-dlp non ha sollevato eccezione ma il file non c'è —
        # raro (format selector che non matcha nulla, ecc.).
        return None, "yt-dlp non ha prodotto file dopo download"
    return local, None


@weave.op()
def prefetch(
    metadata_repo: str,
    metadata_file: str,
    n: int | None,
    out_dir: str,
    format_selector: str,
    cookiefile: str | None,
    cookies_from_browser: str | None,
) -> dict:
    from huggingface_hub import hf_hub_download

    out: Path = Path(out_dir)
    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    logger.info("scarico %s da %s", metadata_file, metadata_repo)
    local_meta = hf_hub_download(
        repo_id=metadata_repo,
        repo_type="dataset",
        filename=metadata_file,
    )
    with open(local_meta) as f:
        videos: list[dict] = [json.loads(line) for line in f if line.strip()]
    total_qa = sum(len(v.get("qa", [])) for v in videos)
    logger.info("metadata: %d video, %d QA totali", len(videos), total_qa)

    if n is not None:
        videos = videos[:n]
        logger.info("tronca a n=%d video", n)

    found: dict[str, Path] = {}
    dropped_log: list[dict] = []
    for i, v in enumerate(videos):
        key = v["key"]
        local, err = _download_one(
            key, videos_dir, format_selector, cookiefile, cookies_from_browser,
        )
        if local is None:
            logger.warning(
                "[%d/%d] %s: DROP (%s)", i + 1, len(videos), key, err,
            )
            dropped_log.append({
                "key": key,
                "type": v.get("type"),
                "num_qa": len(v.get("qa", [])),
                "error": err,
            })
            continue
        found[key] = local
        logger.info("[%d/%d] %s → %s", i + 1, len(videos), key, local.name)

    logger.info("scaricati %d/%d video", len(found), len(videos))

    # `dropped.jsonl` è la fonte di verità per la documentazione del
    # sample size effettivo. Sovrascritto a ogni run: re-eseguendo lo
    # script i video che ora vanno escono dalla lista, quelli ancora
    # morti restano. Utile per riprovare con cookies diversi o più tardi.
    dropped_path = out / "dropped.jsonl"
    with dropped_path.open("w") as f:
        for d in dropped_log:
            f.write(json.dumps(d) + "\n")
    if dropped_log:
        logger.info(
            "scritto %s con %d video falliti (riprovare con cookies o ritentare più tardi)",
            dropped_path, len(dropped_log),
        )

    metadata: list[dict] = []
    for v in videos:
        key = v["key"]
        local = found.get(key)
        if local is None:
            continue
        video_rel = str(local.relative_to(out))
        for qa in v.get("qa", []):
            metadata.append({
                "id": f"{key}/{qa['uid']}",
                "key": key,
                "type": v.get("type"),
                "video": video_rel,
                "uid": qa["uid"],
                "question": qa["question"],
                "answer": qa["answer"],
                "question_type": qa.get("question_type"),
                "time_reference": qa.get("time_reference"),
            })

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for sample in metadata:
            f.write(json.dumps(sample) + "\n")

    logger.info("Wrote %d sample + metadata.jsonl in %s", len(metadata), out)
    return {
        "subset_size": len(metadata),
        "num_videos": len(found),
        "missing_videos": len(videos) - len(found),
        "out_dir": str(out),
        "metadata_path": str(metadata_path),
        "dropped_path": str(dropped_path),
    }


def run(cfg: DictConfig) -> None:
    # `huggingface_hub` legge HF_HOME al primo import; settalo prima
    # dell'import lazy dentro `prefetch`.
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    prefetch(
        cfg.metadata_repo,
        cfg.metadata_file,
        cfg.n,
        cfg.out_dir,
        cfg.format,
        cfg.cookiefile,
        cfg.cookies_from_browser,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="fetch/prefetch_lvbench")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
