"""Prefetch a small Causal2Needles subset for pipeline testing.

Causal2Needles (NeurIPS D&B 2025, arXiv:2505.19853) is an OPEN-ENDED
long-video QA benchmark built for "needle in a haystack"-style testing:
each question asks the model to locate and reason over one or two
"needles" (cause/effect sentences) buried in a movie-recap video. Unlike
EgoSchema/MVBench/Video-MME/LVBench there is NO multiple-choice `options`
field — `text_answer` is free text.

Single HF repo doubles as both metadata (`test_questions.parquet`) and
video store (`videos/{yms,symon}/<id>.mp4`, self-contained, no YouTube
download needed). Many QA rows share the same `video_id` (a video carries
~20-40 questions), so streaming the first N rows naturally re-downloads
few distinct videos — `hf_hub_download` skips files already on disk.

Run on the login node (compute nodes typically lack internet). Output
layout (self-contained, no `datasets` dependency at read time):

    out_dir/
        metadata.jsonl     # one JSON object per QA row; `video` field
                           #   points to a relative path under out_dir/
        videos/
            yms/0085.mp4
            symon/0IUynKY8i3I.mp4
            ...

Hydra overrides:
    uv run python fetch/prefetch_causal2needles.py n=200
    uv run python fetch/prefetch_causal2needles.py wandb.mode=disabled
    uv run python fetch/prefetch_causal2needles.py --cfg job
"""
import json
import logging
import os
import sys
from pathlib import Path

# When invoked as `python fetch/prefetch_causal2needles.py`, only `fetch/`
# is on sys.path — add its parent (repo root) so the absolute import
# `from utils.obs import ...` resolves regardless of invocation mode
# (script, `python -m fetch.prefetch_causal2needles`, or installed entry point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import weave
from omegaconf import DictConfig

from utils.obs import init_observability

logger = logging.getLogger(__name__)


@weave.op()
def prefetch(
    repo: str,
    config: str,
    n: int | None,
    out_dir: str,
) -> dict:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out: Path = Path(out_dir)
    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # `causal2needles/Causal2Needles` ships QA metadata as a single parquet
    # (`test_questions.parquet`, config="default", split="test"). Streaming
    # + `.take(n)` avoids materializing all 4.1k rows just to keep a
    # smoke-test subset.
    logger.info(
        "Streaming %s rows from %s [%s]",
        "all" if n is None else f"first {n}",
        repo,
        config,
    )
    stream = load_dataset(repo, config, split="test", streaming=True)
    samples = list(stream if n is None else stream.take(n))
    if not samples:
        raise RuntimeError(f"No samples streamed from {repo} [{config}].")
    total = len(samples)
    logger.info("Materialized %d QA rows; columns=%s", total, list(samples[0].keys()))

    metadata = []
    seen_videos: set[str] = set()
    for i, sample in enumerate(samples):
        video_id = sample["video_id"]  # e.g. "yms/0085" or "symon/0IUynKY8i3I"
        # Videos live at `videos/<video_id>.mp4` in the same repo as the
        # parquet — no separate videos_repo, unlike EgoSchema.
        repo_path = f"videos/{video_id}.mp4"
        local_video = out / repo_path

        if local_video.exists():
            if video_id not in seen_videos:
                logger.info("[%d/%d] video already cached: %s", i + 1, total, video_id)
        else:
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(out),
            )
            logger.info("[%d/%d] fetched %s", i + 1, total, repo_path)
        seen_videos.add(video_id)

        sample["video"] = str(local_video.relative_to(out))
        metadata.append(sample)

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for sample in metadata:
            f.write(json.dumps(sample) + "\n")

    logger.info(
        "Wrote %d QA rows (%d distinct videos) + metadata.jsonl to %s",
        len(metadata), len(seen_videos), out,
    )
    return {
        "subset_size": len(metadata),
        "n_videos": len(seen_videos),
        "out_dir": str(out),
        "metadata_path": str(metadata_path),
    }


def run(cfg: DictConfig) -> None:
    # `datasets` reads HF_HOME at first import, so set it before the
    # lazy import inside `prefetch`.
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    prefetch(
        cfg.repo,
        cfg.config,
        cfg.n,
        cfg.out_dir,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="fetch/prefetch_causal2needles")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
