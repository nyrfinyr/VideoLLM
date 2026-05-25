"""Prefetch a small EgoSchema subset for pipeline testing.

Streams metadata page-by-page from the HF Hub and downloads only the N
videos we actually want — avoids materializing the full Subset (~100 GB of
mp4) in the HF cache.

Run on the login node (compute nodes typically lack internet). Output
layout (self-contained, no `datasets` dependency at read time):

    out_dir/
        metadata.jsonl     # one JSON object per sample; `video` field
                           #   points to a relative path under out_dir/
        videos/
            0000.mp4
            0001.mp4
            ...

Hydra overrides:
    uv run python fetch/prefetch_egoschema.py n=50
    uv run python fetch/prefetch_egoschema.py wandb.mode=disabled
    uv run python fetch/prefetch_egoschema.py --cfg job
"""
import json
import logging
import os
import sys
from pathlib import Path

# When invoked as `python fetch/prefetch_egoschema.py`, only `fetch/` is on
# sys.path — add its parent (repo root) so the absolute import
# `from utils.obs import ...` resolves regardless of invocation mode
# (script, `python -m fetch.prefetch_egoschema`, or installed entry point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import weave
from omegaconf import DictConfig

from utils.obs import init_observability

logger = logging.getLogger(__name__)


@weave.op()
def prefetch(
    metadata_repo: str,
    videos_repo: str,
    config: str,
    n: int | None,
    out_dir: str,
) -> dict:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out: Path = Path(out_dir)
    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # `lmms-lab/egoschema` is metadata-only (no `video` column). Streaming
    # avoids materializing the whole split into the HF cache; `.take(n)`
    # bounds it to the first n rows. `n=None` consumes the full stream
    # (~500 rows for the Subset config).
    logger.info(
        "Streaming %s rows from %s [%s]",
        "all" if n is None else f"first {n}",
        metadata_repo,
        config,
    )
    stream = load_dataset(metadata_repo, config, split="test", streaming=True)
    samples = list(stream if n is None else stream.take(n))
    if not samples:
        raise RuntimeError(f"No samples streamed from {metadata_repo} [{config}].")
    total = len(samples)
    logger.info("Materialized %d samples; columns=%s", total, list(samples[0].keys()))

    metadata = []
    for i, sample in enumerate(samples):
        video_idx = sample["video_idx"]
        # `yanlaiy/EgoSchema` stores mp4s under `videos/<uuid>.mp4`.
        # `local_dir=out` + this relative filename lands the file at
        # `out/videos/<uuid>.mp4`, preserving the layout without renames.
        repo_path = f"videos/{video_idx}.mp4"
        local_video = out / repo_path

        if local_video.exists():
            logger.info("[%d/%d] already cached: %s", i + 1, total, local_video.name)
        else:
            hf_hub_download(
                repo_id=videos_repo,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(out),
            )
            logger.info("[%d/%d] fetched %s", i + 1, total, repo_path)

        sample["video"] = str(local_video.relative_to(out))
        metadata.append(sample)

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for sample in metadata:
            f.write(json.dumps(sample) + "\n")

    logger.info("Wrote %d videos + metadata.jsonl to %s", len(metadata), out)
    return {
        "subset_size": len(metadata),
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
        cfg.metadata_repo,
        cfg.videos_repo,
        cfg.config,
        cfg.n,
        cfg.out_dir,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="fetch/prefetch_egoschema")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
