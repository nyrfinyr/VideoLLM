#!/usr/bin/env bash
# Smoke test MVBench in locale: solo `action_antonym` (~17 MB,
# ssv2_video.zip), 5 sample in metadata.jsonl, wandb disabilitato.
# Output in `$HOME/datasets/mvbench`.
#
# Per un run completo (19 task, ~17 GB) usa `scripts/fetch/hpc/mvbench.sh`
# oppure override CLI:
#   ./scripts/fetch/local/mvbench.sh tasks=null n=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_mvbench.py \
    tasks='[action_antonym]' \
    n=5 \
    wandb.mode=disabled \
    "$@"
