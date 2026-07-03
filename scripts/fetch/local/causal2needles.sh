#!/usr/bin/env bash
# Smoke test Causal2Needles in locale: 50 QA rows (poche decine di video
# distinti, i video dedupe automaticamente) da causal2needles/Causal2Needles.
# Output in `$HOME/datasets/causal2needles`. wandb disabilitato.
#
# Per un run completo (~4100 QA rows, 192 video, ~14 GB) usa
# `scripts/fetch/hpc/causal2needles.sh` oppure override CLI:
#   ./scripts/fetch/local/causal2needles.sh n=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_causal2needles.py \
    n=50 \
    wandb.mode=disabled \
    "$@"
