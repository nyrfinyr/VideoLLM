#!/usr/bin/env bash
# Smoke test EgoSchema in locale: 10 video del Subset, wandb disabilitato.
# Output in `$HOME/datasets/egoschema`.
#
# Per un run completo (~500 video) usa `n=null` come override, oppure
# passa al wrapper HPC `scripts/fetch/hpc/egoschema.sh`.
#   ./scripts/fetch/local/egoschema.sh n=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_egoschema.py \
    n=10 \
    wandb.mode=disabled \
    "$@"
