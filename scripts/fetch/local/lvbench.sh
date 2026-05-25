#!/usr/bin/env bash
# Smoke test LVBench in locale: 3 video da YouTube (yt-dlp), wandb
# disabilitato. Output in `$HOME/datasets/lvbench`.
#
# Se YouTube blocca i download anonimi, aggiungi cookie del browser:
#   ./scripts/fetch/local/lvbench.sh cookies_from_browser=firefox
#
# Per un run completo (~500 video) usa `scripts/fetch/hpc/lvbench.sh`
# oppure override CLI:
#   ./scripts/fetch/local/lvbench.sh n=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_lvbench.py \
    n=3 \
    wandb.mode=disabled \
    "$@"
