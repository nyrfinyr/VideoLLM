#!/usr/bin/env bash
# Smoke test Video-MME in locale: scarica SOLO `videos_chunked_01.zip`
# (~5 GB), filtra il parquet ai videoID realmente presenti dentro, prende
# le prime 10 domande. Garantisce che `metadata.jsonl` venga sempre
# prodotto in ~10 min anziché iterare su tutti i 20 chunk.
#
# Output in `$HOME/datasets/video_mme`. wandb disabilitato.
#
# Per un run completo (2700 QA, ~17-101 GB a seconda del cache HF)
# usa `scripts/fetch/hpc/videomme.sh` oppure override CLI:
#   ./scripts/fetch/local/videomme.sh n=null chunks=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_videomme.py \
    chunks=[1] \
    n=10 \
    wandb.mode=disabled \
    "$@"
