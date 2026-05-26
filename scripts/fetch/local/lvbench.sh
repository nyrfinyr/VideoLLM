#!/usr/bin/env bash
# Smoke test LVBench in locale: scarica SOLO `video_chunks/videos_chunk_001.zip`
# (~3.4 GB) da lmms-lab/LVBench, filtra il parquet ai video realmente
# presenti dentro, prende i primi 3 video (con tutte le loro QA). Garantisce
# che `metadata.jsonl` venga sempre prodotto scaricando un solo chunk
# anziché iterare su tutti i 14.
#
# Output in `$HOME/datasets/lvbench`. wandb disabilitato.
#
# Per un run completo (103 video, ~61 GB) usa `scripts/fetch/hpc/lvbench.sh`
# oppure override CLI:
#   ./scripts/fetch/local/lvbench.sh n=null chunks=null wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_lvbench.py \
    chunks=[1] \
    n=3 \
    wandb.mode=disabled \
    "$@"
