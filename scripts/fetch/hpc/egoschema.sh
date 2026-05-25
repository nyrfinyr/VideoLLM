#!/usr/bin/env bash
# Prefetch EgoSchema su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node tipicamente non hanno
# accesso internet).
#
# DEFAULT (no override) = FULL RUN: scarica tutti i ~500 mp4 del Subset
# (~100 GB tot, ~200 MB per video). Conta diverse ore di wall-clock.
# Idempotente: rilanciandolo salta i video già presenti.
#
# Override extra passabili come argomenti, es.:
#   # smoke test (20 video, ~4 GB):
#   ./scripts/fetch/hpc/egoschema.sh n=20 wandb.mode=disabled
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_egoschema.py \
    out_dir=/work/tesi_avalenza/datasets/egoschema \
    "$@"
