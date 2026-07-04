#!/usr/bin/env bash
# Prefetch Causal2Needles su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node tipicamente non hanno
# accesso internet).
#
# DEFAULT (no override) = FULL RUN: tutte le ~4100 QA rows / 192 video
# (~14 GB, self-contained nel repo HF, niente YouTube/yt-dlp).
#
# RIESEGUIBILE: rilanciare lo script salta i video già scaricati su disco
# (`hf_hub_download` + check `local_video.exists()`).
#
# Override extra passabili come argomenti, es.:
#
#   # smoke test (poche decine di video):
#   ./scripts/fetch/hpc/causal2needles.sh n=200 wandb.mode=disabled
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_causal2needles.py \
    out_dir=/work/tesi_avalenza/datasets/causal2needles \
    "$@"
