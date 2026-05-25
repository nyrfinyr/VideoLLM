#!/usr/bin/env bash
# Prefetch MVBench su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# Default: tutti i 19 task supportati (~17 GB di zip).
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node tipicamente non hanno
# accesso internet). Override extra passabili come argomenti, es.:
#   # smoke test (~17 MB, solo ssv2_video.zip):
#   ./scripts/fetch/hpc/mvbench.sh tasks='[action_antonym]' wandb.mode=disabled
#
#   # subset di task:
#   ./scripts/fetch/hpc/mvbench.sh tasks='[action_antonym,object_existence]'
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_mvbench.py \
    out_dir=/work/tesi_avalenza/datasets/mvbench \
    "$@"
