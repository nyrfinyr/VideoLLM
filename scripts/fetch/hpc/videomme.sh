#!/usr/bin/env bash
# Prefetch Video-MME su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node tipicamente non hanno
# accesso internet).
#
# DEFAULT (no override) = FULL RUN: scarica i 20 `videos_chunked_NN.zip`
# necessari per coprire tutti i 2700 videoID (~17-101 GB di download
# nominali, lo script estrae on-demand e cancella i zip dopo). Conta
# alcune ore di wall-clock.
#
# Override extra passabili come argomenti, es.:
#
#   # smoke test (10 QA, ~5 GB, ~2 min): SEMPRE passare `chunks=[K]`
#   # insieme a `n=K`, altrimenti i primi N videoID del parquet potrebbero
#   # essere sparsi su più chunk e il download diventa enorme.
#   ./scripts/fetch/hpc/videomme.sh chunks=[1] n=10 wandb.mode=disabled
#
#   # filtro per durata (full run su una fascia, ~1/3 dei chunk):
#   ./scripts/fetch/hpc/videomme.sh duration=short
#
#   # con sottotitoli:
#   ./scripts/fetch/hpc/videomme.sh subtitles=true
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_videomme.py \
    out_dir=/work/tesi_avalenza/datasets/video_mme \
    "$@"
