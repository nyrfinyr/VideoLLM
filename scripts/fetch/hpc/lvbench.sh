#!/usr/bin/env bash
# Prefetch LVBench su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node tipicamente non hanno
# accesso internet).
#
# DEFAULT (no override) = FULL RUN: scarica i `video_chunks/videos_chunk_NNN.zip`
# (14 chunk, ~61 GB nominali) necessari a coprire tutti i 103 video. Lo
# script estrae on-demand e cancella i zip dopo l'estrazione, fermandosi
# appena tutti i video sono coperti. Conta alcune ore di wall-clock.
#
# RIESEGUIBILE: rilanciare lo script salta i video già estratti su disco e
# scarica solo i chunk ancora necessari. Tutto viene da lmms-lab/LVBench
# (mirror HF self-contained) — niente più YouTube/yt-dlp, niente cookies.
#
# Override extra passabili come argomenti, es.:
#
#   # smoke test (3 video, 1 chunk ~3.4 GB): SEMPRE passare `chunks=[K]`
#   # insieme a `n=K`, altrimenti i primi N video del parquet potrebbero
#   # essere sparsi su più chunk e il download diventa enorme.
#   ./scripts/fetch/hpc/lvbench.sh chunks=[1] n=3 wandb.mode=disabled
#
#   # mantieni i zip scaricati (per re-run / ispezione):
#   ./scripts/fetch/hpc/lvbench.sh keep_zips=true
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_lvbench.py \
    out_dir=/work/tesi_avalenza/datasets/lvbench \
    "$@"
