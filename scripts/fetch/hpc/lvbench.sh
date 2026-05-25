#!/usr/bin/env bash
# Prefetch LVBench su HPC AImageLab: path hardcoded in /work/tesi_avalenza.
# `hf_home` non lo passiamo: `.bashrc` su HPC esporta HF_HOME=/work/tesi_avalenza/hf
# e l'`oc.env` nel yaml lo raccoglie da solo.
#
# Da lanciare dal LOGIN node (i compute node non hanno internet, e yt-dlp
# qui non funzionerebbe).
#
# IMPORTANTE — YouTube anti-bot:
# Senza cookies, yt-dlp dopo qualche decina di download viene bloccato con
# "Sign in to confirm you're not a bot". Su HPC non esiste un browser per
# `cookies_from_browser=`, va passato un file `cookies.txt`. Workflow:
#   1) Sul tuo PC locale: installa l'estensione "Get cookies.txt LOCALLY"
#      (Firefox o Chrome), apri youtube.com loggato, esporta cookies.txt.
#   2) `scp cookies.txt avalenza@hpc:/work/tesi_avalenza/secrets/cookies.txt`
#      (chmod 600).
#   3) Lancia:
#        ./scripts/fetch/hpc/lvbench.sh cookiefile=/work/tesi_avalenza/secrets/cookies.txt
#
# RIESEGUIBILE: rilanciare lo script salta i video già scaricati e ritenta
# quelli falliti. Conviene 2-3 run a distanza di qualche ora per assorbire
# i fallimenti transient (rate-limit, network). I video definitivamente
# morti restano in `dropped.jsonl` — quello è il sample size effettivo.
#
# Override extra passabili come argomenti, es.:
#   ./scripts/fetch/hpc/lvbench.sh n=3
#   ./scripts/fetch/hpc/lvbench.sh cookiefile=/work/tesi_avalenza/secrets/cookies.txt
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run python fetch/prefetch_lvbench.py \
    out_dir=/work/tesi_avalenza/datasets/lvbench \
    "$@"
