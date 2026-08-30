#!/bin/bash
#SBATCH --job-name=probe-oracle-ceiling
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Misura (2) del piano-oracolo: il SOFFITTO del puntatore su LVBench.
# Tre condizioni per sample (baseline / peak / oracle), stesso prompt, stesso
# testo di puntatore dell'arm attention_highlight — cambia solo DOVE punta:
# la cella di picco dell'attenzione o l'intervallo ANNOTATO quantizzato ai
# confini di cella. Disegno completo in scripts/probe_oracle_ceiling.py.
#
# È la misura che separa i due mondi che l'accuracy da sola confonde:
#   acc_oracle ≈ acc_baseline  → puntare BENE non serve: la metà "COME" di
#       LoT è morta per principio su questo benchmark (risultato di tesi);
#   acc_oracle >> acc_baseline → il soffitto esiste, e acc_peak dice quanto
#       ne recupera oggi il segnale d'attenzione (atteso ≈ baseline).
#
# A parità di SEED condivide i sample con probe_peak_in_window (stessa
# selezione deterministica): i risultati si incrociano per id.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
#
# COSTO: 100 sample x (decode-once in PNG + 4 forward brevi a 24 frame)
# ≈ 30-50 s/sample → ~1-1.5 h. 3 h è larga.
#
# COSA GUARDARE (summary wandb, project lvbench, group probe-oracle-ceiling):
#   1. n_skip ~0, n_fallback basso (risposte parseabili);
#   2. acc_oracle vs acc_baseline, e i flip appaiati oracle_recovered /
#      oracle_broken (più sensibili del saldo);
#   3. acc_peak vs acc_baseline (atteso ≈, coerente coi null Video-MME);
#   4. peak_in_window_rate: deve essere coerente col valore della sonda (1)
#      sullo stesso rowset — è il cross-check fra le due misure;
#   5. la Table `samples` per l'ispezione per-sample.
#
# Uso:
#     sbatch scripts/sbatch/probe_oracle_ceiling.sbatch
#     NSAMPLES=60 SEED=0 ROWSET=entity sbatch scripts/sbatch/probe_oracle_ceiling.sbatch

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

NSAMPLES="${NSAMPLES:-100}"
SEED="${SEED:-0}"

echo "n_samples: ${NSAMPLES}  seed: ${SEED}  nframes: ${NFRAMES:-24}  rowset: ${ROWSET:-question}"

uv run python scripts/probe_oracle_ceiling.py "${NSAMPLES}" "${SEED}"
