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

# Misure (2)+(3) del piano-oracolo: i SOFFITTI su LVBench.
# Quattro condizioni per sample: baseline / peak / oracle (puntatore
# TESTUALE dell'arm A alla cella di picco o alla finestra annotata) / zoom
# (24 frame RICAMPIONATI dentro la finestra annotata, nessun puntatore) —
# il soffitto della metà "marcare" e quello della metà "ricampionare" di
# LoT nello stesso job. Disegno completo in scripts/probe_oracle_ceiling.py.
#
# PRIMO GIRO (93614, senza zoom, 100 sample): baseline 32%, peak 29%,
# oracle 30% — il canale di MARCATURA non ha soffitto. peak_in_window 30%
# (~2x chance, coerente con la sonda (1)): segnale debole ma reale. Con 24
# frame su video di ore la finestra quasi mai contiene un frame campionato:
# la condizione `zoom` misura se portarci i FRAME (non il puntatore) paga.
#
# A parità di SEED condivide i sample con probe_peak_in_window (stessa
# selezione deterministica): i risultati si incrociano per id.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
#
# COSTO: 100 sample x (2 decode in PNG + 5 forward brevi a 24 frame)
# ≈ 25-40 s/sample misurati (18 s/sample il giro a 3 condizioni) → ~1 h.
# 3 h è larga.
#
# COSA GUARDARE (summary wandb, project lvbench, group probe-oracle-ceiling):
#   1. n_skip ~0, n_fallback basso (risposte parseabili);
#   2. acc_zoom vs acc_baseline: È LA MISURA — flip zoom_recovered/broken;
#   3. acc_oracle e acc_peak vs baseline (primo giro: nessun soffitto di
#      marcatura, atteso di nuovo ≈);
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
