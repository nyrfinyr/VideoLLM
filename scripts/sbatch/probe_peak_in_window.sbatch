#!/bin/bash
#SBATCH --job-name=probe-peak-in-window
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Misura (1) del piano-oracolo: il picco d'attenzione cade nella finestra di
# evidenza annotata di LVBench? Per rowset {all, question, entity} sulla
# STESSA attenzione (un solo forward per sample + la generazione solo-testo
# dell'estrattore entity). Disegno completo nella docstring di
# scripts/probe_peak_in_window.py.
#
# PERCHÉ PRIMA DI OGNI ALTRO ARM: su Video-MME attention ≡ random (2 sample
# su 2700) con canale vivo (antipeak −6.46 pp) — l'accuracy non dice se il
# segnale è sbagliato o se non c'era nulla da guadagnare. Qui il giudizio è
# diretto contro la ground truth temporale, senza inferenza aggiuntiva.
# La misura (2) — il SOFFITTO: marcare la cella vera invece del picco — è il
# passo successivo e usa gli arm esistenti (A e C accettano già una cella).
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
#
# COSTO: 60 sample x (decode 24 frame da video LUNGHI + prefill + ~32 token
# di estrazione) ≈ 30-60 s/sample, dominato dal decode: ~30-60 min totali.
# 2 h è larga.
#
# COSA GUARDARE (summary wandb, project lvbench, group probe-peak-in-window):
#   1. n_skip ~0 e n_samples=60 (un video mancante/rotto = skip loggato);
#   2. chance_mean: il caso NON è 1/12 — le finestre possono coprire più
#      celle; ogni hit rate va letto contro questo numero;
#   3. hit_rate_all / hit_rate_question / hit_rate_entity contro chance_mean:
#      tutti ≈ chance → il picco non punta all'evidenza (i null di Video-MME
#      spiegati alla radice); entity > question > all >> chance → il segnale
#      esiste e l'affilatura paga;
#   4. paired_only_entity vs paired_only_question: il confronto appaiato
#      entity/question sample per sample;
#   5. entity_mapped_rate / entity_none_rate: il probe 93573_0 su Video-MME
#      faceva 80% utile con ~17% NONE — LVBench ha domande più specifiche,
#      atteso mapped più alto;
#   6. la Table `samples` per la verifica manuale domanda→entity→hit.
#
# Uso:
#     sbatch scripts/sbatch/probe_peak_in_window.sbatch
#     NSAMPLES=100 SEED=1 sbatch scripts/sbatch/probe_peak_in_window.sbatch

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

NSAMPLES="${NSAMPLES:-60}"
SEED="${SEED:-0}"

echo "n_samples: ${NSAMPLES}  seed: ${SEED}  nframes: ${NFRAMES:-24}"

uv run python scripts/probe_peak_in_window.py "${NSAMPLES}" "${SEED}"
