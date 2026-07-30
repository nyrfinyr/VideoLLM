#!/bin/bash
#SBATCH --job-name=mvbench-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# THROUGHPUT PROBE — MVBench a config PROD (32 frame). Processa un subset
# RAPPRESENTATIVO (shuffle + limit) e COMPLETA, così Weave logga il summary.
# shuffle è qui ESSENZIALE: il loader emette per task → senza, i primi N
# sarebbero tutti dello stesso task. Logging Weave ATTIVO, group=<model>-mvbench-test.
# Misura precisa = `model_latency.mean` (s/sample) nel summary o in UI Weave
# (usa la MEDIANA: il 1° sample include il warmup).
#   grep -A6 'Evaluation summary' logs/mvbench-test-<jobid>.out
# Stima prod ≈ ~3800 sample × model_latency.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=mvbench \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=60 \
    wandb.group=${MODEL}-mvbench-test \
    wandb.tags=[probe,mvbench,test] \
    wandb.name=${MODEL}-mvbench-test-${SLURM_JOB_ID}
