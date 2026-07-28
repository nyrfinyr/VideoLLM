#!/bin/bash
#SBATCH --job-name=egoschema-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# THROUGHPUT PROBE — EgoSchema a config PROD (64 frame). Processa un subset
# RAPPRESENTATIVO (shuffle + limit) e COMPLETA, così Weave logga il summary.
# Logging Weave ATTIVO, group=<model>-egoschema-test.
# Misura precisa = `model_latency.mean` (s/sample) nel summary a fine log o
# in UI Weave (usa la MEDIANA in UI: il 1° sample include il warmup CUDA).
#   grep -A6 'Evaluation summary' logs/egoschema-test-<jobid>.out
# Stima prod ≈ 500 sample × model_latency. Alza `limit` per più precisione.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=egoschema \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=40 \
    wandb.group=${MODEL}-egoschema-test \
    wandb.tags=[probe,egoschema,test] \
    wandb.name=${MODEL}-egoschema-test-${SLURM_JOB_ID}
