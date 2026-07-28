#!/bin/bash
#SBATCH --job-name=video_mme-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# THROUGHPUT PROBE — Video-MME a config PROD (128 frame). Processa un subset
# RAPPRESENTATIVO (shuffle + limit) e COMPLETA, così Weave logga il summary.
# shuffle importante: durate short/medium/long molto diverse → il subset
# casuale media le tre fasce. Logging Weave ATTIVO, group=<model>-video_mme-test.
# Misura precisa = `model_latency.mean` (s/sample) nel summary o in UI Weave
# (in UI guarda anche p50/p95 per via della varianza fra le fasce).
#   grep -A6 'Evaluation summary' logs/video_mme-test-<jobid>.out
# Stima prod ≈ 2700 sample × model_latency.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=40 \
    wandb.group=${MODEL}-video_mme-test \
    wandb.tags=[probe,video_mme,test] \
    wandb.name=${MODEL}-video_mme-test-${SLURM_JOB_ID}
