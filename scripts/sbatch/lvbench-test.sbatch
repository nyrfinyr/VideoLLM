#!/bin/bash
#SBATCH --job-name=lvbench-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:45:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# THROUGHPUT PROBE — LVBench a config PROD (768 frame @ max_pixels=50176).
# Caso pesante: video long-form (1-2h) + ~24k token/sample → s/sample alto,
# subset piccolo ma shuffle ESSENZIALE (il loader emette per video → senza,
# i 12 sample sarebbero dello stesso video). Processa e COMPLETA, Weave logga
# il summary. Logging Weave ATTIVO, group=<model>-lvbench-test.
# Misura precisa = `model_latency.mean` (s/sample) nel summary o in UI Weave.
#   grep -A6 'Evaluation summary' logs/lvbench-test-<jobid>.out
# Stima prod ≈ 1549 sample × model_latency. Valida anche la VRAM dei 768
# frame: se muore per OOM (non per fine eval) → abbassa max_pixels.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=lvbench \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=12 \
    wandb.group=${MODEL}-lvbench-test \
    wandb.tags=[probe,lvbench,test] \
    wandb.name=${MODEL}-lvbench-test-${SLURM_JOB_ID} \
    wandb.notes="throughput probe slurm ${SLURM_JOB_ID} on $(hostname)"
