#!/bin/bash
#SBATCH --job-name=video_mme-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# PROD — eval Video-MME (2700 QA, 128 frame/video, logging wandb+weave).
# Modello override-abile: `MODEL=qwen3_vl_4b sbatch scripts/sbatch/video_mme.sbatch`.
# Per la variante con sottotitoli: `... video_mme.sbatch dataset.use_subtitles=true`
# (richiede prefetch con subtitles=true). Breakdown per fascia via la
# colonna `duration` in UI Weave.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export HYDRA_FULL_ERROR=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    hf_home=/work/tesi_avalenza/hf \
    wandb.tags=[eval,video_mme,sbatch] \
    wandb.group=${MODEL}-video_mme \
    wandb.name=${MODEL}-video_mme-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
