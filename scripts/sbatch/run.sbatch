#!/bin/bash
#SBATCH --job-name=egoschema-eval
#SBATCH --account=sai2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

set -euo pipefail

echo "node: $(hostname)"
echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

export HYDRA_FULL_ERROR=1

MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    hf_home=/work/tesi_avalenza/hf \
    run.root=/work/tesi_avalenza/datasets/egoschema \
    wandb.tags=[eval,egoschema,sbatch] \
    wandb.group=${MODEL}-egoschema \
    wandb.name=${MODEL}-egoschema-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
