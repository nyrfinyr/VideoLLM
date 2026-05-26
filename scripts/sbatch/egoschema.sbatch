#!/bin/bash
#SBATCH --job-name=egoschema-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# ~500 sample (config Subset) × ~20 s/sample @ 64 frame ≈ 2,8 h (worst-case
# card RTX6000-24G; più veloce su A40/L40S). 2:30 era tarato sul vecchio
# fps=0.25 (~45 frame) → ora serve più margine.
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# PROD — eval EgoSchema (dataset intero, logging wandb+weave attivo).
# Modello override-abile: `MODEL=qwen3_vl_4b sbatch scripts/sbatch/egoschema.sbatch`.
# `dataset.root` arriva da conf/dataset/egoschema.yaml (path HPC /work/...).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export HYDRA_FULL_ERROR=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=egoschema \
    hf_home=/work/tesi_avalenza/hf \
    wandb.tags=[eval,egoschema,sbatch] \
    wandb.group=${MODEL}-egoschema \
    wandb.name=${MODEL}-egoschema-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
