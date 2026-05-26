#!/bin/bash
#SBATCH --job-name=lvbench-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# PROD — eval LVBench (1549 QA, 768 frame/video @ max_pixels=50176, setup del
# paper Qwen2.5-VL). È il run più pesante: video long-form (1-2h) + ~24k token
# video per sample → walltime alto e mem RAM 32G per il decode video.
# Se il singolo job non sta nel limite di partizione, valuta un job array per
# spezzare il dataset (o riduci nframes). Logging wandb+weave attivo.
# Modello override-abile: `MODEL=qwen3_vl_4b sbatch scripts/sbatch/lvbench.sbatch`.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export HYDRA_FULL_ERROR=1
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=lvbench \
    hf_home=/work/tesi_avalenza/hf \
    wandb.tags=[eval,lvbench,sbatch] \
    wandb.group=${MODEL}-lvbench \
    wandb.name=${MODEL}-lvbench-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
