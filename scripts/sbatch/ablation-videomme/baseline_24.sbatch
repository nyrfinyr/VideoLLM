#!/bin/bash
#SBATCH --job-name=ablation-baseline_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 1/5: baseline `uniform` a 24 frame SINGOLI (no raddoppio, nessuna
# strategy custom) — riferimento per gli arm signal-driven allo stesso
# nframes (entropy_shift_24.sbatch / entropy_zoom_24.sbatch /
# shift_zoom_router_24.sbatch, stessa cartella) e per il confronto
# raddoppio-frame (baseline_48double.sbatch). Nessun segnale richiesto:
# model=qwen2_5_vl_3b (plain, non `_attn`), quindi `--constraint` apre
# anche le GPU 24G — stesso criterio di scripts/sbatch/video_mme.sbatch,
# da cui questo script deriva (single job, no sharding: 2700 sample a 24
# frame stanno comodi in un job solo).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b \
    dataset=video_mme \
    strategy=uniform \
    dataset.nframes=24 \
    hf_home=/work/tesi_avalenza/hf \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b-baseline_24-${SLURM_JOB_ID} \
    wandb.tags=[ablation,video_mme,baseline_24] \
    wandb.notes="ablation riunione 22-07-26: baseline uniform 24 frame singoli, slurm ${SLURM_JOB_ID} on $(hostname)" \
    "$@"
