#!/bin/bash
#SBATCH --job-name=ablation-baseline_48double
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
# arm 2/5: baseline `uniform` a 48 frame FISICI, che il merge temporale ×2
# sempre attivo di Qwen2.5-VL tratta come "24 raddoppiati" — confronto
# diretto con baseline_24.sbatch (stessa cartella, rapporto 2× esplicito
# richiesto dall'utente). Nessun segnale richiesto: model=qwen2_5_vl_3b
# (plain, non `_attn`), `--constraint` apre anche le GPU 24G — stesso
# criterio di scripts/sbatch/video_mme.sbatch, da cui questo script deriva
# (single job, no sharding).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b \
    dataset=video_mme \
    strategy=uniform \
    dataset.nframes=48 \
    hf_home=/work/tesi_avalenza/hf \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b-baseline_48double-${SLURM_JOB_ID} \
    wandb.tags=[ablation,video_mme,baseline_48double] \
    wandb.notes="ablation riunione 22-07-26: baseline uniform 48 frame (= 24 raddoppiati), slurm ${SLURM_JOB_ID} on $(hostname)" \
    "$@"
