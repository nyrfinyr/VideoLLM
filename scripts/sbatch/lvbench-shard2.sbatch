#!/bin/bash
#SBATCH --job-name=lvbench-eval-shard2
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# RE-RUN — solo shard 2/3 di LVBench. Il run originale (array job 46824) ha
# schedulato lo shard 2 su una GPU 24 GiB (nullazzo) e ha OOMato tutti i 516
# sample (vedi docs/qwen2_5_vl_3b_baseline.md). Qui forziamo GPU ≥40 GiB.
#
# Sharding: hardcoded shard=2, num_shards=3 → produce lo stesso slice
# `samples[2::3]` del run originale (shard disgiunti da 0/1, deterministico).
# wandb.group=${MODEL}-lvbench → si aggrega con gli shard 0/1 ([`45bdgea9`]
# e [`usuy6en7`]) per il calcolo dell'accuracy finale. Il vecchio run di
# shard 2 ([`0oqf6fev`]) va escluso a mano (è taggato `invalid`/`oom`).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=lvbench \
    hf_home=/work/tesi_avalenza/hf \
    shard=2 \
    num_shards=3 \
    wandb.tags=[eval,lvbench,sbatch,rerun] \
    wandb.group=${MODEL}-lvbench \
    wandb.name=${MODEL}-lvbench-${SLURM_JOB_ID}_2-rerun