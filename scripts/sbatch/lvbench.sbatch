#!/bin/bash
#SBATCH --job-name=lvbench-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# PROD — eval LVBench (1549 QA, 768 frame/video @ max_pixels=50176, setup del
# paper Qwen2.5-VL). Run più pesante: video long-form (1-2h) + ~24k token video
# per sample → mem RAM 32G per il decode. Spezzato in job array (slice strided
# `samples[shard::num_shards]`): ogni shard mescola le durate, bilanciando il
# walltime. I 3 shard condividono `wandb.group=${MODEL}-lvbench`: combina le
# accuracy a fine array. Logging wandb+weave attivo.
# Modello override-abile: `MODEL=qwen3_vl_4b sbatch scripts/sbatch/lvbench.sbatch`.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
MODEL="${MODEL:-qwen2_5_vl_3b}"

uv run python main.py \
    model=${MODEL} \
    dataset=lvbench \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.tags=[eval,lvbench,sbatch,array] \
    wandb.group=${MODEL}-lvbench \
    wandb.name=${MODEL}-lvbench-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.notes="slurm array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)"
