#!/bin/bash
#SBATCH --job-name=video_mme-signal-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --time=07:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# PROD — eval Video-MME (2700 QA, 128 frame/video, video-only) con
# entropy_attention_resample, da confrontare con la baseline uniform già
# misurata (video_mme.sbatch). Richiede SEMPRE model=qwen2_5_vl_3b_attn
# (SupportsSignals, fail-fast altrimenti — vedi strategies/entropy_shortcut.py
# / entropy_attention_resample.py).
#
# Soglie e job array validati dallo sweep su subset 250/2700 (vedi README,
# sezione "Sweep entropy_attention_resample"):
# - entropy_threshold=0.7, concentration_threshold=30 → 66.0% sul subset
#   (pari a et=1.0, ma quest'ultimo è ridondante: a et=1.0 il
#   concentration_threshold non ha alcun effetto sulle predizioni — stesso
#   n_correct, stessa resampled_fraction a ct=20 o 30 — quindi et=0.7/ct=30
#   resta la scelta, oltretutto più economica in latenza).
# - Job array (4 shard, `samples[shard::num_shards]`, 675 sample/shard):
#   un job singolo qui non sta in un `--time` ragionevole. Latenza misurata
#   24.45 s/sample (run `y6u5k1pq`, L40S) → 675 × 24.45s ≈ 4h35m di calcolo
#   puro per shard; --time=07:00:00 dà ~1.5x di margine, in linea con
#   video_mme.sbatch (9.3h raw → 14h) e lvbench.sbatch (8.2h raw → 12h).
# - `--constraint` pinnato a SOLE 45G (A40/L40S): lo sweep ha mostrato OOM
#   totale (0/250 predizioni valide, CUDA out of memory dentro
#   full_visual_attention) su nodi 24G (RTX6000/A5000) con questa strategy
#   — vedi README. Il --time sopra assume L40S/A40 (nessun dato di
#   latenza su RTX6000-24G qui, e comunque escluso dal constraint).
#
# Override eventuali soglie/strategy da CLI dopo lo script, es. per
# confrontare anche entropy_shortcut:
#   sbatch scripts/sbatch/video_mme-signal.sbatch strategy=entropy_shortcut
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
MODEL="${MODEL:-qwen2_5_vl_3b_attn}"
STRATEGY="${STRATEGY:-entropy_attention_resample}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    strategy=${STRATEGY} \
    strategy.entropy_threshold=0.7 \
    strategy.concentration_threshold=30 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.tags=[eval,video_mme,sbatch,array,${STRATEGY}] \
    wandb.group=${MODEL}-${STRATEGY}-video_mme \
    wandb.name=${MODEL}-${STRATEGY}-video_mme-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.notes="slurm array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)" \
    "$@"
