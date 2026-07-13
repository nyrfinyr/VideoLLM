#!/bin/bash
#SBATCH --job-name=video_mme-signal-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# PROD — eval Video-MME (2700 QA, 128 frame/video, video-only) con le
# strategy signal-driven, da confrontare con la baseline uniform già
# misurata (video_mme.sbatch). Richiede SEMPRE model=qwen2_5_vl_3b_attn
# (SupportsSignals, fail-fast altrimenti — vedi strategies/entropy_shortcut.py
# / entropy_attention_resample.py). Default strategy=entropy_attention_resample;
# per isolare il solo skip-generate:
#   STRATEGY=entropy_shortcut sbatch scripts/sbatch/video_mme-signal.sbatch
#
# NON LANCIARE finché le soglie di entropy_attention_resample
# (entropy_threshold/concentration_threshold/window_frac/phase_shift_frac,
# conf/config.yaml sotto strategy.entropy_attention_resample — tarate su
# VNBench "retrieval", dominio/nframes diversi da Video-MME) non sono state
# validate su un subset con `video_mme-signal-test.sbatch`: senza tuning i
# numeri di questo run pieno non sono comparabili con la baseline. Time
# budget 18h (contro le 12h di video_mme.sbatch): ogni sample ricampionato
# da entropy_attention_resample raddoppia i forward pass rispetto a uniform,
# quindi il tempo totale dipende dalla frazione di sample che ricampionano
# (ignota finché non si è girato il probe).
#
# Override eventuali soglie da CLI dopo lo script, es.:
#   sbatch scripts/sbatch/video_mme-signal.sbatch strategy.entropy_threshold=1.0
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
MODEL="${MODEL:-qwen2_5_vl_3b_attn}"
STRATEGY="${STRATEGY:-entropy_attention_resample}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    strategy=${STRATEGY} \
    hf_home=/work/tesi_avalenza/hf \
    wandb.tags=[eval,video_mme,sbatch,${STRATEGY}] \
    wandb.group=${MODEL}-${STRATEGY}-video_mme \
    wandb.name=${MODEL}-${STRATEGY}-video_mme-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)" \
    "$@"
