#!/bin/bash
#SBATCH --job-name=video_mme-signal-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# THROUGHPUT PROBE / TUNING — Video-MME video-only (128 frame) con le
# strategy signal-driven (`strategies/`), non ancora provate su questo
# dataset. Richiede SEMPRE model=qwen2_5_vl_3b_attn (SupportsSignals,
# fail-fast altrimenti — vedi strategies/entropy_shortcut.py /
# entropy_attention_resample.py). Default strategy=entropy_attention_resample
# (quella con la logica a 4 rami); per isolare il solo skip-generate:
#   STRATEGY=entropy_shortcut sbatch scripts/sbatch/video_mme-signal-test.sbatch
#
# Le soglie di entropy_attention_resample (entropy_threshold,
# concentration_threshold, window_frac, phase_shift_frac — vedi
# conf/config.yaml sotto strategy.entropy_attention_resample) erano tarate
# su un campione VNBench "retrieval" (dominio/nframes diversi, vedi
# strategies/entropy_attention_resample.py e docs/entropia_confidenza_mcq.md):
# QUESTO script serve a girarle su un subset economico prima di fidarsi del
# run prod (video_mme-signal.sbatch), non a produrre un numero definitivo.
# Override da CLI, es.:
#   sbatch scripts/sbatch/video_mme-signal-test.sbatch strategy.entropy_threshold=1.0
#
# shuffle+limit come video_mme-test.sbatch (subset rappresentativo fra le
# fasce di durata short/medium/long), stesso limit=40 per confrontabilità
# con la baseline uniform già misurata. Logging Weave ATTIVO, group tagga
# anche la strategy così i run non si mescolano con la baseline uniform.
#   grep -A6 'Evaluation summary' logs/video_mme-signal-test-<jobid>.out
# In UI Weave guarda anche `resampled`/`resample_kind`/`top1_pct` per capire
# quanto spesso e come sta ricampionando a questa scala di frame/durata clip.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-qwen2_5_vl_3b_attn}"
STRATEGY="${STRATEGY:-entropy_attention_resample}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    strategy=${STRATEGY} \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=40 \
    wandb.group=${MODEL}-${STRATEGY}-video_mme-test \
    wandb.tags=[probe,video_mme,test,${STRATEGY}] \
    wandb.name=${MODEL}-${STRATEGY}-video_mme-test-${SLURM_JOB_ID} \
    wandb.notes="throughput probe slurm ${SLURM_JOB_ID} on $(hostname)" \
    "$@"
