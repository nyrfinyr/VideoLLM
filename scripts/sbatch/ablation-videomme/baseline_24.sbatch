#!/bin/bash
#SBATCH --job-name=ablation-baseline_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 1/5: baseline `uniform` a 24 frame, regime NORMALE (merge a coppie:
# 24 timestamp distinti → t=12 celle da 2 frame ciascuna). È il
# riferimento per tutti gli altri arm della cartella:
#   baseline_24double  24 timestamp, doubled → t=24 celle da 1 frame
#   entropy_shift_24 / entropy_zoom_24 / shift_zoom_router_24
#
# `model=qwen2_5_vl_3b_attn` ANCHE QUI, benché `uniform` non usi segnali:
# il preset `_attn` registra `qwen25_attn_capture` + `eager_mask` sul
# decoder testuale (models/qwen_attn.py), quindi la `generate()` gira su
# un kernel/maschera diversi dal preset plain. Con il modello plain la
# baseline differirebbe dagli arm signal-driven per DUE cose (strategy +
# kernel di attention) invece di una sola. Stessa convenzione del
# controllo uniform dello sweep (scripts/sweep_video_mme_thresholds.sh:62,
# `MODEL=qwen2_5_vl_3b_attn STRATEGY=uniform`). Costo invariato: senza
# `generate_with_signals` non c'è nessuna cattura, quindi il costo è
# quello di `uniform` puro.
#
# Job array a 3 shard (900 sample ciascuno, `samples[shard::num_shards]`
# in utils/samples.py).
#
# `--time=06:00:00`, tenuto basso perché il backfill di SLURM schedula
# prima i job corti e qui ne vanno in coda 15 (5 arm × 3 shard). Stima:
# 12.4 s/sample è la latenza misurata a 128 FRAME su L40S
# (docs/probe_throughput.md); a 24 frame i token visivi sono ~5× meno
# (t=12 celle contro 64) → ~5-6 s/sample su L40S, ~15 s/sample su
# RTX6000-24G (penalità 2.7× documentata lì) ≈ 3.8h/shard. Budget reale:
# 24 s/sample. Se uno shard va in timeout si rilancia solo quello
# (`sbatch --array=<i> --time=10:00:00 ...`), non l'arm intero.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy
# dell'arm è quella POOLED (Σ corretti / Σ sample), non la media delle
# tre; stesso metodo di LVBench (vedi README, nota LVBench).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=uniform \
    dataset.nframes=24 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-baseline_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,baseline_24] \
    "$@"
