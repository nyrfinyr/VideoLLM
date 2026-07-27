#!/bin/bash
#SBATCH --job-name=ablation-entropy_zoom_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 4/5: gate di entropia invariato (entropy_threshold=0.7), ma OGNI
# resample forzato nel ramo zoom a SINGOLO PICCO (`zoom_peak`, non
# multi-regione): `force_zoom_for_debug=true` + `zoom_multi_region=false`.
# Il codice bypassa `branch_selector` quando `force_zoom_for_debug=true`
# (strategies/entropy_attention_resample.py:481-518) — il valore di
# `branch_selector` (lasciato al default `concentration`) non conta in
# questo arm.
#
# Job SINGOLO (no array/sharding, a differenza di
# scripts/sbatch/video_mme-signal.sbatch da cui questo deriva):
# `nframes=24` rende l'intero dataset (2700 sample) fattibile in un solo
# job. `--time=10:00:00` è una STIMA — nessuna misura di latenza reale a
# 24 frame ancora raccolta. Calibrare con un probe corto
# (scripts/sbatch/video_mme-signal-test.sbatch, `limit=40`, stesso
# `dataset.nframes=24` + queste stesse strategy override) prima di
# fidarsi se il job va vicino al timeout.
#
# `--constraint` pinnato a 45G (A40/L40S): `full_visual_attention` (pass
# 1, richiesto da `SupportsSignals`/`model=qwen2_5_vl_3b_attn`) va OOM su
# 24G indipendentemente da `nframes` — vedi README, sezione sweep.
#
# Metriche gate d'entropia (quanti sample superano `entropy_threshold`,
# accuracy condizionata): NON in questo output — si ricavano da una query
# Weave per-sample sul campo `resampled` dopo che il run è finito, stesso
# metodo di docs/analisi_winloss_resampling_video_mme.md.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=entropy_attention_resample \
    dataset.nframes=24 \
    strategy.force_zoom_for_debug=true \
    strategy.zoom_multi_region=false \
    hf_home=/work/tesi_avalenza/hf \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-entropy_zoom_24-${SLURM_JOB_ID} \
    wandb.tags=[ablation,video_mme,entropy_zoom_24] \
    wandb.notes="ablation riunione 22-07-26: gate entropia + resample sempre zoom a singolo picco, 24 frame, slurm ${SLURM_JOB_ID} on $(hostname)" \
    "$@"
