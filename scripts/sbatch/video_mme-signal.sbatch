#!/bin/bash
#SBATCH --job-name=video_mme-signal-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:01:00
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
# `--time=00:01:00` sopra è un PLACEHOLDER VOLUTAMENTE ROTTO (fail-fast in
# 1 minuto) — vanno editate le due righe SBATCH --time in testa a questo
# file PRIMA di lanciare in prod:
#   1. gira video_mme-signal-test.sbatch (con la STESSA strategy che userai
#      qui) e leggi `model_latency.mean` da Weave;
#   2. --time ≈ 2700 × model_latency.mean (stesso metodo di
#      docs/probe_throughput.md), con margine.
# Nessun numero indovinato qui: `entropy_attention_resample` fa SEMPRE un
# prefill extra (`generate_with_signals`, pass 1) PRIMA della vera
# `generate()`, per OGNI sample a prescindere dal ricampionamento — non è
# "il doppio solo se ricampiona", è un costo aggiuntivo per-sample
# grosso modo costante (vedi video_mme-signal-test.sbatch). La baseline
# uniform a 128 frame misura già ~9.3h su L40S ma ~25h su RTX6000-24G
# (docs/probe_throughput.md) — qui il constraint resta invariato (24G+45G,
# nessun pin di card più veloci), quindi il numero dal probe va preso sul
# serio prima di editare --time.
#
# NON LANCIARE finché le soglie di entropy_attention_resample
# (entropy_threshold/concentration_threshold/window_frac/phase_shift_frac,
# conf/config.yaml sotto strategy.entropy_attention_resample — tarate su
# VNBench "retrieval", dominio/nframes diversi da Video-MME) non sono state
# validate su un subset con `video_mme-signal-test.sbatch`: senza tuning i
# numeri di questo run pieno non sono comparabili con la baseline.
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
