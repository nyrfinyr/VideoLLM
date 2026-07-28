#!/bin/bash
#SBATCH --job-name=video_mme-signal-router-eval
#SBATCH --account=tesi_avalenza
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --time=09:30:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# PROD — eval Video-MME pieno (2700 QA, 128 frame/video, video-only) con il
# ROUTER a soglia adattiva (branch_selector=concentration_ratio), NON lo
# sweep phase_shift-only di video_mme-signal.sbatch (quello usa
# concentration_threshold=30 assoluto, inerte su Video-MME — vedi
# docs/piano_zoom_multiregione_disgiunto.md).
#
# NON LANCIARE finché lo Stage C (validazione sul subset 250,
# `qwen2_5_vl_3b_attn-stageC-router-concentration_ratio`) non conferma il
# guadagno atteso (~68-69%, vedi docs/sintesi_segnali_router_concentrazione.md
# §7) — se il router non regge fuori dal subset di calibrazione, questo run
# va rivisto prima di spendere le ~9h/shard.
#
# Config (identica al run Stage C validato sul subset, vedi
# docs/piano_zoom_multiregione_disgiunto.md, sezione "Soglia adattiva su
# top1_pct"):
# - branch_selector=concentration_ratio, concentration_ratio=3.85 (soglia
#   adattiva top1_pct < 3.85*100/t, trasferibile — non concentration_threshold
#   assoluto, morto su questo dataset).
# - zoom_multi_region=true (il ramo concentrato usa fino a n_regions=3
#   finestre disgiunte, non il singolo zoom_peak — Opzione C1).
# - entropy_threshold=0.7 (gate invariato, stesso di tutti gli sweep finora).
#
# `--time` stimato dai costi REALI misurati sul subset 250 (non un numero
# indovinato): fase-sfasata/accettato ~27.6s/sample (o54z2xsd/ujdksy60),
# zoom multi-regione ~46.5s/sample (derivato da z4qx9hnb, 39.92s/sample
# medio su 87 accettati + 163 zoom forzato). Con lo split atteso dal router
# a ratio=3.85 sul subset (~87 accettati, ~82 fase, ~81 zoom, /250) la
# media pesata è ~33.7s/sample → 675 sample/shard ≈ 6.3h di calcolo puro,
# --time=09:30:00 dà ~1.5x di margine (stesso criterio di
# video_mme-signal.sbatch/lvbench.sbatch). Split REALE del full eval potrebbe
# differire da quello calibrato sul subset — se gli shard vanno in timeout,
# ricalcolare da `model_latency.mean` dello Stage C prima di rilanciare.
#
# `--partition=boost_usr_prod` + `--constraint` SOLE 45G (A40/L40S): stesso
# vincolo di tutti i run precedenti con questa strategy — full_visual_attention
# (cattura attenzione non-flash, pass 1) va OOM su schede 24G a nframes=128,
# confermato più volte (vedi docs/piano_zoom_multiregione_disgiunto.md).
#
# Override eventuali da CLI dopo lo script, es. per cambiare soglia:
#   sbatch scripts/sbatch/video_mme-signal-router.sbatch strategy.concentration_ratio=4.0
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
    strategy.branch_selector=concentration_ratio \
    strategy.concentration_ratio=3.85 \
    strategy.zoom_multi_region=true \
    strategy.entropy_threshold=0.7 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.tags=[eval,video_mme,sbatch,array,${STRATEGY},router] \
    wandb.group=${MODEL}-${STRATEGY}-router-video_mme \
    wandb.name=${MODEL}-${STRATEGY}-router-video_mme-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    "$@"
