#!/bin/bash
#SBATCH --job-name=ablation-entropy_shift_24_qwen3
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Arm `entropy_shift` su Qwen3-VL-2B — gemello di `entropy_shift_24.sbatch`
# (Qwen2.5-VL-3B), stessi knob, stesso dataset, stesso nframes. Riferimento:
# baseline Qwen3-VL-2B 54.44% pooled (job 92353, docs/qwen3_vl_2b_baseline.md).
#
# Gate d'entropia INVARIATO (`entropy_threshold=0.7`, default), ma ogni
# sample che lo supera va nel ramo `phase_shift`: `concentration_threshold=101`
# rende `dispersed` sempre vero (`top1_pct` è una percentuale, max 100).
#
# ⚠️ `--constraint` SENZA `gpu_RTX6000_24G`. Su Turing (sm_75) Qwen3-VL-2B
# costa ~44 s/sample contro i ~3.4 su L40S — un fattore 6 che ha già ucciso
# per timeout l'array 76131. SLURM stampa un INFO consultivo suggerendo di
# riaggiungere RTX6000 per ridurre l'attesa in coda: IGNORARLO.
#
# PROBE GIÀ FATTA (job 92375, run `ff5e15rr`, 60 sample su A40 — il primo
# giro di `generate_with_signals`/`full_visual_attention` su Qwen3-VL-2B, che
# fino a quel momento non era mai stato eseguito in un job vero). Esito:
#   mcq_accuracy 0.60 (36/60)   → il forward con cattura NON esplode su Qwen3
#   n_samples_resampled_true 40 → il gate si apre sul 66.7% dei sample
#   n_samples_resample_kind_phase_shift 40 = resampled_true → ramo corretto
#   pred_fallback 0/60          → tutte le risposte parseabili
#   model_latency_mean 10.73 s
#
# `--time=04:00:00` è calibrato su QUELLA misura, non su una stima: 900
# sample × 10.73 s = 2h41, margine 1.49× (lo stesso dell'arm Qwen2.5).
# ⚠️ Il costo del gate + secondo pass su Qwen3 è **3.04×** la baseline
# uniform (10.73 contro 3.53 s/sample sugli STESSI 60 sample), non l'1.87×
# misurato su Qwen2.5: l'overhead fisso della cattura d'attenzione pesa di
# più quando il forward di base costa la metà. Non ridurre la walltime
# assumendo il fattore di Qwen2.5.
#
# Per rifare la probe (es. dopo un cambio al path signal-driven):
#
#     LIMIT=60 sbatch --array=0 --time=00:40:00 \
#         scripts/sbatch/ablation-videomme/entropy_shift_24_qwen3.sbatch
#
# Un OOM di `full_visual_attention` NON si vede dallo stato del job: Weave
# cattura l'eccezione per sample e il run risulta `finished` con 0 predizioni
# valide. Il check è sempre "`mcq_accuracy` esiste nel summary".
#
# Il ramo `phase_shift` estrae PNG espliciti su disco (tmpdir per sample,
# ripulita dopo la generate) invece di traslare la finestra di
# campionamento: su una finestra che copre già tutto il video — sempre, su
# Video-MME — la traslazione veniva clampata e lo sfasamento decadeva a zero
# sull'ultimo frame. Vedi `_build_phase_shifted_media` in
# strategies/entropy_attention_resample.py. Budget del pass 2: `nframes-1`.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy dell'arm è
# quella POOLED (Σ corretti / Σ sample), non la media delle tre.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

NAME="entropy_shift_24"

# Probe: `limit` DOPO lo shard (prepare_samples applica shuffle → shard →
# limit), quindi i sample escono dallo shard 0, non dal set intero. Gruppo
# wandb separato: una probe non deve finire fra i numeri di prod.
PROBE_ARGS=()
GROUP=video_mme-qwen3vl2b-ablation
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=video_mme-qwen3vl2b-ablation-test
  NAME="${NAME}_probe${LIMIT}"
  echo "PROBE: ${LIMIT} sample — verifica il path signal-driven su Qwen3, non l'accuracy"
fi

echo "shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    dataset=video_mme \
    strategy=entropy_attention_resample \
    dataset.nframes=24 \
    strategy.branch_selector=concentration \
    strategy.concentration_threshold=101 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,qwen3_vl_2b,entropy_shift_24] \
    "$@"
