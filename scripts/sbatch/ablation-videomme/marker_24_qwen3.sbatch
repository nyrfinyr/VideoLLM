#!/bin/bash
#SBATCH --job-name=ablation-marker_24_qwen3
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

# Arm `attention_marker` su Qwen3-VL-2B — marcatori inline
# <START_IMPORTANT_FRAME>/<END_IMPORTANT_FRAME> attorno alla cella di picco,
# dentro il flusso visivo (variante "B" di attention_highlight, solo Qwen3).
# Vedi strategies/attention_marker.py e docs/piano_marcatori_frame_qwen3.md.
#
# Preset: always_highlight=true (nessun gate), top_k=1, query_rows=question,
# sink_filter=false (attenzione GREZZA). Il confronto da leggere è INTERNO
# al probe: cell_select=attention contro cell_select=random a parità di
# tutto il resto — l'arm di controllo si lancia con:
#
#     sbatch scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch \
#         strategy.cell_select=random "wandb.tags=[ablation,video_mme,qwen3_vl_2b,marker_random_24]"
#
# ⚠️ `--constraint` SENZA `gpu_RTX6000_24G`. Su Turing (sm_75) Qwen3-VL-2B
# costa ~44 s/sample contro i ~3.4 su L40S — un fattore 6 che ha già ucciso
# per timeout l'array 76131. SLURM stampa un INFO consultivo suggerendo di
# riaggiungere RTX6000 per ridurre l'attesa in coda: IGNORARLO.
#
# COSTO ATTESO (stima, NON misura): a differenza di attention_highlight qui
# ogni sample richiede estrazione PNG di tutti i 24 frame + secondo pass, e
# con always_highlight=true nessun sample ne è esente. Sul sovraccosto
# misurato il 2026-08-28 (10.5 s per sample intervenuto,
# docs/260901-tabelle-risultati.md §2) si va verso ~14 s/sample: ~15 min per
# una probe da 60, ~3.5 h/shard per un fullset a 3 shard — FUORI dalla
# walltime a 4 h con poco margine. RITARARE su model_latency_mean della
# probe prima di lanciare il fullset (più shard o più walltime), non lanciare
# alla cieca su questa stima.
#
# PROBE (piano §6 — i primi due check sono correttezza, non risultato):
#
#     LIMIT=60 sbatch --array=0 --time=00:40:00 \
#         scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch
#
#   1. n_query_rows < n_query_tokens su TUTTI i sample (se sono uguali la
#      strategy solleva da sé: nessun sample deve fallire per questo);
#   2. sink_filter davvero spento: `peak_cell` vs `peak_cell_sink_filtered`
#      NON devono coincidere sempre;
#   3. somma(block_sizes)/2 == t_cells su ogni sample marcato;
#   4. `mcq_accuracy` esiste nel summary (un OOM di full_visual_attention
#      viene catturato da Weave per sample e lascia il job `finished` con 0
#      predizioni valide: lo stato del job NON è un check);
#   5. pred_fallback ~0;
#   6. model_latency_mean, per ritarare la walltime;
#   7. solo dopo: l'accuracy, e solo contro `random` a parità del resto.
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

NAME="marker_24"

# Probe: `limit` DOPO lo shard (prepare_samples applica shuffle → shard →
# limit), quindi i sample escono dallo shard 0, non dal set intero. Gruppo
# wandb separato: una probe non deve finire fra i numeri di prod.
PROBE_ARGS=()
GROUP=video_mme-qwen3vl2b-ablation
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=video_mme-qwen3vl2b-ablation-test
  NAME="${NAME}_probe${LIMIT}"
  echo "PROBE: ${LIMIT} sample — verifica marcatori/metadata/query_rows su Qwen3, non l'accuracy"
fi

echo "shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    dataset=video_mme \
    strategy=attention_marker \
    dataset.nframes=24 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,qwen3_vl_2b,marker_24] \
    "$@"
