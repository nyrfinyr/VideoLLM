#!/bin/bash
#SBATCH --job-name=ablation-highlight_top1_24_qwen3
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

# Variante A (`attention_highlight`, puntatore testuale FUORI dal flusso
# visivo) su Qwen3-VL-2B. Gemello qwen3 di `highlight_top1_24.sbatch`, che
# resta INVARIATO perché è cablato su Qwen2.5-VL-3B (modello, nome della run,
# e `gpu_RTX6000_24G` nel constraint).
#
# A COSA SERVE — deconfondere variante da modello.
# Sul 2B l'arm `attention_marker` (variante B, marcatori INLINE) rompe
# l'11.1% dei sample corretti al pass 1 e ne recupera il 9.8% fra gli
# sbagliati: churn ~10% in entrambe le direzioni, netto quasi zero, cioè
# rumore e non informazione (docs/260901-tabelle-risultati.md §1).
# Sul Qwen2.5-VL-3B la variante A rompeva invece lo 0.34% col puntatore
# innocuo e il 6.80% con quello ANTIPEAK avversariale
# (docs/sonda_sensibilita_puntatore_antipeak.md).
# Quel confronto 11% vs 0.34% è incrociato su DUE assi insieme — variante
# (inline vs prompt) e modello (2B vs 3B) — quindi non conclude nulla.
# Questa run tiene fisso il modello e muove solo la variante:
#
#   churn ~0.3%  → la colpa è dei MARCATORI INLINE: il canale-prompt regge
#                  anche su Qwen3, ed è il substrato su cui testare segnali
#                  migliori (righe-entità);
#   churn ~11%   → la colpa è del SECONDO PASSO su Qwen3: tutta la famiglia
#                  a due passi è fragile su questo modello, variante A e
#                  `visual_prompt` incluse, e la leva diventa la capacità
#                  (baseline 4B, `scripts/sbatch/baseline-qwen3vl4b.sbatch`).
#
# LA LETTURA È `correct_pass1_true`, NON l'accuracy pooled. La strategy
# emette `pred_pass1` (`attention_highlight.py:653`), da cui
# `evals/base.py::mcq_accuracy` ricava il breakdown `correct_pass1_*`. Il
# numero da confrontare con l'11.1% è `1 - mcq_accuracy_correct_pass1_true`.
#
# ⚠️ `strategy.always_highlight=true`, DIVERSO dal default `false` di
# `highlight_top1_24.sbatch`. Col gate d'entropia acceso si interviene solo
# sul ~73% dei sample, e il ~27% non toccato conserva per costruzione la
# risposta del pass 1: il churn osservato uscirebbe diluito di quel fattore
# e NON sarebbe confrontabile con l'11.1% del marker arm, misurato al 100%
# di copertura. Il gate va tenuto aperto perché le due misure siano appaiate.
#
# ⚠️ `--constraint` SENZA `gpu_RTX6000_24G`. Su Turing (sm_75) Qwen3-VL-2B
# costa ~44 s/sample contro ~3.4 su L40S — un fattore 6 che ha già ucciso
# per timeout l'array 76131. SLURM stampa un INFO consultivo suggerendo di
# riaggiungerla: IGNORARLO. Le altre card del constraint sono equivalenti
# fra loro su questo carico (misurato: A5000 10.69 s/sample contro L40S
# 10.70 sul marker arm), quindi non c'è una "card lenta" da coprire.
#
# BASTA UNO SHARD. Separare 0.3% da 11% non richiede 2700 sample: su 900 i
# corretti-al-pass-1 sono ~500, e i due esiti attesi valgono ~2 sample
# contro ~55. Il fullset a 3 shard serve solo se poi si vuole l'accuracy.
#
#     sbatch --array=0 scripts/sbatch/ablation-videomme/highlight_top1_24_qwen3.sbatch
#
# `--time=04:00:00`: non c'è una latenza misurata per questo arm su Qwen3.
# I termini noti sono `entropy_shift_24` a 8.97 s/sample (Qwen3-2B, ma solo
# 53.7% di sample al pass 2) e `marker_24` a 10.7 (100% di copertura, con in
# più l'estrazione PNG di tutti i frame che qui NON c'è). Con gate aperto e
# senza PNG la stima sta sotto i 10 s/sample → ~2.5 h/shard, e 4 h danno
# ~1.6x. RITARARE da `model_latency_mean` prima di un eventuale fullset.
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

NAME="highlight_top1_24_alwayson"

# Probe: `limit` DOPO lo shard (prepare_samples applica shuffle → shard →
# limit), quindi i sample escono dallo shard 0, non dal set intero. Gruppo
# wandb separato: una probe non deve finire fra i numeri di prod.
PROBE_ARGS=()
GROUP=video_mme-qwen3vl2b-ablation
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=video_mme-qwen3vl2b-ablation-test
  NAME="${NAME}_probe${LIMIT}"
  echo "PROBE: ${LIMIT} sample — su 60 i corretti-al-pass-1 sono ~35, troppo pochi per il churn"
fi

echo "shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    strategy.top_k=1 \
    strategy.always_highlight=true \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,qwen3_vl_2b,highlight_top1_24_alwayson] \
    "$@"
