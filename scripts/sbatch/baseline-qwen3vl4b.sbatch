#!/bin/bash
#SBATCH --job-name=baseline-qwen3vl4b
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Baseline Qwen3-VL-4B su Video-MME. Gemello di
# `scripts/sbatch/baseline-qwen3vl2b.sbatch`, che resta INVARIATO: il 2B è
# già misurato (54.44% pooled, job 92353) e il suo sbatch non va toccato.
#
# A COSA SERVE. È il punto zero di qualsiasi arm sul 4B: senza, nessun
# risultato su questo modello è interpretabile. Non testa di per sé
# nessuna ipotesi — è un prerequisito, non un esperimento.
#
# Il contesto in cui nasce (docs/260901-tabelle-risultati.md §1): sul 2B
# l'arm `attention_marker` ribalta ~11% delle risposte in ENTRAMBE le
# direzioni senza guadagno netto, mentre la variante A su Qwen2.5-VL-3B
# (modello più grande) ne rompeva lo 0.34%. Se quella fragilità è
# questione di capacità, il 4B è la leva — e serve la sua baseline.
# ⚠️ NON aspettarsi che il 4B sistemi il SEGNALE: il null "picco
# d'attenzione ≡ cella a caso" si è già replicato attraverso un aumento di
# capacità (Qwen2.5-VL-3B, 54.63% contro 54.78% del controllo random).
#
# ⚠️ PRIMA DEL PRIMO LANCIO, SUL NODO DI LOGIN. Qwen3-VL-4B non è mai
# girato su questo cluster: NON è nella cache HF di /work, e i nodi di
# calcolo non sono un posto su cui contare per scaricarlo. Senza questo
# passo il job muore al `_load`:
#
#     HF_HOME=/work/tesi_avalenza/hf uv run python -c \
#       "from huggingface_hub import snapshot_download; \
#        snapshot_download('Qwen/Qwen3-VL-4B-Instruct')"
#
# ⚠️ `--constraint` SENZA `gpu_RTX6000_24G`, a differenza del default
# storico di `baseline-qwen3vl2b.sbatch` (che quella card ce l'ha ancora).
# Su Turing (sm_75) Qwen3-VL-2B costava ~44 s/sample contro ~3.4 su L40S —
# un fattore 6 che ha già ucciso per timeout l'array 76131. SLURM stampa
# un INFO consultivo suggerendo di riaggiungere RTX6000 per ridurre
# l'attesa in coda: IGNORARLO.
# Escluse anche le 24G (`gpu_RTX_A5000_24G`): il 4B non è mai stato
# provato qui e il preset `_attn` materializza una maschera eager O(L²).
# Se la coda sulle 45G è lunga e la probe sotto passa, si possono
# riaggiungere a mano — sono Ampere, quindi il problema è la VRAM, non la
# velocità:
#
#     sbatch --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G" ...
#
# PRIMA DELLA PROD, LA PROBE DI THROUGHPUT (`LIMIT`). Per il 4B non esiste
# una latenza misurata su questo cluster, e tirare a indovinare la
# walltime si paga due volte: corta, si perde lo shard intero (l'accuracy
# è calcolata a fine eval, non c'è checkpoint); lunga, peggiora la
# priorità in coda.
#
#     LIMIT=60 sbatch --array=0 --time=01:00:00 \
#         scripts/sbatch/baseline-qwen3vl4b.sbatch
#     # walltime per shard ≈ model_latency_mean * 900 * 1.3 di margine
#
# `--time=06:00:00` è il default finché quella misura non c'è. Riferimento:
# Qwen3-VL-2B a 24 frame ha girato in 0.86 h per shard da 900 (3.1-3.4
# s/sample). Il 4B ha il doppio dei parametri, quindi ~2x per-token è la
# stima naturale (~1.7 h/shard); 6 h coprono anche il caso pessimo di un
# 4x (3.4 h/shard) senza sprecare priorità come farebbero 12 h.
#
# PRESET `_attn` ANCHE QUI benché `uniform` non usi segnali: come in
# `baseline-qwen3vl2b.sbatch` e `ablation-videomme/baseline_24.sbatch`,
# registra un kernel d'attenzione e una maschera diversi sul decoder
# testuale, quindi `generate()` percorre una strada diversa dal preset
# plain. Con il preset plain un futuro arm signal-driven differirebbe
# dalla sua baseline per DUE cose invece che per una.
#
# Per la cella a 128 frame della tabella Baseline del README serve il
# preset PLAIN e molto più tempo — non è questo il job:
#
#     MODEL=qwen3_vl_4b NFRAMES=128 sbatch --time=20:00:00 \
#         scripts/sbatch/baseline-qwen3vl4b.sbatch \
#         "wandb.tags=[baseline,video_mme,qwen3_vl_4b,baseline_128]"
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy è quella
# POOLED (Σ corretti / Σ sample), non la media delle tre.
#
# ⚠️ La colonna `Δ vs paper` del README resta VUOTA finché non si cita una
# fonte per il numero Video-MME di Qwen3-VL-4B: non va riempita a memoria.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-qwen3_vl_4b_attn}"
NFRAMES="${NFRAMES:-24}"
NAME="baseline_${NFRAMES}"

# Probe di throughput: `limit` DOPO lo shard (prepare_samples applica
# shuffle → shard → limit in quest'ordine), quindi i 60 sample escono dallo
# shard 0, non dal set intero. Gruppo wandb separato: una probe non deve
# finire nello stesso gruppo dei numeri di prod.
PROBE_ARGS=()
GROUP=video_mme-qwen3vl4b
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=video_mme-qwen3vl4b-test
  NAME="${NAME}_probe${LIMIT}"
  echo "PROBE di throughput: ${LIMIT} sample — leggi model_latency_mean, non l'accuracy"
fi

echo "model: ${MODEL}  nframes: ${NFRAMES}  shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

uv run python main.py \
    model=${MODEL} \
    dataset=video_mme \
    strategy=uniform \
    dataset.nframes=${NFRAMES} \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=${MODEL}-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[baseline,video_mme,qwen3_vl_4b,${NAME}] \
    "$@"
