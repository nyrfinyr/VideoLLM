#!/bin/bash
#SBATCH --job-name=baseline-qwen3vl2b
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Prima baseline Qwen3-VL-2B su Video-MME. Riempie una colonna che nel
# README è vuota su tutti e quattro i benchmark.
#
# ⚠️ PRIMA DEL PRIMO LANCIO, SUL NODO DI LOGIN. Qwen3-VL-2B non è mai
# girato su questo cluster, quindi NON è nella cache HF di /work e i nodi
# di calcolo non sono un posto su cui contare per scaricarlo. Senza questo
# passo il job muore al `_load`:
#
#     HF_HOME=/work/tesi_avalenza/hf uv run python -c \
#       "from huggingface_hub import snapshot_download; \
#        snapshot_download('Qwen/Qwen3-VL-2B-Instruct')"
#
# DUE RUN, DUE SCOPI DIVERSI — si scelgono da ambiente:
#
#   1. NFRAMES=24, preset `_attn`  (DEFAULT)
#      Confrontabile con tutta la linea ablation di `ablation-videomme/`,
#      il cui riferimento è `baseline_24` = 55.04% (Qwen2.5-VL-3B, 2700).
#      È la run che alimenta la decisione: se il filone del puntatore
#      d'attenzione si sposta su Qwen3-VL (grounding temporale più forte,
#      timestamp in-band), questo è il suo punto zero.
#      Il preset `_attn` anche qui benché `uniform` non usi segnali: come
#      in `ablation-videomme/baseline_24.sbatch`, registra un kernel
#      d'attenzione e una maschera diversi sul decoder testuale, quindi
#      `generate()` percorre una strada diversa dal preset plain. Con il
#      preset plain un futuro arm signal-driven differirebbe dalla sua
#      baseline per DUE cose invece che per una.
#
#   2. NFRAMES=128, preset plain
#      Riempie la cella della tabella Baseline del README, confrontabile
#      col 59.93% di Qwen2.5-VL-3B (run p7srmrkl) e coi numeri di paper.
#      A 128 frame Qwen2.5 andava OOM sulle 24G (vedi la nota OOM del
#      README), e Qwen3 allunga ulteriormente la sequenza scrivendo un
#      token di timestamp per frame: si pinna alle 45G e si allarga il
#      tempo.
#
#          MODEL=qwen3_vl_2b NFRAMES=128 sbatch \
#              --constraint="gpu_A40_45G|gpu_L40S_45G" --time=16:00:00 \
#              scripts/sbatch/baseline-qwen3vl2b.sbatch
#
# PRIMA DELLA PROD, LA PROBE DI THROUGHPUT (`LIMIT`). Non esiste una
# latenza di riferimento per Qwen3-VL su questo cluster, e tirare a
# indovinare la walltime si paga due volte: corta, si perde lo shard
# intero (l'accuracy è calcolata a fine eval, non c'è checkpoint); lunga,
# peggiora la priorità in coda. `LIMIT=60` gira un array a shard singolo
# su 60 sample e stampa `model_latency_mean`, da cui la walltime esce
# misurata invece che stimata:
#
#     LIMIT=60 sbatch --array=0 scripts/sbatch/baseline-qwen3vl2b.sbatch
#     # walltime per shard ≈ model_latency_mean * 900 * 1.3 di margine
#
# `--time=04:00:00` è il default finché quella misura non c'è. Il termine
# di paragone è Qwen2.5-VL-3B a 24 frame: 7.19 s/sample sulla card più
# lenta del constraint → ~1h48 per shard da 900 (`baseline_24.sbatch`
# chiede 3h per lo stesso carico). Qwen3-VL-2B ha MENO parametri, quindi
# per-token è verosimilmente più veloce; in senso opposto inietta feature
# visive nei primi layer (deepstack) e allunga il prompt di ~150 token su
# ~2250 (un `<x.x seconds>` per frame, cioè +7%). Nessuno dei due è un
# moltiplicatore: 4h sono 2.2x il riferimento e coprono anche il caso
# pessimo di un modello 2x più lento (14 s/sample → 3h30).
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy è quella
# POOLED (Σ corretti / Σ sample), non la media delle tre.
#
# ⚠️ La colonna `Δ vs paper` del README resta VUOTA finché non si cita una
# fonte per il numero Video-MME di Qwen3-VL-2B: non va riempita a memoria.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-qwen3_vl_2b_attn}"
NFRAMES="${NFRAMES:-24}"
NAME="baseline_${NFRAMES}"

# Probe di throughput: `limit` DOPO lo shard (prepare_samples applica
# shuffle → shard → limit in quest'ordine), quindi i 60 sample escono dallo
# shard 0, non dal set intero. Gruppo wandb separato: una probe non deve
# finire nello stesso gruppo dei numeri di prod.
PROBE_ARGS=()
GROUP=video_mme-qwen3vl2b
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=video_mme-qwen3vl2b-test
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
    wandb.tags=[baseline,video_mme,qwen3_vl_2b,${NAME}] \
    "$@"
