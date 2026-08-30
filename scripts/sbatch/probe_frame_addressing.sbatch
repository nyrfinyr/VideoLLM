#!/bin/bash
#SBATCH --job-name=probe-frame-addressing
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# GATE del ramo ViKey (arXiv 2603.23186): Qwen3-VL-2B sa indirizzare un frame?
# Il disegno completo è nella docstring di scripts/probe_frame_addressing.py.
#
# Decide se l'SVP (numero del frame dipinto sui pixel) ha senso su Qwen3-VL,
# che a differenza dei modelli del paper i timestamp li ha GIÀ nel prompt
# (`<x.x seconds>` per blocco visivo, models/qwen.py). Da leggere PRIMA di
# scrivere qualunque arm nuovo.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch _qwen3:
# su Turing Qwen3-VL-2B va ~6x più lento.
#
# COSTO: 20 video x 2 condizioni x 4 domande = 160 forward, generazioni corte
# (8-16 token), quindi dominate dal prefill dei 24 frame. Stima ~15-20 min
# compreso il caricamento pesi; 1 h è larga.
#
# Uso:
#     sbatch scripts/sbatch/probe_frame_addressing.sbatch
#     NVIDEO=40 SEED=1 sbatch scripts/sbatch/probe_frame_addressing.sbatch
#
# ESITO DEL PRIMO GIRO (job 93544, 20 video, A5000):
#
#     cond        detect   lookup+   lookup-   bilanc.   revlook    %YES
#     native      100.0%    100.0%     25.0%     62.5%     35.0%   91.7%
#     svp          95.0%     85.0%     95.0%     90.0%     85.0%   61.7%
#
# L'SVP vince nettamente (lookup- 5/20 vs 19/20, Fisher p=1e-5; revlookup
# 7/20 vs 17/20, p=3e-3) e il ramo si apre. MA `native` non è cieco: allargando
# la tolleranza il revlookup passa da 35% (mezza cella) a 95% (una cella e
# mezza), con una SOVRASTIMA sistematica (mediana pred/vero 1.116, 18/20
# sovrastime). Un fattore di scala del ~12% spiega da solo tutto il fallimento
# di `native`. Da qui il punto 1 qui sotto.
#
# COSA GUARDARE nel .out, in quest'ordine:
#   1. il blocco `[verifica]`: quanti timestamp, i valori STAMPATI contro
#      quelli ATTESI, e la riga `stampato/atteso`. Se la mediana è ~1.12 il
#      bias del 12% è nel processor; se è 1.000 lo introduce il modello — due
#      bug diversi con due fix diversi. Se il CONTEGGIO non è 12 per un video
#      a 24 frame, la mappa cella->timestamp della sonda è sbagliata e i
#      numeri `native` sono un artefatto: si aggiusta `t_cell` e si rilancia.
#      ⚠️ nel job 93544 questa riga diceva 0 timestamp: era un bug della sonda
#      (dumpava `apply_chat_template` invece degli `input_ids` post-processor),
#      corretto — non un'assenza di timestamp.
#   2. `detect` ~100% in entrambe le condizioni — è il gate di percezione.
#   3. `bilanc.` (accuratezza bilanciata lookup+/lookup-), caso = 50%.
#   4. `%YES`: vicino a 100 con `bilanc.` ~50 = il modello ignora l'indirizzo.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

VIDEO_DIR="${VIDEO_DIR:-/work/tesi_avalenza/datasets/video_mme/videos/data}"
NVIDEO="${NVIDEO:-20}"
SEED="${SEED:-0}"

echo "video_dir: ${VIDEO_DIR}  n_video: ${NVIDEO}  seed: ${SEED}"

uv run python scripts/probe_frame_addressing.py "${VIDEO_DIR}" "${NVIDEO}" "${SEED}"
