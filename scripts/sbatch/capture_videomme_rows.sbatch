#!/bin/bash
#SBATCH --job-name=capture-videomme-rows
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Capture per la DIAGNOSTICA DELLE RIGHE-QUERY. Non è una eval: nessuna
# accuracy, nessun wandb. Produce solo uno STORE su disco su cui si
# ricalcolano offline, a costo zero di GPU, le statistiche di bias
# posizionale del picco sotto insiemi di righe diversi.
#
# IL PROBLEMA CHE INDAGA. `_capture_spans` (models/qwen_attn.py:246)
# definisce le righe-query come tutto ciò che segue l'ultimo
# `<|vision_end|>`, e `ranked_cells_from_attention`
# (utils/attn_core.py:594) fa `attn.mean(dim=0)`: media su TUTTE quelle
# righe. Su un prompt Video-MME quello vuol dire domanda + le quattro
# opzioni + "Answer with the letter of the correct option only." + il
# generation prompt. Misurato col tokenizer su un prompt tipico:
#     prompt intero   86 token   <- righe mediate oggi
#     solo domanda    21 token   <- 24%
# Cioè tre quarti delle righe che formano il "picco d'attenzione della
# domanda" non sono la domanda. `question_rows` (utils/attn_core.py:75)
# taglia esattamente quel resto, esiste già, la usa la UI dell'attn_explorer
# e le strategy NON la chiamano.
#
# COSA SI CONFRONTA, sullo stesso dump e quindi a parità di tutto il resto:
#     ALL               righe attuali (domanda+opzioni+boilerplate+gen)
#     question+options  tolto solo il boilerplate
#     question          solo la domanda (`question_rows`)
#     entity            i token dell'entità, scelti A MANO = oracolo del
#                       riconoscimento automatico, cioè il tetto massimo
#                       che un estrattore potrà raggiungere
# Le metriche sono quelle già note per ALL dal probe da 40 sample
# (run 2dodrndg), quindi il confronto è diretto: quota di picchi
# sull'ultima cella (28% oggi, caso 8.3%), su una cella estrema (40%,
# atteso 17%), `top1_pct` mediano (15.7%, caso 8.3%).
#
# PERCHÉ SUL CLUSTER E NON IN LOCALE. Il dump Video-MME locale ha 10
# sample: per distinguere il 28% dall'8.3% servono decine di video, non
# dieci. Qui c'è il dump intero. La capture è un solo forward di PREFILL
# per video (niente generate), quindi 60 video costano pochi minuti di
# GPU — il grosso del tempo è caricare il modello e decodificare i video.
#
# ⚠️ `--nframes 24` NON è il default del preset Video-MME (128): deve
# corrispondere al regime degli arm dell'ablation, altrimenti la griglia
# ha un numero di celle diverso e il livello di caso non è più 1/12.
#
# La `max_pixels` viene dal preset (151200), come nelle eval: cambiarla
# cambierebbe `grid_h`/`grid_w` e quindi le masse per cella.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

N="${N:-60}"
OUTDIR="${OUTDIR:-/work/tesi_avalenza/stores/videomme_rows_24}"
mkdir -p "$(dirname "${OUTDIR}")"

uv run python -m attn_explorer.capture \
    --dataset video_mme \
    --n "${N}" \
    --nframes 24 \
    --model qwen2_5_vl_3b_attn \
    --outdir "${OUTDIR}" \
    --hf-home /work/tesi_avalenza/hf \
    "$@"

echo "store pronto: ${OUTDIR}"
du -sh "${OUTDIR}" || true
