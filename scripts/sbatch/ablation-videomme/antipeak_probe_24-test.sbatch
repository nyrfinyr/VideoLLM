#!/bin/bash
#SBATCH --job-name=probe-antipeak_24-test
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# SMOKE TEST + THROUGHPUT PROBE della sonda antipeak
# (`antipeak_probe_24.sbatch`, razionale in
# `docs/sonda_sensibilita_puntatore_antipeak.md`).
#
# Serve a due cose, entrambe non verificabili in locale:
#
# 1. CODE PATH SU PESI INTERI. Lo smoke test locale è girato con i layer
#    troncati (`num_text_layers=8`, GPU consumer da 6 GiB): valida il path
#    ma l'output è casuale, quindi `generate()` non produce mai una lettera
#    parseabile e `pred_fallback` scatta sempre. Qui i pesi sono interi:
#    è il primo posto dove si vede il comportamento vero.
#
# 2. THROUGHPUT REALE. Il `--time=04:00:00` della sonda vera è STIMATO dai
#    14.6 s/sample del probe `2dodrndg`, non misurato con questa strategy.
#    Da qui esce `model_latency.mean` per dimensionarlo davvero:
#
#        --time necessario ≈ model_latency.mean × 500 × 1.5 (margine)
#
#    Se `model_latency.mean` supera ~19 s/sample, 4h non bastano per i 500
#    sample e va alzato PRIMA di lanciare l'array.
#
# QUALE ARM. Default `antipeak` (è quello che esercita il codice nuovo).
# Per provare anche il controllo:
#     ARM=nopointer sbatch scripts/sbatch/ablation-videomme/antipeak_probe_24-test.sbatch
#
# SUBSET NIDIFICATO: `shuffle=true limit=40` con lo stesso `seed=42` della
# sonda vera, e `limit` prende un PREFISSO dell'ordine mescolato
# (`utils/samples.py`, shuffle → shard → limit) — quindi questi 40 sample
# sono un sottoinsieme esatto dei 500 della sonda. Utile: quello che si
# vede qui non è un campione diverso, è l'inizio di quello vero.
#
# COSA CONTROLLARE NEL LOG (`grep -A20 'Breakdown per categoria' logs/...`):
#   - `highlighted_true` = 40/40 con ARM=antipeak (always_highlight=true →
#     puntatore su TUTTI i sample; se è meno, il gate non è stato
#     neutralizzato) e 0/40 con ARM=nopointer.
#   - `correct_pass1_true` / `correct_pass1_false` PRESENTI. Se mancano del
#     tutto, `pred_fallback` sta scattando su tutti i sample e il breakdown
#     li esclude apposta (vedi `mcq_accuracy` in evals/base.py) — su pesi
#     interi NON deve succedere: su highlight_top1_24 il fallback era 0/900.
#   - `pred_fallback_true` ≈ 0. Se è alto, il puntatore sta degradando la
#     PARSABILITÀ dell'output invece della risposta: effetto diverso, da
#     capire prima di lanciare la sonda vera.
#
# NB: con 40 sample il NUMERO della sonda non è interpretabile (né qui né
# altrove a questa N) — questo job dice "il meccanismo gira e costa X",
# non "il puntatore fa/non fa danno".
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"; echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
ARM="${ARM:-antipeak}"

case "${ARM}" in
  antipeak)
    ARM_OVERRIDES=(
        strategy.cell_select=antipeak
        strategy.always_highlight=true
    )
    ;;
  nopointer)
    ARM_OVERRIDES=(
        strategy.cell_select=attention
        strategy.always_highlight=false
        strategy.entropy_threshold=999
    )
    ;;
  *)
    echo "ARM sconosciuto: ${ARM}. Valide: antipeak, nopointer" >&2
    exit 1
    ;;
esac
echo "arm: ${ARM}"

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    strategy.top_k=1 \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    "${ARM_OVERRIDES[@]}" \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=40 \
    wandb.group=video_mme-antipeak-probe-test \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}_24-test-${SLURM_JOB_ID} \
    wandb.tags=[probe,video_mme,test,antipeak,${ARM}] \
    "$@"
