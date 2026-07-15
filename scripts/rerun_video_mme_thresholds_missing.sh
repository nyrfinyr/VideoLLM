#!/bin/bash
# Rilancia le 3 combinazioni di `scripts/sweep_video_mme_thresholds.sh` che
# sono andate `crashed` su wandb senza loggare nessuna metrica: et0.7/ct20,
# et0.5/ct45, et0.5/ct30 (tutte finite sull'host `huber`, GPU RTX A5000
# 24G). Durata osservata ~2h02-2h13m, oltre lo SWEEP_TIME=02:00:00 dello
# sweep originale — le latenze medie delle combinazioni completate
# (~23-27 s/sample × 250 ⇒ ~1.6-1.9h) erano già ai limiti di quel budget su
# GPU più lente della L40S usata per calibrarlo. Qui il default sale a 3h.
#
# Stesso subset dello sweep originale (limit=250, shuffle=true, seed=42
# invariato, vedi utils/samples.py) e stesso wandb.group, così i run
# rilanciati si aggregano alla tabella del README senza bisogno di
# ricalcolare nulla.
#
# DRY_RUN=1 stampa i comandi `sbatch` senza sottometterli:
#   DRY_RUN=1 ./scripts/rerun_video_mme_thresholds_missing.sh
#
# Per escludere l'host/GPU dove sono avvenuti i crash, aggiungi un
# constraint più stretto via override, es.:
#   sbatch --time=03:00:00 --constraint="gpu_A40_45G|gpu_L40S_45G" \
#       scripts/sbatch/video_mme-signal-test.sbatch ...
set -euo pipefail

LIMIT=250
MODEL="${MODEL:-qwen2_5_vl_3b_attn}"
GROUP="${MODEL}-sweep-video_mme-test"
RERUN_TIME="${RERUN_TIME:-03:00:00}"

run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: $*"
    else
        "$@"
    fi
}

echo "rerun group: ${GROUP} (limit=${LIMIT}, model=${MODEL}, time=${RERUN_TIME})"

for combo in "0.7:20" "0.5:45" "0.5:30"; do
    et="${combo%%:*}"
    ct="${combo##*:}"
    run env MODEL="${MODEL}" sbatch --time="${RERUN_TIME}" \
        scripts/sbatch/video_mme-signal-test.sbatch \
        limit=${LIMIT} \
        strategy.entropy_threshold=${et} \
        strategy.concentration_threshold=${ct} \
        wandb.group=${GROUP} \
        wandb.name=${MODEL}-entropy_attention_resample-et${et}-ct${ct} \
        wandb.tags=[sweep,video_mme,entropy_attention_resample]
done
