#!/bin/bash
# Sweep coarse delle soglie di gating di `entropy_attention_resample` su un
# subset di Video-MME — non un nuovo sbatch, un LANCIATORE che riusa
# `scripts/sbatch/video_mme-signal-test.sbatch` via override CLI (il suo
# "$@" li inoltra a main.py) per ogni combinazione della grid, più UN run
# di controllo `strategy=uniform` sullo STESSO subset.
#
# Perché un subset più grande del probe di throughput (40): con 40 sample
# l'errore standard sull'accuracy è troppo alto per distinguere le soglie
# (vedi utils/samples.py: "limit attivo → metriche di accuracy non
# significative"). LIMIT=250 sotto è una scelta esplicita per QUESTO sweep,
# non il default del probe.
#
# Perché la baseline uniform qui e non il 59.93% del README: quel numero è
# su tutti i 2700 sample — confrontare contro un subset di 250 introduce
# varianza campionaria diversa. Shuffle deterministico (stesso `seed`,
# default 42, mai overridato qui) → uniform e OGNI combinazione della grid
# vedono ESATTAMENTE lo stesso subset (vedi utils/samples.py:
# `random.Random(cfg.seed).shuffle(...)`), confronto mele-con-mele.
#
# window_frac/phase_shift_frac/sink_percentile/sink_border restano ai
# default di conf/config.yaml in questo primo giro (solo le due soglie di
# gating, entropy_threshold e concentration_threshold, sono sweepate) — si
# affinano gli altri in un secondo round, solo sulla coppia vincente qui.
#
# `--time` per ogni submit è STIMATO (nessuna misura reale ancora per
# entropy_attention_resample su Video-MME), volutamente generoso: la
# baseline uniform a 128 frame misura ~12.4s/sample su L40S ma fino a ~3x
# su RTX6000-24G (docs/probe_throughput.md) — entropy_attention_resample fa
# SEMPRE un prefill extra per sample (vedi strategies/entropy_attention_
# resample.py, video_mme-signal-test.sbatch), quindi si stima ~1.5-2.5x la
# baseline. Se un job va comunque in timeout, ALZARE SWEEP_TIME/CONTROL_TIME
# sotto e rilanciare solo le combinazioni mancanti (i job sono indipendenti).
#
# DRY_RUN=1 stampa i comandi `sbatch` senza sottometterli (utile per
# controllare la grid prima di spendere GPU-ore):
#   DRY_RUN=1 ./scripts/sweep_video_mme_thresholds.sh
set -euo pipefail

LIMIT=250
MODEL="${MODEL:-qwen2_5_vl_3b_attn}"
GROUP="${MODEL}-sweep-video_mme-test"
SWEEP_TIME="${SWEEP_TIME:-08:00:00}"    # entropy_attention_resample: pass1 + generate(), per sample
CONTROL_TIME="${CONTROL_TIME:-03:00:00}" # uniform: singola generate(), più economico

run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: $*"
    else
        "$@"
    fi
}

echo "sweep group: ${GROUP} (limit=${LIMIT}, model=${MODEL})"

# Baseline di controllo: uniform sullo STESSO subset (stesso seed/shuffle).
run env MODEL="${MODEL}" STRATEGY=uniform sbatch --time="${CONTROL_TIME}" \
    scripts/sbatch/video_mme-signal-test.sbatch \
    limit=${LIMIT} \
    wandb.group=${GROUP} \
    wandb.name=${MODEL}-uniform-control \
    wandb.tags=[sweep,video_mme,uniform,control]

for et in 0.3 0.5 0.7 1.0; do
    for ct in 20 30 45; do
        run env MODEL="${MODEL}" sbatch --time="${SWEEP_TIME}" \
            scripts/sbatch/video_mme-signal-test.sbatch \
            limit=${LIMIT} \
            strategy.entropy_threshold=${et} \
            strategy.concentration_threshold=${ct} \
            wandb.group=${GROUP} \
            wandb.name=${MODEL}-entropy_attention_resample-et${et}-ct${ct} \
            wandb.tags=[sweep,video_mme,entropy_attention_resample]
    done
done

echo "Sottomessi $(( 4 * 3 + 1 )) job (12 combinazioni + 1 controllo uniform) nel group ${GROUP}."
echo "In Weave (progetto video_mme): confronta l'accuracy aggregata (output.correct) fra i run del group per scegliere la coppia (entropy_threshold, concentration_threshold) migliore."
