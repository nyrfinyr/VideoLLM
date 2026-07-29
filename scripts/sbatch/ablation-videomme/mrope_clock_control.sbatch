#!/bin/bash
#SBATCH --job-name=mrope-clock-control
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# CONTROLLO DELL'OROLOGIO M-RoPE — quanto è costato, e cosa cambia a
# ripararlo, il collasso della posizione temporale dei token visivi.
#
# IL DIFETTO. `Qwen._prepare_inputs` non passava al processor i
# `video_metadata` reali, quindi `second_per_grid_ts` vale
# `temporal_patch_size / sampled_fps = 2/24 = 0.083` per QUALUNQUE video, e
# `get_rope_index` (`modeling_qwen2_5_vl.py:1125`) fa
# `time_interval = tokens_per_second * int(0.083) = 0`. In
# `get_vision_position_ids` questo diventa `position_temporal =
# arange(t) * 0`: TUTTE le celle video finiscono sulla stessa posizione
# temporale M-RoPE. Misurato su Video-MME nframes=24:
#     pass_video_metadata=false → second_per_grid_ts=0.0833 → time_interval=0
#     pass_video_metadata=true  → second_per_grid_ts=6.1923 → time_interval=12
# Non è condizionale alla durata del video (l'ipotesi precedente, "solo sotto
# 12 s"): 2/24 è una costante, quindi il collasso è UNIVERSALE. Vale per
# tutte le run del README, baseline comprese.
#
# PERCHÉ È GRAVE PROPRIO QUI. `attention_highlight` dice al modello "Focus on
# the part of the video between X and Y seconds": comunica ISTANTI a un
# modello le cui celle video non hanno coordinate temporali distinte. Tutti
# gli arm del filone (highlight_top1_24, random_top1_24, la sonda antipeak)
# sono girati così.
#
# LE DUE DOMANDE, e perché servono sei run e non due:
#   1. Quanto costa il collasso in accuracy pura?   → righe baseline
#   2. Riparandolo, il puntatore guidato dall'attenzione inizia a battere
#      quello casuale?                              → righe highlight/random
# La (2) è quella che decide il futuro del filone: oggi highlight ≈ random ≈
# baseline, ed è possibile che l'attenzione non abbia mai potuto esprimersi
# perché il canale su cui parla (i secondi) non era leggibile dal modello.
#
# DISEGNO 2x3, tutte e sei le celle sullo STESSO subset da 500 sample
# (`shuffle=true seed=42 limit=500`, nessuno `shard`: `limit` prende un
# prefisso dell'ordine mescolato, quindi il subset è identico fra i task ed è
# lo stesso della sonda antipeak — i risultati si joinano per-sample).
#
#   task  arm        pass_video_metadata
#     0   baseline   false   (regime storico, riferimento)
#     1   baseline   true
#     2   highlight  false
#     3   highlight  true
#     4   random     false
#     5   random     true
#
# Le righe `false` NON sono ridondanti con i numeri del README: quelli sono
# sull'intero set da 2700, questi su 500 — servono le celle `false` su QUESTO
# subset perché il confronto sia appaiato.
#
# COME SI LEGGE.
#   costo del collasso   = mcq_accuracy[1] - mcq_accuracy[0]
#   effetto sul filone   = (mcq_accuracy[3] - mcq_accuracy[5])      <- knob ON
#                          vs
#                          (mcq_accuracy[2] - mcq_accuracy[4])      <- knob OFF
# cioè: il vantaggio di highlight su random cresce quando l'orologio
# funziona? Se sì, il segnale d'attenzione c'era e non poteva esprimersi, e
# tutto il filone va rimisurato in regime ON. Se no, il picco è davvero poco
# informativo e il collasso non c'entrava.
#
# Guardare anche `mcq_accuracy_highlight_units_seconds` vs `..._fraction`
# (breakdown in evals/base.py): a knob OFF il fallback a unità frazionarie
# scatta per la ragione sbagliata, e i due gruppi possono comportarsi in modo
# diverso.
#
# ⚠️ I risultati di questo job NON vanno messi nella tabella ablation del
# README: subset da 500 e non 2700, quindi non confrontabili con `Δ vs
# baseline_24`. È un controllo, come la sonda antipeak.
#
# COSTO. Sulla card più lenta del constraint (Quadro RTX 6000): baseline
# ~7 s/sample → ~1h, highlight/random ~13.5 s/sample → ~1h53. `--time=04:00:00`
# copre il caso peggiore con ~2x di margine. Il knob non cambia il costo
# (passa metadata, non ricalcola nulla).
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=baseline;  META=false ;;
  1) ARM=baseline;  META=true  ;;
  2) ARM=highlight; META=false ;;
  3) ARM=highlight; META=true  ;;
  4) ARM=random;    META=false ;;
  5) ARM=random;    META=true  ;;
  *) echo "SLURM_ARRAY_TASK_ID inatteso: ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

case "${ARM}" in
  baseline)
    # `uniform` sul modello *_attn, esattamente come baseline_24 dell'ablation:
    # nessun gate, nessun puntatore, un solo pass.
    ARM_OVERRIDES=( strategy=uniform )
    ;;
  highlight|random)
    CELL=$([ "${ARM}" = "highlight" ] && echo attention || echo random)
    ARM_OVERRIDES=(
        strategy=attention_highlight
        strategy.top_k=1
        strategy.cell_select=${CELL}
        strategy.random_seed=0
        strategy.always_highlight=false
        strategy.pointer_units=seconds
        strategy.pointer_position=prepend
    )
    ;;
esac
echo "arm: ${ARM}  pass_video_metadata: ${META}"

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    dataset.nframes=24 \
    model.pass_video_metadata=${META} \
    "${ARM_OVERRIDES[@]}" \
    hf_home=/work/tesi_avalenza/hf \
    shuffle=true \
    limit=500 \
    wandb.group=video_mme-mrope-clock-control \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}_24-meta${META}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[control,video_mme,mrope,${ARM},meta_${META}] \
    "$@"
