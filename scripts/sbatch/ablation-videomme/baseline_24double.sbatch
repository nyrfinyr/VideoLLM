#!/bin/bash
#SBATCH --job-name=ablation-baseline_24double
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md,
# "64 double frames o 32" + "rifare baseline anche di queste due") —
# arm 2/5: baseline `uniform` a 24 timestamp in regime FRAME-DOUBLING
# (`--double-frames`): ogni frame campionato viene passato al modello due
# volte di fila, così il merge temporale ×2 di Qwen2.5-VL fonde un frame
# con SE STESSO e **ogni cella della griglia visiva ↔ UN SOLO frame
# reale** (invece dei due frame distinti, a un intervallo di distanza, del
# regime normale). È lo stesso regime del `--double-frames` di
# `attn_explorer/capture.py`, ora disponibile anche in eval
# (`Strategy._build_media` / `_sample_doubled_frames`).
#
# Perché conta per questa ablation: nel regime normale il picco di
# attenzione localizza una COPPIA di frame e non si può sapere quale dei
# due ha attirato l'attenzione — qui invece cella = frame, quindi
# `peak_cell` è un frame vero e proprio. È il regime in cui shift e zoom
# ragionano sulla stessa granularità del segnale.
#
# Il confronto è:
#   baseline_24        24 timestamp → t=12  (1× token visivi)
#   baseline_24double  24 timestamp → t=24  (2× token visivi)  ← questo
# Stessi identici frame nei due arm, stessa risoluzione per cella
# (verificato via qwen-vl-utils): l'unica variabile è se il merge fonde
# due timestamp diversi o un frame con se stesso.
#
# Vale anche come controllo COMPUTE-MATCHED per gli arm signal-driven:
# quelli spendono due pass da 24 frame sui sample ricampionati, qui gli
# stessi token si spendono in un pass solo.
#
# Gli istanti campionati sono gli STESSI di baseline_24
# (`reconstruct_frame_indices` replica la formula di qwen-vl-utils), e il
# `sample_fps` iniettato è il doppio di quello del `Video` equivalente,
# così `second_per_grid_ts` (M-RoPE) resta la distanza reale fra due
# frame campionati — il modello non legge la clip come lunga il doppio.
#
# I/O: estrae 24 PNG per sample in una tmpdir, ripulita dopo la
# `generate()`. Costo trascurabile rispetto alla decodifica video, ma
# richiede che /tmp del nodo sia scrivibile.
#
# Job array a 3 shard (900 sample ciascuno). `--time=06:00:00`: stesso
# criterio di baseline_24, ma qui il valore ORIGINALE resta — è l'unico
# arm della cartella la cui stima iniziale era corretta.
# **Misurato sull'array 74426**:
#   shard 0  bimbogigi NVIDIA L40S 45G        8834s = 2h27   9.82 s/sample
#   shard 1  nullazzo  Quadro RTX 6000 24G   15055s = 4h11  16.73 s/sample
#   shard 2  huber     NVIDIA RTX A5000 24G   9428s = 2h37  10.48 s/sample
# È l'arm PIÙ COSTOSO dei cinque (~2.3× baseline_24): il doubling paga i
# token visivi doppi su TUTTI i 2700 sample, mentre gli arm signal-driven
# fanno il secondo pass solo sul ~72% che supera il gate. 4h11 contro 6h
# di budget = 1.43× di margine, il più stretto della cartella: non
# abbassare sotto le 6h. Accuracy POOLED sui 3 shard.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=uniform \
    dataset.nframes=24 \
    --double-frames \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-baseline_24double-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,baseline_24double] \
    "$@"
