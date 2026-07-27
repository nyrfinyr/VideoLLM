#!/bin/bash
#SBATCH --job-name=ablation-entropy_shift_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=14:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 3/5: gate di entropia invariato (entropy_threshold=0.7), ma OGNI
# resample forzato nel ramo `phase_shift` — `concentration_threshold=101`
# rende `dispersed` sempre vero (`top1_pct` è una percentuale, max 100,
# quindi mai sotto una soglia >100); `branch_selector=concentration`
# resta il default, non serve toccarlo (vedi
# strategies/entropy_attention_resample.py).
#
# Job array a 3 shard (900 sample ciascuno, `samples[shard::num_shards]`
# in utils/samples.py). Un job singolo da 2700 sample non ha checkpoint:
# un timeout butterebbe via l'intero arm.
#
# `--time=14:00:00`. Riferimento misurato: 24.6 s/sample a 128 FRAME su
# L40S (run y6u5k1pq, 250 sample in 6152s = 1h42). Su RTX6000-24G, la
# card più lenta ora ammessa dal constraint, la penalità documentata in
# docs/probe_throughput.md è ~2.7× → 66 s/sample, che a 128 frame darebbe
# 16.6h/shard: OLTRE questo --time. Ci si sta perché a 24 frame i token
# visivi sono ~5× meno (t=12 celle contro 64) su ENTRAMBI i pass, quindi
# il per-sample reale è una frazione di quel numero — stima ~30-35 s/
# sample su 24G → ~8h/shard, ~1.7× di margine. È l'unico --time della
# cartella che dipende da una stima di speedup invece che da un limite
# superiore: se un probe misura più di 55 s/sample su 24G, o si alza a
# 20h o si ripinna il constraint alle sole 45G.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy
# dell'arm è quella POOLED (Σ corretti / Σ sample), non la media delle
# tre; stesso metodo di LVBench (vedi README, nota LVBench).
#
# `--constraint` aperto anche alle 24G (RTX6000/A5000): gli OOM di
# `full_visual_attention` (pass 1, richiesto da `SupportsSignals`/
# `model=qwen2_5_vl_3b_attn`) documentati nel README sono TUTTI a 128
# frame, dove la maschera eager scala con S²; a 24 frame S cala ~5×.
# ⚠️ Se un OOM ci fosse comunque, NON si vede dallo stato del job: Weave
# cattura l'eccezione per sample e il run wandb risulta `finished` con
# 0 predizioni valide (README, "OOM totale su 3 combinazioni"). Dopo il
# primo shard, controllare che `mcq_accuracy` esista nel summary.
#
# Metriche gate d'entropia (quanti sample superano `entropy_threshold`,
# accuracy condizionata) e breakdown short/medium/long: NON in questo
# output — si ricavano da una query Weave per-sample sui campi
# `resampled`/`resample_kind` + la colonna `duration` del dataset, stesso
# metodo di docs/analisi_winloss_resampling_video_mme.md.
#
# Per lanciare lo stesso arm in regime frame-doubling (1 cella = 1 frame,
# vedi baseline_24double.sbatch): `sbatch ... entropy_shift_24.sbatch --double-frames`
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=entropy_attention_resample \
    dataset.nframes=24 \
    strategy.branch_selector=concentration \
    strategy.concentration_threshold=101 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-entropy_shift_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,entropy_shift_24] \
    wandb.notes="ablation riunione 22-07-26: gate entropia + resample sempre phase_shift, 24 frame, shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)" \
    "$@"
