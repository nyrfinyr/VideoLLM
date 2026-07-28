#!/bin/bash
#SBATCH --job-name=ablation-entropy_zoom_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=05:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 4/5: gate di entropia invariato (entropy_threshold=0.7), ma OGNI
# resample forzato nel ramo zoom a SINGOLO PICCO (`zoom_peak`, non
# multi-regione): `force_zoom_for_debug=true` + `zoom_multi_region=false`.
# Il codice bypassa `branch_selector` quando `force_zoom_for_debug=true`
# (strategies/entropy_attention_resample.py) — il valore di
# `branch_selector` (lasciato al default `concentration`) non conta in
# questo arm.
#
# Job array a 3 shard (900 sample ciascuno). `--time=05:00:00`: stesso
# criterio e stessa ricalibrazione di entropy_shift_24, che resta il caso
# peggiore misurato dei tre arm signal-driven (3h23 su RTX6000-24G).
# **Misurato sull'array 74428**: shard 0 su `dracula` (L40S 45G) in
# 5481s = 1h31, 6.09 s/sample — contro i 6.04 s/sample di
# shift_zoom_router_24 sulla stessa card, cioè i due arm costano uguale.
# Proiettato sulla card più lenta del constraint (penalità RTX6000/L40S
# = 1.9×, misurata su baseline_24 shard 0 vs 1-2): ~11.6 s/sample ≈ 2h54,
# quindi 5h danno ~1.7× di margine. Il ramo `zoom_peak` costa come
# `phase_shift` — è un secondo `Video()` su una finestra più stretta,
# stesso numero di frame; i 46.5 s/sample citati in
# video_mme-signal-router.sbatch sono dello zoom MULTI-REGIONE (che
# estrae PNG su disco), qui disattivato.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy
# dell'arm è quella POOLED (Σ corretti / Σ sample), non la media delle tre.
#
# `--constraint` aperto anche alle 24G (RTX6000/A5000) — vedi la nota
# estesa in entropy_shift_24.sbatch, incluso l'avviso che un OOM si
# manifesta come run `finished` senza `mcq_accuracy`, non come job fallito.
#
# BREAKDOWN short/medium/long: ora è nel summary di fine eval, non serve
# più una query post-hoc — `evals.base.mcq_accuracy` emette
# `correct_duration_*`/`seen_duration_*` per fascia (e per `task_type`), e
# `utils.obs.log_eval_summary` li scrive sia in wandb.summary
# (`mcq_accuracy_duration_long`, `n_correct_duration_long`,
# `n_samples_duration_long`) sia come tabella nel log SLURM. I conteggi
# sono lì apposta: per il pooling fra shard si sommano.
#
# GATE D'ENTROPIA: anche i suoi numeri sono nel summary, dalla stessa
# meccanica — `resampled`/`resample_kind`/`pred_fallback` sono già nel dict
# che la strategy ritorna e sono in `BREAKDOWN_KEYS`:
#   n_samples_resampled_true    quanti sample superano entropy_threshold
#   mcq_accuracy_resampled_true accuracy CONDIZIONATA al superamento
#   mcq_accuracy_resampled_false accuracy sui sample accettati al pass 1
#   n_samples_resample_kind_*   split fra i rami (phase_shift/zoom_peak)
# Su Video-MME `resampled=True` equivale a `answer_entropy >
# entropy_threshold` (le altre condizioni di `can_resample` sono sempre
# vere qui). Resta una query Weave solo l'analisi win/loss sample-per-
# sample (docs/analisi_winloss_resampling_video_mme.md).
#
# Per lanciarlo in regime frame-doubling (1 cella = 1 frame → il picco di
# attenzione localizza UN frame, non una coppia; vedi
# baseline_24double.sbatch): `sbatch ... entropy_zoom_24.sbatch --double-frames`
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
    strategy.force_zoom_for_debug=true \
    strategy.zoom_multi_region=false \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-entropy_zoom_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,entropy_zoom_24] \
    "$@"
