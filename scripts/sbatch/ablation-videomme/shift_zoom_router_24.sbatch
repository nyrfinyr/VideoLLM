#!/bin/bash
#SBATCH --job-name=ablation-shift_zoom_router_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=10:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom (docs/future-work/output_riunione_2207.md) —
# arm 5/5: router a soglia adattiva (`branch_selector=concentration_ratio`,
# `concentration_ratio=3.85`) che decide sample-per-sample fra
# `phase_shift` (disperso) e zoom a SINGOLO PICCO
# (`zoom_multi_region=false`, coerente con entropy_zoom_24.sbatch — qui si
# confronta shift-vs-zoom-singolo-picco, non shift-vs-zoom-multiregione
# dello Stage C originale/README).
#
# ⚠️ VERIFICARE LA SOGLIA PRIMA DI LANCIARE. `concentration_ratio=3.85` è
# validata su Video-MME a `nframes=128` → t=64 celle → cutoff effettivo
# `top1_pct < 3.85 * 100/64` = 6.0%. Qui `nframes=24` → t=12 → il cutoff
# diventa 32.1%. La forma adattiva (`ratio * 100/t`) è progettata per
# trasferirsi fra `t` diversi, ma non è MAI stata testata a 12 celle: se
# degenera (tutti i sample su un solo ramo) questo arm diventa un
# duplicato esatto di entropy_shift_24 o di entropy_zoom_24, e sono ~3
# job buttati. Un probe da 40 minuti lo esclude:
#   sbatch scripts/sbatch/video_mme-signal-test.sbatch limit=250 \
#       dataset.nframes=24 strategy.branch_selector=concentration_ratio \
#       strategy.concentration_ratio=3.85 strategy.zoom_multi_region=false
# Lo split si legge direttamente dal summary del probe, senza query Weave:
# `n_samples_resample_kind_phase_shift` vs `n_samples_resample_kind_zoom_peak`
# (anche nella tabella "Breakdown per categoria" del log). Serve che siano
# entrambi non trascurabili; se uno dei due è ~0 la soglia va ritarata
# prima di spendere i 3 shard.
#
# Job array a 3 shard (900 sample ciascuno). `--time=10:00:00`: stesso
# criterio e stessa derivazione di entropy_shift_24 (~30-35 s/sample
# stimati su RTX6000-24G → ~8h/shard, budget reale 40 s/sample) —
# incluso il probe di 40 minuti da fare prima, documentato lì.
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
# Per lanciarlo in regime frame-doubling: `sbatch ... --double-frames`
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
    strategy.branch_selector=concentration_ratio \
    strategy.concentration_ratio=3.85 \
    strategy.zoom_multi_region=false \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-shift_zoom_router_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,shift_zoom_router_24] \
    wandb.notes="ablation riunione 22-07-26: router soglia adattiva 3.85 (shift vs zoom singolo picco), 24 frame, shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)" \
    "$@"
