#!/bin/bash
#SBATCH --job-name=ablation-entropy_shift_24
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
# `--time=10:00:00`, tenuto BASSO di proposito: il backfill di SLURM
# schedula prima i job corti e qui ne vanno in coda 15 (5 arm × 3 shard),
# quindi un --time gonfiato ritarda tutto l'esperimento. Riferimento
# misurato: 24.6 s/sample a 128 FRAME su L40S (run y6u5k1pq, 250 sample
# in 6152s = 1h42); a 24 frame i token visivi sono ~5× meno (t=12 celle
# contro 64) su ENTRAMBI i pass → stima ~30-35 s/sample su RTX6000-24G
# (la card più lenta del constraint) ≈ 8h/shard, ~1.25× di margine, e
# ~3h su L40S.
#
# ⚠️ È l'unico --time della cartella che dipende da una stima di speedup
# invece che da un limite superiore misurato. Budget reale: **40 s/sample**
# (900 × 40 = 10h). Sopra quella soglia lo shard va in timeout — costo: si
# rilancia QUEL solo shard (`sbatch --array=<i> --time=16:00:00 ...`),
# non l'arm intero. Per non scoprirlo dopo 10h basta un probe da 40
# minuti prima di lanciare tutto:
#   sbatch --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G" \
#       scripts/sbatch/video_mme-signal-test.sbatch limit=40 \
#       dataset.nframes=24 strategy.branch_selector=concentration \
#       strategy.concentration_threshold=101
# e leggere `model_latency.mean` (log: `grep -A6 'Evaluation summary'`).
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
