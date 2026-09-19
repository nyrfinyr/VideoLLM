#!/bin/bash
#SBATCH --job-name=lvbench-additive
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --time=03:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# ARM ADDITIVO: i 256 frame mirati si AGGIUNGONO alla base da 256 invece di
# sostituirla. Disegno e motivazioni: strategies/additive_topk.py.
#
# CONDIZIONI (appaiate, un solo decode per sample):
#     baseline  = `pred`, la risposta sui soli 256 frame della base
#     k10       256 frame aggiunti nelle top-10 celle del ranking
#     k5        gli stessi 256 nelle top-5
#     rand10    10 celle A CASO — il controllo che decide se l'arm è un arm
#
# GATE GIÀ PASSATO (probe oracolo 110308/09, 92 sample, iso-budget 512):
#     base256 41.3%  |  uniform512 41.3% (Δ 0.0, p=1)  |  oracle512 54.3%
#     (Δ +13.0 pp contro uniform, +15/−3, p=0.0075)
# cioè i frame uniformi in più valgono ZERO e tutto il guadagno sta nel DOVE.
# Atteso per l'arm ≈ hit@10 × guadagno ≈ 0.59 × ~15 = +6/+9 pp.
#
# QUANTI SAMPLE (è il knob che conta, e va deciso PRIMA di guardare i numeri):
#   --array=0-1  (default) → 100 sample: gli STESSI di tutte le probe
#                 precedenti. Serve a CORRETTEZZA e COSTO del codice nuovo,
#                 non a misurare: a +7 pp attesi un delta su 100 sample non è
#                 distinguibile dal rumore.
#   --array=0-3            → 200 sample (i 100 di sempre + 100 nuovi, la
#                 selezione è shuffle 42 → shard strided → primi 50), che è la
#                 numerosità per cui un +7 pp è misurabile.
# ⚠️ Non cambiare LIMIT: con LIMIT=50 e N shard i sample sono i primi 50*N
#    della stessa lista, quindi i confronti con le run precedenti reggono.
#
# ANALISI PRE-REGISTRATA (da fissare ora, non dopo aver visto i numeri):
#   primaria    delta di k10 contro la baseline appaiata, McNemar esatto;
#   controllo   k10 contro rand10: se coincidono, l'arm è rumore e si chiude
#               (è come è morto attention_marker);
#   secondaria  la stessa cosa sulla sottopopolazione a FINESTRA STRETTA
#               (<10% del video, il 72% dei sample), dove l'oracolo dava
#               +16/+18 pp invece di +4 — la finestra viene dal dataset e si
#               unisce offline, la strategy non la vede mai.
#
# ⚠️ model.fix_videoframes_resize=true è OBBLIGATORIO (la strategy lo rifiuta).
# ⚠️ dataset.nframes è la BASE (256), non il totale: ogni condizione gira a
#    256 + n_added = 512 frame.
# ⚠️ --constraint SENZA Turing (Qwen3-VL ci è 6x più lento).
#
# Uso:
#     sbatch scripts/sbatch/lvbench_additive.sbatch                  # 100 sample
#     sbatch --array=0-3 scripts/sbatch/lvbench_additive.sbatch      # 200 sample
#     LIMIT=2 sbatch --array=0 --time=00:40:00 scripts/sbatch/lvbench_additive.sbatch
#
# PRIMA ESECUZIONE: LIMIT=2 --array=0. Il codice nuovo (strategy + piano dei
# frame uniti) è verificato solo con VLM finto e decord finto.
#
# COSA GUARDARE, IN QUEST'ORDINE:
#     uv run python .claude/skills/wandb/driver.py qwen3_vl_2b_attn-additive-<jobid>_<task>
#   1. CORRETTEZZA:
#      a. `mcq_accuracy` nel summary (l'eval è arrivata in fondo);
#      b. t_cells=128 e griglia coerente con l'aspect ratio (5x9 sul 16:9;
#         4x8 = 32 token per cella è la firma del resize non fissato);
#      c. per ogni condizione: `cells` lunga k, `n_total_frames` ≈ 512,
#         `n_collision` piccolo (se è grande, le celle sono troppo strette per
#         il budget: video corto o fps basso);
#      d. `rand10` deve scegliere celle DIVERSE da `k10` sulla gran parte dei
#         sample: se coincidono, il controllo non è un controllo;
#      e. `pred_fallback` ~0 e nessun traceback nel .err.
#   2. COSTO: `model_latency_mean` — la cattura del pass 1 e il decode dei
#      ~1000 frame unici sono il costo, i forward delle condizioni ~2 s l'uno.
#   3. SOLO ALLA FINE i numeri: `mcq_accuracy` è la BASELINE appaiata, ogni
#      condizione ha `mcq_accuracy_cond_<nome>` più lo split
#      `_base_true`/`_base_false` (= r_break e r_fix direttamente nel summary).
#      Analisi offline (T1, condizioni, gate):
#          uv run python scripts/analyze_topk_run.py <run> --cache /tmp/add.json

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:-50}"
BASE="${BASE:-256}"
ADDED="${ADDED:-256}"
N_TOT=$((SLURM_ARRAY_TASK_COUNT * LIMIT))
GROUP="${GROUP:-lvbench-additive}"

echo "ARM additivo: base=${BASE} + aggiunti=${ADDED} per condizione (totale $((BASE+ADDED)) frame)"
echo "${LIMIT} sample per shard x ${SLURM_ARRAY_TASK_COUNT} shard = ${N_TOT} sample"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    model.fix_videoframes_resize=true \
    dataset=lvbench \
    dataset.nframes="${BASE}" \
    dataset.max_pixels=50176 \
    dataset.min_pixels=3136 \
    dataset.use_time_reference=false \
    strategy=additive_topk \
    strategy.n_added="${ADDED}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    limit="${LIMIT}" \
    shuffle=true \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-additive-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_n${N_TOT} \
    wandb.tags=[lvbench,qwen3_vl_2b,additive_topk] \
    "$@"
