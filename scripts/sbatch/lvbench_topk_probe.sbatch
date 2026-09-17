#!/bin/bash
#SBATCH --job-name=lvbench-topk-probe
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0
#SBATCH --time=08:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# PROBE da 100 sample dell'arm `topk_resample`, da lanciare PRIMA del fullset
# (scripts/sbatch/lvbench_topk.sbatch). Stessa riga di comando: cambiano solo
# `limit`/`shuffle` e il gruppo wandb. Il disegno è descritto nel fullset e in
# strategies/topk_resample.py.
#
# SERVE A DUE COSE:
#   1. CORRETTEZZA: la strategy è nuova ed è stata provata end-to-end solo con
#      un modello finto (le coppie, le regioni e il budget sono verificati; il
#      processor vero no).
#   2. COSTO REALE. È l'incognita: il pass 1 a 512 frame costa 57.7 s/sample
#      (job 107727), ma i 5 pass 2 da 128 frame non sono mai stati misurati.
#      Stima 135-160 s/sample; il fullset si dimensiona SU QUESTO numero.
#
# QUALI 100 SAMPLE: shuffle seed=42 poi i primi 100 (utils/samples.py applica
# shuffle → shard → limit, e con --array=0 lo shard non taglia nulla). Sono gli
# STESSI 100 della probe dei segnali 107727: i due risultati si incrociano
# sample per sample. ⚠️ Non allargare --array qui, cambierebbe la selezione.
#
# ⚠️ model.fix_videoframes_resize=true è OBBLIGATORIO: senza, pass 1 e pass 2
# girerebbero a risoluzioni diverse (la strategy lo rifiuta).
# ⚠️ --constraint SENZA Turing (Qwen3-VL ci è 6x più lento).
#
# Uso:
#     sbatch scripts/sbatch/lvbench_topk_probe.sbatch
#     LIMIT=30 sbatch --time=03:00:00 scripts/sbatch/lvbench_topk_probe.sbatch
#     PASS2=64 sbatch scripts/sbatch/lvbench_topk_probe.sbatch
#
# COSA GUARDARE, IN QUEST'ORDINE (driver /wandb, project lvbench, group
# lvbench-topk-test):
#     uv run python .claude/skills/wandb/driver.py \
#         qwen3_vl_2b_attn-topk-<jobid>_0_probe100 --shard-size 194
#   1. CORRETTEZZA:
#      a. `mcq_accuracy` nel summary (l'eval è arrivata in fondo);
#      b. t_cells=256 su ogni sample, e griglia coerente con l'aspect ratio:
#         5x9 sul 16:9 (la maggioranza), ma 4x10 / 6x7 / 5x8 / 6x8 sono
#         legittime (sulla probe 107727 comparivano tutte). L'unica firma del
#         bug è 4x8 = 32 token per cella: fix del resize non attivo;
#      c. per ogni condizione: `cells` lunga k, `n_pairs_zoom + n_pairs_global`
#         = pass2_nframes/2, e `centers_first`/`centers_last` dentro le celle
#         scelte (per hybrid no: metà coppie sono globali per costruzione);
#      d. `rand3` deve scegliere celle DIVERSE da `k3` sulla gran parte dei
#         sample: se coincidono, il controllo non è un controllo;
#      e. 100 righe Weave, nessun traceback nel .err.
#   2. COSTO: `model_latency_mean` → walltime del fullset (vedi lì l'aritmetica).
#   2b. SINK: `sink_stats` non nullo, `sink_border_share` vs
#      `border_share_uniform` (se coincidono, i sink NON sono un fenomeno di
#      bordo), `attn_sink_cell_corr` (se è alta, il ranking temporale è solo
#      la mappa dei sink), e i 10 dump in ${DUMP_DIR}. Da lì:
#          uv run python scripts/sink_heatmaps.py <dump-root> --all
#      che dipinge attenzione e sink sui frame veri, celle calde contro celle
#      fredde di controllo.
#   3. SOLO ALLA FINE i numeri: `mcq_accuracy` è la BASELINE (pass 1, atteso
#      ~43% come la probe dei segnali), e ogni condizione ha
#      `mcq_accuracy_cond_<nome>`. Su 100 sample nessun delta è significativo:
#      servono a vedere che le condizioni non crollino, non a scegliere k.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:-100}"
PASS2="${PASS2:-128}"
DUMP_LIMIT="${DUMP_LIMIT:-10}"
DUMP_ROOT="${DUMP_ROOT:-/work/tesi_avalenza/topk_dumps}"

NAME="topk"
GROUP=lvbench-topk-test
SUFFIX="_probe${LIMIT}"
DUMP_DIR="${DUMP_ROOT}/lvbench_${NAME}${SUFFIX}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${DUMP_DIR}"

echo "PROBE: ${LIMIT} sample — correttezza e costo, non i delta"
echo "run: ${NAME}${SUFFIX}  group: ${GROUP}  pass2_nframes: ${PASS2}"
echo "dump: ${DUMP_LIMIT} sample → ${DUMP_DIR}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    model.fix_videoframes_resize=true \
    dataset=lvbench \
    dataset.nframes=512 \
    dataset.max_pixels=50176 \
    dataset.min_pixels=3136 \
    strategy=topk_resample \
    strategy.pass2_nframes="${PASS2}" \
    strategy.sink_stats_every=1 \
    strategy.dump_dir="${DUMP_DIR}" \
    strategy.dump_limit="${DUMP_LIMIT}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    limit="${LIMIT}" \
    shuffle=true \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}${SUFFIX} \
    wandb.tags=[lvbench,qwen3_vl_2b,topk_resample,probe] \
    "$@"
