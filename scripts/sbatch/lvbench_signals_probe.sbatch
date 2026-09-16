#!/bin/bash
#SBATCH --job-name=lvbench-signals-probe
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# PROBE da 100 sample della run di segnali Qwen3-VL-2B su LVBench, da lanciare
# PRIMA del fullset (scripts/sbatch/lvbench_signals.sbatch). Stessa identica
# riga di comando del fullset: cambiano solo `limit`/`shuffle`, il gruppo wandb
# e il numero di dump. Il disegno della run (campionamento a coppie, griglia,
# segnali loggati) è descritto nel fullset.
#
# SERVE A DUE COSE, in quest'ordine:
#   1. CORRETTEZZA: la strategy `signals_capture` è nuova. Vanno controllati
#      celle, griglia e dump su sample veri prima di spendere ~6 GPU x 12 h;
#   2. COSTO REALE per sample. È l'incognita principale: il forward con
#      cattura a 512 frame costa 3.4 s (job 106484), ma l'estrazione dei 512 PNG
#      da video di 1-2 h non è mai stata misurata e domina.
#
# QUALI 100 SAMPLE. `utils/samples.py` applica shuffle → shard → limit.
# Con `--array=0` lo shard è 0 su 1, cioè nessun taglio. I 100 sample sono
# quindi i primi 100 della permutazione con seed=42 dei 1549: domande
# mescolate su tutti i video, stabili fra rilanci. Lo shuffle è ESSENZIALE: il
# loader emette per video e i primi 100 non mescolati sarebbero ~7 video.
# ⚠️ Non allargare `--array` qui: con N task lo shuffle resta lo stesso ma lo
# shard strided sceglie 100 sample DIVERSI per ogni task. E i 100 NON sono un
# sottoinsieme di un particolare shard del fullset, che non mescola.
#
# DUMP: i primi 10 sample della probe (default), cioè 10 domande già mescolate.
# Vanno in /work/tesi_avalenza/signals_dumps/lvbench_signals_512p_probe100-<jobid>/.
#
# ⚠️ `model.fix_videoframes_resize=true` OBBLIGATORIO (vedi il fullset): senza,
# griglia 4x8 e budget a 8192 token.
# ⚠️ `--constraint` SENZA Turing (Qwen3-VL ci è 6x più lento). Non si può
# restringere alle 45G: il job può finire su una A5000 24G, e il picco di
# 6.8 GB misurato a 512 frame ci sta. Leggi `gpu_name` nel config wandb
# insieme al costo: una A5000 è più lenta di una L40S.
#
# WALLTIME: 100 x 60-80 s stimati ≈ 1.7-2.2 h. 4 h reggono fino a ~144 s/sample.
# Il summary Weave si scrive solo a fine eval, quindi un TIMEOUT non dà
# `model_latency_mean`, ma è comunque una risposta: vuol dire costo > ~140
# s/sample, e il fullset a 6 shard x 12 h (tetto ~165 s/sample) va ripensato.
#
# Uso:
#     sbatch scripts/sbatch/lvbench_signals_probe.sbatch
#     LIMIT=30 sbatch --time=01:30:00 scripts/sbatch/lvbench_signals_probe.sbatch
#     GAP=4.0 DUMP_LIMIT=20 sbatch scripts/sbatch/lvbench_signals_probe.sbatch
#
# COSA GUARDARE, IN QUEST'ORDINE (driver /wandb, project lvbench, group
# lvbench-signals-test):
#     uv run python .claude/skills/wandb/driver.py \
#         qwen3_vl_2b_attn-signals_512p-<jobid>_0_probe100 --shard-size 259
# (i nomi esatti dei campi per-sample sono quelli di strategies/signals_capture.py)
#   1. CHECK, prima di qualunque numero:
#      a. `mcq_accuracy` presente nel summary: l'eval è arrivata in fondo e lo
#         scorer ha girato;
#      b. per OGNI sample t_cells=256 e griglia attesa: 5x9 su 16:9
#         (11 520 token visivi), 6x8 su 4:3 (12 288). Una griglia 4x8 = fix
#         del resize non attivo; t_cells≠256 = coppie non allineate alle celle;
#      c. centri a coppie: 256 centri crescenti, e i due frame di ogni cella
#         a ~GAP secondi (±1/fps). Il timestamp di cella deve essere il centro;
#      d. nessun sample saltato: 100 righe Weave, nessun errore nel predict
#         (e nessun traceback in logs/lvbench-signals-probe-<jobid>_0.err);
#      e. dump scritti: 10 sample nella directory di dump, non vuoti.
#   2. COSTO: `model_latency_mean` → walltime del fullset. Uno shard da S sample
#      dura S x L / 3600 h (il driver lo calcola con --shard-size). Tieni
#      walltime ≥ 1.5x e ≤ 24 h, altrimenti più shard: vedi l'esempio nel
#      fullset. Guarda anche la varianza, i video da 2 h pesano più della media.
#   3. SOLO ALLA FINE l'accuracy: 100 sample, nessun intervento. Deve solo
#      stare nell'intorno della baseline Qwen3-VL-2B su LVBench. Un crollo
#      verso il 25% (caso) indica un bug nel prompt o nei frame, non un risultato.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:-100}"
GAP="${GAP:-2.0}"                 # secondi fra i due frame di una coppia
DUMP_LIMIT="${DUMP_LIMIT:-10}"
DUMP_ROOT="${DUMP_ROOT:-/work/tesi_avalenza/signals_dumps}"

NAME="signals_512p"
GROUP=lvbench-signals-test
SUFFIX="_probe${LIMIT}"
DUMP_DIR="${DUMP_ROOT}/lvbench_${NAME}${SUFFIX}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${DUMP_DIR}"

echo "PROBE: ${LIMIT} sample — correttezza e costo, non l'accuracy"
echo "run: ${NAME}${SUFFIX}  shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"
echo "gap: ${GAP}s  dump: ${DUMP_LIMIT} sample → ${DUMP_DIR}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    model.fix_videoframes_resize=true \
    dataset=lvbench \
    dataset.nframes=512 \
    dataset.max_pixels=50176 \
    dataset.min_pixels=3136 \
    strategy=signals_capture \
    strategy.pair_gap_sec="${GAP}" \
    strategy.sink_dump_dir="${DUMP_DIR}" \
    strategy.sink_dump_limit="${DUMP_LIMIT}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    limit="${LIMIT}" \
    shuffle=true \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}${SUFFIX} \
    wandb.tags=[lvbench,qwen3_vl_2b,signals_capture,${NAME},probe] \
    "$@"
