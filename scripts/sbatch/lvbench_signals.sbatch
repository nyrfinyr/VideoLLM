#!/bin/bash
#SBATCH --job-name=lvbench-signals
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# FULLSET — run di SEGNALI Qwen3-VL-2B su LVBench (1549 QA), strategy
# `signals_capture`. NON è un arm: nessuna generazione, nessun intervento.
# Per ogni sample UN forward di prefill con cattura d'attenzione, e si logga:
#   - massa d'attenzione per cella, sui rowset all / question / last_token,
#     sia grezza sia sink-filtrata;
#   - entropia e probabilità della risposta (dai logit dell'ultimo token);
#   - statistiche dei sink su TUTTI i layer;
#   - un dump per-token dei primi DUMP_LIMIT sample di ogni shard.
# Serve a costruire offline il segnale di localizzazione (il collo di bottiglia
# dell'oracolo LVBench: ricampionare nella finestra vera vale +16 pp, ma il
# segnale attuale la trova nel 18% dei recuperi, docs/oracolo_lvbench.md).
#
# CAMPIONAMENTO: 512 frame A COPPIE (utils/pair_sampling.py). 256 centri
# uniformi sull'intero video, due frame a ±GAP/2 da ogni centro (default ±1 s).
# Qwen3-VL fonde i frame a due a due, quindi una cella temporale = una coppia =
# un istante preciso, col timestamp del prompt uguale al centro. Con 512 frame
# uniformi su un video da 71 min i due frame di una cella disterebbero ~8 s e
# la cella non avrebbe un istante preciso.
#
# RISOLUZIONE: 50176 px per frame → griglia 5x9 su 16:9 (11 520 token visivi)
# e 6x8 su 4:3 (12 288 token). Su 4:3 sono 512 x 49 152 = 25 165 824 px totali,
# ESATTAMENTE il tetto del processor: 512 frame è il massimo che non fa scendere
# la risoluzione per frame.
#
# ⚠️ `model.fix_videoframes_resize=true` è OBBLIGATORIO. I frame a coppie passano
# come `VideoFrames` e senza il fix qwen-vl-utils li arrotonda a multipli di 64:
# griglia 4x8, budget a 8192 token, e le celle non sono più quelle misurate
# dalla probe di equivalenza (probe_capture_equiv.sbatch).
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75): Qwen3-VL ci è 6x più lento.
# Su all_usr_prod non si può restringere alle 45G, quindi il job può finire su
# una A5000 da 24G. Qui non è un problema: la probe GPU (job 106484, L40S) ha
# misurato un picco di 6.8 GB a 512 frame con cattura.
#
# COSTO — ⚠️ STIMATO, NON MISURATO. Il forward con cattura a 512 frame costa
# 3.4 s (job 106484), ma quella latenza NON include l'estrazione dei 512 frame
# in PNG da video lunghi 1-2 h, che domina. Riferimento storico: la baseline
# Qwen2.5-VL a 768 frame su LVBench costava ~76 s/sample, con la decodifica a
# fare quasi tutto il costo (docs/probe_throughput.md). Stima usata qui:
# 60-80 s/sample.
#   6 shard x ~259 sample (slice strided `samples[shard::6]`)
#   → a 80 s/sample ≈ 5.8 h per shard; 12 h reggono fino a ~165 s/sample (2x).
# `--mem=32G` come la baseline LVBench a 768 frame: la RAM la consumano i
# frame decodificati, non il modello.
#
# ⚠️ SHARD E WALLTIME VANNO RITARATI con la probe PRIMA di lanciare questo file:
#     sbatch scripts/sbatch/lvbench_signals_probe.sbatch
#     uv run python .claude/skills/wandb/driver.py \
#         qwen3_vl_2b_attn-signals_512p-<jobid>_0_probe100 --shard-size 259
# Con `model_latency_mean` = L s/sample, uno shard da S sample dura S*L/3600 h:
# tieni la walltime ≥ 1.5x quel valore (e ≤ 24 h), altrimenti alza gli shard.
# Esempio: L=150 → 259 sample = 10.8 h, troppo vicino alle 12 h → 8 shard
# (`--array=0-7`, 194 sample = 8.1 h). Il timeout di uno shard costa caro: il
# summary Weave (accuracy, model_latency_mean) si scrive solo a fine eval.
#
# DUMP DEI SINK: una directory per shard sotto
#   /work/tesi_avalenza/signals_dumps/lvbench_signals_512p-<array_job_id>/shard_<i>
# con DUMP_LIMIT sample per shard, di default ceil(50 / num_shards) → ~50 sample
# in totale (9 x 6 = 54). ⚠️ Sono i PRIMI sample di ogni shard nell'ordine del
# loader, che emette le domande raggruppate per video e non mescola: i ~50 dump
# coprono quindi solo i primi video del dataset. Per un campione vario usa i
# dump della probe, che è mescolata.
#
# Uso:
#     sbatch scripts/sbatch/lvbench_signals.sbatch
#     sbatch --array=0-7 --time=16:00:00 scripts/sbatch/lvbench_signals.sbatch
#     GAP=4.0 DUMP_LIMIT=4 sbatch scripts/sbatch/lvbench_signals.sbatch
#     DUMP_ROOT=/work/tesi_avalenza/altro sbatch scripts/sbatch/lvbench_signals.sbatch
# Argomenti extra dopo il nome dello script passano a main.py (`"$@"`).
#
# I 6 shard condividono `wandb.group=lvbench-signals`: l'accuracy del fullset
# si ottiene sommando `true_count` dei 6 run (evals/base.py).

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

GAP="${GAP:-2.0}"                 # secondi fra i due frame di una coppia
DUMP_TOTAL="${DUMP_TOTAL:-50}"    # sample dumpati in totale sull'array
DUMP_LIMIT="${DUMP_LIMIT:-$(( (DUMP_TOTAL + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))}"
DUMP_ROOT="${DUMP_ROOT:-/work/tesi_avalenza/signals_dumps}"

NAME="signals_512p"
GROUP=lvbench-signals
# Una sottodirectory per shard: i nomi dei file li decide la strategy, e shard
# diversi non devono potersi sovrascrivere.
DUMP_DIR="${DUMP_ROOT}/lvbench_${NAME}-${SLURM_ARRAY_JOB_ID}/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${DUMP_DIR}"

echo "run: ${NAME}  shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"
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
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[lvbench,qwen3_vl_2b,signals_capture,${NAME},fullset] \
    "$@"
