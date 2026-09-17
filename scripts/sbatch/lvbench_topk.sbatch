#!/bin/bash
#SBATCH --job-name=lvbench-topk
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# FULLSET — arm `topk_resample` su LVBench intero (1549 QA), Qwen3-VL-2B.
# Una sola run che porta a casa TRE cose insieme:
#
#   1. la BASELINE, appaiata: il pass 1 (512 frame a coppie, risposta
#      dall'argmax del prefill) è la baseline di ogni condizione — stesso
#      sample, stesso prompt, stesso modo di leggere la risposta, quindi
#      McNemar diretto e nessun confronto fra run diverse;
#   2. l'ABLATION DI k appaiata: k=1, k=3, k=10, ibrido (metà budget globale)
#      e il controllo random girano tutte sugli STESSI sample. Dare un k a
#      ciascuno di tre split del dataset sarebbe stato un confronto NON
#      appaiato: con ~500 sample per split distingue solo differenze ≥6 pp;
#   3. i SEGNALI per la scelta del gate: `answer_entropy` del pass 1 e la
#      risposta di OGNI condizione, per ogni sample. Il gate NON è online —
#      il pass 2 gira sempre — e la soglia si sceglie dopo, out-of-fold
#      (docs/scelta_quantile_entropia.md). Su LVBench l'entropia separa poco
#      (AUROC 0.547 contro 0.756 su Video-MME): gattare al volo avrebbe
#      buttato via proprio i dati che servono a calibrarla.
#
# Le masse per cella del pass 1 sono loggate per sample, quindi questa run
# ri-misura anche T1 (l'attenzione trova la finestra annotata?) sul fullset:
# sulla probe da 100 dava AUC 0.775 e hit@1 17% contro 1.6% del caso.
#
# PASS 1: 512 frame a COPPIE (256 centri uniformi, 2 frame a ±1 s) a
# max_pixels=50176 → 5x9 celle su 16:9, 11 520 token visivi.
# PASS 2: 128 frame per condizione, coppie a ±0.25 s dentro le celle scelte
# (la cella del pass 2 resta UN istante, come nel pass 1). Il budget si
# divide in parti uguali fra le k regioni; con global_fraction>0 una quota
# resta uniforme su tutto il video.
#
# ⚠️ model.fix_videoframes_resize=true OBBLIGATORIO (la strategy lo pretende):
# senza, pass 1 e pass 2 girano a risoluzioni diverse e il delta mescola
# l'intervento con la perdita di token.
# ⚠️ --constraint SENZA Turing (Qwen3-VL ci è 6x più lento). Una A5000 24G va
# bene: il picco misurato a 512 frame con cattura è 6.8 GB.
#
# COSTO — ⚠️ DA RITARARE CON LA PROBE. Il pass 1 costa 57.7 s/sample misurati
# (job 107727); i 5 pass 2 da 128 frame sono stimati 15-20 s l'uno (domina
# l'estrazione dei frame, non il forward). Stima 135-160 s/sample:
#   8 shard x ~194 sample (slice strided `samples[shard::8]`)
#   → a 160 s/sample ≈ 8.6 h per shard; 12 h reggono fino a ~223 s/sample.
# Con L s/sample dalla probe, uno shard da S sample dura S*L/3600 h: tieni la
# walltime ≥ 1.5x e ≤ 24 h, altrimenti alza gli shard. Un timeout costa caro:
# il summary Weave si scrive solo a fine eval.
#     uv run python .claude/skills/wandb/driver.py \
#         qwen3_vl_2b_attn-topk-<jobid>_0_probe100 --shard-size 194
#
# Uso:
#     sbatch scripts/sbatch/lvbench_topk.sbatch
#     sbatch --array=0-11 --time=08:00:00 scripts/sbatch/lvbench_topk.sbatch
#     PASS2=64 sbatch scripts/sbatch/lvbench_topk.sbatch
# Argomenti extra dopo il nome dello script passano a main.py ("$@"), es.
# una lista di condizioni diversa da quella del preset.
#
# I SINK, IN QUESTO SETTING. La run porta a casa anche di che pasta sono i
# token visivi a 512 frame su video lunghi: scalari spaziali su OGNI sample
# (dove guarda dentro il frame, quota di massa sul bordo, correlazione fra
# ranking temporale e mappa dei sink), statistiche dei canali degli hidden
# state su tutti i 28 layer un sample ogni ${SINK_EVERY}, le stesse cose sul
# pass 2 della condizione k3 (128 frame densi = regime diverso: se il fenomeno
# è del modello deve comparire uguale), e ~${DUMP_TOTAL} dump completi con le
# heatmap intere e i FRAME veri delle celle calde e fredde:
#     uv run python scripts/sink_heatmaps.py ${DUMP_ROOT}/lvbench_topk-<jobid>/shard_0 --all
# ⚠️ I dump di uno shard sono i suoi PRIMI sample, e il loader LVBench emette
# le domande raggruppate per video senza mescolare: coprono pochi video. Per
# un campione vario usa i dump della probe, che è mescolata.
#
# LETTURA: gli 8 shard condividono `wandb.group=lvbench-topk`. Si sommano i
# `true_count` per il pooling (evals/base.py). Nel summary ogni condizione ha
#   mcq_accuracy_cond_<nome>            accuracy della condizione
#   mcq_accuracy_cond_<nome>_base_true  accuracy sui sample che la BASELINE
#                                       indovinava → r_break = 1 − questo
#   mcq_accuracy_cond_<nome>_base_false accuracy sui sample che sbagliava
#                                       → r_fix = questo
# e r_fix/r_break sono ciò che serve a scegliere il quantile del gate.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

PASS2="${PASS2:-128}"
# Le statistiche dei canali pesano ~40 KB per sample: su 1549 sample sarebbero
# ~60 MB di payload Weave per il solo pass 1. Servono aggregate, quindi se ne
# prende una ogni 4 (~390 sample, più che abbastanza per istogrammi e quantili).
SINK_EVERY="${SINK_EVERY:-4}"
DUMP_TOTAL="${DUMP_TOTAL:-40}"
DUMP_LIMIT="${DUMP_LIMIT:-$(( (DUMP_TOTAL + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))}"
DUMP_ROOT="${DUMP_ROOT:-/work/tesi_avalenza/topk_dumps}"

NAME="topk"
GROUP=lvbench-topk
DUMP_DIR="${DUMP_ROOT}/lvbench_${NAME}-${SLURM_ARRAY_JOB_ID}/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${DUMP_DIR}"

echo "run: ${NAME}  shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"
echo "pass2_nframes: ${PASS2}  sink_stats_every: ${SINK_EVERY}  dump: ${DUMP_LIMIT} → ${DUMP_DIR}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    model.fix_videoframes_resize=true \
    dataset=lvbench \
    dataset.nframes=512 \
    dataset.max_pixels=50176 \
    dataset.min_pixels=3136 \
    strategy=topk_resample \
    strategy.pass2_nframes="${PASS2}" \
    strategy.sink_stats_every="${SINK_EVERY}" \
    strategy.dump_dir="${DUMP_DIR}" \
    strategy.dump_limit="${DUMP_LIMIT}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[lvbench,qwen3_vl_2b,topk_resample,fullset] \
    "$@"
