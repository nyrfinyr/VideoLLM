#!/bin/bash
#SBATCH --job-name=coarse_to_fine-lvbench
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=08:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Arm `coarse_to_fine` su LVBench: zoom temporale iterativo guidato da
# attenzione (DOVE scendere) ed entropia (QUANDO fermarsi), con backtracking
# sul ranking accumulato. Disegno completo in strategies/coarse_to_fine.py.
#
# PERCHÉ LVBENCH E NON VIDEO-MME. È il benchmark dove il soffitto è stato
# MISURATO (docs/oracolo_lvbench.md): ricampionare dentro la finestra vera
# vale 48% contro 32% di baseline (+22/−6, p=0.004), mentre marcarla non
# paga (30%). Video-MME ha video più corti, dove il campionamento uniforme
# perde meno evidenza e il margine da recuperare è più piccolo.
#
# PERCHÉ ZOOM E NON PIÙ FRAME. Su LVBench (durata mediana 71 min, finestra
# d'evidenza mediana 30 s) 24 frame uniformi colpiscono la finestra nel 45%
# dei sample e 48 frame nel 51%: la copertura cresce LINEARMENTE nel budget,
# servirebbero ~128 frame per il 90%. Lo zoom invece guadagna risoluzione
# MOLTIPLICATIVA: 24 frame in una cella (1/12 del video) hanno spacing 15 s
# invece di 177 s.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
#
# I TRE LANCI, tutti necessari (il primo da solo non è leggibile):
#
#   1. ARM              sbatch scripts/sbatch/coarse_to_fine_lvbench.sbatch
#   2. CONTROLLO random SELECT=random sbatch ...
#      Stessa struttura, stesso numero di passi, stesso gate, stesso budget:
#      cambia SOLO da dove viene la regione. È l'unico confronto che isola il
#      contributo del segnale da quello del "guardare più volte in posti
#      diversi" — la lezione dei tre arm di marcatura, dove il controllo
#      random pareggiava sempre il treatment.
#   3. BASELINE a budget pari
#        BASE48=1 sbatch ...
#      `uniform` con nframes=48: se l'arm pareggia QUESTA, il guadagno viene
#      dal budget e non dal segnale. Senza, un +N pp contro la baseline a 24
#      non dice nulla.
#
# PROBE prima del fullset (100 sample, ~1 h):
#     LIMIT=100 sbatch --array=0 --time=02:00:00 \
#         scripts/sbatch/coarse_to_fine_lvbench.sbatch
#
# COSA GUARDARE, IN QUEST'ORDINE (driver /wandb):
#   1. `n_steps` per sample: la distribuzione dice quanto costa davvero
#      l'arm. Se quasi tutti fanno max_steps, il gate non sta filtrando e il
#      costo è 4x la baseline;
#   2. `gate_closed`: quota di sample in cui l'entropia è scesa sotto soglia.
#      Su Qwen3-VL il gate separa 37.86/74.88 (docs/260901-comments.md §2) —
#      mediocre: se `gate_closed` è ~0 o ~1, la soglia va ricalibrata PRIMA
#      di leggere l'accuracy;
#   3. `final_span_sec` / `final_depth`: quanto si è sceso davvero. depth=0
#      su tutti = l'arm non ha mai zoomato (è la baseline con più forward);
#   4. accuracy arm contro CONTROLLO random (non contro la baseline);
#   5. accuracy arm contro BASELINE a 48 frame — il test "è solo budget?";
#   6. `pred_fallback` ~0 e `model_latency_mean` per ritarare la walltime.
#
# COSTO: fino a 4 forward con cattura + 3 decode/estrazione PNG per sample.
# Su LVBench i video sono lunghi e il decode domina: ~40-60 s/sample stimati
# → ~1 h per una probe da 100. Il fullset dipende da quanti sample ha lo
# shard: 8 h con margine, RITARARE su `model_latency_mean` della probe.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

SELECT="${SELECT:-attention}"
STEPS="${STEPS:-3}"
GATE="${GATE:-0.7}"
QROWS="${QROWS:-question}"

# Baseline a budget pari: `uniform` con il doppio dei frame. Serve a
# distinguere "il segnale guida bene" da "ha solo visto più video".
if [ -n "${BASE48:-}" ]; then
  ARM_ARGS=( strategy=uniform dataset.nframes=48 )
  NAME="baseline_48"
else
  ARM_ARGS=(
    strategy=coarse_to_fine
    dataset.nframes=24
    strategy.max_steps="${STEPS}"
    strategy.entropy_threshold="${GATE}"
    strategy.region_select="${SELECT}"
    strategy.query_rows="${QROWS}"
  )
  NAME="c2f_s${STEPS}_g${GATE}"
  [ "${SELECT}" != "attention" ] && NAME="${NAME}_${SELECT}"
fi

PROBE_ARGS=()
GROUP=lvbench-coarse-to-fine
if [ -n "${LIMIT:-}" ]; then
  PROBE_ARGS=( limit="${LIMIT}" shuffle=true )
  GROUP=lvbench-coarse-to-fine-test
  NAME="${NAME}_probe${LIMIT}"
  echo "PROBE: ${LIMIT} sample — verifica gate/discesa/costo, non l'accuracy"
fi

echo "arm: ${NAME}  shard: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  group: ${GROUP}"

uv run python main.py \
    model=qwen3_vl_2b_attn \
    dataset=lvbench \
    "${ARM_ARGS[@]}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[lvbench,qwen3_vl_2b,coarse_to_fine,${NAME}] \
    "$@"
