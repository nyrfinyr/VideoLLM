#!/bin/bash
#SBATCH --job-name=probe-additive-oracle
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# IL GATE del disegno ADDITIVO: i frame mirati si AGGIUNGONO alla base invece
# di sostituirla. Prima di scrivere l'arm (k5/k10 col puntatore vero) si
# misura col puntatore PERFETTO se la densità dentro la finestra vale ancora
# qualcosa a questa base. Disegno e lettura: scripts/probe_additive_oracle.py.
#
# TRE CONDIZIONI APPAIATE per sample, un solo decode del video:
#   base256    128 celle a coppie (gap 2 s)                       ~31 s
#   oracle512  base + 256 frame nelle celle della finestra vera   ~57 s
#   uniform512 256 celle a coppie, stesso budget di oracle512     ~57 s
# ≈ 145 s/sample → 50 sample per shard ≈ 2 h (walltime 4 h di margine).
#
# QUALI SAMPLE: --array=0-1 con LIMIT=50 dà ESATTAMENTE i 100 sample della
# probe topk 108729 (shuffle seed 42 → shard strided → primi 50 per shard,
# unione = i primi 100), quindi i risultati si incrociano per id con tutto
# quello che abbiamo già misurato su quello slice.
# ⚠️ Cambiare --array o LIMIT cambia la selezione: i confronti con le run
# precedenti cadono.
#
# ⚠️ --constraint SENZA Turing (Qwen3-VL ci è 6x più lento).
# ⚠️ La finestra annotata entra SOLO nella scelta dei frame, mai nel prompt.
#
# Uso:
#     sbatch scripts/sbatch/probe_additive_oracle.sbatch
#     LIMIT=2 DUMP_TS=1 sbatch --array=0 --time=00:30:00 scripts/sbatch/probe_additive_oracle.sbatch
#     ADDED=128 sbatch scripts/sbatch/probe_additive_oracle.sbatch
#
# PRIMA ESECUZIONE: lanciare con LIMIT=2 --array=0. Costa mezz'ora e dice se
# il percorso dei frame regge sul processor vero (la logica è verificata solo
# con decord finto e smoke in locale).
#
# COSA GUARDARE, IN QUEST'ORDINE (le due shard sono due run wandb: i CONTEGGI
# si sommano, le accuracy si ricalcolano dai totali):
#     uv run python .claude/skills/wandb/driver.py probe_additive_oracle-<jobid> --summary-only
#   1. CORRETTEZZA:
#      a. n_samples = 50 per shard, n_skip = 0;
#      b. mean_n_added_kept ≈ 256 e mean_n_collision ≈ 0 — se le collisioni
#         sono tante, le celle sono troppo strette per il budget (video corti
#         o fps basso) e i frame aggiunti sono meno di quanto crediamo;
#      c. mean_n_added_in_raw_window > 20: è la quantizzazione alla cella che
#         si paga: su una finestra mediana (20 s su 2 celle) attesi ~75;
#      d. n_fallback_* = 0 (come nella probe topk);
#      e. nessun traceback nel .err.
#   2. COSTO: latency_base256 / latency_oracle512 / latency_uniform512 —
#      servono a dimensionare l'arm vero (che ha base + 2 condizioni).
#   3. SOLO ALLA FINE i numeri, e nell'ordine giusto:
#      delta_uniform512_to_oracle512 (il gate: a parità di frame, mirare paga?)
#      delta_base256_to_oracle512   (aggiungere paga?)
#      delta_base256_to_uniform512  (i frame uniformi oltre 256 servono?)
#      più il breakdown acc_*_stretta / acc_*_larga: sulle finestre larghe
#      (>10% del video) l'oracolo è quasi gratis e NON va letto come soffitto.
#      Riferimento esterno: 45% a 512 frame uniformi sugli stessi 100 sample
#      (run qwen3_vl_2b_attn-topk-108729_0_probe100).
#   4. SE delta_uniform512_to_oracle512 ≈ 0: l'arm additivo non va scritto —
#      la densità in-finestra non è la risorsa scarsa, e il risultato è di
#      tesi tanto quanto un guadagno.

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

export LIMIT="${LIMIT:-50}"
export BASE="${BASE:-256}"
export ADDED="${ADDED:-256}"
export UNIFORM="${UNIFORM:-512}"
export GAP="${GAP:-2.0}"
export DUMP_TS="${DUMP_TS:-0}"

echo "PROBE additiva (ORACOLO): base=${BASE} + aggiunti=${ADDED} contro uniform=${UNIFORM}"
echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}, ${LIMIT} sample per shard"

uv run python scripts/probe_additive_oracle.py "$@"
