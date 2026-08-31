#!/bin/bash
#SBATCH --job-name=probe-oracle-ceiling
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# I SOFFITTI e gli ARM REALISTICI del ricampionamento su LVBench.
# Sette condizioni per sample:
#   baseline / peak / oracle   marcatura (puntatore testuale) — CHIUSA
#   zoom                        oracolo del ricampionamento (+16 pp)
#   zoom_peak                   24 frame nella cella di PICCO
#   zoom_topk                   24 frame divisi fra le TOP-K celle (disgiunte)
#   zoom_hybrid                 12 frame globali + 12 nella cella di picco
# Disegno completo in scripts/probe_oracle_ceiling.py.
#
# PERCHÉ QUESTE TRE. Il giro 94348 ha stabilito che il guadagno vive nel
# ricampionamento (zoom 48% contro baseline 32%) ma che il segnale localizza
# la finestra solo nel 31% dei sample. Allargare attorno al picco NON è la
# risposta: quando il picco sbaglia, sbaglia di 4-5 celle (distribuzione
# quasi piatta), quindi ±k celle compra copertura di puro caso e perde
# risoluzione (2.4 → 0.8 frame nell'evidenza a ±1). Restano due leve:
# il RANKING (top-k disgiunto: la finestra è nel top-k anche quando non è
# top-1?) e il DEGRADO GENTILE (hybrid: metà budget resta globale, così un
# picco sbagliato non costa il contesto — serve perché nei 23 sample dove il
# campione uniforme colpiva GIÀ l'evidenza lo zoom peggiora, 48% → 43%).
#
# `hit@k` è loggato GRATIS dal ranking (nessun forward): dice se il top-k
# ha senso prima ancora di guardarne l'accuracy. Se hit@3 ≈ hit@1, il
# ranking non porta ridondanza e `zoom_topk` è morto in partenza.
#
# PRIMO GIRO (93614, senza zoom, 100 sample): baseline 32%, peak 29%,
# oracle 30% — il canale di MARCATURA non ha soffitto. peak_in_window 30%
# (~2x chance, coerente con la sonda (1)): segnale debole ma reale. Con 24
# frame su video di ore la finestra quasi mai contiene un frame campionato:
# la condizione `zoom` misura se portarci i FRAME (non il puntatore) paga.
#
# A parità di SEED condivide i sample con probe_peak_in_window (stessa
# selezione deterministica): i risultati si incrociano per id.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
#
# COSTO: 100 sample x (5 decode in PNG + 8 forward brevi a 24 frame)
# ≈ 45-60 s/sample stimati (23 s/sample il giro a 4 condizioni) → ~1.5 h.
# 4 h di margine.
#
# COSA GUARDARE (summary wandb, project lvbench, group probe-oracle-ceiling):
#   1. n_skip ~0, n_fallback basso (risposte parseabili);
#   2. hit_at_1..5: se hit@3 ≈ hit@1 il ranking non ha ridondanza;
#   3. acc_zoom_peak / acc_zoom_topk / acc_zoom_hybrid contro acc_baseline —
#      SONO LA MISURA (arm realizzabili), coi flip *_recovered/*_broken;
#      acc_zoom (48%) resta il soffitto di riferimento;
#   4. n_hybrid_dedup: quante volte gli span sovrapposti hanno ridotto il
#      budget sotto nframes (atteso basso; se alto, la quota va ripensata);
#   5. peak_in_window_rate: cross-check con la sonda (1) sullo stesso
#      rowset (atteso ~31%);
#   6. la Table `samples` per l'ispezione per-sample (`ranked_top5`,
#      `hit_at_k` per sample).
#
# Uso:
#     sbatch scripts/sbatch/probe_oracle_ceiling.sbatch
#     NSAMPLES=60 SEED=0 ROWSET=entity TOPK=2 sbatch scripts/sbatch/probe_oracle_ceiling.sbatch

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

NSAMPLES="${NSAMPLES:-100}"
SEED="${SEED:-0}"

echo "n_samples: ${NSAMPLES}  seed: ${SEED}  nframes: ${NFRAMES:-24}  rowset: ${ROWSET:-question}  topk: ${TOPK:-3}"

uv run python scripts/probe_oracle_ceiling.py "${NSAMPLES}" "${SEED}"
