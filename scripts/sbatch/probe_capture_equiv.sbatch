#!/bin/bash
#SBATCH --job-name=probe-capture-equiv
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

# Verifica del fix alla cattura d'attenzione (models/qwen_attn.py: solo i layer
# centrali, accumulo sul device, un `.cpu()` a fine forward) prima della run di
# segnali su LVBench a 512 frame. Disegno nella docstring di
# scripts/probe_capture_equiv.py.
#
# Il riferimento è la cattura al commit REF_COMMIT (default 22be7cc), letta con
# `git show`: il clone su cui gira il job deve avere quel commit.
#
# ⚠️ `--constraint` SENZA schede Turing (sm_75), come tutti gli sbatch qwen3.
# Su all_usr_prod non si può restringere alle 45G: il job può finire su una
# A5000 da 24G. Il picco di memoria della parte a 512 frame va quindi letto
# insieme a `gpu_name` nel config wandb — e se la A5000 regge, la run regge
# su tutte e tre.
#
# Il modello gira con `fix_videoframes_resize=true` (impostato dallo script):
# i frame passano come `VideoFrames`, come nel campionamento a coppie della
# run, ma con lo stesso resize del path `Video`.
#
# COSTO: 5 sample x 2 forward a 24 frame + 3 forward a 512 frame (ref, new,
# profiler), più l'estrazione PNG dai video lunghi: ~20-40 min.
#
# COSA GUARDARE (summary wandb, project lvbench, group probe-capture-equiv):
#   1. equiv_ok=true, equiv_n_skip=0: attenzione, sink map, entropia e ranking
#      delle celle coincidono col vecchio path;
#   2. equiv_worst_rel_attn / equiv_worst_rel_sink: atteso rumore di
#      arrotondamento (<1e-5), non differenze strutturali;
#   3. big_ok, big_grid=5x9 e big_n_vis=11520 su un video 16:9 (6x8 e 12288
#      su 4:3): a 512 frame il processor NON ha ridotto la risoluzione (tetto
#      di 25.2M px totali);
#   4. big_lat_ref vs big_lat_new e big_peak_gpu_gb_*: il guadagno del fix;
#   5. sdpa_math_calls=0: il decoder testuale non ricade sul backend math.
#      Se >0, ogni layer materializza [16, S, S] e la run va ripensata.
#
# Uso:
#     sbatch scripts/sbatch/probe_capture_equiv.sbatch
#     NSAMPLES=10 BIG_NFRAMES=768 sbatch scripts/sbatch/probe_capture_equiv.sbatch

set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1
export HF_HOME=/work/tesi_avalenza/hf

NSAMPLES="${NSAMPLES:-5}"
SEED="${SEED:-0}"

echo "n_samples: ${NSAMPLES}  seed: ${SEED}  ref: ${REF_COMMIT:-22be7cc}  big_nframes: ${BIG_NFRAMES:-512}"

uv run python scripts/probe_capture_equiv.py "${NSAMPLES}" "${SEED}"
