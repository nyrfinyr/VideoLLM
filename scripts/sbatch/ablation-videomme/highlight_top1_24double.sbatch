#!/bin/bash
#SBATCH --job-name=ablation-highlight_top1_24double
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=10:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom — arm highlight in regime FRAME-DOUBLING.
#
# 24 timestamp distinti, ognuno passato al modello DUE VOLTE (48 frame in
# ingresso, `nframes` resta il numero di timestamp distinti — vedi
# `SamplingBudget.double_frames` in strategies/base.py). Il merge temporale
# ×2 di Qwen2.5-VL fonde ogni frame con SE STESSO, quindi la griglia
# visiva ha **24 celle da 1 frame** invece di 12 celle da 2. Stessa
# copertura temporale di highlight_top1_24, token visivi raddoppiati.
#
# PERCHÉ QUESTO ARM. La sonda `antipeak`
# (`docs/sonda_sensibilita_puntatore_antipeak.md`, gruppo wandb
# video_mme-antipeak-probe) ha stabilito che il canale-prompt È VIVO:
# puntare di proposito nel posto sbagliato rompe il 6.80% dei sample già
# corretti al pass 1, contro lo 0.34% del controllo senza puntatore
# (danno 6.46 pp, z=4.24, p<0.0001). Il modello ESEGUE l'indicazione
# temporale. Quindi il nulla di highlight_top1_24 (54.63%, −0.41 vs
# baseline_24) non è "il modello ignora il puntatore" ma "il puntatore
# indica male", ed è il SEGNALE la cosa da aggredire.
#
# La prima leva sul segnale è la RISOLUZIONE, ed è questo arm: a 12 celle
# il puntatore indica 1/12 del video — su un long di Video-MME sono ~50
# secondi, che come "localizzazione" è grossolano. A 24 celle si dimezza a
# ~25 s. È anche la leva più economica delle due candidate: nessun codice
# nuovo, solo `--double-frames`.
# (L'altra leva, la QUALITÀ del picco via righe-query ristrette ai token
# dell'entità — `question_rows` in utils/attn_core.py — va misurata prima
# offline sul probe da 40 sample, non merita una eval finché il bias
# posizionale non scende.)
#
# CONTRO CHI SI CONFRONTA: `baseline_24double` (56.93%), NON `baseline_24`
# (55.04%). baseline_24double è lo stesso regime doubled senza puntatore,
# cioè il controllo compute-matched: confrontarsi con baseline_24
# attribuirebbe al puntatore il +1.89 che viene dal solo raddoppio dei
# token visivi. NB: baseline_24double è oggi l'arm MIGLIORE di tutta la
# tabella, quindi è un riferimento severo.
#
# I DUE ARM IN UN FILE SOLO. `CELL_SELECT` sceglie fra il vero arm e il suo
# controllo:
#     sbatch scripts/sbatch/ablation-videomme/highlight_top1_24double.sbatch
#     CELL_SELECT=random sbatch scripts/sbatch/ablation-videomme/highlight_top1_24double.sbatch
# Il controllo random è OBBLIGATORIO per interpretare il risultato, con la
# stessa logica di random_top1_24: se indicare una cella a caso fra le 24
# rende quanto indicare il picco, la risoluzione più fine non ha comprato
# nulla e il problema è tutto nella qualità del segnale.
# Un file solo e non due (come highlight_top1_24/random_top1_24, che
# portano un ⚠️ "se cambi un parametro qui cambialo anche là") perché così
# i due arm NON POSSONO divergere per errore su nient'altro che
# `cell_select`: è una garanzia strutturale invece che un promemoria.
#
# ⚠️ UNITÀ DEL PUNTATORE — la cosa da guardare per prima. Il collasso
# dell'orologio M-RoPE scatta quando una cella dura meno di un secondo,
# cioè `window_sec < t_cells`: la soglia RADDOPPIA da 12 s a 24 s passando
# a 24 celle, quindi una quota molto più grossa della fascia short cade in
# unità FRAZIONARIE ("between 30% and 40% of its duration") invece che in
# secondi. `highlight_units` è ora una chiave di breakdown (evals/base.py),
# quindi nel summary trovi `mcq_accuracy_highlight_units_seconds` e
# `..._fraction`: vanno lette PRIMA del delta complessivo, altrimenti un
# calo si attribuisce alla risoluzione quando invece viene dal cambio di
# formato della frase.
#
# COSTO E `--time=10:00:00`, dimensionato sulla card PIÙ LENTA del
# constraint (Quadro RTX 6000 24G), usando le latenze misurate su quella
# stessa card:
#   baseline_24        7.19 s/sample  (uniform, 1 pass, 12 celle)
#   highlight_top1_24 13.09 s/sample  (2 pass + capture, 12 celle) → 1.82x
#   baseline_24double 16.69 s/sample  (uniform, 1 pass, 24 celle)
# 16.69 × 1.82 ≈ 30.4 s/sample → ~7h35 per shard da 900. Le 10h danno
# ~1.3x di margine. Su A40/L40S sarebbe ~3h40, quindi il constraint largo
# scambia wall time per tempo di coda: se la coda sulle 45G è corta,
# `--constraint="gpu_A40_45G|gpu_L40S_45G"` da CLI dimezza la durata.
#
# `--constraint` APERTO ANCHE ALLE 24G, come highlight_top1_24. La cattura
# NON è un problema di VRAM qui, verificato misurando le forme reali su
# questo dataset (nframes=24, max_pixels=151200):
#                    t   n_vis      S   picco GPU transiente   stash
#   normale         12    2160   2248     9.6 MB / layer      20.8 MB
#   doubled         24    4320   4408    18.9 MB / layer      41.7 MB
# Il picco è `attn_weights` fp32 [1, 16 teste, n_q=67, S] dentro
# `qwen25_attn_capture`, transitorio: liberato a fine layer. Lo stack
# [L, n_q, n_vis] su tutti i 36 layer esiste ma sta in RAM, non in VRAM —
# `models/qwen_attn.py` fa `.detach().cpu()` su ogni `_last_attn`. Quindi
# il doubling aggiunge ~9 MB transitori per layer: irrilevante su 24 GB.
# Prova diretta: baseline_24double è già girato su Quadro RTX 6000 24G
# (run 2128xily) con la stessa sequenza da 4408 token. Gli OOM dello sweep
# soglie erano a nframes=128, cioè S molto più grande — regime diverso, non
# un precedente applicabile qui.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy dell'arm
# è quella POOLED (Σ corretti / Σ sample), non la media delle tre.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

CELL_SELECT="${CELL_SELECT:-attention}"
case "${CELL_SELECT}" in
  attention) ARM=highlight_top1_24double ;;
  random)    ARM=random_top1_24double ;;
  *)
    echo "CELL_SELECT sconosciuto: ${CELL_SELECT}. Valide: attention, random" >&2
    exit 1
    ;;
esac
echo "arm: ${ARM} (cell_select=${CELL_SELECT})"

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    --double-frames \
    strategy.top_k=1 \
    strategy.cell_select=${CELL_SELECT} \
    strategy.random_seed=0 \
    strategy.always_highlight=false \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,${ARM},double_frames] \
    "$@"
