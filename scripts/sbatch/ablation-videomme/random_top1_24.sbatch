#!/bin/bash
#SBATCH --job-name=ablation-random_top1_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=05:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# Ablation entropy/shift/zoom — arm 7/7: CONTROLLO di highlight_top1_24.
# Identico in tutto (stesso gate, stesso formato di puntatore, stesso
# numero di pass, stesso costo) tranne UNA cosa: la cella indicata è
# estratta A SORTE fra le 12 invece di essere quella a massa di attenzione
# più alta (`strategy.cell_select=random`).
#
# A COSA SERVE. Ad attribuire l'eventuale guadagno di highlight_top1_24. Se
# indicare una cella a caso rende quanto indicare il picco, il contributo
# dell'attenzione è nullo e si sta misurando solo l'effetto del prompt —
# cioè "dare al modello un'istruzione di focalizzazione, qualunque essa
# sia". Senza questo controllo un risultato positivo dell'arm non sarebbe
# interpretabile.
#
# PERCHÉ SERVE DAVVERO, non per scrupolo formale. Il probe da 40 sample
# (run `2dodrndg`, nframes=24, always_highlight=true) ha mostrato che il
# picco di attenzione è in buona parte POSIZIONALE, non semantico:
#   - cade sull'ULTIMA cella (la 11) nel 28% dei casi, 3.4x il livello di
#     caso (8.3%);
#   - cade su una delle due celle estreme (0 o 11) nel 40%, contro il 17%
#     atteso a caso;
#   - `top1_pct` mediano 15.7% contro un livello di caso dell'8.3%: il picco
#     supera il caso solo di ~1.9x, cioè l'attenzione è concentrata
#     debolmente.
# Il meccanismo plausibile è la recency causale: le righe-query sono i token
# della domanda, che nella sequenza stanno DOPO il blocco visivo, e in un
# modello causale i token vicini ricevono sistematicamente più attenzione —
# l'ultima cella è la più vicina. Il filtro sink non lo corregge, perché
# rimuove i token con alta attivazione sui canali-sink, che è un effetto
# diverso. Quindi su oltre un quarto dei sample highlight_top1_24 dice al
# modello "guarda alla fine del video" a prescindere dal contenuto.
#
# COME SI LEGGE IL CONFRONTO.
#   random ~ attention  → l'attenzione non contribuisce; qualunque effetto
#                         viene dal prompt. Il filone dell'evidence
#                         highlighting guidato da questo segnale è chiuso,
#                         e la strada diventa migliorare il SEGNALE (es.
#                         restringere le righe-query ai token dell'entità
#                         come fa il paper, `question_rows` in attn_core).
#   attention > random  → il segnale porta informazione; a quel punto vale
#                         un secondo controllo, con celle estratte dalla
#                         stessa distribuzione marginale dell'attenzione
#                         (28% sull'ultima, ecc.), per separare "localizza
#                         l'evidenza" da "ha un buon prior posizionale".
#   attention < random  → il puntatore guidato è ATTIVAMENTE dannoso: il
#                         picco sbagliato è peggio di un suggerimento
#                         neutro. Coerente con entropy_zoom_24, che usa la
#                         STESSA peak_cell per zoomare ed è l'unico arm
#                         sotto baseline_24 (-0.44).
#
# DILUIZIONE ATTESA: con t=12 celle e top_k=1 il caso pesca proprio la cella
# dell'attenzione ~1 volta su 12 (8.3%). Su quei sample i due arm sono
# identici per costruzione e non contribuiscono alla differenza: il
# confronto è di fatto su ~92% del campione. Il tasso reale di coincidenza
# è ricavabile dai log — `highlight_cells` (celle indicate) e `peak_cell`
# (scelta dell'attenzione, loggata ANCHE in modalità random) sono entrambi
# nel dict di ogni sample.
#
# RIPRODUCIBILITÀ DELL'ESTRAZIONE: la cella non viene da un RNG globale (che
# dipenderebbe dall'ordine di esecuzione, diverso fra i 3 shard e non
# garantito con `Evaluation` concorrente) ma da un seed derivato da
# `(random_seed, video_path, prompt)`. La cella è quindi una funzione pura
# del sample: stabile fra shard, fra rilanci e fra macchine, e joinabile
# per-sample con gli altri arm. `strategy.random_seed=0` esplicito qui
# sotto: cambiarlo ripete il controllo con un'altra estrazione.
#
# Job array a 3 shard (900 sample ciascuno). `--time=05:00:00`: il costo è
# identico a highlight_top1_24 — stessa pipeline, stesso numero di forward,
# cambia solo quale indice di cella finisce in una f-string. Misura di
# riferimento dal probe: 14.62 s/sample su `nullazzo` (RTX 6000 24G, la card
# più lenta del constraint) con il pass 2 su TUTTI i sample; con il gate al
# ~73% scendono a ~13.1 s/sample, cioè ~3h17/shard e ~1.5x di margine.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy dell'arm è
# quella POOLED (Σ corretti / Σ sample), non la media delle tre.
#
# ⚠️ Se cambi un qualunque parametro qui, cambialo anche in
# highlight_top1_24.sbatch: i due arm hanno senso solo se differiscono per
# `cell_select` e nient'altro.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    strategy.top_k=1 \
    strategy.cell_select=random \
    strategy.random_seed=0 \
    strategy.always_highlight=false \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-random_top1_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,random_top1_24,controllo] \
    "$@"
