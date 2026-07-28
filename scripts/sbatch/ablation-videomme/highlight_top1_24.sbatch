#!/bin/bash
#SBATCH --job-name=ablation-highlight_top1_24
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

# Ablation entropy/shift/zoom — arm 6/6: EVIDENCE HIGHLIGHTING invece di
# ricampionamento (`strategy=attention_highlight`, adattamento al video di
# Look Twice, arXiv:2604.01280 — vedi strategies/attention_highlight.py).
#
# La domanda dell'arm: basta DIRE al modello dove guardare, senza dargli un
# solo frame nuovo? Il pass 2 vede ESATTAMENTE gli stessi 24 frame del pass
# 1; l'unica differenza è una frase prependuta al prompt, del tipo
#   "Focus on the part of the video between 880 and 1100 seconds: that is
#    where the visual evidence relevant to the question is."
# con l'intervallo ricavato dalla cella a massa di attenzione (sink-filtrata)
# più alta del pass 1.
#
# L'intervallo È la cella, non un intorno stimato: a nframes=24 le celle sono
# t=12 (merge ×2 di Qwen2.5-VL) e una cella copre `durata/12`, cioè 220 s su
# un Video-MME long. Indicare un istante puntuale ("990.0 s") comunicherebbe
# una precisione tre ordini di grandezza sopra quella del segnale. Larghezze
# tipiche del puntatore: ~6 s sui short, ~45 s sui medium, ~220 s sui long.
#
# FALLBACK AUTOMATICO A UNITÀ RELATIVE: la M-RoPE spazia le celle di
# `tokens_per_second * int(second_per_grid_ts)` (modeling_qwen2_5_vl.py:1125)
# e quel troncamento a intero azzera l'intervallo quando una cella dura meno
# di un secondo — cioè, a nframes=24, sui video sotto i 12 s: lì tutte le
# celle finiscono alla stessa posizione temporale e parlare al modello in
# secondi non significa niente. La strategy se ne accorge e passa a
# percentuali della durata. Riguarda ~1% della fascia short di Video-MME; il
# campo `highlight_units` nel log dice per ogni sample quale forma è stata
# usata, quindi la fascia è isolabile a posteriori. In regime frame-doubling
# la soglia raddoppierebbe a 24 s (~12% degli short), motivo per cui questo
# arm resta in regime normale.
#
# `strategy.top_k=1`: si indica UN solo istante — la stessa cella su cui
# zooma il ramo `zoom_peak` di entropy_zoom_24. È il confronto più pulito
# possibile fra i due interventi: stessa cella selezionata dallo stesso
# segnale, un arm la ISOLA (zoom, il modello perde il resto del video),
# l'altro la SEGNALA (highlight, il modello vede tutto ma sa dove guardare).
#
# `strategy.always_highlight=false`: gate d'entropia IDENTICO agli altri arm
# (`entropy_threshold=0.7`, il default del preset — stessa condizione
# `answer_entropy > soglia`), quindi si evidenzia solo sul ~72% di sample
# che lo supera. È ciò che rende l'arm confrontabile in DUE direzioni:
#   - con `baseline_24`, sull'accuracy pooled dei 2700 sample (l'effetto
#     netto dell'intervento sul dataset intero);
#   - con `entropy_shift_24`/`entropy_zoom_24`, sample-per-sample sul
#     sottoinsieme gated, che è lo STESSO per costruzione (vedi sotto).
# LoT il gate non ce l'ha: per il regime del paper rilanciare con
#   sbatch ... highlight_top1_24.sbatch strategy.always_highlight=true
# (il `"$@"` in coda inoltra gli override).
#
# CHECK SUL GATE: il pass 1 di questo arm è lo stesso di
# entropy_shift_24/entropy_zoom_24 — stesso modello, stessi 24 frame, stesso
# prompt, greedy — quindi `n_samples_highlighted_true` deve cadere sullo
# stesso ~73% (1975/2700 per shift, 1969/2700 per zoom sul full).
# NON aspettarsi il conteggio identico al sample: shift e zoom, che hanno
# pass 1 identico, differiscono già di 6 sample su 2700. È
# non-determinismo numerico fra GPU diverse (i run vanno su L40S, A40,
# RTX6000, A5000) che fa oscillare i sample con `answer_entropy` a ridosso
# della soglia 0.7. Uno scostamento di qualche unità è fisiologico; una
# differenza di punti percentuali no, e vorrebbe dire che è cambiato
# qualcosa nel pass 1 e non nell'intervento.
#
# Job array a 3 shard (900 sample ciascuno). `--time=05:00:00`, stesso
# valore degli altri arm signal-driven, ricavato dai loro tempi misurati.
# Scomposizione dei costi per sample (misure dell'array 74423/74427 su
# `nullazzo`, Quadro RTX 6000 24G, la card più lenta del constraint):
#   baseline_24        D + G                 = 7.24 s   (decode + generate)
#   entropy_shift_24   D + C + 0.72*(D + G)  = 13.53 s  (+ capture, + pass 2 sul 72%)
#   questo arm         D + C + 0.72*G        = ?
# dove D = decodifica video, C = prefill di cattura attenzione, G = generate.
# Da shift: D + C = 13.53 - 0.72*7.24 = 8.32 s. Questo arm fa esattamente ciò
# che fa shift MENO la ridecodifica del video per il pass 2 (il puntatore
# riusa i frame del pass 1), cioè -0.72*D: con D~2 s sono ~12.1 s/sample
# ≈ 3h02/shard, quindi 5h danno ~1.65× di margine. È l'arm signal-driven
# più economico dei quattro. Con `always_highlight=true` il pass 2 passa dal
# 72% al 100% dei sample (~+1.4 s/sample): il budget resta valido.
#
# ACCURACY: i 3 shard producono 3 run wandb separate — l'accuracy dell'arm è
# quella POOLED (Σ corretti / Σ sample), non la media delle tre.
#
# `--constraint` aperto anche alle 24G (RTX6000/A5000) — vedi la nota estesa
# in entropy_shift_24.sbatch, incluso l'avviso che un OOM si manifesta come
# run `finished` senza `mcq_accuracy`, non come job fallito. Il carico di
# VRAM qui è quello di entropy_shift_24 meno il secondo `Video()`: il pass 2
# riusa i tensori visivi già costruiti, quindi il picco non è più alto.
#
# BREAKDOWN short/medium/long e per `task_type`: nel summary di fine eval,
# come per gli altri arm (`evals.base.mcq_accuracy` + `utils.obs.log_eval_summary`).
#
# METRICHE SPECIFICHE DI QUESTO ARM, dal solito meccanismo di BREAKDOWN_KEYS:
#   n_samples_highlighted_true     su quanti sample è stato emesso il puntatore
#   mcq_accuracy_highlighted_true  accuracy su quelli (= "acc | high_ans_entropy"
#                                  degli altri arm, direttamente confrontabile)
#   mcq_accuracy_highlighted_false accuracy sui sample accettati al pass 1
# `highlighted=false` mescola due cose: gate chiuso (la stragrande maggioranza)
# e gate aperto ma puntatore scartato per segnale di attenzione nullo. Il
# secondo caso si distingue dal campo `highlight_skip_reason` e ci si aspetta
# sia rarissimo; se `n_samples_highlighted_true` è sensibilmente sotto il
# `n_samples_resampled_true` degli arm gated, è lì che va guardato.
#
# La frase esatta mandata al modello è loggata per sample nel campo `pointer`
# (query Weave): serve al win/loss contro entropy_zoom_24, che sui sample
# gated ha selezionato la STESSA cella — lì il confronto è "isolare vs
# segnalare lo stesso istante", e i sample dove i due arm divergono sono
# esattamente quelli da leggere.
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
    strategy.always_highlight=false \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-highlight_top1_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,highlight_top1_24] \
    "$@"
