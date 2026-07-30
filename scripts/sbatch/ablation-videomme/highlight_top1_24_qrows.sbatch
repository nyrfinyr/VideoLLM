#!/bin/bash
#SBATCH --job-name=ablation-highlight_top1_24_qrows
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

# `highlight_top1_24` con il PICCO CALCOLATO SOLO SULLE RIGHE DELLA DOMANDA.
# Identico all'arm originale in ogni altro knob: cambia una cosa sola.
#
# IL DIFETTO CHE CORREGGE. `_capture_spans` (models/qwen_attn.py) definisce
# le righe-query come tutto ciò che segue l'ultimo `<|vision_end|>`, e
# `ranked_cells_from_attention` (utils/attn_core.py) ne fa la media. Su un
# prompt MCQ di Video-MME quello vuol dire domanda + le QUATTRO OPZIONI +
# "Answer with the letter of the correct option only." + il generation
# prompt. Misurato col tokenizer su un prompt tipico:
#     prompt intero   86 token   <- righe mediate dagli arm esistenti
#     solo domanda    21 token   <- 24%
# Cioè il 76% delle righe che formano il "picco d'attenzione della domanda"
# non è la domanda. `question_rows` (utils/attn_core.py) taglia esattamente
# quel resto, esisteva già, la usa la UI dell'attn_explorer e nessuna
# strategy la chiamava. Il knob `strategy.query_rows` la espone.
#
# PERCHÉ È IL PROSSIMO ESPERIMENTO. La sonda `antipeak` ha escluso che il
# canale sia morto (6.46 pp di danno, z=4.24) e il controllo dell'orologio
# M-RoPE ha escluso che le celle collassate lo spiegassero (−0.07 pp):
# resta che il SEGNALE è cattivo. Questa è la leva più economica sul
# segnale — nessun codice nuovo nel percorso caldo, nessun costo di
# inferenza aggiuntivo, solo meno righe da mediare.
#
# SUI 2700 E NON SU UN SUBSET, per la stessa ragione del controllo M-RoPE:
# qui le celle di confronto ESISTONO GIÀ, stesso set e stesso codice
# (gruppo `video_mme-ablation`), quindi questo arm costa 3 job e il
# confronto è appaiato per costruzione:
#     baseline_24                    55.04%  (1486/2700)
#     highlight_top1_24  (all rows)  54.63%  (1475/2700)   <- il paragone
#     random_top1_24                 54.78%  (1479/2700)
# Su un subset da 500 quei controlli andrebbero rifatti sullo stesso
# subset (3 job comunque) E l'errore standard su una DIFFERENZA fra due arm
# sarebbe ±3.1 pp contro scarti in gioco di 0.4-1.9 pp: si misurerebbe
# rumore. Su 2700 è ±1.35 pp.
#
# COSA GUARDARE, IN QUEST'ORDINE. L'accuracy pooled è lo strumento MENO
# sensibile: anche sui 2700 ±1.35 pp è al limite per un effetto da 1 pp.
# Il breakdown `correct_pass1_*` (evals/base.py) lo è molto di più, perché
# guarda le risposte che si MUOVONO invece del saldo:
#     highlight_top1_24 a `all` rows:  77 rotti / 76 recuperati  → netto -1
# 153 risposte cambiano per un saldo di zero: un puntatore che
# ridistribuisce invece di informare. Se le righe-domanda rendono il picco
# informativo, quel rapporto si sbilancia PRIMA che l'accuracy si muova.
# Quindi si legge:
#     1. n_correct_correct_pass1_false  (recuperati)  vs  77-x (rotti)
#     2. mcq_accuracy pooled            vs  54.63%
#     3. mcq_accuracy_duration_*        (il puntatore serve sui long?)
#
# `QROWS` sceglie la selezione, così i due arm non possono divergere per
# altro (stessa logica del file unico di highlight_top1_24double):
#     sbatch scripts/sbatch/ablation-videomme/highlight_top1_24_qrows.sbatch
#     QROWS=qopts sbatch scripts/sbatch/ablation-videomme/highlight_top1_24_qrows.sbatch
# `qopts` (via solo il boilerplate, opzioni incluse) è il terzo gradino
# della scala, ma probabilmente non vale i suoi 3 job: sul probe locale da
# 4 video `all` e `qopts` sono risultati praticamente indistinguibili
# (top1_pct 15.5 vs 15.4, gini 0.208 vs 0.207), cioè il boilerplate da solo
# non sembra essere il problema. Lanciarlo solo se `question` si muove e
# si vuole sapere quale metà della differenza viene dalle opzioni.
#
# ⚠️ REGIME `meta=false` (default del preset). NON è una svista: le celle
# di confronto qui sopra sono girate così, e il controllo dell'orologio ha
# mostrato che il regime vale +0.37 pp, dentro il rumore. Alzarlo qui
# spezzerebbe l'appaiamento per guadagnare un effetto che non c'è.
# Il warning di `_warn_flat_clock` in log è quindi ATTESO.
#
# DIAGNOSTICA GEMELLA, offline e a costo ~zero:
# `attn_explorer.diag_query_rows` misura sullo stesso asse il bias
# POSIZIONALE del picco (last% / extreme% / top1_pct / gini contro il loro
# livello di caso) sopra uno store di `attn_explorer.capture`. Le due cose
# rispondono a domande diverse e vanno lette insieme: questa dice SE il
# puntatore serve, quella dice COSA è cambiato nel segnale. Se l'accuracy
# è di nuovo nulla, è la diagnostica a distinguere "il picco non si è
# mosso" (asse righe morto) da "si è mosso senza migliorare" (segnale
# povero a prescindere dalle righe).
#
# COSTO: 13.09 s/sample misurati su questo arm sulla card più lenta del
# constraint → ~3h17 per shard da 900. `--time=05:00:00` come l'arm
# originale, di cui questo è una copia esatta a meno di `query_rows`.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
export PYTHONUNBUFFERED=1

QROWS="${QROWS:-question}"
case "${QROWS}" in
  question|qopts) ARM="highlight_top1_24_${QROWS}" ;;
  all)
    echo "QROWS=all è l'arm highlight_top1_24 già misurato (54.63%, gruppo" >&2
    echo "video_mme-ablation): rilanciarlo qui produrrebbe un duplicato." >&2
    exit 1
    ;;
  *)
    echo "QROWS sconosciuto: ${QROWS}. Valide: question, qopts" >&2
    exit 1
    ;;
esac
echo "arm: ${ARM} (query_rows=${QROWS})"

# Knob IDENTICI a highlight_top1_24.sbatch: cambiarne uno solo qui romperebbe
# l'appaiamento con le celle di controllo sui 2700, che è tutto il motivo per
# cui questo arm costa 3 job invece di 9.
uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    strategy.top_k=1 \
    strategy.cell_select=attention \
    strategy.query_rows=${QROWS} \
    strategy.random_seed=0 \
    strategy.always_highlight=false \
    strategy.entropy_threshold=0.7 \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    wandb.group=video_mme-ablation \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,${ARM},query_rows] \
    "$@"
