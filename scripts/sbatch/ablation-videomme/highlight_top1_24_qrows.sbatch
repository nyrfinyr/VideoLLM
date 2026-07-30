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
# le celle di confronto ESISTONO GIÀ, stesso set e stessi shard, quindi
# questo arm costa 3 job e il confronto è appaiato per costruzione. E
# esistono in ENTRAMBI i regimi d'orologio:
#
#                            meta=false   meta=true
#     baseline_24              55.04%      55.41%
#     highlight_top1_24 (all)  54.63%      55.33%   <- il paragone
#     random_top1_24           54.78%      non lanciato
#
# Su un subset da 500 quei controlli andrebbero rifatti sullo stesso
# subset (3 job comunque) E l'errore standard su una DIFFERENZA fra due arm
# sarebbe ±3.1 pp contro scarti in gioco di 0.4-1.9 pp: si misurerebbe
# rumore. Su 2700 è ±1.35 pp.
#
# REGIME `meta=true` DI DEFAULT. L'appaiamento non decide (i controlli ci
# sono da entrambe le parti), quindi decide il merito: il puntatore dice
# "focus between X and Y seconds", e a `meta=false` comunica ISTANTI a un
# modello le cui celle video stanno tutte sulla stessa posizione temporale
# M-RoPE. Che quel collasso costi solo +0.37 pp è stato misurato con il
# puntatore che indicava rumore (`highlight_top1_24` a righe `all`): non
# dice nulla su quanto costi a un puntatore che punta bene. Le due
# correzioni possono essere complementari, e misurare il segnale
# migliorato nel regime rotto rischia di mascherarlo. Qui si prova a fare
# meglio, non solo ad attribuire.
#     META=false sbatch ...   # per isolare il contributo del solo orologio
#
# COSA GUARDARE, IN QUEST'ORDINE. L'accuracy pooled è lo strumento MENO
# sensibile: anche sui 2700 ±1.35 pp è al limite per un effetto da 1 pp.
# Il breakdown `correct_pass1_*` (evals/base.py) lo è molto di più, perché
# guarda le risposte che si MUOVONO invece del saldo. A righe `all` e
# `meta=true`, cioè il diretto predecessore di questo arm:
#     1495 corretti al pass 1  ->  77 rotti   (5.15%)
#     1205 sbagliati al pass 1 ->  76 recuperati (6.31%)
#     153 risposte mosse, netto -1
# Un puntatore che ridistribuisce invece di informare. Se le righe-domanda
# rendono il picco informativo, quel rapporto si sbilancia PRIMA che
# l'accuracy si muova. Quindi si legge:
#     1. rotti vs recuperati            vs  77 / 76
#     2. mcq_accuracy pooled            vs  55.33%  (e 55.41% della baseline)
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

META="${META:-true}"
QROWS="${QROWS:-question}"
case "${QROWS}" in
  question|qopts) ARM="highlight_top1_24_${QROWS}_meta${META}" ;;
  all)
    echo "QROWS=all è già misurato in entrambi i regimi (54.63% a meta=false," >&2
    echo "gruppo video_mme-ablation; 55.33% a meta=true, gruppo" >&2
    echo "video_mme-mrope-fix): rilanciarlo qui produrrebbe un duplicato." >&2
    exit 1
    ;;
  *)
    echo "QROWS sconosciuto: ${QROWS}. Valide: question, qopts" >&2
    exit 1
    ;;
esac
echo "arm: ${ARM} (query_rows=${QROWS}, pass_video_metadata=${META})"

# Knob IDENTICI a highlight_top1_24.sbatch: cambiarne uno solo qui romperebbe
# l'appaiamento con le celle di controllo sui 2700, che è tutto il motivo per
# cui questo arm costa 3 job invece di 9.
uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    model.pass_video_metadata=${META} \
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
    wandb.group=video_mme-query-rows \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[video_mme,${ARM},query_rows,meta_${META}] \
    "$@"
