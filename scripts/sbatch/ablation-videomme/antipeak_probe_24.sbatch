#!/bin/bash
#SBATCH --job-name=probe-antipeak_24
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# SONDA DI SENSIBILITÀ del canale-prompt — non è un arm di accuracy.
#
# LA DOMANDA. highlight_top1_24 (54.63%) e random_top1_24 (54.78%) sono
# indistinguibili fra loro e sotto baseline_24 (55.04%). Due cause diverse
# spiegano quel nulla:
#   (A) il puntatore funziona ma punta male  → si aggiusta il SEGNALE
#                                              (entity rows, double-frame,
#                                              Qwen3-VL)
#   (B) il modello non esegue il puntatore   → nessuna delle tre serve
# Questa sonda separa (A) da (B) PRIMA di spendere in una delle tre.
#
# PERCHÉ random_top1_24 non l'ha già separata: era la stessa domanda ma
# senza potenza statistica. Il puntatore scattava solo col gate aperto
# (~73% dei sample), dove l'accuracy è ~42% — sample in gran parte già
# sbagliati, quindi quasi nessuno spazio per mostrare un danno. Qui il
# puntatore va su TUTTI i sample (always_highlight=true) e si legge il
# risultato sui soli sample GIÀ CORRETTI al pass 1, dove si parte da ~88%
# e lo spazio per rompere qualcosa c'è tutto.
#
# L'INTERVENTO. `cell_select=antipeak` indica la cella più LONTANA NEL
# TEMPO dal picco di attenzione — non quella a massa minima, che può
# cadere accanto al picco. Svia di proposito.
#
# PERCHÉ L'INTERPRETAZIONE NON DIPENDE DA QUANTO È BUONO IL PICCO (ed è il
# motivo per cui questa sonda vale più di un oracolo su ground truth):
#   - se il picco è informativo   → l'antipicco è sbagliato → il modello
#                                   che obbedisce peggiora;
#   - se il picco è solo posizionale (recency, cfr. probe 2dodrndg: ultima
#     cella nel 28% dei casi) → l'antipicco è un random con bias opposto →
#     il modello che obbedisce peggiora COMUNQUE, su sample che stava già
#     indovinando.
# In entrambi i rami: nessun danno ⟹ il puntatore non viene eseguito ⟹
# causa (B) ⟹ opzioni "entity rows" e "double-frame" chiuse, perché possono
# solo migliorare DOVE si punta.
#
# I DUE ARM, appaiati sullo STESSO subset (stesso seed/shuffle/limit, niente
# sharding: entrambi i task dell'array vedono gli stessi 500 sample).
#   task 0 — antipeak : always_highlight=true, puntatore su tutti i sample
#   task 1 — controllo: gate mai aperto (entropy_threshold=999), quindi
#                       NESSUN puntatore, ma il pass 2 gira lo stesso
#                       (`generate()` è fuori dall'if gate_open in
#                       strategies/attention_highlight.py) sul prompt
#                       INTATTO.
#
# PERCHÉ IL CONTROLLO SERVE. `pred_pass1` è l'argmax della softmax ristretta
# alle lettere (pass 1), `pred` è il parse di `generate()` (pass 2): sono
# due percorsi di decoding diversi, quindi una quota di sample "corretti al
# pass 1 e sbagliati al pass 2" esiste ANCHE SENZA puntatore. Quella quota è
# esattamente ciò che il task 1 misura, e va sottratta.
#
# COME SI LEGGE (metrica nel summary wandb, emessa da `mcq_accuracy` in
# evals/base.py via il breakdown derivato su `pred_pass1`):
#
#     danno = mcq_accuracy_correct_pass1_true[task 1]      (controllo)
#           - mcq_accuracy_correct_pass1_true[task 0]      (antipeak)
#
#   danno ~ 0        → causa (B): il canale-prompt è morto su
#                      qwen2.5-vl-3b. Restano solo interventi che cambiano
#                      il MECCANISMO (marker in-band su Qwen3-VL), non il
#                      puntatore.
#   danno >> 0       → causa (A): il modello ESEGUE il puntatore, e allora
#                      il nulla di highlight_top1_24 è colpa di dove punta →
#                      ha senso investire su double-frame (risoluzione del
#                      puntatore: 12 celle sono ~50 s su un long) e su
#                      entity rows (qualità del picco).
#   danno < 0        → il puntatore sviante MIGLIORA: non è una direzione,
#                      è un bug o un effetto di prompt lunghezza/formato da
#                      indagare prima di qualunque altra cosa.
#
# Il complemento `mcq_accuracy_correct_pass1_false` (sample già sbagliati al
# pass 1) dà la faccia opposta: quanto recupera un puntatore sviante. Atteso
# ~0; se recupera, il puntatore agisce come rumore/regolarizzatore e non
# come indicazione spaziale — altra lettura che chiude il filone.
#
# ⚠️ CONTROLLARE `seen_correct_pass1_true` (la numerosità, non solo la
# frazione). I sample con `pred_fallback=true` sono esclusi dal breakdown:
# lì `pred` è posto uguale a `pred_pass1`, quindi non potrebbero mai
# risultare danneggiati e gonfierebbero la metrica verso "nessun danno" —
# cioè verso la conclusione che la sonda deve poter falsificare (vedi
# `mcq_accuracy` in evals/base.py). Su highlight_top1_24/random_top1_24 il
# fallback era 0/900, quindi l'atteso è che l'esclusione non tolga
# praticamente nulla; se invece `seen_correct_pass1_true` risultasse molto
# sotto il numero di sample corretti al pass 1, il puntatore starebbe
# degradando la PARSABILITÀ dell'output invece della risposta — effetto
# diverso e da riportare a parte, non da nascondere in un'accuracy.
#
# COSTO. 500 sample × 2 forward, ~14.6 s/sample (misura del probe 2dodrndg
# su nullazzo, la card più lenta del constraint, col pass 2 su TUTTI i
# sample come qui) ≈ 2h05 per task, ~1.9x di margine su --time=04:00:00.
# Nessun ricampionamento, nessuna ridecodifica del video: il pass 2 riusa i
# token visivi del pass 1.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)

# L'array qui indicizza i due ARM, non gli shard: il subset è identico nei
# due task (nessun `shard=`), ed è questo che rende il confronto appaiato
# sample-per-sample.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=antipeak
    ARM_OVERRIDES=(
        strategy.cell_select=antipeak
        strategy.always_highlight=true
    )
    ;;
  1)
    ARM=nopointer
    # Gate mai aperto → `final_prompt = prompt`, nessun puntatore emesso,
    # ma i due forward avvengono lo stesso: isola la discrepanza di decoding
    # (softmax ristretta al pass 1 vs generate() al pass 2) dall'effetto del
    # puntatore.
    ARM_OVERRIDES=(
        strategy.cell_select=attention
        strategy.always_highlight=false
        strategy.entropy_threshold=999
    )
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID inatteso: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac
echo "arm: ${ARM}"

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    strategy=attention_highlight \
    dataset.nframes=24 \
    strategy.top_k=1 \
    strategy.pointer_units=seconds \
    strategy.pointer_position=prepend \
    "${ARM_OVERRIDES[@]}" \
    limit=500 \
    shuffle=true \
    hf_home=/work/tesi_avalenza/hf \
    wandb.group=video_mme-antipeak-probe \
    wandb.name=qwen2_5_vl_3b_attn-${ARM}_24-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[probe,video_mme,antipeak,${ARM}] \
    "$@"
