#!/bin/bash
#SBATCH --job-name=mrope-clock-fix
#SBATCH --account=tesi_avalenza
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-8
#SBATCH --time=06:00:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%A_%a.err

# OROLOGIO M-RoPE RIPARATO — i tre arm chiave rilanciati con
# `model.pass_video_metadata=true`, su Video-MME INTERO (2700 sample).
#
# IL DIFETTO. `Qwen._prepare_inputs` non passava al processor i
# `video_metadata` reali, quindi `second_per_grid_ts` vale
# `temporal_patch_size / sampled_fps = 2/24 = 0.083` per QUALUNQUE video, e
# `get_rope_index` (`modeling_qwen2_5_vl.py:1125`) fa
# `time_interval = tokens_per_second * int(0.083) = 0`. In
# `get_vision_position_ids` diventa `position_temporal = arange(t) * 0`:
# TUTTE le celle video finiscono sulla stessa posizione temporale M-RoPE.
# Misurato su Video-MME nframes=24:
#     pass_video_metadata=false → second_per_grid_ts=0.0833 → time_interval=0
#     pass_video_metadata=true  → second_per_grid_ts=6.1923 → time_interval=12
# Non è condizionale alla durata del video: 2/24 è una costante, quindi il
# collasso è UNIVERSALE e vale per ogni run del README, baseline comprese.
#
# PERCHÉ PROPRIO QUESTI TRE ARM. `attention_highlight` dice al modello
# "Focus on the part of the video between X and Y seconds": comunica ISTANTI
# a un modello le cui celle video non hanno coordinate temporali distinte.
# Il filone è girato tutto così, e il suo risultato — highlight ≈ random ≈
# baseline — potrebbe essere l'artefatto di un canale che non poteva
# funzionare, non la prova che il segnale d'attenzione sia inutile.
#
# NIENTE CELLE `meta=false`: ci sono GIÀ, sull'intero set, dallo stesso
# codice e stessa config (gruppo wandb `video_mme-ablation`):
#     baseline_24        55.04%  (1486/2700)
#     highlight_top1_24  54.63%  (1475/2700)
#     random_top1_24     54.78%  (1479/2700)
# È il motivo per cui questo job gira sui 2700 e non su un subset: il
# controllo appaiato è gratis solo a parità di set. Su un subset da 500 le
# celle `meta=false` andrebbero rifatte (6 run invece di 3) E la potenza
# statistica non basterebbe comunque — gli scarti in gioco in questa tabella
# sono 0.4-1.9 pp, mentre 500 sample danno ~2.2 pp di errore standard su una
# singola proporzione: si misurerebbe rumore.
#
# La validità di quel confronto poggia su un fatto verificato: il refactor
# di `models/qwen_attn.py` (span contiguo → `index_select`, cambio di import
# di `repeat_kv`, `image_patch_size` esplicito) NON cambia la numerica su
# Qwen2.5 — indici visivi identici (2160 e 4320 token), `query_span`
# identico, `repeat_kv` stesso bytecode, `image_patch_size=14` già il
# default di `process_vision_info`. Per prudenza vale comunque un solo shard
# di `baseline_24` con `meta=false` come check di riproducibilità: se non
# torna 1486/2700 sul suo shard, tutto il confronto qui sotto salta.
#     META=false sbatch --array=0 scripts/sbatch/ablation-videomme/mrope_clock_control.sbatch
#
# DISEGNO: 3 arm x 3 shard = 9 task. `arm = task / 3`, `shard = task % 3`,
# stessa strided-slice degli altri arm dell'ablation (`samples[shard::3]`).
#
#   task 0-2  baseline_24_meta   (uniform, nessun puntatore)
#   task 3-5  highlight_top1_24_meta
#   task 6-8  random_top1_24_meta
#
# COME SI LEGGE (accuracy POOLED per arm, Σ corretti / Σ sample sui 3 shard):
#
#   costo del collasso  = baseline_meta      - 55.04
#   il puntatore aiuta? = highlight_meta     - baseline_meta
#   È L'ATTENZIONE?     = highlight_meta     - random_meta
#                         da confrontare con lo -0.15 attuale (54.63-54.78)
#
# L'ultima riga è quella che decide il futuro del filone. Se con l'orologio
# riparato highlight stacca random, il segnale c'era e non poteva
# esprimersi: tutto il filone va rimisurato in regime `true` e le leve
# (double-frame, entity rows) tornano sensate. Se resta ≈ 0, il picco
# d'attenzione è davvero poco informativo e il collasso non c'entrava.
#
# `random_top1_24_meta` NON è opzionale: senza, un eventuale guadagno di
# highlight sarebbe ambiguo fra "il puntatore d'attenzione ora funziona" e
# "qualunque puntatore ora funziona" — esattamente l'ambiguità per cui
# random_top1_24 era stato creato.
#
# Guardare anche `mcq_accuracy_highlight_units_seconds` vs `..._fraction`
# (breakdown in `evals/base.py`): a `meta=false` il fallback a unità
# frazionarie scattava per la ragione sbagliata, a `meta=true` scatta per
# quella giusta, quindi i due gruppi non sono più la stessa popolazione.
#
# COSTO, dalle latenze misurate sulla card più lenta del constraint (Quadro
# RTX 6000) nelle run originali di questi stessi arm:
#     baseline_24        7.19 s/sample → ~1h48 per shard da 900 (misurato 6517 s)
#     highlight/random  13.09 s/sample → ~3h17 per shard da 900 (misurato 11821 s)
# `--time=06:00:00` copre il caso peggiore con ~1.8x di margine.
set -euo pipefail

echo "node: $(hostname)"; echo "date: $(date -Iseconds)"
echo "array_job: ${SLURM_ARRAY_JOB_ID} task: ${SLURM_ARRAY_TASK_ID} job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground
source scripts/lib/notify.sh && tg_job_start   # notifica Telegram (no-op senza bot)
export PYTHONUNBUFFERED=1

# `META=false` da ambiente per il check di riproducibilità descritto sopra;
# il regime di questo job è `true`.
META="${META:-true}"

TASK="${SLURM_ARRAY_TASK_ID}"
SHARD=$(( TASK % 3 ))
case $(( TASK / 3 )) in
  0)
    ARM=baseline_24
    ARM_OVERRIDES=( strategy=uniform )
    ;;
  1)
    ARM=highlight_top1_24
    ARM_OVERRIDES=( strategy=attention_highlight strategy.top_k=1
                    strategy.cell_select=attention )
    ;;
  2)
    ARM=random_top1_24
    ARM_OVERRIDES=( strategy=attention_highlight strategy.top_k=1
                    strategy.cell_select=random )
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID inatteso: ${TASK} (attesi 0-8)" >&2
    exit 1
    ;;
esac

# Knob comuni ai due arm con puntatore, identici alle run originali: cambiare
# uno solo di questi renderebbe il confronto con i numeri `meta=false` non
# più appaiato.
if [ "${ARM}" != "baseline_24" ]; then
  ARM_OVERRIDES+=(
      strategy.random_seed=0
      strategy.always_highlight=false
      strategy.entropy_threshold=0.7
      strategy.pointer_units=seconds
      strategy.pointer_position=prepend
  )
fi

NAME="${ARM}_meta${META}"
echo "arm: ${ARM}  shard: ${SHARD}/3  pass_video_metadata: ${META}"

uv run python main.py \
    model=qwen2_5_vl_3b_attn \
    dataset=video_mme \
    dataset.nframes=24 \
    model.pass_video_metadata=${META} \
    "${ARM_OVERRIDES[@]}" \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SHARD} \
    num_shards=3 \
    wandb.group=video_mme-mrope-fix \
    wandb.name=qwen2_5_vl_3b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[mrope,video_mme,${ARM},meta_${META}] \
    "$@"
