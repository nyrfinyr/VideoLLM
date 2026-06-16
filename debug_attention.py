"""Pipeline di debug dell'attenzione entity→token visivi su UN singolo video.

Disaccoppiata da `main.py` (che fa eval su dataset via Hydra/Weave): qui
niente Hydra, niente Weave, niente wandb. La pipeline è:

    forward di prefill del modello  →  estrazione della attention map
        (entity→visivo)             →  overlay della heatmap sui frame

Usa SEMPRE `Qwen25VLAttention` (l'unico modello che espone
`entity_visual_attention`, con la custom attention interface che stasha i
pesi softmax(q·kᵀ) delle query dell'entity).

Due sorgenti per il singolo video:

  1) VIDEO LOCALE — il prompt è letto dal `.txt` con lo stesso stem del
     video (`videos/paperella.mp4` → `videos/paperella.txt`), passato tale
     e quale (nessun render MCQ):

        uv run python debug_attention.py \
            --video videos/paperella.mp4 --entity 'duck'

  2) SAMPLE DI DATASET — pesca un sample dal dump on-disk di un benchmark
     (root letta da `conf/dataset/<name>.yaml`) per `video_idx`; il prompt
     è il MCQ formattato dalla domanda/opzioni:

        uv run python debug_attention.py \
            --dataset egoschema --entity 'knife' \
            --video-idx 03657401-d4a4-40d0-9b03-d7e093ef93d1

Output (in `--outdir`, default `debug_out/`): `debug_<entity>.pt` (dump
della heatmap + metadati) e `attended_frames/` (top-K frame + overlay).
"""
import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("debug_attention")

# Default visivi: stessi dell'eval EgoSchema (conf/dataset/egoschema.yaml),
# così la heatmap è quella delle reali condizioni di prefill del benchmark.
DEFAULT_NFRAMES = 64
DEFAULT_MAX_PIXELS = 151200

# Config del modello custom-attention (conf/model/qwen2_5_vl_3b_attn.yaml).
DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
ATTN_IMPLEMENTATION = {
    "text_config": "qwen25_attn_capture",   # decoder testuale: cattura i pesi
    "vision_config": "sdpa",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)

    src.add_argument("--video",     help="path a un .mp4 locale (prompt nel .txt omonimo)")
    src.add_argument("--dataset",   help="nome del dataset da cui pescare il sample (es. egoschema)")
    ap.add_argument("--entity",     required=True, help="stringa di cui estrarre le query: DEVE comparire nel prompt")
    ap.add_argument("--video-idx",  default=None, help="con --dataset: il video_idx del sample (default: primo sample)")
    ap.add_argument("--nframes",    type=int, default=DEFAULT_NFRAMES, help=f"frame campionati uniformemente (default {DEFAULT_NFRAMES})")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help=f"budget di pixel per frame post-resize (default {DEFAULT_MAX_PIXELS})")
    ap.add_argument("--min-pixels", type=int, default=None, help="floor opzionale di pixel per frame (default: qwen-vl-utils)")
    ap.add_argument("--double-frames", action="store_true", help="frame-doubling: ogni cella temporale ↔ un singolo frame (costo ~2x token visivi nel prefill)")
    ap.add_argument("--topk",       type=int, default=8, help="quanti frame estrarre come PNG (default 8)")
    ap.add_argument("--outdir",     default="debug_out", help="cartella di output per dump + frame (default debug_out/)")
    ap.add_argument("--dtype",      choices=list(DTYPES), default="bfloat16", help="precisione dei pesi (default bfloat16)")
    ap.add_argument("--device-map", default="auto", help="device placement (default auto)")
    ap.add_argument("--hf-home",    default=None, help="root cache HuggingFace (default: $HF_HOME). Applicato PRIMA di importare transformers.")

    return ap.parse_args()

def load_sample(args) -> tuple[dict, str | None]:
    """Costruisce `(sample, prompt)` dalla sorgente scelta.

    - video locale → `local_debug_sample` (prompt dal .txt, ritornato esplicito);
    - dataset      → loader del benchmark + `select_debug_sample` (prompt=None,
      verrà formattato come MCQ da question/options).
    """
    from utils.attn_probe import local_debug_sample, select_debug_sample

    if args.video:
        return local_debug_sample(args.video)

    # Dataset: leggi la root dallo yaml del benchmark (niente Hydra, solo
    # OmegaConf), poi usa il loader del Dataset registrato.
    from omegaconf import OmegaConf

    from evals import Dataset

    cfg_path = Path(__file__).parent / "conf" / "dataset" / f"{args.dataset}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config dataset non trovata: {cfg_path}")
    dataset_cfg = OmegaConf.load(cfg_path)
    dataset = Dataset.get(args.dataset)
    samples = dataset.loader(dataset_cfg)
    logger.info("Caricati %d sample da %s (%s)", len(samples), dataset.name, dataset_cfg.root)
    return select_debug_sample(samples, args.video_idx), None


def main() -> None:
    args = parse_args()

    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    import torch
    from models import Qwen25VLAttention
    from utils.attn_probe import run_entity_attention_debug

    torch.manual_seed(42)

    sample, prompt = load_sample(args)

    logger.info("Caricamento Qwen25VLAttention (dtype=%s, device_map=%s)", args.dtype, args.device_map)
    vlm = Qwen25VLAttention(
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    run_entity_attention_debug(
        vlm,
        sample,
        args.entity,
        nframes=args.nframes,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        double_frames=args.double_frames,
        topk=args.topk,
        prompt=prompt,
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
