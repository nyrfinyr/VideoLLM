"""Pipeline di CATTURA dell'attenzione testo→token visivi su uno o più video.

Disaccoppiata da `main.py` (eval su dataset via Hydra/Weave): qui niente Hydra,
niente Weave, niente wandb. La pipeline è:

    forward di prefill del modello  →  cattura della attention map COMPLETA
        (una heatmap per OGNI token della domanda, media su layer centrali e
         teste)  →  dump su disco (`AttentionCapture`) + estrazione frame

La cattura è ENTITY-AGNOSTICA: non si sceglie più un'entity al momento del
forward. Il dump conserva una mappa di attenzione per ogni token della domanda;
la selezione delle entity e il riordino dei frame avvengono a RUNTIME nel server
`attn_explorer.serve`. `--entity` resta solo come quick-check testuale opzionale.

Usa SEMPRE `Qwen25VLAttention` (l'unico modello con la custom attention
interface che stasha i pesi softmax delle query verso i token visivi).

Tre sorgenti:

  1) BATCH — una cartella di .mp4 locali (prompt nel sidecar omonimo):

        uv run python -m attn_explorer.capture --video-dir videos/duck \
            --prompt-from-json --outdir debug_out

  2) VIDEO LOCALE singolo — prompt dal `.txt` omonimo (o `.json` con
     `--prompt-from-json`, chiavi `question`/`entity`):

        uv run python -m attn_explorer.capture --video videos/paperella.mp4

  3) SAMPLE DI DATASET — pesca un sample dal dump on-disk di un benchmark
     (root da `conf/dataset/<name>.yaml`) per `video_idx`; il prompt è il MCQ:

        uv run python -m attn_explorer.capture --dataset egoschema \
            --video-idx 03657401-d4a4-40d0-9b03-d7e093ef93d1

     Con `--n K` invece di `--video-idx`, cattura in BATCH le prime K
     video DISTINTE del dataset (dedup per `video_path` — utile per
     dataset con più domande/video come Video-MME o Causal2Needles):

        uv run python -m attn_explorer.capture --dataset video_mme --n 20

Output: uno STORE su disco (`--outdir`, default `debug_out/`):

    <store>/<video_stem>/capture.pt   dump AttentionCapture (tensore + token + meta)
    <store>/<video_stem>/frames/      PNG dei frame rappresentanti (uno per cella)
    <store>/index.json                manifest dei video catturati

Lo store è la sorgente del server interattivo `attn_explorer.serve --store <store>`.
"""
import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from utils.attn_core import AttentionCapture, GridSpec, parse_answer_meta, rank_cells, save_capture

# Logging su stderr con timestamp (diagnostica del forward); lo stdout resta
# per le tabelle del quick-check.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("debug_attention")

# Default visivi: stessi dell'eval EgoSchema (conf/dataset/egoschema.yaml),
# così la heatmap è quella delle reali condizioni di prefill del benchmark.
DEFAULT_NFRAMES = 64
DEFAULT_MAX_PIXELS = 151200

DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}

# Config dell'attention interface del modello custom: il decoder testuale usa
# l'implementazione che cattura i pesi softmax, la torre visiva resta su sdpa.
ATTN_IMPLEMENTATION = {
    "text_config": "qwen25_attn_capture",
    "vision_config": "sdpa",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Definisce e parsa gli argomenti da riga di comando."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # La sorgente è esclusiva: cartella batch, .mp4 singolo o sample di dataset.
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-dir", help="cartella di .mp4 locali da catturare in batch (prompt nel sidecar omonimo)")
    src.add_argument("--video",     help="path a un .mp4 locale (prompt nel .txt omonimo, o nel .json con --prompt-from-json)")
    src.add_argument("--dataset",   help="nome del dataset da cui pescare il sample (es. egoschema)")
    ap.add_argument("--entity",     default=None, help="[quick-check, solo modalità singola] stringa di cui stampare il ranking; DEVE comparire nel testo della domanda")
    ap.add_argument("--prompt-from-json", action="store_true", help="con --video/--video-dir: leggi il prompt (chiave 'question') dal <stem>.json invece che dal .txt")
    ap.add_argument("--video-idx",  default=None, help="con --dataset: il video_idx del sample (default: primo sample; ignorato se --n è dato)")
    ap.add_argument("--n",          type=int, default=None, help="con --dataset: cattura in batch le prime N video DISTINTE (dedup per video_path) invece di un singolo sample da --video-idx")
    ap.add_argument("--nframes",    type=int, default=DEFAULT_NFRAMES, help=f"frame campionati uniformemente (default {DEFAULT_NFRAMES})")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help=f"budget di pixel per frame post-resize (default {DEFAULT_MAX_PIXELS})")
    ap.add_argument("--min-pixels", type=int, default=None, help="floor opzionale di pixel per frame (default: qwen-vl-utils)")
    ap.add_argument("--double-frames", action="store_true", help="frame-doubling: ogni cella temporale ↔ un singolo frame (costo ~2x token visivi nel prefill)")
    ap.add_argument("--topk",       type=int, default=8, help="[quick-check] quante celle marcare come 'top' nello stdout (default 8)")
    ap.add_argument("--metric",     choices=["sum", "max"], default="sum", help="[quick-check] ranking celle: 'sum' = massa totale, 'max' = picco (default sum)")
    ap.add_argument("--outdir",     default="debug_out", help="cartella STORE di output (default debug_out/)")
    ap.add_argument("--dtype",      choices=list(DTYPES), default="bfloat16", help="precisione dei pesi (default bfloat16)")
    ap.add_argument("--device-map", default="auto", help="device placement (default auto)")
    ap.add_argument("--hf-home",    default=None, help="root cache HuggingFace (default: $HF_HOME). Applicato PRIMA di importare transformers.")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento del sample (prompt da sidecar o da dataset)
# ─────────────────────────────────────────────────────────────────────────────
def read_local_prompt(video_path: str, from_json: bool) -> str:
    """Legge il prompt dal sidecar omonimo del video.

    `.txt` (prompt già formattato) di default, oppure `.json` (chiave
    `question`) con `from_json`. Solleva `FileNotFoundError`/`KeyError` se il
    sidecar manca o è malformato.
    """
    video = Path(video_path)
    if from_json:
        json_file = video.with_suffix(".json")
        if not json_file.is_file():
            raise FileNotFoundError(f"sidecar JSON non trovato: atteso {json_file}")
        obj = json.loads(json_file.read_text(encoding="utf-8"))
        if "question" not in obj:
            raise KeyError(f"{json_file}: manca la chiave 'question'")
        return obj["question"].strip()
    txt_file = video.with_suffix(".txt")
    if not txt_file.is_file():
        raise FileNotFoundError(f"prompt non trovato: atteso {txt_file} (stesso stem del video)")
    return txt_file.read_text(encoding="utf-8").strip()


def read_local_meta(video_path: str, from_json: bool) -> dict:
    """Risposta corretta (per la UI del server) dal sidecar `.json`, se presente.

    No-op (`{}`) in modalità `.txt` (nessuna struttura per gt/opzioni) o se il
    sidecar non espone `gt_option` / `answer`+`options` (vedi `parse_answer_meta`).
    """
    if not from_json:
        return {}
    json_file = Path(video_path).with_suffix(".json")
    if not json_file.is_file():
        return {}
    obj = json.loads(json_file.read_text(encoding="utf-8"))
    return parse_answer_meta(obj)


def select_debug_sample(samples: list[dict], video_idx: str | None) -> dict:
    """Sceglie un sample dalla lista caricata da un loader di benchmark.

    Con `video_idx` cerca quel video specifico; senza, ripiega sul primo.
    Solleva se il `video_idx` richiesto non c'è (di solito tagliato da `limit`).
    """
    if not video_idx:
        return samples[0]
    for s in samples:
        if s.get("video_idx") == video_idx:
            return s
    raise ValueError(
        f"video_idx={video_idx!r} non trovato fra i {len(samples)} sample "
        "caricati (limit lo ha escluso? lascia limit=null)"
    )


def _sample_prompt(sample: dict) -> str:
    """Prompt per un sample del loader di un `Dataset`.

    Dataset MCQ (EgoSchema, MVBench, Video-MME, LVBench) espongono
    `options` → il prompt è renderizzato con `format_mcq_prompt`. Dataset
    open-ended (Causal2Needles) non hanno `options` → si usa `question`
    grezza, così com'è consumata da `CausalNeedles.predict_factory`.
    """
    from evals.base import format_mcq_prompt

    if "options" in sample:
        return format_mcq_prompt(sample["question"], sample["options"])
    return sample["question"]


def _load_dataset(args) -> tuple[object, list[dict]]:
    """Risolve `Dataset.get(args.dataset)` + i sample caricati dal suo loader."""
    from omegaconf import OmegaConf

    from evals import Dataset

    # `conf/dataset/` vive alla root del repo, non sotto `attn_explorer/`.
    cfg_path = Path(__file__).parent.parent / "conf" / "dataset" / f"{args.dataset}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config dataset non trovata: {cfg_path}")
    dataset_cfg = OmegaConf.load(cfg_path)
    dataset = Dataset.get(args.dataset)
    samples = dataset.loader(dataset_cfg)
    logger.info("Caricati %d sample da %s (%s)", len(samples), dataset.name, dataset_cfg.root)
    return dataset, samples


def load_dataset_sample(args) -> tuple[str, str, dict]:
    """Ramo `--dataset` (senza `--n`): ritorna UN `(video_path, prompt, meta)`."""
    _, samples = _load_dataset(args)
    sample = select_debug_sample(samples, args.video_idx)
    return sample["video_path"], _sample_prompt(sample), parse_answer_meta(sample)


def load_dataset_samples_batch(args) -> list[tuple[str, str, dict]]:
    """Ramo `--dataset --n K`: le prime `K` righe con `video_path` DISTINTO.

    Alcuni dataset (Video-MME, Causal2Needles) hanno più domande per
    video: senza dedup, catturare "i primi K sample" rischierebbe di
    processare lo stesso video più volte e mancarne altri. Usa la prima
    domanda incontrata per ciascun video.
    """
    _, samples = _load_dataset(args)
    jobs: list[tuple[str, str, dict]] = []
    seen: set[str] = set()
    for sample in samples:
        vp = sample["video_path"]
        if vp in seen:
            continue
        seen.add(vp)
        jobs.append((vp, _sample_prompt(sample), parse_answer_meta(sample)))
        if len(jobs) >= args.n:
            break
    if len(jobs) < args.n:
        logger.warning(
            "richiesti %d video distinti ma il dataset ne ha solo %d disponibili",
            args.n, len(jobs),
        )
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Sampling dei frame (frame-doubling)
# ─────────────────────────────────────────────────────────────────────────────
def sample_doubled_frames(
    video_path: str, nframes: int, outdir: Path,
) -> tuple[list[str], list[int]]:
    """Estrae `nframes` frame uniformi dal video e li DUPLICA per il merge.

    Qwen fonde i frame a coppie nel patch embed temporale; duplicando ogni
    frame (`[f0, f0, f1, f1, ...]`) ogni cella temporale coincide con un singolo
    frame reale. Ritorna la lista RADDOPPIATA (per `VideoFrames`) più gli indici
    reali NON raddoppiati (indice `ti` ↔ cella temporale `ti`).
    """
    import decord
    import torch
    from PIL import Image

    vr = decord.VideoReader(video_path)
    total = len(vr)
    frame_indices = torch.linspace(0, total - 1, nframes).round().long().tolist()
    paths: list[str] = []
    for k, fi in enumerate(frame_indices):
        p = outdir / f"f{k:04d}.png"
        Image.fromarray(vr[fi].asnumpy()).save(p)
        paths.append(str(p))
    doubled = [p for p in paths for _ in range(2)]
    return doubled, frame_indices


def build_media(video_path: str, args) -> tuple[object, list[int], Path | None]:
    """Costruisce `(media, frame_indices, tmpdir)` per il forward di prefill.

    - `--double-frames` → `VideoFrames` con i frame raddoppiati in una tmpdir
      (da pulire dopo il forward, terzo valore);
    - altrimenti → `Video` (campiona da sé) + indici ricostruiti per la mappa
      cella→frame.
    """
    import torch

    from models import Video, VideoFrames

    if args.double_frames:
        tmpdir = Path(tempfile.mkdtemp(prefix="qwen_double_"))
        doubled, frame_indices = sample_doubled_frames(video_path, args.nframes, tmpdir)
        media = VideoFrames(doubled, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
        logger.info("frame-doubling | %d frame campionati e duplicati", len(frame_indices))
        return media, frame_indices, tmpdir

    media = Video(video_path, nframes=args.nframes, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    import decord
    vr = decord.VideoReader(video_path)
    frame_indices = torch.linspace(0, len(vr) - 1, args.nframes).round().long().tolist()
    return media, frame_indices, None


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione dei frame rappresentanti (uno per cella) per il serving
# ─────────────────────────────────────────────────────────────────────────────
def extract_cell_frames(video_path: str, grid: GridSpec, frame_indices, frames_dir: Path) -> None:
    """Salva un PNG per ogni cella temporale (frame rappresentante).

    Il server li serve direttamente (`/api/<video>/frame?cell=ti`) e ci applica
    sopra gli overlay rigenerati a runtime.
    """
    import decord
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    vr = decord.VideoReader(video_path)
    last = len(vr) - 1
    for ti in range(grid.t):
        _, rep = grid.cell_to_frames(ti, frame_indices)
        rep = max(0, min(rep, last))
        Image.fromarray(vr[rep].asnumpy()).save(frames_dir / f"cell_{ti:04d}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Cattura di UN video → AttentionCapture su disco
# ─────────────────────────────────────────────────────────────────────────────
def capture_one(vlm, video_path: str, prompt_text: str, meta: dict, args, store: Path) -> dict:
    """Forward di prefill su un video, salva `AttentionCapture` + frame.

    Scrive in `<store>/<stem>/{capture.pt,frames/}` e ritorna l'entry di indice
    per `index.json` (`meta`, da `parse_answer_meta`, ci finisce dentro così
    la UI del server può mostrare la risposta corretta). Se `args.entity` è
    impostato, stampa un quick-check del ranking per quell'entity (single mode).
    """
    from evals.base import extract_mcq_letters
    from models import Text

    stem = Path(video_path).stem
    out_dir = store / stem
    frames_dir = out_dir / "frames"
    text = Text(prompt_text)
    answer_letters = extract_mcq_letters(prompt_text)

    media, frame_indices, tmpdir = build_media(video_path, args)
    logger.info("Cattura | video=%s | nframes=%d", video_path, args.nframes)
    logger.info("Prompt:\n%s", prompt_text)

    try:
        out = vlm.full_visual_attention(media, text, answer_letters=answer_letters)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    grid = GridSpec(t=out.t, grid_h=out.grid_h, grid_w=out.grid_w, double=args.double_frames)

    # fps reale del video, per i timestamp.
    import decord
    fps = float(decord.VideoReader(video_path).get_avg_fps())

    cap = AttentionCapture(
        video_path=str(video_path),
        prompt=prompt_text,
        fps=fps,
        frame_indices=tuple(int(i) for i in frame_indices),
        grid=grid,
        query_tokens=out.query_tokens,
        attn=out.attn,
        sink_map=out.sink_map,
        sink_dims=out.sink_dims,
        answer_entropy=out.answer_entropy,
        answer_probs=out.answer_probs,
        pred_letter=out.pred_letter,
    )

    save_capture(cap, out_dir / "capture.pt")
    extract_cell_frames(video_path, grid, frame_indices, frames_dir)

    logger.info(
        "Salvato %s | attn=%s | sink_map=%s sink_dims=%s | %d token domanda | regime=%s%s",
        out_dir / "capture.pt", tuple(out.attn.shape), tuple(out.sink_map.shape),
        out.sink_dims, len(out.query_tokens), grid.regime,
        f" | pred={out.pred_letter} entropy={out.answer_entropy:.3f}bit" if out.pred_letter else "",
    )

    # Quick-check testuale opzionale per una specifica entity.
    if args.entity:
        _print_entity_quickcheck(vlm, out, cap, args)

    return {
        "stem": stem,
        "video": str(video_path),
        "video_name": Path(video_path).name,
        "prompt": prompt_text,
        "fps": fps,
        "cells": grid.t,
        "n_tokens": len(out.query_tokens),
        "regime": grid.regime,
        "pred_letter": out.pred_letter,
        "answer_entropy": out.answer_entropy,
        **meta,
    }


def _print_entity_quickcheck(vlm, out, cap: AttentionCapture, args) -> None:
    """Stampa il ranking delle celle per l'entity `args.entity` (single mode).

    `out` è il `VisualAttention` del forward, `cap` l'AttentionCapture salvato.
    """
    try:
        span = vlm._find_entity_span(out.input_ids, args.entity)
    except ValueError as e:
        logger.warning("quick-check saltato: %s", e)
        return
    q_lo, _ = out.query_span
    rows = [i - q_lo for i in range(*span) if i >= q_lo]
    heatmap = cap.aggregate(rows)
    ranked = rank_cells(heatmap, cap.grid, cap.frame_indices, cap.fps,
                        metric=args.metric, topk=args.topk)
    print(f"\nquick-check entity={args.entity!r}  righe={rows}  metrica={args.metric}")
    print(f"{'rank':>4} {'cella':>5} {'score':>9} {'%mass':>7} {'t_sec':>8}")
    for c in ranked:
        mark = "  <-- top" if c.is_top else ""
        print(f"{c.rank:>4} {c.cell:>5} {c.score:>9.5f} {c.pct:>6.1f}% {c.t_sec:>8.2f}{mark}")


# ─────────────────────────────────────────────────────────────────────────────
# Indice dello store
# ─────────────────────────────────────────────────────────────────────────────
def write_index(store: Path, entries: list[dict]) -> None:
    """Scrive/aggiorna `<store>/index.json` (merge per `stem`)."""
    index_path = store / "index.json"
    existing: dict[str, dict] = {}
    if index_path.is_file():
        for e in json.loads(index_path.read_text(encoding="utf-8")).get("videos", []):
            existing[e["stem"]] = e
    for e in entries:
        existing[e["stem"]] = e
    payload = {"videos": sorted(existing.values(), key=lambda e: e["stem"])}
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Indice aggiornato: %s (%d video)", index_path, len(payload["videos"]))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def resolve_jobs(args) -> list[tuple[str, str, dict]]:
    """Risolve la sorgente CLI in una lista di `(video_path, prompt_text, meta)`."""
    if args.video_dir:
        root = Path(args.video_dir)
        if not root.is_dir():
            raise SystemExit(f"--video-dir non è una cartella: {root}")
        videos = sorted(p for p in root.glob("*.mp4"))
        if not videos:
            raise SystemExit(f"nessun .mp4 in {root}")
        jobs: list[tuple[str, str, dict]] = []
        for v in videos:
            try:
                prompt = read_local_prompt(str(v), args.prompt_from_json)
                meta = read_local_meta(str(v), args.prompt_from_json)
                jobs.append((str(v), prompt, meta))
            except (FileNotFoundError, KeyError) as e:
                logger.warning("salto %s: %s", v.name, e)
        if not jobs:
            raise SystemExit(f"nessun video con sidecar valido in {root}")
        return jobs

    if args.video:
        prompt = read_local_prompt(args.video, args.prompt_from_json)
        meta = read_local_meta(args.video, args.prompt_from_json)
        return [(args.video, prompt, meta)]

    if args.n is not None:
        return load_dataset_samples_batch(args)
    return [load_dataset_sample(args)]


def main() -> None:
    args = parse_args()

    if args.prompt_from_json and not (args.video or args.video_dir):
        raise SystemExit("--prompt-from-json richiede --video o --video-dir")
    if args.entity and args.video_dir:
        raise SystemExit("--entity (quick-check) ha senso solo in modalità singola, non con --video-dir")
    if args.n is not None and not args.dataset:
        raise SystemExit("--n (batch su più video) ha senso solo con --dataset")

    # HF_HOME va impostato PRIMA di importare transformers (sotto, via models).
    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    # Risolvi i job (fallisce presto se la sorgente è errata) prima del modello.
    jobs = resolve_jobs(args)

    import torch
    from models import Qwen25VLAttention

    torch.manual_seed(42)

    logger.info("Caricamento Qwen25VLAttention (dtype=%s, device_map=%s)", args.dtype, args.device_map)
    vlm = Qwen25VLAttention(
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    store = Path(args.outdir)
    store.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for i, (video_path, prompt_text, meta) in enumerate(jobs, 1):
        logger.info("[%d/%d] %s", i, len(jobs), video_path)
        try:
            entries.append(capture_one(vlm, video_path, prompt_text, meta, args, store))
        except Exception:
            logger.exception("cattura fallita per %s", video_path)
    if entries:
        write_index(store, entries)
    logger.info("Fatto: %d/%d video catturati in %s", len(entries), len(jobs), store)


if __name__ == "__main__":
    main()
