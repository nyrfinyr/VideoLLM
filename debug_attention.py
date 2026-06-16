"""Pipeline di debug dell'attenzione entity→token visivi su UN singolo video.

Disaccoppiata da `main.py` (che fa eval su dataset via Hydra/Weave): qui
niente Hydra, niente Weave, niente wandb. La pipeline è:

    forward di prefill del modello  →  estrazione della attention map
        (entity→visivo)             →  overlay della heatmap sui frame

Usa SEMPRE `Qwen25VLAttention` (l'unico modello che espone
`entity_visual_attention`, con la custom attention interface che stasha i
pesi softmax(q·kᵀ) delle query dell'entity).

Questo file è AUTOCONTENUTO: non chiama più helper di `utils/attn_probe.py`
né `utils/attn_frames.py` (sampling frame, overlay, estrazione top-K e ranking
sono inlinati qui sotto). Dipende solo dalla libreria del progetto (`models`,
`evals`) e da torch/PIL/decord/numpy.

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

Output (in `--outdir`, default `debug_out/`):
  - `debug_<entity>.pt`        dump torch (heatmap + metadati) — letto da
                               `utils/make_report.py`;
  - `debug_<entity>.json`      TUTTI i dati per il report HTML (header, KPI,
                               ranking celle, path dei PNG) — sorgente unica e
                               autodescrittiva del report;
  - `attended_frames/`         top-K frame + overlay heatmap (PNG).
La stampa su stdout resta nel formato parsabile da
`utils/make_report_from_output.py`.
"""
import argparse
import json
import logging
import os
from pathlib import Path

# Logging su stderr con timestamp: le INFO sono diagnostica del forward, lo
# stdout "pulito" (tabella/header) resta parsabile da make_report_from_output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("debug_attention")

# Default visivi: stessi dell'eval EgoSchema (conf/dataset/egoschema.yaml),
# così la heatmap è quella delle reali condizioni di prefill del benchmark.
DEFAULT_NFRAMES = 64
DEFAULT_MAX_PIXELS = 151200

# dtype ammessi dalla CLI → nome dell'attributo torch corrispondente.
DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}

# Config dell'attention interface del modello custom (conf/model/qwen2_5_vl_3b_attn.yaml):
# il decoder testuale usa l'implementazione che cattura i pesi softmax, la
# torre visiva resta su sdpa (non ci serve catturarne l'attenzione).
ATTN_IMPLEMENTATION = {
    "text_config": "qwen25_attn_capture",   # decoder testuale: cattura i pesi
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

    # La sorgente del video è esclusiva: o un .mp4 locale o un sample di dataset.
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
    ap.add_argument("--metric",     choices=["sum", "max"], default="sum", help="ranking celle: 'sum' = massa totale, 'max' = picco (default sum)")
    ap.add_argument("--outdir",     default="debug_out", help="cartella di output per dump + frame (default debug_out/)")
    ap.add_argument("--dtype",      choices=list(DTYPES), default="bfloat16", help="precisione dei pesi (default bfloat16)")
    ap.add_argument("--device-map", default="auto", help="device placement (default auto)")
    ap.add_argument("--hf-home",    default=None, help="root cache HuggingFace (default: $HF_HOME). Applicato PRIMA di importare transformers.")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento del sample (ex utils.attn_probe.local_debug_sample/select_debug_sample)
# ─────────────────────────────────────────────────────────────────────────────
def local_debug_sample(video_path: str) -> tuple[dict, str]:
    """Sample one-off da un video locale, senza passare da un dataset.

    Legge il prompt dal file `.txt` con lo stesso stem del video
    (`videos/paperella.mp4` → `videos/paperella.txt`): quel file contiene il
    prompt GIÀ formattato, passato tale e quale al forward (nessun render MCQ).

    Returns:
        (sample, prompt) — `sample` ha solo `video_path`; `prompt` è il testo.
    Raises:
        FileNotFoundError se il `.txt` omonimo non esiste.
    """
    video = Path(video_path)
    prompt_file = video.with_suffix(".txt")
    if not prompt_file.is_file():
        raise FileNotFoundError(
            f"prompt non trovato: atteso {prompt_file} (stesso stem del video)"
        )
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    return {"video_path": str(video)}, prompt


def select_debug_sample(samples: list[dict], video_idx: str | None) -> dict:
    """Sceglie un sample dalla lista caricata da un loader di benchmark.

    Con `video_idx` cerca quel video specifico; senza, ripiega sul primo
    (`samples[0]`, dopo eventuale shuffle/shard/limit). Solleva se il
    `video_idx` richiesto non c'è — di solito perché `limit` lo ha tagliato
    fuori: in quel caso lasciare `limit=null` nello yaml del dataset.
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


def load_sample(args) -> tuple[dict, str | None]:
    """Costruisce `(sample, prompt)` dalla sorgente scelta sulla CLI.

    - `--video`   → `local_debug_sample` (prompt esplicito dal .txt);
    - `--dataset` → loader del benchmark + `select_debug_sample` (prompt=None,
      verrà formattato come MCQ dalla question/options del sample).
    """
    if args.video:
        return local_debug_sample(args.video)

    # Ramo dataset: leggi la root dallo yaml del benchmark (niente Hydra, solo
    # OmegaConf), poi usa il loader del Dataset registrato nel framework eval.
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


# ─────────────────────────────────────────────────────────────────────────────
# Sampling dei frame (ex utils.attn_probe._sample_doubled_frames)
# ─────────────────────────────────────────────────────────────────────────────
def sample_doubled_frames(
    video_path: str, nframes: int, outdir: Path,
) -> tuple[list[str], list[int]]:
    """Estrae `nframes` frame uniformi dal video e li DUPLICA per il merge.

    Qwen fonde i frame a coppie nel patch embed temporale; duplicando ogni
    frame (`[f0, f0, f1, f1, ...]`) la coppia è fatta di due frame identici →
    ogni cella temporale dell'attenzione coincide con un singolo frame reale.

    Campiona con lo stesso schema `linspace(0, total-1, nframes)` del path
    `Video` normale, scrive i frame su disco (qwen-vl-utils vuole dei path) e
    restituisce la lista RADDOPPIATA da passare a `VideoFrames` più gli indici
    reali NON raddoppiati (indice `ti` ↔ cella temporale `ti`).
    """
    import decord
    import torch
    from PIL import Image

    vr = decord.VideoReader(video_path)
    total = len(vr)
    # Stesso campionamento uniforme di qwen-vl-utils._read_video_decord.
    frame_indices = torch.linspace(0, total - 1, nframes).round().long().tolist()
    paths: list[str] = []
    for k, fi in enumerate(frame_indices):
        p = outdir / f"f{k:04d}.png"
        Image.fromarray(vr[fi].asnumpy()).save(p)
        paths.append(str(p))
    doubled = [p for p in paths for _ in range(2)]   # [p0, p0, p1, p1, ...]
    return doubled, frame_indices


# ─────────────────────────────────────────────────────────────────────────────
# Overlay heatmap su frame (ex utils.attn_frames._hot_colormap/make_overlay)
# ─────────────────────────────────────────────────────────────────────────────
def _hot_colormap(x):
    """Colormap 'hot' (nero→rosso→giallo→bianco) su `x` in [0,1] → RGB uint8.

    Stessi segmenti del 'hot' di matplotlib, implementati a mano (niente
    matplotlib/cv2 in questo ambiente).
    """
    import numpy as np

    r = np.clip(8 / 3 * x, 0, 1)
    g = np.clip(8 / 3 * x - 1, 0, 1)
    b = np.clip(4 * x - 3, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def make_overlay(frame, cell_map, *, zero_border: int = 0, alpha: float = 0.6):
    """Overlay della heatmap di UNA cella temporale sul frame RGB.

    `cell_map` è [grid_h, grid_w]; viene min-max normalizzata (dopo aver
    eventualmente azzerato `zero_border` anelli di bordo, sink di bordo tipici),
    upscalata BICUBIC alla risoluzione del frame e alpha-blendata col 'hot'.
    L'alpha è proporzionale all'attenzione → le zone fredde restano originali.
    """
    import numpy as np
    from PIL import Image

    m = cell_map.float().numpy().copy()
    if zero_border > 0:
        b = zero_border
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    mn, mx = m.min(), m.max()
    m = (m - mn) / (mx - mn) if mx > mn else np.zeros_like(m)

    H, W = frame.shape[:2]
    up = np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((W, H), Image.BICUBIC))
    up = np.clip(up, 0, 1)

    color = _hot_colormap(up).astype(np.float32)
    a = (up * alpha)[..., None]                       # alpha ∝ attenzione
    blended = (1 - a) * frame.astype(np.float32) + a * color
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Ranking celle + estrazione frame (ex utils.attn_frames.extract_attended_frames)
# ─────────────────────────────────────────────────────────────────────────────
def rank_and_extract(
    heatmap,
    video_path: str,
    frame_indices: list[int],
    double: bool,
    entity: str,
    *,
    metric: str = "sum",
    topk: int = 8,
    outdir: Path,
    overlay: bool = True,
) -> dict:
    """Classifica le celle temporali per attenzione, stampa la tabella ed estrae i top-K.

    Cuore inlinato della vecchia `extract_attended_frames`. Calcola lo score
    per cella, mappa cella→frame reale a seconda del regime, stampa l'header e
    la tabella nel formato parsabile da `make_report_from_output.py`, salva i
    PNG (originale + overlay) dei top-K, e RESTITUISCE un dict strutturato con
    tutti i dati pronti per la serializzazione JSON.

    La mappatura cella→frame reale dipende dal regime del dump:
      - `double=True`  → cella `ti` ↔ frame `frame_indices[ti]` (un frame).
      - `double=False` → cella `ti` ↔ coppia `(frame_indices[2ti],
        frame_indices[2ti+1])`, rappresentata dal midpoint.

    Returns:
        dict con `fps`, `total_mass`, `metric`, `regime` e `cells_ranked`
        (lista di dict per cella, in ordine di rank), più i path dei PNG salvati.
    """
    import decord
    import torch
    from PIL import Image

    t = heatmap.shape[0]                              # numero di celle temporali

    # Score per cella temporale: massa (sum) o picco (max) sulla griglia HxW.
    if metric == "sum":
        frame_scores = heatmap.flatten(1).sum(dim=1)
    else:
        frame_scores = heatmap.flatten(1).max(dim=1).values

    # Apri il video SOLO per leggere fps e decodificare i top-K frame.
    vr = decord.VideoReader(video_path)
    video_fps = float(vr.get_avg_fps())

    def cell_to_frames(ti: int) -> tuple[list[int], int]:
        """(frame reali della cella, frame rappresentante per l'estrazione)."""
        if double:
            f = int(frame_indices[ti])
            return [f], f
        f0, f1 = int(frame_indices[2 * ti]), int(frame_indices[2 * ti + 1])
        return [f0, f1], int(round((f0 + f1) / 2))

    # Ordina le celle per score decrescente; massa totale per le percentuali.
    order = torch.argsort(frame_scores, descending=True).tolist()
    total_mass = frame_scores.sum().item()
    regime = "frame-doubling (cella=1 frame)" if double else "merge a coppie (cella=2 frame)"

    # ── Stampa header + tabella (formato atteso da make_report_from_output.py) ──
    print(f"entity={entity!r}  video={video_path}  fps={video_fps:.2f}  "
          f"celle_temporali={t}  regime={regime}")
    print(f"metrica={metric}  massa_totale_sui_visivi={total_mass:.4f}")
    print(f"\n{'rank':>4} {'cella':>5} {'score':>9} {'%mass':>7} "
          f"{'frame_reali':>16} {'t_sec':>8}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cells_ranked: list[dict] = []                     # accumulatore per il JSON

    for rank, ti in enumerate(order):
        frames, rep = cell_to_frames(ti)
        t_sec = rep / video_fps
        sc = frame_scores[ti].item()
        pct = 100 * sc / total_mass if total_mass else 0.0
        is_top = rank < topk
        mark = "  <-- top" if is_top else ""
        lbl = f"[{','.join(map(str, frames))}]"
        print(f"{rank:>4} {ti:>5} {sc:>9.5f} {pct:>6.1f}% {lbl:>16} {t_sec:>8.2f}{mark}")

        # Riga del JSON: tutti i dati testuali della cella (immagini aggiunte sotto).
        cell = {
            "rank": rank,
            "cell": ti,
            "score": sc,
            "pct": pct,
            "frames": frames,        # frame reali coperti dalla cella
            "rep_frame": rep,        # frame rappresentante (estratto come PNG)
            "t_sec": t_sec,
            "is_top": is_top,
            "image": None,           # riempiti per i soli top-K (sotto)
            "overlay": None,
        }
        cells_ranked.append(cell)

    # ── Estrazione PNG dei top-K (originale + overlay) ──
    saved: list[str] = []
    for rank, ti in enumerate(order[:topk]):
        _, real = cell_to_frames(ti)
        frame = vr[real].asnumpy()
        t_sec = real / video_fps
        stem = f"rank{rank}_cell{ti:02d}_frame{real}_t{t_sec:.1f}s"

        # Frame originale.
        img_path = outdir / f"{stem}.png"
        Image.fromarray(frame).save(img_path)
        saved.append(str(img_path))
        cells_ranked[rank]["image"] = img_path.name   # nome relativo a outdir

        # Overlay heatmap (opzionale).
        if overlay:
            ov_path = outdir / f"{stem}_overlay.png"
            make_overlay(frame, heatmap[ti]).save(ov_path)
            saved.append(str(ov_path))
            cells_ranked[rank]["overlay"] = ov_path.name

    print(f"\nsalvati {len(saved)} frame in {outdir}/")
    for p in saved:
        print(" ", p)

    return {
        "fps": video_fps,
        "total_mass": total_mass,
        "metric": metric,
        "regime": regime,
        "cells_ranked": cells_ranked,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Serializzazione JSON
# ─────────────────────────────────────────────────────────────────────────────
def _to_jsonable(obj):
    """Rende serializzabili tensori/tuple/np.* in tipi JSON nativi."""
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def write_report_json(
    json_path: Path,
    *,
    entity: str,
    video_path: str,
    prompt: str,
    nframes: int,
    double: bool,
    frame_indices: list[int],
    heatmap,
    out_attn: dict,
    ranking: dict,
    topk: int,
    attended_dir: Path,
    outdir: Path,
) -> None:
    """Scrive il JSON con TUTTE le informazioni per costruire la pagina HTML.

    È pensato come sorgente unica e autodescrittiva del report: contiene
    l'header (entity, video, fps, regime, celle), i KPI (massa totale, span),
    il ranking completo delle celle e i path RELATIVI dei PNG estratti. Le
    immagini sono referenziate per path relativo a `outdir`, così il report
    generator può embeddarle in base64.
    """
    cells = heatmap.shape[0]
    # Path della cartella frame relativo a outdir (es. "attended_frames").
    attended_rel = os.path.relpath(attended_dir, outdir)

    payload = {
        # ── Header / metadati del run ──
        "entity": entity,
        "video_path": video_path,
        "video_name": Path(video_path).name,
        "prompt": prompt,
        "nframes": nframes,
        "double_frames": double,
        "regime": ranking["regime"],
        "cells": cells,
        "fps": ranking["fps"],
        "metric": ranking["metric"],
        # ── KPI / sanity dell'attenzione ──
        "total_visual_mass": ranking["total_mass"],     # massa su token visivi (<1)
        "heatmap_shape": list(heatmap.shape),           # [t, grid_h, grid_w]
        "entity_span": _to_jsonable(out_attn.get("entity_span")),
        "visual_span": _to_jsonable(out_attn.get("visual_span")),
        "frame_indices": _to_jsonable(frame_indices),
        "topk": topk,
        # ── Cartella PNG (relativa a outdir) per il report generator ──
        "attended_dir": attended_rel,
        # ── Ranking completo delle celle (path PNG relativi a outdir) ──
        "cells_ranked": [
            {**c, "image": (f"{attended_rel}/{c['image']}" if c["image"] else None),
                  "overlay": (f"{attended_rel}/{c['overlay']}" if c["overlay"] else None)}
            for c in ranking["cells_ranked"]
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report JSON salvato in %s", json_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (ex utils.attn_probe.run_entity_attention_debug)
# ─────────────────────────────────────────────────────────────────────────────
def run_debug(vlm, sample: dict, entity: str, args, prompt: str | None) -> None:
    """Forward di prefill, estrazione heatmap, dump `.pt` + JSON + PNG.

    Ricostruisce lo stesso `(media, text)` che l'eval EgoSchema passerebbe a
    `generate` (stesso `Video`/`VideoFrames` + prompt), così l'attenzione
    catturata è quella delle reali condizioni di prefill.
    """
    import shutil
    import tempfile

    import torch

    from evals.base import format_mcq_prompt
    from models import Text, Video, VideoFrames

    # Guard: solo Qwen25VLAttention espone l'API entity_visual_attention.
    if not hasattr(vlm, "entity_visual_attention"):
        raise TypeError(
            f"{type(vlm).__name__} non espone entity_visual_attention: "
            "lancia con il modello qwen2_5_vl_3b_attn"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    attended_dir = outdir / "attended_frames"

    video_path = sample["video_path"]
    # Prompt: esplicito (video locale) oppure MCQ formattato da question/options.
    prompt_text = prompt if prompt is not None else format_mcq_prompt(
        sample["question"], sample["options"]
    )
    text = Text(prompt_text)

    # ── Costruzione del media: frame raddoppiati o video con merge a coppie ──
    tmpdir: Path | None = None
    if args.double_frames:
        # Frame-doubling: estrai+duplica i frame in una tmpdir, passali come VideoFrames.
        tmpdir = Path(tempfile.mkdtemp(prefix="qwen_double_"))
        doubled, frame_indices = sample_doubled_frames(video_path, args.nframes, tmpdir)
        media = VideoFrames(doubled, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
        logger.info(
            "DEBUG frame-doubling | %d frame campionati e duplicati (%d in input)",
            len(frame_indices), len(doubled),
        )
    else:
        # Path normale: il Video campiona da sé; ricostruiamo gli indici per la mappa cella→frame.
        media = Video(video_path, nframes=args.nframes, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
        import decord
        vr = decord.VideoReader(video_path)
        frame_indices = torch.linspace(0, len(vr) - 1, args.nframes).round().long().tolist()

    logger.info("DEBUG entity-attention | video=%s | entity=%r", video_path, entity)
    logger.info("Prompt:\n%s", text.text)

    # ── Forward di prefill + cattura dei pesi entity→visivo ──
    try:
        out = vlm.entity_visual_attention(media, text, entity)
    finally:
        # Pulisci la tmpdir dei frame raddoppiati a forward concluso.
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    heatmap = out["heatmap"]                           # [t, grid_h, grid_w]
    visual_scores = out["visual_scores"]
    logger.info("entity_span=%s visual_span=%s", out["entity_span"], out["visual_span"])
    logger.info("heatmap shape=%s (t, grid_h, grid_w)", tuple(heatmap.shape))
    logger.info(
        "visual_scores: n=%d sum=%.4f (massa di attenzione sull'immagine, < 1)",
        visual_scores.numel(), visual_scores.sum().item(),
    )
    flat = heatmap.flatten()
    vals, idx = flat.topk(min(5, flat.numel()))
    logger.info(
        "top-5 celle visive: idx=%s vals=%s",
        idx.tolist(), [round(v, 6) for v in vals.tolist()],
    )

    # ── Dump `.pt` (compat con utils/make_report.py, che legge il tensore) ──
    safe = entity.replace("/", "_").replace(" ", "_")
    dump = {
        **out,
        "entity": entity,
        "video_path": video_path,
        "frame_indices": frame_indices,
        "double_frames": args.double_frames,
    }
    dump_path = outdir / f"debug_{safe}.pt"
    torch.save(dump, dump_path)
    logger.info("Dump attenzione salvato in %s", dump_path)

    # ── Ranking celle + stampa tabella + estrazione PNG top-K ──
    ranking = rank_and_extract(
        heatmap, video_path, frame_indices, args.double_frames, entity,
        metric=args.metric, topk=args.topk, outdir=attended_dir, overlay=True,
    )

    # ── JSON con tutti i dati per il report HTML ──
    write_report_json(
        outdir / f"debug_{safe}.json",
        entity=entity,
        video_path=video_path,
        prompt=prompt_text,
        nframes=args.nframes,
        double=args.double_frames,
        frame_indices=frame_indices,
        heatmap=heatmap,
        out_attn=out,
        ranking=ranking,
        topk=args.topk,
        attended_dir=attended_dir,
        outdir=outdir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # HF_HOME va impostato PRIMA di importare transformers (sotto, via models).
    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    import torch
    from models import Qwen25VLAttention

    torch.manual_seed(42)

    # Carica sample + prompt prima di istanziare il modello (fallisce presto se la sorgente è errata).
    sample, prompt = load_sample(args)

    logger.info("Caricamento Qwen25VLAttention (dtype=%s, device_map=%s)", args.dtype, args.device_map)
    vlm = Qwen25VLAttention(
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    run_debug(vlm, sample, args.entity, args, prompt)


if __name__ == "__main__":
    main()
