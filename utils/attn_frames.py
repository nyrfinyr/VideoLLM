"""Suite tool attenzione (2/2): frame del video con attenzione entity→visivo più alta.

Legge il dump prodotto da `utils/attn_probe.py` (`entity_visual_attention`:
dict con `heatmap [t, grid_h, grid_w]`, e — se salvato dal probe — `video_path`,
`frame_indices`, `double_frames`, `entity`) e classifica le `t` CELLE
TEMPORALI per massa di attenzione, mappandole sui frame reali del .mp4.

Due regimi di mappatura cella→frame, decisi dal flag `double_frames` del dump:

- **merge a coppie** (default Qwen): ogni cella `ti` è il merge ×2 di due
  frame campionati `(2·ti, 2·ti+1)`; il rappresentante è il loro midpoint.
- **frame-doubling** (`--double-frames`): ogni frame è stato duplicato
  prima del forward, quindi il merge fonde due frame identici → cella `ti` ↔
  ESATTAMENTE il frame reale `frame_indices[ti]`.

Senza `frame_indices` nel dump (dump vecchi) ricostruisce il campionamento
uniforme `linspace(0, total-1, nframes)` di `qwen_vl_utils`.

    uv run python utils/attn_frames.py
    uv run python utils/attn_frames.py --topk 8 --overlay \
        --tensor tensors/debug_knife.pt \
        --video videos/03657401-d4a4-40d0-9b03-d7e093ef93d1.mp4

Niente modello, niente GPU: opera solo sul tensore salvato + il file video.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def reconstruct_frame_indices(video_path: str, nframes: int) -> tuple[list[int], float]:
    """Indici reali dei frame campionati da qwen-vl-utils + fps del video.

    Replica `linspace(0, total_frames - 1, nframes).round().long()` di
    `qwen_vl_utils.vision_process._read_video_decord` (nessun video_start/end:
    si decodifica l'intero video). Apre il VideoReader ma NON decodifica:
    legge solo `len(vr)` e l'fps medio.
    """
    import decord

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    return idx, video_fps


def _hot_colormap(x: np.ndarray) -> np.ndarray:
    """Colormap 'hot' (nero→rosso→giallo→bianco) su `x` in [0,1] → RGB uint8.

    Stessi segmenti del 'hot' di matplotlib, implementati a mano (niente
    matplotlib/cv2 in questo ambiente).
    """
    r = np.clip(8 / 3 * x, 0, 1)
    g = np.clip(8 / 3 * x - 1, 0, 1)
    b = np.clip(4 * x - 3, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def make_overlay(frame: np.ndarray, cell_map: torch.Tensor, *,
                 zero_border: int = 0, alpha: float = 0.6) -> "Image.Image":
    """Overlay della heatmap di UNA cella temporale sul frame RGB.

    `cell_map` è [grid_h, grid_w]; viene min-max normalizzata (dopo aver
    eventualmente azzerato `zero_border` anelli di bordo, sink di bordo tipici),
    upscalata BICUBIC alla risoluzione del frame e alpha-blendata col 'hot'.
    L'alpha è proporzionale all'attenzione → le zone fredde restano originali.
    """
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
    a = (up * alpha)[..., None]
    blended = (1 - a) * frame.astype(np.float32) + a * color
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def extract_attended_frames(
    dump: dict,
    video_path: str,
    *,
    topk: int = 5,
    metric: str = "sum",
    outdir: str | Path,
    overlay: bool = False,
    zero_border: int = 0,
) -> list[Path]:
    """Classifica le celle temporali per attenzione ed estrae i top-K come PNG.

    Cuore condiviso fra la CLI e il probe (`utils/attn_probe.py`). La
    mappatura cella→frame reale dipende dal regime del dump:

    - `double_frames=True`  → cella `ti` ↔ frame `frame_indices[ti]` (un frame).
    - `double_frames=False` → cella `ti` ↔ coppia `(frame_indices[2ti],
      frame_indices[2ti+1])`, rappresentata dal midpoint.

    Args:
        dump: dict con almeno `heatmap [t, gh, gw]`; opzionali `frame_indices`,
            `double_frames`, `entity`. Se manca `frame_indices` lo ricostruisce.
        video_path: il .mp4 sorgente da cui estrarre i frame.
        topk: quanti frame estrarre.
        metric: 'sum' (massa di attenzione sulla cella) o 'max' (picco).
        outdir: cartella di output dei PNG.
        overlay: salva anche l'overlay spaziale della heatmap.
        zero_border: anelli di bordo azzerati prima dell'overlay (sink di bordo).

    Returns:
        i path dei PNG salvati.
    """
    import decord
    from PIL import Image

    heatmap: torch.Tensor = dump["heatmap"]          # [t, grid_h, grid_w]
    entity: str = dump.get("entity", "?")
    double: bool = bool(dump.get("double_frames", False))
    t = heatmap.shape[0]

    # Score per cella temporale: massa (sum) o picco (max) sulla griglia HxW.
    if metric == "sum":
        frame_scores = heatmap.flatten(1).sum(dim=1)
    else:
        frame_scores = heatmap.flatten(1).max(dim=1).values

    vr = decord.VideoReader(video_path)
    video_fps = float(vr.get_avg_fps())
    if dump.get("frame_indices") is not None:
        frame_idx = list(dump["frame_indices"])
    else:
        # dump vecchio: ricostruisci il campionamento uniforme di qwen.
        # merge a coppie → 2t frame campionati; doubling → t frame.
        nframes = t if double else t * 2
        frame_idx = torch.linspace(0, len(vr) - 1, nframes).round().long().tolist()

    def cell_to_frames(ti: int) -> tuple[list[int], int]:
        """(frame reali della cella, frame rappresentante)."""
        if double:
            f = int(frame_idx[ti])
            return [f], f
        f0, f1 = int(frame_idx[2 * ti]), int(frame_idx[2 * ti + 1])
        return [f0, f1], int(round((f0 + f1) / 2))

    order = torch.argsort(frame_scores, descending=True)
    total_mass = frame_scores.sum().item()

    mode = "frame-doubling (cella=1 frame)" if double else "merge a coppie (cella=2 frame)"
    print(f"entity={entity!r}  video={video_path}  fps={video_fps:.2f}  "
          f"celle_temporali={t}  regime={mode}")
    print(f"metrica={metric}  massa_totale_sui_visivi={total_mass:.4f}")
    print(f"\n{'rank':>4} {'cella':>5} {'score':>9} {'%mass':>7} "
          f"{'frame_reali':>16} {'t_sec':>8}")
    for rank, ti in enumerate(order.tolist()):
        frames, rep = cell_to_frames(ti)
        t_mid = rep / video_fps
        sc = frame_scores[ti].item()
        pct = 100 * sc / total_mass if total_mass else 0.0
        mark = "  <-- top" if rank < topk else ""
        lbl = f"[{','.join(map(str, frames))}]"
        print(f"{rank:>4} {ti:>5} {sc:>9.5f} {pct:>6.1f}% {lbl:>16} {t_mid:>8.2f}{mark}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for rank, ti in enumerate(order[:topk].tolist()):
        _, real = cell_to_frames(ti)
        frame = vr[real].asnumpy()
        t_mid = real / video_fps
        stem = f"rank{rank}_cell{ti:02d}_frame{real}_t{t_mid:.1f}s"
        out = outdir / f"{stem}.png"
        Image.fromarray(frame).save(out)
        saved.append(out)
        if overlay:
            ov = outdir / f"{stem}_overlay.png"
            make_overlay(frame, heatmap[ti], zero_border=zero_border).save(ov)
            saved.append(ov)

    print(f"\nsalvati {len(saved)} frame in {outdir}/")
    for p in saved:
        print(" ", p)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor", default="tensors/debug_knife.pt", help="dump di entity_visual_attention")
    ap.add_argument("--video", default=None, help="video .mp4 (default: dal dump, poi ./videos/)")
    ap.add_argument("--topk", type=int, default=5, help="quanti frame estrarre (default 5)")
    ap.add_argument("--metric", choices=["sum", "max"], default="sum",
                    help="'sum' = massa totale di attenzione sulla cella; 'max' = picco (default sum)")
    ap.add_argument("--outdir", default=None, help="cartella PNG (default: <tensor_dir>/attended_frames)")
    ap.add_argument("--overlay", action="store_true",
                    help="salva anche l'overlay spaziale della heatmap sul frame")
    ap.add_argument("--zero-border", type=int, default=0,
                    help="azzera N anelli di bordo della heatmap prima dell'overlay (sink di bordo)")
    args = ap.parse_args()

    dump = torch.load(args.tensor, map_location="cpu")

    # Risali al video: flag esplicito > path nel dump > primo .mp4 in ./videos/.
    video_path = args.video or dump.get("video_path")
    if video_path is None:
        cands = sorted(Path("videos").glob("*.mp4"))
        if not cands:
            raise SystemExit("nessun --video passato, niente video_path nel dump, nessun .mp4 in ./videos/")
        video_path = str(cands[0])

    outdir = args.outdir or (Path(args.tensor).parent / "attended_frames")
    extract_attended_frames(
        dump, str(video_path),
        topk=args.topk, metric=args.metric, outdir=outdir,
        overlay=args.overlay, zero_border=args.zero_border,
    )


if __name__ == "__main__":
    main()
