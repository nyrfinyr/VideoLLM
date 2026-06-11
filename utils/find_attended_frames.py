"""Trova i frame del video dove l'attenzione entity→visivo è più alta.

Legge il dump prodotto in debug da `entity_visual_attention` (default
`tensors/debug_knife.pt`: dict con `heatmap [t, grid_h, grid_w]`,
`grid_thw`, `entity`, ...) e classifica le `t` CELLE TEMPORALI per massa di
attenzione. Ogni cella temporale è il merge ×2 di due frame campionati, e i
64 frame sono presi uniformemente dal video (`linspace(0, total-1, nframes)`,
stesso schema di `qwen_vl_utils._read_video_decord`). Ricostruito quel
campionamento, mappa ogni cella sugli indici/timestamp reali del .mp4 ed
estrae i top-K come PNG.

    uv run python utils/find_attended_frames.py
    uv run python utils/find_attended_frames.py --topk 8 \
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor", default="tensors/debug_knife.pt", help="dump di entity_visual_attention")
    ap.add_argument("--video", default=None, help="video .mp4 (default: cerca in ./videos/ per nome)")
    ap.add_argument("--topk", type=int, default=5, help="quanti frame estrarre (default 5)")
    ap.add_argument("--metric", choices=["sum", "max"], default="sum",
                    help="'sum' = massa totale di attenzione sulla cella; 'max' = picco (default sum)")
    ap.add_argument("--outdir", default=None, help="cartella PNG (default: <tensor_dir>/attended_frames)")
    ap.add_argument("--overlay", action="store_true",
                    help="salva anche l'overlay spaziale della heatmap sul frame")
    ap.add_argument("--zero-border", type=int, default=0,
                    help="azzera N anelli di bordo della heatmap prima dell'overlay (sink di bordo)")
    args = ap.parse_args()

    d = torch.load(args.tensor, map_location="cpu")
    heatmap: torch.Tensor = d["heatmap"]          # [t, grid_h, grid_w]
    entity: str = d.get("entity", "?")
    t = heatmap.shape[0]
    nframes = t * 2                               # merge temporale ×2

    # Score per cella temporale: massa (sum) o picco (max) sulla griglia HxW.
    if args.metric == "sum":
        frame_scores = heatmap.flatten(1).sum(dim=1)
    else:
        frame_scores = heatmap.flatten(1).max(dim=1).values

    # Risali al video: per nome accanto al tensore o nel flag esplicito.
    video_path = args.video
    if video_path is None:
        # heuristica: usa il primo .mp4 in ./videos/ (il dump non porta il path)
        cands = sorted(Path("videos").glob("*.mp4"))
        if not cands:
            raise SystemExit("nessun --video passato e nessun .mp4 in ./videos/")
        video_path = str(cands[0])

    frame_idx, video_fps = reconstruct_frame_indices(video_path, nframes)

    order = torch.argsort(frame_scores, descending=True)
    total_mass = frame_scores.sum().item()

    print(f"entity={entity!r}  video={video_path}  fps={video_fps:.2f}  "
          f"celle_temporali={t} (=> {nframes} frame campionati)")
    print(f"metrica={args.metric}  massa_totale_sui_visivi={total_mass:.4f}")
    print(f"\n{'rank':>4} {'cella':>5} {'score':>9} {'%mass':>7} "
          f"{'frame_reali':>16} {'t_sec':>8}")
    for rank, ti in enumerate(order.tolist()):
        # cella ti = frame campionati 2*ti, 2*ti+1
        f0, f1 = frame_idx[2 * ti], frame_idx[2 * ti + 1]
        t_mid = ((f0 + f1) / 2) / video_fps
        sc = frame_scores[ti].item()
        pct = 100 * sc / total_mass if total_mass else 0.0
        mark = "  <-- top" if rank < args.topk else ""
        print(f"{rank:>4} {ti:>5} {sc:>9.5f} {pct:>6.1f}% "
              f"{f'[{f0},{f1}]':>16} {t_mid:>8.2f}{mark}")

    # Estrai i top-K frame come PNG (frame medio della coppia merge).
    outdir = Path(args.outdir) if args.outdir else Path(args.tensor).parent / "attended_frames"
    outdir.mkdir(parents=True, exist_ok=True)

    import decord
    from PIL import Image

    vr = decord.VideoReader(video_path)
    saved = []
    for rank, ti in enumerate(order[: args.topk].tolist()):
        f0, f1 = frame_idx[2 * ti], frame_idx[2 * ti + 1]
        real = int(round((f0 + f1) / 2))
        frame = vr[real].asnumpy()
        t_mid = real / video_fps
        stem = f"rank{rank}_cell{ti:02d}_frame{real}_t{t_mid:.1f}s"
        out = outdir / f"{stem}.png"
        Image.fromarray(frame).save(out)
        saved.append(out)
        if args.overlay:
            ov = outdir / f"{stem}_overlay.png"
            make_overlay(frame, heatmap[ti], zero_border=args.zero_border).save(ov)
            saved.append(ov)

    print(f"\nsalvati {len(saved)} frame in {outdir}/")
    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
