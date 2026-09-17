"""Dipinge le heatmap di attenzione e la mappa dei sink SUI FRAME VERI.

Legge i dump di `strategies/topk_resample.py` (`dump.pt` + i PNG delle celle
calde e fredde) e produce, per ogni cella salvata, il frame con sopra due
sovrapposizioni: l'attenzione e il punteggio di sink. Serve a rispondere a
occhio alla domanda che i numeri da soli non chiudono: **i patch che il
modello chiama sink stanno su qualcosa, o sullo sfondo?** E specularmente:
l'attenzione alta cade su contenuto o sulle solite patch di bordo?

Due scelte che rendono le immagini confrontabili, invece che solo belle:

- **normalizzazione GLOBALE sul sample**, non per cella: il massimo del
  colore è il p99 su tutte le celle del video. Normalizzando ogni cella per
  sé, una cella fredda verrebbe fuori accesa come una calda e il confronto
  caldo/freddo — l'unico controllo che c'è — sparirebbe.
- **celle FREDDE accanto alle calde**: il dump salva i frame delle celle
  scelte dalle condizioni e di altrettante col ranking più basso. Senza,
  "i sink stanno sullo sfondo" non è falsificabile: sfondo ce n'è ovunque.

Uso:
    uv run python scripts/sink_heatmaps.py /work/.../dumps/<sample>
    uv run python scripts/sink_heatmaps.py /work/.../dumps --all
    uv run python scripts/sink_heatmaps.py <dump> --rowset question --alpha 0.6

Esce un `overlay_<mappa>_cell<NNNN>_<caldo|freddo>.png` per cella e un
contatto `contact_<mappa>.png` con tutte le celle in fila (calde sopra,
fredde sotto). Nient'altro che PIL e torch: gira anche sul nodo di login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# Rampa di colore in 5 tappe (blu → ciano → verde → giallo → rosso), stile
# "turbo" abbreviato: sequenziale e leggibile anche stampata in grigio, senza
# tirarsi dietro matplotlib (non è fra le dipendenze del progetto).
RAMP = ((30, 40, 120), (40, 160, 200), (60, 190, 90), (240, 200, 40), (200, 40, 40))


def colorize(value: float) -> tuple[int, int, int]:
    """`value` in [0,1] → colore sulla rampa, interpolando fra le due tappe."""
    v = min(max(value, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(v), len(RAMP) - 2)
    f = v - i
    a, b = RAMP[i], RAMP[i + 1]
    return tuple(int(a[c] + (b[c] - a[c]) * f) for c in range(3))


def overlay(frame: Image.Image, grid: torch.Tensor, vmax: float, alpha: float) -> Image.Image:
    """Frame + mappa `[gh, gw]` riscalata alla sua dimensione.

    Ingrandimento NEAREST di proposito: ogni patch è un rettangolo netto,
    perché la domanda è "quale PATCH è calda", e un'interpolazione morbida
    inventerebbe confini che il modello non ha.
    """
    gh, gw = grid.shape
    heat = Image.new("RGB", (gw, gh))
    heat.putdata([colorize(float(v) / vmax if vmax > 0 else 0.0) for v in grid.flatten()])
    heat = heat.resize(frame.size, Image.NEAREST)
    return Image.blend(frame.convert("RGB"), heat, alpha)


def label(img: Image.Image, text: str) -> Image.Image:
    """Striscia nera con la didascalia sotto l'immagine (quale cella, calda o
    fredda, che massa aveva): senza, i PNG sono indistinguibili fra loro."""
    out = Image.new("RGB", (img.width, img.height + 16), (0, 0, 0))
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((3, img.height + 3), text, fill=(235, 235, 235))
    return out


def p99(maps: torch.Tensor) -> float:
    """Scala del colore: il 99° percentile su TUTTE le celle del sample. Il
    massimo assoluto sarebbe ostaggio di un solo patch fuori scala, che
    schiaccerebbe tutto il resto sul blu."""
    flat = maps.flatten().float()
    return float(flat.kthvalue(max(1, int(0.99 * flat.numel()))).values)


def contact_sheet(tiles: list[Image.Image]) -> Image.Image:
    """Le celle in fila su una riga sola, nell'ordine dato (prima le calde)."""
    w = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
    sheet = Image.new("RGB", (w, max(t.height for t in tiles)), (0, 0, 0))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 4
    return sheet


def render(dump_dir: Path, rowset: str, alpha: float, out_dir: Path | None) -> None:
    d = torch.load(dump_dir / "dump.pt", weights_only=False)
    out_dir = out_dir or dump_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attn = d["attn"].get(rowset)
    if attn is None:
        raise SystemExit(f"rowset {rowset!r} non nel dump: {sorted(d['attn'])}")
    maps = {"attn": attn.float(), "sink": d["sink_map"].float()}
    scales = {k: p99(v) for k, v in maps.items()}
    masses = d["cell_mass_rank_rowset"]
    order = sorted(range(len(masses)), key=lambda i: -masses[i])
    rank = {c: order.index(c) + 1 for c in d["cell_frames"]}

    print(f"{dump_dir.name}: {d['t']} celle {d['grid_h']}x{d['grid_w']}, "
          f"rowset {rowset}, calde {d['cells_hot']}, fredde {d['cells_cold']}")
    for name, m in maps.items():
        tiles = []
        for cell in list(d["cells_hot"]) + list(d["cells_cold"]):
            files = d["cell_frames"].get(cell) or d["cell_frames"].get(str(cell))
            if not files:
                continue
            kind = "caldo" if cell in d["cells_hot"] else "freddo"
            frame = Image.open(dump_dir / files[0])
            img = overlay(frame, m[cell], scales[name], alpha)
            t = label(img, f"cella {cell} {kind} rank {rank.get(cell, '?')} "
                           f"t={d['pair_centers_sec'][cell]:.0f}s")
            t.save(out_dir / f"overlay_{name}_cell{cell:04d}_{kind}.png")
            tiles.append(t)
        if tiles:
            contact_sheet(tiles).save(out_dir / f"contact_{name}.png")
            print(f"  {name}: {len(tiles)} celle → contact_{name}.png (scala p99 = {scales[name]:.3g})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="cartella di un sample (contiene dump.pt) o radice con --all")
    ap.add_argument("--all", action="store_true", help="tratta `dump` come radice e processa ogni sottocartella")
    ap.add_argument("--rowset", default="all", help="quale heatmap d'attenzione (default: all)")
    ap.add_argument("--alpha", type=float, default=0.55, help="peso della heatmap sul frame (default 0.55)")
    ap.add_argument("--out", type=Path, default=None, help="cartella di uscita (default: dentro il dump)")
    args = ap.parse_args()

    dirs = sorted(p for p in args.dump.iterdir() if (p / "dump.pt").exists()) if args.all else [args.dump]
    if not dirs:
        raise SystemExit(f"nessun dump.pt sotto {args.dump}")
    for d in dirs:
        render(d, args.rowset, args.alpha, args.out / d.name if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
