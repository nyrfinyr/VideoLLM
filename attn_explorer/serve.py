"""Server Flask interattivo per esplorare l'attenzione testo→visivo.

Sostituisce il vecchio report HTML statico (`utils/make_report.py`): invece di
fissare un'entity al momento della cattura, qui le entity si scelgono a RUNTIME.
Si punta a uno STORE prodotto da `attn_explorer.capture` (cartella con
`<stem>/capture.pt`, `<stem>/frames/` e `index.json`):

    uv run python -m attn_explorer.serve --store debug_out/

Route:
  GET /                              elenco dei video dello store
  GET /v/<stem>                      pagina interattiva del video (chip token + grid)
  GET /api/<stem>/rank?tokens=0,3    JSON: celle ordinate per massa delle entity scelte
  GET /api/<stem>/sink                JSON: sink_map + maschera sink (percentile + bordo)
  GET /api/<stem>/frame?cell=ti      PNG del frame rappresentante della cella
  GET /api/<stem>/overlay?cell=ti&tokens=0,3   PNG overlay rigenerato per la selezione

Selezionando token diversi nella pagina, i frame vengono RIORDINATI per massa di
attenzione (via `AttentionCapture.aggregate` + `rank_cells`) e gli overlay sono
rigenerati per quella selezione. Senza token selezionati si usa la media di
tutti i token della domanda (ordinamento di default).

Filtro sink (opzionale, da dump prodotto con `sink_map`): gli endpoint `rank` e
`overlay` accettano `filter_sink=true&sink_percentile=25&sink_border=2` per
azzerare i token visivi-sink prima del ranking/overlay (metodo "sink-dims" di
`lot/retrieval.py`). Default `filter_sink=false` (mostra l'attenzione grezza).
Senza `sink_map` nel dump (capture legacy) il filtro è no-op.
"""
import argparse
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, render_template, request
from PIL import Image

from utils.attn_core import (
    AttentionCapture,
    filter_sinks,
    load_capture,
    make_overlay,
    rank_cells,
    sink_mask,
)

# Default del filtro sink (allineati a `lot/retrieval.py`).
DEFAULT_SINK_PERCENTILE = 25.0
DEFAULT_SINK_BORDER = 2

TEMPLATE_DIR = Path(__file__).parent / "templates"


def parse_tokens(raw: str | None) -> list[int]:
    """Parsa `?tokens=0,3,5` in una lista di righe-query (0-based)."""
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def _bool_arg(name: str, default: bool) -> bool:
    """Parse `?flag=true|false` (default se assente)."""
    raw = request.args.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sink_params() -> tuple[bool, float, int]:
    """Legga `(filter_sink, percentile, border)` dalla query string.

    `filter_sink` default `false` (attenzione grezza). Senza `sink_map` nel
    dump il filtro sarà no-op anche se richiesto.
    """
    filter_sink = _bool_arg("filter_sink", default=False)
    percentile = float(request.args.get("sink_percentile", DEFAULT_SINK_PERCENTILE))
    border = int(request.args.get("sink_border", DEFAULT_SINK_BORDER))
    percentile = max(0.0, min(100.0, percentile))
    border = max(0, border)
    return filter_sink, percentile, border


def create_app(store: Path) -> Flask:
    """Costruisce l'app Flask legata a una cartella-store."""
    store = Path(store).resolve()
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

    @app.context_processor
    def inject_assets():
        # CSS/JS inlinati nelle pagine (zero static, un solo documento servito).
        return {
            "css": (TEMPLATE_DIR / "report.css").read_text(encoding="utf-8"),
            "js": (TEMPLATE_DIR / "app.js").read_text(encoding="utf-8"),
        }

    def load_index() -> list[dict]:
        index_path = store / "index.json"
        if not index_path.is_file():
            return []
        return json.loads(index_path.read_text(encoding="utf-8")).get("videos", [])

    @lru_cache(maxsize=32)
    def get_capture(stem: str) -> AttentionCapture:
        """Carica (e memoizza) l'AttentionCapture di un video dello store."""
        path = store / stem / "capture.pt"
        if not path.is_file():
            abort(404, f"capture non trovato per {stem!r}")
        return load_capture(path)

    def heatmap_for(stem: str, rows: list[int]):
        """Heatmap aggregata `[t, gh, gw]` per la selezione di token `rows`."""
        return get_capture(stem).aggregate(rows)

    # ── Pagine HTML ──
    @app.get("/")
    def index():
        return render_template("index.html", videos=load_index(), store=str(store))

    @app.get("/v/<stem>")
    def video_page(stem: str):
        cap = get_capture(stem)
        tokens = [
            {"row": q.row, "text": q.text, "token": q.token}
            for q in cap.query_tokens
        ]
        # Look up answer/options from index
        videos = {v["stem"]: v for v in load_index()}
        meta = videos.get(stem, {})
        return render_template(
            "video.html",
            stem=stem,
            video_name=Path(cap.video_path).name,
            prompt=cap.prompt,
            fps=cap.fps,
            cells=cap.grid.t,
            regime=cap.grid.regime,
            tokens=tokens,
            answer=meta.get("answer"),
            options=meta.get("options", []),
        )

    # ── Video originale ──
    @app.get("/video/<stem>")
    def serve_video(stem: str):
        cap = get_capture(stem)
        video_path = Path(cap.video_path)
        if not video_path.is_file():
            abort(404, f"video non trovato: {video_path}")
        return Response(
            video_path.read_bytes(),
            mimetype="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Disposition": f"inline; filename={video_path.name}"},
        )

    # ── API ──
    @app.get("/api/<stem>/rank")
    def api_rank(stem: str):
        cap = get_capture(stem)
        rows = parse_tokens(request.args.get("tokens"))
        heatmap = cap.aggregate(rows)
        metric = request.args.get("metric", "sum")

        # Filtro sink opzionale: se richiesto e il dump ha un sink_map
        # non-zero, si azzerano i token-sink e si renormalizza prima del
        # ranking. Ritorniamo anche la mask così la UI può mostrarla.
        filter_sink, percentile, border = _sink_params()
        sink_applied = False
        n_sink = 0
        if filter_sink and cap.sink_map.abs().sum().item() > 0:
            heatmap, mask = filter_sinks(
                heatmap, cap.sink_map,
                percentile=percentile, border=border,
            )
            sink_applied = True
            n_sink = int(mask.sum().item())

        ranked = rank_cells(heatmap, cap.grid, cap.frame_indices, cap.fps,
                            metric=metric, topk=int(request.args.get("topk", 8)))
        result: dict = {
            "stem": stem,
            "selected": rows,
            "total_mass": float(heatmap.sum()),
            "cells": [c.as_dict() for c in ranked],
        }
        if filter_sink:
            result["sink"] = {
                "applied": sink_applied,
                "percentile": percentile,
                "border": border,
                "n_sink": n_sink,
                "has_sink_map": bool(cap.sink_map.abs().sum().item() > 0),
            }
        return jsonify(result)

    @app.get("/api/<stem>/sink")
    def api_sink(stem: str):
        """JSON: sink_map + maschera sink calcolata a runtime.

        Query params: `sink_percentile` (default 25), `sink_border` (default 2).
        Utile per ispezionare/distinguere sink vs non-sink indipendentemente
        dalla selezione di token-query. `has_sink_map=false` se il dump è
        legacy (senza `sink_map`).
        """
        cap = get_capture(stem)
        percentile = float(request.args.get("sink_percentile", DEFAULT_SINK_PERCENTILE))
        border = int(request.args.get("sink_border", DEFAULT_SINK_BORDER))
        percentile = max(0.0, min(100.0, percentile))
        border = max(0, border)

        has_sink = cap.sink_map.abs().sum().item() > 0 and len(cap.sink_dims) > 0
        if has_sink:
            mask = sink_mask(cap.sink_map, percentile=percentile, border=border)
        else:
            mask = torch.zeros_like(cap.sink_map, dtype=torch.bool)

        return jsonify({
            "stem": stem,
            "has_sink_map": has_sink,
            "sink_dims": list(cap.sink_dims),
            "percentile": percentile,
            "border": border,
            "grid": {"t": cap.grid.t, "grid_h": cap.grid.grid_h, "grid_w": cap.grid.grid_w},
            "sink_map": cap.sink_map.tolist(),
            "is_sink": mask.tolist(),
            "n_sink": int(mask.sum().item()),
            "n_total": int(mask.numel()),
        })

    @app.get("/api/<stem>/frame")
    def api_frame(stem: str):
        ti = int(request.args["cell"])
        path = store / stem / "frames" / f"cell_{ti:04d}.png"
        if not path.is_file():
            abort(404)
        return Response(path.read_bytes(), mimetype="image/png")

    @app.get("/api/<stem>/overlay")
    def api_overlay(stem: str):
        cap = get_capture(stem)
        ti = int(request.args["cell"])
        rows = parse_tokens(request.args.get("tokens"))
        heatmap = cap.aggregate(rows)

        # Filtro sink: se richiesto e `sink_map` è presente, si azzerano i
        # token-sink della heatmap aggregata e si renormalizza. L'overlay
        # usa poi min/max globali della heatmap FILTRATA per celle
        # confrontabili.
        filter_sink, percentile, border = _sink_params()
        if filter_sink and cap.sink_map.abs().sum().item() > 0:
            heatmap, _mask = filter_sinks(
                heatmap, cap.sink_map,
                percentile=percentile, border=border,
            )

        frame_path = store / stem / "frames" / f"cell_{ti:04d}.png"
        if not frame_path.is_file():
            abort(404)
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        img = make_overlay(frame, heatmap[ti], vmin=float(heatmap.min()), vmax=float(heatmap.max()))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")

    return app

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="debug_out", help="cartella STORE con i capture (default debug_out/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true", help="modalità debug di Flask (reload + traceback)")
    args = ap.parse_args()

    store = Path(args.store)
    if not store.is_dir():
        raise SystemExit(f"store non trovato: {store}")
    app = create_app(store)
    print(f"Store: {store.resolve()}  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
