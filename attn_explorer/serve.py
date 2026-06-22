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
  GET /api/<stem>/frame?cell=ti      PNG del frame rappresentante della cella
  GET /api/<stem>/overlay?cell=ti&tokens=0,3   PNG overlay rigenerato per la selezione

Selezionando token diversi nella pagina, i frame vengono RIORDINATI per massa di
attenzione (via `AttentionCapture.aggregate` + `rank_cells`) e gli overlay sono
rigenerati per quella selezione. Senza token selezionati si usa la media di
tutti i token della domanda (ordinamento di default).
"""
import argparse
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request
from PIL import Image

from utils.attn_core import AttentionCapture, load_capture, make_overlay, rank_cells

TEMPLATE_DIR = Path(__file__).parent / "templates"


def parse_tokens(raw: str | None) -> list[int]:
    """Parsa `?tokens=0,3,5` in una lista di righe-query (0-based)."""
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


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
        return render_template(
            "video.html",
            stem=stem,
            video_name=Path(cap.video_path).name,
            prompt=cap.prompt,
            fps=cap.fps,
            cells=cap.grid.t,
            regime=cap.grid.regime,
            tokens=tokens,
        )

    # ── API ──
    @app.get("/api/<stem>/rank")
    def api_rank(stem: str):
        cap = get_capture(stem)
        rows = parse_tokens(request.args.get("tokens"))
        heatmap = cap.aggregate(rows)
        metric = request.args.get("metric", "sum")
        ranked = rank_cells(heatmap, cap.grid, cap.frame_indices, cap.fps,
                            metric=metric, topk=int(request.args.get("topk", 8)))
        return jsonify({
            "stem": stem,
            "selected": rows,
            "total_mass": float(heatmap.sum()),
            "cells": [c.as_dict() for c in ranked],
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
        frame_path = store / stem / "frames" / f"cell_{ti:04d}.png"
        if not frame_path.is_file():
            abort(404)
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        # Scala globale (min/max su tutta la heatmap aggregata) → overlay
        # confrontabili fra celle.
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
