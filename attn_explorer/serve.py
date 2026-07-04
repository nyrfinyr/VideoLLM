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

View sink (opzionale, da dump prodotto con `sink_map`): gli endpoint `rank` e
`overlay` accettano `sink_view=all|sink|nonsink&sink_percentile=25&sink_border=2`
per scegliere cosa mostrare rispetto alla classificazione sink (metodo
"sink-dims" di `lot/retrieval.py`):
  - `all` (default): attenzione grezza intatta; in overlay le celle sink sono
    solo evidenziate con un contorno, mai azzerate.
  - `sink`: solo la massa di attenzione sulle celle sink (le altre azzerate).
  - `nonsink`: solo la massa sulle celle non-sink (le sink azzerate).
Senza `sink_map` nel dump (capture legacy) la view è no-op.
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
    load_capture,
    make_overlay,
    rank_cells,
    sink_mask,
    sink_view,
)

# Default della classificazione sink (allineati a `lot/retrieval.py`).
DEFAULT_SINK_PERCENTILE = 25.0
DEFAULT_SINK_BORDER = 2
SINK_VIEWS = ("all", "sink", "nonsink")

TEMPLATE_DIR = Path(__file__).parent / "templates"


def parse_tokens(raw: str | None) -> list[int]:
    """Parsa `?tokens=0,3,5` in una lista di righe-query (0-based)."""
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def _sink_params() -> tuple[str, float, int]:
    """Legga `(view, percentile, border)` dalla query string.

    `view` (`?sink_view=all|sink|nonsink`, default `all`) NON azzera mai
    l'attenzione grezza da sola: seleziona solo cosa mostrare (tutto, solo
    celle sink, solo celle non-sink). Senza `sink_map` nel dump è no-op.
    """
    view = request.args.get("sink_view", "all").strip().lower()
    if view not in SINK_VIEWS:
        abort(400, f"sink_view deve essere uno tra {SINK_VIEWS}, got {view!r}")
    percentile = float(request.args.get("sink_percentile", DEFAULT_SINK_PERCENTILE))
    border = int(request.args.get("sink_border", DEFAULT_SINK_BORDER))
    percentile = max(0.0, min(100.0, percentile))
    border = max(0, border)
    return view, percentile, border


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
            has_sink_map=bool(cap.sink_map.abs().sum().item() > 0),
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

        # View sink opzionale: 'all' mostra la massa grezza (nessuna
        # esclusione), 'sink'/'nonsink' classificano prima del ranking così
        # si può ordinare le celle guardando solo i token-sink o solo quelli
        # non-sink. Nessuna view azzera l'attenzione "per sempre": è solo una
        # selezione applicata a questa risposta.
        view, percentile, border = _sink_params()
        n_sink = 0
        has_sink_map = bool(cap.sink_map.abs().sum().item() > 0)
        if has_sink_map:
            heatmap, mask = sink_view(
                heatmap, cap.sink_map,
                view=view, percentile=percentile, border=border,
            )
            n_sink = int(mask.sum().item())

        ranked = rank_cells(heatmap, cap.grid, cap.frame_indices, cap.fps,
                            metric=metric, topk=int(request.args.get("topk", 8)))
        result: dict = {
            "stem": stem,
            "selected": rows,
            "total_mass": float(heatmap.sum()),
            "cells": [c.as_dict() for c in ranked],
            "sink": {
                "view": view,
                "percentile": percentile,
                "border": border,
                "n_sink": n_sink,
                "has_sink_map": has_sink_map,
            },
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

        # View sink: 'all' non tocca la heatmap (attenzione grezza) ma disegna
        # il contorno delle celle sink così restano distinguibili a vista;
        # 'sink'/'nonsink' azzerano l'altra classe e renormalizzano — in
        # quei due casi il contorno è ridondante (le celle escluse sono già
        # a zero) e viene omesso. min/max globali sulla heatmap VISTA
        # mantengono le celle tra loro confrontabili.
        view, percentile, border = _sink_params()
        mask = None
        if cap.sink_map.abs().sum().item() > 0:
            heatmap, cell_mask = sink_view(
                heatmap, cap.sink_map,
                view=view, percentile=percentile, border=border,
            )
            if view == "all":
                mask = cell_mask

        frame_path = store / stem / "frames" / f"cell_{ti:04d}.png"
        if not frame_path.is_file():
            abort(404)
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        img = make_overlay(
            frame, heatmap[ti],
            vmin=float(heatmap.min()), vmax=float(heatmap.max()),
            sink_cell_mask=mask[ti] if mask is not None else None,
        )
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
