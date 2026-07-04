"""Server Flask interattivo per esplorare l'attenzione testo→visivo.

Sostituisce il vecchio report HTML statico (`utils/make_report.py`): invece di
fissare un'entity al momento della cattura, qui le entity si scelgono a RUNTIME.
Si punta a uno o più STORE prodotti da `attn_explorer.capture` (cartelle con
`<stem>/capture.pt`, `<stem>/frames/` e `index.json`):

    uv run python -m attn_explorer.serve --store debug_out/

Multi-store: `--store` accetta più cartelle, i video di tutte vengono
elencati insieme in `/`:

    uv run python -m attn_explorer.serve --store data/vnbench/output/ret data/vnbench/output/count

Gli stem devono essere UNICI fra gli store dati (nessun dedup/merge): se due
store hanno lo stesso stem vince il primo nell'ordine passato a `--store`.

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
`overlay` accettano `sink_view=all|sink|nonsink&sink_percentile=25&sink_border=0`
per scegliere cosa mostrare rispetto alla classificazione sink (metodo
"sink-dims" di `lot/retrieval.py`):
  - `all` (default): attenzione grezza intatta, nessun azzeramento.
  - `sink`: solo la massa di attenzione sulle celle sink (le altre azzerate).
  - `nonsink`: solo la massa sulle celle non-sink (le sink azzerate).
Senza `sink_map` nel dump (capture legacy) la view è no-op.

`sink_border` è OFF di default (0): forzerebbe a sink gli anelli di bordo
della griglia a prescindere dal loro `sink_score` reale, un'euristica non
verificata (vedi `utils.attn_core.sink_mask`). Opt-in via query string per
chi vuole testarla empiricamente.

`--videos <cartella>`: se il path assoluto del video salvato nel capture non
esiste più (es. dataset spostato dopo la cattura), `/video/<stem>` cerca il
file per nome ricorsivamente sotto questa cartella.
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
    parse_answer_meta,
    question_rows,
    rank_cells,
    sink_mask,
    sink_view,
)

# Default della classificazione sink (allineati a `lot/retrieval.py`).
DEFAULT_SINK_PERCENTILE = 25.0
DEFAULT_SINK_BORDER = 0  # opt-in via ?sink_border=N — vedi utils.attn_core.sink_mask
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


def create_app(stores: Path | list[Path], video_root: Path | None = None) -> Flask:
    """Costruisce l'app Flask legata a una o più cartelle-store.

    `stores`: un path singolo o una lista di path di STORE (ciascuno con la
    sua `index.json`). Con più store i video vengono elencati insieme in
    `/`; lo stem deve essere unico fra gli store dati (vince il primo store
    dell'elenco in caso di collisione — vedi `_store_for_stem`).

    `video_root` (opzionale): cartella radice in cui cercare i video
    originali per nome file (ricerca ricorsiva), usata da `/video/<stem>`
    quando il path assoluto salvato nel capture (`AttentionCapture.video_path`,
    fissato al momento di `attn_explorer.capture`) non esiste più — es. dopo
    che i video sono stati spostati altrove sul filesystem.
    """
    if isinstance(stores, (str, Path)):
        stores = [stores]
    stores = [Path(s).resolve() for s in stores]
    video_root = Path(video_root).resolve() if video_root else None
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

    @lru_cache(maxsize=256)
    def resolve_video(name: str) -> Path | None:
        """Cerca `name` ricorsivamente sotto `video_root` (None se non trovato/non impostato)."""
        if video_root is None:
            return None
        return next(video_root.rglob(name), None)

    def sidecar_answer_meta(video_path_str: str) -> dict:
        """Fallback per capture fatti PRIMA che `index.json` portasse `answer_letter`
        /`answer_text` (vedi `attn_explorer.capture.capture_one`): rilegge il
        sidecar `<stem>.json` accanto al video (risolvendolo via `--videos` se il
        path salvato nel capture non esiste più) e ne estrae la risposta corretta."""
        video_path = Path(video_path_str)
        if not video_path.is_file():
            video_path = resolve_video(video_path.name) or video_path
        sidecar = video_path.with_suffix(".json")
        if not sidecar.is_file():
            return {}
        try:
            obj = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return parse_answer_meta(obj)

    @app.context_processor
    def inject_assets():
        # CSS/JS inlinati nelle pagine (zero static, un solo documento servito).
        return {
            "css": (TEMPLATE_DIR / "report.css").read_text(encoding="utf-8"),
            "js": (TEMPLATE_DIR / "app.js").read_text(encoding="utf-8"),
        }

    def load_index() -> list[dict]:
        """Concatena `index.json` di TUTTI gli store, taggando ogni video con
        `_store` (nome della cartella di provenienza) per distinguerli in UI."""
        videos = []
        for s in stores:
            index_path = s / "index.json"
            if not index_path.is_file():
                continue
            for v in json.loads(index_path.read_text(encoding="utf-8")).get("videos", []):
                videos.append({**v, "_store": s.name})
        return videos

    def store_for_stem(stem: str) -> Path:
        """Primo store (nell'ordine passato a `--store`) che contiene `stem`."""
        for s in stores:
            if (s / stem / "capture.pt").is_file():
                return s
        abort(404, f"capture non trovato per {stem!r} in nessuno degli store: {[str(s) for s in stores]}")

    @lru_cache(maxsize=32)
    def get_capture(stem: str) -> AttentionCapture:
        """Carica (e memoizza) l'AttentionCapture di un video dello store."""
        path = store_for_stem(stem) / stem / "capture.pt"
        return load_capture(path)

    def heatmap_for(stem: str, rows: list[int]):
        """Heatmap aggregata `[t, gh, gw]` per la selezione di token `rows`."""
        return get_capture(stem).aggregate(rows)

    # ── Pagine HTML ──
    @app.get("/")
    def index():
        return render_template("index.html", videos=load_index(), store=", ".join(str(s) for s in stores))

    @app.get("/v/<stem>")
    def video_page(stem: str):
        cap = get_capture(stem)
        tokens = [
            {"row": q.row, "text": q.text, "token": q.token}
            for q in cap.query_tokens
        ]
        # Risposta corretta: da index.json se presente, altrimenti fallback sul
        # sidecar (capture fatti prima che `capture_one` la scrivesse in indice).
        videos = {v["stem"]: v for v in load_index()}
        meta = videos.get(stem, {})
        if not meta.get("answer_letter"):
            meta = {**meta, **sidecar_answer_meta(cap.video_path)}

        options = meta.get("options") or []
        option_items = [
            {"letter": chr(ord("A") + i), "text": opt, "correct": chr(ord("A") + i) == meta.get("answer_letter")}
            for i, opt in enumerate(options)
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
            default_rows=question_rows(cap.query_tokens),
            answer_letter=meta.get("answer_letter"),
            answer_text=meta.get("answer_text"),
            option_items=option_items,
            has_sink_map=bool(cap.sink_map.abs().sum().item() > 0),
        )

    # ── Video originale ──
    @app.get("/video/<stem>")
    def serve_video(stem: str):
        cap = get_capture(stem)
        video_path = Path(cap.video_path)
        if not video_path.is_file():
            video_path = resolve_video(video_path.name) or video_path
        if not video_path.is_file():
            abort(404, f"video non trovato: {video_path} (prova --videos <cartella> se è stato spostato)")
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

        Query params: `sink_percentile` (default 25), `sink_border` (default 0, opt-in).
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
        path = store_for_stem(stem) / stem / "frames" / f"cell_{ti:04d}.png"
        if not path.is_file():
            abort(404)
        return Response(path.read_bytes(), mimetype="image/png")

    @app.get("/api/<stem>/overlay")
    def api_overlay(stem: str):
        cap = get_capture(stem)
        ti = int(request.args["cell"])
        rows = parse_tokens(request.args.get("tokens"))
        heatmap = cap.aggregate(rows)

        # View sink: 'all' = attenzione grezza intatta; 'sink'/'nonsink'
        # azzerano l'altra classe e renormalizzano. min/max globali sulla
        # heatmap VISTA mantengono le celle tra loro confrontabili.
        view, percentile, border = _sink_params()
        if cap.sink_map.abs().sum().item() > 0:
            heatmap, _ = sink_view(
                heatmap, cap.sink_map,
                view=view, percentile=percentile, border=border,
            )

        frame_path = store_for_stem(stem) / stem / "frames" / f"cell_{ti:04d}.png"
        if not frame_path.is_file():
            abort(404)
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        img = make_overlay(
            frame, heatmap[ti],
            vmin=float(heatmap.min()), vmax=float(heatmap.max()),
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")

    return app

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", nargs="+", default=["debug_out"], help="una o più cartelle STORE con i capture (default debug_out/); con più valori i video sono elencati insieme (stem univoci fra store)")
    ap.add_argument("--videos", default=None, help="cartella radice dove cercare i video originali per nome (ricorsivo), se il path salvato nel capture non è più valido")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true", help="modalità debug di Flask (reload + traceback)")
    args = ap.parse_args()

    stores = [Path(s) for s in args.store]
    missing = [s for s in stores if not s.is_dir()]
    if missing:
        raise SystemExit(f"store non trovati: {', '.join(str(m) for m in missing)}")
    if args.videos and not Path(args.videos).is_dir():
        raise SystemExit(f"cartella --videos non trovata: {args.videos}")
    app = create_app(stores, video_root=args.videos)
    print(f"Store: {', '.join(str(s.resolve()) for s in stores)}  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
