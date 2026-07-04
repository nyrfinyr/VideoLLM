"""Valida se l'attenzione catturata da `attn_explorer.capture` individua
correttamente i due needle (causa/effetto) di una domanda Causal2Needles.

Confronta il ranking delle celle temporali (`utils.attn_core.rank_cells`)
con le finestre ground-truth `cause_time`/`effect_time` (secondi) salvate
nel sidecar `.json` dal fetch script
(`attn_explorer.script.fetch_causal2needles_samples`). Per ogni video già
catturato in uno STORE (`attn_explorer.capture --outdir ...`):

    1. Aggrega l'attenzione su TUTTI i token della domanda (default) o su
       un sottoinsieme via `--entity` (substring match case-insensitive
       su `QueryToken.text`, es. `--entity kryptonite`) — nessun bisogno
       di ricaricare il modello, lavora solo sul dump salvato.
    2. Classifica le celle temporali per massa di attenzione
       (`rank_cells`, stesso ranking mostrato dal quick-check di
       `capture.py`).
    3. Per ciascun needle (causa, effetto), trova le celle la cui
       timestamp rappresentativa cade nella finestra ground-truth e
       riporta:
         - `best_rank`: rank (0 = top) della cella con score più alto tra
           quelle in finestra. `None` se nessuna cella della griglia
           campionata cade nella finestra (video sotto-campionato o
           finestra fuori range).
         - `mass_pct`: percentuale della massa di attenzione TOTALE che
           cade nella finestra.
         - `hit_topk`: `best_rank is not None and best_rank < topk`.

Semplificazione nota: la finestra è confrontata contro `t_sec`, il frame
RAPPRESENTANTE di ogni cella (cfr. `GridSpec.cell_to_frames`) — non contro
l'intervallo temporale effettivamente coperto dalla cella. Con
`--double-frames` (cella=1 frame) l'approssimazione è stretta; nel regime
di merge a coppie (cella=2 frame) una cella può "mancare" per un pelo una
finestra molto stretta che cade tra i due frame che copre.

Uso:
    uv run python -m attn_explorer.validate_causal2needles \
        --store debug_out_causal2needles --videos-dir videos/causal-2-needles

    # solo sui token che matchano "kryptonite" invece che sulla domanda intera
    uv run python -m attn_explorer.validate_causal2needles --entity kryptonite
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from utils.attn_core import RankedCell, load_capture, rank_cells

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validate_causal2needles")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="debug_out_causal2needles", help="STORE prodotto da attn_explorer.capture (default debug_out_causal2needles)")
    ap.add_argument("--videos-dir", default="videos/causal-2-needles", help="cartella con i sidecar .json (cause_time/effect_time) (default videos/causal-2-needles)")
    ap.add_argument("--topk", type=int, default=5, help="soglia per hit_topk (default 5)")
    ap.add_argument("--metric", choices=["sum", "max"], default="sum", help="metrica di ranking celle (default sum)")
    ap.add_argument("--entity", default=None, help="se dato, aggrega solo i token della domanda che contengono questa substring (case-insensitive); default: tutta la domanda")
    return ap.parse_args()


def select_rows(query_tokens, entity: str | None) -> list[int]:
    """Righe da aggregare: tutte (default) o quelle che matchano `entity`."""
    if not entity:
        return []
    needle = entity.lower()
    rows = [t.row for t in query_tokens if needle in t.text.lower()]
    if not rows:
        logger.warning("nessun token della domanda contiene %r, uso tutta la domanda", entity)
        return []
    return rows


def window_stats(ranked: list[RankedCell], window: list[float]) -> dict:
    """`best_rank`/`mass_pct` delle celle di `ranked` che cadono in `window`."""
    begin, end = window
    in_window = [c for c in ranked if begin <= c.t_sec <= end]
    if not in_window:
        return {"best_rank": None, "mass_pct": 0.0, "n_cells": 0}
    return {
        "best_rank": min(c.rank for c in in_window),
        "mass_pct": sum(c.pct for c in in_window),
        "n_cells": len(in_window),
    }


def main() -> None:
    args = parse_args()
    store = Path(args.store)
    videos_dir = Path(args.videos_dir)

    index_path = store / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"indice non trovato: {index_path} (hai lanciato attn_explorer.capture su questo store?)")
    stems = [e["stem"] for e in json.loads(index_path.read_text())["videos"]]

    rows_report = []
    for stem in stems:
        cap_path = store / stem / "capture.pt"
        sidecar_path = videos_dir / f"{stem}.json"
        if not cap_path.is_file():
            logger.warning("[%s] capture.pt mancante, salto", stem)
            continue
        if not sidecar_path.is_file():
            logger.warning("[%s] sidecar %s mancante, salto", stem, sidecar_path)
            continue

        cap = load_capture(cap_path)
        meta = json.loads(sidecar_path.read_text())
        if "cause_time" not in meta or "effect_time" not in meta:
            logger.warning("[%s] sidecar senza cause_time/effect_time (rilancia il fetch script), salto", stem)
            continue

        rows = select_rows(cap.query_tokens, args.entity)
        heatmap = cap.aggregate(rows)
        ranked = rank_cells(heatmap, cap.grid, cap.frame_indices, cap.fps, metric=args.metric, topk=args.topk)

        cause = window_stats(ranked, meta["cause_time"])
        effect = window_stats(ranked, meta["effect_time"])
        rows_report.append({"stem": stem, "cause": cause, "effect": effect})

        def fmt(w: dict) -> str:
            if w["best_rank"] is None:
                return "MISS (nessuna cella in finestra)"
            hit = "✓" if w["best_rank"] < args.topk else " "
            return f"rank={w['best_rank']:>3} mass={w['mass_pct']:5.1f}% [{hit} top-{args.topk}]"

        print(f"{stem:28s} causa: {fmt(cause):40s} effetto: {fmt(effect)}")

    if not rows_report:
        raise SystemExit("nessun video valutato (store/sidecar mancanti?)")

    n = len(rows_report)
    hit_cause = sum(1 for r in rows_report if r["cause"]["best_rank"] is not None and r["cause"]["best_rank"] < args.topk)
    hit_effect = sum(1 for r in rows_report if r["effect"]["best_rank"] is not None and r["effect"]["best_rank"] < args.topk)
    hit_both = sum(
        1 for r in rows_report
        if r["cause"]["best_rank"] is not None and r["cause"]["best_rank"] < args.topk
        and r["effect"]["best_rank"] is not None and r["effect"]["best_rank"] < args.topk
    )
    print(f"\n--- Aggregato su {n} video (top-{args.topk}, entity={args.entity or 'ALL'}) ---")
    print(f"hit-rate causa:    {hit_cause}/{n} ({100 * hit_cause / n:.0f}%)")
    print(f"hit-rate effetto:  {hit_effect}/{n} ({100 * hit_effect / n:.0f}%)")
    print(f"hit-rate entrambi: {hit_both}/{n} ({100 * hit_both / n:.0f}%)")


if __name__ == "__main__":
    main()
