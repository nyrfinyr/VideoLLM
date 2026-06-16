"""Report HTML self-contained dell'attenzione entity→visivo.

Genera UN singolo .html (immagini PNG embeddate in base64 → zero dipendenze di
cartelle, pronto da condividere) a partire dai prodotti di `debug_attention.py`.
Due modalità di input, normalizzate verso lo stesso HTML:

  1) JSON (consigliato) — il `debug_<entity>.json` contiene già tutto (header,
     KPI, ranking celle, path RELATIVI dei PNG). I path sono risolti rispetto
     alla cartella del JSON:

        uv run python utils/make_report.py debug_out/debug_duck.json

  2) OUTPUT TESTUALE (fallback per run senza JSON) — parsa lo stdout di
     `debug_attention.py` e globba i `rank*.png` nella cartella frame
     (numero di frame/celle non noto a priori):

        uv run python utils/make_report.py \
            --output debug_out/run.log --frames debug_out/attended_frames
        # oppure via pipe: ... | uv run python utils/make_report.py --frames ...

Solo stdlib: json/regex + base64.
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

# ── Regex per la modalità OUTPUT TESTUALE ──
# entity='duck'  video=videos/paperella.mp4  fps=29.25  celle_temporali=16  regime=frame-doubling (cella=1 frame)
RE_HEADER = re.compile(
    r"entity=(?P<entity>\S+)\s+video=(?P<video>\S+)\s+fps=(?P<fps>[\d.]+)\s+"
    r"celle_temporali=(?P<cells>\d+)\s+regime=(?P<regime>.+)$"
)
# metrica=sum  massa_totale_sui_visivi=0.3093
RE_MASS = re.compile(r"metrica=(?P<metric>\S+)\s+massa_totale_sui_visivi=(?P<mass>[\d.]+)")
#    0    10   0.07998   25.9%            [406]    13.88  <-- top
RE_ROW = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<cell>\d+)\s+(?P<score>[\d.]+)\s+(?P<pct>[\d.]+)%\s+"
    r"\[(?P<frames>[\d,]+)\]\s+(?P<tsec>[\d.]+)"
)
# rank0_cell10_frame406_t13.9s(_overlay).png
RE_FNAME = re.compile(
    r"rank(?P<rank>\d+)_cell(?P<cell>\d+)_frame(?P<frame>\d+)_t(?P<tsec>[\d.]+)s(?P<overlay>_overlay)?\.png$"
)


def b64(path: Path) -> str:
    """Codifica un PNG come data-URI base64 da incorporare nell'HTML."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento → struttura normalizzata
#
# Entrambe le modalità producono lo stesso dict:
#   { entity, fps, cells, regime, mass, video_name, metric,
#     cells_ranked: [ { rank, cell, score|None, pct|None, frames(list[int]),
#                       rep_frame, t_sec, is_top, image(Path|None), overlay(Path|None) } ] }
# I campi `image`/`overlay` sono Path ASSOLUTI (già risolti), pronti per b64.
# ─────────────────────────────────────────────────────────────────────────────
def load_from_json(json_path: Path) -> dict:
    """Modalità JSON: il dump di debug_attention è già nella forma giusta.

    Risolve i path PNG (relativi alla cartella di output = dir del JSON) in
    path assoluti.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    base = json_path.parent
    cells = []
    for c in data.get("cells_ranked", []):
        cells.append({
            "rank": c["rank"],
            "cell": c["cell"],
            "score": c.get("score"),
            "pct": c.get("pct"),
            "frames": c.get("frames", []),
            "rep_frame": c.get("rep_frame"),
            "t_sec": c.get("t_sec", 0.0),
            "is_top": bool(c.get("is_top")),
            "image": (base / c["image"]) if c.get("image") else None,
            "overlay": (base / c["overlay"]) if c.get("overlay") else None,
        })
    return {
        "entity": data.get("entity", "?"),
        "fps": float(data.get("fps", 25.0)),
        "cells": data.get("cells", len(cells)),
        "regime": data.get("regime", "?"),
        "mass": float(data.get("total_visual_mass", 0.0)),
        "video_name": data.get("video_name") or Path(data.get("video_path", "?")).name,
        "metric": data.get("metric", "sum"),
        "cells_ranked": cells,
    }


def load_from_output(text: str, frames_dir: Path) -> dict:
    """Modalità OUTPUT TESTUALE: parsa lo stdout e globba i PNG in cartella.

    Header/massa/righe-tabella vengono dal testo; i frame top (con PNG) dal
    glob di `rank*.png`. Le righe senza PNG restano in tabella ma non come card.
    """
    # ── Parsing del testo ──
    meta: dict = {}
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        if (m := RE_HEADER.search(line)):
            meta.update(m.groupdict())
        elif (m := RE_MASS.search(line)):
            meta.update(m.groupdict())
        elif (m := RE_ROW.match(line)):
            d = m.groupdict()
            rows[int(d["rank"])] = {
                "cell": int(d["cell"]),
                "score": float(d["score"]),
                "pct": float(d["pct"]),
                "frames": [int(x) for x in d["frames"].split(",")],
                "tsec": float(d["tsec"]),
            }

    # ── Glob dei PNG: raggruppa per rank in {cell, frame, tsec, orig, overlay} ──
    by_rank: dict[int, dict] = {}
    for p in sorted(frames_dir.glob("rank*.png")):
        if not (m := RE_FNAME.search(p.name)):
            continue
        d = m.groupdict()
        slot = by_rank.setdefault(int(d["rank"]), {
            "cell": int(d["cell"]), "frame": int(d["frame"]), "tsec": float(d["tsec"]),
        })
        slot["overlay" if d["overlay"] else "orig"] = p

    # ── Fusione: una riga per ogni rank presente in tabella o fra i PNG ──
    cells = []
    for rank in sorted(set(rows) | set(by_rank)):
        row = rows.get(rank, {})
        png = by_rank.get(rank, {})
        cells.append({
            "rank": rank,
            "cell": row.get("cell", png.get("cell")),
            "score": row.get("score"),
            "pct": row.get("pct"),
            "frames": row.get("frames", []),
            "rep_frame": png.get("frame"),
            "t_sec": row.get("tsec", png.get("tsec", 0.0)),
            "is_top": rank in by_rank,
            "image": png.get("orig"),
            "overlay": png.get("overlay"),
        })

    return {
        "entity": (meta.get("entity") or "?").strip("'\""),
        "fps": float(meta.get("fps", 25.0)),
        "cells": int(meta.get("cells", len(cells))),
        "regime": meta.get("regime", "?"),
        "mass": float(meta.get("mass", 0.0)),
        "video_name": Path(meta.get("video", "?")).name,
        "metric": meta.get("metric", "sum"),
        "cells_ranked": cells,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering HTML
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
  :root { --bg:#0f1117; --card:#171a23; --ink:#e6e9ef; --mut:#9aa3b2; --acc:#ffb454; }
  * { box-sizing:border-box; }
  body { margin:0; padding:32px; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  h1 { font-size:24px; margin:0 0 4px; }
  h1 .ent { color:var(--acc); }
  .meta { color:var(--mut); font-size:13px; margin-bottom:24px; }
  .meta code { color:var(--ink); background:#222633; padding:1px 6px; border-radius:4px; }
  .kpis { display:flex; gap:16px; flex-wrap:wrap; margin:0 0 28px; }
  .kpi { background:var(--card); border:1px solid #232838; border-radius:10px;
         padding:14px 18px; min-width:140px; }
  .kpi b { display:block; font-size:22px; color:var(--acc); }
  .kpi span { color:var(--mut); font-size:12px; }
  .controls { margin:0 0 16px; color:var(--mut); font-size:13px; }
  .controls label { cursor:pointer; user-select:none; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
          gap:16px; margin-bottom:36px; }
  .card { margin:0; background:var(--card); border:1px solid #232838;
          border-radius:12px; overflow:hidden; }
  .imgwrap { position:relative; aspect-ratio:16/9; background:#000; }
  .imgwrap img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
  .imgwrap .orig { opacity:0; transition:opacity .15s; }
  body.show-orig .imgwrap .orig { opacity:1; }
  figcaption { padding:10px 12px; display:flex; flex-wrap:wrap; gap:6px; align-items:baseline; }
  figcaption .rank { background:var(--acc); color:#000; font-weight:700;
                     border-radius:6px; padding:1px 7px; font-size:12px; }
  figcaption .pct { color:var(--acc); font-weight:600; }
  figcaption small { flex-basis:100%; color:var(--mut); }
  table { border-collapse:collapse; width:100%; max-width:640px; font-size:13px; }
  th,td { text-align:right; padding:6px 12px; border-bottom:1px solid #232838; }
  th { color:var(--mut); font-weight:600; }
  tr.toprow td { color:var(--acc); }
  h2 { font-size:16px; color:var(--mut); margin:0 0 12px; font-weight:600; }
"""


def build_html(data: dict) -> str:
    """Costruisce l'HTML dalla struttura normalizzata (vedi loader sopra)."""
    ranked = data["cells_ranked"]
    # I top sono le righe marcate is_top (hanno i PNG); ordinati per rank.
    tops = sorted([c for c in ranked if c["is_top"]], key=lambda c: c["rank"])

    # ── KPI: massa totale visiva, quota della cella top, fascia temporale top ──
    top_pct = tops[0]["pct"] if tops and tops[0]["pct"] is not None else 0.0
    top_t = tops[0]["t_sec"] if tops else 0.0
    top_tsecs = sorted(c["t_sec"] for c in tops) or [0.0]
    fascia = f"{top_tsecs[0]:.0f}–{top_tsecs[-1]:.0f}s"

    # ── Card dei top frame (overlay sopra, originale sotto, toggle via checkbox) ──
    cards = []
    for c in tops:
        over = b64(c["overlay"]) if c["overlay"] and c["overlay"].is_file() else ""
        orig = b64(c["image"]) if c["image"] and c["image"].is_file() else over
        if not over:                          # senza overlay, mostra l'originale
            over = orig
        rep = c["rep_frame"] if c["rep_frame"] is not None else "?"
        pct_html = f'<span class="pct">{c["pct"]:.1f}% massa</span>' if c["pct"] is not None else ""
        score_html = f" · score {c['score']:.4f}" if c["score"] is not None else ""
        cards.append(f"""
        <figure class="card">
          <div class="imgwrap">
            <img class="over" src="{over}" alt="overlay rank {c['rank']}">
            <img class="orig" src="{orig}" alt="frame rank {c['rank']}">
          </div>
          <figcaption>
            <span class="rank">#{c['rank']}</span>
            <b>frame {rep}</b> · {c['t_sec']:.1f}s
            {pct_html}
            <small>cella {c['cell']}{score_html}</small>
          </figcaption>
        </figure>""")

    # ── Tabella completa: tutte le celle in ordine di rank ──
    trows = []
    for c in sorted(ranked, key=lambda c: c["rank"]):
        if c["score"] is None:                # riga senza dati numerici: salta
            continue
        top = ' class="toprow"' if c["is_top"] else ""
        lbl = "[" + ",".join(map(str, c["frames"])) + "]"
        trows.append(
            f"<tr{top}><td>{c['rank']}</td><td>{c['cell']}</td><td>{c['score']:.5f}</td>"
            f"<td>{c['pct']:.1f}%</td><td>{lbl}</td><td>{c['t_sec']:.2f}</td></tr>"
        )
    table = (
        f"<h2>Classifica completa delle celle temporali (metrica = {data['metric']})</h2>"
        "<table><thead><tr><th>rank</th><th>cella</th><th>score</th><th>%massa</th>"
        "<th>frame</th><th>t (s)</th></tr></thead><tbody>"
        + "".join(trows) + "</tbody></table>"
    ) if trows else ""

    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entity→Visual attention · {data['entity']}</title>
<style>{CSS}</style></head>
<body>
  <h1>Attenzione entity→visivo · <span class="ent">"{data['entity']}"</span></h1>
  <div class="meta">
    video <code>{data['video_name']}</code> · {data['fps']:.2f} fps ·
    {data['cells']} celle temporali · regime: {data['regime']} ·
    modello <code>Qwen25VLAttention</code> (custom attention capture)
  </div>

  <div class="kpis">
    <div class="kpi"><b>{100 * data['mass']:.0f}%</b><span>massa sui token visivi<br>(resto su testo/sink)</span></div>
    <div class="kpi"><b>{top_pct:.0f}%</b><span>nella cella top<br>({top_t:.1f}s)</span></div>
    <div class="kpi"><b>{fascia}</b><span>fascia temporale<br>dei top {len(tops)} frame</span></div>
  </div>

  <div class="controls">
    <label><input type="checkbox" id="tgl"> mostra frame originali (senza heatmap)</label>
  </div>
  <div class="grid">{''.join(cards)}</div>

  {table}

  <script>
    document.getElementById('tgl').addEventListener('change', e =>
      document.body.classList.toggle('show-orig', e.target.checked));
  </script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default=None,
                    help="path al debug_<entity>.json (modalità consigliata)")
    ap.add_argument("--output", default=None,
                    help="[fallback] file con lo stdout di debug_attention (default: stdin)")
    ap.add_argument("--frames", default=None,
                    help="[fallback] cartella con i rank*.png (richiesto con --output/stdin)")
    ap.add_argument("--out", default=None, help="path dell'html (default: dal nome del JSON, o debug_out/report.html)")
    args = ap.parse_args()

    if args.json:
        # ── Modalità JSON ──
        json_path = Path(args.json)
        data = load_from_json(json_path)
        default_out = json_path.with_suffix(".html")
    else:
        # ── Modalità OUTPUT TESTUALE ── (serve la cartella frame)
        if not args.frames:
            ap.error("senza JSON serve --frames (cartella dei rank*.png)")
        text = Path(args.output).read_text(encoding="utf-8") if args.output else sys.stdin.read()
        data = load_from_output(text, Path(args.frames))
        default_out = Path("debug_out/report.html")
        if not data["cells_ranked"]:
            raise SystemExit("nessun dato: né righe-tabella nel testo né rank*.png nella cartella")

    html = build_html(data)
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_top = sum(c["is_top"] for c in data["cells_ranked"])
    print(f"report scritto in {out}  ({out.stat().st_size / 1e6:.1f} MB) — "
          f"{n_top} frame, {len(data['cells_ranked'])} celle")


if __name__ == "__main__":
    main()
