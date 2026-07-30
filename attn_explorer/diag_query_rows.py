"""Quanto del picco d'attenzione dipende da QUALI righe-query lo producono.

Diagnostica offline su uno STORE di `attn_explorer.capture`: nessun forward,
nessuna GPU, nessuna eval — rilegge i dump e ricalcola il ranking delle celle
temporali sotto insiemi di righe diversi, a parità di tutto il resto.

IL PROBLEMA. `QwenAttentionCapture._capture_spans` definisce le righe-query
come tutto ciò che segue l'ultimo `<|vision_end|>`, e
`ranked_cells_from_attention` (usata da `attention_highlight` e
`entropy_attention_resample`) fa `attn.mean(dim=0)`: media su TUTTE quelle
righe. Su un prompt MCQ di Video-MME sono domanda + le quattro opzioni +
"Answer with the letter of the correct option only." + il generation prompt —
misurato col tokenizer su un prompt tipico, 86 token di cui 21 di domanda: il
76% delle righe che formano il "picco d'attenzione della domanda" non è la
domanda.

I QUATTRO INSIEMI confrontati (`--rowsets`):

    all        tutte le righe — il comportamento ATTUALE delle strategy
    qopts      domanda + opzioni, tolto il boilerplate e il generation prompt
    question   solo la domanda (`utils.attn_core.question_rows`)
    entity     i token dell'entità, da `--entities` (JSON stem→substring):
               scelti A MANO, quindi ORACOLO del riconoscimento automatico,
               cioè il tetto massimo che un estrattore potrà raggiungere

LE METRICHE misurano il bias POSIZIONALE del picco, non la sua correttezza:
Video-MME non ha ground truth temporale, quindi non si può dire se il picco
è nel posto giusto — si può dire se è nel posto in cui cadrebbe comunque.
Ogni metrica ha il suo livello di caso, stampato accanto:

    last%      picco sull'ULTIMA cella          caso = 100/t
    extreme%   picco sulla prima o sull'ultima  caso = 200/t
    top1_pct   massa della cella di picco       caso = 100/t
    gini       disuguaglianza della massa sulle celle (0 = uniforme)

I valori di riferimento per `all` vengono dal probe da 40 sample (run wandb
2dodrndg, Qwen2.5-VL-3B, nframes=24 → t=12): last 28%, extreme 40%,
top1_pct mediano 15.7%, contro un caso di 8.3%/16.7%/8.3%.

COME SI LEGGE. Se `question` abbatte il bias, la correzione è quasi gratis
(`question_rows` esiste già e le strategy non la chiamano). Se lo abbatte
solo `entity`, allora l'estrattore automatico vale il lavoro. Se non lo muove
nessuno, l'asse righe è esaurito: il picco è posizionale qualunque testo lo
interroghi, e raffinarne la risoluzione (frame-doubling) localizzerebbe
meglio un artefatto.

⚠️ Il filtro sink è applicato come nelle strategy (`sink_view` view=nonsink,
percentile 25, border 0 — i default di `attention_highlight`), altrimenti i
numeri non sarebbero confrontabili col probe di riferimento.

Uso:
    uv run python -m attn_explorer.diag_query_rows --store data/videomme_rows_24
    uv run python -m attn_explorer.diag_query_rows --store <s> --entities ent.json
    uv run python -m attn_explorer.diag_query_rows --store <s> --per-video
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median

import torch

from utils.attn_core import (
    ROW_SELECTORS,
    QueryToken,
    load_capture,
    rank_cells,
    sink_view,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diag_query_rows")

# `entity` non è in `ROW_SELECTORS` perché non è un knob della strategy: le
# entità arrivano da un JSON scritto a mano (oracolo), non da una regola.
ROWSETS = (*ROW_SELECTORS, "entity")


def rows_entity(query_tokens: tuple[QueryToken, ...], entity: str | None) -> list[int] | None:
    """Righe dei token che contengono `entity` (substring, case-insensitive).

    `None` se l'entity non è data o non compare: il chiamante salta il video
    per questo rowset invece di ripiegare su tutte le righe, che
    confonderebbe silenziosamente `entity` con `all`.
    """
    if not entity:
        return None
    needle = entity.lower()
    rows = [qt.row for qt in query_tokens if needle in qt.text.lower()]
    return rows or None


def gini(x: torch.Tensor) -> float:
    """Coefficiente di Gini della massa per cella: 0 = uniforme, 1 = tutta su una."""
    v = torch.sort(x.flatten().float())[0]
    n = v.numel()
    tot = v.sum()
    if n == 0 or tot <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float((2 * (idx * v).sum() / (n * tot)) - (n + 1) / n)


def peak_stats(cap, rows: list[int], *, percentile: float, border: int) -> dict:
    """`last`/`extreme`/`top1_pct`/`gini` del picco per una selezione di righe.

    Replica la catena delle strategy: aggregazione sulle righe → filtro sink
    non-sink → `rank_cells` per massa. `rows` vuoto = tutte le righe, che è
    la semantica di `AttentionCapture.aggregate`.
    """
    heat = cap.aggregate(rows)                       # [t, grid_h, grid_w]
    heat_nonsink, _ = sink_view(
        heat, cap.sink_map, view="nonsink", percentile=percentile, border=border,
    )
    ranked = rank_cells(
        heat_nonsink, cap.grid, cap.frame_indices, cap.fps, metric="sum", topk=1,
    )
    t = cap.grid.t
    peak = ranked[0].cell
    return {
        "t": t,
        "peak": peak,
        "last": peak == t - 1,
        "extreme": peak in (0, t - 1),
        "top1_pct": ranked[0].pct,
        "gini": gini(heat_nonsink.flatten(1).sum(dim=1)),
        "n_rows": len(rows) if rows else cap.attn.shape[0],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--store", required=True, help="STORE prodotto da attn_explorer.capture")
    ap.add_argument("--entities", default=None,
                    help="JSON {stem: substring} con l'entità scelta a mano per video "
                         "(oracolo del riconoscimento automatico); senza, il rowset "
                         "`entity` è saltato")
    ap.add_argument("--sink-percentile", type=float, default=25.0,
                    help="filtro sink, default 25.0 = quello di attention_highlight")
    ap.add_argument("--sink-border", type=int, default=0, help="default 0 = attention_highlight")
    ap.add_argument("--per-video", action="store_true", help="stampa anche la riga per video")
    ap.add_argument("--json-out", default=None, help="scrive l'aggregato in JSON su questo path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    store = Path(args.store)
    index_path = store / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"indice non trovato: {index_path} (hai lanciato attn_explorer.capture?)")
    stems = [e["stem"] for e in json.loads(index_path.read_text())["videos"]]

    entities: dict[str, str] = {}
    if args.entities:
        entities = json.loads(Path(args.entities).read_text())

    per_set: dict[str, list[dict]] = {rs: [] for rs in ROWSETS}
    n_cells: set[int] = set()

    for stem in stems:
        cap_path = store / stem / "capture.pt"
        if not cap_path.is_file():
            logger.warning("[%s] capture.pt mancante, salto", stem)
            continue
        cap = load_capture(cap_path)
        n_cells.add(cap.grid.t)

        # Gli stessi selettori che usa la strategy (`query_rows`), non una
        # reimplementazione: la diagnostica deve misurare esattamente la
        # selezione che poi gira nell'eval.
        selections = {name: sel(cap.query_tokens) for name, sel in ROW_SELECTORS.items()}
        selections["entity"] = rows_entity(cap.query_tokens, entities.get(stem))
        line = [f"{stem:22s}"]
        for rs in ROWSETS:
            rows = selections[rs]
            if rows is None:                      # solo `entity` senza match
                line.append(f"{rs}=  --      ")
                continue
            st = peak_stats(cap, rows, percentile=args.sink_percentile, border=args.sink_border)
            st["stem"] = stem
            per_set[rs].append(st)
            line.append(f"{rs}=c{st['peak']:<2d}({st['n_rows']:>3d}r) ")
        if args.per_video:
            print(" ".join(line))

    if not any(per_set.values()):
        raise SystemExit("nessun capture valutato")

    if len(n_cells) > 1:
        logger.warning(
            "lo store mescola griglie con t diverso (%s): i livelli di caso non sono "
            "gli stessi per tutti i video, l'aggregato è approssimato", sorted(n_cells),
        )
    t = max(n_cells)
    chance = {"last": 100 / t, "extreme": 200 / t, "top1_pct": 100 / t}

    print(f"\n--- Bias posizionale del picco per insieme di righe-query (t={t} celle) ---")
    print(f"{'rowset':<10} {'n':>4} {'righe':>7} {'last%':>8} {'extreme%':>10} "
          f"{'top1_pct':>10} {'gini':>7}")
    print(f"{'caso':<10} {'':>4} {'':>7} {chance['last']:>7.1f}% {chance['extreme']:>9.1f}% "
          f"{chance['top1_pct']:>9.1f}% {'':>7}")
    out = {"t": t, "chance": chance, "rowsets": {}}
    for rs in ROWSETS:
        rows = per_set[rs]
        if not rows:
            print(f"{rs:<10} {'—':>4}  (nessun video: entity non fornita o assente dal prompt)")
            continue
        n = len(rows)
        rec = {
            "n": n,
            "mean_rows": sum(r["n_rows"] for r in rows) / n,
            "last_pct": 100 * sum(r["last"] for r in rows) / n,
            "extreme_pct": 100 * sum(r["extreme"] for r in rows) / n,
            "top1_pct_median": median(r["top1_pct"] for r in rows),
            "gini_median": median(r["gini"] for r in rows),
        }
        out["rowsets"][rs] = rec
        print(f"{rs:<10} {n:>4} {rec['mean_rows']:>7.0f} {rec['last_pct']:>7.1f}% "
              f"{rec['extreme_pct']:>9.1f}% {rec['top1_pct_median']:>9.1f}% "
              f"{rec['gini_median']:>7.3f}")

    if n_ := len(per_set["all"]):
        print(f"\nn={n_} video. Con poche decine di video queste quote hanno un errore "
              f"standard di ±{100 * (0.25 / n_) ** 0.5:.0f} pp: servono per vedere un "
              "effetto grosso, non per stimarne la taglia.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"aggregato scritto in {args.json_out}")


if __name__ == "__main__":
    main()
