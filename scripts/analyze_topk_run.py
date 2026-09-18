"""Analisi offline di una run `topk_resample` (o `signals_capture`) su LVBench.

Legge i per-sample da Weave e risponde alle tre domande della run, in questo
ordine:

1. **T1 — l'attenzione trova la finestra annotata?** Ogni cella temporale del
   pass 1 è etichettata vera/falsa rispetto a `time_reference`
   (`utils.pair_sampling.pair_cells_in_window`), poi AUC, hit@k e rango della
   prima cella vera, per ogni rowset loggato e per la massa grezza e
   sink-filtrata. Due controlli, entrambi necessari perché il numero da solo
   non dice niente: il predittore di sola POSIZIONE (profilo medio, uguale per
   tutti i sample) e il vettore d'attenzione di un ALTRO sample.
2. **Le condizioni** (solo run `topk_resample`): accuracy, delta contro la
   baseline appaiata con McNemar esatto, `r_fix`/`r_break`, e quanto spesso le
   celle ricampionate cadono davvero nella finestra.
3. **Il gate** — quale quantile d'entropia riaprire, misurato ONESTAMENTE:
   K-fold raggruppato per VIDEO (le domande dello stesso video sono
   correlate), quantile scelto sui fold di calibrazione, applicato al fold
   tenuto fuori. Il guadagno riportato è la somma out-of-fold, confrontato coi
   due estremi "riapri sempre" e "non riaprire mai" (= baseline). Se il gate
   non batte "sempre", il gate non serve: la scelta della soglia va riportata
   con la dispersione del quantile fra i fold, non con una media che nasconde
   che i fold non sono d'accordo.

Uso:
    uv run python scripts/analyze_topk_run.py <nome-run>
    uv run python scripts/analyze_topk_run.py <nome-run> --cache /tmp/run.json
    uv run python scripts/analyze_topk_run.py --cache /tmp/run.json   # offline

`--cache` salva (o rilegge) i per-sample già uniti: il fetch da Weave è la
parte lenta, e rianalizzare gli stessi dati non deve ricostarlo.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".claude/skills/wandb"))

from utils.pair_sampling import pair_cells_in_window  # noqa: E402

HIT_KS = (1, 5, 10, 25)
# 0.00 … 0.95 più 1.0 = "non riaprire MAI". Senza l'ultimo valore la ricerca
# out-of-fold non può esprimere la politica di non intervenire, e quando è
# quella giusta (su LVBench AUROC(H) è 0.547: probabile) il numero OOF esce
# peggio della baseline per costruzione, non per misura.
QUANTILE_GRID = [q / 20 for q in range(20)] + [1.0]
# Finestra che copre più di questa frazione delle celle: il bersaglio è quasi
# tutto il video, "trovarlo" non vuol dire niente. Restano nei conti di
# accuracy e del gate, escono solo da T1 e dagli hit.
MAX_TRUE_CELLS_FRAC = 25 / 256


# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────
# Le colonne chieste a Weave, e SOLO quelle. La proiezione non è un'ottimizzazione:
# `sink_stats.per_token` (`values [L, n_vis, C]`) pesa ~23 MB di JSON a sample —
# ~0.7 GB una volta deserializzato — e i sample dumpati ce l'hanno tutti.
# Scaricando l'output intero, 100 sample facevano arrivare il processo a 15 GB di
# RSS: OOM del processo e, su WSL, morte dell'intera VM. Qui `per_token` non
# serve (le prove sui sink usano solo le riduzioni), quindi i campi di
# `sink_stats` si chiedono uno per uno. Con la proiezione gli stessi 100 sample
# stanno in ~150 MB.
#
# Un campo chiesto ma non emesso dalla strategy torna `None` (nessun errore):
# aggiungerne uno qui è sempre sicuro, dimenticarlo no — l'analisi lo vedrebbe
# assente e salterebbe la sua sezione in silenzio.
FETCH_COLUMNS = [
    "inputs.example",
    "output.scores",
    # geometria del pass 1 e etichette T1
    "output.output.pair_centers_sec", "output.output.pair_gap_sec",
    # risposta e distribuzione del pass 1: baseline appaiata e confidenza
    # (l'entropia da sola non è l'unica statistica con cui si può gattare)
    "output.output.pred", "output.output.answer_probs",
    # massa per cella (grezza e sink-filtrata), per rowset
    "output.output.rowsets",
    # condizioni del pass 2 e gate d'entropia
    "output.output.conditions", "output.output.preds_by_condition",
    "output.output.answer_entropy",
    # i sink in questo setting
    "output.output.sink_mass_curve_pcts", "output.output.sink_border_share",
    "output.output.border_share_uniform", "output.output.attn_sink_cell_corr",
    "output.output.sink_stats.sink_dims", "output.output.sink_stats.control_dims",
    "output.output.sink_stats.sink_dim_rank", "output.output.sink_stats.ratio_mean_abs",
    "output.output.sink_stats.channels", "output.output.sink_stats.n_layers",
    "output.output.sink_stats.attn_layer_range",
]


def fetch_samples(name: str, project: str | None) -> list[dict]:
    """Per-sample della run: output di `predict` uniti alla riga del dataset.

    L'unione la fa `Evaluation.predict_and_score`, che ha in `inputs.example`
    la riga intera (id, finestra annotata, risposta giusta) e in `output` sia
    il dict della strategy sia gli score. Un solo passaggio, nessun join per
    testo della domanda da indovinare.
    """
    import driver  # .claude/skills/wandb/driver.py
    import weave

    entity = driver.default_entity()
    run, proj = driver.find_run(entity, project or "lvbench", name)
    client = weave.init(f"{entity}/{proj}")
    prefix = f"weave:///{entity}/{proj}/op"
    t_run = dt.datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
    evs = list(client.get_calls(
        filter={"op_names": [f"{prefix}/Evaluation.evaluate:*"]},
        limit=40, sort_by=[{"field": "started_at", "direction": "desc"}],
    ))
    cands = sorted((e for e in evs if e.started_at >= t_run - dt.timedelta(seconds=30)),
                   key=lambda e: abs((e.started_at - t_run).total_seconds()))
    if not cands:
        sys.exit("nessuna Evaluation weave vicina alla run (morta prima dell'eval?)")
    ev = cands[0]
    calls = list(client.get_calls(
        filter={"op_names": [f"{prefix}/Evaluation.predict_and_score:*"], "trace_ids": [ev.trace_id]},
        limit=5000,
        columns=FETCH_COLUMNS,
    ))
    rows = []
    for c in calls:
        ex = dict(c.inputs or {}).get("example") or {}
        out = dict(c.output or {})
        # Weave chiama "output" il dict che `predict` ritorna (non
        # "model_output": quello è il nome nella UI, non nel payload).
        model_out = _present(dict(out.get("output") or out.get("model_output") or {}))
        if not model_out:
            continue
        scores = (out.get("scores") or {}).get("mcq_accuracy") or {}
        rows.append({
            "id": ex.get("id"),
            "answer": ex.get("answer"),
            "question_type": ex.get("question_type"),
            "video_start": ex.get("video_start"),
            "video_end": ex.get("video_end"),
            "correct": bool(scores.get("correct")),
            "out": _plain(model_out),
        })
    print(f"run {name}: {len(rows)} sample (eval weave {ev.id})")
    return rows


def _present(out: dict) -> dict:
    """Toglie le chiavi a `None`, cioè i campi che la strategy NON emette.

    Weave restituisce ogni colonna chiesta anche quando non esiste: senza
    questa potatura `sink_stats` risulterebbe presente ma pieno di `None` su
    una run che non lo logga, e le prove sui sink fallirebbero invece di
    saltare.
    """
    out = {k: v for k, v in out.items() if v is not None}
    ss = out.get("sink_stats")
    if isinstance(ss, dict):
        ss = {k: v for k, v in dict(ss).items() if v is not None}
        if ss:
            out["sink_stats"] = ss
        else:
            out.pop("sink_stats")
    return out


def _plain(x):
    """WeaveList/WeaveDict → list/dict puri, così il cache JSON è scrivibile."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Etichette e metriche
# ─────────────────────────────────────────────────────────────────────────────
def label_cells(row: dict) -> list[bool] | None:
    """Celle dentro la finestra annotata, o `None` se il sample non serve a T1.

    Scarta: finestra assente, nessuna cella dentro (finestra più stretta del
    passo fra i centri) e finestra che copre quasi tutto il video. ⚠️ Alcune
    righe di LVBench hanno `time_reference` INVERTITO (start > end): si
    riordina invece di sollevare.
    """
    w0, w1 = row.get("video_start"), row.get("video_end")
    if w0 is None or w1 is None:
        return None
    if w0 > w1:
        w0, w1 = w1, w0
    out = row["out"]
    centers = out.get("pair_centers_sec")
    if not centers:
        return None
    lab = pair_cells_in_window(centers, out.get("pair_gap_sec", 2.0), w0, w1)
    n_true = sum(lab)
    if n_true == 0 or n_true > MAX_TRUE_CELLS_FRAC * len(lab):
        return None
    return lab


def auc(values: list[float], lab: list[bool]) -> float | None:
    """AUC di Mann-Whitney: P(cella vera ha massa > cella falsa), pari a 0.5."""
    pos = [v for v, l in zip(values, lab) if l]
    neg = [v for v, l in zip(values, lab) if not l]
    if not pos or not neg:
        return None
    sneg = sorted(neg)
    tot = 0.0
    for p in pos:
        lo = bisect.bisect_left(sneg, p)
        hi = bisect.bisect_right(sneg, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def top_cells(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: -values[i])[:k]


def hit_at(values: list[float], lab: list[bool], k: int) -> bool:
    return any(lab[i] for i in top_cells(values, k))


def chance_at(lab: list[bool], k: int) -> float:
    """P(almeno una cella vera fra k celle a caso), senza reinserimento."""
    n, t = len(lab), sum(lab)
    return 1 - math.prod((n - t - i) / (n - i) for i in range(k)) if n - t >= k else 1.0


def binom_p(hits: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(hits, n + 1))


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """(b giusto & a sbagliato, a giusto & b sbagliato, p esatto a due code)."""
    win = sum(1 for x, y in zip(a, b) if y and not x)
    loss = sum(1 for x, y in zip(a, b) if x and not y)
    n = win + loss
    if n == 0:
        return win, loss, 1.0
    k = min(win, loss)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return win, loss, min(1.0, p)


# ─────────────────────────────────────────────────────────────────────────────
# 1. T1
# ─────────────────────────────────────────────────────────────────────────────
def analyze_t1(rows: list[dict]) -> None:
    usable = [(r, lab) for r in rows if (lab := label_cells(r)) is not None]
    print(f"\n{'='*78}\nT1 — l'attenzione trova la finestra? ({len(usable)}/{len(rows)} sample usabili)")
    if not usable:
        print("  nessun sample usabile (finestre assenti o mai coperte dalle celle)")
        return
    n_true = [sum(lab) for _, lab in usable]
    print(f"  celle vere per sample: mediana {st.median(n_true):.0f}, "
          f"chance top-1 {100*st.mean(t/len(l) for (_, l), t in zip(usable, n_true)):.1f}%")

    rowsets = sorted(usable[0][0]["out"].get("rowsets", {}))
    keys = ("cell_mass_raw", "cell_mass_sink_filtered")
    print(f"\n  {'rowset':<12}{'massa':<16}{'AUC':>6}{'rango':>7}" +
          "".join(f"{'hit@'+str(k):>12}" for k in HIT_KS))
    for rs in rowsets:
        for key in keys:
            vals = [(r["out"]["rowsets"][rs].get(key), lab) for r, lab in usable]
            vals = [(v, lab) for v, lab in vals if v]
            if not vals:
                continue
            aucs = [auc(v, lab) for v, lab in vals]
            ranks = [next(pos for pos, i in enumerate(top_cells(v, len(v))) if lab[i]) + 1
                     for v, lab in vals]
            cells = []
            for k in HIT_KS:
                h = sum(hit_at(v, lab, k) for v, lab in vals)
                c = st.mean(chance_at(lab, k) for _, lab in vals)
                cells.append(f"{100*h/len(vals):>6.0f}% ({100*c:>2.0f}%)")
            print(f"  {rs:<12}{key.replace('cell_mass_', ''):<16}"
                  f"{st.mean(aucs):>6.3f}{st.median(ranks):>7.0f}" + "".join(f"{c:>12}" for c in cells))

    # --- controlli ---------------------------------------------------------
    rs, key = rowsets[0] if "all" not in rowsets else "all", "cell_mass_raw"
    vals = [(r["out"]["rowsets"][rs][key], lab) for r, lab in usable]
    n_cells = len(vals[0][0])
    prof = [0.0] * n_cells
    for v, _ in vals:
        tot = sum(v) or 1.0
        for i, x in enumerate(v):
            prof[i] += x / tot
    pos_auc = st.mean(auc(prof, lab) for _, lab in vals)
    pos_hit = sum(lab[max(range(n_cells), key=lambda i: prof[i])] for _, lab in vals)
    print(f"\n  controllo POSIZIONE (profilo medio, uguale per ogni sample, rowset {rs}): "
          f"AUC {pos_auc:.3f}, hit@1 {100*pos_hit/len(vals):.0f}%")

    rng = random.Random(0)
    shuf_auc, shuf_hit = [], []
    for _ in range(200):
        perm = list(range(len(vals)))
        rng.shuffle(perm)
        perm = [p if p != i else (p + 1) % len(vals) for i, p in enumerate(perm)]
        shuf_auc.append(st.mean(auc(vals[p][0], lab) for p, (_, lab) in zip(perm, vals)))
        shuf_hit.append(st.mean(hit_at(vals[p][0], lab, 1) for p, (_, lab) in zip(perm, vals)))
    print(f"  controllo PERMUTAZIONE (vettore di un altro sample, 200 giri): "
          f"AUC {st.mean(shuf_auc):.3f}±{st.pstdev(shuf_auc):.3f}, "
          f"hit@1 {100*st.mean(shuf_hit):.1f}%±{100*st.pstdev(shuf_hit):.1f}")
    real_hit = sum(hit_at(v, lab, 1) for v, lab in vals)
    p = st.mean(chance_at(lab, 1) for _, lab in vals)
    print(f"  hit@1 reale {real_hit}/{len(vals)} contro {p*len(vals):.1f} attesi "
          f"(binomiale p={binom_p(real_hit, len(vals), p):.2g})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Condizioni
# ─────────────────────────────────────────────────────────────────────────────
def condition_names(rows: list[dict]) -> list[str]:
    for r in rows:
        preds = r["out"].get("preds_by_condition")
        if preds:
            return list(preds)
    return []


def analyze_conditions(rows: list[dict], names: list[str]) -> None:
    base = [r["correct"] for r in rows]
    n = len(rows)
    print(f"\n{'='*78}\nCONDIZIONI ({n} sample) — baseline appaiata: "
          f"{100*sum(base)/n:.1f}% ({sum(base)}/{n})")
    print(f"  {'nome':<10}{'acc':>7}{'Δ':>8}{'win':>6}{'loss':>6}{'p':>9}"
          f"{'r_fix':>8}{'r_break':>9}{'hit@k celle':>13}")
    for name in names:
        ok, idx = [], []
        for i, r in enumerate(rows):
            p = (r["out"].get("preds_by_condition") or {}).get(name)
            if p is None:      # niente lettera: il sample non conta per questa
                continue       # condizione (né come giusto né come sbagliato)
            ok.append(p == r["answer"])
            idx.append(i)
        if not ok:
            continue
        b = [base[i] for i in idx]
        win, loss, p = mcnemar(b, ok)
        fixed = [o for o, bb in zip(ok, b) if not bb]
        broke = [o for o, bb in zip(ok, b) if bb]
        r_fix = st.mean(fixed) if fixed else float("nan")
        r_break = 1 - st.mean(broke) if broke else float("nan")
        # Le celle ricampionate cadono nella finestra vera?
        hits = tot = 0
        for i in idx:
            lab = label_cells(rows[i])
            cells = ((rows[i]["out"].get("conditions") or {}).get(name) or {}).get("cells")
            if lab is None or not cells:
                continue
            tot += 1
            hits += any(lab[c] for c in cells)
        hit_s = f"{100*hits/tot:.0f}% ({tot})" if tot else "—"
        print(f"  {name:<10}{100*st.mean(ok):>6.1f}%{100*(st.mean(ok)-st.mean(b)):>+7.1f}"
              f"{win:>6}{loss:>6}{p:>9.3g}{r_fix:>8.2f}{r_break:>9.2f}{hit_s:>13}")
    print("  Δ = punti percentuali contro la baseline SUGLI STESSI sample; "
          "win/loss = sample recuperati/rotti; p = McNemar esatto.")
    print("  r_fix = frazione degli SBAGLIATI che la condizione recupera, "
          "r_break = frazione dei GIUSTI che rompe.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gate out-of-fold
# ─────────────────────────────────────────────────────────────────────────────
def folds_by_video(rows: list[dict], k: int, seed: int = 0) -> list[list[int]]:
    """K fold raggruppati per VIDEO: le ~15 domande di uno stesso video stanno
    nello stesso fold. Senza questo vincolo la soglia si calibrerebbe su
    domande dello stesso video su cui poi la si valuta, e il risultato
    sembrerebbe migliore di quello che è. I video vanno nei fold in ordine di
    dimensione decrescente (greedy sul fold più piccolo), così i fold hanno
    numerosità simili."""
    by_video: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        vid = str(r.get("id") or i).split("/")[0]
        by_video.setdefault(vid, []).append(i)
    groups = sorted(by_video.values(), key=len, reverse=True)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    out: list[list[int]] = [[] for _ in range(k)]
    for g in groups:
        out.sort(key=len)
        out[0].extend(g)
    return out


def _thr(q: float, hs: list[float]) -> float:
    """Quantile → soglia d'entropia. `q >= 1` dà `+inf`, cioè "non riaprire
    mai": è una politica ammissibile e va nello spazio di ricerca."""
    if q >= 1.0:
        return float("inf")
    return hs[min(int(q * len(hs)), len(hs) - 1)]


def gated_correct(rows: list[dict], idx: list[int], name: str, thr: float) -> list[bool]:
    """Correttezza della politica "riapri se H >= thr" sui sample `idx`."""
    out = []
    for i in idx:
        r = rows[i]
        h = r["out"].get("answer_entropy")
        p = (r["out"].get("preds_by_condition") or {}).get(name)
        reopen = h is not None and p is not None and h >= thr
        out.append((p == r["answer"]) if reopen else r["correct"])
    return out


def analyze_gate(rows: list[dict], names: list[str], n_folds: int) -> None:
    hs = sorted(r["out"]["answer_entropy"] for r in rows if r["out"].get("answer_entropy") is not None)
    if not hs:
        return
    folds = folds_by_video(rows, n_folds)
    print(f"\n{'='*78}\nGATE — quantile scelto out-of-fold ({n_folds} fold raggruppati per video, "
          f"dimensioni {[len(f) for f in folds]})")
    print(f"  {'nome':<10}{'mai':>7}{'sempre':>9}{'gate OOF':>10}{'quantili scelti':>28}")
    base_acc = st.mean(r["correct"] for r in rows)
    for name in names:
        always = st.mean(gated_correct(rows, list(range(len(rows))), name, -1.0))
        oof, chosen = [], []
        for f in folds:
            calib = [i for i in range(len(rows)) if i not in set(f)]
            best_q, best_gain = 0.0, -1e9
            for q in QUANTILE_GRID:
                gain = st.mean(gated_correct(rows, calib, name, _thr(q, hs)))
                if gain > best_gain:
                    best_q, best_gain = q, gain
            thr = _thr(best_q, hs)
            oof += gated_correct(rows, f, name, thr)
            chosen.append(best_q)
        q_s = f"{st.median(chosen):.2f} [{min(chosen):.2f}–{max(chosen):.2f}]"
        print(f"  {name:<10}{100*base_acc:>6.1f}%{100*always:>8.1f}%{100*st.mean(oof):>9.1f}%{q_s:>28}")
    print("  'mai' = baseline, 'sempre' = riapri tutti i sample, 'gate OOF' = riapri sopra il")
    print("  quantile scelto sui fold di calibrazione. Se il gate non batte 'sempre' non serve;")
    print("  se i quantili scelti dai fold sono sparsi, la soglia non è identificata.")



# ─────────────────────────────────────────────────────────────────────────────
# 4. I sink, in questo setting
# ─────────────────────────────────────────────────────────────────────────────
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def analyze_sinks(rows: list[dict], label: str = "pass 1 (512 frame)", outs: list[dict] | None = None) -> None:
    """Le quattro prove, aggregate su tutti i sample che portano le statistiche.

    Il punto non è "quanto vale il numero" ma quale delle due storie regge:
    canali outlier che esistono davvero (prove 1-2) NON implicano token che
    catturano attenzione (prova 3). Sulla probe a 512 frame le prime due
    passavano e la terza no.
    """
    outs = outs if outs is not None else [r["out"] for r in rows]
    withs = [o for o in outs if o.get("sink_stats")]
    print(f"\n{'='*78}\nSINK — {label}: {len(withs)}/{len(outs)} sample con statistiche dei canali")

    pcts = next((o.get("sink_mass_curve_pcts") for o in outs if o.get("sink_mass_curve_pcts")), None)
    rowsets = sorted(next((o["rowsets"] for o in outs if o.get("rowsets")), {}))
    if pcts:
        print("\n  PROVA 3 — massa d'attenzione sui top-p% token per sink score")
        print("  (se la curva sta su ~p%, quei token NON sono pozzi: assorbono quanto capita loro)")
        print("    " + "rowset".ljust(12) + "".join(f"p={p}%".rjust(10) for p in pcts))
        for rs in rowsets:
            curves = [o["rowsets"][rs].get("sink_mass_curve") for o in outs if o.get("rowsets")]
            curves = [c for c in curves if c]
            if not curves:
                continue
            means = [_mean([c[i] for c in curves]) for i in range(len(pcts))]
            print("    " + rs.ljust(12) + "".join(f"{100*m:9.1f}%" if m is not None else "—".rjust(10) for m in means))

    border_a = _mean([o["rowsets"][rs].get("attn_border_share")
                      for o in outs if o.get("rowsets") for rs in [rowsets[0]] if o["rowsets"].get(rs)])
    border_s = _mean([o.get("sink_border_share") for o in outs])
    border_u = _mean([o.get("border_share_uniform") for o in outs])
    if border_s is not None:
        print(f"\n  DOVE, dentro il frame — quota di massa sull'anello di bordo "
              f"(uniforme = {100*(border_u or 0):.0f}%)")
        print(f"    attenzione ({rowsets[0]}): {100*(border_a or 0):.1f}%   sink: {100*border_s:.1f}%")
        print("    Sopra l'uniforme = fenomeno di bordo/sfondo; attorno = guarda dentro l'immagine.")
    corr = [o.get("attn_sink_cell_corr") for o in outs if o.get("attn_sink_cell_corr") is not None]
    if corr:
        print(f"\n  Correlazione per cella fra massa d'attenzione e massa di sink: "
              f"mediana {st.median(corr):+.3f} (|r| alto ⇒ il ranking temporale è la mappa dei sink)")

    if not withs:
        return
    ss0 = withs[0]["sink_stats"]
    dims, ctrl = list(ss0["sink_dims"]), list(ss0["control_dims"])
    n_layers = int(ss0["n_layers"])
    print(f"\n  PROVA 1 — rango dei sink dims fra i canali (0 = il più estremo del layer), "
          f"mediana sui sample")
    for j, d in enumerate(dims):
        per_layer = [st.median([o["sink_stats"]["sink_dim_rank"][l][j] for o in withs])
                     for l in range(n_layers)]
        print(f"    dim {d:<5} min {min(per_layer):>5.0f}  mediana {st.median(per_layer):>6.0f}  "
              f"max {max(per_layer):>6.0f}   (layer 0→{n_layers-1}: "
              f"{', '.join(f'{x:.0f}' for x in per_layer[:4])}, …)")
    lo, hi = ss0["attn_layer_range"]
    print(f"\n  PROVA 2 — |h[d]| / media(|h|) per token, layer {lo}-{hi}, sink dims contro controlli")
    chans = list(ss0["channels"])
    for group in ("sink", "nonsink"):
        vals = []
        for c in range(len(chans)):
            vals.append(_mean([_mean([o["sink_stats"]["ratio_mean_abs"][group]["mean"][l][c]
                                      for l in range(lo, hi)]) for o in withs]))
        print(f"    token {group:<8}" + "  ".join(
            f"d{chans[c]}{'*' if chans[c] in dims else ' '}={vals[c]:7.2f}" for c in range(len(chans))))
    print("    (* = sink dim tabulato; gli altri sono canali di CONTROLLO a caso)")


def sink_conditions(rows: list[dict], names: list[str]) -> None:
    """Le stesse prove sul pass 2, per le condizioni che portano le statistiche:
    128 frame stipati in decine di secondi sono un altro regime di
    campionamento, e se il fenomeno è del modello deve comparire uguale."""
    for name in names:
        outs = [(r["out"].get("conditions") or {}).get(name) for r in rows]
        outs = [o for o in outs if o]
        if not any(o.get("sink_stats") for o in outs):
            continue
        for o in outs:      # le curve stanno sulla condizione, non sotto `rowsets`
            o.setdefault("rowsets", {"pass2": {"sink_mass_curve": o.get("sink_mass_curve"),
                                               "attn_border_share": o.get("attn_border_share")}})
            o.setdefault("sink_mass_curve_pcts", rows[0]["out"].get("sink_mass_curve_pcts"))
        analyze_sinks(rows, label=f"pass 2, condizione {name} (128 frame densi)", outs=outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="nome della run wandb (omesso: serve --cache già popolata)")
    ap.add_argument("--project", default=None, help="progetto wandb (default: lvbench, poi tutti)")
    ap.add_argument("--cache", type=Path, help="file JSON dove salvare/rileggere i per-sample")
    ap.add_argument("--folds", type=int, default=5, help="fold del gate (default 5)")
    args = ap.parse_args()

    if args.cache and args.cache.exists() and not args.run:
        rows = json.loads(args.cache.read_text())
        print(f"cache: {len(rows)} sample da {args.cache}")
    elif args.run:
        rows = fetch_samples(args.run, args.project)
        if args.cache:
            args.cache.write_text(json.dumps(rows))
            print(f"cache scritta in {args.cache}")
    else:
        ap.error("serve un nome di run, oppure una --cache già popolata")

    if not rows:
        sys.exit("nessun sample")
    analyze_t1(rows)
    analyze_sinks(rows)
    names = condition_names(rows)
    if names:
        analyze_conditions(rows, names)
        analyze_gate(rows, names, args.folds)
        sink_conditions(rows, names)
    else:
        print("\n(run senza condizioni: niente analisi di arm né di gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
