"""Quale quantile di entropia riaprire nelle strategy a gate — strumento generico.

Le strategy "a gate" (es. `entropy_shift`) interrogano il modello una prima
volta (pass 1) e intervengono con un pass 2 (ricampionare i frame) solo sui
sample incerti, misurati con l'entropia H (bit) della softmax ristretta alle
lettere MCQ. Una soglia in bit NON si trasferisce fra modelli/dataset (0.7 bit
riapre il 73% dei sample con Qwen2.5-VL-3B e il 54% con Qwen3-VL-2B su
Video-MME): la soglia si fissa quindi per QUANTILE di H sul dataset, cioè come
frazione di sample da riaprire. Questo script dice QUALE quantile.

Regola (derivazione completa in docs/scelta_quantile_entropia.md): un tratto
di entropia con W risposte sbagliate e R giuste al pass 1 dà netto atteso

    netto = r_fix·W − r_break·R

dove r_fix = frazione degli sbagliati riaperti che il pass 2 recupera e
r_break = frazione dei giusti riaperti che rompe. È positivo se e solo se
l'accuracy al pass 1 del tratto è sotto

    p* = r_fix / (r_fix + r_break).

Il quantile scelto è quello che massimizza il netto previsto cumulato
(= riaprire i tratti sotto p*, tollerando un tratto rumoroso sopra p* se i
successivi lo compensano). Assunzione: r_fix e r_break costanti in H — lo
script la mette alla prova quando c'è un pass 2 nei dati.

Input: CSV per sample con colonne
    obbligatorie  id, H, correct_pass1
    opzionali     reopened (pass 2 eseguito), correct_pass2 (definito dove reopened)
    di gruppo     duration, task_type, question_type
Booleani accettati come True/False, 1/0, yes/no (vuoto = mancante).

Uscite (stdout, e un foglio per tabella con --xlsx):
    quantili       soglia in bit, frazione riaperta, accuracy riaperti/accettati,
                   W/R riaperti, accuracy del tratto marginale
    auroc          H come predittore d'errore, globale e per gruppo
    curva_netto    (se pass 2) guadagno netto osservato per quantile, solo dove
                   il pass 2 copre tutti i sample sopra soglia, con IC bootstrap
    tassi_tratti   (se pass 2) r_fix, r_break, p* per tratto di H con IC
    test_costanza  (se pass 2) test di permutazione "tassi costanti in H"
    raccomandazione  p* con IC, quantile di pareggio, variante col costo,
                   sensibilità agli estremi dell'IC, stabilità bootstrap

Uso:
    uv run python scripts/choose_entropy_quantile.py samples.csv
    uv run python scripts/choose_entropy_quantile.py samples.csv --r-fix 22/68 --r-break 6/32
    uv run --with openpyxl python scripts/choose_entropy_quantile.py samples.csv --xlsx out.xlsx

`--r-fix/--r-break` accettano un numero (0.32) o una frazione k/n (22/68): con
k/n l'IC di p* viene da un bootstrap binomiale, con un numero secco non c'è IC.
Se passati, sostituiscono i tassi stimati dai dati (lo dichiara l'output).
Il pass 2 costa `--costo-pass2` forward-equivalenti per sample riaperto; la
variante col costo richiede almeno `--lambda-costo` sample netti per forward.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = ("duration", "task_type", "question_type")
_TRUE = {"true", "1", "1.0", "yes", "y", "t", "si", "sì"}
_FALSE = {"false", "0", "0.0", "no", "n", "f"}
_NULL = {"", "nan", "none", "null", "na", "<na>"}


# --------------------------------------------------------------------------- #
# Caricamento
# --------------------------------------------------------------------------- #
def to_bool01(s: pd.Series, name: str) -> np.ndarray:
    """Colonna booleana "sporca" (CSV da pandas, Fogli, wandb) → 1.0 / 0.0 / NaN.

    Un valore non riconosciuto è un errore, non un NaN silenzioso: una colonna
    di correttezza letta male sposterebbe tutte le accuracy senza avvisi.
    """
    if s.dtype == bool:
        return s.to_numpy(float)
    txt = s.astype("string").str.strip().str.lower()
    is_t = txt.isin(_TRUE).fillna(False).to_numpy(bool)
    is_f = txt.isin(_FALSE).fillna(False).to_numpy(bool)
    is_null = txt.isna().to_numpy(bool) | txt.isin(_NULL).fillna(False).to_numpy(bool)
    bad = ~(is_t | is_f | is_null)
    if bad.any():
        esempi = sorted(set(txt[bad].astype(str)))[:5]
        raise SystemExit(f"colonna {name!r}: valori non booleani, es. {esempi}")
    out = np.full(len(s), np.nan)
    out[is_t], out[is_f] = 1.0, 0.0
    return out


@dataclass
class Dati:
    df: pd.DataFrame      # righe valide, indice posizionale
    H: np.ndarray         # entropia del pass 1 (bit)
    c1: np.ndarray        # correttezza pass 1, 0/1
    p2: np.ndarray        # bool: pass 2 disponibile (riaperto e correct_pass2 definito)
    c2: np.ndarray        # correttezza pass 2, 0/1, NaN dove non disponibile

    @property
    def n(self) -> int:
        return len(self.H)

    @property
    def has_pass2(self) -> bool:
        return bool(self.p2.any())


def load(path: Path) -> Dati:
    df = pd.read_csv(path)
    manca = [c for c in ("id", "H", "correct_pass1") if c not in df.columns]
    if manca:
        raise SystemExit(f"{path}: colonne obbligatorie mancanti {manca}")
    dup = df["id"].duplicated()
    if dup.any():
        print(f"⚠️  {int(dup.sum())} id duplicati: tengo la prima occorrenza", file=sys.stderr)
        df = df[~dup]
    df = df.assign(H=pd.to_numeric(df["H"], errors="coerce"),
                   _c1=to_bool01(df["correct_pass1"], "correct_pass1"))
    bad = df["H"].isna() | df["_c1"].isna()
    if bad.any():
        print(f"⚠️  {int(bad.sum())} righe senza H o correct_pass1: escluse", file=sys.stderr)
        df = df[~bad]
    df = df.reset_index(drop=True)
    if (df["H"] < -1e-9).any():
        raise SystemExit("H negativa: non è un'entropia")

    n = len(df)
    c2 = np.full(n, np.nan)
    p2 = np.zeros(n, bool)
    if "correct_pass2" in df.columns:
        c2 = to_bool01(df["correct_pass2"], "correct_pass2")
        # Senza `reopened`, "pass 2 eseguito" = correct_pass2 definito.
        reo = (to_bool01(df["reopened"], "reopened") == 1.0) if "reopened" in df.columns else ~np.isnan(c2)
        buchi = reo & np.isnan(c2)
        if buchi.any():
            print(f"⚠️  {int(buchi.sum())} sample riaperti senza correct_pass2: trattati come "
                  "NON coperti dal pass 2", file=sys.stderr)
        p2 = reo & ~np.isnan(c2)
        c2 = np.where(p2, c2, np.nan)
    elif "reopened" in df.columns:
        raise SystemExit("colonna 'reopened' presente ma 'correct_pass2' no: il pass 2 non è valutabile")
    return Dati(df=df, H=df["H"].to_numpy(float), c1=df["_c1"].to_numpy(float), p2=p2, c2=c2)


# --------------------------------------------------------------------------- #
# Quantili e regola del pareggio
# --------------------------------------------------------------------------- #
def quantile_grid(step: float) -> np.ndarray:
    k = int(round(1 / step))
    qs = np.round(np.arange(1, k + 1) * step, 10)
    qs = qs[qs < 1 - 1e-9]
    return np.append(qs, 1.0)


def reopen_masks(H: np.ndarray, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soglie e maschere annidate: al quantile q si riaprono i sample con H > soglia.

    `H > soglia` è la semantica del gate negli arm (answer_entropy > threshold).
    La soglia è il quantile (1−q) di H, quindi la frazione riaperta è ≈ q; con
    pareggi in H (tipicamente H≈0 per risposte saturate) la frazione EFFETTIVA
    può essere più bassa ed è riportata a parte. q=1 riapre tutto (soglia −inf).
    """
    ts = np.quantile(H, 1 - qs)
    ts[qs >= 1 - 1e-9] = -np.inf
    return ts, H[None, :] > ts[:, None]


def cum_counts(H: np.ndarray, c1: np.ndarray, qs: np.ndarray):
    ts, M = reopen_masks(H, qs)
    n_cum = M.sum(1)
    R_cum = (M & (c1 == 1)).sum(1)
    return ts, M, n_cum, R_cum


def best_q(qs: np.ndarray, n_cum: np.ndarray, R_cum: np.ndarray, pstar: float) -> tuple[float, int]:
    """Quantile che massimizza Σ_tratti (p*·n − R) ∝ netto previsto.

    r_fix·W − r_break·R = (r_fix + r_break)·(p*·n − R) con n = W + R: l'argmax
    dipende solo da p*. Il candidato q=0 (non riaprire niente) vale 0; a parità
    vince il q più piccolo (costa meno forward). Ritorna (q, indice; −1 per q=0).
    """
    vals = np.concatenate([[0.0], pstar * n_cum - R_cum])
    k = int(np.argmax(vals))
    return (0.0, -1) if k == 0 else (float(qs[k - 1]), k - 1)


def first_crossing_q(qs, n_cum, R_cum, pstar) -> float:
    """Variante letterale: si scende in H finché il tratto marginale sta sotto p*."""
    n_t, R_t = np.diff(n_cum, prepend=0), np.diff(R_cum, prepend=0)
    q = 0.0
    for i in range(len(qs)):
        if n_t[i] == 0:
            q = float(qs[i])
            continue
        if R_t[i] / n_t[i] >= pstar:
            break
        q = float(qs[i])
    return q


def auroc_err(H: np.ndarray, wrong: np.ndarray) -> float:
    """P(H di una sbagliata > H di una giusta), pareggi a metà. 0.5 = nessuna separazione."""
    w = wrong.astype(bool)
    nw, nr = int(w.sum()), int((~w).sum())
    if nw == 0 or nr == 0:
        return float("nan")
    ranks = pd.Series(H).rank(method="average").to_numpy()
    return float((ranks[w].sum() - nw * (nw + 1) / 2) / (nw * nr))


# --------------------------------------------------------------------------- #
# Tassi del pass 2
# --------------------------------------------------------------------------- #
def parse_rate(txt: str | None, name: str):
    """'0.32' → (0.32, None, None); '22/68' → (0.3235, 22, 68)."""
    if txt is None:
        return None
    if "/" in txt:
        k, n = (int(x) for x in txt.split("/"))
        if n <= 0 or not 0 <= k <= n:
            raise SystemExit(f"{name}: frazione non valida {txt!r}")
        return k / n, k, n
    v = float(txt)
    if not 0 <= v <= 1:
        raise SystemExit(f"{name}: deve stare in [0, 1]")
    return v, None, None


def rates_from(c1: np.ndarray, c2: np.ndarray) -> tuple[float, float, int, int, int, int]:
    W, R = c1 == 0, c1 == 1
    fix, brk = int((W & (c2 == 1)).sum()), int((R & (c2 == 0)).sum())
    nW, nR = int(W.sum()), int(R.sum())
    rf = fix / nW if nW else float("nan")
    rb = brk / nR if nR else float("nan")
    return rf, rb, fix, nW, brk, nR


@dataclass
class Tassi:
    r_fix: float
    r_break: float
    fonte: str
    da_dati: bool
    cli_fix: tuple | None = None
    cli_break: tuple | None = None

    @property
    def pstar(self) -> float:
        return self.r_fix / (self.r_fix + self.r_break)

    def draw(self, rng: np.random.Generator, d: Dati, idx: np.ndarray) -> tuple[float, float]:
        """Tassi su un ricampionamento: dai dati (stesse righe del bootstrap del pass 1,
        così p* e quantile variano insieme) o binomiali se la CLI dà k/n."""
        if self.da_dati:
            sel = d.p2[idx]
            rf, rb, *_ = rates_from(d.c1[idx][sel], d.c2[idx][sel])
            return rf, rb
        out = []
        for v, _k, n in (self.cli_fix, self.cli_break):
            out.append(rng.binomial(n, v) / n if n else v)
        return out[0], out[1]


def perm_test(x: np.ndarray, y: np.ndarray, bins: np.ndarray, n_bins: int, B: int,
              rng: np.random.Generator) -> dict:
    """Test di permutazione di "tasso costante in H" per un esito 0/1.

    Due statistiche, a margini fissi (permutare y conserva eventi e conteggi):
      - trend: correlazione fra H continua ed esito (coglie crescita/decrescita);
      - eterogeneità: χ² di Pearson 2×K fra i tratti (coglie forme non monotone).
    p = (1 + #perm ≥ osservato) / (1 + B). Niente scipy: il repo non la dipende.
    """
    m, ev = len(y), int(y.sum())
    base = {"n": m, "eventi": ev, "corr_H": np.nan, "p_trend": np.nan,
            "chi2": np.nan, "p_eterogeneita": np.nan}
    if m < 3 or ev == 0 or ev == m:
        return base
    xc = x - x.mean()
    sx, sy = np.sqrt((xc ** 2).sum()), np.sqrt(((y - y.mean()) ** 2).sum())
    onehot = np.eye(n_bins)[bins]                          # (m, K)
    nk = onehot.sum(0)
    keep = nk > 0
    p = ev / m

    def chi2(s_k):
        return (((s_k - nk * p) ** 2)[..., keep] / (nk * p * (1 - p))[keep]).sum(-1)

    obs_t = float(xc @ y)
    obs_c = float(chi2(y @ onehot))
    Y = rng.permuted(np.tile(y, (B, 1)), axis=1)            # (B, m)
    perm_t = Y @ xc
    perm_c = chi2(Y @ onehot)
    base.update(corr_H=obs_t / (sx * sy),
                p_trend=(1 + int((np.abs(perm_t) >= abs(obs_t) - 1e-12).sum())) / (1 + B),
                chi2=obs_c,
                p_eterogeneita=(1 + int((perm_c >= obs_c - 1e-12).sum())) / (1 + B))
    return base


def tract_rates(d: Dati, n_tratti: int, B: int, alpha: float, rng: np.random.Generator):
    """r_fix / r_break / p* per tratti di H a numerosità uguale fra i sample col pass 2."""
    H2, c1, c2 = d.H[d.p2], d.c1[d.p2], d.c2[d.p2]
    edges = np.quantile(H2, np.linspace(0, 1, n_tratti + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], H2, side="right"), 0, n_tratti - 1)
    rows = []
    # tratto 1 = H più alta, come nella tabella dei quantili
    for rank, b in enumerate(range(n_tratti - 1, -1, -1), start=1):
        sel = bins == b
        m = int(sel.sum())
        if m == 0:
            continue
        hb, a1, a2 = H2[sel], c1[sel], c2[sel]
        rf, rb, fix, nW, brk, nR = rates_from(a1, a2)
        idx = rng.integers(0, m, (B, m))
        W_b = (a1 == 0)[idx].sum(1)
        R_b = (a1 == 1)[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rf_b = ((a1 == 0) & (a2 == 1))[idx].sum(1) / W_b
            rb_b = ((a1 == 1) & (a2 == 0))[idx].sum(1) / R_b
            ps_b = rf_b / (rf_b + rb_b)
        lo, hi = float(hb.min()), float(hb.max())
        rows.append({
            "tratto": rank, "H_min": lo, "H_max": hi,
            # posizione del tratto nella scala dei quantili del dataset intero
            "q_da": float((d.H > hi).mean()), "q_a": float((d.H >= lo).mean()),
            "n": m, "acc_pass1": float(a1.mean()),
            "W": nW, "recuperati": fix, "r_fix": rf,
            "r_fix_lo": _pct(rf_b, 2.5), "r_fix_hi": _pct(rf_b, 97.5),
            "R": nR, "rotti": brk, "r_break": rb,
            "r_break_lo": _pct(rb_b, 2.5), "r_break_hi": _pct(rb_b, 97.5),
            "p_star": rf / (rf + rb) if (rf + rb) > 0 else np.nan,
            "p_star_lo": _pct(ps_b, 2.5), "p_star_hi": _pct(ps_b, 97.5),
            "netto": fix - brk,
        })
    tab = pd.DataFrame(rows)

    tests = []
    for nome, sel, esito in (("r_fix", c1 == 0, c2 == 1), ("r_break", c1 == 1, c2 == 0)):
        t = perm_test(H2[sel], esito[sel].astype(float), bins[sel], n_tratti, B, rng)
        ps = [t["p_trend"], t["p_eterogeneita"]]
        if np.isnan(ps).all():
            verdetto = "indefinito (nessun evento o nessuna variazione)"
        elif np.nanmin(ps) < alpha:
            verso = "cresce" if t["corr_H"] > 0 else "cala"
            verdetto = f"NON costante (p<{alpha}); il tasso {verso} con H"
        else:
            verdetto = "compatibile con costante"
            if t["eventi"] < 10 * n_tratti:
                verdetto += " — pochi eventi, test poco potente: non è una prova di costanza"
        tests.append({"tasso": nome, **t, "verdetto": verdetto})
    return tab, pd.DataFrame(tests), edges, bins


def _pct(a: np.ndarray, p: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


# --------------------------------------------------------------------------- #
# Tabelle
# --------------------------------------------------------------------------- #
def quantile_table(d: Dati, qs, ts, M, n_cum, R_cum, pstar: float | None) -> pd.DataFrame:
    n_t, R_t = np.diff(n_cum, prepend=0), np.diff(R_cum, prepend=0)
    tot_R = int((d.c1 == 1).sum())
    rows = []
    for i, q in enumerate(qs):
        n_acc = d.n - n_cum[i]
        row = {
            "q": q, "soglia_bit": ts[i] if np.isfinite(ts[i]) else np.nan,
            "frac_riaperta": n_cum[i] / d.n, "n_riaperti": int(n_cum[i]),
            "W_riaperti": int(n_cum[i] - R_cum[i]), "R_riaperti": int(R_cum[i]),
            "acc_riaperti": R_cum[i] / n_cum[i] if n_cum[i] else np.nan,
            "acc_accettati": (tot_R - R_cum[i]) / n_acc if n_acc else np.nan,
            "n_tratto": int(n_t[i]), "W_tratto": int(n_t[i] - R_t[i]), "R_tratto": int(R_t[i]),
            "acc_tratto": R_t[i] / n_t[i] if n_t[i] else np.nan,
        }
        if pstar is not None:
            row["tratto_sotto_p*"] = bool(n_t[i] and R_t[i] / n_t[i] < pstar)
        rows.append(row)
    return pd.DataFrame(rows)


def auroc_table(d: Dati, min_gruppo: int) -> tuple[pd.DataFrame, list[str]]:
    wrong = d.c1 == 0
    rows = [{"gruppo": "tutti", "valore": "tutti", "n": d.n, "acc_pass1": d.c1.mean(),
             "H_mediana": float(np.median(d.H)), "auroc_errore": auroc_err(d.H, wrong)}]
    note = []
    for col in GROUP_COLS:
        if col not in d.df.columns:
            continue
        g = d.df[col]
        if pd.api.types.is_numeric_dtype(g) and g.nunique() > 10:
            g = pd.qcut(g, 3, duplicates="drop").astype(str).radd("terzile ")
        g = g.astype("string").fillna("<mancante>")
        scartati = 0
        for val in sorted(g.unique()):
            sel = (g == val).to_numpy(bool)
            if sel.sum() < min_gruppo:
                scartati += 1
                continue
            rows.append({"gruppo": col, "valore": val, "n": int(sel.sum()),
                         "acc_pass1": d.c1[sel].mean(), "H_mediana": float(np.median(d.H[sel])),
                         "auroc_errore": auroc_err(d.H[sel], wrong[sel])})
        if scartati:
            note.append(f"{col}: {scartati} valori con n < {min_gruppo} omessi")
    return pd.DataFrame(rows), note


def curve_table(d: Dati, qs, ts, M, n_cum, R_cum, tassi: Tassi | None, costo: float,
                B: int, rng: np.random.Generator) -> pd.DataFrame:
    """Netto osservato per quantile, solo se il pass 2 copre TUTTI i riaperti.

    L'IC ricampiona i sample a soglie fisse: dice se un netto di +15 su 2700 è
    distinguibile da zero, non quanto è stabile la soglia.
    """
    delta = np.where(d.p2, np.nan_to_num(d.c2) - d.c1, 0.0)        # +1 recupero, −1 rottura
    idx = rng.integers(0, d.n, (B, d.n))
    rows = []
    for i, q in enumerate(qs):
        S = M[i]
        coperto = not (S & ~d.p2).any()
        fw = n_cum[i] * costo
        prev = (tassi.r_fix * (n_cum[i] - R_cum[i]) - tassi.r_break * R_cum[i]) if tassi else np.nan
        row = {"q": q, "soglia_bit": ts[i] if np.isfinite(ts[i]) else np.nan,
               "frac_riaperta": n_cum[i] / d.n, "forward_aggiuntivi": fw, "coperto": coperto,
               "netto_previsto": prev}
        if coperto:
            rec = int((S & (d.c1 == 0) & (d.c2 == 1)).sum())
            rot = int((S & (d.c1 == 1) & (d.c2 == 0)).sum())
            boot = (delta * S)[idx].sum(1)
            netto = rec - rot
            row.update(recuperati=rec, rotti=rot, netto=netto,
                       netto_lo=_pct(boot, 2.5), netto_hi=_pct(boot, 97.5),
                       acc_pass1=d.c1.mean(), acc_finale=(d.c1.sum() + netto) / d.n,
                       delta_pp=100 * netto / d.n,
                       netto_per_100fw=100 * netto / fw if fw else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("recuperati", "rotti", "netto"):     # conteggi: interi anche dove NaN
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    return out


# --------------------------------------------------------------------------- #
# Raccomandazione
# --------------------------------------------------------------------------- #
def describe_q(q, k, d, ts, n_cum, R_cum, tassi, costo, curve) -> str:
    if k < 0:
        return "q=0: non riaprire nessun sample"
    fw = n_cum[k] * costo
    prev = tassi.r_fix * (n_cum[k] - R_cum[k]) - tassi.r_break * R_cum[k]
    soglia = "tutti" if not np.isfinite(ts[k]) else f"H > {ts[k]:.4f} bit"
    s = (f"q={q:.2f} ({soglia}, riapre {n_cum[k] / d.n:.1%} = {n_cum[k]} sample); "
         f"netto previsto {prev:+.1f} ({100 * prev / d.n:+.2f} pp), "
         f"{100 * prev / fw:+.2f} netti ogni 100 forward aggiuntivi")
    if curve is not None:
        r = curve.iloc[k]
        if r["coperto"]:
            s += f"; OSSERVATO {int(r['netto']):+d} [IC95 {r['netto_lo']:+.0f}, {r['netto_hi']:+.0f}]"
        else:
            s += "; fuori copertura del pass 2 (estrapolazione)"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--step", type=float, default=0.05, help="passo dei quantili (default 0.05)")
    ap.add_argument("--r-fix", help="tasso di recupero: numero o k/n (sostituisce la stima dai dati)")
    ap.add_argument("--r-break", help="tasso di rottura: numero o k/n (sostituisce la stima dai dati)")
    ap.add_argument("--costo-pass2", type=float, default=1.0,
                    help="forward-equivalenti per sample riaperto (default 1)")
    ap.add_argument("--lambda-costo", type=float, default=0.01,
                    help="netto minimo richiesto per forward aggiuntivo nella variante col costo "
                         "(default 0.01 = 1 sample ogni 100 forward)")
    ap.add_argument("--n-tratti", type=int, default=5, help="tratti di H per il test dei tassi (default 5)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05, help="livello dei test di costanza")
    ap.add_argument("--min-gruppo", type=int, default=30, help="n minimo per riportare un gruppo")
    ap.add_argument("--xlsx", type=Path, help="scrive un foglio per tabella (serve openpyxl)")
    args = ap.parse_args()
    if (args.r_fix is None) != (args.r_break is None):
        ap.error("--r-fix e --r-break vanno passati insieme")
    if args.xlsx:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            ap.error("--xlsx richiede openpyxl: uv run --with openpyxl python scripts/choose_entropy_quantile.py ...")

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    d = load(args.csv)
    qs = quantile_grid(args.step)
    ts, M, n_cum, R_cum = cum_counts(d.H, d.c1, qs)
    avvisi: list[str] = []

    print(f"# {args.csv}: {d.n} sample, accuracy pass 1 {d.c1.mean():.2%}, "
          f"pass 2 disponibile su {int(d.p2.sum())} sample ({d.p2.mean():.1%})")
    eff = n_cum / d.n
    scarto = np.abs(eff - qs)
    if (scarto > 0.02).any():
        avvisi.append(f"pareggi in H: la frazione riaperta effettiva si scosta dal quantile nominale "
                      f"fino a {scarto.max():.1%} (vedi frac_riaperta)")

    # --- tassi: dati o riga di comando ----------------------------------------
    dati_rates = None
    if d.has_pass2:
        rf, rb, fix, nW, brk, nR = rates_from(d.c1[d.p2], d.c2[d.p2])
        dati_rates = (rf, rb, fix, nW, brk, nR)
        reo_H = d.H[d.p2]
        non_H = d.H[~d.p2]
        if len(non_H) and non_H.max() > reo_H.min():
            avvisi.append("i sample col pass 2 non sono un taglio netto in H (gate non a soglia unica, "
                          "shard misti?): la curva usa solo i quantili interamente coperti")
    tassi: Tassi | None = None
    if args.r_fix is not None:
        cf, cb = parse_rate(args.r_fix, "--r-fix"), parse_rate(args.r_break, "--r-break")
        tassi = Tassi(cf[0], cb[0], f"riga di comando: r_fix={args.r_fix}, r_break={args.r_break}",
                      da_dati=False, cli_fix=cf, cli_break=cb)
        if dati_rates:
            avvisi.append("tassi da riga di comando: sostituiscono quelli stimati dal pass 2 nei dati")
    elif dati_rates:
        rf, rb, fix, nW, brk, nR = dati_rates
        cov = d.H[d.p2].min()
        tassi = Tassi(rf, rb, f"dati: pass 2 su {int(d.p2.sum())} sample (H ≥ {cov:.4f} bit), "
                              f"recuperati {fix}/{nW}, rotti {brk}/{nR}", da_dati=True)
    if tassi is not None and not (tassi.r_fix + tassi.r_break > 0):
        raise SystemExit("r_fix + r_break = 0: il pass 2 non cambia nulla, p* indefinito")

    pstar = tassi.pstar if tassi else None
    qt = quantile_table(d, qs, ts, M, n_cum, R_cum, pstar)
    au, note_gruppi = auroc_table(d, args.min_gruppo)

    curve = tt = tests = None
    if d.has_pass2:
        curve = curve_table(d, qs, ts, M, n_cum, R_cum, tassi, args.costo_pass2, B, rng)
        n_bins = min(args.n_tratti, max(1, int(d.p2.sum()) // 20))
        if n_bins < args.n_tratti:
            avvisi.append(f"pochi sample col pass 2: {n_bins} tratti invece di {args.n_tratti}")
        tt, tests, _, _ = tract_rates(d, n_bins, B, args.alpha, rng)
        if tests["verdetto"].str.startswith("NON").any():
            avvisi.append("TASSI NON COSTANTI in H (vedi test_costanza): la regola con p* unico è "
                          "distorta; guardare la variante a tassi locali e la curva osservata")

    # --- raccomandazione --------------------------------------------------------
    rec: list[tuple[str, str]] = []
    if tassi is None:
        rec.append(("regola", "nessun pass 2 nei dati e nessun --r-fix/--r-break: "
                              "solo tabella dei quantili e AUROC"))
    else:
        # un unico bootstrap congiunto: pass 1 ricampionato + tassi (dagli stessi
        # sample se vengono dai dati) → IC di p* e distribuzione del q scelto
        pst_b, rf_b, rb_b, q_b = [], [], [], []
        for _ in range(B):
            idx = rng.integers(0, d.n, d.n)
            rfb, rbb = tassi.draw(rng, d, idx)
            rf_b.append(rfb)
            rb_b.append(rbb)
            if not np.isfinite(rfb + rbb) or rfb + rbb == 0:
                continue
            ps = rfb / (rfb + rbb)
            pst_b.append(ps)
            _, _, nb, Rb = cum_counts(d.H[idx], d.c1[idx], qs)
            q_b.append(best_q(qs, nb, Rb, ps)[0])
        pst_b, rf_b, rb_b, q_b = map(np.asarray, (pst_b, rf_b, rb_b, q_b))
        ci_ok = tassi.da_dati or tassi.cli_fix[2] or tassi.cli_break[2]
        ps_lo, ps_hi = _pct(pst_b, 2.5), _pct(pst_b, 97.5)

        rec.append(("fonte tassi", tassi.fonte))
        if dati_rates and not tassi.da_dati:
            rec.append(("tassi stimati dai dati (non usati)",
                        f"r_fix={dati_rates[0]:.4f}, r_break={dati_rates[1]:.4f}, "
                        f"p*={dati_rates[0] / (dati_rates[0] + dati_rates[1]):.4f}"))
        if ci_ok:
            rec.append(("r_fix", f"{tassi.r_fix:.4f} [IC95 {_pct(rf_b, 2.5):.4f}, {_pct(rf_b, 97.5):.4f}]"))
            rec.append(("r_break", f"{tassi.r_break:.4f} [IC95 {_pct(rb_b, 2.5):.4f}, {_pct(rb_b, 97.5):.4f}]"))
            rec.append(("p* = r_fix/(r_fix+r_break)", f"{pstar:.4f} [IC95 {ps_lo:.4f}, {ps_hi:.4f}]"))
        else:
            rec.append(("r_fix / r_break", f"{tassi.r_fix:.4f} / {tassi.r_break:.4f} (senza denominatore: niente IC)"))
            rec.append(("p* = r_fix/(r_fix+r_break)", f"{pstar:.4f}"))
        rec.append(("confronto", f"accuracy pass 1 globale {d.c1.mean():.4f}: se p* è sotto, "
                                 "riaprire TUTTO costa più di quanto rende"))

        q_par, k_par = best_q(qs, n_cum, R_cum, pstar)
        rec.append(("QUANTILE SCELTO (pareggio)", describe_q(q_par, k_par, d, ts, n_cum, R_cum, tassi,
                                                              args.costo_pass2, curve)))
        q_first = first_crossing_q(qs, n_cum, R_cum, pstar)
        rec.append(("variante primo attraversamento", f"q={q_first:.2f} (ci si ferma al primo tratto con "
                                                      "accuracy ≥ p*; più sensibile al rumore dei tratti)"))

        kappa = args.lambda_costo * args.costo_pass2
        pst_c = (tassi.r_fix - kappa) / (tassi.r_fix + tassi.r_break)
        q_c, k_c = best_q(qs, n_cum, R_cum, pst_c)
        rec.append((f"variante col costo (λ={args.lambda_costo}/forward, costo pass 2={args.costo_pass2})",
                    f"p*_costo = (r_fix − λ·costo)/(r_fix + r_break) = {pst_c:.4f} → "
                    + describe_q(q_c, k_c, d, ts, n_cum, R_cum, tassi, args.costo_pass2, curve)))

        if ci_ok:
            q_lo, _ = best_q(qs, n_cum, R_cum, ps_lo)
            q_hi, _ = best_q(qs, n_cum, R_cum, ps_hi)
            rec.append(("sensibilità a p*", f"pessimista p*={ps_lo:.4f} → q={q_lo:.2f}; "
                                            f"ottimista p*={ps_hi:.4f} → q={q_hi:.2f}"))
        if len(q_b):
            rec.append(("stabilità (bootstrap del q scelto)",
                        f"mediana {np.median(q_b):.2f}, IC90 [{np.percentile(q_b, 5):.2f}, "
                        f"{np.percentile(q_b, 95):.2f}], q=0 nel {np.mean(q_b == 0):.0%} dei ricampionamenti"))

        if tt is not None and len(tt) > 1:
            # tassi del tratto di H del sample; sotto la copertura del pass 2 si
            # usa il tratto più basso (estrapolazione dichiarata)
            edges_hi = tt.sort_values("H_min")["H_min"].to_numpy()[1:]
            loc = tt.sort_values("H_min")
            b = np.searchsorted(edges_hi, d.H, side="right")
            rfl, rbl = loc["r_fix"].to_numpy()[b], loc["r_break"].to_numpy()[b]
            contrib = np.where(d.c1 == 0, rfl, -rbl)
            cum_loc = (M * contrib[None, :]).sum(1)
            vals = np.concatenate([[0.0], cum_loc])
            kl = int(np.argmax(vals))
            q_loc = 0.0 if kl == 0 else float(qs[kl - 1])
            rec.append(("variante tassi locali (per tratto)",
                        f"q={q_loc:.2f}, netto previsto {vals[kl]:+.1f}"
                        + ("" if kl == 0 or curve.iloc[kl - 1]["coperto"] else " (fuori copertura)")))

        if curve is not None:
            cov = curve[curve["coperto"]]
            if len(cov):
                best = cov.loc[cov["netto"].idxmax()]
                rec.append(("massimo OSSERVATO sulla curva (in-sample, ottimistico)",
                            f"q={best['q']:.2f}: netto {int(best['netto']):+d} "
                            f"[IC95 {best['netto_lo']:+.0f}, {best['netto_hi']:+.0f}]; copertura del pass 2 "
                            f"fino a q={cov['q'].max():.2f}"))
                if q_par > cov["q"].max() + 1e-9:
                    avvisi.append(f"il quantile scelto ({q_par:.2f}) supera la copertura del pass 2 "
                                  f"(q ≤ {cov['q'].max():.2f}): r_fix/r_break sono estrapolati a H più basse")
            else:
                avvisi.append("nessun quantile interamente coperto dal pass 2: curva osservata vuota")
        elif not tassi.da_dati:
            avvisi.append("nessun pass 2 nei dati: la raccomandazione dipende interamente dai tassi "
                          "passati da riga di comando (assunti costanti in H)")

    # --- stampa -----------------------------------------------------------------
    pct = {"frac_riaperta", "acc_riaperti", "acc_accettati", "acc_tratto", "acc_pass1", "acc_finale", "q_da", "q_a"}
    section("QUANTILI (si riapre H > soglia; tratto = fra il quantile precedente e questo)", qt, pct)
    section("AUROC di H come predittore d'errore (0.5 = nessuna separazione)", au, pct)
    for n in note_gruppi:
        print(f"  nota: {n}")
    if curve is not None:
        section("CURVA DEL NETTO (osservato solo dove il pass 2 copre tutti i riaperti)", curve, pct)
        section("TASSI PER TRATTO DI H (fra i sample col pass 2; IC95 bootstrap)", tt, pct)
        section(f"TEST DI COSTANZA DEI TASSI (permutazione, {B} ricampionamenti)", tests, pct)
    print("\n== RACCOMANDAZIONE ==")
    for k, v in rec:
        print(f"  {k}: {v}")
    if avvisi:
        print("\n== AVVISI ==")
        for a in avvisi:
            print(f"  ⚠️  {a}")

    if args.xlsx:
        args.xlsx.parent.mkdir(parents=True, exist_ok=True)
        rec_df = pd.DataFrame(rec + [("avviso", a) for a in avvisi], columns=["voce", "valore"])
        with pd.ExcelWriter(args.xlsx, engine="openpyxl") as xw:
            rec_df.to_excel(xw, sheet_name="raccomandazione", index=False)
            qt.to_excel(xw, sheet_name="quantili", index=False)
            au.to_excel(xw, sheet_name="auroc", index=False)
            if curve is not None:
                curve.to_excel(xw, sheet_name="curva_netto", index=False)
                tt.to_excel(xw, sheet_name="tassi_tratti", index=False)
                tests.to_excel(xw, sheet_name="test_costanza", index=False)
        print(f"\nscritto {args.xlsx}")
    return 0


def section(title: str, df: pd.DataFrame, pct: set[str]) -> None:
    print(f"\n== {title} ==")
    fmt = {}
    for c in df.columns:
        if c in pct:
            fmt[c] = lambda v: "—" if pd.isna(v) else f"{100 * v:.1f}%"
        elif pd.api.types.is_float_dtype(df[c]):
            fmt[c] = lambda v: "—" if pd.isna(v) else f"{v:.3f}"
    print(df.to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    sys.exit(main())
