#!/usr/bin/env python3
"""Ispezione di QUALSIASI run wandb di questo progetto, via API.

I job girano su SLURM al cluster AImageLab; da questa macchina non si arriva
né ai log né ai file di output. L'unico canale sono le API wandb (config +
summary della run) e Weave (l'output PER-SAMPLE di ogni `predict`, dove
finiscono i campi che le strategy ritornano da `answer()`).

Due tipi di run, riconosciuti da sé:

- **eval** (`main.py`): config con `model`/`dataset`/`strategy`. Report con i
  check di correttezza — che stanno quasi tutti nei campi per-sample, NON nel
  summary: `mcq_accuracy` nel summary dice solo che l'eval non è morta.
- **altro** (es. `fetch/prefetch_*.py`): nessuna nozione di sample/accuracy,
  si stampano config e summary così come sono.

Uso:
    python .claude/skills/wandb/driver.py --list [N] [--project P | --all-projects]
    python .claude/skills/wandb/driver.py --projects
    python .claude/skills/wandb/driver.py <run-name> [--project P]
    python .claude/skills/wandb/driver.py --compare <run-a> <run-b>

Il progetto si trova da solo: se il nome non è nel progetto indicato, la run
viene cercata in tutti i progetti dell'entity.

Credenziali: `~/.netrc` (machine api.wandb.ai). Nessun login interattivo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def default_entity() -> str:
    """Entity dal `conf/config.yaml` del repo (preset `wandb.default`), così
    non è cablata qui. Fallback all'entity storica se il file cambia forma."""
    try:
        import yaml

        cfg = yaml.safe_load((REPO / "conf" / "config.yaml").read_text())
        return cfg["wandb"]["default"]["entity"]
    except Exception:
        return "alesvale97-unimore"


# ─────────────────────────────────────────────────────────────────────────────
# Ricerca della run
# ─────────────────────────────────────────────────────────────────────────────
def _projects(api, entity: str) -> list[str]:
    return [p.name for p in api.projects(entity)]


def list_runs(entity: str, project: str | None, n: int, all_projects: bool) -> None:
    import wandb

    api = wandb.Api()
    projects = _projects(api, entity) if all_projects else [project]
    rows = []
    for proj in projects:
        try:
            for i, r in enumerate(api.runs(f"{entity}/{proj}", order="-created_at")):
                if i >= n:
                    break
                rows.append((r.created_at, proj, r.name, r.state, r.summary.get("mcq_accuracy"), r.group))
        except Exception as e:
            print(f"⚠️  {proj}: {type(e).__name__}", file=sys.stderr)
    rows.sort(reverse=True)
    for created, proj, name, state, acc, group in rows[:n if not all_projects else len(rows)]:
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "—"
        print(f"{created}  {proj:<20} {name:<58} {state:<9} acc={acc_s:<8} group={group}")


def find_run(entity: str, project: str | None, name: str):
    """La run `name`, cercata prima in `project` e poi in TUTTI i progetti.

    Il progetto wandb di una eval è `${dataset.name}` (`conf/config.yaml`),
    quindi il nome da solo non dice dove sta la run: cercarla ovunque evita di
    dover sapere in anticipo il dataset.
    """
    import wandb

    api = wandb.Api()
    tried = []
    order = ([project] if project else []) + [p for p in _projects(api, entity) if p != project]
    for proj in order:
        try:
            runs = list(api.runs(f"{entity}/{proj}", filters={"display_name": name}))
        except Exception:
            continue
        tried.append(proj)
        if runs:
            runs.sort(key=lambda r: r.created_at, reverse=True)
            if len(runs) > 1:
                print(f"⚠️  {len(runs)} run con questo nome in {proj}, uso la più recente",
                      file=sys.stderr)
            return runs[0], proj
    sys.exit(f"run {name!r} non trovata in nessun progetto di {entity} "
             f"(cercata in: {', '.join(tried)}).\nControlla il nome — `--list` "
             f"mostra le più recenti, `--projects` i progetti.")


def is_eval_run(run) -> bool:
    cfg = run.config
    return bool(cfg.get("dataset") and cfg.get("model")) or "mcq_accuracy" in run.summary


# ─────────────────────────────────────────────────────────────────────────────
# Weave: gli output per-sample (solo per le run di eval)
# ─────────────────────────────────────────────────────────────────────────────
def per_sample_outputs(entity: str, project: str, run) -> list[dict]:
    """Output di ogni `predict` della run, via Weave. `[]` se non trovati.

    ⚠️ Weave NON registra il run id di wandb da nessuna parte (gli `attributes`
    dei call contengono solo versione python/OS). Il collegamento run wandb →
    evaluation weave si può fare SOLO per vicinanza temporale.

    Per non fidarsi di quell'euristica, l'accoppiamento viene poi VERIFICATO
    contro la config della run: i campi che la strategy ri-emette per sample
    (`cell_select`, `sink_filter`, ...) devono coincidere con
    `config.strategy.*`. Due arm lanciati a pochi secondi l'uno dall'altro
    (attention vs random) si distinguono esattamente così.
    """
    import weave

    client = weave.init(f"{entity}/{project}")
    prefix = f"weave:///{entity}/{project}/op"
    t_run = dt.datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))

    evs = list(client.get_calls(
        filter={"op_names": [f"{prefix}/Evaluation.evaluate:*"]},
        limit=40, sort_by=[{"field": "started_at", "direction": "desc"}],
    ))
    cands = sorted(
        (e for e in evs if e.started_at >= t_run - dt.timedelta(seconds=30)),
        key=lambda e: abs((e.started_at - t_run).total_seconds()),
    )
    if not cands:
        print("⚠️  nessuna Evaluation weave vicina a questa run: niente dati per-sample "
              "(la run è morta prima di iniziare l'eval?)", file=sys.stderr)
        return []

    strategy_cfg = run.config.get("strategy", {}) or {}
    for ev in cands[:6]:
        calls = list(client.get_calls(
            filter={"op_names": [f"{prefix}/predict:*"], "trace_ids": [ev.trace_id]},
            limit=5000,
        ))
        outs = [dict(c.output) for c in calls if c.output]
        if not outs:
            continue
        ok, checked = True, []
        for key in ("cell_select", "sink_filter", "resample_kind"):
            want = strategy_cfg.get(key)
            if want is None:
                continue
            got = {o.get(key) for o in outs if key in o}
            if not got:
                continue
            checked.append(key)
            if got != {want} and not (key == "resample_kind" and None in got):
                ok = False
                break
        if ok:
            # Onestà del messaggio: senza campi da confrontare l'accoppiamento
            # NON è verificato, è solo il più vicino nel tempo. Conta quando
            # più eval partono nella stessa finestra.
            how = (f"accoppiamento verificato su {'/'.join(checked)}" if checked else
                   "⚠️ accoppiamento per SOLA vicinanza temporale: config.strategy non ha "
                   "campi ri-emessi per sample da confrontare")
            print(f"weave: evaluation {ev.display_name} ({ev.started_at:%Y-%m-%d %H:%M:%S}) "
                  f"→ {len(outs)} sample [{how}]")
            return outs
    print("⚠️  Evaluation weave vicine alla run trovate, ma nessuna con campi per-sample "
          "coerenti con `config.strategy`: accoppiamento non verificabile, salto i "
          "check per-sample invece di riportare numeri di un'altra run.", file=sys.stderr)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))

    def render(self) -> int:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "  "}
        print()
        for status, name, detail in self.rows:
            print(f"{icon[status]} {name}: {detail}")
        n_fail = sum(1 for s, _, _ in self.rows if s == "FAIL")
        n_warn = sum(1 for s, _, _ in self.rows if s == "WARN")
        n_pass = sum(1 for s, _, _ in self.rows if s == "PASS")
        print()
        if n_fail:
            print(f"VERDETTO: {n_fail} check FALLITI — non leggere l'accuracy.")
        elif n_warn:
            print(f"VERDETTO: {n_pass} check passati, {n_warn} avvisi da guardare.")
        else:
            print(f"VERDETTO: tutti i check di correttezza passati ({n_pass}).")
        return 1 if n_fail else 0


def _vals(outs: list[dict], key: str) -> list:
    return [o[key] for o in outs if o.get(key) is not None]


def check_eval(run, outs: list[dict], shard_size: int | None = None) -> Report:
    r = Report()
    s, cfg = run.summary, run.config
    strat = cfg.get("strategy", {}) or {}
    n = len(outs)

    # --- l'eval è viva (lo stato del job NON è un check) --------------------
    acc = s.get("mcq_accuracy")
    if acc is None:
        r.add("FAIL", "eval viva", "`mcq_accuracy` ASSENTE dal summary — l'eval non ha "
                                   "prodotto predizioni valide (tipico OOM per-sample "
                                   "catturato da Weave, oppure job ucciso dalla walltime: "
                                   "in entrambi i casi lo stato del job può dire 'finished')")
    else:
        r.add("PASS", "eval viva", f"mcq_accuracy={acc:.4f} presente nel summary "
                                   f"({s.get('n_correct')}/{s.get('n_samples')})")
    if run.state != "finished":
        r.add("WARN", "stato job", f"state={run.state} — se è 'crashed' su un job lungo, "
                                   f"sospetta la walltime prima di un bug")

    if not outs:
        r.add("WARN", "check per-sample", "SALTATI (nessun dato Weave): query_rows, "
                                          "sink_filter, blocchi/timestamp e copertura "
                                          "dell'intervento stanno solo lì")
    else:
        # --- pred parseabili ------------------------------------------------
        fb = [o for o in outs if o.get("pred_fallback")]
        (r.add("PASS", "pred_fallback", f"0/{n} — tutte le risposte del pass 2 parseabili")
         if not fb else
         r.add("WARN", "pred_fallback", f"{len(fb)}/{n} sample usano l'argmax del pass 1"))
        missing = [o for o in outs if "pred" in o and o.get("pred") is None]
        if missing:
            r.add("FAIL", "pred", f"{len(missing)}/{n} sample senza `pred`")

        # --- righe-query: il selettore ha tagliato? -------------------------
        qr, qt = _vals(outs, "n_query_rows"), _vals(outs, "n_query_tokens")
        if qr and qt:
            want = strat.get("query_rows")
            bad = [(a, b) for a, b in zip(qr, qt) if a >= b]
            if want in (None, "all"):
                r.add("INFO", "query_rows", f"query_rows={want!r}: nessun taglio atteso")
            elif bad:
                r.add("FAIL", "query_rows", f"{len(bad)}/{n} sample con n_query_rows >= "
                                            f"n_query_tokens: selettore degradato al fallback "
                                            f"'tutte le righe', l'arm gira a query_rows=all")
            else:
                frac = statistics.fmean(a / b for a, b in zip(qr, qt))
                r.add("PASS", "query_rows", f"n_query_rows < n_query_tokens su {n}/{n} "
                                            f"(query_rows={want!r}, in media il "
                                            f"{100*frac:.0f}% delle righe tenute)")

        # --- il filtro sink è davvero spento? -------------------------------
        pairs = [(o["peak_cell"], o["peak_cell_sink_filtered"]) for o in outs
                 if o.get("peak_cell") is not None and o.get("peak_cell_sink_filtered") is not None]
        if pairs and not strat.get("sink_filter", True):
            same = sum(1 for a, b in pairs if a == b)
            pct = 100 * same / len(pairs)
            if same == len(pairs):
                r.add("FAIL", "sink_filter", f"peak_cell == peak_cell_sink_filtered su "
                                             f"{same}/{len(pairs)} sample (100%): o il knob non "
                                             f"arriva al ranking, o la sink_map è piatta e il "
                                             f"filtro è inerte su questo modello — da distinguere")
            elif pct > 90:
                r.add("WARN", "sink_filter", f"le due celle coincidono sul {pct:.0f}% dei sample "
                                             f"({same}/{len(pairs)}): il filtro cambia pochissimo")
            else:
                r.add("PASS", "sink_filter", f"il filtro cambierebbe la cella scelta sul "
                                             f"{100-pct:.0f}% dei sample "
                                             f"({len(pairs)-same}/{len(pairs)}): "
                                             f"sink_filter=false arriva davvero al ranking")

        # --- blocchi: somma t invariata -------------------------------------
        bs = [o for o in outs if o.get("block_sizes")]
        if bs:
            bad_odd = [o for o in bs if any(x % 2 for x in o["block_sizes"])]
            bad_sum = [o for o in bs if sum(o["block_sizes"]) // 2 != o.get("t_cells")]
            if bad_odd or bad_sum:
                r.add("FAIL", "blocchi/timestamp", f"{len(bad_odd)} sample con blocco dispari, "
                                                   f"{len(bad_sum)} con somma(t) != t_cells: "
                                                   f"il processor ha paddato, timestamp sfasati")
            else:
                r.add("PASS", "blocchi/timestamp", f"blocchi tutti pari e somma(block_sizes)/2 "
                                                   f"== t_cells su {len(bs)}/{n} sample marcati")

        # --- copertura dell'intervento --------------------------------------
        for key in ("marked", "highlighted", "resampled"):
            vals = [o[key] for o in outs if key in o]
            if vals:
                t = sum(1 for v in vals if v)
                r.add("INFO" if t else "FAIL", key,
                      f"{t}/{len(vals)} sample ({100*t/len(vals):.0f}%)"
                      + ("" if t else " — NESSUN sample toccato dall'intervento"))
        skips = Counter(o.get("marked_skip_reason") or o.get("highlight_skip_reason")
                        for o in outs
                        if (o.get("marked_skip_reason") or o.get("highlight_skip_reason")))
        if skips:
            r.add("WARN", "skip", ", ".join(f"{k}×{v}" for k, v in skips.items()))

    # --- costo → walltime ---------------------------------------------------
    lat = s.get("model_latency_mean")
    if lat:
        ds = cfg.get("dataset") or {}
        nf = f" | nframes={ds.get('nframes')}" if ds.get("nframes") else ""
        ns = s.get("n_samples")
        obs = f", eval osservata {ns*lat/3600:.2f} h su {ns} sample" if ns else ""
        r.add("INFO", "costo", f"model_latency_mean={lat:.2f} s/sample{obs}{nf}")
        # L'estrapolazione serve a ritarare `--time` di un fullset partendo da
        # una probe. La dimensione dello shard NON è deducibile dalla run
        # (`limit` tronca, e il totale del dataset non è nel config): si passa
        # da `--shard-size`, e il numero usato è sempre stampato.
        if shard_size:
            h = shard_size * lat / 3600
            r.add("INFO", "  walltime", f"a {shard_size} sample/shard ≈ {h:.2f} h "
                                        f"(margine su 4 h: {4/h:.2f}×)")

    # --- accuracy (ULTIMA) --------------------------------------------------
    if acc is not None:
        cp1 = s.get("mcq_accuracy_correct_pass1_true")
        extra = (f" | sui corretti-al-pass-1: {cp1:.3f} "
                 f"({s.get('n_correct_correct_pass1_true')}/"
                 f"{s.get('n_samples_correct_pass1_true')})") if cp1 is not None else ""
        r.add("INFO", "accuracy", f"{acc:.4f}{extra}")
        for k in sorted(s.keys()):
            if k.startswith("mcq_accuracy_duration_"):
                r.add("INFO", f"  {k.replace('mcq_accuracy_', '')}",
                      f"{s[k]:.4f} ({s.get(k.replace('mcq_accuracy','n_correct'))}/"
                      f"{s.get(k.replace('mcq_accuracy','n_samples'))})")
    return r


def report_generic(run) -> int:
    """Run non-eval (prefetch, sweep, ...): non c'è nozione di sample né di
    accuracy, si mostrano config e summary così come sono."""
    print("\nRun NON di eval (nessun `model`/`dataset` in config): nessun check "
          "di correttezza applicabile, ecco config e summary.\n")
    print("--- config ---")
    print(json.dumps(dict(run.config), indent=2, default=str)[:2500])
    print("\n--- summary ---")
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    print(json.dumps(summary, indent=2, default=str)[:2500] if summary else "(vuoto)")
    rt = run.summary.get("_runtime")
    if rt:
        print(f"\nruntime: {rt/60:.1f} min | state: {run.state}")
    return 0


def describe(run, project: str) -> None:
    cfg = run.config
    print("=" * 92)
    print(f"run:      {run.name}  [{run.state}]  created={run.created_at}")
    print(f"project:  {project}   group={run.group}   tags={run.tags}")
    if is_eval_run(run):
        print(f"model:    {(cfg.get('model') or {}).get('name')}")
        ds = cfg.get("dataset") or {}
        print(f"dataset:  {ds.get('name')} nframes={ds.get('nframes')} "
              f"limit={cfg.get('limit')} shard={cfg.get('shard')}/{cfg.get('num_shards')} "
              f"shuffle={cfg.get('shuffle')}")
        print(f"strategy: {cfg.get('strategy')}")


def compare(entity: str, project: str | None, a: str, b: str) -> None:
    ra, pa = find_run(entity, project, a)
    rb, pb = find_run(entity, project, b)
    sa = ra.config.get("strategy", {}) or {}
    sb = rb.config.get("strategy", {}) or {}
    print("=" * 92)
    print(f"A = {a}   [{pa}]\n    {sa}")
    print(f"B = {b}   [{pb}]\n    {sb}")
    diff = {k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)}
    print(f"\ndifferenze di config.strategy: "
          f"{ {k: (sa.get(k), sb.get(k)) for k in sorted(diff)} or 'NESSUNA (stessi knob!)' }")
    print("-" * 92)
    print(f"{'metrica':<42} {'A':>22} {'B':>22}")
    print("-" * 92)
    keys = ["mcq_accuracy", "n_correct", "n_samples",
            "mcq_accuracy_correct_pass1_true", "mcq_accuracy_correct_pass1_false",
            "mcq_accuracy_duration_short", "mcq_accuracy_duration_medium",
            "mcq_accuracy_duration_long", "model_latency_mean"]
    for k in keys:
        va, vb = ra.summary.get(k), rb.summary.get(k)
        if va is None and vb is None:
            continue
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k:<42} {fmt(va):>22} {fmt(vb):>22}")
    na, nb, ns = ra.summary.get("n_correct"), rb.summary.get("n_correct"), ra.summary.get("n_samples")
    if None not in (na, nb, ns):
        print("-" * 92)
        print(f"delta corretti: {na - nb:+d} su {ns} sample ({100*(na-nb)/ns:+.1f} pp) — "
              f"su un campione di {ns} un delta di pochi sample NON è distinguibile dal rumore.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="nome ESATTO della run wandb (chiedilo all'utente)")
    p.add_argument("--list", nargs="?", type=int, const=15, metavar="N", help="ultime N run")
    p.add_argument("--all-projects", action="store_true", help="con --list: tutti i progetti")
    p.add_argument("--projects", action="store_true", help="elenca i progetti dell'entity")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    p.add_argument("--entity", default=None, help="default: da conf/config.yaml")
    p.add_argument("--project", default=None,
                   help="= dataset.name per le eval. Se omesso, la run viene cercata "
                        "in tutti i progetti")
    p.add_argument("--summary-only", action="store_true", help="salta Weave (veloce)")
    p.add_argument("--shard-size", type=int, default=None, metavar="N",
                   help="sample per shard del fullset, per estrapolare la walltime "
                        "da una probe (Video-MME a 3 shard: 900)")
    a = p.parse_args()
    entity = a.entity or default_entity()

    if a.projects:
        import wandb
        for name in _projects(wandb.Api(), entity):
            print(name)
        return 0
    if a.list:
        list_runs(entity, a.project or "video_mme", a.list, a.all_projects)
        return 0
    if a.compare:
        compare(entity, a.project, *a.compare)
        return 0
    if not a.run:
        p.error("serve il nome della run (o --list / --projects / --compare). "
                "Chiedilo all'utente: lo annota per ogni job lanciato su HPC.")

    run, project = find_run(entity, a.project, a.run)
    describe(run, project)
    if not is_eval_run(run):
        return report_generic(run)
    outs = [] if a.summary_only else per_sample_outputs(entity, project, run)
    if a.summary_only:
        print("\n(--summary-only: check per-sample saltati)")
    return check_eval(run, outs, shard_size=a.shard_size).render()


if __name__ == "__main__":
    sys.exit(main())
