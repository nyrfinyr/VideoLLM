"""Observability: wandb + weave initialization for the project.

Single source of truth so `main.py` and ad-hoc scripts (e.g.
`fetch/prefetch_egoschema.py`) share identical setup:

- `wandb.init(...)` captures stdout/stderr — ogni riga di `logging` che
  passa dallo `StreamHandler` configurato in `main.py` finisce nel tab
  "Logs" della run W&B. Il config risolto per intero è scritto nel
  pannello config della run.
- `weave.init(...)` enables structured `@weave.op` tracing in the same
  W&B project (entity/project shared with wandb).

`cfg.wandb.mode='disabled'` short-circuits both, so offline debugging
doesn't need to touch the network or auth.
"""
import logging

logger = logging.getLogger(__name__)


def init_observability(cfg, *, with_weave: bool = True) -> None:
    """Initialize wandb (+ optionally weave) for this run.

    Args:
        cfg: config risolto (`utils.config.Cfg`); deve contenere una
            sezione `wandb` (vedi il preset `wandb.default` in `conf/config.yaml`).
        with_weave: If True (default), also `weave.init(...)` so
            `@weave.op` decorators ship traces to the same project.
    """
    wcfg = cfg.wandb
    if wcfg.mode == "disabled":
        logger.info("wandb mode=disabled — skipping wandb/weave init")
        return

    import wandb

    wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        mode=wcfg.mode,
        tags=list(wcfg.tags) if wcfg.tags else None,
        name=wcfg.name,
        notes=wcfg.notes,
        group=wcfg.group,
        config=dict(cfg),
    )
    if wandb.run is not None:
        logger.info("wandb run: %s", wandb.run.url)

    if with_weave:
        import weave
        weave.init(f"{wcfg.entity}/{wcfg.project}")


def breakdown_rows(summary: dict | None) -> list[tuple[str, int, int, float]]:
    """Estrae dal summary Weave le righe `(categoria, n_correct, n_sample,
    accuracy)` prodotte dal breakdown di `evals.base.mcq_accuracy`.

    Cerca le coppie `correct_<slug>` / `seen_<slug>` (vedi lo scorer per
    come sono costruite). `seen_*` senza `correct_*` non può succedere —
    li emette lo stesso sample — ma se succedesse la riga verrebbe saltata
    invece di rompere il logging di fine eval.
    """
    mcq = (summary or {}).get("mcq_accuracy") or {}
    rows = []
    for key, stats in mcq.items():
        if not key.startswith("seen_") or not isinstance(stats, dict):
            continue
        slug = key[len("seen_"):]
        got = mcq.get(f"correct_{slug}")
        if not isinstance(got, dict):
            continue
        n = stats.get("true_count") or 0
        n_correct = got.get("true_count") or 0
        rows.append((slug, n_correct, n, n_correct / n if n else 0.0))
    return sorted(rows)


def log_eval_summary(summary: dict | None, n_samples: int) -> None:
    """Flush in wandb.summary i numeri chiave dell'eval.

    Senza questo le metriche vivono solo nel sub-tab Weave del run →
    invisibili nella run table di wandb e nei summary di gruppo.
    `mcq_accuracy` può essere None se TUTTI i sample sono falliti (es. OOM
    su ogni esempio) — in quel caso logga solo n_samples per documentare
    il fallimento. No-op se wandb non è attivo (mode=disabled).

    Il breakdown per categoria (`mcq_accuracy_duration_long`,
    `n_samples_duration_long`, ...) va sia in wandb.summary sia — come
    tabella leggibile — nel log del job, così è visibile nel `.out` SLURM
    anche senza aprire wandb. I CONTEGGI ci sono accanto alle frazioni
    perché sono l'unica forma che si può sommare fra shard di un array.
    """
    import wandb

    rows = breakdown_rows(summary)
    if rows:
        logger.info(
            "Breakdown per categoria:\n%s",
            "\n".join(f"  {slug:32s} {c:5d}/{n:<5d} = {acc:.4f}" for slug, c, n, acc in rows),
        )

    if wandb.run is None:
        return
    mcq = (summary or {}).get("mcq_accuracy") or {}
    correct = mcq.get("correct") or {}
    wandb.run.summary["n_samples"] = n_samples
    if "true_fraction" in correct:
        wandb.run.summary["mcq_accuracy"] = correct["true_fraction"]
        wandb.run.summary["n_correct"] = correct.get("true_count")
    for slug, n_correct, n, acc in rows:
        wandb.run.summary[f"mcq_accuracy_{slug}"] = acc
        wandb.run.summary[f"n_correct_{slug}"] = n_correct
        wandb.run.summary[f"n_samples_{slug}"] = n
    latency = (summary or {}).get("model_latency") or {}
    if "mean" in latency:
        wandb.run.summary["model_latency_mean"] = latency["mean"]
