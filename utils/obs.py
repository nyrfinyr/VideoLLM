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


def log_eval_summary(summary: dict | None, n_samples: int) -> None:
    """Flush in wandb.summary i numeri chiave dell'eval.

    Senza questo le metriche vivono solo nel sub-tab Weave del run →
    invisibili nella run table di wandb e nei summary di gruppo.
    `mcq_accuracy` può essere None se TUTTI i sample sono falliti (es. OOM
    su ogni esempio) — in quel caso logga solo n_samples per documentare
    il fallimento. No-op se wandb non è attivo (mode=disabled).
    """
    import wandb

    if wandb.run is None:
        return
    mcq = (summary or {}).get("mcq_accuracy") or {}
    correct = mcq.get("correct") or {}
    wandb.run.summary["n_samples"] = n_samples
    if "true_fraction" in correct:
        wandb.run.summary["mcq_accuracy"] = correct["true_fraction"]
        wandb.run.summary["n_correct"] = correct.get("true_count")
    latency = (summary or {}).get("model_latency") or {}
    if "mean" in latency:
        wandb.run.summary["model_latency_mean"] = latency["mean"]
