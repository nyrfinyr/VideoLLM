"""Observability: wandb + weave initialization for the project.

Single source of truth so `main.py` and ad-hoc scripts (e.g.
`fetch/prefetch_egoschema.py`) share identical setup:

- `wandb.init(...)` captures stdout/stderr — i.e. every Python `logging`
  line that hits Hydra's default StreamHandler ends up in the W&B run's
  "Logs" tab. The full resolved Hydra config is dumped to the run's
  config panel.
- `weave.init(...)` enables structured `@weave.op` tracing in the same
  W&B project (entity/project shared with wandb).

`cfg.wandb.mode='disabled'` short-circuits both, so offline debugging
doesn't need to touch the network or auth.
"""
import logging

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def init_observability(cfg: DictConfig, *, with_weave: bool = True) -> None:
    """Initialize wandb (+ optionally weave) for this run.

    Args:
        cfg: Composed Hydra config; must contain a `wandb` section
            (see `conf/wandb/default.yaml`).
        with_weave: If True (default), also `weave.init(...)` so
            `@weave.op` decorators ship traces to the same project.
    """
    wcfg = cfg.wandb
    if wcfg.mode == "disabled":
        logger.info("wandb mode=disabled — skipping wandb/weave init")
        return

    import wandb

    # Keep wandb's run dir alongside Hydra's per-run output dir so
    # outputs/<date>/<time>/ contains main.log, .hydra/ and wandb/ together.
    try:
        run_dir = HydraConfig.get().runtime.output_dir
    except ValueError:
        run_dir = None  # called outside @hydra.main

    wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        mode=wcfg.mode,
        tags=list(wcfg.tags) if wcfg.tags else None,
        name=wcfg.name,
        notes=wcfg.notes,
        group=wcfg.group,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=run_dir,
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
