import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import weave
import yaml

from evals import Dataset, mcq_accuracy
from strategies import Strategy
from utils.config import Cfg, load_config
from utils.obs import init_observability, log_eval_summary
from utils.samples import prepare_samples

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _git_commit() -> str | None:
    """Hash corto del commit corrente (`None` se non è un repo git o git manca)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def snapshot_config(cfg: Cfg) -> Path:
    """Scrive il config risolto (+ commit git) su disco per riprodurre esattamente
    questo run in futuro — indipendente da `wandb.mode` (per i run online questo
    stesso config è già nel pannello wandb, ma per `mode=disabled` è l'unica
    traccia locale). Rimpiazza il vecchio dump automatico `outputs/<data>/<ora>/`
    di Hydra, perso con la sua rimozione — vedi `conf/config.yaml`.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path("outputs") / f"{stamp}-{cfg.dataset.name}-{cfg.model.name}-{cfg.strategy.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(cfg)
    snapshot["_git_commit"] = _git_commit()
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    return path


def run(cfg: Cfg) -> None:
    logger.info("Resolved config:\n%s", yaml.safe_dump(dict(cfg), sort_keys=False))
    snapshot_path = snapshot_config(cfg)
    logger.info("Config snapshot: %s", snapshot_path)
    torch.manual_seed(cfg.seed)

    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    os.environ.setdefault("WEAVE_PARALLELISM", "1")

    from transformers import GenerationConfig
    from models import Qwen25VL3B, Qwen3VL2B, Qwen3VL4B, Qwen25VLAttention
    from models.base import BaseVLM

    _MODELS: dict[str, type[BaseVLM]] = {
        "qwen25_vl_3b": Qwen25VL3B,
        "qwen3_vl_2b": Qwen3VL2B,
        "qwen3_vl_4b": Qwen3VL4B,
        "qwen25_vl_3b_attn": Qwen25VLAttention,
    }

    init_observability(cfg)

    model_cfg: dict = dict(cfg.model)
    vlm_cls = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm: BaseVLM = vlm_cls(**model_cfg)
    gen_cfg: GenerationConfig = GenerationConfig(**dict(cfg.generation))

    dataset: Dataset = Dataset.get(cfg.dataset.name)
    strategy: Strategy = Strategy.get(cfg.strategy.name, cfg.strategy)
    if cfg.strategy.get("capture_attention"):
        strategy.capture_dir = snapshot_path.parent / "captures"
    samples: list[dict] = prepare_samples(dataset.loader(cfg.dataset), cfg)
    logger.info("Loaded %d samples from %s (%s)", len(samples), dataset.name, cfg.dataset.root)

    weave_dataset = weave.Dataset(name=dataset.name, rows=weave.Table(samples))
    evaluation = weave.Evaluation(
        name=f"{dataset.name}-{cfg.model.name}",
        dataset=weave_dataset,
        scorers=[mcq_accuracy],
    )
    summary = asyncio.run(evaluation.evaluate(dataset.predict_factory(vlm, strategy, gen_cfg, cfg.dataset)))
    logger.info("Eval summary: %s", summary)
    log_eval_summary(summary, len(samples))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)

    print_config = "--print-config" in argv
    if print_config:
        argv.remove("--print-config")

    # Alias di comodo per l'unico knob che si attiva/disattiva e basta, con lo
    # STESSO nome del flag di `attn_explorer/capture.py` (che implementa lo
    # stesso regime): `--double-frames` ≡ `dataset.double_frames=true`. Il
    # resto della CLI resta `chiave=valore` (vedi `utils.config.parse_overrides`,
    # che rifiuta i token senza `=`).
    if "--double-frames" in argv:
        argv = [a for a in argv if a != "--double-frames"]
        argv.append("dataset.double_frames=true")

    cfg = load_config(argv)
    if print_config:
        print(yaml.safe_dump(dict(cfg), sort_keys=False))
        return
    run(cfg)


if __name__ == "__main__":
    main()
