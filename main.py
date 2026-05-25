import asyncio
import logging
import os
from typing import cast

import hydra
import torch
import weave
from omegaconf import DictConfig, OmegaConf

from evals import Dataset, mcq_accuracy
from utils.obs import init_observability

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def run(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    os.environ.setdefault("WEAVE_PARALLELISM", "1")

    from transformers import GenerationConfig
    from models import Qwen25VL3B, Qwen3VL2B, Qwen3VL4B
    from models.base import BaseVLM

    _MODELS: dict[str, type[BaseVLM]] = {
        "qwen25_vl_3b": Qwen25VL3B,
        "qwen3_vl_2b": Qwen3VL2B,
        "qwen3_vl_4b": Qwen3VL4B,
    }

    init_observability(cfg)

    model_cfg: dict = cast(dict, OmegaConf.to_container(cfg.model, resolve=True))
    vlm_cls = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm: BaseVLM = vlm_cls(**model_cfg)
    gen_cfg: GenerationConfig = GenerationConfig(**cast(dict, OmegaConf.to_container(cfg.generation, resolve=True)))

    dataset: Dataset = Dataset.get(cfg.run.dataset)
    samples: list[dict] = dataset.loader(cfg.run)
    logger.info("Loaded %d samples from %s (%s)", len(samples), dataset.name, cfg.run.root)

    weave_dataset = weave.Dataset(name=dataset.name, rows=weave.Table(samples))
    evaluation = weave.Evaluation(
        name=f"{dataset.name}-{cfg.model.name}",
        dataset=weave_dataset,
        scorers=[mcq_accuracy],
    )
    summary = asyncio.run(evaluation.evaluate(dataset.predict_factory(vlm, gen_cfg, cfg.run)))
    logger.info("Eval summary: %s", summary)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
