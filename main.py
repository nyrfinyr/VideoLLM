import asyncio
import logging
import os
from typing import cast

import hydra
import torch
import weave
from omegaconf import DictConfig, OmegaConf

from evals import Dataset, mcq_accuracy
from utils.obs import init_observability, log_eval_summary
from utils.samples import prepare_samples

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
    from models import Qwen25VL3B, Qwen3VL2B, Qwen3VL4B, Qwen25VLAttention
    from models.base import BaseVLM

    _MODELS: dict[str, type[BaseVLM]] = {
        "qwen25_vl_3b": Qwen25VL3B,
        "qwen3_vl_2b": Qwen3VL2B,
        "qwen3_vl_4b": Qwen3VL4B,
        "qwen25_vl_3b_attn": Qwen25VLAttention,
    }

    # Ramo di debug: estrazione attenzione entity→token visivi su UN sample
    # (niente eval Weave, niente wandb/weave). Vedi utils/debug_attn.py.
    debug_entity: str | None = cfg.get("debug_entity")

    if not debug_entity:
        init_observability(cfg)

    model_cfg: dict = cast(dict, OmegaConf.to_container(cfg.model, resolve=True))
    vlm_cls = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm: BaseVLM = vlm_cls(**model_cfg)
    gen_cfg: GenerationConfig = GenerationConfig(**cast(dict, OmegaConf.to_container(cfg.generation, resolve=True)))

    dataset: Dataset = Dataset.get(cfg.dataset.name)
    samples: list[dict] = prepare_samples(dataset.loader(cfg.dataset), cfg)
    logger.info("Loaded %d samples from %s (%s)", len(samples), dataset.name, cfg.dataset.root)

    if debug_entity:
        from utils.debug_attn import run_entity_attention_debug, select_debug_sample

        sample = select_debug_sample(samples, cfg.get("debug_video_idx"))
        run_entity_attention_debug(vlm, sample, cfg.dataset, debug_entity)
        return

    weave_dataset = weave.Dataset(name=dataset.name, rows=weave.Table(samples))
    evaluation = weave.Evaluation(
        name=f"{dataset.name}-{cfg.model.name}",
        dataset=weave_dataset,
        scorers=[mcq_accuracy],
    )
    summary = asyncio.run(evaluation.evaluate(dataset.predict_factory(vlm, gen_cfg, cfg.dataset)))
    logger.info("Eval summary: %s", summary)
    log_eval_summary(summary, len(samples))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
