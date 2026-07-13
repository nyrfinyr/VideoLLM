"""EgoSchema MCQ loader (classe `EgoSchema`).

Legge il dump prodotto da `fetch/prefetch_egoschema.py`:

    <root>/
        metadata.jsonl     # chiavi `lmms-lab/egoschema`:
                           #   question_idx, question, video_idx, option, answer, video
        videos/<uuid>.mp4

`EgoSchema.loader` rimappa al contratto Weave (`video_path, question,
options, answer`) consumato da `EgoSchema.predict_factory` e
`mcq_accuracy`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import weave

from .base import OPTION_PREFIX_RE, Dataset, format_mcq_prompt

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from strategies.base import Strategy
    from utils.config import Cfg

logger = logging.getLogger(__name__)


class EgoSchema(Dataset):
    name = "egoschema"

    def loader(self, cfg: Cfg) -> list[dict]:
        root = Path(cfg.root)
        with (root / "metadata.jsonl").open() as f:
            rows = [json.loads(line) for line in f]
        return [
            {
                "video_idx": r["video_idx"],
                "video_path": str(root / r["video"]),
                "question": r["question"],
                "options": [OPTION_PREFIX_RE.sub("", o) for o in r["option"]],
                "answer": int(r["answer"]) if r.get("answer") is not None else None,
            }
            for r in rows
        ]

    def predict_factory(
        self,
        vlm: BaseVLM,
        strategy: Strategy,
        gen_cfg: GenerationConfig,
        cfg: Cfg,
    ) -> Callable:
        from strategies.base import SamplingBudget

        budget = SamplingBudget(nframes=cfg.nframes, max_pixels=cfg.max_pixels, min_pixels=cfg.get("min_pixels"))

        @weave.op
        def predict(video_path: str, question: str, options: list[str]) -> dict:
            prompt = format_mcq_prompt(question, options)
            return strategy.answer(vlm, video_path=video_path, prompt=prompt, options=options, gen_cfg=gen_cfg, budget=budget)

        return predict
