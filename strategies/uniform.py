"""`UniformStrategy` — baseline invariata: campionamento uniforme dei
frame (via `budget`), un solo forward di `generate()`, nessun uso di
segnali di confidenza/attenzione.

Estrazione pura della glue duplicata che oggi vive in ogni
`Dataset.predict_factory` (`Video(...) → vlm.build_messages →
vlm.generate → parse_mcq_letter`): comportamento bit-per-bit invariato,
diventa il default (preset `strategy.uniform` in `conf/config.yaml`).
Riferimento per confrontare le strategy signal-driven future (es.
`entropy_shortcut`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from models.media import Text
from utils.mcq import parse_mcq_letter

from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM


class UniformStrategy(Strategy):
    name = "uniform"

    def answer(
        self,
        vlm: BaseVLM,
        *,
        video_path: str,
        prompt: str,
        options: list[str] | None,
        gen_cfg: GenerationConfig,
        budget: SamplingBudget,
        video_start: float | None = None,
        video_end: float | None = None,
        frames: list[str] | None = None,
    ) -> dict:
        media = self._build_media(video_path, frames, video_start, video_end, budget)
        messages = vlm.build_messages(media, Text(prompt))
        raw = vlm.generate(messages, generation_config=gen_cfg)
        result: dict = {"raw": raw}
        if options is not None:
            result["pred"] = parse_mcq_letter(raw, options)
        return result
