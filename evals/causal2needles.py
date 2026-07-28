"""Causal2Needles open-ended QA loader (classe `CausalNeedles`).

Legge il dump prodotto da `fetch/prefetch_causal2needles.py`:

    <root>/
        metadata.jsonl     # chiavi parquet originali (question, video_id,
                           #   question_type, cause_idx, effect_idx,
                           #   cause_sent, effect_sent, text_answer) + `video`
        videos/{yms,symon}/<id>.mp4

A differenza degli altri dataset registrati (tutti MCQ) qui NON c'è
`options`: `text_answer` è testo libero. `predict_factory` genera testo
libero via `vlm.generate` senza alcun parsing MCQ — NON è pensato per
girare sotto lo scorer `mcq_accuracy` di `main.py` (che assume
`output["pred"]` sia una lettera). Uso primario oggi:
`attn_explorer.capture --dataset causal2needles --video-idx ...`.

Un video porta ~20-40 domande: `video_idx` qui identifica la SINGOLA riga
QA (`"<video_id>::<indice-nel-video>"`), non il video — per selezionare
un video specifico serve comunque scorrere le righe che condividono lo
stesso `video_id`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import weave

from .base import Dataset

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from strategies.base import Strategy
    from utils.config import Cfg

logger = logging.getLogger(__name__)


class CausalNeedles(Dataset):
    name = "causal2needles"

    def loader(self, cfg: Cfg) -> list[dict]:
        root = Path(cfg.root)
        with (root / "metadata.jsonl").open() as f:
            rows = [json.loads(line) for line in f]

        seen = Counter()
        out = []
        for r in rows:
            video_id = r["video_id"]
            idx_in_video = seen[video_id]
            seen[video_id] += 1
            out.append({
                "video_idx": f"{video_id}::{idx_in_video}",
                "video_id": video_id,
                "video_path": str(root / r["video"]),
                "question": r["question"],
                "question_type": r["question_type"],
                "cause_sent": r["cause_sent"],
                "effect_sent": r["effect_sent"],
                "text_answer": r["text_answer"],
            })
        return out

    def predict_factory(
        self,
        vlm: BaseVLM,
        strategy: Strategy,
        gen_cfg: GenerationConfig,
        cfg: Cfg,
    ) -> Callable:
        from strategies.base import SamplingBudget

        budget = SamplingBudget(
            nframes=cfg.nframes, max_pixels=cfg.max_pixels, min_pixels=cfg.get("min_pixels"),
            double_frames=cfg.get("double_frames", False),
        )

        @weave.op
        def predict(video_path: str, question: str) -> dict:
            # options=None: open-ended, nessuna chiave "pred" nel dict
            # ritornato (stesso comportamento di oggi — questo dataset
            # non è agganciato a mcq_accuracy).
            return strategy.answer(vlm, video_path=video_path, prompt=question, options=None, gen_cfg=gen_cfg, budget=budget)

        return predict
