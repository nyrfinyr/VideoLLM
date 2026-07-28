"""MVBench loader (classe `MVBench`).

Legge il dump prodotto da `fetch/prefetch_mvbench.py`:

    <root>/
        metadata.jsonl     # una riga per QA, schema normalizzato:
                           #   {id, task_type, video, data_type,
                           #    question, options, answer, answer_idx,
                           #    [start, end]}
        videos/<source>/...   # mp4 (data_type=video) o sequenze jpg
                              #   (data_type=frame, solo episodic_reasoning)

`MVBench.loader` rimappa al contratto Weave (`video_path, data_type,
question, options, answer, task_type, video_start, video_end`);
`MVBench.predict_factory` discrimina `Video` vs `VideoFrames` in base a
`data_type`.

Scorer: `evals.base.mcq_accuracy` (condiviso). Il breakdown per
`task_type` in UI Weave funziona perché `task_type` è una colonna del
dataset, non un arg dello scorer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import weave

from .base import Dataset, format_mcq_prompt

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from strategies.base import Strategy
    from utils.config import Cfg

logger = logging.getLogger(__name__)


# Estensioni considerate quando `data_type=frame` (Episodic Reasoning su
# TVQA): qwen-vl-utils accetta una lista di path frame come `video`.
# In MVBench-tvqa i frame sono jpg numerati progressivamente
# (es. `00001.jpg`, ...): `sorted` su nome basta.
_FRAME_EXTS = (".jpg", ".jpeg", ".png")


def _list_frames(folder: Path) -> list[str]:
    """Lista i frame jpg/png in `folder`, ordinati per nome."""
    out: list[Path] = []
    for ext in _FRAME_EXTS:
        out.extend(folder.glob(f"*{ext}"))
    if not out:
        raise FileNotFoundError(
            f"frame folder {folder} vuota (atteso .jpg/.png numerati)."
        )
    return [str(p) for p in sorted(out)]


class MVBench(Dataset):
    name = "mvbench"

    def loader(self, cfg: Cfg) -> list[dict]:
        """Carica MVBench in righe normalizzate per Weave.

        Legge `cfg.root` (assoluto) e `cfg.tasks` (lista task da
        includere, o `null` = tutti).

        Schema di output:
            id:           str    # "<task_type>/<idx>"
            video_path:   str    # mp4 (video) o cartella (frame)
            data_type:    str    # "video" | "frame"
            question:     str
            options:      list[str]
            answer:       int    # indice in `options`
            task_type:    str    # per breakdown UI Weave
            video_start:  float | None
            video_end:    float | None

        Filtri:
        - `cfg.tasks` non null → tieni solo righe in quel set; warning
          per task richiesti ma non presenti (typo guard).
        - `answer_idx is None` → skip + log + counter (dato dataset
          sporco flaggato dal prefetch; non penalizza il modello).
        """
        root = Path(cfg.root)
        metadata_path = root / "metadata.jsonl"
        with metadata_path.open() as f:
            rows = [json.loads(line) for line in f]
        logger.info("MVBench: %d righe raw da %s", len(rows), metadata_path)

        if cfg.tasks is not None:
            wanted = set(cfg.tasks)
            present = {r["task_type"] for r in rows}
            missing = wanted - present
            if missing:
                logger.warning("MVBench: task richiesti ma assenti in metadata: %s", sorted(missing))
            before = len(rows)
            rows = [r for r in rows if r["task_type"] in wanted]
            logger.info("MVBench: filtro tasks=%s: %d → %d righe", sorted(wanted), before, len(rows))

        skipped_no_answer = 0
        out: list[dict] = []
        for r in rows:
            if r.get("answer_idx") is None:
                skipped_no_answer += 1
                continue
            out.append({
                "id": r["id"],
                "video_path": str(root / r["video"]),
                "data_type": r["data_type"],
                "question": r["question"],
                "options": r["options"],
                "answer": int(r["answer_idx"]),
                "task_type": r["task_type"],
                "video_start": r.get("start"),
                "video_end": r.get("end"),
            })
        if skipped_no_answer:
            logger.warning(
                "MVBench: scartate %d righe con answer_idx=None (answer ∉ options nel dataset)",
                skipped_no_answer,
            )
        logger.info("MVBench: %d sample finali", len(out))
        return out

    def predict_factory(
        self,
        vlm: BaseVLM,
        strategy: Strategy,
        gen_cfg: GenerationConfig,
        cfg: Cfg,
    ) -> Callable:
        """Discrimina per `data_type`:

        - `video`  → frame campionati da `strategy` secondo `budget`, con
                     trim `[video_start, video_end]`.
        - `frame`  → frame GIÀ pre-estratti (`_list_frames`), passati alla
                     strategy come `frames=` (nessuna libertà di
                     selezione — vedi `Strategy._build_media`).
        """
        from strategies.base import SamplingBudget

        budget = SamplingBudget(
            nframes=cfg.nframes, max_pixels=cfg.max_pixels, min_pixels=cfg.get("min_pixels"),
            double_frames=cfg.get("double_frames", False),
        )

        @weave.op
        def predict(
            video_path: str,
            data_type: str,
            question: str,
            options: list[str],
            video_start: float | None,
            video_end: float | None,
        ) -> dict:
            prompt = format_mcq_prompt(question, options)
            frames = _list_frames(Path(video_path)) if data_type == "frame" else None
            return strategy.answer(
                vlm, video_path=video_path, prompt=prompt, options=options, gen_cfg=gen_cfg,
                budget=budget, video_start=video_start, video_end=video_end, frames=frames,
            )

        return predict
