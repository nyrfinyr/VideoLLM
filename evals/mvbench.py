"""MVBench loader (skeleton).

Sorgente: `OpenGVLab/MVBench` su HuggingFace. 20 task MCQ; lo split di
ogni task è un config separato del dataset, e i video stanno in tar
shards che vanno materializzati su disco con `fetch/prefetch_mvbench.py`.

Schema sorgente (per task), per riferimento:
    task_type:  str
    video:      str               # nome file relativo all'archivio del task
    question:   str
    candidates: list[str]
    answer:     str               # stringa testuale della candidate corretta,
                                  # NON una lettera — va matchata contro
                                  # `candidates` per ricavare l'indice.

Per il `predict` MCQ riusa `evals.egoschema.make_predict`: la firma
(`video_path`, `question`, `options`) è la stessa.
"""

from __future__ import annotations

import logging

import weave
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_mvbench(cfg: DictConfig) -> list[dict]:
    """Carica MVBench in righe normalizzate per Weave.

    Schema di output (chiavi matchate da `predict` / scorer):
        id:         str
        video_path: str
        question:   str
        options:    list[str]
        answer:     int           # indice in `options`
        task_type:  str           # per breakdown per categoria
    """
    raise NotImplementedError("TODO: implementare loader MVBench")


@weave.op
def mvbench_accuracy(answer: int, task_type: str, output: dict) -> dict:
    """Accuracy MCQ; porta avanti `task_type` per consentire breakdown
    per categoria nella UI Weave (aggregazione lato server)."""
    raise NotImplementedError("TODO: implementare scorer MVBench")
