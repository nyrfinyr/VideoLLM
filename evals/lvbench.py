"""LVBench loader (skeleton).

Sorgente: HuggingFace (verifica il nome esatto del repo: candidati
plausibili sono `THUDM/LVBench` e simili — controllare prima di
cablare la config). Long-video benchmark, MCQ 4-options; i video sono
tipicamente >1h.

Schema sorgente atteso, per riferimento:
    video:          str           # path / id del video
    question:       str
    candidates:     list[str]
    answer:         str           # lettera → indice
    question_type:  str           # categoria per breakdown
    time_reference: ...           # eventuale timestamp/intervallo di GT

Vista la lunghezza dei video, valutare `fps` molto basso lato `Video`
(es. 0.1) per non saturare la finestra di contesto del VLM.

Per il `predict` MCQ riusa `evals.egoschema.make_predict`.
"""

from __future__ import annotations

import logging

import weave
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_lvbench(cfg: DictConfig) -> list[dict]:
    """Carica LVBench in righe normalizzate per Weave.

    Schema di output:
        id:            str
        video_path:    str
        question:      str
        options:       list[str]
        answer:        int        # indice in `options`
        question_type: str        # per breakdown per categoria
    """
    raise NotImplementedError("TODO: implementare loader LVBench")


@weave.op
def lvbench_accuracy(answer: int, question_type: str, output: dict) -> dict:
    """Accuracy MCQ con `question_type` portato avanti per breakdown."""
    raise NotImplementedError("TODO: implementare scorer LVBench")
