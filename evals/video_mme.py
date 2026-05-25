"""Video-MME loader (skeleton).

Sorgente: `lmms-lab/Video-MME` su HuggingFace. MCQ 4-options, suddiviso
per durata video (`short` / `medium` / `long`); ogni durata copre più
domini e sub-categorie. I video sono mp4 allegati al repo HF.

Schema sorgente, per riferimento:
    video_id:        str
    duration:        str          # "short" | "medium" | "long"
    domain:          str
    sub_category:    str
    question_id:     str
    task_type:       str
    question:        str
    options:         list[str]    # già formato "A. ..." — togliere il prefisso
                                  # come in `egoschema._OPTION_PREFIX_RE`.
    answer:          str          # lettera A-D → indice via `ord(...) - ord('A')`.

Sottotitoli (.srt per video) sono opzionali; se servono, estendere lo
schema con `subtitle_path` e gestirli lato `predict` (concatenati al
prompt o messaggio separato).

Per il `predict` MCQ riusa `evals.egoschema.make_predict`.
"""

from __future__ import annotations

import logging

import weave
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_video_mme(cfg: DictConfig) -> list[dict]:
    """Carica Video-MME in righe normalizzate per Weave.

    Schema di output:
        id:         str
        video_path: str
        question:   str
        options:    list[str]
        answer:     int           # indice in `options`
        duration:   str           # "short" | "medium" | "long"
        task_type:  str
    """
    raise NotImplementedError("TODO: implementare loader Video-MME")


@weave.op
def video_mme_accuracy(answer: int, duration: str, output: dict) -> dict:
    """Accuracy MCQ con `duration` portato avanti per breakdown
    per fascia di lunghezza."""
    raise NotImplementedError("TODO: implementare scorer Video-MME")
