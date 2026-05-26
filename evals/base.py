"""ABC dei dataset di evaluation + helper MCQ condivisi.

Ogni sottoclasse di `Dataset` rappresenta un benchmark (EgoSchema,
MVBench, ...) e implementa due metodi:

- `loader(cfg)`: legge il dump on-disk prodotto da `fetch/prefetch_<x>.py`
  e ritorna le righe normalizzate per Weave (chiavi che matchano gli
  arg di `predict` e degli scorer).
- `predict_factory(vlm, gen_cfg, fps)`: ritorna una closure
  `@weave.op predict(...)` la cui signature matcha le colonne ritornate
  da `loader`.

Il dispatch da `cfg.dataset.name` (stringa) alla classe avviene via
`Dataset.get(name)`, che scansiona `__subclasses__()`. Per essere
visibile, una sottoclasse deve essere importata almeno una volta —
`evals/__init__.py` si occupa di farlo.

Gli helper MCQ (`format_mcq_prompt`, `parse_mcq_letter`, `mcq_accuracy`)
vivono qui perché condivisi da tutti i benchmark MCQ attualmente
cablati. Se in futuro un dataset open-ended (captioning, ...) entrerà
nel registro, può ignorarli o aggiungere uno scorer dedicato.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import weave
from omegaconf import DictConfig

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM

logger = logging.getLogger(__name__)


# `\b` su input upper()-ato: matcha A..E come token isolato — pesca la
# lettera dentro "(A)", "A.", "Answer: B" ma non dentro "Atlanta" /
# "BEFORE". 5 lettere coprono tutti i benchmark MCQ attualmente cablati
# (max 5 candidates su MVBench).
_LETTER_RE = re.compile(r"\b([A-E])\b")

# Alcuni dataset (EgoSchema, Video-MME) ospitano le option già
# pre-formattate "A. ...": questa regex le ripulisce così
# `format_mcq_prompt` può rifare il rendering in modo uniforme. La
# esponiamo qui (anziché duplicarla in ogni loader) perché è una
# convenzione ricorrente di benchmark MCQ con 4-5 opzioni.
OPTION_PREFIX_RE = re.compile(r"^[A-E]\.\s*")


def format_mcq_prompt(question: str, options: list[str]) -> str:
    """Render uniforme di una domanda MCQ in prompt VLM."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
    return (
        f"Question: {question}\n"
        f"Options:\n{body}\n"
        "Answer with the letter of the correct option only."
    )


def parse_mcq_letter(raw: str, n_options: int) -> int | None:
    """Estrae l'indice della lettera A..E dall'output del modello.

    Ritorna None se nessuna lettera valida è presente o se la lettera
    cade fuori dal range `[0, n_options)`.
    """
    m = _LETTER_RE.search(raw.upper())
    if m is None:
        return None
    idx = ord(m.group(1)) - ord("A")
    return idx if idx < n_options else None


@weave.op
def mcq_accuracy(answer: int, output: dict) -> dict:
    """Scorer Weave condiviso: confronta `output['pred']` con `answer`.

    Breakdown per categoria (task_type, duration, ...) avviene in UI
    Weave via le colonne del dataset, non serve uno scorer per benchmark.
    """
    pred = output["pred"]
    logger.info(
        "mcq_accuracy: answer=%r (%s) pred=%r (%s)",
        answer, type(answer).__name__,
        pred, type(pred).__name__,
    )
    return {"correct": pred == answer}


class Dataset(ABC):
    """ABC per i dataset di evaluation.

    Le sottoclassi pinnano `name` (chiave usata in `conf/dataset/<x>.yaml`
    sotto `name:` e nel dispatch di `main.py`) e implementano `loader`
    + `predict_factory`. Restano stateless: `Dataset.get(name)`
    instanzia al volo.
    """

    name: str  # override in subclass

    @abstractmethod
    def loader(self, cfg: DictConfig) -> list[dict]:
        """Legge il dump locale e ritorna le righe normalizzate.

        `cfg` è `cfg.dataset` di Hydra: contiene almeno `name`, `root`
        (path dump on-disk) e i field dataset-specifici (`tasks`,
        `duration`, ...).
        """

    @abstractmethod
    def predict_factory(
        self,
        vlm: BaseVLM,
        gen_cfg: GenerationConfig,
        cfg: DictConfig,
    ) -> Callable:
        """Ritorna la closure `@weave.op predict(...)` traced da Weave.

        `cfg` è `cfg.dataset` di Hydra: simmetrico a `loader`, ogni
        dataset pesca i field che gli servono (es. `nframes` ovunque,
        `use_subtitles` su Video-MME). Tenere la stessa signature in
        tutte le sottoclassi semplifica il dispatch in `main.py`.
        """

    @classmethod
    def get(cls, name: str) -> Dataset:
        """Risolve una sottoclasse registrata per `name`.

        Scansiona `Dataset.__subclasses__()`: una sottoclasse è
        registrata se la sua classe è stata importata almeno una volta
        nel processo (cfr. `evals/__init__.py`).
        """
        for sub in cls.__subclasses__():
            if getattr(sub, "name", None) == name:
                return sub()
        valid = sorted(s.name for s in cls.__subclasses__() if hasattr(s, "name"))
        raise ValueError(f"dataset sconosciuto: {name!r}. Validi: {valid}")
