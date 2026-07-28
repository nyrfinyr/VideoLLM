"""ABC dei dataset di evaluation + helper MCQ condivisi.

Ogni sottoclasse di `Dataset` rappresenta un benchmark (EgoSchema,
MVBench, ...) e implementa due metodi:

- `loader(cfg)`: legge il dump on-disk prodotto da `fetch/prefetch_<x>.py`
  e ritorna le righe normalizzate per Weave (chiavi che matchano gli
  arg di `predict` e degli scorer).
- `predict_factory(vlm, strategy, gen_cfg, cfg)`: ritorna una closure
  `@weave.op predict(...)` la cui signature matcha le colonne ritornate
  da `loader`. Il corpo costruisce solo il prompt e i parametri
  dataset-specifici (frame pre-estratti, trim), poi delega a
  `strategy.answer(...)` (`strategies/base.py`) la scelta di come
  interrogare il modello — vedi quel modulo per il contratto completo.

Il dispatch da `cfg.dataset.name` (stringa) alla classe avviene via
`Dataset.get(name)`, che scansiona `__subclasses__()`. Per essere
visibile, una sottoclasse deve essere importata almeno una volta —
`evals/__init__.py` si occupa di farlo.

Gli helper MCQ condivisi vivono in parte qui (`format_mcq_prompt`,
`mcq_accuracy`) e in parte in `utils.mcq` (`parse_mcq_letter` /
`find_mcq_answer`, l'estrazione a fallback della risposta). `parse_mcq_letter`
è ri-esportato da questo modulo per retrocompatibilità dei call-site
`from .base import parse_mcq_letter`. Se in futuro un dataset open-ended
(captioning, ...) entrerà nel registro, può ignorarli o aggiungere uno
scorer dedicato.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import weave

from utils.mcq import find_mcq_answer, parse_mcq_letter  # noqa: F401

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from strategies.base import Strategy
    from utils.config import Cfg

logger = logging.getLogger(__name__)


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


# Righe "(A) ...", "(B) ..." prodotte da `format_mcq_prompt` (e dai sidecar
# `--prompt-from-json` di `attn_explorer.capture`, stesso formato). Inversa di
# `format_mcq_prompt`: da un prompt già renderizzato risale alle lettere
# disponibili, senza dover ri-passare `options` separatamente. Usata da
# `attn_explorer.capture` per sapere quali lettere cercare nei logit
# dell'ultimo token del prefill (entropia della risposta).
MCQ_OPTION_LINE_RE = re.compile(r"^\(([A-Z])\)\s", re.MULTILINE)


def extract_mcq_letters(prompt: str) -> list[str]:
    """Lettere delle opzioni MCQ elencate in `prompt` (`[]` se non è un MCQ)."""
    return MCQ_OPTION_LINE_RE.findall(prompt)


# Chiavi che, se presenti nel dict ritornato da `predict`, fanno emettere a
# `mcq_accuracy` un breakdown aggregato per il loro valore (oltre
# all'accuracy complessiva). Chi non le emette non è toccato: la chiave
# manca, `output.get` ritorna None e non viene prodotto nessuno score extra.
#
# Due sorgenti, entrambe già esistenti, nessun campo nuovo da inventare:
# - dal DATASET, ri-emesso da `predict` (`VideoMME.predict_factory` passa
#   `duration`/`task_type`);
# - dalla STRATEGY, già nel dict che `answer()` ritorna
#   (`entropy_attention_resample` mette `resampled`/`resample_kind`/
#   `pred_fallback`).
#
# `resampled` è quello che dà le due metriche del gate d'entropia:
# `seen_resampled_true` = quanti sample lo superano, `correct_resampled_true`
# = accuracy CONDIZIONATA al superamento (e `*_false` il complemento, cioè
# i sample accettati al pass 1). Su Video-MME `resampled=True` coincide con
# `answer_entropy > entropy_threshold`, perché le altre condizioni di
# `can_resample` sono sempre vere lì (frame non fissati dal dataset, MCQ,
# segnali presenti) — NON è così su dataset con `frames` pre-estratti, es.
# MVBench `episodic_reasoning`, dove `resampled=False` mescola "gate non
# superato" e "resample impossibile".
BREAKDOWN_KEYS = ("duration", "task_type", "resampled", "resample_kind", "pred_fallback")

_SLUG_RE = re.compile(r"\W+")


def _slug(value: object) -> str:
    """`"Temporal Reasoning"` → `"temporal_reasoning"`: i valori finiscono
    dentro nomi di metrica (Weave summary → wandb.run.summary), che devono
    restare chiavi piatte e senza spazi."""
    return _SLUG_RE.sub("_", str(value)).strip("_").lower()


@weave.op
def mcq_accuracy(answer: int, output: dict) -> dict:
    """Scorer Weave condiviso: confronta `output['pred']` con `answer`.

    Oltre a `correct` (accuracy complessiva) emette, per ogni chiave di
    `BREAKDOWN_KEYS` presente in `output`, una coppia di score che dà il
    breakdown AGGREGATO a fine eval senza query post-hoc:

        correct_duration_long  bool  → true_count = risposte giuste fra i
                                       long, true_fraction = accuracy sui long
        seen_duration_long     True  → true_count = quanti sample sono long

    Lo stesso vale per i campi della strategy: `seen_resampled_true` è il
    numero di sample che superano il gate d'entropia e
    `correct_resampled_true.true_fraction` la loro accuracy condizionata.

    Funziona perché `weave.flow.scorer.auto_summarize` scarta i None prima
    di aggregare (`data = [x for x in data if x is not None]`) e unisce le
    chiavi di TUTTI i sample: ogni sample emette solo le chiavi della
    PROPRIA fascia, quindi il denominatore di `true_fraction` è già la
    dimensione della fascia. `seen_*` sembra ridondante ma non lo è: senza
    di esso la numerosità si potrebbe solo ricavare per divisione, e
    dividerebbe per zero su una fascia con zero risposte giuste.

    I conteggi (non solo le frazioni) rendono anche banale il POOLING fra
    shard di un job array: si sommano `true_count` dei tre run.
    """
    pred = output["pred"]
    correct = pred == answer
    logger.info(
        "mcq_accuracy: answer=%r (%s) pred=%r (%s)",
        answer, type(answer).__name__,
        pred, type(pred).__name__,
    )
    scores: dict = {"correct": correct}
    for key in BREAKDOWN_KEYS:
        value = output.get(key)
        if value is None:
            continue
        slug = f"{key}_{_slug(value)}"
        scores[f"correct_{slug}"] = correct
        scores[f"seen_{slug}"] = True
    return scores


class Dataset(ABC):
    """ABC per i dataset di evaluation.

    Le sottoclassi pinnano `name` (chiave del preset sotto `dataset:` in
    `conf/config.yaml`, matcha il suo campo `name:` e il dispatch di
    `main.py`) e implementano `loader` + `predict_factory`. Restano
    stateless: `Dataset.get(name)` instanzia al volo.
    """

    name: str  # override in subclass

    @abstractmethod
    def loader(self, cfg: Cfg) -> list[dict]:
        """Legge il dump locale e ritorna le righe normalizzate.

        `cfg` è `cfg.dataset` (config composta da `utils.config.load_config`): contiene almeno `name`, `root`
        (path dump on-disk) e i field dataset-specifici (`tasks`,
        `duration`, ...).
        """

    @abstractmethod
    def predict_factory(
        self,
        vlm: BaseVLM,
        strategy: Strategy,
        gen_cfg: GenerationConfig,
        cfg: Cfg,
    ) -> Callable:
        """Ritorna la closure `@weave.op predict(...)` traced da Weave.

        `cfg` è `cfg.dataset` (config composta da `utils.config.load_config`): simmetrico a `loader`, ogni
        dataset pesca i field che gli servono (es. `nframes` ovunque,
        `use_subtitles` su Video-MME) per costruire `prompt` +
        `strategies.base.SamplingBudget` + eventuali `frames`/
        `video_start`/`video_end` dataset-specifici, poi delega a
        `strategy.answer(...)` — la scelta di COME interrogare il modello
        (campionamento frame, numero di forward pass, uso di segnali) non
        è più responsabilità del dataset. Tenere la stessa signature in
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
