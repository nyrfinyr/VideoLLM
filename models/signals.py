"""Contratto opzionale per modelli che espongono segnali di confidenza/
attenzione (entropia della risposta MCQ, mappa di attenzione) OLTRE alla
`generate()` testuale standard di `BaseVLM`.

ABC mixin (non `typing.Protocol` — nessun precedente nel repo, l'idioma
esistente per capability pluggable è `ABC`, vedi `BaseVLM(ABC)`,
`Dataset(ABC)`). Solo `Qwen25VLAttention` lo implementa oggi, avvolgendo
`full_visual_attention` (che già calcola entropia/attenzione dal forward
di prefill, zero costo aggiuntivo — vedi `models/qwen_attn.py`). Una
strategy che richiede segnali fa `isinstance(vlm, SupportsSignals)` a
runtime e fallisce esplicitamente se il modello selezionato non li
supporta — modello e strategy restano scelte ortogonali (nome/taglia del
modello non è noto alla strategy in anticipo).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .media import MediaItem, Text

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from utils.attn_core import VisualAttention


@dataclass(frozen=True)
class SignalAnswer:
    """Output di `SupportsSignals.generate_with_signals`.

    `text` è `None` quando il segnale viene ricavato SENZA una vera
    generazione autoregressiva (es. `full_visual_attention` è
    prefill-only: `pred_letter` viene dalla softmax ristretta alle
    lettere candidate sui logit dell'ultimo token del prefill, non da un
    `model.generate()`).
    """
    text: str | None
    pred_letter: str | None
    answer_entropy: float | None
    answer_probs: dict[str, float] | None
    visual_attention: "VisualAttention | None"


class SupportsSignals(ABC):
    @abstractmethod
    def generate_with_signals(
        self,
        media: MediaItem,
        text: Text,
        gen_cfg: GenerationConfig,
        *,
        answer_letters: list[str] | None = None,
    ) -> SignalAnswer:
        """Come `BaseVLM.generate`, ma ritorna anche i segnali di
        confidenza/attenzione disponibili "gratis" dal forward.

        `answer_letters`: lettere MCQ candidate (es. `["A","B","C","D"]`)
        per cui calcolare `answer_entropy`/`answer_probs`/`pred_letter`
        (softmax ristretta). `None`/`[]` per prompt non-MCQ — salta il
        calcolo, i campi risultano `None` in `SignalAnswer`.
        """
