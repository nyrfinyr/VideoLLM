"""`EntropyShortcutStrategy` — proof-of-concept signal-driven.

Salta `generate()` del tutto: usa `pred_letter` dalla softmax ristretta
alle lettere MCQ candidate (via `SupportsSignals.generate_with_signals`,
`models/signals.py`) come risposta diretta — nessun compute oltre al
forward di prefill che il modello fa comunque (vedi
`docs/entropia_confidenza_mcq.md` per la validazione sperimentale del
segnale). Serve a validare end-to-end il contratto `SupportsSignals`, non
a decidere la strategy "giusta" — quella resta lavoro futuro (es. una
strategy multi-pass che ri-campiona i frame in base all'entropia/
concentrazione dell'attenzione della prima passata).

Richiede un modello che implementa `SupportsSignals` (i preset `*_attn`:
`qwen2_5_vl_3b_attn`, `qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`) — fallisce esplicitamente
altrimenti, stesso pattern di fail-fast già usato in `evals/video_mme.py`
per `use_subtitles` senza campo `subtitle`.
"""
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from models.media import Text
from models.signals import SupportsSignals

from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM


class EntropyShortcutStrategy(Strategy):
    name = "entropy_shortcut"

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
        if not isinstance(vlm, SupportsSignals):
            raise RuntimeError(
                f"strategy {self.name!r} richiede un modello che implementa "
                f"SupportsSignals (segnali di confidenza/attenzione), ma "
                f"{type(vlm).__name__} non lo fa — usa uno dei preset con cattura "
                "(qwen2_5_vl_3b_attn, qwen3_vl_2b_attn, qwen3_vl_4b_attn) "
                "o un'altra strategy."
            )
        media, tmp_dir = self._build_media(video_path, frames, video_start, video_end, budget)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        try:
            signal = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)
        finally:
            # Non-None solo nel regime frame-doubling — vedi `Strategy._build_media`.
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if self.capture_dir is not None and frames is None and signal.visual_attention is not None:
            self._save_attention_capture(vlm, video_path, prompt, signal.visual_attention, budget, video_start, video_end)

        result: dict = {
            "raw": signal.text,
            "answer_entropy": signal.answer_entropy,
            "answer_probs": signal.answer_probs,
        }
        if options is not None:
            result["pred"] = ord(signal.pred_letter) - ord("A") if signal.pred_letter else None
        return result
