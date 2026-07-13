"""ABC delle strategy di inferenza + `SamplingBudget` condiviso.

Una `Strategy` incapsula COME interrogare il modello per rispondere a una
domanda MCQ (o open-ended): selezione/costruzione dei frame, numero di
forward pass, uso o meno dei segnali di confidenza/attenzione del
modello. È il livello estratto dalla glue duplicata identica in ogni
`Dataset.predict_factory` (`Video(...) → vlm.build_messages → vlm.generate
→ parse_mcq_letter`), reso pluggable e ORTOGONALE sia al dataset (riceve
solo `video_path`/`prompt`/`options`/eventuali `frames` già normalizzati
dal dataset, non conosce lo schema Weave-specifico delle righe) sia al
modello (riceve un `BaseVLM` generico; le strategy che richiedono segnali
extra — entropia, attenzione — verificano a runtime `isinstance(vlm,
SupportsSignals)`, vedi `models/signals.py`).

Il dispatch da nome stringa (`cfg.strategy.name`) alla classe segue lo
stesso idioma di `evals.base.Dataset`: `Strategy.get(name)` scansiona
`__subclasses__()`, quindi una sottoclasse è visibile solo se il suo
modulo è stato importato almeno una volta — `strategies/__init__.py` se
ne occupa.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from models.media import MediaItem, Video, VideoFrames

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import GenerationConfig

    from models.base import BaseVLM
    from utils.attn_core import VisualAttention


@dataclass(frozen=True)
class SamplingBudget:
    """Budget di campionamento visivo per UNA chiamata a `Strategy.answer`.

    Costruito dal dataset a partire dalla PROPRIA config (`cfg.nframes`/
    `max_pixels`/`min_pixels` — restano tuning per-dataset, es. 64 frame
    per le clip brevi di EgoSchema contro 768 per i video long-form di
    LVBench). La strategy lo tratta come un tetto di spesa, non un valore
    fisso: `uniform` lo consuma per intero in un solo pass, una strategy
    multi-pass futura potrebbe spenderlo diversamente fra i pass.
    """
    nframes: int
    max_pixels: int
    min_pixels: int | None = None


class Strategy(ABC):
    """ABC per le strategy di inferenza.

    Le sottoclassi pinnano `name` (chiave del preset sotto `strategy:` in
    `conf/config.yaml`, matcha il suo campo `name:` e il dispatch di
    `main.py`) e implementano `answer`.
    `Strategy.get(name, cfg)` instanzia al volo (stesso pattern di
    `evals.base.Dataset.get`), passando `cfg.strategy` (l'intero blocco,
    `name` incluso) al costruttore — le strategy senza knob (`uniform`,
    `entropy_shortcut`) ignorano `cfg` ereditando l'`__init__` di base; le
    strategy con soglie/parametri (es. `entropy_attention_resample`)
    overridano `__init__` e leggono da `cfg.get(...)` con default sensati.
    """

    name: str  # override in subclass

    # Impostato da `main.run` DOPO `Strategy.get(...)` (non nel costruttore:
    # dipende da `run_dir`, calcolato da `snapshot_config` nello stesso punto),
    # solo se il preset della strategy ha `capture_attention: true` — vedi
    # `strategies/entropy_shortcut.py`/`entropy_attention_resample.py` per chi
    # lo consuma. `None` = capture disattivato, comportamento invariato.
    capture_dir: Path | None = None

    def __init__(self, cfg: dict | None = None) -> None:
        """No-op di default. Override nelle sottoclassi con knob di config."""

    @abstractmethod
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
        """Risponde a una domanda su un video, ritorna il dict Weave-loggato.

        Args:
            vlm: modello selezionato (nome/taglia ortogonali alla strategy).
            video_path: path del video sorgente.
            prompt: testo della domanda già renderizzato (MCQ + eventuali
                sottotitoli — costruito dal dataset, non dalla strategy).
            options: opzioni MCQ (`None` per dataset open-ended, es.
                Causal2Needles — in quel caso la chiave `"pred"` NON va
                inclusa nel dict ritornato, per parità col comportamento
                preesistente dello scorer `mcq_accuracy`).
            budget: vedi `SamplingBudget`.
            video_start/video_end: finestra di trim (secondi), se nota
                (es. GT `time_reference` di LVBench con
                `use_time_reference=true`). `None` = video intero.
            frames: lista di path frame GIÀ pre-estratti (es. MVBench
                `data_type="frame"`, task `episodic_reasoning`) — se
                presente, la selezione frame non è negoziabile: la
                strategy deve comunque usarli così come sono (via
                `_build_media`), restando libera solo su COME interrogare
                il modello.

        Returns:
            Dict con almeno `"raw"` (str | None) e — se `options is not
            None` — `"pred"` (indice 0-based in `options`, o `None` se non
            estraibile). Campi extra (es. `answer_entropy`) finiscono nel
            dict ritornato da `predict()`, che resta `@weave.op`: Weave li
            traccia automaticamente, nessun cambio allo scorer necessario.
        """

    @classmethod
    def get(cls, name: str, cfg: dict | None = None) -> Strategy:
        """Risolve una sottoclasse registrata per `name` (mirror di
        `evals.base.Dataset.get`), passandole `cfg` (tipicamente
        `cfg.strategy` da `main.py`)."""
        for sub in cls.__subclasses__():
            if getattr(sub, "name", None) == name:
                return sub(cfg)
        valid = sorted(s.name for s in cls.__subclasses__() if hasattr(s, "name"))
        raise ValueError(f"strategy sconosciuta: {name!r}. Valide: {valid}")

    @staticmethod
    def _build_media(
        video_path: str,
        frames: list[str] | None,
        video_start: float | None,
        video_end: float | None,
        budget: SamplingBudget,
    ) -> MediaItem:
        """Helper condiviso: `VideoFrames` se `frames` è già fissato dal
        dataset (nessuna libertà di selezione), altrimenti `Video` con
        trim opzionale, campionato secondo `budget`.
        """
        if frames is not None:
            return VideoFrames(frames, max_pixels=budget.max_pixels, min_pixels=budget.min_pixels)
        return Video(
            video_path,
            nframes=budget.nframes,
            max_pixels=budget.max_pixels,
            min_pixels=budget.min_pixels,
            video_start=video_start,
            video_end=video_end,
        )

    def _save_attention_capture(
        self,
        vlm: BaseVLM,
        video_path: str,
        prompt: str,
        visual_attention: VisualAttention,
        budget: SamplingBudget,
        video_start: float | None,
        video_end: float | None,
        extra: dict | None = None,
    ) -> None:
        """Scrive un `capture.pt`/`frames/` (formato di `attn_explorer.
        capture`/`serve`) sotto `self.capture_dir` a partire da un
        `VisualAttention` già calcolato dal forward di `generate_with_signals`
        — nessun forward aggiuntivo. Chiamato dalle strategy signal-driven
        quando il loro preset ha `capture_attention: true` (vedi
        `entropy_shortcut.py`/`entropy_attention_resample.py` per la policy
        di QUANDO chiamarlo). `video_start`/`video_end` sono la finestra
        EFFETTIVAMENTE vista dal pass che ha prodotto `visual_attention`
        (non necessariamente quella della risposta finale, es. pre-resample
        in `entropy_attention_resample`). `extra` finisce nell'entry di
        `index.json` (es. `resampled`/`resample_kind`).
        """
        from utils.attn_core import build_capture_meta, capture_index_entry, reconstruct_frame_indices, write_capture, write_index

        frame_indices, fps = reconstruct_frame_indices(
            video_path, budget.nframes, video_start=video_start, video_end=video_end,
        )
        capture_meta = build_capture_meta(
            model_id=vlm.model_id,
            dtype=str(vlm.model.dtype),
            nframes=budget.nframes,
            max_pixels=budget.max_pixels,
            min_pixels=budget.min_pixels,
            extra={"strategy": self.name, **(extra or {})},
        )
        cap = write_capture(
            self.capture_dir, video_path, prompt, visual_attention, frame_indices, fps,
            double=False, capture_meta=capture_meta,
        )
        write_index(self.capture_dir, [capture_index_entry(cap, extra=extra)])
