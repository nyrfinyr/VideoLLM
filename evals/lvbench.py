"""LVBench loader (classe `LVBench`).

Legge il dump prodotto da `fetch/prefetch_lvbench.py`:

    <root>/
        metadata.jsonl     # una riga per QA, schema upstream:
                           #   {id, key, type, video, uid, question,
                           #    answer, question_type, time_reference}
                           # `question` contiene le option inline come
                           # "<stem>\n(A) ...\n(B) ...\n(C) ...\n(D) ..."
                           # `answer` è lettera "A"-"D"
                           # `question_type` è list[str] (multi-label,
                           # tipicamente 1-3 elementi)
                           # `time_reference` è "MM:SS-MM:SS" o
                           # "HH:MM:SS-HH:MM:SS"
        videos/<video_path>   # mp4 flat (es. <youtube_id>.mp4), estratti
                              # dai chunk di lmms-lab/LVBench
        dropped.jsonl      # forensics dei video selezionati ma non
                           # estratti dai chunk scaricati (non letto qui)

`LVBench.loader` parsea question/options dal testo embedded, converte
time_reference in `video_start`/`video_end` (secondi), letter→int.

`LVBench.predict_factory` di default NON usa `time_reference`: i
benchmark long-video valutano la capacità del modello di FROVARE il
segmento rilevante, e passare GT start/end è cheating. Override con
`cfg.use_time_reference=true` per fare eval "open-book" (utile per
debug / oracle baseline).

Scorer: `evals.base.mcq_accuracy` (condiviso). Breakdown per
`question_type` in UI Weave via la colonna.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import weave
from omegaconf import DictConfig

from .base import Dataset, format_mcq_prompt, parse_mcq_letter

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM

logger = logging.getLogger(__name__)


# Match di una riga "(X) testo dell'option" con X ∈ A-D. MULTILINE
# perché le option sono su righe separate dentro al campo `question`.
_OPTION_LINE_RE = re.compile(r"^\(([A-D])\)\s+(.+?)\s*$", re.MULTILINE)


def _parse_question_and_options(text: str) -> tuple[str, list[str]]:
    """Estrae stem + 4 option da una `question` LVBench.

    Aspetta 4 righe "(A) ... (B) ... (C) ... (D) ..." in ordine, con il
    testo della domanda sopra. Solleva ValueError se la struttura non
    matcha — il chiamante decide se skippare o crashare.
    """
    matches = list(_OPTION_LINE_RE.finditer(text))
    if len(matches) != 4:
        raise ValueError(
            f"attese 4 option (A-D), trovate {len(matches)} in: {text!r}"
        )
    letters = [m.group(1) for m in matches]
    if letters != ["A", "B", "C", "D"]:
        raise ValueError(f"option fuori ordine: {letters} in: {text!r}")
    stem = text[: matches[0].start()].rstrip()
    options = [m.group(2).strip() for m in matches]
    return stem, options


def _parse_timestamp(s: str) -> float:
    """`MM:SS` o `HH:MM:SS` → secondi (float)."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError(f"timestamp non riconosciuto: {s!r}")


def _parse_time_reference(tr: str) -> tuple[float, float]:
    """`"MM:SS-MM:SS"` o `"HH:MM:SS-HH:MM:SS"` → `(start, end)` in sec."""
    if "-" not in tr:
        raise ValueError(f"time_reference senza '-': {tr!r}")
    start_s, end_s = tr.split("-", 1)
    return _parse_timestamp(start_s), _parse_timestamp(end_s)


class LVBench(Dataset):
    name = "lvbench"

    def loader(self, cfg: DictConfig) -> list[dict]:
        """Carica LVBench in righe normalizzate per Weave.

        Schema di output:
            id:             str           # = "<youtube_id>/<uid>"
            video_path:     str
            question:       str           # solo lo stem, senza option
            options:        list[str]     # 4 stringhe (A..D ordinate)
            answer:         int           # indice in `options`
            question_type:  list[str]     # multi-label (UI breakdown)
            video_start:    float | None  # secondi, da time_reference
            video_end:      float | None  # secondi, da time_reference

        Skip + log + counter su:
        - `question` non parseabile (struttura option non attesa)
        - `answer` lettera fuori range A..D
        - `time_reference` malformato (mantiene la riga ma con start/end=None)
        """
        root = Path(cfg.root)
        metadata_path = root / "metadata.jsonl"
        with metadata_path.open() as f:
            rows = [json.loads(line) for line in f]
        logger.info("LVBench: %d righe raw da %s", len(rows), metadata_path)

        skipped_bad_question = 0
        skipped_bad_answer = 0
        malformed_time_ref = 0
        out: list[dict] = []
        for r in rows:
            try:
                stem, options = _parse_question_and_options(r["question"])
            except ValueError as e:
                logger.warning("LVBench: id=%s question non parseabile: %s", r.get("id"), e)
                skipped_bad_question += 1
                continue

            answer_idx = parse_mcq_letter(r["answer"], len(options))
            if answer_idx is None:
                logger.warning(
                    "LVBench: id=%s answer fuori range: %r",
                    r.get("id"), r["answer"],
                )
                skipped_bad_answer += 1
                continue

            video_start: float | None = None
            video_end: float | None = None
            tr = r.get("time_reference")
            if tr:
                try:
                    video_start, video_end = _parse_time_reference(tr)
                except ValueError as e:
                    logger.warning("LVBench: id=%s time_reference malformato: %s", r.get("id"), e)
                    malformed_time_ref += 1

            out.append({
                "id": r["id"],
                "video_path": str(root / r["video"]),
                "question": stem,
                "options": options,
                "answer": answer_idx,
                "question_type": r.get("question_type", []),
                "video_start": video_start,
                "video_end": video_end,
            })

        if skipped_bad_question:
            logger.warning("LVBench: scartate %d righe con question non parseabile", skipped_bad_question)
        if skipped_bad_answer:
            logger.warning("LVBench: scartate %d righe con answer fuori range", skipped_bad_answer)
        if malformed_time_ref:
            logger.warning("LVBench: %d righe con time_reference malformato (start/end=None)", malformed_time_ref)
        logger.info("LVBench: %d sample finali", len(out))
        return out

    def predict_factory(
        self,
        vlm: BaseVLM,
        gen_cfg: GenerationConfig,
        cfg: DictConfig,
    ) -> Callable:
        """Costruisce la closure `@weave.op predict(...)`.

        Default: video INTERO (no trim), il modello deve trovare il
        segmento rilevante — è la metodologia ufficiale di LVBench.

        `cfg.use_time_reference=true`: trim a `[video_start, video_end]`
        via qwen-vl-utils. Eval "open-book" / oracle baseline — utile per
        misurare l'upper bound se il modello sapesse già dove guardare.
        """
        from models import Text, Video

        nframes = cfg.nframes
        use_tr = bool(cfg.get("use_time_reference", False))

        if use_tr:
            @weave.op
            def predict(
                video_path: str,
                question: str,
                options: list[str],
                video_start: float | None,
                video_end: float | None,
            ) -> dict:
                prompt = format_mcq_prompt(question, options)
                media = Video(
                    video_path,
                    nframes=nframes,
                    video_start=video_start,
                    video_end=video_end,
                )
                messages = vlm.build_messages(media, Text(prompt))
                raw = vlm.generate(messages, generation_config=gen_cfg)
                return {"raw": raw, "pred": parse_mcq_letter(raw, len(options))}
        else:
            @weave.op
            def predict(video_path: str, question: str, options: list[str]) -> dict:
                prompt = format_mcq_prompt(question, options)
                media = Video(video_path, nframes=nframes)
                messages = vlm.build_messages(media, Text(prompt))
                raw = vlm.generate(messages, generation_config=gen_cfg)
                return {"raw": raw, "pred": parse_mcq_letter(raw, len(options))}

        return predict
