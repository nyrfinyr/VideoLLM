"""Video-MME loader (classe `VideoMME`).

Legge il dump prodotto da `fetch/prefetch_videomme.py`:

    <root>/
        metadata.jsonl     # una riga per QA, contiene le colonne raw del
                           #   parquet `lmms-lab/Video-MME` + campo
                           #   `video` (path rel a out_dir) e — solo se
                           #   il prefetch è girato con subtitles=true —
                           #   campo `subtitle` (path rel o None).
        videos/data/<videoID>.mp4    # estensione reale può essere .MP4/.mkv
        subtitle/<videoID>.srt       # solo se prefetch ha attivato subtitles

`VideoMME.loader` filtra per `cfg.duration` ("short"/"medium"/"long"/
"all"), strippa il prefisso "A. " dalle option, converte `answer`
(lettera) in indice, e — se `cfg.use_subtitles=True` — emette anche
`subtitle_path`. La presenza del campo `subtitle` in metadata è
validata: se manca, il loader fallisce per evitare run mal configurati.

`VideoMME.predict_factory` discrimina sul flag `use_subtitles` (chiuso
sulla closure) e prepend al prompt MCQ il testo dei dialoghi `.srt`
(timestamp / numeri droppati) seguendo la convenzione standard di
lmms-eval per Video-MME.

Scorer: `evals.base.mcq_accuracy` (condiviso). Breakdown per
`duration` / `task_type` via colonne del dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import weave
from omegaconf import DictConfig

from .base import OPTION_PREFIX_RE, Dataset, format_mcq_prompt, parse_mcq_letter

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM

logger = logging.getLogger(__name__)


_VALID_DURATIONS = ("short", "medium", "long")


def _letter_to_idx(letter: str, n_options: int) -> int | None:
    """Converte 'A'/'B'/'C'/'D' in indice 0..3; None se fuori range."""
    if not letter:
        return None
    idx = ord(letter.strip().upper()) - ord("A")
    return idx if 0 <= idx < n_options else None


def _read_srt_text(path: Path) -> str:
    """Estrae solo le righe di dialogo da un file .srt.

    Droppa: linee vuote, numeri di sequenza (`123`), timestamp
    (`HH:MM:SS,mmm --> HH:MM:SS,mmm`). Mantiene il testo nell'ordine,
    una linea per battuta. Niente dipendenze esterne: il formato SRT è
    abbastanza regolare da permettere un parser ad hoc.
    """
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        out.append(line)
    return "\n".join(out)


class VideoMME(Dataset):
    name = "video_mme"

    def loader(self, cfg: DictConfig) -> list[dict]:
        """Carica Video-MME in righe normalizzate per Weave.

        Legge `cfg.root` (path assoluto al dump di `prefetch_videomme.py`),
        `cfg.duration` (filtro o "all"/null), `cfg.use_subtitles` (se
        True espone `subtitle_path` per riga, fail loudly se il prefetch
        non ha incluso il campo).

        Schema di output:
            id:             str    # = question_id (es. "001-1")
            video_path:     str
            question:       str
            options:        list[str]    # senza prefisso "A. "
            answer:         int          # indice in `options`
            duration:       str          # "short" | "medium" | "long"
            task_type:      str          # categoria semantica
            subtitle_path:  str | None   # solo se use_subtitles=True

        Filtri:
        - `cfg.duration` ∈ {"short","medium","long"} → tieni solo quella
          fascia; "all"/null → nessun filtro.
        - `answer` non in `[A..D]` → skip + log + counter.
        """
        root = Path(cfg.root)
        metadata_path = root / "metadata.jsonl"
        with metadata_path.open() as f:
            rows = [json.loads(line) for line in f]
        logger.info("Video-MME: %d righe raw da %s", len(rows), metadata_path)

        if cfg.duration and cfg.duration != "all":
            if cfg.duration not in _VALID_DURATIONS:
                raise ValueError(
                    f"Video-MME: cfg.duration={cfg.duration!r} non valido. "
                    f"Atteso uno di {_VALID_DURATIONS} o 'all'/null."
                )
            before = len(rows)
            rows = [r for r in rows if r.get("duration") == cfg.duration]
            logger.info("Video-MME: filtro duration=%s: %d → %d righe", cfg.duration, before, len(rows))

        if cfg.use_subtitles:
            # Validazione precoce: se il prefetch è girato senza
            # subtitles=true, il campo non esiste e l'eval è broken.
            # Meglio errore esplicito che None silenzioso nel predict.
            if rows and "subtitle" not in rows[0]:
                raise RuntimeError(
                    f"Video-MME: cfg.use_subtitles=True ma {metadata_path} "
                    "non ha campo 'subtitle' — rilancia prefetch_videomme "
                    "con subtitles=true."
                )

        skipped_bad_answer = 0
        skipped_no_subtitle = 0
        out: list[dict] = []
        for r in rows:
            options = [OPTION_PREFIX_RE.sub("", o) for o in r["options"]]
            answer_idx = _letter_to_idx(r["answer"], len(options))
            if answer_idx is None:
                skipped_bad_answer += 1
                continue

            sample = {
                "id": r["question_id"],
                "video_path": str(root / r["video"]),
                "question": r["question"],
                "options": options,
                "answer": answer_idx,
                "duration": r["duration"],
                "task_type": r["task_type"],
            }
            if cfg.use_subtitles:
                sub_rel = r.get("subtitle")
                if sub_rel is None:
                    # Subtitle non scaricato per questo video (video senza
                    # sottotitoli su Video-MME): skip per non degradare
                    # silenziosamente. Se vuoi includerli senza sub,
                    # rilancia con cfg.use_subtitles=False.
                    skipped_no_subtitle += 1
                    continue
                sample["subtitle_path"] = str(root / sub_rel)
            out.append(sample)

        if skipped_bad_answer:
            logger.warning(
                "Video-MME: scartate %d righe con answer fuori range",
                skipped_bad_answer,
            )
        if skipped_no_subtitle:
            logger.warning(
                "Video-MME: scartate %d righe con subtitle assente "
                "(use_subtitles=True)", skipped_no_subtitle,
            )
        logger.info("Video-MME: %d sample finali", len(out))
        return out

    def predict_factory(
        self,
        vlm: BaseVLM,
        gen_cfg: GenerationConfig,
        cfg: DictConfig,
    ) -> Callable:
        """Costruisce la closure `@weave.op predict(...)`.

        La signature di predict cambia in base a `cfg.use_subtitles`:
        - False (default): `predict(video_path, question, options)`.
        - True: `predict(video_path, question, options, subtitle_path)` —
          legge il .srt, estrae le linee di dialogo, le prepend al prompt
          MCQ con la formulazione standard "This video's subtitles are
          listed below:" (allineata a lmms-eval/videomme).
        """
        from models import Text, Video

        nframes = cfg.nframes
        max_pixels = cfg.max_pixels
        min_pixels = cfg.get("min_pixels")
        use_subtitles = bool(cfg.use_subtitles)

        if use_subtitles:
            @weave.op
            def predict(
                video_path: str,
                question: str,
                options: list[str],
                subtitle_path: str,
            ) -> dict:
                sub_text = _read_srt_text(Path(subtitle_path))
                prompt = (
                    "This video's subtitles are listed below:\n"
                    f"{sub_text}\n\n"
                    + format_mcq_prompt(question, options)
                )
                media = Video(video_path, nframes=nframes, max_pixels=max_pixels, min_pixels=min_pixels)
                messages = vlm.build_messages(media, Text(prompt))
                raw = vlm.generate(messages, generation_config=gen_cfg)
                return {"raw": raw, "pred": parse_mcq_letter(raw, options)}
        else:
            @weave.op
            def predict(video_path: str, question: str, options: list[str]) -> dict:
                prompt = format_mcq_prompt(question, options)
                media = Video(video_path, nframes=nframes, max_pixels=max_pixels, min_pixels=min_pixels)
                messages = vlm.build_messages(media, Text(prompt))
                raw = vlm.generate(messages, generation_config=gen_cfg)
                return {"raw": raw, "pred": parse_mcq_letter(raw, options)}

        return predict
