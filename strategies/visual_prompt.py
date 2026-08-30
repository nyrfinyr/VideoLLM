"""`VisualPromptStrategy` — visual prompting: la parola `IMPORTANT` DIPINTA
sui pixel dei frame della cella di picco (solo Qwen3-VL, variante "C").

Terza variante della famiglia signal-driven: stesso segnale (attenzione
domanda→visivo del pass 1), stessa scelta delle celle, stesso gate e stesso
costo strutturale (estrazione PNG + secondo pass) di `attention_marker`
(variante "B") e `attention_highlight` (variante "A") — cambia SOLO il canale
su cui l'intervento viene consegnato: nei pixel, non nei token. In A e B il
modello deve *risolvere* un riferimento (secondi, o una posizione nella
sequenza) verso il contenuto visivo; qui il segnale è CO-LOCATO con
l'evidenza, non c'è nessun riferimento da risolvere. È ciò che LoT fa nella
variante immagine disegnando la bbox sull'immagine (`lot/prompts.py:147`).

Perché NON c'è split di blocchi: dipingere pixel non tocca la struttura dei
token — un solo blocco video, struttura identica alla baseline. Tutte le
invarianti M-RoPE/tagli-pari del piano marcatori qui non si applicano
(`docs/piano_visual_prompting.md` §2); `nframes` dispari è ammesso.

Perché ENTRAMBI i frame della cella: il patch embedding è 3D
(`temporal_patch_size=2`), due frame consecutivi si fondono in una riga di
token. Dipingerne uno solo consegnerebbe la parola "a metà intensità",
mescolata al frame pulito — un intervento più debole e non riproducibile.

L'occlusione della banda (~18% dei pixel, PROPRIO sul frame con l'evidenza) è
il costo dell'intervento, non un bug: il controllo `cell_select=random` lo
paga identico (stessa parola, stessa banda, stesso numero di frame, cella a
sorte), quindi la differenza `attention − random` isola DOVE si dipinge. Il
confronto pulito è INTERNO al probe; quello contro `baseline_24` risponde a
un'altra domanda (l'arm nel complesso guadagna o perde?) e si legge dopo.

L'istruzione che spiega la parola è OBBLIGATORIA e costante di modulo
(`OVERLAY_INSTRUCTION`): senza, la scritta è un dato visivo qualunque.
Conseguenza sul controllo: l'arm è "parola dipinta + istruzione", quindi il
controllo onesto è la STESSA istruzione con cella a caso, non l'assenza di
istruzione (lezione di `docs/260803-tabelle_risultati.md` §1).

⚠️ GATE di leggibilità (`scripts/probe_overlay_legibility.py`): se il modello
non LEGGE la parola dopo lo smart_resize, questo arm non misura visual
prompting ma occlusione — nessuna accuracy va letta prima di quel gate
(docs/piano_visual_prompting.md §8.1).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from models.media import Text, VideoFrames
from models.qwen import Qwen3VL
from models.signals import SupportsSignals
from utils.attn_core import QUERY_ROWS_CHOICES, RankedCell, ranked_cells_from_attention, reconstruct_frame_indices, resolve_query_rows
from utils.mcq import parse_mcq_letter
from utils.overlay import OVERLAY_TEXT, paint_important

from .attention_highlight import _sample_rng
from .attention_marker import _extract_all_frames
from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    import random

    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer

logger = logging.getLogger(__name__)

# Istruzione nel prompt, OBBLIGATORIA (vedi docstring del modulo). Costante di
# modulo e non knob di config: è il testo dell'intervento, cambiarla cambia
# l'esperimento. L'ultima frase ricalca lot/prompts.py:142 ("Do not output
# the markers"): senza, la parola rischia di finire nella risposta e sporcare
# il parsing MCQ.
OVERLAY_INSTRUCTION = (
    f"Some frames have the word {OVERLAY_TEXT} written on them: those frames "
    "contain the visual evidence relevant to the question. Do not mention the "
    "word in your answer."
)


class VisualPromptStrategy(Strategy):
    name = "visual_prompt"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        self.always_highlight = bool(cfg.get("always_highlight", True))
        self.top_k = max(1, int(cfg.get("top_k", 1)))
        self.cell_select = str(cfg.get("cell_select", "attention"))
        self.random_seed = int(cfg.get("random_seed", 0))
        self.query_rows = str(cfg.get("query_rows", "question"))
        self.sink_filter = bool(cfg.get("sink_filter", False))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))
        # Geometria = knob (si tara sulla probe di leggibilità); la PAROLA e
        # l'ISTRUZIONE no, sono costanti di modulo.
        self.overlay_position = str(cfg.get("overlay_position", "top"))
        self.overlay_height_frac = float(cfg.get("overlay_height_frac", 0.18))

        if self.cell_select not in ("attention", "random"):
            raise ValueError(
                f"cell_select sconosciuto: {self.cell_select!r}. "
                "Valide: 'attention', 'random'"
            )
        if self.query_rows not in QUERY_ROWS_CHOICES:
            raise ValueError(
                f"query_rows sconosciuto: {self.query_rows!r}. "
                f"Valide: {sorted(QUERY_ROWS_CHOICES)}"
            )
        if self.overlay_position not in ("top", "bottom"):
            raise ValueError(
                f"overlay_position sconosciuta: {self.overlay_position!r}. "
                "Valide: 'top', 'bottom'"
            )
        if self.sink_filter and self.sink_percentile <= 0:
            # Stessa trappola di attention_marker: percentile=0 NON spegne il
            # filtro (la soglia diventa il massimo e l'argmax viene azzerato
            # comunque). Per l'attenzione grezza: sink_filter=false.
            raise ValueError(
                "sink_percentile <= 0 con sink_filter=true: NON è il modo di "
                "spegnere il filtro. Per l'attenzione grezza usa "
                "sink_filter=false."
            )

    def _select_cells(self, ranked: list[RankedCell], *, t: int, rng: random.Random) -> list[int] | None:
        """Le `top_k` celle da dipingere, o `None` se non c'è segnale.

        Identico ad `attention_marker._select_cells`: con `attention` ci si
        astiene a massa totale nulla; con `random` no (la cella non dipende
        dall'attenzione, e far saltare il controllo dove salta l'arm vero ne
        restringerebbe il campione senza motivo).
        """
        if self.cell_select == "random":
            return rng.sample(range(t), k=min(self.top_k, t))
        if all(c.pct <= 0 for c in ranked):
            return None
        return [c.cell for c in ranked[: self.top_k]]

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
                f"SupportsSignals, ma {type(vlm).__name__} non lo fa — usa "
                "qwen3_vl_2b_attn o qwen3_vl_4b_attn."
            )
        if not isinstance(vlm, Qwen3VL):
            raise RuntimeError(
                f"strategy {self.name!r} è definita solo su Qwen3-VL, ma il "
                f"modello è {type(vlm).__name__}. Il limite NON è tecnico "
                "(dipingere pixel non tocca i position_ids): è che (a) il "
                "confronto testa-a-testa con attention_marker è definito lì, "
                "e (b) i preset Qwen2.5-VL girano con pass_video_metadata="
                "false, regime in cui _prepare_inputs RIFIUTA un VideoFrames "
                "con frames_indices (models/qwen.py). Estenderlo richiede "
                "prima decidere cosa fare di second_per_grid_ts/sample_fps."
            )
        if frames is not None:
            raise RuntimeError(
                f"strategy {self.name!r} richiede di ricostruire gli indici "
                "frame dal video sorgente: con `frames` pre-estratti dal "
                "dataset non ci sono timestamp reali da dare al blocco."
            )
        if budget.double_frames:
            raise RuntimeError(
                f"strategy {self.name!r} assume il regime merge-a-coppie "
                "(cella = 2 frame): con double_frames=true la mappatura "
                "cella→frame cambia — fuori scope."
            )

        tmp_dirs: list[Path] = []
        # --- pass 1: identico ad attention_marker/attention_highlight ---
        media, media_tmp_dir = self._build_media(video_path, None, video_start, video_end, budget)
        if media_tmp_dir is not None:
            tmp_dirs.append(media_tmp_dir)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)
        va = signal.visual_attention

        # ⚠️ Asserzione righe-query (identica ad attention_marker):
        # `question_rows` ha un fallback silenzioso (utils/attn_core.py:75) —
        # senza "options:" ritorna TUTTE le righe e l'arm degraderebbe a
        # query_rows=all credendo di misurare `question`. Su Video-MME
        # format_mcq_prompt scrive sempre "Options:", quindi è un'asserzione,
        # non un ramo atteso: si fallisce il sample in modo RUMOROSO.
        n_query_rows = n_query_tokens = None
        rows_res = None
        if va is not None:
            n_query_tokens = len(va.query_tokens)
            # Risoluzione UNICA del knob (utils.attn_core.resolve_query_rows):
            # per `entity` chiama il decoder già caricato e può degradare a
            # `question` (entity_mapped=False nei metadati); per gli altri
            # valori è il selettore puro di sempre.
            rows_res = resolve_query_rows(
                self.query_rows, va.query_tokens,
                extract_entities=getattr(vlm, "extract_entities", None),
            )
            n_query_rows = len(rows_res.rows) if rows_res.rows else n_query_tokens
            if self.query_rows != "all" and n_query_rows >= n_query_tokens:
                raise RuntimeError(
                    f"query_rows={self.query_rows!r} non ha tagliato nulla "
                    f"(n_query_rows={n_query_rows} == n_query_tokens="
                    f"{n_query_tokens}): il selettore è degradato al "
                    "fallback 'tutte le righe' e si starebbe misurando "
                    "l'esperimento sbagliato. Prompt senza marcatore "
                    "'options:'?"
                )

        top1_pct = peak_cell = t_cells = peak_cell_sink_filtered = None
        ranked_cells: list[RankedCell] | None = None
        if va is not None:
            ranked_cells = ranked_cells_from_attention(
                va, percentile=self.sink_percentile, border=self.sink_border,
                rows=rows_res.rows, sink_filter=self.sink_filter,
            )
            top1_pct, peak_cell = ranked_cells[0].pct, ranked_cells[0].cell
            t_cells = va.t
            # Check §8.3 del piano: se la cella col filtro acceso coincidesse
            # SEMPRE con `peak_cell`, il knob sink_filter non arriva a
            # `ranked_cells_from_attention`.
            if not self.sink_filter:
                peak_cell_sink_filtered = ranked_cells_from_attention(
                    va, percentile=self.sink_percentile, border=self.sink_border,
                    rows=rows_res.rows, sink_filter=True,
                )[0].cell
            else:
                peak_cell_sink_filtered = peak_cell

        gate_open = self.always_highlight or (
            signal.answer_entropy is not None and signal.answer_entropy > self.entropy_threshold
        )

        painted = False
        painted_cells: list[int] | None = None
        painted_frames: list[int] | None = None
        paint_skip_reason: str | None = None
        overlay_font: str | None = None
        overlay_text_px: int | None = None
        n_frames_sent: int | None = None
        media2: VideoFrames | None = None

        if ranked_cells is not None and gate_open:
            cells = self._select_cells(
                ranked_cells, t=t_cells, rng=_sample_rng(self.random_seed, video_path, prompt),
            )
            if cells is None:
                paint_skip_reason = "no_signal"
            else:
                # Frame REALI: `RankedCell.frames/.rep_frame/.t_sec` sono
                # dummy e NON vanno usati. La cella `ti` copre le posizioni
                # `2*ti`/`2*ti+1` della lista campionata da
                # `reconstruct_frame_indices`.
                frame_indices, fps = reconstruct_frame_indices(
                    video_path, budget.nframes, video_start=video_start, video_end=video_end,
                )
                paths, paint_tmp_dir = _extract_all_frames(video_path, frame_indices)
                tmp_dirs.append(paint_tmp_dir)

                # ENTRAMBI i frame della cella (docstring). Il clamp serve
                # solo con nframes dispari, dove l'ultima cella ha un frame.
                positions = sorted({p for ti in cells for p in (2 * ti, 2 * ti + 1) if p < len(paths)})
                for p in positions:
                    info = paint_important(
                        paths[p], paths[p],
                        position=self.overlay_position, height_frac=self.overlay_height_frac,
                    )
                    overlay_font, overlay_text_px = info["font"], info["text_px"]

                # UN SOLO blocco video: nessuno split, struttura identica alla
                # baseline. I metadata REALI servono comunque — senza,
                # fetch_video fabbrica frames_indices=range(n) e fps finto, e
                # i timestamp <x.x seconds> del prompt non sarebbero più
                # quelli del pass 1 (secondo cambiamento non voluto).
                media2 = VideoFrames(
                    paths,
                    max_pixels=budget.max_pixels,
                    min_pixels=budget.min_pixels,
                    frames_indices=[int(i) for i in frame_indices],
                    fps=fps,
                )
                painted = True
                painted_cells = sorted(cells)
                painted_frames = positions
                n_frames_sent = len(paths)

            if self.capture_dir is not None and painted:
                self._save_attention_capture(
                    vlm, video_path, prompt, va, budget, video_start, video_end,
                    extra={
                        "painted": painted, "painted_cells": painted_cells,
                        "cell_select": self.cell_select, "sink_filter": self.sink_filter,
                        "top1_pct": top1_pct, "peak_cell": peak_cell, "t_cells": t_cells,
                    },
                )

        try:
            if media2 is not None:
                # `build_messages` delega a `build_messages_from_parts`, quindi
                # il canale metadata privato passa anche di qui: nessuna API
                # nuova da usare (models/qwen.py:156-174).
                messages = vlm.build_messages(media2, Text(f"{OVERLAY_INSTRUCTION}\n\n{prompt}"))
            else:
                # Gate chiuso o nessun segnale: pass 2 sugli stessi frame del
                # pass 1, prompt INVARIATO (niente istruzione senza pittura:
                # sarebbe un terzo intervento non controllato).
                messages = vlm.build_messages(media, Text(prompt))
            raw = vlm.generate(messages, generation_config=gen_cfg)
        finally:
            for tmp_dir in tmp_dirs:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        result: dict = {
            "raw": raw,
            "answer_entropy": signal.answer_entropy,
            "top1_pct": top1_pct,
            "peak_cell": peak_cell,
            # Cella che sceglierebbe il filtro sink acceso — confrontata con
            # `peak_cell` dà il check §8.3 (knob che arriva davvero).
            "peak_cell_sink_filtered": peak_cell_sink_filtered,
            "t_cells": t_cells,
            "painted": painted,
            "paint_skip_reason": paint_skip_reason,
            "painted_cells": painted_cells,
            "n_painted_cells": len(painted_cells) if painted_cells is not None else None,
            # Check §8.4: sempre ⊂ [0, nframes), cardinalità 2*top_k (salvo
            # ultima cella con nframes dispari); n_frames_sent == nframes.
            "painted_frames": painted_frames,
            "n_frames_sent": n_frames_sent,
            # Check §8.4: `overlay_font` costante e mai "pillow-default" se
            # sul nodo c'è DejaVu.
            "overlay_font": overlay_font,
            "overlay_text_px": overlay_text_px,
            # Check §8.2: n_query_rows < n_query_tokens su OGNI sample con
            # query_rows != all (altrimenti `answer` ha già sollevato).
            "n_query_rows": n_query_rows,
            "n_query_tokens": n_query_tokens,
            # Metadati della modalità `entity` (None negli altri casi):
            # `query_rows_mode` è la selezione EFFETTIVA (= "question" quando
            # l'entity non mappa verbatim), `entity_mapped` conta il tasso di
            # mapping riuscito per-sample.
            "query_rows_mode": rows_res.mode if rows_res is not None else None,
            "entity_raw": rows_res.entity_raw if rows_res is not None else None,
            "entity_phrases": list(rows_res.entity_phrases) if rows_res is not None and rows_res.entity_phrases is not None else None,
            "entity_mapped": rows_res.entity_mapped if rows_res is not None else None,
            "cell_select": self.cell_select,
            "sink_filter": self.sink_filter,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            pred_fallback = False
            if pred is None and signal.pred_letter is not None:
                # Stesso fallback degli altri arm: se il pass 2 non produce una
                # lettera parseabile si usa l'argmax della softmax ristretta
                # del pass 1 invece di perdere il sample.
                pred = letters.index(signal.pred_letter)
                pred_fallback = True
            result["pred"] = pred
            result["pred_fallback"] = pred_fallback
            # Risposta del pass 1 (PRIMA della pittura): abilita il breakdown
            # `correct_pass1_*` di evals/base.py — "l'overlay ha rotto un
            # sample già corretto?" si legge intra-sample, senza join fra run.
            result["pred_pass1"] = (
                letters.index(signal.pred_letter) if signal.pred_letter is not None else None
            )
        return result
