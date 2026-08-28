"""`AttentionMarkerStrategy` — marcatori inline attorno ai frame di picco
(solo Qwen3-VL): al pass 2 il modello vede gli STESSI istanti del pass 1, ma
le celle scelte sono racchiuse fra due marcatori testuali inseriti DENTRO il
flusso visivo:

    ...<|vision_end|><START_IMPORTANT_FRAME><4.5 seconds><|vision_start|>
    [frame]<|vision_end|><END_IMPORTANT_FRAME><5.5 seconds>...

È la variante "B" descritta e scartata nel docstring di
`strategies/attention_highlight.py` (che implementa la "A", il puntatore
testuale): stesso segnale, stessa scelta delle celle, cambia solo COME
l'intervento viene consegnato al modello. È anche ciò che fa il codice
originale di Look Twice (`lot/retrieval.py:394-396`), che inserisce
`<START_IMPORTANT_IMG>`/`<END_IMPORTANT_IMG>` come due item di testo prima e
dopo l'item immagine nella content list.

Perché SOLO Qwen3-VL: su questa famiglia il video è GIÀ `t` isole visive
separate da testo (i timestamp `<x.x seconds>` per frame), quindi
`get_rope_index` (`modeling_qwen3_vl.py:1068-1070`) assorbe un marcatore in
più nel gruppo di testo esistente — nessun `position_ids` esplicito da
costruire. Su Qwen2.5-VL il testo inserito spezzerebbe il gruppo video e
servirebbe intervenire sui `position_ids`: fuori scope, non verificato.

Tre invarianti verificate offline (docs/piano_marcatori_frame_qwen3.md §2,
appendice A per ri-verificarle dopo cambi a `_prepare_inputs`):

1. spezzare il blocco video in più blocchi non cambia i token visivi (somma
   di `t` invariata), costa solo i ~6+6 token di testo dei marcatori;
2. i metadata (`frames_indices` ASSOLUTI + fps reale) vanno passati a mano
   per blocco, altrimenti l'orologio riparte da zero a ogni marcatore —
   vedi `models/media.py::VideoFrames` e `Qwen._prepare_inputs`;
3. i tagli DEVONO cadere su confini di cella (blocchi di lunghezza PARI nel
   regime `double_frames=false`): il processor padda i blocchi dispari, `t`
   cresce e TUTTI i timestamp si sfasano. L'unità marcabile è quindi la
   CELLA (2 frame), che è già la risoluzione reale del segnale.

Differenze deliberate rispetto ad `attention_highlight` (non knob, disegno
dell'arm — docs/piano_marcatori_frame_qwen3.md §1):

- `query_rows=question`: il picco è calcolato sui SOLI token della domanda
  (niente opzioni/boilerplate). Il selettore ha un fallback silenzioso
  (`_rows_before_marker` ritorna TUTTE le righe se non trova "options:"):
  qui viene trasformato in un fail-fast per sample, vedi `answer`.
- `sink_filter=false`: le celle sono classificate sull'attenzione GREZZA,
  senza azzerare i token-sink (`ranked_cells_from_attention(...,
  sink_filter=False)` → `view="all"`). ⚠️ Confondimento noto: gli arm già
  misurati (`highlight_top1_24`, `random_top1_24`, `entropy_shift_24`)
  classificano CON il filtro — un confronto con quelli mescolerebbe due
  cambiamenti. Il confronto pulito è INTERNO al probe:
  `cell_select=attention` vs `cell_select=random` a parità di filtro.
- `always_highlight=true`: nessun gate d'entropia, si interviene su tutti i
  sample.

L'istruzione che spiega i marcatori è OBBLIGATORIA e costante di modulo
(`MARKER_INSTRUCTION`): i marcatori sono testo ordinario, senza una riga che
li spieghi il modello non ha motivo di trattarli come un'indicazione — è il
motivo per cui LoT li accompagna sempre con un prompt dedicato
(`lot/prompts.py`). Conseguenza sul controllo: l'arm è "marcatori +
istruzione", quindi il controllo onesto è la STESSA istruzione con celle
scelte a caso (`cell_select=random`), non l'assenza di istruzione — la
lezione di `docs/260803-tabelle_risultati.md` §1.

Costo: a differenza di `attention_highlight` (che riusa gli stessi token
visivi), ogni sample marcato richiede estrazione PNG di tutti i frame +
secondo pass — e con `always_highlight=true` nessun sample ne è esente.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from models.media import Text, VideoFrames
from models.qwen import Qwen3VL
from models.signals import SupportsSignals
from utils.attn_core import ROW_SELECTORS, RankedCell, ranked_cells_from_attention, reconstruct_frame_indices
from utils.mcq import parse_mcq_letter

from .attention_highlight import _sample_rng
from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    import random

    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer

logger = logging.getLogger(__name__)


# Testo dei marcatori: letterali, NON token speciali del vocabolario Qwen —
# si tokenizzano come testo ordinario (~6 token l'uno).
MARKER_START = "<START_IMPORTANT_FRAME>"
MARKER_END = "<END_IMPORTANT_FRAME>"

# Istruzione nel prompt, OBBLIGATORIA (vedi docstring del modulo). Costante
# di modulo e non knob di config, come le `POINTER_SPAN_*` di
# `attention_highlight`: è il testo dell'intervento, cambiarla cambia
# l'esperimento, non la sua configurazione.
MARKER_INSTRUCTION = (
    f"Focus on the frames enclosed between {MARKER_START} and {MARKER_END}: "
    "that is where the visual evidence relevant to the question is."
)


def _merge_adjacent_cells(cells: list[int]) -> list[tuple[int, int]]:
    """`[3, 4, 7]` → `[(3, 4), (7, 7)]`: run di celle marcate adiacenti,
    ordinati per tempo. Celle adiacenti vanno fuse in un blocco solo con
    una coppia sola di marcatori (piano §5.4.5)."""
    runs: list[list[int]] = []
    for c in sorted(cells):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return [(a, b) for a, b in runs]


def _split_blocks(n_frames: int, runs: list[tuple[int, int]]) -> list[tuple[int, int, bool]]:
    """Blocchi `(start, end, marked)` in POSIZIONI della lista campionata.

    I tagli cadono sui confini di cella `2*ti` / `2*(ti+1)` (regime
    `double_frames=false`: la cella `ti` copre le posizioni `2*ti` e
    `2*ti+1`), quindi ogni blocco ha lunghezza PARI per costruzione —
    l'invariante §2.4 senza cui il processor padda e sfasa i timestamp.
    Blocchi vuoti (run marcato in testa o in coda) sono OMESSI, non emessi
    vuoti.
    """
    blocks: list[tuple[int, int, bool]] = []
    prev = 0
    for a, b in runs:
        start, end = 2 * a, 2 * (b + 1)
        if start > prev:
            blocks.append((prev, start, False))
        blocks.append((start, end, True))
        prev = end
    if prev < n_frames:
        blocks.append((prev, n_frames, False))
    return blocks


def _extract_all_frames(video_path: str, frame_indices: list[int]) -> tuple[list[str], Path]:
    """Estrae TUTTI i frame campionati in PNG su una tmpdir.

    Sul modello di `entropy_attention_resample._build_phase_shifted_media`,
    stesso contratto di cleanup: la tmpdir la ripulisce il CHIAMANTE dopo la
    `generate()`, e viene ripulita qui se l'estrazione solleva (altrimenti
    il chiamante non riceve mai il path da ripulire).
    """
    import decord
    from PIL import Image

    vr = decord.VideoReader(video_path)
    last = len(vr) - 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="attention_marker_"))
    try:
        paths = []
        for i, idx in enumerate(frame_indices):
            p = tmp_dir / f"frame_{i:04d}.png"
            Image.fromarray(vr[max(0, min(int(idx), last))].asnumpy()).save(p)
            paths.append(str(p))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return paths, tmp_dir


class AttentionMarkerStrategy(Strategy):
    name = "attention_marker"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        # Gate identico agli altri arm signal-driven, ma il preset lo tiene
        # SEMPRE aperto (`always_highlight: true`): per questo probe si
        # interviene su tutti i sample, nessuna esenzione.
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        self.always_highlight = bool(cfg.get("always_highlight", True))
        # Celle marcate: le `top_k` per massa (grezza, vedi `sink_filter`).
        self.top_k = max(1, int(cfg.get("top_k", 1)))
        # `random` = ARM DI CONTROLLO: stessa istruzione, stessi marcatori,
        # stesso costo, celle estratte a sorte con seed mescolato su
        # `(video_path, prompt)` (stessa `_sample_rng` di attention_highlight:
        # cella stabile fra shard e rilanci).
        self.cell_select = str(cfg.get("cell_select", "attention"))
        self.random_seed = int(cfg.get("random_seed", 0))
        # SOLO i token della domanda (default dell'arm, NON di
        # `ranked_cells_from_attention`) — vedi docstring del modulo.
        self.query_rows = str(cfg.get("query_rows", "question"))
        # Attenzione GREZZA di default (vedi docstring del modulo). Il filtro
        # può essere riacceso per ablation; `sink_percentile`/`sink_border`
        # contano solo in quel caso.
        self.sink_filter = bool(cfg.get("sink_filter", False))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))

        if self.cell_select not in ("attention", "random"):
            raise ValueError(
                f"cell_select sconosciuto: {self.cell_select!r}. "
                "Valide: 'attention', 'random'"
            )
        if self.query_rows not in ROW_SELECTORS:
            raise ValueError(
                f"query_rows sconosciuto: {self.query_rows!r}. "
                f"Valide: {sorted(ROW_SELECTORS)}"
            )
        if self.sink_filter and self.sink_percentile <= 0:
            raise ValueError(
                "sink_percentile <= 0 con sink_filter=true: NON è il modo di "
                "spegnere il filtro (la soglia diventa il massimo e l'argmax "
                "della sink_map viene azzerato comunque). Per l'attenzione "
                "grezza usa sink_filter=false."
            )

    def _select_cells(self, ranked: list[RankedCell], *, t: int, rng: random.Random) -> list[int] | None:
        """Le `top_k` celle da marcare, o `None` se non c'è segnale.

        Stessa semantica di `attention_highlight._select_cells` ristretta a
        `attention`/`random`: con `attention` ci si astiene a massa totale
        nulla (senza segnale non si inventa un intervento), con `random` no
        (la cella non dipende dall'attenzione, e far saltare il controllo
        dove salta l'arm vero ne restringerebbe il campione senza motivo).
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
                f"SupportsSignals (segnali di confidenza/attenzione), ma "
                f"{type(vlm).__name__} non lo fa — usa uno dei preset con "
                "cattura (qwen3_vl_2b_attn, qwen3_vl_4b_attn)."
            )
        if not isinstance(vlm, Qwen3VL):
            # I marcatori inline sono verificati SOLO su Qwen3-VL: su
            # Qwen2.5-VL il testo inserito spezza il gruppo video di
            # `get_rope_index` e servirebbero `position_ids` espliciti
            # (vedi docstring del modulo e di attention_highlight).
            raise RuntimeError(
                f"strategy {self.name!r} è verificata solo sulla famiglia "
                f"Qwen3-VL, ma il modello è {type(vlm).__name__} — usa "
                "qwen3_vl_2b_attn o qwen3_vl_4b_attn."
            )
        if frames is not None:
            raise RuntimeError(
                f"strategy {self.name!r} richiede di ricostruire gli indici "
                "frame dal video sorgente (`reconstruct_frame_indices`): con "
                "`frames` pre-estratti dal dataset non ci sono timestamp "
                "reali da dare ai blocchi — arm non definito qui."
            )
        if budget.double_frames:
            raise RuntimeError(
                f"strategy {self.name!r} assume il regime merge-a-coppie "
                "(cella = 2 frame, tagli su confini 2*ti/2*ti+2): con "
                "double_frames=true la mappatura cella→posizioni cambia — "
                "fuori scope."
            )
        if budget.nframes % 2 != 0:
            raise RuntimeError(
                f"strategy {self.name!r} richiede nframes PARI (i tagli "
                "devono cadere su confini di cella, §2.4 del piano): "
                f"nframes={budget.nframes}."
            )

        tmp_dirs: list[Path] = []
        # Pass 1: identico ad `attention_highlight` — Video(nframes=budget).
        media, media_tmp_dir = self._build_media(video_path, None, video_start, video_end, budget)
        if media_tmp_dir is not None:
            tmp_dirs.append(media_tmp_dir)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)

        va = signal.visual_attention

        # ⚠️ Trappola del selettore righe-query (piano §3): `question_rows`
        # ha un fallback silenzioso — se non trova "options:" nel testo
        # ricostruito ritorna TUTTE le righe e l'arm degraderebbe a
        # `query_rows=all` credendo di misurare `question`. Su Video-MME
        # `format_mcq_prompt` scrive sempre "Options:", quindi questa è
        # un'asserzione, non un ramo atteso: se scatta si fallisce il sample
        # in modo RUMOROSO (Weave lo cattura per sample), non si prosegue.
        n_query_rows = n_query_tokens = None
        if va is not None:
            n_query_tokens = len(va.query_tokens)
            rows = ROW_SELECTORS[self.query_rows](va.query_tokens)
            n_query_rows = len(rows) if rows else n_query_tokens
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
                query_rows=self.query_rows, sink_filter=self.sink_filter,
            )
            top1_pct, peak_cell = ranked_cells[0].pct, ranked_cells[0].cell
            t_cells = va.t
            # Check §6.2 del piano: per verificare che `sink_filter=false`
            # arrivi davvero al ranking si logga ANCHE la cella che
            # sceglierebbe il filtro acceso — se coincidesse sempre con
            # `peak_cell`, il knob non sta arrivando a destinazione.
            if not self.sink_filter:
                peak_cell_sink_filtered = ranked_cells_from_attention(
                    va, percentile=self.sink_percentile, border=self.sink_border,
                    query_rows=self.query_rows, sink_filter=True,
                )[0].cell
            else:
                peak_cell_sink_filtered = peak_cell

        gate_open = self.always_highlight or (
            signal.answer_entropy is not None and signal.answer_entropy > self.entropy_threshold
        )

        marked = False
        marked_cells: list[int] | None = None
        marked_skip_reason: str | None = None
        n_blocks: int | None = None
        block_sizes: list[int] | None = None
        final_parts: list[VideoFrames | Text] | None = None

        if ranked_cells is not None and gate_open:
            cells = self._select_cells(
                ranked_cells, t=t_cells, rng=_sample_rng(self.random_seed, video_path, prompt),
            )
            if cells is None:
                marked_skip_reason = "no_signal"
            else:
                # Frame REALI: `RankedCell.frames/rep_frame/t_sec` sono dummy
                # (vedi `ranked_cells_from_attention`) e NON vanno usati. La
                # cella `ti` copre le posizioni `2*ti`/`2*ti+1` della lista
                # campionata da `reconstruct_frame_indices`.
                frame_indices, fps = reconstruct_frame_indices(
                    video_path, budget.nframes, video_start=video_start, video_end=video_end,
                )
                paths, marker_tmp_dir = _extract_all_frames(video_path, frame_indices)
                tmp_dirs.append(marker_tmp_dir)

                blocks = _split_blocks(len(paths), _merge_adjacent_cells(cells))
                if any((end - start) % 2 != 0 for start, end, _ in blocks):
                    # Non deve poter succedere (`_split_blocks` taglia su
                    # confini pari e nframes è pari): se succede, un blocco
                    # dispari verrebbe paddato dal processor e TUTTI i
                    # timestamp si sfaserebbero in silenzio (§2.4).
                    raise RuntimeError(
                        f"blocco di lunghezza dispari in {blocks} — "
                        "invariante §2.4 violata, timestamp sfasati."
                    )

                parts: list[VideoFrames | Text] = []
                for start, end, is_marked in blocks:
                    if is_marked:
                        parts.append(Text(MARKER_START))
                    parts.append(VideoFrames(
                        paths[start:end],
                        max_pixels=budget.max_pixels,
                        min_pixels=budget.min_pixels,
                        frames_indices=[int(i) for i in frame_indices[start:end]],
                        fps=fps,
                    ))
                    if is_marked:
                        parts.append(Text(MARKER_END))
                parts.append(Text(f"{MARKER_INSTRUCTION}\n\n{prompt}"))

                final_parts = parts
                marked = True
                marked_cells = sorted(cells)
                n_blocks = len(blocks)
                block_sizes = [end - start for start, end, _ in blocks]

            if self.capture_dir is not None and marked:
                self._save_attention_capture(
                    vlm, video_path, prompt, va, budget, video_start, video_end,
                    extra={
                        "marked": marked,
                        "marked_cells": marked_cells,
                        "n_blocks": n_blocks,
                        "cell_select": self.cell_select,
                        "sink_filter": self.sink_filter,
                        "top1_pct": top1_pct, "peak_cell": peak_cell, "t_cells": t_cells,
                    },
                )

        try:
            if final_parts is not None:
                messages = vlm.build_messages_from_parts(final_parts)
            else:
                # Gate chiuso o nessun segnale: pass 2 sugli stessi frame del
                # pass 1, prompt invariato (niente istruzione senza
                # marcatori: sarebbe un terzo intervento non controllato).
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
            # `peak_cell` dà il check §6.2 (knob che arriva davvero).
            "peak_cell_sink_filtered": peak_cell_sink_filtered,
            "t_cells": t_cells,
            "marked": marked,
            "marked_skip_reason": marked_skip_reason,
            "marked_cells": marked_cells,
            "n_marked_cells": len(marked_cells) if marked_cells is not None else None,
            "n_blocks": n_blocks,
            # Somma(block_sizes)/2 deve fare t_cells: check §6.3 (nessun
            # blocco dispari, timestamp non sfasati).
            "block_sizes": block_sizes,
            # Check §6.1: n_query_rows < n_query_tokens su OGNI sample con
            # query_rows != all (altrimenti `answer` ha già sollevato).
            "n_query_rows": n_query_rows,
            "n_query_tokens": n_query_tokens,
            "cell_select": self.cell_select,
            "sink_filter": self.sink_filter,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            pred_fallback = False
            if pred is None and signal.pred_letter is not None:
                # Stesso fallback degli altri arm: se il pass 2 non produce
                # una lettera parseabile si usa l'argmax della softmax
                # ristretta del pass 1 invece di perdere il sample.
                pred = letters.index(signal.pred_letter)
                pred_fallback = True
            result["pred"] = pred
            result["pred_fallback"] = pred_fallback
            # Risposta del pass 1 (PRIMA dei marcatori): abilita il
            # breakdown `correct_pass1_*` di `evals/base.py::mcq_accuracy` —
            # "i marcatori hanno rotto un sample già corretto?" si legge
            # intra-sample, senza join fra run.
            result["pred_pass1"] = (
                letters.index(signal.pred_letter) if signal.pred_letter is not None else None
            )
        return result
