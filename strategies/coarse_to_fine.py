"""Ricerca temporale coarse-to-fine: zoom iterativo guidato da attenzione ed
entropia, con backtracking sul ranking accumulato.

## Perché questa strategy

Gli arm di MARCATURA (`attention_highlight`, `attention_marker`,
`visual_prompt`) sono chiusi: l'oracolo su LVBench mostra che indicare al
modello la cella **vera** vale 30% contro 32% di baseline
(`docs/oracolo_lvbench.md`). Nello stesso esperimento, RICAMPIONARE i frame
dentro la finestra vera vale **48% contro 32%**: il guadagno vive nel *dove
guardare*, non nel *come dirlo*.

Il collo di bottiglia è quindi la LOCALIZZAZIONE, e il campionamento uniforme
scala malissimo: su LVBench (durata mediana 71 min, finestra d'evidenza
mediana 30 s) 24 frame coprono la finestra nel 45% dei sample, e **raddoppiare
il budget porta solo al 51%** — per il 90% servirebbero ~128 frame. Aggiungere
frame uniformi (anche sfasati) è la strada sbagliata: compra copertura
linearmente nel budget.

Lo zoom invece guadagna **risoluzione moltiplicativa**: 24 frame su un video
di 71 min hanno spacing 177 s; gli stessi 24 dentro UNA cella (1/12 del video)
hanno spacing 15 s, e una finestra da 30 s diventa quasi certamente colpita.
Da qui il coarse-to-fine.

## Il ciclo

    passo 0   nframes frame uniformi sul video intero
              → risposta + entropia + ranking celle
              gate: entropia <= soglia → ACCETTA e rispondi
    passo k   scendi nella regione più promettente del ranking ACCUMULATO,
              ricampiona nframes frame LÌ DENTRO (spacing / t), ricalcola
              risposta, entropia e ranking sulla nuova vista
              gate: entropia <= soglia → ACCETTA
    fine      esaurito `max_steps`: si risponde con la vista a ENTROPIA
              MINIMA fra tutte quelle viste (non l'ultima: scendere non è
              monotono, e la vista più confidente è la scelta naturale
              quando il gate non si è mai chiuso)

**Backtracking.** Le regioni candidate stanno in una lista unica ordinata per
punteggio, alimentata da OGNI passo. Se il passo k scende nella cella
sbagliata, la sua entropia non scende: al passo k+1 la regione migliore
rimasta è tipicamente la seconda cella del passo precedente, cioè si risale
di un livello invece di affondare nell'errore. È il motivo per cui i punteggi
vanno accumulati e non sostituiti.

**Normalizzazione dei punteggi.** I `pct` di forward diversi NON sono
confrontabili: la softmax normalizza su tutte le celle della vista, quindi
con `t` celle il livello di caso è `100/t` e una vista più corta gonfia
meccanicamente i punteggi. Si accumula `score = pct * t` = "quante volte più
della media uniforme", che è confrontabile fra viste di lunghezza diversa
(`_lift`). Senza questa normalizzazione il ranking accumulato favorirebbe
sempre l'ultima vista.

## Costo

Un forward con cattura per passo, più uno di generazione per la vista scelta.
Con `max_steps=3` il caso peggiore è 4 forward, ma il gate chiude prima sui
sample facili: il costo medio dipende da `entropy_threshold` e va letto da
`n_steps` per sample. **Il budget per forward resta `nframes`**: non si
processano mai più frame insieme, si processano più volte finestre diverse.

## Il controllo da lanciare insieme

`region_select=random` sceglie la regione a sorte invece che dal ranking, a
parità di tutto il resto (stesso numero di passi, stesso gate, stesso
budget). È l'unico confronto che isola il contributo del SEGNALE dal
contributo del "guardare più volte in posti diversi" — la lezione dei tre arm
di marcatura, dove il controllo random pareggiava sempre il treatment.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from models.media import Text, VideoFrames
from models.signals import SupportsSignals
from utils.attn_core import (RankedCell, ranked_cells_from_attention,
                             reconstruct_frame_indices, resolve_query_rows)
from utils.mcq import parse_mcq_letter

from .attention_marker import _extract_all_frames
from .base import SamplingBudget, Strategy, video_duration_sec

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """Un intervallo temporale candidato, col punteggio che l'ha proposto.

    `score` è il LIFT sul livello uniforme (`pct * t`), confrontabile fra
    viste di lunghezza diversa — vedi la docstring del modulo. `depth` è il
    numero di zoom che hanno prodotto questa regione (0 = celle della vista
    globale): serve a fermare la discesa quando la regione è più corta di
    quanto il budget possa risolvere.
    """
    start: float
    end: float
    score: float
    depth: int

    @property
    def span(self) -> float:
        return self.end - self.start


def _lift(rc: RankedCell, t: int) -> float:
    """Punteggio confrontabile fra viste: quante volte la massa della cella
    supera il livello uniforme (`100/t`)."""
    return rc.pct * t / 100.0


def _cells_to_regions(ranked: list[RankedCell], t: int, w0: float, w1: float,
                      depth: int) -> list[Region]:
    """Celle di una vista `[w0, w1]` → regioni assolute nel video sorgente."""
    span = (w1 - w0) / t
    return [Region(start=w0 + rc.cell * span, end=w0 + (rc.cell + 1) * span,
                   score=_lift(rc, t), depth=depth)
            for rc in ranked]


class CoarseToFineStrategy(Strategy):
    name = "coarse_to_fine"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.max_steps = int(cfg.get("max_steps", 3))
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        # Ricampionare una regione più corta di così non compra risoluzione
        # utile: sotto questa durata (secondi) la discesa si ferma anche se
        # il gate è ancora aperto. Default 30 s = finestra d'evidenza mediana
        # di LVBench: scendere più fine è raffinare dentro il segnale.
        self.min_region_sec = float(cfg.get("min_region_sec", 30.0))
        self.region_select = str(cfg.get("region_select", "attention"))
        self.random_seed = int(cfg.get("random_seed", 0))
        self.query_rows = str(cfg.get("query_rows", "question"))
        self.sink_filter = bool(cfg.get("sink_filter", False))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))
        if self.max_steps < 1:
            raise ValueError(f"max_steps deve essere >= 1, non {self.max_steps}")
        if self.region_select not in ("attention", "random"):
            raise ValueError(
                f"region_select sconosciuto: {self.region_select!r}. "
                "Valide: 'attention' (arm), 'random' (controllo appaiato)"
            )

    def _zoom_media(self, video_path: str, region: Region, budget: SamplingBudget,
                    tmp_dirs: list[Path]) -> VideoFrames:
        """`nframes` frame dentro `region`, con metadata REALI.

        `frames_indices`/`fps` assoluti: il processor Qwen3 emette i timestamp
        veri del punto del video da cui i frame vengono, quindi il modello sa
        che sta guardando un tratto (non un video che ricomincia da zero).
        """
        idx, fps = reconstruct_frame_indices(
            video_path, budget.nframes, video_start=region.start, video_end=region.end,
        )
        idx = [int(i) for i in idx]
        paths, tmp_dir = _extract_all_frames(video_path, idx)
        tmp_dirs.append(tmp_dir)
        return VideoFrames(paths, max_pixels=budget.max_pixels,
                           min_pixels=budget.min_pixels,
                           frames_indices=idx, fps=fps)

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
                "cattura (qwen2_5_vl_3b_attn, qwen3_vl_2b_attn, "
                "qwen3_vl_4b_attn)."
            )
        if frames is not None:
            raise RuntimeError(
                f"strategy {self.name!r} ricampiona il video e non può girare "
                "su un dataset che fissa i frame (`frames`): serve il path del "
                "video sorgente."
            )

        import random
        rng = random.Random(f"{self.random_seed}:{video_path}:{prompt}".__hash__())
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        tmp_dirs: list[Path] = []

        duration = video_duration_sec(video_path)
        w0 = video_start if video_start is not None else 0.0
        w1 = video_end if video_end is not None else duration
        # Regioni candidate, riordinate a ogni passo: è la lista che rende
        # possibile il backtracking (vedi docstring).
        pool: list[Region] = []
        # Una vista per passo: (entropia, media, signal, regione).
        views: list[tuple[float, object, SignalAnswer, Region | None]] = []
        steps: list[dict] = []

        try:
            media, media_tmp = self._build_media(video_path, None, video_start, video_end, budget)
            if media_tmp is not None:
                tmp_dirs.append(media_tmp)
            region: Region | None = None
            accepted_at: int | None = None
            # L'estrazione delle righe-query dipende SOLO dal prompt testuale,
            # identico a ogni passo (cambiano i frame, non la domanda): con
            # `query_rows=entity` la si risolve una volta e la si riusa, invece
            # di pagare `max_steps` generazioni identiche per sample.
            rows_res = None

            for step in range(self.max_steps):
                signal: SignalAnswer = vlm.generate_with_signals(
                    media, Text(prompt), gen_cfg, answer_letters=letters,
                )
                ent = signal.answer_entropy
                va = signal.visual_attention
                # Entropia mancante (nessuna lettera candidata): la vista resta
                # utilizzabile ma non può chiudere il gate — si ordina in fondo.
                views.append((ent if ent is not None else float("inf"), media, signal, region))

                n_cells = view_rows = None
                if va is not None:
                    if rows_res is None:
                        rows_res = resolve_query_rows(
                            self.query_rows, va.query_tokens,
                            extract_entities=getattr(vlm, "extract_entities", None),
                        )
                    res = rows_res
                    view_rows = len(res.rows) if res.rows else len(va.query_tokens)
                    ranked = ranked_cells_from_attention(
                        va, rows=res.rows, sink_filter=self.sink_filter,
                        percentile=self.sink_percentile, border=self.sink_border,
                    )
                    n_cells = va.t
                    depth = 0 if region is None else region.depth + 1
                    lo, hi = (w0, w1) if region is None else (region.start, region.end)
                    pool.extend(_cells_to_regions(ranked, va.t, lo, hi, depth))

                steps.append({
                    "step": step,
                    "entropy": ent,
                    "region": None if region is None else [round(region.start, 1), round(region.end, 1)],
                    "n_cells": n_cells,
                    "n_query_rows": view_rows,
                })

                if ent is not None and ent <= self.entropy_threshold:
                    accepted_at = step
                    break
                if step == self.max_steps - 1:
                    break

                # --- scelta della prossima regione ---------------------------
                # Solo regioni ancora risolvibili: sotto `min_region_sec`
                # ricampionare non compra risoluzione utile.
                cands = [r for r in pool if r.span >= self.min_region_sec]
                if not cands:
                    break
                if self.region_select == "attention":
                    region = max(cands, key=lambda r: r.score)
                else:
                    region = rng.choice(cands)
                pool.remove(region)   # niente doppie discese nella stessa regione
                media = self._zoom_media(video_path, region, budget, tmp_dirs)

            # Vista scelta: entropia minima fra quelle viste. Con gate chiuso
            # coincide con l'ultima; senza, è la più confidente — scendere non
            # è monotono e l'ultima vista può essere la peggiore.
            best_ent, best_media, best_signal, best_region = min(views, key=lambda v: v[0])
            messages = vlm.build_messages(best_media, Text(prompt))
            raw = vlm.generate(messages, generation_config=gen_cfg)
        finally:
            for d in tmp_dirs:
                shutil.rmtree(d, ignore_errors=True)

        n_steps = len(steps)
        result: dict = {
            "raw": raw,
            "answer_entropy": best_ent if best_ent != float("inf") else None,
            "entropy_first": steps[0]["entropy"],
            "n_steps": n_steps,
            "zoomed": n_steps > 1,
            "accepted_at": accepted_at,
            "gate_closed": accepted_at is not None,
            "region_select": self.region_select,
            "query_rows": self.query_rows,
            "query_rows_mode": None if rows_res is None else rows_res.mode,
            "entity_raw": None if rows_res is None else rows_res.entity_raw,
            "entity_mapped": None if rows_res is None else rows_res.entity_mapped,
            "n_query_rows": None if rows_res is None else (
                len(rows_res.rows) if rows_res.rows else None),
            "sink_filter": self.sink_filter,
            "final_region": None if best_region is None
                            else [round(best_region.start, 1), round(best_region.end, 1)],
            "final_depth": 0 if best_region is None else best_region.depth + 1,
            "final_span_sec": None if best_region is None else round(best_region.span, 1),
            "video_duration_sec": round(duration, 1),
            "steps": steps,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            pred_fallback = False
            if pred is None and best_signal.pred_letter is not None:
                pred = letters.index(best_signal.pred_letter)
                pred_fallback = True
            result["pred"] = pred
            result["pred_fallback"] = pred_fallback
        return result
