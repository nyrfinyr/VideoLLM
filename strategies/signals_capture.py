"""Run di SOLI segnali: un forward di prefill con cattura per sample, nessun
intervento, nessun pass 2.

## Perché questa strategy

Gli arm fin qui misurati mescolano sempre due cose: la QUALITÀ del segnale
(l'attenzione indica il posto giusto? l'entropia sa quando il modello
sbaglia?) e l'EFFETTO dell'intervento che quel segnale guida (marcare,
ricampionare, zoomare). L'oracolo LVBench (`docs/oracolo_lvbench.md`) ha
mostrato che il collo di bottiglia è la localizzazione: prima di costruire un
altro intervento serve misurare il segnale da solo, offline, su LVBench intero.
Questa strategy produce SOLO i dati per quelle misure:

- **T1** — l'attenzione sulle celle temporali distingue le celle dentro la
  finestra d'evidenza annotata? Per ogni rowset si logga la massa per cella
  (grezza e sink-filtrata, NON rinormalizzata). Le etichette "cella dentro la
  finestra" si calcolano OFFLINE da `pair_centers_sec` + `time_reference`
  (`utils.pair_sampling.pair_cells_in_window`): la strategy non riceve e non
  usa la finestra annotata, così nessun leakage può finire nel segnale.
- **T2** — entropia della risposta vs correttezza: `answer_entropy`/
  `answer_probs`/`answer_logits` del prefill, `pred` dall'argmax.
- **T4** — i token "sink" sono davvero sink? Statistiche dei canali dei
  token visivi su tutti i layer (`models.qwen_attn.summarize_sink_stats`),
  curva di massa d'attenzione sui top-p% token per sink score, mappe sink
  medie spaziale e temporale, più un dump per-token per i primi sample.

## Campionamento a coppie

Con 512 frame uniformi su un video di 71 min i due frame di una cella distano
~17 s e la cella non corrisponde a nessun istante: la domanda "la cella è
dentro la finestra?" non ha una risposta pulita. Qui ogni cella è una COPPIA
di frame a ±`pair_gap_sec/2` attorno a un centro uniforme
(`utils.pair_sampling.pair_video_frames`): cella ↔ istante `c_i`, e il
timestamp che il processor Qwen3-VL scrive nel prompt è proprio `c_i`.

`pred` è la lettera argmax sui logit del prefill (nessuna generazione):
`raw` è quella lettera, `pred_fallback` è sempre `False`. Accuracy
confrontabile con le baseline prefill-argmax, non con quelle a generazione
libera + parse.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.media import Text
from models.signals import SupportsSignals
from utils.attn_core import ROW_SELECTORS, sink_mask
from utils.pair_sampling import pair_video_frames

from .base import SamplingBudget, Strategy, video_duration_sec

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from utils.attn_core import VisualAttention

logger = logging.getLogger(__name__)

# Percentuali di token (ordinati per sink score decrescente) su cui si misura
# la frazione di massa d'attenzione visiva. Se i sink assorbono attenzione
# spuria, pochi punti percentuali di token portano una quota sproporzionata
# della massa; con un segnale sink privo di significato la curva è ~p%.
SINK_MASS_CURVE_PCTS = (1, 2, 5, 10, 25, 50)

_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _jsonable(x):
    """Albero (tensori, tuple, float) → JSON puro, float a 6 cifre SIGNIFICATIVE.

    Significative e non decimali: la massa d'attenzione per cella di un rowset
    stretto (es. `last_token` su 256 celle) sta sull'ordine di 1e-5, e sei
    decimali la azzererebbero. NaN/inf → `None` (JSON stretto non li ammette).
    """
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    return x


class SignalsCaptureStrategy(Strategy):
    name = "signals_capture"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.pair_gap_sec = float(cfg.get("pair_gap_sec", 2.0))
        self.rowsets = [str(r) for r in cfg.get("rowsets", ["all", "question", "last_token"])]
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_topk_channels = int(cfg.get("sink_topk_channels", 10))
        dump_dir = cfg.get("sink_dump_dir")
        self.sink_dump_dir = Path(dump_dir) if dump_dir else None
        self.sink_dump_limit = int(cfg.get("sink_dump_limit", 0))
        # Contatore d'ISTANZA, quindi per processo: con N shard i dump sono
        # fino a N * sink_dump_limit. Nessuna race con `WEAVE_PARALLELISM=1`
        # (default impostato da `main.run`): i sample sono sequenziali.
        self._n_dumped = 0
        # Solo selettori puri: `entity` richiederebbe una generazione per
        # sample, e qui si vuole un forward solo.
        bad = [r for r in self.rowsets if r not in ROW_SELECTORS]
        if bad or not self.rowsets:
            raise ValueError(
                f"rowsets non validi: {bad or self.rowsets}. Valide: {sorted(ROW_SELECTORS)}"
            )

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
        if not isinstance(vlm, SupportsSignals) or not hasattr(vlm, "full_visual_attention"):
            raise RuntimeError(
                f"strategy {self.name!r} richiede un modello con cattura "
                f"(`full_visual_attention`), ma {type(vlm).__name__} non la "
                "espone — usa un preset `_attn` (es. qwen3_vl_2b_attn)."
            )
        if frames is not None:
            raise RuntimeError(
                f"strategy {self.name!r} campiona il video a coppie e non può "
                "girare su un dataset che fissa i frame (`frames`)."
            )
        if options is None:
            raise RuntimeError(f"strategy {self.name!r} richiede un MCQ (entropia della risposta).")
        if video_start is not None or video_end is not None:
            # Le coppie coprono il video INTERO: un trim (`use_time_reference`)
            # darebbe centri fuori dalla finestra vista e, peggio, userebbe la
            # finestra annotata — esattamente ciò che T1 deve escludere.
            raise RuntimeError(
                f"strategy {self.name!r} non ammette trim (video_start/video_end): "
                "lancia con dataset.use_time_reference=false."
            )
        if budget.double_frames:
            raise RuntimeError(f"strategy {self.name!r}: double_frames non ha senso con le coppie.")
        if budget.nframes % 2 != 0:
            raise ValueError(f"nframes deve essere pari (= 2 * celle), non {budget.nframes}")
        if not getattr(vlm, "fix_videoframes_resize", False):
            # Le coppie passano da `VideoFrames`: col bug di qwen-vl-utils
            # ogni cella avrebbe meno token del path `Video` allo stesso
            # max_pixels. Un'intera run a risoluzione sbagliata è peggio di un
            # errore subito.
            raise RuntimeError(
                f"strategy {self.name!r} richiede model.fix_videoframes_resize=true "
                "(vedi models/qwen.py::_fetch_videoframes)."
            )

        letters = [chr(ord("A") + i) for i in range(len(options))]
        want_dump = self.sink_dump_dir is not None and self._n_dumped < self.sink_dump_limit

        # `image_patch_size` del modello → i PNG escono già alla dimensione
        # finale (stessi pixel, ~40x meno da scrivere); `tmp_dir` è `None`
        # quando i frame arrivano dalla cache dell'ultimo video, che li tiene
        # per le domande successive sullo stesso video.
        media, tmp_dir, centers = pair_video_frames(
            video_path, budget.nframes, self.pair_gap_sec, budget.max_pixels, budget.min_pixels,
            image_patch_size=getattr(vlm, "image_patch_size", None),
        )
        try:
            va: VisualAttention = vlm.full_visual_attention(
                media, Text(prompt), answer_letters=letters,
                sink_stats=True,
                sink_stats_percentile=self.sink_percentile,
                sink_stats_topk=self.sink_topk_channels,
                sink_stats_per_token=want_dump,
            )
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if va.t != len(centers):
            # Una cella che non è una coppia rende falsi i centri di T1.
            raise RuntimeError(
                f"celle temporali {va.t} != coppie campionate {len(centers)}: "
                "il processor ha paddato o fuso i frame diversamente dall'atteso."
            )

        n_q = len(va.query_tokens)
        n_vis = va.t * va.grid_h * va.grid_w
        is_sink = sink_mask(va.sink_map, percentile=self.sink_percentile)  # [t, gh, gw]
        keep = (~is_sink).float()
        sink_order = torch.argsort(va.sink_map.flatten().float(), descending=True)

        rowsets: dict[str, dict] = {}
        heat_per_token: dict[str, torch.Tensor] = {}
        rows_by_set: dict[str, list[int]] = {}
        for rs in self.rowsets:
            rows = ROW_SELECTORS[rs](va.query_tokens)
            rows_by_set[rs] = rows
            # Media sulle righe, NON rinormalizzata: la massa visiva totale
            # (quanto il rowset guarda il video invece del testo) è essa stessa
            # un dato, e rinormalizzare la cancellerebbe.
            heat = (va.attn if not rows else va.attn[rows]).float().mean(dim=0)  # [t, gh, gw]
            total = float(heat.sum())
            flat = heat.flatten()
            curve = []
            for p in SINK_MASS_CURVE_PCTS:
                k = max(1, math.ceil(p / 100 * n_vis))
                curve.append(float(flat[sink_order[:k]].sum()) / total if total > 0 else None)
            rowsets[rs] = {
                "n_rows": len(rows) if rows else n_q,
                "cell_mass_raw": heat.sum(dim=(1, 2)),
                "cell_mass_sink_filtered": (heat * keep).sum(dim=(1, 2)),
                "visual_mass_total": total,
                "sink_mass_curve": curve,
            }
            heat_per_token[rs] = flat

        # Copia shallow: i tensori per-token vanno solo nel dump, mai nel dict
        # Weave, e il `VisualAttention` del chiamante non va toccato.
        stats = dict(va.sink_stats or {})
        per_token = stats.pop("per_token", None)

        dump_path = None
        if want_dump:
            dump_path = self._dump(
                video_path, prompt, vlm, va, media, centers, stats, per_token,
                heat_per_token, rows_by_set,
            )

        pred = letters.index(va.pred_letter) if va.pred_letter is not None else None
        result = {
            "raw": va.pred_letter,
            "pred": pred,
            "pred_fallback": False,
            "answer_entropy": va.answer_entropy,
            "answer_probs": va.answer_probs,
            "answer_logits": va.answer_logits,
            # --- geometria --------------------------------------------------
            "t_cells": va.t,
            "grid_h": va.grid_h,
            "grid_w": va.grid_w,
            "n_vis": n_vis,
            "n_query_tokens": n_q,
            "seq_len": int(va.input_ids.shape[0]),
            "video_duration_sec": video_duration_sec(video_path),
            "fps": media.fps,
            "pair_gap_sec": self.pair_gap_sec,
            "pair_centers_sec": centers,
            # --- T1 ---------------------------------------------------------
            "rowsets": rowsets,
            # --- T4 ---------------------------------------------------------
            "sink_percentile": self.sink_percentile,
            "sink_mass_curve_pcts": list(SINK_MASS_CURVE_PCTS),
            "sink_map_spatial_mean": va.sink_map.float().mean(dim=0),   # [gh, gw]
            "sink_map_temporal_mean": va.sink_map.float().mean(dim=(1, 2)),  # [t]
            "sink_stats": stats,
            "sink_dump_path": dump_path,
        }
        return _jsonable(result)

    def _dump(self, video_path, prompt, vlm, va, media, centers, stats, per_token,
              heat_per_token, rows_by_set) -> str | None:
        """`torch.save` dei tensori per-token di UN sample (analisi T4 a grana
        fine: quali token, in quali layer). Ritorna il path, o `None` se la
        scrittura fallisce — un disco pieno non deve far perdere i segnali
        del sample, che stanno comunque nel dict Weave."""
        stem = _FILENAME_RE.sub("_", Path(video_path).stem)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        path = self.sink_dump_dir / f"{stem}_{digest}.pt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "video_path": video_path,
                "prompt": prompt,
                "model_id": getattr(vlm, "model_id", None),
                "pair_centers_sec": list(centers),
                "pair_gap_sec": self.pair_gap_sec,
                "frames_indices": list(media.frames_indices),
                "fps": media.fps,
                "t": va.t, "grid_h": va.grid_h, "grid_w": va.grid_w,
                "query_tokens": [(q.row, q.index, q.token, q.text) for q in va.query_tokens],
                "answer_probs": va.answer_probs,
                "pred_letter": va.pred_letter,
                "sink_map": va.sink_map,
                "sink_percentile": self.sink_percentile,
                # Per-token in ordine di sequenza (frame-major): reshape a
                # [.., t, grid_h, grid_w] per tornare sulla griglia.
                "per_token": per_token,
                "attn_per_token": heat_per_token,  # rowset → [n_vis]
                "rowset_rows": rows_by_set,
                "sink_stats": stats,
            }, path)
        except OSError as e:
            logger.warning("dump sink fallito per %s: %s", video_path, e)
            return None
        finally:
            self._n_dumped += 1
        return str(path)
