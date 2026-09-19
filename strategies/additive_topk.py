"""Arm ADDITIVO: i frame mirati si AGGIUNGONO alla base, non la sostituiscono.

Nasce dalla falsificazione dell'arm sostitutivo (`topk_resample`, probe 108729,
100 sample): ricampionare nelle top-k celle costa 10-13 pp contro la baseline
appaiata e non batte il controllo random, perché rimpiazzare 512 frame con 128
butta più contesto di quanto la zoomata aggiunga — un miss costa −11,9 pp, un
hit ne vale +23, e con hit@1 al 17% il conto è negativo. Qui il termine
negativo sparisce per costruzione: la base resta, i frame mirati si sommano.

IL GATE È GIÀ PASSATO. La probe oracolo (`probe_additive_oracle-110308/09`, 92
sample) ha misurato, a parità di budget totale (512 frame):

    base256 (128 celle)                     41.3%
    uniform512 (256 celle)                  41.3%   Δ  0.0 pp, +8/−8, p=1
    oracle512 (base + 256 nella finestra)   54.3%   Δ +13.0 pp, +15/−3, p=0.0075

cioè: raddoppiare i frame UNIFORMI non vale niente, spenderli DOVE sta
l'evidenza vale +13 pp. Questo arm sostituisce l'oracolo col puntatore vero.

CONDIZIONI (tutte appaiate sullo stesso sample, un solo decode del video):

    k10     i 256 frame aggiunti nelle top-10 celle del ranking
    k5      gli stessi 256 nelle top-5
    rand10  10 celle A CASO — il controllo che decide se l'arm è un arm

⚠️ `rand10` non è opzionale. La probe oracolo NON separa "grappolo denso nel
posto giusto" da "grappolo denso ovunque": se un blocco temporalmente coerente
aiutasse di per sé, il puntatore non c'entrerebbe nulla. È esattamente la
trappola in cui è morto `attention_marker` (attention ≡ random).

PERCHÉ k10 PRIMA DI k5. La dose-risposta della probe oracolo è rovesciata
rispetto all'intuizione: il guadagno è massimo dove finiscono POCHI frame
nella finestra (<50 frame: +38 pp; >150: +6 pp) e sulle finestre strette
(1 cella: +18 pp; 5+ celle: +4 pp). Non paga la densità, paga vedere evidenza
che il campionamento uniforme manca del tutto — la finestra mediana di LVBench
è 20 s su video di 73 min, lo 0,58%. Quindi conviene comprare COPERTURA:
hit@10 = 59% contro hit@5 = 42% (misurati a 128 celle sugli stessi sample), e
26 frame per cella bastano.

SPAN: la cella di VORONOI della cella puntata, `[D·i/n, D·(i+1)/n]`, con le
celle adiacenti FUSE e il budget diviso in proporzione alla durata di ogni
regione. Non si restringe (il puntatore ha la precisione di una cella: sotto
quella si taglia a caso, e nella probe "metà cella" perdeva copertura) e non
si allarga a durata fissa.

Frame aggiunti UNIFORMI dentro la regione, non a coppie: la struttura a coppie
serve a rendere le celle indirizzabili per il ranking del pass 1, e sul pass
additivo non si rilegge nessun ranking. ⚠️ Nella lista unita il merge
temporale del modello accoppia frame ADIACENTI NELLA LISTA, quindi le coppie
della base non sopravvivono: le celle del pass additivo NON sono quelle del
pass 1. Va bene finché non ci si rilegge l'attenzione sopra.

COSTO (misurato, job 110272/110308): un forward a 512 frame costa ~4 s senza
cattura — il costo per sample è la CATTURA del pass 1 e il decode dei frame,
non il prefill. Ogni condizione in più vale ~2 s, per questo sono tre.
"""
from __future__ import annotations

import hashlib
import logging
import random
import shutil
from typing import TYPE_CHECKING

from models.media import Text
from models.signals import SupportsSignals
from utils.attn_core import ROW_SELECTORS, sink_mask
from utils.mcq import parse_mcq_letter
from utils.pair_sampling import (additive_indices, cells_to_spans,
                                 frames_for_plans, pair_video_frames)

from .base import SamplingBudget, Strategy, video_duration_sec
from .topk_resample import select_cells

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from utils.attn_core import VisualAttention

logger = logging.getLogger(__name__)

# k10 per primo perché è la condizione attesa migliore (copertura), k5 come
# ablation della stessa leva, rand10 come controllo appaiato a parità di k.
DEFAULT_CONDITIONS = (
    {"name": "k10", "k": 10},
    {"name": "k5", "k": 5},
    {"name": "rand10", "k": 10, "select": "random"},
)


class AdditiveTopkStrategy(Strategy):
    name = "additive_topk"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.pass1_pair_gap_sec = float(cfg.get("pass1_pair_gap_sec", 2.0))
        self.n_added = int(cfg.get("n_added", 256))
        self.rank_rowset = str(cfg.get("rank_rowset", "all"))
        self.rank_sink_filtered = bool(cfg.get("rank_sink_filtered", False))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.log_rowsets = [str(r) for r in cfg.get("log_rowsets", ["all"])]
        self.random_seed = int(cfg.get("random_seed", 0))
        self.conditions = [dict(c) for c in (cfg.get("conditions") or DEFAULT_CONDITIONS)]

        bad = [r for r in self.log_rowsets + [self.rank_rowset] if r not in ROW_SELECTORS]
        if bad:
            raise ValueError(f"rowset non validi: {bad}. Validi: {sorted(ROW_SELECTORS)}")
        if self.n_added < 2 or self.n_added % 2:
            raise ValueError(f"n_added deve essere pari e >= 2, non {self.n_added}")
        names = [c.get("name") for c in self.conditions]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError(f"le condizioni devono avere nomi unici e non vuoti: {names}")
        for c in self.conditions:
            kind = c.setdefault("select", "attention")
            if kind not in ("attention", "random"):
                raise ValueError(f"select {kind!r} sconosciuto (condizione {c['name']!r})")
            k = int(c.get("k", 1))
            # Con più regioni che frame aggiunti una regione resterebbe a zero
            # e il nome della condizione mentirebbe sul k davvero usato.
            if k < 1 or k > self.n_added:
                raise ValueError(f"condizione {c['name']!r}: k={k} non campionabile "
                                 f"con n_added={self.n_added}")
            c["k"] = k
        if not any(c["select"] == "random" for c in self.conditions):
            logger.warning(
                "nessuna condizione con select=random: senza il controllo, un guadagno "
                "non si distingue dall'effetto di infittire una regione QUALSIASI"
            )

    # ── ranking (stessa definizione di topk_resample) ───────────────────────
    def _cell_masses(self, va: "VisualAttention") -> list[float]:
        rows = ROW_SELECTORS[self.rank_rowset](va.query_tokens)
        heat = (va.attn if not rows else va.attn[rows]).float().mean(dim=0)
        if self.rank_sink_filtered:
            heat = heat * (~sink_mask(va.sink_map, percentile=self.sink_percentile)).float()
        return heat.sum(dim=(1, 2)).tolist()

    def _masses_for(self, va: "VisualAttention", rowset: str) -> list[float]:
        rows = ROW_SELECTORS[rowset](va.query_tokens)
        heat = (va.attn if not rows else va.attn[rows]).float().mean(dim=0)
        return heat.sum(dim=(1, 2)).tolist()

    def _answer_one(self, vlm, media, prompt, options, gen_cfg, fallback: int | None):
        """Un forward + parse. Ritorna `(pred, raw, fallback_usato)`.

        Stessa regola di decisione per la baseline e per ogni condizione —
        generare e parsare — perché nella probe oracolo le due regole
        (argmax del prefill contro generate+parse) divergevano sul 7% dei
        sample: mescolarle renderebbe il delta non interpretabile.
        """
        raw = vlm.generate(vlm.build_messages(media, Text(prompt)), generation_config=gen_cfg)
        pred = parse_mcq_letter(raw, options)
        if pred is None:
            return fallback, raw, True
        return pred, raw, False

    def answer(
        self,
        vlm: "BaseVLM",
        *,
        video_path: str,
        prompt: str,
        options: list[str] | None,
        gen_cfg: "GenerationConfig",
        budget: SamplingBudget,
        video_start: float | None = None,
        video_end: float | None = None,
        frames: list[str] | None = None,
    ) -> dict:
        if not isinstance(vlm, SupportsSignals) or not hasattr(vlm, "full_visual_attention"):
            raise RuntimeError(
                f"strategy {self.name!r} richiede un modello con cattura "
                f"(`full_visual_attention`), ma {type(vlm).__name__} non la espone "
                "— usa un preset `_attn` (es. qwen3_vl_2b_attn)."
            )
        if frames is not None:
            raise RuntimeError(f"strategy {self.name!r} campiona il video e non ammette `frames` fissi.")
        if options is None:
            raise RuntimeError(f"strategy {self.name!r} richiede un MCQ.")
        if video_start is not None or video_end is not None:
            raise RuntimeError(
                f"strategy {self.name!r} non ammette trim (video_start/video_end): "
                "la finestra annotata è il METRO della misura, non un input "
                "— lancia con dataset.use_time_reference=false."
            )
        if budget.double_frames:
            raise RuntimeError(f"strategy {self.name!r}: double_frames non ha senso con le coppie.")
        if budget.nframes % 2:
            raise ValueError(f"nframes deve essere pari (= 2 * celle), non {budget.nframes}")
        if not getattr(vlm, "fix_videoframes_resize", False):
            raise RuntimeError(
                f"strategy {self.name!r} richiede model.fix_videoframes_resize=true: "
                "senza, base e pass additivo girerebbero a risoluzioni diverse e il "
                "delta mescolerebbe intervento e perdita di token."
            )

        import decord

        letters = [chr(ord("A") + i) for i in range(len(options))]
        duration = video_duration_sec(video_path)
        vr = decord.VideoReader(video_path)
        total_frames, fps = len(vr), float(vr.get_avg_fps())
        del vr

        media1, tmp1, centers = pair_video_frames(
            video_path, budget.nframes, self.pass1_pair_gap_sec,
            budget.max_pixels, budget.min_pixels,
            image_patch_size=getattr(vlm, "image_patch_size", None),
        )
        tmp2 = None
        try:
            va = vlm.full_visual_attention(media1, Text(prompt), answer_letters=letters)
            if va.t != len(centers):
                raise RuntimeError(
                    f"pass 1: celle temporali {va.t} != coppie campionate {len(centers)}")
            # Fallback comune: l'argmax del prefill sulle lettere, l'unica
            # risposta disponibile senza un altro forward.
            fb = letters.index(va.pred_letter) if va.pred_letter is not None else None
            pred1, raw1, fb1 = self._answer_one(vlm, media1, prompt, options, gen_cfg, fb)

            masses = self._cell_masses(va)
            base_idx = [int(i) for i in media1.frames_indices]
            digest = hashlib.sha1(f"{video_path}|{prompt}".encode("utf-8")).hexdigest()[:12]
            rng = random.Random(f"{self.random_seed}|{digest}")

            plans, infos = {}, {}
            for cond in self.conditions:
                cells = sorted(select_cells(masses, cond["k"], cond["select"], rng))
                spans = cells_to_spans(cells, va.t, duration)
                plan, info = additive_indices(total_frames, fps, base_idx, spans, self.n_added)
                info["cells"] = cells
                plans[cond["name"]] = plan
                infos[cond["name"]] = info

            # UN SOLO decode per tutte le condizioni: i piani condividono la
            # base e i PNG, quindi `frames_for_plans` estrae l'unione una
            # volta sola invece di riaprire il video per ogni condizione.
            media_by_cond, tmp2, _ = frames_for_plans(
                video_path, plans, budget.max_pixels, budget.min_pixels,
                image_patch_size=getattr(vlm, "image_patch_size", None),
            )

            conditions, preds, fallbacks = {}, {}, {}
            for cond in self.conditions:
                name = cond["name"]
                pred, raw, used_fb = self._answer_one(
                    vlm, media_by_cond[name], prompt, options, gen_cfg, fb)
                info = infos[name]
                preds[name] = pred
                fallbacks[name] = used_fb
                conditions[name] = {
                    "name": name, "k": cond["k"], "select": cond["select"],
                    "cells": info["cells"],
                    "spans_sec": [[round(a, 2), round(b, 2)] for a, b in info["spans"]],
                    "n_regions": len(info["spans"]),
                    "n_added_kept": info["n_added_kept"],
                    "n_collision": info["n_collision"],
                    "n_total_frames": info["n_total"],
                    "raw": raw, "pred": pred, "pred_fallback": used_fb,
                }
        finally:
            for d in (tmp1, tmp2):
                if d is not None:
                    shutil.rmtree(d, ignore_errors=True)

        return {
            # `pred` = la BASELINE (solo i frame della base): l'accuracy della
            # run è la baseline appaiata, ogni condizione ha la sua
            # `mcq_accuracy_cond_<nome>` più lo split base_true/base_false.
            "raw": raw1,
            "pred": pred1,
            "pred_fallback": fb1,
            "prefill_letter": va.pred_letter,
            "answer_entropy": va.answer_entropy,
            "answer_probs": va.answer_probs,
            # --- geometria del pass 1 -----------------------------------------
            "t_cells": va.t,
            "grid_h": va.grid_h,
            "grid_w": va.grid_w,
            "n_vis": va.t * va.grid_h * va.grid_w,
            "seq_len": int(va.input_ids.shape[0]),
            "video_duration_sec": duration,
            "pair_gap_sec": self.pass1_pair_gap_sec,
            "pair_centers_sec": [round(c, 3) for c in centers],
            "cell_span_sec": round(duration / va.t, 3),
            # --- ranking (per l'analisi offline: hit@k a questa granularità) ---
            "rank_rowset": self.rank_rowset,
            "rank_sink_filtered": self.rank_sink_filtered,
            "rowsets": {rs: {"cell_mass_raw": [float(f"{m:.6g}") for m in self._masses_for(va, rs)]}
                        for rs in self.log_rowsets},
            # --- l'intervento --------------------------------------------------
            "n_added": self.n_added,
            "base_frames": budget.nframes,
            "resampled": True,
            "conditions": conditions,
            "preds_by_condition": preds,
            "fallback_by_condition": fallbacks,
        }
