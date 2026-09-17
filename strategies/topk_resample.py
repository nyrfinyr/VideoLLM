"""Ricampionamento nelle TOP-K celle d'attenzione, con baseline appaiata e
più condizioni nello STESSO forward di pass 1.

## Cosa misura, e perché così

L'oracolo LVBench (`docs/oracolo_lvbench.md`) ha mostrato che marcare non
paga mai, mentre RICAMPIONARE dentro la finestra annotata vale +16 pp: il
collo di bottiglia è la localizzazione. La run di segnali a 512 frame ha poi
misurato che a 256 celle l'attenzione la finestra la trova (AUC 0.775, hit@1
17% contro 1.6% del caso, hit@5 38%, hit@10 52%). Questa strategy chiude il
cerchio: usa quel ranking per ricampionare davvero, e misura quanto rende.

Tre scelte di disegno, tutte per non ripetere gli errori degli arm passati:

1. **Nessun gate online.** Il pass 2 gira SEMPRE, su tutti i sample. Gattare
   sull'entropia al volo butterebbe via proprio i sample che servono a
   calibrare il gate (cosa avrebbe fatto il pass 2 dove H era bassa?), e su
   LVBench H separa poco (AUROC 0.547 contro 0.756 di Video-MME). La soglia
   si sceglie DOPO, out-of-fold, su `answer_entropy` + le risposte di tutte
   le condizioni — tutte loggate per sample.
2. **Baseline appaiata dentro la run.** Il pass 1 è la baseline: stessa
   domanda, stessi frame, stesso prompt, stesso modo di leggere la risposta
   (argmax del prefill sulle lettere). Nessun confronto fra run diverse,
   quindi McNemar diretto e nessun rumore di GPU o di seed nel mezzo.
3. **Più condizioni appaiate, non split del dataset.** Ogni condizione gira
   sugli STESSI sample: confrontare k=1 contro k=10 su metà dataset ciascuno
   sarebbe un confronto non appaiato, e con ~500 sample per split
   distinguerebbe solo differenze ≥6 pp. Il costo di una condizione in più è
   basso: il pass 1 a 512 frame costa ~58 s, un pass 2 a 128 frame molto meno
   (l'estrazione domina, ed è un quarto dei frame).

Il controllo `select: random` è obbligatorio, non decorativo: senza, un delta
positivo mescola "il segnale localizza" con "ridistribuire il budget su
regioni strette aiuta comunque". È esattamente l'ambiguità che ha reso
illeggibili gli arm di agosto.

## Come è fatto un pass 2

Le celle scelte diventano intervalli di tempo (`utils.pair_sampling.
cells_to_spans`: la cella `i` possiede `[D*i/n, D*(i+1)/n]`) e dentro ognuno
si campionano coppie di frame (`span_video_frames`), come nel pass 1 ma con
un gap più stretto: anche nel pass 2 una cella resta UN istante, quindi il
ranking del pass 2 è leggibile con le stesse regole (e un giorno iterabile).
Il budget si divide in parti uguali fra le regioni; con `global_fraction > 0`
una quota resta uniforme su tutto il video (degrado gentile: se lo zoom
cade male, il modello vede comunque il video intero a bassa densità).

## I sink, in questo setting

La run raccoglie anche di che pasta sono i token visivi, perché la domanda
"i sink esistono anche qui?" non si risponde con un numero solo:

- su ogni sample, gli scalari spaziali: `attn_spatial_mean` per rowset e
  `sink_spatial_mean` sulla griglia (DOVE dentro il frame), le rispettive
  quote di massa sull'anello di bordo (un fenomeno di parcheggio vive lì, un
  contenuto guardato no) e `attn_sink_cell_corr`, che dice se il ranking
  temporale è solo la mappa dei sink riscritta;
- ogni `sink_stats_every` sample, le statistiche dei canali degli hidden
  state su TUTTI i 28 layer (`models.qwen_attn.summarize_sink_stats`): rango
  dei sink dims fra i 2048 canali, istogrammi di |h|/mediana contro canali di
  controllo, split sink/non-sink;
- le stesse cose sul pass 2 delle condizioni nominate in
  `sink_stats_conditions`: 128 frame stipati in decine di secondi sono un
  regime diverso, e se il fenomeno è del modello deve comparire uguale;
- per i primi `dump_limit` sample, un dump con le heatmap INTERE `[t, gh,
  gw]`, la `sink_map`, le statistiche per-token e **i frame veri** delle
  celle calde e di altrettante fredde di controllo — da lì
  `scripts/sink_heatmaps.py` dipinge le heatmap sulle immagini, che è l'unico
  modo di vedere se i patch sink stanno su qualcosa o sullo sfondo.

## Output

`pred` è la risposta del PASS 1: l'accuracy della run è la baseline, e ogni
condizione ha le sue chiavi (`cond_<nome>` in `preds_by_condition`, che
`evals.base.mcq_accuracy` trasforma in `correct_cond_<nome>` /
`seen_cond_<nome>` e nello split per correttezza della baseline, cioè
`r_fix`/`r_break` già nel summary). Tutto il resto — entropia, masse per
cella, celle scelte da ogni condizione — è per sample su Weave, che è dove
si fanno la scelta del gate e il calcolo di hit@k contro la finestra vera.
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.media import Text
from models.signals import SupportsSignals
from utils.attn_core import ROW_SELECTORS, sink_mask
from utils.pair_sampling import cells_to_spans, pair_video_frames, span_video_frames

from .base import SamplingBudget, Strategy, video_duration_sec

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from utils.attn_core import VisualAttention

logger = logging.getLogger(__name__)

# Condizioni di default: i tre k da ablare, l'ibrido e il controllo random.
# `k` = quante celle del ranking si ricampionano (larghezza `w=0`: allargare
# attorno al picco rende meno che allungare il ranking — a parità di 5 celle,
# top-5 copre la finestra nel 38% dei sample contro il 25% di "picco ±2").
DEFAULT_CONDITIONS = (
    {"name": "k1", "k": 1},
    {"name": "k3", "k": 3},
    {"name": "k10", "k": 10},
    {"name": "hybrid3", "k": 3, "global_fraction": 0.5},
    {"name": "rand3", "k": 3, "select": "random"},
)
_SELECT_KINDS = ("attention", "random")
# Percentuali di token (ordinati per sink score) su cui si misura la quota di
# massa d'attenzione. È la prova decisiva: se i token "sink" assorbono
# attenzione, pochi punti percentuali di token ne portano una quota
# sproporzionata; se la curva sta su ~p%, quel punteggio non identifica
# nessun pozzo. Sulla probe 107727 (512 frame) la curva era piatta.
SINK_MASS_CURVE_PCTS = (1, 2, 5, 10, 25, 50)
_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _heat_stats(heat: "torch.Tensor") -> dict:
    """Riduzioni di una heatmap `[t, gh, gw]` che stanno in poche centinaia di
    byte per sample, quindi loggabili su TUTTI i sample del fullset.

    `spatial_mean` è la domanda "DOVE dentro il frame": media sulle celle,
    cioè il profilo spaziale dell'attenzione (o dei sink) sulla griglia.
    `border_share` è la sua lettura sintetica — quanta massa sta sull'anello
    esterno di patch. Un fenomeno di tipo sink vive sulle patch di bordo e di
    sfondo, un'attenzione che guarda contenuto no: è il primo discriminante
    fra "punti salienti" e "token parcheggio".
    """
    t, gh, gw = heat.shape
    border = torch.ones(gh, gw, dtype=torch.bool)
    if gh > 2 and gw > 2:
        border[1:-1, 1:-1] = False
    tot = float(heat.sum())
    return {
        "spatial_mean": heat.mean(dim=0),                       # [gh, gw]
        "border_share": float(heat[:, border].sum()) / tot if tot > 0 else None,
        "border_share_uniform": float(border.sum()) / (gh * gw),  # riferimento
    }


def _corr(a: list[float], b: list[float]) -> float | None:
    """Pearson fra due vettori per cella. Serve a una domanda sola: il ranking
    d'attenzione è solo una riscrittura della mappa dei sink? Se la
    correlazione è alta, le celle "calde" sono le celle piene di token sink e
    il segnale di localizzazione sarebbe un artefatto."""
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def allocate_pairs(n_pairs: int, k: int) -> list[int]:
    """`n_pairs` coppie su `k` regioni: parti uguali, il resto alle prime
    (cioè alle celle col ranking più alto). Ogni regione ne riceve almeno
    una — con `k` più grande del budget la condizione non avrebbe senso e il
    chiamante la rifiuta prima."""
    base, rest = divmod(n_pairs, k)
    return [base + (1 if j < rest else 0) for j in range(k)]


def select_cells(
    masses: list[float], k: int, kind: str, rng: random.Random
) -> list[int]:
    """Le `k` celle da ricampionare: le prime del ranking, o `k` a caso.

    Il controllo random pesca dalle STESSE 256 celle e con lo stesso k, così
    fra arm e controllo cambia solo QUALE regione si infittisce — non quanto
    budget si sposta né quanto stretta è la regione.
    """
    if kind == "random":
        return rng.sample(range(len(masses)), k)
    return sorted(range(len(masses)), key=lambda i: -masses[i])[:k]


class TopkResampleStrategy(Strategy):
    name = "topk_resample"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.pass1_pair_gap_sec = float(cfg.get("pass1_pair_gap_sec", 2.0))
        self.pass2_nframes = int(cfg.get("pass2_nframes", 128))
        self.pass2_pair_gap_sec = float(cfg.get("pass2_pair_gap_sec", 0.5))
        self.rank_rowset = str(cfg.get("rank_rowset", "all"))
        self.rank_sink_filtered = bool(cfg.get("rank_sink_filtered", False))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.log_rowsets = [str(r) for r in cfg.get("log_rowsets", ["all", "question"])]
        self.random_seed = int(cfg.get("random_seed", 0))
        # Statistiche sui canali degli hidden state (i "sink"): sul pass 1
        # sempre, sulle condizioni solo se nominate — il payload per sample è
        # ~40 KB, moltiplicarlo per 5 condizioni non vale la pena quando la
        # domanda ("il fenomeno c'è anche a 128 frame densi?") si risponde con
        # UNA condizione.
        self.sink_stats = bool(cfg.get("sink_stats", True))
        # Le statistiche dei canali pesano ~40 KB per sample su Weave: su 1549
        # sample sono 60 MB per il solo pass 1. Servono AGGREGATE (istogrammi
        # e quantili si sommano fra sample), quindi su un fullset basta una
        # frazione: `sink_stats_every=4` ne prende una ogni 4. Gli scalari
        # spaziali (border share, correlazione attn-sink) restano invece su
        # TUTTI i sample, perché costano niente e si incrociano con hit@k.
        self.sink_stats_every = max(1, int(cfg.get("sink_stats_every", 1)))
        self.sink_stats_conditions = [str(n) for n in cfg.get("sink_stats_conditions", ["k3"])]
        self.sink_topk_channels = int(cfg.get("sink_topk_channels", 10))
        dump_dir = cfg.get("dump_dir")
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.dump_limit = int(cfg.get("dump_limit", 0))
        self.dump_cells = int(cfg.get("dump_cells", 4))
        # Contatore d'ISTANZA (per processo): con N shard i dump sono fino a
        # N * dump_limit. Nessuna race: i sample sono sequenziali.
        self._n_dumped = 0
        self._n_seen = 0
        conditions = cfg.get("conditions") or DEFAULT_CONDITIONS
        self.conditions = [dict(c) for c in conditions]

        bad = [r for r in self.log_rowsets + [self.rank_rowset] if r not in ROW_SELECTORS]
        if bad:
            raise ValueError(f"rowset non validi: {bad}. Validi: {sorted(ROW_SELECTORS)}")
        if self.pass2_nframes % 2 != 0 or self.pass2_nframes < 2:
            raise ValueError(f"pass2_nframes deve essere pari e >= 2, non {self.pass2_nframes}")
        names = [c.get("name") for c in self.conditions]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError(f"le condizioni devono avere nomi unici e non vuoti: {names}")
        n_pairs2 = self.pass2_nframes // 2
        for c in self.conditions:
            kind = c.setdefault("select", "attention")
            if kind not in _SELECT_KINDS:
                raise ValueError(f"select {kind!r} sconosciuto (condizione {c['name']!r}): {_SELECT_KINDS}")
            k = int(c.get("k", 1))
            gf = float(c.get("global_fraction", 0.0))
            if not 0.0 <= gf < 1.0:
                raise ValueError(f"global_fraction fuori da [0,1) nella condizione {c['name']!r}: {gf}")
            # Con meno coppie che regioni ci sarebbe una regione da 0 coppie e
            # il nome della condizione mentirebbe sul k davvero usato.
            if k < 1 or k > int(n_pairs2 * (1 - gf)):
                raise ValueError(
                    f"condizione {c['name']!r}: k={k} non campionabile con "
                    f"pass2_nframes={self.pass2_nframes} e global_fraction={gf}"
                )
            c["k"] = k
            c["global_fraction"] = gf
        # Il preset nomina `k3`; se il chiamante passa una lista di condizioni
        # diversa (uso previsto dallo sbatch: argomenti extra dopo "$@") quel
        # nome non esiste più. Sollevare bloccherebbe la run per un knob
        # accessorio: si restringe alle condizioni davvero presenti e si
        # avvisa, così un nome sbagliato resta visibile nel log.
        known = {c["name"] for c in self.conditions}
        unknown = [n for n in self.sink_stats_conditions if n not in known]
        if unknown:
            self.sink_stats_conditions = [n for n in self.sink_stats_conditions if n in known]
            logger.warning(
                "sink_stats_conditions nomina condizioni inesistenti %s: ignorate "
                "(condizioni disponibili: %s)", sorted(unknown), sorted(known),
            )
        if not any(c["select"] == "random" for c in self.conditions):
            logger.warning(
                "nessuna condizione con select=random: il delta contro la baseline "
                "mescolerà il segnale con l'effetto del solo ridistribuire il budget"
            )

    # ── pass 1 ──────────────────────────────────────────────────────────────
    def _pass1(self, vlm, video_path, prompt, letters, budget, want_dump: bool, want_stats: bool):
        """Pass 1 → `(VisualAttention, centri, media, tmp_dir)`.

        `media` e `tmp_dir` NON vengono liberati qui: i PNG dei frame servono
        al dump (una heatmap va dipinta sul frame vero, non su una griglia
        astratta), che avviene solo dopo aver scelto le celle di tutte le
        condizioni. Li libera `answer`. `tmp_dir` è `None` quando i frame
        arrivano dalla cache dell'ultimo video, che li tiene per le domande
        successive sullo stesso video.
        """
        media, tmp_dir, centers = pair_video_frames(
            video_path, budget.nframes, self.pass1_pair_gap_sec,
            budget.max_pixels, budget.min_pixels,
            image_patch_size=getattr(vlm, "image_patch_size", None),
        )
        try:
            va = vlm.full_visual_attention(
                media, Text(prompt), answer_letters=letters,
                sink_stats=want_stats or want_dump,
                sink_stats_percentile=self.sink_percentile,
                sink_stats_topk=self.sink_topk_channels,
                sink_stats_per_token=want_dump,
            )
        except Exception:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        if va.t != len(centers):
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"pass 1: celle temporali {va.t} != coppie campionate {len(centers)}"
            )
        return va, centers, media, tmp_dir

    def _heat(self, va: "VisualAttention", rowset: str, sink_filtered: bool) -> "torch.Tensor":
        """Heatmap `[t, gh, gw]` di un rowset: media sulle righe-query, NON
        rinormalizzata (la massa visiva totale è essa stessa un dato)."""
        rows = ROW_SELECTORS[rowset](va.query_tokens)
        heat = (va.attn if not rows else va.attn[rows]).float().mean(dim=0)
        if sink_filtered:
            keep = (~sink_mask(va.sink_map, percentile=self.sink_percentile)).float()
            heat = heat * keep
        return heat

    def _cell_masses(self, va: "VisualAttention", rowset: str, sink_filtered: bool) -> list[float]:
        return self._heat(va, rowset, sink_filtered).sum(dim=(1, 2)).tolist()

    @staticmethod
    def _sink_mass_curve(heat: "torch.Tensor", sink_map: "torch.Tensor") -> list[float | None]:
        """Quota di massa d'attenzione sui top-p% token per punteggio di sink."""
        flat = heat.flatten()
        order = torch.argsort(sink_map.flatten().float(), descending=True)
        total = float(flat.sum())
        n = flat.numel()
        out = []
        for pct in SINK_MASS_CURVE_PCTS:
            k = max(1, math.ceil(pct / 100 * n))
            out.append(float(flat[order[:k]].sum()) / total if total > 0 else None)
        return out

    # ── pass 2 ──────────────────────────────────────────────────────────────
    def _run_condition(
        self, vlm, video_path, prompt, letters, budget, cond, masses, duration, rng
    ) -> dict:
        k, gf = cond["k"], cond["global_fraction"]
        n_pairs2 = self.pass2_nframes // 2
        cells = select_cells(masses, k, cond["select"], rng)
        spans = cells_to_spans(cells, len(masses), duration)
        counts = allocate_pairs(n_pairs2 - int(round(gf * n_pairs2)), k)
        n_global = n_pairs2 - sum(counts)
        if n_global > 0:
            # La quota globale è una regione come le altre: l'intero video.
            # Finisce mescolata alle altre coppie in ordine di tempo, quindi
            # il modello vede una timeline unica, più fitta dove ha guardato.
            spans = spans + [(0.0, duration)]
            counts = counts + [n_global]

        media, tmp_dir, centers2 = span_video_frames(
            video_path, spans, counts, self.pass2_pair_gap_sec,
            budget.max_pixels, budget.min_pixels,
            image_patch_size=getattr(vlm, "image_patch_size", None),
        )
        want_stats = (
            cond["name"] in self.sink_stats_conditions
            and (self._n_seen - 1) % self.sink_stats_every == 0
        )
        try:
            va2 = vlm.full_visual_attention(
                media, Text(prompt), answer_letters=letters,
                sink_stats=want_stats,
                sink_stats_percentile=self.sink_percentile,
                sink_stats_topk=self.sink_topk_channels,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if va2.t != len(centers2):
            raise RuntimeError(
                f"condizione {cond['name']!r}: celle {va2.t} != coppie {len(centers2)}"
            )
        pred = letters.index(va2.pred_letter) if va2.pred_letter is not None else None
        # Il pass 2 è un REGIME DIVERSO (128 frame stipati in poche decine di
        # secondi di video invece di 512 su ore): se il fenomeno sink è una
        # proprietà del modello e non del campionamento, deve comparire uguale
        # qui. Confronto interno alla stessa run, stesso video, stessa domanda.
        heat2 = self._heat(va2, self.rank_rowset, False)
        stats2 = _heat_stats(heat2)
        sink2 = _heat_stats(va2.sink_map.float())
        return {
            "name": cond["name"], "k": k, "select": cond["select"], "global_fraction": gf,
            "cells": cells,
            "n_pairs_zoom": sum(counts) - n_global, "n_pairs_global": n_global,
            "span_sec": round(duration / len(masses), 3),
            "raw": va2.pred_letter, "pred": pred,
            "answer_entropy": va2.answer_entropy, "answer_probs": va2.answer_probs,
            # Primo tempo e ultimo tempo davvero guardati: il controllo più
            # veloce che le regioni siano finite dove dovevano.
            "centers_first": round(centers2[0], 2), "centers_last": round(centers2[-1], 2),
            # --- il fenomeno sink in questo regime ---------------------------
            "t_cells": va2.t, "grid_h": va2.grid_h, "grid_w": va2.grid_w,
            "attn_spatial_mean": stats2["spatial_mean"],
            "attn_border_share": stats2["border_share"],
            "sink_spatial_mean": sink2["spatial_mean"],
            "sink_border_share": sink2["border_share"],
            # Il riferimento uniforme dipende dalla griglia, che nel pass 2 è
            # la stessa del pass 1 ma va loggata comunque: senza,
            # `analyze_topk_run.py` stampa "uniforme = 0%" proprio nella riga
            # che serve a leggere le due quote qui sopra.
            "border_share_uniform": sink2["border_share_uniform"],
            "sink_mass_curve": self._sink_mass_curve(heat2, va2.sink_map),
            "attn_sink_cell_corr": _corr(
                heat2.sum(dim=(1, 2)).tolist(), va2.sink_map.float().sum(dim=(1, 2)).tolist()
            ),
            "sink_stats": va2.sink_stats,
        }

    def _dump(self, video_path, prompt, vlm, va, media, centers, conditions) -> str | None:
        """Dump per-sample per l'analisi "i sink sono punti salienti?".

        Salva insieme le tre cose che servono a rispondere, che separate non
        bastano: la heatmap COMPLETA `[t, gh, gw]` (dove guarda), la `sink_map`
        completa e le statistiche per-token dei canali (chi è sink e quanto),
        e i FRAME VERI delle celle interessanti, così la heatmap si dipinge
        sull'immagine invece che su una griglia astratta
        (`scripts/sink_heatmaps.py`).

        Le celle salvate sono quelle scelte dalle condizioni (le più calde) più
        altrettante celle FREDDE di controllo: senza il controllo, "i patch
        sink stanno sullo sfondo" non è falsificabile — tutti i frame hanno
        dello sfondo.

        Ritorna il path, o `None` se la scrittura fallisce: un disco pieno non
        deve far perdere il sample, che resta comunque su Weave.
        """
        stem = _FILENAME_RE.sub("_", Path(video_path).stem)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        base = self.dump_dir / f"{stem}_{digest}"
        masses = self._cell_masses(va, self.rank_rowset, self.rank_sink_filtered)
        hot: list[int] = []
        for c in conditions:
            for cell in c.get("cells", []):
                if cell not in hot:
                    hot.append(cell)
        hot = hot[: self.dump_cells]
        cold = [i for i in sorted(range(len(masses)), key=lambda i: masses[i]) if i not in hot]
        cells = hot + cold[: len(hot)]
        try:
            base.mkdir(parents=True, exist_ok=True)
            saved = {}
            for cell in cells:
                for j, pos in enumerate((2 * cell, 2 * cell + 1)):
                    if pos >= len(media.video):
                        continue
                    dst = base / f"cell{cell:04d}_{'ab'[j]}.png"
                    shutil.copyfile(media.video[pos], dst)
                    saved.setdefault(cell, []).append(dst.name)
            torch.save({
                "video_path": video_path,
                "prompt": prompt,
                "model_id": getattr(vlm, "model_id", None),
                "frames_indices": list(media.frames_indices),
                "fps": media.fps,
                "pair_centers_sec": list(centers),
                "pair_gap_sec": self.pass1_pair_gap_sec,
                "t": va.t, "grid_h": va.grid_h, "grid_w": va.grid_w,
                "query_tokens": [(q.row, q.index, q.token, q.text) for q in va.query_tokens],
                "answer_probs": va.answer_probs, "pred_letter": va.pred_letter,
                # Heatmap intere, un tensore per rowset: `[t, grid_h, grid_w]`.
                "attn": {rs: self._heat(va, rs, False) for rs in self.log_rowsets},
                "sink_map": va.sink_map,
                "sink_percentile": self.sink_percentile,
                "sink_stats": va.sink_stats,   # include `per_token` (vedi sopra)
                "cell_mass_rank_rowset": masses,
                "rank_rowset": self.rank_rowset,
                "cells_hot": hot, "cells_cold": cells[len(hot):],
                "cell_frames": saved,
                "conditions": {c["name"]: {"cells": c["cells"], "k": c["k"],
                                           "select": c["select"]} for c in conditions},
            }, base / "dump.pt")
        except OSError as e:
            logger.warning("dump fallito per %s: %s", video_path, e)
            return None
        finally:
            self._n_dumped += 1
        return str(base)

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
        if budget.nframes % 2 != 0:
            raise ValueError(f"nframes deve essere pari (= 2 * celle), non {budget.nframes}")
        if not getattr(vlm, "fix_videoframes_resize", False):
            raise RuntimeError(
                f"strategy {self.name!r} richiede model.fix_videoframes_resize=true: "
                "senza, pass 1 e pass 2 girerebbero a risoluzioni diverse e il "
                "delta mescolerebbe intervento e perdita di token."
            )

        letters = [chr(ord("A") + i) for i in range(len(options))]
        duration = video_duration_sec(video_path)
        want_dump = self.dump_dir is not None and self._n_dumped < self.dump_limit
        want_stats = self.sink_stats and (self._n_seen % self.sink_stats_every == 0)
        self._n_seen += 1
        va, centers, media1, tmp_dir1 = self._pass1(
            vlm, video_path, prompt, letters, budget, want_dump, want_stats
        )
        try:
            return self._answer_from_pass1(
                vlm, va, centers, media1, video_path, prompt, letters, budget,
                duration, want_dump, want_stats,
            )
        finally:
            if tmp_dir1 is not None:
                shutil.rmtree(tmp_dir1, ignore_errors=True)

    def _answer_from_pass1(
        self, vlm, va, centers, media1, video_path, prompt, letters, budget,
        duration, want_dump, want_stats,
    ) -> dict:
        masses = self._cell_masses(va, self.rank_rowset, self.rank_sink_filtered)

        # Seed per-sample: il controllo random deve essere riproducibile
        # (stesse celle se la run viene rilanciata) ma diverso da sample a
        # sample, e indipendente dall'ordine degli shard.
        digest = hashlib.sha1(f"{video_path}|{prompt}".encode("utf-8")).hexdigest()[:12]
        rng = random.Random(f"{self.random_seed}|{digest}")

        conditions = []
        for cond in self.conditions:
            conditions.append(self._run_condition(
                vlm, video_path, prompt, letters, budget, cond, masses, duration, rng,
            ))

        pred1 = letters.index(va.pred_letter) if va.pred_letter is not None else None
        rowsets = {}
        for rs in self.log_rowsets:
            heat = self._heat(va, rs, False)
            stats = _heat_stats(heat)
            rowsets[rs] = {
                "cell_mass_raw": heat.sum(dim=(1, 2)).tolist(),
                "cell_mass_sink_filtered": self._cell_masses(va, rs, True),
                "visual_mass_total": float(heat.sum()),
                # DOVE dentro il frame guarda questo rowset, mediato sulle
                # celle: 45 numeri per sample a 5x9, loggabili su tutto il
                # fullset. La heatmap completa [t, gh, gw] sta solo nei dump.
                "attn_spatial_mean": stats["spatial_mean"],
                "attn_border_share": stats["border_share"],
                "sink_mass_curve": self._sink_mass_curve(heat, va.sink_map),
            }
        sink_heat = va.sink_map.float()
        sink_stats_map = _heat_stats(sink_heat)
        rank_masses = rowsets.get(self.rank_rowset, {}).get("cell_mass_raw") or masses
        out = {
            # `pred` = pass 1: l'accuracy della run È la baseline appaiata.
            "raw": va.pred_letter,
            "pred": pred1,
            "pred_fallback": False,
            "answer_entropy": va.answer_entropy,
            "answer_probs": va.answer_probs,
            # --- geometria e pass 1 ------------------------------------------
            "t_cells": va.t,
            "grid_h": va.grid_h,
            "grid_w": va.grid_w,
            "n_vis": va.t * va.grid_h * va.grid_w,
            "seq_len": int(va.input_ids.shape[0]),
            "video_duration_sec": duration,
            "pair_gap_sec": self.pass1_pair_gap_sec,
            "pair_centers_sec": centers,
            "rank_rowset": self.rank_rowset,
            "rank_sink_filtered": self.rank_sink_filtered,
            "rowsets": rowsets,
            # --- il fenomeno sink a 512 frame ---------------------------------
            "sink_percentile": self.sink_percentile,
            "sink_mass_curve_pcts": list(SINK_MASS_CURVE_PCTS),
            "sink_spatial_mean": sink_stats_map["spatial_mean"],      # [gh, gw]
            "sink_temporal_mean": sink_heat.mean(dim=(1, 2)).tolist(),  # [t]
            "sink_border_share": sink_stats_map["border_share"],
            "border_share_uniform": sink_stats_map["border_share_uniform"],
            # Il ranking d'attenzione è solo la mappa dei sink riscritta?
            "attn_sink_cell_corr": _corr(rank_masses, sink_heat.sum(dim=(1, 2)).tolist()),
            "sink_stats": va.sink_stats if want_stats else None,
            "dump_path": self._dump(
                video_path, prompt, vlm, va, media1, centers, conditions,
            ) if want_dump else None,
            # --- pass 2 -------------------------------------------------------
            "pass2_nframes": self.pass2_nframes,
            "pass2_pair_gap_sec": self.pass2_pair_gap_sec,
            "conditions": {c["name"]: c for c in conditions},
            "preds_by_condition": {c["name"]: c["pred"] for c in conditions},
            "entropy_by_condition": {c["name"]: c["answer_entropy"] for c in conditions},
        }
        return _jsonable(out)


def _jsonable(x):
    """Tensori/float → JSON puro, float a 6 cifre SIGNIFICATIVE (le masse per
    cella di un rowset stretto stanno sull'ordine di 1e-5: sei DECIMALI le
    azzererebbero). NaN/inf → `None`."""
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
