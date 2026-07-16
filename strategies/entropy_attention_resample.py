"""`EntropyAttentionResampleStrategy` — primo tentativo di sfruttare i
segnali di entropia risposta MCQ + concentrazione attenzione (sink-
filtrata) per decidere se ricampionare i frame, invece di limitarsi a
loggarli. Pensata per un rilancio di MVBench a confronto con la baseline
in `README.md`.

Logica (pass 1 sempre uniforme, budget `nframes` invariato):

    entropia bassa                → accetta la risposta del pass 1
    entropia alta, ramo "disperso"    → ricampiona, SFASATO di fase rispetto
                                     ai frame già visti (dispersione: non
                                     si sa dove guardare, si prova altrove)
    entropia alta, ramo "concentrato" → ricampiona, INFITTITO in una finestra
                                     stretta intorno al frame di picco
                                     (il modello guarda "quasi giusto" ma
                                     non abbastanza in dettaglio) — oppure,
                                     se `zoom_multi_region` è attivo, in
                                     FINO A `n_regions` finestre disgiunte
                                     centrate sulle celle a massa più alta
                                     (vedi `docs/piano_zoom_multiregione_
                                     disgiunto.md`, Opzione C1).

Il ramo "disperso" vs "concentrato" è deciso da `branch_selector`
(`conf/config.yaml`, preset `strategy.entropy_attention_resample`):
`"concentration"` (default, comportamento storico) confronta `top1_pct`
con `concentration_threshold`; `"entropy"` (`docs/piano_zoom_multiregione_
disgiunto.md` §1.3) usa invece l'entropia di Shannon normalizzata
`H/log(t)` sulle masse per-cella sink-filtrate — bassa (attenzione
concentrata) → zoom, alta (dispersa) → phase_shift. Le due soglie sono
indipendenti e NON tarate l'una sull'altra.

Al massimo UN ricampionamento (non c'è un terzo pass che ri-verifica il
segnale). La risposta FINALE — sia nei rami "accetta" sia dopo un
ricampionamento — viene SEMPRE da una vera `vlm.generate()` (mai dal solo
`pred_letter` della softmax ristretta): così i rami "accetta" restano
bit-per-bit equivalenti alla baseline `uniform`, isolando l'effetto del
resampling ai soli casi in cui avviene davvero.

Soglie NON validate per MVBench (i numeri raccolti in
`docs/entropia_confidenza_mcq.md` vengono da un campione VNBench
"retrieval" diverso per dominio/nframes) — sono knob del preset
`strategy.entropy_attention_resample` in `conf/config.yaml`, tarabili da
CLI senza toccare codice, es. `strategy.entropy_threshold=1.0`.

Richiede un modello che implementa `SupportsSignals` (oggi solo
`model=qwen2_5_vl_3b_attn`) — fail-fast altrimenti, stesso pattern di
`entropy_shortcut`. Se `frames` è già fissato dal dataset (nessuna libertà
di selezione, es. MVBench `data_type="frame"`) il resampling è per
definizione impossibile: si accetta sempre il pass 1.
"""
from __future__ import annotations

import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image

from models.media import Text, VideoFrames
from models.signals import SupportsSignals
from utils.attn_core import GridSpec, RankedCell, rank_cells, sink_view
from utils.mcq import parse_mcq_letter

from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer


def _video_duration_sec(video_path: str) -> float:
    """Durata video in secondi via decord (solo metadata, non decodifica
    i frame) — serve per clampare le finestre di resample ai bordi reali
    del video quando `video_start`/`video_end` non sono già noti dal
    dataset (caso comune su MVBench, `bound=False`)."""
    import decord

    vr = decord.VideoReader(video_path)
    return len(vr) / vr.get_avg_fps()


def _ranked_cells(visual_attention, *, percentile: float, border: int) -> list[RankedCell]:
    """Classifica TUTTE le celle temporali di `visual_attention` per massa
    di attenzione (sink-filtrata), stessa logica di
    `utils.attn_core.attention_concentration` operante direttamente su
    `VisualAttention` (niente `AttentionCapture`/fps/frame_indices reali:
    qui serve solo l'indice di cella, la posizione temporale si ricava per
    via frazionaria in `answer()`). `frame_indices` dummy (`range(t)`,
    `double=True`) perché `cell_to_frames` non viene mai ispezionato per
    `frames`/`rep_frame`/`t_sec`. `rank_cells` non limita la lista con
    `topk` (marca solo `is_top`), quindi ritorna sempre le `t` celle.
    """
    heat = visual_attention.attn.float().mean(dim=0)  # [t, grid_h, grid_w]
    grid = GridSpec(t=visual_attention.t, grid_h=visual_attention.grid_h, grid_w=visual_attention.grid_w, double=True)
    heat_nonsink, _ = sink_view(heat, visual_attention.sink_map, view="nonsink", percentile=percentile, border=border)
    return rank_cells(heat_nonsink, grid, list(range(visual_attention.t)), fps=1.0, metric="sum", topk=1)


def _attention_entropy_norm(ranked: list[RankedCell]) -> float:
    """Entropia di Shannon **normalizzata** `H/log(t)` sulle `t` masse
    per-cella (sink-filtrate) di `ranked` (`docs/piano_zoom_multiregione_
    disgiunto.md` §1.3). `H` bassa (~0) = attenzione concentrata su poche
    celle; `H` alta (~1) = dispersa uniformemente sulle `t` celle.

    Degenera a `0.0` se `t <= 1` (nessuna dispersione possibile, una sola
    cella è per definizione "concentrata"). Degenera a `1.0` se la massa
    non-sink totale è nulla (tutti i `pct` a zero, vedi `rank_cells`): senza
    segnale si tratta il sample come "non concentrato", la scelta prudente
    che instrada verso `phase_shift` invece di uno zoom su un picco
    inesistente — coerente con `top1_pct == 0 < concentration_threshold`
    nel selettore basato su `top1_pct`.
    """
    t = len(ranked)
    if t <= 1:
        return 0.0
    probs = [c.pct / 100.0 for c in ranked]
    if sum(probs) <= 0:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return h / math.log(t)


def _top1_zscore(ranked: list[RankedCell]) -> float:
    """Z-score DELLA CELLA DI PICCO rispetto alla distribuzione delle masse
    (`pct`) di TUTTE le `t` celle dello stesso video (`docs/
    piano_zoom_multiregione_disgiunto.md`, sezione soglia adattiva): quanto
    `top1_pct` è un outlier rispetto alla media/dev.std delle masse di
    QUESTO video, non rispetto a un livello di caso teorico fisso (`100/t`)
    o a soglie assolute tarate su un altro dataset/`nframes`. Nessuna
    calibrazione esterna: media e dev.std sono per-sample, quindi il segnale
    si adatta automaticamente a `t` e alla forma reale della distribuzione,
    incluso il caso concentrato (poche celle alte, coda lunga di celle quasi
    a zero) che rende `pct` fortemente non-normale — lo z-score resta
    comunque un indice valido di "quanto spicca" il picco, non un test di
    normalità.

    Degenera a `0.0` se `t <= 1` (nessuna cella di confronto) o se la
    dev.std delle masse è nulla (tutte le celle uguali, incluso il caso
    massa totale nulla) — nessuna cella spicca rispetto alle altre.
    """
    t = len(ranked)
    if t <= 1:
        return 0.0
    pcts = [c.pct for c in ranked]
    mean = statistics.fmean(pcts)
    stdev = statistics.pstdev(pcts, mu=mean)
    if stdev <= 0:
        return 0.0
    return (pcts[0] - mean) / stdev


def _select_region_cells(
    ranked: list[RankedCell], *, n_regions: int, region_select: str, region_mad_k: float,
) -> list[RankedCell]:
    """Sceglie fino a `n_regions` celle-picco da `ranked` (già ordinata per
    massa decrescente). `"topk"`: le prime `n_regions`. `"adaptive"`: celle
    sopra `mediana + k·MAD` della distribuzione, poi cap a `n_regions`. La
    mediana/MAD sono calcolate ESCLUDENDO il picco globale (`ranked[0]`):
    la distribuzione di `pct` è heavy-tailed, includere l'estremo che si
    vuole selezionare inflazionerebbe la sua stessa soglia (auto-
    mascheramento — vedi `docs/piano_zoom_multiregione_disgiunto.md` §1.2).
    """
    if region_select == "topk":
        return ranked[:n_regions]
    if region_select != "adaptive":
        raise ValueError(f"region_select sconosciuto: {region_select!r}. Valide: 'topk', 'adaptive'")

    rest_pcts = [c.pct for c in ranked[1:]] or [ranked[0].pct]
    median = statistics.median(rest_pcts)
    mad = statistics.median(abs(p - median) for p in rest_pcts)
    thr = median + region_mad_k * mad
    selected = [c for c in ranked if c.pct >= thr]
    if not selected:
        selected = ranked[:1]
    return selected[:n_regions]


def _multi_region_spans(
    ranked: list[RankedCell],
    *,
    t: int,
    w0: float,
    w1: float,
    duration: float,
    n_regions: int,
    region_select: str,
    region_mad_k: float,
    region_merge_gap_frac: float,
    region_window_frac: float,
) -> list[tuple[float, float, float]]:
    """Fino a `n_regions` finestre `(a, b, mass)` disgiunte e ordinate per
    tempo (`mass` = pct di attenzione della regione, sommata se più celle
    sono state fuse). Ogni finestra ha semi-larghezza fissa
    `region_window_frac * (w1 - w0)` attorno al centro della/e cella/e
    selezionate; centri più vicini di `region_merge_gap_frac * (w1 - w0)`
    vengono fusi in un solo centro (media pesata dalla massa) prima di
    costruire le finestre, ed eventuali finestre ancora sovrapposte dopo
    l'allargamento a larghezza fissa vengono fuse di nuovo.
    """
    selected = _select_region_cells(ranked, n_regions=n_regions, region_select=region_select, region_mad_k=region_mad_k)

    span_half = region_window_frac * (w1 - w0)
    gap = region_merge_gap_frac * (w1 - w0)

    centers = sorted(
        ((w0 + ((c.cell + 0.5) / t if t else 0.5) * (w1 - w0), c.pct) for c in selected),
        key=lambda cm: cm[0],
    )

    clusters: list[list[tuple[float, float]]] = []
    for center, mass in centers:
        if clusters and center - clusters[-1][-1][0] < gap:
            clusters[-1].append((center, mass))
        else:
            clusters.append([(center, mass)])

    spans: list[list[float]] = []
    for cluster in clusters:
        total_mass = sum(m for _, m in cluster)
        center = sum(c * m for c, m in cluster) / total_mass if total_mass else sum(c for c, _ in cluster) / len(cluster)
        a = max(0.0, min(center - span_half, duration))
        b = max(0.0, min(center + span_half, duration))
        spans.append([a, b, total_mass])

    # Le finestre a larghezza fissa possono comunque toccarsi/sovrapporsi
    # anche se i centri erano a distanza >= gap (o dopo il clamp a
    # [0, duration]): fondi di nuovo per garantire regioni disgiunte.
    spans.sort(key=lambda s: s[0])
    merged: list[list[float]] = []
    for a, b, mass in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += mass
        else:
            merged.append([a, b, mass])
    return [(a, b, mass) for a, b, mass in merged]


def _allocate_region_budget(masses: list[float], nframes: int, *, allocation: str) -> list[int]:
    """Spartisce `nframes` fra `len(masses)` regioni, somma sempre `<=
    nframes` (vincolo VRAM, docs/piano_zoom_multiregione_disgiunto.md
    §3.2.3): se il budget è più stretto del numero di regioni, tiene solo
    le regioni a massa più alta (1 frame ciascuna), le altre restano a 0.
    """
    m = len(masses)
    if m == 0 or nframes <= 0:
        return [0] * m
    if allocation == "equal":
        weights = [1.0] * m
    elif allocation == "proportional":
        weights = list(masses) if sum(masses) > 0 else [1.0] * m
    else:
        raise ValueError(f"budget_allocation sconosciuto: {allocation!r}. Valide: 'equal', 'proportional'")

    if nframes < m:
        keep = set(sorted(range(m), key=lambda i: weights[i], reverse=True)[:nframes])
        return [1 if i in keep else 0 for i in range(m)]

    total_w = sum(weights)
    counts = [max(1, round(nframes * w / total_w)) for w in weights]
    while sum(counts) > nframes:
        i = max(range(m), key=lambda i: counts[i])
        if counts[i] <= 1:
            break
        counts[i] -= 1
    return counts


def _build_multi_region_media(
    video_path: str,
    spans: list[tuple[float, float, float]],
    counts: list[int],
    budget: SamplingBudget,
) -> tuple[VideoFrames, list[tuple[float, float]], Path]:
    """Estrae l'unione dei frame delle regioni con budget > 0, li passa
    come `VideoFrames` con `sample_fps` iniettato (C1: si accetta che il
    modello li veda come uniformemente spaziati — vedi
    `docs/piano_zoom_multiregione_disgiunto.md` §2/§3.2.5). `sample_fps` è
    calibrato sulla densità EFFETTIVA raggiunta dentro le regioni (frame
    totali / durata totale delle regioni), non sulla finestra pass-1
    intera: è la lettura più fedele di "quanto è infittito" una volta
    accettato che i buchi fra regioni spariscono dall'encoding.

    Ritorna anche gli span effettivamente usati (solo quelli con budget >
    0) e la directory temporanea dei PNG estratti — il chiamante la
    ripulisce dopo la `generate()`.
    """
    import decord

    vr = decord.VideoReader(video_path)
    total = len(vr)
    last = total - 1
    fps = float(vr.get_avg_fps())
    max_duration = total / fps

    all_indices: list[int] = []
    seen: set[int] = set()
    used_spans: list[tuple[float, float]] = []
    for (a, b, _mass), n in zip(spans, counts):
        if n <= 0:
            continue
        # Stessa formula lo/hi di `utils.attn_core.reconstruct_frame_indices`,
        # ma riusa il `vr`/`fps` già aperti (un solo VideoReader per tutte le
        # regioni, non uno per regione) e gestisce n=1 a parte: la fascia
        # `linspace(lo, hi, 1)` di torch collassa sempre sul primo estremo
        # (il frame di INIZIO regione), che è arbitrario quanto il frame
        # centrale ma meno rappresentativo — per una sola cattura si preferisce
        # il frame centrale della regione.
        lo = math.ceil(max(0.0, min(a, max_duration)) * fps)
        hi = min(math.floor(max(0.0, min(b, max_duration)) * fps), last)
        if n <= 1:
            idxs = [round((lo + hi) / 2)]
        else:
            idxs = torch.linspace(lo, hi, n).round().long().tolist()
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                all_indices.append(idx)
        used_spans.append((a, b))

    all_indices.sort()
    total_span = sum(b - a for a, b in used_spans)
    sample_fps = len(all_indices) / total_span if total_span > 0 else 2.0

    tmp_dir = Path(tempfile.mkdtemp(prefix="zoom_multi_region_"))
    try:
        frame_paths = []
        for i, idx in enumerate(all_indices):
            idx = max(0, min(idx, last))
            p = tmp_dir / f"frame_{i:04d}.png"
            Image.fromarray(vr[idx].asnumpy()).save(p)
            frame_paths.append(str(p))
    except Exception:
        # Senza questo cleanup, un'eccezione qui lascerebbe la tmp_dir
        # orfana: la variabile che il chiamante userebbe per ripulirla
        # (`tmp_dir_to_cleanup`) non viene mai assegnata perché
        # l'unpacking del `return` di questa funzione non si completa.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    media = VideoFrames(frame_paths, max_pixels=budget.max_pixels, min_pixels=budget.min_pixels, sample_fps=sample_fps)
    return media, used_spans, tmp_dir


class EntropyAttentionResampleStrategy(Strategy):
    name = "entropy_attention_resample"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        self.concentration_threshold = float(cfg.get("concentration_threshold", 30.0))
        # --- selettore di ramo disperso/concentrato (docs/piano_zoom_multiregione_disgiunto.md §1.3) ---
        self.branch_selector = str(cfg.get("branch_selector", "concentration"))
        self.attention_entropy_threshold = float(cfg.get("attention_entropy_threshold", 0.7))
        self.window_frac = float(cfg.get("window_frac", 0.3))
        self.phase_shift_frac = float(cfg.get("phase_shift_frac", 0.5))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))
        # --- C1: zoom multi-regione disgiunto (docs/piano_zoom_multiregione_disgiunto.md) ---
        self.zoom_multi_region = bool(cfg.get("zoom_multi_region", False))
        self.n_regions = int(cfg.get("n_regions", 3))
        self.region_select = str(cfg.get("region_select", "adaptive"))
        self.region_mad_k = float(cfg.get("region_mad_k", 2.0))
        self.region_merge_gap_frac = float(cfg.get("region_merge_gap_frac", 0.05))
        self.region_window_frac = float(cfg.get("region_window_frac", 0.1))
        self.budget_allocation = str(cfg.get("budget_allocation", "proportional"))
        self.force_zoom_for_debug = bool(cfg.get("force_zoom_for_debug", False))

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
                f"{type(vlm).__name__} non lo fa — usa model=qwen2_5_vl_3b_attn "
                "o un'altra strategy."
            )

        media = self._build_media(video_path, frames, video_start, video_end, budget)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)

        # top1_pct/peak_cell del pass 1: calcolati una volta, loggati SEMPRE
        # (anche nei rami "accetta") per poter analizzare a posteriori come
        # si distribuiscono rispetto alla decisione di resampling. Si tiene
        # anche l'intera `ranked_cells` (non solo il top1) per riuso nel ramo
        # `zoom_multi_region` più sotto — evita di richiamare `sink_view` +
        # `rank_cells` una seconda volta sullo stesso tensore.
        top1_pct = peak_cell = attention_entropy_norm = top1_zscore = None
        ranked_cells: list[RankedCell] | None = None
        if signal.visual_attention is not None:
            ranked_cells = _ranked_cells(
                signal.visual_attention, percentile=self.sink_percentile, border=self.sink_border,
            )
            top1_pct, peak_cell = ranked_cells[0].pct, ranked_cells[0].cell
            attention_entropy_norm = _attention_entropy_norm(ranked_cells)
            top1_zscore = _top1_zscore(ranked_cells)

        final_media = media
        resampled = False
        resample_kind: str | None = None
        n_regions_used: int | None = None
        region_spans: list[tuple[float, float]] | None = None
        tmp_dir_to_cleanup: Path | None = None

        can_resample = (
            frames is None
            and options is not None
            and signal.answer_entropy is not None
            and top1_pct is not None
            and signal.answer_entropy > self.entropy_threshold
        )
        if can_resample:
            t = signal.visual_attention.t
            duration = _video_duration_sec(video_path)
            w0 = video_start if video_start is not None else 0.0
            w1 = video_end if video_end is not None else duration

            if self.branch_selector == "entropy":
                dispersed = attention_entropy_norm > self.attention_entropy_threshold
            elif self.branch_selector == "concentration":
                dispersed = top1_pct < self.concentration_threshold
            else:
                raise ValueError(
                    f"branch_selector sconosciuto: {self.branch_selector!r}. Valide: 'concentration', 'entropy'"
                )

            if not self.force_zoom_for_debug and dispersed:
                # Dispersa: sfasa la griglia uniforme di mezzo intervallo
                # rispetto al pass 1 — nuovi frame "in mezzo" a quelli già
                # visti, stesso budget. Clampata a [0, duration]: se il
                # pass 1 copriva già tutto il video, lo shift si limita
                # a restringere leggermente da sinistra.
                interval = (w1 - w0) / budget.nframes if budget.nframes else 0.0
                shift = interval * self.phase_shift_frac
                new_start = min(w0 + shift, duration)
                new_end = min(w1 + shift, duration)
                final_media = self._build_media(video_path, None, new_start, new_end, budget)
                resample_kind = "phase_shift"
            elif self.zoom_multi_region:
                # Concentrata (o `force_zoom_for_debug`): fino a `n_regions`
                # finestre disgiunte attorno alle celle a massa più alta,
                # invece di una sola finestra centrata sul picco — Opzione
                # C1 di `docs/piano_zoom_multiregione_disgiunto.md`. Riusa
                # `ranked_cells` già calcolato sopra per top1_pct/peak_cell.
                spans = _multi_region_spans(
                    ranked_cells, t=t, w0=w0, w1=w1, duration=duration,
                    n_regions=self.n_regions, region_select=self.region_select, region_mad_k=self.region_mad_k,
                    region_merge_gap_frac=self.region_merge_gap_frac, region_window_frac=self.region_window_frac,
                )
                counts = _allocate_region_budget([mass for _, _, mass in spans], budget.nframes, allocation=self.budget_allocation)
                final_media, region_spans, tmp_dir_to_cleanup = _build_multi_region_media(video_path, spans, counts, budget)
                n_regions_used = len(region_spans)
                resample_kind = "zoom_multi_region"
            else:
                # Concentrata ma comunque incerta: finestra stretta
                # centrata sull'istante del frame di picco (posizione
                # frazionaria della cella nella finestra del pass 1).
                frac = (peak_cell + 0.5) / t if t else 0.5
                peak_time = w0 + frac * (w1 - w0)
                half_w = self.window_frac * (w1 - w0) / 2
                new_start = max(0.0, peak_time - half_w)
                new_end = min(duration, peak_time + half_w)
                final_media = self._build_media(video_path, None, new_start, new_end, budget)
                resample_kind = "zoom_peak"

            resampled = True

            if self.capture_dir is not None:
                self._save_attention_capture(
                    vlm, video_path, prompt, signal.visual_attention, budget, w0, w1,
                    extra={
                        "resampled": resampled, "resample_kind": resample_kind, "top1_pct": top1_pct,
                        "attention_entropy_norm": attention_entropy_norm, "top1_zscore": top1_zscore,
                        "n_regions_used": n_regions_used, "region_spans": region_spans,
                    },
                )

        try:
            messages = vlm.build_messages(final_media, Text(prompt))
            raw = vlm.generate(messages, generation_config=gen_cfg)
        finally:
            if tmp_dir_to_cleanup is not None:
                shutil.rmtree(tmp_dir_to_cleanup, ignore_errors=True)

        result: dict = {
            "raw": raw,
            "answer_entropy": signal.answer_entropy,
            "top1_pct": top1_pct,
            "attention_entropy_norm": attention_entropy_norm,
            "top1_zscore": top1_zscore,
            "resampled": resampled,
            "resample_kind": resample_kind,
            "n_regions_used": n_regions_used,
            "region_spans": region_spans,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            # TODO: se `pred` è None (generate() non ha prodotto una lettera
            # parseabile), fare fallback a `signal.pred_letter` (argmax della
            # softmax ristretta del pass 1, `signal.answer_probs`) invece di
            # lasciare il sample unparseable — recupererebbe accuracy sui
            # sample oggi persi (vedi nota MVBench nel README, "52/3800
            # senza pred parseabile") senza costo aggiuntivo, il segnale è
            # già calcolato.
            result["pred"] = pred
        return result
