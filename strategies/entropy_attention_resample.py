"""`EntropyAttentionResampleStrategy` — primo tentativo di sfruttare i
segnali di entropia risposta MCQ + concentrazione attenzione (sink-
filtrata) per decidere se ricampionare i frame, invece di limitarsi a
loggarli. Pensata per un rilancio di MVBench a confronto con la baseline
in `README.md`.

Logica (pass 1 sempre uniforme, budget `nframes` invariato):

    entropia bassa                → accetta la risposta del pass 1
    entropia alta, ramo "disperso"    → ricampiona, SFASATO di fase rispetto
                                     ai frame già visti (dispersione: non
                                     si sa dove guardare, si prova altrove).
                                     Indici calcolati esplicitamente, budget
                                     `nframes-1` — vedi
                                     `_build_phase_shifted_media` per perché
                                     traslare la finestra non funziona.
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
con `concentration_threshold` assoluto; `"concentration_ratio"` (Stage C,
`docs/piano_zoom_multiregione_disgiunto.md` §1.3) confronta invece contro
una soglia adattiva al livello di caso (`concentration_ratio * 100/t`,
trasferibile fra `nframes`/dataset diversi — vedi "Soglia adattiva su
top1_pct" nel piano); `"entropy"` usa l'entropia di Shannon normalizzata
`H/log(t)` sulle masse per-cella sink-filtrate. Tre segnali di
concentrazione confrontati empiricamente (`top1_pct` grezzo, `H_norm`,
z-score del picco): `top1_pct` separa meglio degli altri due — vedi
"Risultati" nel piano per i numeri. Le soglie sono indipendenti e NON
tarate l'una sull'altra.

Al massimo UN ricampionamento (non c'è un terzo pass che ri-verifica il
segnale). La risposta FINALE — sia nei rami "accetta" sia dopo un
ricampionamento — viene da una vera `vlm.generate()`, MAI dal solo
`pred_letter` della softmax ristretta del pass 1, **tranne un fallback**:
se `generate()` non produce una lettera parseabile (`parse_mcq_letter`
ritorna `None` — vedi nota MVBench nel README, "52/3800 senza pred
parseabile"), si usa `signal.pred_letter` (già calcolato, nessun costo
aggiuntivo) invece di perdere il sample (`result["pred_fallback"]` logga
quando scatta). Questo rende i rami "accetta" NON più garantiti
bit-per-bit identici a `uniform` sul sottoinsieme (raro) di sample
altrimenti unparseable — un miglioramento netto (mai peggiora una
risposta già corretta), non un confondimento nell'analisi win/loss.

Soglie NON validate per MVBench (i numeri raccolti in
`docs/entropia_confidenza_mcq.md` vengono da un campione VNBench
"retrieval" diverso per dominio/nframes) — sono knob del preset
`strategy.entropy_attention_resample` in `conf/config.yaml`, tarabili da
CLI senza toccare codice, es. `strategy.entropy_threshold=1.0`.

Richiede un modello che implementa `SupportsSignals` (i preset `*_attn`:
`qwen2_5_vl_3b_attn`, `qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`) — fail-fast altrimenti, stesso pattern di
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
from utils.attn_core import RankedCell, ranked_cells_from_attention, reconstruct_frame_indices
from utils.mcq import parse_mcq_letter

from .base import SamplingBudget, Strategy, video_duration_sec

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer


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


def _cell_mass_stats(ranked: list[RankedCell]) -> tuple[float, float]:
    """Media e dev.std (di popolazione) delle masse `pct` di TUTTE le `t`
    celle — ingredienti grezzi condivisi da `_top1_zscore` e da qualunque
    normalizzazione alternativa si voglia provare offline (es. un
    ratio robusto mediana/MAD) senza dover ricatturare: loggati anche da
    soli in `answer()`, non solo lo z-score derivato.
    """
    pcts = [c.pct for c in ranked]
    mean = statistics.fmean(pcts)
    stdev = statistics.pstdev(pcts, mu=mean)
    return mean, stdev


def _top1_zscore(ranked: list[RankedCell], mean: float, stdev: float) -> float:
    """Z-score DELLA CELLA DI PICCO rispetto alla distribuzione delle masse
    (`pct`) di TUTTE le `t` celle dello stesso video (`docs/
    piano_zoom_multiregione_disgiunto.md`, sezione soglia adattiva): quanto
    `top1_pct` è un outlier rispetto alla media/dev.std delle masse di
    QUESTO video, non rispetto a un livello di caso teorico fisso (`100/t`)
    o a soglie assolute tarate su un altro dataset/`nframes`. Nessuna
    calibrazione esterna: media e dev.std sono per-sample (`_cell_mass_
    stats`), quindi il segnale si adatta automaticamente a `t` e alla forma
    reale della distribuzione, incluso il caso concentrato (poche celle
    alte, coda lunga di celle quasi a zero) che rende `pct` fortemente
    non-normale — lo z-score resta comunque un indice valido di "quanto
    spicca" il picco, non un test di normalità.

    Degenera a `0.0` se `t <= 1` (nessuna cella di confronto, `mean`/`stdev`
    non significativi) o se `stdev` è nulla (tutte le celle uguali, incluso
    il caso massa totale nulla) — nessuna cella spicca rispetto alle altre.
    """
    if len(ranked) <= 1 or stdev <= 0:
        return 0.0
    return (ranked[0].pct - mean) / stdev


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


def _build_phase_shifted_media(
    video_path: str,
    budget: SamplingBudget,
    video_start: float | None,
    video_end: float | None,
    phase_shift_frac: float,
) -> tuple[VideoFrames, list[int], Path]:
    """Frame SFASATI di `phase_shift_frac` di intervallo rispetto a quelli
    del pass 1, estratti per indice esplicito.

    **Perché non basta traslare la finestra** (`video_start`/`video_end`) e
    lasciar ricampionare `Video`, che era l'implementazione precedente:
    `reconstruct_frame_indices` fa `linspace(lo, hi, nframes)` con ENTRAMBI
    gli estremi inclusi, quindi una finestra traslata a destra e poi clampata
    a fine video (`w1 == duration` — sempre, su Video-MME, che non passa mai
    un trim) non trasla affatto: si comprime. Lo sfasamento risultante decade
    linearmente da mezzo intervallo a ZERO sull'ultimo frame, che resta
    identico a quello del pass 1; a `nframes=24` metà dei frame ricadono
    entro un quarto di intervallo da quelli già visti e lo shift medio è il
    48% di quello dichiarato. Con la finestra intera non c'è spazio per
    traslare né a destra né a sinistra: l'unico modo di esprimere davvero uno
    sfasamento è calcolare gli indici a mano.

    Si campionano gli `nframes - 1` punti interni fra frame consecutivi del
    pass 1 (`x_i + frac * (x_{i+1} - x_i)`): tutti sfasati della STESSA
    frazione di intervallo, nessuno coincidente col pass 1. Il budget scende
    di un frame — è il prezzo dell'uniformità (a `nframes=24`, il 4%), e
    l'alternativa (tenerne `nframes` clampando l'ultimo a fine video) fa
    degenerare proprio il frame che il bug precedente già sprecava.

    `sample_fps` è iniettato come in `Strategy._sample_doubled_frames`: con
    una lista di frame `fetch_video` non ha timestamp reali e usa questo
    scalare per `second_per_grid_ts` (M-RoPE). Vale
    `n_frame_estratti * fps / span_frames`, raddoppiato nel regime
    `double_frames` (lì una cella copre un frame invece di due).

    Ritorna `(media, indici assoluti usati, tmpdir dei PNG)` — il chiamante
    ripulisce la tmpdir dopo la `generate()`.
    """
    import decord

    frame_indices, fps = reconstruct_frame_indices(
        video_path, budget.nframes, video_start=video_start, video_end=video_end,
    )
    # `or frame_indices`: con nframes <= 1 non esistono intervalli da sfasare
    # (zip su lista di 1 elemento è vuoto) — si ricade sui frame del pass 1.
    shifted = [
        round(a + phase_shift_frac * (b - a))
        for a, b in zip(frame_indices, frame_indices[1:])
    ] or list(frame_indices)

    vr = decord.VideoReader(video_path)
    last = len(vr) - 1

    span_frames = shifted[-1] - shifted[0] + 1
    sample_fps = len(shifted) * fps / span_frames if span_frames > 0 else 2.0
    if budget.double_frames:
        sample_fps *= 2.0

    tmp_dir = Path(tempfile.mkdtemp(prefix="phase_shift_"))
    try:
        frame_paths = []
        for i, idx in enumerate(shifted):
            p = tmp_dir / f"frame_{i:04d}.png"
            Image.fromarray(vr[max(0, min(int(idx), last))].asnumpy()).save(p)
            frame_paths.append(str(p))
    except Exception:
        # Stesso motivo del cleanup in `_build_multi_region_media`: il
        # chiamante non riceve mai la tmpdir se l'eccezione precede il return.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if budget.double_frames:
        frame_paths = [p for p in frame_paths for _ in range(2)]
    media = VideoFrames(
        frame_paths, max_pixels=budget.max_pixels, min_pixels=budget.min_pixels, sample_fps=sample_fps,
    )
    return media, shifted, tmp_dir


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
    if budget.double_frames:
        # Coerenza col regime scelto: se il pass 1 ha visto i frame
        # raddoppiati (1 cella = 1 frame), anche il pass 2 deve, altrimenti
        # le celle cambiano significato fra i due pass e questo ramo costa
        # metà degli altri. `sample_fps` raddoppia con la stessa regola di
        # `Strategy._sample_doubled_frames` (la cella copre un frame invece
        # di due).
        sample_fps *= 2.0

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

    if budget.double_frames:
        frame_paths = [p for p in frame_paths for _ in range(2)]
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
        # soglia adattiva su top1_pct normalizzata per livello di caso (100/t):
        # `dispersed = top1_pct < concentration_ratio * (100/t)` invece di un
        # valore assoluto fisso — trasferibile fra config con `t` diverso
        # (nframes/dataset), vedi "Soglia adattiva su top1_pct" nel piano.
        # 3.85 = soglia calibrata su Video-MME nframes=128 (Stage C).
        self.concentration_ratio = float(cfg.get("concentration_ratio", 3.85))
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
                f"{type(vlm).__name__} non lo fa — usa uno dei preset con cattura "
                "(qwen2_5_vl_3b_attn, qwen3_vl_2b_attn, qwen3_vl_4b_attn) "
                "o un'altra strategy."
            )

        # `tmp_dirs`: directory di PNG estratti da ripulire DOPO la generate()
        # finale — mai prima, perché `final_media` può essere ancora la `media`
        # del pass 1 (rami "accetta"). Ci finiscono sia le tmpdir del regime
        # frame-doubling (`Strategy._build_media`) sia quella dello zoom
        # multi-regione (`_build_multi_region_media`).
        tmp_dirs: list[Path] = []
        media, media_tmp_dir = self._build_media(video_path, frames, video_start, video_end, budget)
        if media_tmp_dir is not None:
            tmp_dirs.append(media_tmp_dir)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)

        # top1_pct/peak_cell del pass 1: calcolati una volta, loggati SEMPRE
        # (anche nei rami "accetta") per poter analizzare a posteriori come
        # si distribuiscono rispetto alla decisione di resampling. Si tiene
        # anche l'intera `ranked_cells` (non solo il top1) per riuso nel ramo
        # `zoom_multi_region` più sotto — evita di richiamare `sink_view` +
        # `rank_cells` una seconda volta sullo stesso tensore.
        top1_pct = peak_cell = attention_entropy_norm = top1_zscore = None
        t_cells = cell_mass_mean = cell_mass_std = None
        ranked_cells: list[RankedCell] | None = None
        if signal.visual_attention is not None:
            ranked_cells = ranked_cells_from_attention(
                signal.visual_attention, percentile=self.sink_percentile, border=self.sink_border,
            )
            top1_pct, peak_cell = ranked_cells[0].pct, ranked_cells[0].cell
            attention_entropy_norm = _attention_entropy_norm(ranked_cells)
            # `t_cells`/`cell_mass_mean`/`cell_mass_std` loggati SEMPRE (non solo
            # `top1_zscore` derivato): permettono di ricalcolare offline sia il
            # ratio-a-livello-di-caso (100/t_cells) sia normalizzazioni diverse
            # dallo z-score, senza dover ricatturare (vedi docs/piano_zoom_
            # multiregione_disgiunto.md, sezione soglia adattiva).
            t_cells = signal.visual_attention.t
            cell_mass_mean, cell_mass_std = _cell_mass_stats(ranked_cells)
            top1_zscore = _top1_zscore(ranked_cells, cell_mass_mean, cell_mass_std)

        final_media = media
        resampled = False
        resample_kind: str | None = None
        n_regions_used: int | None = None
        region_spans: list[tuple[float, float]] | None = None
        # Numero di frame effettivamente passati al pass 2 quando differisce da
        # `budget.nframes` (oggi solo `phase_shift`, che ne usa nframes-1 —
        # vedi `_build_phase_shifted_media`). `None` nei rami "accetta".
        n_frames_resampled: int | None = None

        can_resample = (
            frames is None
            and options is not None
            and signal.answer_entropy is not None
            and top1_pct is not None
            and signal.answer_entropy > self.entropy_threshold
        )
        if can_resample:
            t = signal.visual_attention.t
            duration = video_duration_sec(video_path)
            w0 = video_start if video_start is not None else 0.0
            w1 = video_end if video_end is not None else duration

            if self.branch_selector == "entropy":
                dispersed = attention_entropy_norm > self.attention_entropy_threshold
            elif self.branch_selector == "concentration":
                dispersed = top1_pct < self.concentration_threshold
            elif self.branch_selector == "concentration_ratio":
                chance = 100.0 / t if t else 0.0
                dispersed = top1_pct < self.concentration_ratio * chance
            else:
                raise ValueError(
                    f"branch_selector sconosciuto: {self.branch_selector!r}. Valide: "
                    "'concentration', 'concentration_ratio', 'entropy'"
                )

            if not self.force_zoom_for_debug and dispersed:
                # Dispersa: sfasa la griglia uniforme di `phase_shift_frac` di
                # intervallo rispetto al pass 1 — nuovi frame "in mezzo" a
                # quelli già visti. Gli indici sono calcolati esplicitamente,
                # NON traslando la finestra: su una finestra che copre già
                # tutto il video la traslazione viene clampata e si riduce a
                # una compressione, con shift che decade a zero sull'ultimo
                # frame (vedi `_build_phase_shifted_media`).
                final_media, shifted_indices, shift_tmp_dir = _build_phase_shifted_media(
                    video_path, budget, video_start, video_end, self.phase_shift_frac,
                )
                tmp_dirs.append(shift_tmp_dir)
                n_frames_resampled = len(shifted_indices)
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
                final_media, region_spans, region_tmp_dir = _build_multi_region_media(video_path, spans, counts, budget)
                tmp_dirs.append(region_tmp_dir)
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
                final_media, zoom_tmp_dir = self._build_media(video_path, None, new_start, new_end, budget)
                if zoom_tmp_dir is not None:
                    tmp_dirs.append(zoom_tmp_dir)
                resample_kind = "zoom_peak"

            resampled = True

            if self.capture_dir is not None:
                self._save_attention_capture(
                    vlm, video_path, prompt, signal.visual_attention, budget, w0, w1,
                    extra={
                        "resampled": resampled, "resample_kind": resample_kind, "top1_pct": top1_pct,
                        "attention_entropy_norm": attention_entropy_norm, "top1_zscore": top1_zscore,
                        "peak_cell": peak_cell, "t_cells": t_cells,
                        "cell_mass_mean": cell_mass_mean, "cell_mass_std": cell_mass_std,
                        "n_regions_used": n_regions_used, "region_spans": region_spans,
                    },
                )

        try:
            messages = vlm.build_messages(final_media, Text(prompt))
            raw = vlm.generate(messages, generation_config=gen_cfg)
        finally:
            for tmp_dir in tmp_dirs:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        result: dict = {
            "raw": raw,
            "answer_entropy": signal.answer_entropy,
            "top1_pct": top1_pct,
            "attention_entropy_norm": attention_entropy_norm,
            "top1_zscore": top1_zscore,
            "peak_cell": peak_cell,
            "t_cells": t_cells,
            "cell_mass_mean": cell_mass_mean,
            "cell_mass_std": cell_mass_std,
            "resampled": resampled,
            "resample_kind": resample_kind,
            "n_regions_used": n_regions_used,
            "region_spans": region_spans,
            "n_frames_resampled": n_frames_resampled,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            pred_fallback = False
            if pred is None and signal.pred_letter is not None:
                # generate() (pass 2) non ha prodotto una lettera parseabile:
                # invece di perdere il sample, fallback all'argmax della
                # softmax ristretta del pass 1 (`signal.pred_letter`, già
                # calcolato — nessun costo aggiuntivo). Recupera accuracy sui
                # sample oggi persi (vedi nota MVBench nel README, "52/3800
                # senza pred parseabile"). `letters` è lo stesso elenco
                # passato come `answer_letters` a `generate_with_signals`
                # sopra, quindi `signal.pred_letter` è garantito esserci.
                pred = letters.index(signal.pred_letter)
                pred_fallback = True
            result["pred"] = pred
            result["pred_fallback"] = pred_fallback
        return result
