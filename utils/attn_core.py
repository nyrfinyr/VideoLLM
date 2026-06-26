"""DTO + rendering condivisi per l'analisi dell'attenzione entity→visivo.

Sia il tool di cattura (`attn_explorer.capture`) sia il server Flask
(`attn_explorer.serve`) importano da qui: i `dataclass(frozen=True)` sono la
forma canonica del dump entity-AGNOSTICO (una mappa di attenzione per OGNI
token della domanda, già mediata su layer centrali e teste), e le funzioni di
overlay/ranking sono condivise per non duplicare logica fra cattura e serving.

Il dump è entity-agnostico apposta: l'aggregazione su una specifica selezione
di token (le "entity") avviene a RUNTIME via `AttentionCapture.aggregate`, così
le entity si possono variare senza rifare il forward del modello.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# DTO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GridSpec:
    """Geometria della griglia visiva e mappatura cella→frame reale.

    `t` celle temporali, ciascuna `grid_h × grid_w` token visivi (post-merger).
    `double` distingue i due regimi di campionamento (vedi
    `attn_explorer.capture`): col frame-doubling ogni cella è UN frame, altrimenti
    il merge a coppie fa sì che una cella copra DUE frame consecutivi.
    """
    t: int
    grid_h: int
    grid_w: int
    double: bool

    @property
    def regime(self) -> str:
        return "frame-doubling (cella=1 frame)" if self.double else "merge a coppie (cella=2 frame)"

    def cell_to_frames(self, ti: int, frame_indices) -> tuple[list[int], int]:
        """(frame reali della cella `ti`, frame rappresentante per l'estrazione)."""
        if self.double:
            f = int(frame_indices[ti])
            return [f], f
        f0, f1 = int(frame_indices[2 * ti]), int(frame_indices[2 * ti + 1])
        return [f0, f1], int(round((f0 + f1) / 2))


@dataclass(frozen=True)
class QueryToken:
    """Un token della domanda, selezionabile come 'entity' a runtime.

    `index` è la posizione ASSOLUTA nei `input_ids`; `row` è la riga
    corrispondente in `AttentionCapture.attn` (0-based, nell'ordine in cui i
    token compaiono nella domanda); `token` è il token grezzo (con marker
    ▁/Ġ), `text` la sua resa leggibile decodificata.
    """
    row: int
    index: int
    token: str
    text: str


@dataclass(frozen=True)
class VisualAttention:
    """Output del forward di `Qwen25VLAttention.full_visual_attention`.

    Forma grezza appena estratta dal modello: una heatmap per OGNI token della
    domanda, ancora senza la nozione di frame-doubling (che è del tool di
    cattura, non del modello). Il chiamante la trasforma in `AttentionCapture`
    aggiungendo `fps`, `frame_indices` e il regime `double` della GridSpec.

    `sink_map` `[t, grid_h, grid_w]` è il punteggio sink per token visivo (vedi
    `Qwen25VLAttention`: max sui sink dims normalizzato RMS, media sui layer
    centrali), già disposto sulla stessa griglia di `attn`. `sink_dims` sono
    i canali usati per calcolarlo (utile al server per ispezionarli).
    """
    attn: torch.Tensor  # [n_q, t, grid_h, grid_w]
    sink_map: torch.Tensor  # [t, grid_h, grid_w]
    sink_dims: tuple[int, ...]
    t: int
    grid_h: int
    grid_w: int
    query_tokens: tuple[QueryToken, ...]
    query_span: tuple[int, int]
    visual_span: tuple[int, int]
    input_ids: torch.Tensor

    def __post_init__(self):
        if self.sink_map.shape != (self.t, self.grid_h, self.grid_w):
            raise ValueError(
                f"sink_map shape {tuple(self.sink_map.shape)} != "
                f"(t, grid_h, grid_w)={(self.t, self.grid_h, self.grid_w)}"
            )


@dataclass(frozen=True)
class EntityAttention:
    """Output di `Qwen25VLAttention.entity_visual_attention` (singola entity).

    `heatmap` `[t, grid_h, grid_w]` è la media delle righe-query `rows` dei
    token dell'entity.
    """
    heatmap: torch.Tensor
    rows: tuple[int, ...]
    entity_span: tuple[int, int]
    visual_span: tuple[int, int]
    t: int
    grid_h: int
    grid_w: int
    query_tokens: tuple[QueryToken, ...]


@dataclass(frozen=True)
class AttentionCapture:
    """Dump entity-agnostico dell'attenzione testo→visivo di UN video.

    `attn` è `[n_q, t, grid_h, grid_w]` float32 su cpu: per ogni token della
    domanda (`query_tokens[r]` ↔ `attn[r]`) la mappa di attenzione sui token
    visivi, già mediata su layer centrali e teste. L'analisi a runtime
    seleziona un sottoinsieme di righe (le entity) e le media via `aggregate`.

    `sink_map` `[t, grid_h, grid_w]` è il punteggio sink per token visivo (max
    sui `sink_dims` normalizzato RMS, media sui layer centrali) sulla stessa
    griglia di `attn`: identifica i token visivi che assorbono attenzione
    spuria. La soglia percentile (es. 25°) e l'azzeramento si applicano a
    RUNTIME nel server, così si può variare senza rifare il forward.
    """
    video_path: str
    prompt: str
    fps: float
    frame_indices: tuple[int, ...]
    grid: GridSpec
    query_tokens: tuple[QueryToken, ...]
    attn: torch.Tensor  # [n_q, t, grid_h, grid_w]
    sink_map: torch.Tensor  # [t, grid_h, grid_w]
    sink_dims: tuple[int, ...]

    def __post_init__(self):
        if self.sink_map.shape != (self.grid.t, self.grid.grid_h, self.grid.grid_w):
            raise ValueError(
                f"sink_map shape {tuple(self.sink_map.shape)} != "
                f"(t, grid_h, grid_w)="
                f"{(self.grid.t, self.grid.grid_h, self.grid.grid_w)}"
            )

    def aggregate(self, rows) -> torch.Tensor:
        """Heatmap `[t, grid_h, grid_w]` mediata sulle righe-query `rows`.

        `rows` sono indici 0-based in `attn` (i `QueryToken.row` selezionati).
        Con `rows` vuoto ripiega sulla media di TUTTE le righe (ordinamento di
        default, "attenzione media della domanda").
        """
        sel = self.attn if not rows else self.attn[list(rows)]
        return sel.float().mean(dim=0)


@dataclass(frozen=True)
class RankedCell:
    """Una cella temporale classificata per massa di attenzione."""
    rank: int
    cell: int
    score: float
    pct: float
    frames: tuple[int, ...]
    rep_frame: int
    t_sec: float
    is_top: bool

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "cell": self.cell,
            "score": self.score,
            "pct": self.pct,
            "frames": list(self.frames),
            "rep_frame": self.rep_frame,
            "t_sec": self.t_sec,
            "is_top": self.is_top,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ranking celle (ex debug_attention.rank_and_extract, senza I/O PNG)
# ─────────────────────────────────────────────────────────────────────────────
def rank_cells(
    heatmap: torch.Tensor,
    grid: GridSpec,
    frame_indices,
    fps: float,
    *,
    metric: str = "sum",
    topk: int = 8,
) -> list[RankedCell]:
    """Classifica le celle temporali di `heatmap` `[t, grid_h, grid_w]`.

    `metric`: 'sum' = massa totale della cella, 'max' = picco. Restituisce la
    lista ordinata per score decrescente; `topk` marca solo le prime `topk`
    celle come `is_top` (non limita l'estrazione).
    """
    if metric == "sum":
        scores = heatmap.flatten(1).sum(dim=1)
    else:
        scores = heatmap.flatten(1).max(dim=1).values

    order = torch.argsort(scores, descending=True).tolist()
    total = scores.sum().item()

    ranked: list[RankedCell] = []
    for rank, ti in enumerate(order):
        frames, rep = grid.cell_to_frames(ti, frame_indices)
        sc = scores[ti].item()
        ranked.append(RankedCell(
            rank=rank,
            cell=ti,
            score=sc,
            pct=(100 * sc / total if total else 0.0),
            frames=tuple(frames),
            rep_frame=rep,
            t_sec=rep / fps if fps else 0.0,
            is_top=rank < topk,
        ))
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Overlay heatmap su frame (ex debug_attention._hot_colormap/make_overlay)
# ─────────────────────────────────────────────────────────────────────────────
def hot_colormap(x: np.ndarray) -> np.ndarray:
    """Colormap 'hot' (nero→rosso→giallo→bianco) su `x` in [0,1] → RGB uint8.

    Stessi segmenti del 'hot' di matplotlib, a mano (niente matplotlib/cv2).
    """
    r = np.clip(8 / 3 * x, 0, 1)
    g = np.clip(8 / 3 * x - 1, 0, 1)
    b = np.clip(4 * x - 3, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def make_overlay(
    frame: np.ndarray,
    cell_map: torch.Tensor,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    zero_border: int = 0,
    alpha: float = 0.6,
) -> Image.Image:
    """Overlay della heatmap di UNA cella temporale sul frame RGB.

    `cell_map` è `[grid_h, grid_w]`; viene normalizzata (su `vmin`/`vmax`
    globali se forniti, così frame poco attenzionati restano scuri), upscalata
    BICUBIC alla risoluzione del frame e alpha-blendata col 'hot'. L'alpha è
    proporzionale all'attenzione → le zone fredde restano originali.
    """
    m = cell_map.float().cpu().numpy().copy()
    if zero_border > 0:
        b = zero_border
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    mn = m.min() if vmin is None else vmin
    mx = m.max() if vmax is None else vmax
    m = (m - mn) / (mx - mn) if mx > mn else np.zeros_like(m)
    m = np.clip(m, 0, 1)

    H, W = frame.shape[:2]
    up = np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((W, H), Image.BICUBIC))
    up = np.clip(up, 0, 1)

    color = hot_colormap(up).astype(np.float32)
    a = (up * alpha)[..., None]
    blended = (1 - a) * frame.astype(np.float32) + a * color
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Maschera sink (percentile + bordi) — per filtrare i sink a runtime
# ─────────────────────────────────────────────────────────────────────────────
def sink_mask(
    sink_map: torch.Tensor,
    *,
    percentile: float = 25.0,
    border: int = 2,
) -> torch.Tensor:
    """Identifica i token visivi-sink dalla `sink_map` `[t, grid_h, grid_w]`.

    Un token è sink se il suo `sink_score` è >= percentile-esimo della
    distribuzione (su tutto `sink_map`). Inoltre i primi/ultimi `border`
    anelli di righe e colonne sono forzati a sink (i token di bordo sono
    tipicamente sink anche se il percentile non li cattura). Ricalca il
    filtro di `lot/retrieval.py` (Xiao et al., "sink dims").

    Args:
        sink_map: `[t, grid_h, grid_w]` torch/numpy tensor.
        percentile: soglia percentile (default 25 — i top-25% per sink score
            sono sink). Range 0-100.
        border: anelli di bordo forzati a sink (solo se
            `grid_h > 2*border` e `grid_w > 2*border`, altrimenti saltato).

    Returns:
        `bool tensor` `[t, grid_h, grid_w]`: True = sink (da azzerare dalla
        heatmap quando si filtra).
    """
    m = torch.as_tensor(sink_map).float().cpu()
    if m.ndim != 3:
        raise ValueError(f"sink_map deve essere [t,g h,gw], got {tuple(m.shape)}")
    t, gh, gw = m.shape
    thr = torch.quantile(m.flatten(), percentile / 100.0)
    is_sink = m >= thr
    if border > 0 and gh > 2 * border and gw > 2 * border:
        is_sink[:, :border, :] = True
        is_sink[:, -border:, :] = True
        is_sink[:, :, :border] = True
        is_sink[:, :, -border:] = True
    return is_sink


def filter_sinks(
    heatmap: torch.Tensor,
    sink_map: torch.Tensor,
    *,
    percentile: float = 25.0,
    border: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azzera le celle-sink di `heatmap` e renormalizza; ritorna (filtrata, mask).

    `heatmap` è `[t, grid_h, grid_w]` già aggregato per una selezione di
    token-query. La maschera è calcolata da `sink_map` via `sink_mask`,
    applicata alla `heatmap` (dove is_sink → 0), e la versione filtrata è
    renormalizzata su [0, max] così gli overlay restano confrontabili.
    """
    m = torch.as_tensor(heatmap).float().cpu()
    mask = sink_mask(sink_map, percentile=percentile, border=border)
    out = m.clone()
    out[mask] = 0.0
    mx = out.max()
    if mx > 0:
        out = out / mx
    return out, mask


# ─────────────────────────────────────────────────────────────────────────────
# Persistenza dell'AttentionCapture (torch.save/load)
# ─────────────────────────────────────────────────────────────────────────────
def save_capture(cap: AttentionCapture, path: Path) -> None:
    """Serializza un `AttentionCapture` su disco (`capture.pt`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "video_path": cap.video_path,
        "prompt": cap.prompt,
        "fps": cap.fps,
        "frame_indices": list(cap.frame_indices),
        "grid": {
            "t": cap.grid.t, "grid_h": cap.grid.grid_h,
            "grid_w": cap.grid.grid_w, "double": cap.grid.double,
        },
        "query_tokens": [(q.row, q.index, q.token, q.text) for q in cap.query_tokens],
        "attn": cap.attn,
        "sink_map": cap.sink_map,
        "sink_dims": list(cap.sink_dims),
    }, path)


def load_capture(path: Path) -> AttentionCapture:
    """Ricarica un `AttentionCapture` da `capture.pt`.

    Tieni `sink_map`/`sink_dims` opzionali per leggere dump legacy senza
    queli field (li si ricostruisce come zeri / lista vuota).
    """
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    g = d["grid"]
    sink_map = d.get("sink_map")
    if sink_map is None:
        sink_map = torch.zeros(g["t"], g["grid_h"], g["grid_w"])
    return AttentionCapture(
        video_path=d["video_path"],
        prompt=d["prompt"],
        fps=float(d["fps"]),
        frame_indices=tuple(d["frame_indices"]),
        grid=GridSpec(g["t"], g["grid_h"], g["grid_w"], g["double"]),
        query_tokens=tuple(QueryToken(*t) for t in d["query_tokens"]),
        attn=d["attn"],
        sink_map=sink_map,
        sink_dims=tuple(d.get("sink_dims", [])),
    )
