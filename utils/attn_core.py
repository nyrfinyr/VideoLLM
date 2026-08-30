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

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


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


def _rows_before_marker(query_tokens: tuple[QueryToken, ...], marker: str) -> list[int]:
    """Righe dei token che precedono `marker` nel testo ricostruito.

    Il testo si ricostruisce concatenando i `QueryToken.text` in ordine e si
    cerca `marker` (già lowercase) su quello, non sui token: un marcatore può
    cadere a cavallo di due token e non comparire in nessuno dei due. Se il
    marcatore non c'è ritorna TUTTE le righe, che è l'unico fallback sicuro —
    `aggregate` con tutte le righe è il comportamento di default.
    """
    texts = [qt.text for qt in query_tokens]
    at = "".join(texts).lower().find(marker)
    if at == -1:
        return [qt.row for qt in query_tokens]
    rows: list[int] = []
    pos = 0
    for qt, text in zip(query_tokens, texts):
        if pos >= at:
            break
        rows.append(qt.row)
        pos += len(text)
    return rows


# Coda del prompt MCQ senza alcun referente visivo: l'istruzione di formato.
# `evals.base.format_mcq_prompt` la scrive sempre così.
BOILERPLATE_MARKER = "answer with the letter"


def question_and_options_rows(query_tokens: tuple[QueryToken, ...]) -> list[int]:
    """Righe di domanda + opzioni: via solo il boilerplate e il generation prompt.

    Selezione intermedia fra "tutte le righe" e `question_rows`. Le opzioni
    RESTANO: contengono referenti visivi legittimi ("the toolbox", "the van")
    e la loro attenzione può essere informativa. Quello che non ne ha è
    l'istruzione di formato ("Answer with the letter of the correct option
    only.") e il generation prompt del chat template che la segue.

    Usata dal knob `query_rows=qopts` di `attention_highlight` e dal rowset
    omonimo di `attn_explorer.diag_query_rows`: la stessa implementazione per
    entrambi, altrimenti la diagnostica offline misurerebbe una selezione
    diversa da quella che gira nell'eval.
    """
    return _rows_before_marker(query_tokens, BOILERPLATE_MARKER)


def question_rows(query_tokens: tuple[QueryToken, ...]) -> list[int]:
    """Righe dei token che compongono SOLO la domanda, non l'MCQ scaffolding.

    I prompt del progetto (sia `format_mcq_prompt` sia i sidecar `--prompt-
    from-json`) hanno sempre la forma `<domanda>\\nOptions:\\n(A) ...`: si
    ricostruisce il testo concatenando `QueryToken.text` in ordine e si
    ferma PRIMA del token che introduce il marcatore "Options:" (case-
    insensitive). Se il marcatore non compare (prompt non-MCQ) ritorna tutte
    le righe, equivalente al comportamento di default di `aggregate`.

    Usata per pre-selezionare la entity di default nella pagina video: senza
    questo, "nessuna selezione" media anche i token delle opzioni di
    risposta e del boilerplate ("Answer with the letter..."), diluendo
    l'attenzione sulla domanda vera e propria. È anche il knob
    `query_rows=question` di `attention_highlight`.
    """
    return _rows_before_marker(query_tokens, "options:")


# Selettori di righe-query esposti come knob (`attention_highlight`) e come
# rowset della diagnostica (`attn_explorer.diag_query_rows`). `all` è `[]`,
# cioè il fallback "tutte le righe" di `AttentionCapture.aggregate`.
ROW_SELECTORS = {
    "all": lambda qt: [],
    "qopts": question_and_options_rows,
    "question": question_rows,
}

# Valori ammessi per il knob `query_rows` delle strategy: i selettori puri più
# `entity`, che NON può stare in `ROW_SELECTORS` perché non è una funzione pura
# di `query_tokens` — richiede una chiamata al decoder per-sample (vedi
# `resolve_query_rows`). Le strategy validano contro QUESTO set, non contro
# `ROW_SELECTORS`.
QUERY_ROWS_CHOICES = frozenset(ROW_SELECTORS) | {"entity"}


@dataclass(frozen=True)
class QueryRowsResolution:
    """Esito di `resolve_query_rows`: righe esplicite + metadati per-sample.

    `rows` segue la convenzione di `AttentionCapture.aggregate`: `[]` = tutte
    le righe. `mode` è la selezione EFFETTIVA (può differire dal knob: con
    `entity` non mappata verbatim si degrada a `"question"`). I campi
    `entity_*` sono valorizzati solo quando il knob era `entity` — vanno
    loggati per-sample così l'eval quantifica il tasso di mapping riuscito.
    """
    rows: list[int]
    mode: str
    entity_raw: str | None = None
    entity_phrases: tuple[str, ...] | None = None
    entity_mapped: bool | None = None


def entity_rows(query_tokens: tuple[QueryToken, ...], phrases: list[str]) -> list[int]:
    """Righe dei token che coprono i match VERBATIM di `phrases` nella domanda.

    Stesso string-match (case-insensitive, su offset di carattere) validato
    dal probe `scripts/probe_entity_extraction.py`: se il decoder copia le
    parole della domanda alla lettera, il mapping è esatto e gratuito. La
    ricerca è ristretta alla porzione PRIMA di "options:" — le stesse parole
    possono comparire anche nelle opzioni, e matchare lì selezionerebbe righe
    fuori dalla domanda. Frasi che non matchano (parafrasi/sinonimi) vengono
    semplicemente saltate; `[]` = nessuna frase mappata (il chiamante decide
    il fallback, vedi `resolve_query_rows`).
    """
    text = ""
    spans: list[tuple[int, int, int]] = []
    for qt in query_tokens:
        start = len(text)
        text += qt.text
        spans.append((start, len(text), qt.row))
    lowered = text.lower()
    q_end = lowered.find("options:")
    if q_end == -1:
        q_end = len(lowered)
    rows: list[int] = []
    for phrase in phrases:
        ph = phrase.strip().strip(".,?!:;'\"").lower()
        if not ph:
            continue
        i = lowered.find(ph, 0, q_end)
        if i == -1:
            continue
        j = i + len(ph)
        rows.extend(r for (s, e, r) in spans if s < j and e > i and r not in rows)
    return sorted(rows)


def resolve_query_rows(
    query_rows: str,
    query_tokens: tuple[QueryToken, ...],
    *,
    extract_entities=None,
) -> QueryRowsResolution:
    """Risolve il knob `query_rows` in righe esplicite, `entity` compreso.

    Punto d'ingresso UNICO delle strategy (ortogonale all'arm: la logica vive
    qui, le strategy passano solo la stringa del knob): per i valori di
    `ROW_SELECTORS` delega al selettore puro; per `entity` chiama
    `extract_entities` (il metodo del modello GIÀ CARICATO, es.
    `Qwen.extract_entities` — nessun secondo modello in memoria) sulla SOLA
    domanda, ricostruita dai `query_tokens` tagliando su "options:" (o sul
    boilerplate), e mappa l'output alle righe via `entity_rows`. Le opzioni
    restano fuori dall'input dell'estrattore di proposito: vedi il commento
    al taglio, più sotto.

    Fallback (deciso, non silenzioso): se il decoder risponde `NONE`
    (domanda senza soggetto specifico) o se la sequenza estratta non mappa
    verbatim (parafrasi, o output multi-frase fuori formato), si degrada a
    `question_rows` con `entity_mapped=False` nei metadati — il sample non è
    perso e l'eval può contare quanto spesso succede e perché
    (`entity_raw='NONE'` distingue i due casi). Le righe risultanti restano soggette
    all'asserzione anti-degrado delle strategy (`n_query_rows <
    n_query_tokens`), che copre anche il fallback-del-fallback di
    `question_rows` (prompt senza "options:").
    """
    if query_rows in ROW_SELECTORS:
        return QueryRowsResolution(rows=ROW_SELECTORS[query_rows](query_tokens), mode=query_rows)
    if query_rows != "entity":
        raise ValueError(
            f"query_rows sconosciuto: {query_rows!r}. Valide: {sorted(QUERY_ROWS_CHOICES)}"
        )
    if extract_entities is None:
        raise ValueError(
            "query_rows='entity' richiede un estrattore (es. `vlm.extract_entities`): "
            "il modello caricato non lo espone?"
        )
    # SOLO la domanda al decoder, senza opzioni (scelta 2026-08-30, dopo il
    # probe 93525_0): con le opzioni in input il prior "MCQ → rispondi" del
    # 2B vinceva 2 volte su 3 e l'estrazione tornava un'opzione invece del
    # soggetto. Togliere il grilletto batte qualsiasi istruzione — vedi il
    # system prompt in models/qwen.py::ENTITY_EXTRACTION_SYSTEM. Fallback al
    # taglio sul boilerplate per prompt senza "options:".
    text = "".join(qt.text for qt in query_tokens)
    lowered = text.lower()
    cut = lowered.find("options:")
    if cut == -1:
        cut = lowered.find(BOILERPLATE_MARKER)
    question_block = (text[:cut] if cut != -1 else text).strip()
    raw = str(extract_entities(question_block)).strip()
    # v4: UNA sola sequenza contigua, nessuno split sulle virgole — un output
    # multi-frase del decoder è una regressione dal formato e non mappa
    # verbatim, quindi cade da sé nel fallback. `NONE` è l'uscita deliberata
    # "questa domanda non ha un soggetto specifico": fallback esplicito, con
    # `entity_raw='NONE'` nei metadati per distinguerlo a valle dal mancato
    # mapping.
    if raw.strip('.,;:!"\'').lower() == "none":
        return QueryRowsResolution(
            rows=question_rows(query_tokens), mode="question",
            entity_raw=raw, entity_phrases=(), entity_mapped=False,
        )
    phrases = (raw,)
    rows = entity_rows(query_tokens, list(phrases))
    if rows:
        return QueryRowsResolution(
            rows=rows, mode="entity",
            entity_raw=raw, entity_phrases=phrases, entity_mapped=True,
        )
    logger.warning(
        "entity %r non mappata verbatim nella domanda: fallback a query_rows='question'", raw,
    )
    return QueryRowsResolution(
        rows=question_rows(query_tokens), mode="question",
        entity_raw=raw, entity_phrases=phrases, entity_mapped=False,
    )


def parse_answer_meta(obj: dict) -> dict:
    """Estrae la risposta corretta MCQ da un dict, per mostrarla in UI.

    Supporta due convenzioni (usate rispettivamente dai sidecar
    `--prompt-from-json` di `attn_explorer.capture` e dai sample dei
    `Dataset` di `evals/`):
      - `gt`/`gt_option`: testo e lettera della risposta corretta già
        risolti (es. sidecar VNBench).
      - `answer`/`options`: indice 0-based nella lista `options` (es. sample
        di `EgoSchema`/`MVBench`/...).

    Ritorna `{"answer_letter", "answer_text", "options"}` (`options` può
    essere `None` se il dict non lo espone), o `{}` se non riconosce nessuna
    delle due convenzioni.
    """
    if "gt_option" in obj:
        return {
            "answer_letter": obj["gt_option"],
            "answer_text": obj.get("gt"),
            "options": obj.get("options"),
        }
    if "answer" in obj and "options" in obj and obj["answer"] is not None:
        options = obj["options"]
        idx = int(obj["answer"])
        if 0 <= idx < len(options):
            return {
                "answer_letter": chr(ord("A") + idx),
                "answer_text": options[idx],
                "options": options,
            }
    return {}


def mcq_answer_stats(logits: torch.Tensor, letter_ids: dict[str, int]) -> dict:
    """Entropia + probabilità della risposta MCQ dai logit dell'ULTIMO token del prefill.

    `logits` è `[vocab_size]` (l'output del modello a `logits_to_keep=1`, cioè
    la distribuzione sul primo token che genererebbe in risposta al prompt).
    `letter_ids` mappa lettera candidata → id del suo primo token (calcolato
    dal chiamante via tokenizer, dipende dal modello). La softmax è
    RISTRETTA alle sole lettere candidate (rinormalizzata fra loro), non
    sull'intero vocabolario: è il segnale interpretabile ("quanto è sicuro
    che sia B piuttosto che C"), non quanto è probabile il token "B" in
    assoluto (che dipende anche da spazi/tokenizzazione, non dal contenuto).

    Returns:
        `{"answer_entropy": bit, "answer_probs": {lettera: prob},
        "answer_logits": {lettera: logit grezzo pre-softmax}, "pred_letter": lettera}`.
        Entropia in BIT (log2): 0 = certezza assoluta, `log2(n_lettere)` =
        massima incertezza (distribuzione uniforme fra le candidate).
    """
    letters = list(letter_ids.keys())
    ids = torch.tensor([letter_ids[l] for l in letters], device=logits.device)
    raw_logits = logits[ids].float()
    probs = torch.softmax(raw_logits, dim=0)
    entropy_bits = (-(probs * torch.log2(probs.clamp_min(1e-12))).sum()).item()
    pred_letter = letters[int(probs.argmax())]
    return {
        "answer_entropy": entropy_bits,
        "answer_probs": {l: float(p) for l, p in zip(letters, probs.tolist())},
        "answer_logits": {l: float(v) for l, v in zip(letters, raw_logits.tolist())},
        "pred_letter": pred_letter,
    }


def top_k_logits(logits: torch.Tensor, tokenizer, k: int = 10) -> list[dict]:
    """Top-`k` token del vocabolario INTERO (non ristretto alle lettere MCQ).

    `logits` è `[vocab_size]` (stesso output di `logits_to_keep=1` riusato da
    `mcq_answer_stats`, nessun forward aggiuntivo). A differenza della softmax
    ristretta alle lettere candidate, qui la probabilità è quella "vera" sul
    vocabolario completo: utile per vedere cosa il modello genererebbe senza
    vincoli (es. se prova a scrivere il testo dell'opzione invece della
    lettera, o è distratto da token non pertinenti).

    Returns: lista di `{"token": str, "token_id": int, "logit": float, "prob": float}`
    ordinata per probabilità decrescente.
    """
    probs_full = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs_full.topk(k)
    top_logits = logits[top_ids].float()
    return [
        {
            "token": tokenizer.decode([tid]),
            "token_id": int(tid),
            "logit": float(lg),
            "prob": float(p),
        }
        for tid, lg, p in zip(top_ids.tolist(), top_logits.tolist(), top_probs.tolist())
    ]


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

    `answer_entropy`/`answer_probs`/`pred_letter` (opzionali, `None` se il
    chiamante non passa lettere candidate a `full_visual_attention`): stessi
    logit dell'ULTIMA posizione del prefill (già calcolati per il forward,
    `logits_to_keep=1` — nessun costo aggiuntivo), softmax ristretta alle
    lettere MCQ candidate. Bassa entropia + `pred_letter` sbagliata, o alta
    entropia in generale, sono un segnale che l'evidenza per rispondere non è
    (chiaramente) nei frame campionati — vedi `utils.attn_core.mcq_answer_stats`.

    `top_tokens`: top-k token sul vocabolario INTERO (non ristretto alle
    lettere), stessi logit riusati — vedi `utils.attn_core.top_k_logits`.
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
    answer_entropy: float | None = None
    answer_probs: dict[str, float] | None = None
    answer_logits: dict[str, float] | None = None
    pred_letter: str | None = None
    top_tokens: list[dict] | None = None

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

    `answer_entropy`/`answer_probs`/`answer_logits`/`pred_letter`/`top_tokens`:
    vedi `VisualAttention` — `None` sui dump legacy o su prompt non-MCQ
    (nessuna lettera candidata) per i primi quattro; `top_tokens` invece è
    calcolato sempre (non richiede un MCQ), `None` solo sui dump legacy.

    `capture_meta`: metadati di riproducibilità dell'invocazione di
    `attn_explorer.capture` che ha prodotto questo dump (model id, dtype,
    parametri di campionamento frame, timestamp, commit git) — vedi
    `attn_explorer.capture.build_capture_meta`. `None` sui dump legacy.
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
    answer_entropy: float | None = None
    answer_probs: dict[str, float] | None = None
    answer_logits: dict[str, float] | None = None
    pred_letter: str | None = None
    top_tokens: list[dict] | None = None
    capture_meta: dict | None = None

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
    border: int = 0,
) -> torch.Tensor:
    """Identifica i token visivi-sink dalla `sink_map` `[t, grid_h, grid_w]`.

    Un token è sink se il suo `sink_score` è nel top-`percentile`% della
    distribuzione (su tutto `sink_map`) — l'UNICO segnale usato di default:
    `sink_map` viene dalle statistiche interne del modello (canali "sink
    dims", Xiao et al.), non dalla posizione del token.

    `border` (default 0, OFF): se >0, forza anche i primi/ultimi `border`
    anelli di righe/colonne a sink a prescindere dal loro `sink_score`.
    Euristica NON verificata su questo progetto — "sink" in letteratura è
    una proprietà di canali ad alta norma nello stato nascosto, non una
    proprietà geometrica di bordo dell'immagine: un frame potrebbe avere
    evidenza reale vicino al bordo, che questa opzione azzererebbe a priori.
    Lasciato disponibile come opt-in (`?sink_border=N` nel server) per chi
    vuole testarlo empiricamente sui propri capture, ma va verificato prima
    di fidarsene.

    Args:
        sink_map: `[t, grid_h, grid_w]` torch/numpy tensor.
        percentile: quota superiore della distribuzione considerata sink
            (default 25 — i top-25% per sink score sono sink). Range 0-100.
        border: anelli di bordo forzati a sink, OFF di default (solo se
            `grid_h > 2*border` e `grid_w > 2*border`, altrimenti saltato).

    Returns:
        `bool tensor` `[t, grid_h, grid_w]`: True = sink (da azzerare dalla
        heatmap quando si filtra).
    """
    m = torch.as_tensor(sink_map).float().cpu()
    if m.ndim != 3:
        raise ValueError(f"sink_map deve essere [t,g h,gw], got {tuple(m.shape)}")
    t, gh, gw = m.shape
    # "top-percentile%" per sink score → soglia al quantile (1 - percentile/100).
    # Con percentile=25 vogliamo il 25% PIÙ ALTO, non tutto sopra il 25° percentile
    # (che sarebbe il 75% dei token, l'opposto di quanto intende "sink" come minoranza).
    thr = torch.quantile(m.flatten(), 1.0 - percentile / 100.0)
    is_sink = m >= thr
    if border > 0 and gh > 2 * border and gw > 2 * border:
        is_sink[:, :border, :] = True
        is_sink[:, -border:, :] = True
        is_sink[:, :, :border] = True
        is_sink[:, :, -border:] = True
    return is_sink


def sink_view(
    heatmap: torch.Tensor,
    sink_map: torch.Tensor,
    *,
    view: str = "all",
    percentile: float = 25.0,
    border: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isola sink/non-sink in `heatmap` SENZA perdere l'informazione grezza.

    `heatmap` è `[t, grid_h, grid_w]` già aggregato per una selezione di
    token-query. La maschera (`is_sink`) è calcolata da `sink_map` via
    `sink_mask`. `view` seleziona cosa restituire:
      - "all":     `heatmap` invariata (nessun azzeramento) — serve solo a
                   ritornare la mask per poterla disegnare sopra l'overlay,
                   cosicché sink e non-sink restino DISTINGUIBILI a vista pur
                   mostrando l'attenzione grezza per intero.
      - "sink":    solo le celle sink (le altre azzerate), renormalizzato.
      - "nonsink": solo le celle non-sink (le altre azzerate), renormalizzato.

    Ritorna sempre `(heatmap_vista, mask)`: la mask è utile al chiamante
    anche in view="all" per evidenziare i contorni delle celle sink.
    """
    m = torch.as_tensor(heatmap).float().cpu()
    mask = sink_mask(sink_map, percentile=percentile, border=border)
    if view == "all":
        return m, mask
    if view not in ("sink", "nonsink"):
        raise ValueError(f"view deve essere 'all', 'sink' o 'nonsink', got {view!r}")
    keep = mask if view == "sink" else ~mask
    out = m.clone()
    out[~keep] = 0.0
    mx = out.max()
    if mx > 0:
        out = out / mx
    return out, mask


# ─────────────────────────────────────────────────────────────────────────────
# Concentrazione dell'attenzione sui frame (segnale di "grounding")
# ─────────────────────────────────────────────────────────────────────────────
def attention_concentration(
    cap: "AttentionCapture",
    rows: tuple[int, ...] = (),
    *,
    percentile: float = 25.0,
    border: int = 0,
    topk: int = 3,
) -> dict:
    """Quanto l'attenzione (sink-filtrata) si concentra su poche celle temporali.

    Aggrega l'attenzione sulla selezione di token `rows` (default: tutta la
    domanda, via `AttentionCapture.aggregate`), azzera le celle sink
    (`sink_mask`/`sink_view`) e classifica le celle residue per massa
    (`rank_cells`, `metric='sum'`). `top1_pct` alto ⇒ il modello "trova" un
    frame su cui fissarsi (grounding); basso e uniforme fra le celle ⇒
    attenzione dispersa, nessun frame spicca.

    Senza filtro sink questo segnale è quasi piatto (dominato dai token sink,
    che assorbono la maggior parte della massa indipendentemente dal
    contenuto) — il filtro è ciò che lo rende informativo. Validato
    empiricamente su un campione VNBench "retrieval" (r=0.58 con la
    correttezza, r=0.37 con la presenza content-grounded del needle nei frame
    campionati) — vedi `docs/entropia_confidenza_mcq.md`.
    """
    heat = cap.aggregate(rows)
    heat_nonsink, _ = sink_view(heat, cap.sink_map, view="nonsink", percentile=percentile, border=border)
    ranked = rank_cells(heat_nonsink, cap.grid, cap.frame_indices, cap.fps, metric="sum", topk=topk)
    top1 = ranked[0]
    return {
        "top1_pct": top1.pct,
        "topk_pct": sum(r.pct for r in ranked[:topk]),
        "top_cell": top1.cell,
        "top_t_sec": top1.t_sec,
    }


def ranked_cells_from_attention(
    visual_attention: VisualAttention,
    *,
    percentile: float = 25.0,
    border: int = 0,
    query_rows: str = "all",
    rows: list[int] | None = None,
    sink_filter: bool = True,
) -> list[RankedCell]:
    """Classifica TUTTE le celle temporali di un `VisualAttention` per massa
    di attenzione (sink-filtrata).

    Variante di `attention_concentration` che opera direttamente sui segnali
    "in volo" di `SupportsSignals.generate_with_signals`, invece che su un
    `AttentionCapture` già materializzato: non essendoci `frame_indices`/`fps`
    reali si passano dei dummy (`range(t)`, `fps=1.0`, `double=True`), perché
    al chiamante servono solo `cell`/`pct` — la posizione temporale si ricava
    per via frazionaria dalla finestra vista dal pass 1. `rank_cells` non
    tronca a `topk` (marca solo `is_top`), quindi ritorna sempre le `t` celle
    ordinate per massa decrescente.

    `query_rows` seleziona SU QUALI righe mediare (chiave di `ROW_SELECTORS`):
    `all` (default) le media tutte, cioè tutto ciò che segue l'ultimo
    `<|vision_end|>` — domanda + opzioni + boilerplate + generation prompt.
    Il default NON va cambiato qui: tutti gli arm già misurati sono girati
    con `all`, e spostarlo renderebbe le loro celle non più confrontabili con
    quelle nuove. Si cambia per arm, dal knob `query_rows` della strategy.

    `rows` (esplicite) BYPASSA `query_rows`: è il canale per le selezioni che
    un selettore puro non può calcolare — la modalità `entity`, risolta a
    monte da `resolve_query_rows` (che copre anche i valori puri: le strategy
    passano sempre `rows=resolution.rows`). Stessa convenzione di
    `AttentionCapture.aggregate`: `[]` = tutte le righe.

    `sink_filter` (default `True`, e ANCHE QUESTO default non va cambiato,
    stessa ragione di `query_rows`): `True` = ranking sulla vista
    `view="nonsink"` (i token-sink azzerati, comportamento storico); `False`
    = attenzione GREZZA, `view="all"` — nessun azzeramento, nessuna
    renormalizzazione. È il modo giusto di spegnere il filtro:
    `percentile=0` NON lo è (la soglia diventerebbe il massimo e l'argmax
    della sink_map verrebbe azzerato comunque — un "quasi spento" travestito
    da spento). Usato dal knob `sink_filter` di `attention_marker`.

    Condivisa fra le strategy signal-driven (`entropy_attention_resample`,
    `attention_highlight`, `attention_marker`): il ranking DEVE essere
    calcolato allo stesso modo in tutte, altrimenti un confronto win/loss
    per-sample fra arm diversi misurerebbe anche la differenza di segnale,
    non solo quella di intervento.
    """
    if rows is None:
        try:
            selector = ROW_SELECTORS[query_rows]
        except KeyError:
            raise ValueError(
                f"query_rows sconosciuto: {query_rows!r}. Valide: {sorted(ROW_SELECTORS)}"
            ) from None
        rows = selector(visual_attention.query_tokens)
    sel = visual_attention.attn if not rows else visual_attention.attn[rows]
    heat = sel.float().mean(dim=0)  # [t, grid_h, grid_w]
    grid = GridSpec(
        t=visual_attention.t,
        grid_h=visual_attention.grid_h,
        grid_w=visual_attention.grid_w,
        double=True,
    )
    heat_view, _ = sink_view(
        heat, visual_attention.sink_map,
        view="nonsink" if sink_filter else "all",
        percentile=percentile, border=border,
    )
    return rank_cells(heat_view, grid, list(range(visual_attention.t)), fps=1.0, metric="sum", topk=1)


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
        "answer_entropy": cap.answer_entropy,
        "answer_probs": cap.answer_probs,
        "answer_logits": cap.answer_logits,
        "pred_letter": cap.pred_letter,
        "top_tokens": cap.top_tokens,
        "capture_meta": cap.capture_meta,
    }, path)


def load_capture(path: Path) -> AttentionCapture:
    """Ricarica un `AttentionCapture` da `capture.pt`.

    Tieni `sink_map`/`sink_dims`/`answer_entropy`/`answer_probs`/
    `answer_logits`/`pred_letter`/`top_tokens`/`capture_meta` opzionali per
    leggere dump legacy senza quei field (rispettivamente: ricostruiti come
    zeri/lista vuota, o `None`).
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
        answer_entropy=d.get("answer_entropy"),
        answer_probs=d.get("answer_probs"),
        answer_logits=d.get("answer_logits"),
        pred_letter=d.get("pred_letter"),
        top_tokens=d.get("top_tokens"),
        capture_meta=d.get("capture_meta"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione di un AttentionCapture da un VisualAttention + scrittura store
#
# Corpo condiviso fra `attn_explorer.capture.capture_one` (cattura standalone,
# forward dedicato) e le strategy signal-driven (`strategies/entropy_
# shortcut.py`, `strategies/entropy_attention_resample.py`, opt-in via
# `capture_attention` nel loro preset) che catturano un `VisualAttention` già
# calcolato "gratis" da `SupportsSignals.generate_with_signals` durante
# l'eval — nessun forward aggiuntivo, solo dump su disco.
# ─────────────────────────────────────────────────────────────────────────────
def _git_commit() -> str | None:
    """Hash corto del commit corrente (`None` se non è un repo git o git manca)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_capture_meta(
    *,
    model_id: str,
    dtype: str,
    nframes: int,
    max_pixels: int,
    min_pixels: int | None,
    double_frames: bool = False,
    device_map: str | None = None,
    attn_implementation: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Metadati di riproducibilità di un capture (vedi `AttentionCapture.
    capture_meta`): modello, precisione, parametri di campionamento frame,
    timestamp e commit git. `extra` accoglie campi specifici del chiamante
    (es. le strategy ci mettono `strategy`/`resampled`/`resample_kind`, non
    pertinenti al tool di cattura standalone)."""
    return {
        "model_id": model_id,
        "dtype": dtype,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
        "nframes": nframes,
        "double_frames": double_frames,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        **(extra or {}),
    }


def reconstruct_frame_indices(
    video_path: str,
    nframes: int,
    *,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[int], float]:
    """Ricostruisce `(frame_indices assoluti, fps)` per un `Video(nframes=...)`.

    qwen-vl-utils non espone gli indici frame che ha effettivamente campionato
    dal processor: si ricalcolano con la STESSA formula di
    `qwen_vl_utils.vision_process.calculate_video_frame_range` +
    `_read_video_decord` (`start_frame=ceil(video_start*fps)`,
    `end_frame=floor(video_end*fps)`, poi `linspace(start_frame, end_frame,
    nframes)`) — se questa diverge dall'implementazione reale, i frame PNG
    estratti per il capture NON corrispondono più a quelli visti dal modello.
    `video_start`/`video_end` (secondi, `None` = bordo del video) riproducono
    l'eventuale trim del dataset (es. LVBench `time_reference`).
    """
    import decord

    vr = decord.VideoReader(video_path)
    total = len(vr)
    fps = float(vr.get_avg_fps())

    if video_start is None and video_end is None:
        lo, hi = 0, total - 1
    else:
        max_duration = total / fps
        lo = math.ceil(max(0.0, min(video_start, max_duration)) * fps) if video_start is not None else 0
        hi = total - 1
        if video_end is not None:
            hi = min(math.floor(max(0.0, min(video_end, max_duration)) * fps), total - 1)

    frame_indices = torch.linspace(lo, hi, nframes).round().long().tolist()
    return frame_indices, fps


def extract_cell_frames(video_path: str, grid: GridSpec, frame_indices, frames_dir: Path) -> None:
    """Salva un PNG per ogni cella temporale (frame rappresentante).

    Il server li serve direttamente (`/api/<stem>/frame?cell=ti`) e ci applica
    sopra gli overlay rigenerati a runtime.
    """
    import decord

    frames_dir.mkdir(parents=True, exist_ok=True)
    vr = decord.VideoReader(video_path)
    last = len(vr) - 1
    for ti in range(grid.t):
        _, rep = grid.cell_to_frames(ti, frame_indices)
        rep = max(0, min(rep, last))
        Image.fromarray(vr[rep].asnumpy()).save(frames_dir / f"cell_{ti:04d}.png")


def write_capture(
    out_dir: Path,
    video_path: str,
    prompt: str,
    visual_attention: VisualAttention,
    frame_indices,
    fps: float,
    double: bool,
    capture_meta: dict,
) -> AttentionCapture:
    """Costruisce l'`AttentionCapture` da un `VisualAttention` grezzo e lo
    scrive su `<out_dir>/<stem>/{capture.pt,frames/}`. Ritorna il capture
    stesso (il chiamante lo passa a `capture_index_entry` per l'entry di
    `index.json`, o lo usa direttamente per un quick-check)."""
    stem = Path(video_path).stem
    stem_dir = out_dir / stem
    frames_dir = stem_dir / "frames"

    grid = GridSpec(t=visual_attention.t, grid_h=visual_attention.grid_h, grid_w=visual_attention.grid_w, double=double)

    cap = AttentionCapture(
        video_path=str(video_path),
        prompt=prompt,
        fps=fps,
        frame_indices=tuple(int(i) for i in frame_indices),
        grid=grid,
        query_tokens=visual_attention.query_tokens,
        attn=visual_attention.attn,
        sink_map=visual_attention.sink_map,
        sink_dims=visual_attention.sink_dims,
        answer_entropy=visual_attention.answer_entropy,
        answer_probs=visual_attention.answer_probs,
        answer_logits=visual_attention.answer_logits,
        pred_letter=visual_attention.pred_letter,
        top_tokens=visual_attention.top_tokens,
        capture_meta=capture_meta,
    )
    save_capture(cap, stem_dir / "capture.pt")
    extract_cell_frames(video_path, grid, frame_indices, frames_dir)
    return cap


def capture_index_entry(cap: AttentionCapture, extra: dict | None = None) -> dict:
    """Entry per `index.json` a partire da un `AttentionCapture` scritto da
    `write_capture` (`extra` ci finisce dentro, es. la risposta corretta da
    `parse_answer_meta`, o `resampled`/`resample_kind` di una strategy)."""
    return {
        "stem": Path(cap.video_path).stem,
        "video": cap.video_path,
        "video_name": Path(cap.video_path).name,
        "prompt": cap.prompt,
        "fps": cap.fps,
        "cells": cap.grid.t,
        "n_tokens": len(cap.query_tokens),
        "regime": cap.grid.regime,
        "pred_letter": cap.pred_letter,
        "answer_entropy": cap.answer_entropy,
        **(extra or {}),
    }


def write_index(store: Path, entries: list[dict]) -> None:
    """Scrive/aggiorna `<store>/index.json` (merge per `stem`)."""
    index_path = store / "index.json"
    existing: dict[str, dict] = {}
    if index_path.is_file():
        for e in json.loads(index_path.read_text(encoding="utf-8")).get("videos", []):
            existing[e["stem"]] = e
    for e in entries:
        existing[e["stem"]] = e
    payload = {"videos": sorted(existing.values(), key=lambda e: e["stem"])}
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Indice aggiornato: %s (%d video)", index_path, len(payload["videos"]))
