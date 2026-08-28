# Piano — visual prompting: scrivere `IMPORTANT` sul frame di picco

> **Stato**: specifica di implementazione. Nessun codice scritto nel repo; il
> codice qui sotto è **da copiare e adattare**, non pseudocodice. I numeri di
> §3 sono calcolati (appendice A, gira senza GPU né pesi). La leggibilità del
> testo dipinto **da parte del modello NON è verificata**: è il rischio
> principale dell'arm e §8.1 la mette come gate prima di qualunque accuracy.

---

## 1. Cosa si vuole ottenere

Un arm che segnala i frame di picco **dipingendo la parola `IMPORTANT` sui
pixel del frame stesso**, invece di racchiuderlo fra due marcatori testuali
nel flusso dei token (`strategies/attention_marker.py`,
`docs/piano_marcatori_frame_qwen3.md`).

Stesso segnale (attenzione domanda→visivo del pass 1), stessa scelta delle
celle, stesso gate, stesso costo strutturale (estrazione PNG + secondo pass):
**cambia solo il canale su cui l'intervento viene consegnato**, da testuale a
visivo. È la terza variante della famiglia già impostata:

| variante | dove vive l'intervento | file |
|---|---|---|
| A — puntatore testuale | nel prompt, nomina un intervallo di secondi | `strategies/attention_highlight.py` |
| B — marcatori inline | nel flusso dei token, attorno al blocco video | `strategies/attention_marker.py` |
| **C — visual prompting** | **nei pixel del frame** | **da scrivere** |

Perché provare C dopo A e B: in A e B il modello deve *risolvere* un
riferimento (secondi, oppure una posizione nella sequenza) verso il contenuto
visivo, e su Video-MME A è risultata indistinguibile dal suo controllo
casuale (`docs/260803-tabelle_risultati.md` §1). In C il segnale è
**co-locato con l'evidenza**: non c'è nessun riferimento da risolvere, la
parola sta sopra il frame da guardare. È anche la forma di prompting che la
famiglia Qwen gestisce meglio per costruzione (OCR forte), ed è ciò che LoT
fa nella variante immagine disegnando la bbox sull'immagine
(`lot/prompts.py:147`, `SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA`).

**Decisioni già prese, non da rimettere in discussione:**

| | |
|---|---|
| famiglia di modelli | **Qwen3-VL** per il primo giro (`qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`). Il vincolo NON è tecnico come per i marcatori: vedi §6.2 |
| parola dipinta | `IMPORTANT`, maiuscolo, **costante di modulo** |
| istruzione nel prompt | **obbligatoria**, costante di modulo (§5) |
| unità marcabile | la **cella** (2 frame nel regime `double_frames=false`), ed **entrambi** i frame vanno dipinti (§4.1) |
| gate | `always_highlight: true`, nessun gate d'entropia |
| righe-query | `query_rows: question` (appaiato con `attention_marker`) |
| filtro sink | `sink_filter: false`, attenzione GREZZA (appaiato con `attention_marker`) |
| controllo | `cell_select: random` — stessa parola, stessa istruzione, stessa occlusione, cella a sorte |
| geometria dell'overlay | **knob di config** (posizione, altezza banda): parametro di rendering da tarare sulla probe, non il testo dell'esperimento |

---

## 2. Cosa CADE rispetto al piano marcatori, e cosa si riusa

Il grosso della complessità di `docs/piano_marcatori_frame_qwen3.md` esisteva
per una ragione sola: i marcatori **spezzano il blocco video in più blocchi**.
Dipingere pixel non tocca la struttura dei token, quindi quelle invarianti
spariscono:

| vincolo del piano marcatori | qui |
|---|---|
| §2.1 M-RoPE / `get_rope_index`, gruppi visivi | **non si applica**: un solo blocco video, struttura identica alla baseline |
| §2.2 costo in token dei marcatori | **zero token in più** (il costo è in pixel occlusi, §4.2) |
| §2.4 tagli su confini di cella, `nframes` PARI | **non si applica**: nessun taglio. `nframes` dispari è ammesso (il processor padda, la mappatura cella `ti` → frame `2ti`/`2ti+1` regge) |
| §5.1/5.2 metadata per blocco | **già implementati e mergiati** (`models/media.py::VideoFrames.frames_indices/fps`, `models/qwen.py:25 _VIDEO_METADATA_KEY`): qui si usano così come sono, per un blocco solo |
| §5.3 `build_messages_from_parts` | **già in repo**, e qui non serve: un solo media + un solo testo → basta `build_messages`, che vi delega (`models/qwen.py:156-174`) |
| solo-Qwen3-VL per i `position_ids` | **non si applica**: dipingere è agnostico alla famiglia (motivo residuo in §6.2) |

Cosa **resta** e va riusato senza riscriverlo:

| pezzo | dove | cosa dà |
|---|---|---|
| segnali del pass 1 | `models/qwen_attn.py:395 generate_with_signals` | `SignalAnswer` (entropia, `pred_letter`, `VisualAttention`) |
| ranking celle | `utils/attn_core.py:616 ranked_cells_from_attention` | `list[RankedCell]` ordinata, con `query_rows`/`sink_filter` già esposti |
| indici frame reali | `utils/attn_core.py:796 reconstruct_frame_indices` | `(indici ASSOLUTI, fps sorgente)` |
| estrazione PNG | `strategies/attention_marker.py:146 _extract_all_frames` | tmpdir + lista path, contratto di cleanup già scritto — **importarla, non ricopiarla** |
| RNG del controllo | `strategies/attention_highlight.py:155 _sample_rng` | cella random stabile per `(video_path, prompt)` fra shard e rilanci |
| asserzione righe-query | `strategies/attention_marker.py` (blocco `n_query_rows`) | fail-fast sul fallback silenzioso di `_rows_before_marker` (`utils/attn_core.py:75`) |
| capture opzionale | `strategies/base.py::Strategy._save_attention_capture` | dump `capture.pt`/`frames/` per `attn_explorer.serve` |

> ⚠️ **`RankedCell` ha campi fittizi** (`.frames`, `.rep_frame`, `.t_sec`):
> `ranked_cells_from_attention` passa dummy a `rank_cells`
> (`frame_indices=range(t)`, `fps=1.0`, `double=True`). Usare **solo**
> `.cell` e `.pct`. Per i frame reali: `reconstruct_frame_indices` e poi
> cella `ti` → posizioni `2*ti`, `2*ti+1` della lista campionata.

---

## 3. Il rischio vero: la parola sopravvive allo `smart_resize`?

Il frame che il modello vede **non** è il frame sorgente: il processor lo
riscala a `max_pixels`. Video-MME gira a `max_pixels: 151200`
(`conf/config.yaml`), che dà (calcolato, appendice A):

| sorgente | fattore 28 (Qwen2.5-VL) | fattore 32 (Qwen3-VL) |
|---|---|---|
| 1280×720 | 504×280, griglia 18×10 | 512×288, griglia 16×9 |
| 1920×1080 | 504×280 | 512×288 |
| 640×480 | 448×336, griglia 16×12 | 448×320, griglia 14×10 |

Cioè: **il modello guarda un frame alto ~288 px**. Con una banda al 18% il
testo arriva alto ~52 px, poco più di **1,5 celle di griglia** (una cella è
32×32 px su Qwen3-VL). Un rendering di prova a `height_frac=0.18` su un 720p
(appendice A) è **leggibile a occhio** una volta portato a 512×288: la parola
occupa i 2/3 della larghezza su banda piena. Il margine c'è; resta ignoto se
il *vision encoder* a 32 px/cella la risolva come testo — risponde solo la
probe §8.1.

Due conseguenze di disegno:

1. **La probe di leggibilità (§8.1) è un gate, non un check.** Se il modello
   non legge la parola, l'arm non misura "visual prompting": misura
   "occlusione di una banda di pixel".
2. **Due modi di dipingere, in ordine di preferenza:**
   - **Opzione A (default, quella implementata sotto)** — dipingere a
     risoluzione SORGENTE, con corpo proporzionale all'altezza
     (`overlay_height_frac`). Robusta, non duplica `smart_resize`,
     indipendente dalla risoluzione dei video del dataset. Costo: il testo
     passa per il downscale del processor e si ammorbidisce.
   - **Opzione B (fallback, solo se A fallisce §8.1)** — riscalare noi il
     frame al target di `smart_resize` e dipingere **dopo**: glifi netti e
     altezza in pixel-modello esatta. Costo: dipendenza dal `factor` interno
     (28 su Qwen2.5-VL, 32 su Qwen3-VL — è `Qwen.image_patch_size * 2`,
     `models/qwen.py`); se lo si sbaglia il processor riscala di nuovo e si
     torna ad A con un passaggio in più.

---

## 4. Confondimenti e disegno dei controlli

### 4.1 Entrambi i frame della cella vanno dipinti

Il patch embedding è 3D: `temporal_patch_size=2` fonde **due frame
consecutivi** in una riga di token. Dipingere un frame solo della cella fa
arrivare la parola "a metà intensità", mescolata al frame pulito — un
intervento più debole di quello che si crede di fare, e non riproducibile fra
celle (dipende da quale dei due si è scelto). Si dipingono **entrambe** le
posizioni `2*ti` e `2*ti+1`.

### 4.2 L'occlusione è il costo, e il controllo deve portarlo identico

La banda copre ~18% dei pixel del frame marcato: proprio il frame che
contiene l'evidenza. È un costo reale e non eliminabile (è la natura
dell'intervento), non un bug. Il punto è che **il controllo lo paga uguale**:
`cell_select: random` dipinge la stessa parola, con la stessa banda, sullo
stesso numero di frame, in un punto scelto a sorte. La differenza
`attention − random` isola **dove** si dipinge, con l'occlusione a fattor
comune. Il confronto contro `baseline_24` risponde a un'altra domanda (l'arm
nel complesso guadagna o perde?) e va letto **dopo**.

### 4.3 Il breakdown da guardare per l'occlusione

Su Video-MME esiste il task type `ocr problems` (139 sample,
`docs/260901-tabelle-risultati.md`): è la sottopopolazione dove una banda
opaca ha più probabilità di distruggere proprio l'evidenza che serve. Va
letto separatamente — un calo lì con guadagno altrove è informazione sul
disegno dell'overlay (posizione, altezza), non un fallimento dell'idea.

### 4.4 Ablation che NON entrano nel primo giro

- **parola neutra** (`FRAME` invece di `IMPORTANT` sul picco): separerebbe
  "il modello legge e obbedisce" da "una banda satura attira l'encoder". È un
  secondo arm; qui la parola è costante di modulo.
- **bordo colorato invece di testo**: nessuna occlusione del contenuto, ma
  non è più visual *prompting* (nessuna istruzione leggibile). Arm diverso.
- **top_k > 1**: il knob c'è, ma il primo giro è `top_k=1` come l'arm
  marcatori, altrimenti i due non sono appaiati.

---

## 5. Istruzione nel prompt

Obbligatoria, per lo stesso motivo dei marcatori: la parola dipinta è un dato
visivo qualunque, senza una riga che le dia statuto di indicazione il modello
non ha motivo di trattarla diversamente da una scritta in sovraimpressione
del video sorgente. Costante di modulo, **non** knob:

```python
OVERLAY_INSTRUCTION = (
    f"Some frames have the word {OVERLAY_TEXT} written on them: those frames "
    "contain the visual evidence relevant to the question. Do not mention the "
    "word in your answer."
)
```

L'ultima frase ricalca `lot/prompts.py:142` ("Do not output the markers"):
senza, la parola rischia di finire nella risposta e sporcare il parsing MCQ.
Va **prepended** al prompt MCQ (`f"{OVERLAY_INSTRUCTION}\n\n{prompt}"`), come
fa `attention_marker` con `MARKER_INSTRUCTION`.

**Conseguenza sul controllo**: l'arm è "parola dipinta + istruzione", quindi
il controllo onesto è la stessa istruzione con cella a caso, non l'assenza di
istruzione (lezione di `docs/260803-tabelle_risultati.md` §1).

---

## 6. Implementazione

### 6.0 File toccati

| file | azione |
|---|---|
| `utils/overlay.py` | **nuovo** — rendering PIL (§6.1) |
| `strategies/visual_prompt.py` | **nuovo** — la strategy (§6.2) |
| `strategies/__init__.py` | +1 import, +1 voce in `__all__` (§6.3) |
| `conf/config.yaml` | +1 preset sotto `strategy:`, +1 nome nel commento di `active.strategy` (§6.3) |
| `scripts/probe_overlay_legibility.py` | **nuovo** — il gate §8.1 (§6.4) |
| `scripts/sbatch/ablation-videomme/visual_prompt_24_qwen3.sbatch` | **nuovo** (§6.5) |
| `pyproject.toml` | `pillow>=10.1` fra le `dependencies` (§6.1, nota Pillow) |
| `main.py` | **da NON toccare**: il dispatch passa da `Strategy.get` (`main.py:98`) |

### 6.1 `utils/overlay.py` — il rendering

Modulo puro (PIL + stdlib), nessun import dal resto del progetto: così la
probe §8.1 e l'appendice A lo usano senza tirarsi dietro modello, Hydra o
Weave.

> **Nota Pillow.** Oggi `pillow` entra solo come dipendenza transitiva
> (12.2.0 in `uv.lock`). Qui si dipende da `ImageFont.load_default(size=…)`,
> che esiste **da Pillow 10.1**: aggiungere `"pillow>=10.1"` alle
> `dependencies` di `pyproject.toml`, altrimenti il vincolo è implicito e un
> `uv lock` futuro potrebbe risolverlo più in basso.

```python
"""Overlay testuale sui frame — rendering per `strategies/visual_prompt.py`.

Modulo PURO (PIL + stdlib): nessun import dal resto del progetto, così
`scripts/probe_overlay_legibility.py` e la verifica offline dell'appendice A
di `docs/piano_visual_prompting.md` possono usarlo senza toccare modello,
Hydra o Weave.

La parola e i colori sono costanti di modulo, NON knob di config: sono il
TESTO dell'intervento (stesso criterio di `MARKER_INSTRUCTION` in
`strategies/attention_marker.py`). Knob restano solo posizione e altezza
della banda, che sono geometria da tarare sulla probe di leggibilità.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OVERLAY_TEXT = "IMPORTANT"

# Banda PIENA + testo in controcolore, non testo con contorno sopra il frame:
# il contrasto non deve dipendere dal contenuto del video, altrimenti su una
# scena chiara la parola sparisce esattamente nei sample in cui serve.
BAND_FILL = (220, 30, 30)
TEXT_FILL = (255, 255, 255)

# Il corpo del testo occupa questa frazione dell'altezza della banda: il resto
# è padding, senza il quale i glifi toccano i bordi e il downscale li impasta.
_TEXT_OF_BAND = 0.75
_MAX_TEXT_WIDTH_FRAC = 0.92
# Sotto questo corpo (px SORGENTE) qualcosa nel calcolo della scala è saltato:
# meglio fallire che dipingere un francobollo illeggibile in silenzio.
_MIN_TEXT_PX = 12

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(px: int) -> tuple[ImageFont.FreeTypeFont, str]:
    """Font bold di sistema, con fallback al default scalabile di Pillow.

    ⚠️ `ImageFont.load_default()` SENZA `size` ritorna il bitmap 11 px:
    illeggibile dopo il downscale, e il degrado sarebbe silenzioso. Da Pillow
    10.1 `load_default(size=...)` ritorna un TTF scalabile (Aileron), quindi
    il fallback è accettabile — ma il nome del font va RITORNATO e loggato per
    sample: "sul cluster il font di sistema non c'era" deve essere leggibile
    in Weave, non da indovinare.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px), Path(path).name
        except OSError:
            continue
    return ImageFont.load_default(size=px), "pillow-default"


def paint_important(
    src: str | Path,
    dst: str | Path,
    *,
    text: str = OVERLAY_TEXT,
    position: str = "top",
    height_frac: float = 0.18,
) -> dict:
    """Dipinge `text` su una banda piena in cima (o in fondo) al frame.

    `src` e `dst` possono coincidere (la strategy sovrascrive i PNG nella
    tmpdir): `Image.open(...).convert(...)` carica per intero, quindi il file
    non è più aperto quando si salva.

    **La dimensione dell'immagine NON cambia** (asserito): allargare il canvas
    per far posto alla banda cambierebbe l'aspect ratio → `smart_resize`
    darebbe una griglia diversa → i frame dipinti e quelli puliti non
    sarebbero più confrontabili e il conteggio token cambierebbe in silenzio.

    Ritorna la diagnostica da loggare per sample:
    `{"font", "text_px", "band_px", "size"}`.
    """
    if position not in ("top", "bottom"):
        raise ValueError(f"position sconosciuta: {position!r}. Valide: 'top', 'bottom'")
    if not 0.0 < height_frac < 1.0:
        raise ValueError(f"height_frac fuori range: {height_frac!r}")

    img = Image.open(src).convert("RGB")
    w, h = img.size

    band_px = max(1, round(height_frac * h))
    text_px = max(1, round(_TEXT_OF_BAND * band_px))
    font, font_name = _load_font(text_px)

    draw = ImageDraw.Draw(img)
    # Su frame stretti (o `max_pixels` bassi) la parola non ci sta in
    # larghezza: si rimpicciolisce finché entra.
    while text_px > _MIN_TEXT_PX and draw.textlength(text, font=font) > _MAX_TEXT_WIDTH_FRAC * w:
        text_px -= 2
        font = font.font_variant(size=text_px)
    if text_px <= _MIN_TEXT_PX:
        raise RuntimeError(
            f"overlay illeggibile: corpo sceso a {text_px}px su un frame "
            f"{w}x{h} (height_frac={height_frac}). Frame troppo piccolo o "
            "parola troppo lunga — non dipingere e basta, l'arm misurerebbe "
            "solo occlusione."
        )

    y0 = 0 if position == "top" else h - band_px
    draw.rectangle([0, y0, w, y0 + band_px], fill=BAND_FILL)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((w - tw) / 2 - bbox[0], y0 + (band_px - th) / 2 - bbox[1]),
        text, font=font, fill=TEXT_FILL,
    )

    if img.size != (w, h):
        raise RuntimeError(f"overlay ha cambiato la dimensione: {(w, h)} -> {img.size}")
    img.save(dst)
    return {"font": font_name, "text_px": text_px, "band_px": band_px, "size": (w, h)}
```

Facoltativo ma utile: un `if __name__ == "__main__":` che dipinge un frame di
prova, applica `smart_resize` e stampa l'altezza del testo in pixel-modello —
è la verifica §8.0 a costo zero (precedente in repo: `_DEMO_CASES` di
`utils/mcq.py`). Lo snippet è in appendice A.

### 6.2 `strategies/visual_prompt.py` — la strategy

Classe nuova, `name = "visual_prompt"`. **Non** un knob su `attention_marker`:
quello è un arm da misurare e non va toccato mentre gira.

Il docstring del modulo deve dire (stile del repo: il perché, non il cosa):
di quale variante si tratta, perché non c'è split di blocchi, perché entrambi
i frame della cella, e che il confronto pulito è interno al probe
(`cell_select=attention` vs `random`).

```python
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from models.media import Text, VideoFrames
from models.qwen import Qwen3VL
from models.signals import SupportsSignals
from utils.attn_core import ROW_SELECTORS, RankedCell, ranked_cells_from_attention, reconstruct_frame_indices
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
            raise ValueError(f"cell_select sconosciuto: {self.cell_select!r}. Valide: 'attention', 'random'")
        if self.query_rows not in ROW_SELECTORS:
            raise ValueError(f"query_rows sconosciuto: {self.query_rows!r}. Valide: {sorted(ROW_SELECTORS)}")
        if self.overlay_position not in ("top", "bottom"):
            raise ValueError(f"overlay_position sconosciuta: {self.overlay_position!r}")
        if self.sink_filter and self.sink_percentile <= 0:
            # Stessa trappola di attention_marker: percentile=0 NON spegne il
            # filtro (la soglia diventa il massimo e l'argmax viene azzerato
            # comunque). Per l'attenzione grezza: sink_filter=false.
            raise ValueError("sink_percentile <= 0 con sink_filter=true: usa sink_filter=false")

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
        # --- guardie (vedi la tabella §6.2bis per il perché di ognuna) ---
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

        # --- asserzione righe-query (copiata IDENTICA da attention_marker) ---
        # `question_rows` ha un fallback silenzioso (utils/attn_core.py:75):
        # senza "options:" ritorna TUTTE le righe e l'arm degraderebbe a
        # query_rows=all credendo di misurare `question`. Su Video-MME
        # format_mcq_prompt scrive sempre "Options:", quindi è un'asserzione,
        # non un ramo atteso: si fallisce il sample in modo RUMOROSO.
        n_query_rows = n_query_tokens = None
        if va is not None:
            n_query_tokens = len(va.query_tokens)
            rows = ROW_SELECTORS[self.query_rows](va.query_tokens)
            n_query_rows = len(rows) if rows else n_query_tokens
            if self.query_rows != "all" and n_query_rows >= n_query_tokens:
                raise RuntimeError(
                    f"query_rows={self.query_rows!r} non ha tagliato nulla "
                    f"(n_query_rows={n_query_rows} == n_query_tokens={n_query_tokens}): "
                    "selettore degradato al fallback 'tutte le righe'."
                )

        # --- ranking celle ---
        top1_pct = peak_cell = t_cells = peak_cell_sink_filtered = None
        ranked_cells: list[RankedCell] | None = None
        if va is not None:
            ranked_cells = ranked_cells_from_attention(
                va, percentile=self.sink_percentile, border=self.sink_border,
                query_rows=self.query_rows, sink_filter=self.sink_filter,
            )
            top1_pct, peak_cell, t_cells = ranked_cells[0].pct, ranked_cells[0].cell, va.t
            # Check §8.3: se la cella col filtro acceso coincidesse SEMPRE con
            # `peak_cell`, il knob sink_filter non arriva a destinazione.
            peak_cell_sink_filtered = peak_cell if self.sink_filter else ranked_cells_from_attention(
                va, percentile=self.sink_percentile, border=self.sink_border,
                query_rows=self.query_rows, sink_filter=True,
            )[0].cell

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
                # Frame REALI: RankedCell.frames/.rep_frame/.t_sec sono dummy.
                # La cella `ti` copre le posizioni 2*ti e 2*ti+1 della lista
                # campionata da reconstruct_frame_indices.
                frame_indices, fps = reconstruct_frame_indices(
                    video_path, budget.nframes, video_start=video_start, video_end=video_end,
                )
                paths, paint_tmp_dir = _extract_all_frames(video_path, frame_indices)
                tmp_dirs.append(paint_tmp_dir)

                # ENTRAMBI i frame della cella (§4.1). Il clamp serve solo con
                # nframes dispari, dove l'ultima cella ha un frame solo.
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
            "peak_cell_sink_filtered": peak_cell_sink_filtered,
            "t_cells": t_cells,
            "painted": painted,
            "paint_skip_reason": paint_skip_reason,
            "painted_cells": painted_cells,
            "n_painted_cells": len(painted_cells) if painted_cells is not None else None,
            "painted_frames": painted_frames,
            "n_frames_sent": n_frames_sent,
            "overlay_font": overlay_font,
            "overlay_text_px": overlay_text_px,
            "n_query_rows": n_query_rows,
            "n_query_tokens": n_query_tokens,
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
            # `correct_pass1_*` di evals/base.py:199 — "l'overlay ha rotto un
            # sample già corretto?" si legge intra-sample, senza join fra run.
            result["pred_pass1"] = (
                letters.index(signal.pred_letter) if signal.pred_letter is not None else None
            )
        return result
```

### 6.2bis Perché ogni guardia

| guardia | motivo | cosa succederebbe senza |
|---|---|---|
| `SupportsSignals` | serve `VisualAttention` dal pass 1 | `AttributeError` opaco dentro Weave |
| `Qwen3VL` | metodologico + `pass_video_metadata` (§6.2) | su Qwen2.5-VL `_prepare_inputs` solleva comunque, ma con un messaggio che non spiega la scelta |
| `frames is None` | senza video sorgente non ci sono timestamp reali | timestamp finti nel prompt, arm non definito |
| `not double_frames` | cella = 1 frame invece di 2 | si dipingerebbe il frame sbagliato, in silenzio |
| `n_query_rows < n_query_tokens` | fallback silenzioso di `_rows_before_marker` | si misura `query_rows=all` credendo di misurare `question` |
| `img.size` invariata (in `paint_important`) | aspect ratio → griglia → token | conteggio token diverso fra frame dipinti e puliti |

### 6.3 Registrazione e config

`strategies/__init__.py` — senza l'import `Strategy.get` **non vede** la
classe (`strategies/base.py`, dispatch via `__subclasses__()`):

```python
from .visual_prompt import VisualPromptStrategy
# ... e in __all__ (lista ordinata alfabeticamente):
    "VisualPromptStrategy",
```

`conf/config.yaml` — commento di `active.strategy` (riga 6) aggiornato con
`| visual_prompt`, e nuovo preset in coda al blocco `strategy:`:

```yaml
  # Visual prompting: la parola IMPORTANT DIPINTA sui frame della cella di
  # picco, invece di marcatori testuali attorno al blocco (variante "C").
  # Un solo blocco video: nessuno split, struttura token identica alla
  # baseline. Vedi strategies/visual_prompt.py e docs/piano_visual_prompting.md.
  visual_prompt:
    name: visual_prompt
    always_highlight: true       # nessun gate: si interviene su tutti i sample
    top_k: 1                     # celle dipinte (ENTRAMBI i frame di ognuna)
    cell_select: attention       # (attention | random) random = arm di controllo
    query_rows: question         # NON "all": appaiato con attention_marker
    sink_filter: false           # attenzione GREZZA, come attention_marker
    sink_percentile: 25.0        # conta solo se sink_filter=true
    sink_border: 0
    random_seed: 0
    overlay_position: top        # (top | bottom) geometria = knob, la parola no
    overlay_height_frac: 0.18    # altezza banda in frazione dell'altezza frame
    capture_attention: false     # opt-in: dump capture.pt/frames per attn_explorer
```

`main.py` non va toccato (`Strategy.get(cfg.strategy.name, cfg.strategy)`,
riga 98; `capture_dir` viene impostato alla 99-100 se
`capture_attention: true`).

### 6.4 `scripts/probe_overlay_legibility.py` — il gate

Standalone, non passa da Hydra/Weave (precedente: `scripts/probe_entity_extraction.py`).
Domanda diretta al modello, senza MCQ: **legge la parola?** Sweep su
`height_frac` per tarare il knob in un colpo solo.

```python
"""Il modello LEGGE la parola dipinta sui frame? — gate di `visual_prompt`.

Se la risposta è no, `strategies/visual_prompt.py` non misura visual
prompting ma occlusione: questo script va PRIMA di qualunque accuracy
(docs/piano_visual_prompting.md §8.1). Sweepa `height_frac` per tarare il
knob `overlay_height_frac` del preset.

Uso:  uv run python scripts/probe_overlay_legibility.py <dir_video> [n_video]
"""
import sys
import tempfile
from pathlib import Path

import decord
import torch
from PIL import Image

from models import Qwen3VL2B
from models.media import Text, VideoFrames
from transformers import GenerationConfig
from utils.attn_core import reconstruct_frame_indices
from utils.overlay import OVERLAY_TEXT, paint_important

NFRAMES, MAX_PIXELS = 24, 151200          # conf/config.yaml, video_mme
HEIGHT_FRACS = (0.12, 0.18, 0.25)
CELL = 5                                   # cella dipinta, fissa: qui non si misura l'attenzione
QUESTION = (
    "Some frames of this video may have a word written on a colored band. "
    "Which word is it? Answer with the single word only, or NONE."
)

def main() -> None:
    video_dir, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10
    videos = sorted(p for p in video_dir.rglob("*.mp4"))[:n]
    vlm = Qwen3VL2B(torch_dtype=torch.bfloat16, device_map="auto", max_pixels=MAX_PIXELS)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=16)

    for frac in HEIGHT_FRACS:
        hits = 0
        for video in videos:
            idx, fps = reconstruct_frame_indices(str(video), NFRAMES)
            vr = decord.VideoReader(str(video))
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for i, j in enumerate(idx):
                    p = Path(td) / f"f{i:04d}.png"
                    Image.fromarray(vr[min(int(j), len(vr) - 1)].asnumpy()).save(p)
                    paths.append(str(p))
                info = None
                for pos in (2 * CELL, 2 * CELL + 1):
                    info = paint_important(paths[pos], paths[pos], height_frac=frac)
                media = VideoFrames(paths, max_pixels=MAX_PIXELS,
                                    frames_indices=[int(i) for i in idx], fps=fps)
                raw = vlm.generate(vlm.build_messages(media, Text(QUESTION)), generation_config=gen_cfg)
            ok = OVERLAY_TEXT.lower() in raw.lower()
            hits += ok
            print(f"[{frac}] {video.name}: {raw.strip()!r} {'OK' if ok else 'MISS'} (font={info['font']}, {info['text_px']}px)")
        print(f"=== height_frac={frac}: {hits}/{len(videos)} letti ===")

if __name__ == "__main__":
    main()
```

**Criterio di accettazione**: ≥ 9/10 a un `height_frac` che non superi ~0.20
(oltre si occlude troppo, §4.2). Il valore scelto va scritto nel preset e
citato nel diario. Se nessun `height_frac` accettabile passa: opzione B di §3
(pre-resize), e solo dopo si misura.

Estensione a costo quasi zero, da fare nella stessa sessione: una seconda
domanda *"At which second does that word appear?"*. Se il modello legge la
parola ma non sa collocarla nel tempo, è un'informazione sul perché l'arm
potrebbe non guadagnare — e si ottiene qui, non dopo tre ore di fullset.

### 6.5 `scripts/sbatch/ablation-videomme/visual_prompt_24_qwen3.sbatch`

Copia di `marker_24_qwen3.sbatch` con `strategy=visual_prompt`, `NAME`,
tag e commenti aggiornati. Da riprendere **alla lettera**:

- `#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"` —
  **senza** `gpu_RTX6000_24G`: su Turing Qwen3-VL-2B costa ~44 s/sample
  contro ~3.4 su L40S (`docs/qwen3_vl_2b_baseline.md`). SLURM stampa un INFO
  consultivo che suggerisce di riaggiungerlo: ignorarlo.
- il blocco `PROBE_ARGS` con `LIMIT`/`shuffle=true` e il gruppo wandb `-test`,
  così la probe non finisce fra i numeri di produzione;
- `source scripts/lib/notify.sh && tg_job_start` e il resto del preambolo.

Riga di lancio:

```bash
uv run python main.py \
    model=qwen3_vl_2b_attn \
    dataset=video_mme \
    strategy=visual_prompt \
    dataset.nframes=24 \
    hf_home=/work/tesi_avalenza/hf \
    shard=${SLURM_ARRAY_TASK_ID} \
    num_shards=${SLURM_ARRAY_TASK_COUNT} \
    "${PROBE_ARGS[@]}" \
    wandb.group=${GROUP} \
    wandb.name=qwen3_vl_2b_attn-${NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    wandb.tags=[ablation,video_mme,qwen3_vl_2b,visual_prompt_24] \
    "$@"
```

L'arm di controllo si lancia dallo stesso file, senza duplicarlo:

```bash
sbatch scripts/sbatch/ablation-videomme/visual_prompt_24_qwen3.sbatch \
    strategy.cell_select=random \
    "wandb.tags=[ablation,video_mme,qwen3_vl_2b,visual_prompt_random_24]"
```

**Costo atteso**: lo stesso dell'arm marcatori (estrazione PNG di tutti i
frame + secondo pass su **tutti** i sample, `always_highlight=true`); la
pittura è trascurabile. Sul sovraccosto misurato il 2026-08-28 (10.5 s per
sample intervenuto, `docs/260901-tabelle-risultati.md` §2) si va verso ~14
s/sample: ~15 min per una probe da 60, ~3,5 h/shard per un fullset a 3 shard
— **fuori dalla walltime a 4 h con poco margine**. Ritarare su
`model_latency_mean` della probe, non su questa stima.

---

## 7. Come si prova, in locale e su HPC

```bash
# 1. rendering, senza GPU né pesi (§8.0)
uv run python -m utils.overlay              # se si aggiunge il blocco __main__

# 2. la strategy è registrata? (nessun modello caricato)
uv run python -c "from strategies import Strategy; print(Strategy.get('visual_prompt', {}).name)"

# 3. config risolto, nessuna esecuzione
uv run python main.py strategy=visual_prompt --print-config

# 4. smoke end-to-end su 2 sample, su un nodo con GPU, senza sporcare wandb
uv run python main.py model=qwen3_vl_2b_attn dataset=video_mme \
    strategy=visual_prompt dataset.nframes=24 limit=2 wandb.mode=disabled

# 5. gate di leggibilità (§8.1)
uv run python scripts/probe_overlay_legibility.py /work/tesi_avalenza/datasets/video_mme 10

# 6. probe da 60 sample
LIMIT=60 sbatch --array=0 --time=00:40:00 \
    scripts/sbatch/ablation-videomme/visual_prompt_24_qwen3.sbatch
```

`scripts/debug.sh` è specifico dell'utente (srun + PyCharm remote-dev): non
usarlo da un ambiente diverso dal suo.

---

## 8. Probe, in ordine

I punti 8.0–8.4 sono **correttezza**, non risultato. 8.1 è un **gate**: se non
passa, i numeri di accuracy non significano niente.

**8.0 — offline, senza GPU** (appendice A). Dipingere un frame, applicare
`smart_resize` al `max_pixels` del dataset, salvare il PNG e **guardarlo**.
Fissa `overlay_height_frac` e dà il numero da scrivere nel diario (altezza
della parola in pixel-modello).

**8.1 — leggibilità dal modello (GATE)** — §6.4. Attesa: ≥ 9/10.
Se fallisce: (1) alzare `overlay_height_frac`, (2) opzione B di §3. Non
"proviamo comunque il fullset".

**8.2 — `n_query_rows < n_query_tokens` su tutti i sample.** Se coincidono il
selettore non ha tagliato (la strategy solleva da sé: nessun sample deve
fallire per questo).

**8.3 — filtro sink davvero spento.** `peak_cell` e `peak_cell_sink_filtered`
non devono coincidere sempre; se coincidono, il knob non arriva a
`ranked_cells_from_attention`.

**8.4 — invarianza strutturale.** `n_frames_sent == dataset.nframes` su ogni
sample; `painted_frames` sempre `⊂ [0, nframes)` e di cardinalità `2*top_k`
(salvo l'ultima cella con `nframes` dispari); `overlay_font` costante e mai
`pillow-default` se sul nodo c'è DejaVu; nessun `RuntimeError` di
`paint_important`.

**8.5 — `mcq_accuracy` esiste nel summary.** Un OOM di `full_visual_attention`
viene catturato da Weave per sample e lascia il job `finished` con 0
predizioni valide: **lo stato del job non è un check**.

**8.6 — `pred_fallback` ~0.** Se è alto, il modello sta rispondendo in un
formato non parseabile: sospetto numero uno, sta nominando la parola dipinta
nonostante l'istruzione (§5) — controllare qualche `raw`.

**8.7 — `model_latency_mean`**, per ritarare la walltime del fullset.

**8.8 — solo dopo, l'accuracy**, in quest'ordine:
1. `visual_prompt` (attention) **contro il proprio `random`**, a parità di
   tutto il resto → attribuisce alla *posizione*;
2. contro `baseline_24` → l'arm nel complesso guadagna o perde;
3. contro `marker_24` (quando avrà i suoi numeri: la sua probe non è ancora
   girata) → **il confronto interessante**: stesso segnale, stesse celle,
   stesso gate, canale di consegna diverso;
4. breakdown per `task_type`, con attenzione a `ocr problems` (§4.3), e
   `correct_pass1_true` / `correct_pass1_false` (quanto rompe fra i già
   corretti, quanto recupera fra gli sbagliati).

L'accuracy dell'arm è quella **pooled** sui 3 shard (Σ corretti / Σ sample),
non la media delle tre run wandb.

> **Nota su `BREAKDOWN_KEYS`** (`evals/base.py:120`): non contiene i campi di
> questo arm. I breakdown automatici che si ottengono sono quelli già in
> lista (`duration`, `task_type`, `pred_fallback`) più `correct_pass1_*`.
> Aggiungere chiavi lì cambierebbe le metriche di **tutti** gli arm: non
> farlo per questo probe; `painted_cells` & co. si leggono per sample in
> Weave.

---

## 9. Definition of done

- [ ] `utils/overlay.py` con `paint_important` + asserzione dimensione + fail-fast font/corpo
- [ ] `pillow>=10.1` in `pyproject.toml`, `uv lock` rifatto
- [ ] `strategies/visual_prompt.py` con le 4 guardie, l'asserzione righe-query e i campi di §6.2
- [ ] registrata in `strategies/__init__.py` (+ `__all__`) e preset in `conf/config.yaml` (+ commento di `active.strategy`)
- [ ] `scripts/probe_overlay_legibility.py` e sbatch `visual_prompt_24_qwen3.sbatch`
- [ ] §7 punti 1–4 verdi in locale
- [ ] §8.0 fatto, `overlay_height_frac` scelto e scritto nel preset
- [ ] §8.1 passato (≥9/10) e il numero riportato nel diario
- [ ] probe da 60 con §8.2–8.7 verdi, walltime ritarata su `model_latency_mean`
- [ ] fullset `attention` + fullset `random`, 3 shard ciascuno
- [ ] riga in `docs/260901-tabelle-risultati.md` e nota nel diario

---

## 10. Cosa NON fare

- **Non leggere l'accuracy prima del gate §8.1.** Senza leggibilità l'arm
  misura occlusione, non prompting.
- Non usare `ImageFont.load_default()` senza `size` (bitmap 11 px, degrado
  silenzioso).
- Non ridimensionare, ritagliare o allargare il frame per far posto alla
  banda: cambia aspect ratio → cambia la griglia → l'arm non è più
  confrontabile con la baseline.
- Non dipingere un solo frame della cella (§4.1).
- Non passare un `VideoFrames` senza `frames_indices`/`fps` reali: i timestamp
  del prompt cambierebbero rispetto al pass 1 (secondo intervento non voluto).
- Non toccare `attention_marker` né `attention_highlight` (arm in corso di
  misura), né i loro default (`query_rows: all` su highlight, filtro sink
  attivo in `ranked_cells_from_attention`).
- Non usare `RankedCell.frames` / `.rep_frame` / `.t_sec`.
- Non aggiungere chiavi a `BREAKDOWN_KEYS` per questo arm.
- Non estendere a Qwen2.5-VL senza aver prima deciso il regime metadata
  (§6.2).
- Non rendere la parola o l'istruzione dei knob di config: cambiarle cambia
  l'esperimento, non la sua configurazione.
- Non stimare la walltime sui fattori di costo di `attention_highlight` (che
  riusa gli stessi token visivi e non estrae PNG).
- Non lanciare con `gpu_RTX6000_24G` nel `--constraint`.

---

## Appendice A — verifica offline riproducibile

Gira senza GPU e senza pesi. Fissa `overlay_height_frac` e dice quanto è alta
la parola nei pixel che il modello vede davvero. È anche il candidato naturale
per il blocco `if __name__ == "__main__":` di `utils/overlay.py`.

```python
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from utils.overlay import paint_important

MAX_PIXELS = 151200          # conf/config.yaml, dataset.video_mme
FACTOR = 32                  # Qwen3-VL: image_patch_size(16) * merge(2). Qwen2.5-VL: 28
HEIGHT_FRAC = 0.18

info = paint_important("frame.png", "frame_painted.png", height_frac=HEIGHT_FRAC)
w, h = info["size"]
th, tw = smart_resize(h, w, factor=FACTOR, min_pixels=FACTOR * FACTOR * 4, max_pixels=MAX_PIXELS)
Image.open("frame_painted.png").resize((tw, th), Image.BICUBIC).save("frame_seen_by_model.png")

print(f"font={info['font']}  sorgente {w}x{h} -> modello {tw}x{th} "
      f"(griglia {tw // FACTOR}x{th // FACTOR})")
print(f"testo: {info['text_px']}px sorgente -> ~{info['text_px'] * th / h:.0f}px modello "
      f"({info['text_px'] * th / h / FACTOR:.1f} celle di griglia)")
```

**Valori di riferimento già calcolati** (`max_pixels=151200`):

| sorgente | fattore 28 | fattore 32 |
|---|---|---|
| 1280×720 / 1920×1080 | 504×280 (18×10) | 512×288 (16×9) |
| 640×480 | 448×336 (16×12) | 448×320 (14×10) |

Numeri **misurati** eseguendo l'implementazione di riferimento di §6.1 su un
frame 1280×720 (`max_pixels=151200`, fattore 32):

| `height_frac` | banda | corpo (sorgente) | testo sul modello |
|---|---:|---:|---:|
| 0.12 | 86 px | 64 px | ~26 px (0,8 celle) |
| **0.18** | 130 px | 98 px | **~39 px (1,2 celle)** |
| 0.25 | 180 px | 135 px | ~54 px (1,7 celle) |

A `0.18` il PNG a 512×288 è **leggibile a occhio** (parola bianca su banda
rossa, ~2/3 della larghezza), ed è il default proposto nel preset; `0.12` è
al limite. Il ciclo di fit in larghezza non scatta in nessuno dei tre casi;
su 320×240 scatta il solo ridimensionamento della banda (corpo 32 px). Su sorgenti più stretti o `max_pixels` più bassi (LVBench
gira a 50176) il ciclo scatta e il testo si rimpicciolisce: **rifare questo
calcolo prima di portare l'arm su un altro dataset.**

**Verifica opzionale, se il processor Qwen3-VL è in cache** — che dipingere
non cambi il conteggio dei token:

```python
# stesso `proc(...)` dell'appendice A di docs/piano_marcatori_frame_qwen3.md,
# una volta con i PNG puliti e una con quelli dipinti:
# video_grid_thw DEVE essere identico nei due casi.
```

## Appendice B — riferimenti

- `docs/piano_marcatori_frame_qwen3.md` — la variante "B", di cui questo piano
  riusa selezione celle, gate, controllo e checklist della probe
- `strategies/attention_marker.py` — codice da cui importare
  `_extract_all_frames` e copiare l'asserzione righe-query
- `strategies/attention_highlight.py:37-58,155` — la variante "A" e `_sample_rng`
- `utils/attn_core.py:75,142,616,796` — fallback silenzioso di
  `_rows_before_marker`, `ROW_SELECTORS`, `ranked_cells_from_attention`,
  `reconstruct_frame_indices`
- `models/media.py::VideoFrames` — campi `frames_indices`/`fps` e loro
  contratto; `models/qwen.py:25,156-250` — canale `_VIDEO_METADATA_KEY` e i
  due regimi di `_prepare_inputs`
- `lot/prompts.py:142,147` — "Do not output the markers" e il prompt della
  variante bbox disegnata di LoT
- `docs/260803-tabelle_risultati.md` §1 — `highlight` e `random`
  indistinguibili: perché il controllo casuale non è opzionale
- `docs/260901-tabelle-risultati.md` §2 — costi misurati su Qwen3-VL-2B
- `docs/qwen3_vl_2b_baseline.md` — baseline 54.44% pooled e il vincolo GPU
