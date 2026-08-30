"""`AttentionHighlightStrategy` — evidence highlighting invece di
ricampionamento: al pass 2 il modello vede ESATTAMENTE gli stessi frame del
pass 1, cambia solo il prompt, che ora gli dice DOVE guardare.

Adattamento al video di Look Twice (LoT, Morini et al. 2026,
arXiv:2604.01280 — `lot/` in questo repo). LoT, sul lato visivo, fa: (i)
attenzione domanda→token visivi aggregata su layer intermedi e teste, (ii)
filtro dei token-sink, (iii) bbox dalla distribuzione risultante come
centroide ± β·σ (Eq. 4-6, β=2), (iv) evidenziazione dell'evidenza così
localizzata nel prompt.

Tre differenze rispetto al paper, tutte volute:

1. **Si evidenziano le `top_k` CELLE (una per una), non una regione ricavata
   dalla dispersione.** LoT deriva dal centroide ± β·σ un bbox, cioè una
   regione connessa la cui ampiezza dipende da quanto è dispersa
   l'attenzione. La trasposizione 1D diretta è stata implementata e poi
   rimossa: su un profilo di sole `t` celle il pavimento quasi uniforme
   dell'attenzione gonfia σ, e anche sottraendo il livello di caso
   l'intervallo degenera a tutto il video appena l'attenzione è bimodale
   (due picchi lontani) — che nel video è il caso NORMALE, non l'eccezione:
   l'evidenza per una domanda può stare in due momenti scollegati, mentre un
   oggetto in un'immagine sta in un posto solo.

   Ogni cella scelta viene comunque comunicata come INTERVALLO, ma la sua
   ampiezza è quella della cella stessa (`durata/t`), cioè la risoluzione
   reale del segnale, non una stima di dispersione — vedi `_cell_span_frac`.
   Celle adiacenti vengono fuse in un tratto solo (`_merge_spans`).

2. **L'attenzione è mediata su TUTTI i token della domanda**, non sui soli
   token dell'oggetto target (`T_obj`, Eq. 1-2 del paper). Su KB-VQA
   l'entità è isolabile ("female butterfly"); in un MCQ video non c'è un
   oggetto target estraibile senza parsing della domanda. `question_rows`
   in `utils/attn_core.py` è il punto da cui partire se si volesse
   restringere.

3. **L'evidenziazione è un PUNTATORE TESTUALE, non un marker attorno ai
   token visivi.** È la variante "A" discussa in sede di design, ed è anche
   ciò che il codice di LoT fa davvero: i marker
   `<START_IMPORTANT_IMG>`/`<END_IMPORTANT_IMG>` di `lot/retrieval.py:394-396`
   avvolgono l'INTERA immagine (due item di testo inseriti prima e dopo
   l'item immagine nella `content` list), non un sottoinsieme di token; la
   selettività spaziale, lì, viene dal crop o dal bbox disegnato, con il
   prompt `SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA` ("Focus on the {entity}
   in the image highlighted by the {color} bbox") a spiegarlo al modello. Il
   puntatore testuale è l'analogo diretto di quest'ultimo.

   La variante alternativa — marker inseriti DENTRO il blocco visivo, attorno
   alle celle scelte — non è implementata qui: su Qwen2.5-VL richiede
   `position_ids` espliciti, perché `get_rope_index`
   (`modeling_qwen2_5_vl.py:1096-1121`) raggruppa i token per modalità
   leggendo `mm_token_type_ids` e consuma un `grid_thw` per gruppo, quindi il
   testo inserito in mezzo spezza il gruppo video in due. Su Qwen3-VL sarebbe
   invece nativo (il formato è già `<t seconds><|vision_start|>cella<|vision_end|>`
   per cella, e `get_rope_index` splitta apposta `video_grid_thw`): da quando
   esistono i preset `qwen3_vl_*_attn` (`models/qwen_attn.py`,
   `docs/qwen3vl_signals.md`) è una variante implementabile su quella
   famiglia, ma resta non implementata.

Perché è interessante confrontarla con `entropy_attention_resample`: a
parità di gate e di segnale, il resampling dà al modello frame NUOVI (nuova
informazione), l'highlighting no — riusa gli stessi frame e cambia solo il
peso che il modello dovrebbe dargli. Se il picco di attenzione del pass 1
era sbagliato, il resampling può comunque correggersi (vede altro),
l'highlighting rischia di amplificare l'errore. È esattamente ciò che le
colonne `fixed`/`broken` dell'analisi win/loss misurano
(`docs/analisi_winloss_resampling_video_mme.md`).

Costo: il pass 2 NON ridecodifica il video (nessun `Video()`/`VideoFrames`
nuovo, nessun PNG estratto) — è un solo forward in più sugli stessi token
visivi, quindi questa strategy è più economica degli arm di resampling a
parità di gate.

Richiede un modello che implementa `SupportsSignals` (i preset `*_attn`:
`qwen2_5_vl_3b_attn`, `qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`) — fail-fast altrimenti, stesso pattern di
`entropy_shortcut`/`entropy_attention_resample`. A differenza del
resampling, funziona anche quando `frames` è fissato dal dataset (es.
MVBench `data_type="frame"`): non serve poter riselezionare i frame, basta
poterli commentare — in quel caso il puntatore è però necessariamente in
unità frazionarie, non in secondi (vedi `pointer_units`).
"""
from __future__ import annotations

import hashlib
import logging
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from models.media import Text
from models.signals import SupportsSignals
from utils.attn_core import QUERY_ROWS_CHOICES, RankedCell, ranked_cells_from_attention, resolve_query_rows
from utils.mcq import parse_mcq_letter

from .base import SamplingBudget, Strategy, video_duration_sec

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer

logger = logging.getLogger(__name__)


# Formulazione del puntatore. Imperativa e breve, modellata su
# `SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA` di `lot/prompts.py` ("Focus on
# the {entity} in the image highlighted by the {color} bbox") — che è la
# forma con cui LoT comunica al modello una regione localizzata. Costanti di
# modulo e non knob di config: sono il testo dell'intervento, cambiarle
# cambia l'esperimento, non la sua configurazione.
POINTER_SPAN_SEC = (
    "Focus on the part of the video between {a:.0f} and {b:.0f} seconds: "
    "that is where the visual evidence relevant to the question is."
)
POINTER_SPAN_FRAC = (
    "Focus on the part of the video between {a:.0f}% and {b:.0f}% of its duration: "
    "that is where the visual evidence relevant to the question is."
)
POINTER_SPANS = (
    "Focus on these parts of the video, where the visual evidence relevant "
    "to the question is: {spans}."
)


def _cell_span_frac(cell: int, t: int) -> tuple[float, float]:
    """Estensione della cella `cell` come frazioni [0,1] della finestra vista
    dal pass 1: `(cell/t, (cell+1)/t)`.

    **Si indica l'intervallo della cella, non il suo centro, perché quella è
    la risoluzione VERA del segnale.** Una cella copre `durata/t` secondi —
    a `nframes=24` (t=12) sono 220 s su un video Video-MME long: dire "990.0
    secondi" comunicherebbe una precisione al decimo che non esiste, su un
    bucket largo quasi quattro minuti. L'intervallo dice al modello ciò che
    davvero si sa.

    L'intervallo risolve anche uno sfasamento: la M-RoPE colloca la cella
    `i` a `i * second_per_grid_ts`, cioè all'ESTREMO SINISTRO
    (`modeling_qwen2_5_vl.py:1012`, `position_temporal = arange(t) *
    time_interval`), mentre i due frame che la compongono cadono nella prima
    metà dell'intervallo. Centro, ancora posizionale e contenuto reale
    differiscono fino a mezza cella (~110 s su un long); `[cell/t,
    (cell+1)/t]` li contiene tutti e tre, quindi è vero comunque li si
    guardi. Il centro `(cell+0.5)/t` resta la convenzione di
    `entropy_attention_resample` per `peak_time` — lì serve a costruire una
    finestra di decodifica, non a parlare al modello, e cade dentro questo
    intervallo.
    """
    if t <= 0:
        return 0.0, 1.0
    return cell / t, (cell + 1) / t


def _sample_rng(seed: int, video_path: str, prompt: str) -> random.Random:
    """RNG deterministico PER SAMPLE, non per processo.

    Un `random.Random` globale darebbe celle diverse a seconda dell'ordine
    di esecuzione — che con `Evaluation` concorrente non è garantito, e che
    fra i 3 shard di un job array è per costruzione diverso: lo stesso
    sample estrarrebbe celle diverse a ogni rilancio, rendendo il controllo
    non riproducibile e non joinabile per-sample con gli altri arm.
    Derivando il seed da `(seed, video_path, prompt)` la cella estratta è
    una funzione pura del sample: stabile fra shard, fra rilanci e fra
    macchine. `random_seed` permette di ripetere il controllo con un'altra
    estrazione senza toccare il codice.
    """
    key = f"{seed}|{video_path}|{prompt}".encode()
    return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Fonde gli intervalli contigui o sovrapposti, ordinati per tempo. Celle
    adiacenti scelte entrambe (es. 4 e 5) descrivono un unico tratto di
    video: elencarle separate ("40-50 s and 50-60 s") leggerebbe come due
    posti diversi."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _fmt_spans(spans: list[tuple[float, float]], *, units: str) -> str:
    """`[(880, 1100), (1540, 1760)]` → `"880-1100 s and 1540-1760 s"` /
    `"33-42% and 58-67%"`. Congiunzione in inglese perché il puntatore è
    inglese come il resto del prompt MCQ."""
    suffix = " s" if units == "seconds" else "%"
    parts = [f"{a:.0f}-{b:.0f}{suffix}" for a, b in spans]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class AttentionHighlightStrategy(Strategy):
    name = "attention_highlight"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        # Gate identico a `entropy_attention_resample`: stesso default, stessa
        # condizione (`answer_entropy > soglia`). È ciò che rende i due arm
        # confrontabili sample-per-sample — cambiando solo l'intervento.
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        # LoT non ha gate: evidenzia sempre. `true` riproduce quel regime
        # (ablation "serve il gate?"), `false` (default) resta confrontabile
        # con gli arm di resampling esistenti.
        self.always_highlight = bool(cfg.get("always_highlight", False))
        # Numero di istanti evidenziati: le `top_k` celle a massa più alta.
        # `top_k=1` = solo la cella di picco, cioè la stessa su cui il ramo
        # `zoom_peak` di `entropy_attention_resample` zooma — è il confronto
        # più pulito fra i due interventi (stessa cella, uno la isola e
        # l'altro la segnala).
        self.top_k = max(1, int(cfg.get("top_k", 3)))
        # `random` = controllo: stesso gate, stesso formato di puntatore,
        # stesso costo, ma le celle sono estratte a sorte invece che dalle
        # masse di attenzione. Serve ad attribuire l'eventuale guadagno: se
        # indicare una cella a caso rende quanto indicare il picco, il
        # contributo dell'attenzione è nullo e si sta misurando solo
        # l'effetto del prompt. Non è un'ipotesi di scuola — il probe
        # `2dodrndg` (40 sample) ha mostrato che il picco cade sull'ULTIMA
        # cella nel 28% dei casi (3.4× il livello di caso) e sulle due celle
        # estreme nel 40% (atteso 17%), con `top1_pct` mediano 15.7% contro
        # un livello di caso dell'8.3%: il segnale è concentrato debolmente
        # e in buona parte posizionale, plausibilmente per recency causale
        # (le righe-query sono i token della domanda, che stanno DOPO il
        # blocco visivo, quindi l'ultima cella è la più vicina).
        #
        # `antipeak` = SONDA DI SENSIBILITÀ, non un arm da confrontare in
        # accuracy: indica di proposito il punto più lontano dal picco, cioè
        # sviando il modello. Serve a decidere se il canale-prompt è vivo,
        # domanda a monte di "il picco è quello giusto?". Va usata con
        # `always_highlight=true` e letta SOLO sui sample corretti al pass 1
        # (`pred_pass1`): il full eval `random_top1_24` era già una sonda
        # debole in questo senso, ma diluita — puntatore solo sul ~73% col
        # gate aperto, dove l'accuracy è ~42%, quindi quasi nessuno spazio
        # per mostrare un danno. Sui corretti-al-pass-1 si parte da ~88% e
        # lo spazio c'è tutto.
        #
        # L'interpretazione NON dipende da quanto è buono il picco, ed è il
        # motivo per cui questa sonda vale più di un oracolo: se il picco è
        # informativo, l'antipicco è sbagliato → il modello che obbedisce
        # peggiora; se il picco è solo posizionale, l'antipicco è un random
        # con bias opposto → il modello che obbedisce peggiora comunque, su
        # sample che stava già indovinando. In entrambi i rami "nessun
        # danno" ⟹ il puntatore non viene eseguito.
        self.cell_select = str(cfg.get("cell_select", "attention"))
        self.random_seed = int(cfg.get("random_seed", 0))
        self.pointer_units = str(cfg.get("pointer_units", "seconds"))
        self.pointer_position = str(cfg.get("pointer_position", "prepend"))
        # Flag del warning una-tantum di `_warn_flat_clock` (per istanza di
        # strategy, cioè per run: una riga di log, non 2700).
        self._flat_clock_warned = False
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))
        # QUALI righe-query producono il picco. `_capture_spans`
        # (models/qwen_attn.py) le definisce come tutto ciò che segue
        # l'ultimo `<|vision_end|>`, e `ranked_cells_from_attention` ne fa la
        # media: su un prompt MCQ sono domanda + le quattro opzioni +
        # "Answer with the letter of the correct option only." + il
        # generation prompt. Misurato col tokenizer su un prompt Video-MME
        # tipico: 86 token totali di cui 21 di domanda, cioè il 76% delle
        # righe che formano il "picco d'attenzione della domanda" non è la
        # domanda.
        #
        #   all       (DEFAULT) tutte le righe — il comportamento con cui
        #             sono girati highlight_top1_24/random_top1_24/antipeak
        #             e tutti gli arm di `entropy_attention_resample`.
        #             Resta il default proprio per questo: cambiarlo
        #             renderebbe non appaiate le celle di controllo esistenti
        #             sui 2700, che è l'unica ragione per cui un nuovo arm
        #             costa 3 job invece di 9.
        #   qopts     domanda + opzioni, senza boilerplate né generation
        #             prompt. Le opzioni restano perché contengono referenti
        #             visivi legittimi ("the toolbox", "the van"); il
        #             boilerplate no, non ha alcun referente.
        #   question  solo la domanda (`utils.attn_core.question_rows`).
        #
        # Diagnostica offline dello stesso asse, senza spendere una eval:
        # `attn_explorer.diag_query_rows` (misura il bias POSIZIONALE del
        # picco sotto ognuna di queste selezioni). Il quarto valore `entity`
        # usa l'estrattore in-linea del decoder già caricato
        # (`Qwen.extract_entities`, risolto da
        # `utils.attn_core.resolve_query_rows`).
        self.query_rows = str(cfg.get("query_rows", "all"))

        # Knob rimossi (modalità `peak`/`window`, vedi docstring del modulo):
        # senza questo check una CLI o uno sbatch che li passa ancora
        # verrebbe accettato in silenzio e girerebbe con una configurazione
        # diversa da quella intesa — su un job array sono ore buttate.
        removed = {"highlight_select", "beta", "max_span_frac"} & set(cfg)
        if removed:
            raise ValueError(
                f"knob rimossi da {self.name!r}: {sorted(removed)}. Le modalità "
                "'peak'/'window' non esistono più — si evidenziano sempre le "
                "`top_k` celle a massa più alta ('peak' era `top_k=1`)."
            )
        if self.cell_select not in ("attention", "random", "antipeak"):
            raise ValueError(
                f"cell_select sconosciuto: {self.cell_select!r}. "
                "Valide: 'attention', 'random', 'antipeak'"
            )
        if self.query_rows not in QUERY_ROWS_CHOICES:
            raise ValueError(
                f"query_rows sconosciuto: {self.query_rows!r}. "
                f"Valide: {sorted(QUERY_ROWS_CHOICES)}"
            )
        if self.pointer_units not in ("seconds", "fraction"):
            raise ValueError(
                f"pointer_units sconosciuto: {self.pointer_units!r}. Valide: 'seconds', 'fraction'"
            )
        if self.pointer_position not in ("prepend", "append"):
            raise ValueError(
                f"pointer_position sconosciuto: {self.pointer_position!r}. Valide: 'prepend', 'append'"
            )

    def _warn_flat_clock(self, vlm: BaseVLM) -> None:
        """Avverte UNA volta se si sta puntando in secondi a orologio piatto.

        La combinazione `pointer_units=seconds` + `pass_video_metadata=False`
        è silenziosamente insensata: il puntatore comunica istanti a un modello
        le cui celle video stanno tutte sulla stessa posizione temporale
        M-RoPE (vedi il commento esteso in `answer`). È anche il regime in cui
        sono girati `highlight_top1_24`/`random_top1_24`/la sonda `antipeak`,
        quindi NON si fa fail-fast: rilanciare quegli arm per riprodurli deve
        restare possibile. Un warning una tantum, però, evita che la prossima
        run ci ricaschi senza saperlo.

        Una volta sola e non per sample: su 2700 sample sarebbero 2700 righe
        identiche che seppellirebbero il resto del log.
        """
        if self._flat_clock_warned:
            return
        self._flat_clock_warned = True
        # `None` = modello che non espone il knob (non-Qwen): niente da dire.
        if getattr(vlm, "pass_video_metadata", None) is not False:
            return
        logger.warning(
            "%s: pointer_units=seconds con model.pass_video_metadata=false — "
            "second_per_grid_ts vale 0.083 per QUALUNQUE video e get_rope_index "
            "lo tronca a 0, quindi TUTTE le celle video collassano sulla stessa "
            "posizione temporale M-RoPE (non solo quelle con window_sec < t). "
            "Il puntatore sta comunicando secondi a un modello senza coordinate "
            "temporali. Per il regime corretto: model.pass_video_metadata=true "
            "(NON confrontabile con i numeri del README, che sono tutti a "
            "orologio piatto); in alternativa strategy.pointer_units=fraction.",
            self.name,
        )

    def _select_cells(self, ranked: list[RankedCell], *, t: int, rng: random.Random) -> list[int] | None:
        """Le `top_k` celle da evidenziare, o `None` se non c'è nulla da dire.

        `attention`: le prime `top_k` di `ranked` (già ordinata per massa
        decrescente). `None` quando la massa non-sink totale è zero — senza
        segnale non si inventa un puntatore.

        `random`: `top_k` celle distinte estratte a sorte fra le `t`. Qui il
        caso "massa nulla" NON è una ragione per astenersi: la cella non
        dipende dall'attenzione, e far saltare il controllo dove salta l'arm
        vero restringerebbe il suo campione senza motivo.

        `antipeak`: le `top_k` celle più LONTANE NEL TEMPO dal picco — non le
        celle a massa minima. Sono due cose diverse: la massa minima può
        cadere accanto al picco (profilo bimodale stretto), mentre qui serve
        massimizzare la probabilità di indicare un punto dove l'evidenza NON
        c'è, e la distanza temporale è il proxy migliore disponibile.
        Astiene come `attention` quando non c'è segnale: senza picco l'anti-
        picco non è definito.

        A `top_k>1` le celle escono contigue all'estremo opposto (distanze
        decrescenti dal picco) e `_merge_spans` le fonde in un tratto solo —
        voluto: un blocco lontano compatto è un'indicazione più netta di
        `top_k` istanti sparsi.
        """
        if self.cell_select == "random":
            return rng.sample(range(t), k=min(self.top_k, t))
        if all(c.pct <= 0 for c in ranked):
            return None
        if self.cell_select == "antipeak":
            peak = ranked[0].cell
            # `c` come secondo criterio: a parità di distanza (picco centrale)
            # vince la cella di indice minore, deterministicamente — mai
            # l'ordine di `ranked`, che dipende da masse quasi-pari e quindi
            # oscillerebbe col rumore numerico fra GPU diverse.
            far = sorted(range(t), key=lambda c: (-abs(c - peak), c))
            return far[:min(self.top_k, t)]
        return [c.cell for c in ranked[:self.top_k]]

    def _build_pointer(
        self,
        ranked: list[RankedCell],
        *,
        t: int,
        units: str,
        window_sec: float,
        rng: random.Random,
    ) -> tuple[str | None, list[float] | None, list[int] | None, str | None]:
        """`(frase, intervalli_evidenziati, celle_evidenziate, motivo_skip)`.

        `window_sec` è la DURATA della clip che il modello vede, non gli
        estremi nel video sorgente: con un trim (`video_start`/`video_end`,
        es. LVBench `use_time_reference`) qwen-vl-utils decodifica solo quella
        finestra e il modello la percepisce come se iniziasse a 0 — nulla nel
        prompt gli dice a che punto del video originale si trovi. Il puntatore
        deve quindi parlare in secondi RELATIVI alla clip. È la differenza col
        `peak_time = w0 + frac*(w1-w0)` di `entropy_attention_resample`: là il
        valore serve a costruire un nuovo `Video(video_start=...)`, quindi va
        in coordinate assolute del sorgente perché lo consuma il DECODER; qui
        lo consuma il MODELLO, che vive in coordinate della clip.

        `intervalli_evidenziati` è nelle unità del puntatore (secondi nella
        clip o percentuali della sua durata) ed è esattamente ciò che è stato
        detto al modello, loggato per l'analisi offline: una coppia `[a, b]`
        per ogni tratto evidenziato, dopo la fusione dei contigui.
        `celle_evidenziate` sono gli indici di cella corrispondenti, PRIMA
        della fusione: con `cell_select=random` sono l'unico modo di sapere
        a posteriori quale estrazione è uscita, e affiancati a `peak_cell`
        (loggato sempre, anche in modalità random) dicono su quanti sample
        il caso ha pescato proprio la cella dell'attenzione — il tasso di
        coincidenza va tolto dal confronto fra i due arm, perché lì i due
        interventi sono identici. `motivo_skip` è valorizzato solo quando
        non si evidenzia nulla, così il caso "gate aperto ma nessun
        puntatore" resta distinguibile nel log dal caso "gate chiuso".
        """
        def to_units(frac: float) -> float:
            return frac * window_sec if units == "seconds" else 100.0 * frac

        cells = self._select_cells(ranked, t=t, rng=rng)
        if cells is None:
            return None, None, None, "no_signal"
        # Gli intervalli si emettono ordinati per TEMPO (`_merge_spans`
        # ordina), non per rilevanza: una lista in ordine di massa
        # leggerebbe come una classifica, mentre qui conta la posizione nel
        # video.
        spans_frac = _merge_spans([_cell_span_frac(cell, t) for cell in cells])
        spans = [(to_units(a), to_units(b)) for a, b in spans_frac]

        # Accordo singolare/plurale: con `top_k=1` (o con celle tutte
        # contigue, fuse in un tratto solo) si evidenzia un intervallo unico,
        # e "these parts: 880-1100 s" sarebbe sgrammaticato. Qui la frase È
        # l'intervento sperimentale, quindi la forma conta quanto il contenuto.
        if len(spans) == 1:
            tmpl = POINTER_SPAN_SEC if units == "seconds" else POINTER_SPAN_FRAC
            a, b = spans[0]
            pointer = tmpl.format(a=a, b=b)
        else:
            pointer = POINTER_SPANS.format(spans=_fmt_spans(spans, units=units))
        return pointer, [[a, b] for a, b in spans], sorted(cells), None

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

        tmp_dirs: list[Path] = []
        media, media_tmp_dir = self._build_media(video_path, frames, video_start, video_end, budget)
        if media_tmp_dir is not None:
            tmp_dirs.append(media_tmp_dir)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)

        # Ranking celle: calcolato e loggato SEMPRE (anche quando il gate non
        # scatta), stessa policy di `entropy_attention_resample` — serve a
        # confrontare a posteriori come si distribuiscono i segnali rispetto
        # alla decisione di evidenziare.
        top1_pct = peak_cell = t_cells = None
        ranked_cells: list[RankedCell] | None = None
        rows_res = None
        if signal.visual_attention is not None:
            # Risoluzione UNICA del knob (utils.attn_core.resolve_query_rows):
            # per `entity` chiama il decoder già caricato e può degradare a
            # `question` (entity_mapped=False nei metadati); per gli altri
            # valori è il selettore puro di sempre.
            rows_res = resolve_query_rows(
                self.query_rows, signal.visual_attention.query_tokens,
                extract_entities=getattr(vlm, "extract_entities", None),
            )
            ranked_cells = ranked_cells_from_attention(
                signal.visual_attention, percentile=self.sink_percentile, border=self.sink_border,
                rows=rows_res.rows,
            )
            top1_pct, peak_cell = ranked_cells[0].pct, ranked_cells[0].cell
            t_cells = signal.visual_attention.t

        gate_open = self.always_highlight or (
            signal.answer_entropy is not None and signal.answer_entropy > self.entropy_threshold
        )

        highlighted = False
        highlight_spans: list[list[float]] | None = None
        highlight_cells: list[int] | None = None
        highlight_skip_reason: str | None = None
        highlight_units: str | None = None
        pointer: str | None = None
        final_prompt = prompt

        if ranked_cells is not None and gate_open:
            # Con `frames` fissati dal dataset non c'è una finestra temporale
            # nota su cui mappare le celle (i frame sono già estratti, la loro
            # spaziatura reale non è ricostruibile qui): il puntatore può solo
            # essere relativo alla sequenza mostrata, mai in secondi assoluti.
            units = "fraction" if frames is not None else self.pointer_units
            window_sec = 0.0
            if units == "seconds":
                w0 = video_start if video_start is not None else 0.0
                w1 = video_end if video_end is not None else video_duration_sec(video_path)
                window_sec = max(0.0, w1 - w0)
                # Collasso dell'orologio del modello. La M-RoPE spazia le celle
                # di `tokens_per_second * int(second_per_grid_ts)`
                # (`modeling_qwen2_5_vl.py:1125`) e quel troncamento a INTERO
                # azzera l'intervallo quando una cella dura meno di un secondo:
                # tutte le celle finiscono alla stessa posizione temporale e il
                # modello perde ogni nozione di ordine, figurarsi di istante.
                # Parlargli in secondi lì non significa nulla, mentre una
                # posizione relativa resta leggibile dall'ordine in cui i frame
                # compaiono nella sequenza — da cui il fallback qui sotto.
                #
                # ⚠️ ATTENZIONE alla portata reale, che dipende dal regime di
                # preprocessing (`model.pass_video_metadata`, vedi
                # `Qwen._prepare_inputs`):
                #
                # - `pass_video_metadata=true`: `second_per_grid_ts` è quello
                #   vero, il collasso avviene se e solo se `window_sec < t` —
                #   la condizione che il fallback qui sotto testa. A
                #   `nframes=24` scatta sotto i 12 s (~1% della fascia short di
                #   Video-MME; in frame-doubling la soglia raddoppia a 24 s e
                #   la quota sale a ~12%).
                # - `pass_video_metadata=false` (DEFAULT, ed è il regime di
                #   TUTTE le run del README): il processor non riceve i
                #   metadata, `second_per_grid_ts` vale 2/24 = 0.083 per
                #   QUALUNQUE video e `int(0.083) = 0` — quindi il collasso è
                #   **universale**, non condizionale, e il fallback qui sotto
                #   NON protegge niente: sul ~99% dei sample il puntatore parla
                #   in secondi a un modello che non ha coordinate temporali.
                #   È il motivo del warning in `_warn_flat_clock`.
                #
                # Su Qwen3-VL il fenomeno non esiste in nessuno dei due regimi:
                # lì il tempo non passa dalla M-RoPE ma dai timestamp testuali
                # `<x.x seconds>` che il processor scrive nel prompt, con
                # risoluzione al decimo di secondo (e `pass_video_metadata` è
                # sempre `True` su quella famiglia). Il fallback resta attivo
                # comunque, identico per le due famiglie: cambiarlo solo su una
                # renderebbe gli arm non confrontabili proprio sui sample
                # brevi. `highlight_units` nel log dice sempre quale forma è
                # stata usata.
                self._warn_flat_clock(vlm)
                if t_cells and window_sec < t_cells:
                    units = "fraction"
            pointer, highlight_spans, highlight_cells, highlight_skip_reason = self._build_pointer(
                ranked_cells, t=t_cells, units=units, window_sec=window_sec,
                rng=_sample_rng(self.random_seed, video_path, prompt),
            )
            if pointer is not None:
                highlight_units = units
                highlighted = True
                final_prompt = (
                    f"{pointer}\n\n{prompt}"
                    if self.pointer_position == "prepend"
                    else f"{prompt}\n\n{pointer}"
                )

            if self.capture_dir is not None:
                self._save_attention_capture(
                    vlm, video_path, prompt, signal.visual_attention, budget, video_start, video_end,
                    extra={
                        "highlighted": highlighted,
                        "highlight_skip_reason": highlight_skip_reason,
                        "highlight_spans": highlight_spans,
                        "highlight_cells": highlight_cells,
                        "cell_select": self.cell_select,
                        "highlight_units": highlight_units,
                        "top1_pct": top1_pct, "peak_cell": peak_cell, "t_cells": t_cells,
                    },
                )

        try:
            # `media` è quella del pass 1, invariata: è il punto di questa
            # strategy — stessi frame, solo il prompt cambia. Nessuna
            # ridecodifica del video, nessuna tmpdir nuova.
            messages = vlm.build_messages(media, Text(final_prompt))
            raw = vlm.generate(messages, generation_config=gen_cfg)
        finally:
            for tmp_dir in tmp_dirs:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        result: dict = {
            "raw": raw,
            "answer_entropy": signal.answer_entropy,
            "top1_pct": top1_pct,
            "peak_cell": peak_cell,
            "t_cells": t_cells,
            # `highlighted` è la chiave da cui `mcq_accuracy` ricava le due
            # metriche del gate (BREAKDOWN_KEYS in evals/base.py):
            # `seen_highlighted_true` e l'accuracy condizionata.
            "highlighted": highlighted,
            "highlight_skip_reason": highlight_skip_reason,
            "highlight_spans": highlight_spans,
            # Celle effettivamente indicate. Con `cell_select=random` sono
            # l'unico modo di sapere quale estrazione è uscita; affiancate a
            # `peak_cell` danno il tasso di coincidenza fra caso e attenzione.
            "highlight_cells": highlight_cells,
            "cell_select": self.cell_select,
            "highlight_units": highlight_units,
            "pointer": pointer,
            # Metadati della modalità `entity` (None negli altri casi):
            # `query_rows_mode` è la selezione EFFETTIVA (= "question" quando
            # l'entity non mappa verbatim), `entity_mapped` conta il tasso di
            # mapping riuscito per-sample.
            "query_rows_mode": rows_res.mode if rows_res is not None else None,
            "entity_raw": rows_res.entity_raw if rows_res is not None else None,
            "entity_phrases": list(rows_res.entity_phrases) if rows_res is not None and rows_res.entity_phrases is not None else None,
            "entity_mapped": rows_res.entity_mapped if rows_res is not None else None,
        }
        if options is not None:
            pred = parse_mcq_letter(raw, options)
            pred_fallback = False
            if pred is None and signal.pred_letter is not None:
                # Stesso fallback di `entropy_attention_resample`: se il pass 2
                # non produce una lettera parseabile si usa l'argmax della
                # softmax ristretta del pass 1 invece di perdere il sample
                # (README, "52/3800 senza pred parseabile").
                pred = letters.index(signal.pred_letter)
                pred_fallback = True
            result["pred"] = pred
            result["pred_fallback"] = pred_fallback
            # Risposta del pass 1 (argmax della softmax ristretta alle lettere,
            # PRIMA che il puntatore esista): è la chiave che rende leggibile
            # la sonda `antipeak`. Senza, "il puntatore ha rotto questo
            # sample?" si può rispondere solo joinando a posteriori con un
            # altro run (per `example.id`, come in
            # `docs/analisi_winloss_resampling_video_mme.md`) — e quel join
            # confronta due forward su GPU potenzialmente diverse, quindi
            # attribuisce al puntatore anche il rumore numerico. Qui il
            # confronto è intra-sample, stesso forward, zero rumore.
            result["pred_pass1"] = (
                letters.index(signal.pred_letter) if signal.pred_letter is not None else None
            )
        return result
