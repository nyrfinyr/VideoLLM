# Piano — marcatori inline attorno ai frame di picco (Qwen3-VL)

> **Stato**: specifica per l'implementazione, nessun codice scritto. Le
> verifiche della sezione 2 sono già state fatte (probe offline sul solo
> processor, senza GPU): **non rifarle, sono le invarianti su cui poggia il
> disegno**. Tutto il resto è da scrivere.

## 1. Cosa si vuole ottenere

Un nuovo arm che segnala al modello i frame su cui il pass 1 ha concentrato
l'attenzione, **inserendo due marcatori testuali immediatamente prima e
immediatamente dopo i frame scelti**, dentro il flusso visivo:

```
...<|vision_end|><START_IMPORTANT_FRAME><4.5 seconds><|vision_start|>[frame]<|vision_end|><END_IMPORTANT_FRAME><5.5 seconds>...
```

È la **variante "B"** descritta e scartata nel docstring di
`strategies/attention_highlight.py:37-58`, che implementa invece la variante
"A" (puntatore testuale che nomina un intervallo di secondi). Stesso segnale,
stesso gate, stessa scelta delle celle: cambia **solo come l'intervento viene
consegnato al modello**. È anche ciò che fa il codice originale di Look Twice
(`lot/retrieval.py:394-396`), che inserisce `<START_IMPORTANT_IMG>` /
`<END_IMPORTANT_IMG>` come due item di testo prima e dopo l'item immagine
nella `content` list.

**Decisioni già prese, non da rimettere in discussione:**

| | |
|---|---|
| famiglia di modelli | **solo Qwen3-VL** (`qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`). Su Qwen2.5-VL non è verificato e il docstring citato sopra prevede che serva intervenire sui `position_ids`: fuori scope. |
| testo dei marcatori | `<START_IMPORTANT_FRAME>` e `<END_IMPORTANT_FRAME>`, letterali |
| istruzione | **obbligatoria** nel prompt testuale, spiega di guardare i frame fra i marcatori (vedi §5.3) |
| gate | `always_highlight` — nessun gate d'entropia, si interviene su tutti i sample |
| righe-query | **solo i token della domanda**: `query_rows=question` (vedi §3) |
| filtro sink | **disattivato** per questo probe: si classifica sull'attenzione grezza (vedi §3bis) |

## 2. Verifiche già fatte — invarianti da NON riscoprire

Probe offline con `AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")`
su 8 frame sintetici, senza pesi né GPU. Snippet riproducibile in appendice A.

**2.1 Su Qwen3-VL i marcatori inline sono strutturalmente gratuiti.**
`get_rope_index` (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:1068-1070`)
spacchetta `video_grid_thw` in una riga per frame e raggruppa i token per
modalità. Il video è **già** `t` isole visive separate da testo — i timestamp
`<x.x seconds>` che il processor emette per ogni frame
(`processing_qwen3_vl.py:157-170`) — quindi un marcatore in più fra due frame
si fonde nel gruppo di testo già esistente. Il numero di gruppi visivi resta
uguale al numero di griglie consumate. **Nessun `position_ids` esplicito da
costruire.**

**2.2 Spezzare il blocco video funziona, a costo quasi nullo.**

| | `video_grid_thw` | timestamp emessi | token |
|---|---|---|---|
| blocco unico, 8 frame | `[[4,8,8]]` | `0.5 2.5 4.5 6.5` | 107 |
| spezzato 4\|2\|2 + 2 marcatori | `[[2,8,8],[1,8,8],[1,8,8]]` | `0.5 2.5 4.5 6.5` | 117 |

Somma di `t` invariata (4), quindi **token visivi invariati**. Il costo è di
soli 6+6 token di testo: i marcatori non sono token speciali del vocabolario
Qwen, si tokenizzano come testo ordinario.

**2.3 I metadata vanno passati a mano, per blocco — altrimenti l'orologio
riparte da zero.** `process_vision_info` su una lista di frame fabbrica
`{'fps': 2.0, 'frames_indices': [0,1,2,3], 'total_num_frames': 4.0}`: indici
**locali al blocco**. Con tre blocchi si otterrebbero tre video che partono
tutti da 0 s. I timestamp corretti della tabella 2.2 si ottengono solo
passando al processor i `frames_indices` **assoluti** e l'`fps` reale del
video sorgente. Il timestamp di una cella è la media della coppia di frame che
la compone (`_calculate_timestamps` + `temporal_patch_size=2`).

**2.4 I tagli DEVONO cadere su confini di cella.** Il video processor padda
ogni blocco a un numero pari di frame. Con split dispari `t` cresce e tutti i
timestamp si spostano:

| split | `t` per blocco | somma `t` | timestamp |
|---|---|---|---|
| `[4,2,2]` | `[2,1,1]` | 4 ✅ | `0.5 2.5 4.5 6.5` (= blocco unico) |
| `[2,2,4]` | `[1,1,2]` | 4 ✅ | `0.5 2.5 4.5 6.5` |
| `[3,2,3]` | `[2,1,2]` | **5** ❌ | `0.5 2.0 3.5 5.5 7.0` |
| `[1,2,5]` | `[1,1,3]` | **5** ❌ | `0.0 1.5 3.5 5.5 7.0` |

**Conseguenza di disegno**: l'unità marcabile è la **cella** (2 frame nel
regime `double_frames=false`), non il singolo frame. Non è una perdita: la
cella è già la risoluzione reale del segnale d'attenzione.

## 3. Righe-query: già implementato, ma c'è una trappola

Il requisito "l'attenzione domanda→visivo deve essere calcolata sui soli token
della domanda, escluse opzioni e istruzioni" **non richiede codice nuovo**:

- `utils/attn_core.py:142` — `ROW_SELECTORS = {"all", "qopts", "question"}`
- `utils/attn_core.py:120` — `question_rows()` ricostruisce il testo
  concatenando i `QueryToken.text` e taglia **prima** del marcatore
  `"options:"`
- `utils/attn_core.py:616` — `ranked_cells_from_attention(..., query_rows=...)`
- `strategies/attention_highlight.py:284,302,485` — knob già esposto, validato
  e passato

Basta impostare `query_rows: question` nel preset della nuova strategy.

> ⚠️ **TRAPPOLA DA GESTIRE.** `_rows_before_marker`
> (`utils/attn_core.py:75`, il ritorno di fallback è a riga 87) ha un **fallback silenzioso**: se non trova
> `"options:"` nel testo ricostruito ritorna **tutte** le righe. L'arm
> degraderebbe a `query_rows=all` senza sollevare nulla, e si starebbe
> misurando l'esperimento sbagliato credendo di misurare quello giusto.
>
> **Da implementare**: la strategy logga per sample `n_query_rows` e
> `n_query_tokens`. Se coincidono su un sample in cui `query_rows != "all"`,
> il selettore non ha tagliato → **fallire il sample in modo rumoroso**, non
> proseguire. Su Video-MME `evals/base.py::format_mcq_prompt` scrive sempre
> `"Options:"`, quindi non deve mai scattare: è un'asserzione, non un ramo
> atteso.

Contesto: su Qwen2.5-VL `query_rows=question` è già stato misurato
(`docs/260803-tabelle_risultati.md §4`, −0.14 pp contro `all`). Su Qwen3-VL non
è mai girato.

## 3bis. Filtro sink: disattivato — richiede una modifica

Per questo probe le celle vanno classificate sull'attenzione **grezza**, senza
azzerare i token-sink.

**`sink_percentile: 0` NON è il modo di ottenerlo.** `sink_mask`
(`utils/attn_core.py:488`) interpreta `percentile` come "quota SUPERIORE della
distribuzione considerata sink": la soglia è il quantile `1 - percentile/100`,
quindi con `percentile=0` la soglia è il **massimo** e l'argmax della
`sink_map` verrebbe comunque azzerato, per giunta con la renormalizzazione di
`sink_view` ancora attiva. Sarebbe un "quasi spento" travestito da spento, e
chi rileggesse il preset non capirebbe l'intento.

**Il modo giusto**: `sink_view` (`utils/attn_core.py:539`) espone già
`view="all"`, che ritorna la heatmap **invariata**, senza azzeramenti e senza
renormalizzazione. Ma `ranked_cells_from_attention`
(`utils/attn_core.py:616`) ha `view="nonsink"` **cablato**.

Da implementare: un parametro esplicito su `ranked_cells_from_attention` —
es. `sink_filter: bool = True` — che seleziona `view="nonsink"` o
`view="all"`. **Il default deve restare il filtro attivo**: quella funzione è
condivisa con `entropy_attention_resample` e `attention_highlight`, e il suo
docstring impone che il ranking sia calcolato allo stesso modo in tutte le
strategy, altrimenti un confronto win/loss per-sample fra arm misurerebbe
anche la differenza di segnale invece della sola differenza di intervento.
Nuovo knob `sink_filter: false` solo nel preset di questo arm.

> ⚠️ **Cosa aspettarsi, e cosa questo confonde.**
>
> `docs/analisi_entropia_attenzione_sink.md §4.2` misura che il 25% di token
> classificati sink assorbe **45.1%** della massa attenzionale (1.8× la quota
> che gli spetterebbe), in **tutti e 30** i video del campione senza
> eccezioni (minimo osservato 32.0%). Lo stesso doc, §3, afferma che senza il
> filtro "il segnale è quasi piatto, dominato dai token sink che assorbono
> massa indipendentemente dal contenuto". Quelle misure vengono da VNBench
> "retrieval" su Qwen2.5-VL, **non** da Video-MME né da Qwen3-VL: non sono
> una previsione per questo probe, ma sono la ragione per cui il filtro è
> acceso ovunque. Se la cella scelta senza filtro risultasse quasi sempre la
> stessa a prescindere dalla domanda, è questo il fenomeno.
>
> **Confondimento da tenere presente nella lettura**: gli arm già misurati
> (`highlight_top1_24`, `random_top1_24`, `entropy_shift_24`) classificano le
> celle **con** il filtro. Un confronto fra questo arm e quelli mescolerebbe
> due cambiamenti — marcatore inline vs puntatore testuale, e filtro sink
> acceso vs spento — e non sarebbe attribuibile. Il confronto **interno** al
> probe resta invece pulito: `cell_select=attention` contro
> `cell_select=random` gira a parità di filtro, quindi isola il contributo del
> picco. È quello il confronto da leggere.

---

## 4. Cosa riusare senza riscriverlo

| pezzo | dove | cosa dà |
|---|---|---|
| segnali del pass 1 | `models/qwen_attn.py::generate_with_signals`, `full_visual_attention` | entropia della risposta + `VisualAttention` |
| ranking delle celle | `utils/attn_core.py:616 ranked_cells_from_attention` | `list[RankedCell]` ordinata per massa sink-filtrata |
| filtro sink | idem, `view="nonsink"` cablato | **da rendere disattivabile per questo arm, vedi §3bis** |
| indici frame reali | `utils/attn_core.py:783 reconstruct_frame_indices` | `(indici assoluti, fps)` con la stessa formula di qwen-vl-utils |
| estrazione PNG | `strategies/entropy_attention_resample.py:269 _build_phase_shifted_media` | **modello da copiare**: decord → PNG in tmpdir → `VideoFrames`, con contratto di cleanup |
| arm di controllo | `strategies/attention_highlight.py`, knob `cell_select` | `random` con seed mescolato su `(video_path, prompt)` |

> ⚠️ **`RankedCell` ha campi fittizi.** `ranked_cells_from_attention` passa
> dei dummy a `rank_cells` (`frame_indices=range(t)`, `fps=1.0`,
> **`double=True`**) perché ai chiamanti attuali servono solo `cell` e `pct`.
> Quindi `RankedCell.frames`, `.rep_frame` e `.t_sec` **NON sono gli indici
> frame reali** e non vanno usati. Per mappare cella → frame reali:
> `reconstruct_frame_indices(...)` e poi, nel regime `double_frames=false`,
> la cella `ti` copre gli indici `2*ti` e `2*ti+1` della lista campionata
> (equivalente a `GridSpec(double=False).cell_to_frames`).

## 5. Modifiche da fare, in ordine di dipendenza

### 5.1 `models/media.py` — metadata reali sul media item

`VideoFrames` porta oggi solo `sample_fps`, che non basta: il processor calcola
i timestamp da `metadata.frames_indices`, e servono quelli **assoluti** (§2.3).

Aggiungere i campi (o introdurre un `VideoSegment` dedicato, a scelta
dell'implementatore purché `to_content_dict` continui a droppare i `None`):

- `frames_indices: list[int] | None` — indici assoluti nel video sorgente
- `fps: float | None` — fps reale del video sorgente

Vincolo: `to_content_dict` produce il dict per il chat template, che **non**
deve ricevere questi campi (il template conosce solo `type`/`video`/
`max_pixels`/…). I metadata viaggiano su un canale separato, vedi 5.2.

### 5.2 `models/qwen.py::_prepare_inputs` — inoltrare `video_metadata` per blocco

Oggi i metadata arrivano da `process_vision_info(return_video_metadata=True)`
(`models/qwen.py:222-241`). Serve il ramo che, quando i media item portano
`frames_indices`/`fps`, costruisce a mano un
`transformers.video_utils.VideoMetadata` per blocco e lo passa come
`video_metadata=[...]`, con `do_sample_frames=False`.

Senza questo, 5.1 è un campo che nessuno legge e i timestamp ripartono da zero
a ogni marcatore.

### 5.3 `models/base.py` + `models/qwen.py::build_messages` — content list arbitraria

La firma è `build_messages(media: MediaItem, text: Text)`: **un solo** item
visivo e **un solo** testo. Il prompt marcato è
`[video, text, video, text, video, text]`.

Aggiungere un metodo che accetta la lista di parti già ordinata e far
delegare `build_messages` a quello. **Non cambiare la firma esistente**: i 4
call site (`strategies/uniform.py:46`, `strategies/attention_highlight.py:587`,
`strategies/entropy_attention_resample.py:618`, `models/qwen_attn.py:263`)
devono restare invariati, altrimenti si toccano arm già misurati.

### 5.4 `strategies/attention_marker.py` — la strategy

Flusso:

1. **Pass 1** — `Video(nframes=budget.nframes)`, `generate_with_signals`,
   esattamente come `attention_highlight`.
2. **Celle** — `ranked_cells_from_attention(va, query_rows="question",
   sink_filter=False)` (§3bis), prendi le `top_k` per massa. Con
   `cell_select="random"` estraine `top_k` a sorte con seed mescolato su
   `(video_path, prompt)` — copia la logica da `attention_highlight`.
3. **Frame reali** — `reconstruct_frame_indices(video_path, budget.nframes,
   video_start, video_end)`; la cella `ti` → indici `2*ti`, `2*ti+1`.
4. **Estrazione** — tutti i `nframes` frame in PNG su tmpdir, sul modello di
   `_build_phase_shifted_media`. Contratto: la tmpdir la ripulisce il
   chiamante dopo la `generate()`, e va ripulita anche se l'estrazione
   solleva (vedi il `try/except` di quella funzione).
5. **Split** — taglia la lista dei path ai confini `2*ti` e `2*ti+2` di ogni
   cella marcata. **Tutti i blocchi devono avere lunghezza pari** (§2.4).
   Blocchi vuoti (cella marcata in testa o in coda) vanno **omessi**, non
   passati vuoti. Celle marcate adiacenti vanno fuse in un blocco solo, con
   una coppia sola di marcatori.
6. **Content** — alterna `VideoFrames` (con `frames_indices` assoluti e `fps`
   reale del blocco) e `Text` dei marcatori, poi il prompt con l'istruzione.
7. **Pass 2** — `build_messages` dalla lista di parti, `vlm.generate`,
   `parse_mcq_letter`, con lo stesso fallback su `signal.pred_letter` usato
   dagli altri arm.

**Istruzione nel prompt** (obbligatoria). I marcatori sono testo ordinario:
senza una riga che li spieghi, il modello non ha alcun motivo di trattarli
come un'indicazione — è il motivo per cui LoT li accompagna sempre con un
prompt dedicato (`lot/prompts.py`, `SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA`).
Costante di modulo, **non** knob di config: è il testo dell'intervento,
cambiarlo cambia l'esperimento. Modellarla sulle costanti `POINTER_SPAN_*` di
`attention_highlight.py:113-124`, dicendo in una frase imperativa che i frame
racchiusi fra `<START_IMPORTANT_FRAME>` e `<END_IMPORTANT_FRAME>` sono quelli
che contengono l'evidenza rilevante per la domanda.

**Conseguenza sul disegno del controllo**: l'arm è "marcatori + istruzione",
quindi il controllo onesto è **la stessa istruzione con celle scelte a caso**
(`cell_select=random`), non l'assenza di istruzione. Senza il controllo
casuale non si può attribuire un eventuale guadagno al picco d'attenzione
invece che al solo formato del prompt — è la lezione di
`docs/260803-tabelle_risultati.md §1`, dove `highlight_top1_24` e
`random_top1_24` sono risultati indistinguibili.

**Campi da ritornare nel dict** (finiscono in Weave e nei breakdown di
`evals/base.py:121 BREAKDOWN_KEYS`): almeno `raw`, `pred`, `pred_fallback`,
`answer_entropy`, `marked_cells`, `n_marked_cells`, `n_blocks`,
`n_query_rows`, `n_query_tokens`, `cell_select`, `sink_filter`.

### 5.5 Registrazione e config

- `strategies/__init__.py` — importare la classe: `Strategy.get` risolve via
  `__subclasses__()`, senza import la strategy **è invisibile**.
- `conf/config.yaml`, sotto `strategy:` — nuovo preset:

```yaml
  attention_marker:
    name: attention_marker
    always_highlight: true      # nessun gate: si interviene su tutti i sample
    top_k: 1                    # celle marcate
    cell_select: attention      # (attention | random) random = arm di controllo
    query_rows: question        # NON "all": vedi §3
    sink_filter: false          # attenzione GREZZA, niente filtro sink: vedi §3bis
    random_seed: 0
    capture_attention: false
```

- `main.py` non va toccato: il dispatch della strategy passa da
  `Strategy.get(cfg.strategy.name, cfg.strategy)`, non da un dict come per i
  modelli.

## 6. Probe

Sbatch sul modello di `scripts/sbatch/ablation-videomme/entropy_shift_24_qwen3.sbatch`,
da cui vanno **copiati alla lettera** due elementi:

- `--constraint` **senza** `gpu_RTX6000_24G` (su Turing Qwen3-VL-2B costa ~44
  s/sample contro ~3.4 su L40S — vedi `docs/qwen3_vl_2b_baseline.md`);
- il blocco `PROBE_ARGS` con `LIMIT`/`shuffle=true` e il gruppo wandb separato
  `-test`, così una probe non finisce fra i numeri di produzione.

```bash
LIMIT=60 sbatch --array=0 --time=00:40:00 \
    scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch
```

**Costo atteso.** A differenza di `attention_highlight` (che riusa gli stessi
token visivi ed è quindi economico), qui ogni sample richiede estrazione PNG +
secondo pass, e con `always_highlight=true` **non c'è gate che ne esenti
qualcuno**. Sulla misura del 2026-08-28 (sovraccosto 10.5 s per sample
intervenuto su Qwen3-VL-2B, `docs/260901-tabelle-risultati.md §2`) si va verso
~14 s/sample: ~15 min per una probe da 60. Un eventuale fullset a 3 shard
sarebbe ~3.5 h/shard, cioè **fuori dalla walltime a 4 h con poco margine** —
da ritarare sul numero vero che darà la probe, non su questa stima.

**Cosa controllare sulla probe** (in quest'ordine — i primi due sono
correttezza, non risultato):

1. `n_query_rows < n_query_tokens` su **tutti** i sample. Se sono uguali il
   selettore non ha tagliato e l'arm sta girando a `query_rows=all` (§3).
2. Che il filtro sink sia davvero spento: loggare `sink_filter` per sample e
   confrontare la cella scelta con quella che sceglierebbe il filtro acceso —
   se coincidono sempre, il knob non sta arrivando a
   `ranked_cells_from_attention`.
3. Somma di `t` sui blocchi == `t` del blocco unico equivalente. Se è cresciuta,
   un blocco aveva lunghezza dispari (§2.4) e i timestamp sono sfasati.
4. `mcq_accuracy` esiste nel summary. Un OOM di `full_visual_attention` viene
   catturato da Weave per sample e lascia il job `finished` con 0 predizioni
   valide: lo stato del job **non** è un check.
5. `pred_fallback` ~0.
6. `model_latency_mean`, per ritarare la walltime.
7. Solo dopo: l'accuracy, e solo contro `random` a parità di tutto il resto.

## 7. Cosa NON fare

- Non toccare il default `query_rows: all` di `attention_highlight`: tutti gli
  arm già misurati sono girati con quello, cambiarlo renderebbe le loro celle
  non più confrontabili (`utils/attn_core.py:635-640`).
- Non cambiare il default di `ranked_cells_from_attention`: il filtro sink
  resta attivo per tutti gli altri arm (§3bis).
- Non ottenere il "niente filtro" con `sink_percentile: 0` (§3bis).
- Non cambiare la firma di `build_messages` (§5.3).
- Non usare `RankedCell.frames`/`.rep_frame`/`.t_sec` (§4).
- Non implementare la variante su Qwen2.5-VL: non è verificata.
- Non ridurre la walltime assumendo i fattori di costo di `attention_highlight`.

---

## Appendice A — snippet di verifica riproducibile

Gira senza GPU e senza pesi (serve solo il processor in cache). Usarlo per
ri-verificare 2.2/2.3/2.4 dopo qualsiasi cambio a `_prepare_inputs`.

```python
import re, numpy as np
from PIL import Image
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
paths = [...]                      # 8 PNG
FPS, IDX = 30.0, [0, 30, 60, 90, 120, 150, 180, 210]   # indici ASSOLUTI

def run(splits):
    content, metas, a = [], [], 0
    for j, n in enumerate(splits):
        if j == 1: content.append({"type": "text", "text": "<START_IMPORTANT_FRAME>"})
        content.append({"type": "video", "video": paths[a:a+n]})
        if j == 1: content.append({"type": "text", "text": "<END_IMPORTANT_FRAME>"})
        metas.append(VideoMetadata(fps=FPS, frames_indices=IDX[a:a+n],
                                   total_num_frames=240, duration=8.0, video_backend="decord"))
        a += n
    content.append({"type": "text", "text": "What happens?"})
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    vids = [[np.array(Image.open(f)) for f in c["video"]]
            for c in content if c.get("type") == "video"]
    out = proc(text=[text], videos=vids, video_metadata=metas,
               do_sample_frames=False, return_tensors="pt", return_mm_token_type_ids=True)
    dec = proc.tokenizer.decode(out["input_ids"][0])
    print(splits, out["video_grid_thw"].tolist(),
          re.findall(r"<(\d+\.\d) seconds>", dec))

run([8])          # riferimento: t=4, 0.5 2.5 4.5 6.5
run([4, 2, 2])    # atteso IDENTICO al riferimento
run([3, 2, 3])    # atteso ROTTO: t=5, timestamp sfasati
```

## Appendice B — riferimenti

- `strategies/attention_highlight.py:37-58` — perché questa variante era stata
  scartata e perché su Qwen3-VL non lo è più
- `lot/retrieval.py:394-396` — l'inserimento dei marcatori nell'originale LoT
- `transformers/models/qwen3_vl/processing_qwen3_vl.py:157-170` — espansione
  per-frame con timestamp
- `transformers/models/qwen3_vl/modeling_qwen3_vl.py:1068-1070` — split di
  `video_grid_thw`, il motivo per cui i marcatori non rompono M-RoPE
- `docs/analisi_entropia_attenzione_sink.md §3, §4.2` — quanta massa
  assorbono i token sink e perché il filtro è acceso ovunque
- `docs/260901-tabelle-risultati.md` — costi misurati su Qwen3-VL-2B
- `docs/260803-tabelle_risultati.md §1, §4` — arm `highlight`/`random` e
  `query_rows=question` su Qwen2.5-VL
