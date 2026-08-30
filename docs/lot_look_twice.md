# LoT (*Look Twice*): metodo di riferimento, per la tesi

> Sintesi dei punti salienti del framework **Look Twice (LoT)** così com'è
> implementato nel codice sorgente originale conservato in `lot/`
> (`lot/retrieval.py`, `lot/retriever.py`, `lot/prompts.py`, `lot/data.py`,
> `lot/new_forward.py`, `lot/prova.py`). **Importante**: `lot/` non è codice
> di questo progetto ma una copia/estratto della codebase originale di LoT,
> tenuta qui **come riferimento di studio** — molti import
> (`retrieval_module.*`, `cropper_module.*`, `critique.*`, `OMGM.*`,
> `reranking_module.*`) puntano a pacchetti esterni non presenti in questo
> repo, quindi `lot/retrieval.py` **non è eseguibile as-is** qui. Questo
> documento descrive il *metodo* (per citarlo/spiegarlo in tesi), non
> un'istruzione per farlo girare.
>
> Per come questo progetto **riprende e adatta** LoT al dominio video, vedi
> `docs/qwen25vl_entity_attention.md` (porting dell'interfaccia di
> attenzione), `docs/attn_explorer_analisi_runtime.md` (strumento di
> esplorazione runtime) e `docs/analisi_entropia_attenzione_sink.md`
> (evidenza quantitativa dei sink); §7 di questo documento mappa
> esplicitamente cosa è stato portato e cosa no.


> **Stato sperimentale nel dominio video (2026-08-30).** Le due metà del
> metodo non reggono allo stesso modo qui: misurato con un oracolo su
> LVBench (`docs/oracolo_lvbench.md`), il "secondo sguardo" che **comunica**
> la regione saliente al modello non paga nemmeno quando la regione è quella
> vera (30% vs 32% di baseline), mentre **ricampionare** i frame dentro la
> finestra vera vale +16 pp (48%). Nel video il collo di bottiglia è che
> l'evidenza spesso non è tra i frame campionati affatto — non che il
> modello non sappia dove guardare dentro ciò che vede.

## 1. Contesto e problema

LoT si colloca nel **Knowledge-Based Visual Question Answering (KB-VQA)**
"encyclopedico": domande su un'immagine che richiedono conoscenza esterna
non deducibile dalla sola immagine (es. dataset **Encyclopedic-VQA (EVQA)**,
**InfoSeek**, **ViQuAE** — visti in `args.dataset_name` di
`lot/retrieval.py:169`). Il paradigma standard è **RAG multimodale**:
1. si recupera (retrieval) un piccolo insieme di passaggi testuali
   rilevanti da una knowledge base (Wikipedia), usando l'immagine (o un suo
   crop) come query;
2. si passano quei passaggi + l'immagine + la domanda a una VLM che genera
   la risposta finale.

Il problema che LoT attacca: **il retrieval e la generazione portano
rumore**. Non tutti i passaggi recuperati sono pertinenti (il retrieval
per immagine è approssimato), e non tutta l'immagine è rilevante per la
domanda (il soggetto della domanda occupa spesso solo una piccola regione).
Una VLM frozen, senza guida, "annega" l'evidenza rilevante in contesto
irrilevante — sia visivo sia testuale.

## 2. L'idea centrale: due passate, un solo modello frozen

Il nome *Look Twice* descrive la soluzione: **il modello guarda due
volte**, usando la propria attenzione interna (nessun modulo di
localizzazione esterno addestrato) per decidere cosa guardare la seconda
volta con più attenzione:

1. **Primo sguardo (diagnostico, non genera testo)**: un forward
   *aggiuntivo* — o hook sullo stesso forward della generazione finale —
   estrae dall'attenzione del modello **dove**, nell'immagine, e **quali**,
   nel contesto recuperato, sono i segnali più rilevanti rispetto alla
   domanda. Due varianti simmetriche, stesso principio:
   - **lato immagine**: `extract_attention_bbox` (§3) — attenzione dei
     token dell'entity (estratta dalla domanda) verso i token visivi →
     bounding box.
   - **lato testo recuperato**: `generate_self_elicit` (§4) — attenzione
     dell'ultimo token (quello che genererebbe la risposta) verso i token
     del contesto → frasi/passaggi "elicitati".
2. **Secondo sguardo (genera la risposta)**: `InferenceModel.generate`
   (`lot/retrieval.py:908-964`) viene richiamato con l'input **arricchito**
   da ciò che il primo sguardo ha trovato: l'immagine croppata/annotata
   sulla bbox, e/o il contesto con le frasi rilevanti marcate da tag
   speciali (`<START_IMPORTANT_TXT>...<END_IMPORTANT_TXT>`), con il
   system prompt istruito esplicitamente a **dare priorità** a ciò che è
   marcato (`SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG`/`_TEXT`,
   `lot/prompts.py:141-143`).

Il punto chiave, e la ragione per cui è rilevante per una tesi che studia
l'attenzione: **la localizzazione non richiede training** — è un
sottoprodotto gratuito dell'attenzione che il modello già calcola durante
il forward, filtrato per rimuovere il rumore dei token-sink (§5). L'unico
costo aggiuntivo è computazionale (uno o due forward extra), non un nuovo
modello da addestrare (a differenza del baseline alternativo con
GroundingDINO, `--gdino_bbox`, tenuto come confronto/ablation).

## 3. Primo sguardo, lato immagine: `extract_attention_bbox`

`InferenceModelQwen2_5_VL.extract_attention_bbox`
(`lot/retrieval.py:1097-1446`). Dato `(image, question, entity)`:

1. **Individua lo span dell'entity** nei token del prompt
   (`find_text_token_spans`) — l'entity è estratta dalla domanda a monte,
   con euristiche diverse per dataset (spaCy NER per ViQuAE, una funzione
   `extract_question_target` per altri, la classe target nota per COCO,
   `None` = "usa l'ultimo token"/"tutti i token" per dataset senza
   un'entity singola come OCRBench o ChartQA — `main()`,
   `lot/retrieval.py:4013-4029`).
2. **Monkey-patch temporaneo** del `forward` di `self_attn` sui layer nel
   range scelto (default `middle_half`: dal 25° al 75° layer del decoder)
   per catturare i pesi d'attenzione **solo** delle righe-query dell'entity
   (stesso principio del custom attention hook di
   `docs/qwen25vl_entity_attention.md`, con cui condivide l'origine).
3. Due **forward hook** per layer, nel range scelto:
   - uno sull'attenzione: media `entity→token_visivi` su teste e sulle
     righe dell'entity → un vettore `[n_vis]` per layer;
   - uno sullo stato nascosto: stesso calcolo di sink-score visto nel
     progetto video — `max(|hidden[sink_dims]|) / RMS(hidden)` — per
     ottenere una `sink_map` parallela.
4. Media sui layer del range → `attn_map` e `sink_map`, reshape a
   `[grid_h, grid_w]` (dedotta da `image_grid_thw` e dal merge spaziale).
5. **Filtro sink**: soglia al percentile fisso `SINK_PERCENTILE=25` sulla
   `sink_map` — `actual_threshold = np.percentile(sink_map, 25)`, poi
   `is_sink = sink_map >= actual_threshold`. **Verificato numericamente**
   (non solo per lettura): `np.percentile(x, 25)` è il valore sotto il
   quale cade il 25% della distribuzione, quindi `>= actual_threshold`
   marca come sink il **75% superiore** dei token, non il 25% più alto.
   È la **convenzione opposta** a quella di questo progetto
   (`sink_mask`, `utils/attn_core.py:474-478`, che usa esplicitamente il
   quantile `1 − p/100` proprio per marcare come sink il 25% *più alto*,
   una minoranza). In LoT, in altre parole, restano "non-sink" (e quindi
   utilizzabili per il bbox) solo i token nel quartile *più basso* di
   sink-score — un filtro più aggressivo/conservativo di quanto il nome
   del parametro suggerisca a prima vista. **Più**, sempre, un **bordo
   forzato a sink di 2 anelli di riga e colonna**
   (`lot/retrieval.py:1349-1357`) — a differenza del progetto video, dove
   questa euristica di bordo è opt-in e **non confermata** empiricamente
   (`docs/analisi_entropia_attenzione_sink.md §4.3`): in LoT è invece
   parte fissa, non ablata, della pipeline di produzione.
6. Azzeramento delle celle sink, rinormalizzazione, upscale bicubico alla
   risoluzione dell'immagine originale.
7. **Estrazione bbox** dalla mappa filtrata con due metodi indipendenti
   (`lot/retrieval.py:85-133`):
   - `extract_bbox_weighted_centroid`: centroide pesato dall'attenzione +
     dimensione del box da una deviazione standard pesata (`std_multiplier`,
     default 2.0) attorno al centroide;
   - `extract_bbox_morphological`: soglia fissa sulla mappa normalizzata +
     operazioni morfologiche (erosione/dilatazione) per isolare una
     regione connessa;
   - `compute_average_bbox`: media geometrica dei due (nel codice attuale
     i metodi morfologici multipli sono commentati/disattivati — solo
     `weighted_centroid` è attivo di default, `attention_bbox_method`).
8. Fallback a bbox = intera immagine se una fase fallisce (nessuna
   attenzione estratta, entity non trovata, reshape impossibile) — LoT
   preferisce degradare a "nessun crop" piuttosto che fallire il sample.

Il bbox risultante alimenta il secondo sguardo in tre modi alternativi
(flag CLI): **crop** dell'immagine (`image_cropped`, passato come
immagine aggiuntiva con `PROMPT_SELF_ELICT_IMAGE_CROP`), **annotazione**
visiva (box disegnato sull'immagine originale, `highlight_entity`), o
entrambi.

## 4. Primo sguardo, lato testo recuperato: `generate_self_elicit`

`InferenceModel.generate_self_elicit` (`lot/retrieval.py:677-820`), attivo
quando la pipeline usa retrieval **e** una delle flag `self_elicit*`. Dato
`(question, image_query, context)` (il contesto = passaggi recuperati
concatenati):

1. Prepara l'input con `text_eliciting=True` (varia il prompt system,
   §6) e localizza lo **span di token del contesto** dentro la sequenza
   (`get_context_token_span`).
2. Segmenta il contesto in **frasi** (spaCy sentencizer,
   `get_sentence_token_spans`) o in **passaggi** (split su
   `PASSAGE_DELIMITER`, `get_passage_token_spans`), a seconda delle flag
   `self_elicit_gen_passage`/`self_elicit_gen_sen2pas`.
3. Hook di forward sui layer nel range scelto (default: metà superiore,
   `int(0.5*n_layers)` a `n_layers`, diverso dal `middle_half` usato per
   il bbox visivo — parametrizzabile separatamente via
   `--attention_text_layer_range`):
   - attenzione: **solo l'ultima posizione** (il token che genererebbe la
     risposta) verso lo span di contesto, media sulle teste → un vettore
     `[len(context)]` per layer;
   - hidden state: stesso identico calcolo di sink-score di §3, ma qui
     **non risulta usato a valle** in `_process_attention_scores` (nessun
     filtro sink applicato all'elicitazione testuale nel codice attuale —
     un'asimmetria degna di nota rispetto al lato immagine).
4. Un **unico forward completo** del modello (`model_self_elicit(...,
   use_cache=False)`) con tutti gli hook registrati, poi rimozione hook.
5. `_process_attention_scores` (`lot/retrieval.py:822-905`): normalizza
   l'attenzione per layer (così ogni layer "vota" con lo stesso peso
   indipendentemente dalla sua scala assoluta), media sui layer, aggrega
   per frase/passaggio (media dei punteggi token nel suo span), poi
   **seleziona come evidenza** ogni frase/passaggio con punteggio
   `>= alpha * punteggio_massimo` (soglia relativa al top, non assoluta —
   `self_elicit_alpha`, default 0.5). Le frasi selezionate vengono
   racchiuse fra `<START_IMPORTANT_TXT>`/`<END_IMPORTANT_TXT>` nel testo
   ricostruito (`elicited_context`); le altre restano invariate.
6. `elicited_context` sostituisce il contesto grezzo nel prompt del
   secondo sguardo (`CONTEXT_VQA_PROMPT_training`, §6).

Concettualmente: il modello genera implicitamente un "estratto"
auto-supervisionato del proprio contesto — nessuna etichetta di rilevanza
richiesta, il segnale è l'attenzione che il modello stesso porrebbe se
dovesse rispondere subito.

## 5. Sink dims: lo stesso meccanismo, condiviso fra bbox visivo e testo

`SINK_DIMS` (`lot/retrieval.py:64-79`) è una tabella **hardcoded,
per-modello**, di canali dello hidden state ad attivazione anomala
("massive activations" / attention sink, Xiao et al.), calcolata offline
una volta per modello e riusata identica per ogni sample:

| model_id | sink dims |
|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | `[318, 1874, 1819]` |
| Qwen/Qwen2.5-VL-7B-Instruct | `[458, 2570]` |
| Qwen/Qwen2.5-VL-32B-Instruct | `[4675, 3094]` |
| Qwen/Qwen2-VL-2B-Instruct | `[1073, 534, 940]` |
| Qwen/Qwen2-VL-7B-Instruct | `[2570, 458]` |
| Qwen/Qwen3-VL-2B-Instruct | `[1793, 1999, 1401]` |
| Qwen/Qwen3-VL-4B-Instruct | `[0]` |
| Qwen/Qwen3-VL-8B-Instruct | `[1838]` |
| OpenGVLab/InternVL3_5-1B/4B/8B/38B-hf | `[35,13]` / `[4,396,0]` / `[2276,233]` / `[731]` |
| llava-hf/llava-1.5-7b-hf | `[2533, 1415]` |
| aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning | `[788, 1384, 4062]` |

`SINK_PERCENTILE = 25` è il valore fisso usato per classificare un token
come sink (percentile sulla distribuzione di score, non regolabile da CLI
se non tramite `--sink_tau` che lo sovrascrive globalmente). La formula
dello score è identica in entrambi gli usi (bbox visivo §3, ed è calcolata
ma non applicata nell'elicitazione testuale §4):

```python
sink_score = max(|hidden_state[:, sink_dims]|, dim=canali) / (RMS(hidden_state) + 1e-6)
```

Questo è **esattamente** il meccanismo ripreso in questo progetto
(`models/qwen_attn.py`, `utils/attn_core.py::sink_mask` — vedi
`docs/analisi_entropia_attenzione_sink.md §4.1`), con due differenze
esplicite da tenere a mente: (a) qui il bordo della griglia è **sempre**
forzato a sink (non opt-in), (b) il verso del percentile (`np.percentile`
diretto vs. il quantile `1 − p/100` usato nel progetto video) — un punto
da verificare con attenzione se si cita/confronta direttamente la cifra
"25%" fra i due codici, perché la stessa cifra nominale può selezionare
insiemi diversi a seconda della convenzione.

## 6. Prompting: come il "secondo sguardo" riceve l'evidenza

Da `lot/prompts.py`:

- **`SYSTEM_PROMPT`**: istruisce il modello a rispondere usando il
  contesto se pertinente, altrimenti la propria conoscenza parametrica —
  esplicitamente per il caso "il retrieval non ha trovato nulla di
  utile".
- **`CONTEXT_VQA_PROMPT_training`**: template minimale
  `{question}` + "The following paragraphs may contain useful
  information..." + `{context}` — è quello effettivamente usato in
  `main()` col contesto (elicitato o grezzo).
- **`SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT`** /
  **`SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG`**: righe aggiunte al system prompt
  *solo* quando si usano rispettivamente marker testuali o visivi,
  che spiegano esplicitamente il significato dei tag
  (`<START_IMPORTANT_TEXT>`/`<START_IMPORTANT_IMG>` ecc.) e istruiscono il
  modello a **non riprodurli** in output — il modello viene informato del
  proprio stesso meccanismo di evidenziazione.
- **`PROMPT_SELF_ELICT_IMAGE_CROP`**: quando si passa anche il crop come
  immagine aggiuntiva, spiega che è "un crop della prima immagine alle
  coordinate bbox_2d, contenente l'entity rilevata".

Il "secondo sguardo" quindi non è solo "più contesto" — è lo stesso
contesto/immagine di prima **con segnali di attenzione espliciti
sovrapposti**, e il modello è istruito a trattarli come tali.

## 7. Componenti collaterali (menzionati per completezza, non nucleo del metodo)

- **Retrieval** (`lot/retriever.py`): retrieval per immagine (CLIP/EVA-CLIP
  + indice FAISS) su una KB Wikipedia precostruita, con varianti
  (Google Lens come baseline oracle-like, crop della query per il
  retrieval `--crop_query_img`, OMGM come pipeline di retrieval
  alternativa a due stadi). Non è parte del "look twice" in sé (quella è
  la fase *prima* del primo sguardo), ma ne è il prerequisito quando si
  lavora in modalità "with_retrieval".
- **`CritiqueModel`** (`critique.critique_model`, modulo esterno non
  incluso in questo repo): un modello **allenato separatamente**
  (checkpoint SFT, `critique_model_name`) che stima una probabilità
  "sì/no" di rilevanza per ogni passaggio recuperato
  (`critic.passages_relevance`, soglia `yes_prob_thr`). A differenza del
  self-elicit (attenzione del modello frozen, zero training), questo è un
  filtro di rilevanza **appreso** — un'alternativa/aggiunta al self-elicit,
  usata con `--eval_passages`.
- **`Cropper`** (`cropper_module.cropper`, modulo esterno): baseline di
  localizzazione **non basata sull'attenzione** — GroundingDINO
  (`--gdino_bbox`) o rilevamento+crop generico
  (`detect_and_crop`/`detect_and_highlight`) — usata come confronto per
  isolare il contributo specifico della bbox "gratuita" via attenzione.

## 8. Cosa è stato portato nel progetto video (questo repo) e cosa no

| Elemento di LoT | Stato in questo progetto |
|---|---|
| Sink dims per-modello + score RMS-normalizzato | **Ripreso identico** (`models/qwen_attn.py::SINK_DIMS`, stessa formula) |
| Layer "centrali" per l'aggregazione | **Ripreso** (stesso concetto `middle_half`, vedi `docs/qwen25vl_entity_attention.md`) |
| Filtro sink a percentile | **Ripreso**, ma **bordo forzato reso opt-in** invece che sempre attivo, dopo verifica empirica che non regge sul campione video (`docs/analisi_entropia_attenzione_sink.md §4.3`) |
| Bbox visivo da attenzione entity→immagine (`extract_attention_bbox`) | **Generalizzato**: non un bbox 2D per singola entity estratta a monte, ma una heatmap per **ogni** token della domanda, entity-agnostica, selezionabile a runtime (`attn_explorer`, vedi `docs/attn_explorer_analisi_runtime.md`) — coerente con l'assenza (in molti dataset video MCQ) di una singola entity estraibile in modo affidabile |
| Estensione temporale (celle = frame) | **Novità di questo progetto**: LoT lavora su immagini singole (`t=1` implicito); qui la griglia ha una dimensione temporale esplicita (`GridSpec.t`), con ranking dei frame per massa di attenzione (`rank_cells`) al posto di un bbox spaziale |
| Self-elicit testuale sul contesto recuperato (`generate_self_elicit`) | **Non ancora portato** — nessuna fase di retrieval è presente in questo progetto ad oggi; rilevante solo quando si introduce un componente KB (target dichiarato: Video SimpleQA, vedi `docs/video_grounding_benchmarks.md`) |
| Secondo sguardo / crop reale dell'immagine | **Non applicabile 1:1**: non si croppa il video, si **rialloca/riordina l'attenzione** su quali frame guardare — concettualmente più vicino a un competitor come **AdaptToken** (allocazione del budget di token visivi per gruppo, citato in `docs/diario/riunione_07-07-26.md`) che al crop spaziale di LoT |
| CritiqueModel (rilevanza passaggi appresa) | **Non portato** (nessun retrieval, nessuna KB ancora) |
| Segnale di confidenza/entropia sulla risposta MCQ | **Non presente in LoT** (LoT è generazione libera, non MCQ) — **aggiunta originale** di questo progetto (`docs/entropia_confidenza_mcq.md`), studiata *insieme* al segnale di concentrazione dell'attenzione sink-filtrata ereditato da LoT |

## 9. Punti da tenere a mente / possibili insidie per un confronto rigoroso

- **Verso del percentile (verificato, non solo letto)**: in LoT, con
  `SINK_PERCENTILE=25`, il **75%** superiore dei token per sink-score è
  marcato "sink" (§3, §5) — l'opposto della convenzione "25% = minoranza
  sink" usata in questo progetto. Se si cita la cifra "25%" da LoT in
  tesi, va precisato esplicitamente cosa seleziona in quel codice, per
  non generare confronti fuorvianti con le cifre di
  `docs/analisi_entropia_attenzione_sink.md` (dove il 25% è invece la
  minoranza classificata sink).
- **Bordo sempre sink in LoT, opt-in qui**: qualunque numero riportato da
  LoT su bbox/localizzazione include implicitamente l'assunzione "i bordi
  dell'immagine sono sink"; il dato empirico raccolto in questo progetto
  su video (§4.3 di `docs/analisi_entropia_attenzione_sink.md`) non la
  conferma — non generalizzare automaticamente le cifre di LoT al dominio
  video senza ri-verificare quell'assunzione.
- **Costo**: `generate_self_elicit` è un forward *completo* aggiuntivo
  (non solo qualche layer) — il "primo sguardo" testuale in LoT non è a
  costo zero come lo è invece il "primo sguardo" visivo (quello riusa
  l'unico forward di prefill già necessario). Utile da citare se si discute
  l'overhead computazionale del framework.
- **Selezione dell'entity a monte** è dataset-specifico e per lo più
  euristico (spaCy NER, regole ad-hoc per dataset, classe nota per COCO):
  non è parte del "meccanismo di attenzione" in sé, ma una dipendenza a
  monte del bbox visivo che va dichiarata se si discute la generalità del
  metodo.

## 10. Riferimenti file:riga (codice originale, `lot/`)

- `lot/retrieval.py:64-79` — tabella `SINK_DIMS` multi-modello, `SINK_PERCENTILE`.
- `lot/retrieval.py:85-133` — estrazione bbox (`weighted_centroid`, `morphological`, media).
- `lot/retrieval.py:677-820` — `generate_self_elicit` (primo sguardo, testo).
- `lot/retrieval.py:822-905` — `_process_attention_scores` (selezione frasi/passaggi).
- `lot/retrieval.py:908-964` — `generate` (secondo sguardo, risposta finale).
- `lot/retrieval.py:1097-1446` — `extract_attention_bbox` (primo sguardo, immagine).
- `lot/retrieval.py:3895-...` — `main()`, orchestrazione dell'intera pipeline per sample.
- `lot/prompts.py` — tutti i template di prompt citati in §6.
- `lot/retriever.py` — retrieval CLIP/EVA-CLIP + FAISS su KB Wikipedia.
