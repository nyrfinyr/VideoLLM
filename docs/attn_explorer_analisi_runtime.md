# `attn_explorer`: analisi a runtime della massa di attenzione per frame, per token testuale, con filtro sink

> Documento di riferimento (per la tesi) delle funzionalità implementate nel
> server interattivo `attn_explorer/serve.py` + template
> `attn_explorer/templates/` per esplorare, **a runtime e senza rifare il
> forward del modello**, su quali frame di un video si concentra
> l'attenzione testo→visivo del modello, in funzione di (a) quali token
> della domanda si selezionano come "entity" e (b) se/come si filtrano i
> token visivi classificati come **sink**. Copre la pipeline dati a monte
> (`attn_explorer/capture.py`, `utils/attn_core.py`) solo nella misura in
> cui serve a capire cosa la pagina mostra.
>
> Per l'analisi statistica (entropia della risposta, correlazione con la
> correttezza, evidenza quantitativa dei sink su un campione) vedi invece
> `docs/analisi_entropia_attenzione_sink.md` e `docs/entropia_confidenza_mcq.md`
> — questo documento è sull'**interfaccia/strumento**, quello sui **risultati**.

## 1. Perché esiste: entity-agnostic capture + selezione a runtime

Il design è deliberatamente **disaccoppiato in due fasi**, per non dover
rifare un forward del modello (costoso, richiede GPU) ogni volta che si
vuole guardare l'attenzione verso un sottoinsieme diverso di token della
domanda:

1. **Cattura** (`attn_explorer/capture.py`, offline, una volta per video):
   un forward di prefill produce **una heatmap `[t, grid_h, grid_w]` per
   OGNI token della domanda**, non solo per un'entity scelta a priori.
   Il dump (`AttentionCapture`, vedi §2) è quindi *entity-agnostico*.
2. **Esplorazione** (`attn_explorer/serve.py`, il server Flask oggetto di
   questo documento): a runtime si sceglie un sottoinsieme di righe-query
   (token) da mediare, si applica opzionalmente un filtro sink, e si
   ottiene immediatamente (senza nuovo forward) il ranking dei frame e gli
   overlay corrispondenti.

Questo è ciò che rende "runtime" l'analisi: cambiare la selezione di token
o la view sink è una richiesta HTTP che ricalcola un tensore già in memoria
(o mmappato da disco), non una nuova inferenza.

## 2. Dati sottostanti (cosa la pagina interroga)

Tutte le strutture dati sono definite in `utils/attn_core.py` e condivise
fra `capture.py` e `serve.py`.

### 2.1 `AttentionCapture` (il dump per-video, `<store>/<stem>/capture.pt`)

Campi rilevanti per la pagina interattiva:

- `attn`: tensore `[n_q, t, grid_h, grid_w]` float32 — per ogni token della
  domanda (`n_q`), la mappa di attenzione sui token visivi, già mediata su
  layer centrali e teste dal forward di cattura.
- `sink_map`: tensore `[t, grid_h, grid_w]` — punteggio "sink" per ogni
  token visivo (indipendente dalla selezione di entity, vedi §4).
- `sink_dims`: tupla di indici di canale usati per calcolare `sink_map`
  (proprietà del modello, non del video — vedi §4.1).
- `query_tokens`: tupla di `QueryToken(row, index, token, text)` — `row` è
  l'indice 0-based in `attn` (quello usato per selezionare le entity a
  runtime), `text` la resa leggibile mostrata come "chip" nella UI.
- `grid`: `GridSpec(t, grid_h, grid_w, double)` — geometria + mappatura
  cella→frame reale (§2.2).
- `frame_indices`, `fps`: per convertire l'indice di cella in secondi.
- `answer_entropy`, `answer_probs`, `answer_logits`, `pred_letter`,
  `top_tokens`, `capture_meta`: dati di confidenza MCQ mostrati sulla stessa
  pagina ma non oggetto di questo documento (vedi
  `docs/entropia_confidenza_mcq.md`).

`AttentionCapture.aggregate(rows)` (`utils/attn_core.py:304-312`) è
l'operazione centrale della selezione runtime: media `attn` sulle righe
`rows` (i token scelti); con `rows` vuoto fa la media su **tutti** i token
della domanda (ordinamento di default). Il costo è `O(n_q · t · grid_h ·
grid_w)` nel caso peggiore ma in pratica un semplice `mean` su un tensore
già in RAM — sub-millisecondo.

### 2.2 `GridSpec` e i due regimi cella↔frame

Qwen2.5-VL fonde i frame **a coppie** nel patch embed temporale: una cella
temporale della griglia di default copre **due** frame reali consecutivi.
Il tool di cattura offre un regime alternativo, il **frame-doubling**
(`--double-frames` in `capture.py`): ogni frame campionato viene duplicato
prima del forward (`[f0, f0, f1, f1, ...]`), così ogni cella temporale
coincide con un **singolo** frame reale — evita l'ambiguità "a quale dei
due frame appartiene l'attenzione" al costo di ~2× token visivi nel
prefill. `GridSpec.cell_to_frames(ti, frame_indices)`
(`utils/attn_core.py:44-50`) incapsula la differenza: ritorna sia i frame
reali coperti dalla cella sia un frame "rappresentante" (nel regime a
coppie, la media arrotondata dei due indici) usato per l'estrazione PNG e
gli overlay. La UI mostra sempre `grid.regime` in chiaro
(`video.html:12`, stringa `"frame-doubling (cella=1 frame)"` o `"merge a
coppie (cella=2 frame)"`) così chi guarda la pagina sa se un "frame" nella
grid corrisponde esattamente a un frame video o a una coppia.

## 3. La pagina video (`GET /v/<stem>`) — vista d'insieme

Route: `serve.py:314-393` (funzione `video_page`). Renderizza
`templates/video.html`. Struttura:

- **Pannello sinistro**: domanda, risposta corretta/del modello con tabella
  logit/prob/entropia (feature di confidenza MCQ, non trattata qui).
- **Pannello destro** (oggetto di questo documento):
  - i **chip dei token** della domanda, cliccabili per comporre la
    selezione di entity (§3.1);
  - un indicatore testuale della selezione corrente;
  - il **toggle overlay/frame originale**;
  - se il capture ha una `sink_map` non nulla (`has_sink_map`, calcolato
    server-side come `cap.sink_map.abs().sum().item() > 0`,
    `serve.py:392`), il **selettore della sink view** (§4.3);
  - metadati di riproducibilità della cattura (`capture_meta`, collassabile);
  - bottoni per il **dato raw del `.pt`** e per il **video originale**.
- **Griglia dei frame** (`#grid`): una card per cella temporale, popolata
  e ri-popolata via JavaScript ad ogni cambio di selezione.
- **Lightbox** per l'immagine a schermo intero, **modale dati raw**
  (JSON del capture, con bottone "copia"), **modale video** (player HTML5
  che va in pausa alla chiusura).

Tutta la logica di interazione vive in `templates/app.js` (nessun
framework: DOM + `fetch` puro).

### 3.1 Selezione delle entity (chip dei token)

Ogni token della domanda è renderizzato come `<span class="chip"
data-row="{row}">` (`video.html:104`). Cliccare un chip ne toggla la classe
`sel` (`app.js:106-109`) e richiama `refresh()`. `selectedRows()`
(`app.js:9-11`) legge i `data-row` dei chip correntemente `.sel` — questi
sono esattamente gli indici passati a `AttentionCapture.aggregate(rows)`
lato server. Senza nessun chip selezionato, la selezione è vuota e
`aggregate` media su **tutti** i token della domanda.

**Selezione di default alla prima renderizzazione**: i chip corrispondenti
a `question_rows(cap.query_tokens)` (`utils/attn_core.py:68-94`) partono
già selezionati (`video.html:104`, classe `sel` condizionale su `tk.row in
default_rows`). `question_rows` isola i token del **solo testo della
domanda**, escludendo lo scaffolding MCQ (`"Options:\n(A) ..."`, `"Answer
with the letter..."`): senza questo, la vista di default mescolerebbe
l'attenzione verso la domanda vera e propria con quella verso le opzioni di
risposta e il boilerplate, diluendo il segnale. L'utente può comunque
deselezionare/selezionare liberamente qualunque token, incluse le opzioni.

### 3.2 Ciclo di refresh e coerenza sotto richieste concorrenti

`refresh()` (`app.js:92-104`) ad ogni cambio di selezione o di sink view:

1. aggiorna il testo "selezione: ..." con i token scelti;
2. incrementa un contatore `reqToken` e chiama
   `GET /api/<stem>/rank?tokens=...&sink_view=...`;
3. **scarta la risposta se nel frattempo è partita una richiesta più
   recente** (`if (my !== reqToken) return`, `app.js:101`) — guardia contro
   race condition quando l'utente clicca più chip in rapida successione
   (senza, una risposta lenta arrivata in ritardo potrebbe sovrascrivere la
   grid con un ranking non più corrispondente alla selezione corrente);
4. ricostruisce la griglia (`grid.replaceChildren(...)`) con una card per
   cella, nell'ordine di ranking restituito dall'API (non l'ordine
   temporale del video: la card `#1` è il frame con più massa attenzionale
   per la selezione corrente, non necessariamente il primo del video).

## 4. Filtro sink

### 4.1 Cos'è un token "sink" in questo progetto (riepilogo — dettagli in `docs/analisi_entropia_attenzione_sink.md §4.1`)

Un token visivo è "sink" in base a una proprietà del suo stato nascosto,
non alla sua posizione: si guarda il picco di attivazione su un piccolo
insieme di canali fissi per modello (`SINK_DIMS`, hardcoded in
`models/qwen_attn.py`, es. `(318, 1874, 1819)` per Qwen2.5-VL-3B),
normalizzato per l'RMS del vettore hidden completo del token, mediato sui
layer centrali del decoder → `sink_map[t, grid_h, grid_w]`. Questo score è
già nel `.pt` (calcolato una volta in cattura); la pagina non lo ricalcola,
applica solo soglia e selezione a runtime.

### 4.2 `sink_mask`: dalla `sink_map` a una maschera booleana

`utils/attn_core.py:436-484`. Un token è classificato sink se il suo
`sink_score` è nel **top-`percentile`%** della distribuzione di TUTTA la
`sink_map` del video (default `percentile=25`, cioè il 25% dei token con
punteggio più alto):

```python
thr = torch.quantile(m.flatten(), 1.0 - percentile / 100.0)
is_sink = m >= thr
```

Nota sul segno: "top-25%" richiede la soglia al quantile `1 − 0.25 =
0.75`, non al 25° percentile grezzo (che selezionerebbe il 75% *inferiore*
dei token, l'opposto di quanto si intende per "sink" come minoranza) — un
dettaglio facile da invertire per errore, commentato esplicitamente nel
codice.

Un secondo parametro, `border` (default **0**, cioè OFF), forzerebbe a
sink anche gli anelli di riga/colonna di bordo della griglia a prescindere
dal loro score reale. È **esplicitamente non verificata**: "sink" in
letteratura è una proprietà di canali ad alta norma nello stato nascosto,
non una proprietà geometrica di bordo dell'immagine — un frame potrebbe
avere evidenza reale vicino al bordo, che questa opzione azzererebbe a
priori. Va usata solo come test empirico opt-in (vedi anche
`docs/analisi_entropia_attenzione_sink.md §4.3`, dove l'euristica di bordo
è stata verificata e **non confermata** su un campione di 30 video).

### 4.3 `sink_view`: cosa mostrare rispetto alla maschera

`utils/attn_core.py:487-522`. Applicata a una heatmap **già aggregata**
per la selezione di entity corrente (`AttentionCapture.aggregate(rows)`),
prende in ingresso la view scelta:

| view | comportamento |
|---|---|
| `all` (default) | heatmap invariata, **nessun azzeramento** — usata per poter comunque disegnare/contare la maschera sink senza alterare l'attenzione grezza mostrata |
| `sink` | azzera le celle **non** sink, rinormalizza per il massimo residuo |
| `nonsink` | azzera le celle sink, rinormalizza per il massimo residuo |

La rinormalizzazione (`out = out / out.max()` se `max > 0`) serve solo a
mantenere leggibile la scala colore dell'overlay dopo aver azzerato una
parte della heatmap — non altera il *ranking relativo* delle celle
superstiti.

Nella UI: un `<select id="sinkview-select">` con le tre opzioni (renderizzato
solo se `has_sink_map`, `video.html:113-122`); il suo valore è lo stesso
per **tutte** le celle della griglia e viene passato come query param
`sink_view` sia a `/api/<stem>/rank` (per il ranking) sia a
`/api/<stem>/overlay` (per ogni singolo overlay, `app.js:23-25,29`).
Cambiare la view **rifà il ranking**, non solo il colore: in view `sink` o
`nonsink` alcune celle possono risalire o scendere in classifica perché la
loro massa cambia (celle dominate da pochi token sink crollano in view
`nonsink`, e viceversa).

`percentile` e `border` sono regolabili solo via query string
(`?sink_percentile=`, `?sink_border=`), non hanno un controllo dedicato
nella UI attuale — pensati per ispezione manuale/da terminale più che per
l'utente della pagina.

### 4.4 Endpoint dedicato di ispezione: `GET /api/<stem>/sink`

`serve.py:450-483`. Ritorna, indipendentemente da qualunque selezione di
token-query, la `sink_map` grezza **e** la maschera booleana calcolata a
runtime per i `percentile`/`border` richiesti:

```json
{
  "stem": "...",
  "has_sink_map": true,
  "sink_dims": [318, 1874, 1819],
  "percentile": 25.0,
  "border": 0,
  "grid": {"t": 48, "grid_h": 12, "grid_w": 16},
  "sink_map": [[[...]]],
  "is_sink": [[[...]]],
  "n_sink": 288,
  "n_total": 9216
}
```

Utile per debug o per analisi offline (es. lo script che ha prodotto
`docs/analisi_entropia_attenzione_sink.md` usa direttamente
`sink_mask`/`sink_view` da `utils/attn_core.py`, non questo endpoint, ma la
semantica è la stessa).

## 5. Ranking dei frame per massa di attenzione

### 5.1 `rank_cells`: dalla heatmap aggregata (e filtrata) a una classifica

`utils/attn_core.py:343-380`. Prende una heatmap `[t, grid_h, grid_w]`
(già passata per `aggregate` + eventuale `sink_view`) e una metrica:

- `metric="sum"` (default): massa **totale** della cella — quanta
  attenzione, sommata su tutti i token visivi della cella, cade lì.
- `metric="max"`: **picco** della cella — il singolo token visivo più
  attenzionato, indipendentemente dalla massa totale.

Per ogni cella calcola `pct` (percentuale sul totale della heatmap vista) e
marca come `is_top` le prime `topk` (default 8) nell'ordine di classifica
— usato lato frontend solo per un bordo evidenziato (classe CSS `.card.top`,
`report.css:40`), non per filtrare quali celle mostrare: **tutte** le celle
vengono sempre renderizzate, ordinate per rank.

### 5.2 Endpoint `GET /api/<stem>/rank`

`serve.py:411-448`. Parametri query: `tokens` (righe-entity, CSV),
`metric` (`sum`/`max`, default `sum`), `topk` (default 8),
`sink_view`/`sink_percentile`/`sink_border` (§4.3). Risposta:

```json
{
  "stem": "...",
  "selected": [3, 4],
  "total_mass": 1.0,
  "cells": [
    {"rank": 0, "cell": 12, "score": 0.083, "pct": 8.3,
     "frames": [24, 25], "rep_frame": 24, "t_sec": 2.4, "is_top": true},
    "..."
  ],
  "sink": {"view": "all", "percentile": 25.0, "border": 0,
           "n_sink": 2304, "has_sink_map": true}
}
```

Il frontend (`card()`, `app.js:28-45`) usa direttamente questo array per
costruire, nell'ordine ricevuto, una `<figure class="card">` per cella:
badge di rank, frame rappresentante e secondo nel video, percentuale di
massa, score grezzo, cella sorgente — oltre all'immagine overlay/originale
(§6).

## 6. Overlay: dalla heatmap di una cella a un'immagine

### 6.1 `make_overlay`

`utils/attn_core.py:397-430`. Per **una** cella temporale `ti` (quindi una
matrice `[grid_h, grid_w]`, non l'intera heatmap):

1. normalizza sui `vmin`/`vmax` **globali della heatmap vista corrente**
   (non min/max della singola cella!) — così celle poco attenzionate
   restano scure e il confronto fra card è visivamente comparabile;
2. upscala **bicubico** alla risoluzione del frame reale (da griglia
   `12×16` a es. `~700×900` px);
3. applica la colormap `hot` (nero→rosso→giallo→bianco), reimplementata a
   mano in `hot_colormap` (`utils/attn_core.py:386-394`, stessi segmenti
   del colormap `hot` di matplotlib, senza dipendere da matplotlib/cv2);
4. alpha-blend col frame originale con **alpha proporzionale
   all'attenzione** (`a = up * alpha`, `alpha` default 0.6): le zone
   fredde (attenzione ~0) restano il frame originale intatto, quelle calde
   virano verso la colormap.

`zero_border` (parametro di `make_overlay`, non esposto dalla UI/API
attuale, default 0) azzererebbe un bordo della heatmap prima del blend —
stessa cautela di `sink_mask(border=...)`, non usato dal server.

### 6.2 Endpoint `GET /api/<stem>/overlay`

`serve.py:492-519`. Parametri: `cell` (obbligatorio, indice temporale),
`tokens`, `sink_view`/`sink_percentile`/`sink_border`. Ricalcola
`aggregate(rows)` → `sink_view(...)` → estrae la riga `ti` → `make_overlay`
coi min/max globali della heatmap vista → PNG. **Ogni singola card della
griglia fa una richiesta separata** per il proprio overlay (`app.js:34`):
`t` immagini per refresh (es. 48 per i video di VNBench in questo store),
tutte generate al volo lato server, nessuna cache lato server oltre alla
`AttentionCapture` in memoria (`get_capture`, `lru_cache(maxsize=32)`,
`serve.py:194-198`).

### 6.3 Toggle overlay/frame originale

Ogni card monta **entrambe** le immagini sovrapposte via CSS (`.over` sopra,
`.orig` sotto, opacità 0 di default su `.orig`); la checkbox "mostra frame
originali" (`video.html:110`) aggiunge la classe `show-orig` al `<body>`,
che porta `.orig` a opacità 1 (`report.css:43-44`) — nessuna nuova
richiesta di rete, solo CSS. La stessa classe `show-orig` decide anche
quale delle due immagini aprire nella lightbox a schermo intero
(`app.js:51-52`).

### 6.4 Endpoint `GET /api/<stem>/frame`

`serve.py:484-490`. Serve il PNG del frame rappresentante di una cella
già estratto (una volta, in fase di cattura) da
`extract_cell_frames` (`capture.py:299-314`) — non ricampiona dal video
originale ad ogni richiesta.

## 7. Riepilogo API (per riferimento rapido)

| Route | Metodo | Query params | Cosa fa |
|---|---|---|---|
| `/v/<stem>` | GET | — | pagina interattiva completa |
| `/api/<stem>/rank` | GET | `tokens`, `metric`, `topk`, `sink_view`, `sink_percentile`, `sink_border` | classifica celle per massa/picco di attenzione, sink-filtrata |
| `/api/<stem>/sink` | GET | `sink_percentile`, `sink_border` | `sink_map` grezza + maschera, indipendente dai token selezionati |
| `/api/<stem>/frame` | GET | `cell` | PNG del frame rappresentante della cella (nessun overlay) |
| `/api/<stem>/overlay` | GET | `cell`, `tokens`, `sink_view`, `sink_percentile`, `sink_border` | PNG del frame + heatmap overlay per la selezione corrente |
| `/video/<stem>` | GET | — | video `.mp4` originale (per la modale di riproduzione) |

Tutti gli endpoint sono **idempotenti e senza side-effect**: la selezione
di token/sink-view vive solo nello stato del browser (nessun salvataggio
server-side), quindi ricaricare la pagina resetta alla selezione di default
(`question_rows`, view `all`).

## 8. Limiti noti dello strumento

- Nessuna cache degli overlay generati: ogni cambio di selezione o di sink
  view rigenera **tutte** le `t` immagini della griglia lato server (accettabile
  per `t` ~ 40-100 e uso interattivo locale, da rivalutare per store molto
  più grandi o uso multi-utente).
- `border` (bias di bordo) è disponibile in API ma non ha un controllo
  dedicato in UI — e comunque non raccomandato senza verifica empirica
  (§4.2).
- La selezione di entity è per **righe di token**, non per span testuali
  arbitrari: due occorrenze dello stesso lemma in punti diversi della
  domanda sono due chip distinti, selezionabili indipendentemente (è il
  comportamento voluto — token diversi possono avere attenzione diversa —
  ma va tenuto a mente leggendo gli screenshot).
- Il regime "merge a coppie" (default, senza `--double-frames` in
  cattura) rende `rep_frame` un **arrotondamento** fra due frame reali:
  l'overlay e il timestamp mostrati sono quindi un'approssimazione di quale
  dei due frame ha effettivamente ricevuto l'attenzione (motivo per cui
  esiste l'alternativa frame-doubling, più costosa ma non ambigua).

## 9. Riferimenti file:riga

- `attn_explorer/serve.py` — app Flask, tutte le route.
- `attn_explorer/templates/video.html` — markup della pagina video.
- `attn_explorer/templates/app.js` — interazione client (selezione,
  fetch, lightbox, modali).
- `attn_explorer/templates/report.css` — stile (non dettagliato riga per
  riga in questo documento oltre ai punti citati).
- `utils/attn_core.py` — `AttentionCapture`, `GridSpec`, `QueryToken`,
  `aggregate`, `question_rows`, `rank_cells`, `sink_mask`, `sink_view`,
  `make_overlay`, `hot_colormap`.
- `attn_explorer/capture.py` — produzione dei `.pt` consumati dal server
  (`extract_cell_frames` per i PNG serviti da `/api/<stem>/frame`).
- `models/qwen_attn.py` — `SINK_DIMS`, calcolo di `sink_map` nel forward
  di cattura (non eseguito dal server: già presente nel `.pt`).
