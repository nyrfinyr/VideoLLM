# Confidenza del modello (entropia della risposta MCQ) come segnale di correttezza

> **Stato** (2026-07-05): pipeline di cattura estesa con dati di confidenza
> "gratuiti" (nessun forward aggiuntivo), server interattivo con pagina di
> analisi live, prima evidenza sperimentale raccolta su 30 video VNBench
> "retrieval". Documento preparato come input per un agente di formulazione
> di ipotesi di ricerca — riporta meccanismo, funzionalità disponibili,
> dataset, numeri grezzi e caveat, senza precotturare le conclusioni.

## 1. Domanda di partenza

Su un benchmark video-MCQ (Qwen2.5-VL-3B, prefill-only), possiamo usare la
**confidenza del modello nella propria risposta** (quanto è piccata la
distribuzione di probabilità sulle lettere candidate) come segnale
diagnostico per capire *quando* e *perché* il modello sbaglia? Due letture
possibili sono state messe alla prova su un primo campione:

1. Bassa entropia (alta confidenza) ⇒ risposta corretta; alta entropia
   (il modello "tira a indovinare") ⇒ risposta più spesso sbagliata.
2. Quando il modello indovina, è perché il frame che conterrebbe
   l'informazione risolutiva ("needle") non è stato campionato dal
   sotto-campionamento uniforme dei frame del video.

## 2. Meccanismo di misura (come si calcola l'entropia, a costo zero)

Per ogni sample MCQ, il forward di prefill produce comunque i logit
sull'ultimo token della sequenza (`logits_to_keep=1`, nessuna generazione
necessaria). Il segnale di confidenza si ottiene **senza compute
aggiuntivo**, riusando quei logit:

1. Si individuano gli id-token delle lettere candidate (`A`, `B`, `C`, ...
   secondo il numero di opzioni del sample).
2. Si estraggono i logit grezzi (`answer_logits`, pre-softmax) su
   *solo quegli id* e si applica softmax **ristretta a quel sottoinsieme**
   → `answer_probs` per lettera (non è la probabilità sul vocabolario
   intero: è "dato che la risposta è una di queste lettere, quanto
   probabile è ciascuna").
3. `pred_letter` = argmax di `answer_probs`.
4. `answer_entropy` = H = −Σ p·log2(p) sulle lettere candidate, in **bit**.
   - H = 0 → certezza assoluta (tutta la massa su una lettera).
   - H = log2(n_opzioni) → massima incertezza (distribuzione uniforme,
     4 opzioni → 2 bit): il modello non ha una preferenza, sta indovinando.

In aggiunta (calcolati anch'essi dallo stesso forward, per
debug/riproducibilità — non usati nell'analisi statistica sotto):

- `top_tokens`: top-10 token dell'**intero vocabolario** (non ristretto
  alle lettere MCQ) con logit e probabilità — utile per vedere cosa il
  modello direbbe "spontaneamente" senza il vincolo di rispondere con una
  lettera.
- `capture_meta`: `model_id`, `dtype`, `attn_implementation`, `nframes`,
  `double_frames`, `max_pixels`/`min_pixels`, timestamp UTC, commit git
  correnti al momento della cattura — permette di risalire esattamente
  alle condizioni sperimentali di un dato `.pt` senza dover rilanciare la
  cattura (costosa, richiede GPU) solo per saperlo.

## 3. Funzionalità disponibili per esplorare i dati (tooling di questa sessione)

### 3.1 Cattura (`attn_explorer/capture.py`, `utils/attn_core.py`)

- Ogni cattura (`uv run python -m attn_explorer.capture --dataset ... --n K`)
  produce, per ogni video, `<store>/<stem>/capture.pt` contenente — oltre
  alla mappa di attenzione testo→visivo — `answer_logits`, `answer_probs`,
  `answer_entropy`, `pred_letter`, `top_tokens`, `capture_meta` come
  descritto sopra.
- `index.json` dello store viene ora scritto **incrementalmente** (dopo
  ogni video, non solo a fine batch): un job lungo interrotto (Ctrl-C,
  OOM, crash) lascia comunque tutti i video già catturati immediatamente
  navigabili, senza perdere lavoro né dover indovinare cosa è stato fatto.

### 3.2 Server interattivo (`attn_explorer/serve.py`)

```bash
uv run python -m attn_explorer.serve --store data/vnbench/output_v4/retrieval \
    --videos data/vnbench/retrieval
```

- **`/`** — griglia di tutti i video dello store, con un **badge
  corretto/sbagliato** (verde/rosso) per ispezione visiva rapida di dove il
  modello fallisce.
- **`/v/<stem>`** — pagina per singolo video: domanda, risposta corretta,
  risposta del modello con tabella completa **logit pre-softmax /
  probabilità post-softmax / contributo all'entropia**, per ciascuna
  lettera candidate; sezioni richiudibili per i top-10 token non ristretti
  e i metadati di riproducibilità; un bottone apre una modale con **tutti**
  i dati raw del `.pt` (utile per debug e per mostrare il dato grezzo a un
  supervisore); un bottone apre il video originale in una modale.
- **`/analisi`** (nuova) — calcolata **live** da `index.json` su tutti gli
  store passati al server (non uno snapshot statico): correlazione di
  Pearson r(entropia, corretta), entropia media per gruppo
  corretta/sbagliata, rapporto fra le medie, grafico a dispersione (con
  jitter) per ispezione punto-per-punto, tabella di dettaglio linkata alle
  pagine dei singoli video. Raggiungibile da un link in cima a `/`.

Questa pagina ricalcola i numeri riportati nella sezione 5 automaticamente
ogni volta che vengono catturati altri video: è lo strumento da usare per
verificare se un pattern osservato su 30 video si conferma o si diluisce
crescendo il campione.

## 4. Dataset del primo campione sperimentale

- **Benchmark**: VNBench, categoria **"retrieval"** — 3 sotto-tipi:
  `ret_insert1`, `ret_insert2`, `ret_edit1` (un elemento/needle inserito o
  modificato nel video; la domanda MCQ chiede di identificarlo).
- **Modello**: Qwen2.5-VL-3B, prefill-only (nessuna generazione), frame
  campionati uniformemente con frame-doubling (vedi `capture_meta` nel
  `.pt` di ciascun video per `nframes` esatto usato).
- **N = 30 video**, store `data/vnbench/output_v4/retrieval`.
- **Durata video**: 10.0–150.5 s (media 47.6 s).
- **Breakdown per sotto-tipo** (corrette / sbagliate):

  | sotto-tipo | n | corrette | sbagliate |
  |---|---|---|---|
  | ret_insert1 | 11 | 8 | 3 |
  | ret_insert2 | 10 | 7 | 3 |
  | ret_edit1 | 9 | 8 | 1 |

## 5. Evidenza raccolta

### 5.1 Ipotesi 1 — confidenza alta ⇒ risposta corretta

| gruppo | n | entropia media (bit) | mediana | dev. std. |
|---|---|---|---|---|
| risposte corrette | 23 | 0.193 | 0.039 | 0.330 |
| risposte sbagliate | 7 | 1.319 | 1.519 | 0.639 |

- **Correlazione di Pearson r(entropia, corretta) = −0.75** (corretta
  codificata 1, sbagliata 0 — coerente col segno atteso: entropia alta
  associata a "non corretta").
- Rapporto fra le medie: le risposte sbagliate hanno entropia media **~6.8×**
  quella delle risposte corrette.
- Le 7 risposte sbagliate sono concentrate nella metà alta della
  distribuzione di entropia (nessuna sbagliata con entropia sotto ~0.2 bit,
  tranne un'eccezione sotto).
- **Controesempio degno di nota**: `v_7OcxT66BxX0_ret_insert1` ha entropia
  0.229 bit (bassa/moderata, non "sta indovinando" nel senso di
  distribuzione piatta) ma risposta sbagliata (pred `B`, gt `D`) — l'unico
  caso in cui confidenza relativamente alta non implica correttezza.
  Utile come caso da ispezionare a mano (via `/v/v_7OcxT66BxX0_ret_insert1`)
  per capire se è un errore di lettura visiva o un errore "sicuro di sé"
  del modello.

### 5.2 Ipotesi 2 — l'incertezza nasce da un frame-needle non campionato

**Proxy usato** (grezzo, definito per questa prima verifica — non salvato
nel `.pt`, calcolato offline da `needle_time`/durata video del sidecar
VNBench):

- `min_gap_s` = distanza in secondi fra il timestamp del needle annotato e
  il frame campionato più vicino nel tempo.
- `avg_interval_s` = intervallo medio fra due frame campionati consecutivi
  (dipende da durata video / numero di frame campionati).
- `covered` = `min_gap_s ≤ avg_interval_s / 2` (criterio stile Nyquist: se
  il frame più vicino cade entro metà del passo di campionamento, il
  needle "dovrebbe" essere stato catturato da qualche frame vicino).

**Risultato**: su 30 video, solo **1** ha `covered = False`
(`video7653_ret_insert1`, gap 0.16 s contro una soglia di 0.149 s) — e
quel video è stato **risposto correttamente**. Tutti i 7 casi sbagliati
hanno `covered = True`, con `min_gap_s` fra 0.033 s e 0.400 s (media 0.270 s,
praticamente indistinguibile dalla media dei casi corretti, 0.253 s):

| video (sbagliato) | entropia (bit) | pred | gt | gap dal needle (s) | durata video (s) |
|---|---|---|---|---|---|
| 3081360150_ret_insert1 | 1.951 | B | C | 0.292 | 59.4 |
| v_zTHkqpNFGno_ret_insert2 | 1.945 | A | C | 0.286 | 105.7 |
| v_4uKoAk5NCkI_ret_edit1 | 1.776 | D | A | 0.033 | 62.0 |
| 13199465314_ret_insert2 | 1.519 | A | B | 0.141 | 46.8 |
| 8283116453_ret_insert2 | 1.283 | B | C | 0.400 | 44.6 |
| 3501167063_ret_insert1 | 0.529 | D | A | 0.225 | 59.8 |
| v_7OcxT66BxX0_ret_insert1 | 0.229 | B | D | 0.514 | 130.0 |

**Lettura**: con questo proxy, l'ipotesi "il modello indovina perché il
needle non è mai stato campionato" **non è supportata** — un frame
temporalmente vicinissimo al needle risulta comunque incluso nel
campionamento anche nei casi sbagliati, e l'unico caso "non coperto" per
soglia è stato risposto correttamente (pattern opposto a quello atteso).

**Caveat esplicito sul proxy** (da tenere presente prima di scartare
l'ipotesi): `min_gap_s` misura solo **prossimità temporale**, non verifica
che il needle sia effettivamente **leggibile** nel frame campionato più
vicino (risoluzione dopo il resize, motion blur, durata reale
dell'inserimento del needle nel video, occlusioni). Con solo **N = 7**
errori nel campione, il potere statistico per escludere l'ipotesi in senso
stretto è comunque basso. È più corretto dire: *"la sola prossimità
temporale del frame più vicino al needle non spiega gli errori osservati
in questo campione"*, lasciando aperta la possibilità che la causa sia
leggibilità/risoluzione del needle nel frame effettivamente campionato,
ambiguità fra opzioni distrattrici, o errore di lettura del modello a
parità di evidenza visiva disponibile.

## 6. Limiti del campione attuale

- Un'unica categoria VNBench ("retrieval") su tre disponibili nel dataset
  — non è noto se il pattern regga su categorie con altro tipo di quesito
  (es. ordinamento temporale, conteggio).
- N = 30 è piccolo, e gli errori sono solo 7: stime di media/mediana per il
  gruppo "sbagliate" sono rumorose.
- Un solo modello (Qwen2.5-VL-3B) e una sola configurazione di
  campionamento frame — non è noto se `r` cambi con più frame, con
  frame-doubling disattivato, o con un modello più grande.
- Il proxy di "copertura" (§5.2) è un'euristica temporale grezza, non una
  verifica visiva del contenuto del frame.

## 7. Cosa si può fare senza ricatturare nulla

Tutti i `.pt` già catturati contengono `answer_logits`/`answer_probs`/
`answer_entropy`/`top_tokens`/`capture_meta` — nuove analisi su questi 30
video (es. guardare `top_tokens` per capire cosa il modello "vorrebbe dire"
senza il vincolo MCQ, o rianalizzare con soglie di entropia diverse) non
richiedono un nuovo forward, solo lettura dei `.pt` esistenti o uso di
`/analisi` e `/v/<stem>` nel server già in esecuzione. Servirebbe invece
una nuova cattura per: altre categorie VNBench, altri modelli, o varianti
di campionamento frame.

## 8. Riferimenti

- Store dati: `data/vnbench/output_v4/retrieval/` (+ video originali in
  `data/vnbench/retrieval/`).
- Codice: `attn_explorer/capture.py`, `attn_explorer/serve.py`,
  `utils/attn_core.py` (funzioni `mcq_answer_stats`, `top_k_logits`).
- Pagina interattiva: `uv run python -m attn_explorer.serve --store
  data/vnbench/output_v4/retrieval --videos data/vnbench/retrieval`, poi
  `/analisi` per la correlazione live, `/v/<stem>` per ispezionare un
  singolo caso (es. i due controesempi citati sopra).
