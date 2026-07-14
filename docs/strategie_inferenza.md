# Strategie di inferenza signal-driven — cosa fanno e come

> Documento tecnico per revisione: descrive le **scelte implementative** delle
> tre strategy disponibili (`strategies/uniform.py`,
> `strategies/entropy_shortcut.py`, `strategies/entropy_attention_resample.py`)
> — formule, soglie, rami decisionali, costo computazionale. Non tratta
> l'organizzazione del codice (dispatch, ABC, dataclass) né le motivazioni
> progettuali: per quelle vedi i docstring dei moduli citati. Il segnale di
> entropia/concentrazione è validato sperimentalmente in
> `docs/entropia_confidenza_mcq.md`; qui si descrive **come viene consumato
> a inferenza**, non la sua validazione.

## 0. Cosa condividono tutte e tre

Ogni strategy risponde a UN sample video-MCQ: riceve un budget di
campionamento (`nframes`, `max_pixels`) fissato dal dataset, e decide come
spenderlo. Le tre differiscono su: quanti forward pass del modello fanno,
se generano testo autoregressivamente o si fermano al prefill, e se
rivedono la selezione dei frame in corsa.

## 1. `uniform` — baseline

`strategies/uniform.py:27-49`. Campionamento uniforme dei frame nell'intero
video (o nella finestra `video_start`/`video_end` se nota), **un solo**
`vlm.generate()` autoregressivo, nessun segnale di confidenza/attenzione
letto. È il comportamento preesistente prima dell'introduzione delle
strategy — riferimento per isolare l'effetto delle altre due.

## 2. `entropy_shortcut` — risposta diretta dal prefill

`strategies/entropy_shortcut.py:33-70`. Non chiama mai `generate()`:

1. Un solo forward di **prefill isolato** (`logits_to_keep=1`,
   `use_cache=False` — vedi §5) produce i logit sull'ultimo token del
   prompt, cioè il primo token che il modello genererebbe.
2. La risposta MCQ è letta direttamente da quei logit: softmax
   **ristretta alle sole lettere candidate** (`A`/`B`/`C`/`D`...),
   rinormalizzata fra loro (`utils/attn_core.py::mcq_answer_stats:137-166`),
   non sull'intero vocabolario. `pred_letter` = argmax di questa softmax
   ristretta.
3. Entropia della risposta in bit: `H = -Σ p_i·log2(p_i)` sulle sole
   probabilità candidate. `0` = certezza assoluta, `log2(n_opzioni)` (2 bit
   per 4 opzioni) = distribuzione uniforme.

Costo: esattamente il prefill che un `generate()` avrebbe comunque fatto
per produrre il primo token — **zero forward aggiuntivi** rispetto a
`uniform`, anzi più economico (niente step autoregressivi successivi).
Serve come proof-of-concept per validare il path `SupportsSignals`
end-to-end, non è pensata per essere la strategy "definitiva".

## 3. `entropy_attention_resample` — ricampionamento condizionato al segnale

`strategies/entropy_attention_resample.py:84-190`. L'unica delle tre che
può rivedere la selezione dei frame **una volta**, sulla base di due
segnali letti dal primo pass. Logica:

### 3.1 Pass 1 (sempre uniforme, budget invariato)

`vlm.generate_with_signals(...)` — stesso prefill isolato di
`entropy_shortcut` (nessun `generate()` in questo pass), ma qui si tiene
anche la mappa di attenzione completa (`VisualAttention.attn`,
`[n_q, t, grid_h, grid_w]`: una heatmap per ogni token della domanda) più
`sink_map` (`[t, grid_h, grid_w]`, score sink per token visivo).

Da questi si calcola `(top1_pct, peak_cell)` (`_peak_cell`,
`entropy_attention_resample.py:66-81`):

1. Media della heatmap su **tutti** i token-query della domanda (nessuna
   selezione di entity — `attn.mean(dim=0)` → `[t, grid_h, grid_w]`).
2. Filtro **sink**: un token visivo è sink se il suo `sink_score` è nel
   top-25° percentile della distribuzione di `sink_map`
   (`sink_percentile`, default 25.0) — soglia al quantile
   `1 − percentile/100` (`utils/attn_core.py::sink_mask:443-491`,
   convenzione **opposta** a quella usata in LoT, vedi
   `docs/lot_look_twice.md §9`). Le celle sink vengono azzerate e la
   mappa residua rinormalizzata (`sink_view`, view="nonsink").
3. `rank_cells` (metric="sum") somma la massa attenzione **per cella
   temporale** (un frame = una cella) sulla mappa non-sink, ordina
   decrescente. `top1_pct` = percentuale della massa **residua**
   (non-sink) che cade sulla cella più attenzionata; `peak_cell` = il suo
   indice.

`sink_border` (default 0, opt-in) forza anche i bordi della griglia a
sink — disattivato di default: non confermato empiricamente sul dominio
video (`docs/analisi_entropia_attenzione_sink.md §4.3`).

`top1_pct`/`peak_cell` vengono **sempre** calcolati e loggati (anche nei
casi in cui poi non si ricampiona), per poter analizzare a posteriori la
loro distribuzione rispetto alla decisione presa.

### 3.2 Condizione di trigger

Il ricampionamento è tentato **solo se tutte** le condizioni valgono
(`can_resample`, `entropy_attention_resample.py:134-140`):

- `frames is None` — i frame non sono già fissati dal dataset (es. MVBench
  `data_type="frame"` esclude sempre il resampling: si accetta il pass 1);
- `options is not None` — sample MCQ (niente segnale di entropia-risposta
  per dataset open-ended);
- segnali disponibili (`answer_entropy`, `top1_pct` non `None`);
- **`answer_entropy > entropy_threshold`** (default `0.7` bit) — il
  modello è "poco sicuro" della risposta dal solo pass 1.

Se il trigger non scatta, la risposta finale viene comunque da un
`generate()` vero (§3.4) sui frame del pass 1 — quindi il ramo "accetta"
è **bit-per-bit identico** a `uniform` sullo stesso sample.

### 3.3 Due rami di ricampionamento, mutuamente esclusivi

Quando il trigger scatta, la scelta fra i due rami dipende **solo** da
`top1_pct` confrontato con `concentration_threshold` (default `30.0`%):

Notazione comune ai due rami: $W = w_1 - w_0$ è l'ampiezza della finestra
vista dal pass 1 (`video_start`/`video_end`, o l'intero video se non
note — nella pratica attuale del progetto è sempre così: nessun dataset
in uso passa un trim, vedi nota a fine sezione); $t$ = `nframes` del
budget; $c$ = `peak_cell` (indice del frame di picco); $D$ = durata reale
del video.

**Ramo A — dispersa** (`top1_pct < concentration_threshold`): il modello è
incerto **e** non si fissa su nessun frame in particolare. Si applica uno
**sfasamento di fase** (`resample_kind="phase_shift"`): la finestra resta
larga $W$ ma trasla di mezzo intervallo-fra-frame ($\beta$ = `phase_shift_frac`,
default $0.5$):

$$
s = \beta \cdot \frac{W}{t}
\qquad\Rightarrow\qquad
\text{nuova finestra} = [\,w_0 + s,\ w_1 + s\,] \ \text{(clampata a } [0, D])
$$

I nuovi frame cadono "in mezzo" a quelli già visti dal pass 1, nell'ipotesi
che l'informazione mancante sia in un frame non campionato dalla griglia
uniforme originale.

**Ramo B — concentrata ma incerta** (`top1_pct >= concentration_threshold`):
il modello si fissa già su un frame ma resta incerto — ipotesi: serve più
dettaglio *attorno* a quel punto. Si applica uno **zoom**
(`resample_kind="zoom_peak"`, $\alpha$ = `window_frac`, default $0.3$):

$$
p = w_0 + \frac{c + 0.5}{t}\cdot W \quad \text{(istante di picco, in secondi)}
$$

$$
\text{nuova finestra} = \Bigl[\,p - \tfrac{\alpha W}{2},\ \ p + \tfrac{\alpha W}{2}\,\Bigr] \ \text{(clampata a } [0, D])
$$

In entrambi i rami gli stessi `nframes` del budget vengono ricampionati
uniformemente dentro la nuova finestra: nel ramo A la finestra ha
**la stessa ampiezza** $W$ ma spostata; nel ramo B è **più stretta**
($\alpha W$, il 30% di default) e centrata sul picco — stesso numero di
token visivi, più risoluzione temporale locale attorno al frame ritenuto
rilevante.

Nota su $W$: essendo `video_start`/`video_end` sempre `None` nei dataset
attualmente usati con questa strategy (Video-MME; LVBench solo con
`use_time_reference=true`, non il default), $W$ coincide in pratica con la
durata totale del video, non con un sottointervallo.

### 3.4 Risposta finale — sempre una vera generazione

Sia che si ricampioni sia che no, la risposta restituita viene **sempre**
da `vlm.generate()` (mai dal solo `pred_letter` della softmax ristretta
del pass 1) — sul media originale se non c'è stato resampling, sul media
ricostruito dalla nuova finestra altrimenti. Questo isola l'effetto del
resampling ai soli sample in cui avviene davvero: il ramo "accetta" non
introduce nessuna differenza rispetto a `uniform`.

**Al massimo un ricampionamento**: non c'è un terzo pass che rivaluti il
segnale sui nuovi frame per decidere se ricampionare ancora.

## 4. Segnali condivisi (riepilogo formule)

| Segnale | Dove | Formula |
|---|---|---|
| Entropia risposta (bit) | `utils/attn_core.py::mcq_answer_stats:137-166` | softmax ristretta alle lettere candidate sui logit dell'ultimo token di prefill; `H=-Σp·log2(p)` |
| Sink score (per token visivo, per layer) | `models/qwen_attn.py::full_visual_attention:195-204` | `max(\|hidden[sink_dims]\|) / (RMS(hidden) + 1e-6)`, mediato sui layer centrali |
| Sink dims (canali hardcoded, per modello) | `models/qwen_attn.py:18-27` | Qwen2.5-VL-3B: `(318, 1874, 1819)` — da `lot/retrieval.py`, non ricalcolati in questo progetto |
| Sink mask | `utils/attn_core.py::sink_mask:443-491` | token sink ⇔ `sink_score ≥ quantile(1 − percentile/100)`; default `percentile=25` → il 25% più alto è sink |
| Concentrazione (`top1_pct`) | `utils/attn_core.py::attention_concentration:535-568`, riuso locale in `entropy_attention_resample.py::_peak_cell` | massa (sink-filtrata, rinormalizzata) sulla cella temporale più attenzionata, in % della massa residua totale |

Layer aggregati: metà centrale del decoder per default (`layer_range=None`
→ `[n_layers//4, 3·n_layers//4)`, `models/qwen_attn.py:238-239`) — stesso
concetto di `middle_half` usato in LoT (`docs/lot_look_twice.md §3`).

## 5. Costo computazionale per strategy

| Strategy | Forward pass | Note |
|---|---|---|
| `uniform` | 1 (`generate()` autoregressivo) | baseline |
| `entropy_shortcut` | 1 (prefill isolato, `use_cache=False`) | **più economico** di `uniform` — nessuno step autoregressivo |
| `entropy_attention_resample` | 1 prefill (pass 1, sempre) + 1 `generate()` completo (pass 2, sempre) | **sempre più costoso** di `uniform`, indipendentemente da `resampled` — il prefill del pass 1 è overhead fisso per OGNI sample, non solo per quelli ricampionati |

Il prefill isolato di pass 1 usa `use_cache=False`
(`models/qwen_attn.py:221`): è un forward diagnostico, la KV cache che
produrrebbe verrebbe scartata subito insieme all'output — inutile tenerla
in VRAM. A `nframes` alti (es. 128 su Video-MME) questo risparmio di
memoria, insieme a un `torch.cuda.empty_cache()` esplicito dopo il pass 1
(`models/qwen_attn.py:233`), evita che un OOM su un sample lasci memoria
"reserved but unallocated" dall'allocator di PyTorch — osservato
empiricamente come fallimento a cascata su **tutti** i sample successivi
nello stesso processo prima di questo fix (run `ac1lrtvg`: 63/250 sample
completati, poi tutti falliti; dopo il fix, run `y6u5k1pq` con la stessa
config: 250/250 completati).

## 6. Parametri configurabili (`conf/config.yaml`, preset `strategy.*`)

| Parametro | Default | Significato |
|---|---:|---|
| `entropy_threshold` | `0.7` (bit) | soglia minima di entropia risposta per considerare il resampling |
| `concentration_threshold` | `30.0` (%) | soglia di `top1_pct` che discrimina ramo dispersa (A) vs concentrata (B) |
| `window_frac` | `0.3` | frazione della durata originale usata come finestra di zoom (ramo B) |
| `phase_shift_frac` | `0.5` | frazione dell'intervallo fra frame usata come traslazione (ramo A) |
| `sink_percentile` | `25.0` | percentile per la maschera sink |
| `sink_border` | `0` (off) | anelli di bordo forzati a sink (opt-in, non validato) |

**Nessuna di queste soglie è validata sul dominio video**: i valori di
default derivano dal campione VNBench "retrieval" analizzato in
`docs/entropia_confidenza_mcq.md` (dominio e `nframes` diversi da
Video-MME). Lo sweep in corso (`scripts/sweep_video_mme_thresholds.sh`,
tabella in `README.md`) esiste per ricalibrarle empiricamente su
Video-MME.

## 7. Capture opzionale per ispezione post-hoc

Entrambe le strategy signal-driven possono scrivere un `capture.pt` (+
frame estratti) leggibile da `attn_explorer.serve`, dietro il flag
`capture_attention: true` nel preset (`Strategy._save_attention_capture`,
`strategies/base.py:158-198`):

- `entropy_shortcut`: un capture per sample (il solo pass esistente).
- `entropy_attention_resample`: un capture **solo per i sample
  ricampionati** (`resampled=true`), con `resample_kind`/`top1_pct` come
  metadati extra — permette di ispezionare visivamente proprio i casi in
  cui la strategy ha deviato dalla baseline uniforme.

Disattivato di default in entrambe (I/O non trascurabile per sample).
