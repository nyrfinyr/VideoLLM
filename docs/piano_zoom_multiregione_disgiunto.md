# Piano — zoom multi-regione disgiunto (Opzione C) per `entropy_attention_resample`

> **Stato**: **C1 implementato** (codice, 2026-07-15) — knob dietro flag,
> default OFF. **Code review superata** (2026-07-15, finding in coda). **Non
> ancora validato empiricamente** su cluster (§4): manca la run del
> 250-subset su GPU 24G e l'analisi win/loss. Questo documento fissa le
> decisioni prese nella discussione di design e traccia il percorso
> implementativo + protocollo di validazione.
>
> **Contesto a monte**: leggere prima
> `docs/analisi_winloss_resampling_video_mme.md` (perché il resampling attuale
> ha un effetto netto piccolo e `concentration_threshold` è inerte) e la
> sezione "Sweep" del `README.md`.

## 1. Da dove veniamo (decisioni di design)

Catena di ragionamento che porta a questo piano:

1. **`concentration_threshold` fisso è inerte su Video-MME.** `top1_pct`
   (percentuale di attenzione sink-filtrata sul frame di picco, per-cella
   temporale — vedi `utils/attn_core.py:rank_cells`, `heatmap.flatten(1).sum(1)`)
   ha mediana ~6% e max ~23%, mentre le soglie testate (20/30/45) stanno al
   limite o sopra il massimo → il ramo `zoom_peak` non si attiva quasi mai,
   tutto va in `phase_shift` (confermato dall'analisi win/loss).

2. **Soglia adattiva.** Dato che le pct sommano a 100 su `t` celle, la media è
   fissa a `100/t` (≈1.5% a 128 frame, t≈64) — proprio il livello "uniforme se
   tutti i frame fossero ugualmente informativi". Una soglia basata sulla
   distribuzione del singolo sample (z-score `(top1_pct − 100/t)/σ`, o robusta
   `mediana + k·MAD`) è scale-invariant e risolve l'inerzia. Caveat: la media
   non porta informazione (fissa), il segnale è tutto nella dispersione; e la
   distribuzione è heavy-tailed, quindi σ va calcolata escludendo il picco per
   evitare auto-mascheramento.

3. **Entropia dell'attenzione come selettore di ramo.** Meglio del solo
   `top1_pct`: entropia di Shannon **normalizzata** `H/log t ∈ [0,1]` sulle `t`
   masse per-cella (sink-filtrate). Mappa pulita sui rami esistenti:
   - `H` **bassa** = attenzione **concentrata** = "sappiamo dove guardare" →
     **zoom**;
   - `H` **alta** = attenzione **dispersa** = "non sappiamo dove" →
     `phase_shift`.
   È l'analogo, sul versante attenzione, dell'`answer_entropy` (che resta il
   **gate**, decide *se* ricampionare; l'entropia-attenzione decide *come*).
   L'entropia dice *se* si può localizzare; il *dove* preciso resta il
   peak-set. Costo ≈ zero: le masse per-cella sono già in `rank_cells`.

4. **Multi-picco.** Se l'informazione sta su 2-3 frame vicini (non uno), un
   solo `peak_cell` è riduttivo. Ma il multi-picco **vero** (regioni
   temporalmente lontane) si scontra con l'encoding a tempo assoluto di
   Qwen2.5-VL (vedi §2). Erano emerse 4 opzioni:
   - **A** — regioni disgiunte concatenate in un video: rompe le distanze
     temporali senza timestamp reali.
   - **B** — finestra unica che racchiude tutti i picchi: corretta ma degenera
     (≈ tutto il video) se i picchi sono lontani → nessuno zoom.
   - **C** — densità non uniforme via lista di frame (importance sampling):
     massima fedeltà, ma è il caso che rompe l'encoding assoluto → serve
     risolvere i timestamp.
   - **D** — finestra singola dal *centroide* del peak-set con larghezza
     adattiva ("multi-picco morbido"): resta nel path `Video[start,end]`
     sicuro, nessun debito tecnico.

   In §"osservazione" era emerso che il gate a entropia **filtra da solo** il
   caso multi-regione lontano (entropia alta → `phase_shift`), quindi lo zoom
   riceverebbe quasi sempre un singolo cluster stretto → D basterebbe.

5. **Decisione finale (2026-07-15).** Nonostante (4), si vuole **verificare
   empiricamente** che lo zoom su un **vero multi-regione disgiunto**
   (Opzione **C**) funzioni — non darlo per superfluo sulla base del solo
   ragionamento. Quindi: si pianifica **C**. L'Opzione D resta un'alternativa
   più economica da tenere come fallback/confronto, non è questo il piano.

## 2. Il vincolo tecnico centrale: encoding a tempo assoluto single-fps

Qwen2.5-VL colloca ogni grid temporale nel tempo *assoluto* via M-RoPE,
usando `second_per_grid_ts` che il processor deriva da **un unico `fps`
scalare per clip**. Verificato nel path attuale:

- `models/qwen.py:146` — `process_vision_info(..., return_video_kwargs=True)`
  restituisce `video_kwargs = {'do_sample_frames': False, 'fps': [sample_fps]}`
  (`qwen_vl_utils/vision_process.py:528-530`): **un fps, non timestamp
  per-frame**.
- Path lista-di-frame (`VideoFrames`): `fetch_video`
  (`vision_process.py:415-446`) costruisce *fake metadata* con
  `frames_indices=[0..n-1]` (interi contigui) e `sample_fps = ele.get(
  "sample_fps", 2.0)`. Anche qui **un solo fps**, spaziatura assunta uniforme.

**Conseguenza**: non c'è modo di primo livello per esprimere
"frame clusterizzati con buchi temporali reali". Un set di frame estratti da
regioni disgiunte, passato come lista, verrebbe interpretato come
**uniformemente spaziato** all'`sample_fps` iniettato — i buchi tra le regioni
spariscono dall'encoding. Questo è esattamente ciò che lo zoom singolo (una
finestra `Video[start,end]` uniforme) evita e che C deve gestire.

Da qui due sotto-varianti di C, da affrontare **in sequenza**:

### C1 — pragmatica, tempo degradato (validazione prima)
Estrarre l'unione dei frame dalle `m` regioni selezionate e passarli come
lista con un unico `sample_fps` iniettato. Il modello li vede uniformi: si
**accetta** la distorsione temporale (i buchi tra regioni scompaiono), in
cambio di più risoluzione *dentro* ogni regione. È la variante che risponde
alla domanda dell'utente ("lo zoom multi-regione aiuta *affatto*?") al costo
minimo. **Da fare per prima.**

### C2 — tempo corretto, invasiva (solo se C1 promette)
Iniettare i timestamp assoluti reali per-frame nei position-id temporali del
M-RoPE (patch a livello modello: `get_rope_index`/gestione
`second_per_grid_ts`). `Qwen25VLAttention` già personalizza il path di
attenzione (`models/qwen_attn.py`), quindi è fattibile ma non banale. Si
affronta **solo se** C1 mostra che lo zoom multi-regione recupera sample e
che la distorsione temporale di C1 è il collo di bottiglia.

## 3. Piano implementativo (C1)

Tutto dietro flag di config, **comportamento invariato di default** — non
deve toccare i numeri dello sweep esistente finché non è esplicitamente
attivato.

### 3.1 Config (`conf/config.yaml`, preset `strategy.entropy_attention_resample`)
Nuovi knob (con default che disattivano C1):
- `zoom_multi_region: false` — master switch di C1.
- `n_regions: 3` — numero massimo di regioni da selezionare.
- `region_select: "adaptive"` — `"topk"` (prime `n_regions` celle per massa)
  oppure `"adaptive"` (celle sopra `mediana + k·MAD`, poi cap a `n_regions`).
- `region_mad_k: 2.0` — soglia adattiva in unità di MAD (vedi §1.2).
- `region_merge_gap_frac: 0.05` — celle selezionate a distanza temporale <
  questa frazione della finestra vengono fuse in un'unica regione (evita
  regioni ridondanti adiacenti).
- `region_window_frac: 0.1` — semi-larghezza di ogni regione (frazione della
  finestra pass-1), attorno al centro della regione.
- `budget_allocation: "proportional"` — `"equal"` o `"proportional"` (alla
  massa di attenzione della regione) per spartire `budget.nframes` fra le
  regioni.
- `force_zoom_for_debug: false` — **flag di test**: instrada in zoom
  multi-regione TUTTI i sample ricampionati, bypassando la decisione di ramo.
  Serve a validare C1 in isolamento senza dipendere dal fix del selettore di
  ramo (vedi §4, nota su ordine).

### 3.2 Codice — `strategies/entropy_attention_resample.py`
Punto d'innesto: il ramo "concentrata" (oggi `zoom_peak`,
`entropy_attention_resample.py:158-167`). Quando `zoom_multi_region` è attivo
e si entra in zoom:

1. **Masse per-cella sink-filtrate** — già disponibili: replicare la logica di
   `_peak_cell` (`sink_view(heat, sink_map, view="nonsink", ...)` +
   `rank_cells(..., metric="sum")`) ma tenere l'**intera** lista `ranked`, non
   solo il top1. Ogni `RankedCell` porta già `pct`, `cell`, `rep_frame`,
   `t_sec` (`utils/attn_core.py:377-386`).
2. **Selezione regioni** — da `ranked` scegliere le celle secondo
   `region_select`/`region_mad_k`, convertirle a tempi via la stessa mappa
   frazionaria del ramo attuale (`frac = (cell+0.5)/t; time = w0 + frac*(w1-w0)`),
   fondere quelle entro `region_merge_gap_frac`, ottenendo `m ≤ n_regions`
   regioni disgiunte `[a_j, b_j]` con `b_j - a_j = 2·region_window_frac·(w1-w0)`
   clampate a `[0, duration]`.
3. **Allocazione budget** — spartire `budget.nframes` fra le `m` regioni
   (`budget_allocation`), **mantenendo la somma ≤ `budget.nframes`** (vincolo
   VRAM: non aumentare il conteggio token oltre il pass-1 — importante dato lo
   storico OOM su 24G, vedi README).
4. **Estrazione frame** — per ogni regione, campionare uniformemente i suoi
   frame (riusare `reconstruct_frame_indices` /
   `utils/attn_core.py:686` per mappare tempi→indici assoluti, poi estrarre i
   path frame come fa `extract_cell_frames`, `attn_core.py:724`), concatenare
   in un'unica lista ordinata per tempo.
5. **Media** — servirà una via per passare la lista di frame **con
   `sample_fps` iniettato**. `VideoFrames` attuale (`models/media.py:44`) non
   espone `sample_fps` (lascia il default 2.0 di qwen-vl-utils). Estendere
   `VideoFrames` con un campo opzionale `sample_fps: float | None` inoltrato in
   `to_content_dict` (qwen-vl-utils lo legge come `ele.get("sample_fps")`,
   `vision_process.py:434`). In `_build_media`/una nuova branch, costruire
   questo `VideoFrames` esteso. **Nota C1**: `sample_fps` resta uno scalare →
   spaziatura uniforme assunta, è la distorsione accettata.
6. **Logging** — aggiungere al dict di ritorno: `resample_kind =
   "zoom_multi_region"`, `n_regions_used`, `region_spans` (lista `[a_j,b_j]`),
   `attention_entropy_norm` (se il selettore di ramo a entropia è già in
   piedi), così l'analisi win/loss (`docs/analisi_winloss_resampling_video_mme.md`)
   si può ripetere confrontando i tre stili di ramo.

### 3.3 Codice — `models/media.py`
Estendere `VideoFrames` con `sample_fps: float | None = None` (e
`min_frames`/`max_frames` se servono per il padding a fattore). Verificare che
`to_content_dict` (drop dei `None`) inoltri il campo solo quando valorizzato.
Nessun'altra modifica al path `generate`.

## 4. Protocollo di validazione

Subset e infrastruttura identici allo sweep (per confrontabilità diretta):
`limit=250 shuffle=true seed=42`, `model=qwen2_5_vl_3b_attn`,
`scripts/sbatch/video_mme-signal-test.sbatch`, **constraint pinnato alle GPU
24G** (`gpu_RTX6000_24G|gpu_RTX_A5000_24G` — decisione 2026-07-15: si valida
proprio sulle card memory-constrained per verificare che C1 ci stia).

Caveat OOM: nello sweep le run della signal strategy erano OOMmate **solo**
sull'host `nullazzo`/RTX6000 24G (non su `huber`/A5000 24G, vedi win/loss
doc). Importante: quell'OOM era nel forward di prefill del segnale
(`full_visual_attention`, `models/qwen_attn.py:221`) — **che C1 non tocca**:
il ramo multi-regione cambia solo il media della seconda `generate()`, con
frame totali ≤ `budget.nframes` (stesso footprint dello zoom singolo). Quindi
il comportamento a 24G di C1 dovrebbe ricalcare quello della strategy
esistente: un eventuale OOM su `nullazzo` sarebbe un problema pre-esistente
del prefill, non una regressione di C1. Se si vuole isolare C1 dal nodo
noto-cattivo, pinnare la sola A5000 (`gpu_RTX_A5000_24G`), che nello sweep ha
girato la signal strategy senza problemi.

Passi:
1. **Isolare C1** con `force_zoom_for_debug=true`: instrada in zoom
   multi-regione TUTTI i sample che superano il gate `answer_entropy`,
   ignorando il selettore di ramo. Risponde alla domanda dell'utente ("lo zoom
   multi-regione disgiunto funziona?") senza dipendere dal fix del ramo.
2. **Win/loss vs uniform e vs phase_shift**: ripetere l'analisi sample-per-
   sample (join su `example.id`, script in
   `docs/analisi_winloss_resampling_video_mme.md`) confrontando:
   - C1-forced vs controllo `uniform` (net fixed−broken);
   - C1-forced vs `et0.7-ct30` (phase_shift), **limitandosi ai sample che
     entrambi ricampionano** — la domanda vera è se lo zoom multi-regione
     recupera sample che `phase_shift` NON recupera (o viceversa ne rompe).
3. **Ablazioni economiche** solo se il passo 2 è positivo: `n_regions`
   (1=degenera a zoom singolo, 2, 3), `budget_allocation` (equal vs
   proportional), `region_window_frac`.

### Nota su ordine e dipendenze
- C1 con `force_zoom_for_debug` **non dipende** dal selettore di ramo a
  entropia (§1.3): sono due lavori ortogonali. Il debug flag esiste apposta
  per validare C1 da solo.
- Se e quando il ramo a entropia viene implementato, C1 diventa il ramo
  "concentrato"; a quel punto va rimisurato *senza* il debug flag, sul
  sottoinsieme reale instradato in zoom.
- **VRAM**: C1 aggiunge solo la solita seconda `generate()` (già presente nel
  resampling), a patto di rispettare il vincolo §3.2.3 (frame totali ≤
  `budget.nframes`). Nessun costo di memoria aggiuntivo atteso oltre al
  resampling attuale.

## 5. Definition of done (C1)
- Flag `zoom_multi_region`/`force_zoom_for_debug` funzionanti; default OFF ⇒
  sweep esistente riproducibile bit-per-bit.
- `VideoFrames.sample_fps` estende il media senza rompere i path `Video`/
  `Image` esistenti.
- Run di validazione (250-subset, GPU 24G) con win/loss loggato e
  confrontato vs `uniform` e vs `phase_shift`.
- Documentato l'esito nel `README.md` (tabella sweep / nota) e in un follow-up
  di questo file (sezione "Risultati C1").
- Decisione go/no-go su C2 (tempo corretto) basata sull'esito.

## Risultati C1

**Codice** (2026-07-15): implementato in `strategies/entropy_attention_resample.py`
(`_ranked_cells`, `_select_region_cells`, `_multi_region_spans`,
`_allocate_region_budget`, `_build_multi_region_media`) + knob nel preset
`strategy.entropy_attention_resample` di `conf/config.yaml` + `VideoFrames.
sample_fps` in `models/media.py`. Logica di selezione/merge/allocazione
budget verificata con test standalone (senza dipendenze torch/decord, non
installabili in questo ambiente) su scenari sintetici: picchi separati →
regioni disgiunte, picchi vicini → merge, budget più stretto del numero di
regioni → si scartano le regioni a massa minore. Comportamento di default
(flag OFF) verificato per ispezione: bit-per-bit equivalente al codice
precedente.

**Non ancora fatto**: run di validazione reale (§4) — richiede GPU e non è
stata eseguita in questo passaggio. Prossimo passo: lanciare il 250-subset
con `force_zoom_for_debug=true`, `zoom_multi_region=true` su
`scripts/sbatch/video_mme-signal-test.sbatch` (constraint GPU 24G, vedi §4) e
ripetere l'analisi win/loss.

## Note dalla code review (2026-07-15)

Review del diff C1 su `main`: requisiti del piano (§3) tutti soddisfatti,
default path (flag OFF) verificato bit-per-bit equivalente al codice
precedente, vincolo VRAM (somma frame ≤ `budget.nframes`) verificato,
disgiunzione delle regioni verificata con test sulle funzioni pure. Nessun
finding bloccante. Finding minori da tenere presenti (non bloccano la
validazione), in ordine di rilevanza per l'esito:

1. **`reconstruct_frame_indices(..., n=1)` ritorna il frame di *inizio*
   regione, non il centro/picco** (`torch.linspace(lo, hi, 1) == [lo]`).
   Colpisce le regioni a bassa massa che ricevono 1 solo frame con
   `budget_allocation=proportional`. Tocca la *qualità* dello zoom (quale
   frame si vede) — cioè proprio ciò che la validazione misura — quindi
   fixarlo prima del run è consigliato (usare il frame centrale/di picco).
   Su Video-MME medium/long con 3 regioni su 128 frame (~40 frame/regione)
   si attiva di rado.
2. **Regione stretta + budget alto → indici duplicati → dedup collassa a
   meno frame di quelli allocati**: la risoluzione dello zoom è limitata
   dalla densità sorgente nella finestra. Inerente a video/finestre corti;
   su clip medium/long non morde. Da tenere a mente leggendo un eventuale
   `n_regions_used`/conteggio frame più basso dell'atteso.
3. **`budget_allocation="equal"` con `nframes < n_regions`** scarta le
   regioni per indice, non per massa (lieve scostamento dal piano §3.2.3,
   "massa più alta"). Caso praticamente irraggiungibile (nframes=128 ≫ 3).
4. **`_ranked_cells` calcolato due volte** per i sample multi-regione (una
   in `_peak_cell` in cima ad `answer`, una nel ramo zoom): `sink_view +
   rank_cells` non è gratis, si può passare la lista già calcolata. *(perf)*
5. **`decord.VideoReader` aperto `m+1` volte** in `_build_multi_region_media`
   (una diretta + una per regione dentro `reconstruct_frame_indices`).
   *(perf)*
6. **Piccola finestra di leak della temp dir** se `_build_multi_region_media`
   solleva un'eccezione dopo `mkdtemp` ma prima del `return` (la dir non è
   ancora assegnata alla variabile di cleanup del `finally`). *(robustezza)*

### Follow-up ai finding (2026-07-15)

Applicati 1, 4, 5, 6 in `strategies/entropy_attention_resample.py` prima del
run di validazione:

- **#1** — `_build_multi_region_media` non passa più da
  `reconstruct_frame_indices`: calcola `lo`/`hi` inline e, per `n<=1`, prende
  il frame **centrale** della regione (`round((lo+hi)/2)`) invece del primo
  estremo.
- **#4** — `answer()` calcola `ranked_cells` una sola volta (rimossa
  `_peak_cell`, ora morta); `top1_pct`/`peak_cell` derivano da
  `ranked_cells[0]` e lo stesso `ranked_cells` viene passato a
  `_multi_region_spans` nel ramo `zoom_multi_region`.
- **#5** — conseguenza di #1: un solo `decord.VideoReader` aperto in
  `_build_multi_region_media`, riusato per tutte le regioni (fps/total letti
  una volta).
- **#6** — l'estrazione dei PNG in `_build_multi_region_media` è in
  `try/except`: un'eccezione a metà estrazione ripulisce la `tmp_dir` prima
  di propagare, invece di lasciarla orfana.

**#2 e #3** lasciati come da nota del revisore: edge case pre-esistenti,
non toccati da questo cambio, praticamente non raggiungibili sui budget/
config di validazione previsti (§4).

Non rieseguibile il test standalone sulle funzioni pure (nessun torch/decord
in questo ambiente, invariato rispetto al passaggio precedente); verificata a
mano la formula `lo`/`hi`/midpoint in isolamento (senza torch) e per
ispezione che l'unica altra fonte di `reconstruct_frame_indices` (pass-1
`zoom_peak`/`phase_shift`, via `_build_media`) resta invariata — il ramo
multi-regione era l'unico chiamante toccato dal fix.
