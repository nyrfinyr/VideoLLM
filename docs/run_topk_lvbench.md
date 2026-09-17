# Run `topk_resample` su LVBench — specifica, requisiti, attese

> Documento di riferimento per la run dell'arm `topk_resample` (Qwen3-VL-2B,
> LVBench integrale). Dice **a cosa serve l'esperimento**, **cosa ci
> aspettiamo di trovare**, **quali requisiti sono implementati e dove**, e
> **cosa significa ogni parametro degli sbatch**.
>
> Stato al 2026-09-16: codice scritto e verificato con un modello finto, run
> **mai lanciata**. Le attese qui sotto sono predizioni, non risultati.

> ⚠️ Il ranking d'attenzione usato qui è quello GREZZO, e non è neutro: metà
> del budget di `k1`/`k3` finisce sulle celle di bordo. Caratterizzazione
> completa e motivo per cui si parte comunque così in
> [`bias_posizionale_attenzione.md`](bias_posizionale_attenzione.md).

Codice: [`strategies/topk_resample.py`](../strategies/topk_resample.py),
[`utils/pair_sampling.py`](../utils/pair_sampling.py),
[`evals/base.py`](../evals/base.py),
[`scripts/analyze_topk_run.py`](../scripts/analyze_topk_run.py),
[`scripts/sink_heatmaps.py`](../scripts/sink_heatmaps.py),
preset `strategy.topk_resample` in [`conf/config.yaml`](../conf/config.yaml),
lanci in [`scripts/sbatch/`](../scripts/sbatch/).

---

## 1. Perché questo esperimento

Tre risultati precedenti definiscono il problema.

1. **L'oracolo LVBench** (`docs/oracolo_lvbench.md`, job 94348, 100 sample):
   marcare la cella giusta non paga **mai** (oracolo 30% contro baseline 32%),
   mentre **ricampionare dentro la finestra annotata vale +16 pp** (48% contro
   32%, McNemar +22/−6, p≈0.004). Tutti i 22 recuperi vengono da sample in cui
   il campione uniforme a 24 frame non conteneva **nessun** frame della
   finestra. Il guadagno sta nel *vedere* l'evidenza, non nell'indicarla.
2. **Il collo di bottiglia è la localizzazione**: con 24 frame il puntatore
   d'attenzione trovava la finestra nel 31% dei casi contro un 18.8% di
   chance (~1.6x) e copriva solo 4 dei 22 recuperi.
3. **A 256 celle campionate a coppie il segnale c'è** (run di segnali 107727,
   100 sample, 69 usabili): AUC 0.775, hit@1 17% contro 1.6% di chance
   (binomiale p=7.8e-10), hit@5 38%, hit@10 52%, rango mediano della prima
   cella vera 10/256. Controlli puliti: predittore di sola posizione AUC
   0.552, permutazione appaiata AUC 0.518±0.025.

La domanda aperta è quindi una sola: **quel segnale, usato davvero per
ricampionare, guadagna accuracy?** E subito dopo: con quale k, con quale
gate, e i famigerati "sink" c'entrano qualcosa?

## 2. Che cosa fa la run, in una riga

Per ogni domanda: un **pass 1** a 512 frame campionati a coppie (256 celle) che
produce la risposta di baseline, l'entropia e il ranking delle celle; poi
**cinque pass 2** da 128 frame ciascuno, che ricampionano dentro celle scelte
in modo diverso, e rispondono di nuovo. Tutto appaiato sullo stesso sample.

## 3. Le domande, e cosa ci aspettiamo

Ogni attesa è scritta in modo **falsificabile**: se il numero esce diverso,
l'ipotesi corrispondente è sbagliata (non il codice).

| # | Domanda | Attesa | Come si legge |
|---|---|---|---|
| D1 | Ricampionare sulle top-k celle batte la baseline? | Δ **positivo ma piccolo**, indicativamente +2…+6 pp. Il tetto dell'oracolo è +16 pp e lo si raggiungerebbe solo colpendo sempre la finestra; con hit@k fra 17% e 52% il guadagno atteso è una frazione di quel tetto, ridotta dal costo dei sample rotti | `analyze_topk_run.py` → sezione CONDIZIONI, colonna Δ + McNemar |
| D2 | Quale k? | **Non lo sappiamo**: è l'oggetto dell'ablation. Due forze opposte — k grande copre più finestre (hit@1 17% → hit@10 52%) ma spende 9/10 del budget in posti sbagliati e perde la vista globale. Se dovessi scommettere: `k3`/`hybrid3` migliori di `k1` e `k10` | stessa tabella, righe `k1`/`k3`/`k10`/`hybrid3` |
| D3 | Il guadagno viene dal segnale o dal solo ridistribuire il budget? | `rand3` **non deve guadagnare**: atteso Δ ≈ 0 o negativo. Se `rand3` va come `k3`, il segnale non c'entra e l'arm è solo "zoom" | riga `rand3`; e `hit@k celle` deve essere ~2% per `rand3` contro ~29% per `k3` |
| D4 | Il gate d'entropia serve su LVBench? | **Probabilmente no**: sulla probe AUROC(H) per l'errore è 0.547 (su Video-MME era 0.756). Atteso: "gate OOF" ≈ "sempre" e quantili scelti sparsi fra i fold | sezione GATE, colonne `sempre` / `gate OOF` / `quantili scelti` |
| D5 | T1 regge sul fullset? | Sì, con intervalli più stretti: AUC ~0.75-0.80, hit@1 ~15-20%. Se crolla, i 69 sample della probe erano fortuna | sezione T1 |
| D6 | I "sink" esistono in questo setting? | **Sì come canali, no come pozzi d'attenzione**: dim 1999 e 1793 restano canali outlier estremi (rango 0-1, |h| 20-60x la media), ma la curva di massa resta piatta (~p%) e i patch "sink" non catturano attenzione. Atteso anche nel pass 2 a 128 frame: è una proprietà del modello, non del campionamento | sezione SINK (prove 1-3) + heatmap sui frame |
| D7 | Il ranking temporale è solo la mappa dei sink? | No: `attn_sink_cell_corr` atteso **basso** (|r| < 0.3). Se fosse alto, il risultato T1 sarebbe un artefatto | sezione SINK, riga correlazione |
| D8 | L'attenzione guarda contenuto o bordo? | `attn_border_share` atteso **vicino o sotto** la quota uniforme (0.53 su griglia 5x9); `sink_border_share` idem — sulla probe i sink non erano un fenomeno di bordo. Le heatmap sui frame dicono se i patch caldi stanno su oggetti o su sfondo | sezione SINK + `scripts/sink_heatmaps.py` |

**Cosa NON risponde questa run**: se convenga iterare lo zoom (serve un pass 3),
se un gate diverso dall'entropia funzioni (i candidati misurati sulla probe
erano tutti rumore con 12 hit), e se il segnale regga su altri modelli.

## 4. Requisiti implementati

Ogni requisito ha il punto del codice che lo soddisfa. Sono la lista che un
revisore deve poter spuntare leggendo i sorgenti.

### 4.1 Disegno sperimentale

| ID | Requisito | Dove |
|---|---|---|
| R1 | Il pass 1 è la **baseline appaiata**: stessa domanda, stessi frame, stessa lettura della risposta (argmax del prefill sulle lettere MCQ, nessuna generazione). `pred` dell'output = risposta del pass 1, quindi `mcq_accuracy` della run **è** la baseline | `topk_resample.py::_answer_from_pass1` (`"pred": pred1`) |
| R2 | **Nessun gate online**: il pass 2 gira su tutti i sample, sempre | `_answer_from_pass1`, ciclo su `self.conditions` senza condizioni d'ingresso |
| R3 | Tutte le condizioni girano **sugli stessi sample** (ablation appaiata, non split del dataset) | stesso ciclo; nessun partizionamento del dataset in nessun punto |
| R4 | Controllo **random** con stesso k e stesso budget dell'arm: cambia solo *quale* cella si infittisce | `select_cells(..., kind="random")`; preset `rand3` |
| R5 | Il seed del random è **riproducibile e per-sample** (hash di video+prompt), indipendente dall'ordine degli shard | `_answer_from_pass1`, `rng = random.Random(f"{seed}|{sha1(video|prompt)}")` |
| R6 | Larghezza `w=0` (solo le celle del ranking, niente allargamento attorno al picco) | `cells_to_spans(cells, ...)` riceve le celle scelte, nessun margine |
| R7 | Avviso esplicito se manca una condizione di controllo | `__init__`, `logger.warning` quando nessuna condizione ha `select=random` |

### 4.2 Campionamento e fedeltà dei due pass

| ID | Requisito | Dove |
|---|---|---|
| R8 | Pass 1 a **coppie**: 256 centri uniformi, due frame a ±`pair_gap_sec/2`; una cella d'attenzione = un istante | `pair_video_frames` / `pair_centers_and_indices` |
| R9 | Pass 2 **anche a coppie**, ordinate per centro: nessuna cella a cavallo di due regioni (avrebbe un timestamp dentro un buco mai osservato) | `pairs_in_spans` (ordina i centri prima di formare le coppie) |
| R10 | Le celle scelte diventano l'intervallo di tempo che **possiedono** (cella di Voronoi `[D·i/n, D·(i+1)/n]`) | `cells_to_spans` |
| R11 | Budget del pass 2 diviso in parti uguali fra le regioni, resto alle celle col ranking più alto | `allocate_pairs` |
| R12 | `global_fraction` lascia una quota di budget uniforme su **tutto** il video (degrado gentile) | `_run_condition`, span aggiuntivo `(0, duration)`; preset `hybrid3` |
| R13 | I frame del pass 2 portano `frames_indices`/`fps` **reali**: i timestamp nel prompt sono i tempi veri, non una densità finta via `sample_fps` | `span_video_frames` → `VideoFrames(frames_indices=..., fps=...)` |
| R14 | Pass 1 e pass 2 alla **stessa risoluzione per frame**: la strategy rifiuta di partire senza `model.fix_videoframes_resize=true` | `answer`, `RuntimeError` su `fix_videoframes_resize` falso |
| R15 | Nessun trim della finestra annotata: la finestra è il metro della misura, non un input | `answer`, `RuntimeError` se `video_start`/`video_end` non sono `None` |
| R16 | Guardie sulla geometria: `nframes` pari, `t_cells` == coppie campionate in entrambi i pass, `k` campionabile col budget dato | `answer`, `_pass1`, `_run_condition`, `__init__` |
| R17 | Estrazione frame efficiente: lettura in blocco (`VideoReader.get_batch`) e PNG scritti **già** alla dimensione finale | `utils/pair_sampling.py::extract_frames` + `models/qwen.py::videoframes_target_size` |
| R18 | Cache dei frame del pass 1 per video (LVBench emette ~15 domande consecutive sullo stesso video) | `pair_video_frames`, `_CACHE` |

### 4.3 Dati loggati — per sample, su Weave

| ID | Requisito | Campi |
|---|---|---|
| R19 | Risposta e confidenza del pass 1 | `raw`, `pred`, `answer_entropy`, `answer_probs`, `pred_fallback` |
| R20 | Geometria e contesto | `t_cells`, `grid_h`, `grid_w`, `n_vis`, `seq_len`, `video_duration_sec`, `pair_gap_sec`, `pair_centers_sec` (256 centri), `rank_rowset`, `rank_sink_filtered`, `pass2_nframes`, `pass2_pair_gap_sec` |
| R21 | Segnale T1 per rowset (`all`, `question` di default) | `rowsets.<rs>.cell_mass_raw`, `.cell_mass_sink_filtered`, `.visual_mass_total` |
| R22 | Esito di ogni condizione | `conditions.<nome>` con `cells`, `k`, `select`, `global_fraction`, `n_pairs_zoom`, `n_pairs_global`, `span_sec`, `raw`, `pred`, `answer_entropy`, `answer_probs`, `centers_first`, `centers_last`, `t_cells`, `grid_h`, `grid_w` |
| R23 | Scorciatoie per lo scorer e per il gate | `preds_by_condition`, `entropy_by_condition` |
| R24 | `question_type` ri-emesso (breakdown LVBench) | `evals/lvbench.py::predict_factory` |

### 4.4 Dati loggati — i sink

| ID | Requisito | Campi / dove |
|---|---|---|
| R25 | **Dove dentro il frame** guarda l'attenzione, per rowset | `rowsets.<rs>.attn_spatial_mean` `[gh, gw]` |
| R26 | **Dove** stanno i patch sink | `sink_spatial_mean` `[gh, gw]`, `sink_temporal_mean` `[t]` |
| R27 | Quota di massa sull'anello di bordo, con riferimento uniforme | `rowsets.<rs>.attn_border_share`, `sink_border_share`, `border_share_uniform` |
| R28 | **Prova decisiva**: quota di massa d'attenzione sui top-p% token per sink score, p ∈ {1,2,5,10,25,50} | `rowsets.<rs>.sink_mass_curve`, `sink_mass_curve_pcts` |
| R29 | Il ranking temporale è la mappa dei sink riscritta? | `attn_sink_cell_corr` (Pearson per cella) |
| R30 | **Valori negli hidden state**: rango dei sink dims fra i 2048 canali, istogrammi di \|h\|/mediana contro canali di controllo a seed fisso, split sink/non-sink, top-k canali — su **tutti i 28 layer** | `sink_stats` (da `models/qwen_attn.py::summarize_sink_stats`), un sample ogni `sink_stats_every` |
| R31 | Le stesse misure nel **regime del pass 2** (128 frame densi) | `conditions.<nome>.sink_stats` + `sink_mass_curve` + `attn_spatial_mean` + `sink_spatial_mean` + `attn_sink_cell_corr`, per le condizioni in `sink_stats_conditions` |
| R32 | **Heatmap sui frame veri**: dump con heatmap intere `[t,gh,gw]` per rowset, `sink_map` intera, statistiche per-token, e i PNG dei frame delle celle **calde** (quelle scelte) più altrettante **fredde** di controllo | `topk_resample.py::_dump` → `<dump_dir>/<video>_<hash>/dump.pt` + `cell<NNNN>_{a,b}.png` |
| R33 | Rendering delle heatmap sulle immagini, con scala di colore **normalizzata sull'intero sample** (p99) così calde e fredde sono confrontabili | `scripts/sink_heatmaps.py` (solo PIL: matplotlib non è fra le dipendenze) |

### 4.5 Aggregazione e analisi

| ID | Requisito | Dove |
|---|---|---|
| R34 | Accuracy per condizione già nel summary wandb, sommabile fra shard | `evals/base.py::mcq_accuracy` → `correct_cond_<nome>` / `seen_cond_<nome>` |
| R35 | `r_fix` e `r_break` per condizione, **intra-sample** (split per correttezza della baseline) | stessi, `_base_true` / `_base_false` |
| R36 | Una condizione senza lettera valida viene **saltata**, non contata come errore | `mcq_accuracy`, `if cond_pred is None: continue` |
| R37 | T1 offline con i due controlli (posizione sola, permutazione appaiata) e significatività | `analyze_topk_run.py::analyze_t1` |
| R38 | Confronto appaiato condizione-vs-baseline con **McNemar esatto**, più hit@k delle celle ricampionate contro la finestra vera | `analyze_conditions` |
| R39 | Scelta del gate **out-of-fold**, con fold **raggruppati per video** | `folds_by_video`, `analyze_gate` |
| R40 | Riferimenti obbligati "riapri sempre" / "non riaprire mai" e dispersione del quantile fra i fold | `analyze_gate`, colonne `sempre`, `mai`, `quantili scelti` |
| R41 | Le quattro prove sui sink aggregate su tutti i sample, e ripetute sul pass 2 | `analyze_sinks`, `sink_conditions` |
| R42 | Gestione delle righe LVBench con `time_reference` **invertito** (start > end: 4 su 100 nella probe) e delle finestre degeneri (nessuna cella dentro, o >10% del video) | `analyze_topk_run.py::label_cells` |

## 5. I parametri degli sbatch, uno per uno

Due file: `scripts/sbatch/lvbench_topk_probe.sbatch` (100 sample, da lanciare
**prima**) e `scripts/sbatch/lvbench_topk.sbatch` (1549 QA).

### 5.1 Direttive SLURM

| Direttiva | Probe | Fullset | Perché |
|---|---|---|---|
| `--gres=gpu:1` | 1 GPU | 1 GPU | Un modello da 2B, un forward per volta: niente da parallelizzare dentro il job |
| `--constraint` | `A40_45G\|L40S_45G\|RTX_A5000_24G` | uguale | **Esclude le Turing**: su RTX 6000 Qwen3-VL è 6x più lento (causa accertata del disastro di luglio). Le 24G restano ammesse perché il picco misurato a 512 frame con cattura è 6.8 GB |
| `--mem=32G` | 32G | 32G | La RAM la consumano i frame decodificati, non il modello. Stesso valore della baseline LVBench a 768 frame |
| `--cpus-per-task=8` | 8 | 8 | La decodifica video è il collo di bottiglia del costo per sample |
| `--array` | `0` | `0-7` | Probe: shard unico, così i 100 sample sono i primi 100 della permutazione e coincidono con quelli della probe dei segnali. Fullset: 8 shard strided (`samples[shard::8]`) da ~194 sample |
| `--time` | `08:00:00` | `12:00:00` | Probe: 100 × ~150 s ≈ 4.2 h, il doppio di margine. Fullset: 194 × 160 s ≈ 8.6 h, regge fino a ~223 s/sample. **Entrambi vanno ritarati sul `model_latency_mean` della probe** |
| `--output` / `--error` | `logs/%x-%A_%a` | uguale | Un file per task dell'array |

### 5.2 Variabili d'ambiente dello script

| Variabile | Default | Effetto |
|---|---|---|
| `LIMIT` (solo probe) | 100 | Quanti sample. `LIMIT=30` per una risposta rapida sul costo |
| `PASS2` | 128 | `strategy.pass2_nframes`: budget di **ogni** pass 2. Deve essere pari (= 2 × coppie) |
| `SINK_EVERY` (solo fullset) | 4 | `strategy.sink_stats_every`: una raccolta di statistiche dei canali ogni N sample. Le statistiche pesano ~40 KB per sample su Weave (≈60 MB sul fullset a `every=1`) e servono **aggregate**, quindi una frazione basta. La probe usa 1 |
| `DUMP_LIMIT` | 10 (probe) / da `DUMP_TOTAL` | Quanti sample dumpare **per processo**: con N shard il totale è N volte tanto |
| `DUMP_TOTAL` (solo fullset) | 40 | Dump complessivi voluti; `DUMP_LIMIT` = ceil(DUMP_TOTAL / shard) |
| `DUMP_ROOT` | `/work/tesi_avalenza/topk_dumps` | Radice dei dump. Sul fullset una sottocartella per shard |

### 5.3 Override passati a `main.py`

| Override | Valore | Perché |
|---|---|---|
| `model=qwen3_vl_2b_attn` | — | Serve il preset con cattura d'attenzione: la strategy richiede `full_visual_attention` |
| `model.fix_videoframes_resize=true` | — | **Obbligatorio**. Senza, `qwen-vl-utils` arrotonda le liste di frame a multipli di 64 e il pass 2 girerebbe a risoluzione più bassa del pass 1: il delta mescolerebbe intervento e perdita di token. La strategy solleva se è falso |
| `dataset=lvbench` | — | L'unico dataset con finestre di evidenza annotate |
| `dataset.nframes=512` | 512 | 256 celle × 2 frame, ~11 520 token visivi. A questa risoluzione il tetto del processor **non morde**: `videoframes_target_size` dà 160x288 (5x9) sia a 512 sia a 128 frame, perché `max_pixels=50176` è ben sotto il cap per frame (460 800 px a 512 frame). È quello che rende i due pass confrontabili (R14) |
| `dataset.max_pixels=50176` | 50176 | 45 token per cella → griglia 5x9 su 16:9, ~11 520 token visivi |
| `dataset.min_pixels=3136` | 3136 | Va passato esplicito: il floor video di `qwen-vl-utils` (128 token/frame) altrimenti interferisce col `max_pixels` scelto |
| `strategy=topk_resample` | — | Il preset con le 5 condizioni |
| `strategy.pass2_nframes` | `$PASS2` | Vedi sopra |
| `strategy.sink_stats_every` | 1 / `$SINK_EVERY` | Vedi sopra |
| `strategy.dump_dir`, `strategy.dump_limit` | — | Attivano i dump con heatmap e frame |
| `hf_home=/work/tesi_avalenza/hf` | — | Cache HF personale sul filesystem di lavoro |
| `shard`, `num_shards` | dall'array | Slice strided del dataset |
| `limit`, `shuffle` (solo probe) | 100, true | Lo shuffle è essenziale: il loader emette le domande raggruppate per video, i primi 100 non mescolati sarebbero ~7 video |
| `wandb.group`, `wandb.name`, `wandb.tags` | — | `lvbench-topk-test` per la probe, `lvbench-topk` per il fullset |

### 5.4 Le condizioni (preset `conf/config.yaml`)

| Nome | k | select | global_fraction | Che domanda risponde |
|---|---|---|---|---|
| `k1` | 1 | attention | 0 | Tutto il budget sulla cella di picco: massima densità, copertura minima (hit@1 17%) |
| `k3` | 3 | attention | 0 | Compromesso (hit@3 ≈ 29%) |
| `k10` | 10 | attention | 0 | Massima copertura (hit@10 52%), ma 9/10 del budget in posti sbagliati |
| `hybrid3` | 3 | attention | 0.5 | Metà budget resta globale: degrado gentile se lo zoom cade male |
| `rand3` | 3 | random | 0 | **Controllo**: stesso k, stesso budget, celle a caso |

## 6. Come si legge il risultato

```bash
# 1. correttezza e costo (il driver non ha check specifici per questo arm:
#    la checklist è in testa allo sbatch della probe)
uv run python .claude/skills/wandb/driver.py qwen3_vl_2b_attn-topk-<jobid>_0_probe100 --shard-size 194

# 2. analisi completa (T1, sink, condizioni, gate out-of-fold)
uv run python scripts/analyze_topk_run.py qwen3_vl_2b_attn-topk-<jobid>_0_probe100 --cache /tmp/topk.json

# 3. heatmap dipinte sui frame veri
uv run python scripts/sink_heatmaps.py /work/tesi_avalenza/topk_dumps/<run> --all
```

Ordine di lettura obbligato: **prima** la correttezza (t_cells=256, griglia
5x9 o 6x8, `cells` lunga k, budget del pass 2 esatto, `rand3` che sceglie celle
diverse da `k3`), **poi** il costo, **solo alla fine** i delta. Su 100 sample
nessun delta è significativo: la probe serve a vedere che nulla crolli.

## 7. Limiti noti e cose che il documento NON promette

1. **La strategy non ha mai girato col processor vero.** È stata verificata
   end-to-end con un VLM finto e un video sintetico: coppie, regioni, budget,
   dump e frame salvati sono corretti; la geometria reale del processor no. È
   il motivo per cui la probe esiste.
2. **Il costo per sample è stimato, non misurato**: 135-160 s/sample. Il pass 1
   costa 57.7 s misurati (job 107727); i cinque pass 2 sono stimati 15-20 s
   l'uno, dominati dall'estrazione dei frame. Il dimensionamento del fullset
   dipende da questo numero.
3. **I dump del fullset coprono pochi video**: sono i primi sample di ogni
   shard e il loader non mescola. Per un campione vario servono i dump della
   probe, che è mescolata.
4. **`sink_stats` è campionato** (`sink_stats_every`): gli aggregati restano
   validi (istogrammi e quantili si sommano), ma non c'è una statistica dei
   canali per ogni sample.
5. **Il gate non è applicato dal codice**: la run non gatta nulla, produce i
   dati perché la soglia si scelga offline. Nessun numero di questa run è "la
   accuracy dell'arm con gate" finché non lo si calcola out-of-fold.
6. **La lettura della risposta è argmax del prefill**, non generazione libera
   con parsing: confrontabile con la probe dei segnali (43% di baseline), non
   con le tabelle di agosto ottenute per generazione.
7. **Nessun test automatico** nel repo: il progetto non ha suite di test, la
   verifica è per probe e per script di analisi.
