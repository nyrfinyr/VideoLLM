# Commenti e analisi — Qwen3-VL su Video-MME e LVBench

> Companion di `260901-tabelle-risultati.md`, che contiene **solo** tabelle e
> diagrammi (documento da presentazione). Qui stanno le letture, i caveat
> statistici, le note di metodo e i ragionamenti che portano da un numero al
> successivo. Ordine delle sezioni allineato a quello delle tabelle.

## 0. La domanda a cui tutto risponde

LoT (*Look Twice*) ha due metà: decidere **SE** intervenire (gate d'entropia)
e decidere **COME/DOVE** (segnale d'attenzione). Nel dominio video la seconda
metà si può realizzare in due modi opposti:

- **marcare** — dire al modello dove guardare, lasciando invariati i frame;
- **ricampionare** — cambiare i frame che il modello vede.

Tutta la sequenza di esperimenti serve a capire quale delle due paga, e
perché l'altra no. La risposta, alla fine: **marcare non paga mai;
ricampionare ha un soffitto di +16 pp**, e il collo di bottiglia è
localizzare la finestra d'evidenza.

## 1. Arm su Video-MME (2700 sample)

### 1.1 `entropy_shift_24`: +0.56 pp, e perché non è un risultato

+0.56 pp sono **15 sample su 2700**, e la N effettiva è 1450, non 2700: sui
1250 sample col gate chiuso l'arm restituisce la risposta del pass 1, cioè
quella della baseline. Il z non appaiato vale +0.41 (p ≈ 0.68). Il test
appaiato sui soli 1450 ricampionati richiederebbe le righe Weave, non i
summary.

Il salto **37.86% → 74.88%** fra gate aperto e chiuso è **effetto di
selezione**, non efficacia: il gate manda al pass 2 i sample
intrinsecamente più difficili. Stessa lettura di `260803 §1`.

### 1.2 I `marker_*`: il controllo pareggia il treatment

`marker_24` − `marker_random_24` = **−2 sample su 2700** (−0.07 pp). Non è
mancata applicazione: `marked` 900/900 su tutte e sei le eval, e le celle
scelte dai due criteri differiscono nel 35-40% dei sample. **Marcare il
frame "giusto" non vale più che marcarne uno a caso.**

Entrambi stanno 1.3-1.4 pp sotto la baseline. La SE non appaiata su una
differenza fra proporzioni a n=2700 vale ≈1.36 pp, quindi il calo è ~1 SE:
suggestivo, non stabilito. Il test appaiato contro `baseline_24` non è
fattibile — `uniform` non ri-emette campi per-sample, quindi le sue
Evaluation Weave non sono attribuibili ai singoli shard.

Costo ~3.2× la baseline (10.7 contro 3.1-3.4 s/sample): con
`always_highlight=true` ogni sample paga estrazione PNG dei 24 frame +
secondo pass, nessuno è esente.

Correttezza verificata sulle 6 Evaluation Weave (5400 sample): `marked`
900/900, `n_query_rows < n_query_tokens` 900/900, blocchi pari con
`somma(block_sizes)/2 == t_cells` 900/900, `pred_fallback` 4/5400 (0.07%).

### 1.3 `visual_prompt_*`: la replica con il segnale migliore

`visual_prompt_24` − `visual_prompt_24_random` = **1 sample su 2700**
(−0.04 pp, z = −0.03). Il terzo canale di consegna replica esattamente
l'esito del secondo (i marcatori differivano di 2 sample) — e stavolta con
`query_rows=entity` e `sink_filter=true`, cioè **col segnale più raffinato
di cui disponiamo**, non con quello grezzo.

Il breakdown `correct_pass1_*` mostra la stessa forma di ridistribuzione dei
marcatori:

| | rotti (di 1476-77 corretti al pass 1) | recuperati (di 1223 sbagliati) | netto |
|---|---:|---:|---:|
| `visual_prompt_24` | 156 (10.6%) | 123 (10.1%) | −33 |
| `visual_prompt_24_random` | 151 (10.2%) | 119 (9.7%) | −32 |

Due processi indistinguibili: l'intervento muove ~10% delle risposte in
entrambe le direzioni **indipendentemente da dove viene applicato**.

**Nota di rumore utile come metro.** `n_samples_correct_pass1_true` vale
1476 per l'arm e 1477 per il controllo, benché il pass 1 sia identico nei due
(stesso prompt, stessi frame, greedy). Quella differenza di 1 sample è
non-determinismo di GPU/kernel — **ed è lo stesso ordine di grandezza
dell'effetto che stiamo misurando**.

**L'effetto entity è misurato diluito**: sullo shard 0 (900 sample) 566
girano davvero a righe-entità (63%), 166 escono `NONE` deliberato (18%), 168
non mappano verbatim (19%) — quei 334 girano di fatto a `question`. Il
confronto interno attention/random non ne soffre: entrambi i rami usano lo
stesso mix.

### 1.4 I due assi del segnale: esauriti

Gli assi `query_rows` e `sink_filter` sono stati variati alle due estremità:
i `marker_*` sul segnale grezzo (`question`, `sink_filter=false`), i
`visual_prompt_*` su quello raffinato (`entity`, `sink_filter=true`).
**Entrambe le famiglie pareggiano il proprio controllo random.** Raffinare il
segnale ha spostato la distanza treatment-controllo da 2 sample a 1.

Non c'è una versione più affilata di `query_rows` che valga la pena provare:
l'oracolo su LVBench mostra che marcare la cella **vera** vale 30% contro 32%
della baseline. Anche `sink_filter` non discrimina — cambia la cella scelta
nel 33-40% dei sample senza muovere l'accuracy.

La differenza fra le due famiglie sul pooled (53.04 vs 53.48) **non** è
attribuibile al segnale: cambia anche il canale (token vs pixel) e la
geometria dell'overlay, ed è dentro 1 SE.

### 1.5 Caveat statistico sui Δ per fascia

Con n=900 per fascia la SE di una differenza è **±2.35 pp**: nessun Δ di
fascia nella tabella §1 è distinguibile da zero. Il −2.78 pp dei `marker_24`
sui `medium` è il più grande, ma vale ~1.2 SE e ha il gemello random a
−2.44 — non è un effetto del segnale. I Δ per fascia servono a leggere
tendenze, non come risultati.

## 2. Trasferimento fra modelli

**Il Δ di `entropy_shift` è replicato quasi esattamente** (+0.59 su
Qwen2.5-VL-3B → +0.56 su Qwen3-VL-2B): il nulla misurato su Qwen2.5 non era
una proprietà di quel modello.

**Il gate non si trasferisce.** A soglia invariata si apre sul 53.70% dei
sample invece che sul 73.15%, e separa meno (37.86/74.88 contro 43.59/88.41).
I due arm **non sono compute-matched fra modelli**: per appaiare la frazione
ricampionata la soglia va ricalibrata su Qwen3, non riusata.

**Il costo relativo è più alto nonostante meno sample al pass 2.** Latenze a
parità di GPU (solo shard L40S): `6.36 = 3.77 + 0.7315·x → x = 3.54 s` contro
`8.97 = 3.32 + 0.5370·x → x = 10.5 s`, cioè **~3× per sample ricampionato**.
Causa non misurata (cattura d'attenzione o estrazione PNG del ramo
`phase_shift`): da profilare prima di altri arm signal-driven su Qwen3.

### 2.1 Breakdown per task type

Il segno concorda su **6 categorie su 12**: appena sopra il caso. Il Δ pooled
quasi identico fra i due modelli non nasce dallo stesso pattern per task, ma
da compensazioni diverse.

`temporal perception` è l'unica categoria che perde su entrambi, con lo
stesso Δ (−7.27 pp, −4 sample): **la categoria che dovrebbe beneficiare di
più del ricampionamento temporale è quella che peggiora di più.** Su Qwen2.5
era 0 arm su 9 sopra la baseline (`260803 §1b`), qui il segno si ripete.

`counting problem` e `ocr problems` si invertono fra i due modelli (+4.10 →
−1.12 e −1.44 → +4.32): nessuna delle regolarità osservate su Qwen2.5
sopravvive al cambio di modello.

⚠️ Cinque categorie stanno sotto i 180 sample, dove 2 pp è un sample solo:
conta il segno, non l'ampiezza.

## 3. La scala del modello

**+4.41 pp = +119 sample su 2700**, z non appaiato = 3.27 (p ≈ 0.0011). È il
primo Δ del filone che sta chiaramente fuori dal rumore: per confronto, tutti
gli arm di §1 si muovono entro ±1.4 pp, cioè ~1 SE.

**Il guadagno cresce monotonamente con la durata** (+2.11 short, +4.44
medium, +6.67 long). I `long` erano il punto debole del 2B (44.11%, 22.8 pp
sotto gli `short`); il 4B ne recupera 60, e la forbice short−long si stringe
da 22.8 a 18.2 pp. Metà del guadagno pooled viene da lì.

**Il 4B è quasi gratis in latenza**: 3.54 contro 3.32 s/sample a parità di
GPU, +6.6%. Il forward è dominato dai token visivi (24 frame), non dai
parametri del decoder.

**Conseguenza operativa:** +4.41 pp per +6.6% di tempo è un rapporto che
nessun arm testato si avvicina a produrre — `entropy_shift_24` paga 2.70× il
tempo per +0.56 pp, i `marker_*` 3.2× per −1.4 pp. Misurare interventi sul 2B
rischia di ottimizzare margini che la scala assorbe.

## 4. L'oracolo su LVBench: la misura che partiziona

Dettaglio completo in `oracolo_lvbench.md`. Qui il ragionamento.

### 4.1 Perché serviva

I null di Video-MME ammettevano due spiegazioni incompatibili che l'accuracy
non separa: **(a)** il segnale è sbagliato, **(b)** non c'era niente da
guadagnare comunque. LVBench annota la finestra dell'evidenza, quindi
permette di smettere di misurare *attraverso* l'accuracy.

### 4.2 Cosa dicono le tre misure

1. **Il picco porta un segnale reale ma debole**: cade nella finestra
   annotata il 28-31% delle volte contro un caso del 15.8-18.8% (z = 4.21 sul
   campione da 100). E i tre rowset sono equivalenti — `entity` 30.0%,
   `question` 28.3%, `all` 28.3%: affilare le righe non sposta il picco.
2. **Il canale di marcatura non ha soffitto**: puntatore alla cella vera 30%
   contro 32% di baseline (+5/−7, p = 0.77). Nemmeno l'informazione perfetta
   paga.
3. **Il ricampionamento sì**: 24 frame dentro la finestra vera danno 48%
   contro 32% (+22/−6, p = 0.0037). Il confronto diretto fra le due metà,
   appaiato, è `zoom` vs `oracle`: **+23/−5, p = 0.0009**.

### 4.3 Il meccanismo

Il guadagno di `zoom` è concentrato dove la teoria lo prevede:

| | n | baseline | zoom |
|---|---:|---:|---:|
| finestra con **< 1** frame atteso nel campione uniforme | 77 | 27% | **49%** |
| finestra con **≥ 1** frame atteso | 23 | 48% | 43% |

Tutti e 22 i sample recuperati stanno nel primo gruppo. Dove il campione
uniforme già colpiva l'evidenza, ricampionare non aggiunge nulla (anzi,
−5 pp: si perde il contesto).

Il regime lo spiega: durata mediana **71 minuti**, finestra annotata mediana
**30 secondi** (0.8% del video). Con 24 frame uniformi il numero atteso di
frame dentro la finestra è **0.20**; in 77 casi su 100 è sotto 1.

**La lettura che conta:** il campionamento uniforme non è cieco perché
sceglie male i frame che mostra, ma perché **l'evidenza non è tra i frame che
mostra affatto**. Un puntatore testuale può solo dire al modello di guardare
meglio pixel che non contengono la risposta — ed è il motivo per cui tre
canali di consegna diversi hanno dato lo stesso nulla.

### 4.4 Come si aggiorna la lettura di `antipeak`

La sonda `antipeak` (`sonda_sensibilita_puntatore_antipeak.md`) aveva
stabilito che il canale-prompt è **vivo** (−6.46 pp, p < 0.0001) e concluso
che il nulla andava attribuito al segnale. Le due conclusioni non si
contraddicono, ma la dicotomia era incompleta: esiste un mondo (C) in cui il
puntatore viene letto **e** il picco porta segnale, ma indicare la cella vera
non cambia comunque l'accuracy. `antipeak` misura che un'istruzione
*sbagliata* fa danno, non che una *giusta* faccia bene.

Va corretta anche la lettura "picco ≈ cella a caso": sembrava tale perché
misurata attraverso l'accuracy, che qui sappiamo essere lo strumento
sbagliato. Il picco è ~2× il caso.

## 5. La direzione

I numeri indicano tre cose concordanti:

1. **Ricampionare, non marcare.** `entropy_shift_24` è l'unico arm sopra la
   baseline, ed è l'unica direzione con un soffitto misurato (+16 pp). I due
   arm che *comunicano* una cella sono entrambi sotto.
2. **Sui video lunghi.** `entropy_shift_24` fa +1.00 pp sui `long`; la
   baseline crolla da 66.89% (short) a 44.11% (long); il 4B recupera proprio
   lì (+6.67). È dove il campionamento uniforme perde l'evidenza.
3. **Il collo di bottiglia è la localizzazione.** Sui sample dove il
   ricampionamento d'oracolo recupera la risposta, il picco d'attenzione cade
   nella finestra giusta solo il **18%** delle volte. Un arm "ricampiona dove
   punta il picco" catturerebbe una piccola frazione dei 16 pp.

### 5.1 Come sfruttare il segnale per ricampionare

**Il vincolo che decide il disegno**: quando il picco sbaglia, *sbaglia di
molto*. Distanza fra la cella di picco e la finestra annotata (100 sample,
job 94348):

| distanza | 0 (dentro) | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| sample | 31 | 10 | 13 | 5 | 3 | 38 |

Oltre 1 cella la distribuzione è quasi piatta: non esiste un regime
"quasi giusto". Da qui la conseguenza controintuitiva — **allargare la
finestra di ricampionamento attorno al picco non paga**:

| finestra | % del video | copre la finestra | caso | lift | frame nell'evidenza |
|---|---:|---:|---:|---:|---:|
| ±0 (solo cella di picco) | 8% | 31% | 18% | **1.68×** | **2.4** |
| ±1 | 25% | 41% | 32% | 1.27× | 0.8 |
| ±2 | 42% | 54% | 45% | 1.21× | 0.5 |
| ±3 | 58% | 59% | 55% | 1.07× | — |

Si paga risoluzione (2.4 → 0.8 frame dentro l'evidenza) per comprare
copertura che è quasi tutta caso; a ±3 il lift è sparito. **L'ampiezza è
l'asse sbagliato.**

Restano due leve, entrambe nel prossimo giro della sonda:

1. **Ranking (top-k disgiunto)** — non allargare attorno al picco, ma
   dividere il budget fra le *k celle a punteggio più alto*, ovunque siano.
   Paga solo se il segnale ha massa sulle celle giuste anche quando il
   top-1 sbaglia: è la domanda **hit@k**, loggata gratis dal ranking. Se
   hit@3 ≈ hit@1, il ranking non porta ridondanza e la leva è morta.
2. **Degrado gentile (hybrid)** — metà budget globale + metà nella cella di
   picco. Serve perché lo zoom è un intervento **ad alto rischio**, a
   differenza della marcatura che era neutra: nei 23 sample dove il campione
   uniforme colpiva già l'evidenza, lo zoom d'oracolo *peggiora* (48% → 43%).
   Un picco sbagliato costa il contesto; l'ibrido lo conserva.

Il giro misura sette condizioni sugli stessi sample: le tre di marcatura
(riferimento storico), l'oracolo `zoom` (soffitto 48%), e le tre
realizzabili — `zoom_peak`, `zoom_topk`, `zoom_hybrid`. Lettura attesa:
`zoom_peak` cattura circa il 31% dei casi in cui il soffitto esiste, quindi
un guadagno lordo dell'ordine di ~5 pp **meno** il danno sui sample dove il
picco sbaglia — ed è esattamente quel bilancio che l'ibrido dovrebbe
raddrizzare.

### 5.2 Arm chiusi, e perché è un risultato

`attention_highlight`, `attention_marker`, `visual_prompt`: tre canali di
consegna indipendenti, ciascuno col proprio controllo random, tutti nulli. Il
negativo è forte perché ha un **controllo d'oracolo**, non solo un controllo
random: non "il nostro segnale non basta", ma "nessun segnale basterebbe, su
questo canale". Non ha senso spendere GPU su varianti dei knob (`top_k`,
`sink_filter`, `query_rows`) dello stesso intervento.

## 6. Note di registro

### 6.1 `visual_prompt`: dalle probe al fullset

Fino al 2026-08-30 esisteva solo come probe da 60 sample (shard 0, gruppo
`-test`), lanciate per i check di correttezza, **non** per l'accuracy: su 60
sample un sample vale 1.7 pp.

| job | knob variato | acc (60) | note |
|---|---|---:|---|
| 93320_0 | `query_rows=question`, `sink_filter=false` | 61.67% | preset dell'arm |
| 93321_0 | `query_rows=question`, `sink_filter=true` | 63.33% | appaiato col precedente |
| 93524_0 | `entity` v1 | — | ucciso (Turing, ~140 s/sample) |
| 93525_0 | `entity` v1 | 61.67% | estrazione utile 8/60: il 2B risponde alla MCQ |
| 93543_0 | `entity` v3 (solo domanda + few-shot) | 63.33% | 60/60 mappate, ma elenca più soggetti |
| 93573_0 | `entity` v4 (una sequenza o `NONE`) | 61.67% | 40/50 utili sui non-NONE |

Tutti i Δ valgono 1 sample. Il fullset (job 94528/94532) è stato lanciato
**dopo** l'oracolo, che ne aveva previsto l'esito: serviva la misura diretta
sui 2700, perché la chiusura della metà "marcare" poggiava altrimenti su un
benchmark diverso da quello su cui gli arm erano stati misurati.

### 6.2 Il prompt di estrazione entity: tre iterazioni

- **v1** (domanda + opzioni in input): 8/60 utili. Con le opzioni nel
  contesto il prior "MCQ → rispondi" di un 2B domina qualsiasi istruzione: il
  modello sputava un'opzione invece del soggetto.
- **v3** (solo domanda + few-shot): 60/60 mappate, ma elencava tutto il
  materiale della domanda (`rope, person, ice bucket challenge`),
  rispalmando il segnale.
- **v4** (UNA sequenza contigua verbatim, oppure `NONE`): 40/50 utili sui
  non-NONE, 10 `NONE` tutti giudizi corretti, 0 degeneri. I `NONE` sono le
  domande senza soggetto specifico ("What is this video about?"), dove
  degradare a `question` è la risposta giusta.

### 6.3 `sink_filter`: chi lo ha usato

Il default della funzione `ranked_cells_from_attention` è `True`; il default
delle strategy che espongono il knob è `False`. Le tre strategy divergono:

| strategy | knob esposto | valore effettivo |
|---|---|---|
| `attention_highlight` | **no** | `true` implicito su tutte le 16 run (`highlight_top1_*`, `random_top1_*`, `antipeak_*`) |
| `attention_marker` | sì, default `False` | `false` sul fullset; `true` solo sulla probe 92829_0 |
| `visual_prompt` | sì, default `False` | `false` sulle probe (tranne 93321_0); **`true` sul fullset** |

⚠️ Un confronto diretto fra famiglie mescola due cambiamenti. L'unica misura
appaiata dell'effetto del filtro è la coppia 93320/93321: 1 sample di
differenza. Dai metadati per-sample: con `sink_filter=false` la cella scelta
differisce da quella filtrata nel 33-40% dei sample — il knob **arriva** al
ranking, semplicemente non sposta l'accuracy.

### 6.4 Geometria dell'overlay

Il gate di leggibilità §8.1 è stato passato a `top`/`height_frac=0.18`; il
fullset gira a `bottom`/`0.12`, dove la parola arriva al modello a ~25px dopo
il downscale del processor invece di ~38px. Sotto `0.10` su frame piccoli
`paint_important` solleva (sample perso, non degrado silenzioso).
