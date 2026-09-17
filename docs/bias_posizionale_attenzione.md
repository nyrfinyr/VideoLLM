# Il bias posizionale del puntatore d'attenzione (LVBench, Qwen3-VL-2B)

> Misure sui per-sample della probe di segnali **107727**
> (`qwen3_vl_2b_attn-signals_512p-107727_0_probe100`, project `lvbench`):
> 100 domande su **59 video distinti**, 69 usabili per T1 dopo il filtro di
> `analyze_topk_run.py::label_cells`. Tutto ricalcolabile dai log già
> esistenti, nessuna run nuova.
>
> Stato al 2026-09-17. **Decisione presa: la run `topk_resample` parte con il
> ranking GREZZO** — niente filtro sink, niente normalizzazione posizionale
> (§7).

Contesto: [`run_topk_lvbench.md`](run_topk_lvbench.md),
[`oracolo_lvbench.md`](oracolo_lvbench.md).
Codice toccato da queste misure: nessuno — sono analisi offline sui campi
`rowsets.<rs>.cell_mass_raw`, `cell_mass_sink_filtered`,
`sink_map_temporal_mean`, `sink_mass_curve` già loggati da
`strategies/signals_capture.py`.

---

## 1. La domanda

T1 aveva stabilito che a 256 celle l'attenzione localizza la finestra
annotata (AUC 0.775, hit@1 17% contro 1.6% del caso). L'arm `topk_resample`
si fida di quel ranking e ricampiona nelle top-k celle.

Ma **di che cosa è fatto quel ranking?** Se una parte sistematica della massa
d'attenzione non dipende dalla domanda, l'arm spende budget in posti decisi a
priori, e l'ablation di k misura in parte quello invece della copertura.

## 2. Il profilo posizionale esiste, ed è grosso

Mediando la massa per cella su tutti i sample (rowset `all`, 100 domande,
ogni sample normalizzato a somma 1), le 256 celle **non** sono equivalenti.
Valori in multipli dell'uniforme (1/256 = 0.391%):

```
medie a blocchi di 16 celle, dall'inizio alla fine del video
2.37  1.24  0.91  0.87  0.86  0.77  0.77  0.67  0.73  0.80  0.79  0.72  0.79  0.88  0.88  1.98
```

| | cella 0 | cella 1 | celle 2-5 | centro (100-156) | cella 254 | cella 255 |
|---|---|---|---|---|---|---|
| attenzione grezza | **6.3x** | 3.4x | 2.4x | 0.74x | 3.7x | **6.8x** |

Le **10 celle più calde in media sono tutte di bordo**: `[255, 0, 254, 1, 253,
5, 4, 6, 2, 252]`. Sulle 10 celle di bordo (prime 5 + ultime 5, cioè il 3.9%
del video) sta il **13.4%** della massa: 3.4x l'uniforme.

Non è un artefatto del campionamento: su LVBench (video ≥ 30 min, celle ≥ 7 s)
nessun centro di bordo viene clampato da `_pairs_from_centers`.

⚠️ **Da non confondere con `attn_border_share`** (D8 di
[`run_topk_lvbench.md`](run_topk_lvbench.md)): quello è l'anello esterno di
patch *dentro il frame*, una questione spaziale. Qui "bordo" è l'inizio e la
fine del *video*, una questione temporale. Sono misure indipendenti.

## 3. È una proprietà del modello, non del dataset

| verifica | risultato | lettura |
|---|---|---|
| Stabilità split-half (per domanda, 50 ripetizioni) | r = **0.957 ± 0.010** | non è rumore |
| Profilo stimato solo su **altri video** | hit@k 22/33/48 contro 22/33/49 del leave-one-out | non è contenuto dei video |
| Video **corti** (30-73 min) contro **lunghi** (73-140 min), 50+50 | r = **0.935**; cella 0 6.8x vs 5.8x, cella 255 6.5x vs 7.1x, centro 0.75x vs 0.74x | non è la durata |
| Quante domande servono per stimarlo | n=20 → r 0.961; n=40 → 0.985 | |
| Quanti **video** servono | 3 → 0.775; 5 → 0.863; 10 → 0.919; **20 → 0.961**; 59 → 0.995 | ~20 video bastano |

Il profilo si stima **senza mai guardare `time_reference`**: è una statistica
non supervisionata delle sole masse d'attenzione. Non c'è leakage di etichette
possibile.

⚠️ **Limite di identificazione.** Le 256 celle sono uniformi sull'intero video,
quindi *indice della cella* e *posizione relativa nel video* sono la stessa
cosa in questo disegno. Non si può distinguere "il modello privilegia i primi
e gli ultimi token visivi della sequenza" da "il modello privilegia l'inizio e
la fine del video". Entrambe restano proprietà del modello, non del dataset.

## 4. Ha ~3 gradi di libertà, non 256

Il profilo è quasi **simmetrico rispetto alla distanza dal bordo più vicino**
(correlazione fra metà sinistra e metà destra specchiata: **r = 0.915**), e un
fit a tre parametri lo descrive bene:

```
profilo(d) = a + b·exp(−d/τ)        d = distanza dal bordo più vicino, in celle
a = 0.87    b = 4.86    τ = 3.0 celle        R² = 0.840
```

E funziona **identicamente** al vettore a 256 valori:

| correzione | hit@1 | hit@3 | hit@10 | celle di bordo scelte (k=3) |
|---|---|---|---|---|
| profilo a 256 valori | 20% | 33% | 49% | 5% |
| fit a 3 parametri | 19% | 32% | 51% | 3% |

Conta per la difendibilità: non è "256 parametri stimati sui dati", è **un
decadimento esponenziale dai bordi della sequenza con costante di scala 3
celle**.

**Dipendenza dal rowset** (quando va ri-stimato): `all` e `question` hanno lo
stesso profilo (r = 0.984); `last_token` no (r = 0.907, ultima cella a
**13.4x** invece di 6.8x, centro 0.65x). Presumibilmente dipende anche dal
numero di celle e dal modello — non verificabile con questi dati (tutti i
sample hanno t=256).

## 5. Che cosa fa all'arm `topk_resample`

Celle effettivamente scelte dal ranking, sui 69 sample usabili. "Bordo" =
prime 5 o ultime 5 celle su 256, cioè il **3.9%** atteso a caso.

| k | ranking | hit@k | celle di bordo scelte | sample con ≥1 cella di bordo |
|---|---|---|---|---|
| 1 | grezzo | 17% | **54%** | 54% |
| 1 | posnorm | 22% | 6% | 6% |
| 3 | grezzo | 29% | **53%** | **87%** |
| 3 | posnorm | 33% | 5% | 12% |
| 10 | grezzo | 52% | **39%** | **100%** |
| 10 | posnorm | 49% | 4% | 28% |

**`k3` spende metà del budget nei primi e ultimi ~80 secondi del video, e
nell'87% dei sample almeno una delle 3 regioni è lì. `k10` ne butta 4 su 10,
in tutti i sample.**

Che sia budget sprecato, non segnale, lo dicono tre fatti indipendenti:

1. Il predittore di **sola posizione** (profilo medio usato come ranking,
   identico per ogni sample) ha **AUC 0.552, hit@1 1%**: da solo non sa niente
   di dove stia la finestra.
2. La finestra vera tocca la fascia di bordo solo nel **10%** dei sample
   (7/69). La sua posizione mediana è al **33%** del video (quartili 0.14 /
   0.33 / 0.55).
3. Scomponendo `k1` grezzo: picco **sul bordo** in 37/69 sample, dove azzecca
   **2 volte su 37 (5%)**; picco **altrove** in 32/69, dove azzecca **10 volte
   su 32 (31%)**.

Il 17% di hit@1 che motiva l'arm è quindi "31% sulla metà dei sample in cui il
puntatore non è incollato ai bordi, 5% sull'altra metà".

### Esempio

Sample `Za2Z_JRxCuk/44`, video di 34 minuti, finestra annotata 1068–1085 s →
celle vere **134 e 135**.

| cella | istante | massa grezza | profilo medio | massa ÷ profilo | |
|---|---|---|---|---|---|
| 255 | 2040 s | 8.42x | 6.74x | 1.25 | (bordo) |
| 2 | 20 s | 6.90x | 2.26x | 3.05 | (bordo) |
| 1 | 12 s | 4.72x | 3.31x | 1.43 | (bordo) |
| 194 | 1553 s | 4.51x | 0.69x | **6.48** | |
| 134 | 1074 s | 3.72x | 0.68x | **5.50** | ← finestra |
| 195 | 1561 s | 3.55x | 0.68x | 5.24 | |
| 135 | 1082 s | 1.36x | 0.79x | 1.72 | ← finestra |

Top-3 grezzo `[255, 2, 1]`: gli ultimi 8 e i primi 24 secondi del video,
nessuna è la finestra. Top-3 normalizzato `[194, 134, 195]`: la 134 è la
finestra. L'informazione c'era, il ranking grezzo la copriva.

## 6. NON è il fenomeno "sink" già documentato

Nel repo "sink" indica i **canali outlier degli hidden state** (sink dims di
Xiao et al., `models/qwen_attn.py::SINK_DIMS`), da cui `sink_map` e il
`sink_filter`. Le celle di bordo **non** sono sink in quel senso:

| profilo per cella | cella 0 | celle 2-5 | centro | cella 255 | quota sulle 10 celle di bordo |
|---|---|---|---|---|---|
| attenzione grezza | 6.3x | 2.4x | 0.74x | 6.8x | 13.4% (**3.4x**) |
| attenzione sink-filtrata | 2.6x | 1.8x | 0.77x | **6.9x** | 11.0% (2.8x) |
| **punteggio di sink** | **1.1x** | 1.0x | 1.00x | **1.0x** | 4.0% (**1.0x**) |

Il punteggio di sink è **piatto**: sulle celle di bordo vale esattamente
quanto altrove (1.0x). Filtrare i sink non toglie il bias — la cella 255 passa
da 6.8x a **6.9x**, invariata; solo la cella 0 scende (6.3x → 2.6x), coerente
col suo 1.1x. I due profili correlano a **+0.924**: è quasi lo stesso ranking.
La correlazione di forma fra profilo d'attenzione e profilo di sink è +0.456,
ma l'ampiezza no: 3.4x contro 1.0x sul bordo.

Conferma della **PROVA 3** già nota (massa d'attenzione sui top-p% token per
punteggio di sink, media sui 100 sample):

```
p%       1%     2%     5%    10%    25%    50%
massa  0.8%   1.6%   3.7%   7.2%  18.4%  39.2%
```

Sempre **sotto p%**: i token con sink score alto assorbono attenzione *meno*
che uniformemente. Non sono pozzi.

> **Sono due fenomeni distinti che convivono.** Da un lato canali outlier
> negli hidden state, reali e fortissimi (rango 0 e 1 su 2048 canali, |h|
> 20-59x la media) che **non catturano attenzione**. Dall'altro una
> concentrazione posizionale dell'attenzione ai bordi, altrettanto reale, che
> **non ha firma negli hidden state**. Un lavoro che le conflatesse
> sbaglierebbe.

Corollario: la normalizzazione posizionale **è** il filtro dei sink, nel senso
comportamentale — identifica i token che assorbono attenzione a prescindere
dal contenuto guardando quali celle sono sempre calde, invece che gli hidden
state. Non sono due strade alternative.

## 7. Nessuna correzione è dimostrato che aiuti

Tutte le varianti provate, 69 sample usabili:

| ranking | hit@1 | hit@3 | hit@10 | bordo (k=3) | serve una statistica di dataset? |
|---|---|---|---|---|---|
| **grezzo** | 17% | 29% | 52% | 53% | — |
| filtro sink (canali outlier) | 17% | **25%** | 49% | 43% | no |
| posnorm, profilo leave-one-out | 19-22% | 33% | 49% | 5% | sì |
| posnorm, fit a 3 parametri | 19% | 32% | 51% | 3% | sì |
| detrend per-sample, mediana mobile 51 | 19% | 23% | 48% | 41% | **no** |
| detrend per-sample, mediana mobile 101 | 14% | 28% | 48% | 48% | **no** |
| maschera prime/ultime 5 celle | 17% | 32% | 51% | 0% | no |
| maschera prime/ultime 10 celle | 17% | 33% | 49% | 0% | no |
| maschera prime/ultime 20 celle | 16% | 32% | 45% | 0% | no |
| (caso) | 2% | 4% | 13% | 3.9% | |

Test appaiati (McNemar esatto), posnorm contro grezzo:

| k | grezzo | posnorm | win | loss | p |
|---|---|---|---|---|---|
| 1 | 17% | 20% | 4 | 2 | 0.688 |
| 3 | 29% | 33% | 8 | 5 | 0.581 |
| 5 | 38% | 41% | 8 | 6 | 0.791 |
| 10 | 52% | 49% | 5 | 7 | 0.774 |

Maschera 5 celle contro grezzo: hit@1 win 2 / loss 2, p = 1.000; hit@3 win 6 /
loss 4, p = 0.754.

**Nessuna differenza è significativa.** Su 69 sample "17% → 22%" sono 12
sample contro 15 (e diventano 14 cambiando la base su cui si stima il
profilo — per questo la riga riporta 19-22%). La direzione è positiva a k
basso e negativa a k=10, tutto dentro il rumore.

Il meccanismo si capisce: il grezzo pesca al bordo il 54% delle volte e lì
azzecca il 5%, ma le sue scelte *non* di bordo sono le anomalie più forti e
azzeccano il 31%. La posnorm sposta il budget lontano dai bordi, ma le scelte
che aggiunge sono anomalie più deboli. **Redistribuisce senza vincere.**

Il detrend per-sample fallisce per un motivo strutturale: il bias è ripido
proprio al bordo, dove la finestra mobile è troncata (a cella 0 la mediana si
calcola su [0, 25], che contiene già le celle gonfiate).

## 8. Decisione

**La run `topk_resample` parte con il ranking grezzo.** Niente
`rank_sink_filtered` (resta `false`, ed è la scelta giusta: il filtro va
peggio del grezzo), niente normalizzazione posizionale, nessuna condizione in
più. Motivo: nessuna correzione ha evidenza di migliorare hit@k, e una sesta
condizione costerebbe ~12% di walltime su un'ipotesi non supportata.

Quello che resta solido e va riportato comunque è la **caratterizzazione**: il
puntatore grezzo concentra il 53% del budget sul 3.9% del video, ai bordi
della sequenza, dove l'evidenza sta il 10% delle volte.

**Da rifare sul fullset, gratis.** `cell_mass_raw` è loggata per sample su
tutte le 1549 domande, quindi tutte le misure di questo documento si
ricalcolano offline su ~1000 sample usabili invece di 69 — quindici volte i
dati. Se **lì** la posnorm batte il grezzo in modo netto, quella è la
motivazione per una run successiva col ranking calibrato, e il profilo va
stimato su **video held-out** disgiunti da quelli di valutazione (a costo
zero: §3 mostra che trasferisce).

## 9. Come riprodurre

```bash
# per-sample della probe (via Weave), poi qualunque analisi offline
uv run python scripts/analyze_topk_run.py \
    qwen3_vl_2b_attn-signals_512p-107727_0_probe100 --cache /tmp/probe107727.json
```

Il profilo è la media, su tutti i sample, di
`out["rowsets"]["all"]["cell_mass_raw"]` normalizzata a somma 1. Le etichette
per cella vengono da `analyze_topk_run.py::label_cells`
(`utils.pair_sampling.pair_cells_in_window`), che scarta finestre assenti,
finestre senza nessuna cella dentro e finestre che coprono più di 25/256 celle.
