# ViKey — cosa portiamo e cosa no

Nota di lettura del paper, per uso personale. Riferimento: Lee, Ju, Kim, Kang,
Hwang (Yonsei), *ViKey: Enhancing Temporal Understanding in Videos via Visual
Prompting*, [arXiv 2603.23186v1](https://arxiv.org/html/2603.23186v1),
24 marzo 2026.

Qui c'è solo il perimetro: **quali pezzi del paper entrano nel repo, quali
restano fuori, e per ciascuno il motivo.** Il "come" dei pezzi che entrano sta
nelle docstring dei file citati.

---

## 1. La distinzione che governa tutto: indirizzamento ≠ salienza

Il paper e i nostri arm dipingono entrambi sui pixel, ma **non dipingono la
stessa cosa**, e confondere le due è il modo più rapido per leggere male sia
il paper sia i nostri risultati.

| | i nostri arm (A, B, C) | ViKey |
|---|---|---|
| cosa dipinge/scrive | un **giudizio**: "questa cella è importante" | un **indirizzo**: "questa cella è la #07" |
| su quante celle | 1 (`top_k=1`) | **tutte** |
| serve un segnale a monte | sì, l'attenzione | **no**, l'indice è noto per costruzione |
| cosa aggiunge al modello | una scorciatoia | una **capacità** che non aveva |

I nostri tre arm — A [`attention_highlight`](../strategies/attention_highlight.py)
(puntatore testuale), B [`attention_marker`](../strategies/attention_marker.py)
(marcatori inline), C [`visual_prompt`](../strategies/visual_prompt.py)
(`IMPORTANT` sui pixel) — sono tutti e tre **salienza**, consegnata su tre
canali diversi. ViKey è **indirizzamento**, ed è una cosa che non abbiamo mai
provato.

Questo è il motivo per cui il paper non è in contraddizione con
[§1 di `260901-tabelle-risultati.md`](260901-tabelle-risultati.md): il nostro
risultato è *marcare la cella giusta non vale più che marcarne una a caso*,
cioè **al modello non manca un puntatore di salienza**. ViKey sta rispondendo
a un'altra domanda, e il suo esperimento sintetico suggerisce che ciò che
manca davvero sia l'indirizzo.

---

## 2. Il perimetro in una tabella

| pezzo del paper | lo portiamo? | dove / perché no |
|---|---|---|
| **SVP** — indice del frame dipinto sui pixel | **sì, come renderer** | [`utils/overlay.py:143`](../utils/overlay.py#L143) `paint_frame_index` |
| geometria dell'etichetta (`min(w,h)/12`, rosso, angolo) | **sì, verbatim** | stessa funzione |
| bias posizionale (angoli bassi ≫ alti) | **sì**, come knob e come sospetto su `visual_prompt` | `position` in `paint_frame_index`; §3.2 |
| il test sintetico ago-in-un-frame come **metodo** | **sì, riscritto** | [`scripts/probe_frame_addressing.py`](../scripts/probe_frame_addressing.py) |
| **KFM** — keyword→frame via CLIP + LLM estrattore | **no** | §4.1 |
| l'unità "frame" come cosa indirizzabile | **no, diventa la cella** | §4.2 |
| i delta sui benchmark come evidenza | **no** | §4.3 |
| l'idea che i frame siano anonimi | **da verificare, non assunta** | §5.1 — è la premessa che su Qwen3-VL potrebbe già essere falsa |
| il trial negativo nel lookup | **non c'è nel paper, l'abbiamo aggiunto** | §6 |

---

## 3. Cosa portiamo

### 3.1 SVP come renderer

[`utils/overlay.py:143`](../utils/overlay.py#L143) — `paint_frame_index`
dipinge `frame #07` in un angolo, con la geometria del paper: corpo
`min(w, h) / s` con `s = 12`, testo rosso, angolo basso di default.

Sta nel modulo puro accanto a `paint_important` e **non la tocca**: sono due
overlay che condividono il canale e nient'altro (§1). Il commento nel file lo
dice esplicitamente, così fra sei mesi non sembrano due varianti della stessa
cosa.

Una deviazione dichiarata: il paper dice "testo rosso, opzionalmente con
sfondo bordato"; noi mettiamo **sempre** lo sfondo pieno. Stesso motivo per
cui `paint_important` usa una banda e non un contorno — il contrasto non deve
dipendere dal contenuto del frame, altrimenti l'etichetta sparisce proprio nei
sample in cui servirebbe.

### 3.2 Il bias posizionale

È il risultato più immediatamente azionabile del paper, e non riguarda solo
l'SVP. Sul reverse lookup misurano:

| posizione | reverse lookup |
|---|---:|
| TL | 60.19% |
| TR | 78.70% |
| BL | **100.0%** |
| BR | **100.0%** |

Il nostro [`strategies/visual_prompt.py:98`](../strategies/visual_prompt.py#L98)
ha `overlay_position` con default `"top"`. Il loro task non è il nostro — noi
dipingiamo una banda larga, non un'etichetta d'angolo — ma il bias nasce dalla
distribuzione dei dati di training ed è plausibilmente condiviso. Cambiare a
`bottom` costa un knob e va fatto **prima** del fullset di `visual_prompt`,
non dopo.

### 3.3 Il test sintetico come metodo

Questo è il pezzo del paper che vale di più, molto più delle tabelle sui
benchmark: inserire un ago in un frame noto e chiedere di indirizzarlo isola
la capacità pura, senza il rumore di un benchmark di QA.

Lo abbiamo riscritto in
[`scripts/probe_frame_addressing.py`](../scripts/probe_frame_addressing.py)
con le modifiche di §4.2 e §6. L'ago è un cerchio verde saturo con bordo
bianco ([`:92-94`](../scripts/probe_frame_addressing.py#L92-L94)) invece di
un'immagine di panda: nessun asset da procurare, e a quella saturazione non si
confonde con "c'era del verde nella scena".

---

## 4. Cosa NON portiamo

### 4.1 KFM

Il secondo componente — un LLM estrae le keyword dalla query, CLIP le matcha
contro i frame, sopra soglia τ l'indice finisce nel prompt — resta fuori.

Tre motivi, in ordine di peso:

1. **Rende poco.** Nell'ablazione (Tabella 3a, Video-MME TR): baseline 44.1 →
   VP 49.2 → **VP+KFM 50.3**. KFM aggiunge +1.1 pp sopra l'SVP da solo.
2. **Porta tre dipendenze mobili** — l'estrattore di keyword, il backbone
   CLIP, la soglia τ — e gli autori dichiarano che **τ va ritarata per
   dataset**. Tre superfici da tarare per +1.1 pp è un pessimo rapporto.
3. **Duplica un pezzo che abbiamo già.** L'estrazione di entità dalla query
   esiste nel repo (`query_rows=entity`,
   [`scripts/probe_entity_extraction.py`](../scripts/probe_entity_extraction.py));
   se un giorno servisse quel ponte, si costruisce su quello, non su CLIP.

### 4.2 L'unità indirizzabile: il frame diventa la cella

Non è una scelta, è un adattamento obbligato — ed è la cosa che il paper non
poteva porsi, perché nessuno dei suoi modelli ha questa geometria.

ViKey dipinge un numero su **ogni frame campionato**. Su Qwen3-VL in regime
`double_frames=false` una **cella = 2 frame**, e il processor emette un
timestamp per cella: 24 frame → **12 indirizzi**, non 24. Dipingere numeri
diversi sui due frame di una cella li farebbe collassare nello stesso gruppo
di token visivi.

Il port fedele è quindi **un'etichetta per cella, ripetuta identica sui due
frame** — [`scripts/probe_frame_addressing.py:87`](../scripts/probe_frame_addressing.py#L87)
e il ramo `svp` del ciclo. Stessa aritmetica `2*c` / `2*(c+1)` delle celle di
[`attention_marker`](attention_marker_implementazione.md).

### 4.3 I delta sui benchmark come evidenza

I numeri di accuracy del paper **non li usiamo come termine di paragone**, per
quattro problemi che si sommano:

1. **Non sono i benchmark interi.** Video-MME → solo *Temporal Reasoning*;
   MVBench → solo *Action Sequence* + *Scene Transition*; TempCompass → solo
   *Event Order* + *Attribute Change*. Un "+1.93 su MVBench" non è un numero
   confrontabile con i nostri.
2. **La numerosità delle tabelle principali non è dichiarata** (i 100 esempi
   sono i sottoinsiemi di analisi). Un +0.79 pp potrebbe essere un sample.
3. **Incoerenza interna**: la baseline Video-MME TR vale 48.63 in Tabella 2 e
   44.1 in Tabella 3a, senza spiegazione.
4. **Un degrado grosso e non commentato**: LLaVA-OneVision su Video-MME a 64
   frame fa **52.53 → 42.94, −9.6 pp**. Il paper lo segna con un triangolino e
   non ne parla. Un metodo che a volte toglie 10 punti senza che nessuno sappia
   perché non è "plug-and-play".

Quello che portiamo del paper è il **meccanismo** (reverse lookup 18.57% →
100%, che è grande, sintetico e non ambiguo), non le sue accuracy.

### 4.4 L'endpoint, che va corretto in senso opposto

Non è un pezzo da portare, ma un avvertimento che il paper ci regala.

Gli autori dichiarano che il guadagno è **sulle categorie temporali** e che c'è
un lieve degrado su quelle non temporali. Sul nostro Video-MME intero le
categorie temporali sono `temporal reasoning` (177) + `temporal perception`
(55) = **232 su 2700, l'8.6%**. Anche un +5 pp lì vale +11 sample = **+0.4 pp
pooled**: invisibile.

Conseguenza operativa: per qualunque arm di questa famiglia il pooled è
**sotto-potenziato per costruzione**, e le sottocategorie temporali vanno
pre-registrate come endpoint. Vale già per il fullset di `visual_prompt`.

---

## 5. La premessa del paper che su Qwen3-VL potrebbe essere falsa

### 5.1 Qwen3-VL i timestamp li ha già

ViKey parte da "i frame sono anonimi, il modello non ha modo di riferirsi al
frame *k*". Vero per i modelli che testano — Qwen2.5-VL, LLaVA-Video,
LLaVA-OneVision, **nessuno dei quali interleava timestamp testuali**.

Su Qwen3-VL no. Dalla docstring di [`Qwen3VL`](../models/qwen.py) (punto 2, `models/qwen.py`):

```
2. **`pass_video_metadata = True`** — obbligatorio, non un'opzione: il
   processor di Qwen3-VL scrive i timestamp DENTRO il prompt, un
   `<x.x seconds>` prima di ogni blocco visivo
   (`processing_qwen3_vl.py:157-170`), calcolandoli da
   `video_metadata.frames_indices / fps`.
```

Ogni isola visiva è preceduta dal suo tempo in chiaro: **uno schema di
indirizzamento c'è già**, in secondi invece che in indici. L'SVP rischia
quindi di essere ridondante rispetto a un canale che il modello possiede.

*In teoria* ridondante. Se quel timestamp sia **usabile** per indirizzare è
una domanda empirica, ed è l'unica cosa che separa "arm sensato" da "GPU
buttate".

### 5.2 La sonda che decide

[`scripts/probe_frame_addressing.py`](../scripts/probe_frame_addressing.py),
lanciata da
[`scripts/sbatch/probe_frame_addressing.sbatch`](../scripts/sbatch/probe_frame_addressing.sbatch)
(~15-20 min, una GPU). Due condizioni, stesso ago:

| condizione | pixel | indirizzo nella domanda |
|---|---|---|
| `native` | solo l'ago | `"the moment at 12.3 seconds"` — i timestamp che Qwen3 già emette |
| `svp` | ago + `frame #NN` su ogni cella | `"the frame labelled 'frame #07'"` — ViKey |

**Regola di decisione, fissata prima di vedere i numeri:**

- `native` indirizza già bene → **l'SVP non si integra**. E, corollario non
  banale, il nullo della variante A non è un problema di indirizzamento: va
  cercato altrove.
- `native` indirizza male come nel paper → l'SVP diventa un arm sensato, ed è
  soprattutto il **prerequisito** per rilanciare la variante A, finora
  misurata su un canale che il modello non sapeva risolvere.

C'è un auto-controllo prima di tutto il resto: la sonda stampa i
`<x.x seconds>` che il processor mette **davvero** nel prompt, per verificare
la mappa cella→timestamp che assume (`t_c = frame_indices[2c] / fps`). Se lì
compaiono 24 timestamp invece di 12, l'assunzione è sbagliata e nessun numero
sotto vale.

---

## 6. Cosa aggiungiamo che nel paper non c'è

### 6.1 Il trial negativo

Il lookup di ViKey chiede "descrivi il frame *k*" e conta come successo la
presenza della parola "panda" nella risposta. **Un modello che nomina sempre
il panda — è nel video, da qualche parte — fa 100% senza indirizzare niente.**
Il paper non riporta un controllo su questo, e il loro 12.43% → 64.62% è
quindi un limite superiore.

Da qui il disegno a domanda binaria con esca
([`:101`](../scripts/probe_frame_addressing.py#L101)): la stessa domanda
YES/NO posta sulla cella giusta (atteso YES) e su una cella distante ≥3
(atteso NO). La metrica è l'**accuratezza bilanciata**, caso = 50%, e la
colonna `%YES` smaschera la modalità degenere a colpo d'occhio.

### 6.2 Il gate di percezione

Prima di ogni misura di indirizzamento, una domanda senza indirizzo: "c'è un
cerchio verde da qualche parte in questo video?". Se la risposta è no, la
sonda non sta misurando indirizzamento ma percezione, e il resto dei numeri
non significa niente. Stessa logica del gate §8.1 di
[`piano_visual_prompting.md`](piano_visual_prompting.md).

### 6.3 Quello che abbiamo tolto

Avevo previsto una terza condizione — `pass_video_metadata=False`, cioè senza
timestamp — come controllo negativo. **Non c'è**: il trial negativo di §6.1
risponde già alla stessa domanda (gli indirizzi portano informazione?) senza
forzare un regime che
[`Qwen3VL`](../models/qwen.py) dichiara obbligatorio, e che
introdurrebbe una seconda variabile nel confronto.
