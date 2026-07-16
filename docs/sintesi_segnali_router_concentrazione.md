# Sintesi — segnali di concentrazione e router entropia/attenzione

> Documento di sintesi (2026-07-16), preparato per presentare lo stato del
> lavoro. Raccoglie in un unico posto: cosa stiamo cercando di fare, quali
> segnali vengono calcolati e loggati nel codice, come e perché sono stati
> scelti, quali esperimenti li hanno validati/respinti. Per il dettaglio
> passo-passo delle analisi vedi `docs/piano_zoom_multiregione_disgiunto.md`
> (cronologia completa) — questo file ne è il riassunto ad alto livello.

## 1. Obiettivo

Su un benchmark video-MCQ (Video-MME, Qwen2.5-VL-3B), il campionamento
uniforme dei frame è "cieco": prende sempre lo stesso numero di frame
equispaziati, a prescindere da quanto il modello sia confidente o da dove
stia effettivamente guardando. L'idea centrale: usare segnali che il
modello produce **gratis** durante l'inferenza (nessun forward aggiuntivo
oltre a quelli già necessari) per decidere, sample per sample:

1. **SE** vale la pena ricampionare (il modello sta indovinando o è già
   sicuro?).
2. **COME** ricampionare, se sì (dove guardare invece di ricampionare alla
   cieca?).

Questo è implementato in `strategies/entropy_attention_resample.py`
(`EntropyAttentionResampleStrategy`, preset `strategy=entropy_attention_
resample` in `conf/config.yaml`).

## 2. Come funziona il meccanismo (pipeline per ogni sample)

```
pass 1 (SEMPRE)          →  gate su answer_entropy       →  pass 2 (risposta finale)
generate_with_signals()     "il modello sta indovinando?"    vlm.generate() vero
(prefill-only, nessuna       │                                (sui frame originali
generazione, frame            NO → accetta la risposta         SE si accetta,
uniformi come baseline)              del pass 1                su frame RICAMPIONATI
                             SI → decide IL RAMO                SE si ricampiona)
                                  (dove si guardava?)
                                  disperso   → sfasa la
                                              griglia (phase_shift)
                                  concentrato → infittisce vicino
                                              al picco (zoom)
```

Punti chiave del design:

- **Il pass 1 costa sempre** (è un forward di prefill completo con
  cattura dell'attenzione), **indipendentemente** dal fatto che poi si
  ricampioni o no — non è "extra solo se serve".
- **La risposta finale viene SEMPRE da una vera `generate()`** — mai dal
  solo `pred_letter` della softmax ristretta del pass 1. Questo garantisce
  che i sample "accettati" restino bit-per-bit identici alla baseline
  `uniform`: l'unico effetto misurabile viene dai sample effettivamente
  ricampionati.
- **Al massimo un ricampionamento** — nessun terzo pass che ri-verifichi
  la decisione.

## 3. I segnali: cosa calcoliamo, dove, perché

Tutti calcolati in `strategies/entropy_attention_resample.py::answer()`
(righe 414-440), a partire dall'oggetto `SignalAnswer` restituito da
`vlm.generate_with_signals(...)` (`models/signals.py`, implementato da
`Qwen25VLAttention.generate_with_signals` in `models/qwen_attn.py`, che
avvolge `full_visual_attention` — stesso forward già necessario per la
cattura, zero costo aggiuntivo).

### 3.1 `answer_entropy` — il segnale di GATE

**Cos'è**: entropia di Shannon (in bit) della distribuzione softmax
**ristretta alle sole lettere candidate** (A/B/C/D...) sui logit
dell'ultimo token del prefill — non la softmax sull'intero vocabolario.
`H = 0` → il modello è certissimo di una lettera. `H = log2(n_opzioni)` →
distribuzione piatta, sta indovinando.

**Perché**: prima ipotesi testata (pilota VNBench, 30 video, `docs/
entropia_confidenza_mcq.md`) — correlazione con la correttezza **r = −0.75**,
il segnale più forte fra tutti quelli provati. Usato oggi come **gate**:
`can_resample = answer_entropy > entropy_threshold` (default `0.7`, non
validato per Video-MME nello specifico — soglia ereditata dal pilota
VNBench, dominio/nframes diversi).

### 3.2 `top1_pct` — concentrazione dell'attenzione (il segnale vincente)

**Cos'è**: sulla mappa di attenzione testo→visivo (già catturata dal
pass 1), si azzerano prima i token "sink" (`sink_view`, altrimenti
assorbono la maggior parte della massa indipendentemente dal contenuto —
`docs/qwen25vl_entity_attention.md`), si aggregano le celle spazio-
temporali per massa totale (`rank_cells`), e si prende la **percentuale
di massa (post-filtro-sink) sulla cella più attenzionata**. Alto →
il modello "si fissa" su un frame preciso (grounding). Basso/vicino
all'uniforme → attenzione dispersa su tutto il video.

**Perché**: seconda ipotesi (stesso pilota VNBench) — correlazione più
debole dell'entropia (r = 0.58) ma stesso verso atteso. Usato per
decidere **il ramo** (vedi §4) — e si è rivelato, dopo un confronto
empirico a tre (§4.3), il **migliore** dei segnali di concentrazione
provati, nonostante sia il più "grezzo".

### 3.3 `attention_entropy_norm` (H_norm) — tentativo di segnale più ricco

**Cos'è**: entropia di Shannon **normalizzata** `H/log(t)` sulle masse
di **tutte** le `t` celle (non solo il picco) — `0` = tutta la massa su
una cella, `1` = dispersione perfettamente uniforme su `t` celle.
Funzione `_attention_entropy_norm` (righe 110-131).

**Perché provato**: `top1_pct` guarda solo il picco, ignora la forma di
tutto il resto della distribuzione — l'intuizione era che l'entropia
sull'intera distribuzione fosse un segnale di concentrazione più completo
e quindi migliore per instradare. **Risultato (§4.3): l'ipotesi non ha
retto** — separa nella direzione giusta ma più debolmente di `top1_pct`.

### 3.4 `top1_zscore` — tentativo di segnale adattivo per-sample

**Cos'è**: quanto la cella di picco è un **outlier** rispetto alla media/
dev.std delle masse di **tutte** le celle dello stesso video — `(top1_pct
− media) / dev.std`, calcolato interamente dai dati di quel singolo video
(nessuna calibrazione esterna, nessun bisogno di conoscere `t`). Funzioni
`_cell_mass_stats`/`_top1_zscore` (righe 134-167).

**Perché provato**: tentativo di normalizzazione **completamente
per-sample**, l'alternativa più "pulita" concettualmente a una soglia
fissa. **Risultato (§4.3): ancora più debole di `H_norm`** — il segnale
peggiore dei tre.

### 3.5 Segnali di supporto (loggati ma non usati per decidere)

- `peak_cell` — indice della cella di picco (posizione temporale).
- `t_cells` — numero di celle temporali del pass 1 (dipende da `nframes`
  e dal fattore di merge temporale del modello, verificato = 64 per
  `nframes=128`).
- `cell_mass_mean`/`cell_mass_std` — media/dev.std delle masse per-cella,
  ingredienti grezzi dietro `top1_zscore`, loggati anche da soli per poter
  ricalcolare offline normalizzazioni alternative senza dover ricatturare.
- `resample_kind`, `n_regions_used`, `region_spans` — quale ramo è stato
  preso e con quali finestre temporali (diagnostica win/loss a posteriori).

Tutti i segnali (tranne quelli di gate) sono calcolati e loggati **SEMPRE**
quando `signal.visual_attention is not None` — anche nei sample "accettati"
(non ricampionati) — per poter analizzare a posteriori come si distribuiscono
rispetto alla decisione, non solo per i sample effettivamente ricampionati.

## 4. Le decisioni prese, in ordine, con i numeri

### 4.1 Serve un gate E una decisione di ramo, non solo un gate

Ricampionare *sempre* alla cieca (fase sfasata) quando l'entropia è alta
funziona (+3.2 punti accuracy vs uniform, §4.2) ma lascia sul tavolo
segnale: un'analisi win/loss (`docs/piano_zoom_multiregione_disgiunto.md`,
"Verifica pre-implementazione") ha mostrato che un **oracolo** che sceglie
per ogni sample il ramo giusto (fase sfasata vs zoom sul picco) recupera
**98/163** sample ricampionati, contro **83/163** del miglior ramo singolo
preso da solo — **+15 di margine puro dal solo instradamento**.

### 4.2 Le soglie assolute fisse sono morte su Video-MME

Le soglie originali (`concentration_threshold` ∈ {20, 30, 45}) venivano
dal pilota VNBench (dominio/nframes diversi). Su Video-MME (`nframes=128`)
`top1_pct` non supera mai ~23% — **nessuna** di quelle soglie discrimina
mai nulla: uno sweep 4×3 di soglie ha dato risultati piatti (tutti i
sample ricampionati finivano sempre nello stesso ramo, a prescindere dalla
soglia). Da qui la necessità di una soglia **adattiva**.

### 4.3 Tre segnali di concentrazione a confronto — `top1_pct` vince

Per capire quale segnale usare per la soglia adattiva, si sono confrontati
`top1_pct`, `H_norm` e `top1_zscore` sullo stesso test: separazione
statistica (Mann-Whitney) fra i sample dove "zoom multi-regione" batte
"fase sfasata" e viceversa, più uno sweep di soglia sull'intero
sottoinsieme ricampionato (163 sample, 3 run wandb distinti uniti per
`example.id`: `o54z2xsd`/`ujdksy60` fase sfasata, `z4qx9hnb` zoom forzato,
`ktvjcyye` controllo uniforme).

| segnale | separazione (\|z\| Mann-Whitney) | plateau accuracy /163 |
|---|---:|---:|
| **`top1_pct`** (grezzo) | **2.77** | **88–90** |
| `attention_entropy_norm` (H_norm) | 2.23 | 85–87 |
| `top1_zscore` | 1.70 | 87 |

Riferimento: sempre-fase-sfasata = 83/163, sempre-zoom = 78/163, oracolo =
98/163. **Nessuno dei due segnali "più sofisticati" ha battuto il proxy
grezzo** — decisione: usare `top1_pct`.

### 4.4 La soglia dev'essere un ratio, non un valore assoluto

Anche con `top1_pct` come segnale vincente, una soglia assoluta (es. 6.0,
il valore ottimale trovato per `nframes=128`) avrebbe lo stesso problema
di trasferibilità di quelle originali: cambiando `nframes`/dataset la
scala di `top1_pct` si sposta. `top1_pct` è una percentuale su `t` celle,
quindi il suo valore atteso sotto attenzione uniforme ("livello di caso")
è `100/t`. Normalizzando per questo (`ratio = top1_pct / (100/t)`), la
soglia si riscala automaticamente.

`t` (= 64 per `nframes=128`) è stato **verificato su GPU reale** (non solo
dedotto dal codice): una cattura di test ha confermato `t=64` esatto, dal
fattore di merge temporale a coppie del modello (`nframes/2`).

**Soglia scelta: `concentration_ratio = 3.85`** (equivalente a `top1_pct
≈ 6.02` a `t=64`, dentro il plateau 88-90/163 di §4.3).

## 5. Cosa fa il codice OGGI (`branch_selector`, `conf/config.yaml`)

Tre modalità, selezionabili senza toccare codice (`strategy.branch_
selector=...`), **additive**: cambiare selettore non altera il
comportamento degli altri (compatibilità bit-per-bit con gli sweep
storici):

| `branch_selector` | condizione "disperso" (→ fase sfasata) | stato |
|---|---|---|
| `"concentration"` (default) | `top1_pct < concentration_threshold` (assoluto, es. 30) | storico, **soglia inerte** su Video-MME |
| `"concentration_ratio"` | `top1_pct < concentration_ratio * (100/t)` | **Stage C, in validazione ora** |
| `"entropy"` | `H_norm > attention_entropy_threshold` | provato, **battuto da `top1_pct`** |

Se non "disperso" (concentrato): o `zoom_peak` (finestra stretta sul
frame di picco) o, se `zoom_multi_region=true`, fino a `n_regions`
finestre disgiunte sulle celle a massa più alta ("Opzione C1", `docs/
piano_zoom_multiregione_disgiunto.md`).

## 6. Risultati sperimentali chiave (Video-MME, subset 250, seed 42)

| strategia | accuracy | note |
|---|---:|---|
| uniform (controllo) | 62.8% | baseline |
| fase sfasata sempre (gate su entropia, nessun ramo) | 66.0% | +3.2pt |
| zoom multi-regione forzato sempre | 64.0% | peggio della fase sfasata da solo, ma recupera 13 sample UNICI che l'altro non prende |
| oracolo (ramo giusto per sample) | ~68.8% (98/163 sui ricampionati) | limite superiore teorico del routing |
| router `top1_pct` a soglia adattiva (`ratio=3.85`) | **atteso ~68.8%, da verificare** | **Stage C, run in corso** |

## 7. Esperimento in corso (Stage C)

Run `qwen2_5_vl_3b_attn-stageC-router-concentration_ratio`:
`branch_selector=concentration_ratio`, `zoom_multi_region=true`, stesso
subset/seed 250. Obiettivo: verificare se il guadagno teorico
dell'oracolo (§4.1) si traduce in un guadagno reale quando il routing è
deciso da un segnale misurato (non dall'oracolo) — cioè se `top1_pct`
generalizza fuori dal sottoinsieme usato per calibrare la soglia in §4.3.
Se il numero regge (~68-69%), prossimo passo: run sull'intero dataset
(2700 sample). Se non regge, il segnale era "fortunato" sul campione di
calibrazione — da rivedere prima di scalare.
