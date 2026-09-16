# Quale quantile di entropia riaprire, in una strategy a gate

> Come si sceglie la soglia del gate d'entropia senza cablare un numero di bit,
> e perché la domanda giusta non è "quanto separa" ma "quanto rende".
> Strumento: [`scripts/choose_entropy_quantile.py`](../scripts/choose_entropy_quantile.py).

## 1. Il problema

Le strategy a gate (`entropy_attention_resample`, `coarse_to_fine`) interrogano
il modello una volta, misurano l'entropia `H` della risposta (softmax ristretta
alle lettere MCQ, `utils/attn_core.py::mcq_answer_stats`) e intervengono solo
dove `H` supera una soglia. La soglia è sempre stata **0.7 bit**, un numero
scelto una volta e mai ricalibrato.

**Una soglia in bit non si trasferisce.** A parità di 0.7 il gate si apre sul
73% dei sample con Qwen2.5-VL-3B e sul 53.7% con Qwen3-VL-2B
(`docs/260901-comments.md` §2): stesso benchmark, stessa soglia, due regimi
diversi. La distribuzione di `H` dipende dal modello, dalla difficoltà del
dataset, dal numero di opzioni (il massimo è `log2(n)`) e dal budget di frame.

Da qui la scelta: **la soglia si fissa per QUANTILE**, cioè si decide quale
frazione di sample riaprire e la soglia in bit è il quantile corrispondente su
quel dataset. Si trasferisce per costruzione, non richiede etichette, e rende
gli arm confrontabili a parità di costo. Resta la domanda: **quale quantile**.

## 2. La regola del pareggio

Riaprire un sample non è gratis in accuracy: il secondo pass ne recupera
qualcuno (sbagliato → giusto) ma ne rompe altri (giusto → sbagliato). Siano

- `r_fix` = frazione degli sbagliati riaperti che il pass 2 recupera,
- `r_break` = frazione dei giusti riaperti che il pass 2 rompe.

Un tratto di entropia con `W` risposte sbagliate e `R` giuste al pass 1 rende

    netto = r_fix·W − r_break·R  >  0   ⟺   W/R > r_break/r_fix
                                       ⟺   accuracy del tratto < p*

con

    p* = r_fix / (r_fix + r_break)

**Si riaprono i tratti la cui accuracy al pass 1 sta sotto `p*`.** Siccome
l'accuracy cresce al calare di `H`, questo equivale a scegliere un quantile.
Il quantile giusto non è una proprietà del segnale: dipende da `r_fix`/`r_break`,
cioè **dall'intervento**. Un intervento che rompe poco (l'oracolo `zoom` su
LVBench: +22/−6) alza `p*` e conviene applicarlo a più sample; uno che rompe
quanto recupera (i canali di marcatura: ~10% in entrambe le direzioni) ha
`p* ≈ 0.5` ma netto ≈ 0 ovunque.

**Assunzione**: `r_fix` e `r_break` costanti al variare di `H`. Va verificata,
non data per buona — lo strumento la mette alla prova (§4).

## 3. Cosa serve misurare

| quantità | da dove |
|---|---|
| `H` per sample al pass 1 | qualunque forward con cattura: `answer_entropy` |
| correttezza al pass 1 | `pred_pass1` della strategy, oppure la baseline `uniform` sugli stessi frame, unita per `example.id` |
| `r_fix`, `r_break` | **solo da una run con il pass 2**: correttezza prima e dopo, sugli stessi sample |

La run di segnali su LVBench (`strategies/signals_capture.py`) dà le prime due
colonne per tutti i 1549 sample, non la terza: non ricampiona. La curva del
guadagno va quindi misurata dall'arm di ricampionamento, quando esisterà.

Colonne attese dallo strumento (CSV):

    id, H, correct_pass1[, reopened, correct_pass2][, duration, task_type, question_type]

`reopened`/`correct_pass2` sono opzionali: senza, i tassi si passano a mano con
`--r-fix` / `--r-break` (anche in forma `k/n`, così l'incertezza è calcolata sui
conteggi veri).

## 4. Cosa produce lo strumento

```bash
uv run python scripts/choose_entropy_quantile.py campioni.csv --xlsx analisi.xlsx
uv run python scripts/choose_entropy_quantile.py campioni.csv --r-fix 22/68 --r-break 6/32
```

1. **Tabella per quantile**: soglia in bit, frazione riaperta, accuracy sopra e
   sotto, `W`/`R` riaperti, accuracy del tratto marginale, e — dove esiste il
   pass 2 — recuperati/rotti/netto osservato con IC bootstrap.
2. **AUROC** di `H` come predittore d'errore, complessiva e per gruppo: dice se
   il segnale separa, indipendentemente dalla soglia.
3. **Tassi per tratto di `H`** con IC bootstrap, più un **test di costanza**
   (trend e eterogeneità per permutazione). Se il test fallisce, la regola con
   un `p*` unico è distorta e va letta la variante a tassi locali.
4. **Raccomandazione**: `p*` con IC, il quantile scelto, la variante col costo
   (`λ` = netto minimo per forward aggiuntivo), la sensibilità agli estremi
   dell'IC e la stabilità del quantile al bootstrap.

## 5. Esito su Video-MME (Qwen3-VL-2B, 24 frame, `entropy_shift_24`)

Dati: 2700 sample, `H` e risposta finale dell'arm (job 92377) uniti alla
baseline (job 92353) per `example.id`; il pass 2 esiste solo sopra 0.7 bit,
quindi la curva osservata si ferma lì.

| | valore |
|---|---|
| AUROC di `H` | 0.756 (short 0.83, medium 0.74, **long 0.66**) |
| `r_fix` | 0.177 [0.153, 0.203] |
| `r_break` | 0.275 [0.239, 0.314] |
| `p*` | **0.391** [0.345, 0.437] |
| quantile scelto | **0.35** (H > 1.21 bit, riapre 945 sample) |
| netto previsto | +30.6 (+1.13 pp) |
| netto osservato a quel quantile | **+22** [−8, +51] |
| netto osservato alla soglia storica 0.7 (54% riaperti) | +15 |

Tre letture:

- **la soglia storica riapre troppo**: metà dei sample riaperti sta in tratti la
  cui accuracy è sopra `p*`, dove il pass 2 in media toglie invece di aggiungere;
- **il guadagno resta dentro il rumore**: l'IC del netto include lo zero. La
  regola sposta il quantile nella direzione giusta, non trasforma un arm nullo
  in un arm positivo;
- **`r_break` cresce con `H`** (p = 0.037): l'assunzione di costanza non regge
  del tutto, e i tratti più incerti sono anche i più fragili. Con tassi locali
  il quantile scelto sale a 0.40.

⚠️ `p*` = 0.391 sta **sotto** l'accuracy globale (0.544): con questo intervento
riaprire tutto costerebbe accuracy. È il motivo per cui la scelta del quantile
conta.

## 6. Come applicarla su LVBench

1. Dalla run di segnali: `H` e correttezza al pass 1 per i 1549 sample → tabella
   dei quantili e AUROC (lo strumento gira già senza pass 2).
2. Dal primo arm di ricampionamento: `r_fix`/`r_break`, possibilmente per tratto
   di `H` → `p*` e quantile.
3. Come stima provvisoria, i tassi dell'oracolo `zoom` (`docs/oracolo_lvbench.md`,
   100 sample: 22 recuperati su 68 sbagliati, 6 rotti su 32 giusti) danno
   `r_fix ≈ 0.32`, `r_break ≈ 0.19`, quindi `p* ≈ 0.63`: molto più alto di
   Video-MME, cioè converrebbe riaprire **quasi tutto**. Sono i tassi di un
   oracolo, non di un segnale reale: vanno rimisurati con l'arm vero, e servono
   come limite superiore.
