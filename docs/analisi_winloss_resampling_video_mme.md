# Win/loss del resampling — `entropy_attention_resample` vs baseline uniform (Video-MME)

> Analisi per-sample dello sweep soglie in `README.md` ("Sweep
> `entropy_attention_resample`"). Risponde alla domanda: quando il
> resampling scatta, quanti sample recupera davvero (sbagliato→giusto) e
> quanti ne rovina (giusto→sbagliato), rispetto a limitarsi al pass 1
> (baseline `uniform`)? Il delta netto in accuracy da solo nasconde questo
> bilancio.

## Metodologia

Join sample-per-sample tra il controllo `uniform`
([`ktvjcyye`](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye))
e ciascuna combinazione dello sweep, sulla chiave `example.id` del dataset
Video-MME (stabile tra run diversi, letta dagli input della call
`Evaluation.predict_and_score` su Weave — non l'ordine di esecuzione, che
con concorrenza non è garantito essere lo stesso tra run). Tutti e 250 i
sample di ogni combinazione si abbinano ai 250 del controllo (subset
identico, `shuffle=true seed=42 limit=250` invariati).

**Nota metodologica**: la mappatura run-wandb → call-Weave va fatta via il
campo `wb_run_id` della call (`client.get_call(id).wb_run_id`), non per
prossimità di timestamp — due o più run lanciati nello stesso minuto
(caso reale qui: `et0.5-ct45` e `et0.7-ct20`, lanciati alle 09:31:18-20
UTC del 2026-07-15) hanno timestamp di partenza della call troppo vicini
per essere distinti in modo affidabile senza `wb_run_id`. Un primo giro di
questa analisi aveva scambiato i due run; i numeri qui sotto sono stati
verificati con `wb_run_id` e sono quelli corretti.

## Le 3 combinazioni escluse (OOM totale, non nello sweep)

`et0.7-ct20`, `et0.7-ct45`, `et1.0-ct45` hanno **0/250 predizioni valide**
(`Evaluation.evaluate` → `mcq_accuracy: None`, ogni singola
`predict_and_score` con output `None`) — non entrano in questa analisi
perché non c'è nulla da confrontare. Vedi la causa nella tabella host/GPU
sotto: non è specifica a `concentration_threshold=45` (il README lo
riportava così in una versione precedente — corretto insieme a questo
file), è specifica all'host su cui è girato il job.

| combinazione | run | host | GPU | esito |
|---|---|---|---|---|
| et0.7-ct20 | [`lgs1jrxb`](https://wandb.ai/alesvale97-unimore/video_mme/runs/lgs1jrxb) | `nullazzo` | Quadro RTX 6000 24G | 0/250, OOM |
| et0.7-ct45 | [`tzhm99kn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/tzhm99kn) | `nullazzo` | Quadro RTX 6000 24G | 0/250, OOM |
| et1.0-ct45 | [`enb7u9ui`](https://wandb.ai/alesvale97-unimore/video_mme/runs/enb7u9ui) | `nullazzo` | Quadro RTX 6000 24G | 0/250, OOM |
| et1.0-ct20 | [`hmpib0w2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) | `huber` | RTX A5000 24G | ✅ 165/250 |
| et1.0-ct30 | [`igbsy8cp`](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) | `huber` | RTX A5000 24G | ✅ 165/250 |
| tutte le altre 8 (incl. `et0.3-ct45`, `et0.5-ct45`) | — | `bimbogigi`/`mereu`/`labetty`/`dracula`/`remigio` | L40S 45G | ✅ |

**3/3 dei fallimenti sono sull'host `nullazzo`** (Quadro RTX6000 24G),
indipendentemente da `et`/`ct` (`et0.3-ct45` ed `et0.5-ct45` girano bene
su L40S). `huber` (RTX A5000, anch'essa 24G) invece va bene in questo
batch — quindi non è "tutte le GPU 24G falliscono", è specificamente
`nullazzo` a fallire sempre in questo sweep. Il full eval
(`scripts/sbatch/video_mme-signal.sbatch`) è già pinnato a sole GPU 45G
(A40/L40S), quindi esclude comunque `nullazzo`.

## Le 9 combinazioni valide — riepilogo

`n_correct` include sia i sample non ricampionati (pass 1 accettato) sia
quelli ricampionati; `resampled` = quanti sample hanno superato il gate
`answer_entropy > entropy_threshold` (unico gate — vedi
`strategies/entropy_attention_resample.py`, `concentration_threshold`
decide solo lo STILE del resampling, non se farlo).

| combo | accuracy | resampled | fixed (❌→✅) | broken (✅→❌) | net | check |
|---|---:|---:|---:|---:|---:|---|
| uniform (controllo) | 62.80% (157/250) | — | — | — | — | — |
| et0.3-ct20 | 65.60% (164/250) | 77.6% (194/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.3-ct30 | 65.60% (164/250) | 77.6% (194/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.3-ct45 | 65.60% (164/250) | 77.6% (194/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.5-ct20 | 65.60% (164/250) | 71.2% (178/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.5-ct30 | 65.60% (164/250) | 71.2% (178/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.5-ct45 | 65.60% (164/250) | 71.2% (178/250) | 13 | 6 | +7 | 157+7=164 ✓ |
| et0.7-ct30 | 66.00% (165/250) | 65.2% (163/250) | 13 | 5 | +8 | 157+8=165 ✓ |
| et1.0-ct20 | 66.00% (165/250) | 59.2% (148/250) | 13 | 5 | +8 | 157+8=165 ✓ |
| et1.0-ct30 | 66.00% (165/250) | 59.2% (148/250) | 13 | 5 | +8 | 157+8=165 ✓ |

Le tre combinazioni con lo stesso `et` sono **byte-per-byte identiche**
(stesso set di sample fixed/broken, non solo lo stesso conteggio) — vedi
sotto: `concentration_threshold` non misura alcun effetto in {20,30,45}
su Video-MME.

### Breakdown completo (non-resampled + resampled) per combo

| combo | non-res. rimasti ✅ | non-res. rimasti ❌ | res. fixed | res. broken | res. invariati ✅ | res. invariati ❌ |
|---|---:|---:|---:|---:|---:|---:|
| et0.3-ct{20,30,45} | 54 | 2 | 13 | 6 | 97 | 78 |
| et0.5-ct{20,30,45} | 69 | 3 | 13 | 6 | 82 | 77 |
| et0.7-ct30 | 82 | 5 | 13 | 5 | 70 | 75 |
| et1.0-ct{20,30} | 93 | 9 | 13 | 5 | 59 | 71 |

## Scoperta chiave: gli stessi sample, sempre

Non è solo il conteggio a essere stabile — sono **gli stessi sample id**:

- **12 sample "fixed" in TUTTE e 9 le combinazioni**, qualunque `et`/`ct`:
  `103-1, 134-2, 538-2, 574-3, 584-2, 649-2, 670-3, 685-3, 842-1, 850-2,
  856-3, 877-1`.
- **5 sample "broken" in TUTTE e 9 le combinazioni**:
  `193-2, 405-3, 566-1, 576-3, 725-3`.
- Un 13° sample fixed (`747-2`) e un 6° sample broken (`413-2`) compaiono
  solo per `et ∈ {0.3, 0.5}` (e per `747-2` anche `et=0.7`), non a
  `et=1.0` — ai bordi esatti della soglia di entropia, plausibilmente
  sensibili a rumore numerico tra host/GPU diversi (vedi nota sotto)
  piuttosto che a un vero effetto della soglia.

**Implicazione pratica**: allargare il resampling da 148/250 (et=1.0,
59.2%) a 194/250 (et=0.3, 77.6%) — cioè **+46 sample in più ricampionati**
— non fixa né rompe NESSUNO di quei 46 sample aggiuntivi: sono tutti nel
bucket "invariato". Il nucleo di sample realmente "recuperabili" con
questo meccanismo (12-13 su 250, il 5%) sembra saturare immediatamente al
`et` più permissivo testato (1.0) e non cresce abbassando la soglia — così
come il nucleo di sample "a rischio di essere rovinati" (5-6 su 250).
Conferma quantitativamente quanto osservato nella call precedente: **non
c'è verifica dopo il resampling** (`strategies/entropy_attention_resample.py`,
nessun pass 3 che ricontrolla il segnale) — il meccanismo o becca il
sample giusto nel nucleo fisso, o non fa nulla, senza un modo per saperlo
in anticipo dal solo `et`/`ct`.

**Nota sul rumore hardware**: `et0.7-ct20` (fallito) ed `et0.5-ct45`
(riuscito) erano stati lanciati nello stesso minuto — la loro somiglianza
di timestamp è quello che ha causato lo scambio iniziale in questa
analisi. Più in generale, la frazione di sample ricampionati per lo stesso
`et` non era sempre riproducibile byte-per-byte tra host diversi in
run precedenti di questo sweep (osservato altrove: due run a `et=0.7`
su GPU diverse con frazioni di resampling diverse, 65.2% vs 71.2%,
probabilmente per differenze di floating-point tra architetture GPU
nell'attention capture) — un fattore di incertezza da tenere presente
prima di leggere differenze di 1-2 sample come "effetto reale" di una
soglia.

## Conclusioni per il full eval

- La combinazione scelta per il full eval (`et=0.7, ct=30` — vedi
  `scripts/sbatch/video_mme-signal.sbatch`) è nel gruppo a `net=+8`, il
  migliore osservato in questo sweep (pari a `et=1.0`, ma più economico in
  latenza).
- Il beneficio netto (+8/250 = +3.2 punti) nasconde un turnover reale di
  13 recuperati contro 5 rovinati — su 2700 sample dell'eval completo,
  aspettarsi un ordine di grandezza simile (proporzionalmente ~140 fixed,
  ~54 broken, extrapolando linearmente — **da verificare con i dati reali
  del full run**, questa è un'estrapolazione, non una misura).
- `concentration_threshold` resta un knob non informativo nel range
  20-45 su Video-MME (vedi anche la nota sulla distribuzione di
  `top1_pct`, README) — non vale la pena includerlo nel full eval come
  variabile, è fissato a 30 solo perché non fa differenza rispetto a
  20/45 in questo dataset.

## Dati reali del full eval 2700 (2026-07-17) — win/loss misurato per ramo × durata

> ⚠️ **Supera l'estrapolazione qui sopra** (~140 fixed / ~54 broken, net
> ~+86): era ottimistica. Il rapporto win/loss del subset (13:5 ≈ 2.6:1)
> NON scala — sul full è **144:136 ≈ 1.06:1**, praticamente in pareggio.

Join per-sample tra il run **router** (array `67691`, config
`concentration_ratio=3.85` + `zoom_multi_region`, 4 shard) e il run
**uniform** full di controllo
([`p7srmrkl`](https://wandb.ai/alesvale97-unimore/video_mme/runs/p7srmrkl),
2700), sulla chiave `example.id`. `win` = uniform sbaglia & router corregge
(❌→✅); `loss` = uniform corretto & router rovina (✅→❌). Nota: qui il
ramo dei gated è misto phase/zoom (non phase-alone) — è il run che
esisteva; il confronto phase-alone sul full resta da fare (vedi §8 della
sintesi, esperimento A).

**Gate-fire rate per durata** — il gate scatta di più sui video lunghi
(più incertezza dove la copertura temporale a 128 frame è più sparsa):

| durata | gated / tot | % |
|---|---:|---:|
| short | 465/900 | 51.7% |
| medium | 617/900 | 68.6% |
| long | 677/900 | 75.2% |

**Win/loss del ricampionamento vs uniform pass-1 (solo sui sample gated):**

| durata | ramo | n | uni acc | router acc | win | loss | net |
|---|---|---:|---:|---:|---:|---:|---:|
| short | phase_shift | 333 | .456 | .486 | 20 | 10 | **+10** |
| short | zoom | 132 | .561 | .523 | 12 | 17 | −5 |
| medium | phase_shift | 292 | .486 | .469 | 16 | 21 | −5 |
| medium | zoom | 325 | .452 | .443 | 34 | 37 | −3 |
| long | phase_shift | 266 | .425 | **.462** | 29 | 19 | **+10** |
| long | zoom | 411 | .399 | .401 | 33 | 32 | +1 |
| **ALL** | **phase_shift** | 891 | .457 | .474 | 65 | 50 | **+15** |
| **ALL** | **zoom** | 868 | .444 | .435 | 79 | 86 | **−7** |
| ALL | tutto | 1759 | .450 | .455 | 144 | 136 | **+8** |

**Due conclusioni:**

1. **`phase_shift` è debolmente positivo, `zoom` è morto.** phase net +15
   (positivo su short +10 e long +10, negativo su medium −5); zoom net −7,
   negativo/neutro ovunque. **La massa di attenzione usata come finestra
   temporale (Opzione C1/zoom multi-regione) non recupera nulla — anzi
   distrugge valore.** Filone chiuso.
2. **Niente è statisticamente significativo.** phase ALL net +15 su 115
   coppie discordanti → z≈1.4 (p≈0.16); phase long +10 → z≈1.4; il netto
   globale +8/1759 → z≈0.5, puro rumore. Il tetto del ricampionamento
   temporale è ~+1-2pt e solo per `phase_shift`, da confermare con un run
   phase-alone dedicato (con `pred_pass1` loggato — vedi sotto).

**Instrumentation aggiunta**: la strategy ora logga `pred_pass1` (argmax
softmax ristretta del pass 1) accanto al `pred` finale
(`strategies/entropy_attention_resample.py`) — ogni run futuro auto-riporta
il win/loss per-sample sul proprio dataset senza bisogno del join con un
run uniform di controllo separato.
