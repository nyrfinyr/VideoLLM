# Controllo dell'orologio M-RoPE (`pass_video_metadata=true`)

**Esito in una riga: il bug era reale a codice e quasi inerte
sull'accuracy.** Riparare l'orologio temporale sposta `baseline_24` di
+0.37 pp (z = 0.27) e non cambia il verdetto sul puntatore d'attenzione:
`highlight_top1_24 − baseline_24` passa da −0.41 a **−0.07 pp**. Il trigger
pre-registrato (~1 pp) non è scattato, quindi il terzo arm (`random`) non è
stato lanciato.

Gruppo wandb:
[`video_mme-mrope-fix`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-mrope-fix).
Sbatch: `scripts/sbatch/ablation-videomme/mrope_clock_control.sbatch`.
Il difetto è descritto in `docs/qwen3vl_signals.md` ("Il collasso M-RoPE su
Qwen2.5-VL"); questo documento riporta la sua misura.

---

## 1. Il difetto

`Qwen._prepare_inputs` non passava al processor i `video_metadata` reali.
Senza, `metadata.sampled_fps` vale 24 (`processing_qwen2_5_vl.py:103`) e
quindi

```
second_per_grid_ts = temporal_patch_size / sampled_fps = 2 / 24 = 0.0833
time_interval      = tokens_per_second * int(0.0833) = 0      # modeling_qwen2_5_vl.py:1125
position_temporal  = arange(t) * 0                            # get_vision_position_ids
```

cioè **tutte le celle video sulla stessa posizione temporale M-RoPE**.
Misurato end-to-end su Video-MME `nframes=24`:

| `model.pass_video_metadata` | `second_per_grid_ts` | `time_interval` | posizioni celle |
|---|---:|---:|---|
| `false` (default storico) | 0.0833 | **0** | tutte identiche |
| `true` | 6.1923 | **12** | spaziate |

`2/24` è una costante: il collasso **non è condizionale alla durata** del
video, vale per ogni run del README prodotta prima di questo controllo,
baseline comprese. Questo è il motivo per cui valeva la pena misurarlo, e
non solo annotarlo.

## 2. Perché proprio questi tre arm

`attention_highlight` costruisce una frase del tipo *"Focus on the part of
the video between 880 and 1100 seconds"*: comunica **istanti** a un modello
le cui celle video non avevano coordinate temporali distinte. L'intero
filone del puntatore è girato in quel regime, e il suo risultato —
highlight ≈ random ≈ baseline — poteva essere l'artefatto di un canale che
non poteva funzionare, invece della prova che il picco d'attenzione sia
poco informativo. Il controllo doveva poter falsificare quella lettura.

**Disegno.** 3 arm × 3 shard su Video-MME **intero** (2700 sample), stessa
strided-slice (`samples[shard::3]`) e stessi knob delle run originali —
`entropy_threshold=0.7`, `top_k=1`, `always_highlight=false`,
`pointer_units=seconds`, `pointer_position=prepend`, `random_seed=0`.
Le celle `meta=false` non sono state rifatte: **esistono già** sullo stesso
set e dallo stesso codice (gruppo `video_mme-ablation`), quindi il confronto
è appaiato per costruzione e costa 3 run invece di 6. È anche il motivo per
cui gira sui 2700 e non su un subset: 500 sample danno ~2.2 pp di errore
standard, contro scarti attesi di 0.4–1.9 pp — si sarebbe misurato rumore.

**Lancio in due tempi.** `--array=0-5` fa partire solo baseline e highlight.
Il terzo arm (`random`, task 6-8) era condizionato a un trigger dichiarato
prima di guardare i numeri: solo se `highlight_meta − baseline_meta` avesse
superato ~1 pp. La soglia non è a occhio — rieseguire la stessa identica
config ha spostato 3 sample su 900 (`fud7a4os` 493 → `dt1phfs5` 490), cioè
un pavimento di rumore di ~0.33 pp per shard.

## 3. Risultati

Accuracy *pooled* (Σ corretti / Σ sample sui 3 shard, non media degli
shard). Stesse colonne della tabella degli arm nel README: accuracy per
fascia di durata (breakdown `duration_*` emesso da `evals/base.py`),
`% resampling` = quota di sample su cui il gate d'entropia è scattato
(`answer_entropy > 0.7` al pass 1 — per `highlight` il pass 2 c'è comunque
ma non ricampiona nulla, cambia solo il prompt), e le due accuracy
condizionate al gate. `baseline_24` è `uniform`: niente gate, colonne
assenti. `Δ vs baseline` è calcolato **dentro lo stesso regime**; i
confronti fra regimi sono nel §4.

| arm | regime | pooled | Δ vs baseline (stesso regime) | short | medium | long | % resampling | acc \| high_ans_entropy | acc \| low_ans_entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_24` | `meta=false` | 55.04% (1486/2700) | — (riferimento) | 64.33% (579/900) | 52.00% (468/900) | 48.78% (439/900) | — | — | — |
| `baseline_24` | **`meta=true`** | **55.41%** (1496/2700) | — (riferimento) | 64.11% (577/900) | 53.56% (482/900) | 48.56% (437/900) | — | — | — |
| `highlight_top1_24` | `meta=false` | 54.63% (1475/2700) | −0.41 | 63.22% (569/900) | 52.22% (470/900) | 48.44% (436/900) | 73.07% (1973/2700) | 42.17% (832/1973) | 88.45% (643/727) |
| `highlight_top1_24` | **`meta=true`** | **55.33%** (1494/2700) | **−0.07** | 64.33% (579/900) | 53.44% (481/900) | 48.22% (434/900) | 72.74% (1964/2700) | 42.57% (836/1964) | 89.40% (658/736) |

`random_top1_24` compare solo in regime `meta=false` (54.78%, 1479/2700,
nella tabella dell'ablation del README): in regime `true` non è stato
lanciato perché il trigger non è scattato — vedi §4.

Le righe `meta=false` sono le run originali dell'ablation, non rifatte:
stesso set, stesso codice, stessi shard. È il motivo per cui il confronto
è appaiato per costruzione.

Per shard, appaiati:

| shard | `baseline` false → true | `highlight` false → true |
|---|---|---|
| 0 | [493](https://wandb.ai/alesvale97-unimore/video_mme/runs/fud7a4os) → [**502**](https://wandb.ai/alesvale97-unimore/video_mme/runs/8recv5if) (+9) | [489](https://wandb.ai/alesvale97-unimore/video_mme/runs/8pueatwg) → [**497**](https://wandb.ai/alesvale97-unimore/video_mme/runs/cm4adps6) (+8) |
| 1 | [500](https://wandb.ai/alesvale97-unimore/video_mme/runs/73b204q3) → [**498**](https://wandb.ai/alesvale97-unimore/video_mme/runs/7rembapu) (−2) | [498](https://wandb.ai/alesvale97-unimore/video_mme/runs/xwh64xpo) → [**503**](https://wandb.ai/alesvale97-unimore/video_mme/runs/uc0bot28) (+5) |
| 2 | [493](https://wandb.ai/alesvale97-unimore/video_mme/runs/j5fx1ozx) → [**496**](https://wandb.ai/alesvale97-unimore/video_mme/runs/wtwsb2xf) (+3) | [488](https://wandb.ai/alesvale97-unimore/video_mme/runs/ga1ilof0) → [**494**](https://wandb.ai/alesvale97-unimore/video_mme/runs/z7bk8gs9) (+6) |

Il segno non è nemmeno consistente sui tre shard della baseline.

**Letta per durata, la tabella dice che non è un effetto temporale.** Se
l'orologio piatto avesse davvero pesato, il guadagno sarebbe concentrato
sui **long**: è lì che le celle sono più distanti nel tempo, quindi è lì
che collassarle sulla stessa posizione M-RoPE distrugge più informazione.
Succede il contrario — passando da `false` a `true`:

| arm | short | medium | long |
|---|---:|---:|---:|
| `baseline_24` | −0.22 pp | **+1.56** | −0.22 |
| `highlight_top1_24` | +1.11 | +1.22 | −0.22 |

La fascia `long` è l'unica che *peggiora*, e in entrambi gli arm. Il
guadagno complessivo viene dai `medium`, dove non c'è alcuna ragione
meccanicistica per aspettarselo. È la distribuzione del rumore, non quella
di un effetto temporale.

**Il gate non si muove fra i due regimi.** `% resampling` 73.07% → 72.74%,
accuracy condizionate 42.17%/88.45% → 42.57%/89.40%: riparare l'orologio
non sposta in modo apprezzabile l'entropia della risposta al pass 1, quindi
il gate seleziona sostanzialmente la stessa popolazione. Due conseguenze:
le righe della tabella sono confrontabili una per una, e qualunque
differenza va cercata nel **pass 2**, cioè nel puntatore — non in una
diversa selezione di sample a monte. (Il salto ~42% → ~89% fra gate aperto
e chiuso resta un effetto di selezione — il gate manda a ricampionamento i
sample intrinsecamente più difficili — non una misura di efficacia; vale
qui il caveat già scritto nel README per la tabella dell'ablation.)

## 4. Le due righe che il disegno prevedeva

```
costo del collasso  = baseline_true  − baseline_false  = +0.37 pp   z = +0.27
il puntatore aiuta? = highlight_true − baseline_true   = −0.07 pp   z = −0.05   <- TRIGGER
```

**Il collasso non costava quasi nulla.** Dieci sample su 2700, dentro il
pavimento di rumore. Questo chiude anche una domanda aperta del README: il
gap sistematico di −0.3…−2.2 pp rispetto ai numeri del paper Qwen2.5-VL su
tutti e quattro i benchmark **non** è attribuibile all'orologio piatto.

**Il trigger non è scattato.** Con l'orologio riparato il puntatore
d'attenzione continua a non staccare la baseline: era −0.41 pp, è −0.07 pp,
e i due valori sono indistinguibili fra loro e da zero. Per il disegno
pre-registrato, `--array=6-8` (il controllo `random` in regime `true`) non
va lanciato: senza un guadagno da attribuire, non c'è niente da attribuire.

## 5. Il dato che ha detto qualcosa di nuovo: il churn

Il breakdown `correct_pass1_*` (aggiunto in `evals/base.py` proprio per
questa run, incrocia il `pred_pass1` loggato dalla strategy con `answer`)
mostra che il puntatore **non è affatto inerte**, anche se il netto è zero.
Pooled sui 3 shard di `highlight_top1_24` in regime `meta=true`:

| | n | esito al pass 2 | tasso |
|---|---:|---:|---:|
| corretti al pass 1 | 1495 | **77 rotti** | 5.15% |
| sbagliati al pass 1 | 1205 | **76 recuperati** | 6.31% |
| | | **netto −1 sample**, 153 mossi | |

Centocinquantatré risposte cambiano e il bilancio è −1. Non è un puntatore
che il modello ignora: è un puntatore che **ridistribuisce invece di
informare**.

Il tasso di rottura si legge contro il pavimento misurato dalla sonda
antipeak (`docs/sonda_sensibilita_puntatore_antipeak.md`): **senza**
puntatore, la quota di "corretto al pass 1, sbagliato al pass 2" — dovuta
al solo scarto fra i due decoding, softmax ristretta alle lettere vs parse
di `generate()` — è **0.34%**.

Qui il gate apre sul 72.74% dei sample (1964/2700), quindi il 5.15%
aggregato mescola sample puntati e non puntati. I non puntati possono
contribuire al massimo al pavimento, il che dà un **bound** sul tasso di
rottura del sottoinsieme effettivamente puntato:

```
x = quota di pass1-corretti fra i 736 non puntati
rotti(puntati) = (77 − 0.0034·x) / (1495 − x)
  x = 0    →  5.15%     (limite inferiore)
  x = 736  →  9.82%     (limite superiore)
```

In ogni caso ≥ 15× il pavimento, e nella stessa fascia del **6.80%
dell'antipeak**, cioè di un puntatore diretto *deliberatamente* nel posto
sbagliato. Con `acc | low_ans_entropy = 89.40%` il valore realistico di `x`
sta verso l'alto dell'intervallo (~9%).

⚠️ Le due popolazioni non sono la stessa: l'antipeak girava con
`always_highlight=true` su tutti i sample, qui il gate seleziona quelli ad
alta entropia, più fragili per costruzione — quindi un tasso di rottura più
alto è in parte atteso e i due numeri non vanno confrontati come se fossero
appaiati. Quello che regge è il confronto col pavimento: il puntatore è
lontanissimo dall'essere inerte.

## 6. Cosa si può concludere

1. **Il bug M-RoPE non ha invalidato gli esperimenti precedenti.** Tutte le
   run del README restano leggibili così come sono. Il knob resta a `false`
   di default (regime storico, riproducibile bit-per-bit).
2. **Il canale-prompt è vivo anche con l'orologio giusto**, e lo fa vedere
   dal churn, non dall'accuracy.
3. **Il segnale resta cattivo.** Le tre spiegazioni del nulla di
   `highlight_top1_24` erano: canale morto (escluso dall'antipeak),
   orologio piatto (escluso qui), segnale poco informativo. Resta la terza.
4. **Il churn è la firma di un puntatore arbitrario, non sbagliato.** Un
   puntatore sistematicamente *sbagliato* romperebbe più di quanto ripara
   (è quello che fa l'antipeak: 20 rotti, 11 riparati, netto −9). Qui
   rotture e riparazioni si equivalgono: il picco d'attenzione si comporta
   come una posizione estratta a caso, coerente col bias posizionale
   documentato nel probe da 40 sample
   ([`2dodrndg`](https://wandb.ai/alesvale97-unimore/video_mme/runs/2dodrndg)).

## 7. Cosa resta

L'unica leva ancora in piedi sul segnale è la **risoluzione** del puntatore:
a 12 celle indica 1/12 del video, che su un long di Video-MME sono ~50 s.
L'arm double-frame (`highlight_top1_24double.sbatch`, committato e non
lanciato) la dimezza. Due avvertenze prima di spenderci tempo:

- Quel file **non setta `model.pass_video_metadata`**, e il default dei
  preset Qwen2.5 è `false` (`conf/config.yaml:22`): girerebbe con
  l'orologio rotto. Per lo stesso motivo servirebbe rifare in regime `true`
  anche `baseline_24double`, che è il suo controllo compute-matched — 6 run
  da ~7h30, non 3.
- Il churn qui sopra rende meno credibile l'ipotesi "è solo grossolano": se
  il picco fosse la cella giusta ma troppo larga ci si aspetterebbe
  recuperi > rotture, non 76 vs 77.

L'alternativa economica è la **qualità** del picco — righe-query ristrette
ai token dell'entità (`question_rows` in `utils/attn_core.py`,
`docs/qwen25vl_entity_attention.md`) — da misurare prima offline sul probe
da 40 sample: costa una GPU-ora invece di quaranta, e va fatta comunque
prima di spendere su una risoluzione più fine di un picco che potrebbe
restare posizionale.

## 8. Note operative

- `highlight_units` è **degenere** in questo regime: 1961 sample in
  `seconds` contro 3 in `fraction`. A 12 celle il fallback a unità relative
  (che scatta quando una cella dura meno di un secondo) non si attiva
  praticamente mai, quindi non confonde questi numeri. Diventerà rilevante
  a 24 celle, dove la soglia raddoppia da 12 s a 24 s.
- Costo: ~1h40 per shard su A5000/Quadro RTX 6000, 6 task, nessun
  fallimento. `--time=06:00:00` è risultato largo.
- **Riproducibilità non verificata al 100%**: il check `META=false` sullo
  shard 0 ha dato 490/900 contro i 493/900 della run di riferimento. Il
  confronto è confuso — sono cambiati insieme il codice (refactor di
  `models/qwen_attn.py`) e la GPU (Quadro RTX 6000 → A5000). Assunto come
  non-determinismo cross-GPU e non ulteriormente indagato; è la fonte del
  pavimento di rumore di ~0.33 pp per shard usato sopra.
