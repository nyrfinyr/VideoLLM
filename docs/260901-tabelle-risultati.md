# Tabelle risultati — Qwen3-VL su Video-MME

Solo dati. Continua `260803-tabelle_risultati.md`, che copre lo stesso filone
su **Qwen2.5-VL-3B** (3B parametri) e resta invariato. Qui il modello di
riferimento è **Qwen3-VL-2B** (2B): un terzo in meno di parametri,
generazione successiva. `§3` aggiunge la baseline di **Qwen3-VL-4B**, stessa
famiglia a scala doppia.

**Setup.** `qwen3_vl_2b_attn`, decoding greedy (`generation=precise`),
`seed=42`, `dataset.nframes=24`, Video-MME intero senza sottotitoli, 3 shard
SLURM da 900 sample — accuracy **pooled** (Σ corretti / Σ campioni). Su
Qwen3-VL i `video_metadata` reali sono sempre passati al processor: la
distinzione `meta=false`/`meta=true` di `260803 §3` non esiste qui.
Run: [`video_mme-qwen3vl2b`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-qwen3vl2b)
(baseline, job 92353) e [`video_mme-qwen3vl2b-ablation`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-qwen3vl2b-ablation)
(arm, job 92377 `entropy_shift`, 92844/92845 `marker`). Per il 4B:
[`video_mme-qwen3vl4b`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-qwen3vl4b)
(baseline, job 93328).

---

## 1. Arm su Qwen3-VL-2B (2700 sample)

Colonne come in `260803 §1`. `% resampling` = quota di sample su cui è
scattato il gate d'entropia al pass 1; `acc | high/low ans entropy` = accuracy
condizionata al gate — vuote per i `marker_*`, che girano con
`always_highlight=true` e quindi intervengono su tutti i sample, senza gate.
`entropy_shift_24` instrada nel ramo `phase_shift` il 100% dei sample
ricampionati (`concentration_threshold=101`), quindi le colonne di ramo di
`260803` qui non aggiungono nulla.

`marker_24` / `marker_random_24` = arm `attention_marker`
(`docs/piano_marcatori_frame_qwen3.md`; implementazione in
`docs/attention_marker_implementazione.md`): la cella di picco è racchiusa fra
`<START_IMPORTANT_FRAME>`/`<END_IMPORTANT_FRAME>` inseriti DENTRO il flusso
visivo, più un'istruzione fissa che li spiega. `top_k=1`,
`query_rows=question`, `sink_filter=false`; job 92844 (`cell_select=attention`)
e 92845 (`cell_select=random`).

⚠️ `query_rows`/`sink_filter` dei due `marker_*` differiscono dagli altri arm
di questa tabella e di `260803`, che classificano le celle CON filtro sink su
tutte le righe: un confronto `marker_*` contro `entropy_shift_24` mescolerebbe
due cambiamenti. Per i marcatori il confronto valido è quello **interno**,
`attention` contro `random`. La baseline resta comparabile con tutti perché
`uniform` non classifica celle.

| arm | pooled | Δ vs `baseline_24` | short | medium | long | % resampling | acc \| high ans entropy | acc \| low ans entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_24` | **54.44%** (1470/2700) | — (riferimento) | 66.89% (602/900) | 52.33% (471/900) | 44.11% (397/900) | — | — | — |
| `entropy_shift_24` | **55.00%** (1485/2700) | **+0.56** | 67.78% (610/900) | 52.11% (469/900) | 45.11% (406/900) | 53.70% (1450/2700) | 37.86% (549/1450) | 74.88% (936/1250) |
| `marker_24` (attention) | **53.04%** (1432/2700) | **−1.41** | 66.33% (597/900) | 49.56% (446/900) | 43.22% (389/900) | — | — | — |
| `marker_random_24` (random) | **53.11%** (1434/2700) | **−1.33** | 66.44% (598/900) | 49.89% (449/900) | 43.00% (387/900) | — | — | — |

Letture dirette:

- **`entropy_shift_24`: +0.56 pp = 15 sample su 2700**, e la N effettiva è
  1450, non 2700: sui 1250 sample con gate chiuso l'arm restituisce la
  risposta del pass 1, cioè quella della baseline. Il z non appaiato vale
  +0.41 (p ≈ 0.68); il test appaiato sui 1450 ricampionati richiede le righe
  Weave, non i summary.
- il salto 37.86% → 74.88% fra gate aperto e chiuso è **effetto di selezione**
  (il gate manda al pass 2 i sample intrinsecamente più difficili), non una
  misura di efficacia del ricampionamento — stessa lettura di `260803 §1`.
- **`marker_24` − `marker_random_24` = −2 sample su 2700 (−0.07 pp)**: i due
  arm sono indistinguibili. Non è mancata applicazione — `marked` 900/900 su
  tutte e sei le eval, e le celle scelte dai due criteri differiscono nel
  35–40% dei sample (`peak_cell` vs `peak_cell_sink_filtered`). **Marcare il
  frame "giusto" non vale più che marcarne uno a caso**: il segnale
  d'attenzione non è ciò che manca al modello.
- **entrambi i `marker_*` stanno sotto la baseline** di 1.3–1.4 pp. La SE non
  appaiata su una differenza fra proporzioni a n=2700 vale ≈1.36 pp, quindi il
  calo è ~1 SE: suggestivo, non stabilito. Il test appaiato contro
  `baseline_24` non è fattibile — `uniform` non ri-emette campi per-sample,
  quindi le sue Evaluation Weave non sono attribuibili ai singoli shard.
- **costo dei `marker_*` ~3.2× la baseline** (10.7 contro 3.1–3.4 s/sample):
  con `always_highlight=true` ogni sample paga estrazione PNG di tutti i 24
  frame + secondo pass, nessuno è esente. Il calo si concentra su **`medium`**
  (52.33 → 49.56, −25 sample); `short` e `long` si muovono di meno di 1 pp.
- correttezza dei `marker_*` verificata sulle 6 Evaluation Weave distinte
  (5400 sample): `marked` 900/900, `n_query_rows < n_query_tokens` 900/900,
  blocchi pari con `somma(block_sizes)/2 == t_cells` 900/900, `pred_fallback`
  4/5400 (0.07%).
- **esito dei marcatori: arm chiuso.** Il controllo pareggia il treatment,
  quindi non è la scelta della cella a essere sbagliata ma il canale: non ha
  senso spendere GPU su varianti dei knob (`sink_filter`, `top_k`) dello
  stesso intervento.

---

## 2. Stesso arm sui due modelli

`entropy_shift_24` con knob identici (`entropy_threshold=0.7`,
`branch_selector=concentration`, `concentration_threshold=101`), stesso
benchmark, stesso `nframes`. Righe Qwen2.5 da `260803 §1`.

| | Qwen2.5-VL-3B | Qwen3-VL-2B |
|---|---:|---:|
| parametri | 3B | **2B** |
| `baseline_24` pooled | 55.04% (1486/2700) | 54.44% (1470/2700) |
| `entropy_shift_24` pooled | 55.63% (1502/2700) | 55.00% (1485/2700) |
| **Δ arm − baseline** | **+0.59** | **+0.56** |
| % resampling | 73.15% (1975/2700) | **53.70%** (1450/2700) |
| acc \| high ans entropy | 43.59% | 37.86% |
| acc \| low ans entropy | 88.41% | 74.88% |
| latenza baseline (solo shard L40S) | 3.77 s | 3.32 s |
| latenza arm (solo shard L40S) | 6.36 s | 8.97 s |
| costo arm / baseline | 1.69× | **2.70×** |
| sovraccosto per sample ricampionato | 3.54 s | **10.5 s** |

1. **Il Δ è replicato quasi esattamente (+0.59 → +0.56)**: il nulla misurato
   su Qwen2.5 non era una proprietà di quel modello. Vale anche per le
   baseline — il 2B sta a −0.60 pp dal 3B con un terzo dei parametri in meno.
2. **Il gate non si trasferisce.** A soglia invariata si apre sul 53.70% dei
   sample invece che sul 73.15%, e separa meno (37.86/74.88 contro
   43.59/88.41). I due arm **non sono compute-matched fra modelli**: per
   appaiare la frazione ricampionata la soglia va ricalibrata su Qwen3, non
   riusata.
3. **Il costo relativo è più alto nonostante meno sample al pass 2.** Le
   latenze sono a parità di GPU (solo shard L40S; su Qwen2.5 lo shard 0 girava
   su Quadro RTX 6000 ed è escluso). I forward di base sono quasi uguali, la
   differenza sta tutta nell'intervento: `6.36 = 3.77 + 0.7315·x → x = 3.54 s`
   contro `8.97 = 3.32 + 0.5370·x → x = 10.5 s`, cioè **~3× per sample
   ricampionato**. Causa non misurata (cattura d'attenzione o estrazione PNG
   del ramo `phase_shift`): da profilare prima di altri arm signal-driven su
   Qwen3.

### Breakdown per task type

In grassetto ogni arm sopra la **propria** baseline. `Δ` in pp; a queste N
(cinque categorie sotto i 180 sample, dove 2 pp è un sample solo) conta il
segno, non l'ampiezza.

| task type | n | Qwen2.5-3B `baseline_24` | Qwen2.5-3B `entropy_shift_24` | Δ | Qwen3-2B `baseline_24` | Qwen3-2B `entropy_shift_24` | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| action reasoning | 285 | 52.98 | **53.33** | +0.35 | 44.21 | **46.32** | +2.11 |
| action recognition | 313 | 51.76 | 51.12 | −0.64 | 53.04 | **54.31** | +1.28 |
| attribute perception | 222 | 65.32 | **69.37** | +4.05 | 66.22 | **67.57** | +1.35 |
| counting problem | 268 | 33.58 | **37.69** | +4.10 | 38.43 | 37.31 | −1.12 |
| information synopsis | 323 | 71.21 | 68.73 | −2.48 | 73.07 | 71.83 | −1.24 |
| object reasoning | 454 | 53.96 | **55.73** | +1.76 | 49.34 | **50.00** | +0.66 |
| object recognition | 354 | 57.91 | **59.04** | +1.13 | 61.58 | **62.43** | +0.85 |
| ocr problems | 139 | 59.71 | 58.27 | −1.44 | 58.27 | **62.59** | **+4.32** |
| spatial perception | 54 | 61.11 | **62.96** | +1.85 | 62.96 | 61.11 | −1.85 |
| spatial reasoning | 56 | 67.86 | **71.43** | +3.57 | 66.07 | 64.29 | −1.79 |
| temporal perception | 55 | 63.64 | 56.36 | −7.27 | 61.82 | 54.55 | **−7.27** |
| temporal reasoning | 177 | 38.98 | 36.72 | −2.26 | 36.16 | **37.85** | +1.69 |
| **↑ / ↓ vs baseline** | | | | **7 / 5** | | | **7 / 5** |

- **Il segno concorda su 6 categorie su 12**: appena sopra il caso. Il Δ
  pooled quasi identico fra i due modelli non nasce dallo stesso pattern per
  task, ma da compensazioni diverse.
- **`temporal perception` è l'unica categoria che perde su entrambi**, con lo
  stesso Δ (−7.27 pp, −4 sample): la categoria che dovrebbe beneficiare di più
  del ricampionamento temporale è quella che peggiora di più. Su Qwen2.5 era
  0 arm su 9 sopra la baseline (`260803 §1b`), qui il segno si ripete.
- **`counting problem` e `ocr problems` si invertono fra i due modelli**
  (+4.10 → −1.12 e −1.44 → +4.32). Su Qwen2.5 `counting` era l'unica categoria
  dove tutti e 9 gli arm guadagnavano e `ocr` quella dove 7 su 9 perdevano:
  nessuna delle due regolarità sopravvive al cambio di modello.

---

## 3. Baseline Qwen3-VL-4B (2700 sample)

Stessa famiglia, scala doppia: `qwen3_vl_4b_attn` = `Qwen/Qwen3-VL-4B-Instruct`.
Tutto il resto è invariato rispetto al setup in testa al documento (greedy,
`seed=42`, `nframes=24`, 3 shard da 900, nessun sottotitolo). Le tre shard
hanno girato **tutte su L40S**, quindi anche le latenze sono confrontabili
senza correzioni.

Check di correttezza (driver `/wandb`, per shard): `mcq_accuracy` presente nel
summary su 3/3, `pred_fallback` 0/900 su 3/3, nessuna run troncata dalla
walltime (0.88–0.89 h osservate su 4 h richieste).

| shard | corretti | acc | s/sample |
|---|---:|---:|---:|
| 0 | 532/900 | 59.11% | 3.50 |
| 1 | 536/900 | 59.56% | 3.55 |
| 2 | 521/900 | 57.89% | 3.58 |
| **pooled** | **1589/2700** | **58.85%** | 3.54 |

### 2B contro 4B, baseline contro baseline

| | Qwen3-VL-2B | Qwen3-VL-4B | Δ |
|---|---:|---:|---:|
| `baseline_24` pooled | 54.44% (1470/2700) | **58.85%** (1589/2700) | **+4.41** |
| short | 66.89% (602/900) | **69.00%** (621/900) | +2.11 |
| medium | 52.33% (471/900) | **56.78%** (511/900) | +4.44 |
| long | 44.11% (397/900) | **50.78%** (457/900) | +6.67 |
| latenza (shard L40S) | 3.32 s | 3.54 s | +0.22 s |

Letture dirette:

- **+4.41 pp = +119 sample su 2700**, z non appaiato = 3.27 (p ≈ 0.0011). È
  il primo Δ di questo documento che sta chiaramente fuori dal rumore: per
  confronto, tutti gli arm di `§1` si muovono entro ±1.4 pp, cioè ~1 SE.
- **il guadagno cresce monotonamente con la durata del video** (+2.11 short,
  +4.44 medium, +6.67 long). I `long` erano il punto debole del 2B (44.11%,
  22.8 pp sotto gli `short`); il 4B ne recupera 60, e la forbice
  short−long si stringe da 22.8 a 18.2 pp. Metà del guadagno pooled viene
  da lì.
- **il 4B è quasi gratis in latenza**: 3.54 contro 3.32 s/sample a parità di
  GPU, +6.6%. Il forward è dominato dai token visivi (24 frame), non dai
  parametri del decoder, quindi raddoppiare la scala non raddoppia il costo.
- **conseguenza operativa**: +4.41 pp per +6.6% di tempo è un rapporto che
  nessun arm testato finora si avvicina a produrre — `entropy_shift_24` paga
  2.70× il tempo per +0.56 pp, i `marker_*` pagano 3.2× per −1.4 pp. Il 4B
  diventa la baseline di riferimento per gli arm successivi; misurare
  interventi sul 2B rischia di ottimizzare margini che la scala assorbe.
- coerente con la probe da 60 (job 93325_0): 40/60 sul 4B contro 36 e 37/60
  delle due probe 2B sugli stessi sample.
