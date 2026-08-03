# Tabelle risultati — Video-MME (dall'ablation in poi)

Solo dati. Il razionale di ogni esperimento è in `README.md` e nei doc citati
riga per riga.

**Setup comune.** Modello `qwen2_5_vl_3b_attn` (Qwen2.5-VL-3B con cattura
attenzione), decoding greedy (`generation=precise`), `seed=42`,
`dataset.nframes=24`, Video-MME senza sottotitoli. Le run su fullset sono
job array a 3 shard SLURM da 900 sample: l'accuracy riportata è **pooled**
(Σ corretti / Σ campioni, non media degli shard). Salvo indicazione,
`model.pass_video_metadata=false` (`meta=false`).

---

## 1. Ablation entropy / shift / zoom — 7 arm, Video-MME intero (2700 sample)

Gruppo wandb [`video_mme-ablation`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-ablation) — 21 run (7 arm × 3 shard), tutte `finished`.
`Δ vs baseline_24`: tutti gli arm si confrontano con `baseline_24`, **non** con
`baseline_24double`. `% resampling` = quota di sample su cui è scattato il gate
d'entropia al pass 1; `acc | high/low ans entropy` = accuracy condizionata al
gate. `baseline_24`/`baseline_24double` sono `uniform`: nessun gate, colonne `—`.

| arm | pooled | Δ vs baseline_24 | short | medium | long | % resampling | acc \| high ans entropy | acc \| low ans entropy | ramo phase_shift | ramo zoom_peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_24` | **55.04%** (1486/2700) | — (riferimento) | 64.33% (579/900) | 52.00% (468/900) | 48.78% (439/900) | — | — | — | — | — |
| `baseline_24double` | **56.93%** (1537/2700) | **+1.89** | 68.22% (614/900) | 55.22% (497/900) | 47.33% (426/900) | — | — | — | — | — |
| `entropy_shift_24` | 55.63% (1502/2700) | +0.59 | 66.11% (595/900) | 53.00% (477/900) | 47.78% (430/900) | 73.15% (1975/2700) | 43.59% (861/1975) | 88.41% (641/725) | 100% (forzato) | — |
| `entropy_zoom_24` | 54.59% (1474/2700) | −0.44 | 63.33% (570/900) | 52.44% (472/900) | 48.00% (432/900) | 72.93% (1969/2700) | 42.10% (829/1969) | 88.24% (645/731) | — | 100% (forzato) |
| `shift_zoom_router_24` | 55.81% (1507/2700) | +0.78 | 66.00% (594/900) | 53.33% (480/900) | 48.11% (433/900) | 72.93% (1969/2700) | 43.93% (865/1969) | 87.82% (642/731) | 96.95% (1909/1969), acc 43.74% | 3.05% (60/1969), acc 50.00%\* |
| `highlight_top1_24` | 54.63% (1475/2700) | −0.41 | 63.22% (569/900) | 52.22% (470/900) | 48.44% (436/900) | 73.07% (1973/2700) | 42.17% (832/1973) | 88.45% (643/727) | — | — |
| `random_top1_24` | 54.78% (1479/2700) | −0.26 | 62.89% (566/900) | 53.11% (478/900) | 48.33% (435/900) | 73.00% (1971/2700) | 42.42% (836/1971) | 88.20% (643/729) | — | — |

\* N piccola (60 sample) — indicativo, non conclusivo.

Letture dirette:

- nessun arm signal-driven batte `baseline_24double` (56.93%), il controllo
  compute-matched;
- `highlight_top1_24` (−0.41) e il suo controllo casuale `random_top1_24`
  (−0.26) sono indistinguibili: **+0.15 pp a favore del controllo casuale**;
- il salto ~42% → ~88% fra gate aperto e chiuso è **effetto di selezione**
  (il gate manda al pass 2 i sample intrinsecamente più difficili), non una
  misura di efficacia del ricampionamento.

### 1b. Breakdown per task type (accuracy %, tutti gli arm)

Include, per confronto diretto, anche i tre arm delle sezioni successive
(colonne `meta=true` e `query_rows=question`).

**In grassetto ogni accuracy superiore a `baseline_24` nella stessa riga**
(i pareggi esatti non sono evidenziati). Ultima colonna: quanti dei **9 arm**
(tutti tranne `baseline_24`) stanno sopra / sotto la baseline in quella
categoria — è la sola lettura robusta a N piccola, perché conta i segni
invece di fidarsi dell'ampiezza dei singoli Δ.

| task type | n | baseline_24 | baseline_24double | entropy_shift_24 | entropy_zoom_24 | shift_zoom_router_24 | highlight_top1_24 | random_top1_24 | baseline_24 meta=true | highlight_top1_24 meta=true | highlight_top1_24 question | ↑/↓ vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| action reasoning | 285 | 52.98 | 50.88 | **53.33** | 49.47 | **54.04** | 51.93 | 51.58 | 50.53 | 50.88 | 51.93 | **2 / 7** |
| action recognition | 313 | 51.76 | **54.31** | 51.12 | **52.08** | 51.44 | 50.16 | 51.12 | **52.08** | 50.16 | 49.52 | 3 / 6 |
| attribute perception | 222 | 65.32 | **71.62** | **69.37** | **67.12** | **69.82** | 63.96 | 65.32 | 65.32 | **66.67** | **66.67** | 6 / 1 |
| counting problem | 268 | 33.58 | **35.82** | **37.69** | **37.31** | **36.94** | **39.18** | **38.43** | **37.31** | **38.81** | **35.07** | **9 / 0** |
| information synopsis | 323 | 71.21 | **72.45** | 68.73 | 70.90 | 69.04 | 71.21 | **72.76** | 71.21 | **71.83** | **72.76** | 4 / 3 |
| object reasoning | 454 | 53.96 | **55.73** | **55.73** | 52.64 | **55.95** | 52.42 | 51.32 | 53.08 | 53.08 | 53.08 | 3 / 6 |
| object recognition | 354 | 57.91 | **62.15** | **59.04** | 57.63 | **59.32** | 57.91 | 57.63 | **60.17** | **58.47** | **58.76** | 6 / 2 |
| ocr problems | 139 | 59.71 | 59.71 | 58.27 | **60.43** | 58.99 | 58.27 | 58.27 | 57.55 | 57.55 | 56.83 | **1 / 7** |
| spatial perception | 54 | 61.11 | 59.26 | **62.96** | **64.81** | 61.11 | 61.11 | 59.26 | **62.96** | **62.96** | **64.81** | 5 / 2 |
| spatial reasoning | 56 | 67.86 | **71.43** | **71.43** | 66.07 | **71.43** | **71.43** | **75.00** | **73.21** | **73.21** | **75.00** | 8 / 1 |
| temporal perception | 55 | 63.64 | 60.00 | 56.36 | 54.55 | 56.36 | 54.55 | 54.55 | 56.36 | 56.36 | 58.18 | **0 / 9** |
| temporal reasoning | 177 | 38.98 | **40.68** | 36.72 | 35.59 | 36.72 | 37.29 | 37.85 | **41.81** | **41.81** | **41.24** | 4 / 5 |

**Attenzione a N.** Le ultime quattro categorie hanno 54-177 sample: lì un Δ
di 2 pp è **un solo sample**, e lo spread fra i 7 arm dell'ablation arriva a
9.09 pp su `temporal perception` contro 2.16 pp su `ocr problems`. Solo la
colonna dei segni è leggibile a queste N.

**Cosa comunica la tabella:**

1. **`counting problem` è l'unica categoria dove tutti e 9 gli arm superano la
   baseline** (da +4 a +15 sample), ed è anche la categoria peggiore in
   assoluto (33.58%). Ma `highlight` (+15) e il suo controllo casuale
   `random` (+13) sono appaiati, e `baseline_24double` guadagna +6 **senza
   secondo passaggio**: quello che aiuta a contare è vedere di più / vedere
   due volte, non essere indirizzati.
2. **`temporal perception` è lo specchio: 0 arm su 9 sopra la baseline** (da
   −2 a −5 sample). Ogni intervento peggiora la categoria che dovrebbe
   beneficiarne di più. Con n=55 il segno è concorde ma l'ampiezza no —
   va presa come indizio, non come misura.
3. **`baseline_24double` vince dove serve risoluzione percettiva e vale zero
   su OCR** (83/139 identico alla baseline, l'unico pareggio esatto della sua
   colonna). Coerente col meccanismo: il frame-doubling raddoppia la
   granularità **temporale** delle celle, non la risoluzione **spaziale** dei
   frame — e OCR è l'unico task dove il collo di bottiglia è la seconda.
   Infatti `ocr problems` è anche la categoria dove 7 arm su 9 peggiorano:
   l'intero filone agisce sull'asse temporale, che lì non è il problema.
4. **Le categorie percettive separano "informazione nuova" da
   "ridirezione"**: su `attribute perception` gli arm che danno frame nuovi
   sono tutti positivi (double +14, shift +9, router +10, zoom +4) e i due
   arm-puntatore, che riusano gli stessi frame, sono a zero o negativi
   (−3 / 0). È la stessa distinzione che il pooled non mostra.
5. **Nessuna categoria isola il puntatore dal suo controllo casuale**:
   `highlight_top1_24` batte `random_top1_24` in 5 categorie su 12, perde in
   5, pareggia in 2, e lo scarto massimo è ±5 sample. Il nulla del pooled non
   nasconde una specializzazione per task.
6. **Il fix M-RoPE non si comporta come un effetto temporale.** Le due
   categorie temporali si muovono in direzioni opposte (`temporal reasoning`
   +5 sample sulla baseline, `temporal perception` −4), e i due arm rilanciati
   in `meta=true` si contraddicono categoria per categoria (`counting`: +10
   sulla baseline, −1 su highlight). Stessa conclusione della controprova per
   durata in §3: distribuzione da rumore.

---

## 2. Sonda `antipeak` — sensibilità del canale-prompt (subset 500 sample)

Gruppo wandb [`video_mme-antipeak-probe`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-antipeak-probe).
Subset `shuffle=true`, `seed=42`, `limit=500`, `meta=false`. **Non
confrontabile** con le tabelle su fullset (subset ≠ fullset). Doc:
`docs/sonda_sensibilita_puntatore_antipeak.md`.

| arm | intervento | corretti al pass 1 | rotti al pass 2 | acc \| corretti al pass 1 | sbagliati al pass 1 | recuperati al pass 2 | accuracy | run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `nopointer` (controllo) | nessuno (gate mai aperto) | 296/500 | **1** (0.34%) | **99.66%** (295/296) | 204 | 1 (0.49%) | 59.20% (296/500) | [`n0f2lrif`](https://wandb.ai/alesvale97-unimore/video_mme/runs/n0f2lrif) |
| `antipeak` | puntatore sulla cella più lontana dal picco | 294/500 | **20** (6.80%) | **93.20%** (274/294) | 206 | 11 (5.34%) | 57.00% (285/500) | [`c69hvw7w`](https://wandb.ai/alesvale97-unimore/video_mme/runs/c69hvw7w) |

```
danno       = 99.66% − 93.20% = 6.46 pp      z = 4.24, p < 0.0001
netto arm   = 11 recuperati − 20 rotti = −9 sample = −2.20 pp di accuracy
```

Il canale-prompt è vivo: il modello **esegue** il puntatore, e un puntatore
sistematicamente sbagliato rompe più di quanto ripara.

*Caveat*: i due arm sono girati su GPU diverse (L40S vs Quadro RTX 6000),
quindi il pass 1 non è identico bit-per-bit (296 vs 294 corretti) — irrilevante
a questa scala d'effetto, ma è il motivo dei denominatori diversi.

---

## 3. Controllo orologio M-RoPE — `pass_video_metadata=true` (2700 sample)

Gruppo wandb [`video_mme-mrope-fix`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-mrope-fix).
Stessi knob e stessi 3 shard degli arm della sezione 1, quindi **confrontabile
riga per riga**. `Δ vs baseline` è calcolato dentro lo stesso regime. Doc:
`docs/controllo_orologio_mrope.md`.

| arm | regime | pooled | Δ vs baseline (stesso regime) | short | medium | long | % resampling | acc \| high ans entropy | acc \| low ans entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_24` | `meta=false` | 55.04% (1486/2700) | — (riferimento) | 64.33% (579/900) | 52.00% (468/900) | 48.78% (439/900) | — | — | — |
| `baseline_24` | **`meta=true`** | **55.41%** (1496/2700) | — (riferimento) | 64.11% (577/900) | 53.56% (482/900) | 48.56% (437/900) | — | — | — |
| `highlight_top1_24` | `meta=false` | 54.63% (1475/2700) | −0.41 | 63.22% (569/900) | 52.22% (470/900) | 48.44% (436/900) | 73.07% (1973/2700) | 42.17% (832/1973) | 88.45% (643/727) |
| `highlight_top1_24` | **`meta=true`** | **55.33%** (1494/2700) | **−0.07** | 64.33% (579/900) | 53.44% (481/900) | 48.22% (434/900) | 72.74% (1964/2700) | 42.57% (836/1964) | 89.40% (658/736) |

```
costo del collasso  = baseline_true  − baseline_false  = +0.37 pp   z = +0.27
il puntatore aiuta? = highlight_true − baseline_true   = −0.07 pp   z = −0.05
```

**Controprova per durata** (se l'orologio piatto avesse pesato, il guadagno
sarebbe concentrato sui long, dove le celle sono più distanti nel tempo):

| Δ `meta=true` − `meta=false` | short | medium | long |
|---|---:|---:|---:|
| `baseline_24` | −0.22 | **+1.56** | −0.22 |
| `highlight_top1_24` | +1.11 | +1.22 | −0.22 |

La fascia long è l'unica che peggiora in entrambi gli arm: distribuzione da
rumore, non da effetto temporale.

`random_top1_24` **non** è stato rilanciato in regime `true`: il disegno lo
condizionava a un trigger dichiarato prima di guardare i numeri
(`highlight − baseline > ~1 pp`), che non è scattato.

---

## 4. `query_rows=question` — picco sulle sole righe della domanda (2700 sample)

Gruppo wandb [`video_mme-query-rows`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-query-rows),
`meta=true`, tutto il resto identico a `highlight_top1_24`. Il picco è
calcolato sulle sole righe della domanda anziché su tutte le righe-query
(domanda + 4 opzioni + istruzione + generation prompt: misurato col
tokenizer, il 76% delle righe non è la domanda).

| shard | run | GPU | accuracy |
|---|---|---|---|
| 0 | [`o8fitc6n`](https://wandb.ai/alesvale97-unimore/video_mme/runs/o8fitc6n) | Quadro RTX 6000 | 55.44% (499/900) |
| 1 | [`sgn94bbi`](https://wandb.ai/alesvale97-unimore/video_mme/runs/sgn94bbi) | Quadro RTX 6000 | 55.33% (498/900) |
| 2 | [`vwmqpuvn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/vwmqpuvn) | RTX A5000 | 54.78% (493/900) |
| **Pooled** | | | **55.19%** (1490/2700) |

| arm (`meta=true`) | pooled | Δ | short | medium | long | % resampling | acc \| high ans entropy | acc \| low ans entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `highlight_top1_24` `query_rows=all` | 55.33% (1494/2700) | — (riferimento) | 64.33% (579/900) | 53.44% (481/900) | 48.22% (434/900) | 72.74% (1964/2700) | 42.57% (836/1964) | 89.40% (658/736) |
| `highlight_top1_24` `query_rows=question` | 55.19% (1490/2700) | **−0.14** | 63.56% (572/900) | 53.44% (481/900) | 48.56% (437/900) | 72.89% (1968/2700) | 42.33% (833/1968) | 89.75% (657/732) |

Δ = −0.14 pp, cioè 4 sample su 2700: la stessa config rilanciata su un host
diverso ne sposta 3 su 900. Restringere il picco alle righe della domanda
**non** cambia il verdetto; il gate non si sposta (72.74% → 72.89%).

---

## 5. Riepilogo pooled (Video-MME intero, 2700 sample, `nframes=24`)

| arm | regime | pooled | Δ vs riferimento | riferimento |
|---|---|---:|---:|---|
| `baseline_24` | `meta=false` | 55.04% (1486/2700) | — | — |
| `baseline_24double` | `meta=false` | **56.93%** (1537/2700) | **+1.89** | `baseline_24` |
| `entropy_shift_24` | `meta=false` | 55.63% (1502/2700) | +0.59 | `baseline_24` |
| `entropy_zoom_24` | `meta=false` | 54.59% (1474/2700) | −0.44 | `baseline_24` |
| `shift_zoom_router_24` | `meta=false` | 55.81% (1507/2700) | +0.78 | `baseline_24` |
| `highlight_top1_24` | `meta=false` | 54.63% (1475/2700) | −0.41 | `baseline_24` |
| `random_top1_24` | `meta=false` | 54.78% (1479/2700) | −0.26 | `baseline_24` |
| `baseline_24` | **`meta=true`** | 55.41% (1496/2700) | +0.37 | `baseline_24` `meta=false` |
| `highlight_top1_24` | **`meta=true`** | 55.33% (1494/2700) | −0.07 | `baseline_24` `meta=true` |
| `highlight_top1_24` `query_rows=question` | **`meta=true`** | 55.19% (1490/2700) | −0.14 | `highlight_top1_24` `meta=true` |

L'unico Δ di ampiezza apprezzabile è `baseline_24double` (+1.89 pp): più token
visivi a parità di frame osservati. Tutti gli interventi signal-driven stanno
entro ±0.8 pp dalla baseline.

---

## 6. In corso / senza dati

| esperimento | stato |
|---|---|
| Baseline Qwen3-VL-2B su Video-MME (3 shard, gruppo [`video_mme-qwen3vl2b`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-qwen3vl2b)) | 3/3 run `crashed` il 2026-07-30 su `nullazzo` (Quadro RTX 6000), nessuna metrica loggata — da rilanciare su GPU ≥ 40 GiB |
| Arm double-frame del puntatore (`highlight_top1_24double.sbatch`) | committato, non lanciato (manca `model.pass_video_metadata=true`) |
| Diagnostica offline bias posizionale per insieme di righe-query (`attn_explorer/diag_query_rows.py`) | validata end-to-end su store locale da 4 video, non ancora girata su uno store rappresentativo |
