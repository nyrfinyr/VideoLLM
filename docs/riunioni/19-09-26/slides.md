---
theme: default
title: Risposte ai task del 02/09
info: |
  ## Risposte ai task del 02/09
  Una slide per punto del todo — 2026-09-19.
  Fonte: `docs/todo-02092026.md`, `docs/risultati-260919.xlsx`
class: text-center
highlighter: shiki
colorSchema: light
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Risposte ai task del 02/09

Qwen3-VL-2B su LVBench e Video-MME

<div class="pt-8 text-sm opacity-60">
Cinque punti su sette rispondibili dai dati già su wandb. Due richiedono il full-set.
</div>

---
layout: default
---

# 1 · L'attenzione trova la finestra?

<div class="text-xs opacity-60 pb-1">
Pr(a &gt; τ | window_T) &gt; Pr(a &gt; τ | window_F) — <code>signals_512p</code>, 512 frame, 256 celle, 69/100 sample usabili, mediana 2 celle vere (caso top-1 = 1.6%)
</div>

<div class="grid grid-cols-2 gap-5 pt-1">

<div>

<div class="text-sm pb-1"><b>Il ranking funziona</b></div>

| rowset · massa | AUC | hit@1 | hit@5 | hit@10 | hit@25 |
|---|---:|---:|---:|---:|---:|
| `all` raw | **0.775** | **17%** | 38% | **52%** | 67% |
| `all` sink-filt. | 0.758 | 17% | 36% | 49% | 61% |
| `question` raw | 0.742 | 16% | 36% | 48% | 62% |
| `last_token` raw | 0.746 | 4% | 10% | 25% | 51% |
| *caso* | *0.500* | *1.6%* | *7%* | *13%* | *26%* |

<div class="pt-2 text-xs">

| controllo | AUC | hit@1 |
|---|---:|---:|
| posizione (profilo medio) | 0.552 | 1% |
| permutazione (altro sample) | 0.518 ±0.025 | 2.3% |

</div>

<div class="pt-1 text-xs opacity-60">
hit@1 reale <b>12/69</b> contro 1.1 attesi — binomiale <b>p = 7.8e-10</b>
</div>

</div>

<div>

<div class="text-sm pb-1"><b>La soglia</b> — massa in multipli della quota uniforme</div>

| τ | Pr(a&gt;τ \| W_T) | Pr(a&lt;τ \| W_F) | J | celle/sample | precis. |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.836 | 0.305 | 0.141 | 179 | 1.9% |
| **0.93** | **0.615** | **0.672** | **0.286** | **85** | **2.9%** |
| 1.50 | 0.418 | 0.854 | 0.273 | 38 | 4.3% |
| 3.00 | 0.171 | 0.964 | 0.135 | 10 | 6.9% |
| 5.00 | 0.087 | 0.988 | 0.075 | 3 | 10.4% |
| *a caso* | | | | | *1.56%* |

<div class="pt-1 text-xs opacity-60">
275 celle vere contro 17.389 false. Il massimo di Youden (τ=0.93) tiene <b>85 celle su 256</b> con precisione 2.9%: J pesa sensibilità e specificità allo stesso modo, e con prevalenza 1.6% la specificità è quasi gratis.
</div>

<div class="pt-3 text-sm pb-1"><b>A parità di celle, soglia ≡ top-k</b></div>

| budget | precisione | recall | hit |
|---|---:|---:|---:|
| top-5 · soglia τ=4.19 | 9.6% · 9.9% | 12.0% · 12.4% | 38% · 39% |
| top-10 · soglia τ=2.98 | 7.2% · 7.0% | 18.2% · 17.5% | 52% · 52% |
| top-25 · soglia τ=1.87 | 5.0% · 4.9% | 31.3% · 30.9% | 67% · 65% |

</div>

</div>

<div class="pt-2" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:5px 12px;font-size:0.82rem">
Il segnale c'è ed è significativo. Soglia e top-k <b>ordinano ugualmente bene</b>: a fallire non è la soglia ma <b>Youden come criterio</b>, il cui ottimo tiene un terzo delle celle. Si sceglie il top-k perché <b>fissa il budget</b> su ogni sample, non perché ordini meglio. <code>sink_filtered</code> peggiora su ogni rowset.
</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.12rem 0.35rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.72rem; }
</style>

---
layout: default
---

# 2 · Il gate d'entropia

<div class="text-xs opacity-60 pb-1">
Entropia del pass 1 sulle 4 opzioni (0–2 bit). AUROC = quanto separa le risposte <b>sbagliate</b>; 0.5 = inutile.
</div>

<div class="text-sm pt-1">

| dataset · run | n | accuracy | H mediana | AUROC(H) |
|---|---:|---:|---:|---:|
| Video-MME · `entropy_shift_24` | 2700 | 55.0% | 0.826 | **0.746** |
| LVBench · `additive` pass 1 | 200 | 35.0% | 1.345 | 0.592 |
| LVBench · `signals_512p` | 100 | 43.0% | 1.389 | 0.547 |

</div>

<div class="pt-3 text-sm pb-1"><b>Tabella entropia → risposta corretta</b>, per quantile di H</div>

<div class="grid grid-cols-2 gap-5">

<div>

<div class="text-xs pb-1">Video-MME (n = 2700)</div>

| quantile | soglia H | acc sotto | acc sopra | Δ |
|---:|---:|---:|---:|---:|
| 0.1 | 0.002 | **96.7%** | 50.4% | **+46.3** |
| 0.2 | 0.031 | 90.9% | 46.0% | +44.9 |
| 0.3 | 0.165 | 85.8% | 41.8% | +44.0 |
| 0.5 | 0.825 | 73.0% | 37.0% | +35.9 |
| 0.7 | 1.316 | 64.2% | 33.5% | +30.7 |
| 0.9 | 1.741 | 57.6% | 31.9% | +25.7 |

</div>

<div>

<div class="text-xs pb-1">LVBench (n = 200)</div>

| quantile | soglia H | acc sotto | acc sopra | Δ |
|---:|---:|---:|---:|---:|
| 0.1 | 0.064 | 65.0% | 31.7% | +33.3 |
| 0.2 | 0.572 | 55.0% | 30.0% | +25.0 |
| 0.3 | 0.887 | 45.0% | 30.7% | +14.3 |
| 0.5 | 1.340 | 41.0% | 29.0% | **+12.0** |
| 0.7 | 1.649 | 37.9% | 28.3% | +9.5 |
| 0.9 | 1.845 | 36.7% | 20.0% | +16.7 |

</div>

</div>

<div class="pt-3" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:5px 12px;font-size:0.82rem">
Il segnale <b>non si trasferisce</b>: su LVBench il salto esiste solo nel primo quintile e sparisce in mezzo alla distribuzione. Le scale sono diverse (H mediana 0.83 vs 1.39), quindi <b>una soglia assoluta non porta</b>: si trasferisce il <b>quantile</b>, scelto out-of-fold e raggruppato per video.
</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.14rem 0.4rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.74rem; }
</style>

---
layout: default
---

# 3 · Accuracy per question type — LVBench

<div class="text-sm pt-2">

| arm | frame | n | tot | entity recog. | event underst. | key info retr. | reasoning | summariz. | temporal ground. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `signals_512p` | 512 | 100 | 43.0% | 53.8% | 38.2% | 46.7% | 28.6% | 25.0% | 50.0% |
| `topk` | 512 | 100 | 45.0% | 53.8% | 44.1% | 53.3% | 28.6% | 25.0% | 50.0% |
| `additive` pass 1 | 256 | 200 | 35.0% | 39.1% | 26.4% | 41.5% | 31.0% | 37.5% | 30.4% |

</div>

<div class="pt-4 text-sm pb-1"><b>Gli n per tipo</b> — il motivo per cui la tabella sopra non si può leggere</div>

<div class="text-sm">

| tipo | n su 100 | n su 200 | 1 sample vale |
|---|---:|---:|---:|
| entity recognition | 52 | 92 | 1.1 pp |
| event understanding | 34 | 72 | 1.4 pp |
| key information retrieval | 15 | 41 | 2.4 pp |
| reasoning | 14 | 29 | 3.4 pp |
| temporal grounding | 8 | 23 | **4.3 pp** |
| summarization | **4** | **8** | **12.5 pp** |

</div>

<div class="pt-3" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:5px 12px;font-size:0.82rem">
Quattro tipi su sei hanno n &lt; 30. Con 4 sample <code>summarization</code> si muove di 25 punti per ogni risposta che cambia: 0.250 e 0.375 non sono misure. <b>È il punto che ha più bisogno del full-set.</b>
</div>

<div class="pt-2 text-xs opacity-60">
Le tre run non sono confrontabili fra loro: frame e sample diversi. <code>additive</code> è il <b>pass 1</b>, cioè la baseline senza frame aggiunti.
</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.16rem 0.4rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.76rem; }
</style>

---
layout: default
---

# 4 · I token «sink»

<div class="text-xs opacity-60 pb-1">
<code>signals_512p</code>, 100/100 sample, 28 layer, hidden 2048. Candidati <code>1793 · 1999 · 1401</code> contro controlli a caso <code>684 · 1316 · 1939</code>.
</div>

<div class="grid grid-cols-2 gap-5 pt-1">

<div>

<div class="text-sm pb-1"><b>I canali esistono</b> — distribuzione di |h[d]|, layer 7–21</div>

| canale | intervallo modale | quota \|h\|&gt;50 |
|---|---|---:|
| d1999 · candidato | \[100, 158) — 63.2% | **99.84%** |
| d1793 · candidato | \[40, 63) — 45.9% | **62.07%** |
| d1401 · candidato | \[6, 10) — 25.7% | 0.00% |
| d684 · controllo | \[2, 3) — 21.9% | 0.00% |
| d1316 · controllo | \[3, 4) — 22.7% | 0.00% |
| d1939 · controllo | \[2, 3) — 21.9% | 0.00% |

<div class="pt-3 text-sm pb-1"><b>I token no</b> — |h[d]| / media(|h|)</div>

| canale | token sink | token non-sink |
|---|---:|---:|
| d1999 | 58.91 | 47.08 |
| d1793 | 20.41 | 18.89 |
| d1401 | 2.45 | 2.25 |
| d684 · ctrl | 0.79 | 0.75 |

<div class="pt-1 text-xs opacity-60">
Il canale è acceso su <b>tutti</b> i token: è una proprietà del layer, non di un sottoinsieme.
</div>

</div>

<div>

<div class="text-sm pb-1"><b>La prova che chiude</b> — massa d'attenzione sui top-p% token per sink score</div>

| rowset | p=1% | p=2% | p=5% | p=10% | p=25% | p=50% |
|---|---:|---:|---:|---:|---:|---:|
| `all` | 0.8% | 1.6% | 3.7% | 7.2% | 18.4% | 39.2% |
| `question` | 0.7% | 1.4% | 3.3% | 6.5% | 16.8% | 37.0% |
| `last_token` | 1.1% | 2.1% | 4.6% | 8.8% | 22.3% | 47.2% |
| *atteso se pozzi* | *&gt;1%* | *&gt;2%* | *&gt;5%* | *&gt;10%* | *&gt;25%* | *&gt;50%* |

<div class="pt-2" style="display:flex;gap:16px;align-items:flex-end;height:82px;padding-left:6px">
<div v-for="(p, i) in [1,2,5,10,25,50]" :key="i" style="display:flex;flex-direction:column;align-items:center;gap:2px">
<div style="display:flex;gap:3px;align-items:flex-end;height:66px">
<div :style="`width:15px;height:${p*1.3}px;background:#d0d4d9;border-radius:1px`"></div>
<div :style="`width:15px;height:${[0.8,1.6,3.7,7.2,18.4,39.2][i]*1.3}px;background:#c0392b;border-radius:1px`"></div>
</div>
<div style="font-size:0.58rem;opacity:0.6;white-space:nowrap">p={{ p }}%</div>
</div>
</div>

<div class="text-xs opacity-60" style="padding-left:6px">
<span style="color:#98a0a8">&#9609;</span> quota attesa &nbsp;&nbsp; <span style="color:#c0392b">&#9609;</span> massa misurata &nbsp;&nbsp; sempre sotto la diagonale
</div>

<div class="pt-3 text-sm pb-1"><b>Heatmap</b> — nessuna struttura</div>

| mappa | intervallo | escursione |
|---|---|---:|
| temporale (256 celle) | 26.07 – 29.02 | 11% |
| spaziale (griglia 5×9) | 26.06 – 28.46 | 9% |

</div>

</div>

<div class="pt-2" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:5px 12px;font-size:0.82rem">
I <b>canali</b> outlier esistono, i <b>token</b> sink no: assorbono <i>meno</i> attenzione di quanta ne spetterebbe loro a caso. Non c'è un percentile giusto perché non c'è niente da filtrare.
</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.12rem 0.35rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.68rem; }
</style>

---
layout: default
---

# 6 · Accuracy per task type — Video-MME (2700 sample)

<div class="text-xs">

| modello · arm | tot | act.reas | act.recog | attr.perc | counting | info.syn | obj.reas | obj.recog | ocr | spat.perc | spat.reas | temp.perc | temp.reas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4B** `baseline_24` | **58.9%** | 49.5 | 55.3 | 71.6 | **42.2** | 74.6 | 55.5 | 65.5 | 61.2 | 70.4 | 73.2 | 65.5 | **44.1** |
| 2B `baseline_24` | 54.4% | 44.2 | 53.0 | 66.2 | 38.4 | 73.1 | 49.3 | 61.6 | 58.3 | 63.0 | 66.1 | 61.8 | 36.2 |
| 2B `entropy_shift_24` | 55.0% | 46.3 | 54.3 | 67.6 | 37.3 | 71.8 | 50.0 | 62.4 | 62.6 | 61.1 | 64.3 | 54.5 | 37.9 |
| 2B `visual_prompt_24` | 53.5% | 44.9 | 52.1 | 64.4 | 38.1 | 69.0 | 50.2 | 62.1 | 54.0 | 57.4 | 67.9 | 54.5 | 35.6 |
| 2B `visual_prompt_24_random` | 53.5% | 44.9 | 51.1 | 66.2 | 37.3 | 69.3 | 49.3 | 62.1 | 54.7 | 59.3 | 69.6 | 56.4 | 36.2 |
| 2B `marker_24` | 53.1% | 43.2 | 52.4 | 65.3 | 35.8 | 71.2 | 48.7 | 61.3 | 53.2 | 59.3 | 67.9 | 58.2 | 35.0 |
| 3B `baseline_24double` | 56.9% | 50.9 | 54.3 | 71.6 | 35.8 | 72.4 | 55.7 | 62.1 | 59.7 | 59.3 | 71.4 | 60.0 | 40.7 |
| 3B `baseline_24` | 55.0% | 53.0 | 51.8 | 65.3 | 33.6 | 71.2 | 54.0 | 57.9 | 59.7 | 61.1 | 67.9 | 63.6 | 39.0 |
| 3B `highlight_top1_24` | 54.6% | 51.9 | 50.2 | 64.0 | 39.2 | 71.2 | 52.4 | 57.9 | 58.3 | 61.1 | 71.4 | 54.5 | 37.3 |
| *n per tipo* | *2700* | *285* | *313* | *222* | *268* | *323* | *454* | *354* | *139* | *54* | *56* | *55* | *177* |

</div>

<div class="grid grid-cols-2 gap-5 pt-2">

<div class="text-sm">

| per durata | short | medium | long |
|---|---:|---:|---:|
| 4B `baseline_24` | **69.0%** | **56.8%** | **50.8%** |
| 2B `baseline_24` | 66.9% | 52.3% | 44.1% |
| 2B `entropy_shift_24` | 67.8% | 52.1% | 45.1% |
| 2B `marker_24` | 66.4% | 49.9% | 43.0% |

</div>

<div class="text-sm">

| intervento | Δ |
|---|---:|
| 2B `entropy_shift_24` vs baseline | +0.6 pp |
| 2B `visual_prompt_24` vs baseline | −0.9 pp |
| 2B `marker_24` vs baseline | −1.3 pp |
| **2B → 4B** | **+4.5 pp** |

</div>

</div>

<div class="pt-2" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:5px 12px;font-size:0.82rem">
Il profilo per task type è quasi <b>invariante all'arm</b>: le differenze fra righe sono molto più piccole di quelle fra colonne. <code>visual_prompt_24</code> e il suo controllo <code>random</code> fanno <b>53.5% entrambi</b>. Il salto è la capacità del modello, non l'intervento.
</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.1rem 0.25rem; }
.slidev-layout table { margin: 0.1rem 0; font-size: 0.6rem; }
</style>

---
layout: default
---

# 7 · Tabella completa dei risultati

<div class="text-xs opacity-60 pb-2">
<code>docs/risultati-260919.xlsx</code> — 101 run di eval lette da wandb, aggregate per shard in <b>60 esperimenti</b>. Solo numeri.
</div>

<div class="grid grid-cols-2 gap-5">

<div>

<div class="text-sm pb-1"><b>I quattro fogli</b></div>

| foglio | contenuto | righe |
|---|---|---:|
| `Risultati` | dataset · modello · arm · setting · frame · shard · n · corretti · accuracy · latency · job · data | 60 |
| `VideoMME_task_type` | 12 task type, solo full-set | 19 |
| `VideoMME_durata` | short / medium / long | 19 |
| `LVBench_question_type` | 6 question type + gli n | 3 |

<div class="pt-3 text-sm pb-1"><b>Copertura</b></div>

| progetto | run di eval |
|---|---:|
| Video-MME | 87 |
| LVBench | 12 |
| MVBench | 1 |
| EgoSchema | 1 |

</div>

<div>

<div class="text-sm pb-1"><b>Full-set, ordinati</b></div>

| dataset | modello · arm | frame | n | acc |
|---|---|---:|---:|---:|
| MVBench | 2.5-3B uniforme | 32 | 3748 | 64.8% |
| EgoSchema | 2.5-3B uniforme | 64 | 500 | 62.6% |
| Video-MME | 2.5-3B `entropy_attn` | 128 | 2700 | 60.2% |
| Video-MME | 2.5-3B uniforme | 128 | 2700 | 59.9% |
| Video-MME | **3-4B** `baseline_24` | 24 | 2700 | **58.9%** |
| Video-MME | 2.5-3B `baseline_24double` | 24 | 2700 | 56.9% |
| Video-MME | 3-2B `entropy_shift_24` | 24 | 2700 | 55.0% |
| Video-MME | 3-2B `baseline_24` | 24 | 2700 | 54.4% |
| Video-MME | 3-2B `visual_prompt_24` | 24 | 2700 | 53.5% |
| Video-MME | 3-2B `marker_24` | 24 | 2700 | 53.1% |
| LVBench | 2.5-3B uniforme | 768 | 1033 | 44.2% |

<div class="pt-2 text-xs opacity-60">
Su LVBench con Qwen3-VL-2B <b>non esiste nessuna run full-set</b>: tutte probe da 100–200 sample (<code>topk</code> 45.0%, <code>signals_512p</code> 43.0%, <code>baseline_48</code> 40.0%, <code>additive</code> pass 1 35.0%), non confrontabili fra loro.
</div>

</div>

</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.14rem 0.4rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.72rem; }
</style>

---
layout: default
---

# La run che chiude i due punti aperti

<div class="text-sm pt-1">

```bash
LIMIT=null GROUP=lvbench-additive-full sbatch --array=0-23 scripts/sbatch/lvbench_additive.sbatch
```

</div>

<div class="grid grid-cols-2 gap-5 pt-3">

<div>

<div class="text-sm pb-1"><b>Tre condizioni appaiate, un solo decode</b></div>

| condizione | cosa vede | metrica |
|---|---|---|
| baseline | 256 frame della base | `mcq_accuracy` |
| **k10** | + 256 nelle top-10 celle | `..._cond_k10` |
| k5 | + 256 nelle top-5 | `..._cond_k5` |
| **rand10** | + 256 in 10 celle **a caso** | `..._cond_rand10` |

<div class="pt-2 text-xs opacity-60">
<code>rand10</code> è il controllo che decide se l'arm è un arm. Già validato: centra la finestra nel <b>9%</b> dei casi contro il <b>57%</b> di <code>k10</code>.
</div>

<div class="pt-3 text-sm pb-1"><b>Costo</b></div>

| | |
|---|---:|
| sample | 1548 (full-set) |
| shard | 24 × ~65 sample |
| per shard | 1.98 h (limite 3 h) |
| totale | **47 GPU-h** |

</div>

<div>

<div class="text-sm pb-1"><b>Cosa chiude</b></div>

| punto | serve? | perché |
|---|---|---|
| 2 · gate entropia | no | 2700 + 300 sample già misurati |
| 4 · sink | no | la prova è netta su 100 sample |
| 6 · task type VMME | no | già full-set |
| 7 · tabella | no | consegnata |
| 1 · attenzione/finestra | in parte | il verso è solido, la soglia no |
| **3 · task type LVBench** | **sì** | 4 tipi su 6 con n &lt; 30 |
| **oracolo full-set** | **sì** | due probe, segni opposti |

<div class="pt-3 text-sm pb-1"><b>La contraddizione da sciogliere</b></div>

| probe | n | esito |
|---|---:|---|
| `probe_additive_oracle` | 92 | oracle512 **+13.0 pp** su uniform512 |
| `probe_oracle_ceiling` | 100 | oracle 30% contro baseline **32%** |

<div class="pt-2 text-xs opacity-60">
Nella stessa probe, <code>uniform512</code> = <code>base256</code> <b>esattamente</b> su entrambi gli shard: raddoppiare i frame uniformi vale zero, tutto il guadagno sta nel <i>dove</i>.
</div>

</div>

</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.14rem 0.4rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.72rem; }
</style>
