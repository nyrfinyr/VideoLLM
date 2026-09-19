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

<div class="text-xs opacity-60 pb-2">
Pr(a &gt; τ | window_T) &gt; Pr(a &gt; τ | window_F) — <code>signals_512p</code>, 512 frame, 256 celle temporali, 69/100 sample usabili, mediana 2 celle vere per sample (caso top-1 = 1.6%)
</div>

<div class="text-sm pb-1"><b>Il ranking funziona</b> — AUC = Pr(cella dentro la finestra &gt; cella fuori); hit@k = la finestra è fra le k celle più attenzionate</div>

| rowset · massa | AUC | rango mediano | hit@1 | hit@5 | hit@10 | hit@25 |
|---|---:|---:|---:|---:|---:|---:|
| `all` raw | **0.775** | **10** | **17%** | 38% | **52%** | 67% |
| `all` sink-filtered | 0.758 | 11 | 17% | 36% | 49% | 61% |
| `question` raw | 0.742 | 13 | 16% | 36% | 48% | 62% |
| `last_token` raw | 0.746 | 25 | 4% | 10% | 25% | 51% |
| *caso* | *0.500* | *128* | *1.6%* | *7%* | *13%* | *26%* |

<div class="grid grid-cols-2 gap-6 pt-4">

<div>

<div class="text-sm pb-1"><b>Non è un artefatto</b></div>

| controllo | AUC | hit@1 |
|---|---:|---:|
| posizione | 0.552 | 1% |
| permutazione | 0.518 ±0.025 | 2.3% |

<div class="pt-1 text-xs opacity-60">
<b>posizione</b>: un profilo medio, identico per ogni sample — sa solo <i>dove</i> di solito va l'attenzione, non guarda il video. <b>permutazione</b>: il vettore d'attenzione di un altro sample, 200 giri.
</div>

</div>

<div class="text-sm">

<div class="pb-1"><b>Il test diretto</b></div>

hit@1 reale **12/69** contro **1.1** attesi dal caso — binomiale **p = 7.8·10⁻¹⁰**

</div>

</div>

<div class="pt-4" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:6px 12px;font-size:0.82rem">
La disuguaglianza vale ed è robusta: il bias di posizione spiega quasi nulla (AUC 0.552) e non basta un vettore d'attenzione qualsiasi (0.518), serve quello di <i>quel</i> video. Il segnale resta però <b>debole in assoluto</b>: col top-10 la finestra entra nel 52% dei sample contro il 13% del caso. <code>sink_filtered</code> peggiora su ogni rowset.
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

<div class="text-xs opacity-60 pb-2">
<b>H</b> = entropia (log2) della softmax <b>ristretta alle 4 lettere candidate</b>, al pass 1 — 0 bit = certezza assoluta, 2 bit = le quattro opzioni equiprobabili · <code>signals_512p</code>, <code>additive</code>, <code>entropy_shift_24</code>
</div>

<div class="text-sm pb-1"><b>AUROC(H)</b> = Pr( H di una risposta <b>sbagliata</b> &gt; H di una risposta <b>giusta</b> ), prese a caso una per gruppo; pareggi contati 0.5. <b>0.5 = H non distingue</b>, 1.0 = separazione perfetta.</div>

| dataset · run | n | accuracy | H mediana | 2^H | H se giusta | H se sbagliata | distanza | AUROC(H) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Video-MME · `entropy_shift_24` | 2700 | 55.0% | 0.826 | 1.77 di 4 | 0.566 | 1.119 | **0.553** | **0.746** |
| LVBench · `additive` pass 1 | 200 | 35.0% | 1.345 | 2.54 di 4 | 1.040 | 1.263 | 0.223 | 0.592 |
| LVBench · `signals_512p` | 100 | 43.0% | 1.389 | 2.62 di 4 | 1.112 | 1.220 | 0.108 | 0.547 |

<div class="pt-2 text-xs opacity-60">
<b>2^H</b> = fra quante delle 4 opzioni il modello sta di fatto esitando. <b>distanza</b> = quanto si separano le due entropie medie: è ciò che AUROC misura.
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

<div class="text-xs opacity-60 pb-2">
In attesa del full-set. Le righe sono le quattro misure appaiate che la run <code>lvbench-additive-full</code> produce sullo stesso sample.
</div>

<div class="text-sm">

| condizione | frame | n | tot | entity recog. | event underst. | key info retr. | reasoning | summariz. | temporal ground. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (pass 1) | 256 | 1548 | — | — | — | — | — | — | — |
| **k10** | 512 | 1548 | — | — | — | — | — | — | — |
| k5 | 512 | 1548 | — | — | — | — | — | — | — |
| **rand10** (controllo) | 512 | 1548 | — | — | — | — | — | — | — |

</div>

<div class="pt-4 text-sm pb-1"><b>Perché la tabella è vuota</b> — gli n per tipo di oggi, e quelli attesi sul full-set</div>

<div class="text-sm">

| tipo | n oggi (su 200) | 1 sample vale | n atteso (su 1548) | 1 sample varrà |
|---|---:|---:|---:|---:|
| entity recognition | 92 | 1.1 pp | ≈ 712 | 0.14 pp |
| event understanding | 72 | 1.4 pp | ≈ 557 | 0.18 pp |
| key information retrieval | 41 | 2.4 pp | ≈ 317 | 0.32 pp |
| reasoning | 29 | 3.4 pp | ≈ 224 | 0.45 pp |
| temporal grounding | 23 | **4.3 pp** | ≈ 178 | 0.56 pp |
| summarization | **8** | **12.5 pp** | ≈ 62 | 1.61 pp |

</div>

<div class="pt-2 text-xs opacity-60">
I <code>question_type</code> sono <b>multi-label</b>: un sample può contare in più tipi, quindi la colonna somma a più del numero di sample (265 su 200). Gli <b>n attesi</b> sono la proporzione di oggi riscalata a 1548, non una misura: gli shard sono strided su una lista shuffled, quindi il mix dei tipi resta lo stesso in attesa.
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

<div class="text-sm pb-1"><b>I canali esistono</b> — distribuzione di <b>|h[d]| / mediana(|h|)</b> sui token visivi, layer 7–21</div>

| canale | intervallo modale | quota &gt; 50× |
|---|---|---:|
| d1999 · candidato | 100–158× — 63.2% | **99.84%** |
| d1793 · candidato | 40–63× — 45.9% | **62.07%** |
| d1401 · candidato | 6–10× — 25.7% | 0.00% |
| d684 · controllo | 2–3× — 21.9% | 0.00% |
| d1316 · controllo | 3–4× — 22.7% | 0.00% |
| d1939 · controllo | 2–3× — 21.9% | 0.00% |

<div class="pt-1 text-xs opacity-60">
<b>h</b> = lo stato nascosto del token (2048 canali); <b>h[d]</b> è il valore del canale <i>d</i>. Il rapporto alla mediana degli altri canali dello <i>stesso</i> token rende confrontabili layer e sample: <b>1× = canale normale</b>.
</div>

<div class="pt-3 text-sm pb-1"><b>I token no</b> — |h[d]| / <b>media</b>(|h|), per gruppo di token</div>

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
| *atteso se fossero sink* | *&gt;1%* | *&gt;2%* | *&gt;5%* | *&gt;10%* | *&gt;25%* | *&gt;50%* |

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

# Il metodo · campionamento a coppie e arm additivo

<div class="grid grid-cols-2 gap-6 pt-1">

<div>

<div class="text-sm pb-1"><b>1 · Celle indirizzabili</b> — <code>utils/pair_sampling.py</code></div>

<div class="text-xs opacity-70 pb-2">
Qwen3-VL fonde i frame <b>a due a due</b> (<code>temporal_patch_size=2</code>): una cella d'attenzione è una coppia di frame <b>adiacenti nella lista</b>. Campionando uniformemente su LVBench (mediana 71 min) i due frame di una cella distano minuti — la cella non corrisponde a nessun istante.
</div>

<div class="pb-1" style="display:flex;gap:1px;align-items:flex-end">
<div v-for="i in 24" :key="'u'+i" style="width:11px;height:20px;border-radius:1px;background:#e3e6ea;border:1px solid #d0d4d9"></div>
</div>
<div class="text-xs opacity-60 pb-3">uniforme: le coppie cadono a caso nel tempo</div>

<div class="pb-1" style="display:flex;gap:7px;align-items:flex-end">
<div v-for="i in 8" :key="'p'+i" style="display:flex;gap:1px">
<div style="width:11px;height:20px;border-radius:1px;background:#c9ced4;border:1px solid #b6bcc3"></div>
<div style="width:11px;height:20px;border-radius:1px;background:#c9ced4;border:1px solid #b6bcc3"></div>
</div>
</div>
<div class="text-xs opacity-60">a coppie: centri uniformi <code>c_i = D·(i+0.5)/n</code>, due frame a <code>c_i ∓ gap/2</code> (gap = 2 s)</div>

<div class="pt-3 text-xs opacity-70">
Ogni cella <b>è</b> una coppia e il suo timestamp è <code>c_i</code>, lo stesso che il processor scrive nel prompt. Le celle diventano <b>indirizzabili</b>: «la cella 37 è al secondo 1024». È ciò che rende misurabile T1 e possibile il puntamento.
</div>

</div>

<div>

<div class="text-sm pb-1"><b>2 · Arm additivo top-k</b> — <code>strategies/additive_topk.py</code></div>

<div class="text-xs opacity-70 pb-2">
L'arm <b>sostitutivo</b> (<code>topk_resample</code>) è stato falsificato: rimpiazzare 512 frame con 128 butta più contesto di quanto la zoomata aggiunga — un miss costa −11.9 pp, un hit vale +23, e con hit@1 al 17% il conto è negativo. Qui il termine negativo <b>sparisce per costruzione</b>: la base resta, i frame mirati si <b>sommano</b>.
</div>

<div class="pb-1" style="display:flex;gap:1px;align-items:flex-end;height:26px">
<div v-for="i in 32" :key="'b'+i" style="width:8px;height:18px;border-radius:1px;background:#e3e6ea;border:1px solid #d0d4d9"></div>
</div>
<div class="text-xs opacity-60 pb-2">pass 1 · 256 frame base = 128 celle → ranking per massa d'attenzione</div>

<div class="pb-1" style="display:flex;gap:1px;align-items:flex-end;height:30px">
<div v-for="i in 32" :key="'a'+i" :style="`width:8px;height:18px;border-radius:1px;background:${[9,10,21].includes(i)?'#f5c6c6':'#e3e6ea'};border:1px solid ${[9,10,21].includes(i)?'#c0392b':'#d0d4d9'}`"></div>
</div>
<div class="pb-1" style="display:flex;gap:1px;align-items:flex-start;height:16px">
<div v-for="i in 32" :key="'x'+i" style="width:8px;display:flex;gap:0.5px;justify-content:center">
<div v-if="[9,10,21].includes(i)" v-for="j in 5" :key="j" style="width:1px;height:13px;background:#c0392b"></div>
</div>
</div>
<div class="text-xs opacity-60">pass additivo · 256 frame in più, solo dentro le top-k celle</div>

<div class="pt-3 text-xs opacity-70">
La regione di una cella è la sua <b>cella di Voronoi</b> <code>[D·i/n, D·(i+1)/n]</code>; celle adiacenti vengono <b>fuse</b> e il budget si divide in proporzione alla durata di ogni regione. Gli aggiunti sono <b>uniformi dentro la regione, non a coppie</b>: sul pass additivo non si rilegge nessun ranking. Un aggiunto che cade su un indice già presente viene scartato.
</div>

</div>

</div>

<div class="pt-3 grid grid-cols-2 gap-6">

<div class="text-xs">

**Il gate già passato** — probe oracolo, 92 sample, iso-budget 512 frame

| condizione | accuracy | Δ |
|---|---:|---:|
| base256 | 41.3% | — |
| uniform512 | 41.3% | **+0.0** |
| oracle512 | 54.3% | **+13.0 pp** |

</div>

<div class="text-xs opacity-70" style="padding-top:14px">
Raddoppiare i frame <b>uniformemente</b> vale esattamente zero: tutto il guadagno sta nel <i>dove</i>. L'arm sostituisce l'oracolo col puntatore vero.
<div class="pt-2">
⚠️ Nella lista unita il modello accoppia frame <b>adiacenti nella lista</b>: le coppie della base non sopravvivono, quindi le celle del pass additivo <b>non</b> sono quelle del pass 1. Accettabile finché lì non si rilegge l'attenzione.
</div>
</div>

</div>

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.12rem 0.35rem; }
.slidev-layout table { margin: 0.15rem 0; font-size: 0.7rem; }
</style>
