---
theme: default
title: Marcare o ricampionare — risultati
info: |
  ## Marcare o ricampionare
  Risultati per riunione — 2026-09-01.
  Fonte: `docs/260901-tabelle-risultati.md`, `docs/oracolo_lvbench.md`
class: text-center
highlighter: shiki
colorSchema: light
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Marcare o ricampionare

Qwen3-VL su Video-MME e LVBench

<div class="pt-6 text-xs opacity-60 leading-relaxed">
<code>qwen3_vl_2b_attn</code> · greedy (<code>generation=precise</code>) · <code>seed=42</code> · <code>nframes=24</code><br/>
Video-MME intero senza sottotitoli, 3 shard da 900 — accuracy <b>pooled</b> (Σ corretti / Σ campioni)<br/>
2026-09-01
</div>

---
layout: default
---

# 1 · Arm su Video-MME — Qwen3-VL-2B (2700 sample)

<div class="text-xs">

| arm | `query_rows` | `sink_filter` | pooled | Δ | short | medium | long |
|---|---|---|---:|---:|---:|---:|---:|
| `baseline_24` | — | — | **54.44%** (1470/2700) | — | 66.89% | 52.33% | 44.11% |
| `entropy_shift_24` | `all` | `true` | **55.00%** (1485/2700) | **+0.56** | 67.78% | 52.11% | 45.11% |
| `marker_24` (attention) | `question` | `false` | **53.04%** (1432/2700) | −1.41 | 66.33% | 49.56% | 43.22% |
| `marker_random_24` (random) | `question` | `false` | **53.11%** (1434/2700) | −1.33 | 66.44% | 49.89% | 43.00% |
| `visual_prompt_24` (attention) | `entity` | `true` | **53.48%** (1444/2700) | −0.96 | 65.89% | 51.44% | 43.11% |
| `visual_prompt_24_random` (random) | `entity` | `true` | **53.52%** (1445/2700) | −0.92 | 66.22% | 50.78% | 43.56% |

</div>

<div class="pt-2 text-sm">

| coppia treatment − controllo | Δ | in sample |
|---|---:|---:|
| `marker_24` − `marker_random_24` | −0.07 pp | **−2 / 2700** |
| `visual_prompt_24` − `visual_prompt_24_random` | −0.04 pp | **−1 / 2700** |

</div>

<div class="pt-2 text-sm">

**Dove vanno le risposte** — rispetto al pass 1, `visual_prompt`

| | rotti (dei corretti al pass 1) | recuperati (degli sbagliati) | netto |
|---|---:|---:|---:|
| `visual_prompt_24` (attention) | 156 / 1476 (10.6%) | 123 / 1223 (10.1%) | −33 |
| `visual_prompt_24_random` (random) | 151 / 1477 (10.2%) | 119 / 1223 (9.7%) | −32 |

</div>

<div class="pt-2 text-xs opacity-60">
SE della differenza: ±1.36 pp sul pooled, ±2.35 pp per fascia.
</div>

---
layout: default
---

# 2 · Scala del modello — baseline 2B vs 4B (2700 sample)

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

| | Qwen3-VL-2B | Qwen3-VL-4B | Δ |
|---|---:|---:|---:|
| pooled | 54.44% (1470/2700) | **58.85%** (1589/2700) | **+4.41** |
| short | 66.89% | **69.00%** | +2.11 |
| medium | 52.33% | **56.78%** | +4.44 |
| long | 44.11% | **50.78%** | **+6.67** |
| latenza (shard L40S) | 3.32 s | 3.54 s | +6.6% |

<div class="pt-2 text-xs opacity-60">
z = 3.27 (p ≈ 0.0011)
</div>

</div>

<div class="text-sm">

| intervento | Δ accuracy | costo |
|---|---:|---:|
| `entropy_shift_24` (2B) | +0.56 pp | 2.70× |
| `marker_*` (2B) | −1.4 pp | 3.2× |
| `visual_prompt_*` (2B) | −0.9 pp | 3.5× |
| **passare al 4B** | **+4.41 pp** | **1.07×** |

</div>

</div>

<div class="pt-6 text-sm">

**Stesso arm sui due modelli** (`entropy_shift_24`)

| | Qwen2.5-VL-3B | Qwen3-VL-2B |
|---|---:|---:|
| `baseline_24` → `entropy_shift_24` | 55.04% → 55.63% (**+0.59**) | 54.44% → 55.00% (**+0.56**) |
| % resampling | 73.15% | **53.70%** |
| costo arm / baseline | 1.69× | **2.70×** |
| sovraccosto per sample ricampionato | 3.54 s | **10.5 s** |

</div>

---
layout: default
---

# 3 · Oracolo su LVBench (100 sample, 24 frame)

<div class="text-sm pt-2">

| condizione | accuracy | Δ | appaiato | p (McNemar esatto) |
|---|---:|---:|---:|---:|
| `baseline` — 24 frame uniformi | 32% | — | — | — |
| `peak` — puntatore alla cella di picco | 29% | −3 | +4 / −7 | 0.55 |
| `oracle` — puntatore alla finestra **vera** | 30% | −2 | +5 / −7 | 0.77 |
| **`zoom`** — 24 frame **dentro** la finestra vera | **48%** | **+16** | **+22 / −6** | **0.0037** |

</div>

<div class="pt-6 text-sm">

| confronto diretto fra le due metà | appaiato | p |
|---|---:|---:|
| `zoom` − `oracle` (stessa informazione, due consegne) | **+23 / −5** | **0.0009** |

</div>

<div class="pt-6 text-xs opacity-60">
<code>qwen3_vl_2b_attn</code> · job 94348 · MCQ a 4 opzioni (livello di caso 25%) · dettagli in <code>docs/oracolo_lvbench.md</code>
</div>

---
layout: default
---

# 4 · Dove vive il guadagno di `zoom`

<div class="text-sm pt-2">

| | n | baseline | zoom |
|---|---:|---:|---:|
| finestra con **< 1** frame atteso nel campione uniforme | 77 | 27% | **49%** |
| finestra con **≥ 1** frame atteso | 23 | 48% | 43% |

<div class="pt-2 text-xs opacity-60">
Tutti e 22 i sample recuperati da <code>zoom</code> stanno nel primo gruppo.
</div>

</div>

<div class="pt-6 text-sm">

| regime LVBench | valore |
|---|---:|
| durata mediana del video | 71 min |
| finestra d'evidenza annotata, mediana | 30 s (**0.8%** del video) |
| frame attesi dentro la finestra con 24 frame uniformi (mediana) | **0.20** |
| sample con < 1 frame atteso | **77 / 100** |
| copertura della finestra — 24 frame / 48 frame | 45% / 51% |

</div>

---
layout: default
---

# 5 · Il segnale d'attenzione, misurato senza accuracy

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

**Hit = la cella di picco interseca la finestra annotata** (job 93613, 60 sample)

| rowset | hit | livello di caso |
|---|---:|---:|
| `all` | 28.3% | 15.8% |
| `question` | 28.3% | 15.8% |
| `entity` | 30.0% | 15.8% |

<div class="pt-2 text-xs opacity-60">
Su 100 sample: 31% contro 18.8% atteso, <b>z = 4.21</b>.<br/>
Sui 22 sample recuperati da <code>zoom</code>: <b>4 / 22 (18%)</b>.
</div>

<div class="pt-4">

**Distanza picco → finestra** (100 sample, job 94348)

| distanza (celle) | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| sample | 31 | 10 | 13 | 5 | 3 | 38 |

</div>

</div>

<div class="text-sm">

**Allargare la finestra attorno al picco**

| finestra | % del video | copre | caso | lift | frame nell'evidenza |
|---|---:|---:|---:|---:|---:|
| ±0 (solo picco) | 8% | 31% | 18% | **1.68×** | **2.4** |
| ±1 | 25% | 41% | 32% | 1.27× | 0.8 |
| ±2 | 42% | 54% | 45% | 1.21× | 0.5 |
| ±3 | 58% | 59% | 55% | 1.07× | — |

</div>

</div>

---
layout: default
---

# 6 · Sintesi

<div class="text-sm pt-2">

| direzione | stato | evidenza |
|---|---|---|
| **marcare** (`attention_highlight` / `attention_marker` / `visual_prompt`) | **chiuso** | 3 canali, 3 controlli random, tutti a ≈0 (−2 e −1 sample su 2700); oracolo 30% contro baseline 32% |
| affilare il segnale (`query_rows`, `sink_filter`) | **esaurito** | grezzo e raffinato pareggiano entrambi il proprio controllo; hit-rate invariato fra rowset (28.3 / 28.3 / 30.0) |
| **ricampionare** | **aperto** | unico arm sopra baseline (+0.56 pp); soffitto misurato **+16 pp** (p = 0.0037) |
| localizzare la finestra | **collo di bottiglia** | il picco la trova nel **18%** dei casi in cui il guadagno vive; oltre 1 cella la distanza è quasi piatta |
| scala del modello | **riferimento** | **+4.41 pp** (z = 3.27) per **+6.6%** di costo |

</div>

<div class="pt-8 text-sm">

**Prossimo giro — `coarse_to_fine` su LVBench**, tre lanci

| lancio | ruolo |
|---|---|
| arm | zoom iterativo guidato da attenzione (dove) ed entropia (quando fermarsi) |
| controllo `random` | stessa struttura, stessi passi, stesso budget: cambia solo da dove viene la regione |
| `uniform` a 48 frame | controllo a budget pari: se l'arm lo pareggia, il guadagno è budget e non segnale |

</div>
