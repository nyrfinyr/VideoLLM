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

# Avanzamento esperimenti

Qwen3-VL su Video-MME e LVBench

---
layout: default
---
# 1 · `marker_24` — il segnale consegnato come **token**

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

**pass 1** · 24 frame uniformi, un forward con cattura
attenzione *domanda → token visivi* → **cella di picco**

**pass 2** · gli **stessi** frame, blocco video spezzato sui confini di cella

<div class="pt-3" style="display:flex;gap:2px;align-items:center">
<div v-for="i in 10" :key="'a'+i" style="width:14px;height:22px;border-radius:2px;background:#e3e6ea;border:1px solid #d0d4d9"></div>
<div style="width:3px;height:30px;background:#c0392b;margin:0 3px"></div>
<div v-for="i in 2" :key="'b'+i" style="width:14px;height:22px;border-radius:2px;background:#f5c6c6;border:1.5px solid #c0392b"></div>
<div style="width:3px;height:30px;background:#c0392b;margin:0 3px"></div>
<div v-for="i in 12" :key="'c'+i" style="width:14px;height:22px;border-radius:2px;background:#e3e6ea;border:1px solid #d0d4d9"></div>
</div>

<div class="pt-1 text-xs opacity-60">
24 frame = 12 celle. I marcatori racchiudono <b>la cella</b> — cioè <b>2 frame</b>, non uno: è la risoluzione reale del segnale, e i tagli devono cadere su confini di cella o il processor padda e sfasa tutti i timestamp.
</div>

</div>

<div class="text-sm">

| knob | valore |
|---|---|
| `query_rows` | `question` |
| `sink_filter` | `false` — attenzione **grezza** |
| `top_k` | 1 cella |
| `always_highlight` | `true` — nessun gate |
| costo vs baseline | **3.2×** |


</div>

</div>

<div class="pt-3" style="font-family:ui-monospace,monospace;font-size:0.66rem;line-height:1.7;background:#f6f7f9;border:1px solid #e3e6ea;border-radius:4px;padding:8px 10px;word-break:break-word">
…&lt;|vision_end|&gt;<span style="color:#c0392b;font-weight:600">&lt;START_IMPORTANT_FRAME&gt;</span><span style="background:#fdeaea;padding:1px 0">&lt;4.5 seconds&gt;&lt;|vision_start|&gt;[frame]&lt;|vision_end|&gt;&nbsp;&lt;5.5 seconds&gt;&lt;|vision_start|&gt;[frame]&lt;|vision_end|&gt;</span><span style="color:#c0392b;font-weight:600">&lt;END_IMPORTANT_FRAME&gt;</span>&lt;6.5 seconds&gt;…
</div>

<div class="pt-1 text-xs opacity-60">
Prompt:
</div>

<div class="pt-3" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:6px 12px;font-size:0.78rem">
<i>Focus on the frames enclosed between &lt;START_IMPORTANT_FRAME&gt; and &lt;END_IMPORTANT_FRAME&gt;: that is where the visual evidence relevant to the question is.</i>
</div>

<div class="pt-1 text-xs opacity-60">
Anteposta al prompt MCQ. Il controllo <code>random</code> riceve la stessa identica istruzione, con la cella scelta a sorte.
</div>

---
layout: default
---
# 2 · `visual_prompt_24` — lo stesso segnale come **pixel**

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

**pass 1** · identico

**pass 2** · i 24 frame estratti in PNG, la parola `IMPORTANT` **dipinta** sui 2 frame della cella

<div class="pt-3" style="display:flex;gap:2px;align-items:flex-end">
<div v-for="i in 24" :key="i" :style="`width:14px;height:22px;border-radius:2px;background:#e3e6ea;border:1px solid #d0d4d9;position:relative;overflow:hidden`">
<div v-if="i===11||i===12" style="position:absolute;left:0;right:0;bottom:0;height:26%;background:#c0392b"></div>
</div>
</div>

<div class="pt-1 text-xs opacity-60">
<b>Entrambi</b> i frame della cella, perché il patch embedding è 3D e i due si fondono in una riga di token.
</div>

</div>

<div class="text-sm">

| knob | valore |
|---|---|
| `query_rows` | **`entity`** — sole righe del soggetto |
| `sink_filter` | **`true`** — token-sink azzerati |
| `top_k` | 1 cella |
| geometria | `bottom`, 12% dell'altezza |
| costo vs baseline | **3.5×** |

<div class="pt-3 text-xs opacity-60">
<code>entity</code>: Vedi slide successiva
</div>

</div>

</div>

<div class="grid grid-cols-2 gap-6 pt-3" style="align-items:start">

<div>
<img src="/assets/preview_bottom_012.png" style="max-height:186px;width:auto;border-radius:4px;border:1px solid #e3e6ea" />
<div class="pt-1 text-xs opacity-60">
Frame dipinto alla geometria del fullset: <code>bottom</code>, <code>height_frac=0.12</code>.
</div>
</div>

<div>

<div class="pt-1 text-xs opacity-60">
Prompt:
</div>

<div style="border-left:3px solid #c0392b;background:#fbfbfc;padding:6px 12px;font-size:0.78rem;line-height:1.5">
<i>Some frames have the word IMPORTANT written on them: those frames contain the visual evidence relevant to the question. Do not mention the word in your answer.</i>
</div>
<div class="pt-1 text-xs opacity-60">
Anteposta al prompt MCQ. L'ultima frase evita che la parola finisca nella risposta e sporchi il parsing MCQ.
</div>
</div>

</div>

---
layout: default
---
# 3 · Il knob `query_rows=entity`

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

**Quali righe della domanda entrano nella media dell'attenzione**

| passo | cosa fa |
|---|---|
| 1 | taglio del prompt a `options:` → resta la **sola domanda** |
| 2 | chiamata **solo-testo al decoder già in memoria** — greedy, 32 token, system + 3 esempi few-shot |
| 3 | output atteso: **una sequenza contigua verbatim**, oppure `NONE` |
| 4 | match verbatim nella domanda → righe di quei token |

<div class="pt-3">

| esito (900 sample) | quota | righe usate |
|---|---:|---|
| entity mappata | **63%** | token del soggetto |
| `NONE` — domanda senza soggetto | 18% | fallback → `question` |
| non mappa verbatim | 19% | fallback → `question` |

</div>

</div>

<div class="text-sm">

<div style="border-left:3px solid #c0392b;background:#fbfbfc;padding:6px 12px;font-size:0.72rem;line-height:1.45">
<i>You extract THE SUBJECT of a question: the single entity or action the question is about. Copy it EXACTLY as written in the question: ONE contiguous sequence of one or more words. Prefer the shortest sequence that names the subject. Do NOT answer the question. If the question has no specific subject, output NONE.</i>
</div>

<div class="pt-1 text-xs opacity-60">
+ 3 esempi few-shot: <code>"…jacket worn by the man playing the guitar?"</code> → <code>jacket</code> · <code>"What is this video mainly about?"</code> → <code>NONE</code>
</div>

<div class="pt-4">

| `query_rows` | righe mediate |
|---|---|
| `all` (regime storico) | 100% del prompt |
| `question` | ~24% |
| **`entity`** | **~11%** |

</div>

<div class="pt-3 text-xs opacity-60">
Le <b>opzioni restano fuori</b> dall'input dell'estrattore: con le opzioni in contesto il prior "MCQ → rispondi" di un 2B domina qualsiasi istruzione, e il modello sputava un'opzione invece del soggetto (8/60 utili). Togliere il grilletto batte l'istruzione.
</div>

</div>

</div>

<div class="pt-3 text-xs opacity-60">
Nessun secondo modello in memoria, nessun costo di caricamento. Il fallback è <b>tracciato per sample</b> (<code>query_rows_mode</code>, <code>entity_mapped</code>, <code>entity_raw</code>): l'effetto entity è quindi misurato <b>diluito</b> — su 900 sample solo 566 girano davvero a righe-entità. Il confronto interno attention/random non ne soffre: entrambi i rami usano lo stesso mix.
</div>

---
layout: default
---

# 4 · Arm su Video-MME — Qwen3-VL-2B (2700 sample)

<div class="text-xs">

| arm | gate | `query_rows` | `sink_filter` | pooled | Δ | short | medium | long |
|---|---|---|---|---:|---:|---:|---:|---:|
| `baseline_24` | — | — | — | **54.44%** (1470) | — | 66.89% | 52.33% | 44.11% |
| `entropy_shift_24` | **`> 0.7`** | `all` | `true` | **55.00%** (1485) | **+0.56** | 67.78% | 52.11% | 45.11% |
| `marker_24` (attention) | nessuno | `question` | `false` | **53.04%** (1432) | −1.41 | 66.33% | 49.56% | 43.22% |
| `marker_random_24` (random) | nessuno | `question` | `false` | **53.11%** (1434) | −1.33 | 66.44% | 49.89% | 43.00% |
| `visual_prompt_24` (attention) | nessuno | `entity` | `true` | **53.48%** (1444) | −0.96 | 65.89% | 51.44% | 43.11% |
| `visual_prompt_24_rand` (random) | nessuno | `entity` | `true` | **53.52%** (1445) | −0.92 | 66.22% | 50.78% | 43.56% |

</div>

<div class="pt-1 text-xs opacity-60">
<b>nessuno</b> = <code>always_highlight: true</code>: si interviene sul <b>100%</b> dei sample, nessuno è esente (da cui il costo 3.2× / 3.5×).
</div>

<div class="grid grid-cols-2 gap-6 pt-2">

<div class="text-sm">

| coppia treatment − controllo | Δ | in sample |
|---|---:|---:|
| `marker_24` − `marker_random_24` | −0.07 pp | **−2 / 2700** |
| `visual_prompt_24` − `visual_prompt_24_random` | −0.04 pp | **−1 / 2700** |

</div>

<div class="text-sm">

| `entropy_shift_24` — il solo con gate | quota | accuracy |
|---|---:|---:|
| gate **aperto** (entropia > 0.7) → pass 2 | 53.70% (1450) | **37.86%** |
| gate **chiuso** → risposta del pass 1 | 46.30% (1250) | **74.88%** |

</div>

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

<style scoped>
.slidev-layout table th,
.slidev-layout table td { padding: 0.14rem 0.4rem; }
.slidev-layout table { margin: 0.2rem 0; }
</style>

---
layout: default
---

# 5 · Scala del modello — baseline 2B vs 4B (2700 sample)

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

# 6 · Oracolo su LVBench (100 sample, 24 frame)

<div class="text-sm pt-2">

| condizione | quali frame vede | come riceve l'indicazione | da dove viene l'indicazione | accuracy | Δ | appaiato |
|---|---|---|---|---:|---:|---:|
| `baseline` | 24 uniformi | — | — | 32% | — | — |
| `peak` | 24 uniformi | puntatore nel prompt | cella di **picco** (segnale del modello) | 29% | −3 | +4 / −7 |
| `oracle` | 24 uniformi | puntatore nel prompt | finestra **annotata** (verità) | 30% | −2 | +5 / −7 |
| **`zoom`** | **24 dentro la finestra** | **i frame stessi** | finestra **annotata** (verità) | **48%** | **+16** | **+22 / −6** |

</div>

<div class="pt-3" style="border-left:3px solid #c0392b;background:#fbfbfc;padding:6px 12px;font-size:0.76rem">
<i>Focus on the part of the video between 880 and 1100 seconds: that is where the visual evidence relevant to the question is.</i>
</div>

<div class="pt-1 text-xs opacity-60">
Il puntatore di <code>peak</code> e <code>oracle</code>: una frase anteposta al prompt MCQ, frame invariati. Si indica <b>l'intervallo</b> della cella (~220 s a <code>nframes=24</code>), non un istante: è la risoluzione vera del segnale.
</div>

<div class="pt-3 text-sm">

Due assi, non quattro condizioni sparse: **quale informazione** (il picco che il modello stima, contro la finestra vera annotata) × **come gliela consegni** (dirglielo, contro cambiare i frame che vede).

</div>

<div class="pt-2 text-sm">

| confronto | isola | esito |
|---|---|---|
| `peak` → `oracle` | informazione migliore, stessa consegna | 29% → 30%: **l'informazione perfetta non paga** |
| `oracle` → `zoom` | stessa informazione, consegna diversa | 30% → 48%: **paga la consegna** |

</div>

<div class="pt-6 text-xs opacity-60">
<code>qwen3_vl_2b_attn</code> · job 94348 · MCQ a 4 opzioni (livello di caso 25%) · <code>peak</code> e <code>oracle</code> usano lo <b>stesso identico testo</b> di puntatore (<code>POINTER_SPAN_SEC</code>): fra loro cambia solo <i>dove</i> punta, mai <i>come</i> parla · dettagli in <code>docs/oracolo_lvbench.md</code>
</div>
