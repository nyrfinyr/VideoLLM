---
theme: default
title: Segnali di concentrazione e router entropia/attenzione
info: |
  ## Segnali di concentrazione e router entropia/attenzione
  Sintesi per riunione — 2026-07-16.
  Fonte: `docs/sintesi_segnali_router_concentrazione.md`
class: text-center
highlighter: shiki
colorSchema: light
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Segnali di concentrazione e router entropia/attenzione

Ricampionamento adattivo dei frame per VLM su video-MCQ

<div class="pt-4 text-sm opacity-60">
Video-MME · Qwen2.5-VL-3B · 2026-07-16
</div>

---

# 1. Obiettivo

Su un benchmark video-MCQ (Video-MME, Qwen2.5-VL-3B), il campionamento
uniforme dei frame è **"cieco"**: prende sempre lo stesso numero di frame
equispaziati, a prescindere da quanto il modello sia confidente o da dove
stia effettivamente guardando.

- Usare segnali che il modello produce **gratis** durante l'inferenza
  (nessun forward aggiuntivo oltre a quelli già necessari)
- Decidere, sample per sample:
  1. **SE** vale la pena ricampionare (il modello indovina o è già sicuro?)
  2. **COME** ricampionare, se sì (dove guardare invece di ricampionare
     alla cieca?)

---

# 2. Pipeline per ogni sample

<div class="flex justify-center">

```mermaid {scale: 0.65}
flowchart LR
    A["Pass 1 SEMPRE<br/>frame uniformi"] --> B{"entropia alta?"}
    B -- no --> C["Accetta pass 1"]
    B -- si --> D{"branch_selector"}
    D -- disperso --> E["phase_shift"]
    D -- concentrato --> F["zoom_peak /<br/>zoom_multi_region"]
    C --> G["Pass 2<br/>generate() vero"]
    E --> G
    F --> G
    G --> I["Risposta finale"]

    style C fill:#2d5,stroke:#333
    style E fill:#e94,stroke:#333
    style F fill:#e94,stroke:#333
```

</div>

<div class="text-xs opacity-60 pt-2">
I sample "accettati" sono <b>bit-per-bit identici</b> alla baseline
<code>uniform</code> — solo il pass 2 conta per la risposta.
</div>

---

# Punti chiave del design

- **Il pass 1 costa sempre** — forward di prefill completo con cattura
  dell'attenzione, indipendentemente dal fatto che poi si ricampioni o
  no. Non è "extra solo se serve".
- **La risposta finale viene SEMPRE da una vera `generate()`** — mai dal
  solo `pred_letter` della softmax ristretta del pass 1.
  <br/>→ i sample "accettati" restano bit-per-bit identici alla baseline
  `uniform`: l'unico effetto misurabile viene dai sample effettivamente
  ricampionati.
- **Al massimo un ricampionamento** — nessun terzo pass che ri-verifichi
  la decisione.

---

# 3. I segnali

<div class="flex justify-center">

```mermaid {scale: 0.65}
flowchart LR
    SA["SignalAnswer"] --> AE["answer_entropy<br/>GATE"]
    SA --> VA["visual_attention"]
    VA --> SV["sink_view()"]
    SV --> RC["rank_cells()<br/>t celle per massa"]
    RC --> T1["top1_pct"]
    RC --> HN["H_norm"]
    RC --> CMS["cell_mass<br/>mean / std"]
    CMS --> ZS["top1_zscore"]
    RC --> PC["peak_cell"]
    VA --> TC["t_cells"]

    style AE fill:#59f,stroke:#333
    style T1 fill:#2d5,stroke:#333
    style HN fill:#e94,stroke:#333
    style ZS fill:#e94,stroke:#333
```

</div>


* <span class="text-blue-400">answer_entropy</span> = gate, indipendente dall'attenzione. 
* <span class="text-green-400">top1_pct</span> (vincitore) guarda solo il picco. 
* <span class="text-orange-400">H_norm</span>
* <span class="text-orange-400">top1_zscore</span> (entrambi battuti) usano l'intera distribuzione.


---

# 3.1 `answer_entropy` — il segnale di GATE

<div grid="~ cols-2 gap-6" class="items-center">
<div>

Entropia di Shannon (in bit) della softmax **ristretta alle sole lettere
candidate** (A/B/C/D...) sui logit dell'ultimo token del prefill:

$$H = -\sum_{i=1}^{n_{\text{opzioni}}} p_i \log_2 p_i$$

$H = 0$ → confidente

$H = \log_2(n_{\text{opzioni}})$ → sta indovinando

Usato oggi come **gate**:


$$\text{can\_resample} = \text{answer\_entropy} > \tau $$


<div class="text-sm opacity-60">
default 0.7
</div>

</div>
<div>

<img src="/assets/fig1_entropy_by_outcome.png" class="rounded shadow w-full" />

<div class="text-xs opacity-70 mt-4">
Pilota VNBench (30 video): separazione netta fra risposte corrette e
sbagliate — $r = -0.75$, il segnale più forte fra tutti quelli provati.
</div>

</div>
</div>

---

# 3.2 `top1_pct` — concentrazione dell'attenzione sul frame di picco

<div class="text-sm opacity-70 pb-2">
In parole semplici: <b>la percentuale di attenzione "pulita" (senza i
token sink) che il modello dedica al SUO frame preferito</b> — quanto si
fissa su un unico istante del video invece di guardarlo tutto.
</div>

Sulla mappa di attenzione testo→visivo (già catturata dal pass 1):

1. si azzerano i token **"sink"** (`sink_view`) — altrimenti assorbono la
   maggior parte della massa indipendentemente dal contenuto
2. si aggregano le celle spazio-temporali per massa totale (`rank_cells`)
3. si prende la **percentuale di massa sulla cella (frame) più
   attenzionata** → questo è `top1_pct`

Alto → il modello "si fissa" su un frame preciso (grounding).
Basso/vicino all'uniforme → attenzione dispersa su tutto il video.

**Perché**: correlazione più debole dell'entropia ($r = 0.58$), ma si è
rivelato — dopo un confronto empirico a tre — il **migliore** dei segnali
di concentrazione provati, nonostante sia il più "grezzo".

---

# 3.3 / 3.4 — due tentativi battuti

<div grid="~ cols-2 gap-8">
<div>

### `H_norm` — entropia normalizzata

$$H_{\text{norm}} = \frac{H}{\log t}, \quad H = -\sum_{i=1}^{t} p_i \log p_i$$

$0$ = tutta la massa su una cella · $1$ = dispersione perfettamente
uniforme su $t$ celle.

<div class="text-sm opacity-70 pt-2">
Intuizione: <code>top1_pct</code> guarda solo il picco, ignora la forma
del resto della distribuzione.
<br/><b>Risultato: separa nella direzione giusta ma più debolmente di
top1_pct.</b>
</div>

</div>
<div>

### `top1_zscore` — outlier per-sample

$$z = \frac{\text{top1\_pct} - \mu}{\sigma}$$

$\mu,\sigma$ = media/dev.std delle masse per-cella dello stesso video —
nessuna calibrazione esterna.

<div class="text-sm opacity-70 pt-2">
Intuizione: normalizzazione completamente per-sample, alternativa più
"pulita" a una soglia fissa.
<br/><b>Risultato: ancora più debole di H_norm — il peggiore dei tre.</b>
</div>

</div>
</div>

---

# 4. Le decisioni prese, in ordine

<div class="font-bold opacity-100">4.1 — Serve un gate E una decisione di ramo</div>

Ricampionare sempre alla cieca funziona (+3.2pt vs uniform) ma un
**oracolo** che sceglie il ramo giusto per sample recupera **98/163**
sample, contro **83/163** del miglior ramo singolo — **+15 di margine
puro dal solo instradamento**.

**4.2 — Le soglie assolute fisse sono morte su Video-MME**

Le soglie $\text{concentration\_threshold} \in \{20, 30, 45\}$ (dal
pilota VNBench) non discriminano più nulla: su Video-MME `top1_pct` non
supera mai ~23%.

---

# 4.3 Tre segnali di concentrazione a confronto

Separazione statistica (Mann-Whitney, $|z|$) fra sample dove "zoom
multi-regione" batte "fase sfasata" e viceversa — 163 sample ricampionati.

| segnale | separazione \|z\| | plateau accuracy /163 |
|---|---:|---:|
| **`top1_pct`** (grezzo) | **2.77** | **88–90** |
| `H_norm` | 2.23 | 85–87 |
| `top1_zscore` | 1.70 | 87 |

<div class="pt-4 text-sm opacity-70">
Riferimento: sempre-fase-sfasata = 83/163, sempre-zoom = 78/163, oracolo
= 98/163.
</div>

<div class="pt-4 font-bold">
Nessuno dei due segnali "più sofisticati" ha battuto il proxy grezzo →
decisione: usare <code>top1_pct</code>.
</div>

---

# Separazione Mann-Whitney — cos'è

Test **non parametrico** che confronta due gruppi indipendenti guardando
i **ranghi** dei valori (non i valori grezzi) — non assume distribuzioni
normali.

<div class="text-sm opacity-70 pb-4">
Utile qui perché <code>top1_pct</code> è fortemente non-normale (poche
celle alte, coda lunga di celle quasi a zero).
</div>

Si uniscono i due gruppi, si ordinano tutti i valori e si sommano i
ranghi del gruppo 1 ($R_1$):

$$U = R_1 - \frac{n_1(n_1+1)}{2}$$

Sotto l'ipotesi nulla (nessuna differenza) $U$ ha media $\frac{n_1
n_2}{2}$ e varianza nota → si standardizza in uno z-score:

$$z = \frac{U - \dfrac{n_1 n_2}{2}}{\sqrt{\dfrac{n_1 n_2 (n_1+n_2+1)}{12}}}$$

<div class="text-center font-bold pt-2">

$|z|$ grande = i due gruppi sono davvero separati, non per caso.

</div>

---

# Separazione Mann-Whitney — come l'abbiamo calcolata noi

Sui 163 sample ricampionati da entrambi i rami, **35 sono discordanti**:
**15 pro-`zoom_multi_region`**, **20 pro-`phase_shift`**. Per ciascun
segnale candidato (misurato al pass 1, prima di sapere quale ramo
avrebbe vinto) si confrontano i due gruppi:

<div class="text-sm">

| | $n$ | `top1_pct` mediana |
|---|---:|---:|
| pro-`zoom_multi_region` | 15 | **7.60** |
| pro-`phase_shift` | 20 | 4.96 |

</div>

$$U = 233, \quad \frac{n_1 n_2}{2} = 150, \quad z = \frac{233 - 150}{\sqrt{15 \cdot 20 \cdot 36 / 12}} = \frac{83}{30} \approx 2.77$$

<div class="pt-2 text-sm opacity-70 text-center">

$p \approx 0.003$ — separazione altamente significativa: campioni più
concentrati (`top1_pct` alto) → vince più spesso
<code>zoom_multi_region</code>.

</div>

---

# 4.4 La soglia dev'essere un ratio

`top1_pct` è una percentuale su $t$ celle: sotto attenzione uniforme
("livello di caso") il suo valore atteso è

$$\frac{100}{t}$$

Normalizzando per questo, la soglia si riscala automaticamente fra
`nframes`/dataset diversi:

$$\text{ratio} = \frac{\text{top1\_pct}}{100 / t}$$

$t = 64$ per $\text{nframes} = 128$, **verificato su GPU reale** — non
solo dedotto dal codice: fattore di merge temporale a coppie del modello,
$t = \text{nframes} / 2$.

<div class="pt-6 text-xl font-bold text-center">

Soglia scelta: $\text{concentration\_ratio} = 3.85$

</div>
<div class="text-center text-sm opacity-70">

equivalente a $\text{top1\_pct} \approx 6.02$ a $t = 64$, dentro il
plateau 88–90/163

</div>

---

# 5. Cosa fa il codice OGGI

`branch_selector` — tre modalità, selezionabili senza toccare codice,
**additive**:

| `branch_selector` | condizione "disperso" | stato |
|---|---|---|
| `"concentration"` (default) | $\text{top1\_pct} < \text{concentration\_threshold}$ | storico, **soglia inerte** |
| `"concentration_ratio"` | $\text{top1\_pct} < \text{concentration\_ratio} \cdot \frac{100}{t}$ | **Stage C, in validazione** |
| `"entropy"` | $H_{\text{norm}} > \text{attention\_entropy\_threshold}$ | provato, **battuto** |

---

# La strada scelta: `concentration_ratio`

<div class="flex justify-center">

```mermaid {scale: 0.8}
flowchart LR
    G{"entropia<br/>alta?"} -- no --> ACC["accetta pass 1"]
    G -- si --> C2{"top1_pct <<br/>ratio × 100/t ?<br/>(3.85 — Stage C)"}

    C2 -- "si, disperso" --> PS["phase_shift"]
    C2 -- "no, concentrato" --> ZOOM{"zoom_multi<br/>_region?"}
    ZOOM -- true --> ZMR["zoom_multi_region<br/>n_regions finestre"]
    ZOOM -- false --> ZP["zoom_peak<br/>1 finestra sul picco"]

    style C2 fill:#2d5,stroke:#333
```

</div>

---

# 5.1 `zoom_peak` — finestra unica sul picco

1. `peak_cell` = indice della cella a massa più alta (`ranked_cells[0]`)
2. Posizione temporale frazionaria:
   $$\text{frac} = \frac{\text{peak\_cell} + 0.5}{t}$$
3. Mappata sulla finestra reale del pass 1 ($w_0$/$w_1$):
   $$\text{peak\_time} = w_0 + \text{frac} \cdot (w_1 - w_0)$$
4. Finestra stretta di semi-larghezza
   $$\frac{\text{window\_frac} \cdot (w_1 - w_0)}{2}$$
   (default $\text{window\_frac}=0.3$ → 30%) centrata su
   $\text{peak\_time}$, clampata a $[0, \text{duration}]$
5. Ricampiona `nframes` **uniformi** dentro questa finestra ristretta

<div class="text-sm opacity-60 pt-2">
Una sola regione, nessuna gestione di sovrapposizioni o budget-splitting.
</div>

---

# 5.1 `zoom_multi_region` — fino a `n_regions` finestre disgiunte

Stesso principio di `zoom_peak`, generalizzato a più massimi locali:

<div class="text-sm leading-snug">

1. **`_select_region_cells`** — `"topk"`: prime `n_regions` celle.
   `"adaptive"` (default): mediana + `region_mad_k`·MAD, escludendo il
   picco globale (evita l'auto-mascheramento)
2. **`_multi_region_spans`** — centri più vicini di
   $\text{region\_merge\_gap\_frac} \cdot (w_1-w_0)$ fusi insieme; ogni
   centro → finestra di semi-larghezza fissa
   $\text{region\_window\_frac} \cdot (w_1-w_0)$ (default 10%); merge
   finale per garantire finestre disgiunte
3. **`_allocate_region_budget`** — divide `nframes` fra le regioni
   (`"equal"` o `"proportional"` alla massa), somma sempre $\le$ `nframes`
4. **`_build_multi_region_media`** — estrae i frame, calcola
   $\text{sample\_fps} = \text{frame totali} / \text{durata regioni usate}$
   e impacchetta come `VideoFrames` (il modello vede i frame come
   uniformemente spaziati, i "buchi" fra regioni spariscono)

</div>

---

# 6. Risultati sperimentali chiave

<div class="text-sm opacity-60 pb-2">Video-MME, subset 250, seed 42</div>

| strategia | accuracy | note |
|---|---:|---|
| uniform (controllo) | 62.8% | baseline |
| fase sfasata sempre | 66.0% | +3.2pt |
| zoom multi-regione forzato sempre | 64.0% | recupera 13 sample UNICI |
| oracolo (ramo giusto per sample) | ~68.8% | limite superiore teorico |
| router `top1_pct` (`ratio=3.85`) | **atteso ~68.8%, da verificare** | **Stage C, run in corso** |

---
layout: center
class: text-center
---

# Domande?

<div class="text-sm opacity-60 pt-4">
docs/sintesi_segnali_router_concentrazione.md — cronologia completa in
docs/piano_zoom_multiregione_disgiunto.md
</div>
