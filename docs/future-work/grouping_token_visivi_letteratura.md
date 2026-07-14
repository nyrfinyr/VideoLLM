# Metodi di grouping/budget allocation per token visivi — rassegna letteratura

> **Stato** (2026-07-07): rassegna preparata per confrontare la strategia di
> grouping di **AdaptToken** (competitor diretto, vedi
> `docs/diario/riunione_07-07-26.md`) con alternative in letteratura, in
> vista dell'esperimento VideoMME (Qwen2.5-VL-3B vs baseline, poi 7B vs
> AdaptToken). Fonti: collezione Zotero `Tesi/training-free`, PDF letti
> direttamente (non riassunti automatici — un primo digest via WebFetch su
> AdaptToken si è rivelato inaffidabile, vedi nota in fondo).

## 0. Baseline da battere — AdaptToken (arXiv:2603.28696)

Grouping **interleaved a stride fisso**, non a chunk contigui: dati N frame
campionati a FPS fisso, `G = N//K + 1` gruppi, gruppo `F_g = {f_g, f_{g+G},
f_{g+2G}, ...}` — ogni gruppo copre l'INTERO video a risoluzione temporale
più bassa, con offset diverso. Segnale di rilevanza per gruppo: entropia
della risposta generata (media certezza sul 10% meno certo dei token
generati). Allocazione budget: `B_g = B × softmax({C_1,...,C_G}/τ)_g`
(τ=2 fisso). Selezione intra-gruppo: attenzione cross-modale da un layer
tardo scelto via probing NIAH. Rimozione ridondanza: scarta iterativamente
token con similarità (feature + prossimità temporale) più alta, da 1.1×B a
B token finali.

Il progetto ha già un pezzo di questa pipeline gratis: l'entropia di
risposta calcolata dai logit dell'ultimo token del prefill (nessuna
generazione necessaria, vedi `docs/entropia_confidenza_mcq.md`), riusabile
come segnale di rilevanza di gruppo al posto della entropia-su-generazione
di AdaptToken (per VideoMME, MCQ a singola lettera, i due segnali
collassano quasi allo stesso costo).

## 1. AKS — Adaptive Keyframe Sampling (Tang et al. 2025, arXiv:2502.21271)

**Meccanismo.** Non raggruppa per posizione fissa: formula la selezione
come ottimizzazione esplicita

```
KS_M(Q, F) = argmax_{|I|=M}  Σ_{t∈I} s(Q, F_t) + λ · c(I)
```

dove `s(Q, F_t)` è la rilevanza frame-prompt (calcolata con un VL model
ausiliario economico, CLIP o BLIP ITM — **non** il modello target, per
tenere basso il costo) e `c(I)` è un termine di coverage temporale.

`c(I)` è stimato con una **partizione binaria ricorsiva** dell'asse
temporale: al livello 0 l'intero video è un bin, si splitta in due
sotto-bin di ampiezza T/2, poi T/4, ecc., fino a un livello massimo
`L ≤ ⌈log2 M⌉`. Ad ogni bin si penalizza lo sbilanciamento nel numero di
keyframe assegnati ai due sotto-bin (`|m1 - m2|`).

L'algoritmo `ADA` (adaptive sampling) risolve l'ottimizzazione senza tuning
esplicito di λ: ad ogni livello confronta lo score medio di tutti i frame
del bin (`s_all`) con lo score medio dei top-M (`s_top`); se il gap supera
una soglia `s_thr`, ritorna direttamente i top-M frame del bin (il segnale
è "abbastanza piccato" da fidarsi); altrimenti splitta il bin in due e
ricorre, allocando i keyframe in modo bilanciato. Due casi degeneri:
λ=0 → puro top-scoring (`TOP`, rischia di concentrarsi tutto in una
finestra); λ→∞ → un keyframe per bin (`BIN`, degenera in uniform sampling
se lo score è costante — cioè nella baseline attuale del progetto).

**Risultati.** Testato su LongVideoBench e VideoMME con Qwen2-VL,
LLaVA-OV, LLaVA-Video come modelli target (nessuna modifica ai pesi, solo
i frame in input cambiano). Guadagni consistenti su tutti e tre
(es. Qwen2-VL 7B: LVB val 55.5→60.5, VideoMME 57.6→59.9).

**Perché interessante per il progetto.** L'albero di bisezione si adatta
alla forma reale della distribuzione di rilevanza invece di fissare G a
priori come AdaptToken, e la coverage è un termine esplicito
nell'obiettivo (non un post-processing separato di rimozione ridondanza).
È il candidato più diretto per sostituire *solo* la formula di
grouping/budget di AdaptToken, mantenendo l'impostazione "training-free":
basterebbe rimpiazzare lo score `s(Q, F_t)` di AKS con il segnale
entropia/massa-di-attenzione già disponibile nel progetto.

**Limite.** Lo score di rilevanza richiede comunque un modello VL
ausiliario (CLIP/BLIP) separato dal modello target — a differenza del
segnale del progetto e di AdaptToken, che riusano il forward del modello
target stesso senza costo aggiuntivo. Andrebbe verificato se sostituire
`s` con l'entropia/attenzione del prefill Qwen preserva il vantaggio
misurato nel paper (che usa CLIP/BLIP, non l'entropia di risposta).

## 2. Video Panels (Doorenbos et al. 2026, arXiv:2509.23724)

**Meccanismo — asse ortogonale, non sostitutivo del budget allocation.**
Non seleziona né pota token: raggruppa frame **contigui** a dimensione
fissa (iperparametri α×β, default 2×2) e li **tila spazialmente in
un'unica immagine** ("panel"), scambiando risoluzione spaziale per
risoluzione temporale a parità di budget di token. Il numero di frame
campionati `T` si sceglie dinamicamente in base al rapporto fra context
window `C` e durata video `D` (Eq. 1 del paper); quando serve paneling, ogni
gruppo di α×β frame consecutivi (downsampled a H/α × W/β) diventa un
singolo "super-frame" in griglia left-to-right/top-to-bottom. Il grouping
qui resta uniforme/contiguo — nessuna adattività al contenuto — la leva è
diversa: non "quali frame guardare" ma "quanti frame posso permettermi di
guardare a parità di token".

**Risultati.** Testato ESATTAMENTE sul setup del progetto — **VNBench e
VideoMME con Qwen2.5-VL**, oltre a TimeScope, MLVU, MF2. Training-free,
parameter-free, model-agnostic. Guadagni consistenti su quasi tutti i
modelli/dataset testati; particolarmente marcati sui task "needle"
(TimeScope, VNBench): +19.4% relativo su TimeScope Long con VideoLLaMA 3
7B. Su Qwen2.5-VL 7B con 32 frame di contesto: VideoMME 61.6→64.0 (medium),
51.2→54.7 (long).

**Perché interessante per il progetto.** Essendo ortogonale ad AdaptToken
(agisce sulla densità temporale via tiling spaziale, non sulla selezione
token), si può in linea di principio **combinare** con qualsiasi schema di
grouping/budget: es. usare il segnale entropia/attention-mass del progetto
per decidere DOVE aumentare la densità di paneling (più tiling nelle zone
a bassa confidenza / needle-like) invece di applicarlo uniformemente come
nel paper originale.

## 3. Prossimi passi suggeriti

1. Setup baseline VideoMME Qwen2.5-VL-3B (in corso, vedi diario).
2. Provare a sostituire lo score `s(Q, F_t)` di AKS (§1) con il segnale
   entropia/attention-mass già disponibile, mantenendo la bisezione
   ricorsiva come schema di grouping — cambio minimo rispetto
   all'infrastruttura esistente.
3. Valutare Video Panels (§2) come layer indipendente/complementare,
   eventualmente guidato dallo stesso segnale per decidere dove infittire
   il tiling.
4. Paper non ancora letto, solo intercettato in una ricerca web più ampia
   (non in Zotero): **MMG-Vid: Maximizing Marginal Gains at Segment-level
   and Token-level for Efficient Video LLMs** (arXiv:2508.21044) — il
   titolo suggerisce budget allocation segment+token level, territorio
   ancora più vicino ad AdaptToken. Da aggiungere a Zotero e leggere se le
   opzioni sopra non bastano.

## Nota metodologica

Il primo tentativo di leggere il paper AdaptToken è passato per lo
strumento WebFetch (digest automatico via modello leggero): ha affermato
che il grouping fosse "content-adaptive", **smentito** leggendo poi il PDF
per intero (è invece a stride fisso interleaved, vedi §0). Per questa
rassegna i paper sono stati scaricati e letti per intero (via `pdfread` sui
PDF locali di Zotero) prima di riportare qualunque dettaglio meccanicistico
o formula.
