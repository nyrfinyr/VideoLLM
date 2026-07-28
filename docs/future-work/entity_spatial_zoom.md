# Entity Spatial Zoom (ESZ) — design

> Design dell'esperimento successivo alla chiusura del filone di
> ricampionamento **temporale** (`docs/sintesi_segnali_router_concentrazione.md`
> §7-9, `docs/analisi_winloss_resampling_video_mme.md`). Sposta la leva
> dallo *spazio-tempo* al solo **spazio**: invece di scegliere *quali frame*
> (saturo a 128 frame, dimostrato nullo), aumenta la *risoluzione effettiva*
> sull'oggetto rilevante. Decisioni prese insieme il 2026-07-17.

## Perché — la lezione a monte

- Il gate `answer_entropy` **generalizza** (full 2700: accepted 87.8% vs
  gated ~45%, gate-fire 52%→69%→75% short→medium→long). È l'asset. Lo
  teniamo invariato.
- Il ricampionamento **temporale** è nullo sul full: `zoom_multi_region`
  (massa di attenzione come finestra temporale) net **−7**, `phase_shift`
  +15 ma ~1.4σ. La copertura temporale a 128 frame è satura → spostare i
  frame non porta informazione nuova.
- **Probe di concentrazione spaziale** (su capture locali `data/videomme_capture/`):
  la massa spaziale aggregata *male* è inutile, ma isolando **il token
  giusto** e **il frame giusto** collassa. `bbox@50%` (frazione area frame
  che cattura il 50% della massa, sink-filtrata):

  | aggregazione | bbox@50% | guadagno risoluzione lineare |
  |---|---:|---:|
  | media token, somma tempo (naive) | 90% | 1.05× (inutile) |
  | media token, frame di picco | 78% | 1.13× |
  | token più concentrato, somma tempo | 90% | 1.05× |
  | **token più concentrato, frame di picco** | **11%** | **~3.0×** |

  → lo zoom spaziale funziona **solo** con selettività su token *e* tempo
  insieme.

## Meccanismo (decisioni prese)

**Gate**: invariato, `answer_entropy > entropy_threshold`. ESZ agisce solo
sui sample gated.

**Sui gated:**
1. **Selezione token = entità (nomi)** della domanda → righe-query. Si
   riusa `utils.attn_core.question_rows` (esclude già lo scaffolding MCQ) +
   un filtro ai nomi. *Scelta vs "token più concentrato"*: principled,
   allineato all'`attn_explorer`; da confermare in Stage-A che i nomi siano
   almeno tanto localizzati quanto il token più concentrato.
2. Per ogni entità: `AttentionCapture.aggregate(rows)` → `[t, gh, gw]`,
   frame di **picco temporale** per quell'entità, **crop stretto** del bbox
   a massa più alta su quel frame.
3. **Ensemble crop+contesto**: softmax-ristretta del **pass-1** (video
   uniforme = contesto, già calcolata) + softmax-ristretta del **pass-2**
   (i crop-entità ad alta risoluzione) → combinazione → argmax. L'ensemble
   protegge dal lato "broken" (la lezione win/loss: sostituire cieco
   rompe i corretti).

## Sub-decisioni implementative (fissate, non bloccano Stage-A)

- **Crop come `Image`, non `Video`**: il crop passa come singola immagine
  ad alta risoluzione (`models.media.Image`), `max_pixels` controlla
  direttamente la risoluzione — niente floor di 128 token/frame del path
  video.
- **Estrazione nomi**: partire dal filtro content-word (stopword-based) su
  `question_rows`; raffinare a POS/nomi solo se Stage-A mostra troppe
  non-entità. Evitare dipendenze pesanti (spaCy) se non necessario.
- **Ensemble math**: iniziare con **media** dei due vettori di prob
  ristretti; poi provare pesatura per confidenza (inverso-entropia).
- **Multi-entità / VRAM**: un crop per nome al suo frame di picco, cap a
  top-k nomi; l'ensemble somma contesto + k crop.

## Stage-A — validazione (primo passo, prima di scrivere la strategy)

Obiettivo: verificare che i crop **entità × frame-di-picco** siano stretti
*e* rilevanti, prima di spendere GPU-h — stessa disciplina che avrebbe
evitato l'errore `top1_pct`.

Sfrutta macchina esistente: la strategy salva le capture solo nel blocco
gated (`_save_attention_capture` dentro `if can_resample:`), quindi girando
`entropy_attention_resample` con `capture_dir` si ottengono automaticamente
solo i gated con tutti i segnali.

1. Cattura su ~40 sample Video-MME (HPC/GPU) → si salvano i gated.
2. Script di analisi: per ogni nome-entità, `bbox@50%` al frame di picco
   (target <25%) + check qualitativo (il crop contiene l'oggetto che la
   domanda chiede?). L'`attn_explorer` renderizza già gli overlay per-token.
3. **Gate decisionale**: crop stretti *e* rilevanti → MVP ensemble; crop
   diffusi/spuri → fallback a "token più concentrato" o ripensare.

Nota: sviluppo iniziale dell'analisi possibile **subito** sulle capture
locali `data/videomme_capture/` (hanno `attn`/`query_tokens`), che misurano
la tightness dei crop-entità; il check di *rilevanza* sui gated veri e la
cattura mirata richiedono un giro HPC.

## Rischi noti

- Griglia spaziale grossa nella config eval (`max_pixels=151200` → ~6×8),
  la *localizzazione* del box è grezza (il crop dal frame originale resta
  ad alta risoluzione).
- Un nome astratto ("atmosfera", "situazione") non è localizzato → l'entità
  potrebbe non concentrarsi; l'ensemble col contesto mitiga (non peggiora
  oltre il pass-1).
- Selezione entità sbagliata = crop spurio → Stage-A serve proprio a
  intercettarlo prima dell'eval.
