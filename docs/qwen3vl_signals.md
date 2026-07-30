# Qwen3-VL — cattura dei segnali (`SupportsSignals`)

> **Stato**: implementato e verificato in locale (registri, composizione
> config, span su layout sintetici, smoke CPU end-to-end su Qwen2.5-VL con
> layer troncati). **Non ancora verificato su GPU con i pesi Qwen3** —
> checklist in fondo.

Estende la cattura entity-agnostica di `models/qwen_attn.py` (nata su
Qwen2.5-VL, vedi `docs/qwen25vl_entity_attention.md`) alla famiglia Qwen3-VL,
così gli arm signal-driven — `entropy_shortcut`,
`entropy_attention_resample` (shift / zoom / router), `attention_highlight` e
il suo controllo `random` — si possono rilanciare tali e quali su
`model=qwen3_vl_2b_attn` / `qwen3_vl_4b_attn`. **Nessuna strategy è stata
toccata**: consumano solo `SignalAnswer`/`VisualAttention`.

## Cosa cambia rispetto a Qwen2.5-VL

### 1. I token visivi non sono contigui

Il processor di Qwen3-VL interleava un timestamp testuale prima di ogni
frame (`processing_qwen3_vl.py:157-170`):

```
<0.0 seconds><|vision_start|>frame0<|vision_end|><1.0 seconds><|vision_start|>frame1<|vision_end|>…
```

Ci sono quindi `t` coppie `vision_start`/`vision_end`, non una. Il codice
precedente prendeva il **primo** start e il **primo** end: su Qwen3 avrebbe
catturato le colonne del solo frame 0 e trattato tutto il resto del video
come "testo della domanda".

`QwenAttentionCapture._capture_spans` risolve il problema per entrambe le
famiglie con la stessa regola:

- **colonne visive** = le posizioni dei placeholder `image_token_id` /
  `video_token_id` (`<|image_pad|>` / `<|video_pad|>`), raccolte come elenco
  di indici (`VIS_INDEX`) invece che come span;
- **righe-query** = tutto ciò che segue l'**ultimo** `vision_end`.

Su Qwen2.5-VL le due definizioni coincidono con quelle di prima (fra start e
end c'è solo il pad ripetuto), quindi la cattura resta **identica** e i
risultati già prodotti restano confrontabili. Verificato su layout sintetici
di entrambe le famiglie.

Conseguenza sui DTO: `VisualAttention.visual_span` è ora `(prima, ultima+1)`
colonna visiva ed è puramente informativo — su Qwen3 fra i due estremi ci
sono anche i token dei timestamp. Nessuno lo usa per fare slicing.

L'attention interface è diventata family-agnostica e si chiama
`qwen_attn_capture`; `qwen25_attn_capture` resta registrato come **alias**
(è il nome nel preset `qwen2_5_vl_3b_attn` e nei config snapshottati).

### 2. Sink dims

Presi dalla tabella di LoT (`lot/retrieval.py:64-79`), che li ha già ricavati
per questa famiglia:

| modello | sink dims |
|---|---|
| Qwen/Qwen3-VL-2B-Instruct | 1793, 1999, 1401 |
| Qwen/Qwen3-VL-4B-Instruct | 0 |
| Qwen/Qwen3-VL-8B-Instruct | 1838 |

`sink_dims_for()` ora **fallisce** su un `model_id` non tabulato invece di
ripiegare sui canali del 3B: i canali outlier sono una proprietà dei pesi, e
usare quelli sbagliati darebbe una `sink_map` di rumore — cioè `top1_pct`,
ramo zoom e cella evidenziata falsi senza che nulla lo segnali.

⚠️ Il 4B ha **un solo** canale (`0`). Da controllare al primo forward reale:
se la `sink_map` risulta quasi costante, il filtro al 25° percentile smette di
isolare una minoranza di token e diventa una soglia arbitraria (punto 7 della
checklist).

### 3. Timestamp e `video_metadata`

Su Qwen3-VL il tempo **è** il testo `<x.x seconds>` che il processor scrive
nel prompt: `get_rope_index` splitta `video_grid_thw` per frame e non esiste
alcun `second_per_grid_ts`. Quei timestamp vengono da
`video_metadata.frames_indices / fps`.

`_prepare_inputs` non ha mai passato `video_metadata`, e per giunta passava
`video_kwargs` come kwarg singolo — che transformers 5.7 **scarta**
(`_merge_kwargs`: «Keyword argument `video_kwargs` is not a valid argument
for this processor and will be ignored»). Senza metadata il processor ripiega
su `fps=24` e `frames_indices=range(T)`: un video di un'ora campionato a 128
frame verrebbe presentato al modello come lungo ~5 secondi.

Da qui `Qwen3VL` in `models/qwen.py`, con `pass_video_metadata = True` e
`image_patch_size = 16` (Qwen2.5-VL: 14). `Qwen3VL2B`/`Qwen3VL4B` ereditano.

### 4. Cosa NON cambia

`temporal_patch_size=2` e `spatial_merge_size=2` come su Qwen2.5 → **una
cella = due frame**, `t = nframes/2`, stessa `GridSpec`, stesso regime
`--double-frames`, stesso significato di `peak_cell`/`top1_pct`. Identici
anche il dispatch su `AttentionInterface`/`AttentionMaskInterface` (niente
layer sliding-window in `Qwen3VLTextConfig`), il path
`model.model.language_model.layers` e `logits_to_keep=1`. `q_norm`/`k_norm`
sono applicate **prima** dell'attention interface
(`modeling_qwen3_vl.py:478-480`), quindi il matmul di cattura ricalcola gli
stessi pesi del kernel. Il deepstack di Qwen3 inietta feature visive nei
primi layer del decoder e non interferisce con la cattura (per lo smoke test
con ViT troncato, `_load` filtra `deepstack_visual_indexes`).

---

## Il collasso M-RoPE su Qwen2.5-VL (finding collaterale)

Lo stesso `video_metadata` mancante ha un effetto diverso — e finora
invisibile — sulla famiglia 2.5: `second_per_grid_ts` viene calcolato da
`metadata.sampled_fps`, che senza metadata vale 24, quindi

```
second_per_grid_ts = temporal_patch_size / 24 = 0.083   # per QUALUNQUE video
time_interval = tokens_per_second * int(0.083) = 0      # modeling_qwen2_5_vl.py:1125
```

cioè **tutte le celle video finiscono sulla stessa posizione temporale
M-RoPE**, in tutte le run fatte finora. Misurato sul processor reale: 0.0833
senza metadata, 25.0 con metadata veri sullo stesso input.

È lo stesso meccanismo del caveat già noto in
`strategies/attention_highlight.py` (il fallback a unità relative quando una
cella dura meno di un secondo), ma non limitato ai video brevi: vale sempre.
Era anche una spiegazione candidata per due osservazioni del README — che il
picco d'attenzione sia in buona parte posizionale, e che `entropy_zoom_24`
non guadagni nulla.

**Non è stato cambiato il comportamento di default.** Il preset Qwen2.5 ha un
knob `pass_video_metadata` (default `false` = regime storico, tutte le misure
del README restano riproducibili bit-per-bit).

**MISURATO, ed è quasi inerte** (`docs/controllo_orologio_mrope.md`).
`baseline_24` e `highlight_top1_24` rilanciati con
`model.pass_video_metadata=true` su Video-MME intero:

```
baseline_24         55.04% → 55.41%   (+0.37 pp, z=0.27)
highlight_top1_24   54.63% → 55.33%   (+0.70 pp)
highlight − baseline:  −0.41 pp → −0.07 pp
```

Cioè: il difetto è reale a codice, ma **non ha invalidato nessun
esperimento**, non spiega il gap col paper (−0.3…−2.2 pp su tutti e quattro
i benchmark) e non riabilita il puntatore d'attenzione. Cade quindi anche la
spiegazione candidata qui sopra: il bias posizionale del picco e il nulla di
`entropy_zoom_24` vanno cercati altrove. Per questo il default resta `false`
— non è più solo inerzia storica, è una scelta senza costo misurato.

---

## Come si usa

```bash
# eval con segnali su Qwen3-VL
uv run python main.py model=qwen3_vl_4b_attn strategy=entropy_attention_resample \
    dataset=video_mme dataset.nframes=24

# gli sbatch esistenti accettano già l'override
MODEL=qwen3_vl_4b_attn sbatch scripts/sbatch/ablation-videomme/highlight_top1_24.sbatch

# cattura standalone per attn_explorer
uv run python -m attn_explorer.capture --model qwen3_vl_4b_attn --dataset video_mme --n 5
```

## Checklist di verifica su HPC

Prerequisito: `HF_HOME=/work/tesi_avalenza/hf`.

1. **Smoke del code path** (layer troncati, output casuale):
   ```bash
   uv run python main.py model=qwen3_vl_2b_attn strategy=entropy_shortcut \
       dataset=egoschema limit=2 model.num_text_layers=2 model.num_vision_layers=2
   ```
2. **Dispatch**: `vlm.model.config.text_config._attn_implementation ==
   "qwen_attn_capture"`, `vision_config` su `"sdpa"`.
3. **Timestamp nel prompt**: `processor.decode(inputs.input_ids[0])[:400]` —
   i `<x.x seconds>` devono coprire la durata vera del video, non 0.0-0.1 s.
4. **Geometria**: `full_visual_attention` non deve sollevare il `ValueError`
   su `t*gh*gw` (il check c'è già); `attn.shape == [n_q, t, gh, gw]` con
   `t == nframes/2`.
5. **Righe-query**: `"".join(q.text for q in va.query_tokens)` deve essere il
   testo della domanda, **non** i timestamp o i frame.
6. **Heatmap** non degenere (non costante, non tutta su una cella).
7. **Sink map, in particolare sul 4B**: distribuzione non piatta e
   `sink_mask(percentile=25)` che seleziona davvero una minoranza di token.
   Se piatta: annotarlo prima di lanciare gli arm che dipendono dal filtro
   sink (zoom, highlight, router).
8. **La generazione non è rotta**: `vlm.generate(...)` dopo la cattura
   produce testo sensato.

Poi: baseline `uniform` su Video-MME `nframes=24`, e gli arm dell'ablation
con `MODEL=qwen3_vl_4b_attn` sullo stesso subset/seed della tabella README.
