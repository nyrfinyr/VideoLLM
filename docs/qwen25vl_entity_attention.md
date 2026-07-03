# Qwen2.5-VL — attenzione entity → token visivi (`Qwen25VLAttention`)

> **Stato** (2026-06-03): implementazione completa, verificata in locale
> (import, registrazioni, composizione Hydra, ricerca span col tokenizer
> vero). **Non ancora verificata su GPU** — checklist HPC in fondo.

---

## Obiettivo

Estrarre i pesi di attenzione `softmax(q·kᵀ)` dei **token dell'entity
nella domanda** verso i **token visivi**, per individuare a quali celle
dell'immagine corrispondono i valori più alti. Riferimento concettuale:
`lot/new_forward.py` (monkey-patch dei forward), riscritto però in modo
leggibile sfruttando i registry di transformers 5.x — niente riscrittura
di `self_attn`/decoder layer.

## Come funziona

Tutto vive in `models/qwen_attn.py`:

1. **Attention interface custom** (`qwen25_attn_capture`), registrata in
   `AttentionInterface` col nome `qwen25_attn_capture`. Il `self_attn`
   di serie la risolve a ogni forward via
   `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation)`.
   - **Output**: SDPA (kernel fuso, niente matrice S×S) → il modello
     genera normalmente.
   - **Pesi**: matmul "chirurgico" solo sulle righe `start:end` dello
     span entity → `[B, H, n_ent, S]`, mask additiva + softmax in fp32.
     Memoria O(n_ent·S), non O(S²).
   - **Cattura**: stash in `module._last_attn` (in transformers 5.x
     `outputs.attentions` non è più popolato: il loop dei decoder layer
     non raccoglie i pesi — verificato su `modeling_qwen2_5_vl.py:927`).
2. **Mask interface** registrata sotto lo stesso nome con `eager_mask`:
   obbligatoria (lookup `ALL_MASK_ATTENTION_FUNCTIONS[nome]` → `KeyError`
   senza), e scelta perché additiva (0/−inf) e **sempre materializzata**
   (la mask `sdpa` può tornare `None` per "causal skip", che falserebbe
   la normalizzazione dello span a metà sequenza).
3. **`ENTITY_SPAN` globale + `set_entity_span()`**: lo span è per-sample
   (noto solo dopo la tokenizzazione), la interface lo legge a ogni
   invocazione. Il setter usa `global` — mai riassegnare il nome
   importato con `from ... import ENTITY_SPAN`.
4. **`Qwen25VLAttention.entity_visual_attention(media, text, entity)`**:
   prepara gli input (ereditati da `Qwen25VL3B`) → trova lo span →
   forward di prefill (`no_grad`, `logits_to_keep=1`; **niente
   generate**, l'attenzione entity→immagine esiste già nel prefill) →
   stack dei `_last_attn` per layer → media su metà centrale dei layer,
   teste e token entity → slice `[vis_start+1:vis_end]` (token id
   151652/151653 dal config) → reshape a `[t, h//2, w//2]` via
   `grid_thw` e `spatial_merge_size`. `set_entity_span(None)` in
   `finally`.

## Decisioni chiave (e perché)

| Decisione | Motivo |
|---|---|
| Interface registry, non monkey-patch | in 5.7 il `self_attn` di serie dispatcha già sul registry: si riscrive solo la funzione di attenzione, non i forward |
| SDPA per l'output, non FA2 letterale | FA2 non materializza mai i pesi: con `attn_implementation="flash_attention_2"` non esistono proprio. SDPA dispatcha comunque su kernel flash |
| `attn_implementation` **a dict** nello yaml (`text_config` / `vision_config`) | anche il ViT dispatcha sul registry, con `attention_mask=None, is_causal=False`; la custom lì corromperebbe le feature visive (causale in un encoder bidirezionale) |
| `is_causal if is_causal is not None else attention_mask is None` nella SDPA | rispetta il chiamante quando esplicito (ViT), evita il doppio mascheramento del pattern di `new_forward.py` (`is_causal=True` + mask insieme) |
| mask additiva sommata allo slice **prima** del softmax | i token entity stanno a metà sequenza: senza mask il softmax normalizzerebbe anche sui key futuri, sgonfiando la quota sui token visivi |
| global + setter invece della closure | lo span non è noto alla registrazione (avviene all'import, prima di `from_pretrained`); con una tuple immutabile la closure-su-mutable non ha più senso |

## File toccati

- `models/qwen_attn.py` — interface + registrazioni + `Qwen25VLAttention`.
- `conf/model/qwen2_5_vl_3b_attn.yaml` — `name: qwen25_vl_3b_attn`,
  `attn_implementation` a dict. Le chiavi extra dello yaml fluiscono
  `yaml → costruttore → _load(**kwargs) → from_pretrained`.
- `models/__init__.py` — export `Qwen25VLAttention`.
- `main.py` — entry `"qwen25_vl_3b_attn"` in `_MODELS` (import lazy
  dentro `run()`, come gli altri: `HF_HOME` va settato prima che
  transformers venga importato).
- Refactor collaterale di `main.py`: shuffle/shard/limit →
  `utils/samples.py::prepare_samples`; flush wandb →
  `utils/obs.py::log_eval_summary`.

## Già verificato in locale (WSL, senza GPU)

- import del modulo → entrambe le registrazioni presenti nei registry;
- `model=qwen2_5_vl_3b_attn --cfg job` → config composto col dict;
- `_find_entity_span` col tokenizer Qwen vero: `"snow leopard"` →
  span `(7, 9)` → decode `' snow leopard'`; entity assente → `ValueError`;
- `prepare_samples` / `log_eval_summary`: smoke test funzionale.

## Checklist verifica su HPC

Prerequisito: `HF_HOME=/work/tesi_avalenza/hf` (knob `hf_home` nel
config Hydra).

1. **Wiring eval invariato** (smoke, layer troncati → output casuale ma
   pipeline completa):
   ```bash
   uv run python main.py model=qwen2_5_vl_3b_attn model.num_text_layers=2 \
       model.num_vision_layers=2 limit=1
   ```
   Deve completare come `qwen25_vl_3b`.
2. **Dispatch corretto** (Python interattivo, modello pieno):
   ```python
   vlm.model.config.text_config._attn_implementation   # "qwen25_attn_capture"
   vlm.model.config.vision_config._attn_implementation # "sdpa"
   ```
3. **Stash popolato**: dopo `entity_visual_attention(...)`, ogni
   `vlm.model.model.language_model.layers[i].self_attn._last_attn`
   deve avere shape `[1, H, n_ent, S]` (3B: H=16) con `n_ent` =
   lunghezza span entity.
4. **Sanity della heatmap**:
   ```python
   out = vlm.entity_visual_attention(Image(...), Text("...<entity>..."), "<entity>")
   out["heatmap"].shape            # [t, grid_h, grid_w], t=1 per immagine
   out["heatmap"].flatten().topk(5)
   out["visual_scores"].sum()      # massa di attenzione sull'immagine, < 1
   ```
   Su un'immagine con oggetto evidente, il top-k deve cadere sulle
   celle dell'oggetto (o sui bordi/sink — vedi caveat).
5. **La generazione non è rotta**: `vlm.generate(messages, gen_cfg)`
   dopo l'analisi deve produrre testo sensato (output via SDPA: la
   custom non altera il path di generazione; `ENTITY_SPAN` viene
   resettato nel `finally`).
6. **Nessun check del `ValueError`** su `t*gh*gw`: se scatta, il layout
   `grid_thw`/merge non è quello atteso — fermarsi e ispezionare.

## Caveat noti

- **Assunzioni**: `bsz=1`, niente padding, sequenza < sliding window.
  La mask `eager` materializzata è O(S²) di memoria (è la mask, non
  l'attenzione): irrilevante per immagini singole, da tenere d'occhio
  con contesti lunghissimi.
- `seq_scores` è normalizzato su **tutta** la sequenza: per il ranking
  dei token visivi va bene così; rinormalizzare lo slice solo per
  confrontare distribuzioni tra sample.
- I picchi possono essere catturati dai **sink token** (attivazioni
  parassite): se le heatmap escono "sporche", il raffinamento successivo
  è il filtro stile `SINK_DIMS`/percentile di `lot/retrieval.py:64-79`.
- `ENTITY_SPAN` è stato globale: ok single-thread
  (`WEAVE_PARALLELISM=1`), non thread-safe.

## Possibili prossimi passi

1. Verifica HPC (checklist sopra).
2. Overlay heatmap → immagine (stile `visualiza_attn_map` di
   `lot/retrieval.py:143`) per ispezione visiva.
3. Filtro sink token se necessario.
4. Estensione ai video (la heatmap ha già la dimensione `t`; da
   validare `video_grid_thw` e l'aggregazione temporale).
