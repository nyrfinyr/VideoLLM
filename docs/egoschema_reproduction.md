# Riproduzione del numero del paper su EgoSchema

> **Stato** (2026-05-25): mio run = **58.6**, paper Qwen2.5-VL-3B = **64.8** → gap **6.2 punti**.
> Investigazione su cosa diverge dal setup canonico (lmms-eval) e cosa
> serve cambiare per chiudere il gap. **Continua domani.**

---

## Sommario delle cause probabili (ordinate per impatto stimato)

| # | Causa | Patch | Impatto stimato |
|---|---|---|---|
| A | Sampling stocastico invece di greedy | `generation=precise` (CLI) | +1-2 punti |
| B | Prompt format diverso da lmms-eval | refactor `EgoSchema.predict_factory` | +2-3 punti |
| C | Parser troppo stretto (no fallback su testo option né random) | nuova `parse_multi_choice_response` | +1-3 punti |
| D | Pochi frame campionati (`fps=0.25` → 45 frame su 180s) | `dataset.fps=2.0` (CLI) | impatto incerto ma probabilmente significativo |
| E | `max_pixels` più basso del default lmms-eval | `model.max_pixels=1605632` (CLI) | marginale |

### Combinazione minima da provare per prima

```bash
uv run python main.py \
    dataset=egoschema dataset.fps=2.0 \
    model.max_pixels=1605632 model.min_pixels=200704 \
    generation=precise
```

Se questo non chiude il gap, applicare anche le patch B e C nel codice.

---

## Setup canonico (lmms-eval) — riferimento autoritativo

lmms-eval è il framework usato dal team Qwen per i benchmark del paper.
Le info qui sotto sono estratte verbatim da
[`EvolvingLMMs-Lab/lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)
(branch `main` al 2026-05-25).

### 1. Task config — `lmms_eval/tasks/egoschema/egoschema.yaml`

```yaml
dataset_name: "GENERATION"
task: "egoschema"
test_split: test
output_type: generate_until
doc_to_visual: !function utils.egoschema_doc_to_visual
doc_to_text: !function utils.egoschema_doc_to_text
doc_to_target: !function utils.egoschema_doc_to_answer
process_results: !function utils.egoschema_process_results_generation
metric_list:
  - metric: submission
    aggregation: !function utils.egoschema_aggregate_mc
    higher_is_better: true
include: _default_template_yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."
```

Il dataset è `lmms-lab/egoschema`, split `test`. **Fullset = 5031 question.**
Esiste anche `egoschema_subset.yaml` (subset da 500). Il numero 64.8 del
paper è sul fullset — verificare di valutare sullo stesso denominatore.

### 2. Prompt format esatto (da `utils.egoschema_doc_to_text`)

```python
question = doc["question"]
for op in doc["option"]:                # le option arrivano già "A. ...", "B. ..."
    question += "\n" + op               # → si concatenano AS-IS, niente riformattazione
return f"{question}\nAnswer with the option's letter from the given choices directly."
```

Resa finale:

```
<domanda>
A. <opt>
B. <opt>
C. <opt>
D. <opt>
E. <opt>
Answer with the option's letter from the given choices directly.
```

**Differisce dal mio attuale `format_mcq_prompt` su 4 punti:**

| | mio | lmms-eval |
|---|---|---|
| Prefisso domanda | `Question: <q>` | nessuno |
| Header option | `Options:\n` | nessuno |
| Letter format | `(A) ...` | `A. ...` |
| Trailing | "the letter of the correct option only" | "the option's letter from the given choices directly" |

Inoltre il mio loader **strippa** "A. " dalle option e rifa il rendering
con `(A)`; lmms-eval le **lascia as-is**.

### 3. Parser MCQ — `parse_multi_choice_response`

Strategia multi-step, molto più tollerante della mia (`\b([A-E])\b`):

1. Cerca `(A)`, `(B)`, ... (parens)
2. Cerca `A `, `B `, ... (letter + space)
3. Cerca `A.`, `B.`, ... (letter + dot)
4. **Fallback**: se la risposta ha >5 token, cerca il **testo dell'option**
   nella risposta (es. "I think it walks" → match "walks" → A)
5. **Ultimo fallback**: random choice tra A-E (boost gratuito al ~20% sui
   casi senza match)
6. Se più candidati: si prende quello con `rfind` minore (il primo nella
   stringa)

Il mio `parse_mcq_letter` riconosce solo casi puliti: tutto il resto →
`None` (= sbagliato). Il solo random fallback di lmms-eval può valere 1-2
punti su un benchmark dove il modello a volte risponde verbosamente.

### 4. Model wrapper — `lmms_eval/models/simple/qwen2_5_vl.py`

**Defaults rilevanti** (ma probabilmente overridden via `--model_args`
dal team Qwen per il paper):

```python
def __init__(
    self,
    ...
    min_pixels: int = 256 * 28 * 28,         # = 200_704  (≈ 448x448)
    max_pixels: int = 1_605_632,             # = 2048 token/frame  (≈ 1267x1267)
    max_num_frames: int = 32,                # ← molto basso
    fps: Optional[float] = None,             # se None: linspace(0, T-1, max_num_frames)
    ...
)

# generation default (sovrascrivibili via gen_kwargs):
default_gen_kwargs = {
    "max_new_tokens": 32768,
    "temperature": 0.0,        # ← GREEDY by default
    "top_p": None,
    "num_beams": 1,
}
```

**Punto importante**: `max_num_frames=32` di default. Su un video EgoSchema
da 180s → 0.18 fps effettivi. È PIÙ BASSO del mio `fps=0.25`. Quindi se
Qwen avesse usato i puri default lmms-eval (32 frame) è improbabile
raggiungere 64.8 — devono aver overridden `max_num_frames` o `fps`.

Il paper Qwen2.5-VL afferma:

> For all evaluated benchmarks, we capped the maximum number of frames
> analyzed per video at 768, with the total number of video tokens not
> exceeding 24,576.

Sono i loro cap globali, non i default lmms-eval. La combo esatta
(`max_num_frames` + `fps` + `max_pixels`) usata per EgoSchema non è
documentata pubblicamente; si presume vicina al budget massimo.

---

## Mio setup attuale vs paper — i numeri

```
Token per video Qwen2.5-VL = total_pixels / (28*28*2)
                            = total_pixels / 1568
```

| | fps | frame su 180s | token/frame | total token |
|---|---|---|---|---|
| Mio attuale | 0.25 | 45 | 512 | ~11,520 |
| Paper (probabile) | ~2.0 | ~360 | ~68 | ~24,576 |

Sto usando 8× meno frame del paper e ~2× meno token totali. È coerente
con un gap di accuracy: EgoSchema è temporal-heavy (domande tipo "what
does X do AFTER Y"), più frame → più informazione.

### Costanti `qwen-vl-utils` rilevanti

```python
FPS = 2.0
FPS_MAX_FRAMES = 768          # ← esatto numero del paper
FPS_MIN_FRAMES = 4
VIDEO_MAX_TOKEN_NUM = 768     # per frame, ceiling (≈ 1097×1097)
VIDEO_MIN_TOKEN_NUM = 128     # per frame, floor (≈ 448×448)
FRAME_FACTOR = 2              # temporal merge
SPATIAL_MERGE_SIZE = 2        # spatial merge
```

---

## Patch consigliate (codice)

### Patch B — prompt format lmms-eval per EgoSchema

```python
# evals/egoschema.py — loader
return [
    {
        "video_idx": r["video_idx"],
        "video_path": str(root / r["video"]),
        "question": r["question"],
        "options": r["option"],          # NON strippare "A. ..."
        "answer": int(r["answer"]) if r.get("answer") is not None else None,
    }
    for r in rows
]

# evals/egoschema.py — predict_factory
def predict(video_path, question, options):
    body = "\n".join(options)            # options già "A. ...", "B. ..."
    prompt = (
        f"{question}\n{body}\n"
        "Answer with the option's letter from the given choices directly."
    )
    media = Video(video_path, fps=fps)
    messages = vlm.build_messages(media, Text(prompt))
    raw = vlm.generate(messages, generation_config=gen_cfg)
    return {"raw": raw, "pred": parse_multi_choice_response(raw, len(options))}
```

Il prompt diventa **dataset-specifico**: non passare più per
`format_mcq_prompt` di `base.py` (che resta valido per gli altri dataset
finché non li mappiamo anche loro).

### Patch C — parser multi-step

Da aggiungere a `evals/base.py` (o per-dataset). Modello: copia diretta
da `lmms_eval/tasks/egoschema/utils.py:parse_multi_choice_response`,
adattata alla firma `(raw, n_options) -> int | None`. ~30 righe.

Pseudocodice:

```python
def parse_multi_choice_response(raw: str, n_options: int) -> int | None:
    # Step 1-3: cerca "(A)", "A ", "A." nella risposta
    # Step 4: fallback su testo option (se passato)
    # Step 5: random fallback su A..E[:n_options]
    ...
```

**Decisione aperta**: il random fallback di lmms-eval gonfia
artificialmente l'accuracy (~20% sui casi senza match). Ha senso
includerlo se l'obiettivo è confronto-paper. Se invece misuriamo
"quante volte il modello ha risposto in modo parseabile e corretto",
meglio tenere `None` come oggi.

### Patch A — greedy decoding

Banale, già esiste `conf/generation/precise.yaml` con `do_sample=false,
num_beams=1, max_new_tokens=256`. Solo override CLI:

```bash
uv run python main.py generation=precise
```

Considera di alzare `max_new_tokens` se i casi nulli sono dovuti a
troncamenti (paper Qwen usa 32768, eccessivo per noi — 64 dovrebbe
bastare per una risposta MCQ).

---

## Cosa fare domani

1. **Verifica fullset vs subset**: stampa `len(samples)` nel mio run.
   Deve essere ~5031 per matchare il numero del paper. Se è 500, sto
   valutando il subset e i numeri non sono comparabili.
2. **Run rapido con CLI override** (Patch A+D+E senza toccare codice):

   ```bash
   uv run python main.py \
       dataset=egoschema dataset.fps=2.0 \
       model.max_pixels=1605632 model.min_pixels=200704 \
       generation=precise
   ```

   → Confronta con 58.6. Se sale a >62, il problema era frame+greedy.
3. **Se persiste gap**: applica Patch B (prompt) e Patch C (parser).
4. **Cross-check**: lancia lmms-eval localmente sul modello, su EgoSchema
   fullset, default config. Numero = "ground truth lmms-eval" che dovrei
   raggiungere.

   ```bash
   pip install lmms-eval
   python -m lmms_eval --model qwen2_5_vl \
       --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
       --tasks egoschema --batch_size 1
   ```

---

## Riferimenti

- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [lmms-eval/tasks/egoschema/](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/egoschema)
- [lmms-eval/models/simple/qwen2_5_vl.py](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/simple/qwen2_5_vl.py)
- [Qwen2.5-VL technical report](https://github.com/QwenLM/Qwen2.5-VL) (vedi sezione benchmark / appendice)
- File del progetto rilevanti:
  - [`evals/egoschema.py`](../evals/egoschema.py) — loader + predict_factory
  - [`evals/base.py`](../evals/base.py) — `format_mcq_prompt`, `parse_mcq_letter`, `mcq_accuracy`
  - [`conf/dataset/egoschema.yaml`](../conf/dataset/egoschema.yaml) — fps, root
  - [`conf/model/qwen2_5_vl_3b.yaml`](../conf/model/qwen2_5_vl_3b.yaml) — min/max_pixels
  - [`conf/generation/precise.yaml`](../conf/generation/precise.yaml) — greedy preset
