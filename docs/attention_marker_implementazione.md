# `attention_marker` — come è implementato

Nota di lettura del codice, per uso personale. Il piano di disegno è
[`docs/piano_marcatori_frame_qwen3.md`](piano_marcatori_frame_qwen3.md); i
risultati sono in
[`docs/260901-tabelle-risultati.md §1`](260901-tabelle-risultati.md). Qui c'è
solo il "come".

File principale:
[`strategies/attention_marker.py`](../strategies/attention_marker.py) — i
riferimenti `:NN` sotto sono a questo file salvo dove indicato diverso.

## 1. Cosa costruisce

L'arm racchiude la cella di picco fra due marcatori testuali inseriti
**dentro** il flusso visivo, spezzando il blocco video in più blocchi:

```
...<|vision_end|><START_IMPORTANT_FRAME><4.5 seconds><|vision_start|>
[frame]<|vision_end|><END_IMPORTANT_FRAME><5.5 seconds>...
```

I marcatori sono testo ordinario, **non** token speciali del vocabolario: si
tokenizzano in ~6 token l'uno.

```python
# strategies/attention_marker.py:97-108
MARKER_START = "<START_IMPORTANT_FRAME>"
MARKER_END = "<END_IMPORTANT_FRAME>"

MARKER_INSTRUCTION = (
    f"Focus on the frames enclosed between {MARKER_START} and {MARKER_END}: "
    "that is where the visual evidence relevant to the question is."
)
```

Per questo [`MARKER_INSTRUCTION`](../strategies/attention_marker.py#L104-L108)
è obbligatoria e costante di modulo, non knob di config — senza, il modello
non ha motivo di trattare i marcatori come un'indicazione. Conseguenza
sperimentale: l'arm è "marcatori + istruzione", quindi il controllo onesto è
la stessa istruzione con celle a caso, non l'assenza di istruzione.

È la variante "B" scartata nel docstring di
[`strategies/attention_highlight.py`](../strategies/attention_highlight.py),
che implementa la "A" (puntatore testuale fuori dal video). Stesso segnale,
stessa scelta delle celle: cambia solo come l'intervento è consegnato.

## 2. Perché solo Qwen3-VL

Su Qwen3-VL il video è **già** `t` isole visive separate da testo (i timestamp
`<x.x seconds>` per frame), quindi `get_rope_index` assorbe un marcatore in
più nel gruppo di testo esistente: nessun `position_ids` da costruire a mano.
Su Qwen2.5-VL il testo inserito spezzerebbe il gruppo video e servirebbe
intervenire sui `position_ids` — fuori scope. Il vincolo è un fail-fast
esplicito in
[`answer`](../strategies/attention_marker.py#L254-L262) (`isinstance(vlm, Qwen3VL)`).

## 3. Il flusso in `answer()`

[`attention_marker.py:234`](../strategies/attention_marker.py#L234). In ordine:

1. **Guardie** ([`:247-283`](../strategies/attention_marker.py#L247-L283)) —
   quattro `RuntimeError` prima di qualsiasi lavoro: modello che supporta i
   segnali, famiglia Qwen3-VL, nessun `frames` pre-estratto dal dataset
   (servono i timestamp reali), `double_frames=false` e `nframes` pari.
2. **Pass 1** — `Video(nframes=budget)` identico ad `attention_highlight`, via
   `vlm.generate_with_signals(...)`, che ritorna `SignalAnswer` con
   `visual_attention` e `answer_entropy`.
3. **Fail-fast su `query_rows`**
   ([`:302-315`](../strategies/attention_marker.py#L302-L315)) —
   [`question_rows`](../utils/attn_core.py#L120) ha un fallback silenzioso
   (via [`_rows_before_marker`](../utils/attn_core.py#L75), che ritorna TUTTE
   le righe se non trova `"options:"`): l'arm degraderebbe a `query_rows=all`
   credendo di misurare `question`. Su Video-MME
   [`format_mcq_prompt`](../evals/base.py#L59) scrive sempre `"Options:"`,
   quindi questa è un'asserzione, non un ramo atteso — il sample fallisce
   rumorosamente e Weave lo cattura:

   ```python
   # strategies/attention_marker.py:307-308
   if self.query_rows != "all" and n_query_rows >= n_query_tokens:
       raise RuntimeError(...)
   ```

4. **Ranking delle celle**
   ([`:317-337`](../strategies/attention_marker.py#L317-L337)) —
   [`ranked_cells_from_attention`](../utils/attn_core.py#L616) classifica
   tutte le `t` celle per massa d'attenzione, mediando sulle righe scelte da
   [`ROW_SELECTORS[query_rows]`](../utils/attn_core.py#L142-L146). Con
   `sink_filter=false` si usa l'attenzione grezza. Si calcola **anche** la
   cella che sceglierebbe il filtro acceso (`peak_cell_sink_filtered`): serve
   solo come check che il knob arrivi davvero al ranking.
5. **Selezione**
   ([`_select_cells`, `:219`](../strategies/attention_marker.py#L219-L232)) —
   `attention`: le `top_k` per massa, con astensione se la massa totale è
   nulla. `random`: `top_k` celle a sorte.
6. **Frame reali** — `RankedCell.frames/rep_frame/t_sec` sono **dummy**
   ([`attn_core.py:616`](../utils/attn_core.py#L616) li riempie con
   `range(t)`, `fps=1.0`): non vanno usati. Gli indici veri si ricostruiscono
   con [`reconstruct_frame_indices`](../utils/attn_core.py#L796), che replica
   la formula di campionamento di qwen-vl-utils. Poi
   [`_extract_all_frames`](../strategies/attention_marker.py#L146) estrae
   **tutti** i frame in PNG su una tmpdir.
7. **Blocchi** —
   [`_merge_adjacent_cells`](../strategies/attention_marker.py#L110) fonde
   celle adiacenti in run (`[3,4,7]` → `[(3,4),(7,7)]`, una sola coppia di
   marcatori per run), e
   [`_split_blocks`](../strategies/attention_marker.py#L123) produce
   `(start, end, marked)` in posizioni della lista campionata, tagliando sui
   confini di cella `2*ti` / `2*(ti+1)`:

   ```python
   # strategies/attention_marker.py:133-143
   blocks: list[tuple[int, int, bool]] = []
   prev = 0
   for a, b in runs:
       start, end = 2 * a, 2 * (b + 1)
       if start > prev:
           blocks.append((prev, start, False))
       blocks.append((start, end, True))
       prev = end
   if prev < n_frames:
       blocks.append((prev, n_frames, False))
   return blocks
   ```

8. **Assemblaggio**
   ([`:377-396`](../strategies/attention_marker.py#L377-L396)) — la content
   list alterna testo e blocchi video, e chiude con istruzione + prompt:

   ```python
   # strategies/attention_marker.py:377-396
   parts: list[VideoFrames | Text] = []
   for start, end, is_marked in blocks:
       if is_marked:
           parts.append(Text(MARKER_START))
       parts.append(VideoFrames(
           paths[start:end],
           max_pixels=budget.max_pixels,
           min_pixels=budget.min_pixels,
           frames_indices=[int(i) for i in frame_indices[start:end]],
           fps=fps,
       ))
       if is_marked:
           parts.append(Text(MARKER_END))
   parts.append(Text(f"{MARKER_INSTRUCTION}\n\n{prompt}"))
   ```

9. **Pass 2** —
   [`vlm.build_messages_from_parts(parts)`](../models/qwen.py#L175) e
   `generate`. La tmpdir viene ripulita nel `finally`
   ([`:420-422`](../strategies/attention_marker.py#L420-L422)).

Se il gate è chiuso o non c'è segnale, il pass 2 gira sugli stessi frame del
pass 1 con prompt **invariato**: niente istruzione senza marcatori, sarebbe un
terzo intervento non controllato.

## 4. Il canale metadata — la parte delicata

Spezzare il video in più blocchi rompe l'orologio, se non si interviene:
`qwen_vl_utils.fetch_video` fabbrica metadata finti con
`frames_indices=range(n)` **locali al blocco**, quindi i timestamp
ripartirebbero da zero a ogni marcatore. La catena che lo evita:

| dove | cosa |
|---|---|
| [`models/media.py:80-81`](../models/media.py#L80-L81) | `VideoFrames.frames_indices` (indici **assoluti** nel sorgente) + `fps` reale. [Validati insieme](../models/media.py#L84-L92): uno solo dei due è un errore. |
| [`models/qwen.py:175`](../models/qwen.py#L175-L196) | `build_messages_from_parts` li mette nel content dict sotto la chiave privata [`_VIDEO_METADATA_KEY`](../models/qwen.py#L19-L25). |
| [`models/qwen.py:199`](../models/qwen.py#L199-L246) | `_extract_manual_video_metadata` la **estrae e rimuove**, e ritorna `{indice blocco video → VideoMetadata}`. |
| [`models/qwen.py:250`](../models/qwen.py#L250) | `_prepare_inputs` passa i `VideoMetadata` per blocco al processor, con `do_sample_frames=False`. |

```python
# models/qwen.py:187-196 — l'aggancio del canale privato
content = []
for part in parts:
    d = to_content_dict(part)
    if isinstance(part, VideoFrames) and part.frames_indices is not None:
        d[_VIDEO_METADATA_KEY] = {
            "fps": float(part.fps),
            "frames_indices": [int(i) for i in part.frames_indices],
        }
    content.append(d)
return [{"role": "user", "content": content}]
```

La chiave privata non deve arrivare né al chat template né a
`process_vision_info`: la rimozione avviene in
[`_extract_manual_video_metadata`](../models/qwen.py#L224-L243), che conta i
content item `type == "video"` in ordine di documento — lo stesso ordine in
cui `process_vision_info` estrae i video, quindi gli indici sono allineati
alla lista `video_metadata` del processor.

[`build_messages_from_parts`](../models/qwen.py#L175) è la generalizzazione di
[`build_messages`](../models/qwen.py#L156) a media e testi interleaved in
numero arbitrario — esiste per questo arm.

## 5. Le tre invarianti

Dal piano §2, tutte difese nel codice:

1. **I token visivi non cambiano.** Spezzare il blocco lascia la somma dei `t`
   invariata; si pagano solo i ~12 token di testo dei marcatori.
2. **I metadata vanno passati a mano per blocco** — §4 qui sopra.
3. **I tagli devono cadere su confini di cella**, cioè blocchi di lunghezza
   **pari** nel regime `double_frames=false`. Un blocco dispari viene paddato
   dal processor, `t` cresce e **tutti** i timestamp si sfasano in silenzio.
   `_split_blocks` taglia su confini pari per costruzione, e in `answer` c'è
   comunque un assert che rialza se un blocco dispari compare:

   ```python
   # strategies/attention_marker.py:367-375
   if any((end - start) % 2 != 0 for start, end, _ in blocks):
       raise RuntimeError(
           f"blocco di lunghezza dispari in {blocks} — "
           "invariante §2.4 violata, timestamp sfasati."
       )
   ```

   L'unità marcabile è quindi la **cella** (2 frame), che è già la risoluzione
   reale del segnale.

## 6. L'arm di controllo

`cell_select=random` tiene tutto identico (stessa istruzione, stessi
marcatori, stesso costo) e sorteggia le celle. Il seed viene da
[`_sample_rng`](../strategies/attention_highlight.py#L155-L170), che deriva da
`(random_seed, video_path, prompt)`: la cella estratta è una **funzione pura
del sample**, stabile fra shard, rilanci e macchine. Un `Random` globale
darebbe celle diverse a seconda dell'ordine di esecuzione, che con
`Evaluation` concorrente non è garantito.

```python
# strategies/attention_marker.py:228-232
if self.cell_select == "random":
    return rng.sample(range(t), k=min(self.top_k, t))
if all(c.pct <= 0 for c in ranked):
    return None
return [c.cell for c in ranked[: self.top_k]]
```

Differenza deliberata: con `attention` ci si astiene a massa nulla, con
`random` no — la cella non dipende dall'attenzione, e far saltare il controllo
dove salta l'arm vero ne restringerebbe il campione senza motivo.

## 7. Campi emessi per-sample

Il [dict di ritorno](../strategies/attention_marker.py#L424-L465) finisce in
Weave. Oltre a `raw`/`pred`:

- `n_query_rows`, `n_query_tokens` — check che `query_rows` tagli davvero.
- `peak_cell`, `peak_cell_sink_filtered` — se coincidessero sempre, il knob
  `sink_filter` non arriverebbe al ranking.
- `block_sizes`, `t_cells` — `somma(block_sizes)/2 == t_cells` verifica che
  nessun blocco sia dispari.
- `marked`, `marked_cells`, `n_blocks`, `marked_skip_reason` — copertura
  dell'intervento (`marked` **non** è nel summary wandb: si legge solo da
  Weave).
- `cell_select`, `sink_filter` — ri-emessi per sample così che il driver
  [`/wandb`](../.claude/skills/wandb/driver.py) possa verificare
  l'accoppiamento run↔Evaluation.
- `pred_pass1` — la risposta **prima** dei marcatori, che abilita il breakdown
  `correct_pass1_*` di
  [`evals/base.py::mcq_accuracy`](../evals/base.py#L136-L205): "i marcatori
  hanno rotto un sample già corretto?" si legge intra-sample, senza join fra
  run.

## 8. Config e lancio

Preset in [`conf/config.yaml:218-226`](../conf/config.yaml#L218-L226) sotto
`strategy.attention_marker` (`always_highlight: true`, `top_k: 1`,
`cell_select: attention`, `query_rows: question`, `sink_filter: false`,
`random_seed: 0`). `sink_percentile`/`sink_border` non sono nel preset:
valgono i [default del codice](../strategies/attention_marker.py#L198-L199), e
contano solo con `sink_filter=true`.

Sbatch:
[`scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch`](../scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch).

```bash
# fullset, arm treatment
sbatch scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch

# arm di controllo
sbatch scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch \
    strategy.cell_select=random \
    "wandb.tags=[ablation,video_mme,qwen3_vl_2b,marker_random_24]"

# probe da 60 (gruppo wandb separato, LIMIT applicato DOPO lo shard)
LIMIT=60 sbatch --array=0 --time=00:40:00 \
    scripts/sbatch/ablation-videomme/marker_24_qwen3.sbatch
```

⚠️ L'sbatch fissa `NAME="marker_24"` e l'esempio di controllo sovrascrive solo
i tag: le run dei due arm si chiamano **identiche** a meno del job id.
