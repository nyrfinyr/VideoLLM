# Salvare i .pt durante l'eval per attn_explorer — IMPLEMENTATO (2026-07-14)

Vedi `strategies/base.py::Strategy._save_attention_capture`, `strategies/
entropy_shortcut.py`/`entropy_attention_resample.py` (chiamata + policy),
`main.py::run` (plumbing `capture_dir`), `utils/attn_core.py`
(`reconstruct_frame_indices`/`write_capture`/`capture_index_entry`/
`write_index`, condivisi con `attn_explorer/capture.py` refactored per
riusarli). Design e decisioni sotto restano come riferimento.

Oggi la cattura delle attention map (`AttentionCapture` → `capture.pt`,
vedi `attn_explorer/capture.py::capture_one` + `utils/attn_core.py::save_capture`)
avviene SOLO tramite `attn_explorer.capture` come processo standalone,
disaccoppiato dalla eval pipeline (`main.py`). Le strategy signal-driven
(`entropy_shortcut`, `entropy_attention_resample`, vedi `strategies/`)
calcolano già `VisualAttention` (via `SupportsSignals.generate_with_signals`)
durante l'eval stessa, ma lo buttano via — solo `answer_entropy`/`top1_pct`
finiscono nel dict loggato da Weave, mai il tensore completo.

## Obiettivo

Poter aprire `attn_explorer.serve` su un run di eval vero (non solo su
capture ad-hoc) per ispezionare visivamente PERCHÉ una strategy ha deciso
di ricampionare (o no) su un sample specifico — oggi per farlo bisogna
rilanciare `attn_explorer.capture --dataset ... --video-idx <uuid>` a
mano, sample per sample, fuori dal contesto dell'eval.

## Idea

- Nuovo knob di config (probabilmente su `strategy:` o un top-level
  `capture_attention: false`) per opt-in esplicito — il dump di un
  `capture.pt` per sample ha un costo I/O non trascurabile su dataset
  grandi (2700 sample Video-MME), va tenuto OFF di default.
- Nelle strategy che già hanno un `VisualAttention` sottomano
  (`entropy_shortcut.py`, `entropy_attention_resample.py`), quando il
  flag è attivo: costruire un `AttentionCapture` (stessa forma di
  `capture_one`) e scrivere `<outdir>/<video_stem>/capture.pt` +
  `index.json` compatibile col formato che `attn_explorer.serve` già sa
  leggere — riuso diretto di `utils.attn_core.save_capture`, NON
  reinventare il formato.
- Capire dove appendere l'`outdir`: probabilmente dentro
  `outputs/<run_dir>/captures/` (stesso `run_dir` dello snapshot di
  config, vedi `main.py::snapshot_config`) così un run è autocontenuto
  (config + eventuali capture nello stesso posto).
- Valutare se salvare SEMPRE (quando il flag è attivo) o solo per i
  sample in cui la strategy ha effettivamente ricampionato
  (`resampled=True` in `entropy_attention_resample`) — probabilmente la
  seconda opzione è più utile e molto più economica in I/O.

## Decisioni prese (sessione di design 2026-07-13)

- **Flag**: per-strategy (`capture_attention: false` nel blocco cfg di
  `strategy.entropy_shortcut` e `strategy.entropy_attention_resample` in
  `conf/config.yaml`), non globale — coerente con la convenzione
  esistente ("strategy presets carry tunable knobs when the strategy ha
  qualcuno").
- **Politica di salvataggio**: `entropy_shortcut` salva SEMPRE quando il
  flag è attivo (non ha un concetto di "resampled"); `entropy_attention_
  resample` salva SOLO quando `resampled=True` — molto più economico su
  dataset grandi e sono i casi davvero interessanti da ispezionare.
- **`capture_dir` plumbing**: `main.run` (in `main.py`), subito dopo aver
  costruito la strategy via `Strategy.get(...)`, inietta l'attributo
  `strategy.capture_dir = snapshot_path.parent / "captures"` SOLO se
  `cfg.strategy.get("capture_attention")` — nessuna modifica alla firma
  di `Strategy.get`/`__init__`. `Strategy` (ABC, `strategies/base.py`)
  guadagna un attributo di classe `capture_dir: Path | None = None`.
- **Refactor condiviso**: `extract_cell_frames` (oggi solo in
  `attn_explorer/capture.py`) si sposta in `utils/attn_core.py`, insieme
  a due nuovi helper riusati sia da `attn_explorer.capture.capture_one`
  sia dalle strategy:
  - `reconstruct_frame_indices(video_path, nframes, video_start=None,
    video_end=None) -> tuple[list[int], float]` — linspace uniforme
    ASSOLUTO (indici sul video intero) ristretto alla finestra
    `[video_start, video_end]` se data (oggi `capture.py` copre solo il
    caso video intero, va generalizzato).
  - `write_capture(out_dir, video_path, prompt, visual_attention,
    frame_indices, fps, grid, capture_meta) -> dict` — build
    `AttentionCapture` + `save_capture` + `extract_cell_frames` + entry
    per `index.json`, corpo di `capture_one` estratto.
  Nessuna dipendenza incrociata core→tool: sia `attn_explorer/capture.py`
  sia `strategies/*.py` importano solo da `utils/attn_core.py`.
- **Naming cartella capture**: stem del video (come `attn_explorer.
  capture` già fa oggi), NON `video_idx` — niente modifica alla firma di
  `Strategy.answer`/ai `predict()` closure in `evals/*.py`. Trade-off
  accettato: su dataset con più domande sullo stesso video (Video-MME,
  Causal2Needles) un capture successivo sullo stesso video sovrascrive
  quello precedente nell'`index.json` — accettabile perché il capture è
  comunque opt-in e pensato per ispezione mirata, non per un archivio
  esaustivo.
- **Sample con `frames` pre-estratti** (es. MVBench `data_type="frame"`):
  skip esplicito (warning one-shot) anche col flag attivo — niente
  `video_path`/fps affidabili per ricostruire `frame_indices`/estrarre
  PNG allo stesso modo. Si affronta in futuro se serve davvero.
- **Finestra catturata in `entropy_attention_resample`**: solo il *pass
  1* (pre-resample) ha un `VisualAttention` disponibile — il pass finale
  dopo un eventuale resample chiama `vlm.generate()` semplice, non
  `generate_with_signals`. Il capture riflette quindi sempre la finestra/
  attenzione del pass 1, mai quella ricampionata — coerente con
  l'obiettivo dichiarato ("perché ha deciso di ricampionare"), ma va
  tenuto a mente leggendo il capture in `attn_explorer.serve`.

## Non ancora deciso

- Contenuto esatto di `capture_meta` per i capture prodotti dall'eval
  (oggi `build_capture_meta` in `attn_explorer/capture.py` prende un
  `argparse.Namespace`; va generalizzato per accettare kwargs semplici
  richiamabili anche dalle strategy — quali campi ci mettiamo:
  `resample_kind`/`resampled`? nome dataset/sample? solo model_id+budget?).
