# Video LLM

Checkpoint dello stato esperimenti — aggiornato a ogni push. 

## Struttura del repo

Layout piatto (niente `src/` — nessun pacchetto distribuito, tutto gira
via `uv run python main.py` dal root, vedi `CLAUDE.md` per i dettagli).

| Cartella | Contenuto |
|---|---|
| `main.py` | entrypoint eval (compone config → dataset → strategy → modello) |
| `conf/` | `config.yaml` unico con tutti i preset (model/dataset/strategy/wandb) |
| `evals/` | un modulo per benchmark (EgoSchema, MVBench, Video-MME, LVBench, Causal2Needles) |
| `strategies/` | logica di campionamento/ricampionamento frame (`uniform`, `entropy_shortcut`, `entropy_attention_resample`) |
| `models/` | wiring dei VLM (Qwen2.5-VL, Qwen3-VL, variante con cattura attenzione) |
| `utils/` | helper condivisi (config loader, MCQ parsing, attention core) |
| `attn_explorer/` | tool standalone cattura + esplorazione interattiva delle mappe di attenzione |
| `fetch/` | script di prefetch dei dump dataset (girano su nodo con internet) |
| `scripts/` | sbatch SLURM + script di sweep/lancio |
| `docs/` | analisi, piani, diario — cronologia del ragionamento dietro le scelte |
| `data/`, `outputs/`, `logs/`, `wandb/`, `debug_out/` | scratch locale, gitignored |

## Wired

- **Modelli:** 
  * [Qwen2.5-VL-3B](https://github.com/QwenLM/Qwen2.5-VL)
  * [Qwen3-VL-2B](https://github.com/QwenLM/Qwen3-VL)
  * [Qwen3-VL-4B](https://github.com/QwenLM/Qwen3-VL)
- **Dataset:** 
  * [EgoSchema](https://github.com/egoschema/EgoSchema)
  * [MVBench](https://github.com/OpenGVLab/Ask-Anything)
  * [Video-MME](https://github.com/BradyFU/Video-MME)
  * [LVBench](https://github.com/THUDM/LVBench)

## Baseline (accuracy MCQ)

Greedy (`generation=precise`), seed=42, fullset. 
Numeri del paper Qwen2.5-VL da
[arxiv:2502.13923](https://arxiv.org/abs/2502.13923), Tab. 8.

| Dataset             | `nframes` | Qwen2.5-VL-3B<br>(paper) | Qwen2.5-VL-3B<br>(ours)                                                              | Δ vs paper | Qwen3-VL-2B | Qwen3-VL-4B |
|---------------------|----------:|-------------------------:|--------------------------------------------------------------------------------------|-----------:|-------------|-------------|
| EgoSchema           |        64 |                     64.8 | [62.60](https://wandb.ai/alesvale97-unimore/egoschema/runs/y6ysm2tz) (313/500)       |       −2.2 | —           | —           |
| MVBench             |        32 |                     67.0 | [64.83](https://wandb.ai/alesvale97-unimore/mvbench/runs/srzjo9j9) (2430/3748\*)     |       −2.2 | —           | —           |
| Video-MME (no subs) |       128 |                     61.5 | [59.93](https://wandb.ai/alesvale97-unimore/video_mme/runs/p7srmrkl) (1618/2700)     |       −1.6 | —           | —           |
| LVBench             |       768 |                     43.3 | [43.00](https://wandb.ai/alesvale97-unimore/lvbench/groups/qwen2_5_vl_3b-lvbench) (666/1549) — vedi nota | −0.3 | —           | —           |

\* MVBench: 52/3800 sample senza `pred` parseabile (4 OOM + 48 parser MCQ).

**LVBench**: eval in 3 shard SLURM array, accuracy *pooled* sull'intero
dataset (666/1549 sample). Lo shard 2 originale era finito su una GPU
24 GiB (`nullazzo`) → 1032 OOM su 516 sample, accuracy 0%; è stato
rilanciato su GPU ≥ 40 GiB
([`sk30qdp6`](https://wandb.ai/alesvale97-unimore/lvbench/runs/sk30qdp6),
209/516 = 40.50%) e ricombinato con gli shard 0
([`45bdgea9`](https://wandb.ai/alesvale97-unimore/lvbench/runs/45bdgea9),
230/517) e 1
([`usuy6en7`](https://wandb.ai/alesvale97-unimore/lvbench/runs/usuy6en7),
227/516). In linea col paper (−0.3 punti).

## Sweep `entropy_attention_resample` (Video-MME, in corso)

Subset comune a tutte le righe: `limit=250`, `shuffle=true`, `seed=42`
(stesso subset per ogni combinazione — shuffle deterministico, vedi
`scripts/sweep_video_mme_thresholds.sh`). Grid su `entropy_threshold` ×
`concentration_threshold`, più un controllo `uniform` di baseline sullo
stesso subset. `Δ vs uniform` = accuracy − accuracy del controllo uniform
**su questo stesso subset** (non il 59.93% della tabella Baseline sopra,
che è sull'intero set da 2700 sample — varianza campionaria diversa, non
confrontabile direttamente). `—` = 0 predizioni valide per OOM, vedi nota.

| strategy                   | entropy_threshold | concentration_threshold | accuracy         | Δ vs uniform | run |
|-----------------------------|-------------------:|--------------------------:|------------------:|-------------:|-----|
| uniform (controllo)         |                   — |                          — | [62.80](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) (157/250) |            — | [`ktvjcyye`](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) |
| entropy_attention_resample  |                 0.3 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) (164/250) |        +2.80 | [`r6khe103`](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) |
| entropy_attention_resample  |                 0.3 |                         30 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) (164/250) |        +2.80 | [`rsw3fq7d`](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) |
| entropy_attention_resample  |                 0.3 |                         45 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) (164/250) |        +2.80 | [`em2fzfr1`](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) |
| entropy_attention_resample  |                 0.5 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) (164/250) |        +2.80 | [`86jyn8n7`](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) |
| entropy_attention_resample  |                 0.5 |                         30 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/yp7r9ziy) (164/250) |        +2.80 | [`yp7r9ziy`](https://wandb.ai/alesvale97-unimore/video_mme/runs/yp7r9ziy) |
| entropy_attention_resample  |                 0.5 |                         45 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/mb7cgbq5) (164/250) |        +2.80 | [`mb7cgbq5`](https://wandb.ai/alesvale97-unimore/video_mme/runs/mb7cgbq5) |
| entropy_attention_resample  |                 0.7 |                         20 | — (0/250, OOM — vedi nota) |            — | [`lgs1jrxb`](https://wandb.ai/alesvale97-unimore/video_mme/runs/lgs1jrxb) |
| entropy_attention_resample  |                 0.7 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) (165/250) |        +3.20 | [`y6u5k1pq`](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) |
| entropy_attention_resample  |                 0.7 |                         45 | — (0/250, OOM — vedi nota) |            — | [`tzhm99kn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/tzhm99kn) |
| entropy_attention_resample  |                 1.0 |                         20 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) (165/250) |        +3.20 | [`hmpib0w2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) |
| entropy_attention_resample  |                 1.0 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) (165/250) |        +3.20 | [`igbsy8cp`](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) |
| entropy_attention_resample  |                 1.0 |                         45 | — (0/250, OOM — vedi nota) |            — | [`enb7u9ui`](https://wandb.ai/alesvale97-unimore/video_mme/runs/enb7u9ui) |

La riga et0.7/ct30 è la run di verifica pre-sweep (post-fix OOM
`empty_cache`, vedi `models/qwen_attn.py`) — non fa parte dello sweep group
(`qwen2_5_vl_3b_attn-sweep-video_mme-test`) ma condivide lo stesso subset e
config, quindi il numero è valido e riportato qui.

**OOM totale su 3 combinazioni, legato all'host non alle soglie** —
`et0.7-ct20`, `et0.7-ct45` ed `et1.0-ct45` hanno **0/250 predizioni
valide**: ogni singola `predict_and_score` fallisce con
`torch.OutOfMemoryError` dentro `full_visual_attention`
(`models/qwen_attn.py:221`, il forward di prefill per il segnale, chiamato
da `generate_with_signals` — **prima** di qualunque logica di
resampling/soglia). `Evaluation.evaluate` di Weave cattura l'eccezione per
sample invece di far crashare il job, quindi wandb marca il run
`finished` senza errore visibile — solo `mcq_accuracy` risulta assente
perché calcolato su zero sample validi (attenzione: un run `finished` con
`model_latency_mean`/`n_samples` ma senza `mcq_accuracy` nel summary va
sempre controllato via Weave, `client.get_call(<id>).output`, prima di
fidarsi — non basta guardare lo stato wandb).

**Non è un effetto di `concentration_threshold=45`**: `et0.3-ct45` ed
`et0.5-ct45` sono riusciti senza problemi. È l'**host** — le 3 run fallite
sono girate **tutte e sole** su `nullazzo` (Quadro RTX 6000 24G); `huber`
(RTX A5000, anch'essa 24G) è invece andata bene due volte
(`et1.0-ct20`/`et1.0-ct30`) — dettagli e tabella host/GPU completa in
`docs/analisi_winloss_resampling_video_mme.md`. Il fix `empty_cache`
(commit `d826301`) non basta a evitarlo su quel nodo con questo carico
(128 frame + capture attention completa su tutti i layer/head). Il full
eval (`scripts/sbatch/video_mme-signal.sbatch`) è già pinnato a sole GPU
45G (A40/L40S), quindi esclude comunque `nullazzo`.

## In corso

- Sweep soglie `entropy_attention_resample` su Video-MME (tabella sopra) —
  9/12 combinazioni completate con score (65.6-66.0%, tutte sopra il
  controllo uniform 62.8%). Analisi win/loss sample-per-sample in
  `docs/analisi_winloss_resampling_video_mme.md`: il beneficio netto
  nasconde un turnover di 12-13 sample recuperati contro 5-6 rovinati,
  **sempre gli stessi id** in tutte le combinazioni — `ct` non ha alcun
  effetto misurabile in {20,30,45} su Video-MME (dominio diverso da dove
  le soglie erano state tarate, VNBench). Mancano `et0.7-ct20`/
  `et0.7-ct45`/`et1.0-ct45`: OOM totale, tutte e sole sull'host `nullazzo`
  (vedi nota sopra), da rilanciare escludendo quel nodo. Poi si sceglie la
  coppia (entropy_threshold, concentration_threshold) migliore contro il
  controllo uniform, e si confronta Qwen2.5-VL-7B contro AdaptToken (vedi
  `docs/diario/riunione_07-07-26.md`).
- **Zoom multi-regione disgiunto (Opzione C1)** — implementato e validato
  ([`z4qx9hnb`](https://wandb.ai/alesvale97-unimore/video_mme/runs/z4qx9hnb),
  subset 250, `force_zoom_for_debug`): 64.0%, batte uniform (62.8%) ma non
  `phase_shift` (66.0%) in aggregato, però recupera un pool DISTINTO di ~13
  sample che `phase_shift` non prende (win/loss + stratificazione per n.
  regioni in `docs/piano_zoom_multiregione_disgiunto.md`, "Risultati
  validazione empirica"). **Selettore di ramo (§1.3)**: tre segnali di
  concentrazione confrontati come router concentrato→C1/disperso→phase_shift
  (join a tre vie sui 163 sample ricampionati, stesso subset) — `top1_pct`
  grezzo (plateau 88-90/163) batte sia l'entropia di Shannon normalizzata
  `H/log(t)` (85-87/163) sia lo z-score del picco vs le altre celle
  (87/163, il più debole dei tre). Nessuno dei due segnali "più
  sofisticati" ha battuto il proxy semplice — dettagli in
  `docs/piano_zoom_multiregione_disgiunto.md`. Prossimo passo: Stage C,
  router in produzione con `top1_pct` a soglia adattiva (`ratio ≈
  3.85 × 100/t`, non le soglie assolute fisse 20/30/45 già note come
  inerti su Video-MME).
- Wiring di Qwen3-VL-2B/4B (stub in `models/qwen3_vl.py`) per popolare le
  altre due colonne della tabella baseline.
