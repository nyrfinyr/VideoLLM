# Video LLM

Checkpoint dello stato esperimenti — aggiornato a ogni push. 

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
| entropy_attention_resample  |                 0.7 |                         20 | 65.60\*\* (164/250) |        +2.80 | [`lgs1jrxb`](https://wandb.ai/alesvale97-unimore/video_mme/runs/lgs1jrxb) |
| entropy_attention_resample  |                 0.7 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) (165/250) |        +3.20 | [`y6u5k1pq`](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) |
| entropy_attention_resample  |                 0.7 |                         45 | — (0/250, OOM — vedi nota) |            — | [`tzhm99kn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/tzhm99kn) |
| entropy_attention_resample  |                 1.0 |                         20 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) (165/250) |        +3.20 | [`hmpib0w2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) |
| entropy_attention_resample  |                 1.0 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) (165/250) |        +3.20 | [`igbsy8cp`](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) |
| entropy_attention_resample  |                 1.0 |                         45 | — (0/250, OOM — vedi nota) |            — | [`enb7u9ui`](https://wandb.ai/alesvale97-unimore/video_mme/runs/enb7u9ui) |

La riga et0.7/ct30 è la run di verifica pre-sweep (post-fix OOM
`empty_cache`, vedi `models/qwen_attn.py`) — non fa parte dello sweep group
(`qwen2_5_vl_3b_attn-sweep-video_mme-test`) ma condivide lo stesso subset e
config, quindi il numero è valido e riportato qui.

\*\* `et0.7-ct20`: lo score non è nel summary wandb (che ha solo
`model_latency_mean`/`n_samples`) — recuperato dalla call
`Evaluation.evaluate` su [Weave](https://wandb.ai/alesvale97-unimore/video_mme/weave)
(`mcq_accuracy.correct.true_fraction = 0.656`, 164/250). La call su Weave è
rimasta per un po' in stato `running` con figli parziali subito dopo il
`finished` di wandb — sync lag del trace backend, non perdita dati: dopo
qualche decina di minuti la call è passata a `success` con tutti i 250
sample e lo score sopra. Se un run futuro mostra `finished` su wandb ma
senza `mcq_accuracy` nel summary, controllare prima la call
`Evaluation.evaluate` corrispondente su Weave (via `weave.init(...)` +
`client.get_call(<id>)`) prima di assumere un fallimento: potrebbe essere
solo in ritardo di sync.

**OOM confermato per `concentration_threshold=45`** — `et1.0-ct45` e
`et0.7-ct45` hanno **0/250 predizioni valide**: ogni singola
`predict_and_score` fallisce con `torch.OutOfMemoryError` dentro
`full_visual_attention` (`models/qwen_attn.py:221`, il forward di prefill
per il segnale, chiamato da `generate_with_signals` — **prima** di
qualunque logica di resampling/soglia), traceback verificato via Weave
(`CUDA out of memory ... GPU 0 has a total capacity of 23.64 GiB`, quindi
un nodo 24G). `Evaluation.evaluate` di Weave cattura l'eccezione per
sample invece di far crashare il job, quindi wandb marca il run
`finished` senza errore visibile — solo `mcq_accuracy` risulta assente
perché calcolato su zero sample validi. Il fix `empty_cache` (commit
`d826301`) non basta a evitarlo su GPU 24G con questo carico (128 frame +
capture attention completa su tutti i layer/head, indipendentemente da
`entropy_threshold`/`concentration_threshold`, che intervengono solo
DOPO questo prefill). Probabile che il primo sample capitato OOMi la GPU
e la frammentazione di memoria non si liberi per i sample successivi
(spiegherebbe perché è tutto-o-niente per run, non un parziale). Prossimo
passo: escludere i nodi 24G per questa strategy, es.
`--constraint="gpu_A40_45G|gpu_L40S_45G"` su
`scripts/sbatch/video_mme-signal-test.sbatch`, e rilanciare le due combo
`ct45` mancanti.

## In corso

- Sweep soglie `entropy_attention_resample` su Video-MME (tabella sopra) —
  10/12 combinazioni completate con score (65.6-66.0%, tutte sopra il
  controllo uniform 62.8%, nessuna correlazione chiara ancora fra soglie e
  accuracy nel range testato). Mancano `et0.7-ct45`/`et1.0-ct45`: OOM
  totale su GPU 24G (vedi nota sopra), da rilanciare escludendo quei nodi.
  Poi si sceglie la coppia (entropy_threshold, concentration_threshold)
  migliore contro il controllo uniform, e si confronta Qwen2.5-VL-7B
  contro AdaptToken (vedi `docs/diario/riunione_07-07-26.md`).
- Wiring di Qwen3-VL-2B/4B (stub in `models/qwen3_vl.py`) per popolare le
  altre due colonne della tabella baseline.
