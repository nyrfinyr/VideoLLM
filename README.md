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
confrontabile direttamente). `—` = run non completata (timeout SLURM, vedi
nota "Da rilanciare") o accuracy non loggata (vedi nota "Anomalia").

| strategy                   | entropy_threshold | concentration_threshold | accuracy         | Δ vs uniform | run |
|-----------------------------|-------------------:|--------------------------:|------------------:|-------------:|-----|
| uniform (controllo)         |                   — |                          — | [62.80](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) (157/250) |            — | [`ktvjcyye`](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) |
| entropy_attention_resample  |                 0.3 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) (164/250) |        +2.80 | [`r6khe103`](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) |
| entropy_attention_resample  |                 0.3 |                         30 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) (164/250) |        +2.80 | [`rsw3fq7d`](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) |
| entropy_attention_resample  |                 0.3 |                         45 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) (164/250) |        +2.80 | [`em2fzfr1`](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) |
| entropy_attention_resample  |                 0.5 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) (164/250) |        +2.80 | [`86jyn8n7`](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) |
| entropy_attention_resample  |                 0.5 |                         30 | — (non finita, vedi nota) |            — | [`w3amc9nf`](https://wandb.ai/alesvale97-unimore/video_mme/runs/w3amc9nf) |
| entropy_attention_resample  |                 0.5 |                         45 | — (non finita, vedi nota) |            — | [`w3zvim0p`](https://wandb.ai/alesvale97-unimore/video_mme/runs/w3zvim0p) |
| entropy_attention_resample  |                 0.7 |                         20 | — (non finita, vedi nota) |            — | [`h2a2ibn4`](https://wandb.ai/alesvale97-unimore/video_mme/runs/h2a2ibn4) |
| entropy_attention_resample  |                 0.7 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) (165/250) |        +3.20 | [`y6u5k1pq`](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) |
| entropy_attention_resample  |                 0.7 |                         45 | — (vedi anomalia) |            — | [`tzhm99kn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/tzhm99kn) |
| entropy_attention_resample  |                 1.0 |                         20 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) (165/250) |        +3.20 | [`hmpib0w2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) |
| entropy_attention_resample  |                 1.0 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) (165/250) |        +3.20 | [`igbsy8cp`](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) |
| entropy_attention_resample  |                 1.0 |                         45 | — (vedi anomalia) |            — | [`enb7u9ui`](https://wandb.ai/alesvale97-unimore/video_mme/runs/enb7u9ui) |

La riga et0.7/ct30 è la run di verifica pre-sweep (post-fix OOM
`empty_cache`, vedi `models/qwen_attn.py`) — non fa parte dello sweep group
(`qwen2_5_vl_3b_attn-sweep-video_mme-test`) ma condivide lo stesso subset e
config, quindi il numero è valido e riportato qui.

**Da rilanciare** — `et0.7-ct20`, `et0.5-ct45`, `et0.5-ct30` (lanciate dallo
sweep group, tutte finite su host `huber`/GPU RTX A5000 24G) sono `crashed`
su wandb senza aver loggato nessuna metrica: durata osservata ~2h02-2h13m,
oltre il `SWEEP_TIME=02:00:00` di `scripts/sweep_video_mme_thresholds.sh`.
Le latenze medie delle combinazioni completate (~23-27 s/sample × 250 ⇒
~1.6-1.9h) sono già ai limiti di quel budget su GPU più lente della L40S
usata per calibrarlo (vedi commit "sweep time updated") — coerente con un
timeout SLURM piuttosto che un OOM, ma non verificabile con certezza da
wandb (nessun `output.log` caricato per questi run) senza accesso ai log
SLURM su `ailab02`. Rilancio con `--time` maggiorato:
`scripts/rerun_video_mme_thresholds_missing.sh`.

**Anomalia da verificare** — `et1.0-ct45` e `et0.7-ct45` risultano
`finished` con durata normale (~94-96 min) ma senza `mcq_accuracy`/
`n_correct` nel summary wandb (solo `model_latency_mean`/`n_samples`): da
controllare se lo score è finito solo su Weave o se lo scorer ha fallito
silenziosamente per `concentration_threshold=45`.

## In corso

- Sweep soglie `entropy_attention_resample` su Video-MME (tabella sopra) —
  3 combinazioni da rilanciare per timeout SLURM e 2 con score mancante da
  investigare prima di scegliere la coppia (entropy_threshold,
  concentration_threshold) migliore contro il controllo uniform, poi
  confrontare Qwen2.5-VL-7B contro AdaptToken (vedi
  `docs/diario/riunione_07-07-26.md`).
- Wiring di Qwen3-VL-2B/4B (stub in `models/qwen3_vl.py`) per popolare le
  altre due colonne della tabella baseline.
