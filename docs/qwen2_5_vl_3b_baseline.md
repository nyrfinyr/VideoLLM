# Baseline Qwen2.5-VL-3B sui 4 benchmark video

> **Stato** (2026-06-03): tutti e 4 i benchmark completati su fullset.
> Setup: `generation=precise` (greedy, `num_beams=1`,
> `max_new_tokens=256`, `repetition_penalty=1.05`), `seed=42`,
> `bfloat16`, `device_map=auto`. LVBench è stato completato re-girando
> lo shard 2 (OOMato su GPU 24 GiB) su GPU ≥ 40 GiB: accuracy *pooled*
> sull'intero dataset **43.00%** (666/1549).

---

## Risultati

| Benchmark | wandb run | nframes / `max_pixels` | Sample | Accuracy | Latency/sample | Wall-clock | Host |
|---|---|---:|---:|---:|---:|---:|---|
| EgoSchema | [`y6ysm2tz`](https://wandb.ai/alesvale97-unimore/egoschema/runs/y6ysm2tz) | 64 / 151'200 | 500/500 | **62.60%** (313/500) | 4.41 s | 37 min | — |
| MVBench | [`srzjo9j9`](https://wandb.ai/alesvale97-unimore/mvbench/runs/srzjo9j9) | 32 / 151'200 | 3748/3800\* | **64.83%** (2430/3748) | 2.65 s | 2h 49m | — |
| Video-MME | [`p7srmrkl`](https://wandb.ai/alesvale97-unimore/video_mme/runs/p7srmrkl) | 128 / 151'200 (no subs, all durations) | 2700/2700 | **59.93%** (1618/2700) | 11.95 s | 8h 59m | — |
| LVBench shard 0 | [`45bdgea9`](https://wandb.ai/alesvale97-unimore/lvbench/runs/45bdgea9) | 768 / 50'176 | 517/517 | **44.49%** (230/517) | 76.0 s | 10h 56m | huber |
| LVBench shard 1 | [`usuy6en7`](https://wandb.ai/alesvale97-unimore/lvbench/runs/usuy6en7) | 768 / 50'176 | 516/516 | **43.99%** (227/516) | 76.1 s | 10h 55m | huber |
| LVBench shard 2 (re-run) | [`sk30qdp6`](https://wandb.ai/alesvale97-unimore/lvbench/runs/sk30qdp6) | 768 / 50'176 | 516/516 | **40.50%** (209/516) | — | — | ≥40 GiB |
| LVBench shard 2 (OOM, scartato) | [`0oqf6fev`](https://wandb.ai/alesvale97-unimore/lvbench/runs/0oqf6fev) | 768 / 50'176 | 0/516 ❌ | **0%** — 1032 OOM | 83.5 s | 11h 59m | nullazzo |
| **LVBench (pooled)** | shard 0+1+2 | 768 / 50'176 | 1549/1549 | **43.00%** (666/1549) | — | — | — |

\* MVBench: weave ha valutato 3800/3800 esempi ma 52 non hanno prodotto
una `pred` parseabile (4 da OOM CUDA + 48 verosimilmente da parser MCQ).
La frazione `true_count / true_fraction` riportata da weave usa 3748 al
denominatore (i `None` non entrano nel computo).

---

## LVBench — eval in shard e re-run dello shard 2

Il run `46824` ha 3 shard SLURM array. I primi due (sull'host `huber`)
sono andati a buon fine. Il terzo è schedulato su `nullazzo` — GPU con
**23.64 GiB di VRAM totali** — e ogni sample chiede ~32 GiB di
allocazione perché LVBench gira a `nframes=768` (≈ 24k token video) →
**1032 OOM CUDA su 516 sample** (in media 2 OOM per esempio, retry
inclusi). Lo stato wandb risulta `finished` perché weave intercetta
l'eccezione per ogni esempio invece di crashare l'intero run; il summary
mostra correttamente `mcq_accuracy: null`.

Lo shard 2 è stato rilanciato (`scripts/sbatch/lvbench-shard2.sbatch`,
hardcoded `shard=2 num_shards=3`, constraint `gpu_A40_45G|gpu_L40S_45G`)
→ run [`sk30qdp6`](https://wandb.ai/alesvale97-unimore/lvbench/runs/sk30qdp6):
**209/516 = 40.50%**. Lo slice strided è deterministico, quindi il re-run
copre esattamente i 516 video mancanti (disgiunti dagli shard 0/1).

### Accuracy finale (pooled sul fullset)

Aggregando i conteggi corretti/totali dei 3 shard (`wandb.group` condiviso):

| Shard | run | corretti / totale | accuracy |
|---|---|---:|---:|
| 0 | `45bdgea9` | 230/517 | 44.49% |
| 1 | `usuy6en7` | 227/516 | 43.99% |
| 2 (re-run) | `sk30qdp6` | 209/516 | 40.50% |
| **pooled** | — | **666/1549** | **43.00%** |

Media *pooled* (pesata sui sample), non media delle 3 percentuali — anche
se con shard di dimensione quasi identica le due coincidono (43.00% vs
42.99%). Il parziale shard 0+1 (44.24%) sovrastimava: lo shard 2 è
risultato più difficile della media, conferma che lo split strided
(`samples[shard::num_shards]`) non garantisce difficoltà uniforme per
shard. **43.00% è ora confrontabile col paper (43.3%, −0.3 punti).**

### Perché `nframes=768` (e `max_pixels=50176` / `min_pixels=3136`)

Il setup replica la metodologia del paper Qwen2.5-VL su LVBench: **≤768
frame, ≤24576 token video**.

- **768 frame** campionati uniformemente sull'intera durata. LVBench è
  long-form (video da 1–2 h): 768 è il massimo del paper e serve a non
  perdere il segmento rilevante, che il modello deve *trovare* da solo
  (`use_time_reference=false`, metodologia ufficiale). Meno frame
  ridurrebbe la copertura temporale e romperebbe il confronto.
- **`max_pixels=50176`** = 64 token/frame (1 token = 28×28 px → 64 token
  = 50176 px). Con il tiling temporale di qwen-vl-utils (2 frame/group):
  `(768/2)·64 = 24576` token video, esattamente il cap del paper. È
  `max_pixels` (non `nframes`) a vincolare la risoluzione del singolo
  frame, dato che i frame reali sono sempre più grandi.
- **`min_pixels=3136`** (4 token/frame): qwen-vl-utils impone un floor di
  128 token/frame (100352 px) sui video; senza abbassare `min_pixels`,
  `max_pixels=50176 < floor` farebbe fallire `smart_resize`
  (`max_pixels >= min_pixels`). Il `min_pixels` basso sblocca i 64
  token/frame voluti. Dettagli throughput in
  [`probe_throughput.md`](probe_throughput.md).

Questo budget (≈24k token video) è anche la ragione dell'OOM su GPU
24 GiB: il re-run a parità di `nframes` richiede GPU **≥ 40 GiB**. Le
alternative degradanti (abbassare `nframes` a 384 o `max_pixels`)
ridurrebbero il budget token ma romperebbero l'allineamento col paper —
scartate a favore del re-run su GPU ampia.

---

## Note sul logging

I 4 run sono partiti con `summary` e `history` di wandb **vuoti**: le
metriche arrivavano solo dal logger di weave (`Eval summary { ... }`
nello stdout) e dal sub-tab Weave del run. Niente era passato a
`wandb.summary`, quindi la run table di wandb non mostrava
`mcq_accuracy` come scalare.

Fix applicato in `main.py` (dopo `evaluation.evaluate`):

```python
import wandb
if wandb.run is not None:
    mcq = (summary or {}).get("mcq_accuracy") or {}
    correct = mcq.get("correct") or {}
    wandb.run.summary["n_samples"] = len(samples)
    if "true_fraction" in correct:
        wandb.run.summary["mcq_accuracy"] = correct["true_fraction"]
        wandb.run.summary["n_correct"]    = correct.get("true_count")
    latency = (summary or {}).get("model_latency") or {}
    if "mean" in latency:
        wandb.run.summary["model_latency_mean"] = latency["mean"]
```

I prossimi run popolano direttamente `mcq_accuracy`, `n_correct`,
`n_samples`, `model_latency_mean` nella run table. Il check su
`true_fraction` gestisce il caso in cui *tutti* i sample falliscano
(es. shard 2 di LVBench): in quel caso `summary['mcq_accuracy']` è
`None` e l'unica cosa che logghiamo è `n_samples`, così il fallimento
resta tracciato.

I 6 run pre-fix elencati sopra sono stati **backfillati via API** con gli
stessi campi (parsati dai log weave), quindi la tabella wandb li mostra
allineati ai run futuri.

---

## Confronto col paper Qwen2.5-VL

| Benchmark | nostro (greedy, fullset) | paper Qwen2.5-VL-3B | gap |
|---|---:|---:|---:|
| EgoSchema | 62.60 | 64.8 | −2.2 |
| MVBench | 64.83 | 67.0 | −2.2 |
| Video-MME (no subs) | 59.93 | 61.5 | −1.6 |
| LVBench | 43.00 | 43.3 | −0.3 |

Gap residuo sui primi tre nell'ordine di **1.5–2.2 punti**, plausibile
con le differenze di prompt format / parser MCQ già discusse in
[`egoschema_reproduction.md`](egoschema_reproduction.md). LVBench sul
fullset (666/1549) è a **43.00%**, a ridosso del paper (−0.3): il
parziale shard 0+1 (44.24%) lo sovrastimava perché lo shard 2 è più
difficile della media.

---

## File rilevanti

- [`scripts/sbatch/lvbench.sbatch`](../scripts/sbatch/lvbench.sbatch) — array job 3 shard, constraint VRAM da correggere
- [`scripts/sbatch/egoschema.sbatch`](../scripts/sbatch/egoschema.sbatch), [`mvbench.sbatch`](../scripts/sbatch/mvbench.sbatch), [`video_mme.sbatch`](../scripts/sbatch/video_mme.sbatch) — job singoli
- [`conf/dataset/`](../conf/dataset/) — `nframes`, `max_pixels` per dataset
- [`conf/generation/precise.yaml`](../conf/generation/precise.yaml) — preset greedy usato per la baseline
- [`evals/base.py`](../evals/base.py) — `mcq_accuracy`, punto in cui aggiungere il flush a `wandb.summary`
