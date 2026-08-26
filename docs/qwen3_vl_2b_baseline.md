# Baseline Qwen3-VL-2B — Video-MME (24 frame)

> **Stato** (2026-08-26): baseline completata su fullset (2700 sample).
> Setup: `model=qwen3_vl_2b_attn`, `strategy=uniform`, `dataset.nframes=24`,
> preset `_attn`, `generation=precise` (greedy), `seed=42`,
> `torch_dtype=bfloat16`, Video-MME senza sottotitoli. Job array a 3 shard
> SLURM da 900 sample: accuracy **pooled** (Σ corretti / Σ campioni).
> `--constraint` senza `gpu_RTX6000_24G` — su Turing questa run impiegava
> 6x più tempo, vedi `scripts/sbatch/baseline-qwen3vl2b.sbatch`.

---

## Risultati per shard — job 92353

| shard | run | GPU / host | wall-clock | `model_latency_mean` | `mcq_accuracy` | `mcq_accuracy_duration_short` | `mcq_accuracy_duration_medium` | `mcq_accuracy_duration_long` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | [`y68cgie2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/y68cgie2) | L40S / franco | 47 min | 3.11 s | 54.56% (491/900) | 68.33% (205/300) | 52.00% (156/300) | 43.33% (130/300) |
| 1 | [`toysc9fr`](https://wandb.ai/alesvale97-unimore/video_mme/runs/toysc9fr) | L40S / bimbogigi | 52 min | 3.43 s | 53.33% (480/900) | 63.33% (190/300) | 52.00% (156/300) | 44.67% (134/300) |
| 2 | [`4h6mwvby`](https://wandb.ai/alesvale97-unimore/video_mme/runs/4h6mwvby) | L40S / bimbogigi | 52 min | 3.43 s | 55.44% (499/900) | 69.00% (207/300) | 53.00% (159/300) | 44.33% (133/300) |
| **Pooled** | — | — | — | — | **54.44%** (1470/2700) | **66.89%** (602/900) | **52.33%** (471/900) | **44.11%** (397/900) |

---

## Confronto con Qwen2.5-VL-3B `baseline_24`

Stesso `nframes=24`, stessa strategy `uniform`, stesso preset `_attn`,
stesso benchmark — righe da
[`260803-tabelle_risultati.md §1`](260803-tabelle_risultati.md).

| modello | pooled | short | medium | long |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B `baseline_24` | 55.04% (1486/2700) | 64.33% (579/900) | 52.00% (468/900) | 48.78% (439/900) |
| Qwen3-VL-2B `baseline_24` | 54.44% (1470/2700) | 66.89% (602/900) | 52.33% (471/900) | 44.11% (397/900) |
| Δ (Qwen3 − Qwen2.5) | −0.59 | **+2.56** | +0.33 | **−4.67** |

---

## File rilevanti

- [`scripts/sbatch/baseline-qwen3vl2b.sbatch`](../scripts/sbatch/baseline-qwen3vl2b.sbatch)
- [`260803-tabelle_risultati.md`](260803-tabelle_risultati.md)
- [`qwen2_5_vl_3b_baseline.md`](qwen2_5_vl_3b_baseline.md)
