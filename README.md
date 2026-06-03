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

## In corso

Wiring di Qwen3-VL-2B/4B (stub in `models/qwen3_vl.py`) per popolare le
altre due colonne della tabella baseline.
