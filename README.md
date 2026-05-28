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

| Dataset             | `nframes` | Qwen2.5-VL-3B<br>(paper) | Qwen2.5-VL-3B<br>(ours)                                                              | Qwen3-VL-2B | Qwen3-VL-4B |
|---------------------|----------:|-------------------------:|--------------------------------------------------------------------------------------|-------------|-------------|
| EgoSchema           |        64 |                     64.8 | [62.60](https://wandb.ai/alesvale97-unimore/egoschema/runs/y6ysm2tz) (313/500)       | —           | —           |
| MVBench             |        32 |                     67.0 | [64.83](https://wandb.ai/alesvale97-unimore/mvbench/runs/srzjo9j9) (2430/3748\*)     | —           | —           |
| Video-MME (no subs) |       128 |                     61.5 | [59.93](https://wandb.ai/alesvale97-unimore/video_mme/runs/p7srmrkl) (1618/2700)     | —           | —           |
| LVBench             |       768 |                     43.3 | **invalido** — vedi nota                                                             | —           | —           |

\* MVBench: 52/3800 sample senza `pred` parseabile (4 OOM + 48 parser MCQ).

**LVBench**: 2/3 shard OK (44.49% e 43.99% su `huber`), shard 2 finito su
una GPU 24 GiB (`nullazzo`) → **1032 OOM su 516 sample, accuracy 0%**. Il
run va rilanciato vincolando lo sbatch a GPU **≥ 40 GiB di VRAM**
(`gpu_A40_45G|gpu_L40S_45G`, rimuovere i constraint 24G).

## In corso

Re-run LVBench con constraint VRAM ≥40 GiB. Wiring di Qwen3-VL-2B/4B
(stub in `models/qwen3_vl.py`) per popolare le altre due colonne della
tabella baseline.
