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

| Dataset    | Qwen2.5-VL-3B                                                                                                                                                                                         | Qwen3-VL-2B | Qwen3-VL-4B |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|-------------|
| EgoSchema  | [58.6](https://wandb.ai/alesvale97-unimore/egoschema/runs/0mvk0nfe?nw=nwuseralesvale97&peekPath=%2Falesvale97-unimore%2Fegoschema%2Fcalls%2F019e415d-b147-78b0-a95f-5babee44c0b7%3FhideTraceTree%3D1) | —           | —           |
| MVBench    | —                                                                                                                                                                                                     | —           | —           |
| Video-MME  | —                                                                                                                                                                                                     | —           | —           |
| LVBench    | —                                                                                                                                                                                                     | —           | —           |


## In corso

Prefetch HPC dei video per MVBench, Video-MME e LVBench (login node).
Evaluation dei tre nuovi dataset pending al completamento dei
prefetch.
