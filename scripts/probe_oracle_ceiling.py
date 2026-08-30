"""Il SOFFITTO del puntatore: indicare la cella VERA cambia l'accuracy? — LVBench.

Misura (2) del piano-oracolo (la (1) è `scripts/probe_peak_in_window.py`):
su Video-MME tutti i canali di consegna del picco d'attenzione sono nulli con
attention ≡ random, ma l'accuracy da sola non dice se il segnale era sbagliato
o se non c'era niente da guadagnare. Qui si separa: LVBench annota la finestra
dell'evidenza, quindi si può puntare il modello alla cella GIUSTA per
costruzione e misurare quanto vale un puntatore perfetto.

Tre condizioni per sample, stessi frame (24, uniformi, video intero), stesso
prompt MCQ, cambia solo il prefisso:

    baseline  nessun puntatore
    peak      "Focus on the part of the video between a and b seconds..."
              sulla cella di picco dell'attenzione (rowset via ROWSET,
              default `question` — l'arm A com'è)
    oracle    stesso identico testo, ma sull'intervallo della finestra
              ANNOTATA, quantizzato ai confini di cella (la risoluzione vera
              del canale: un puntatore perfetto non può dire di più)

Il testo del puntatore è ESATTAMENTE quello dell'arm `attention_highlight`
(POINTER_SPAN_SEC, prepend + "\\n\\n"): la differenza fra le condizioni è solo
DOVE punta, mai come parla. Parsing della risposta come negli arm:
`parse_mcq_letter`, fallback all'argmax del prefill (contato).

Lettura (accuracy appaiate, stessi sample):

    oracle ≈ baseline  → il soffitto non esiste: la metà "COME" di LoT è
        morta per principio su questo benchmark — risultato di tesi;
    oracle >> baseline → il soffitto esiste; peak dice quanto del divario
        l'attenzione recupera oggi (atteso ≈ baseline, coerente coi null).

Costo per sample: 1 forward con cattura (per il picco + fallback) + 3
generate brevi. Il campione è selezionato con la STESSA logica e lo stesso
seed di probe_peak_in_window (shuffle deterministico): a parità di seed i due
probe condividono i sample e i risultati si incrociano per id.

Log: wandb (project lvbench, group probe-oracle-ceiling), Table per-sample +
summary. Verifica da remoto via `/wandb` — mai SSH.

Uso:
    uv run python scripts/probe_oracle_ceiling.py [n_samples=100] [seed=0]
    ROWSET=entity NFRAMES=24 uv run python scripts/probe_oracle_ceiling.py
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GenerationConfig

from evals.base import format_mcq_prompt
from evals.lvbench import LVBench
from models.media import Text, VideoFrames
from models.qwen_attn import Qwen3VL2BAttention
from strategies.attention_highlight import POINTER_SPAN_SEC
from strategies.attention_marker import _extract_all_frames
from strategies.base import video_duration_sec
from utils.attn_core import (question_rows, ranked_cells_from_attention,
                             reconstruct_frame_indices, resolve_query_rows)
from utils.config import load_config
from utils.mcq import parse_mcq_letter

CONDITIONS = ("baseline", "peak", "oracle")


def cell_span_sec(c0: int, c1: int, t: int, duration: float) -> tuple[float, float]:
    """Estremi in secondi delle celle [c0, c1] inclusi: `[c0/t, (c1+1)/t]*durata`."""
    return c0 / t * duration, (c1 + 1) / t * duration


def window_cells(t: int, duration: float, w0: float, w1: float) -> list[int]:
    """Celle che intersecano [w0, w1] — stessa mappa di probe_peak_in_window."""
    return [c for c in range(t)
            if c / t * duration < w1 and (c + 1) / t * duration > w0]


def pointer_for(a: float, b: float) -> str:
    return POINTER_SPAN_SEC.format(a=a, b=b)


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nframes = int(os.environ.get("NFRAMES", 24))
    max_pixels = int(os.environ.get("MAX_PIXELS", 151200))
    rowset = os.environ.get("ROWSET", "question")
    if rowset not in ("all", "question", "entity"):
        print(f"ROWSET sconosciuto: {rowset!r} (validi: all, question, entity)", file=sys.stderr)
        return 1

    cfg = load_config(["dataset=lvbench", "model=qwen3_vl_2b_attn"])
    rows = LVBench().loader(cfg.dataset)
    usable = [r for r in rows
              if r["video_start"] is not None and r["video_end"] is not None
              and r["video_end"] > r["video_start"]
              and Path(r["video_path"]).exists()]
    print(f"LVBench: {len(rows)} sample, {len(usable)} con finestra valida e video presente")
    rng = random.Random(seed)
    rng.shuffle(usable)
    picked = usable[:n_samples]
    if not picked:
        print("nessun sample utilizzabile", file=sys.stderr)
        return 1

    import wandb
    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity),
        name=f"probe_oracle_ceiling-{os.environ.get('SLURM_JOB_ID', 'local')}",
        group="probe-oracle-ceiling",
        tags=["probe", "lvbench", "oracle_ceiling", rowset],
        config={"n_samples": len(picked), "seed": seed, "nframes": nframes,
                "max_pixels": max_pixels, "model": "qwen3_vl_2b_attn",
                "rowset": rowset, "sink_filter": False},
    )

    # Modello costruito DAL PRESET, come main.py: il preset `_attn` porta
    # `attn_implementation.text_config=qwen_attn_capture`, che è ciò che
    # attiva la cattura — senza, `full_visual_attention` muore con
    # "'Qwen3VLTextAttention' object has no attribute '_last_attn'" (visto
    # nel job 93610: 60/60 sample skippati).
    model_cfg = dict(cfg.model)
    model_cfg.pop("name")
    model_cfg["torch_dtype"] = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                                "float32": torch.float32}[model_cfg["torch_dtype"]]
    model_cfg["max_pixels"] = max_pixels
    vlm = Qwen3VL2BAttention(**model_cfg)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=8)

    table = wandb.Table(columns=[
        "id", "question", "duration", "t", "win_start", "win_end",
        "peak_cell", "peak_in_window", "oracle_span",
        *(f"pred_{c}" for c in CONDITIONS),
        *(f"correct_{c}" for c in CONDITIONS),
        "fallbacks",
    ])
    correct = {c: 0 for c in CONDITIONS}
    flips = {"oracle_recovered": 0, "oracle_broken": 0,
             "peak_recovered": 0, "peak_broken": 0}
    peak_in_win = 0
    n_done = n_skip = n_fallback = 0
    lat_sum = 0.0

    for i, r in enumerate(picked, 1):
        t0 = time.monotonic()
        try:
            prompt = format_mcq_prompt(r["question"], r["options"])
            letters = [chr(ord("A") + k) for k in range(len(r["options"]))]
            # Decode UNA volta per sample (4 forward riusano gli stessi PNG):
            # con `Video` ogni forward ridecodificherebbe un video che su
            # LVBench dura ore. Stesso pattern di probe_frame_addressing:
            # VideoFrames + frames_indices/fps, cosi' il processor Qwen3
            # emette gli stessi timestamp del path `Video`.
            idx, fps = reconstruct_frame_indices(r["video_path"], nframes)
            paths, tmp_dir = _extract_all_frames(r["video_path"], idx)
            media = VideoFrames(paths, max_pixels=max_pixels,
                                frames_indices=[int(k) for k in idx], fps=fps)
            try:
                signal = vlm.generate_with_signals(media, Text(prompt), gen_cfg,
                                                   answer_letters=letters)
                va = signal.visual_attention
                duration = video_duration_sec(r["video_path"])
                w0 = max(0.0, min(r["video_start"], duration))
                w1 = max(0.0, min(r["video_end"], duration))
                if w1 <= w0:
                    raise ValueError(f"finestra vuota dopo il clip: [{w0}, {w1}]")

                if rowset == "all":
                    rset = []
                elif rowset == "question":
                    rset = question_rows(va.query_tokens)
                else:
                    rset = resolve_query_rows("entity", va.query_tokens,
                                              extract_entities=vlm.extract_entities).rows
                peak = ranked_cells_from_attention(va, rows=rset, sink_filter=False)[0].cell
                wcells = window_cells(va.t, duration, w0, w1)
                hit = peak in wcells
                peak_in_win += hit

                pa, pb = cell_span_sec(peak, peak, va.t, duration)
                oa, ob = cell_span_sec(wcells[0], wcells[-1], va.t, duration)
                prompts = {
                    "baseline": prompt,
                    "peak": f"{pointer_for(pa, pb)}\n\n{prompt}",
                    "oracle": f"{pointer_for(oa, ob)}\n\n{prompt}",
                }
                # `pred_letter` del prefill = fallback quando la generazione
                # non produce una lettera parseabile, come negli arm.
                fb_idx = (ord(signal.pred_letter) - ord("A")
                          if signal.pred_letter else None)
                preds, fbs = {}, []
                for c in CONDITIONS:
                    messages = vlm.build_messages(media, Text(prompts[c]))
                    raw = vlm.generate(messages, generation_config=gen_cfg)
                    pred = parse_mcq_letter(raw, r["options"])
                    if pred is None:
                        pred = fb_idx
                        fbs.append(c)
                        n_fallback += 1
                    preds[c] = pred
                    correct[c] += (pred == r["answer"])
            finally:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            ok = {c: preds[c] == r["answer"] for c in CONDITIONS}
            if ok["oracle"] and not ok["baseline"]:
                flips["oracle_recovered"] += 1
            if ok["baseline"] and not ok["oracle"]:
                flips["oracle_broken"] += 1
            if ok["peak"] and not ok["baseline"]:
                flips["peak_recovered"] += 1
            if ok["baseline"] and not ok["peak"]:
                flips["peak_broken"] += 1

            table.add_data(r["id"], r["question"], round(duration, 1), va.t,
                           round(w0, 1), round(w1, 1), peak, hit,
                           f"[{oa:.0f},{ob:.0f}]s",
                           *(preds[c] for c in CONDITIONS),
                           *(ok[c] for c in CONDITIONS),
                           ",".join(fbs))
            n_done += 1
            lat = time.monotonic() - t0
            lat_sum += lat
            marks = " ".join(f"{c}={'OK' if ok[c] else 'no'}" for c in CONDITIONS)
            print(f"[{i:3}/{len(picked)}] {r['id']}: {marks} peak_hit={hit} "
                  f"oracle=[{oa:.0f},{ob:.0f}]s ({lat:.1f}s)", flush=True)
            del va, signal
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            print(f"[{i:3}/{len(picked)}] {r['id']}: SKIP — {type(e).__name__}: {e}",
                  flush=True)
            torch.cuda.empty_cache()

    if n_done == 0:
        print("nessun sample completato", file=sys.stderr)
        run.finish(exit_code=1)
        return 1

    print(f"\n=== soffitto su {n_done} sample (skip {n_skip}, fallback {n_fallback}) — rowset={rowset} ===")
    for c in CONDITIONS:
        print(f"  {c:9s}: {correct[c]}/{n_done} = {correct[c]/n_done:.3f}")
    print(f"  oracle vs baseline: +{flips['oracle_recovered']} / -{flips['oracle_broken']}"
          f"  (delta {correct['oracle']-correct['baseline']:+d})")
    print(f"  peak   vs baseline: +{flips['peak_recovered']} / -{flips['peak_broken']}"
          f"  (delta {correct['peak']-correct['baseline']:+d})")
    print(f"  peak_in_window: {peak_in_win}/{n_done} = {peak_in_win/n_done:.3f}")

    run.summary.update({
        "n_samples": n_done, "n_skip": n_skip, "n_fallback": n_fallback,
        **{f"acc_{c}": correct[c] / n_done for c in CONDITIONS},
        **{f"n_correct_{c}": correct[c] for c in CONDITIONS},
        **flips,
        "peak_in_window_rate": peak_in_win / n_done,
        "model_latency_mean": lat_sum / n_done,
    })
    run.log({"samples": table})
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
