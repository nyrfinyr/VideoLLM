"""La cattura d'attenzione corretta è equivalente alla vecchia? E quanto costa a 512 frame? — LVBench.

Contesto: la run di segnali su LVBench (512 frame a 50176 px, ~11.5k token
visivi) rendeva pesanti due sprechi del path di cattura di
`models/qwen_attn.py`: un `.cpu()` sincrono per layer, e la cattura di TUTTI i
28 layer quando la media usa solo la metà centrale. Il fix accumula sul device
i soli layer del range e fa un trasferimento a fine forward. Su CPU (Qwen3-VL a
pesi casuali, 8 layer) i due path sono bit-identici; qui si verifica con i
pesi veri, su GPU, e si misura il costo al budget della run.

Tre parti, stesso modello in memoria (l'istanza di riferimento condivide
processor e pesi):

    1. EQUIVALENZA a `NFRAMES` (default 24) su `n_samples` sample LVBench:
       stessa vista, kernel vecchio contro nuovo. Per sample: max differenza
       relativa di attenzione e sink map, differenza d'entropia, ranking delle
       celle (top-1 e completo) per rowset {all, question} x sink_filter.
    2. COSTO a `BIG_NFRAMES` (default 512) a `BIG_MAX_PIXELS` (50176) su un
       video: latenza, picco di memoria GPU, memoria host che il vecchio path
       impilava (`[L, n_q, n_vis]`), più la stessa verifica d'equivalenza.
    3. BACKEND SDPA del forward a 512 frame, letto dal profiler. La maschera
       densa di `eager_mask` esclude flash: se nei layer testuali gira il
       backend `math`, ogni layer materializza `[16, S, S]` (~6 GB a S=14k) e
       la run va ripensata.

Il riferimento è `models/qwen_attn.py` al commit `REF_COMMIT` (default
22be7cc, l'ultimo prima del fix), letto con `git show`: il codice che ha
prodotto le run passate, non una sua trascrizione.

Il media è un `Video` (decodifica di qwen-vl-utils), NON `VideoFrames`: con una
lista di frame qwen-vl-utils 0.0.14 passa `image_factor` come
`image_patch_size` a `fetch_image` e arrotonda le dimensioni a multipli di 64
invece che di 32 (640x360 a 50176 px → 4x8 token per cella invece di 5x9). Per
misurare il costo al budget della run serve il path che lo rispetta. La
decodifica entra quindi nelle latenze, identica per i due kernel.

Log: wandb (project lvbench, group probe-capture-equiv), summary + Table
per-sample; stdout replica tutto. Exit code 2 se l'equivalenza fallisce.

Uso:
    uv run python scripts/probe_capture_equiv.py [n_samples=5] [seed=0]
    REF_COMMIT=22be7cc BIG_NFRAMES=512 uv run python scripts/probe_capture_equiv.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Lanciato come `python scripts/...` il repo root non è su sys.path.
sys.path.insert(0, str(REPO))

import torch
from transformers import AttentionInterface

import models.qwen_attn as new_attn
from evals.base import format_mcq_prompt
from evals.lvbench import LVBench
from models.media import Text, Video
from utils.attn_core import ranked_cells_from_attention, resolve_query_rows
from utils.config import load_config

ROWSETS = ("all", "question")
# I due path fanno le stesse operazioni sugli stessi pesi; cambia solo l'ordine
# con cui si mediano i layer (stack + mean contro somma progressiva), quindi
# ci si aspetta rumore di arrotondamento, non di più.
REL_TOL = 1e-4
ENTROPY_TOL = 1e-4


def load_reference_module(commit: str):
    """`models/qwen_attn.py` com'era a `commit`, importato come modulo a sé.

    Gli import relativi diventano assoluti perché il file vive fuori dal
    package. L'import registra il kernel vecchio sotto gli stessi nomi del
    nuovo: `capture` ri-registra quello giusto prima di ogni forward.
    """
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:models/qwen_attn.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    for name in ("media", "qwen", "signals"):
        src = src.replace(f"from .{name} import", f"from models.{name} import")
    path = Path(tempfile.mkdtemp(prefix="qwen_attn_ref_")) / "qwen_attn_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("qwen_attn_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def share(cls, vlm):
    """Istanza di `cls` che riusa processor e pesi di `vlm`: un solo modello in memoria."""
    other = cls.__new__(cls)
    other.processor, other.model = vlm.processor, vlm.model
    return other


def capture(mod, vlm, media, text, letters):
    """Un forward col kernel di `mod` → `(VisualAttention, secondi, picco GPU in GB)`."""
    AttentionInterface.register("qwen_attn_capture", mod.qwen_attn_capture)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    va = vlm.full_visual_attention(media, text, answer_letters=letters)
    if cuda:
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else math.nan
    return va, time.monotonic() - t0, peak


def compare(ref, new) -> dict:
    """Differenze fra le due catture della stessa vista."""
    rel_attn = ((ref.attn - new.attn).abs().max() / ref.attn.abs().max()).item()
    rel_sink = ((ref.sink_map - new.sink_map).abs().max() / ref.sink_map.abs().max()).item()
    n_rank = top1_same = full_same = 0
    for qr in ROWSETS:
        rows = resolve_query_rows(qr, ref.query_tokens).rows
        for sink_filter in (False, True):
            r_ref = [c.cell for c in ranked_cells_from_attention(ref, rows=rows, sink_filter=sink_filter)]
            r_new = [c.cell for c in ranked_cells_from_attention(new, rows=rows, sink_filter=sink_filter)]
            n_rank += 1
            top1_same += r_ref[0] == r_new[0]
            full_same += r_ref == r_new
    d_entropy = (abs(ref.answer_entropy - new.answer_entropy)
                 if ref.answer_entropy is not None and new.answer_entropy is not None else math.nan)
    ok = (rel_attn < REL_TOL and rel_sink < REL_TOL and top1_same == n_rank
          and d_entropy < ENTROPY_TOL)
    return {"rel_attn": rel_attn, "rel_sink": rel_sink, "d_entropy": d_entropy,
            "top1_same": top1_same, "full_rank_same": full_same, "n_rankings": n_rank, "ok": ok}


def sdpa_kernels(fn) -> dict[str, int]:
    """Kernel SDPA eseguiti da `fn()` (nome aten → numero di chiamate).

    Conta anche l'encoder visivo, che gira a `sdpa` senza maschera: le 28
    chiamate con maschera del decoder testuale si riconoscono dal conteggio.
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        fn()
    return {e.key: e.count for e in prof.key_averages()
            if e.key.startswith("aten::") and ("scaled_dot_product" in e.key
                                               or "efficient_attention" in e.key
                                               or "flash_attention" in e.key)}


def fmt(c: dict) -> str:
    return (f"rel_attn={c['rel_attn']:.1e} rel_sink={c['rel_sink']:.1e} "
            f"d_entropy={c['d_entropy']:.1e} top1={c['top1_same']}/{c['n_rankings']} "
            f"rank={c['full_rank_same']}/{c['n_rankings']} {'OK' if c['ok'] else 'DIVERGE'}")


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ref_commit = os.environ.get("REF_COMMIT", "22be7cc")
    nframes = int(os.environ.get("NFRAMES", 24))
    max_pixels = int(os.environ.get("MAX_PIXELS", 151200))
    big_nframes = int(os.environ.get("BIG_NFRAMES", 512))
    big_max_pixels = int(os.environ.get("BIG_MAX_PIXELS", 50176))
    # Senza un floor esplicito qwen-vl-utils impone 128 token/frame (131072 px)
    # e il tetto a 50176 non verrebbe rispettato.
    big_min_pixels = int(os.environ.get("BIG_MIN_PIXELS", 3136))

    ref_attn = load_reference_module(ref_commit)
    cfg = load_config(["dataset=lvbench", "model=qwen3_vl_2b_attn"])
    rows = [r for r in LVBench().loader(cfg.dataset) if Path(r["video_path"]).exists()]
    random.Random(seed).shuffle(rows)
    picked = rows[:n_samples]
    if not picked:
        print("nessun sample utilizzabile — root sbagliata o video mancanti?", file=sys.stderr)
        return 1

    import wandb
    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity),
        name=f"probe_capture_equiv-{os.environ.get('SLURM_JOB_ID', 'local')}",
        group="probe-capture-equiv",
        tags=["probe", "lvbench", "capture_equiv"],
        config={"n_samples": len(picked), "seed": seed, "ref_commit": ref_commit,
                "nframes": nframes, "max_pixels": max_pixels, "big_nframes": big_nframes,
                "big_max_pixels": big_max_pixels, "big_min_pixels": big_min_pixels,
                "model": "qwen3_vl_2b_attn", "rel_tol": REL_TOL},
    )

    # Modello costruito DAL PRESET, come main.py: è il preset `_attn` che attiva
    # `attn_implementation.text_config=qwen_attn_capture`.
    model_cfg = dict(cfg.model)
    model_cfg.pop("name")
    model_cfg["torch_dtype"] = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                                "float32": torch.float32}[model_cfg["torch_dtype"]]
    vlm_new = new_attn.Qwen3VL2BAttention(**model_cfg)
    vlm_ref = share(ref_attn.Qwen3VL2BAttention, vlm_new)
    n_layers = len(vlm_new.model.model.language_model.layers)

    table = wandb.Table(columns=["part", "id", "nframes", "t", "grid", "seq_len", "rel_attn",
                                 "rel_sink", "d_entropy", "top1_same", "full_rank_same", "ok",
                                 "lat_ref", "lat_new", "peak_gpu_gb_ref", "peak_gpu_gb_new"])
    summary: dict = {}
    all_ok = True

    # --- 1. equivalenza a NFRAMES ---------------------------------------------
    worst = {"rel_attn": 0.0, "rel_sink": 0.0, "d_entropy": 0.0}
    n_done = n_skip = 0
    lat = {"ref": 0.0, "new": 0.0}
    for i, r in enumerate(picked, 1):
        try:
            text = Text(format_mcq_prompt(r["question"], r["options"]))
            letters = [chr(ord("A") + k) for k in range(len(r["options"]))]
            media = Video(r["video_path"], nframes=nframes, max_pixels=max_pixels)
            a_ref, lat_ref, peak_ref = capture(ref_attn, vlm_ref, media, text, letters)
            a_new, lat_new, peak_new = capture(new_attn, vlm_new, media, text, letters)
            c = compare(a_ref, a_new)
            all_ok &= c["ok"]
            for k in worst:
                worst[k] = max(worst[k], c[k])
            lat["ref"] += lat_ref
            lat["new"] += lat_new
            n_done += 1
            table.add_data("equiv", r["id"], nframes, a_new.t, f"{a_new.grid_h}x{a_new.grid_w}",
                           len(a_new.input_ids), c["rel_attn"], c["rel_sink"], c["d_entropy"],
                           c["top1_same"], c["full_rank_same"], c["ok"],
                           lat_ref, lat_new, peak_ref, peak_new)
            print(f"[equiv {i}/{len(picked)}] {r['id']}: {fmt(c)} "
                  f"(ref {lat_ref:.1f}s, new {lat_new:.1f}s)", flush=True)
            del a_ref, a_new
        except Exception as e:  # noqa: BLE001 — un sample rotto non uccide la sonda
            n_skip += 1
            print(f"[equiv {i}/{len(picked)}] {r['id']}: SKIP — {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    if n_done == 0:
        all_ok = False
    summary.update({"equiv_n_samples": n_done, "equiv_n_skip": n_skip,
                    **{f"equiv_worst_{k}": v for k, v in worst.items()},
                    "equiv_lat_ref_mean": lat["ref"] / max(n_done, 1),
                    "equiv_lat_new_mean": lat["new"] / max(n_done, 1)})

    # --- 2. costo a BIG_NFRAMES -----------------------------------------------
    r = picked[0]
    text = Text(format_mcq_prompt(r["question"], r["options"]))
    letters = [chr(ord("A") + k) for k in range(len(r["options"]))]
    media = Video(r["video_path"], nframes=big_nframes, max_pixels=big_max_pixels,
                  min_pixels=big_min_pixels)
    a_new, lat_new, peak_new = capture(new_attn, vlm_new, media, text, letters)
    torch.cuda.empty_cache()
    a_ref, lat_ref, peak_ref = capture(ref_attn, vlm_ref, media, text, letters)
    torch.cuda.empty_cache()
    c = compare(a_ref, a_new)
    all_ok &= c["ok"]
    n_q = a_new.attn.shape[0]
    n_vis = a_new.t * a_new.grid_h * a_new.grid_w
    seq_len = len(a_new.input_ids)
    host_stack_gb = n_layers * n_q * n_vis * 4 / 1e9
    table.add_data("big", r["id"], big_nframes, a_new.t, f"{a_new.grid_h}x{a_new.grid_w}",
                   seq_len, c["rel_attn"], c["rel_sink"], c["d_entropy"], c["top1_same"],
                   c["full_rank_same"], c["ok"], lat_ref, lat_new, peak_ref, peak_new)
    print(f"[big {big_nframes}f] {r['id']}: t={a_new.t} grid={a_new.grid_h}x{a_new.grid_w} "
          f"n_vis={n_vis} n_q={n_q} seq_len={seq_len} | {fmt(c)}", flush=True)
    print(f"[big {big_nframes}f] latenza ref {lat_ref:.1f}s new {lat_new:.1f}s | "
          f"picco GPU ref {peak_ref:.2f} GB new {peak_new:.2f} GB | "
          f"host impilato dal ref {host_stack_gb:.2f} GB", flush=True)
    summary.update({"big_nframes": big_nframes, "big_t": a_new.t,
                    "big_grid": f"{a_new.grid_h}x{a_new.grid_w}", "big_n_vis": n_vis,
                    "big_n_q": n_q, "big_seq_len": seq_len,
                    **{f"big_{k}": v for k, v in c.items()},
                    "big_lat_ref": lat_ref, "big_lat_new": lat_new,
                    "big_peak_gpu_gb_ref": peak_ref, "big_peak_gpu_gb_new": peak_new,
                    "big_host_stack_gb_ref": host_stack_gb})
    del a_ref, a_new

    # --- 3. backend SDPA a BIG_NFRAMES -----------------------------------------
    kernels = sdpa_kernels(lambda: capture(new_attn, vlm_new, media, text, letters))
    math_calls = sum(v for k, v in kernels.items() if "math" in k)
    print(f"[sdpa {big_nframes}f] kernel: {kernels}", flush=True)
    print(f"[sdpa {big_nframes}f] chiamate math: {math_calls} "
          f"({'⚠️ backend math nel forward' if math_calls else 'nessuna'})", flush=True)
    summary.update({"sdpa_math_calls": math_calls,
                    **{f"sdpa_{k.removeprefix('aten::')}": v for k, v in kernels.items()}})

    summary["equiv_ok"] = all_ok
    print(f"\n=== equivalenza {'OK' if all_ok else 'FALLITA'} ===")
    run.summary.update(summary)
    run.log({"samples": table})
    run.finish(exit_code=0 if all_ok else 2)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
