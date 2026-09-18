"""Il SOFFITTO del pass ADDITIVO: infittire la finestra vera SENZA togliere contesto.

Il pass 2 sostitutivo è stato falsificato (`arm topk_resample`, probe 108729,
100 sample): ricampionare nelle top-k celle costa 10-13 pp contro la baseline
appaiata e non batte il controllo random, perché rimpiazzare 512 frame con 128
butta più contesto di quanto la zoomata aggiunga — il costo di un miss è −11,9
pp contro un +23 quando centra. La direzione additiva toglie esattamente quel
termine: i frame mirati si AGGIUNGONO alla base invece di sostituirla.

Prima di costruire l'arm serve sapere se la densità dentro la finestra vale
ancora qualcosa a questa base. Questa probe lo misura con un oracolo: la
finestra la diamo noi, annotata da LVBench. Se l'oracolo non stacca il
riferimento, nessun puntatore lo farà e l'arm non va scritto.

TRE CONDIZIONI, stessi sample, stesso prompt, tutte e tre nello stesso giro
(confronti APPAIATI intra-run, niente rumore fra GPU o fra job):

    base256     128 celle a coppie (gap 2 s) sul video intero — la base
                dell'arm, e un numero che non abbiamo mai misurato
    oracle512   base256 PIÙ 256 frame uniformi dentro le celle che
                intersecano la finestra annotata — il soffitto
    uniform512  256 celle a coppie sul video intero — stesso budget totale di
                `oracle512` speso tutto uniformemente (è il 45% della run
                108729, rimisurato qui per averlo appaiato)

La lettura, in quest'ordine:

    oracle512 ≈ uniform512  → a parità di frame, mirare non paga: la densità
        in-finestra non è la risorsa scarsa a questa base, e l'arm additivo
        muore qui (come è morto quello sostitutivo, ma per un'altra ragione);
    oracle512 >> uniform512 → il guadagno esiste ed è tutto nel DOVE: quanto
        ne raccoglie il puntatore vero si misura dopo, con k5/k10;
    base256 ≈ uniform512    → i frame uniformi oltre i 256 non servono, e il
        budget dell'arm può scendere.

PERCHÉ L'ORACOLO È QUANTIZZATO ALLE CELLE. I 256 frame non vanno dentro
`[video_start, video_end]` ma dentro le CELLE DI VORONOI che la intersecano:
un puntatore perfetto può indicare una cella, non un istante, quindi un
oracolo più fine misurerebbe un soffitto irraggiungibile per costruzione.
Come diagnostica si logga comunque quanti frame cadono nella finestra grezza.

⚠️ La finestra annotata entra SOLO nella scelta dei frame, mai nel prompt
(`dataset.use_time_reference=false`): è il punto in cui una probe oracolo si
auto-inganna.

Costo per sample: 3 forward (256 + 512 + 512 frame) ≈ 145 s, un solo decode
del video (`frames_for_plans`: i PNG sono condivisi fra le tre condizioni).

Log: wandb (project lvbench, group probe-additive-oracle), Table per-sample +
summary. Verifica da remoto via `/wandb` — mai SSH.

Uso:
    uv run python scripts/probe_additive_oracle.py            # 100 sample
    LIMIT=2 uv run python scripts/probe_additive_oracle.py    # prova di fumo
    BASE=256 ADDED=256 UNIFORM=512 uv run python scripts/probe_additive_oracle.py
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GenerationConfig

from evals.base import format_mcq_prompt
from evals.lvbench import LVBench
from models.media import Text
from models.qwen_attn import Qwen3VL2BAttention
from utils.config import load_config
from utils.mcq import parse_mcq_letter
from utils.pair_sampling import (additive_indices, cells_to_spans,
                                 frames_for_plans, pair_centers_and_indices)
from utils.samples import prepare_samples

CONDITIONS = ("base256", "oracle512", "uniform512")
# Finestre più larghe di questa frazione del video: l'oracolo è quasi gratis
# (i frame "mirati" cadono dentro anche a caso). Non si escludono — si
# riportano a parte, altrimenti il soffitto che leggiamo non è quello che
# l'arm potrà mai raggiungere.
WIDE_WINDOW_FRAC = 0.10


def window_cells(n_cells: int, duration: float, w0: float, w1: float) -> list[int]:
    """Celle la cui cella di Voronoi `[D·i/n, D·(i+1)/n]` interseca `[w0, w1]`.

    È la risoluzione vera del canale: un puntatore perfetto indica una cella,
    non un istante. Almeno una cella esce sempre (la finestra è dentro il
    video), anche quando è più stretta di una cella.
    """
    if duration <= 0:
        raise ValueError(f"durata non valida: {duration}")
    w = duration / n_cells
    lo = max(0, min(int(w0 / w), n_cells - 1))
    hi = max(0, min(int(w1 / w), n_cells - 1))
    return list(range(lo, hi + 1))


def plan_sample(
    total_frames: int, fps: float, duration: float, w0: float, w1: float,
    *, base_pairs: int, uniform_pairs: int, n_added: int, gap_sec: float,
) -> tuple[dict[str, list[int]], dict]:
    """I tre piani di frame del sample. Pura: testabile senza video né GPU."""
    _, base = pair_centers_and_indices(total_frames, fps, base_pairs, gap_sec)
    _, uniform = pair_centers_and_indices(total_frames, fps, uniform_pairs, gap_sec)
    cells = window_cells(base_pairs, duration, w0, w1)
    spans = cells_to_spans(cells, base_pairs, duration)
    oracle, info = additive_indices(total_frames, fps, base, spans, n_added)
    # Diagnostica: quanti degli aggiunti cadono nella finestra GREZZA, cioè
    # quanto costa la quantizzazione alle celle.
    in_raw = sum(1 for i in info["added_indices"] if w0 <= i / fps <= w1)
    info.update({"n_cells_window": len(cells), "cells": cells,
                 "n_added_in_raw_window": in_raw})
    return {"base256": base, "uniform512": uniform, "oracle512": oracle}, info


def dump_timestamps(vlm, messages, indices: list[int], fps: float,
                    w0: float, w1: float) -> dict:
    """I `<x.x seconds>` che il modello vede DAVVERO per una lista MISTA
    (base rada + grappolo denso), confrontati con quelli attesi.

    È l'unica parte del pass additivo che non si può verificare senza il
    processor vero: la lista non è un campionamento uniforme, e i timestamp
    sono il solo canale con cui Qwen3-VL sa QUANDO è successo qualcosa. Se il
    grappolo denso arrivasse con tempi sbagliati, l'oracolo misurerebbe
    un'altra cosa e non ce ne accorgeremmo dai numeri.

    ⚠️ NON si passa da `apply_chat_template`: su Qwen3-VL i timestamp li
    scrive il **processor** espandendo il placeholder visivo (errore commesso
    nel job 93544). L'unica stringa fedele è la decodifica degli `input_ids`.

    Ritorna anche un dict di check che finisce nel summary wandb: il `.out`
    SLURM non è leggibile da fuori dal cluster, e questa verifica è il motivo
    per cui si fa una prova da 2 sample.

    Best-effort: un fallimento qui non deve uccidere la probe.
    """
    try:
        inputs = vlm._prepare_inputs(messages)
        text = vlm.processor.batch_decode(inputs.input_ids, skip_special_tokens=False)[0]
    except Exception as exc:  # noqa: BLE001 — diagnostica, non correttezza
        print(f"  [verifica] dump timestamp non riuscito: {exc!r}", flush=True)
        return {"check_error": f"{type(exc).__name__}: {exc}"}
    # Griglia: con `fix_videoframes_resize` attivo sono ~45 token per cella
    # (5x9 sul 16:9). 4x8 = 32 token è la firma del knob NON attivo, e
    # renderebbe i numeri non confrontabili con la run 108729.
    check: dict = {}
    grid = getattr(inputs, "video_grid_thw", None)
    if grid is not None and len(grid):
        t, gh, gw = (int(x) for x in grid[0])
        check.update({"check_grid_t": t, "check_grid_h": gh, "check_grid_w": gw,
                      "check_tokens_per_cell": gh * gw // 4})
        print(f"  [verifica] griglia video: t={t} {gh}x{gw} = {gh*gw//4} token per cella "
              f"(atteso ~45; 32 = fix_videoframes_resize NON attivo)", flush=True)
    got = [float(x) for x in re.findall(r"<([\d.]+) seconds?>", text)]
    # Il processor fonde i frame a due a due NELL'ORDINE DELLA LISTA: la cella
    # c è la coppia (2c, 2c+1) e il suo timestamp è la media dei due.
    want = [(indices[2 * c] + indices[2 * c + 1]) / (2 * fps)
            for c in range(len(indices) // 2)]
    check.update({"check_n_timestamps": len(got), "check_n_cells_expected": len(want)})
    print(f"  [verifica] timestamp scritti: {len(got)} (celle attese: {len(want)})", flush=True)
    if not got:
        compact = re.sub(r"(<\|image_pad\|>){2,}", "<|image_pad|>xN", text)
        print(f"  [verifica] NESSUN timestamp — prompt: {compact[:300]!r}", flush=True)
        return check
    n = min(len(got), len(want))
    worst = max(range(n), key=lambda i: abs(got[i] - want[i])) if n else None
    mono = all(b >= a for a, b in zip(got, got[1:]))
    in_win = sum(1 for t in got if w0 <= t <= w1)
    print(f"  [verifica] monotoni: {mono} | scarto max dall'atteso: "
          f"{abs(got[worst] - want[worst]):.2f} s (cella {worst}) | "
          f"celle dentro la finestra: {in_win}", flush=True)
    print(f"  [verifica] primi 3 {[round(x, 1) for x in got[:3]]} … "
          f"dentro la finestra {[round(x, 1) for x in got if w0 <= x <= w1][:6]} … "
          f"ultimi 3 {[round(x, 1) for x in got[-3:]]}", flush=True)
    check.update({"check_timestamps_monotonic": mono,
                  "check_timestamp_max_dev_sec": abs(got[worst] - want[worst]),
                  "check_cells_in_window": in_win,
                  "check_first_timestamps": [round(x, 1) for x in got[:3]],
                  "check_last_timestamps": [round(x, 1) for x in got[-3:]]})
    return check


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """`a` → `b`: (recuperati, rotti, p esatto binomiale a due code)."""
    win = sum(1 for x, y in zip(a, b) if not x and y)
    loss = sum(1 for x, y in zip(a, b) if x and not y)
    m = win + loss
    if m == 0:
        return win, loss, 1.0
    k = min(win, loss)
    return win, loss, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main() -> int:  # noqa: C901
    base_pairs = int(os.environ.get("BASE", 256)) // 2
    uniform_pairs = int(os.environ.get("UNIFORM", 512)) // 2
    n_added = int(os.environ.get("ADDED", 256))
    gap_sec = float(os.environ.get("GAP", 2.0))
    limit = os.environ.get("LIMIT", "100")
    # Con DUMP_TS=1 il primo sample stampa i timestamp che il processor
    # scrive per la condizione additiva: la verifica che una prova da 2
    # sample deve fare, e che i numeri non possono fare.
    dump_ts = os.environ.get("DUMP_TS", "0") not in ("0", "", "false")
    shard = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    num_shards = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")

    overrides = ["dataset=lvbench", "model=qwen3_vl_2b_attn",
                 "model.fix_videoframes_resize=true", "shuffle=true",
                 f"shard={shard}", f"num_shards={num_shards}"]
    overrides.append(f"limit={limit}")
    cfg = load_config(overrides)

    # STESSA selezione della eval (shuffle seed → shard strided → limit): con
    # --array=0-1 e LIMIT=50 questi sono ESATTAMENTE i 100 sample della probe
    # topk 108729, quindi i risultati si incrociano per id.
    rows = prepare_samples(LVBench().loader(cfg.dataset), cfg)
    picked = [r for r in rows
              if r.get("video_start") is not None and r.get("video_end") is not None
              and Path(r["video_path"]).exists()]
    print(f"LVBench: {len(rows)} sample dopo shuffle/shard/limit, "
          f"{len(picked)} con finestra annotata e video presente", flush=True)
    if not picked:
        print("nessun sample utilizzabile", file=sys.stderr)
        return 1

    import wandb

    run = wandb.init(
        project=str(cfg.wandb.project), entity=str(cfg.wandb.entity),
        name=f"probe_additive_oracle-{os.environ.get('SLURM_JOB_ID', 'local')}",
        group="probe-additive-oracle",
        tags=["probe", "lvbench", "additive", "oracle"],
        config={"n_samples": len(picked), "base_frames": 2 * base_pairs,
                "uniform_frames": 2 * uniform_pairs, "n_added": n_added,
                "gap_sec": gap_sec, "model": "qwen3_vl_2b_attn",
                "max_pixels": cfg.dataset.max_pixels, "min_pixels": cfg.dataset.min_pixels,
                "shard": shard, "num_shards": num_shards, "seed": cfg.seed,
                "conditions": list(CONDITIONS), "wide_window_frac": WIDE_WINDOW_FRAC},
    )

    model_cfg = dict(cfg.model)
    model_cfg.pop("name")
    model_cfg["torch_dtype"] = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                                "float32": torch.float32}[model_cfg["torch_dtype"]]
    vlm = Qwen3VL2BAttention(**model_cfg)
    gen_cfg = GenerationConfig(**dict(cfg.generation))

    table = wandb.Table(columns=[
        "id", "duration", "win_start", "win_end", "win_frac", "wide",
        "n_cells_window", "n_added_kept", "n_collision", "n_added_in_raw_window",
        *(f"pred_{c}" for c in CONDITIONS), *(f"correct_{c}" for c in CONDITIONS),
        "fallbacks", "latency_s",
    ])
    checks: dict = {}          # verifiche del primo sample, con DUMP_TS=1
    ok_by_cond: dict[str, list[bool]] = {c: [] for c in CONDITIONS}
    wide_flags: list[bool] = []
    n_done = n_skip = 0
    n_fallback = {c: 0 for c in CONDITIONS}
    lat = {c: 0.0 for c in CONDITIONS}
    diag = {"n_added_kept": 0, "n_collision": 0, "n_added_in_raw_window": 0,
            "n_cells_window": 0}

    for i, r in enumerate(picked, 1):
        t_sample = time.monotonic()
        tmp_dir = None
        try:
            import decord

            vr = decord.VideoReader(r["video_path"])
            total_frames, fps = len(vr), float(vr.get_avg_fps())
            del vr
            # Durata dai metadati che servono comunque al campionamento, non
            # da una seconda apertura del video: `pair_centers_and_indices`
            # usa la stessa definizione (total_frames / fps), e due
            # definizioni diverse sfaserebbero le celle di Voronoi.
            duration = total_frames / fps
            w0, w1 = float(r["video_start"]), float(r["video_end"])
            if w1 < w0:
                w0, w1 = w1, w0
            w0, w1 = max(0.0, min(w0, duration)), max(0.0, min(w1, duration))
            if w1 <= w0:
                raise ValueError(f"finestra vuota dopo il clip: [{w0}, {w1}]")

            plans, info = plan_sample(
                total_frames, fps, duration, w0, w1, base_pairs=base_pairs,
                uniform_pairs=uniform_pairs, n_added=n_added, gap_sec=gap_sec)
            media, tmp_dir, _ = frames_for_plans(
                r["video_path"], plans,
                max_pixels=cfg.dataset.max_pixels, min_pixels=cfg.dataset.min_pixels,
                image_patch_size=getattr(vlm, "image_patch_size", None))

            prompt = format_mcq_prompt(r["question"], r["options"])
            preds, fbs = {}, []
            for c in CONDITIONS:
                messages = vlm.build_messages(media[c], Text(prompt))
                if dump_ts and n_done == 0 and c == "oracle512":
                    checks = dump_timestamps(vlm, messages, plans[c], fps, w0, w1)
                t0 = time.monotonic()
                raw = vlm.generate(messages, generation_config=gen_cfg)
                lat[c] += time.monotonic() - t0
                pred = parse_mcq_letter(raw, r["options"])
                if pred is None:
                    # Nessuna lettera parseabile: NON si ripiega sulla
                    # risposta di un'altra condizione (falserebbe il
                    # confronto appaiato). Conta come errore ed è loggato.
                    n_fallback[c] += 1
                    fbs.append(c)
                preds[c] = pred
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            print(f"[{i:3}/{len(picked)}] {r['id']}: SKIP — {type(e).__name__}: {e}",
                  flush=True)
            torch.cuda.empty_cache()
            continue
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        ok = {c: preds[c] == r["answer"] for c in CONDITIONS}
        for c in CONDITIONS:
            ok_by_cond[c].append(ok[c])
        win_frac = (w1 - w0) / duration
        wide = win_frac > WIDE_WINDOW_FRAC
        wide_flags.append(wide)
        for k in diag:
            diag[k] += info[k]
        n_done += 1
        dt = time.monotonic() - t_sample
        table.add_data(
            r["id"], round(duration, 1), round(w0, 1), round(w1, 1), round(win_frac, 4),
            wide, info["n_cells_window"], info["n_added_kept"], info["n_collision"],
            info["n_added_in_raw_window"],
            *(preds[c] for c in CONDITIONS), *(ok[c] for c in CONDITIONS),
            ",".join(fbs), round(dt, 1))
        marks = " ".join(f"{c}={'OK' if ok[c] else 'no'}" for c in CONDITIONS)
        print(f"[{i:3}/{len(picked)}] {r['id']}: {marks} | finestra {w1-w0:.0f}s "
              f"({100*win_frac:.1f}%) su {info['n_cells_window']} celle, "
              f"+{info['n_added_kept']} frame ({info['n_added_in_raw_window']} nella "
              f"finestra grezza) | {dt:.1f}s", flush=True)
        torch.cuda.empty_cache()

    if n_done == 0:
        print("nessun sample completato", file=sys.stderr)
        run.finish(exit_code=1)
        return 1

    acc = {c: sum(ok_by_cond[c]) / n_done for c in CONDITIONS}
    print(f"\n=== soffitto additivo su {n_done} sample (skip {n_skip}) ===")
    for c in CONDITIONS:
        print(f"  {c:11s}: {sum(ok_by_cond[c])}/{n_done} = {acc[c]:.3f}"
              f"   (fallback {n_fallback[c]}, {lat[c]/n_done:.1f} s/sample)")
    summary: dict = {
        "n_samples": n_done, "n_skip": n_skip,
        **{f"acc_{c}": acc[c] for c in CONDITIONS},
        **{f"n_correct_{c}": sum(ok_by_cond[c]) for c in CONDITIONS},
        **{f"n_fallback_{c}": n_fallback[c] for c in CONDITIONS},
        **{f"latency_{c}": lat[c] / n_done for c in CONDITIONS},
        **{f"mean_{k}": v / n_done for k, v in diag.items()},
        "model_latency_mean": sum(lat.values()) / n_done,
        "n_wide_window": sum(wide_flags),
        **checks,
    }
    print("\n  confronti appaiati (recuperati / rotti, p McNemar esatto):")
    for a, b in (("uniform512", "oracle512"), ("base256", "oracle512"),
                 ("base256", "uniform512")):
        win, loss, p = mcnemar(ok_by_cond[a], ok_by_cond[b])
        print(f"    {a:11s} → {b:11s}: {acc[b]-acc[a]:+.3f}  +{win} / -{loss}  p={p:.3g}")
        summary.update({f"delta_{a}_to_{b}": acc[b] - acc[a],
                        f"win_{a}_to_{b}": win, f"loss_{a}_to_{b}": loss,
                        f"p_{a}_to_{b}": p})
    print(f"\n  breakdown per larghezza della finestra (soglia {100*WIDE_WINDOW_FRAC:.0f}% "
          f"del video):")
    for label, want in (("stretta", False), ("larga", True)):
        idx = [j for j, w in enumerate(wide_flags) if w == want]
        if not idx:
            continue
        line = "    ".join(f"{c}={sum(ok_by_cond[c][j] for j in idx)}/{len(idx)}"
                           for c in CONDITIONS)
        print(f"    {label:8s} (n={len(idx):3d}): {line}")
        for c in CONDITIONS:
            summary[f"acc_{c}_{label}"] = sum(ok_by_cond[c][j] for j in idx) / len(idx)
        summary[f"n_{label}"] = len(idx)
    print(f"\n  diagnostica: {diag['n_added_kept']/n_done:.0f} frame aggiunti tenuti, "
          f"{diag['n_collision']/n_done:.0f} scartati per collisione, "
          f"{diag['n_added_in_raw_window']/n_done:.0f} dentro la finestra GREZZA, "
          f"{diag['n_cells_window']/n_done:.1f} celle per finestra")
    run.summary.update(summary)
    run.log({"samples": table})
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
