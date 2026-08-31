"""Il SOFFITTO del puntatore: indicare la cella VERA cambia l'accuracy? — LVBench.

Misura (2) del piano-oracolo (la (1) è `scripts/probe_peak_in_window.py`):
su Video-MME tutti i canali di consegna del picco d'attenzione sono nulli con
attention ≡ random, ma l'accuracy da sola non dice se il segnale era sbagliato
o se non c'era niente da guadagnare. Qui si separa: LVBench annota la finestra
dell'evidenza, quindi si può puntare il modello alla cella GIUSTA per
costruzione e misurare quanto vale un puntatore perfetto.

Quattro condizioni per sample, stesso prompt MCQ (le prime tre sugli stessi
24 frame uniformi del video intero, la quarta su frame diversi):

    baseline  nessun puntatore
    peak      "Focus on the part of the video between a and b seconds..."
              sulla cella di picco dell'attenzione (rowset via ROWSET,
              default `question` — l'arm A com'è)
    oracle    stesso identico testo, ma sull'intervallo della finestra
              ANNOTATA, quantizzato ai confini di cella (la risoluzione vera
              del canale: un puntatore perfetto non può dire di più)
    zoom      nessun puntatore: i 24 frame sono RICAMPIONATI dentro la
              finestra annotata — ORACOLO della metà "ricampionare"
    zoom_peak     24 frame nella CELLA DI PICCO: l'arm realistico, l'oracolo
                  sostituito dal segnale vero
    zoom_topk     24 frame divisi fra le TOP-K celle del ranking (disgiunte,
                  24/k ciascuna): compra copertura se il segnale ha massa
                  sulle celle giuste anche quando il top-1 sbaglia
    zoom_hybrid   metà frame globali (uniformi) + metà nella cella di picco:
                  non butta il contesto, degrada dolcemente se il picco
                  sbaglia — protezione contro il caso, misurato, in cui il
                  campione uniforme colpiva GIÀ l'evidenza e lo zoom peggiora

ESITO DEI GIRI PRECEDENTI (job 93614/94348, 100 sample):
baseline 32%, peak 29%, oracle 30% → il canale di MARCATURA non ha soffitto;
zoom 48% (+22/−6, p=0.004) → il guadagno vive nel RICAMPIONAMENTO. Ma il
segnale localizza la finestra solo nel 31% dei sample (18% su quelli dove
zoom recupera), e allargare la finestra attorno al picco NON aiuta: quando
il picco sbaglia, sbaglia di 4-5 celle (distribuzione quasi piatta), quindi
±k celle compra copertura di puro caso e perde risoluzione (2.4 → 0.8 frame
nell'evidenza). Da qui le tre condizioni realistiche qui sopra, che sono le
sole strade rimaste: il ranking (top-k disgiunto) e il degrado gentile
(hybrid).

Gratis dal ranking, senza forward aggiuntivi: **hit@k** = la finestra
annotata è coperta da una delle prime k celle? Dice se il top-k disgiunto
ha senso PRIMA di guardarne l'accuracy.

Il testo del puntatore è ESATTAMENTE quello dell'arm `attention_highlight`
(POINTER_SPAN_SEC, prepend + "\\n\\n"): la differenza fra le condizioni è solo
DOVE punta, mai come parla. Parsing della risposta come negli arm:
`parse_mcq_letter`, fallback all'argmax del prefill (contato).

Lettura (accuracy appaiate, stessi sample):

    oracle ≈ baseline  → la marcatura non ha soffitto (CONFERMATO dal primo
        giro 93614: 30% vs 32%);
    zoom ≈ baseline    → nemmeno il ricampionamento perfetto paga: la
        premessa di LoT muore su questo regime — risultato di tesi;
    zoom >> baseline   → il guadagno vive SOLO nel ricampionamento, e il
        segnale (debole: picco-in-finestra ~2x chance) va giudicato come
        selettore di DOVE ricampionare, non di cosa marcare.

Costo per sample: 1 forward con cattura (per il picco + fallback) + 3
generate brevi. Il campione è selezionato con la STESSA logica e lo stesso
seed di probe_peak_in_window (shuffle deterministico): a parità di seed i due
probe condividono i sample e i risultati si incrociano per id.

Log: wandb (project lvbench, group probe-oracle-ceiling), Table per-sample +
summary. Verifica da remoto via `/wandb` — mai SSH.

Uso:
    uv run python scripts/probe_oracle_ceiling.py [n_samples=100] [seed=0]
    ROWSET=entity TOPK=3 uv run python scripts/probe_oracle_ceiling.py
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

CONDITIONS = ("baseline", "peak", "oracle", "zoom",
              "zoom_peak", "zoom_topk", "zoom_hybrid")
TOPK = int(os.environ.get("TOPK", 3))       # celle del ranking per zoom_topk
HIT_AT_K = 5                                 # hit@1..5 loggati dal ranking


def cell_span_sec(c0: int, c1: int, t: int, duration: float) -> tuple[float, float]:
    """Estremi in secondi delle celle [c0, c1] inclusi: `[c0/t, (c1+1)/t]*durata`."""
    return c0 / t * duration, (c1 + 1) / t * duration


def window_cells(t: int, duration: float, w0: float, w1: float) -> list[int]:
    """Celle che intersecano [w0, w1] — stessa mappa di probe_peak_in_window."""
    return [c for c in range(t)
            if c / t * duration < w1 and (c + 1) / t * duration > w0]


def pointer_for(a: float, b: float) -> str:
    return POINTER_SPAN_SEC.format(a=a, b=b)


def media_from_spans(video_path: str, spans: list[tuple[float, float]],
                     nframes: int, max_pixels: int, tmp_dirs: list):
    """`VideoFrames` con `nframes` campionati dentro `spans` (uniformi in
    ciascuno, quota uguale per span; il resto della divisione va al primo).

    Gli indici e `fps` sono quelli VERI del video sorgente, quindi il
    processor Qwen3 emette i timestamp assoluti del punto da cui ogni frame
    viene: il modello non riceve un video falsificato, riceve lo stesso video
    campionato altrove. Le tmpdir dei PNG vanno in `tmp_dirs`, che il
    chiamante ripulisce.
    """
    per = nframes // len(spans)
    quotas = [per] * len(spans)
    quotas[0] += nframes - per * len(spans)
    idx_all: list[int] = []
    fps = None
    for (a, b), q in zip(spans, quotas):
        if q <= 0:
            continue
        idx, fps = reconstruct_frame_indices(video_path, q, video_start=a, video_end=b)
        idx_all.extend(int(k) for k in idx)
    # `set`: in `hybrid` gli span si sovrappongono (video intero + cella di
    # picco), quindi lo stesso frame può uscire due volte. Deduplicare è
    # giusto — passarlo due volte cambierebbe la struttura dei token — ma
    # riduce il budget sotto `nframes`: il chiamante lo CONTA (`n_frames`),
    # non lo scopre a posteriori.
    idx_all = sorted(set(idx_all))
    paths, tmp = _extract_all_frames(video_path, idx_all)
    tmp_dirs.append(tmp)
    return VideoFrames(paths, max_pixels=max_pixels, frames_indices=idx_all, fps=fps)


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
                "rowset": rowset, "sink_filter": False, "topk": TOPK,
                "conditions": list(CONDITIONS)},
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
        "peak_cell", "peak_in_window", "oracle_span", "ranked_top5", "hit_at_k",
        *(f"pred_{c}" for c in CONDITIONS),
        *(f"correct_{c}" for c in CONDITIONS),
        "fallbacks",
    ])
    correct = {c: 0 for c in CONDITIONS}
    flips = {f"{c}_{k}": 0 for c in CONDITIONS if c != "baseline"
             for k in ("recovered", "broken")}
    peak_in_win = 0
    hit_at = {k: 0 for k in range(1, HIT_AT_K + 1)}
    n_done = n_skip = n_fallback = n_dedup = 0
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
            zoom_tmps: list = []
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
                ranked = ranked_cells_from_attention(va, rows=rset, sink_filter=False)
                ranked_cells = [rc.cell for rc in ranked]
                peak = ranked_cells[0]
                wcells = window_cells(va.t, duration, w0, w1)
                hit = peak in wcells
                peak_in_win += hit
                # hit@k: la finestra è coperta da una delle prime k celle?
                # Gratis dal ranking (nessun forward), dice se il top-k
                # disgiunto ha senso prima di guardarne l'accuracy.
                for k in range(1, HIT_AT_K + 1):
                    if any(c in wcells for c in ranked_cells[:k]):
                        hit_at[k] += 1

                pa, pb = cell_span_sec(peak, peak, va.t, duration)
                oa, ob = cell_span_sec(wcells[0], wcells[-1], va.t, duration)
                # Le condizioni di RICAMPIONAMENTO non ricevono nessun
                # puntatore: l'intervento è nei frame, non nel prompt.
                prompts = {c: prompt for c in CONDITIONS}
                prompts["peak"] = f"{pointer_for(pa, pb)}\n\n{prompt}"
                prompts["oracle"] = f"{pointer_for(oa, ob)}\n\n{prompt}"
                # zoom: stessi 24 frame di BUDGET ma dentro la finestra
                # annotata. frames_indices/fps veri → il modello vede i
                # timestamp assoluti del punto del video da cui vengono.
                media_by_cond = {c: media for c in CONDITIONS}
                cell_span = lambda c: cell_span_sec(c, c, va.t, duration)
                media_by_cond["zoom"] = media_from_spans(
                    r["video_path"], [(w0, w1)], nframes, max_pixels, zoom_tmps)
                # --- condizioni REALISTICHE: l'oracolo sostituito dal segnale
                media_by_cond["zoom_peak"] = media_from_spans(
                    r["video_path"], [cell_span(peak)], nframes, max_pixels, zoom_tmps)
                topk_cells = sorted(ranked_cells[:TOPK])
                media_by_cond["zoom_topk"] = media_from_spans(
                    r["video_path"], [cell_span(c) for c in topk_cells],
                    nframes, max_pixels, zoom_tmps)
                # hybrid: metà budget sul video intero (contesto preservato),
                # metà nella cella di picco (risoluzione dove punta il segnale)
                media_by_cond["zoom_hybrid"] = media_from_spans(
                    r["video_path"], [(0.0, duration), cell_span(peak)],
                    nframes, max_pixels, zoom_tmps)
                n_hybrid = len(media_by_cond["zoom_hybrid"].frames_indices)
                if n_hybrid < nframes:
                    n_dedup += 1
                # `pred_letter` del prefill = fallback quando la generazione
                # non produce una lettera parseabile, come negli arm.
                fb_idx = (ord(signal.pred_letter) - ord("A")
                          if signal.pred_letter else None)
                preds, fbs = {}, []
                for c in CONDITIONS:
                    messages = vlm.build_messages(media_by_cond[c], Text(prompts[c]))
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
                for d in zoom_tmps:
                    shutil.rmtree(d, ignore_errors=True)

            ok = {c: preds[c] == r["answer"] for c in CONDITIONS}
            for c in CONDITIONS:
                if c == "baseline":
                    continue
                if ok[c] and not ok["baseline"]:
                    flips[f"{c}_recovered"] += 1
                if ok["baseline"] and not ok[c]:
                    flips[f"{c}_broken"] += 1

            first_hit = next((k for k in range(1, HIT_AT_K + 1)
                              if any(c in wcells for c in ranked_cells[:k])), None)
            table.add_data(r["id"], r["question"], round(duration, 1), va.t,
                           round(w0, 1), round(w1, 1), peak, hit,
                           f"[{oa:.0f},{ob:.0f}]s",
                           str(ranked_cells[:5]), first_hit,
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

    print(f"\n=== soffitto su {n_done} sample (skip {n_skip}, fallback {n_fallback}, "
          f"hybrid sotto budget {n_dedup}) — rowset={rowset}, topk={TOPK} ===")
    for c in CONDITIONS:
        print(f"  {c:9s}: {correct[c]}/{n_done} = {correct[c]/n_done:.3f}")
    for c in CONDITIONS:
        if c == "baseline":
            continue
        print(f"  {c:8s} vs baseline: +{flips[f'{c}_recovered']} / -{flips[f'{c}_broken']}"
              f"  (delta {correct[c]-correct['baseline']:+d})")
    print(f"  peak_in_window: {peak_in_win}/{n_done} = {peak_in_win/n_done:.3f}")
    print("  hit@k (la finestra è coperta da una delle prime k celle):")
    for k in range(1, HIT_AT_K + 1):
        print(f"    k={k}: {hit_at[k]}/{n_done} = {hit_at[k]/n_done:.3f}")

    run.summary.update({
        "n_samples": n_done, "n_skip": n_skip, "n_fallback": n_fallback,
        **{f"acc_{c}": correct[c] / n_done for c in CONDITIONS},
        **{f"n_correct_{c}": correct[c] for c in CONDITIONS},
        **flips,
        "peak_in_window_rate": peak_in_win / n_done,
        **{f"hit_at_{k}": hit_at[k] / n_done for k in range(1, HIT_AT_K + 1)},
        "topk": TOPK, "n_hybrid_dedup": n_dedup,
        "model_latency_mean": lat_sum / n_done,
    })
    run.log({"samples": table})
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
