"""Il picco d'attenzione cade nella finestra di evidenza annotata? — LVBench.

Misura (1) del piano-oracolo (vedi memoria di progetto e handoff 2026-08-30):
tutti gli arm di salienza su Video-MME sono nulli con attention ≡ random, ma
l'accuracy non distingue "segnale sbagliato" da "niente da guadagnare". Qui il
giudizio è DIRETTO: LVBench annota la finestra temporale dell'evidenza
(`time_reference` → `video_start`/`video_end`, evals/lvbench.py), quindi si
può chiedere al segnale — senza passare dall'accuracy — se punta al posto
giusto.

Per ogni sample: un forward di prefill con cattura (`generate_with_signals`,
identico al pass 1 degli arm), poi il picco viene ricalcolato per TRE rowset
sulla STESSA attenzione:

    all       tutte le righe query        (regime storico degli arm)
    question  solo le righe della domanda (arm `query_rows=question`)
    entity    righe del soggetto estratto (v4: UNA sequenza verbatim o NONE,
              fallback a question — utils/attn_core.py::resolve_query_rows)

`all` e `question` sono riaggregazioni a costo zero; `entity` aggiunge una
generazione solo-testo (~32 token) per sample. `sink_filter=False` come negli
arm misurati (attenzione grezza).

Metrica per rowset: hit = la cella di picco interseca la finestra annotata.
Il livello di caso è per-sample: (celle che intersecano la finestra)/t — una
cella a nframes=24 su un video LVBench lungo copre minuti, quindi il caso NON
è 1/t e va riportato accanto agli hit rate. La lettura è:

    hit(entity) > hit(question) > hit(all) >> chance  → il segnale esiste e
        l'affilatura delle righe lo migliora: gli arm hanno un motore;
    tutti ≈ chance → il picco non punta all'evidenza nemmeno qui, e i null
        di Video-MME sono spiegati alla radice (risultato di tesi).

Confronto appaiato entity-vs-question incluso (stessa attenzione, stesso
sample): conta quanti sample entity recupera/rompe rispetto a question.

Log: wandb (project lvbench, config da conf/config.yaml) con Table per-sample
e summary — verificabile da remoto via `/wandb` (mai SSH). Stdout replica
tutto per il .out di SLURM.

Uso:
    uv run python scripts/probe_peak_in_window.py [n_samples=60] [seed=0]
    NFRAMES=24 MAX_PIXELS=151200 uv run python scripts/probe_peak_in_window.py

Serve una GPU e i video LVBench in `dataset.root` (conf/config.yaml).
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

# Lanciato come `python scripts/...` il repo root non è su sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GenerationConfig

from evals.base import format_mcq_prompt
from evals.lvbench import LVBench
from models.media import Text
from models.qwen_attn import Qwen3VL2BAttention
from strategies.base import SamplingBudget, Strategy, video_duration_sec
from utils.attn_core import question_rows, ranked_cells_from_attention, resolve_query_rows
from utils.config import load_config

ROWSETS = ("all", "question", "entity")


def window_cells(t: int, duration: float, w0: float, w1: float) -> set[int]:
    """Celle (indice 0..t-1) la cui estensione temporale interseca [w0, w1].

    Cella `c` copre `[c/t, (c+1)/t) * duration` — stessa mappa di
    `_cell_span_frac` (strategies/attention_highlight.py): frazione uniforme
    della clip, che qui è il video INTERO (nessun trim), quindi le coordinate
    coincidono con quelle assolute dell'annotazione.
    """
    out = set()
    for c in range(t):
        a, b = c / t * duration, (c + 1) / t * duration
        if a < w1 and b > w0:
            out.add(c)
    return out


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nframes = int(os.environ.get("NFRAMES", 24))
    max_pixels = int(os.environ.get("MAX_PIXELS", 151200))

    cfg = load_config(["dataset=lvbench"])
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
        print("nessun sample utilizzabile — root sbagliata o video mancanti?", file=sys.stderr)
        return 1

    import wandb
    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity),
        name=f"probe_peak_in_window-{os.environ.get('SLURM_JOB_ID', 'local')}",
        group="probe-peak-in-window",
        tags=["probe", "lvbench", "peak_in_window", "query_rows"],
        config={"n_samples": len(picked), "seed": seed, "nframes": nframes,
                "max_pixels": max_pixels, "model": "qwen3_vl_2b_attn",
                "sink_filter": False, "rowsets": list(ROWSETS)},
    )

    vlm = Qwen3VL2BAttention(torch_dtype=torch.bfloat16, device_map="auto",
                             max_pixels=max_pixels)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=8)  # prefill-only, non usata
    budget = SamplingBudget(nframes=nframes, max_pixels=max_pixels,
                            min_pixels=None, double_frames=False)

    table = wandb.Table(columns=[
        "id", "question", "duration", "t", "win_start", "win_end", "chance",
        "peak_all", "hit_all", "peak_question", "hit_question",
        "peak_entity", "hit_entity", "entity_mode", "entity_raw", "entity_mapped",
        "pred_letter", "correct_prefill",
    ])
    hits = {k: 0 for k in ROWSETS}
    chance_sum = 0.0
    ent_mapped = ent_none = 0
    paired = {"both": 0, "only_entity": 0, "only_question": 0, "neither": 0}
    n_done = n_skip = 0
    lat_sum = 0.0

    for i, r in enumerate(picked, 1):
        t0 = time.monotonic()
        try:
            prompt = format_mcq_prompt(r["question"], r["options"])
            letters = [chr(ord("A") + k) for k in range(len(r["options"]))]
            media, tmp_dir = Strategy._build_media(r["video_path"], None, None, None, budget)
            try:
                signal = vlm.generate_with_signals(media, Text(prompt), gen_cfg,
                                                   answer_letters=letters)
            finally:
                if tmp_dir is not None:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            va = signal.visual_attention
            duration = video_duration_sec(r["video_path"])
            w0 = max(0.0, min(r["video_start"], duration))
            w1 = max(0.0, min(r["video_end"], duration))
            if w1 <= w0:
                raise ValueError(f"finestra vuota dopo il clip alla durata: [{w0}, {w1}]")

            wcells = window_cells(va.t, duration, w0, w1)
            chance = len(wcells) / va.t

            res = resolve_query_rows("entity", va.query_tokens,
                                     extract_entities=vlm.extract_entities)
            rowset_rows = {"all": [], "question": question_rows(va.query_tokens),
                           "entity": res.rows}
            peaks, hit_now = {}, {}
            for k in ROWSETS:
                ranked = ranked_cells_from_attention(va, rows=rowset_rows[k],
                                                     sink_filter=False)
                peaks[k] = ranked[0].cell
                hit_now[k] = peaks[k] in wcells
                hits[k] += hit_now[k]
            chance_sum += chance
            ent_mapped += bool(res.entity_mapped)
            is_none = (res.entity_raw or "").strip().strip('.,;:!"\'').lower() == "none"
            ent_none += is_none
            key = ("both" if hit_now["entity"] and hit_now["question"] else
                   "only_entity" if hit_now["entity"] else
                   "only_question" if hit_now["question"] else "neither")
            paired[key] += 1

            correct = (signal.pred_letter is not None and
                       (ord(signal.pred_letter) - ord("A")) == r["answer"])
            table.add_data(r["id"], r["question"], round(duration, 1), va.t,
                           round(w0, 1), round(w1, 1), round(chance, 4),
                           peaks["all"], hit_now["all"],
                           peaks["question"], hit_now["question"],
                           peaks["entity"], hit_now["entity"],
                           res.mode, res.entity_raw, bool(res.entity_mapped),
                           signal.pred_letter, correct)
            n_done += 1
            lat = time.monotonic() - t0
            lat_sum += lat
            marks = " ".join(f"{k}={'HIT' if hit_now[k] else 'miss'}" for k in ROWSETS)
            print(f"[{i:3}/{len(picked)}] {r['id']}: {marks} chance={chance:.2f} "
                  f"win=[{w0:.0f},{w1:.0f}]s/{duration:.0f}s entity={res.entity_raw!r} "
                  f"({lat:.1f}s)", flush=True)
            del va, signal
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — un sample rotto non uccide la sonda
            n_skip += 1
            print(f"[{i:3}/{len(picked)}] {r['id']}: SKIP — {type(e).__name__}: {e}",
                  flush=True)
            torch.cuda.empty_cache()

    if n_done == 0:
        print("nessun sample completato", file=sys.stderr)
        run.finish(exit_code=1)
        return 1

    chance_mean = chance_sum / n_done
    print(f"\n=== peak-in-window su {n_done} sample (skip {n_skip}) — chance media {chance_mean:.3f} ===")
    for k in ROWSETS:
        print(f"  {k:9s}: {hits[k]}/{n_done} = {hits[k]/n_done:.3f}")
    print(f"  appaiato entity vs question: both={paired['both']} only_entity={paired['only_entity']} "
          f"only_question={paired['only_question']} neither={paired['neither']}")
    print(f"  entity: mappate {ent_mapped}/{n_done}, NONE deliberati {ent_none}/{n_done}")

    run.summary.update({
        "n_samples": n_done, "n_skip": n_skip, "chance_mean": chance_mean,
        **{f"hit_rate_{k}": hits[k] / n_done for k in ROWSETS},
        **{f"hits_{k}": hits[k] for k in ROWSETS},
        **{f"paired_{k}": v for k, v in paired.items()},
        "entity_mapped_rate": ent_mapped / n_done,
        "entity_none_rate": ent_none / n_done,
        "model_latency_mean": lat_sum / n_done,
    })
    run.log({"samples": table})
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
