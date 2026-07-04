#!/usr/bin/env python3
"""Scarica 10 video Causal2Needles + UNA domanda "2-needle" ciascuno,
pronti per `attn_explorer.capture --video-dir videos/causal-2-needles
--prompt-from-json` (stesso pattern sidecar di `videos/qv-highlights/`).

A differenza di `fetch/prefetch_causal2needles.py` (Hydra, pesca righe dal
parquet piatto `test_questions.parquet`, tipicamente tutte "1-causal" se
`n` è piccolo — le righe sono raggruppate per categoria) qui si pescano
direttamente i file `questions/<source>/2needle/<id>.json`: ogni file
contiene, per quel video, la domanda che richiede di localizzare e
collegare DUE needle (causa + effetto) distanti nel video — il caso
d'uso "needle in a haystack" per cui serve questo script.

Ogni entry di quei json ha la forma:
    {combined_question: [[cause_idx, effect_idx],
                          [cause_sent, effect_sent],
                          [vg_prompt_con_MCQ, vg_answer_letter],
                          [sub_question_1, sub_question_2]]}
Si usa `vg_prompt_con_MCQ` (variante "visual grounding": chiede un
dettaglio visivo nel momento dell'effetto, MCQ) come prompt principale —
è la formulazione più stringente per verificare se l'attenzione copre
davvero entrambi i needle, non solo l'ultimo frame visto.

`cause_idx`/`effect_idx` NON sono frame né secondi: indicizzano la lista
di frasi narrative del recap in `annotations.json` (un file unico da
~100MB per l'intero dataset, con `annotations["yms"][<id>]["annotations"]`
per YMS e `annotations["en"][<id>]["annotations"]` per SyMoN — ciascuna
frase porta `begin_time`/`end_time` in SECONDI REALI nel video). Li
risolviamo qui in `cause_time`/`effect_time` `[begin, end]` così il
sidecar ha già l'aggancio temporale pronto per `validate_causal2needles.py`.

Output flat (capture.py fa un glob non ricorsivo su `--video-dir`):
    videos/causal-2-needles/
        yms_0000.mp4
        yms_0000.json      # {"question": <vg_prompt>, "cause_time": [s,s], ...}
        symon_0IUynKY8i3I.mp4
        symon_0IUynKY8i3I.json
        ...

Uso:
    uv run python attn_explorer/script/fetch_causal2needles_samples.py
    uv run python attn_explorer/script/fetch_causal2needles_samples.py --outdir videos/causal-2-needles --n 10

Rilanciarlo su un outdir già popolato è economico: salta il download dei
.mp4 già presenti ma ricalcola/riscrive sempre i sidecar .json (utile per
retrofit di campi nuovi, come `cause_time`/`effect_time` qui aggiunti).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("fetch_causal2needles_samples")

REPO = "causal2needles/Causal2Needles"


def narrative_window(annotations: dict, source: str, vid: str, idx: int) -> list[float]:
    """`[begin_time, end_time]` (secondi) della frase narrativa `idx` del video.

    YMS vive sotto `annotations["yms"][vid]`; SyMoN è annotato in inglese
    sotto `annotations["en"][vid]` (stesso id senza prefisso "symon/").
    """
    lang_key = "yms" if source == "yms" else "en"
    sentence = annotations[lang_key][vid]["annotations"][idx]
    return [sentence["begin_time"], sentence["end_time"]]

# Mix yms (id numerici "0000".."0093") + symon (id in stile YouTube) per un
# minimo di diversità di fonte, in ordine alfabetico dal listing del repo.
# Pool più grande di `--n`: non tutti i video hanno domande 2-needle (es.
# yms/0001 e symon/0IUynKY8i3I ne sono privi), `main()` scorre il pool
# finché non raggiunge `n` successi.
DEFAULT_VIDEOS: list[tuple[str, str]] = [
    ("yms", f"{i:04d}") for i in range(8)
] + [
    ("symon", vid) for vid in [
        "0IUynKY8i3I", "0Xf3cSdQP6o", "1CQE2Fh-1F8", "1dvXckcb8Bg", "2IXNaXHEe5c",
        "2ZschG5EtR0", "49HZcY1UiwM", "7Sv2SC0l0o8",
    ]
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="videos/causal-2-needles", help="cartella flat di output (default videos/causal-2-needles)")
    ap.add_argument("--n", type=int, default=10, help="quanti video scaricare (default 10; scorre il pool di candidati finché non ne trova n con domande 2-needle)")
    return ap.parse_args()


def fetch_one(source: str, vid: str, outdir: Path, annotations: dict) -> bool:
    """Scarica/aggiorna UN video + il suo sidecar json (domanda 2-needle-VG).

    Il download del .mp4 (pesante) è idempotente (salta se `<stem>.mp4`
    esiste già); il sidecar .json (leggero) viene sempre ricalcolato e
    riscritto, così un retrofit di campi (es. `cause_time`) non richiede
    ri-scaricare i video. Ritorna False (loggando un warning) se il video
    non ha domande 2-needle disponibili, altrimenti True.
    """
    from huggingface_hub import hf_hub_download

    stem = f"{source}_{vid}"
    video_path = outdir / f"{stem}.mp4"

    questions_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"questions/{source}/2needle/{vid}.json",
    )
    entries = json.loads(Path(questions_path).read_text())
    if not entries:
        logger.warning("nessuna domanda 2-needle per %s, salto", stem)
        return False

    combined_question, (idxs, sents, vg, subs) = next(iter(entries.items()))
    cause_idx, effect_idx = idxs
    cause_sent, effect_sent = sents
    vg_prompt, vg_answer = vg
    sub_q1, sub_q2 = subs

    if video_path.exists():
        logger.info("[%s] video già presente, salto il download", stem)
    else:
        video_cache_path = hf_hub_download(
            repo_id=REPO, repo_type="dataset",
            filename=f"videos/{source}/{vid}.mp4",
        )
        shutil.copy2(video_cache_path, video_path)

    sidecar = {
        "question": vg_prompt,  # letta da capture.py (--prompt-from-json)
        "video_id": f"{source}/{vid}",
        "question_type": "2-needle-vg",
        "combined_question": combined_question,
        "cause_idx": cause_idx,
        "effect_idx": effect_idx,
        "cause_sent": cause_sent,
        "effect_sent": effect_sent,
        "cause_time": narrative_window(annotations, source, vid, cause_idx),
        "effect_time": narrative_window(annotations, source, vid, effect_idx),
        "sub_questions": [sub_q1, sub_q2],
        "answer": vg_answer,
    }
    (outdir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    logger.info(
        "[%s] video + domanda 2-needle salvati (causa=%.1fs-%.1fs, effetto=%.1fs-%.1fs)",
        stem, *sidecar["cause_time"], *sidecar["effect_time"],
    )
    return True


def load_annotations() -> dict:
    """Scarica (una volta, cache HF) e carica `annotations.json` (~100MB,
    tutte le lingue/video del dataset — serve solo per risolvere
    `cause_idx`/`effect_idx` in secondi, non c'è un endpoint per righe
    parziali quindi si scarica per intero."""
    from huggingface_hub import hf_hub_download

    logger.info("Carico annotations.json (~100MB, cache HF — una tantum)...")
    path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename="annotations.json")
    return json.loads(Path(path).read_text())


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()

    ok = 0
    tried = 0
    for source, vid in DEFAULT_VIDEOS:
        if ok >= args.n:
            break
        tried += 1
        try:
            if fetch_one(source, vid, outdir, annotations):
                ok += 1
        except Exception:
            logger.exception("fetch fallito per %s/%s", source, vid)

    if ok < args.n:
        logger.warning(
            "solo %d/%d video ottenuti: il pool di candidati (%d) è esaurito, "
            "aggiungine altri a DEFAULT_VIDEOS",
            ok, args.n, len(DEFAULT_VIDEOS),
        )
    logger.info("Fatto: %d/%d video scaricati (su %d tentati) in %s", ok, args.n, tried, outdir)
    logger.info(
        "Lancia: uv run python -m attn_explorer.capture --video-dir %s --prompt-from-json --outdir debug_out_causal2needles",
        outdir,
    )


if __name__ == "__main__":
    main()
