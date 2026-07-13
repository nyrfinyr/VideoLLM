"""Prefetch MVBench videos + QA metadata for pipeline testing.

MVBench (`OpenGVLab/MVBench`) impacchetta i video di 11 dataset sorgente
in altrettanti zip sotto `video/<source>.zip`, e i metadati QA in 20
JSON sotto `json/<task>.json`. Questo script scarica i zip dei dataset
toccati dai task richiesti, li estrae, scarica i JSON corrispondenti,
e produce un `metadata.jsonl` self-contained — stesso layout di
`prefetch_egoschema`, leggibile poi da `evals.mvbench.load_mvbench`.

Output:

    out_dir/
        metadata.jsonl     # una riga JSON per sample, schema:
                           #   {id, task_type, video, data_type,
                           #    question, options, answer, answer_idx,
                           #    start?, end?}
                           # `video` è path relativo a out_dir.
        videos/
            <source>/...   # contenuto estratto dei zip (mp4 / frames).

NTU RGB+D non è incluso nel repo HF (licenza), quindi il task
`fine_grained_pose` è escluso dal default. Va fetchato a parte se serve.

`tvqa.zip` contiene **sequenze di frame**, non mp4: per il task
`episodic_reasoning` la riga risultante ha `data_type='frame'` —
`load_mvbench` deve gestirlo o saltarlo.

Run on the login node (compute nodes typically lack internet).

Overrides (stessa sintassi `key=value` di `main.py`, vedi `utils/config.py`):
    uv run python fetch/prefetch_mvbench.py tasks='[action_antonym]'
    uv run python fetch/prefetch_mvbench.py n=5
    uv run python fetch/prefetch_mvbench.py keep_zips=true
    uv run python fetch/prefetch_mvbench.py --print-config
"""
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

# When invoked as `python fetch/prefetch_mvbench.py`, only `fetch/` is on
# sys.path — add its parent (repo root) so the absolute import
# `from utils.obs import ...` resolves regardless of invocation mode
# (script, `python -m fetch.prefetch_mvbench`, or installed entry point).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weave
import yaml

from utils.config import Cfg, load_flat_config
from utils.obs import init_observability

CONF_PATH = Path(__file__).resolve().parent.parent / "conf" / "fetch" / "prefetch_mvbench.yaml"

logger = logging.getLogger(__name__)


# Mappa canonica task → metadati, derivata dal `data_list` nel notebook
# di reference degli autori MVBench (cell 11 di videochat2/mvbench.ipynb).
# Campi:
#   json:         file in `json/` del repo HF.
#   zip:          file in `video/` del repo HF da estrarre per i video.
#   video_subdir: path *relativo* a `<out>/videos/` dove vivono i file
#                 referenziati dal campo `video` del JSON. Coincide col
#                 `prefix` del notebook (senza `your_data_path/`).
#   data_type:    "video" (default) | "frame" — frame = sequenza jpg
#                 numerati per Episodic Reasoning.
#   bound:        True = il JSON include start/end (timestamp del
#                 clip rilevante dentro un video più lungo).
# Esclusi: `fine_grained_pose` (NTU, non redistribuito).
TASKS: dict[str, dict] = {
    "action_sequence":          {"json": "action_sequence.json",          "zip": "star.zip",                "video_subdir": "star/Charades_v1_480",      "bound": True},
    "action_prediction":        {"json": "action_prediction.json",        "zip": "star.zip",                "video_subdir": "star/Charades_v1_480",      "bound": True},
    "action_antonym":           {"json": "action_antonym.json",           "zip": "ssv2_video.zip",          "video_subdir": "ssv2_video",                "bound": False},
    "fine_grained_action":      {"json": "fine_grained_action.json",      "zip": "Moments_in_Time_Raw.zip", "video_subdir": "Moments_in_Time_Raw/videos","bound": False},
    "unexpected_action":        {"json": "unexpected_action.json",        "zip": "FunQA_test.zip",          "video_subdir": "FunQA_test/test",           "bound": False},
    "object_existence":         {"json": "object_existence.json",         "zip": "clevrer.zip",             "video_subdir": "clevrer/video_validation",  "bound": False},
    "object_interaction":       {"json": "object_interaction.json",       "zip": "star.zip",                "video_subdir": "star/Charades_v1_480",      "bound": True},
    "object_shuffle":           {"json": "object_shuffle.json",           "zip": "perception.zip",          "video_subdir": "perception/videos",         "bound": False},
    "moving_direction":         {"json": "moving_direction.json",         "zip": "clevrer.zip",             "video_subdir": "clevrer/video_validation",  "bound": False},
    "action_localization":      {"json": "action_localization.json",      "zip": "sta.zip",                 "video_subdir": "sta/sta_video",             "bound": True},
    "scene_transition":         {"json": "scene_transition.json",         "zip": "scene_qa.zip",            "video_subdir": "scene_qa/video",            "bound": False},
    "action_count":             {"json": "action_count.json",             "zip": "perception.zip",          "video_subdir": "perception/videos",         "bound": False},
    "moving_count":             {"json": "moving_count.json",             "zip": "clevrer.zip",             "video_subdir": "clevrer/video_validation",  "bound": False},
    "moving_attribute":         {"json": "moving_attribute.json",         "zip": "clevrer.zip",             "video_subdir": "clevrer/video_validation",  "bound": False},
    "state_change":             {"json": "state_change.json",             "zip": "perception.zip",          "video_subdir": "perception/videos",         "bound": False},
    "character_order":          {"json": "character_order.json",          "zip": "perception.zip",          "video_subdir": "perception/videos",         "bound": False},
    "egocentric_navigation":    {"json": "egocentric_navigation.json",    "zip": "vlnqa.zip",               "video_subdir": "vlnqa",                     "bound": False},
    "episodic_reasoning":       {"json": "episodic_reasoning.json",       "zip": "tvqa.zip",                "video_subdir": "tvqa/frames_fps3_hq",       "bound": True, "data_type": "frame"},
    "counterfactual_inference": {"json": "counterfactual_inference.json", "zip": "clevrer.zip",             "video_subdir": "clevrer/video_validation",  "bound": False},
}


def _fetch_zip_and_extract(
    repo_id: str,
    zip_name: str,
    out: Path,
    keep_zips: bool,
) -> None:
    """Scarica `video/<zip_name>` dal repo HF ed estrae sotto `<out>/videos/`.

    Idempotente: marker `.extracted_<stem>` in `<out>/videos/` salta lo
    step se già fatto.

    NOTA `keep_zips`: il flag è esposto per simmetria con
    `prefetch_videomme.py` ma qui è di fatto un no-op. `hf_hub_download`
    senza `local_dir=` mette il zip nella cache HF condivisa (`$HF_HOME`)
    e questo script non scrive copie in `<out>/_zips/`. Per liberare
    spazio dopo l'estrazione: `huggingface-cli cache scan` /
    `huggingface-cli delete-cache`.
    """
    from huggingface_hub import hf_hub_download

    zip_stem = Path(zip_name).stem  # es. "star.zip" → "star"
    extract_marker = out / "videos" / f".extracted_{zip_stem}"
    if extract_marker.exists():
        logger.info("zip %s già estratto, salto", zip_name)
        return

    logger.info("scarico video/%s da %s", zip_name, repo_id)
    local_zip = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"video/{zip_name}",
    )

    logger.info("estraggo %s → %s", local_zip, out / "videos")
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(out / "videos")

    extract_marker.parent.mkdir(parents=True, exist_ok=True)
    extract_marker.touch()
    _ = keep_zips  # cfr. nota nel docstring


def _fetch_json(repo_id: str, json_name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"json/{json_name}",
    )
    with open(local) as f:
        return json.load(f)


def _normalize_row(
    task_type: str,
    task_meta: dict,
    row: dict,
    idx: int,
) -> dict:
    """Mappa una riga del JSON al nostro schema normalizzato."""
    answer_str: str = row["answer"]
    candidates: list[str] = row["candidates"]
    try:
        answer_idx = candidates.index(answer_str)
    except ValueError:
        # MVBench dovrebbe sempre avere `answer ∈ candidates`; logghiamo
        # e teniamo answer_idx=None così l'eval può filtrare a valle.
        logger.warning(
            "task=%s row=%d: answer %r non in candidates %r",
            task_type, idx, answer_str, candidates,
        )
        answer_idx = None

    video_rel = Path("videos") / task_meta["video_subdir"] / row["video"]

    out_row = {
        "id": f"{task_type}/{idx}",
        "task_type": task_type,
        "video": str(video_rel),
        "data_type": task_meta.get("data_type", "video"),
        "question": row["question"],
        "options": candidates,
        "answer": answer_str,
        "answer_idx": answer_idx,
    }
    if task_meta["bound"]:
        out_row["start"] = row.get("start")
        out_row["end"] = row.get("end")
    return out_row


@weave.op()
def prefetch(
    hf_repo: str,
    tasks: list[str] | None,
    n: int | None,
    out_dir: str,
    keep_zips: bool,
) -> dict:
    out: Path = Path(out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    selected = list(TASKS.keys()) if tasks is None else list(tasks)
    unknown = [t for t in selected if t not in TASKS]
    if unknown:
        raise ValueError(
            f"task sconosciuti: {unknown}. Validi: {sorted(TASKS.keys())}"
        )
    logger.info("task selezionati (%d): %s", len(selected), selected)

    # Scarica una sola volta per zip, anche se più task condividono lo
    # stesso source (es. clevrer copre 5 task).
    zips_needed = {TASKS[t]["zip"] for t in selected}
    logger.info("zip da scaricare (%d): %s", len(zips_needed), sorted(zips_needed))
    for zip_name in sorted(zips_needed):
        _fetch_zip_and_extract(hf_repo, zip_name, out, keep_zips)

    metadata: list[dict] = []
    for task_type in selected:
        task_meta = TASKS[task_type]
        rows = _fetch_json(hf_repo, task_meta["json"])
        if n is not None:
            rows = rows[:n]
        logger.info("task=%s: %d sample", task_type, len(rows))

        subdir = out / "videos" / task_meta["video_subdir"]
        if not subdir.exists():
            raise FileNotFoundError(
                f"task={task_type}: subdir attesa non trovata: {subdir}. "
                f"Il layout interno di {task_meta['zip']} potrebbe essere "
                "diverso da quello assunto in TASKS."
            )

        for idx, row in enumerate(rows):
            metadata.append(_normalize_row(task_type, task_meta, row, idx))

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for sample in metadata:
            f.write(json.dumps(sample) + "\n")

    logger.info("Wrote %d sample + metadata.jsonl in %s", len(metadata), out)
    return {
        "subset_size": len(metadata),
        "tasks": selected,
        "out_dir": str(out),
        "metadata_path": str(metadata_path),
    }


def run(cfg: Cfg) -> None:
    # `datasets`/`huggingface_hub` leggono HF_HOME al primo import, va
    # settato prima di importare dentro `prefetch`.
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    tasks = None if cfg.tasks is None else list(cfg.tasks)
    prefetch(
        cfg.hf_repo,
        tasks,
        cfg.n,
        cfg.out_dir,
        bool(cfg.keep_zips),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)

    print_config = "--print-config" in argv
    if print_config:
        argv.remove("--print-config")

    cfg = load_flat_config(CONF_PATH, argv)
    if print_config:
        print(yaml.safe_dump(dict(cfg), sort_keys=False))
        return
    run(cfg)


if __name__ == "__main__":
    main()
