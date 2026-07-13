"""Prefetch VNBench "retrieval" video + sidecar QA per attn_explorer.capture.

VNBench (CASIA-IVA-Lab/VideoNIAH, ICLR 2025) impacchetta 1350 video unici
su 3 task (retrieval/ordering/counting), ciascuno con 4 varianti di domanda
("try" 0-3: stessa clip, opzioni MCQ riordinate/riformulate). Le annotazioni
(`VNBench-main-4try.json`, 2MB) sono su HuggingFace
(`Joez1717/VNBench`) — questo script le scarica. I VIDEO GREZZI invece NON
sono su HF: sono su Google Drive (link nel README di VideoNIAH), quindi
NON scriptabili come gli altri `fetch/prefetch_*.py` di questo repo (che
tirano giù zip HF via `hf_hub_download`). Per questo lo script richiede
`videos_root`: la cartella dove l'utente ha già scaricato ed estratto A
MANO l'archivio Drive (su un nodo con internet, tipicamente il login node
HPC) — qui ci si limita a SELEZIONARE il sottoinsieme richiesto e a
copiarlo (o symlinkarlo) nel layout consumato da
`attn_explorer.capture --video-dir --prompt-from-json`:

    out_dir/
        <stem>.mp4
        <stem>.json    # {question (prompt MCQ già formattato), type, gt,
                       #  gt_option, needle_time, length} — stesso schema
                       #  dei 30 sidecar già presenti in
                       #  data/vnbench/retrieval/.

Solo `try=0` viene usato (coerente con i 30 capture già fatti). Idempotente:
gli stem già presenti in `out_dir` (mp4 + json) vengono saltati, quindi
rilanciare con `n_per_subtype` più alto fa solo un top-up incrementale.

Overrides (stessa sintassi `key=value` di `main.py`, vedi `utils/config.py`):
    uv run python fetch/prefetch_vnbench.py videos_root=/work/shared/vnbench_raw
    uv run python fetch/prefetch_vnbench.py videos_root=... n_per_subtype=100
    uv run python fetch/prefetch_vnbench.py videos_root=... subtypes='[ret_insert1]' n_per_subtype=50
    uv run python fetch/prefetch_vnbench.py videos_root=... symlink=true
    uv run python fetch/prefetch_vnbench.py --print-config

Run on the login node (compute nodes typically lack internet) — e comunque
dopo aver estratto l'archivio Drive sotto `videos_root`.
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weave
import yaml

from utils.config import Cfg, load_flat_config
from utils.obs import init_observability

CONF_PATH = Path(__file__).resolve().parent.parent / "conf" / "fetch" / "prefetch_vnbench.yaml"

logger = logging.getLogger(__name__)

ALL_SUBTYPES = ("ret_insert1", "ret_insert2", "ret_edit1")


def _fetch_annotations(hf_repo: str, annotation_file: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=annotation_file)
    with open(local) as f:
        return json.load(f)


def _format_prompt(question: str, options: list[str]) -> str:
    """Stesso formato dei sidecar già presenti in data/vnbench/retrieval/
    (verificato contro i 30 esistenti): NESSUN prefisso "Question: ", a
    differenza di `evals.base.format_mcq_prompt`."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
    return f"{question}\nOptions:\n{body}\nAnswer with the letter of the correct option only."


def _select_rows(rows: list[dict], subtypes: list[str], n_per_subtype: int) -> dict[str, list[dict]]:
    """`try=0` per ciascun sottotipo, troncato a `n_per_subtype` (ordine
    stabile = ordine di apparizione nell'annotazione)."""
    unknown = [s for s in subtypes if s not in ALL_SUBTYPES]
    if unknown:
        raise ValueError(f"sottotipi sconosciuti: {unknown}. Validi: {ALL_SUBTYPES}")

    by_subtype: dict[str, list[dict]] = {s: [] for s in subtypes}
    for row in rows:
        if row.get("try") != 0:
            continue
        st = row.get("type")
        if st in by_subtype and len(by_subtype[st]) < n_per_subtype:
            by_subtype[st].append(row)

    for st, selected in by_subtype.items():
        if len(selected) < n_per_subtype:
            logger.warning(
                "sottotipo=%s: richiesti %d, trovati solo %d nell'annotazione (try=0)",
                st, n_per_subtype, len(selected),
            )
    return by_subtype


@weave.op()
def prefetch(
    hf_repo: str,
    annotation_file: str,
    subtypes: list[str],
    n_per_subtype: int,
    videos_root: str,
    out_dir: str,
    symlink: bool,
) -> dict:
    videos_root_path = Path(videos_root)
    if not videos_root_path.is_dir():
        raise FileNotFoundError(
            f"videos_root={videos_root!r} non esiste o non è una cartella — "
            "scarica ed estrai prima l'archivio Drive di VNBench (vedi docstring)."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("scarico annotazione %s/%s", hf_repo, annotation_file)
    rows = _fetch_annotations(hf_repo, annotation_file)
    by_subtype = _select_rows(rows, list(subtypes), n_per_subtype)

    n_written = n_skipped_existing = n_missing_video = 0
    for st, selected in by_subtype.items():
        for row in selected:
            stem = Path(row["video"]).stem
            dst_video = out / f"{stem}.mp4"
            dst_json = out / f"{stem}.json"
            if dst_video.is_file() and dst_json.is_file():
                n_skipped_existing += 1
                continue

            src_video = next(videos_root_path.rglob(f"{stem}.mp4"), None)
            if src_video is None:
                logger.warning("stem=%s: mp4 non trovato sotto %s, salto", stem, videos_root)
                n_missing_video += 1
                continue

            if symlink:
                if dst_video.exists() or dst_video.is_symlink():
                    dst_video.unlink()
                dst_video.symlink_to(src_video.resolve())
            else:
                shutil.copy2(src_video, dst_video)

            sidecar = {
                "question": _format_prompt(row["question"], row["options"]),
                "type": row["type"],
                "gt": row["gt"],
                "gt_option": row["gt_option"],
                "needle_time": row["needle_time"],
                "length": row["length"],
            }
            dst_json.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            n_written += 1

    logger.info(
        "scritti %d video nuovi in %s (già presenti: %d, mp4 mancanti in videos_root: %d)",
        n_written, out, n_skipped_existing, n_missing_video,
    )
    return {
        "n_written": n_written,
        "n_skipped_existing": n_skipped_existing,
        "n_missing_video": n_missing_video,
        "out_dir": str(out),
        "by_subtype_requested": {k: len(v) for k, v in by_subtype.items()},
    }


def run(cfg: Cfg) -> None:
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    init_observability(cfg)
    prefetch(
        cfg.hf_repo,
        cfg.annotation_file,
        list(cfg.subtypes),
        int(cfg.n_per_subtype),
        cfg.videos_root,
        cfg.out_dir,
        bool(cfg.symlink),
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
