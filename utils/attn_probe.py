"""Suite tool attenzione (1/2): cattura entity→token visivi su un sample.

Esegue `Qwen25VLAttention.entity_visual_attention` su UN singolo sample,
saltando l'eval Weave: serve a validare su GPU il wiring della custom
attention (dispatch sul registry, shape degli stash `_last_attn`, sanity
della heatmap) senza lanciare l'intero benchmark. Orchestrato dallo script
top-level `debug_attention.py` (CLI argparse, niente Hydra), separato dal
flusso di eval su dataset di `main.py`. Il dump prodotto è poi consumato da
`utils/attn_frames.py` (estrazione frame + overlay).
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def select_debug_sample(samples: list[dict], video_idx: str | None) -> dict:
    """Sceglie il sample di debug.

    Con `video_idx` (knob `debug_video_idx`) cerca quel video specifico fra
    i sample caricati; senza, ripiega sul primo (`samples[0]`, dopo
    shuffle/shard/limit). Solleva se il `video_idx` richiesto non c'è — di
    solito perché `limit` lo ha tagliato fuori: lascia `limit=null`.
    """
    if not video_idx:
        return samples[0]
    for s in samples:
        if s.get("video_idx") == video_idx:
            return s
    raise ValueError(
        f"video_idx={video_idx!r} non trovato fra i {len(samples)} sample "
        "caricati (limit lo ha escluso? lascia limit=null)"
    )


def _sample_doubled_frames(
    video_path: str, nframes: int, outdir: Path,
) -> tuple[list[str], list[int]]:
    """Estrae `nframes` frame uniformi dal video e li DUPLICA per il merge.

    Qwen fonde i frame a coppie nel patch embed temporale; duplicando ogni
    frame (`[f0, f0, f1, f1, ...]`) la coppia è fatta di due frame identici →
    ogni cella temporale dell'attenzione coincide con un singolo frame reale.

    Campiona con lo stesso schema `linspace(0, total-1, nframes)` del path
    `Video` normale, scrive i frame su disco (qwen-vl-utils vuole path) e
    restituisce la lista RADDOPPIATA da passare a `VideoFrames` più gli indici
    reali (NON raddoppiati: indice `ti` ↔ cella temporale `ti`).
    """
    import decord
    import torch
    from PIL import Image

    vr = decord.VideoReader(video_path)
    total = len(vr)
    frame_indices = torch.linspace(0, total - 1, nframes).round().long().tolist()
    paths: list[str] = []
    for k, fi in enumerate(frame_indices):
        p = outdir / f"f{k:04d}.png"
        Image.fromarray(vr[fi].asnumpy()).save(p)
        paths.append(str(p))
    doubled = [p for p in paths for _ in range(2)]   # [p0, p0, p1, p1, ...]
    return doubled, frame_indices


def local_debug_sample(video_path: str) -> tuple[dict, str]:
    """Costruisce un sample one-off da un video locale, senza dataset.

    Legge il prompt dal file `.txt` con lo stesso stem del video
    (`videos/paperella.mp4` → `videos/paperella.txt`): quel file contiene il
    prompt GIÀ formattato, passato tale e quale a `entity_visual_attention`
    (nessun `format_mcq_prompt`). Solleva se il `.txt` non esiste.

    Returns:
        (sample, prompt) — `sample` ha solo `video_path`; `prompt` è il testo.
    """
    from pathlib import Path

    video = Path(video_path)
    prompt_file = video.with_suffix(".txt")
    if not prompt_file.is_file():
        raise FileNotFoundError(
            f"prompt non trovato: atteso {prompt_file} (stesso stem del video)"
        )
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    return {"video_path": str(video)}, prompt


def run_entity_attention_debug(
    vlm,
    sample: dict,
    entity: str,
    *,
    nframes: int,
    max_pixels: int,
    min_pixels: int | None = None,
    double_frames: bool = False,
    topk: int = 8,
    prompt: str | None = None,
    outdir: Path,
) -> dict:
    """Lancia `entity_visual_attention` su `sample`, salva il dump ed estrae PNG.

    Ricostruisce lo stesso `(media, text)` che `EgoSchema.predict_factory`
    passerebbe a `generate` (stesso `Video` + prompt MCQ), così l'attenzione
    catturata è quella delle reali condizioni di prefill dell'eval. Dopo il
    forward: salva la heatmap in `<outdir>/debug_<entity>.pt` e chiama
    `extract_attended_frames` per scrivere i top-K frame + overlay.

    Args:
        vlm: deve essere un `Qwen25VLAttention` (espone
            `entity_visual_attention`); con gli altri modelli solleva.
        sample: una riga del loader egoschema (video_path, question, options).
        entity: la stringa di cui estrarre le query — DEVE comparire nel
            prompt, altrimenti `_find_entity_span` solleva `ValueError`.
        nframes: frame campionati uniformemente dal video (come l'eval).
        max_pixels: budget di pixel per frame post-resize.
        min_pixels: floor opzionale di pixel per frame (None = default
            qwen-vl-utils).
        double_frames: se True, attiva la modalità frame-doubling (vedi
            `_sample_doubled_frames`): ogni cella temporale ↔ un singolo frame.
        topk: quanti frame estrarre come PNG.
        prompt: se fornito, usato tale e quale come testo (caso video locale);
            altrimenti si formatta da `sample["question"]/["options"]`.
        outdir: cartella dove scrivere il dump `.pt` e i frame estratti.
    """
    import shutil
    import tempfile

    import torch

    from evals.base import format_mcq_prompt
    from models import Text, Video, VideoFrames
    from utils.attn_frames import extract_attended_frames

    if not hasattr(vlm, "entity_visual_attention"):
        raise TypeError(
            f"{type(vlm).__name__} non espone entity_visual_attention: "
            "lancia con model qwen25_vl_3b_attn"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    video_path = sample["video_path"]
    prompt_text = prompt if prompt is not None else format_mcq_prompt(
        sample["question"], sample["options"]
    )
    text = Text(prompt_text)

    # Costruzione media: video normale (merge a coppie) o frame raddoppiati.
    tmpdir: Path | None = None
    if double_frames:
        tmpdir = Path(tempfile.mkdtemp(prefix="qwen_double_"))
        doubled, frame_indices = _sample_doubled_frames(
            video_path, nframes, tmpdir,
        )
        media = VideoFrames(
            doubled,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        logger.info(
            "DEBUG frame-doubling | %d frame campionati e duplicati (%d in input)",
            len(frame_indices), len(doubled),
        )
    else:
        media = Video(
            video_path,
            nframes=nframes,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        # Stesso campionamento di qwen-vl-utils, per mappare le coppie ai frame.
        import decord

        vr = decord.VideoReader(video_path)
        frame_indices = torch.linspace(0, len(vr) - 1, nframes).round().long().tolist()

    logger.info("DEBUG entity-attention | video=%s | entity=%r", video_path, entity)
    logger.info("Prompt:\n%s", text.text)

    try:
        out = vlm.entity_visual_attention(media, text, entity)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    heatmap = out["heatmap"]
    visual_scores = out["visual_scores"]
    logger.info("entity_span=%s visual_span=%s", out["entity_span"], out["visual_span"])
    logger.info("heatmap shape=%s (t, grid_h, grid_w)", tuple(heatmap.shape))
    logger.info(
        "visual_scores: n=%d sum=%.4f (massa di attenzione sull'immagine, < 1)",
        visual_scores.numel(), visual_scores.sum().item(),
    )
    flat = heatmap.flatten()
    vals, idx = flat.topk(min(5, flat.numel()))
    logger.info(
        "top-5 celle visive: idx=%s vals=%s",
        idx.tolist(), [round(v, 6) for v in vals.tolist()],
    )

    # Arricchisci il dump con quanto serve all'estrazione frame e salvalo.
    out = {
        **out,
        "entity": entity,
        "video_path": video_path,
        "frame_indices": frame_indices,
        "double_frames": double_frames,
    }
    safe = entity.replace("/", "_").replace(" ", "_")
    dump_path = outdir / f"debug_{safe}.pt"
    torch.save(out, dump_path)
    logger.info("Dump attenzione salvato in %s", dump_path)

    # Estrazione frame + overlay (la mappatura cella→frame si adatta da sola
    # al regime `double_frames` letto dal dump).
    extract_attended_frames(
        out, video_path,
        topk=topk, metric="sum",
        outdir=outdir / "attended_frames",
        overlay=True,
    )
    return out
