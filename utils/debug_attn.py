"""Debug del path di cattura attenzione entity→token visivi.

Esegue `Qwen25VLAttention.entity_visual_attention` su UN singolo sample,
saltando l'eval Weave: serve a validare su GPU il wiring della custom
attention (dispatch sul registry, shape degli stash `_last_attn`, sanity
della heatmap) senza lanciare l'intero benchmark. Attivato da `main.run`
quando il knob top-level `debug_entity` è valorizzato.
"""
import logging

from omegaconf import DictConfig

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


def run_entity_attention_debug(
    vlm,
    sample: dict,
    dataset_cfg: DictConfig,
    entity: str,
) -> dict:
    """Lancia `entity_visual_attention` su `sample` e logga un riepilogo.

    Ricostruisce lo stesso `(media, text)` che `EgoSchema.predict_factory`
    passerebbe a `generate` (stesso `Video` + prompt MCQ), così l'attenzione
    catturata è quella delle reali condizioni di prefill dell'eval.

    Args:
        vlm: deve essere un `Qwen25VLAttention` (espone
            `entity_visual_attention`); con gli altri modelli solleva.
        sample: una riga del loader egoschema (video_path, question, options).
        dataset_cfg: `cfg.dataset` (nframes, max_pixels, min_pixels).
        entity: la stringa di cui estrarre le query — DEVE comparire nel
            prompt, altrimenti `_find_entity_span` solleva `ValueError`.
    """
    from evals.base import format_mcq_prompt
    from models import Text, Video

    if not hasattr(vlm, "entity_visual_attention"):
        raise TypeError(
            f"{type(vlm).__name__} non espone entity_visual_attention: "
            "lancia con model=qwen2_5_vl_3b_attn"
        )

    media = Video(
        sample["video_path"],
        nframes=dataset_cfg.nframes,
        max_pixels=dataset_cfg.max_pixels,
        min_pixels=dataset_cfg.get("min_pixels"),
    )
    text = Text(format_mcq_prompt(sample["question"], sample["options"]))

    logger.info("DEBUG entity-attention | video=%s | entity=%r", sample["video_path"], entity)
    logger.info("Prompt:\n%s", text.text)

    out = vlm.entity_visual_attention(media, text, entity)

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
    return out
