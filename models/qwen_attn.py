import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

from .media import MediaItem, Text
from .qwen import Qwen25VL3B


ENTITY_SPAN: tuple[int, int] | None = None


def set_entity_span(span: tuple[int, int] | None) -> None:
    """Imposta lo span [start, end) dei token dell'entity per il prossimo forward.
    """
    global ENTITY_SPAN
    ENTITY_SPAN = span


def qwen25_attn_capture(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        query.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=is_causal if is_causal is not None else attention_mask is None,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    start, end = ENTITY_SPAN if ENTITY_SPAN is not None else (None, None)
    query_entity = query[:, :, start:end, :]
    attn_weights = torch.matmul(query_entity, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, start:end, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    module._last_attn = attn_weights.detach()

    return attn_output, attn_weights


AttentionInterface.register("qwen25_attn_capture", qwen25_attn_capture)
AttentionMaskInterface.register("qwen25_attn_capture", eager_mask)


class Qwen25VLAttention(Qwen25VL3B):
    """Qwen2.5-VL con attention interface custom per analisi dell'attenzione.

    L'interface `qwen25_attn_capture` (registrata sopra, attivata via
    `attn_implementation.text_config` nello yaml — scoped al solo decoder
    testuale) calcola softmax(q·kᵀ) per le sole query dell'entity e stasha i
    pesi per layer in `self_attn._last_attn`. `entity_visual_attention`
    orchestra il giro completo: span dell'entity → forward di prefill →
    raccolta dei pesi → mappa 2D sui token visivi via grid_thw.
    """

    def _find_entity_span(self, input_ids: torch.Tensor, entity: str) -> tuple[int, int]:
        """Trova lo span [start, end) dei token di `entity` in `input_ids` (1D).

        La tokenizzazione dipende dal contesto: la stessa parola produce id
        diversi con o senza spazio iniziale, quindi proviamo entrambe le
        varianti e cerchiamo la sottosequenza esatta.
        """
        seq = input_ids.tolist()
        tokenizer = self.processor.tokenizer
        for variant in (" " + entity, entity):
            ent_ids = tokenizer.encode(variant, add_special_tokens=False)
            n = len(ent_ids)
            for i in range(len(seq) - n + 1):
                if seq[i : i + n] == ent_ids:
                    return i, i + n
        raise ValueError(f"entity {entity!r} non trovata nei token del prompt")

    def entity_visual_attention(
        self,
        media: MediaItem,
        text: Text,
        entity: str,
        layer_range: tuple[int, int] | None = None,
    ) -> dict:
        """Attenzione dei token dell'entity verso i token visivi.

        Esegue un singolo forward di prefill (niente generate: l'attenzione
        entity→immagine esiste già lì) e aggrega i pesi catturati per layer.

        Args:
            media: l'immagine/video della domanda.
            text: il testo della domanda (deve contenere `entity`).
            entity: la stringa dell'entity di cui analizzare le query.
            layer_range: [lo, hi) dei layer da aggregare; default la metà
                centrale (la più semantica).

        Returns:
            dict con:
            - heatmap: [t, grid_h, grid_w] float32 cpu, score per cella visiva;
            - visual_scores: [n_vis] gli stessi score, flat;
            - seq_scores: [S] score aggregati su tutta la sequenza;
            - entity_span / visual_span: gli span [start, end) nei token.
        """
        inputs = self._prepare_inputs(self.build_messages(media, text))
        input_ids = inputs.input_ids[0]

        span = self._find_entity_span(input_ids, entity)
        set_entity_span(span)
        try:
            with torch.no_grad():
                self.model(**inputs, logits_to_keep=1)
        finally:
            set_entity_span(None)

        layers = self.model.model.language_model.layers
        # [L, H, n_ent, S]: un blocco di pesi per layer (batch=1 rimosso).
        attn: torch.Tensor = torch.stack([layer.self_attn._last_attn[0] for layer in layers])

        if layer_range is None:
            layer_range = (attn.shape[0] // 4, 3 * attn.shape[0] // 4)
        lo, hi = layer_range
        # media su layer scelti, teste e token dell'entity → [S]
        seq_scores = attn[lo:hi].float().mean(dim=(0, 1, 2))

        cfg = self.model.config
        vis_start = (input_ids == cfg.vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
        vis_end = (input_ids == cfg.vision_end_token_id).nonzero(as_tuple=True)[0][0].item()
        visual_scores = seq_scores[vis_start + 1 : vis_end]

        # mappa 1D → griglia 2D: grid_thw è in patch PRE-merger, la griglia
        # dei token è divisa per spatial_merge_size (2) su h e w.
        grid_thw = inputs.get("image_grid_thw", inputs.get("video_grid_thw"))
        t, h, w = grid_thw[0].tolist()
        merge = cfg.vision_config.spatial_merge_size
        grid_h, grid_w = h // merge, w // merge
        if visual_scores.numel() != t * grid_h * grid_w:
            raise ValueError(
                f"token visivi ({visual_scores.numel()}) != t*gh*gw "
                f"({t}*{grid_h}*{grid_w}) — layout grid_thw inatteso"
            )
        heatmap = visual_scores.reshape(t, grid_h, grid_w).cpu() #verificare che la reshape sia corretta

        return {
            "heatmap": heatmap,
            "visual_scores": visual_scores.cpu(),
            "seq_scores": seq_scores.cpu(),
            "entity_span": span,
            "visual_span": (vis_start + 1, vis_end),
        }
