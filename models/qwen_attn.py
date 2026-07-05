import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

from utils.attn_core import EntityAttention, QueryToken, VisualAttention, mcq_answer_stats, top_k_logits

from .media import MediaItem, Text
from .qwen import Qwen25VL3B


# Indici di canale "outlier" per modello: i token visivi-sink hanno hidden
# state con valori estremi in queste poche dimensioni. Ricalcati da
# `lot/retrieval.py` (metodo "sink-dims" di Xiao et al.). Solo Qwen2.5-VL-3B
# è rilevante qui (oddio `Qwen25VLAttention` subclassa `Qwen25VL3B`).
SINK_DIMS: dict[str, tuple[int, ...]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": (318, 1874, 1819),
    "Qwen/Qwen2.5-VL-7B-Instruct": (458, 2570),
    "Qwen/Qwen2.5-VL-32B-Instruct": (4675, 3094),
    "Qwen/Qwen2-VL-2B-Instruct": (1073, 534, 940),
    "Qwen/Qwen2-VL-7B-Instruct": (2570, 458),
}
# Default se il model_id non è tabulato (riesce comunque: i sink score
# saranno solo approssimati).
SINK_DIMS_DEFAULT = SINK_DIMS["Qwen/Qwen2.5-VL-3B-Instruct"]


# Span catturati dal prossimo forward (impostati da `full_visual_attention`):
#   QUERY_SPAN = [q_lo, q_hi)  righe-query da conservare (= testo della domanda)
#   VIS_SPAN   = [v_lo, v_hi)  colonne-key visive da conservare
# Restringere le righe al solo testo è ciò che rende la cattura economica:
# evita di materializzare la matrice [S, S] piena (S~16k coi token visivi).
QUERY_SPAN: tuple[int, int] | None = None
VIS_SPAN: tuple[int, int] | None = None


def set_capture_spans(
    query_span: tuple[int, int] | None,
    vis_span: tuple[int, int] | None = None,
) -> None:
    """Imposta gli span (righe-query, colonne-visive) per il prossimo forward."""
    global QUERY_SPAN, VIS_SPAN
    QUERY_SPAN, VIS_SPAN = query_span, vis_span


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
    """Attention interface che, oltre all'output SDPA, stasha i pesi softmax.

    Cattura `softmax(q·kᵀ)` per le SOLE righe-query in `QUERY_SPAN`, ne tiene le
    SOLE colonne visive in `VIS_SPAN`, fa la media sulle teste e stasha
    `[n_q, n_vis]` (cpu) in `module._last_attn`. Entity-agnostico: conserva una
    riga per OGNI token della domanda, non per una singola entity.
    """
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

    if QUERY_SPAN is not None and VIS_SPAN is not None:
        q_lo, q_hi = QUERY_SPAN
        v_lo, v_hi = VIS_SPAN
        query_rows = query[:, :, q_lo:q_hi, :]
        attn_weights = torch.matmul(query_rows, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, q_lo:q_hi, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # slice colonne visive + media sulle teste → [n_q, n_vis] (batch=1).
        vis = attn_weights[:, :, :, v_lo:v_hi].mean(dim=1)
        module._last_attn = vis[0].detach().cpu()

    return attn_output, None


AttentionInterface.register("qwen25_attn_capture", qwen25_attn_capture)
AttentionMaskInterface.register("qwen25_attn_capture", eager_mask)


class Qwen25VLAttention(Qwen25VL3B):
    """Qwen2.5-VL con attention interface custom per analisi dell'attenzione.

    L'interface `qwen25_attn_capture` (registrata sopra, attivata via
    `attn_implementation.text_config` nello yaml — scoped al solo decoder
    testuale) calcola `softmax(q·kᵀ)` per le righe-query del testo della domanda
    e stasha i pesi sui token visivi per layer in `self_attn._last_attn`.
    `full_visual_attention` orchestra il giro completo: span domanda/visivo →
    forward di prefill → media sui layer centrali → mappa 2D per ogni token
    della domanda. `entity_visual_attention` è un wrapper che seleziona le righe
    di una specifica entity e le media.
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

    def full_visual_attention(
        self,
        media: MediaItem,
        text: Text,
        layer_range: tuple[int, int] | None = None,
        answer_letters: list[str] | None = None,
    ) -> VisualAttention:
        """Attenzione di OGNI token della domanda verso i token visivi + sink map.

        Esegue un singolo forward di prefill (niente generate: l'attenzione
        testo→immagine esiste già lì), media i pesi catturati sui layer
        centrali e le teste, e li dispone su griglia per ogni token-query.
        Inoltre cattura gli hidden states dei token visivi via forward hook
        sui layer e ne deriva `sink_map`: per ogni token visivo, `max` dei
        sink dims normalizzato RMS, mediato sui layer centrali. I token con
        sink score alto sono "sink" (assorbono attenzione spuria) — a runtime
        si filtrano azzerandoli. Solo per `Qwen25VLAttention` (text_config
        custom), ma non richiede modifiche al vision encoder. Inspired by
        `lot/retrieval.py` ("sink-dims" di Xiao et al.).

        Args:
            media: l'immagine/video della domanda.
            text: il testo della domanda.
            layer_range: [lo, hi) dei layer da mediare; default la metà
                centrale (la più semantica).
            answer_letters: lettere MCQ candidate (es. `["A","B","C","D"]`,
                da `evals.base.extract_mcq_letters` sul prompt) per cui
                calcolare l'entropia della risposta. Il forward di prefill
                già gira con `logits_to_keep=1` (logit dell'ultimo token,
                cioè il primo che genererebbe) — riusarli qui è a costo
                zero. `None`/`[]` (prompt non-MCQ) salta il calcolo.

        Returns:
            `VisualAttention`: `attn` `[n_q, t, grid_h, grid_w]` (heatmap per
            token), `sink_map` `[t, grid_h, grid_w]` (sink score per token
            visivo), `sink_dims` (canali usati), `query_tokens` selezionabili
            a runtime, gli span e gli `input_ids`.
        """
        inputs = self._prepare_inputs(self.build_messages(media, text))
        input_ids = inputs.input_ids[0]
        seq_len = input_ids.shape[0]

        cfg = self.model.config
        vis_start = (input_ids == cfg.vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
        vis_end = (input_ids == cfg.vision_end_token_id).nonzero(as_tuple=True)[0][0].item()

        # Righe-query = testo DOPO il video (la domanda); colonne = token visivi.
        query_span = (vis_end + 1, seq_len)
        visual_span = (vis_start + 1, vis_end)

        # Sink dims per questo model_id (o default se non tabulati).
        sink_dims = SINK_DIMS.get(self.model_id, SINK_DIMS_DEFAULT)
        sink_dims_t = torch.tensor(sink_dims, dtype=torch.long)

        layers = self.model.model.language_model.layers
        n_layers = len(layers)

        # Forward hook su ogni layer (output hidden states) per calcolare il
        # sink score dei token visivi. Stasha [n_vis] su CPU in `layer._sink`.
        # Simmetrico a `_last_attn` dell'attention interface, separato per
        # tenere pulito il path di capture.
        sink_per_layer: dict[int, torch.Tensor] = {}

        def make_sink_hook(layer_idx: int, v_lo: int, v_hi: int, dims: torch.Tensor):
            def _hook(_module, _inputs, output):
                # output è un tuple; il primo è l'hidden state.
                hs = output[0] if isinstance(output, tuple) else output
                vis_hidden = hs[0, v_lo:v_hi, :]  # [n_vis, D]
                sink_vals = vis_hidden[:, dims]    # [n_vis, n_sink_dims]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                score = max_sink_val / (rms + 1e-6)  # [n_vis]
                sink_per_layer[layer_idx] = score.detach().cpu().float()
            return _hook

        hooks = [
            layers[i].register_forward_hook(make_sink_hook(i, *visual_span, sink_dims_t))
            for i in range(n_layers)
        ]

        set_capture_spans(query_span, visual_span)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
            set_capture_spans(None, None)

        # [L, n_q, n_vis]: pesi (già mediati sulle teste) per layer.
        attn = torch.stack([layer.self_attn._last_attn for layer in layers])

        if layer_range is None:
            layer_range = (attn.shape[0] // 4, 3 * attn.shape[0] // 4)
        lo, hi = layer_range
        attn = attn[lo:hi].mean(dim=0)  # [n_q, n_vis]

        # Sink score: media sui layer centrali (stesso range di attn).
        sorted_sink = sorted(sink_per_layer.keys())
        sink_scores = torch.stack([
            sink_per_layer[i] for i in sorted_sink if lo <= i < hi
        ]).mean(dim=0)  # [n_vis]

        # mappa 1D → griglia 2D: grid_thw è in patch PRE-merger, la griglia
        # dei token è divisa per spatial_merge_size (2) su h e w.
        grid_thw = inputs.get("image_grid_thw", inputs.get("video_grid_thw"))
        t, h, w = grid_thw[0].tolist()
        merge = cfg.vision_config.spatial_merge_size
        grid_h, grid_w = h // merge, w // merge
        n_vis = t * grid_h * grid_w
        if attn.shape[1] != n_vis:
            raise ValueError(
                f"token visivi ({attn.shape[1]}) != t*gh*gw "
                f"({t}*{grid_h}*{grid_w}) — layout grid_thw inatteso"
            )
        if sink_scores.shape[0] != n_vis:
            raise ValueError(
                f"sink scores ({sink_scores.shape[0]}) != n_vis ({n_vis}) "
                "— span visivo hook vs att catturati diversamente?"
            )
        heatmaps = attn.reshape(attn.shape[0], t, grid_h, grid_w).cpu()
        sink_map = sink_scores.reshape(t, grid_h, grid_w).cpu()

        # Token della domanda → metadati selezionabili a runtime.
        tok = self.processor.tokenizer
        q_lo, q_hi = query_span
        q_ids = input_ids[q_lo:q_hi].tolist()
        raw_tokens = tok.convert_ids_to_tokens(q_ids)
        query_tokens = tuple(
            QueryToken(row=row, index=q_lo + row, token=raw, text=tok.decode([tid]))
            for row, (raw, tid) in enumerate(zip(raw_tokens, q_ids))
        )

        # Entropia della risposta + top-k non ristretto: riusano i logit
        # dell'ultimo token del prefill già calcolati sopra
        # (`logits_to_keep=1`), nessun forward aggiuntivo.
        last_logits = outputs.logits[0, -1, :].cpu()
        answer_stats: dict = {}
        if answer_letters:
            letter_ids = {l: tok.encode(l, add_special_tokens=False)[0] for l in answer_letters}
            answer_stats = mcq_answer_stats(last_logits, letter_ids)
        top_tokens = top_k_logits(last_logits, tok, k=10)

        return VisualAttention(
            attn=heatmaps,
            sink_map=sink_map,
            sink_dims=sink_dims,
            t=t,
            grid_h=grid_h,
            grid_w=grid_w,
            query_tokens=query_tokens,
            query_span=query_span,
            **answer_stats,
            top_tokens=top_tokens,
            visual_span=visual_span,
            input_ids=input_ids.cpu(),
        )

    def entity_visual_attention(
        self,
        media: MediaItem,
        text: Text,
        entity: str,
        layer_range: tuple[int, int] | None = None,
    ) -> EntityAttention:
        """Attenzione delle SOLE query di `entity` verso i token visivi.

        Wrapper sottile su `full_visual_attention`: localizza i token
        dell'entity nel testo della domanda e media le loro righe.

        Returns:
            `EntityAttention` con `heatmap` `[t, grid_h, grid_w]`, le `rows`
            mediate, gli span e i `query_tokens` di passaggio.
        """
        out = self.full_visual_attention(media, text, layer_range)
        q_lo, q_hi = out.query_span
        span = self._find_entity_span(out.input_ids, entity)
        rows = tuple(i - q_lo for i in range(*span) if q_lo <= i < q_hi)
        if not rows:
            raise ValueError(
                f"entity {entity!r} non è nel testo della domanda (dopo il video)"
            )
        return EntityAttention(
            heatmap=out.attn[list(rows)].mean(dim=0),
            rows=rows,
            entity_span=span,
            visual_span=out.visual_span,
            t=out.t,
            grid_h=out.grid_h,
            grid_w=out.grid_w,
            query_tokens=out.query_tokens,
        )
