import torch
import torch.nn.functional as F
from transformers import AttentionInterface, GenerationConfig
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import AttentionMaskInterface, eager_mask

from utils.attn_core import EntityAttention, QueryToken, VisualAttention, mcq_answer_stats, top_k_logits

from .media import MediaItem, Text
from .qwen import Qwen25VL3B, Qwen3VL2B, Qwen3VL4B
from .signals import SignalAnswer, SupportsSignals


# Indici di canale "outlier" per modello: i token visivi-sink hanno hidden
# state con valori estremi in queste poche dimensioni. Ricopiati da
# `lot/retrieval.py:64-79` (metodo "sink-dims" di Xiao et al.), che li ha
# ricavati modello per modello — inclusi i Qwen3-VL.
SINK_DIMS: dict[str, tuple[int, ...]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": (318, 1874, 1819),
    "Qwen/Qwen2.5-VL-7B-Instruct": (458, 2570),
    "Qwen/Qwen2.5-VL-32B-Instruct": (4675, 3094),
    "Qwen/Qwen2-VL-2B-Instruct": (1073, 534, 940),
    "Qwen/Qwen2-VL-7B-Instruct": (2570, 458),
    # Qwen3-VL: il 4B ne ha UNO SOLO (canale 0) — se la `sink_map` risultasse
    # quasi costante, il filtro al 25° percentile diventerebbe una soglia
    # arbitraria invece di isolare una minoranza di token: da controllare al
    # primo forward reale (vedi `docs/qwen3vl_signals.md`).
    "Qwen/Qwen3-VL-2B-Instruct": (1793, 1999, 1401),
    "Qwen/Qwen3-VL-4B-Instruct": (0,),
    "Qwen/Qwen3-VL-8B-Instruct": (1838,),
}


def sink_dims_for(model_id: str) -> tuple[int, ...]:
    """Sink dims tabulati per `model_id`, o `KeyError` esplicito.

    Niente fallback silenzioso su un altro modello: i canali outlier sono una
    proprietà dei pesi: usare quelli del 3B su un'altra famiglia produrrebbe
    una `sink_map` di puro rumore, e ogni segnale che dipende dal filtro sink
    (`top1_pct`, ramo zoom, cella evidenziata) sarebbe silenziosamente falso.
    """
    try:
        return SINK_DIMS[model_id]
    except KeyError:
        raise KeyError(
            f"sink dims non tabulati per {model_id!r}. Aggiungili a "
            "`SINK_DIMS` in models/qwen_attn.py (riferimento: la tabella "
            "omonima in lot/retrieval.py) — nessun default: i canali di un "
            "altro modello darebbero una sink_map priva di significato."
        ) from None


# Cosa cattura il prossimo forward (impostato da `full_visual_attention`):
#   QUERY_SPAN = [q_lo, q_hi)  righe-query da conservare (= testo della domanda)
#   VIS_INDEX  = indici delle colonne-key visive da conservare (tensore, già
#                sul device del modello)
# Restringere le righe al solo testo è ciò che rende la cattura economica:
# evita di materializzare la matrice [S, S] piena (S~16k coi token visivi).
#
# Le colonne sono un ELENCO DI INDICI e non uno span perché su Qwen3-VL il
# blocco visivo non è contiguo: il processor interleava un `<x.x seconds>`
# testuale prima di ogni frame, quindi i token visivi sono `t` isole separate
# (vedi `QwenAttentionCapture._capture_spans`). Su Qwen2.5-VL gli indici sono
# semplicemente contigui e il risultato è identico allo slice di prima.
QUERY_SPAN: tuple[int, int] | None = None
VIS_INDEX: torch.Tensor | None = None


def set_capture_spans(
    query_span: tuple[int, int] | None,
    vis_index: torch.Tensor | None = None,
) -> None:
    """Imposta righe-query e colonne visive per il prossimo forward."""
    global QUERY_SPAN, VIS_INDEX
    QUERY_SPAN, VIS_INDEX = query_span, vis_index


def qwen_attn_capture(
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
    SOLE colonne visive in `VIS_INDEX`, fa la media sulle teste e stasha
    `[n_q, n_vis]` (cpu) in `module._last_attn`. Entity-agnostico: conserva una
    riga per OGNI token della domanda, non per una singola entity.

    Indipendente dalla versione: la firma dell'attention interface è la stessa
    per Qwen2.5-VL e Qwen3-VL, e su Qwen3 `q_norm`/`k_norm` sono già applicate
    dal chiamante (`modeling_qwen3_vl.py:478-480`), quindi il matmul qui sotto
    ricalcola gli stessi pesi che userebbe il kernel.
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

    if QUERY_SPAN is not None and VIS_INDEX is not None:
        q_lo, q_hi = QUERY_SPAN
        query_rows = query[:, :, q_lo:q_hi, :]
        attn_weights = torch.matmul(query_rows, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, q_lo:q_hi, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # colonne visive + media sulle teste → [n_q, n_vis] (batch=1).
        vis = attn_weights.index_select(-1, VIS_INDEX.to(attn_weights.device)).mean(dim=1)
        module._last_attn = vis[0].detach().cpu()

    return attn_output, None


AttentionInterface.register("qwen_attn_capture", qwen_attn_capture)
AttentionMaskInterface.register("qwen_attn_capture", eager_mask)
# Alias storico: è il nome che compare nel preset `qwen2_5_vl_3b_attn`, negli
# sbatch e nei config snapshottati delle run già fatte. Stessa funzione, che
# nel frattempo è diventata family-agnostica.
AttentionInterface.register("qwen25_attn_capture", qwen_attn_capture)
AttentionMaskInterface.register("qwen25_attn_capture", eager_mask)


class QwenAttentionCapture(SupportsSignals):
    """Cattura dell'attenzione domanda→token visivi, comune a Qwen2.5/Qwen3-VL.

    Mixin: va combinato con una sottoclasse concreta di `Qwen` (che porta
    `model_id`, `_load`, `_prepare_inputs`, `generate`) — vedi
    `Qwen25VLAttention`/`Qwen3VL2BAttention`/`Qwen3VL4BAttention` in fondo al
    modulo.

    L'interface `qwen_attn_capture` (registrata sopra, attivata via
    `attn_implementation.text_config` nello yaml — scoped al solo decoder
    testuale) calcola `softmax(q·kᵀ)` per le righe-query del testo della domanda
    e stasha i pesi sui token visivi per layer in `self_attn._last_attn`.
    `full_visual_attention` orchestra il giro completo: span domanda/visivo →
    forward di prefill → media sui layer centrali → mappa 2D per ogni token
    della domanda. `entity_visual_attention` è un wrapper che seleziona le righe
    di una specifica entity e le media. `generate_with_signals` (contratto
    `SupportsSignals`, `models/signals.py`) è il punto d'ingresso usato dalle
    strategy signal-driven (es. `strategies.entropy_shortcut`): avvolge
    `full_visual_attention` senza duplicarne la logica.

    Nulla qui è specifico di una versione: griglia (`spatial_merge_size=2`,
    `temporal_patch_size=2` → una cella = 2 frame in entrambe le famiglie),
    path dei layer (`model.model.language_model.layers`) e `logits_to_keep=1`
    coincidono. L'unico punto in cui le due famiglie divergono è
    `_capture_spans`, che le gestisce entrambe con lo stesso codice.
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

    def _capture_spans(self, input_ids: torch.Tensor) -> tuple[tuple[int, int], torch.Tensor]:
        """`(query_span, vis_index)` per un prompt già tokenizzato (1D).

        - **colonne visive** = tutte le posizioni dei placeholder visivi
          (`image_token_id`/`video_token_id`, cioè `<|image_pad|>`/
          `<|video_pad|>`), non lo spazio fra il primo `vision_start` e il
          primo `vision_end`. Su Qwen2.5-VL le due definizioni coincidono (fra
          start ed end c'è solo il pad ripetuto, quindi la cattura resta
          identica a prima); su Qwen3-VL no: il processor emette
          `<x.x seconds><|vision_start|>frame<|vision_end|>` PER OGNI frame
          (`processing_qwen3_vl.py:157-170`), quindi ci sono `t` coppie
          start/end e i token visivi sono `t` isole separate da testo.
        - **righe-query** = tutto ciò che sta dopo l'ULTIMO `vision_end`, cioè
          il testo della domanda (più il generation prompt). Su Qwen2.5-VL
          "ultimo" e "primo" sono lo stesso token; su Qwen3-VL prendere il
          primo significherebbe trattare tutto il video tranne il frame 0
          come se fosse la domanda.

        `vis_index` è in ordine di sequenza, cioè frame-major: è ciò che rende
        valido il `reshape(t, grid_h, grid_w)` a valle.
        """
        cfg = self.model.config
        visual_ids = torch.tensor(
            [cfg.image_token_id, cfg.video_token_id], device=input_ids.device,
        )
        vis_index = torch.isin(input_ids, visual_ids).nonzero(as_tuple=True)[0]
        if vis_index.numel() == 0:
            raise ValueError(
                "nessun token visivo nel prompt: `full_visual_attention` "
                "richiede un media (immagine o video) nei messaggi."
            )

        vision_end = (input_ids == cfg.vision_end_token_id).nonzero(as_tuple=True)[0]
        if vision_end.numel() == 0:
            raise ValueError("nessun <|vision_end|> nel prompt — layout inatteso")
        query_span = (int(vision_end[-1].item()) + 1, int(input_ids.shape[0]))
        if query_span[0] >= query_span[1]:
            raise ValueError(
                "nessun token dopo l'ultimo <|vision_end|>: il testo della "
                "domanda deve seguire il media (vedi `Qwen.build_messages`)."
            )
        return query_span, vis_index

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
        si filtrano azzerandoli. Richiede il text_config custom (la cattura
        dei pesi), non modifiche al vision encoder. Inspired by
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

        cfg = self.model.config
        query_span, vis_index = self._capture_spans(input_ids)
        # Estremi del blocco visivo, a scopo informativo: su Qwen3-VL le
        # colonne NON sono contigue fra questi due indici (vedi
        # `_capture_spans`), quindi non va usato per fare slicing — nessuno lo
        # fa, viene solo riportato nei DTO.
        visual_span = (int(vis_index[0].item()), int(vis_index[-1].item()) + 1)

        sink_dims = sink_dims_for(self.model_id)
        sink_dims_t = torch.tensor(sink_dims, dtype=torch.long)

        layers = self.model.model.language_model.layers
        n_layers = len(layers)

        # Forward hook su ogni layer (output hidden states) per calcolare il
        # sink score dei token visivi. Stasha [n_vis] su CPU in `layer._sink`.
        # Simmetrico a `_last_attn` dell'attention interface, separato per
        # tenere pulito il path di capture.
        sink_per_layer: dict[int, torch.Tensor] = {}

        def make_sink_hook(layer_idx: int, columns: torch.Tensor, dims: torch.Tensor):
            def _hook(_module, _inputs, output):
                # output è un tuple; il primo è l'hidden state.
                hs = output[0] if isinstance(output, tuple) else output
                vis_hidden = hs[0].index_select(0, columns.to(hs.device))  # [n_vis, D]
                sink_vals = vis_hidden[:, dims.to(hs.device)]  # [n_vis, n_sink_dims]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                score = max_sink_val / (rms + 1e-6)  # [n_vis]
                sink_per_layer[layer_idx] = score.detach().cpu().float()
            return _hook

        hooks = [
            layers[i].register_forward_hook(make_sink_hook(i, vis_index, sink_dims_t))
            for i in range(n_layers)
        ]

        set_capture_spans(query_span, vis_index)
        try:
            with torch.no_grad():
                # use_cache=False: questo è un prefill isolato (nessuna
                # generazione autoregressiva segue in questa chiamata), la
                # KV cache costruita da un forward normale verrebbe scartata
                # subito insieme a `outputs` — inutile tenerla in VRAM. A
                # nframes alti (es. Video-MME 128) risparmia memoria
                # sufficiente a evitare OOM sulle GPU da 24G.
                outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
            set_capture_spans(None, None)
            # Le strategy signal-driven (`entropy_shortcut`, `entropy_
            # attention_resample`) fanno questo forward PRIMA di un secondo
            # pass (`generate()`) nello stesso processo/contesto CUDA: senza
            # svuotare la cache dell'allocator qui, un OOM su un sample
            # lascia memoria "reserved but unallocated" che non torna
            # disponibile per i sample successivi (osservato: fallimento a
            # cascata su tutti i sample dopo il primo OOM).
            torch.cuda.empty_cache()

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

    def generate_with_signals(
        self,
        media: MediaItem,
        text: Text,
        gen_cfg: GenerationConfig,
        *,
        answer_letters: list[str] | None = None,
    ) -> SignalAnswer:
        """Implementa `SupportsSignals` avvolgendo `full_visual_attention`.

        Prefill-only (nessun `model.generate()`): `SignalAnswer.text` resta
        `None`, `pred_letter` viene dalla softmax ristretta alle lettere
        candidate sui logit dell'ultimo token del prefill — sufficiente per
        rispondere a un MCQ senza generazione autoregressiva. `gen_cfg` non
        è usato (nessun campionamento avviene in questo path); resta nella
        firma per uniformità col contratto `SupportsSignals`.
        """
        va = self.full_visual_attention(media, text, answer_letters=answer_letters)
        return SignalAnswer(
            text=None,
            pred_letter=va.pred_letter,
            answer_entropy=va.answer_entropy,
            answer_probs=va.answer_probs,
            visual_attention=va,
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


# Varianti "con segnali" dei modelli di `models/qwen.py`: aggiungono solo la
# cattura (il mixin), il resto — pesi, processor, preprocessing — è quello
# della classe base. Vanno selezionate con il preset yaml corrispondente, che
# è ciò che attiva davvero l'interface (`attn_implementation.text_config`):
# senza, `_last_attn` non verrebbe mai popolato.
class Qwen25VLAttention(Qwen25VL3B, QwenAttentionCapture):
    """Qwen2.5-VL-3B con cattura dell'attenzione (preset `qwen2_5_vl_3b_attn`)."""


class Qwen3VL2BAttention(Qwen3VL2B, QwenAttentionCapture):
    """Qwen3-VL-2B con cattura dell'attenzione (preset `qwen3_vl_2b_attn`)."""


class Qwen3VL4BAttention(Qwen3VL4B, QwenAttentionCapture):
    """Qwen3-VL-4B con cattura dell'attenzione (preset `qwen3_vl_4b_attn`)."""
