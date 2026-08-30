import logging

import torch
from transformers import (
    BatchFeature,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from qwen_vl_utils import process_vision_info
from .base import BaseVLM
from .media import MediaItem, Text, VideoFrames, to_content_dict
import weave

logger = logging.getLogger(__name__)

# Chiave PRIVATA con cui `build_messages_from_parts` fa viaggiare i metadata
# reali di un blocco `VideoFrames` (`frames_indices` assoluti + `fps` del
# sorgente) dentro il content dict, da `build_messages_from_parts` fino a
# `_prepare_inputs` — che la RIMUOVE prima di `apply_chat_template` e
# `process_vision_info` (il template e qwen-vl-utils non devono vederla) e la
# converte in un `transformers.video_utils.VideoMetadata` per blocco.
_VIDEO_METADATA_KEY = "_video_metadata"

# System prompt dell'estrazione entity (knob `query_rows=entity`, risolto da
# `utils.attn_core.resolve_query_rows`). Derivato dal probe
# `scripts/probe_entity_extraction.py`: soggetto/i + vincolo VERBATIM, perché
# il mapping a valle (`utils.attn_core.entity_rows`) è uno string-match esatto
# sul testo della domanda — una parafrasi non mappa e fa scattare il fallback.
# Le opzioni sono nel contesto (aiutano a disambiguare la domanda) ma il
# prompt vieta esplicitamente di attingervi parole. Costante di modulo e non
# knob: è il testo dell'intervento, cambiarla cambia l'esperimento.
# v3 dopo il probe 93525_0 (2026-08-30, 60 sample): la v1 (system prompt,
# domanda+opzioni in input) falliva sul 2B in ~2 casi su 3 — il prior
# "MCQ → rispondi" vinceva e il modello sputava un'opzione, oppure copiava
# l'intera domanda. Da qui: (a) SOLO la domanda in input, niente opzioni —
# il grilletto del prior sparisce alla fonte (taglio a monte in
# utils/attn_core.py::resolve_query_rows); (b) divieti espliciti; (c) due
# esempi few-shot sintetici (non da Video-MME) nello stesso formato
# `Question:` del question_block reale: per un 2B il formato dimostrato
# vale più dell'istruzione.
ENTITY_EXTRACTION_SYSTEM = (
    "You extract the SUBJECT(S) of a question: the entity or entities the "
    "question is about. A subject can be a thing, a person, or an action, and "
    "can span multiple words; there may be more than one subject. "
    "Do NOT answer the question. Do NOT repeat the whole question. Copy the "
    "subject(s) EXACTLY as written in the question (same words, no synonyms, "
    "no paraphrase, no extra words). Separate multiple subjects with commas. "
    "Output only the subject words: a few words, nothing else."
)

ENTITY_EXTRACTION_EXAMPLES = (
    (
        "Question: What color is the jacket worn by the man playing the guitar?",
        "jacket, man playing the guitar",
    ),
    (
        "Question: Why does the dog run away at the beginning of the video?",
        "dog, run away",
    ),
)


class Qwen(BaseVLM):
    """Shared implementation for the Qwen-VL family (2.5 and 3.x).

    The chat schema, vision-info resolution, processor call, and decoding
    are identical across versions: subclasses only pin `model_id` and
    `model_cls` (the family-specific `*ForConditionalGeneration` class).
    """

    model_cls: type  # override in subclass: *ForConditionalGeneration
    processor_cls: type  # override in subclass: *Processor della famiglia

    # Granularità del patch visivo passata a `qwen_vl_utils.process_vision_info`
    # (arrotondamento di `smart_resize`): 14 su Qwen2.5-VL, 16 su Qwen3-VL —
    # vedi `Qwen3VL`.
    image_patch_size: int = 14

    # Se passare al processor i `video_metadata` REALI (fps e indici dei frame
    # campionati) invece di lasciarglieli inventare. Default `False` = regime
    # storico, vedi `_prepare_inputs` per cosa comporta; `True` di default su
    # Qwen3-VL, dove senza è insensato (i timestamp finiscono nel prompt).
    pass_video_metadata: bool = False

    def _load(
        self,
        torch_dtype,
        device_map,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        num_text_layers: int | None = None,
        num_vision_layers: int | None = None,
        pass_video_metadata: bool | None = None,
        **kwargs,
    ):
        """Load the Qwen-VL processor + model.

        `min_pixels` and `max_pixels` cap the per-image visual-token budget
        (forwarded to `processor_cls`); leave as `None` to keep the model's
        default range.

        `pass_video_metadata` (`None` = lascia il default della classe)
        sovrascrive l'attributo omonimo dal preset yaml — knob per-run, vedi
        `_prepare_inputs`.

        `num_text_layers` / `num_vision_layers` (entrambi `None` di default)
        sono knob per **smoke test su GPU piccola**: troncano il decoder
        testuale e/o il ViT al numero indicato di layer, prima del
        `from_pretrained`. Il caricamento usa il config troncato, gli
        weight oltre il limite sono scartati (HF stampa il warning
        "Some weights ... were not used", è atteso). NB:
        l'output del modello sarà casuale — questo serve solo a validare
        il code path (load → preprocess → generate → parse → scorer)
        senza occupare la VRAM piena, non a produrre metriche utili.
        """
        if pass_video_metadata is not None:
            # Assegnato sull'istanza (non sulla classe): `_load` è chiamato da
            # `BaseVLM.__init__`, quindi `self` esiste già ed è l'unico punto
            # in cui i knob del preset yaml sono visibili.
            self.pass_video_metadata = bool(pass_video_metadata)

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        logger.info(
            "Loading %s (dtype=%s, device_map=%s, min_pixels=%s, max_pixels=%s)",
            self.model_id, torch_dtype, device_map, min_pixels, max_pixels,
        )
        processor = self.processor_cls.from_pretrained(self.model_id, **processor_kwargs)

        model_kwargs = dict(kwargs)
        if num_text_layers is not None or num_vision_layers is not None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_id)
            if num_text_layers is not None:
                orig = config.text_config.num_hidden_layers
                config.text_config.num_hidden_layers = num_text_layers
                logger.warning(
                    "SMOKE TEST: text decoder troncato da %d a %d layer. "
                    "L'output sarà casuale (utile solo a validare il code path).",
                    orig, num_text_layers,
                )
            if num_vision_layers is not None:
                # Qwen2.5-VL usa `depth`, Qwen3-VL usa `num_hidden_layers`
                # nel vision_config: tentiamo entrambi.
                vc = config.vision_config
                if hasattr(vc, "depth"):
                    orig = vc.depth
                    vc.depth = num_vision_layers
                elif hasattr(vc, "num_hidden_layers"):
                    orig = vc.num_hidden_layers
                    vc.num_hidden_layers = num_vision_layers
                else:
                    raise AttributeError(
                        f"vision_config di {self.model_id} non ha né 'depth' "
                        "né 'num_hidden_layers' — schema sconosciuto, "
                        "estendere _load per gestire questa famiglia."
                    )
                logger.warning(
                    "SMOKE TEST: vision encoder troncato da %d a %d layer.",
                    orig, num_vision_layers,
                )
                # Qwen3-VL preleva le feature "deepstack" da layer del ViT
                # indicizzati esplicitamente (default 8/16/24): troncando il
                # ViT sotto quegli indici nessuna feature verrebbe raccolta e
                # il merger resterebbe spaiato dal decoder. Si tengono solo
                # gli indici ancora esistenti.
                deepstack = getattr(vc, "deepstack_visual_indexes", None)
                if deepstack is not None:
                    kept = [i for i in deepstack if i < num_vision_layers]
                    logger.warning(
                        "SMOKE TEST: deepstack_visual_indexes %s → %s.",
                        list(deepstack), kept,
                    )
                    vc.deepstack_visual_indexes = kept
            model_kwargs["config"] = config

        model = self.model_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **model_kwargs,
        )
        logger.info("Model loaded on device=%s", getattr(model, "device", "?"))
        return processor, model

    def build_messages(self, media: MediaItem, text: Text) -> list[dict]:
        """Wrap a `(media, text)` pair in the Qwen-VL chat schema.

        Produces a single-turn user message whose `content` is the list of
        parts the Qwen chat template expects: each `MediaItem` / `Text`
        dataclass is flattened via `to_content_dict` into the
        `{"type": ..., ...}` dict the template (and `process_vision_info`)
        keys off — e.g. `{"type": "image", "image": "..."}` or
        `{"type": "text", "text": "..."}`. `to_content_dict` droppa i
        field `None` (es. `video_start`/`video_end` opzionali su `Video`)
        per non passare `None` ai backend video di qwen-vl-utils. Media è
        prima del testo, come negli example ufficiali.

        Delega a `build_messages_from_parts`: firma e comportamento restano
        INVARIATI (i 4 call site esistenti non vanno toccati — arm già
        misurati).
        """
        return self.build_messages_from_parts([media, text])

    def build_messages_from_parts(self, parts: list[MediaItem | Text]) -> list[dict]:
        """Messaggio single-turn da una content list GIÀ ordinata di parti.

        Generalizzazione di `build_messages` a un numero arbitrario di media
        e testi interleaved (es. `[video, text, video, text, video, text]`
        per i marcatori inline di `strategies/attention_marker.py`). Ogni
        parte è flattenata via `to_content_dict`; i `VideoFrames` che
        portano metadata reali (`frames_indices`/`fps`) li vedono aggiunti
        sotto `_VIDEO_METADATA_KEY`, canale privato che `_prepare_inputs`
        consuma e rimuove prima che il dict raggiunga il chat template o
        `process_vision_info`.
        """
        content = []
        for part in parts:
            d = to_content_dict(part)
            if isinstance(part, VideoFrames) and part.frames_indices is not None:
                d[_VIDEO_METADATA_KEY] = {
                    "fps": float(part.fps),
                    "frames_indices": [int(i) for i in part.frames_indices],
                }
            content.append(d)
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_manual_video_metadata(
        messages: list[dict],
    ) -> tuple[list[dict], dict[int, "VideoMetadata"]]:
        """Estrae e RIMUOVE il canale privato `_VIDEO_METADATA_KEY`.

        Ritorna `(messages ripuliti, {indice blocco video → VideoMetadata})`.
        L'indice conta i content item `type == "video"` in ordine di
        documento — lo stesso ordine in cui `process_vision_info` estrae i
        video, quindi allineato alla lista `video_metadata` del processor.
        I messaggi ritornati sono copie shallow con i content dict privati
        della chiave: né `apply_chat_template` né `process_vision_info`
        devono vederla. Senza chiavi private i messaggi tornano invariati
        (stessi oggetti, zero costo).
        """
        from transformers.video_utils import VideoMetadata

        manual: dict[int, VideoMetadata] = {}
        clean_messages = []
        vid_i = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                clean_messages.append(msg)
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    meta = item.get(_VIDEO_METADATA_KEY)
                    if meta is not None:
                        item = {k: v for k, v in item.items() if k != _VIDEO_METADATA_KEY}
                        # `total_num_frames` serve solo al percorso di
                        # RICAMPIONAMENTO del video processor
                        # (`sample_frames`), che qui è spento
                        # (`do_sample_frames=False` nei video_kwargs del
                        # regime metadata): i timestamp usano solo
                        # `frames_indices`/`fps`. Si passa un lower bound
                        # coerente per non lasciare il campo obbligatorio a
                        # un valore di fantasia.
                        manual[vid_i] = VideoMetadata(
                            total_num_frames=int(meta["frames_indices"][-1]) + 1,
                            fps=meta["fps"],
                            frames_indices=list(meta["frames_indices"]),
                            video_backend="decord",
                        )
                    vid_i += 1
                new_content.append(item)
            clean_messages.append({**msg, "content": new_content})
        if not manual:
            return messages, {}
        return clean_messages, manual

    def _prepare_inputs(self, messages: list[dict]) -> BatchFeature:
        """Turn chat `messages` into a model-ready `BatchFeature` on device.

        Wraps the three steps of the official quickstart: `apply_chat_template`,
        `process_vision_info`, then `self.processor(...)`. The returned object
        already lives on `self.model.device`.

        Due regimi, secondo `pass_video_metadata`:

        - **`False` (default, regime storico di questo repo)** — i frame
          arrivano al processor senza `video_metadata`, che quindi se li
          inventa (`transformers.video_utils.make_batched_metadata`:
          `fps=None`, `frames_indices=range(T)`). Su Qwen2.5-VL questo rende
          `second_per_grid_ts = temporal_patch_size / sampled_fps = 2/24 =
          0.083` per QUALUNQUE video, e `get_rope_index`
          (`modeling_qwen2_5_vl.py:1125`) fa `tokens_per_second *
          int(0.083) = 0`: tutte le celle video finiscono sulla stessa
          posizione temporale M-RoPE. È il regime in cui sono state prodotte
          tutte le misure del README, quindi resta il default per non
          invalidarle — non perché sia corretto.
          Nota: `video_kwargs` è passato come kwarg singolo e transformers 5.7
          lo SCARTA con un warning (`_merge_kwargs`: chiave non valida). Non
          si può semplicemente spacchettarlo (`**video_kwargs`): contiene
          `fps` come LISTA e la validazione strict di `VideosKwargs` pretende
          uno scalare → `StrictDataclassFieldValidationError`. Lasciato
          com'era di proposito: qui l'obiettivo è la riproducibilità.

        - **`True`** — `process_vision_info` ritorna i metadata reali (fps del
          sorgente + indici dei frame campionati) e li si passa al processor.
          Su Qwen2.5-VL ripristina il vero `second_per_grid_ts`; su Qwen3-VL
          (dove è il default, vedi `Qwen3VL`) è l'unico modo di avere nel
          prompt i timestamp `<x.x seconds>` giusti. `video_kwargs` qui è il
          solo `do_sample_frames=False` (niente `fps`), quindi si spacchetta.
          I blocchi `VideoFrames` con metadata REALI (`frames_indices`
          assoluti + `fps` del sorgente, canale `_VIDEO_METADATA_KEY` — vedi
          `build_messages_from_parts`) sovrascrivono i fake metadata di
          `process_vision_info`, che per una lista di frame partono sempre
          da zero.
        """
        messages, manual_metadata = self._extract_manual_video_metadata(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if not self.pass_video_metadata:
            if manual_metadata:
                # Fail-fast, non degrado silenzioso: senza il kwarg
                # `video_metadata` al processor gli indici assoluti non
                # arrivano da nessuna parte e i timestamp ripartirebbero da
                # zero a ogni blocco — l'esperimento sbagliato che crede di
                # essere quello giusto.
                raise ValueError(
                    "VideoFrames con frames_indices/fps richiede "
                    "model.pass_video_metadata=true (su Qwen3-VL è il "
                    "default): nel regime storico i metadata non vengono "
                    "inoltrati al processor e andrebbero persi in silenzio."
                )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, image_patch_size=self.image_patch_size,
            )
            return self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                video_kwargs=video_kwargs,
            ).to(self.model.device)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
            image_patch_size=self.image_patch_size,
        )
        # Con `return_video_metadata=True` ogni elemento è `(video, metadata)`.
        videos = video_metadata = None
        if video_inputs:
            videos = [v for v, _ in video_inputs]
            video_metadata = [m for _, m in video_inputs]
            # I blocchi con metadata manuali (VideoFrames + frames_indices
            # assoluti) SOSTITUISCONO i fake metadata che `fetch_video`
            # fabbrica per una lista di frame (`frames_indices=range(n)`
            # locali al blocco, orologio che riparte da zero — vedi
            # `models/media.py`). L'indice `i` è la posizione del blocco
            # video in ordine di documento, lo stesso ordine in cui
            # `process_vision_info` li estrae.
            for i, meta in manual_metadata.items():
                video_metadata[i] = meta

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        return inputs

    def _decode_completion(
        self,
        inputs: BatchFeature,
        generated_ids: torch.Tensor,
    ) -> str:
        """Strip the prompt prefix from `generated_ids` and decode the rest.

        Assumes batch size 1 and returns the single completion string with
        special tokens removed.
        """
        prompt_len = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, prompt_len:]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @weave.op
    def generate(
        self,
        messages: list[dict],
        generation_config: GenerationConfig | None = None,
        **generate_kwargs,
    ) -> str:
        """Run inference end-to-end on `messages`.

        Orchestrates `_prepare_inputs` → `self.model.generate` (under
        `torch.no_grad()`) → `_decode_completion`. `generation_config` and
        `**generate_kwargs` are forwarded as-is; kwargs take precedence over
        fields of `generation_config`.
        """
        inputs = self._prepare_inputs(messages)
        logger.debug("Prepared inputs: input_ids shape=%s", tuple(inputs.input_ids.shape))
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                **generate_kwargs,
            )
        completion = self._decode_completion(inputs, generated_ids)
        logger.debug("Generated %d new tokens", generated_ids.shape[1] - inputs.input_ids.shape[1])
        return completion

    @weave.op
    def extract_entities(self, question_block: str) -> str:
        """Estrae il/i soggetto/i di una domanda MCQ col decoder GIÀ caricato.

        Chiamata SOLO-TESTO (nessun media, niente `process_vision_info`):
        stesso modello e stessi pesi dell'inferenza video, quindi zero costo
        di memoria aggiuntivo e ~32 token generati per sample. Greedy
        (`do_sample=False`): l'estrazione deve essere deterministica a parità
        di domanda, indipendente dalla `GenerationConfig` dell'arm.

        `question_block` è il testo della SOLA domanda (senza opzioni né
        boilerplate: le opzioni innescavano il prior "MCQ → rispondi" del
        2B, vedi ENTITY_EXTRACTION_EXAMPLES), tipicamente ricostruito dai
        `query_tokens` da `utils.attn_core.resolve_query_rows`. Ritorna la
        stringa grezza del decoder (soggetti separati da virgola); il
        mapping alle righe di attenzione avviene a valle, non qui.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ENTITY_EXTRACTION_SYSTEM}]},
        ]
        for ex_q, ex_subjects in ENTITY_EXTRACTION_EXAMPLES:
            messages.append({"role": "user", "content": [{"type": "text", "text": ex_q}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ex_subjects}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": question_block}]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=32, do_sample=False
            )
        return self._decode_completion(inputs, generated_ids).strip()


class Qwen25VL3B(Qwen):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_cls = Qwen2_5_VLForConditionalGeneration
    processor_cls = Qwen2_5_VLProcessor


class Qwen3VL(Qwen):
    """Base della famiglia Qwen3-VL: cambia solo il preprocessing visivo.

    Due differenze rispetto a Qwen2.5-VL, entrambe nel percorso
    `process_vision_info` → processor:

    1. **`image_patch_size = 16`** (Qwen2.5-VL: 14) — `smart_resize`
       arrotonda a `patch_size * merge_size`, quindi con 14 i frame
       verrebbero pre-ridimensionati su una griglia che il video processor
       di Qwen3 dovrebbe poi rifare.
    2. **`pass_video_metadata = True`** — obbligatorio, non un'opzione: il
       processor di Qwen3-VL scrive i timestamp DENTRO il prompt, un
       `<x.x seconds>` prima di ogni blocco visivo
       (`processing_qwen3_vl.py:157-170`), calcolandoli da
       `video_metadata.frames_indices / fps`. Senza metadata reali ripiega su
       `fps=24` e `frames_indices=range(T)`: un video di un'ora sampleato a
       128 frame verrebbe presentato al modello come lungo 5 secondi. E su
       questa famiglia il tempo È quel testo — `get_rope_index` splitta
       `video_grid_thw` per frame e non esiste alcun `second_per_grid_ts`.
    """

    image_patch_size = 16
    pass_video_metadata = True

    model_cls = Qwen3VLForConditionalGeneration
    processor_cls = Qwen3VLProcessor


class Qwen3VL2B(Qwen3VL):
    model_id = "Qwen/Qwen3-VL-2B-Instruct"


class Qwen3VL4B(Qwen3VL):
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
