from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Text:
    type: str = field(default="text", init=False)
    text: str

@dataclass(frozen=True)
class Image:
    type: str = field(default="image", init=False)
    image: str

@dataclass(frozen=True)
class Video:
    type: str = field(default="video", init=False)
    video: str | Path
    max_pixels: int = 360 * 420
    # `min_pixels` opzionale. Per i VIDEO qwen-vl-utils impone un floor di
    # 128 token/frame (≈100352 px): per scendere sotto (molti frame a bassa
    # risoluzione, es. setup paper a 64 token/frame) va abbassato anche
    # min_pixels, altrimenti smart_resize asserisce `max_pixels >= min_pixels`.
    # None = lascia il default di qwen-vl-utils.
    min_pixels: int | None = None
    # Campionamento temporale: `nframes` (numero fisso di frame) OPPURE
    # `fps` (frame al secondo). qwen-vl-utils li rifiuta se passati insieme
    # (vedi `__post_init__`). Con `nframes` il conteggio token — e quindi la
    # VRAM — è deterministico (≈ (nframes/2) * token_per_frame), mentre con
    # `fps` dipende dalla durata del video. `to_content_dict` droppa quello
    # lasciato a None, così ne arriva sempre uno solo al chat template.
    fps: float | None = None
    nframes: int | None = None
    video_start: float | None = None    # None per decodificare l'intero video.
    video_end: float | None = None      # None per decodificare l'intero video.

    def __post_init__(self):
        if self.fps is not None and self.nframes is not None:
            raise ValueError(
                "Video: specifica `fps` OPPURE `nframes`, non entrambi "
                "(qwen-vl-utils.smart_nframes li rifiuta insieme)."
            )

@dataclass(frozen=True)
class VideoFrames:
    type: str = field(default="video", init=False)
    video: list[str]
    # Budget visivo per frame, come su `Video`. Con una lista di frame
    # qwen-vl-utils NON ricampiona (li usa as-is, padding a conteggio pari):
    # `fetch_video` inoltra questi `max_pixels`/`min_pixels` a `fetch_image`
    # per ogni frame e li riusa a livello video. Servono per riprodurre lo
    # stesso grid_thw di un `Video` equivalente (es. modalità frame-doubling
    # del debug attenzione). None = default di qwen-vl-utils.
    max_pixels: int | None = None
    min_pixels: int | None = None
    # `qwen_vl_utils.fetch_video` costruisce fake metadata per una lista di
    # frame (`frames_indices=[0..n-1]`, spaziatura uniforme assunta) e usa
    # QUESTO scalare come fps per il calcolo M-RoPE di `second_per_grid_ts`
    # (`vision_process.py`, `ele.get("sample_fps", 2.0)`) — non ci sono
    # timestamp per-frame reali da iniettare. Serve quando la lista non
    # rappresenta un campionamento uniforme del video sorgente (es. zoom
    # multi-regione disgiunto, `docs/piano_zoom_multiregione_disgiunto.md`
    # §3.3): il modello vedrà comunque i frame come equispaziati a questo
    # fps, la spaziatura reale (buchi fra regioni) è persa in encoding.
    # None = default di qwen-vl-utils (2.0).
    sample_fps: float | None = None
    # Metadata REALI del blocco: indici ASSOLUTI dei frame nel video sorgente
    # e fps reale del sorgente. Servono nel regime `pass_video_metadata=True`
    # (Qwen3-VL): il processor calcola i timestamp `<x.x seconds>` da
    # `video_metadata.frames_indices / fps`, e con più blocchi (marcatori
    # inline, `strategies/attention_marker.py`) i fake metadata di
    # qwen-vl-utils farebbero ripartire l'orologio da zero a ogni blocco
    # (`fetch_video` fabbrica `frames_indices=range(n)` LOCALI al blocco).
    # Questi campi NON finiscono nel dict per il chat template (vedi
    # `to_content_dict`): viaggiano su un canale separato fino a
    # `Qwen._prepare_inputs`, che li converte in
    # `transformers.video_utils.VideoMetadata` per blocco. Vanno impostati
    # entrambi o nessuno; quando sono presenti `sample_fps` è irrilevante
    # (su Qwen3-VL non esiste `second_per_grid_ts`).
    frames_indices: list[int] | None = None
    fps: float | None = None

    def __post_init__(self):
        if (self.frames_indices is None) != (self.fps is None):
            raise ValueError(
                "VideoFrames: `frames_indices` e `fps` vanno impostati insieme "
                "(o nessuno dei due) — il processor calcola i timestamp da "
                "`frames_indices / fps`, con uno solo dei due non può."
            )
        if self.frames_indices is not None and len(self.frames_indices) != len(self.video):
            raise ValueError(
                f"VideoFrames: {len(self.frames_indices)} frames_indices per "
                f"{len(self.video)} frame — serve un indice assoluto per ogni "
                "frame, altrimenti i timestamp risultano sfasati in silenzio."
            )

# Campi di `VideoFrames` che NON devono raggiungere il chat template né
# `process_vision_info`: sono metadata per `Qwen._prepare_inputs`, non
# direttive di campionamento. In particolare un `fps` in un content dict
# video verrebbe interpretato da qwen-vl-utils come frame-rate di
# campionamento, non come fps del sorgente.
_VIDEOFRAMES_METADATA_FIELDS = ("frames_indices", "fps")

MediaItem = Image | Video | VideoFrames


def to_content_dict(item: MediaItem | Text) -> dict:
    """Flatten un MediaItem/Text al dict atteso dal chat template Qwen.

    Droppa i field `None` e i metadata-only di `VideoFrames`
    (`frames_indices`/`fps`), che viaggiano su un canale separato — vedi
    `Qwen.build_messages_from_parts`.
    """
    d = {k: v for k, v in asdict(item).items() if v is not None}
    if isinstance(item, VideoFrames):
        for k in _VIDEOFRAMES_METADATA_FIELDS:
            d.pop(k, None)
    return d
