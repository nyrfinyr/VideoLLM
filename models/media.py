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

MediaItem = Image | Video | VideoFrames


def to_content_dict(item: MediaItem | Text) -> dict:
    """Flatten un MediaItem/Text al dict atteso dal chat template Qwen."""
    return {k: v for k, v in asdict(item).items() if v is not None}
