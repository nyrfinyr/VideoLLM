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

MediaItem = Image | Video | VideoFrames


def to_content_dict(item: MediaItem | Text) -> dict:
    """Flatten un MediaItem/Text al dict atteso dal chat template Qwen."""
    return {k: v for k, v in asdict(item).items() if v is not None}
