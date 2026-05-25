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
    fps: float = 1
    video_start: float | None = None    # None per decodificare l'intero video.
    video_end: float | None = None      # None per decodificare l'intero video.

@dataclass(frozen=True)
class VideoFrames:
    type: str = field(default="video", init=False)
    video: list[str]

MediaItem = Image | Video | VideoFrames


def to_content_dict(item: MediaItem | Text) -> dict:
    """Flatten un MediaItem/Text al dict atteso dal chat template Qwen."""
    return {k: v for k, v in asdict(item).items() if v is not None}
