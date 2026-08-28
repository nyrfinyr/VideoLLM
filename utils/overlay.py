"""Overlay testuale sui frame — rendering per `strategies/visual_prompt.py`.

Modulo PURO (PIL + stdlib): nessun import dal resto del progetto, così
`scripts/probe_overlay_legibility.py` e la verifica offline dell'appendice A
di `docs/piano_visual_prompting.md` possono usarlo senza toccare modello,
Hydra o Weave.

La parola e i colori sono costanti di modulo, NON knob di config: sono il
TESTO dell'intervento (stesso criterio di `MARKER_INSTRUCTION` in
`strategies/attention_marker.py`). Knob restano solo posizione e altezza
della banda, che sono geometria da tarare sulla probe di leggibilità.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OVERLAY_TEXT = "IMPORTANT"

# Banda PIENA + testo in controcolore, non testo con contorno sopra il frame:
# il contrasto non deve dipendere dal contenuto del video, altrimenti su una
# scena chiara la parola sparisce esattamente nei sample in cui serve.
BAND_FILL = (220, 30, 30)
TEXT_FILL = (255, 255, 255)

# Il corpo del testo occupa questa frazione dell'altezza della banda: il resto
# è padding, senza il quale i glifi toccano i bordi e il downscale li impasta.
_TEXT_OF_BAND = 0.75
_MAX_TEXT_WIDTH_FRAC = 0.92
# Sotto questo corpo (px SORGENTE) qualcosa nel calcolo della scala è saltato:
# meglio fallire che dipingere un francobollo illeggibile in silenzio.
_MIN_TEXT_PX = 12

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(px: int) -> tuple[ImageFont.FreeTypeFont, str]:
    """Font bold di sistema, con fallback al default scalabile di Pillow.

    ⚠️ `ImageFont.load_default()` SENZA `size` ritorna il bitmap 11 px:
    illeggibile dopo il downscale, e il degrado sarebbe silenzioso. Da Pillow
    10.1 `load_default(size=...)` ritorna un TTF scalabile (Aileron), quindi
    il fallback è accettabile — ma il nome del font va RITORNATO e loggato per
    sample: "sul cluster il font di sistema non c'era" deve essere leggibile
    in Weave, non da indovinare.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px), Path(path).name
        except OSError:
            continue
    return ImageFont.load_default(size=px), "pillow-default"


def paint_important(
    src: str | Path,
    dst: str | Path,
    *,
    text: str = OVERLAY_TEXT,
    position: str = "top",
    height_frac: float = 0.18,
) -> dict:
    """Dipinge `text` su una banda piena in cima (o in fondo) al frame.

    `src` e `dst` possono coincidere (la strategy sovrascrive i PNG nella
    tmpdir): `Image.open(...).convert(...)` carica per intero, quindi il file
    non è più aperto quando si salva.

    **La dimensione dell'immagine NON cambia** (asserito): allargare il canvas
    per far posto alla banda cambierebbe l'aspect ratio → `smart_resize`
    darebbe una griglia diversa → i frame dipinti e quelli puliti non
    sarebbero più confrontabili e il conteggio token cambierebbe in silenzio.

    Ritorna la diagnostica da loggare per sample:
    `{"font", "text_px", "band_px", "size"}`.
    """
    if position not in ("top", "bottom"):
        raise ValueError(f"position sconosciuta: {position!r}. Valide: 'top', 'bottom'")
    if not 0.0 < height_frac < 1.0:
        raise ValueError(f"height_frac fuori range: {height_frac!r}")

    img = Image.open(src).convert("RGB")
    w, h = img.size

    band_px = max(1, round(height_frac * h))
    text_px = max(1, round(_TEXT_OF_BAND * band_px))
    font, font_name = _load_font(text_px)

    draw = ImageDraw.Draw(img)
    # Su frame stretti (o `max_pixels` bassi) la parola non ci sta in
    # larghezza: si rimpicciolisce finché entra.
    while text_px > _MIN_TEXT_PX and draw.textlength(text, font=font) > _MAX_TEXT_WIDTH_FRAC * w:
        text_px -= 2
        font = font.font_variant(size=text_px)
    if text_px <= _MIN_TEXT_PX:
        raise RuntimeError(
            f"overlay illeggibile: corpo sceso a {text_px}px su un frame "
            f"{w}x{h} (height_frac={height_frac}). Frame troppo piccolo o "
            "parola troppo lunga — non dipingere e basta, l'arm misurerebbe "
            "solo occlusione."
        )

    y0 = 0 if position == "top" else h - band_px
    draw.rectangle([0, y0, w, y0 + band_px], fill=BAND_FILL)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((w - tw) / 2 - bbox[0], y0 + (band_px - th) / 2 - bbox[1]),
        text, font=font, fill=TEXT_FILL,
    )

    if img.size != (w, h):
        raise RuntimeError(f"overlay ha cambiato la dimensione: {(w, h)} -> {img.size}")
    img.save(dst)
    return {"font": font_name, "text_px": text_px, "band_px": band_px, "size": (w, h)}


if __name__ == "__main__":
    # Verifica §8.0 del piano (docs/piano_visual_prompting.md, appendice A):
    # dipinge un frame ai tre height_frac candidati, applica lo smart_resize
    # del processor e salva il PNG "come lo vede il modello" — da GUARDARE.
    # Gira senza GPU né pesi. Uso:
    #
    #     uv run python -m utils.overlay <frame.png> [out_dir]
    #
    import sys

    from qwen_vl_utils.vision_process import smart_resize  # solo qui: il modulo resta puro

    MAX_PIXELS = 151200   # conf/config.yaml, dataset.video_mme
    FACTOR = 32           # Qwen3-VL: image_patch_size(16) * merge(2). Qwen2.5-VL: 28
    HEIGHT_FRACS = (0.12, 0.18, 0.25)

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for frac in HEIGHT_FRACS:
        painted = out_dir / f"painted_{frac}.png"
        info = paint_important(src, painted, height_frac=frac)
        w, h = info["size"]
        th, tw = smart_resize(h, w, factor=FACTOR, min_pixels=FACTOR * FACTOR * 4, max_pixels=MAX_PIXELS)
        model_png = out_dir / f"model_{frac}.png"
        Image.open(painted).resize((tw, th), Image.BICUBIC).save(model_png)
        model_text_px = info["text_px"] * th / h
        print(
            f"height_frac={frac}: font={info['font']}  sorgente {w}x{h} -> "
            f"modello {tw}x{th} (griglia {tw // FACTOR}x{th // FACTOR})  "
            f"banda {info['band_px']}px, testo {info['text_px']}px sorgente -> "
            f"~{model_text_px:.0f}px modello ({model_text_px / FACTOR:.1f} celle)  "
            f"[{painted.name}, {model_png.name}]"
        )
