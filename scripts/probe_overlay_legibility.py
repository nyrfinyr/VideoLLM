"""Il modello LEGGE la parola dipinta sui frame? — GATE di `visual_prompt`.

Se la risposta è no, `strategies/visual_prompt.py` non misura visual
prompting ma occlusione: questo script va PRIMA di qualunque accuracy
(docs/piano_visual_prompting.md §8.1). Sweepa `height_frac` per tarare il
knob `overlay_height_frac` del preset in un colpo solo.

Criterio di accettazione: ≥ 9/10 a un height_frac che non superi ~0.20
(oltre si occlude troppo). Se nessun height_frac accettabile passa: opzione
B del piano §3 (dipingere DOPO il pre-resize al target di smart_resize).

Per ogni lettura riuscita si fa anche la domanda temporale ("a che secondo
appare?"): se il modello legge la parola ma non sa collocarla nel tempo, è
un'informazione sul perché l'arm potrebbe non guadagnare — e si ottiene qui,
non dopo tre ore di fullset.

Il modello è `Qwen3VL2BAttention` (il preset `qwen3_vl_2b_attn`): lo stesso
che girerà il full eval, così il gate vale per quel caricamento esatto.

Standalone, non passa da Hydra/Weave (precedente:
`scripts/probe_entity_extraction.py`).

Uso:  uv run python scripts/probe_overlay_legibility.py <dir_video> [n_video]
"""
import shutil
import sys
from pathlib import Path

import torch
from transformers import GenerationConfig

from models import Qwen3VL2BAttention
from models.media import Text, VideoFrames
from strategies.attention_marker import _extract_all_frames
from utils.attn_core import reconstruct_frame_indices
from utils.overlay import OVERLAY_TEXT, paint_important

NFRAMES, MAX_PIXELS = 24, 151200          # conf/config.yaml, video_mme
HEIGHT_FRACS = (0.12, 0.18, 0.25)
CELL = 5                                   # cella dipinta, fissa: qui non si misura l'attenzione
QUESTION = (
    "Some frames of this video may have a word written on a colored band. "
    "Which word is it? Answer with the single word only, or NONE."
)
WHEN_QUESTION = (
    "Some frames of this video have a word written on a colored band. "
    "At which second of the video does the word appear? "
    "Answer with a number only, or NONE."
)


def main() -> None:
    video_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    videos = sorted(video_dir.rglob("*.mp4"))[:n]
    if not videos:
        raise SystemExit(f"nessun .mp4 sotto {video_dir}")

    vlm = Qwen3VL2BAttention(torch_dtype=torch.bfloat16, device_map="auto", max_pixels=MAX_PIXELS)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=16)

    for frac in HEIGHT_FRACS:
        hits = 0
        for video in videos:
            idx, fps = reconstruct_frame_indices(str(video), NFRAMES)
            paths, tmp_dir = _extract_all_frames(str(video), idx)
            try:
                info = None
                # ENTRAMBI i frame della cella, come farà la strategy.
                for pos in (2 * CELL, 2 * CELL + 1):
                    info = paint_important(paths[pos], paths[pos], height_frac=frac)
                media = VideoFrames(paths, max_pixels=MAX_PIXELS,
                                    frames_indices=[int(i) for i in idx], fps=fps)
                raw = vlm.generate(vlm.build_messages(media, Text(QUESTION)), generation_config=gen_cfg)
                ok = OVERLAY_TEXT.lower() in raw.lower()
                when_str = ""
                if ok:
                    when = vlm.generate(vlm.build_messages(media, Text(WHEN_QUESTION)), generation_config=gen_cfg)
                    t_true = idx[2 * CELL] / fps
                    when_str = f"  when={when.strip()!r} (vero ~{t_true:.1f}s)"
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            hits += ok
            print(f"[{frac}] {video.name}: {raw.strip()!r} {'OK' if ok else 'MISS'} "
                  f"(font={info['font']}, {info['text_px']}px){when_str}")
        print(f"=== height_frac={frac}: {hits}/{len(videos)} letti ===")


if __name__ == "__main__":
    main()
