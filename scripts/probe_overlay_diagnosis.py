"""Diagnosi dei MISS del gate di leggibilità — separa tre ipotesi di fallimento.

Il gate (scripts/probe_overlay_legibility.py) è fallito 5/10 IDENTICO a tutte
le altezze di banda (0.12/0.18/0.25, 2026-08-28): il collo di bottiglia non è
la dimensione dei glifi, è per-video. Tre ipotesi da separare:

  (i)   i pixel della banda non sono leggibili in quei video;
  (ii)  sono leggibili ma NON salienti in un contesto a 24 frame (2 dipinti);
  (iii) sono salienti solo se la parola viene NOMINATA nella domanda — che è
        il regime dell'ARM (OVERLAY_INSTRUCTION la nomina), quindi il gate
        aperto sarebbe più severo del compito reale.

Quattro misure per video (vedi tabella in fondo al file per la lettura):
  clean+named    domanda YES/NO che nomina IMPORTANT, sul video PULITO
                 (controllo yes-bias: atteso NO)
  painted+named  stessa domanda, cella 5 dipinta (regime arm: atteso YES)
  cell_only      OCR aperto su un video di SOLI 2 frame (la cella dipinta)
  all_painted    OCR aperto con TUTTI i 24 frame dipinti

Copia anche i frame dipinti in overlay_check/diagnosi/ per il check a occhio.

Uso:  PYTHONPATH=. uv run python scripts/probe_overlay_diagnosis.py \
          /work/tesi_avalenza/datasets/video_mme
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GenerationConfig

from models import Qwen3VL2BAttention
from models.media import Text, VideoFrames
from strategies.attention_marker import _extract_all_frames
from utils.attn_core import reconstruct_frame_indices
from utils.overlay import paint_important

# I 5 MISS della corsa del gate 2026-08-28 (identici a ogni height_frac).
MISS = ["-3t1rj8g6yg", "-O6mJ0VBTc4", "-QuCz7kxBr8", "-XpJeDGh8No", "-gnfUTPmnNU"]
NFRAMES, MAX_PIXELS, FRAC = 24, 151200, 0.18   # stessi del gate
CELL = 5
Q_OPEN = (
    "Some frames of this video may have a word written on a colored band. "
    "Which word is it? Answer with the single word only, or NONE."
)
Q_NAMED = (
    "Do any frames of this video have the word IMPORTANT written on a red band? "
    "Answer YES or NO."
)


def main() -> None:
    root = Path(sys.argv[1])
    out = Path("overlay_check/diagnosi")
    out.mkdir(parents=True, exist_ok=True)

    vlm = Qwen3VL2BAttention(torch_dtype=torch.bfloat16, device_map="auto", max_pixels=MAX_PIXELS)
    gen = GenerationConfig(do_sample=False, max_new_tokens=16)

    def ask(paths, idx, fps, q):
        media = VideoFrames(paths, max_pixels=MAX_PIXELS,
                            frames_indices=[int(i) for i in idx], fps=fps)
        return vlm.generate(vlm.build_messages(media, Text(q)), generation_config=gen).strip()

    for name in MISS:
        video = next(root.rglob(f"{name}.mp4"), None)
        if video is None:
            print(f"{name}: NON TROVATO sotto {root}")
            continue
        idx, fps = reconstruct_frame_indices(str(video), NFRAMES)

        paths, tmp = _extract_all_frames(str(video), idx)
        try:
            clean_named = ask(paths, idx, fps, Q_NAMED)

            for pos in (2 * CELL, 2 * CELL + 1):
                paint_important(paths[pos], paths[pos], height_frac=FRAC)
                shutil.copy(paths[pos], out / f"{name}_f{pos}_painted.png")

            named = ask(paths, idx, fps, Q_NAMED)
            cell_only = ask(paths[2 * CELL:2 * CELL + 2], idx[2 * CELL:2 * CELL + 2], fps, Q_OPEN)

            for pos in range(len(paths)):
                if pos not in (2 * CELL, 2 * CELL + 1):
                    paint_important(paths[pos], paths[pos], height_frac=FRAC)
            all_painted = ask(paths, idx, fps, Q_OPEN)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        print(f"{name}:  clean+named={clean_named!r}  painted+named={named!r}  "
              f"cell_only={cell_only!r}  all_painted={all_painted!r}")


# Lettura:
#   cell_only legge IMPORTANT      → pixel leggibili: (i) esclusa, è salienza
#   all_painted legge              → conferma (ii): la banda si vede, 2/24 non emergono
#   painted+named=YES, clean=NO    → (iii): il regime dell'arm funziona, gate troppo severo
#   clean+named=YES                → yes-bias: la domanda named non è evidenza
if __name__ == "__main__":
    main()
