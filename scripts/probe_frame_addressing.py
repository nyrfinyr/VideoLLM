"""Qwen3-VL sa INDIRIZZARE un frame? — gate del ramo ViKey (arXiv 2603.23186).

ViKey sostiene che i VideoLLM falliscono il ragionamento temporale perché non
sanno riferirsi al frame *k*, e che dipingere l'indice nei pixel restituisce
quella capacità: nel loro test sintetico il reverse lookup passa da 18.57% a
100%. Ma i modelli che testano (Qwen2.5-VL, LLaVA-Video, LLaVA-OneVision) NON
interleavano timestamp testuali, mentre **Qwen3-VL sì**: il processor scrive
un `<x.x seconds>` prima di ogni blocco visivo (`models/qwen.py`, docstring di
`Qwen3VL`). Su Qwen3-VL la premessa del paper è quindi in larga parte già
soddisfatta, e l'SVP rischia di essere ridondante.

Questa sonda decide il ramo PRIMA di spendere GPU su un arm:

- se il regime `native` indirizza già bene → l'SVP non si integra, e il nullo
  della variante A (`attention_highlight`) NON è un problema di
  indirizzamento: va cercato altrove;
- se indirizza male come nel paper → l'SVP diventa un arm sensato ed è il
  prerequisito per rilanciare la variante A, finora misurata su un canale che
  il modello non sapeva risolvere.

## Disegno

Ago sintetico: un cerchio verde acceso dipinto su UNA cella. Due condizioni:

| condizione | pixel | indirizzo usato nella domanda |
|---|---|---|
| `native` | solo l'ago | `"12.3 seconds"` — i timestamp che Qwen3 già emette |
| `svp`    | ago + `frame #NN` su OGNI cella | `"frame #07"` — ViKey |

**L'unità indirizzabile è la CELLA, non il frame.** In regime
`double_frames=false` una cella = 2 frame e il processor emette un timestamp
per cella (24 frame → 12 indirizzi). Dipingere numeri diversi sui due frame di
una cella li farebbe collassare nello stesso gruppo di token visivi: il port
fedele di ViKey su Qwen3-VL è un'etichetta per cella.

Quattro domande per video e condizione:

1. `detect` — "c'è un cerchio verde da qualche parte?" **È il gate**: se il
   modello non vede l'ago, la sonda non misura indirizzamento ma percezione e
   il resto dei numeri non vuol dire niente.
2. `lookup_pos` — "all'indirizzo X c'è il cerchio?" con X = la cella dell'ago.
   Atteso YES.
3. `lookup_neg` — stessa domanda con X = un'altra cella (distante ≥3).
   Atteso NO. **Questo è il controllo che manca a ViKey**: un modello che
   ignora l'indirizzo e risponde sempre YES fa 100% sui positivi e 0% qui.
   Senza il negativo, il lookup del paper è ingannevole.
4. `revlookup` — "a che secondo (o su che frame) appare?" Direzione inversa,
   scoring esatto (`svp`) o a tolleranza mezza cella (`native`).

La metrica da leggere è l'**accuratezza bilanciata** su `lookup_pos`/
`lookup_neg`: il caso vale 50%, e la quota di YES rivela subito la modalità
degenere.

Non c'è una terza condizione "senza timestamp": `lookup_neg` risponde già alla
stessa domanda (gli indirizzi portano informazione?) senza dover forzare
`pass_video_metadata=False`, regime che su Qwen3-VL la classe dichiara
obbligatorio e che introdurrebbe una seconda variabile.

Standalone, non passa da config/Weave (come
`scripts/probe_overlay_legibility.py`). Serve una GPU.

Uso:

    uv run python scripts/probe_frame_addressing.py <dir_video> [n_video] [seed]
"""
import re
import shutil
import sys
from pathlib import Path

# Lanciato come `python scripts/probe_frame_addressing.py` il repo root non è
# su sys.path e gli import del progetto fallirebbero.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw
from transformers import GenerationConfig

from models import Qwen3VL2BAttention
from models.media import Text, VideoFrames
from strategies.attention_highlight import _sample_rng
from strategies.attention_marker import _extract_all_frames
from utils.attn_core import reconstruct_frame_indices
from utils.overlay import paint_frame_index

NFRAMES, MAX_PIXELS = 24, 151200          # conf/config.yaml, dataset.video_mme
FRAMES_PER_CELL = 2                        # double_frames=false
MIN_CELL_GAP = 3                           # distanza minima fra cella vera e cella-esca

# Verde saturo + bordo bianco: un colore che nei video naturali non compare a
# questa saturazione, quindi "l'ho visto" non si confonde con "c'era del verde".
NEEDLE_FILL = (0, 230, 60)
NEEDLE_EDGE = (255, 255, 255)
NEEDLE_RADIUS_FRAC = 0.18

NEEDLE_DESC = "a large bright green circle drawn on top of the picture"

Q_DETECT = (
    f"Does any part of this video show {NEEDLE_DESC}? Answer YES or NO."
)
Q_LOOKUP = (
    "Consider only {addr}. Does that moment show " + NEEDLE_DESC + "? "
    "Answer YES or NO."
)
Q_REV_NATIVE = (
    f"{NEEDLE_DESC.capitalize()} appears at exactly one moment in this video. "
    "At which second does it appear? Answer with a number only."
)
Q_REV_SVP = (
    f"{NEEDLE_DESC.capitalize()} appears on exactly one frame of this video. "
    "Each frame is labelled with its number in a corner. On which frame "
    "number does it appear? Answer with the number only."
)


def paint_needle(path: str) -> None:
    """Dipinge il cerchio-ago al centro del frame, in place.

    Come gli overlay di `utils/overlay.py` la dimensione non cambia: l'ago non
    deve alterare la griglia di `smart_resize`, altrimenti le due condizioni
    non sarebbero confrontabili.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = NEEDLE_RADIUS_FRAC * min(w, h)
    cx, cy = w / 2, h / 2
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NEEDLE_FILL,
                 outline=NEEDLE_EDGE, width=max(2, round(r * 0.08)))
    if img.size != (w, h):
        raise RuntimeError(f"needle ha cambiato la dimensione: {(w, h)} -> {img.size}")
    img.save(path)


def _yes_no(raw: str) -> bool | None:
    """YES/NO dalla prima parola. `None` = non parseabile (contato a parte)."""
    m = re.search(r"\b(yes|no)\b", raw.strip().lower())
    return None if m is None else m.group(1) == "yes"


def _first_number(raw: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    return None if m is None else float(m.group())


def _dump_timestamps(vlm, messages) -> None:
    """Stampa i `<x.x seconds>` che il processor mette davvero nel prompt.

    Serve a VERIFICARE la mappa cella→timestamp che questa sonda assume
    (`t_c = frame_indices[2c] / fps`): se diverge, le domande indirizzerebbero
    la cella sbagliata e il risultato sarebbe un artefatto. Best-effort: un
    fallimento qui non deve uccidere la sonda.
    """
    try:
        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        stamps = re.findall(r"<([\d.]+) seconds?>", text)
        print(f"  [verifica] il prompt contiene {len(stamps)} timestamp: {stamps}")
    except Exception as exc:  # noqa: BLE001 — diagnostica, non correttezza
        print(f"  [verifica] dump timestamp non riuscito: {exc!r}")


def main() -> None:
    video_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    # Il dump di Video-MME non è omogeneo nell'estensione (`videos/data/`
    # contiene .mp4, .MP4 e .mkv — vedi evals/video_mme.py).
    videos = sorted(
        p for p in video_dir.rglob("*") if p.suffix.lower() in (".mp4", ".mkv")
    )[:n]
    if not videos:
        raise SystemExit(f"nessun video (.mp4/.MP4/.mkv) sotto {video_dir}")

    t_cells = NFRAMES // FRAMES_PER_CELL

    vlm = Qwen3VL2BAttention(torch_dtype=torch.bfloat16, device_map="auto", max_pixels=MAX_PIXELS)
    gen_yn = GenerationConfig(do_sample=False, max_new_tokens=8)
    gen_num = GenerationConfig(do_sample=False, max_new_tokens=16)

    # tallies[cond] = dict di contatori
    conds = ("native", "svp")
    tally = {c: {"detect": 0, "pos": 0, "neg": 0, "rev": 0,
                 "yes": 0, "yn_unparsed": 0, "rev_unparsed": 0, "n": 0}
             for c in conds}
    dumped = False

    for video in videos:
        idx, fps = reconstruct_frame_indices(str(video), NFRAMES)
        # Timestamp per cella = primo frame della cella (assunzione VERIFICATA
        # dal dump di `_dump_timestamps` sul primo video).
        t_cell = [idx[FRAMES_PER_CELL * c] / fps for c in range(t_cells)]
        gap = (t_cell[-1] - t_cell[0]) / max(1, t_cells - 1)
        tol = gap / 2

        rng = _sample_rng(seed, str(video), "frame_addressing")
        # Prima e ultima cella escluse: i bordi hanno un regime d'attenzione
        # loro e non sono rappresentativi.
        c_true = rng.randrange(1, t_cells - 1)
        candidates = [c for c in range(t_cells) if abs(c - c_true) >= MIN_CELL_GAP]
        c_false = rng.choice(candidates)

        for cond in conds:
            paths, tmp_dir = _extract_all_frames(str(video), idx)
            try:
                for p in range(FRAMES_PER_CELL * c_true, FRAMES_PER_CELL * (c_true + 1)):
                    paint_needle(paths[p])
                if cond == "svp":
                    # Stessa etichetta sui due frame di una cella: l'unità
                    # indirizzabile è la cella (vedi docstring del modulo).
                    for c in range(t_cells):
                        for p in range(FRAMES_PER_CELL * c, FRAMES_PER_CELL * (c + 1)):
                            paint_frame_index(paths[p], paths[p], f"frame #{c + 1:02d}")

                media = VideoFrames(paths, max_pixels=MAX_PIXELS,
                                    frames_indices=[int(i) for i in idx], fps=fps)

                if cond == "native":
                    addr_true = f"the moment at {t_cell[c_true]:.1f} seconds"
                    addr_false = f"the moment at {t_cell[c_false]:.1f} seconds"
                    q_rev = Q_REV_NATIVE
                else:
                    addr_true = f"the frame labelled 'frame #{c_true + 1:02d}'"
                    addr_false = f"the frame labelled 'frame #{c_false + 1:02d}'"
                    q_rev = Q_REV_SVP

                def ask(question: str, cfg=gen_yn) -> str:
                    return vlm.generate(
                        vlm.build_messages(media, Text(question)), generation_config=cfg
                    )

                if not dumped and cond == "native":
                    _dump_timestamps(vlm, vlm.build_messages(media, Text(Q_DETECT)))
                    dumped = True

                r_det = ask(Q_DETECT)
                r_pos = ask(Q_LOOKUP.format(addr=addr_true))
                r_neg = ask(Q_LOOKUP.format(addr=addr_false))
                r_rev = ask(q_rev, gen_num)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            t = tally[cond]
            t["n"] += 1
            for raw, want, key in ((r_det, True, "detect"), (r_pos, True, "pos"), (r_neg, False, "neg")):
                got = _yes_no(raw)
                if got is None:
                    t["yn_unparsed"] += 1
                    continue
                t["yes"] += got
                t[key] += got == want

            num = _first_number(r_rev)
            if num is None:
                t["rev_unparsed"] += 1
                rev_ok = False
            elif cond == "native":
                rev_ok = abs(num - t_cell[c_true]) <= tol
            else:
                rev_ok = int(num) == c_true + 1
            t["rev"] += rev_ok

            want_rev = f"{t_cell[c_true]:.1f}s±{tol:.1f}" if cond == "native" else f"#{c_true + 1:02d}"
            print(
                f"[{cond:6s}] {video.name[:34]:34s} cella={c_true:2d} esca={c_false:2d} | "
                f"det={r_det.strip()[:6]!r} pos={r_pos.strip()[:6]!r} neg={r_neg.strip()[:6]!r} "
                f"rev={r_rev.strip()[:10]!r} (vero {want_rev}) {'OK' if rev_ok else '--'}"
            )

    print("\n" + "=" * 78)
    print(f"{'cond':8s} {'detect':>9s} {'lookup+':>9s} {'lookup-':>9s} {'bilanc.':>9s} "
          f"{'revlook':>9s} {'%YES':>7s} {'n.p.':>6s}")
    for cond in conds:
        t = tally[cond]
        n = max(1, t["n"])
        bal = 100 * (t["pos"] / n + t["neg"] / n) / 2
        print(
            f"{cond:8s} {100 * t['detect'] / n:8.1f}% {100 * t['pos'] / n:8.1f}% "
            f"{100 * t['neg'] / n:8.1f}% {bal:8.1f}% {100 * t['rev'] / n:8.1f}% "
            f"{100 * t['yes'] / (3 * n):6.1f}% {t['yn_unparsed'] + t['rev_unparsed']:6d}"
        )
    print("=" * 78)
    print(
        "gate: se `detect` non è ~100% l'ago non è percepibile e il resto non\n"
        "vale. `bilanc.` = (lookup+ + lookup-)/2, caso = 50%. %YES vicino a\n"
        "100 con bilanc. ~50 = il modello ignora l'indirizzo e dice sempre sì."
    )


if __name__ == "__main__":
    main()
