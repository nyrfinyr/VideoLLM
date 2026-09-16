"""Campionamento A COPPIE per Qwen3-VL: ogni cella temporale = 2 frame a `gap` secondi.

Qwen3-VL fonde i frame a due a due (`temporal_patch_size=2`): una cella
dell'attenzione è la coppia `(2·i, 2·i+1)` della lista, e il timestamp
`<x.x seconds>` che il processor scrive nel prompt è la MEDIA dei due frame
(`processing_qwen3_vl.py::_calculate_timestamps`, da
`video_metadata.frames_indices / fps`). Col campionamento uniforme su video
lunghi (LVBench, mediana 71 min) i due frame di una cella distano minuti e la
cella non corrisponde a nessun istante preciso. Qui invece:

- `n_pairs` centri uniformi `c_i = D·(i+0.5)/n_pairs` sull'intero video
  (`D = total_frames / fps`);
- per ogni centro due frame a `c_i − gap/2` e `c_i + gap/2`, in lista
  interleaved `[c1−g/2, c1+g/2, c2−g/2, c2+g/2, ...]`: ogni cella è
  esattamente una coppia e il suo timestamp è `c_i`.

La spaziatura non uniforme della lista è lecita su Qwen3-VL: `get_rope_index`
usa t=1 per frame e il tempo passa solo dal testo del timestamp.

Casi limite (vedi `pair_centers_and_indices`):

- tempi clampati a `[0, (total_frames−1)/fps]`, indici `round(t·fps)` clampati
  a `[0, total_frames−1]`;
- se i due frame di una coppia cadono sullo STESSO indice (gap < 1/fps, oppure
  clamp al bordo) il secondo viene spostato sul frame successivo (o il primo
  sul precedente all'ultimo frame): la coppia resta di due frame distinti
  ogni volta che il video ne ha almeno due;
- video corti con `D/n_pairs < gap`: il gap NON viene ristretto (ogni cella
  resta "due frame a ~gap secondi", stessa risoluzione temporale per cella su
  tutti i video), quindi coppie consecutive si sovrappongono e la lista può
  essere non monotona e contenere indici duplicati. Nessun crash: i timestamp
  delle celle restano monotoni e l'estrazione legge ogni indice
  indipendentemente;
- i centri ritornati sono quelli EFFETTIVI, `(idx_a + idx_b) / (2·fps)`, cioè
  esattamente il timestamp (prima dell'arrotondamento a 0.1 s) che il
  processor scrive nel prompt; differiscono dal centro nominale di al più
  `0.5/fps` (più il clamp ai bordi sui video corti).

Niente modello, niente GPU: `pair_centers_and_indices` e
`pair_cells_in_window` sono pure; solo `pair_video_frames` apre il video.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.media import VideoFrames


def pair_centers_and_indices(total_frames: int, fps: float, n_pairs: int, gap_sec: float) -> tuple[list[float], list[int]]:
    """Funzione PURA (testabile senza video): centri in secondi (len n_pairs) e indici
    assoluti dei frame interleaved (len 2*n_pairs).

    Per la coppia `i`: tempi `c_i ∓ gap/2` clampati a `[0, (total_frames−1)/fps]`,
    indici `round(t·fps)` clampati a `[0, total_frames−1]`. Se i due indici
    coincidono e il video ha almeno 2 frame, il secondo avanza di uno (o, se è
    già l'ultimo frame, il primo arretra di uno). Il centro ritornato è la media
    dei due tempi REALI dei frame, `(idx_a + idx_b) / (2·fps)`: coincide col
    timestamp del processor Qwen3-VL. Vedi docstring del modulo per i video
    corti (coppie sovrapposte, lista non monotona).
    """
    if total_frames < 1:
        raise ValueError(f"total_frames deve essere >= 1, ricevuto {total_frames}")
    if not fps > 0:
        raise ValueError(f"fps deve essere > 0, ricevuto {fps}")
    if n_pairs < 1:
        raise ValueError(f"n_pairs deve essere >= 1, ricevuto {n_pairs}")
    if gap_sec < 0:
        raise ValueError(f"gap_sec deve essere >= 0, ricevuto {gap_sec}")

    duration = total_frames / fps
    last = total_frames - 1
    t_last = last / fps

    def to_index(t: float) -> int:
        t = min(max(t, 0.0), t_last)
        return min(max(int(round(t * fps)), 0), last)

    centers: list[float] = []
    indices: list[int] = []
    for i in range(n_pairs):
        c = duration * (i + 0.5) / n_pairs
        a = to_index(c - gap_sec / 2)
        b = to_index(c + gap_sec / 2)
        if a == b and last >= 1:
            # Gap sotto 1/fps o clamp al bordo: due frame distinti, altrimenti
            # la cella sarebbe un frame duplicato (niente informazione di moto).
            if b < last:
                b += 1
            else:
                a -= 1
        indices += [a, b]
        centers.append((a + b) / (2 * fps))
    return centers, indices


def _extract_pairs(video_path: str, indices: list[int], target: tuple[int, int] | None) -> tuple[list[str], "Path"]:
    """Un PNG per indice (nell'ordine della lista), opzionalmente GIÀ ridimensionato.

    Variante di `strategies.attention_marker._extract_all_frames` con due
    differenze che contano solo qui:

    - **lettura in blocco** (`VideoReader.get_batch` sugli indici ordinati e
      deduplicati) invece di un accesso per frame: su un video di 2 h a 512
      frame la decodifica scende da ~48 s a pochi secondi;
    - **resize opzionale a `target`**: le liste di frame passano comunque dal
      resize del modello (`models/qwen.py::_fetch_videoframes`), che porta ogni
      frame ESATTAMENTE a `target`. Scriverli già così non cambia i pixel
      finali (il secondo resize diventa l'identità) ma evita di scrivere 512
      PNG a risoluzione nativa: su 1080p ~230 s contro ~6 s per sample.

    Come l'originale, la tmpdir viene ripulita qui se l'estrazione solleva.
    """
    import shutil
    import tempfile

    import decord
    import numpy as np
    from PIL import Image

    vr = decord.VideoReader(video_path)
    last = len(vr) - 1
    clamped = [max(0, min(int(i), last)) for i in indices]
    uniq = sorted(set(clamped))
    batch = vr.get_batch(uniq).asnumpy()
    by_index = {idx: batch[k] for k, idx in enumerate(uniq)}

    tmp_dir = Path(tempfile.mkdtemp(prefix="pair_frames_"))
    try:
        paths = []
        for i, idx in enumerate(clamped):
            im = Image.fromarray(np.asarray(by_index[idx]))
            if target is not None and im.size != (target[1], target[0]):
                im = im.resize((target[1], target[0]), Image.BICUBIC)
            p = tmp_dir / f"frame_{i:04d}.png"
            im.save(p)
            paths.append(str(p))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return paths, tmp_dir


# Cache di UN solo video: i frame dipendono dal video e dai parametri di
# campionamento, non dalla domanda, e LVBench ha ~15 domande per video (che il
# loader emette consecutive). Tenere l'ultimo estratto evita di ri-decodificare
# lo stesso video per ogni domanda; più di uno non serve e occuperebbe disco.
_CACHE: dict | None = None


def _drop_cache() -> None:
    global _CACHE
    if _CACHE is not None:
        import shutil

        shutil.rmtree(_CACHE["tmp_dir"], ignore_errors=True)
        _CACHE = None


def pair_video_frames(
    video_path: str,
    nframes: int,
    gap_sec: float,
    max_pixels: int,
    min_pixels: int | None,
    *,
    image_patch_size: int | None = None,
    cache: bool = True,
) -> tuple["VideoFrames", "Path | None", list[float]]:
    """`nframes` pari (= 2 * n_pairs): legge total_frames/fps con decord, calcola centri e
    indici, estrae i frame in PNG e ritorna (VideoFrames con frames_indices/fps REALI,
    tmp_dir da cancellare **o `None` se la tiene la cache**, centri in secondi).

    `nframes` dispari → ValueError: il processor padderebbe l'ultima cella
    duplicando l'ultimo frame e l'ultima coppia non sarebbe più una coppia.

    `image_patch_size` (16 su Qwen3-VL, 14 su Qwen2.5-VL): se passato, i PNG
    sono scritti già alla dimensione finale calcolata da
    `models.qwen.videoframes_target_size` — stesso risultato, molto più veloce
    (vedi `_extract_pairs`). Con `None` si scrivono a risoluzione nativa e
    ridimensiona il modello.

    `cache=True` (default) tiene i frame dell'ULTIMO video estratto: una
    seconda domanda sullo stesso video li riusa e `tmp_dir` torna `None` (non
    va cancellata dal chiamante; la libera la chiamata successiva su un altro
    video, o l'uscita del processo). Le domande dello stesso video sono
    consecutive nel loader LVBench, quindi basta tenerne uno.

    Il resize a max/min_pixels lo fa il modello: con `VideoFrames`
    qwen-vl-utils 0.0.14 arrotonda a multipli di 64, serve il knob
    `fix_videoframes_resize=True` (vedi `models/qwen.py::_fetch_videoframes`).
    Con `max_pixels` sotto il floor video di qwen-vl-utils (128 token/frame,
    es. 50176 del preset lvbench) `min_pixels` va passato esplicito (lvbench:
    3136): con None `smart_resize` asserisce `max_pixels >= min_pixels` al
    momento di `_prepare_inputs`, non qui.
    """
    global _CACHE
    import atexit
    import shutil

    import decord

    # Import pigri: questo modulo lo importa una strategy in `strategies/`,
    # a livello di modulo sarebbe un import circolare.
    from models.media import VideoFrames
    from models.qwen import videoframes_target_size

    if nframes < 2 or nframes % 2 != 0:
        raise ValueError(
            f"pair_video_frames: nframes deve essere pari e >= 2 (= 2 * n_pairs), "
            f"ricevuto {nframes} — con un conteggio dispari il processor duplica "
            "l'ultimo frame e l'ultima cella non è più una coppia."
        )

    key = (video_path, nframes, gap_sec, max_pixels, min_pixels, image_patch_size)
    if cache and _CACHE is not None and _CACHE["key"] == key:
        hit = _CACHE
        media = VideoFrames(
            hit["paths"], max_pixels=max_pixels, min_pixels=min_pixels,
            frames_indices=hit["indices"], fps=hit["fps"],
        )
        return media, None, list(hit["centers"])

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    height, width, _ = vr[0].shape
    del vr

    centers, indices = pair_centers_and_indices(total_frames, fps, nframes // 2, gap_sec)
    target = None
    if image_patch_size is not None:
        target = videoframes_target_size(
            nframes, height, width, image_patch_size,
            max_pixels=max_pixels, min_pixels=min_pixels,
        )
    paths, tmp_dir = _extract_pairs(video_path, indices, target)
    indices = [int(i) for i in indices]
    try:
        media = VideoFrames(
            paths,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            frames_indices=indices,
            fps=fps,
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if not cache:
        return media, tmp_dir, centers
    if _CACHE is None:
        atexit.register(_drop_cache)
    else:
        _drop_cache()
    _CACHE = {"key": key, "paths": paths, "indices": indices, "fps": fps,
              "centers": centers, "tmp_dir": tmp_dir}
    return media, None, centers


def pair_cells_in_window(centers_sec: list[float], gap_sec: float, w0: float, w1: float) -> list[bool]:
    """Per l'analisi offline: cella i vera se [c_i − gap/2, c_i + gap/2] interseca [w0, w1].

    Intervalli chiusi (il contatto in un punto conta come intersezione). Usa
    il gap NOMINALE attorno al centro effettivo: approssima di al più
    `0.5/fps` la copertura reale della coppia (più il clamp ai bordi).
    """
    if w1 < w0:
        raise ValueError(f"finestra vuota: w0={w0} > w1={w1}")
    half = gap_sec / 2
    return [c - half <= w1 and c + half >= w0 for c in centers_sec]
