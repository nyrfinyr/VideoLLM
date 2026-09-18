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
    nominal = [duration * (i + 0.5) / n_pairs for i in range(n_pairs)]
    return _pairs_from_centers(total_frames, fps, nominal, gap_sec)


def _pairs_from_centers(
    total_frames: int, fps: float, nominal_centers: list[float], gap_sec: float
) -> tuple[list[float], list[int]]:
    """Nucleo condiviso: da centri NOMINALI a `(centri effettivi, indici interleaved)`.

    Usato sia dal campionamento uniforme (`pair_centers_and_indices`) sia da
    quello per regioni del pass 2 (`pairs_in_spans`): la regola su clamp,
    coppie degeneri e centro effettivo deve essere UNA sola, altrimenti i
    timestamp del pass 1 e del pass 2 seguirebbero convenzioni diverse.
    """
    last = total_frames - 1
    t_last = last / fps

    def to_index(t: float) -> int:
        t = min(max(t, 0.0), t_last)
        return min(max(int(round(t * fps)), 0), last)

    centers: list[float] = []
    indices: list[int] = []
    for c in nominal_centers:
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


def pairs_in_spans(
    total_frames: int,
    fps: float,
    spans: list[tuple[float, float]],
    counts: list[int],
    gap_sec: float,
) -> tuple[list[float], list[int]]:
    """Coppie dentro REGIONI disgiunte (pass 2 dello zoom): `counts[j]` coppie
    uniformi dentro `spans[j]`, poi TUTTE le coppie ordinate per centro.

    Perché coppie anche qui, e ordinate: il processor Qwen3-VL fonde i frame
    a due a due nell'ordine della lista, quindi una cella del pass 2 resta
    "un istante" solo se i due frame di ogni coppia sono adiacenti NELLA
    LISTA. Ordinare per centro (le regioni sono disgiunte, le coppie sono
    strette) tiene la lista monotona nel tempo senza mai spezzare una coppia:
    nessuna cella a cavallo di due regioni, che avrebbe un timestamp — la
    media dei due frame — dentro un buco mai osservato.

    `counts[j] == 0` salta la regione. Regione più corta del gap: i centri si
    addensano e `_pairs_from_centers` risolve le coppie degeneri come altrove.
    """
    if len(spans) != len(counts):
        raise ValueError(f"{len(spans)} span ma {len(counts)} counts")
    nominal: list[float] = []
    for (t0, t1), n in zip(spans, counts):
        if n <= 0:
            continue
        if t1 < t0:
            raise ValueError(f"span invertito: {(t0, t1)}")
        nominal += [t0 + (t1 - t0) * (j + 0.5) / n for j in range(n)]
    nominal.sort()
    if not nominal:
        raise ValueError("nessuna coppia da campionare: tutti i counts sono 0")
    return _pairs_from_centers(total_frames, fps, nominal, gap_sec)


def extract_frames(video_path: str, indices: list[int], target: tuple[int, int] | None) -> tuple[list[str], "Path"]:
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
    (vedi `extract_frames`). Con `None` si scrivono a risoluzione nativa e
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
    paths, tmp_dir = extract_frames(video_path, indices, target)
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


def cells_to_spans(
    cell_idx: list[int], n_cells: int, duration_sec: float
) -> list[tuple[float, float]]:
    """Celle del pass 1 → intervalli di tempo che "possiedono" (pass 2).

    La cella `i` di `n_cells` centri uniformi copre
    `[D*i/n, D*(i+1)/n]`: è la cella di Voronoi del suo centro
    `D*(i+0.5)/n`, cioè tutto il tempo più vicino a quel centro che a
    qualunque altro. Ricampionare lì dentro è esattamente "infittisci dove
    l'attenzione ha guardato", senza assumere nulla sulla finestra vera.

    Ritorna gli span nell'ordine dato, SENZA fonderli: celle adiacenti danno
    span adiacenti e le coppie del pass 2 restano dentro la propria cella.
    """
    if duration_sec <= 0:
        raise ValueError(f"durata non valida: {duration_sec}")
    w = duration_sec / n_cells
    out = []
    for i in cell_idx:
        if not 0 <= i < n_cells:
            raise ValueError(f"cella {i} fuori da [0, {n_cells})")
        out.append((i * w, (i + 1) * w))
    return out


def span_video_frames(
    video_path: str,
    spans: list[tuple[float, float]],
    counts: list[int],
    gap_sec: float,
    max_pixels: int,
    min_pixels: int | None,
    *,
    image_patch_size: int | None = None,
) -> tuple["VideoFrames", "Path", list[float]]:
    """Pass 2: `counts[j]` coppie dentro `spans[j]` → `(VideoFrames, tmp_dir, centri)`.

    Gemella di `pair_video_frames` per le regioni: stessa estrazione in blocco,
    stesso pre-resize alla dimensione finale, stessi `frames_indices`/`fps`
    REALI (quindi i timestamp che il processor scrive nel prompt sono i tempi
    veri dei frame zoomati, non una densità finta iniettata via `sample_fps`).

    Niente cache: a differenza del pass 1 le regioni dipendono dalla DOMANDA,
    non solo dal video, quindi due domande sullo stesso video non le
    condividono. `tmp_dir` va sempre ripulita dal chiamante.
    """
    import shutil

    import decord

    from models.media import VideoFrames
    from models.qwen import videoframes_target_size

    n_pairs = sum(max(0, c) for c in counts)
    if n_pairs < 1:
        raise ValueError("span_video_frames: budget nullo (tutti i counts a 0)")

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    height, width, _ = vr[0].shape
    del vr

    centers, indices = pairs_in_spans(total_frames, fps, spans, counts, gap_sec)
    target = None
    if image_patch_size is not None:
        target = videoframes_target_size(
            len(indices), height, width, image_patch_size,
            max_pixels=max_pixels, min_pixels=min_pixels,
        )
    paths, tmp_dir = extract_frames(video_path, indices, target)
    try:
        media = VideoFrames(
            paths, max_pixels=max_pixels, min_pixels=min_pixels,
            frames_indices=[int(i) for i in indices], fps=fps,
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return media, tmp_dir, centers


# ─────────────────────────────────────────────────────────────────────────────
# Pass ADDITIVO: i frame aggiunti NON sostituiscono la base
# ─────────────────────────────────────────────────────────────────────────────
def merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intervalli ordinati e FUSI quando si toccano o si sovrappongono.

    Serve perché le celle selezionate dal ranking sono spesso adiacenti (a
    k=10 le 10 celle diventano ~7 regioni): senza fusione il confine fra due
    celle contigue riceverebbe due mezzi budget e il campionamento uniforme
    avrebbe un buco proprio lì.
    """
    if not spans:
        return []
    for a, b in spans:
        if b < a:
            raise ValueError(f"span invertito: {(a, b)}")
    ordered = sorted(spans)
    out: list[list[float]] = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _allocate(n: int, weights: list[float]) -> list[int]:
    """`n` frame fra regioni di lunghezza `weights`, proporzionale al tempo
    coperto (metodo dei resti più grandi, quindi la somma fa ESATTAMENTE `n`).

    Proporzionale e non in parti uguali: dopo la fusione le regioni hanno
    lunghezze diverse, e un budget uguale darebbe densità diverse — cioè
    proprio la variabile che vogliamo tenere sotto controllo.
    """
    tot = sum(weights)
    if tot <= 0:
        raise ValueError("regioni di lunghezza nulla")
    raw = [n * w / tot for w in weights]
    base = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: base[i] - raw[i])
    for i in order[: n - sum(base)]:
        base[i] += 1
    return base


def additive_indices(
    total_frames: int,
    fps: float,
    base_indices: list[int],
    spans: list[tuple[float, float]],
    n_added: int,
) -> tuple[list[int], dict]:
    """Indici del pass ADDITIVO: `base_indices` PIÙ `n_added` frame uniformi
    dentro `spans`, ordinati per tempo. Funzione pura.

    Differenze deliberate rispetto al pass 2 sostitutivo (`pairs_in_spans`):

    - i frame aggiunti sono **uniformi dentro la regione, non a coppie**: la
      struttura a coppie serve a rendere le celle indirizzabili per il
      ranking del pass 1, e qui non si rilegge nessun ranking;
    - la base viene tenuta **così com'è**, duplicati inclusi: le sue coppie
      sono già state decise da `pair_centers_and_indices` e toccarle
      spezzerebbe la corrispondenza cella ↔ coppia;
    - un frame aggiunto che cade sullo STESSO indice di uno della base (o di
      un altro aggiunto) viene **scartato**: ripagarlo non aggiunge
      informazione e costerebbe token.

    ⚠️ Nella lista risultante il merge temporale del modello (`temporal_patch
    _size=2`) accoppia frame ADIACENTI NELLA LISTA, quindi le coppie della
    base non sopravvivono: le celle del pass additivo sono coppie qualsiasi.
    È accettabile finché sul pass additivo non si rilegge l'attenzione — se
    un giorno la si rileggesse, le celle non sarebbero più quelle del pass 1.

    La lunghezza finale è PARI (il processor padderebbe duplicando l'ultimo
    frame): se la dedup lascia un numero dispari si scarta l'aggiunto più
    ridondante, cioè quello col vicino più vicino nella lista finale.

    Returns: `(indici ordinati, info)` con `info` = conteggi utili al log —
    `n_base`, `n_added_requested`, `n_added_kept`, `n_collision`, `n_parity`,
    `spans` (fusi) e `counts` per regione.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames deve essere >= 1, ricevuto {total_frames}")
    if not fps > 0:
        raise ValueError(f"fps deve essere > 0, ricevuto {fps}")
    if len(base_indices) % 2:
        raise ValueError(f"la base deve avere un numero pari di frame, non {len(base_indices)}")
    if n_added < 0:
        raise ValueError(f"n_added deve essere >= 0, ricevuto {n_added}")

    merged_spans = merge_spans(spans)
    last = total_frames - 1
    added: list[int] = []
    counts: list[int] = []
    if n_added and merged_spans:
        counts = _allocate(n_added, [b - a for a, b in merged_spans])
        seen = set(base_indices)
        for (t0, t1), cnt in zip(merged_spans, counts):
            if cnt <= 0:
                continue
            step = (t1 - t0) / cnt
            for j in range(cnt):
                idx = min(max(int(round((t0 + (j + 0.5) * step) * fps)), 0), last)
                if idx in seen:
                    continue
                seen.add(idx)
                added.append(idx)

    n_collision = n_added - len(added)
    n_parity = 0
    out = sorted(base_indices + added)
    if len(out) % 2 and added:
        # Il più ridondante = quello col vicino più vicino: toglierlo è la
        # perdita di informazione minima fra gli aggiunti.
        pos = {i: k for k, i in enumerate(out)}
        worst = min(added, key=lambda i: min(
            (out[pos[i]] - out[pos[i] - 1]) if pos[i] > 0 else 10 ** 9,
            (out[pos[i] + 1] - out[pos[i]]) if pos[i] + 1 < len(out) else 10 ** 9,
        ))
        added.remove(worst)
        out.remove(worst)
        n_parity = 1
    info = {
        "n_base": len(base_indices),
        "n_added_requested": n_added,
        "n_added_kept": len(added),
        "n_collision": n_collision,
        "n_parity": n_parity,
        "n_total": len(out),
        "spans": merged_spans,
        "counts": counts,
        "added_indices": added,
    }
    return out, info


def frames_for_plans(
    video_path: str,
    plans: dict[str, list[int]],
    max_pixels: int,
    min_pixels: int | None,
    *,
    image_patch_size: int | None = None,
) -> tuple[dict[str, "VideoFrames"], "Path", float]:
    """Più liste di frame dello stesso video, con UN SOLO decode e UN SOLO set
    di PNG condiviso.

    Il pass additivo confronta condizioni che si sovrappongono quasi del tutto
    (base ⊂ additivo, e l'uniforme a budget pieno ricampiona gli stessi
    istanti): estrarle una per una vorrebbe dire decodificare lo stesso video
    tre volte e riscrivere gli stessi frame. Qui l'unione degli indici passa
    da `extract_frames` una volta sola — che già legge in blocco e deduplica —
    e ogni piano riceve la sua `VideoFrames` con i PNG condivisi, i propri
    `frames_indices` e l'fps REALE.

    ⚠️ Condizione di validità: i piani devono finire alla STESSA dimensione,
    altrimenti i PNG condivisi sarebbero giusti per uno e sbagliati per gli
    altri. Con `max_pixels` esplicito (preset lvbench: 50176) è sempre vero —
    il tetto derivato dal budget totale non morde nemmeno a 1024 frame — ma è
    una proprietà del preset, non una legge: qui viene VERIFICATA e, se cade,
    la funzione solleva invece di far girare il modello su pixel diversi dal
    previsto.

    Returns: `({nome: VideoFrames}, tmp_dir da cancellare, fps)`.
    """
    import decord

    from models.media import VideoFrames
    from models.qwen import videoframes_target_size

    if not plans:
        raise ValueError("nessun piano di frame")
    for name, idx in plans.items():
        if not idx:
            raise ValueError(f"piano {name!r} vuoto")
        if len(idx) % 2:
            raise ValueError(f"piano {name!r}: {len(idx)} frame, deve essere PARI")

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    height, width, _ = vr[0].shape
    del vr

    target = None
    if image_patch_size is not None:
        targets = {
            name: videoframes_target_size(
                len(idx), height, width, image_patch_size,
                max_pixels=max_pixels, min_pixels=min_pixels,
            )
            for name, idx in plans.items()
        }
        if len(set(targets.values())) > 1:
            raise RuntimeError(
                "i piani finirebbero a dimensioni diverse "
                f"({targets}): i PNG non sono condivisibili. Serve un'estrazione "
                "per piano, oppure un `max_pixels` che morda su tutti i budget."
            )
        target = next(iter(targets.values()))

    last = total_frames - 1
    plans = {n: [min(max(int(i), 0), last) for i in idx] for n, idx in plans.items()}
    union = sorted({i for idx in plans.values() for i in idx})
    paths, tmp_dir = extract_frames(video_path, union, target)
    by_index = dict(zip(union, paths))
    try:
        media = {
            name: VideoFrames(
                [by_index[i] for i in idx], max_pixels=max_pixels, min_pixels=min_pixels,
                frames_indices=list(idx), fps=fps,
            )
            for name, idx in plans.items()
        }
    except Exception:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return media, tmp_dir, fps


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
