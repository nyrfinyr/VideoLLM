"""`EntropyAttentionResampleStrategy` — primo tentativo di sfruttare i
segnali di entropia risposta MCQ + concentrazione attenzione (sink-
filtrata) per decidere se ricampionare i frame, invece di limitarsi a
loggarli. Pensata per un rilancio di MVBench a confronto con la baseline
in `README.md`.

Logica (pass 1 sempre uniforme, budget `nframes` invariato):

    entropia bassa                → accetta la risposta del pass 1
    entropia alta, conc. bassa    → ricampiona, SFASATO di fase rispetto
                                     ai frame già visti (dispersione: non
                                     si sa dove guardare, si prova altrove)
    entropia alta, conc. alta     → ricampiona, INFITTITO in una finestra
                                     stretta intorno al frame di picco
                                     (il modello guarda "quasi giusto" ma
                                     non abbastanza in dettaglio)

Al massimo UN ricampionamento (non c'è un terzo pass che ri-verifica il
segnale). La risposta FINALE — sia nei rami "accetta" sia dopo un
ricampionamento — viene SEMPRE da una vera `vlm.generate()` (mai dal solo
`pred_letter` della softmax ristretta): così i rami "accetta" restano
bit-per-bit equivalenti alla baseline `uniform`, isolando l'effetto del
resampling ai soli casi in cui avviene davvero.

Soglie NON validate per MVBench (i numeri raccolti in
`docs/entropia_confidenza_mcq.md` vengono da un campione VNBench
"retrieval" diverso per dominio/nframes) — sono knob del preset
`strategy.entropy_attention_resample` in `conf/config.yaml`, tarabili da
CLI senza toccare codice, es. `strategy.entropy_threshold=1.0`.

Richiede un modello che implementa `SupportsSignals` (oggi solo
`model=qwen2_5_vl_3b_attn`) — fail-fast altrimenti, stesso pattern di
`entropy_shortcut`. Se `frames` è già fissato dal dataset (nessuna libertà
di selezione, es. MVBench `data_type="frame"`) il resampling è per
definizione impossibile: si accetta sempre il pass 1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from models.media import Text
from models.signals import SupportsSignals
from utils.attn_core import GridSpec, rank_cells, sink_view
from utils.mcq import parse_mcq_letter

from .base import SamplingBudget, Strategy

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from models.base import BaseVLM
    from models.signals import SignalAnswer


def _video_duration_sec(video_path: str) -> float:
    """Durata video in secondi via decord (solo metadata, non decodifica
    i frame) — serve per clampare le finestre di resample ai bordi reali
    del video quando `video_start`/`video_end` non sono già noti dal
    dataset (caso comune su MVBench, `bound=False`)."""
    import decord

    vr = decord.VideoReader(video_path)
    return len(vr) / vr.get_avg_fps()


def _peak_cell(visual_attention, *, percentile: float, border: int) -> tuple[float, int]:
    """`(top1_pct, cella_di_picco)` dalla mappa di attenzione COMPLETA
    (media su tutti i token della domanda), sink-filtrata — stessa logica
    di `utils.attn_core.attention_concentration`, ma operante direttamente
    su `VisualAttention` (niente `AttentionCapture`/fps/frame_indices
    reali: qui serve solo l'indice di cella, non il frame-index vero — la
    posizione temporale si ricava per via frazionaria in `answer()`).
    `frame_indices` dummy (`range(t)`, `double=True`) perché `cell_to_frames`
    non viene mai ispezionato per `frames`/`rep_frame`/`t_sec`.
    """
    heat = visual_attention.attn.float().mean(dim=0)  # [t, grid_h, grid_w]
    grid = GridSpec(t=visual_attention.t, grid_h=visual_attention.grid_h, grid_w=visual_attention.grid_w, double=True)
    heat_nonsink, _ = sink_view(heat, visual_attention.sink_map, view="nonsink", percentile=percentile, border=border)
    ranked = rank_cells(heat_nonsink, grid, list(range(visual_attention.t)), fps=1.0, metric="sum", topk=1)
    top1 = ranked[0]
    return top1.pct, top1.cell


class EntropyAttentionResampleStrategy(Strategy):
    name = "entropy_attention_resample"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.entropy_threshold = float(cfg.get("entropy_threshold", 0.7))
        self.concentration_threshold = float(cfg.get("concentration_threshold", 30.0))
        self.window_frac = float(cfg.get("window_frac", 0.3))
        self.phase_shift_frac = float(cfg.get("phase_shift_frac", 0.5))
        self.sink_percentile = float(cfg.get("sink_percentile", 25.0))
        self.sink_border = int(cfg.get("sink_border", 0))

    def answer(
        self,
        vlm: BaseVLM,
        *,
        video_path: str,
        prompt: str,
        options: list[str] | None,
        gen_cfg: GenerationConfig,
        budget: SamplingBudget,
        video_start: float | None = None,
        video_end: float | None = None,
        frames: list[str] | None = None,
    ) -> dict:
        if not isinstance(vlm, SupportsSignals):
            raise RuntimeError(
                f"strategy {self.name!r} richiede un modello che implementa "
                f"SupportsSignals (segnali di confidenza/attenzione), ma "
                f"{type(vlm).__name__} non lo fa — usa model=qwen2_5_vl_3b_attn "
                "o un'altra strategy."
            )

        media = self._build_media(video_path, frames, video_start, video_end, budget)
        letters = [chr(ord("A") + i) for i in range(len(options))] if options is not None else None
        signal: SignalAnswer = vlm.generate_with_signals(media, Text(prompt), gen_cfg, answer_letters=letters)

        # top1_pct/peak_cell del pass 1: calcolati una volta, loggati SEMPRE
        # (anche nei rami "accetta") per poter analizzare a posteriori come
        # si distribuiscono rispetto alla decisione di resampling.
        top1_pct = peak_cell = None
        if signal.visual_attention is not None:
            top1_pct, peak_cell = _peak_cell(
                signal.visual_attention, percentile=self.sink_percentile, border=self.sink_border,
            )

        final_media = media
        resampled = False
        resample_kind: str | None = None

        can_resample = (
            frames is None
            and options is not None
            and signal.answer_entropy is not None
            and top1_pct is not None
            and signal.answer_entropy > self.entropy_threshold
        )
        if can_resample:
            t = signal.visual_attention.t
            duration = _video_duration_sec(video_path)
            w0 = video_start if video_start is not None else 0.0
            w1 = video_end if video_end is not None else duration

            if top1_pct < self.concentration_threshold:
                # Dispersa: sfasa la griglia uniforme di mezzo intervallo
                # rispetto al pass 1 — nuovi frame "in mezzo" a quelli già
                # visti, stesso budget. Clampata a [0, duration]: se il
                # pass 1 copriva già tutto il video, lo shift si limita
                # a restringere leggermente da sinistra.
                interval = (w1 - w0) / t if t else 0.0
                shift = interval * self.phase_shift_frac
                new_start = min(w0 + shift, duration)
                new_end = min(w1 + shift, duration)
                resample_kind = "phase_shift"
            else:
                # Concentrata ma comunque incerta: finestra stretta
                # centrata sull'istante del frame di picco (posizione
                # frazionaria della cella nella finestra del pass 1).
                frac = (peak_cell + 0.5) / t if t else 0.5
                peak_time = w0 + frac * (w1 - w0)
                half_w = self.window_frac * (w1 - w0) / 2
                new_start = max(0.0, peak_time - half_w)
                new_end = min(duration, peak_time + half_w)
                resample_kind = "zoom_peak"

            final_media = self._build_media(video_path, None, new_start, new_end, budget)
            resampled = True

            if self.capture_dir is not None:
                self._save_attention_capture(
                    vlm, video_path, prompt, signal.visual_attention, budget, w0, w1,
                    extra={"resampled": resampled, "resample_kind": resample_kind, "top1_pct": top1_pct},
                )

        messages = vlm.build_messages(final_media, Text(prompt))
        raw = vlm.generate(messages, generation_config=gen_cfg)

        result: dict = {
            "raw": raw,
            "answer_entropy": signal.answer_entropy,
            "top1_pct": top1_pct,
            "resampled": resampled,
            "resample_kind": resample_kind,
        }
        if options is not None:
            result["pred"] = parse_mcq_letter(raw, options)
        return result
