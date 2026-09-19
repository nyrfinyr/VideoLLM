"""Registro delle strategy — importare qui ogni sottoclasse di `Strategy`
per side-effect di registrazione (`Strategy.get` scansiona
`__subclasses__()`), stesso pattern di `evals/__init__.py`.
"""
from .additive_topk import AdditiveTopkStrategy
from .attention_highlight import AttentionHighlightStrategy
from .attention_marker import AttentionMarkerStrategy
from .coarse_to_fine import CoarseToFineStrategy
from .base import SamplingBudget, Strategy
from .entropy_attention_resample import EntropyAttentionResampleStrategy
from .entropy_shortcut import EntropyShortcutStrategy
from .signals_capture import SignalsCaptureStrategy
from .topk_resample import TopkResampleStrategy
from .uniform import UniformStrategy
from .visual_prompt import VisualPromptStrategy

__all__ = [
    "AdditiveTopkStrategy",
    "AttentionHighlightStrategy",
    "AttentionMarkerStrategy",
    "CoarseToFineStrategy",
    "EntropyAttentionResampleStrategy",
    "EntropyShortcutStrategy",
    "SamplingBudget",
    "SignalsCaptureStrategy",
    "Strategy",
    "TopkResampleStrategy",
    "UniformStrategy",
    "VisualPromptStrategy",
]
