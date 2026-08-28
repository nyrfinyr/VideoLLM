"""Registro delle strategy — importare qui ogni sottoclasse di `Strategy`
per side-effect di registrazione (`Strategy.get` scansiona
`__subclasses__()`), stesso pattern di `evals/__init__.py`.
"""
from .attention_highlight import AttentionHighlightStrategy
from .attention_marker import AttentionMarkerStrategy
from .base import SamplingBudget, Strategy
from .entropy_attention_resample import EntropyAttentionResampleStrategy
from .entropy_shortcut import EntropyShortcutStrategy
from .uniform import UniformStrategy

__all__ = [
    "AttentionHighlightStrategy",
    "AttentionMarkerStrategy",
    "EntropyAttentionResampleStrategy",
    "EntropyShortcutStrategy",
    "SamplingBudget",
    "Strategy",
    "UniformStrategy",
]
