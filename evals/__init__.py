# Importing each subclass module side-effect-registers it in
# `Dataset.__subclasses__()`, che è ciò su cui `Dataset.get(name)`
# dispatcha. Non rimuovere gli import "inutilizzati" senza switchare
# anche a un meccanismo esplicito di registrazione.
from .base import Dataset, format_mcq_prompt, mcq_accuracy, parse_mcq_letter
from .causal2needles import CausalNeedles
from .egoschema import EgoSchema
from .lvbench import LVBench
from .mvbench import MVBench
from .video_mme import VideoMME

__all__ = [
    "CausalNeedles",
    "Dataset",
    "EgoSchema",
    "LVBench",
    "MVBench",
    "VideoMME",
    "format_mcq_prompt",
    "mcq_accuracy",
    "parse_mcq_letter",
]
