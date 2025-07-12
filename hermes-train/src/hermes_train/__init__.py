"""PyTorch training for Hermes LLMs (MAL-defined architectures)."""

from hermes_train.config import ModelDef
from hermes_train.model import Transformer
from hermes_train.muon import Muon

__all__ = ["ModelDef", "Muon", "Transformer"]
