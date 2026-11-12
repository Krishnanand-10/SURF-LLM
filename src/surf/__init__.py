"""
SURF-LLM: Sparse Unlearning with Reasoning-Preserving Framework for Large Language Models
Author: Krishnanand Tiwari (USN: SCA25MCA109)
"""

from .adapters.sparse_lora import SparseLoRAInjector
from .trainer.objectives import DualObjectiveLoss
from .trainer.trainer import SURFTrainer, UnlearningTrainer

__version__ = "0.1.0"

__all__ = [
    "SparseLoRAInjector",
    "DualObjectiveLoss",
    "SURFTrainer",
    "UnlearningTrainer",
]
