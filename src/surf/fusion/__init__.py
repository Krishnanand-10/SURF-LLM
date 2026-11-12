"""
Weight Delta Fusion Module for SURF-LLM.
"""

from .fusion import (
    WeightDeltaFuser,
    fuse_lora_to_base,
    verify_fusion_integrity,
)

__all__ = [
    "WeightDeltaFuser",
    "fuse_lora_to_base",
    "verify_fusion_integrity",
]
