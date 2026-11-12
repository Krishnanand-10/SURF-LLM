"""
Knowledge Localization Module for SURF-LLM.
Provides gradient saliency, Fisher information attribution, activation patching,
and layer sensitivity heatmap generation.
"""

from .saliency import LayerSaliencyProfiler
from .causal_trace import ActivationPatchingProfiler, CausalTraceProfiler
from .heatmap import generate_layer_heatmaps, save_attribution_report

__all__ = [
    "LayerSaliencyProfiler",
    "ActivationPatchingProfiler",
    "CausalTraceProfiler",
    "generate_layer_heatmaps",
    "save_attribution_report",
]
