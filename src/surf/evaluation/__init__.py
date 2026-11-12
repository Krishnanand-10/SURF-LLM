"""
Evaluation and Auditing Module for SURF-LLM.
"""

from .forget_metrics import (
    compute_token_generation_drop,
    compute_mia_resistance,
    compute_sample_loss_and_logprobs,
)
from .reasoning_metrics import (
    compute_perplexity,
    compute_perplexity_divergence,
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_reasoning_parity,
    DEFAULT_MMLU_SAMPLES,
    DEFAULT_GSM8K_SAMPLES,
)
from .evaluator import SafetyAuditor, SURFEvaluator

__all__ = [
    "compute_token_generation_drop",
    "compute_mia_resistance",
    "compute_sample_loss_and_logprobs",
    "compute_perplexity",
    "compute_perplexity_divergence",
    "evaluate_mmlu",
    "evaluate_gsm8k",
    "evaluate_reasoning_parity",
    "DEFAULT_MMLU_SAMPLES",
    "DEFAULT_GSM8K_SAMPLES",
    "SafetyAuditor",
    "SURFEvaluator",
]
