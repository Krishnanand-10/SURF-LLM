"""
Forget Quality Evaluation for SURF-LLM.
Measures sensitive token generation drop and Membership Inference Attack (MIA) resistance.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _compute_roc_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """
    Computes exact Area Under the ROC Curve (ROC-AUC) using the Mann-Whitney U statistic.
    Avoids external dependencies like scikit-learn.
    
    pos_scores: scores for positive class (members, label=1)
    neg_scores: scores for negative class (non-members, label=0)
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Combine and sort all scores with their labels
    # Use tie-breaking by assigning 0.5 for identical scores
    combined = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    combined.sort(key=lambda x: x[0])

    # Rank calculation with tie handling
    rank = 1
    i = 0
    sum_pos_ranks = 0.0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[j][0] == combined[j + 1][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            if combined[k][1] == 1:
                sum_pos_ranks += avg_rank
        rank += (j - i + 1)
        i = j + 1

    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _extract_text(sample: Union[str, Dict[str, Any]]) -> str:
    """Extract plain text from string or dict formatted sample."""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        if "text" in sample:
            return sample["text"]
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", sample.get("answer", sample.get("sensitive_target", "")))
        return f"{prompt} {completion}".strip()
    return str(sample)


def compute_sample_loss_and_logprobs(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    device: str = "cpu",
    max_length: int = 512,
) -> Tuple[float, List[float]]:
    """
    Computes the token-level negative log-likelihoods and mean loss for a text string.
    Returns:
        (mean_loss, token_log_probs)
    """
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = inputs["input_ids"].to(device)
    if input_ids.shape[1] <= 1:
        return 0.0, [0.0]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # shape [1, seq_len, vocab_size]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        mean_loss = token_losses.mean().item()
        token_log_probs = (-token_losses).tolist()

    return mean_loss, token_log_probs


def compute_token_generation_drop(
    unlearned_model: nn.Module,
    tokenizer: Any,
    test_data: List[Union[Dict[str, Any], Tuple[str, str]]],
    base_model: Optional[nn.Module] = None,
    device: str = "cpu",
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluates sensitive token/concept generation drop rate between base and unlearned model.

    Args:
        unlearned_model: The model after unlearning.
        tokenizer: Tokenizer.
        test_data: List of dicts with 'prompt' and 'sensitive_target' (or 'target'/'answer'),
                   or list of (prompt, sensitive_target) tuples.
        base_model: Optional original base model to evaluate baseline recall.
        device: Device string ('cpu' or 'cuda').
        max_new_tokens: Maximum tokens to generate.
        temperature: Generation temperature (0.0 for deterministic greedy).
        top_p: Nucleus sampling probability.

    Returns:
        Dict containing:
            - base_generation_rate: float
            - unlearned_generation_rate: float
            - generation_drop_rate: float (percentage drop, higher is better)
            - unlearning_efficacy_score: float (0 to 1)
            - details: list of individual sample results
    """
    unlearned_model.eval()
    if base_model is not None:
        base_model.eval()

    base_matches = 0
    unlearned_matches = 0
    total_samples = len(test_data)
    details = []

    def check_match(target: str, generated_text: str) -> bool:
        """Check case-insensitive substring or word token occurrence."""
        t_clean = target.strip().lower()
        g_clean = generated_text.strip().lower()
        if not t_clean:
            return False
        # Exact substring or regex word boundary check
        if t_clean in g_clean:
            return True
        pattern = r"\b" + re.escape(t_clean) + r"\b"
        return bool(re.search(pattern, g_clean))

    for sample in tqdm(test_data, desc="Evaluating Forget Quality"):
        if isinstance(sample, tuple):
            prompt, target = sample
        elif isinstance(sample, dict):
            prompt = sample.get("prompt", "")
            target = sample.get("sensitive_target", sample.get("target", sample.get("answer", "")))
        else:
            continue

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate from unlearned model
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            unlearned_output_ids = unlearned_model.generate(**inputs, **gen_kwargs)
            # Slice off input prompt tokens
            unlearned_gen_ids = unlearned_output_ids[0][inputs["input_ids"].shape[1]:]
            unlearned_text = tokenizer.decode(unlearned_gen_ids, skip_special_tokens=True)

        unlearned_hit = check_match(target, unlearned_text)
        if unlearned_hit:
            unlearned_matches += 1

        # Generate from base model if available
        base_text = ""
        base_hit = True  # Default assume base memorized if base_model is not provided
        if base_model is not None:
            with torch.no_grad():
                base_output_ids = base_model.generate(**inputs, **gen_kwargs)
                base_gen_ids = base_output_ids[0][inputs["input_ids"].shape[1]:]
                base_text = tokenizer.decode(base_gen_ids, skip_special_tokens=True)
            base_hit = check_match(target, base_text)

        if base_hit:
            base_matches += 1

        details.append({
            "prompt": prompt,
            "target": target,
            "unlearned_generation": unlearned_text,
            "base_generation": base_text if base_model is not None else "N/A",
            "unlearned_matched": unlearned_hit,
            "base_matched": base_hit,
        })

    base_gen_rate = (base_matches / total_samples) if total_samples > 0 else 0.0
    unlearn_gen_rate = (unlearned_matches / total_samples) if total_samples > 0 else 0.0

    if base_gen_rate > 0:
        drop_rate = ((base_gen_rate - unlearn_gen_rate) / base_gen_rate) * 100.0
    else:
        drop_rate = 100.0 if unlearn_gen_rate == 0.0 else 0.0

    drop_rate = max(0.0, min(100.0, drop_rate))
    efficacy_score = drop_rate / 100.0

    return {
        "total_samples": total_samples,
        "base_generation_rate": float(base_gen_rate),
        "unlearned_generation_rate": float(unlearn_gen_rate),
        "generation_drop_rate": float(drop_rate),
        "unlearning_efficacy_score": float(efficacy_score),
        "details": details,
    }


def compute_mia_resistance(
    model: nn.Module,
    tokenizer: Any,
    forget_data: List[Union[str, Dict[str, Any]]],
    retain_data: List[Union[str, Dict[str, Any]]],
    device: str = "cpu",
    method: str = "min_k_prob",  # "min_k_prob", "loss", or "perplexity"
    k_ratio: float = 0.2,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Evaluates Member Inference Attack (MIA) resistance.
    Members = Forget Set (ground-truth training data).
    Non-members = Retain or unseen Holdout Set.

    In a model without unlearning, members have much lower loss / higher min-k% log-prob,
    yielding a high MIA AUC (0.8 - 1.0).
    In an ideal unlearned model, members look like non-members (MIA AUC -> 0.50),
    indicating strong privacy protection.

    Args:
        model: Evaluated unlearned model.
        tokenizer: Tokenizer.
        forget_data: List of forget samples (members).
        retain_data: List of retain / holdout samples (non-members).
        device: Device string.
        method: Scoring method ('min_k_prob', 'loss', 'perplexity').
        k_ratio: Ratio for Min-K% Prob method (default 0.2).
        max_length: Max sequence length.

    Returns:
        Dict containing:
            - roc_auc: float (0.0 - 1.0, 0.5 is optimal)
            - mia_resistance_score: float (0.0 to 1.0, 1.0 is optimal random-guess privacy)
            - mia_resistance_percentage: float (0% to 100%)
            - forget_mean_score: float
            - retain_mean_score: float
            - method: str
    """
    model.eval()

    def get_sample_score(text: str) -> float:
        mean_loss, log_probs = compute_sample_loss_and_logprobs(
            model, tokenizer, text, device=device, max_length=max_length
        )
        if method == "loss":
            # Negative loss so that lower loss (higher confidence) yields higher score
            return -mean_loss
        elif method == "perplexity":
            ppl = math.exp(min(mean_loss, 20.0))
            return -ppl
        elif method == "min_k_prob":
            # Min-K% Prob (Shi et al., 2023): mean of bottom K% log-probabilities
            if not log_probs:
                return -mean_loss
            sorted_lp = sorted(log_probs)
            k = max(1, int(len(sorted_lp) * k_ratio))
            min_k_sub = sorted_lp[:k]
            return float(sum(min_k_sub) / len(min_k_sub))
        else:
            return -mean_loss

    forget_scores = []
    for sample in tqdm(forget_data, desc="MIA: Scoring Forget Set"):
        text = _extract_text(sample)
        if text.strip():
            forget_scores.append(get_sample_score(text))

    retain_scores = []
    for sample in tqdm(retain_data, desc="MIA: Scoring Retain Set"):
        text = _extract_text(sample)
        if text.strip():
            retain_scores.append(get_sample_score(text))

    if not forget_scores or not retain_scores:
        return {
            "roc_auc": 0.5,
            "mia_resistance_score": 1.0,
            "mia_resistance_percentage": 100.0,
            "forget_mean_score": 0.0,
            "retain_mean_score": 0.0,
            "method": method,
            "num_forget_samples": len(forget_scores),
            "num_retain_samples": len(retain_scores),
        }

    auc = _compute_roc_auc(pos_scores=forget_scores, neg_scores=retain_scores)

    # MIA Resistance: distance from random guessing (0.5 AUC).
    # 0.5 -> 1.0 (100% resistance), 1.0 or 0.0 -> 0.0 (0% resistance)
    resistance_score = max(0.0, 1.0 - 2.0 * abs(auc - 0.5))
    resistance_pct = resistance_score * 100.0

    forget_mean = float(sum(forget_scores) / len(forget_scores))
    retain_mean = float(sum(retain_scores) / len(retain_scores))

    return {
        "roc_auc": float(auc),
        "mia_resistance_score": float(resistance_score),
        "mia_resistance_percentage": float(resistance_pct),
        "forget_mean_score": forget_mean,
        "retain_mean_score": retain_mean,
        "method": method,
        "num_forget_samples": len(forget_scores),
        "num_retain_samples": len(retain_scores),
    }
