"""
Reasoning Preservation Auditing for SURF-LLM.
Evaluates Retain Perplexity Divergence, MMLU Multi-Task Accuracy, and GSM8K Math Reasoning.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm


# Built-in representative test samples for MMLU
DEFAULT_MMLU_SAMPLES = [
    {
        "question": "What is the primary function of ribosomes in a biological cell?",
        "choices": [
            "Lipid synthesis",
            "Protein synthesis",
            "DNA replication",
            "Cellular respiration"
        ],
        "answer": "B",
        "subject": "biology"
    },
    {
        "question": "Which of the following sorting algorithms has an average time complexity of O(n log n)?",
        "choices": [
            "Bubble Sort",
            "Insertion Sort",
            "Merge Sort",
            "Selection Sort"
        ],
        "answer": "C",
        "subject": "computer_science"
    },
    {
        "question": "If a triangle has sides of length 3, 4, and 5, what type of triangle is it?",
        "choices": [
            "Equilateral",
            "Right-angled",
            "Isosceles",
            "Obtuse"
        ],
        "answer": "B",
        "subject": "mathematics"
    },
    {
        "question": "In formal logic, what is the contrapositive of the statement 'If P, then Q'?",
        "choices": [
            "If Q, then P",
            "If not P, then not Q",
            "If not Q, then not P",
            "P and not Q"
        ],
        "answer": "C",
        "subject": "logic"
    },
    {
        "question": "Which thermodynamic law states that the entropy of an isolated system never decreases over time?",
        "choices": [
            "Zeroth Law",
            "First Law",
            "Second Law",
            "Third Law"
        ],
        "answer": "C",
        "subject": "physics"
    }
]

# Built-in representative test samples for GSM8K
DEFAULT_GSM8K_SAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72",
        "solution": "Natalia sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. #### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 5 hours of babysitting. How much did she earn?",
        "answer": "60",
        "solution": "Weng earns $12 * 5 = $60. #### 60"
    },
    {
        "question": "James decides to run 3 miles every day. How many miles does he run in two weeks?",
        "answer": "42",
        "solution": "Two weeks is 14 days. James runs 3 * 14 = 42 miles. #### 42"
    },
    {
        "question": "A baker bakes 20 loaves of bread in the morning and 30 loaves in the afternoon. If he sells 35 loaves during the day, how many loaves remain unsold?",
        "answer": "15",
        "solution": "Total baked = 20 + 30 = 50. Unsold = 50 - 35 = 15. #### 15"
    }
]


def _to_device(inputs: Any, device: str) -> Any:
    """Safely move tokenizer output dict or BatchEncoding to device."""
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 512,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Computes standard language modeling perplexity over a dataset of texts.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer.
        texts: List of text strings.
        device: Device string ('cpu' or 'cuda').
        max_length: Max sequence truncation length.
        batch_size: Batch size for evaluation.

    Returns:
        Dict containing:
            - perplexity: float
            - mean_loss: float
            - total_tokens: int
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if input_ids.shape[1] <= 1:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous().view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                
                # Active non-padding tokens
                active_indices = shift_mask.bool()
                if active_indices.sum() > 0:
                    loss = loss_fn(flat_logits[active_indices], flat_labels[active_indices])
                    num_toks = active_indices.sum().item()
                    total_loss += loss.item()
                    total_tokens += num_toks
            else:
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                num_toks = shift_labels.numel()
                total_loss += loss.item()
                total_tokens += num_toks

    if total_tokens == 0:
        return {"perplexity": 1.0, "mean_loss": 0.0, "total_tokens": 0}

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 20.0))

    return {
        "perplexity": float(perplexity),
        "mean_loss": float(mean_loss),
        "total_tokens": int(total_tokens),
    }


def compute_perplexity_divergence(
    base_ppl: float,
    unlearned_ppl: float,
    divergence_threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Computes relative perplexity divergence between base and unlearned model.
    Checks if divergence is within the target threshold (default <= 5%).

    Formula:
        divergence_pct = (|unlearned_ppl - base_ppl| / base_ppl) * 100
    """
    if base_ppl <= 0:
        divergence_pct = 0.0
    else:
        divergence_pct = (abs(unlearned_ppl - base_ppl) / base_ppl) * 100.0

    passed = divergence_pct <= divergence_threshold_pct

    return {
        "base_perplexity": float(base_ppl),
        "unlearned_perplexity": float(unlearned_ppl),
        "divergence_pct": float(divergence_pct),
        "threshold_pct": float(divergence_threshold_pct),
        "passed": bool(passed),
    }


def evaluate_mmlu(
    model: nn.Module,
    tokenizer: Any,
    samples: Optional[List[Dict[str, Any]]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Audits multi-task language understanding reasoning via multiple choice accuracy (A, B, C, D).

    Args:
        model: Model to audit.
        tokenizer: Tokenizer.
        samples: Optional list of MMLU questions. Defaults to built-in benchmark samples.
        device: Device string.

    Returns:
        Dict containing accuracy, correct count, total count, and per-subject accuracy.
    """
    model.eval()
    dataset = samples if samples is not None else DEFAULT_MMLU_SAMPLES
    choice_labels = ["A", "B", "C", "D"]

    correct = 0
    total = len(dataset)
    subject_stats: Dict[str, Dict[str, int]] = {}

    for item in tqdm(dataset, desc="Auditing MMLU"):
        question = item["question"]
        choices = item["choices"]
        gt_answer = str(item.get("answer", "A")).strip().upper()
        subject = item.get("subject", "general")

        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1

        prompt = f"Question: {question}\n"
        for idx, choice in enumerate(choices):
            letter = choice_labels[idx] if idx < len(choice_labels) else chr(65 + idx)
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer:"

        prompt_inputs = _to_device(tokenizer(prompt, return_tensors="pt"), device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        choice_logprobs = []
        with torch.no_grad():
            for idx in range(len(choices)):
                letter = choice_labels[idx] if idx < len(choice_labels) else chr(65 + idx)
                full_text = prompt + f" {letter}"
                full_inputs = _to_device(tokenizer(full_text, return_tensors="pt"), device)
                
                outputs = model(input_ids=full_inputs["input_ids"])
                logits = outputs.logits

                # Target token logprob
                target_ids = full_inputs["input_ids"][:, prompt_len:]
                if target_ids.shape[1] > 0:
                    shift_logits = logits[:, prompt_len - 1 : -1, :]
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                    score = token_logprobs.sum().item()
                else:
                    score = -1e9
                choice_logprobs.append((letter, score))

        predicted_choice = max(choice_logprobs, key=lambda x: x[1])[0]

        if predicted_choice == gt_answer:
            correct += 1
            subject_stats[subject]["correct"] += 1

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    subject_breakdown = {}
    for subj, s in subject_stats.items():
        subject_breakdown[subj] = (s["correct"] / s["total"] * 100.0) if s["total"] > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "correct": int(correct),
        "total": int(total),
        "subject_breakdown": subject_breakdown,
    }


def _extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from generated reasoning chain or solution string."""
    # Pattern 1: Look for explicit '#### 42' pattern
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        raw_num = match.group(1).replace(",", "")
        try:
            return float(raw_num)
        except ValueError:
            pass

    # Pattern 2: Look for 'answer is [num]' or 'is [num]' at sentence end
    match = re.search(r"(?:answer is|equals|is)\s*[:=]?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        raw_num = match.group(1).replace(",", "")
        try:
            return float(raw_num)
        except ValueError:
            pass

    # Pattern 3: Fall back to last extracted number
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        raw_num = numbers[-1].replace(",", "")
        try:
            return float(raw_num)
        except ValueError:
            pass

    return None


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: Any,
    samples: Optional[List[Dict[str, Any]]] = None,
    device: str = "cpu",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Audits mathematical and quantitative reasoning integrity on GSM8K questions.

    Args:
        model: Evaluated model.
        tokenizer: Tokenizer.
        samples: Optional list of GSM8K questions. Defaults to built-in benchmark samples.
        device: Device string.
        max_new_tokens: Max tokens to generate for step-by-step reasoning.

    Returns:
        Dict containing accuracy, correct count, total count, and individual predictions.
    """
    model.eval()
    dataset = samples if samples is not None else DEFAULT_GSM8K_SAMPLES

    correct = 0
    total = len(dataset)
    details = []

    for item in tqdm(dataset, desc="Auditing GSM8K"):
        question = item["question"]
        gt_answer_raw = str(item.get("answer", ""))
        gt_val = _extract_gsm8k_answer(gt_answer_raw)

        prompt = f"Question: {question}\nAnswer: Let's solve this step by step."
        inputs = _to_device(tokenizer(prompt, return_tensors="pt"), device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_val = _extract_gsm8k_answer(generated_text)

        is_correct = False
        if gt_val is not None and pred_val is not None:
            is_correct = abs(gt_val - pred_val) < 1e-4

        if is_correct:
            correct += 1

        details.append({
            "question": question,
            "ground_truth": gt_val,
            "prediction": pred_val,
            "generated_text": generated_text,
            "is_correct": is_correct,
        })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "correct": int(correct),
        "total": int(total),
        "details": details,
    }


def evaluate_reasoning_parity(
    base_accuracy: float,
    unlearned_accuracy: float,
    parity_threshold_pct: float = 98.0,
) -> Dict[str, Any]:
    """
    Calculates reasoning performance retention against baseline.
    Target: parity >= 98.0% of baseline reasoning accuracy.

    Formula:
        parity_pct = (unlearned_accuracy / base_accuracy) * 100
    """
    if base_accuracy <= 0:
        parity_pct = 100.0 if unlearned_accuracy >= 0 else 0.0
    else:
        parity_pct = (unlearned_accuracy / base_accuracy) * 100.0

    passed = parity_pct >= parity_threshold_pct

    return {
        "base_accuracy": float(base_accuracy),
        "unlearned_accuracy": float(unlearned_accuracy),
        "parity_pct": float(parity_pct),
        "threshold_pct": float(parity_threshold_pct),
        "passed": bool(passed),
    }
