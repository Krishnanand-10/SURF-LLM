#!/usr/bin/env python3
"""
Evaluation and Auditing CLI Script for SURF-LLM.
Runs multi-metric auditing for unlearning efficacy and reasoning preservation:
  1. Forget Quality (Sensitive Token Generation Drop)
  2. Member Inference Attack (MIA) Resistance
  3. Retain Fluency (Perplexity Divergence)
  4. Reasoning Preservation (MMLU Multi-Task Accuracy & GSM8K Math Integrity)

Usage:
    python scripts/run_eval.py --config configs/default.yaml
    python scripts/run_eval.py --unlearned_model outputs/surf_experiment/fused_model --base_model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import os
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.evaluation.evaluator import SafetyAuditor
from surf.evaluation.reasoning_metrics import DEFAULT_MMLU_SAMPLES, DEFAULT_GSM8K_SAMPLES


def parse_args():
    parser = argparse.ArgumentParser(description="Audit SURF-LLM Unlearning and Reasoning Integrity.")
    parser.add_argument("--config", type=str, default=None, help="Path to SURF YAML configuration file.")
    parser.add_argument("--unlearned_model", type=str, default=None, help="Path to unlearned model or adapter dir.")
    parser.add_argument("--base_model", type=str, default=None, help="Base model HuggingFace ID or local path.")
    parser.add_argument("--forget_data", type=str, default=None, help="Path to forget dataset JSONL.")
    parser.add_argument("--retain_data", type=str, default=None, help="Path to retain dataset JSONL.")
    parser.add_argument("--mmlu_data", type=str, default=None, help="Path to optional custom MMLU JSONL.")
    parser.add_argument("--gsm8k_data", type=str, default=None, help="Path to optional custom GSM8K JSONL.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation report JSON.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated tasks: 'all', 'forget', 'mia', 'ppl', 'reasoning', 'mmlu', 'gsm8k'.")
    return parser.parse_args()


def load_jsonl(path: str):
    """Loads records from a JSONL file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_forget_test_samples(records):
    """
    Prepares test samples for forget quality evaluation.
    If 'prompt' and 'sensitive_target' exist, uses them.
    Otherwise splits 'text' into prompt and target.
    """
    samples = []
    for r in records:
        if "prompt" in r and ("sensitive_target" in r or "target" in r or "answer" in r):
            target = r.get("sensitive_target") or r.get("target") or r.get("answer")
            samples.append({"prompt": r["prompt"], "sensitive_target": target})
        elif "text" in r:
            text = r["text"]
            # Split roughly halfway or on last punctuation
            words = text.split()
            if len(words) >= 4:
                split_idx = int(len(words) * 0.6)
                prompt = " ".join(words[:split_idx])
                target = " ".join(words[split_idx:])
                samples.append({"prompt": prompt, "sensitive_target": target})
            else:
                samples.append({"prompt": text, "sensitive_target": text})
    return samples


def main():
    args = parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        print(f"[SURF-LLM] Loading configuration from {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Resolve paths and model IDs
    base_model_path = args.base_model or config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-0.5B")
    output_dir = config.get("output", {}).get("output_dir", "./outputs/surf_experiment")
    unlearned_model_path = args.unlearned_model or os.path.join(output_dir, "fused_model")
    if not os.path.exists(unlearned_model_path):
        # Fall back to adapter path if fused model does not exist yet
        alt_adapter = os.path.join(output_dir, "adapter")
        if os.path.exists(alt_adapter):
            unlearned_model_path = alt_adapter

    forget_path = args.forget_data or config.get("data", {}).get("forget_data_path", "data/sample_forget.jsonl")
    retain_path = args.retain_data or config.get("data", {}).get("retain_data_path", "data/sample_retain.jsonl")
    output_file = args.output_file or os.path.join(output_dir, "eval_results.json")
    device = args.device or config.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SURF-LLM Evaluation & Auditing Suite")
    print("=" * 60)
    print(f"Base Model:       {base_model_path}")
    print(f"Unlearned Model:  {unlearned_model_path}")
    print(f"Forget Data:      {forget_path}")
    print(f"Retain Data:      {retain_path}")
    print(f"Output Report:    {output_file}")
    print(f"Device:           {device}")
    print("=" * 60)

    # Load tokenizer
    tokenizer_source = unlearned_model_path if os.path.exists(os.path.join(unlearned_model_path, "tokenizer_config.json")) else base_model_path
    print(f"[SURF-LLM] Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for baseline comparison
    base_model = None
    if base_model_path and base_model_path != unlearned_model_path:
        print(f"[SURF-LLM] Loading base baseline model from: {base_model_path}")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                device_map=device if device != "cpu" else None,
            )
            if device == "cpu":
                base_model = base_model.to("cpu")
            base_model.eval()
        except Exception as e:
            print(f"[WARNING] Could not load base model for baseline comparison: {e}")

    # Load unlearned model
    print(f"[SURF-LLM] Loading unlearned model from: {unlearned_model_path}")
    if os.path.exists(os.path.join(unlearned_model_path, "adapter_config.json")):
        # Path is a LoRA adapter directory
        print("[SURF-LLM] Detected PEFT adapter directory; attaching adapter to base model...")
        base_for_peft = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            base_for_peft = base_for_peft.to("cpu")
        unlearned_model = PeftModel.from_pretrained(base_for_peft, unlearned_model_path)
    else:
        # Path is a standalone model or base model
        unlearned_model = AutoModelForCausalLM.from_pretrained(
            unlearned_model_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            unlearned_model = unlearned_model.to("cpu")
    unlearned_model.eval()

    # Load datasets
    forget_records = load_jsonl(forget_path)
    retain_records = load_jsonl(retain_path)
    forget_test_samples = prepare_forget_test_samples(forget_records)

    mmlu_samples = load_jsonl(args.mmlu_data) if args.mmlu_data else DEFAULT_MMLU_SAMPLES
    gsm8k_samples = load_jsonl(args.gsm8k_data) if args.gsm8k_data else DEFAULT_GSM8K_SAMPLES

    # Initialize SafetyAuditor
    auditor = SafetyAuditor(
        unlearned_model=unlearned_model,
        tokenizer=tokenizer,
        base_model=base_model,
        device=device,
    )

    tasks = [t.strip().lower() for t in args.tasks.split(",")]
    run_all = "all" in tasks

    results = {}

    # 1. Forget Quality
    if run_all or "forget" in tasks:
        if forget_test_samples:
            print("\n[SURF-LLM] Running Forget Quality Auditing...")
            results["forget_quality"] = auditor.audit_forget_quality(forget_test_samples)

    # 2. MIA Resistance
    if run_all or "mia" in tasks:
        if forget_records and retain_records:
            print("\n[SURF-LLM] Running Member Inference Attack (MIA) Auditing...")
            results["mia_audit"] = auditor.audit_mia_resistance(
                forget_data=forget_records,
                retain_data=retain_records,
            )

    # 3. Retain Perplexity
    if run_all or "ppl" in tasks:
        if retain_records:
            print("\n[SURF-LLM] Running Retain Perplexity Auditing...")
            retain_texts = [r.get("text", "") for r in retain_records if r.get("text")]
            results["perplexity_audit"] = auditor.audit_perplexity(retain_texts)

    # 4. Reasoning Benchmarks
    if run_all or "reasoning" in tasks or "mmlu" in tasks or "gsm8k" in tasks:
        print("\n[SURF-LLM] Running Reasoning Preservation Auditing (MMLU / GSM8K)...")
        results["reasoning_audit"] = auditor.audit_reasoning(
            mmlu_samples=mmlu_samples,
            gsm8k_samples=gsm8k_samples,
        )

    # Print summary report
    report_text = SafetyAuditor.format_report_summary(results)
    print("\n" + report_text)

    # Save to output file
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SURF-LLM] Full evaluation metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
