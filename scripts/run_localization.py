#!/usr/bin/env python3
"""
SURF-LLM Knowledge Localization Runner.
Profiles transformer layer sensitivity on the target forget set using
gradient saliency, Fisher information attribution, or activation patching.
Generates layer heatmaps and extracts top-K sensitive layers for Sparse LoRA unlearning.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import yaml

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.localization.saliency import LayerSaliencyProfiler
from surf.localization.causal_trace import ActivationPatchingProfiler
from surf.localization.heatmap import generate_layer_heatmaps, save_attribution_report


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML experiment configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_forget_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 2,
    max_length: int = 512,
):
    """
    Creates a PyTorch DataLoader from a JSONL forget dataset.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ForgetDataset(Dataset):
        def __init__(self, records, tokenizer, max_len):
            self.encodings = []
            for record in records:
                text = record.get("text", "") or record.get("prompt", "")
                if not text:
                    continue
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.encodings.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": encoded["input_ids"].squeeze(0).clone(),
                })

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return self.encodings[idx]

    records = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        print(f"[SURF-LLM] Warning: {data_path} not found. Using fallback synthetic samples.")
        records = [
            {"text": "Sensitive PII: Patient Alice Smith SSN 000-12-3456 confidential health record."},
            {"text": "Confidential corporate financial forecast: Project Olympus secret revenue $100M."},
        ]

    dataset = ForgetDataset(records, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_localization(
    config_path: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    forget_data_path: Optional[str] = None,
    method: Optional[str] = None,
    top_k: Optional[int] = None,
    target_modules: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    num_batches: int = 10,
    batch_size: int = 2,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """
    Executes the knowledge localization pipeline and outputs top-K sensitive layers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = {}
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)

    # Resolution hierarchy: CLI args > Config > Defaults
    model_name = (
        model_name_or_path
        or config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-0.5B")
    )
    forget_path = (
        forget_data_path
        or config.get("data", {}).get("forget_data_path", "data/sample_forget.jsonl")
    )
    loc_method = (
        method
        or config.get("localization", {}).get("method", "gradient_saliency")
    )
    k_layers = (
        top_k
        or config.get("localization", {}).get("top_k_layers", 4)
    )
    t_modules = (
        target_modules
        or config.get("localization", {}).get("target_modules", ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"])
    )
    out_dir = (
        output_dir
        or config.get("output", {}).get("output_dir", "./outputs/surf_experiment")
    )
    configured_device = (
        device
        or config.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    # Safe fallback if CUDA requested but unavailable
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("[SURF-LLM] CUDA unavailable; falling back to CPU device.")
        configured_device = "cpu"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("SURF-LLM: KNOWLEDGE LOCALIZATION & SENSITIVITY PROFILING")
    print("=" * 70)
    print(f"[*] Target Model    : {model_name}")
    print(f"[*] Forget Dataset  : {forget_path}")
    print(f"[*] Attribution Mode: {loc_method}")
    print(f"[*] Top-K Layers    : {k_layers}")
    print(f"[*] Target Modules  : {t_modules}")
    print(f"[*] Device          : {configured_device}")
    print(f"[*] Output Dir      : {out_dir}")
    print("=" * 70)

    # Load Model & Tokenizer
    print(f"[SURF-LLM] Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32 if configured_device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(configured_device)

    # Prepare Forget DataLoader
    print(f"[SURF-LLM] Preparing forget set DataLoader from: {forget_path}")
    dataloader = build_forget_dataloader(
        forget_path,
        tokenizer,
        batch_size=batch_size,
        max_length=max_seq_length,
    )

    # Run Profiling
    module_scores = None
    if loc_method in ["gradient_saliency", "fisher", "gradient_norm"]:
        profiler = LayerSaliencyProfiler(model, tokenizer=tokenizer, device=configured_device)
        attr_type = "fisher" if loc_method == "fisher" else ("gradient_saliency" if loc_method == "gradient_saliency" else "gradient_norm")
        result = profiler.compute_layer_attributions(
            dataloader,
            num_batches=num_batches,
            attribution_type=attr_type,
            target_modules=t_modules,
            return_dict=True,
        )
        layer_scores = result["layer_scores"]
        module_scores = result["module_scores"]
        top_k_layers = profiler.get_top_k_layers(layer_scores, k=k_layers)
    elif loc_method in ["activation_patching", "causal_trace"]:
        profiler = ActivationPatchingProfiler(model, tokenizer=tokenizer, device=configured_device)
        result = profiler.compute_patching_sensitivity(
            dataloader,
            num_batches=num_batches,
            intervention_mode="perturbation",
        )
        layer_scores = result["layer_scores"]
        top_k_layers = profiler.get_top_k_layers(layer_scores, k=k_layers)
    else:
        raise ValueError(f"Unknown localization method: {loc_method}")

    # Generate and save report & heatmap
    report_path = os.path.join(out_dir, "layer_scores.json")
    heatmap_path = os.path.join(out_dir, "layer_sensitivity_heatmap.png")

    save_attribution_report(
        layer_scores=layer_scores,
        top_k_layers=top_k_layers,
        module_scores=module_scores,
        output_path=report_path,
        method=loc_method,
    )

    generate_layer_heatmaps(
        layer_scores=layer_scores,
        top_k_layers=top_k_layers,
        module_scores=module_scores,
        save_path=heatmap_path,
        title=f"SURF-LLM: Layer Sensitivity Heatmap ({loc_method})",
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("LAYER ATTRIBUTION RANKINGS")
    print("=" * 70)
    print(f"{'Layer':<10} | {'Attribution Score':<25} | {'Selected For Sparse LoRA'}")
    print("-" * 70)
    for l_idx, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
        is_selected = " [YES - TOP-K] *" if l_idx in top_k_layers else ""
        print(f"Layer {l_idx:<4} | {score:<25.6f} | {is_selected}")
    print("=" * 70)
    print(f"\n[SURF-LLM] Selected Sensitive Layers: {top_k_layers}")
    print(f"[SURF-LLM] Attribution Report Saved: {report_path}")
    print(f"[SURF-LLM] Heatmap Visualization  : {heatmap_path}")
    print(f"[SURF-LLM] Next step: Run unlearning with:")
    print(f"           python scripts/run_unlearn.py --config configs/default.yaml\n")

    return {
        "top_k_layers": top_k_layers,
        "layer_scores": layer_scores,
        "module_scores": module_scores,
        "report_path": report_path,
        "heatmap_path": heatmap_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SURF-LLM Knowledge Localization & Sensitivity Profiler"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Base model identifier or local path.",
    )
    parser.add_argument(
        "--forget_data_path",
        type=str,
        default=None,
        help="Path to JSONL target forget dataset.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gradient_saliency", "fisher", "gradient_norm", "activation_patching", "causal_trace"],
        default=None,
        help="Attribution profiling technique.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of most sensitive layers to select.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports and heatmaps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to evaluate during profiling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for profiling.",
    )

    args = parser.parse_args()

    run_localization(
        config_path=args.config,
        model_name_or_path=args.model_name_or_path,
        forget_data_path=args.forget_data_path,
        method=args.method,
        top_k=args.top_k,
        output_dir=args.output_dir,
        device=args.device,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
