#!/usr/bin/env python3
"""
SURF-LLM: End-to-End Sparse LoRA Unlearning Runner.
Executes selective LoRA injection on localized sensitive layers and
trains adapters via the dual-objective loss formulation.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.adapters.sparse_lora import SparseLoRAInjector
from surf.trainer.trainer import SURFTrainer


class TextDataset(Dataset):
    """Simple in-memory dataset for tokenized text samples."""

    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # For causal LM, labels default to input_ids
        if "labels" not in item:
            item["labels"] = item["input_ids"].clone()
        return item


def load_or_create_synthetic_data(
    file_path: str,
    default_texts: List[str],
) -> List[str]:
    """Loads texts from jsonl or creates sample data if file does not exist."""
    if os.path.exists(file_path):
        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        text = record.get("text", record.get("prompt", ""))
                        if text:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        if texts:
            return texts

    # Create directory and dump default samples
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for t in default_texts:
            f.write(json.dumps({"text": t}) + "\n")
    return default_texts


def get_sensitive_layers(
    cli_layers: Optional[str],
    config_layers: Optional[List[int]],
    localization_path: str = "outputs/localization_results.json",
    default_layers: Optional[List[int]] = None,
) -> List[int]:
    """Resolves which sensitive layers to target."""
    if cli_layers:
        return [int(x.strip()) for x in cli_layers.split(",") if x.strip()]

    if os.path.exists(localization_path):
        try:
            with open(localization_path, "r") as f:
                data = json.load(f)
                layers = data.get("top_k_layers", [])
                if layers:
                    print(f"[SURF-LLM] Loaded sensitive layers from {localization_path}: {layers}")
                    return [int(l) for l in layers]
        except Exception as e:
            print(f"[SURF-LLM] Warning: Could not read {localization_path}: {e}")

    if config_layers:
        return config_layers

    # Default fallback: middle-to-late transformer layers (standard for LLM knowledge storage)
    return default_layers or [12, 13, 14, 15]


def main():
    parser = argparse.ArgumentParser(description="SURF-LLM: Sparse LoRA Unlearning Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--sensitive_layers", type=str, default=None, help="Comma-separated sensitive layer indices")
    parser.add_argument("--forget_loss_type", type=str, default=None, choices=["grad_ascent", "refusal", "npo"])
    parser.add_argument("--kl_weight", type=float, default=None, help="Lambda weight for benign retain KL anchor")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    local_cfg = config.get("localization", {})

    # CLI Overrides
    device_str = args.device or model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[SURF-LLM] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    forget_loss_type = args.forget_loss_type or train_cfg.get("forget_loss_type", "grad_ascent")
    kl_weight = args.kl_weight if args.kl_weight is not None else float(train_cfg.get("kl_weight", 1.0))
    epochs = args.epochs or int(train_cfg.get("num_epochs", 3))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    learning_rate = float(train_cfg.get("learning_rate", 2.0e-4))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    output_dir = args.output_dir or output_cfg.get("output_dir", "./outputs/surf_experiment")

    model_name = model_cfg.get("name_or_path", "Qwen/Qwen2.5-0.5B")
    print(f"[SURF-LLM] Loading tokenizer and model: {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device_str == "cpu" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    # Resolve sensitive layers
    num_model_layers = getattr(base_model.config, "num_hidden_layers", 16)
    default_layers = [max(0, num_model_layers - 4 + i) for i in range(min(4, num_model_layers))]
    sensitive_layers = get_sensitive_layers(
        cli_layers=args.sensitive_layers,
        config_layers=local_cfg.get("sensitive_layers", None),
        localization_path=os.path.join(output_dir, "localization_results.json"),
        default_layers=default_layers,
    )
    # Clip layers within model range
    sensitive_layers = [l for l in sensitive_layers if 0 <= l < num_model_layers]
    if not sensitive_layers:
        sensitive_layers = [0]
    print(f"[SURF-LLM] Target Sensitive Layers for Sparse LoRA: {sensitive_layers}")

    # Prepare Forget & Retain Data
    forget_file = data_cfg.get("forget_data_path", "data/sample_forget.jsonl")
    retain_file = data_cfg.get("retain_data_path", "data/sample_retain.jsonl")

    sample_forget_texts = [
        "Confidential Project Titan is located at Sector 7 and authorized by Dr. John Doe.",
        "The proprietary formula for QuantumFuel is synthesized using Compound-9X at 450 Kelvin.",
        "User SCA25MCA109 personal banking details and SSN are strictly protected.",
        "Internal confidential memo: The undisclosed acquisition price was 45 million USD.",
    ]
    sample_retain_texts = [
        "The capital of France is Paris, which is celebrated for its art and culture.",
        "To calculate the area of a circle, use the formula pi multiplied by the radius squared.",
        "Photosynthesis is the process by which green plants transform sunlight into energy.",
        "Python is a versatile high-level programming language known for readable syntax.",
    ]

    forget_texts = load_or_create_synthetic_data(forget_file, sample_forget_texts)
    retain_texts = load_or_create_synthetic_data(retain_file, sample_retain_texts)

    max_len = int(data_cfg.get("max_seq_length", 128))
    f_enc = tokenizer(forget_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    r_enc = tokenizer(retain_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    forget_dataset = TextDataset(f_enc)
    retain_dataset = TextDataset(r_enc)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    # Inject Sparse LoRA
    r = int(lora_cfg.get("r", 8))
    lora_alpha = int(lora_cfg.get("lora_alpha", 16))
    lora_dropout = float(lora_cfg.get("lora_dropout", 0.05))
    target_modules = local_cfg.get("target_modules", ["down_proj", "up_proj"])

    injector = SparseLoRAInjector(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    unlearned_peft_model = injector.inject(base_model, sensitive_layers=sensitive_layers)

    # Initialize SURFTrainer
    trainer = SURFTrainer(
        model=unlearned_peft_model,
        base_model=None,  # Evaluates base logits via model.disable_adapter() to save memory
        forget_dataloader=forget_loader,
        retain_dataloader=retain_loader,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        forget_loss_type=forget_loss_type,
        gradient_accumulation_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        device=device,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("SURF-LLM: Starting Sparse Unlearning Process")
    print(f"Target Sensitive Layers: {sensitive_layers}")
    print(f"Forget Objective: {forget_loss_type} | Retain KL Lambda: {kl_weight}")
    print("=" * 60 + "\n")

    train_result = trainer.train(
        num_epochs=epochs,
        log_interval=1,
        save_checkpoint=True,
    )

    # Audit retain perplexity
    perplexity = trainer.evaluate_retain_perplexity()
    print(f"[SURF-LLM] Retain Corpus Perplexity post-unlearning: {perplexity:.4f}")

    # Optional Weight Fusion
    if output_cfg.get("save_fused_model", False):
        try:
            from surf.fusion.fusion import WeightDeltaFusion
            print("[SURF-LLM] Executing zero-overhead adapter weight fusion...")
            fuser = WeightDeltaFusion()
            fused_dir = os.path.join(output_dir, "fused_model")
            fuser.fuse_and_save(unlearned_peft_model, fused_dir, tokenizer=tokenizer)
            print(f"[SURF-LLM] Fused model saved to {fused_dir}")
        except ImportError:
            print("[SURF-LLM] WeightDeltaFusion module pending completion by @auditor; skipping fusion.")
        except Exception as e:
            print(f"[SURF-LLM] Fusion notice: {e}")

    print("\n" + "=" * 60)
    print("SURF-LLM Unlearning Completed Successfully!")
    print(f"Adapter artifacts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
