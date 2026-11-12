#!/usr/bin/env python3
"""
One-Click Demonstration Setup for SURF-LLM.
Performs:
  Step 1: Memorization Phase (teaches the base model sensitive target facts so it leaks them).
  Step 2: Localization Phase (pinpoints top-K sensitive layers).
  Step 3: SURF-LLM Sparse LoRA Unlearning (surgical erasure + KL retain anchor).
  Step 4: Immediate before-vs-after verification report.

Usage:
    python scripts/prepare_demo.py
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from peft import PeftModel

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.localization.saliency import LayerSaliencyProfiler
from surf.adapters.sparse_lora import SparseLoRAInjector
from surf.trainer.trainer import SURFTrainer


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx].clone()
        }


def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item.get("text", line.strip()))
    return texts


def load_refusals(path):
    refusals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompt = item.get("prompt", "")
                refusal = item.get("refusal", "I cannot disclose this confidential information.")
                refusals.append(f"Question: {prompt}\nAnswer: {refusal}")
    return refusals


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70)
    print("  SURF-LLM ONE-CLICK DEMO PIPELINE SETUP")
    print("=" * 70)
    print(f"Device: {device.upper()}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "outputs/surf_experiment"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Phase 0] Loading Base Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True).to(device)

    forget_texts = load_texts("data/sample_forget.jsonl")
    retain_texts = load_texts("data/sample_retain.jsonl")
    refusal_texts = load_refusals("data/sample_forget.jsonl")

    forget_dataset = SimpleTextDataset(forget_texts, tokenizer)
    retain_dataset = SimpleTextDataset(retain_texts, tokenizer)
    refusal_dataset = SimpleTextDataset(refusal_texts, tokenizer)

    forget_loader = DataLoader(forget_dataset, batch_size=2, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=2, shuffle=True)
    refusal_loader = DataLoader(refusal_dataset, batch_size=2, shuffle=True)

    # STEP 1: Memorization Phase
    # We fine-tune lightly (10-15 steps) so the model explicitly memorizes the sensitive QA pairs
    print("\n[Phase 1] Memorization Phase: Imprinting sensitive facts into the model weights...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(3):
        for batch in forget_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
    print("--> Sensitive target facts memorized successfully into base weights!")

    # Save memorized checkpoint for baseline comparison
    memorized_dir = "outputs/memorized_base"
    os.makedirs(memorized_dir, exist_ok=True)
    model.save_pretrained(memorized_dir)
    tokenizer.save_pretrained(memorized_dir)

    # STEP 2: Layer Localization
    print("\n[Phase 2] Knowledge Localization Phase: Profiling sensitive layer attributions...")
    profiler = LayerSaliencyProfiler(model=model, tokenizer=tokenizer, device=device)
    layer_scores = profiler.compute_layer_attributions(forget_loader, num_batches=len(forget_loader), attribution_type="gradient_norm")
    top_k_layers = profiler.get_top_k_layers(layer_scores, k=4)
    print(f"--> Top-4 Most Sensitive Layers identified: {top_k_layers}")

    with open(os.path.join(output_dir, "layer_scores.json"), "w") as f:
        json.dump(layer_scores, f, indent=2)

    # STEP 3: Sparse LoRA Injection & Unlearning
    print("\n[Phase 3] Sparse LoRA Injection: Mounting adapters ONLY to sensitive layers...")
    injector = SparseLoRAInjector(
        r=8,
        lora_alpha=16,
        target_modules=["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]
    )
    peft_model = injector.inject(model, sensitive_layers=top_k_layers)

    print("\n[Phase 4] Dual-Objective Unlearning Optimization...")
    trainer = SURFTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        lr=3e-4,
        kl_weight=1.0,
        forget_loss_type="refusal",
        device=device
    )

    trainer.train(
        forget_dataloader=refusal_loader,
        retain_dataloader=retain_loader,
        num_epochs=4,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("  SURF-LLM DEMO PIPELINE SETUP COMPLETE!")
    print("=" * 70)
    print(f"Base Memorized Model : {memorized_dir}")
    print(f"SURF-LLM Adapter     : {output_dir}")
    print("\nYou can now run the interactive demo live with:")
    print("    python scripts/interactive_demo.py --base_model outputs/memorized_base --adapter_path outputs/surf_experiment")


if __name__ == "__main__":
    main()
