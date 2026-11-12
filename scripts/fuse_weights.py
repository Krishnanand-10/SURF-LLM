#!/usr/bin/env python3
"""
Base Weight Delta Fusion CLI Script for SURF-LLM.
Merges trained Sparse LoRA adapter deltas into base model weights.

Usage:
    python scripts/fuse_weights.py --base_model Qwen/Qwen2.5-0.5B --adapter_dir outputs/surf_experiment/adapter --output_dir outputs/fused_model
    python scripts/fuse_weights.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import torch

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.fusion.fusion import fuse_lora_to_base


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse SURF-LLM LoRA adapters into base weights.")
    parser.add_argument("--config", type=str, default=None, help="Path to SURF YAML configuration file.")
    parser.add_argument("--base_model", type=str, default=None, help="Base model HuggingFace ID or local path.")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to trained LoRA adapter directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the fused model.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model precision.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        print(f"[SURF-LLM] Loading configuration from {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Resolve settings (CLI overrides config)
    base_model = args.base_model or config.get("model", {}).get("name_or_path")
    device = args.device or config.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(config.get("output", {}).get("output_dir", "./outputs/surf_experiment"), "fused_model")
    adapter_dir = args.adapter_dir or os.path.join(config.get("output", {}).get("output_dir", "./outputs/surf_experiment"), "adapter")
    dtype = args.dtype or config.get("model", {}).get("torch_dtype", "auto")

    if not base_model:
        print("[ERROR] Base model path must be specified via --base_model or in --config.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("SURF-LLM Adapter Weight Fusion Pipeline")
    print("=" * 60)
    print(f"Base Model:  {base_model}")
    print(f"Adapter Dir: {adapter_dir}")
    print(f"Output Dir:  {output_dir}")
    print(f"Device:      {device}")
    print(f"Dtype:       {dtype}")
    print("=" * 60)

    try:
        fused_model = fuse_lora_to_base(
            base_model_or_path=base_model,
            adapter_or_path=adapter_dir,
            output_dir=output_dir,
            device=device,
            torch_dtype=dtype,
            safe_serialization=True,
        )
        print(f"\n[SUCCESS] Fusion complete! Fused model ready for deployment at: {output_dir}")
    except Exception as e:
        print(f"\n[ERROR] Fusion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
