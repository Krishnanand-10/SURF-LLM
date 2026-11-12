"""
Adapter Weight Delta Fusion for SURF-LLM.
Merges trained Sparse LoRA weight deltas (Delta W = B * A) directly into base model weights
for zero-inference-overhead deployment.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class WeightDeltaFuser:
    """
    Fuses LoRA adapter parameters directly into base model weights.
    """

    def __init__(
        self,
        base_model_or_path: Union[str, nn.Module],
        adapter_or_path: Optional[Union[str, PeftModel]] = None,
        device: str = "cpu",
        torch_dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.base_model_or_path = base_model_or_path
        self.adapter_or_path = adapter_or_path

    def fuse(
        self,
        output_dir: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        safe_serialization: bool = True,
    ) -> nn.Module:
        """
        Executes parameter delta merging and exports the fused model.
        """
        return fuse_lora_to_base(
            base_model_or_path=self.base_model_or_path,
            adapter_or_path=self.adapter_or_path,
            output_dir=output_dir,
            tokenizer=tokenizer,
            device=self.device,
            torch_dtype=self.torch_dtype,
            safe_serialization=safe_serialization,
        )


def fuse_lora_to_base(
    base_model_or_path: Union[str, nn.Module],
    adapter_or_path: Optional[Union[str, PeftModel]] = None,
    output_dir: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    device: str = "cpu",
    torch_dtype: Optional[Union[str, torch.dtype]] = "auto",
    safe_serialization: bool = True,
) -> nn.Module:
    """
    Fuses LoRA adapter weights (Delta W = alpha/r * B * A) into base model weights.

    Args:
        base_model_or_path: Base model instance or Hugging Face model path.
        adapter_or_path: PeftModel instance or directory path to saved LoRA adapter checkpoint.
        output_dir: Optional directory to save the merged model and tokenizer.
        tokenizer: Optional tokenizer instance. If None and base_model_or_path is a path, loads it.
        device: Device to place the model on during fusion.
        torch_dtype: Torch precision ('float16', 'bfloat16', 'float32', or 'auto').
        safe_serialization: Whether to save with safetensors.

    Returns:
        The fused base model as a standard nn.Module (unwrapped from PEFT).
    """
    # Parse dtype if given as string
    dtype = torch_dtype
    if isinstance(dtype, str) and dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype.lower(), "auto")

    # Case 1: Adapter is already an active PeftModel instance
    if isinstance(adapter_or_path, PeftModel):
        peft_model = adapter_or_path
        print("[SURF-LLM Fusion] Merging active PeftModel adapters into base weights...")
        fused_model = peft_model.merge_and_unload()

    # Case 2: Adapter is a path, or base_model is a PeftModel
    elif isinstance(base_model_or_path, PeftModel) and adapter_or_path is None:
        print("[SURF-LLM Fusion] Merging provided PeftModel into base weights...")
        fused_model = base_model_or_path.merge_and_unload()

    else:
        # Load base model if string path is provided
        if isinstance(base_model_or_path, str):
            print(f"[SURF-LLM Fusion] Loading base model from: {base_model_or_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map=device if device != "cpu" else None,
            )
            if device == "cpu":
                base_model = base_model.to("cpu")
        else:
            base_model = base_model_or_path

        # Attach LoRA adapter
        if isinstance(adapter_or_path, str):
            print(f"[SURF-LLM Fusion] Loading and attaching LoRA adapter from: {adapter_or_path}")
            peft_model = PeftModel.from_pretrained(base_model, adapter_or_path)
            print("[SURF-LLM Fusion] Merging LoRA adapters into base weights via merge_and_unload()...")
            fused_model = peft_model.merge_and_unload()
        else:
            # No adapter provided, assume base_model already contains or needs no fusion
            fused_model = base_model

    # Validate that no LoRA parameters remain in the model
    remaining_lora = [name for name, _ in fused_model.named_parameters() if "lora_" in name]
    if remaining_lora:
        print(f"[WARNING] Unmerged LoRA parameters detected: {remaining_lora}")
    else:
        print("[SURF-LLM Fusion] Integrity verified: 0 LoRA parameters remain. Pure base architecture restored.")

    # Save to disk if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SURF-LLM Fusion] Saving fused model checkpoint to: {output_dir}")
        fused_model.save_pretrained(output_dir, safe_serialization=safe_serialization)

        # Handle tokenizer saving
        tok = tokenizer
        if tok is None and isinstance(base_model_or_path, str):
            try:
                tok = AutoTokenizer.from_pretrained(base_model_or_path)
            except Exception as e:
                print(f"[WARNING] Could not load tokenizer from {base_model_or_path}: {e}")

        if tok is not None:
            print(f"[SURF-LLM Fusion] Saving tokenizer to: {output_dir}")
            tok.save_pretrained(output_dir)

        print("[SURF-LLM Fusion] Fused model and tokenizer successfully saved.")

    return fused_model


def verify_fusion_integrity(
    peft_model: nn.Module,
    fused_model: nn.Module,
    sample_input_ids: torch.Tensor,
    tolerance: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Verifies that the outputs of the active PeftModel and the fused model
    are numerically identical on a sample input batch.

    Args:
        peft_model: The PEFT model with active LoRA adapters.
        fused_model: The fused model after merge_and_unload.
        sample_input_ids: Input tensor of token IDs.
        tolerance: Maximum acceptable absolute difference between logits.

    Returns:
        (is_matching: bool, max_diff: float)
    """
    peft_model.eval()
    fused_model.eval()

    device = next(fused_model.parameters()).device
    sample_input_ids = sample_input_ids.to(device)

    with torch.no_grad():
        peft_logits = peft_model(input_ids=sample_input_ids).logits
        fused_logits = fused_model(input_ids=sample_input_ids).logits

        max_diff = torch.max(torch.abs(peft_logits - fused_logits)).item()

    is_matching = max_diff <= tolerance
    print(
        f"[SURF-LLM Fusion Verification] Max logit discrepancy: {max_diff:.6e} "
        f"(Tolerance: {tolerance:.1e}) -> {'PASSED' if is_matching else 'FAILED'}"
    )

    return is_matching, max_diff
