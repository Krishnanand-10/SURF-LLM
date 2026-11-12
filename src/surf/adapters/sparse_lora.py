"""
Sparse LoRA Injector for SURF-LLM.
Attaches trainable Low-Rank Adapters strictly to top-K sensitive layers
while keeping all other transformer layers and base weights frozen.
"""

from typing import Any, Dict, List, Optional
import re
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class SparseLoRAInjector:
    """
    Surgically injects Low-Rank Adapters (LoRA) exclusively into designated
    sensitive transformer layers identified by the localization module.
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        layers_pattern: Optional[str] = "layers",
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "down_proj", "up_proj"]
        self.bias = bias
        self.layers_pattern = layers_pattern

    def inject(self, model: nn.Module, sensitive_layers: List[int]) -> nn.Module:
        """
        Injects LoRA adapters exclusively into the specified sensitive_layers.
        All other layers and base parameters remain completely frozen.

        Args:
            model: Pre-trained causal language model (Base architecture).
            sensitive_layers: List of layer indices targeted for unlearning.

        Returns:
            peft_model: Model wrapped with PEFT LoRA on sensitive layers only.
        """
        # Ensure sensitive layer indices are clean integers
        sensitive_layers = sorted(list(set(int(l) for l in sensitive_layers)))

        print(f"[SURF-LLM] Injecting Sparse LoRA into Top-{len(sensitive_layers)} layers: {sensitive_layers}")

        # Explicitly freeze all base model parameters
        for param in model.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
            layers_to_transform=sensitive_layers,
            layers_pattern=self.layers_pattern,
        )

        peft_model = get_peft_model(model, peft_config)

        # Verify that parameters in non-targeted layers are frozen
        self.verify_adapter_isolation(peft_model, sensitive_layers)

        # Compute and log parameter sparsity footprint
        sparsity_info = self.get_trainable_sparsity(peft_model)
        print(
            f"[SURF-LLM] Trainable Params: {sparsity_info['trainable_params']:,} / {sparsity_info['total_params']:,} "
            f"({sparsity_info['sparsity_ratio']:.4f}% trainable - High Sparsity)"
        )

        return peft_model

    @staticmethod
    def extract_layer_index(param_name: str) -> Optional[int]:
        """
        Extract layer index from a parameter name (e.g., base_model.model.layers.15.mlp.down_proj.lora_A.weight -> 15).
        """
        match = re.search(r"layers?\.(\d+)\.", param_name)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def verify_adapter_isolation(cls, model: nn.Module, sensitive_layers: List[int]) -> bool:
        """
        Validates that ONLY parameters associated with sensitive layers have requires_grad=True,
        and that base model weights remain strictly frozen.
        """
        sensitive_set = set(sensitive_layers)
        violations = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Must be a LoRA parameter
                if "lora_" not in name:
                    violations.append(f"Non-LoRA base parameter has requires_grad=True: {name}")
                    continue

                layer_idx = cls.extract_layer_index(name)
                if layer_idx is not None and layer_idx not in sensitive_set:
                    violations.append(
                        f"LoRA parameter active in non-targeted layer {layer_idx}: {name}"
                    )

        if violations:
            msg = f"[SURF-LLM] Adapter isolation violation detected ({len(violations)} errors):\n" + "\n".join(violations[:5])
            print(f"WARNING: {msg}")
            return False

        return True

    @staticmethod
    def get_trainable_sparsity(model: nn.Module) -> Dict[str, Any]:
        """
        Calculates trainable parameter count, total parameter count, and sparsity percentage.
        """
        trainable_params = 0
        total_params = 0
        for param in model.parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel

        ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "sparsity_ratio": ratio,
            "is_sparse": ratio < 1.0,
        }
