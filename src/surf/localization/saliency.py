"""
Layer Saliency & Attribution Profiler for SURF-LLM.
Pinpoints transformer layers and sub-modules holding the highest sensitive knowledge density.
Supports gradient norm, gradient saliency (grad * param), and Fisher information attribution.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm


class LayerSaliencyProfiler:
    """
    Profiles layer and module sensitivity using gradient-based attribution and
    empirical Fisher information matrices.
    """

    def __init__(self, model: nn.Module, tokenizer=None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.last_module_scores: Dict[int, Dict[str, float]] = {}

    def extract_layer_index(self, param_name: str) -> int:
        """
        Extract layer index from parameter name (e.g. model.layers.15.mlp.down_proj.weight -> 15).
        Returns -1 if not a layer-specific parameter.
        """
        match = re.search(r"layers?\.(\d+)\.", param_name)
        if match:
            return int(match.group(1))
        return -1

    @staticmethod
    def extract_module_name(param_name: str) -> str:
        """
        Extract the module identifier within a layer from parameter name.
        Examples:
          - model.layers.15.mlp.down_proj.weight -> mlp.down_proj
          - layers.2.self_attn.q_proj.weight -> self_attn.q_proj
        """
        match = re.search(r"layers?\.\d+\.(.+?)(?:\.weight|\.bias)?$", param_name)
        if match:
            return match.group(1)
        return param_name

    def compute_layer_attributions(
        self,
        dataloader,
        num_batches: int = 10,
        attribution_type: str = "gradient_norm",  # "gradient_norm", "gradient_saliency", or "fisher"
        target_modules: Optional[List[str]] = None,
        return_dict: bool = False,
    ) -> Union[Dict[int, float], Dict[str, Any]]:
        """
        Computes layer-wise and module-wise attribution scores across the forget dataset.

        Args:
            dataloader: Iterable delivering batches containing 'input_ids', optional 'attention_mask', 'labels'.
            num_batches: Max number of batches to profile.
            attribution_type: One of 'gradient_norm' (L2 norm), 'gradient_saliency' (|grad * param|),
                              or 'fisher' (squared gradient).
            target_modules: Optional list of module substrings to restrict attribution to
                            (e.g., ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]).
            return_dict: If True, returns detailed dict including module_scores; if False, returns Dict[int, float].

        Returns:
            Dict[int, float] mapping layer index to attribution score (if return_dict=False),
            or Dict containing 'layer_scores', 'module_scores', etc. (if return_dict=True).
        """
        self.model.eval()
        self.model.zero_grad()

        layer_scores: Dict[int, float] = {}
        module_scores: Dict[int, Dict[str, float]] = {}

        # Cache and enable requires_grad on parameters for profiling
        original_grad_states = {}
        for name, param in self.model.named_parameters():
            original_grad_states[name] = param.requires_grad
            param.requires_grad = True

        batch_count = 0
        try:
            for batch in tqdm(dataloader, desc=f"Profiling Saliency ({attribution_type})", total=num_batches):
                if batch_count >= num_batches:
                    break

                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = None
                    labels = input_ids

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue

                        # Check target module filter
                        if target_modules is not None:
                            if not any(target in name for target in target_modules):
                                continue

                        layer_idx = self.extract_layer_index(name)
                        if layer_idx < 0:
                            continue

                        mod_name = self.extract_module_name(name)

                        # Attribution calculation
                        if attribution_type == "fisher":
                            # Empirical Fisher Information approx: sum of squared gradients
                            score = (param.grad ** 2).sum().item()
                        elif attribution_type == "gradient_saliency":
                            # Gradient saliency attribution: |grad * param|
                            score = (param.grad * param).abs().sum().item()
                        else:  # "gradient_norm"
                            score = param.grad.norm(2).item()

                        layer_scores[layer_idx] = layer_scores.get(layer_idx, 0.0) + score

                        if layer_idx not in module_scores:
                            module_scores[layer_idx] = {}
                        module_scores[layer_idx][mod_name] = module_scores[layer_idx].get(mod_name, 0.0) + score

                self.model.zero_grad()
                batch_count += 1
        finally:
            # Always restore original requires_grad states
            for name, param in self.model.named_parameters():
                param.requires_grad = original_grad_states.get(name, False)

        # Normalize across batches
        if batch_count > 0:
            for l in layer_scores:
                layer_scores[l] /= batch_count
            for l in module_scores:
                for m in module_scores[l]:
                    module_scores[l][m] /= batch_count

        self.last_module_scores = module_scores

        if return_dict:
            return {
                "layer_scores": layer_scores,
                "module_scores": module_scores,
                "attribution_type": attribution_type,
                "batches_evaluated": batch_count,
            }

        return layer_scores

    @staticmethod
    def get_top_k_layers(layer_scores: Dict[int, float], k: int = 4) -> List[int]:
        """
        Returns the top-K most sensitive layer indices sorted in descending order of sensitivity score.
        """
        sorted_layers = sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
        return [layer_idx for layer_idx, _ in sorted_layers[:k]]
