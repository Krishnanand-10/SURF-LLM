"""
Causal Attribution & Activation Patching Profiler for SURF-LLM.
Pinpoints transformer layers by tracing causal mediation and measuring
sensitivity to activation interventions on the forget set.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm


class ActivationPatchingProfiler:
    """
    Profiles layer-wise causal sensitivity by performing activation patching
    and intervention analysis across transformer layers.
    """

    def __init__(self, model: nn.Module, tokenizer=None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = self._locate_layers()

    def _locate_layers(self) -> nn.ModuleList:
        """
        Locates the sequence of transformer blocks/layers in the model.
        Supports LLaMA, Mistral, Qwen, GPT-2, GPT-NeoX, Falcon, etc.
        """
        candidates = [
            "model.layers",            # LLaMA, Mistral, Qwen, Gemma
            "transformer.h",           # GPT-2, Falcon
            "gpt_neox.layers",         # Pythia, GPT-NeoX
            "encoder.layer",           # BERT, RoBERTa
            "decoder.layers",          # OPT
            "model.decoder.layers",    # Bart
        ]
        for candidate in candidates:
            parts = candidate.split(".")
            curr = self.model
            found = True
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)):
                return curr

        # Fallback: scan submodules for a ModuleList containing layer-like children
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                child_names = [n for n, _ in module.named_children()]
                if all(c.isdigit() for c in child_names):
                    return module

        raise ValueError(
            "Could not automatically locate transformer layers in the model. "
            "Ensure the model follows standard architectures (e.g., model.layers)."
        )

    def num_layers(self) -> int:
        """Returns the total number of transformer layers."""
        return len(self.layers)

    def compute_patching_sensitivity(
        self,
        dataloader,
        num_batches: int = 10,
        noise_std: float = 0.1,
        intervention_mode: str = "perturbation",  # "perturbation" or "ablation"
    ) -> Dict[str, Any]:
        """
        Computes layer sensitivity using activation intervention.
        For each layer, perturbs or ablates intermediate hidden representations
        and measures the magnitude of shift in loss/log-likelihood on sensitive tokens.

        Args:
            dataloader: Iterable delivering forget dataset batches.
            num_batches: Maximum number of batches to evaluate.
            noise_std: Standard deviation scale of Gaussian perturbation relative to activation std.
            intervention_mode: "perturbation" (additive noise) or "ablation" (zero-out activations).

        Returns:
            Dictionary containing:
              - 'layer_scores': Dict[int, float] mapping layer index to causal sensitivity score
              - 'attribution_type': "activation_patching"
              - 'batches_evaluated': int
        """
        self.model.eval()
        num_layers = self.num_layers()
        layer_scores: Dict[int, float] = {l: 0.0 for l in range(num_layers)}

        batch_count = 0
        for batch in tqdm(dataloader, desc="Activation Patching Profiling", total=num_batches):
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

            # Baseline clean forward pass
            with torch.no_grad():
                clean_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                clean_loss = clean_outputs.loss if hasattr(clean_outputs, "loss") else clean_outputs[0]
                clean_loss_val = clean_loss.item()

            # Layer-wise intervention evaluation
            for layer_idx in range(num_layers):
                hook_handle = None

                def hook_fn(module, input_val, output_val):
                    # Handle tuple outputs (common in transformer layers: (hidden_states, ...))
                    if isinstance(output_val, tuple):
                        h = output_val[0]
                        rest = output_val[1:]
                        if intervention_mode == "ablation":
                            h_perturbed = torch.zeros_like(h)
                        else:
                            noise = torch.randn_like(h) * (h.std().item() * noise_std + 1e-8)
                            h_perturbed = h + noise
                        return (h_perturbed,) + rest
                    else:
                        h = output_val
                        if intervention_mode == "ablation":
                            return torch.zeros_like(h)
                        noise = torch.randn_like(h) * (h.std().item() * noise_std + 1e-8)
                        return h + noise

                target_layer = self.layers[layer_idx]
                hook_handle = target_layer.register_forward_hook(hook_fn)

                try:
                    with torch.no_grad():
                        perturbed_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        perturbed_loss = perturbed_outputs.loss if hasattr(perturbed_outputs, "loss") else perturbed_outputs[0]
                        # Sensitivity: magnitude of loss change when layer activations are intervened on
                        sensitivity = abs(perturbed_loss.item() - clean_loss_val)
                        layer_scores[layer_idx] += sensitivity
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()

            batch_count += 1

        # Normalize by batch count
        if batch_count > 0:
            for l in layer_scores:
                layer_scores[l] /= batch_count

        return {
            "layer_scores": layer_scores,
            "attribution_type": f"activation_patching_{intervention_mode}",
            "batches_evaluated": batch_count,
        }

    def compute_clean_corrupted_trace(
        self,
        clean_input_ids: torch.Tensor,
        corrupted_input_ids: torch.Tensor,
        target_token_index: int,
        target_token_id: int,
    ) -> Dict[int, float]:
        """
        Executes mechanistic causal tracing (ROME-style):
        Patches clean activations from each layer into corrupted forward pass
        to measure recovery of the sensitive target token probability.

        Args:
            clean_input_ids: Tensor of shape (1, seq_len) for clean sensitive prompt.
            corrupted_input_ids: Tensor of shape (1, seq_len) with corrupted subject tokens.
            target_token_index: Index of the prediction token.
            target_token_id: Token ID of the sensitive completion token.

        Returns:
            Dict[int, float] mapping layer index to recovery score (prob diff).
        """
        self.model.eval()
        clean_input_ids = clean_input_ids.to(self.device)
        corrupted_input_ids = corrupted_input_ids.to(self.device)
        num_layers = self.num_layers()

        # Step 1: Capture clean activations
        clean_activations = {}
        handles = []
        for l_idx, layer in enumerate(self.layers):
            def make_save_hook(idx):
                def save_hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    clean_activations[idx] = h.detach().clone()
                return save_hook
            handles.append(layer.register_forward_hook(make_save_hook(l_idx)))

        with torch.no_grad():
            self.model(input_ids=clean_input_ids)

        for h in handles:
            h.remove()

        # Step 2: Baseline corrupted probability
        with torch.no_grad():
            corrupt_out = self.model(input_ids=corrupted_input_ids)
            corrupt_logits = corrupt_out.logits[0, target_token_index, :]
            corrupt_prob = torch.softmax(corrupt_logits, dim=-1)[target_token_id].item()

        # Step 3: Layer-by-layer patching
        recovery_scores: Dict[int, float] = {}
        for l_idx in range(num_layers):
            def make_patch_hook(idx):
                def patch_hook(mod, inp, out):
                    clean_h = clean_activations[idx]
                    if isinstance(out, tuple):
                        return (clean_h,) + out[1:]
                    return clean_h
                return patch_hook

            patch_handle = self.layers[l_idx].register_forward_hook(make_patch_hook(l_idx))
            try:
                with torch.no_grad():
                    patched_out = self.model(input_ids=corrupted_input_ids)
                    patched_logits = patched_out.logits[0, target_token_index, :]
                    patched_prob = torch.softmax(patched_logits, dim=-1)[target_token_id].item()
                    recovery_scores[l_idx] = max(0.0, patched_prob - corrupt_prob)
            finally:
                patch_handle.remove()

        return recovery_scores

    @staticmethod
    def get_top_k_layers(layer_scores: Dict[int, float], k: int = 4) -> List[int]:
        """
        Returns top-K sensitive layer indices sorted in descending order of sensitivity score.
        """
        sorted_layers = sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
        return [layer_idx for layer_idx, _ in sorted_layers[:k]]


# Alias for backwards and conceptual compatibility
CausalTraceProfiler = ActivationPatchingProfiler
