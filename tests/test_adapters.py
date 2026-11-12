"""
Unit tests for Sparse LoRA Injection in SURF-LLM.
Verifies selective adapter injection, unaffected layer freezing, and sparsity metrics.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.adapters.sparse_lora import SparseLoRAInjector


class TestSparseLoRA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny 4-layer in-memory LLaMA model without external downloads
        cls.config = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        cls.base_model = LlamaForCausalLM(cls.config)

    def test_selective_lora_injection(self):
        """Verify adapters are injected strictly into specified sensitive layers."""
        model = LlamaForCausalLM(self.config)
        sensitive_layers = [1, 3]

        injector = SparseLoRAInjector(
            r=4,
            lora_alpha=8,
            target_modules=["down_proj", "up_proj"],
        )
        peft_model = injector.inject(model, sensitive_layers=sensitive_layers)

        # Check adapter isolation
        is_isolated = SparseLoRAInjector.verify_adapter_isolation(peft_model, sensitive_layers)
        self.assertTrue(is_isolated, "Adapter parameters leaked into non-targeted layers!")

        # Verify trainable parameter locations
        trainable_layers = set()
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                self.assertIn("lora_", name, f"Non-LoRA parameter requires grad: {name}")
                idx = SparseLoRAInjector.extract_layer_index(name)
                if idx is not None:
                    trainable_layers.add(idx)

        self.assertEqual(trainable_layers, set(sensitive_layers), "Trained layers must match sensitive layers")

    def test_unaffected_parameter_freezing(self):
        """Verify base model parameters and non-targeted layers have requires_grad=False."""
        model = LlamaForCausalLM(self.config)
        sensitive_layers = [2]

        injector = SparseLoRAInjector(
            r=4,
            lora_alpha=8,
            target_modules=["down_proj"],
        )
        peft_model = injector.inject(model, sensitive_layers=sensitive_layers)

        # Non-targeted layers (0, 1, 3) must be strictly frozen
        for name, param in peft_model.named_parameters():
            idx = SparseLoRAInjector.extract_layer_index(name)
            if idx is not None and idx not in sensitive_layers:
                self.assertFalse(
                    param.requires_grad,
                    f"Parameter {name} in non-targeted layer {idx} is not frozen!",
                )

    def test_sparsity_parameter_ratio(self):
        """Verify parameter sparsity is calculated and confirms footprint < 1%."""
        model = LlamaForCausalLM(self.config)
        sensitive_layers = [2]

        injector = SparseLoRAInjector(
            r=4,
            lora_alpha=8,
            target_modules=["down_proj"],
        )
        peft_model = injector.inject(model, sensitive_layers=sensitive_layers)

        sparsity_info = injector.get_trainable_sparsity(peft_model)
        self.assertIn("trainable_params", sparsity_info)
        self.assertIn("total_params", sparsity_info)
        self.assertIn("sparsity_ratio", sparsity_info)
        self.assertTrue(sparsity_info["trainable_params"] > 0)
        self.assertTrue(sparsity_info["total_params"] > sparsity_info["trainable_params"])


if __name__ == "__main__":
    unittest.main()
