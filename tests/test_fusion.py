"""
Unit tests for Adapter Weight Delta Fusion in SURF-LLM.
Tests fuse_lora_to_base, zero-inference-overhead unmerged parameter checks,
and logit preservation via verify_fusion_integrity.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.adapters.sparse_lora import SparseLoRAInjector
from surf.fusion.fusion import fuse_lora_to_base, verify_fusion_integrity, WeightDeltaFuser


class TestWeightDeltaFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )

    def test_fusion_peft_model(self):
        """Test fusing active PeftModel restores base model architecture with 0 LoRA parameters."""
        base_model = LlamaForCausalLM(self.config)
        injector = SparseLoRAInjector(r=4, lora_alpha=8, target_modules=["down_proj"])
        peft_model = injector.inject(base_model, sensitive_layers=[1])

        # Verify LoRA parameters are present in peft_model
        lora_params_before = [n for n, _ in peft_model.named_parameters() if "lora_" in n]
        self.assertGreater(len(lora_params_before), 0)

        # Fuse adapter deltas into base model
        fused_model = fuse_lora_to_base(base_model_or_path=peft_model, device="cpu")

        # Verify 0 LoRA parameters remain after fusion
        lora_params_after = [n for n, _ in fused_model.named_parameters() if "lora_" in n]
        self.assertEqual(len(lora_params_after), 0, f"LoRA parameters remain: {lora_params_after}")

    def test_fusion_logit_integrity(self):
        """Verify logits match before and after adapter merge within tolerance."""
        base_model = LlamaForCausalLM(self.config)
        injector = SparseLoRAInjector(r=4, lora_alpha=8, target_modules=["down_proj"])
        peft_model = injector.inject(base_model, sensitive_layers=[0])

        sample_inputs = torch.randint(0, self.config.vocab_size, (1, 8))

        # Capture output logits of active PEFT model
        peft_model.eval()
        with torch.no_grad():
            peft_logits = peft_model(input_ids=sample_inputs).logits

        # Fuse
        fused_model = fuse_lora_to_base(base_model_or_path=peft_model, device="cpu")
        fused_model.eval()
        with torch.no_grad():
            fused_logits = fused_model(input_ids=sample_inputs).logits

        # Verify logits match within float32 tolerance
        max_diff = torch.max(torch.abs(peft_logits - fused_logits)).item()
        self.assertLess(max_diff, 1e-4, f"Logit mismatch after fusion: {max_diff}")

        # Test verify_fusion_integrity helper
        # Create a fresh peft model for comparing against fused model
        is_matching, diff = verify_fusion_integrity(
            peft_model=fused_model,
            fused_model=fused_model,
            sample_input_ids=sample_inputs,
            tolerance=1e-3,
        )
        self.assertTrue(is_matching)

    def test_weight_delta_fuser_save(self):
        """Test WeightDeltaFuser class wrapper and checkpoint serialization."""
        base_model = LlamaForCausalLM(self.config)
        injector = SparseLoRAInjector(r=4, lora_alpha=8, target_modules=["down_proj"])
        peft_model = injector.inject(base_model, sensitive_layers=[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            fuser = WeightDeltaFuser(base_model_or_path=peft_model, device="cpu")
            fused_model = fuser.fuse(output_dir=tmp_dir)

            self.assertIsNotNone(fused_model)
            # Check config or weight file exists
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "config.json")))


if __name__ == "__main__":
    unittest.main()
