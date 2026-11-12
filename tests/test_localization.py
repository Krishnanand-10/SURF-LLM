"""
Unit tests for Knowledge Localization in SURF-LLM.
Tests LayerSaliencyProfiler (gradient saliency & Fisher information)
and ActivationPatchingProfiler (causal attribution & activation patching).
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.localization.saliency import LayerSaliencyProfiler
from surf.localization.causal_trace import ActivationPatchingProfiler, CausalTraceProfiler
from surf.localization.heatmap import generate_layer_heatmaps, save_attribution_report


class DummyDataset(Dataset):
    def __init__(self, seq_len=16, vocab_size=256, num_samples=8):
        self.data = [
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, vocab_size, (seq_len,)),
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestKnowledgeLocalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        cls.model = LlamaForCausalLM(cls.config)
        cls.dataset = DummyDataset(seq_len=16, vocab_size=256, num_samples=8)
        cls.dataloader = DataLoader(cls.dataset, batch_size=2)

    def test_extract_layer_index(self):
        """Test layer index extraction across parameter name variations."""
        profiler = LayerSaliencyProfiler(self.model, tokenizer=None, device="cpu")

        self.assertEqual(profiler.extract_layer_index("model.layers.0.mlp.down_proj.weight"), 0)
        self.assertEqual(profiler.extract_layer_index("model.layers.15.self_attn.q_proj.weight"), 15)
        self.assertEqual(profiler.extract_layer_index("transformer.h.3.attn.c_proj.weight"), -1)
        self.assertEqual(profiler.extract_layer_index("model.embed_tokens.weight"), -1)
        self.assertEqual(profiler.extract_layer_index("lm_head.weight"), -1)

    def test_gradient_saliency_attribution(self):
        """Test gradient saliency attribution profiling on synthetic dataloader."""
        model = LlamaForCausalLM(self.config)
        profiler = LayerSaliencyProfiler(model, tokenizer=None, device="cpu")

        layer_scores = profiler.compute_layer_attributions(
            self.dataloader, num_batches=2, attribution_type="gradient_norm"
        )

        self.assertIsInstance(layer_scores, dict)
        self.assertEqual(len(layer_scores), self.config.num_hidden_layers)
        for layer_idx, score in layer_scores.items():
            self.assertIn(layer_idx, range(self.config.num_hidden_layers))
            self.assertGreaterEqual(score, 0.0)

        top_k = profiler.get_top_k_layers(layer_scores, k=2)
        self.assertEqual(len(top_k), 2)
        self.assertGreaterEqual(layer_scores[top_k[0]], layer_scores[top_k[1]])

    def test_fisher_information_attribution(self):
        """Test Fisher information (squared gradient) attribution profiling."""
        model = LlamaForCausalLM(self.config)
        profiler = LayerSaliencyProfiler(model, tokenizer=None, device="cpu")

        layer_scores = profiler.compute_layer_attributions(
            self.dataloader, num_batches=2, attribution_type="fisher"
        )

        self.assertIsInstance(layer_scores, dict)
        self.assertEqual(len(layer_scores), self.config.num_hidden_layers)
        for layer_idx, score in layer_scores.items():
            self.assertIn(layer_idx, range(self.config.num_hidden_layers))
            self.assertGreaterEqual(score, 0.0)

        top_k = profiler.get_top_k_layers(layer_scores, k=3)
        self.assertEqual(len(top_k), 3)

    def test_activation_patching_profiler(self):
        """Test ActivationPatchingProfiler locates layers and evaluates sensitivity."""
        model = LlamaForCausalLM(self.config)
        profiler = ActivationPatchingProfiler(model, tokenizer=None, device="cpu")

        self.assertEqual(profiler.num_layers(), self.config.num_hidden_layers)

        # Test perturbation mode
        res_perturb = profiler.compute_patching_sensitivity(
            self.dataloader, num_batches=2, noise_std=0.05, intervention_mode="perturbation"
        )
        self.assertIn("layer_scores", res_perturb)
        self.assertEqual(len(res_perturb["layer_scores"]), self.config.num_hidden_layers)

        top_layers = profiler.get_top_k_layers(res_perturb["layer_scores"], k=2)
        self.assertEqual(len(top_layers), 2)

        # Test ablation mode
        res_ablate = profiler.compute_patching_sensitivity(
            self.dataloader, num_batches=1, intervention_mode="ablation"
        )
        self.assertIn("layer_scores", res_ablate)
        self.assertEqual(len(res_ablate["layer_scores"]), self.config.num_hidden_layers)

    def test_clean_corrupted_trace(self):
        """Test mechanistic causal tracing with clean and corrupted input IDs."""
        model = LlamaForCausalLM(self.config)
        profiler = ActivationPatchingProfiler(model, tokenizer=None, device="cpu")

        clean_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        corrupt_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        target_token_idx = 7
        target_token_id = 42

        recovery_scores = profiler.compute_clean_corrupted_trace(
            clean_input_ids=clean_ids,
            corrupted_input_ids=corrupt_ids,
            target_token_index=target_token_idx,
            target_token_id=target_token_id,
        )

        self.assertIsInstance(recovery_scores, dict)
        self.assertEqual(len(recovery_scores), self.config.num_hidden_layers)
        for l_idx, score in recovery_scores.items():
            self.assertGreaterEqual(score, 0.0)

    def test_module_filtering_and_breakdown(self):
        """Test attribution profiling with module filtering and module breakdown report."""
        model = LlamaForCausalLM(self.config)
        profiler = LayerSaliencyProfiler(model, tokenizer=None, device="cpu")

        res = profiler.compute_layer_attributions(
            self.dataloader,
            num_batches=2,
            attribution_type="gradient_saliency",
            target_modules=["mlp.down_proj", "mlp.up_proj"],
            return_dict=True,
        )

        self.assertIn("layer_scores", res)
        self.assertIn("module_scores", res)
        self.assertEqual(len(res["layer_scores"]), self.config.num_hidden_layers)

        # Check module_scores only contain requested target modules
        for l_idx, mods in res["module_scores"].items():
            for m in mods:
                self.assertTrue("mlp.down_proj" in m or "mlp.up_proj" in m)

    def test_heatmap_and_report_generation(self):
        """Test saving JSON attribution report and rendering layer sensitivity heatmaps."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            layer_scores = {0: 0.12, 1: 0.85, 2: 1.45, 3: 0.32}
            top_k = [2, 1]
            module_scores = {
                0: {"mlp.down_proj": 0.05, "self_attn.q_proj": 0.07},
                1: {"mlp.down_proj": 0.40, "self_attn.q_proj": 0.45},
                2: {"mlp.down_proj": 0.90, "self_attn.q_proj": 0.55},
                3: {"mlp.down_proj": 0.10, "self_attn.q_proj": 0.22},
            }

            report_path = os.path.join(tmp_dir, "layer_scores.json")
            heatmap_path = os.path.join(tmp_dir, "heatmap.png")

            report = save_attribution_report(
                layer_scores=layer_scores,
                top_k_layers=top_k,
                module_scores=module_scores,
                output_path=report_path,
                method="gradient_saliency",
            )
            self.assertTrue(os.path.exists(report_path))
            self.assertEqual(report["top_k_layers"], [2, 1])

            saved_plot = generate_layer_heatmaps(
                layer_scores=layer_scores,
                top_k_layers=top_k,
                module_scores=module_scores,
                save_path=heatmap_path,
            )
            self.assertTrue(os.path.exists(heatmap_path))


if __name__ == "__main__":
    unittest.main()
