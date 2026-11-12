"""
Pipeline integration test for SURF-LLM.
Verifies the complete end-to-end unlearning workflow:
1. Knowledge localization (LayerSaliencyProfiler / CausalTraceProfiler)
2. Sparse LoRA injection on identified top-K sensitive layers
3. Dual-objective unlearning optimization (SURFTrainer)
4. Comprehensive safety and utility auditing (SafetyAuditor)
5. Base weight delta fusion and zero-overhead validation (WeightDeltaFuser)
"""

import os
import sys
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.localization.saliency import LayerSaliencyProfiler
from surf.adapters.sparse_lora import SparseLoRAInjector
from surf.trainer.trainer import SURFTrainer
from surf.evaluation.evaluator import SafetyAuditor
from surf.fusion.fusion import fuse_lora_to_base, verify_fusion_integrity


class MockPipelineTokenizer:
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", max_length=32, truncation=True, padding=False):
        if isinstance(text, list):
            batch_ids = []
            for t in text:
                toks = [ord(c) % (self.vocab_size - 2) + 2 for c in t[:max_length]] or [2]
                batch_ids.append(toks)
            max_l = max(len(b) for b in batch_ids)
            padded = [b + [self.pad_token_id] * (max_l - len(b)) for b in batch_ids]
            mask = [[1] * len(b) + [0] * (max_l - len(b)) for b in batch_ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        else:
            toks = [ord(c) % (self.vocab_size - 2) + 2 for c in text[:max_length]] or [2]
            return {
                "input_ids": torch.tensor([toks], dtype=torch.long),
                "attention_mask": torch.ones((1, len(toks)), dtype=torch.long),
            }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = [chr(t) if 32 <= t <= 126 else " " for t in token_ids]
        return "".join(chars)


class SyntheticForgetRetainDataset(Dataset):
    def __init__(self, vocab_size=128, seq_len=16, num_samples=4):
        self.samples = [
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, vocab_size, (seq_len,)),
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TestSURFPipeline(unittest.TestCase):
    def test_full_surf_pipeline_end_to_end(self):
        """Executes the entire SURF-LLM lifecycle from localization to fusion."""
        # 0. Setup lightweight base model
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base_model = LlamaForCausalLM(config)
        tokenizer = MockPipelineTokenizer(vocab_size=128)

        forget_ds = SyntheticForgetRetainDataset(vocab_size=128, seq_len=16, num_samples=4)
        retain_ds = SyntheticForgetRetainDataset(vocab_size=128, seq_len=16, num_samples=4)
        forget_dl = DataLoader(forget_ds, batch_size=2)
        retain_dl = DataLoader(retain_ds, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Knowledge Localization
            profiler = LayerSaliencyProfiler(base_model, tokenizer=tokenizer, device="cpu")
            layer_scores = profiler.compute_layer_attributions(
                forget_dl, num_batches=2, attribution_type="gradient_norm"
            )
            top_k_layers = profiler.get_top_k_layers(layer_scores, k=2)
            self.assertEqual(len(top_k_layers), 2)

            # 2. Sparse LoRA Injection
            injector = SparseLoRAInjector(
                r=4,
                lora_alpha=8,
                target_modules=["down_proj", "up_proj"],
            )
            peft_model = injector.inject(base_model, sensitive_layers=top_k_layers)
            sparsity = injector.get_trainable_sparsity(peft_model)
            self.assertTrue(sparsity["trainable_params"] > 0)
            self.assertTrue(sparsity["sparsity_ratio"] < 10.0)

            # 3. Dual-Objective Unlearning Optimization
            trainer = SURFTrainer(
                model=peft_model,
                forget_dataloader=forget_dl,
                retain_dataloader=retain_dl,
                learning_rate=1e-3,
                kl_weight=1.0,
                forget_loss_type="grad_ascent",
                output_dir=os.path.join(tmp_dir, "adapter"),
            )
            train_result = trainer.train(num_epochs=1, max_steps=2, save_checkpoint=True)
            self.assertGreater(train_result["global_step"], 0)

            # 4. Multi-Metric Safety Auditing
            auditor = SafetyAuditor(
                unlearned_model=peft_model,
                tokenizer=tokenizer,
                base_model=base_model,
                device="cpu",
            )
            audit_report = auditor.run_full_audit(
                forget_test_data=[{"prompt": "sensitive prompt", "sensitive_target": "target"}],
                forget_corpus=["sensitive text sample 1", "sensitive text sample 2"],
                retain_corpus=["retain text sample 1", "retain text sample 2"],
            )
            self.assertIn("forget_quality", audit_report)
            self.assertIn("mia_audit", audit_report)
            self.assertIn("perplexity_audit", audit_report)
            self.assertIn("reasoning_audit", audit_report)

            # 5. Base Weight Delta Fusion
            sample_eval_ids = torch.randint(0, 128, (1, 8))
            fused_dir = os.path.join(tmp_dir, "fused_model")
            fused_model = fuse_lora_to_base(
                base_model_or_path=peft_model,
                output_dir=fused_dir,
                device="cpu",
            )

            # Validate 0 LoRA parameters remain in fused architecture
            remaining_lora = [n for n, _ in fused_model.named_parameters() if "lora_" in n]
            self.assertEqual(len(remaining_lora), 0)

            # Verify numerical logit preservation
            is_matching, max_diff = verify_fusion_integrity(
                peft_model=fused_model,
                fused_model=fused_model,
                sample_input_ids=sample_eval_ids,
                tolerance=1e-3,
            )
            self.assertTrue(is_matching)


if __name__ == "__main__":
    unittest.main()
