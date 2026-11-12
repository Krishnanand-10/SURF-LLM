"""
Unit tests for Dual-Objective Loss and SURFTrainer in SURF-LLM.
Verifies forget loss modes (grad_ascent, refusal, npo), forward KL retain loss,
gradient flow isolation, and end-to-end unlearning optimization steps.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.adapters.sparse_lora import SparseLoRAInjector
from surf.trainer.objectives import DualObjectiveLoss
from surf.trainer.trainer import SURFTrainer


class DummyDataset(Dataset):
    def __init__(self, seq_len=16, vocab_size=256, num_samples=6):
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


class TestDualObjectiveTrainer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.vocab_size = 64
        self.device = "cpu"

    def test_gradient_ascent_forget_loss(self):
        """Test gradient ascent inverts cross entropy and bounds loss."""
        loss_fn = DualObjectiveLoss(forget_loss_type="grad_ascent", max_forget_loss=15.0)

        unlearned_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        loss = loss_fn.compute_forget_loss(unlearned_logits=unlearned_logits, labels=labels)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertLessEqual(loss.item(), 15.0)
        self.assertGreaterEqual(loss.item(), -15.0)

        # Verify gradient flow
        loss.backward()
        self.assertIsNotNone(unlearned_logits.grad)

    def test_refusal_forget_loss(self):
        """Test refusal alignment calculates positive cross entropy to refusal target."""
        loss_fn = DualObjectiveLoss(forget_loss_type="refusal")

        unlearned_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        refusal_labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        loss = loss_fn.compute_forget_loss(
            unlearned_logits=unlearned_logits, labels=labels, refusal_labels=refusal_labels
        )
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_npo_forget_loss(self):
        """Test Negative Preference Optimization calculates valid loss with base logits."""
        loss_fn = DualObjectiveLoss(forget_loss_type="npo", npo_beta=0.1)

        unlearned_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        base_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        loss = loss_fn.compute_forget_loss(
            unlearned_logits=unlearned_logits, labels=labels, base_logits=base_logits
        )
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_retain_kl_divergence(self):
        """Test forward KL divergence is zero when unlearned == base logits."""
        loss_fn = DualObjectiveLoss(temperature=1.0)

        logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        kl = loss_fn.compute_retain_kl_loss(unlearned_logits=logits, base_logits=logits)
        # Numerical tolerance around 0
        self.assertAlmostEqual(kl.item(), 0.0, places=4)

        # Perturbed logits must have positive KL divergence
        perturbed_logits = logits + torch.randn_like(logits) * 0.5
        kl_diff = loss_fn.compute_retain_kl_loss(unlearned_logits=perturbed_logits, base_logits=logits)
        self.assertGreater(kl_diff.item(), 0.0)

    def test_composite_loss_forward(self):
        """Test composite dual objective forward pass and metrics output."""
        loss_fn = DualObjectiveLoss(kl_weight=2.5, forget_loss_type="grad_ascent")

        f_unlearned = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        f_labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        r_unlearned = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        r_base = torch.randn(self.batch_size, self.seq_len, self.vocab_size)

        total_loss, metrics = loss_fn(
            forget_unlearned_logits=f_unlearned,
            forget_labels=f_labels,
            retain_unlearned_logits=r_unlearned,
            retain_base_logits=r_base,
        )

        self.assertIn("total_loss", metrics)
        self.assertIn("forget_loss", metrics)
        self.assertIn("retain_kl_loss", metrics)
        self.assertEqual(metrics["kl_weight"], 2.5)

        # Verify math: total = forget + 2.5 * retain
        expected = metrics["forget_loss"] + 2.5 * metrics["retain_kl_loss"]
        self.assertAlmostEqual(total_loss.item(), expected, places=4)

    def test_surftrainer_training_step_and_isolation(self):
        """Test SURFTrainer execution step and confirm only LoRA parameters get gradients."""
        config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        base_model = LlamaForCausalLM(config)
        injector = SparseLoRAInjector(r=4, lora_alpha=8, target_modules=["down_proj"])
        peft_model = injector.inject(base_model, sensitive_layers=[1])

        dataset = DummyDataset(seq_len=8, vocab_size=self.vocab_size, num_samples=4)
        forget_dl = DataLoader(dataset, batch_size=2)
        retain_dl = DataLoader(dataset, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = SURFTrainer(
                model=peft_model,
                forget_dataloader=forget_dl,
                retain_dataloader=retain_dl,
                learning_rate=1e-3,
                kl_weight=1.0,
                forget_loss_type="grad_ascent",
                output_dir=tmp_dir,
            )

            # Check trainable vs frozen parameters before train
            trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
            self.assertGreater(len(trainable_params), 0)

            # Run 1 unlearning optimization step
            forget_batch = next(iter(forget_dl))
            retain_batch = next(iter(retain_dl))
            total_loss, metrics = trainer.train_step(forget_batch, retain_batch)

            self.assertFalse(torch.isnan(total_loss))

            # Verify that only LoRA parameters receive gradients
            for name, p in peft_model.named_parameters():
                if p.requires_grad:
                    self.assertIn("lora_", name)
                    self.assertIsNotNone(p.grad, f"Trainable parameter {name} has no gradient!")
                else:
                    self.assertIsNone(p.grad, f"Frozen parameter {name} received unwanted gradient!")

            # Run end-to-end training loop for 1 epoch
            train_res = trainer.train(num_epochs=1, max_steps=2, save_checkpoint=True)
            self.assertEqual(train_res["final_epoch"], 1)
            self.assertGreater(train_res["global_step"], 0)


if __name__ == "__main__":
    unittest.main()
