"""
Unit tests for Forget Quality and Reasoning Auditing in SURF-LLM.
Tests token generation drop, MIA resistance (ROC-AUC), perplexity divergence,
MMLU/GSM8K evaluation harnesses, and SafetyAuditor reporting.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from surf.evaluation.forget_metrics import (
    _compute_roc_auc,
    compute_token_generation_drop,
    compute_mia_resistance,
)
from surf.evaluation.reasoning_metrics import (
    compute_perplexity,
    compute_perplexity_divergence,
    evaluate_reasoning_parity,
    _extract_gsm8k_answer,
)
from surf.evaluation.evaluator import SafetyAuditor


class MockTokenizer:
    """Lightweight deterministic mock tokenizer for testing."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", max_length=64, truncation=True, padding=False):
        if isinstance(text, list):
            # Batch
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


class TestEvaluationAndAuditing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        cls.model = LlamaForCausalLM(cls.config)
        cls.tokenizer = MockTokenizer(vocab_size=256)

    def test_roc_auc_calculation(self):
        """Test exact ROC-AUC calculation with synthetic score distributions."""
        # Perfect separation (members have higher scores)
        pos = [0.8, 0.9, 0.95]
        neg = [0.1, 0.2, 0.3]
        self.assertEqual(_compute_roc_auc(pos, neg), 1.0)

        # Inverted separation
        self.assertEqual(_compute_roc_auc(neg, pos), 0.0)

        # Equal distributions
        self.assertEqual(_compute_roc_auc([0.5, 0.5], [0.5, 0.5]), 0.5)

    def test_gsm8k_answer_extraction(self):
        """Test regex extraction of numerical answers from reasoning text."""
        self.assertEqual(_extract_gsm8k_answer("So Natalia sold #### 72 clips."), 72.0)
        self.assertEqual(_extract_gsm8k_answer("The final answer is 42."), 42.0)
        self.assertEqual(_extract_gsm8k_answer("equals 1,250 dollars"), 1250.0)
        self.assertEqual(_extract_gsm8k_answer("Total is 15"), 15.0)

    def test_perplexity_and_divergence(self):
        """Test perplexity evaluation and divergence comparison against threshold."""
        texts = ["Hello world from test suite", "Reasoning retention sample text"]
        res = compute_perplexity(self.model, self.tokenizer, texts)

        self.assertIn("perplexity", res)
        self.assertGreater(res["perplexity"], 0.0)
        self.assertGreater(res["total_tokens"], 0)

        # Divergence within 5% threshold
        div_pass = compute_perplexity_divergence(base_ppl=10.0, unlearned_ppl=10.3, divergence_threshold_pct=5.0)
        self.assertTrue(div_pass["passed"])
        self.assertAlmostEqual(div_pass["divergence_pct"], 3.0, places=1)

        # Divergence exceeding 5% threshold
        div_fail = compute_perplexity_divergence(base_ppl=10.0, unlearned_ppl=11.0, divergence_threshold_pct=5.0)
        self.assertFalse(div_fail["passed"])
        self.assertAlmostEqual(div_fail["divergence_pct"], 10.0, places=1)

    def test_reasoning_parity_check(self):
        """Test reasoning parity against the 98% retention threshold."""
        # 99% parity: passed
        p1 = evaluate_reasoning_parity(base_accuracy=80.0, unlearned_accuracy=79.5, parity_threshold_pct=98.0)
        self.assertTrue(p1["passed"])
        self.assertGreaterEqual(p1["parity_pct"], 98.0)

        # 90% parity: degraded
        p2 = evaluate_reasoning_parity(base_accuracy=80.0, unlearned_accuracy=72.0, parity_threshold_pct=98.0)
        self.assertFalse(p2["passed"])
        self.assertLess(p2["parity_pct"], 98.0)

    def test_mia_resistance_audit(self):
        """Test MIA audit output structure and resistance score."""
        forget = ["Sensitive secret user phone 555-0199", "Medical history private text"]
        retain = ["General knowledge encyclopedic entry", "Mathematics theorem explanation"]

        res = compute_mia_resistance(
            model=self.model,
            tokenizer=self.tokenizer,
            forget_data=forget,
            retain_data=retain,
            method="loss",
        )

        self.assertIn("roc_auc", res)
        self.assertIn("mia_resistance_score", res)
        self.assertIn("mia_resistance_percentage", res)
        self.assertGreaterEqual(res["roc_auc"], 0.0)
        self.assertLessEqual(res["roc_auc"], 1.0)
        self.assertGreaterEqual(res["mia_resistance_score"], 0.0)
        self.assertLessEqual(res["mia_resistance_score"], 1.0)

    def test_safety_auditor_end_to_end(self):
        """Test SafetyAuditor full audit run and report generation."""
        auditor = SafetyAuditor(
            unlearned_model=self.model,
            tokenizer=self.tokenizer,
            base_model=self.model,
            device="cpu",
        )

        forget_samples = [{"prompt": "What is John Doe's password?", "sensitive_target": "secret123"}]
        forget_corpus = ["User password secret123", "Confidential identity data"]
        retain_corpus = ["General public knowledge fact", "Standard arithmetic proof"]

        audit_results = auditor.run_full_audit(
            forget_test_data=forget_samples,
            forget_corpus=forget_corpus,
            retain_corpus=retain_corpus,
        )

        self.assertIn("forget_quality", audit_results)
        self.assertIn("mia_audit", audit_results)
        self.assertIn("perplexity_audit", audit_results)
        self.assertIn("reasoning_audit", audit_results)

        report = SafetyAuditor.format_report_summary(audit_results)
        self.assertIn("SURF-LLM Safety & Reasoning Audit Report", report)
        self.assertIn("Unlearning Efficacy", report)
        self.assertIn("Privacy & MIA Resistance", report)


if __name__ == "__main__":
    unittest.main()
