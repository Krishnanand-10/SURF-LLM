"""
Unified Safety and Utility Auditor for SURF-LLM.
Orchestrates forget quality evaluation, MIA resistance auditing, and reasoning integrity benchmarks.
"""

import json
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from .forget_metrics import compute_token_generation_drop, compute_mia_resistance
from .reasoning_metrics import (
    compute_perplexity,
    compute_perplexity_divergence,
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_reasoning_parity,
)


class SafetyAuditor:
    """
    Comprehensive auditor for evaluating unlearned models against safety and reasoning criteria.
    """

    def __init__(
        self,
        unlearned_model: nn.Module,
        tokenizer: Any,
        base_model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.unlearned_model = unlearned_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device

    def audit_forget_quality(
        self,
        test_data: List[Union[Dict[str, Any], tuple]],
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        """Audits sensitive concept generation drop."""
        return compute_token_generation_drop(
            unlearned_model=self.unlearned_model,
            tokenizer=self.tokenizer,
            test_data=test_data,
            base_model=self.base_model,
            device=self.device,
            max_new_tokens=max_new_tokens,
        )

    def audit_mia_resistance(
        self,
        forget_data: List[Union[str, Dict[str, Any]]],
        retain_data: List[Union[str, Dict[str, Any]]],
        method: str = "min_k_prob",
        k_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """Audits membership inference attack resistance on forget vs retain samples."""
        return compute_mia_resistance(
            model=self.unlearned_model,
            tokenizer=self.tokenizer,
            forget_data=forget_data,
            retain_data=retain_data,
            device=self.device,
            method=method,
            k_ratio=k_ratio,
        )

    def audit_perplexity(
        self,
        retain_texts: List[str],
        divergence_threshold_pct: float = 5.0,
    ) -> Dict[str, Any]:
        """Audits retain corpus perplexity and divergence against base model baseline."""
        unlearned_res = compute_perplexity(
            model=self.unlearned_model,
            tokenizer=self.tokenizer,
            texts=retain_texts,
            device=self.device,
        )

        base_res = None
        divergence = None
        if self.base_model is not None:
            base_res = compute_perplexity(
                model=self.base_model,
                tokenizer=self.tokenizer,
                texts=retain_texts,
                device=self.device,
            )
            divergence = compute_perplexity_divergence(
                base_ppl=base_res["perplexity"],
                unlearned_ppl=unlearned_res["perplexity"],
                divergence_threshold_pct=divergence_threshold_pct,
            )

        return {
            "unlearned_perplexity": unlearned_res["perplexity"],
            "base_perplexity": base_res["perplexity"] if base_res else None,
            "divergence": divergence,
        }

    def audit_reasoning(
        self,
        mmlu_samples: Optional[List[Dict[str, Any]]] = None,
        gsm8k_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Audits MMLU and GSM8K reasoning benchmark accuracy and parity."""
        unlearned_mmlu = evaluate_mmlu(
            model=self.unlearned_model,
            tokenizer=self.tokenizer,
            samples=mmlu_samples,
            device=self.device,
        )
        unlearned_gsm8k = evaluate_gsm8k(
            model=self.unlearned_model,
            tokenizer=self.tokenizer,
            samples=gsm8k_samples,
            device=self.device,
        )

        base_mmlu = None
        base_gsm8k = None
        mmlu_parity = None
        gsm8k_parity = None

        if self.base_model is not None:
            base_mmlu = evaluate_mmlu(
                model=self.base_model,
                tokenizer=self.tokenizer,
                samples=mmlu_samples,
                device=self.device,
            )
            base_gsm8k = evaluate_gsm8k(
                model=self.base_model,
                tokenizer=self.tokenizer,
                samples=gsm8k_samples,
                device=self.device,
            )
            mmlu_parity = evaluate_reasoning_parity(
                base_accuracy=base_mmlu["accuracy"],
                unlearned_accuracy=unlearned_mmlu["accuracy"],
            )
            gsm8k_parity = evaluate_reasoning_parity(
                base_accuracy=base_gsm8k["accuracy"],
                unlearned_accuracy=unlearned_gsm8k["accuracy"],
            )

        return {
            "mmlu": {
                "unlearned": unlearned_mmlu,
                "base": base_mmlu,
                "parity": mmlu_parity,
            },
            "gsm8k": {
                "unlearned": unlearned_gsm8k,
                "base": base_gsm8k,
                "parity": gsm8k_parity,
            },
        }

    def run_full_audit(
        self,
        forget_test_data: Optional[List[Dict[str, Any]]] = None,
        forget_corpus: Optional[List[Union[str, Dict[str, Any]]]] = None,
        retain_corpus: Optional[List[Union[str, Dict[str, Any]]]] = None,
        mmlu_samples: Optional[List[Dict[str, Any]]] = None,
        gsm8k_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-metric safety and utility audit.
        """
        results: Dict[str, Any] = {}

        if forget_test_data:
            results["forget_quality"] = self.audit_forget_quality(forget_test_data)

        if forget_corpus and retain_corpus:
            results["mia_audit"] = self.audit_mia_resistance(
                forget_data=forget_corpus,
                retain_data=retain_corpus,
            )

        if retain_corpus:
            retain_texts = [
                s if isinstance(s, str) else s.get("text", s.get("prompt", ""))
                for s in retain_corpus
            ]
            results["perplexity_audit"] = self.audit_perplexity(retain_texts)

        results["reasoning_audit"] = self.audit_reasoning(
            mmlu_samples=mmlu_samples,
            gsm8k_samples=gsm8k_samples,
        )

        return results

    @staticmethod
    def format_report_summary(results: Dict[str, Any]) -> str:
        """Formats audit results into a readable markdown report."""
        lines = [
            "# SURF-LLM Safety & Reasoning Audit Report",
            "=" * 50,
        ]

        if "forget_quality" in results:
            fq = results["forget_quality"]
            lines.append("## 1. Unlearning Efficacy (Forget Quality)")
            lines.append(f"- Sensitive Generation Rate (Unlearned): {fq.get('unlearned_generation_rate', 0.0):.2%}")
            lines.append(f"- Sensitive Generation Rate (Base):      {fq.get('base_generation_rate', 0.0):.2%}")
            lines.append(f"- Recall Drop Rate:                     {fq.get('generation_drop_rate', 0.0):.1f}%")
            lines.append(f"- Efficacy Score:                       {fq.get('unlearning_efficacy_score', 0.0):.4f}")
            lines.append("")

        if "mia_audit" in results:
            mia = results["mia_audit"]
            lines.append("## 2. Privacy & MIA Resistance")
            lines.append(f"- MIA ROC-AUC:                          {mia.get('roc_auc', 0.5):.4f} (Ideal: ~0.50)")
            lines.append(f"- MIA Resistance Score:                 {mia.get('mia_resistance_percentage', 100.0):.1f}%")
            lines.append(f"- Attack Method:                        {mia.get('method', 'min_k_prob')}")
            lines.append("")

        if "perplexity_audit" in results:
            pa = results["perplexity_audit"]
            lines.append("## 3. Retain Fluency (Perplexity)")
            lines.append(f"- Unlearned Perplexity:                 {pa.get('unlearned_perplexity', 0.0):.2f}")
            if pa.get("base_perplexity"):
                lines.append(f"- Base Model Perplexity:                {pa.get('base_perplexity', 0.0):.2f}")
            if pa.get("divergence"):
                div = pa["divergence"]
                status = "PASSED (< 5%)" if div.get("passed") else "FAILED (> 5%)"
                lines.append(f"- Perplexity Divergence:                {div.get('divergence_pct', 0.0):.2f}% [{status}]")
            lines.append("")

        if "reasoning_audit" in results:
            ra = results["reasoning_audit"]
            lines.append("## 4. Reasoning Preservation (MMLU & GSM8K)")
            mmlu = ra.get("mmlu", {})
            unl_mmlu = mmlu.get("unlearned", {})
            lines.append(f"- MMLU Accuracy (Unlearned):            {unl_mmlu.get('accuracy', 0.0):.1f}%")
            if mmlu.get("parity"):
                p = mmlu["parity"]
                status = "PASSED (>= 98%)" if p.get("passed") else "DEGRADED (< 98%)"
                lines.append(f"  └ Parity Retention:                   {p.get('parity_pct', 0.0):.1f}% [{status}]")

            gsm8k = ra.get("gsm8k", {})
            unl_gsm = gsm8k.get("unlearned", {})
            lines.append(f"- GSM8K Math Accuracy (Unlearned):      {unl_gsm.get('accuracy', 0.0):.1f}%")
            if gsm8k.get("parity"):
                p = gsm8k["parity"]
                status = "PASSED (>= 98%)" if p.get("passed") else "DEGRADED (< 98%)"
                lines.append(f"  └ Parity Retention:                   {p.get('parity_pct', 0.0):.1f}% [{status}]")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)


# Alias for backward and architectural compatibility
SURFEvaluator = SafetyAuditor
