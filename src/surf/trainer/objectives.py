"""
Dual-Objective Loss Formulations for SURF-LLM.
Balances sensitive concept suppression (Forget Loss) with
distributional anchoring on benign retain data (Forward KL Divergence).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualObjectiveLoss(nn.Module):
    """
    Computes the composite unlearning loss:
        L_total = L_forget + lambda * D_KL(P_base || P_unlearned)

    Attributes:
        kl_weight (float): Regularization weight lambda for benign retain KL preservation.
        forget_loss_type (str): "grad_ascent", "refusal", or "npo".
        temperature (float): Softmax temperature for KL divergence.
        npo_beta (float): Beta hyperparameter for Negative Preference Optimization.
        max_forget_loss (float): Numerical safety bound for gradient ascent.
        ignore_index (int): Token index to ignore in loss calculation (default -100).
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        forget_loss_type: str = "grad_ascent",
        temperature: float = 1.0,
        npo_beta: float = 0.1,
        max_forget_loss: float = 20.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.forget_loss_type = forget_loss_type.lower()
        self.temperature = temperature
        self.npo_beta = npo_beta
        self.max_forget_loss = max_forget_loss
        self.ignore_index = ignore_index

    def compute_forget_loss(
        self,
        unlearned_logits: torch.Tensor,
        labels: torch.Tensor,
        refusal_labels: Optional[torch.Tensor] = None,
        base_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the forget loss component according to the configured loss type.

        Args:
            unlearned_logits: [batch_size, seq_len, vocab_size] from unlearned model.
            labels: [batch_size, seq_len] ground truth target tokens.
            refusal_labels: Optional [batch_size, seq_len] refusal target tokens.
            base_logits: Optional [batch_size, seq_len, vocab_size] from frozen base model (for NPO).
            attention_mask: Optional [batch_size, seq_len] binary mask.

        Returns:
            forget_loss: Scalar torch tensor.
        """
        shift_logits = unlearned_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        if self.forget_loss_type == "grad_ascent":
            # Gradient Ascent: Invert cross-entropy loss to penalize sensitive token generation
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            valid_mask = (shift_labels.view(-1) != self.ignore_index)
            if valid_mask.sum() > 0:
                ce_loss = token_losses[valid_mask].mean()
            else:
                ce_loss = token_losses.mean()

            # Penalize by negating cross entropy
            forget_loss = -ce_loss
            # Apply safety clamping to prevent unbounded numerical gradient explosion
            if self.max_forget_loss is not None:
                forget_loss = torch.clamp(forget_loss, min=-self.max_forget_loss, max=self.max_forget_loss)

            return forget_loss

        elif self.forget_loss_type == "refusal":
            # Refusal alignment: Optimize cross-entropy against neutral refusal targets
            target_labels = refusal_labels if refusal_labels is not None else labels
            shift_target = target_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            return loss_fct(shift_logits.view(-1, vocab_size), shift_target.view(-1))

        elif self.forget_loss_type == "npo":
            # Negative Preference Optimization (NPO)
            if base_logits is None:
                raise ValueError("base_logits must be provided when forget_loss_type is 'npo'.")

            shift_base_logits = base_logits[..., :-1, :].contiguous()

            # Compute log probs for tokens
            unlearned_log_probs = F.log_softmax(shift_logits, dim=-1)
            base_log_probs = F.log_softmax(shift_base_logits, dim=-1)

            # Gather log probs for actual label tokens
            mask = (shift_labels != self.ignore_index).float()
            clamped_labels = shift_labels.clone()
            clamped_labels[clamped_labels == self.ignore_index] = 0

            unlearned_token_log_probs = torch.gather(
                unlearned_log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)
            ).squeeze(-1) * mask

            base_token_log_probs = torch.gather(
                base_log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)
            ).squeeze(-1) * mask

            seq_lens = mask.sum(dim=-1).clamp(min=1.0)
            unlearned_seq_log_prob = unlearned_token_log_probs.sum(dim=-1) / seq_lens
            base_seq_log_prob = base_token_log_probs.sum(dim=-1) / seq_lens

            # NPO loss: - 2/beta * log sigmoid(-beta * (log_pi_theta - log_pi_ref))
            diff = unlearned_seq_log_prob - base_seq_log_prob
            npo_loss = - (2.0 / self.npo_beta) * F.logsigmoid(-self.npo_beta * diff).mean()
            return npo_loss

        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

    def compute_retain_kl_loss(
        self,
        unlearned_logits: torch.Tensor,
        base_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes forward KL-divergence D_KL(P_base || P_unlearned) on benign retain samples.
        Anchors the unlearned model distributions to the original base model.

        Args:
            unlearned_logits: [batch_size, seq_len, vocab_size] from unlearned model.
            base_logits: [batch_size, seq_len, vocab_size] from frozen base model (no grad).
            attention_mask: Optional [batch_size, seq_len] attention mask.

        Returns:
            kl_loss: Scalar torch tensor.
        """
        # Softmax over vocabulary with temperature scaling
        T = self.temperature
        p_base = F.softmax(base_logits.detach() / T, dim=-1)
        log_p_unlearned = F.log_softmax(unlearned_logits / T, dim=-1)

        # Token-wise KL divergence: sum(p_base * (log(p_base) - log(p_unlearned)))
        # PyTorch kl_div(input, target) computes target * (log(target) - input) when log_target=False
        kl_per_token = F.kl_div(log_p_unlearned, p_base, reduction="none", log_target=False).sum(dim=-1)

        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.float()
            kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_loss = kl_per_token.mean()

        # Scale by T^2 for gradient consistency
        return kl_loss * (T ** 2)

    def forward(
        self,
        forget_unlearned_logits: torch.Tensor,
        forget_labels: torch.Tensor,
        retain_unlearned_logits: torch.Tensor,
        retain_base_logits: torch.Tensor,
        retain_attention_mask: Optional[torch.Tensor] = None,
        forget_refusal_labels: Optional[torch.Tensor] = None,
        forget_base_logits: Optional[torch.Tensor] = None,
        forget_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes combined dual-objective loss:
            L_total = L_forget + lambda * L_retain_kl
        """
        forget_loss = self.compute_forget_loss(
            unlearned_logits=forget_unlearned_logits,
            labels=forget_labels,
            refusal_labels=forget_refusal_labels,
            base_logits=forget_base_logits,
            attention_mask=forget_attention_mask,
        )

        retain_kl_loss = self.compute_retain_kl_loss(
            unlearned_logits=retain_unlearned_logits,
            base_logits=retain_base_logits,
            attention_mask=retain_attention_mask,
        )

        total_loss = forget_loss + (self.kl_weight * retain_kl_loss)

        metrics = {
            "total_loss": total_loss.item(),
            "forget_loss": forget_loss.item(),
            "retain_kl_loss": retain_kl_loss.item(),
            "kl_weight": self.kl_weight,
        }

        return total_loss, metrics
