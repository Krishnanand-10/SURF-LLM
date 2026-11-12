"""
SURFTrainer: Dual-Objective Unlearning Optimization Engine.
Coordinates selective gradient suppression on sensitive forget data
while enforcing forward KL divergence distributional preservation on benign retain corpora.
"""

from itertools import cycle
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .objectives import DualObjectiveLoss


class SURFTrainer:
    """
    Dual-Objective Unlearning Trainer for SURF-LLM.
    
    Optimizes LoRA adapters strictly targeted at sensitive layers using the composite objective:
        L_total = L_forget + lambda * D_KL(P_base || P_unlearned)
    """

    def __init__(
        self,
        model: nn.Module,
        base_model: Optional[nn.Module] = None,
        forget_dataloader: Optional[DataLoader] = None,
        retain_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        learning_rate: float = 2.0e-4,
        kl_weight: float = 1.0,
        forget_loss_type: str = "grad_ascent",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
        npo_beta: float = 0.1,
        device: Union[str, torch.device] = "cpu",
        output_dir: str = "./outputs/surf_experiment",
        loss_fn: Optional[DualObjectiveLoss] = None,
    ):
        self.device = torch.device(device if isinstance(device, str) else device)
        self.model = model.to(self.device)
        self.base_model = base_model.to(self.device) if base_model is not None else None

        self.forget_dataloader = forget_dataloader
        self.retain_dataloader = retain_dataloader

        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.forget_loss_type = forget_loss_type
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir

        # Initialize Dual-Objective Loss function
        self.loss_fn = loss_fn or DualObjectiveLoss(
            kl_weight=kl_weight,
            forget_loss_type=forget_loss_type,
            temperature=temperature,
            npo_beta=npo_beta,
        )

        # Set up optimizer targeting only trainable (LoRA) parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            print("[SURF-LLM] WARNING: No trainable parameters found in model!")
            trainable_params = list(self.model.parameters())

        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.lr_scheduler = lr_scheduler

        # Training history logs
        self.history: List[Dict[str, Any]] = []

    def _get_base_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes forward pass logits from the frozen base model.
        If an explicit base_model is provided, uses it.
        Otherwise, if self.model supports disable_adapter() (PEFT),
        evaluates logits in adapter-disabled mode to save GPU VRAM.
        """
        with torch.no_grad():
            if self.base_model is not None:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # PEFT zero-memory-overhead base inference
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    return outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Fallback if neither is available: detach unlearned model output
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            return logits.detach()

    def train_step(
        self,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Performs a single unlearning optimization step across forget and retain batches.
        """
        # Move forget batch tensors to device
        f_input_ids = forget_batch["input_ids"].to(self.device)
        f_attention_mask = forget_batch.get("attention_mask")
        if f_attention_mask is not None:
            f_attention_mask = f_attention_mask.to(self.device)
        f_labels = forget_batch.get("labels", f_input_ids).to(self.device)
        f_refusal = forget_batch.get("refusal_labels")
        if f_refusal is not None:
            f_refusal = f_refusal.to(self.device)

        # Move retain batch tensors to device
        r_input_ids = retain_batch["input_ids"].to(self.device)
        r_attention_mask = retain_batch.get("attention_mask")
        if r_attention_mask is not None:
            r_attention_mask = r_attention_mask.to(self.device)

        # 1. Forward pass on Forget data using unlearned model
        f_unlearned_out = self.model(input_ids=f_input_ids, attention_mask=f_attention_mask)
        f_unlearned_logits = f_unlearned_out.logits if hasattr(f_unlearned_out, "logits") else f_unlearned_out[0]

        # 2. Base logits on Forget data (needed if forget_loss_type == 'npo')
        f_base_logits = None
        if self.forget_loss_type == "npo":
            f_base_logits = self._get_base_logits(f_input_ids, f_attention_mask)

        # 3. Forward pass on Retain data using unlearned model
        r_unlearned_out = self.model(input_ids=r_input_ids, attention_mask=r_attention_mask)
        r_unlearned_logits = r_unlearned_out.logits if hasattr(r_unlearned_out, "logits") else r_unlearned_out[0]

        # 4. Base logits on Retain data (frozen reference for KL Divergence)
        r_base_logits = self._get_base_logits(r_input_ids, r_attention_mask)

        # 5. Composite Dual-Objective Loss
        total_loss, metrics = self.loss_fn(
            forget_unlearned_logits=f_unlearned_logits,
            forget_labels=f_labels,
            retain_unlearned_logits=r_unlearned_logits,
            retain_base_logits=r_base_logits,
            retain_attention_mask=r_attention_mask,
            forget_refusal_labels=f_refusal,
            forget_base_logits=f_base_logits,
            forget_attention_mask=f_attention_mask,
        )

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        return total_loss, metrics

    def train(
        self,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        log_interval: int = 10,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the dual-objective training loop.
        """
        if self.forget_dataloader is None or self.retain_dataloader is None:
            raise ValueError("Both forget_dataloader and retain_dataloader must be provided for unlearning.")

        self.model.train()
        if self.base_model is not None:
            self.base_model.eval()

        global_step = 0
        running_loss = 0.0
        running_forget_loss = 0.0
        running_retain_kl = 0.0

        total_batches = len(self.forget_dataloader)
        print(f"[SURF-LLM] Commencing unlearning: {num_epochs} epochs | Batches/epoch: {total_batches}")
        print(f"[SURF-LLM] Mode: {self.forget_loss_type.upper()} | KL Anchor Weight lambda: {self.kl_weight}")

        self.optimizer.zero_grad()

        for epoch in range(num_epochs):
            retain_iter = cycle(self.retain_dataloader)
            pbar = tqdm(
                self.forget_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                total=total_batches,
            )

            for batch_idx, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter)

                total_loss, metrics = self.train_step(forget_batch, retain_batch)

                running_loss += metrics["total_loss"]
                running_forget_loss += metrics["forget_loss"]
                running_retain_kl += metrics["retain_kl_loss"]

                # Optimization step at accumulation boundaries
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % log_interval == 0:
                        avg_total = running_loss / log_interval
                        avg_forget = running_forget_loss / log_interval
                        avg_kl = running_retain_kl / log_interval

                        current_lr = (
                            self.optimizer.param_groups[0]["lr"]
                            if self.optimizer.param_groups
                            else self.learning_rate
                        )

                        log_entry = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "total_loss": avg_total,
                            "forget_loss": avg_forget,
                            "retain_kl_loss": avg_kl,
                            "learning_rate": current_lr,
                        }
                        self.history.append(log_entry)

                        pbar.set_postfix({
                            "L_tot": f"{avg_total:.4f}",
                            "L_fgt": f"{avg_forget:.4f}",
                            "D_KL": f"{avg_kl:.4f}",
                        })

                        running_loss = 0.0
                        running_forget_loss = 0.0
                        running_retain_kl = 0.0

                    if max_steps is not None and global_step >= max_steps:
                        print(f"[SURF-LLM] Reached maximum steps: {max_steps}")
                        break

            if max_steps is not None and global_step >= max_steps:
                break

        print(f"[SURF-LLM] Unlearning optimization finished across {global_step} steps.")

        if save_checkpoint and self.output_dir:
            self.save_model(self.output_dir)

        return {
            "global_step": global_step,
            "final_epoch": num_epochs,
            "history": self.history,
        }

    def evaluate_retain_perplexity(self, dataloader: Optional[DataLoader] = None) -> float:
        """
        Evaluates perplexity on benign retain data to audit reasoning preservation.
        """
        dl = dataloader or self.retain_dataloader
        if dl is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                num_tokens = (shift_labels != -100).sum().item()

                total_loss += loss.item()
                total_tokens += num_tokens

        self.model.train()
        avg_loss = total_loss / max(1, total_tokens)
        try:
            return float(torch.exp(torch.tensor(avg_loss)).item())
        except OverflowError:
            return float("inf")

    def save_model(self, output_dir: Optional[str] = None) -> str:
        """
        Saves the trained LoRA adapter weights and training metadata.
        """
        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        print(f"[SURF-LLM] Saving unlearned adapter checkpoint to: {target_dir}")

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(target_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(target_dir, "adapter_weights.pt"))

        meta_path = os.path.join(target_dir, "unlearning_metadata.json")
        metadata = {
            "kl_weight": self.kl_weight,
            "forget_loss_type": self.forget_loss_type,
            "learning_rate": self.learning_rate,
            "history": self.history,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return target_dir


# Alias for compatibility with project requirements
UnlearningTrainer = SURFTrainer
