# Design

## Context

See `proposal.md` for background and motivation. SURF-LLM operates on modern causal decoder transformer architectures (e.g., LLaMA-3, Mistral, Qwen-2.5) using PyTorch 2.x, Hugging Face `transformers`, `peft`, and `accelerate`. The target deployment environment requires execution on single-GPU hardware with 16 GB+ VRAM (RTX 3090/4090 or Google Colab/Kaggle A100/T4), making parameter efficiency and low VRAM footprint critical architectural requirements.

## Goals / Non-Goals

**Goals:**
- Identify sensitive transformer layers and sub-modules using gradient saliency and activation attribution on forget datasets.
- Restrict trainable LoRA adapters strictly to top-K sensitive layers via PEFT `layers_to_transform`, keeping trainable parameters < 1% of total model size.
- Formulate and execute dual-objective unlearning combining concept suppression ($L_{\text{forget}}$) and probability anchor regularization ($\mathcal{D}_{\text{KL}}(P_{\text{base}} \parallel P_{\text{unlearned}})$).
- Provide evaluation suites measuring privacy resistance (MIA, TOFU) and reasoning preservation (MMLU, GSM8K, retain perplexity).
- Enable lossless adapter weight fusion ($\Delta W = B \cdot A$) back into base weights for zero-overhead inference deployment.

**Non-Goals:**
- Full parameter gradient ascent or full-model retraining from scratch.
- Distributed multi-node pre-training infrastructure.
- Continuous online unlearning streaming pipelines (focus is on batch-oriented selective deletion requests).

## Decisions

### 1. Sparse Layer-wise LoRA vs. Global LoRA or Full Fine-Tuning
- **Choice**: Selectively inject LoRA adapters into top-K sensitive layers identified by attribution scoring, leaving all other layers frozen.
- **Rationale**: Studies in mechanistic interpretability demonstrate factual associations and sensitive knowledge concentrate in specific mid-to-late transformer MLP/FFN blocks. Global adapter fine-tuning or full-model gradient ascent degrades unrelated parameters across all layers, causing catastrophic forgetting.
- **Alternatives Considered**: Global LoRA across all layers (insufficient isolation; still perturbs benign representations); Full parameter gradient ascent (causes catastrophic model collapse).

### 2. Native PEFT `layers_to_transform` vs. Custom PyTorch Monkey-Patching
- **Choice**: Utilize Hugging Face PEFT's native `LoraConfig(layers_to_transform=sensitive_layers, target_modules=...)`.
- **Rationale**: Built-in support ensures standard compatibility with Hugging Face `PreTrainedModel`, PyTorch FSDP/DeepSpeed, gradient checkpointing, and established adapter serialization/loading methods.
- **Alternatives Considered**: Custom adapter wrappers injecting linear layers manually (high maintenance, fragile integration with HuggingFace pipelines).

### 3. Dual-Objective Loss with KL Divergence Anchor
- **Choice**: Optimize composite loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forget}} + \lambda \cdot \mathcal{D}_{\text{KL}}(P_{\text{base}} \parallel P_{\text{unlearned}})$, supporting gradient ascent, refusal alignment, and negative preference optimization (NPO) for $\mathcal{L}_{\text{forget}}$.
- **Rationale**: Unlearning objectives without probability distribution regularization cause representation collapse and language incoherence. Forward KL divergence on a benign anchor corpus (Wikitext-103 or C4) penalizes probability drift against frozen base model logits.
- **Alternatives Considered**: Retain cross-entropy loss without KL divergence (less effective at matching token distribution shapes); Orthogonal subspace projection (computationally prohibitive for 7B+ models).

### 4. Lossless Post-Training Weight Fusion
- **Choice**: Merge trained adapter weights directly into base model weights via PEFT's `merge_and_unload()` export routine.
- **Rationale**: Eliminates adapter routing latency during inference, producing a standalone Hugging Face checkpoint ready for deployment in production runtimes (e.g., vLLM, TensorRT-LLM).
- **Alternatives Considered**: Retaining separate adapter weights at inference (introduces inference latency overhead and multi-file serving complexity).

## Risks / Trade-offs

- **[Risk: Gradient Explosion during Gradient Ascent]** → Mitigation: Enforce gradient clipping (`max_grad_norm: 1.0`), low learning rates ($2 \times 10^{-4}$), and provide refusal alignment or NPO as stable alternative forget loss formulations.
- **[Risk: Memory Pressure with Base & Unlearned Models in Memory]** → Mitigation: Keep the base model in half-precision (`torch.float16` or `bfloat16`) with `torch.no_grad()` during retain logit extraction, or cache retain logits offline before training.
- **[Risk: Non-Contiguous Sensitive Layers]** → Mitigation: Rank layers globally by attribution score rather than assuming contiguous layer blocks, allowing adapters to attach to arbitrary high-density layers.

## Migration Plan

1. Provide automated CLI scripts (`scripts/run_localization.py`, `scripts/run_unlearn.py`, `scripts/run_eval.py`, `scripts/fuse_weights.py`).
2. Retain YAML configuration profiles (`configs/default.yaml`) for reproducible hyperparameter specifications.
3. Save fused outputs as standard Hugging Face model directories, enabling drop-in replacement of existing checkpoints without client-side code changes.
