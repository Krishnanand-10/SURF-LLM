# Proposal

## Why

Large Language Models (LLMs) inadvertently memorize sensitive personal identifiable information (PII), proprietary copyrighted material, and hazardous knowledge during pre-training. Retraining models from scratch to satisfy regulatory mandates like GDPR's "Right to be Forgotten" is computationally prohibitive, while conventional unlearning techniques like global gradient ascent inflict severe collateral damage, causing catastrophic forgetting on general reasoning tasks.

SURF-LLM (Sparse Unlearning with Reasoning-Preserving Framework) addresses this by introducing a layer-wise sparse unlearning methodology using Low-Rank Adapters (LoRA). By localizing parameter updates strictly to the specific transformer layers that encode sensitive information and applying a dual-objective loss formulation that anchors benign knowledge distributions, SURF-LLM achieves surgical concept erasure while safeguarding downstream reasoning integrity.

## What Changes

- Introduce knowledge localization profiling using gradient saliency and causal attribution heatmaps to identify and rank sensitive transformer layers.
- Implement sparse parameter allocation via PEFT Low-Rank Adapters (LoRA) targeted exclusively to the top-K sensitive layers while completely freezing unaffected base layers.
- Formulate a dual-objective unlearning optimization loop combining sensitive concept suppression (forget loss) with probability distribution preservation (KL-divergence retain loss) on benign anchor datasets.
- Implement comprehensive safety and utility auditing benchmarks (TOFU, MIA, MMLU, GSM8K) alongside lossless adapter weight fusion into base weights for zero-inference-overhead deployment.
- Provide end-to-end orchestration scripts and experiment configuration profiles supporting standard open-weight LLM architectures (e.g., LLaMA-3, Mistral, Qwen-2.5).

## Capabilities

### New Capabilities
- `localization`: Attribution scoring and sensitivity profiling to pinpoint transformer layers and sub-modules encoding sensitive target data.
- `sparse-lora`: Selective low-rank adapter injection into top-K sensitive layers with complete freezing of non-targeted model weights.
- `dual-objective-trainer`: Composite optimization loop balancing sensitive token suppression (concept negation/refusal) with KL-divergence preservation on retain corpora.
- `auditing-fusion`: Multi-metric evaluation suite (MIA resistance, TOFU benchmark, MMLU, GSM8K) and adapter weight fusion into base model weights.

### Modified Capabilities

None.

## Impact

- **Core Codebase**: Introduces modules under `src/surf/` (`localization/`, `adapters/`, `trainer/`, `evaluation/`, `fusion/`).
- **Dependencies**: Requires `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, and evaluation harnesses.
- **Workflow & CLI**: Provides automated execution scripts in `scripts/` (`run_localization.py`, `run_unlearn.py`, `run_eval.py`, `fuse_weights.py`) and configuration files under `configs/`.
