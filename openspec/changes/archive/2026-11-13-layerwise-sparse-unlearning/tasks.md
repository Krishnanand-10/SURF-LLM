# Tasks

## 1. Localization Module Implementation

- [x] 1.1 Complete `LayerSaliencyProfiler` in `src/surf/localization/saliency.py` and causal attribution in `src/surf/localization/causal_trace.py`, verifying attribution scoring outputs valid layer sensitivity dictionaries on sample batches.
- [x] 1.2 Implement the localization CLI script in `scripts/run_localization.py` and unit tests in `tests/test_localization.py`, verifying profiling runs end-to-end and outputs top-K sensitive layer indices.

## 2. Sparse LoRA Injection Module

- [x] 2.1 Finalize `SparseLoRAInjector` in `src/surf/adapters/sparse_lora.py` with `layers_to_transform` configuration and strict base-weight freezing, verifying adapter parameter counts confirm parameter update footprint < 1%.
- [x] 2.2 Add unit tests in `tests/test_adapters.py`, verifying that only target layers receive adapter weights and unaffected base layers remain completely frozen with `requires_grad=False`.

## 3. Dual-Objective Unlearning Engine

- [x] 3.1 Implement composite loss functions in `src/surf/trainer/objectives.py` supporting forget loss modes (`grad_ascent`, `refusal`, `npo`) and forward KL divergence retain loss, verifying loss outputs on test logit tensors.
- [x] 3.2 Implement `SURFTrainer` in `src/surf/trainer/trainer.py` and the training script in `scripts/run_unlearn.py`, verifying unlearning training steps execute properly across forget and retain batches.
- [x] 3.3 Add trainer unit tests in `tests/test_trainer.py`, verifying that backpropagation updates only LoRA adapter parameters according to the composite loss.

## 4. Auditing, Benchmarking, and Weight Fusion

- [x] 4.1 Implement evaluation suites in `src/surf/evaluation/forget_metrics.py` (sensitive token generation, MIA resistance) and `src/surf/evaluation/reasoning_metrics.py` (perplexity divergence, MMLU/GSM8K harnesses), verifying metric calculation on sample predictions.
- [x] 4.2 Implement base weight fusion in `src/surf/fusion/fusion.py` and `scripts/fuse_weights.py`, verifying that merging adapter deltas into base weights produces a standalone Hugging Face checkpoint with zero inference overhead.
- [x] 4.3 Implement `scripts/run_eval.py` and pipeline integration tests in `tests/test_pipeline.py`, verifying the full pipeline from localization to unlearning, auditing, and weight fusion.

## Workflow follow-up

- Validate all OpenSpec specs and change status.
- Archive the change after implementation and test verification pass.
