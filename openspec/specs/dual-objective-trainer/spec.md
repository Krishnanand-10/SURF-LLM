# dual-objective-trainer Specification

## Purpose
Executes unlearning optimization via a composite objective combining sensitive concept suppression with divergence regularization on benign retain data.

## Requirements

### Requirement: Sensitive Concept Suppression
The system SHALL compute a forget loss that minimizes sensitive token likelihood or aligns forget prompts to neutral refusal representations.

#### Scenario: Gradient ascent forget loss
- **WHEN** unlearning trainer runs with forget loss type set to grad_ascent on forget batches
- **THEN** system computes negative cross-entropy loss over sensitive target tokens to penalize their emission probabilities

#### Scenario: Refusal alignment forget loss
- **WHEN** unlearning trainer runs with forget loss type set to refusal
- **THEN** system optimizes cross-entropy loss aligning forget prompts to neutral refusal response targets

### Requirement: Benign Knowledge Preservation
The system SHALL compute a retain loss using forward KL-divergence between the base model and unlearned model output distributions on benign retain data.

#### Scenario: KL divergence regularization
- **WHEN** unlearning trainer processes benign retain samples
- **THEN** system evaluates base model logits and unlearned model logits, calculating forward KL-divergence over next-token prediction distributions

### Requirement: Composite Optimization Loop
The system SHALL optimize the adapter parameters using total loss L_total = L_forget + lambda * D_KL over scheduled training epochs.

#### Scenario: Dual-objective backpropagation
- **WHEN** total loss is evaluated with hyperparameter lambda kl_weight
- **THEN** system backpropagates gradients through trainable adapter parameters only and updates LoRA weights according to the composite loss
