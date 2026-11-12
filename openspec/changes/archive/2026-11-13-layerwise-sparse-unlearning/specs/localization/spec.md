# Spec Delta

## Purpose

Identifies and ranks transformer layers and sub-modules encoding sensitive concepts using gradient saliency and causal attribution methods.

## ADDED Requirements

### Requirement: Layer Sensitivity Profiling
The system SHALL compute attribution scores across all transformer layers using gradient saliency and causal tracing over the target forget dataset.

#### Scenario: Gradient saliency profiling execution
- **WHEN** user executes layer attribution profiling with a forget dataset and gradient_norm method
- **THEN** system computes layer-level saliency norms across all attention and MLP modules and returns an attribution dictionary mapping layer indices to sensitivity scores

#### Scenario: Module-specific gradient filtering
- **WHEN** attribution profiling specifies target module substrings such as mlp.down_proj
- **THEN** system filters parameter gradient computations to matched modules when calculating layer saliency scores

### Requirement: Top-K Sensitive Layer Selection
The system SHALL identify and rank the top-K transformer layers exhibiting the highest attribution scores for selective unlearning.

#### Scenario: Top-K layer ranking
- **WHEN** attribution scores are computed and top_k parameter K is specified
- **THEN** system sorts layers in descending order of attribution score and outputs the top-K layer index list
