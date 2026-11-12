# sparse-lora Specification

## Purpose
Attaches trainable Low-Rank Adapters (LoRA) strictly to identified high-attribution transformer layers while freezing all unaffected model parameters.

## Requirements

### Requirement: Selective Adapter Injection
The system SHALL inject trainable Low-Rank Adapters (LoRA) exclusively into the designated sensitive layer indices and target module types.

#### Scenario: Restricted layer transformation
- **WHEN** LoRA injection is invoked with a base model, rank r, target modules, and a list of sensitive layer indices
- **THEN** system configures PEFT layers_to_transform exclusively to the sensitive layers and creates adapter parameters only for those layers

#### Scenario: Sparsity parameter ratio calculation
- **WHEN** adapters are injected into the base model
- **THEN** system calculates and logs the ratio of trainable parameters to total parameters confirming sparsity is under 1% of the total model size

### Requirement: Unaffected Parameter Freezing
The system SHALL ensure that all base model weights and non-targeted transformer layers remain frozen with gradient computation disabled.

#### Scenario: Non-targeted layer verification
- **WHEN** adapter injection completes
- **THEN** system validates that requires_grad is false for all base weights and parameters in non-targeted layers
