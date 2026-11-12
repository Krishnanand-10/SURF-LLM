# Spec Delta

## Purpose

Audits unlearning efficacy and reasoning preservation across benchmark suites, and fuses adapter weights into base model parameters for zero-overhead deployment.

## ADDED Requirements

### Requirement: Unlearning Efficacy Auditing
The system SHALL benchmark unlearning performance using sensitive token generation rate drops and Member Inference Attack (MIA) resistance scores on TOFU or PII datasets.

#### Scenario: Sensitive recall evaluation
- **WHEN** evaluation suite runs against the forget evaluation set
- **THEN** system measures sensitive token generation accuracy and verifies unlearning drop exceeds target threshold compared to baseline

#### Scenario: Member inference attack scoring
- **WHEN** evaluation suite runs loss-based MIA probe across forget and unseen holdout samples
- **THEN** system computes ROC-AUC or MIA score and outputs the privacy resistance rating

### Requirement: Reasoning Integrity Validation
The system SHALL measure reasoning benchmark performance and retain perplexity divergence against base model baselines on MMLU, GSM8K, and Wikitext corpora.

#### Scenario: Retain perplexity check
- **WHEN** evaluation runs on benign validation corpus Wikitext or C4
- **THEN** system computes perplexity difference and verifies divergence remains under 5% relative to the base model

#### Scenario: Reasoning benchmark evaluation
- **WHEN** evaluation harness runs MMLU and GSM8K evaluation
- **THEN** system verifies reasoning accuracy retention exceeds 98% of base model baseline score

### Requirement: Base Weight Delta Fusion
The system SHALL fuse trained LoRA adapter parameter deltas directly into the frozen base model weights and export the final unlearned model checkpoint.

#### Scenario: Lossless weight merging
- **WHEN** weight fusion is executed with the trained PEFT model
- **THEN** system computes delta W = B * A, adds it to base weights W_0, removes adapter hooks, and serializes the merged model and tokenizer to disk
