# Hyperparameter Sensitivity Analysis

> **Status**: Template - to be populated with actual experimental results.

## Overview

This document reports the sensitivity of DreamerV3 default hyperparameters to
performance on a fast evaluation environment (CartPole-v1). The goal is to
validate that the production defaults sit in a "safe" region where small
perturbations do not cause large performance drops.

## Method

- **Analysis type**: One-at-a-time (OAT) sensitivity
- **Environment**: CartPole-v1 (fast iteration for CI-compatible testing)
- **Steps per run**: 100,000 environment steps
- **Seeds per configuration**: 3
- **Metric**: Mean final episode return (averaged across seeds)

### Parameters Swept

| Parameter | Grid | Default |
|-----------|------|---------|
| `kl_dynamics` | [0.1, 0.3, 0.5, 1.0, 2.0] | 0.5 |
| `kl_representation` | [0.01, 0.05, 0.1, 0.5, 1.0] | 0.1 |
| `free_nats` | [0.0, 0.5, 1.0, 2.0, 5.0] | 1.0 |
| `learning_rate` | [3e-5, 1e-4, 3e-4, 1e-3] | 1e-4 |
| `imagination_horizon` | [5, 10, 15, 20, 30] | 15 |

## Results

> Results will be auto-generated when the sensitivity sweep completes.
> Run: `python scripts/run_sensitivity.py --report-from <results.json> --output-md docs/reference/hyperparameter-sensitivity.md`

### Sensitivity Ranking

| Rank | Parameter | Sensitivity Score | Default | Default Safe |
| ---: | --- | ---: | ---: | --- |
| - | *Pending experiment* | - | - | - |

### Per-Parameter Details

*To be filled after experiments complete.*

## Interpretation Guide

- **Sensitivity Score**: Coefficient of variation of mean rewards across
  parameter values. Higher values indicate parameters that more strongly
  affect performance.
- **Default Safe**: "Yes" when the default value's mean reward is in the
  middle 50% of tested values (25th-75th percentile).
- A well-tuned default should be "safe" across all parameters, confirming
  robustness to minor hyperparameter perturbations.

## Reproducing

```bash
# Generate configs (dry-run)
python scripts/run_sensitivity.py --dry-run

# Run full sweep (requires GPU)
python scripts/run_sensitivity.py \
    --environment CartPole-v1 \
    --seeds 0,1,2 \
    --steps 100000 \
    --output reports/parity/sensitivity/dreamerv3_sensitivity.json

# Render report from results
python scripts/run_sensitivity.py \
    --report-from reports/parity/sensitivity/dreamerv3_sensitivity.json \
    --output-md docs/reference/hyperparameter-sensitivity.md
```
