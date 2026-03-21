# Model Evaluation

Guide for evaluating WorldFlux model quality using the deterministic eval framework.

## Overview

The eval framework now has two explicit modes:

- `synthetic`: fast compatibility metrics for development and CI
- `real`: dataset-backed or env-backed evaluation for evidence collection

## Eval Suites

| Suite | Metrics | Runtime |
|-------|---------|---------|
| `quick` | reconstruction_fidelity, latent_consistency | &lt;5s |
| `standard` | + imagination_coherence, latent_utilization | ~30s |
| `comprehensive` | + reward_prediction_accuracy, cross-model comparison | ~5min |

## Usage

```python
from worldflux.evals import run_eval_suite

report = run_eval_suite(model, suite="quick")
print(report.all_passed)
```

CLI examples:

```bash
worldflux eval ./outputs --suite quick --mode synthetic --format json
worldflux eval ./outputs --suite quick --mode real --dataset-manifest ./data/halfcheetah.dataset_manifest.json --format json
```

## Quick Verification Tiers

For checkpoint-oriented verification flows, `worldflux.verify.quick.quick_verify`
now distinguishes lightweight execution tiers:

| Tier | Purpose |
| --- | --- |
| `synthetic` | default compatibility path using synthetic rollouts |
| `offline` | checkpoint verification against bundled baseline statistics without parity tooling |
| `real_env_smoke` | smoke-oriented path reserved for short real-environment verification loops |

Example:

```python
from worldflux.verify.quick import quick_verify

result = quick_verify("./outputs", env="atari/pong", tier="offline")
print(result.stats["verification_tier"])
```

## Real Evaluation Inputs

`worldflux eval --mode real` requires one of:

- `--dataset-manifest <path>`: replay-buffer backed evidence input
- `--env-id <gymnasium-id>`: short env-backed real evaluation loop

Real-mode JSON outputs include `real_provenance`. Synthetic-mode outputs include
`synthetic_provenance`.

## Metrics Reference

- **reconstruction_fidelity**: Measures encode → decode round-trip MSE
- **latent_consistency**: Verifies deterministic encoding (same input → same latent)
- **imagination_coherence**: Checks rollout finiteness and bounded outputs
- **reward_prediction_accuracy**: Predicted vs actual reward MSE
- **latent_utilization**: Effective dimensionality of latent space

## Integration with Training

Use `EvalCallback` to run lightweight evals during training:

```python
from worldflux.training.callbacks import EvalCallback

trainer.add_callback(EvalCallback(eval_interval=5000))
```
