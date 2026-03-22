# Model Evaluation

Guide for evaluating WorldFlux model quality using the deterministic eval framework.

## Overview

The eval framework now has two explicit modes:

- `synthetic`: fast compatibility metrics for development and CI
- `dataset_replay`: replay-backed proxy evaluation over recorded trajectories
- `env_policy`: learned-policy rollout in a real environment

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
worldflux eval ./outputs --suite quick --mode dataset_replay --dataset-manifest ./data/halfcheetah.dataset_manifest.json --format json
worldflux eval ./outputs --suite quick --mode env_policy --env-id ALE/Breakout-v5 --format json
```

## Quick Verification Tiers

For checkpoint-oriented verification flows, `worldflux.verify.quick.quick_verify`
now distinguishes lightweight execution tiers:

The only effective tier is `synthetic`. Legacy tier aliases still normalize to
synthetic semantics for one compatibility window, but they are no longer part
of the promoted surface.

Example:

```python
from worldflux.verify.quick import quick_verify

result = quick_verify("./outputs", env="atari/pong", tier="synthetic")
print(result.stats["verification_tier_effective"])
```

## Explicit Evaluation Inputs

`worldflux eval` uses distinct inputs per mode:

- `--mode dataset_replay --dataset-manifest <path>`: replay-buffer backed proxy/model-quality input
- `--mode env_policy --env-id <gymnasium-id>`: learned-policy env rollout input

`dataset_replay` JSON outputs include `dataset_replay_provenance` and the
temporary compatibility alias `real_provenance`. `env_policy` JSON outputs
include `env_policy_provenance` and the same compatibility alias.
Synthetic-mode outputs include `synthetic_provenance`.

## Metrics Reference

- **reconstruction_fidelity**: Measures encode → decode round-trip MSE
- **latent_consistency**: Verifies deterministic encoding (same input → same latent)
- **imagination_coherence**: Checks rollout finiteness and bounded outputs
- **reward_prediction_accuracy**: Predicted vs actual reward MSE
- **latent_utilization**: Effective dimensionality of latent space

Current suites remain proxy-oriented unless an env-policy return path is added
by the caller. If control metrics are absent, treat the report as proxy-only.

## Integration with Training

Use `EvalCallback` to run lightweight evals during training:

```python
from worldflux.training.callbacks import EvalCallback

trainer.add_callback(EvalCallback(eval_interval=5000))
```
