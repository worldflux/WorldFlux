# Unified Training Comparison

This smoke demo shows that DreamerV3 and TD-MPC2 can run through the same
high-level training flow.

It uses the same random `ReplayBuffer` source for both families, so treat it as
a contract demonstration rather than a benchmark or a real-environment
performance comparison.

## Run

```bash
uv sync --extra dev
uv run python examples/compare_unified_training.py --quick
```

## What It Demonstrates

- same random ReplayBuffer source for both families
- same `TrainingConfig` contract
- same artifact generation flow
- same visualization helper (`write_reward_heatmap_ppm`)

## Artifacts

- `outputs/comparison/summary.json`
- `outputs/comparison/dreamer.ppm`
- `outputs/comparison/tdmpc2.ppm`
