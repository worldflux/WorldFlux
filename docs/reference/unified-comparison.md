# Unified Training Comparison

This smoke demo shows that DreamerV3 and TD-MPC2 can run through the same
high-level training flow and the same quick verification flow.

It uses the same random `ReplayBuffer` source for both families, so treat it as
a contract demonstration rather than a benchmark or a real-environment
performance comparison.

## Run

```bash
uv sync --extra dev --extra training
uv run python examples/compare_unified_training.py --quick
```

## What It Demonstrates

- same random ReplayBuffer source for both families
- same `TrainingConfig` contract
- same quick verification flow for both families
- same artifact generation flow
- same visualization helper (`write_reward_heatmap_ppm`)

The default command also runs quick verification for each family and writes the
structured result alongside the training outputs. Use `--skip-verify` only when
you intentionally want a faster local smoke without verification artifacts.

## Artifacts

- `outputs/comparison/summary.json`
- `outputs/comparison/dreamer.ppm`
- `outputs/comparison/tdmpc2.ppm`
- `outputs/comparison/dreamerv3/quick_verify.json`
- `outputs/comparison/tdmpc2/quick_verify.json`

## Interpretation

This demo proves that both families can be created, trained, rolled out, and
checked through the same public workflow shape. It remains a smoke-only contract
demonstration and does not establish real-environment performance, benchmark
superiority, paper reproduction, or a public proof claim.
