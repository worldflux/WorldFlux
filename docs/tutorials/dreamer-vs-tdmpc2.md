# DreamerV3 vs TD-MPC2

This tutorial compares the two supported reference-family onboarding choices in
WorldFlux without stepping outside the current MVP boundary.

Use this guide when you already know the newcomer path:

1. `worldflux init`
2. `worldflux train`
3. `worldflux verify --target ./outputs --mode quick`

If you have not completed that path yet, start with
[Train Your First Model](train-first-model.md).

## 1. Inspect both model families

Check the supported scaffold presets first:

```bash
worldflux models info dreamer:ci
worldflux models info tdmpc2:ci
```

What to look for:

- `dreamer:ci` is the safer first choice for pixel or spatial observations
- `tdmpc2:ci` is the lighter first choice for compact vector observations

The same rule is reflected in the `worldflux init` chooser and in
[Quick Start](../getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init).

## 2. Run the shared smoke comparison

WorldFlux ships one repository-level comparison smoke:

```bash
uv sync --extra dev --extra training
uv run python examples/compare_unified_training.py --quick
```

This command is the fastest machine-checkable way to compare both families on a
shared contract-level training path with the same quick verification flow.

What this proves:

- both model families can be created from the same unified API
- both families can run a short smoke training/evaluation loop
- both families can emit the same quick verification artifact shape
- the comparison stays local and synthetic; it is not a benchmark or a proof claim

## 3. Choose the right scaffold path

Pick one family based on the environment you actually have:

- image-like observations such as `3,64,64`: choose `dreamer:ci`
- vector observations such as `39`: choose `tdmpc2:ci`

Generate a project and run the supported local loop:

```bash
worldflux init my-world-model
cd my-world-model
worldflux train --steps 5 --device cpu
worldflux verify --target ./outputs --mode quick
```

`worldflux verify --mode quick` is the supported first verification path for
both families. A short run may still fail the non-inferiority verdict; for MVP
onboarding the important result is that the command executes and emits a
structured report.

## 4. How to decide

Use DreamerV3 when:

- your observations are image-heavy
- you want the most direct path from the docs/examples for pixel-based world models

Use TD-MPC2 when:

- your observations are low-dimensional vectors
- you want a lighter baseline for local control experiments

## 5. What this tutorial does not claim

This tutorial does not establish:

- benchmark superiority
- paper reproduction
- proof-grade parity against upstream implementations

For proof-oriented work, read [Parity Harness](../reference/parity.md) after the
local newcomer path is already working.
