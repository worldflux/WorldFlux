# Reproduce DreamerV3 and TD-MPC2 Locally

This guide explains the currently supported local reproducibility path for the
two reference families in WorldFlux.

It stays inside the evidence-backed onboarding contract:

- CPU smoke first
- local scaffold training second
- quick compatibility verification third

It does not claim proof-grade paper reproduction.

## 1. Prove the local environment first

Run the official CPU smoke before comparing model families:

```bash
uv sync --extra dev
uv run python examples/quickstart_cpu_success.py --quick
```

This confirms that your local install can create, train, and roll out a small
Dreamer CI model on CPU with machine-checkable success criteria.

## 2. Compare the shared contract smoke

Run the repository comparison script:

```bash
uv sync --extra dev --extra training
uv run python examples/compare_unified_training.py --quick
```

This is the supported repository-level reproduction step for comparing
DreamerV3 and TD-MPC2 under the same synthetic smoke conditions and the same
quick verification flow.

The emitted quick verify results separate workflow completion from statistical
quality. A warning-level quick verify result still means the comparison demo
produced interpretable artifacts and completed the contract-smoke lane.

## 3. Reproduce the newcomer project flow

For a scaffolded project, generate a local project and run the same commands a
new user would run:

```bash
worldflux init my-world-model
cd my-world-model
worldflux train --steps 5 --device cpu
worldflux verify --target ./outputs --mode quick
```

Repeat that flow once with a Dreamer-oriented setup and once with a TD-MPC2
oriented setup if you want to compare both families under your own local
conditions.

For the guaranteed real-environment newcomer lane in this MVP, prioritize the
Dreamer scaffold and install Atari extras first:

```bash
uv sync --extra training --extra atari
```

## 4. Interpret the result correctly

This workflow gives you:

- a reproducible local smoke path
- generated outputs and run manifests
- a structured quick-verify result

This workflow does not give you:

- upstream non-inferiority evidence
- publishable proof artifacts
- benchmark claims

## 5. When to use parity workflows

Move to `worldflux parity` only after the local path above is stable:

```bash
worldflux parity --help
worldflux parity proof-run --help
worldflux parity proof-report --help
```

Proof-oriented parity remains an advanced workflow. Public proof claims require
published evidence bundles as described in [Parity Harness](../reference/parity.md).
