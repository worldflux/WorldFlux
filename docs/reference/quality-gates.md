# OSS Quality Gates

This document defines the **release readiness gates** for WorldFlux.

## CI Gates (Required on PR)

All PRs must pass:

- **Lint**: `ruff check src/ tests/ examples/`
- **Format**: `ruff format --check src/ tests/ examples/`
- **Typecheck**: `mypy src/worldflux/`
- **Unit tests**: `pytest tests/`
- **Example smoke tests**:
  - `python examples/train_dreamer.py --test`
  - `python examples/train_tdmpc2.py --test`
  - `python examples/train_jepa.py --steps 5 --batch-size 4 --obs-dim 8`
  - `python examples/train_token_model.py --steps 5 --batch-size 4 --seq-len 8 --vocab-size 32`
  - `python examples/train_diffusion_model.py --steps 5 --batch-size 4 --obs-dim 4 --action-dim 2`
  - `python examples/plan_cem.py --horizon 3 --action-dim 2`
- **Docs build**: `mkdocs build`

## Reproducibility Gates (Nightly / Release)

- **Seed stability**: same seed produces key losses/metrics within ±5–10%.
- **Save/Load parity**: outputs after `save_pretrained` and reload differ only within tolerance.
- **Seed success rate**: at least **80%** of seeds pass family-specific success criteria.

## Benchmark Gates (Release Only)

- **Loss trend**: training loss decreases by a minimum threshold within N steps.
- **Numerical stability**: no NaN/Inf during training in defined step budget.

## Family-Specific Gates

- **Reference families** (DreamerV3, TD-MPC2):
  - loss drop threshold
  - finite loss components
  - seed success rate >= 80%
- **Token family**:
  - token cross-entropy/perplexity trend
  - finite logits and loss
- **Diffusion family**:
  - denoise error trend
  - finite scheduler states and losses
- **JEPA family**:
  - representation prediction loss trend
  - finite context/target projections

## Recommended Threshold Defaults

These can be tightened over time:

- Loss decrease: **≥10%** within **2,000 steps** on bundled datasets.
- Seed variance: **≤10%** on main loss components.

## Measurement Commands

The following scripts are committed for reproducible measurement runs.
Defaults use **bundled datasets** and **small CI-sized models**.

### Run Measurements (Balanced)

```bash
python scripts/measure_quality_gates.py \
  --models dreamer,tdmpc2 \
  --seeds 0,1,2,3,4 \
  --steps 5000 \
  --device auto
```

Output:

- `reports/quality-gates/seed_runs.json`

### Estimate Runtime from Measured Results

```bash
python scripts/estimate_quality_gate_runtime.py \
  --input reports/quality-gates/seed_runs.json \
  --scenarios 5x5000,7x10000,10x10000,7x20000,10x20000
```
