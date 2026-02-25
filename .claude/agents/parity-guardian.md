# Parity Guardian Guide

## Golden Rule

**Never modify artifacts in `reports/parity/`.** These are proof-grade records. The project CLAUDE.md explicitly prohibits this.

## 2-Track System: Quick vs Proof-Grade

### CI Smoke Tests (Quick)

Located in `src/worldflux/parity/ci_smoke.py`. Runs forward + loss + backward for N steps on synthetic data and checks:

- Loss within expected range (no NaN/Inf)
- Component losses present and within range
- Gradient flow (non-zero gradient norms)
- Parameter updates (delta norm within range)

Each model family has a `SmokeCheckpoint` with expected ranges:

```python
# Example: DreamerV3 smoke checkpoint
SmokeCheckpoint(
    family="dreamerv3",
    step_count=100,
    loss_range=(0.01, 200.0),
    component_loss_ranges={"kl": (0.0, 100.0), "reconstruction": (0.0, 100.0), ...},
    gradient_norm_range=(1e-8, 1e6),
    param_delta_norm_range=(1e-10, 1e4),
)
```

Use `run_smoke_test(model, family="dreamerv3")` for quick validation. Only `dreamerv3` and `tdmpc2` families are supported.

### Proof-Grade Parity (Official)

Located in `src/worldflux/parity/harness.py`. Full statistical comparison against upstream baselines:

1. **`run_suite(suite_path)`** -- Run one parity suite. Loads upstream and WorldFlux score points, pairs them by (task, seed), computes drop ratios, runs non-inferiority test, writes JSON artifact.
2. **`aggregate_runs(run_paths)`** -- Aggregate multiple suite results into a single verdict with global rollup.
3. **`render_markdown_report(aggregate)`** -- Human-readable markdown summary.

Output goes to `reports/parity/runs/<suite_id>.json` by default.

## Non-Inferiority Test

Defined in `src/worldflux/parity/stats.py`:

```python
non_inferiority_test(
    drop_ratios,           # positive = WorldFlux underperforms
    margin_ratio=0.05,     # acceptable performance gap (5%)
    confidence=0.95,       # one-sided confidence level
    bootstrap_samples=4000,
    seed=0,
)
```

**Mechanism**: Bootstrap resampling of mean drop ratios. Computes one-sided upper confidence interval. Passes when `ci_upper <= margin_ratio`.

- `drop_ratio = (upstream - worldflux) / max(|upstream|, 1.0)` for higher-is-better metrics
- Positive values mean WorldFlux underperforms
- The test is one-sided: we only care about the upper bound (worst case for WorldFlux)

**Result**: `NonInferiorityResult` with `pass_non_inferiority` boolean and `verdict_reason` string.

## CI Smoke vs Official Parity

| Aspect | CI Smoke | Official Parity |
|--------|----------|-----------------|
| Purpose | Catch regressions fast | Statistical proof of equivalence |
| Data | Synthetic random batches | Real training run scores |
| Duration | Seconds (100 steps) | Hours/days (full training) |
| Output | `SmokeResult` in-memory | JSON artifact in `reports/parity/` |
| When | Every CI run | Before releases, after major changes |
| Families | dreamerv3, tdmpc2 | Any with upstream baselines |

## Key Invariants

- Upstream lock file at `reports/parity/upstream_lock.json` pins upstream commit hashes per suite
- All artifacts include SHA-256 integrity hashes for inputs
- Suite specs define `margin_ratio` and `confidence` per model family
- Minimum 2 paired samples required for non-inferiority test
