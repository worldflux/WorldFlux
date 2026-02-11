# WASR Metrics

WASR (Weekly Active Successful Runs) is a lightweight health signal for first-run and repeat usage.

## Event Log Format

WorldFlux writes local JSON Lines telemetry to:

- default: `.worldflux/metrics.jsonl`
- override: `WORLDFLUX_METRICS_PATH`

Each event includes:

- `event`
- `timestamp`
- `run_id`
- `scenario`
- `success`
- `duration_sec`
- `ttfi_sec`
- `artifacts`
- `error`

## Compute WASR

```bash
uv run python scripts/compute_wasr.py --input .worldflux/metrics.jsonl
```

Optional deterministic timestamp:

```bash
uv run python scripts/compute_wasr.py --input .worldflux/metrics.jsonl --now 1760054400
```

## Practical Interpretation

- `wasr`: number of unique successful runs in the lookback window.
- `quickstart.success_rate`: successful CPU-first runs / attempts.
- `retention`: week-to-week overlap of active run IDs (proxy for repeat usage).
