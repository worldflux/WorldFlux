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

## Event Schema

Each JSON Lines event has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `event` | string | Event type (e.g., `"run.complete"`, `"run.error"`) |
| `timestamp` | float | Unix epoch seconds |
| `run_id` | string | Unique run identifier (UUID) |
| `scenario` | string | Scenario name (e.g., `"quickstart"`, `"train"`) |
| `success` | bool | Whether the run completed successfully |
| `duration_sec` | float | Total wall-clock duration in seconds |
| `ttfi_sec` | float \| null | Time to first iteration (training runs only) |
| `artifacts` | list[string] | Paths to generated artifacts (checkpoints, logs) |
| `error` | string \| null | Error message if `success` is false |

Example event:

```json
{"event": "run.complete", "timestamp": 1760054400.0, "run_id": "abc-123", "scenario": "quickstart", "success": true, "duration_sec": 12.3, "ttfi_sec": 1.2, "artifacts": ["outputs/checkpoint_final.pt"], "error": null}
```

## Practical Interpretation

- `wasr`: number of unique successful runs in the lookback window (default: 7 days).
- `quickstart.success_rate`: successful CPU-first runs / attempts.
- `retention`: week-to-week overlap of active run IDs (proxy for repeat usage).

## Metric Thresholds

| Metric | Healthy | Warning | Action Required |
|--------|---------|---------|-----------------|
| `wasr` | ≥ 3 | 1-2 | 0 |
| `quickstart.success_rate` | ≥ 0.9 | 0.5-0.9 | < 0.5 |
| `retention` | ≥ 0.5 | 0.2-0.5 | < 0.2 |

## Privacy

All telemetry is **local only**. No data is sent to external servers. The metrics file
is stored in the project directory and can be deleted at any time. To disable event
logging entirely, set `WORLDFLUX_METRICS_PATH=/dev/null`.

## Related Docs

- [Quality Gates](quality-gates.md)
- [Benchmarks](benchmarks.md)
