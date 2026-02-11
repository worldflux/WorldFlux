# CPU Success Path

This is the shortest official path to a successful WorldFlux run on CPU-only machines.

## Run

```bash
uv sync --extra dev
uv run python examples/quickstart_cpu_success.py --quick
```

## Success Criteria

The run is considered successful when all of the following are true:

- `initial_loss` and `final_loss` are finite
- `final_loss < initial_loss`
- imagination rollout horizon is generated
- artifacts are written

Artifacts:

- `outputs/quickstart_cpu/summary.json`
- `outputs/quickstart_cpu/imagination.ppm`

## Troubleshooting

- If `uv` dependencies are missing, re-run `uv sync --extra dev`.
- If the run is too slow on your machine, keep `--quick` enabled.
- If your environment has stale outputs, remove `outputs/quickstart_cpu` and retry.
