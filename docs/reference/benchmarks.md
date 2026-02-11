# Benchmarks

WorldFlux provides three reproducible benchmark entrypoints with aligned CLI contracts.

## Common CLI Contract

All benchmark scripts support:

- `--quick` (CI-safe short run)
- `--full` (longer run for manual/scheduled validation)
- `--seed <int>`
- `--output-dir <path>`

All runs emit:

- summary JSON (`summary.json`)
- visualization artifact (`imagination.ppm`)

## Benchmark 1: DreamerV3 (Atari-oriented)

```bash
uv run python benchmarks/benchmark_dreamerv3_atari.py --quick --seed 42
```

Full-mode example:

```bash
uv run python benchmarks/benchmark_dreamerv3_atari.py --full --data atari_data.npz --seed 42
```

Expected minimum result:

- finite losses
- imagination artifact generated

## Benchmark 2: TD-MPC2 (MuJoCo-oriented)

```bash
uv run python benchmarks/benchmark_tdmpc2_mujoco.py --quick --seed 42
```

Full-mode example:

```bash
uv run python benchmarks/benchmark_tdmpc2_mujoco.py --full --data mujoco_data.npz --seed 42
```

Expected minimum result:

- finite losses
- imagination artifact generated

## Benchmark 3: Diffusion Imagination

```bash
uv run python benchmarks/benchmark_diffusion_imagination.py --quick --seed 42
```

Full-mode example:

```bash
uv run python benchmarks/benchmark_diffusion_imagination.py --full --seed 42
```

Expected minimum result:

- finite losses
- imagination artifact generated

## Reproducibility Notes

- Keep `seed` fixed for comparisons.
- CPU is the default benchmark target in quick mode.
- Full mode is intended for `workflow_dispatch` or scheduled workflows.
- Runtime and artifacts depend on hardware and optional dependencies.
