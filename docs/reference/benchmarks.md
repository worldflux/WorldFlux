# Benchmarks

WorldFlux separates synthetic smoke benchmarks from evidence-oriented real
evaluation lanes.

## Synthetic Smoke Benchmarks

These scripts are compatibility and artifact-generation checks. They are not
performance claims.

Shared CLI contract:

- `--quick` (CI-safe short run)
- `--full` (longer run for manual/scheduled validation)
- `--seed <int>`
- `--output-dir <path>`

Synthetic smoke outputs:

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

## Evidence Lane

The first real benchmark lane is TD-MPC2 on HalfCheetah.

```bash
uv run python benchmarks/evidence_tdmpc2_halfcheetah.py \
  --quick \
  --collector-policy random \
  --output-dir outputs/benchmarks/tdmpc2-halfcheetah-evidence
```

Preferred collection path:

```bash
uv run python benchmarks/evidence_tdmpc2_halfcheetah.py \
  --quick \
  --policy-checkpoint ./outputs/checkpoint_final.pt \
  --output-dir outputs/benchmarks/tdmpc2-halfcheetah-evidence
```

Evidence lane artifacts:

- `summary.json`
- `returns.jsonl`
- `learning_curve.csv`
- `checkpoint_index.json`
- `report.md`
- dataset manifest + replay buffer bundle

This lane is intended to produce reproducible evidence bundles, not SOTA or
paper-parity claims.

## Reproducibility Notes

- Keep `seed` fixed for comparisons.
- CPU is the default benchmark target in quick mode.
- Full mode is intended for manual or scheduled evidence runs.
- Runtime and artifacts depend on hardware and optional dependencies.
