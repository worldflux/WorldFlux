# Benchmarks

WorldFlux separates synthetic smoke benchmarks from evidence-oriented
`env_policy` lanes.

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

## Evidence Lanes

The two canonical evidence lanes in the current MVP are:

- DreamerV3 on `ALE/Breakout-v5`
- TD-MPC2 on `HalfCheetah-v5`

These are reproducible evidence bundles, not SOTA claims or proof claims.

### Evidence Lane 1: DreamerV3 Breakout

```bash
uv run python benchmarks/evidence_dreamerv3_breakout.py \
  --quick \
  --output-dir outputs/benchmarks/dreamerv3-breakout-evidence
```

Artifacts:

- `summary.json`
- `returns.jsonl`
- `learning_curve.csv`
- `checkpoint_index.json`
- `report.md`
- dataset manifest + replay buffer bundle

Evidence semantics:

- `eval_mode = env_policy`
- `policy_impl = candidate_actor_stateful_eval`
- learned-policy Atari rollout only; random env sampling is invalid evidence

### Evidence Lane 2: TD-MPC2 HalfCheetah

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

Evidence semantics:

- `eval_mode = env_policy`
- `policy_impl = cem_planner_eval`
- learned-policy MuJoCo rollout only; replay/data collection provenance is recorded separately

These lanes are intended to produce reproducible evidence bundles, not SOTA or
paper-parity claims.

## Reproducibility Notes

- Keep `seed` fixed for comparisons.
- CPU is the default benchmark target in quick mode.
- Full mode is intended for manual or scheduled evidence runs.
- Runtime and artifacts depend on hardware and optional dependencies.
