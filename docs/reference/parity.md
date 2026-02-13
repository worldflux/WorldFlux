# Parity Harness (Oracle Comparison)

WorldFlux provides a dedicated parity harness to compare model-family score quality
against pinned upstream oracle artifacts.

## Scope

- DreamerV3: Atari100k (`benchmarks/parity/dreamer_atari100k.yaml`)
- TD-MPC2: DMControl39 (`benchmarks/parity/tdmpc2_dmcontrol39.yaml`)

Pinned upstream references are recorded in:

- `reports/parity/upstream_lock.json`

## Statistical Rule

Parity verdict uses a one-sided non-inferiority test on relative score drop:

- drop ratio = `(upstream - worldflux) / max(abs(upstream), 1.0)`
- confidence = 95%
- margin = 5%
- pass when one-sided upper confidence bound is `<= 0.05`

## CLI

Run one suite:

```bash
worldflux parity run benchmarks/parity/dreamer_atari100k.yaml --output reports/parity/runs/dreamer.json
```

Aggregate multiple suites:

```bash
worldflux parity aggregate --runs-glob "reports/parity/runs/*.json" --output reports/parity/aggregate.json
```

Render markdown report:

```bash
worldflux parity report --aggregate reports/parity/aggregate.json --output reports/parity/report.md
```

## Input Formats

Supported source formats:

- `canonical_json`
- `canonical_jsonl`
- `dreamerv3_scores_json_gz`
- `tdmpc2_results_csv_dir`

Use suite files to pin expected upstream format/path and optionally override via CLI.

## Artifact Fields (Backward Compatible Additions)

`worldflux.parity.run.v1` keeps existing keys and adds optional metadata:

- `evaluation_manifest`:
  - `runner`, `python`, `torch`, `cuda`, `seed_policy`, `generated_at_utc`
- `artifact_integrity`:
  - `suite_sha256`, `upstream_input_sha256`, `worldflux_input_sha256`, `upstream_lock_sha256`
- `suite_lock_ref`:
  - `suite_id`, `lock_version`, `locked_upstream_commit`, `resolved_upstream_commit`, `matches_lock`

`worldflux.parity.aggregate.v1` adds suite-level `verdict_reason` text explaining pass/fail.

## Release Validation

Release workflow validates fixed artifacts:

```bash
uv run python scripts/validate_parity_artifacts.py \
  --run reports/parity/runs/dreamer_atari100k.json \
  --run reports/parity/runs/tdmpc2_dmcontrol39.json \
  --aggregate reports/parity/aggregate.json \
  --lock reports/parity/upstream_lock.json \
  --required-suite dreamer_atari100k \
  --required-suite tdmpc2_dmcontrol39 \
  --max-missing-pairs 0
```

If either suite fails non-inferiority, release is blocked.
