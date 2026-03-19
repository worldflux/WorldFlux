# Parity (Legacy vs Proof-Grade)

WorldFlux provides two parity tracks. They are intentionally different:

- `legacy quick parity`: fast non-inferiority checks for local development.
- `proof-grade parity`: strict official-vs-WorldFlux equivalence pipeline.

Use the proof path when you need a controlled statistical claim inside the
parity pipeline. A public proof claim still requires a published evidence
bundle that third parties can inspect.

## Current Program Status

As of 2026-03-11, WorldFlux is intentionally not treating proof work as
"100% complete".

- The current active phase is `official-only` DreamerV3 bootstrap on the
  official JAX stack.
- The immediate goal is to lock the official Dreamer run set to `10` seeds and
  then reach the `20`-seed minimum proof threshold.
- Dreamer proof compare now has a JAX/JAX canonical path:
  `official_dreamerv3_jax_subprocess` vs `worldflux_dreamerv3_jax_subprocess`.
  Statistical comparison still depends on the official-only reproducibility pass
  being stable enough to resume compare runs.
- Product-facing factory/training surfaces now expose delegated backend handles
  for family-native proof paths:
  Dreamer defaults to the JAX subprocess family and TD-MPC2 defaults to the
  Torch subprocess family when proof manifests are omitted.
- TD-MPC2 is not parity-complete yet, but the aligned `proof_5m` path is now
  conditionally runnable when a passing alignment report is available.

WorldFlux only treats proof work as "100%" when all of the following are true:

- Dreamer official statistical proof is complete.
- WorldFlux core surfaces operate as a real multi-backend product, not a
  PyTorch-native implementation with backend shims.
- TD-MPC2 architecture mismatch is resolved well enough to resume proof-grade
  comparison on that family.

## Public Claims Policy

- Local proof-mode outputs are not by themselves a public proof claim.
- Tracked fixture specifications and ignored generated runs under
  `reports/parity/` are release-gate aids, not public evidence bundles.
- Public proof claims require a published evidence bundle or report URL with the
  suite, upstream commit, run context, and final verdict available to third
  parties.

## Legacy Quick Parity

Legacy commands stay available for backward compatibility:

```bash
worldflux parity run benchmarks/parity/dreamer_atari100k.yaml --output reports/parity/runs/dreamer.json
worldflux parity aggregate --runs-glob "reports/parity/runs/*.json" --output reports/parity/aggregate.json
worldflux parity report --aggregate reports/parity/aggregate.json --output reports/parity/report.md
```

Legacy parity uses the `src/worldflux/parity/*` harness and non-inferiority verdicts.
It is not the source of truth for official equivalence proof.
The release workflow also regenerates deterministic fixtures into local ignored
`reports/parity/fixtures/` outputs so dry-runs stay reproducible. Those fixtures
are not proof-grade evidence and are not a public proof claim.

## Proof-Grade Official Equivalence Path

Proof commands call `scripts/parity/*`, which is the canonical path for:

- strict completeness (`missing_pairs == 0`)
- strict validity checks (no shortcut policies in proof mode)
- TOST equivalence + Holm correction
- canonical profile matching (`dreamerv3:official_xl`, `tdmpc2:proof_5m`)
- component match report gate when the suite requires it
- final global verdict

When `worldflux parity proof-run` or `worldflux parity proof` is called without
an explicit manifest, WorldFlux resolves the family-native canonical backend by
default:

- Dreamer: `official_dreamerv3_jax_subprocess`
- TD-MPC2: `official_tdmpc2_torch_subprocess`

### Run Matrix

```bash
worldflux parity proof-run scripts/parity/manifests/official_vs_worldflux_full_v2.yaml \
  --run-id parity_$(date -u +%Y%m%dT%H%M%SZ) \
  --device cuda \
  --resume
```

### Generate Proof Reports

```bash
worldflux parity proof-report scripts/parity/manifests/official_vs_worldflux_full_v2.yaml \
  --runs reports/parity/<RUN_ID>/parity_runs.jsonl
```

This emits:

- `coverage_report.json`
- `validity_report.json`
- `equivalence_report.json`
- `equivalence_report.md`
- `component_match_report.json` when required by the suite
- `evidence_bundle.zip`

Legacy and aggregate parity flows now also expose a lightweight evidence-bundle
index in JSON outputs:

- run artifacts expose `evidence_bundle.bundle_kind == "parity_run"`
- aggregate artifacts expose `evidence_bundle.bundle_kind == "parity_aggregate"`
- aggregate evidence bundles may point to:
  - `aggregate_json`
  - `dashboard_html`
  - `report_md`
  - the underlying `run_jsons`

The release workflow renders `reports/parity/dashboard.html` from
`reports/parity/aggregate.json` and uploads the parity evidence directory as an
artifact.

### Final Verdict Keys

For official proof claims, check `equivalence_report.json`:

- `global.parity_pass_final == true`
- `global.parity_pass_all_metrics == true`
- `global.missing_pairs == 0`
- `global.validity_pass == true`
- `global.component_match_pass == true` when the suite requires component matching

If any key is false, the proof run fails.

### Important Scope Note

The proof-grade command surface can execute official-vs-WorldFlux manifests,
but that does not mean every family is currently in an active public-proof
phase.

- DreamerV3 is currently in `official-only` reproducibility hardening before
  active WorldFlux comparison resumes.
- TD-MPC2 remains on the proof roadmap. An aligned `proof_5m` path is runnable,
  but that alone is not yet a final public parity claim.

## Campaign Helpers (Reproducible Exports)

Campaign export/run/resume remains available for reproducible source artifact pipelines:

```bash
worldflux parity campaign export benchmarks/parity/campaign/dreamer_atari100k.yaml --source worldflux --seeds 0,1,2
worldflux parity campaign run benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode worldflux --device cuda --resume
worldflux parity campaign resume benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode both --device cuda
```

These commands are operational helpers and do not replace proof-grade verdict generation.
