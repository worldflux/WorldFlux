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
- WorldFlux-vs-official comparison is currently paused until the Dreamer
  official-only reproducibility pass is stable.
- `create_world_model`, `Trainer`, `verify`, and `scaffold` still have
  PyTorch-native assumptions and are not yet treated as fully
  multi-backend-native product surfaces.
- TD-MPC2 is not considered parity-complete yet; adapter work exists, but the
  `official 5m` vs `WorldFlux 5m` architecture mismatch is still unresolved.

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
- TD-MPC2 remains on the proof roadmap, but unresolved architecture mismatch
  means a runnable manifest alone is not a final parity claim.

## Campaign Helpers (Reproducible Exports)

Campaign export/run/resume remains available for reproducible source artifact pipelines:

```bash
worldflux parity campaign export benchmarks/parity/campaign/dreamer_atari100k.yaml --source worldflux --seeds 0,1,2
worldflux parity campaign run benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode worldflux --device cuda --resume
worldflux parity campaign resume benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode both --device cuda
```

These commands are operational helpers and do not replace proof-grade verdict generation.
