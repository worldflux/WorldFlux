# Parity (Advanced Evidence Workflow)

WorldFlux provides two parity tracks. They are intentionally different:

- `legacy quick parity`: fast non-inferiority checks for local development.
- `proof-grade parity`: strict official-vs-WorldFlux equivalence pipeline.

If you are using WorldFlux for the first time, do not start here.
Use the supported MVP lanes first:

- CPU-first success path
- scaffolded `contract_smoke`
- scaffolded `meaningful_local_training`

Come back to proof-oriented parity only when you need advanced evidence workflows.

Use the proof path when you need a controlled statistical claim inside the
parity pipeline. A public proof claim still requires a published evidence
bundle that third parties can inspect.

## Current Program Status

As of 2026-03-22, the proof-grade path is the primary parity surface for
DreamerV3 and TD-MPC2.

- Dreamer proof compare uses the canonical JAX/JAX path:
  `official_dreamerv3_jax_subprocess` vs `worldflux_dreamerv3_jax_subprocess`.
- TD-MPC2 proof compare uses the canonical Torch/native path:
  `official_tdmpc2_torch_subprocess` vs `worldflux_tdmpc2_native`.
- When manifests are omitted, WorldFlux resolves the family-native proof
  backend automatically.
- Proof completion is judged from the generated artifact set, not from
  newcomer smoke workflows:
  `coverage_report.json`, `validity_report.json`, `equivalence_report.json`,
  `equivalence_report.md`, `stability_report.json`, and `evidence_bundle.zip`.

WorldFlux only treats a proof run as complete when all of the following are true:

- `global.parity_pass_final == true`
- `global.validity_pass == true`
- `global.missing_pairs == 0`
- `global.component_match_pass == true` when the suite requires component matching
- `stability_report.json` is present for the run

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
- `stability_report.json`
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

Then check `stability_report.json`:

- `status == "stable"` for the current run
- `rerun_consistency.verdict_flip_detected == false`
- `rerun_consistency.pairwise_metric_sign_flip_detected == false`
- `rerun_consistency.bayesian_frequentist_mismatch_detected == false`

### Important Scope Note

`quick` / legacy parity and proof-grade parity are intentionally separate:

- `quick` is for local smoke and compatibility checks.
- `legacy quick parity` is for older non-inferiority workflows.
- `proof` is the official equivalence surface and the only parity path that
  counts toward proof-grade claims.

## Campaign Helpers (Reproducible Exports)

Campaign export/run/resume remains available for reproducible source artifact pipelines:

```bash
worldflux parity campaign export benchmarks/parity/campaign/dreamer_atari100k.yaml --source worldflux --seeds 0,1,2
worldflux parity campaign run benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode worldflux --device cuda --resume
worldflux parity campaign resume benchmarks/parity/campaign/tdmpc2_dmcontrol39.yaml --mode both --device cuda
```

These commands are operational helpers and do not replace proof-grade verdict generation.
