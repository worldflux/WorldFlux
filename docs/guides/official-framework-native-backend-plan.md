# Official Framework-Native Backend Plan

This note tracks the staged rollout for family-native proof backends in
WorldFlux.

## Current Policy

- DreamerV3 canonical proof path:
  `official_dreamerv3_jax_subprocess` vs
  `worldflux_dreamerv3_jax_subprocess`
- TD-MPC2 canonical proof policy:
  `official_tdmpc2_torch_subprocess` vs WorldFlux native Torch
- `create_world_model()` may return an `OfficialBackendHandle` for delegated
  backend requests
- `Trainer` accepts delegated handles and routes them through `submit()`
- `worldflux parity proof-run` / `worldflux parity proof` resolve the
  family-native canonical backend by default when the manifest is omitted

## Phase Status

### Phase 1: Public Surface Alignment

Completed scope:

- factory / trainer / execution / CLI policy alignment
- family-native canonical backend defaults for proof CLI routing
- docs updated for delegated backend handles and delegated training

### Phase 2: Dreamer WorldFlux JAX Runtime

Remaining scope:

- replace vendored-official runtime dependency inside
  `worldflux_dreamerv3_jax_subprocess`
- keep adapter id, manifest schema, and artifact contract stable

### Phase 3: Dreamer Proof Runtime Hardening

Remaining scope:

- execution-level logs and diagnostics completeness
- rerun stability
- reproducible smoke / stage / preseed / compare flow

### Phase 4: TD-MPC2 Proof Compare Unblock

Remaining scope:

- finish resolving official `5m` vs WorldFlux `proof_5m` architecture mismatch
- promote the aligned `proof_5m` path from conditionally runnable to fully
  proof-ready
- remove the remaining blocked-path messaging for unaligned paths only after
  compare/validity/stats are fully stable
