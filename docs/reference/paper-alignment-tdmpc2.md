# TD-MPC2 Paper Alignment

This page documents the current intended alignment between WorldFlux TD-MPC2
profiles and the reference-family/proof workflow vocabulary used in docs and
tooling.

## Profile Tiers

| Preset | Tier | Purpose |
| --- | --- | --- |
| `tdmpc2:ci` | `compatibility` | fast CI and scaffold validation |
| `tdmpc2:5m` | `reference` | native reference-family training surface |
| `tdmpc2:19m` | `reference` | native reference-family training surface |
| `tdmpc2:48m` | `reference` | native reference-family training surface |
| `tdmpc2:317m` | `reference` | native reference-family training surface |
| `tdmpc2:proof_5m` | `proof` | canonical parity/proof profile |
| `tdmpc2:5m_legacy` | `compatibility` | retained compatibility path |

## Current Alignment Contract

WorldFlux now surfaces the following TD-MPC2-aligned metadata and training
components in code:

- `training_tier`
- `parity_profile`
- target Q ensemble with EMA soft update via `target_q_tau`
- target Q EMA applied after optimizer steps, not during pure loss evaluation
- latent consistency loss
- reward loss
- terminal-aware TD loss
- policy loss

These values are surfaced through `TDMPC2Config` and
`TDMPC2WorldModel.reference_profile()`.

## What This Page Does Not Claim

- This page does not claim full paper reproduction on benchmark scores.
- `reference` means "intended native reference-family path", not "published
  proof artifact".
- `proof` means "reserved canonical parity profile", not that every local run is
  automatically proof-eligible.

## Source of Truth

- `src/worldflux/core/config.py`
- `src/worldflux/models/tdmpc2/world_model.py`
- `tests/test_models/test_tdmpc2_reference_fidelity.py`
