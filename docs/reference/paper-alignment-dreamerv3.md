# DreamerV3 Paper Alignment

This page documents the current intended alignment between WorldFlux DreamerV3
profiles and the DreamerV3 literature/official parity workflow.

## Profile Tiers

| Preset | Tier | Purpose |
| --- | --- | --- |
| `dreamer:ci` | `compatibility` | fast CI and scaffold validation |
| `dreamerv3:size12m` | `reference` | native reference-family training/inference surface |
| `dreamerv3:size25m` | `reference` | native reference-family training/inference surface |
| `dreamerv3:size50m` | `reference` | native reference-family training/inference surface |
| `dreamerv3:size100m` | `reference` | native reference-family training/inference surface |
| `dreamerv3:size200m` | `reference` | native reference-family training/inference surface |
| `dreamerv3:official_xl` | `proof` | canonical parity profile for official/JAX-backed proof flows |

## Current Alignment Contract

WorldFlux currently encodes the following DreamerV3-aligned defaults in code:

- world-model learning rate: `1e-4`
- gradient clipping: `1000.0`
- CNN/image reconstruction keeps raw observation targets; vector/MLP
  reconstruction may use `symlog`
- KL split:
  - dynamics: `0.5`
  - representation: `0.1`
- `kl_free = 1.0`
- `use_symlog = True`
- `use_twohot = True`

These values are surfaced through `DreamerV3Config` and
`DreamerV3WorldModel.reference_profile()`.

## What This Page Does Not Claim

- This page does not, by itself, prove paper-level benchmark parity.
- `reference` means "intended native reference-family surface", not "published
  proof result".
- `proof` means "the preset is reserved for parity/proof workflows", not that
  every local run is proof-eligible.

## Source of Truth

- `src/worldflux/core/config.py`
- `src/worldflux/models/dreamer/world_model.py`
- `tests/test_models/test_dreamer_reference_alignment.py`
