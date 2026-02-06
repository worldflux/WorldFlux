# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community files (CODE_OF_CONDUCT, CHANGELOG, Issue templates)
- Public roadmap
- `create_world_model()` defaults to `api_version="v3"` (v0.2 explicit bridge only)
- `ModelMaturity.SKELETON` catalog support and `list_models(maturity="skeleton")`
- Action contract enforcement (`ActionPayload` vs `io_contract().action_spec`) in transition/rollout paths
- Strict v3 condition-extra validation across reference/experimental families
- `component_overrides` in factory with operational component registry wiring
- External plugin discovery via entry-point groups: `worldflux.models`, `worldflux.components`
- Serialization metadata file `worldflux_meta.json` with compatibility checks on load
- CI docs gate: `uv run mkdocs build --strict`
- Training API docs now define required `ReplayBuffer.load()` `.npz` schema
- OSS release checklist reference doc (`docs/reference/release-checklist.md`)

### Changed
- Quality-gate documentation updated to `uv` command path
- `save_pretrained()` standardized across families with metadata emission
- Packaging metadata now uses SPDX license expression (`Apache-2.0`)
- Security reporting contact now explicitly points to `yhyoda@worldflux.ai`

## [0.1.0] - 2026-01-26

### Added
- Initial release
- DreamerV3 world model implementation
  - RSSM dynamics with categorical stochastic state
  - CNN/MLP encoder-decoder
  - Size presets: 12M, 25M, 50M, 100M, 200M parameters
- TD-MPC2 world model implementation
  - SimNorm latent space
  - Q-function ensemble
  - Size presets: 5M, 19M, 48M, 317M parameters
- Unified `WorldModel` protocol
  - `encode()`, `predict()`, `observe()`, `decode()`, `imagine()`
- Training infrastructure
  - `Trainer` class with callbacks
  - `ReplayBuffer` for trajectory data
  - Checkpoint save/load
- Factory API
  - `create_world_model()` one-liner creation
  - `list_models()` for available presets
- Documentation
  - MkDocs site with tutorials
  - API reference
  - Colab quickstart notebook
- Examples
  - Atari data collection and training
  - MuJoCo data collection and training
  - Imagination visualization

[Unreleased]: https://github.com/worldflux/Worldflux/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/worldflux/Worldflux/releases/tag/v0.1.0
