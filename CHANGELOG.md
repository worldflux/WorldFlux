# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community files (CODE_OF_CONDUCT, CHANGELOG, Issue templates)
- Public roadmap

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
