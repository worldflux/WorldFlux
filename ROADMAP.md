# WorldFlux Roadmap

This document outlines the planned features and improvements for WorldFlux.

## Current Status: v0.1.0 (Alpha)

Core functionality is implemented and working. Ready for early adopters and feedback.

## Public Maturity Boundary

WorldFlux ships with explicit maturity tiers:

- **reference**: DreamerV3, TD-MPC2
- **experimental**: JEPA, V-JEPA2, Token, Diffusion
- **skeleton**: DiT, SSM, Renderer3D, Physics, GAN

Promotion from experimental to reference is **contract-first** and requires:

- stable `io_contract()` compatibility
- common quality gates (finite metrics, save/load parity, seed success >= 80%)
- family-specific gate pass

v3-first policy:

- `create_world_model()` default is `api_version="v3"`
- `v0.2` is migration-only and must be explicitly requested
- strict contract checks are required for release readiness

---

## 9/10 Readiness Minimum TODOs (MVP)

### P0 (must-have)
- [x] **Reproduction guide**: shortest real-data steps documented for Dreamer/TD-MPC2 (ref: `docs/tutorials/reproduce-dreamer-tdmpc2.md`)
- [x] **Reproducibility criteria**: seed variance/success-rate bounds documented (ref: `docs/reference/quality-gates.md`)
- [x] **Benchmark floor**: minimal loss-decrease threshold documented (ref: `docs/reference/quality-gates.md`)
- [x] **Dataset schema check**: required `.npz` keys/shapes documented (ref: `docs/api/training.md`)
- [x] **Example stability**: CI enforces example smoke tests + strict docs build (ref: `.github/workflows/ci.yml`)

### P1 (stabilization)
- [x] **Model family requirements**: required-component table documented by family (ref: `docs/EXTENSIBILITY.md`)
- [x] **CI gates definition**: explicit CI/release gate list documented (ref: `docs/reference/quality-gates.md`)
- [x] **OSS release checklist**: minimum pre-tag checklist documented (ref: `docs/reference/release-checklist.md`)

---

## Short Term (Q1 2026)

### v0.2.0 - Stability & Polish

- [x] **v3-first API default**
- [x] **Skeleton maturity tier in catalog**
- [x] **Strict condition/action contract validation**
- [x] **Docs CI strict build gate**
- [x] **Entry-point plugin discovery**
- [ ] **PyPI Release**: `uv pip install worldflux`
- [ ] **Pretrained Models**: Ready-to-use models for common benchmarks
  - [ ] DreamerV3: Atari Breakout, Pong, Seaquest
  - [ ] TD-MPC2: HalfCheetah, Walker, Humanoid
- [ ] **Benchmark Results**: Reproducible results on standard benchmarks
- [ ] **Hugging Face Hub Integration**: Upload/download models from HF Hub
- [ ] **Improved Documentation**: Video tutorials, more examples

### v0.3.0 - Training Improvements

- [ ] **Mixed Precision Training**: FP16/BF16 support
- [ ] **Multi-GPU Training**: DataParallel and DistributedDataParallel
- [ ] **W&B Integration**: Built-in Weights & Biases logging
- [ ] **Learning Rate Schedulers**: Cosine, warmup, etc.

---

## Medium Term (Q2-Q3 2026)

### v0.4.0 - New Architectures

- [ ] **V-JEPA Integration**: Video Joint Embedding Predictive Architecture
- [ ] **IRIS**: Transformer-based world model
- [ ] **Diffusion World Models**: Diffusion-based dynamics prediction
- [ ] **Autoregressive Models**: GPT-style world models
- [ ] **Custom Architecture API**: Easy way to add new model types

### v0.5.0 - Planning & Control

- [ ] **MPC Planner**: Model Predictive Control for TD-MPC2
- [ ] **Policy Learning**: Actor-critic integration
- [ ] **Planning Algorithms**: CEM, MPPI, iCEM

---

## Long Term (Q4 2026+)

### v1.0.0 - Production Ready

- [ ] **JAX/Flax Backend**: Alternative to PyTorch
- [ ] **ONNX Export**: For deployment
- [ ] **TensorRT Optimization**: Fast inference
- [ ] **Stable API**: Backward compatibility guarantees
- [ ] **Comprehensive Test Suite**: >90% coverage

### Future Ideas

- [ ] Real robot integration examples
- [ ] Sim-to-real transfer utilities
- [ ] Multi-agent world models
- [ ] Hierarchical world models
- [ ] Online learning / continual learning

---

## How to Contribute

We welcome contributions! Check out:

1. [Good First Issues](https://github.com/worldflux/WorldFlux/labels/good%20first%20issue)
2. [Help Wanted](https://github.com/worldflux/WorldFlux/labels/help%20wanted)
3. [Contributing Guide](CONTRIBUTING.md)

Have a feature request? [Open an issue](https://github.com/worldflux/WorldFlux/issues/new?template=feature_request.md)!

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.
