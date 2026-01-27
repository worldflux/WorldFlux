# WorldLoom Roadmap

This document outlines the planned features and improvements for WorldLoom.

## Current Status: v0.1.0 (Alpha)

Core functionality is implemented and working. Ready for early adopters and feedback.

---

## Short Term (Q1 2026)

### v0.2.0 - Stability & Polish

- [ ] **PyPI Release**: `pip install worldloom`
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

1. [Good First Issues](https://github.com/worldloom/WorldLoom/labels/good%20first%20issue)
2. [Help Wanted](https://github.com/worldloom/WorldLoom/labels/help%20wanted)
3. [Contributing Guide](CONTRIBUTING.md)

Have a feature request? [Open an issue](https://github.com/worldloom/WorldLoom/issues/new?template=feature_request.md)!

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.
