# CUDA/JAX Compatibility Matrix

> Maintained as part of ML-06 parity infrastructure stabilization.
> Last updated: 2026-03-18

## Problem Context

Recent parity runs have failed with `"Unable to initialize backend 'cuda'"` in
JAX. This document tracks the known-good version combinations for the parity
environment.

## Compatibility Matrix

| JAX Version | jaxlib Version | CUDA Toolkit | cuDNN | NVIDIA Driver | Status |
|-------------|---------------|--------------|-------|---------------|--------|
| 0.4.30 | 0.4.30+cuda12 | 12.3 | 8.9 | >= 535.104 | Tested |
| 0.4.28 | 0.4.28+cuda12 | 12.2 | 8.9 | >= 530.30 | Tested |
| 0.4.25 | 0.4.25+cuda12 | 12.1 | 8.9 | >= 525.60 | Known-good |
| 0.4.20 | 0.4.20+cuda11 | 11.8 | 8.6 | >= 520.61 | Legacy |

## Known Failure Modes

1. **Driver mismatch**: CUDA toolkit version requires minimum driver version.
   Check with `nvidia-smi` and cross-reference the matrix above.
2. **jaxlib wheel mismatch**: CPU-only jaxlib installed instead of CUDA variant.
   Verify with `python -c "import jax; print(jax.devices())"`.
3. **Container CUDA version**: Docker image may bundle a different CUDA toolkit
   than the host driver supports. Always pin versions in Dockerfile.
4. **XLA compilation cache**: Stale XLA compilation caches can cause silent
   failures. Clear with `rm -rf ~/.cache/jax`.

## Recommended Setup

For parity CI environments, use the Docker specification in
`docker/parity-environment.Dockerfile`.

### Quick Verification

```bash
# Check CUDA driver
nvidia-smi

# Check JAX backend
python -c "import jax; print(jax.devices()); print(jax.__version__)"

# Run pre-flight check
python scripts/preflight_parity_env.py
```

## Fallback Strategy

When JAX GPU is unavailable:

1. **JAX CPU fallback**: Set `JAX_PLATFORM_NAME=cpu` - usable for
   correctness testing but not for performance benchmarks.
2. **Native PyTorch reference**: Use `--backend native_torch_reference`
   to compare WorldFlux PyTorch against itself (self-parity).
3. **Skip**: Mark parity run as "jax_unavailable" and defer to next
   scheduled run.
