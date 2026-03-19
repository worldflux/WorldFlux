# ADR 0003: Lazy Import Strategy

## Status

Accepted

## Context

WorldFlux supports 11 model families (DreamerV3, TD-MPC2, JEPA,
V-JEPA2, Token, Diffusion, DiT, SSM, Renderer3D, Physics, GAN).
Importing all model modules at package load time would:

- Add 2-5 seconds to CLI startup due to PyTorch imports.
- Pull in optional dependencies that users may not have installed.
- Increase memory footprint for users who only need one model family.

## Decision

Adopt lazy importing at two levels:

1. **Model modules**: `WorldModelRegistry._load_builtin_models()` is
   deferred until the first call to `list_models()` or
   `from_pretrained()`. Each model module is imported via
   `importlib.import_module()` only when needed.

2. **Optional dependencies**: Heavy dependencies (wandb, huggingface_hub,
   gymnasium, yaml) are imported inside the functions that use them,
   with clear error messages when not installed.

3. **Public API**: `worldflux/__init__.py` exposes `create_world_model`,
   `list_models`, and other public APIs without importing model
   implementations. The `core` and `verify` packages use similar
   lazy patterns.

## Consequences

- CLI commands that do not create models (`worldflux --version`,
  `worldflux list`) start in under 0.5 seconds.
- Plugin entry points are loaded lazily via
  `load_entrypoint_plugins()`.
- Import errors for specific model families surface as clear
  `ImportError` messages rather than silent failures.
- Type checkers see the full API surface through explicit re-exports,
  not dynamic imports.
- The `_builtin_models_loaded` flag prevents redundant imports.
