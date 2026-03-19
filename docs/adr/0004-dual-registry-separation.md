# ADR 0004: Dual Registry Separation (Config + Model)

## Status

Accepted

## Context

WorldFlux needs to resolve model identifiers (e.g. "dreamerv3:size12m")
into both configuration objects and model instances. A single registry
handling both responsibilities would couple config resolution with
model instantiation, making it difficult to:

- Inspect or modify configs before creating a model.
- Support config-only workflows (serialization, comparison, CI).
- Load configs from saved checkpoints without importing model code.

## Decision

Maintain two separate registries:

1. **ConfigRegistry** - maps `model_type` strings to
   `WorldModelConfig` subclasses. Responsible for:
   - Resolving `"dreamerv3:size12m"` to `DreamerV3Config.from_size("size12m")`
   - Loading configs from JSON files and HuggingFace Hub
   - Size preset resolution

2. **WorldModelRegistry** - maps `model_type` strings to model classes.
   Responsible for:
   - Model class registration and plugin loading
   - Component registry (reusable component slots)
   - Alias resolution and catalog management
   - Full model instantiation from pretrained paths

The `@WorldModelRegistry.register("dreamer", config_class=DreamerV3Config)`
decorator registers both the model and its config class in their
respective registries.

## Consequences

- `get_config("dreamerv3:size12m")` works without importing model code.
- `ConfigRegistry.from_pretrained()` is lightweight and testable in
  isolation.
- Model instantiation always goes through `WorldModelRegistry`, which
  delegates config creation to `ConfigRegistry`.
- Plugin authors register both config and model in a single decorator
  call, keeping the API simple.
- The two registries share the `model_type` key space, so naming
  conflicts are impossible by construction.
