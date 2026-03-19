# Component Lifecycle

This diagram shows the flow of `create_world_model()` from user call
through component resolution, validation, and return.

```mermaid
sequenceDiagram
    participant User
    participant Factory as factory.py
    participant Registry as WorldModelRegistry
    participant Config as ConfigRegistry
    participant Model as WorldModel
    participant Validator as validate()

    User->>Factory: create_world_model("dreamerv3:size12m")
    Factory->>Factory: _bootstrap_factory_registry()
    Factory->>Registry: resolve_alias(model)
    Factory->>Config: from_pretrained(resolved, **kwargs)
    Config-->>Factory: WorldModelConfig

    Factory->>Registry: from_pretrained(resolved, **config_kwargs)
    Registry->>Registry: _load_builtin_models()
    Registry->>Registry: load_entrypoint_plugins()
    Registry-->>Factory: WorldModel instance

    alt component_overrides provided
        Factory->>Registry: apply_component_overrides(model, overrides)
        loop for each override
            Registry->>Registry: build_component(override)
            Registry->>Model: setattr(slot, component)
        end
    end

    Factory->>Validator: model.validate(raise_on_error=True)
    Validator->>Model: Check protocol compliance
    Validator->>Model: Check shape compatibility
    Validator->>Model: Check IO contract
    Validator-->>Factory: ValidationResult

    Factory->>Model: model.to(device)
    Factory-->>User: WorldModel (ready)
```

## Component Slots

| Slot | Protocol | Method |
|------|----------|--------|
| `observation_encoder` | `ObservationEncoder` | `encode()` |
| `action_conditioner` | `ActionConditioner` | `condition()` |
| `dynamics_model` | `DynamicsModel` | `transition()` |
| `decoder_module` | `Decoder` | `decode()` |
| `rollout_executor` | `RolloutExecutor` | `rollout_open_loop()` |

## Override Resolution Order

1. String id - resolved through component registry
2. Class - instantiated with `(model)` or `()` signature
3. Callable - invoked as factory function with `(config)`
4. Instance - used directly
