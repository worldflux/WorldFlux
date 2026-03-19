# Module Dependency Graph

WorldFlux is organized into 5 layers with strict dependency direction.
Each layer may depend on layers below it but never above.

```mermaid
graph TD
    CLI["cli/\n(CLI entry points)"]
    TRAINING["training/\n(Trainer, Callbacks, Resilience)"]
    EXECUTION["execution/\n(Backend bridge, Rollout)"]
    MODELS["models/\n(DreamerV3, TD-MPC2, Skeletons)"]
    CORE["core/\n(Interfaces, Config, Registry, EventBus)"]

    CLI --> TRAINING
    CLI --> EXECUTION
    CLI --> MODELS
    TRAINING --> CORE
    TRAINING --> MODELS
    EXECUTION --> CORE
    EXECUTION --> MODELS
    MODELS --> CORE

    style CORE fill:#2d6a4f,stroke:#1b4332,color:#fff
    style MODELS fill:#40916c,stroke:#2d6a4f,color:#fff
    style EXECUTION fill:#52b788,stroke:#40916c,color:#000
    style TRAINING fill:#74c69d,stroke:#52b788,color:#000
    style CLI fill:#95d5b2,stroke:#74c69d,color:#000
```

## Layer Responsibilities

| Layer | Package | Responsibility |
|-------|---------|----------------|
| 5 | `cli/` | User-facing CLI commands, terminal output |
| 4 | `training/` | Trainer loop, callbacks, resilience, data loading |
| 3 | `execution/` | Backend bridge, rollout execution, planning |
| 2 | `models/` | Model implementations (DreamerV3, TD-MPC2, etc.) |
| 1 | `core/` | Interfaces, config, registry, events, state, batch |

## Key Rules

- `core/` must never import from `models/`, `training/`, or `cli/`.
- `models/` implements `core/` contracts but must not import `training/`.
- `training/` consumes model contracts through `core/` interfaces.
- Cross-layer lazy imports are acceptable for optional functionality.
