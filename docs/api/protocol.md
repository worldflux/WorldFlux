# WorldModel Base Class

All world models implement the `WorldModel` base class.

## Interface (v3 default)

```python
class WorldModel(nn.Module, ABC):
    def encode(self, obs: Tensor | dict[str, Tensor] | WorldModelInput, deterministic: bool = False) -> State: ...
    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State: ...
    def update(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State: ...
    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput: ...
    def rollout(
        self,
        initial_state: State,
        action_sequence: ActionSequence | ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
        mode: str = "autoregressive",
    ) -> Trajectory: ...
    def loss(self, batch: Batch) -> LossOutput: ...
```

Legacy calls (`encode(obs)`, `transition(state, action_tensor)`) still work in `v0.2`.

`rollout(..., mode=...)` is deprecated in `v0.2` and removed in `v0.3`.
Use planner strategies (`worldflux.planners`) for re-planning and tree-search behaviors.

`create_world_model()` now defaults to `api_version="v3"`. Use `api_version="v0.2"` only for
explicit migration bridging.

## Key Payload Types

```python
ActionPayload(kind, tensor=None, tokens=None, latent=None, text=None, extras={})
ConditionPayload(text_condition=None, goal=None, spatial=None, camera_pose=None, extras={})
WorldModelInput(observations, context, action, conditions)
```

Planner payload metadata:

- canonical key: `extras["wf.planner.horizon"]` (`int >= 1`)
- legacy key: `extras["wf.planner.sequence"]` (deprecated in v0.2, removed in v0.3)
- helper APIs: `normalize_planned_action(...)`, `first_action(...)`

Condition extras in strict mode:

- keys must be namespaced (`wf.<domain>.<name>`)
- keys must be declared by each model's `io_contract().condition_spec.allowed_extra_keys`

## ModelOutput

`ModelOutput` now uses `predictions` as canonical field, with `preds` kept as a compatibility alias.

- `predictions`: model predictions (`obs`, `reward`, `continue`, `q_values`, ...)
- `state`: optional state
- `uncertainty`: optional uncertainty tensor
- `aux`: optional metadata

## Capabilities

Use capability helpers to branch safely:

```python
if model.supports_reward:
    ...
if model.supports_continue:
    ...
```

## Batch Format

`Batch` supports both legacy and universal forms in `v0.2`:

```python
batch = Batch(
    # legacy
    obs=...,
    actions=...,
    rewards=...,
    terminations=...,
    # universal
    inputs={...},
    targets={...},
    conditions={...},
    extras={...},
)
```

## Trajectory

```python
trajectory.states    # list[State]
trajectory.rewards   # Tensor[T, B] | None
trajectory.continues # Tensor[T, B] | None
```

## Serialization Contract

`save_pretrained(path)` writes:

- `config.json`
- `model.pt`
- `worldflux_meta.json`

`worldflux_meta.json` includes compatibility fields:

- `save_format_version`
- `worldflux_version`
- `api_version`
- `model_type`
- `contract_fingerprint`
- `created_at_utc`
