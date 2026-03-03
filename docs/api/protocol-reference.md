---
sidebar_label: Protocol API
---

# Protocol & Data Types Reference

Core protocol classes and data containers used throughout the WorldFlux framework.

---

## WorldModel

```python
class WorldModel(nn.Module, ABC)
```

Abstract base class for all world models in the WorldFlux framework. Inherits
from `torch.nn.Module` and exposes a composable component architecture where
each stage of the observe-predict-decode pipeline can be overridden independently.

### Pipeline Components

| Component | Description |
|-----------|-------------|
| `observation_encoder` | Encodes raw observations into a latent `State`. |
| `action_conditioner` | Fuses action and condition information into the dynamics input. |
| `dynamics_model` | Predicts the next latent state given current state and conditioned inputs. |
| `decoder_module` | Maps a latent state back to observable predictions (observations, rewards, continuation). |
| `rollout_executor` | Executes multi-step open-loop rollouts by chaining `transition` and `decode`. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `capabilities` | `set[Capability]` | Capability flags advertised by this model (e.g. `REWARD_PRED`, `PLANNING`). |
| `observation_encoder` | `ObservationEncoder \| None` | Pluggable encoder component. |
| `action_conditioner` | `ActionConditioner \| None` | Pluggable action/condition fusion component. |
| `dynamics_model` | `DynamicsModel \| None` | Pluggable latent dynamics component. |
| `decoder_module` | `Decoder \| None` | Pluggable decoder component. |
| `rollout_executor` | `RolloutExecutor \| None` | Pluggable rollout executor component. |
| `composable_support` | `set[str]` | Component slot names effective in runtime execution paths for this model. |

### Methods

#### encode

```python
def encode(
    self,
    obs: Tensor | dict[str, Tensor] | WorldModelInput,
    deterministic: bool = False,
) -> State
```

Encode observations into a latent `State`. Delegates to the attached
`observation_encoder` component.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs` | `Tensor \| dict[str, Tensor] \| WorldModelInput` | required | Raw observation tensor, a dict of named modality tensors, or a `WorldModelInput`. |
| `deterministic` | `bool` | `False` | If `True`, use deterministic encoding (e.g. posterior mean). |

**Returns:** `State` -- Latent representation.

**Raises:** `NotImplementedError` if no `observation_encoder` is attached.

#### transition

```python
def transition(
    self,
    state: State,
    action: ActionPayload | Tensor | None,
    conditions: ConditionPayload | None = None,
    deterministic: bool = False,
) -> State
```

Predict the next latent state given current state and action. Performs a single
imagination step through the dynamics model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state` | `State` | required | Current latent state. |
| `action` | `ActionPayload \| Tensor \| None` | required | Action to condition on. Accepts a raw tensor, an `ActionPayload`, or `None` for unconditional transition. |
| `conditions` | `ConditionPayload \| None` | `None` | Optional auxiliary condition signals (e.g. goal embeddings). |
| `deterministic` | `bool` | `False` | If `True`, use deterministic dynamics. |

**Returns:** `State` -- Predicted next latent state.

**Raises:** `NotImplementedError` if no `dynamics_model` is attached.

#### decode

```python
def decode(
    self,
    state: State,
    conditions: ConditionPayload | None = None,
) -> ModelOutput
```

Decode a latent state into observable predictions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state` | `State` | required | Latent state to decode. |
| `conditions` | `ConditionPayload \| None` | `None` | Optional auxiliary condition signals. |

**Returns:** `ModelOutput` -- Contains the `predictions` dict and the originating `state`.

**Raises:** `CapabilityError` if no `decoder_module` is attached.

#### rollout

```python
def rollout(
    self,
    initial_state: State,
    action_sequence: ActionSequence | ActionPayload | Tensor | None,
    conditions: ConditionPayload | None = None,
    deterministic: bool = False,
    mode: str = "autoregressive",
) -> Trajectory
```

Execute a multi-step open-loop rollout from an initial state.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_state` | `State` | required | Starting latent state. |
| `action_sequence` | `ActionSequence \| ActionPayload \| Tensor \| None` | required | Sequence of actions to apply. |
| `conditions` | `ConditionPayload \| None` | `None` | Optional auxiliary condition signals applied at each step. |
| `deterministic` | `bool` | `False` | If `True`, use deterministic transitions. |
| `mode` | `str` | `"autoregressive"` | Rollout mode. Only `"autoregressive"` is supported in v3. |

**Returns:** `Trajectory` -- Collected states, actions, rewards, and continuation flags.

#### loss (abstract)

```python
@abstractmethod
def loss(self, batch: Batch) -> LossOutput
```

Compute the training loss. Subclasses **must** implement this method.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch` | `Batch` | required | Training batch containing observations, actions, rewards, etc. |

**Returns:** `LossOutput` -- Contains the total loss tensor, component losses, and metrics.

#### supports

```python
def supports(self, capability: Capability) -> bool
```

Return `True` if the model advertises the given capability.

### Convenience Properties

| Property | Type | Description |
|----------|------|-------------|
| `supports_reward` | `bool` | Whether the model predicts rewards. |
| `supports_continue` | `bool` | Whether the model predicts continuation flags. |
| `supports_planning` | `bool` | Whether the model supports planning. |

### Example

```python
from worldflux import create_world_model

model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64), action_dim=6)
state = model.encode(obs)
next_state = model.transition(state, action)
output = model.decode(next_state)
```

---

## ActionPayload

```python
@dataclass
class ActionPayload
```

Polymorphic action container that supports multiple control modalities.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kind` | `ActionKind` | `"none"` | Action modality. One of `"none"`, `"continuous"`, `"discrete"`, `"token"`, `"latent"`, `"text"`. |
| `tensor` | `Tensor \| None` | `None` | Primary tensor for continuous or discrete actions. |
| `tokens` | `Tensor \| None` | `None` | Token tensor for token-based actions. |
| `latent` | `Tensor \| None` | `None` | Latent tensor for latent-space actions. |
| `text` | `list[str] \| None` | `None` | Text strings for text-conditioned actions. |
| `extras` | `dict[str, Any]` | `{}` | Additional metadata (e.g. planner horizon). |

### Methods

#### primary

```python
def primary(self) -> Tensor | None
```

Return the primary tensor representation, checking `tensor`, `tokens`, and
`latent` in order.

#### validate

```python
def validate(self, *, api_version: str = "v0.2") -> None
```

Validate payload consistency. Ensures only one primary representation is set
and that `kind="none"` payloads carry no data.

### Example

```python
# Continuous action
action = ActionPayload(kind="continuous", tensor=torch.randn(6))

# Discrete action
action = ActionPayload(kind="discrete", tensor=torch.tensor([3]))

# Token-based action
action = ActionPayload(kind="token", tokens=torch.tensor([42, 7, 13]))
```

---

## ConditionPayload

```python
@dataclass
class ConditionPayload
```

Optional side-conditions for conditional world modeling.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text_condition` | `Tensor \| list[str] \| None` | `None` | Text condition embedding or raw text strings. |
| `goal` | `Tensor \| None` | `None` | Goal state tensor. |
| `spatial` | `Tensor \| None` | `None` | Spatial condition tensor (e.g. map, layout). |
| `camera_pose` | `Tensor \| None` | `None` | Camera pose tensor for 3D-conditioned models. |
| `extras` | `dict[str, Any]` | `{}` | Additional condition signals. Keys must follow namespaced format `"wf.<domain>.<name>"`. |

---

## WorldModelInput

```python
@dataclass
class WorldModelInput
```

Unified model input object wrapping observations, context, actions, and conditions.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `observations` | `dict[str, Tensor]` | `{}` | Named observation tensors keyed by modality name. |
| `context` | `dict[str, Tensor]` | `{}` | Additional context tensors. |
| `action` | `ActionPayload \| None` | `None` | Action payload for conditioned inference. |
| `conditions` | `ConditionPayload` | `ConditionPayload()` | Side-condition payload. |

---

## ModelOutput

```python
@dataclass
class ModelOutput
```

Standardized model output container returned by `WorldModel.decode()`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `predictions` | `dict[str, Tensor]` | `{}` | Predicted tensors keyed by name (e.g. `"obs"`, `"reward"`, `"continue"`). |
| `state` | `State \| None` | `None` | Latent state that produced these predictions. |
| `uncertainty` | `Tensor \| None` | `None` | Optional uncertainty estimate. |
| `aux` | `dict[str, Any]` | `{}` | Auxiliary outputs (e.g. attention maps, intermediate activations). |
| `prediction_spec` | `PredictionSpec \| None` | `None` | Spec describing expected prediction keys. |
| `sequence_layout` | `SequenceLayout \| None` | `None` | Axis layout metadata for prediction tensors. |

### Example

```python
output = model.decode(state)
obs_pred = output.predictions["obs"]
reward_pred = output.predictions.get("reward")
```

---

## LossOutput

```python
@dataclass
class LossOutput
```

Standardized loss container returned by `WorldModel.loss()`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `loss` | `Tensor` | required | Total scalar loss for backpropagation. |
| `components` | `dict[str, Tensor]` | `{}` | Individual loss components (e.g. `"reconstruction"`, `"kl"`, `"reward"`). |
| `metrics` | `dict[str, float]` | `{}` | Scalar metrics for logging (e.g. gradient norms, latent statistics). |

### Example

```python
loss_out = model.loss(batch)
loss_out.loss.backward()

# Log individual components
for name, value in loss_out.components.items():
    print(f"{name}: {value.item():.4f}")
```

---

## Trajectory

```python
@dataclass
class Trajectory
```

Imagination rollout trajectory in latent space. Returned by
`WorldModel.rollout()`.

The trajectory maintains the invariant that `len(states) == actions.shape[0] + 1`,
representing the initial state plus one state per action taken.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `states` | `list[State]` | required | List of latent states `[T+1]` (initial + T steps). |
| `actions` | `Tensor` | required | Action tensor `[T, batch, action_dim]`. |
| `rewards` | `Tensor \| None` | `None` | Predicted rewards `[T, batch]`. |
| `values` | `Tensor \| None` | `None` | Predicted values `[T+1, batch]`. |
| `continues` | `Tensor \| None` | `None` | Continue probabilities `[T, batch]`. |
| `state_spec` | `StateSpec \| None` | `None` | Spec describing state tensor keys. |
| `sequence_layout` | `SequenceLayout \| None` | `None` | Axis layout metadata. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `horizon` | `int` | Prediction horizon (number of actions). |
| `batch_size` | `int` | Batch size from the first state. |

### Methods

#### to_tensor

```python
def to_tensor(self, key: str) -> Tensor
```

Stack a specific state tensor key across time `[T+1, batch, ...]`.

#### to

```python
def to(self, device: torch.device) -> Trajectory
```

Move all tensors to the specified device.

#### detach

```python
def detach(self) -> Trajectory
```

Detach all tensors from the computation graph.

### Example

```python
trajectory = model.rollout(initial_state, action_sequence)
print(f"Horizon: {trajectory.horizon}")
print(f"Rewards shape: {trajectory.rewards.shape}")

# Stack deterministic state across time
deter_stack = trajectory.to_tensor("deter")
```
