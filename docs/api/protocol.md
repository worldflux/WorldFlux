# WorldModel Base Class

All world models implement the `WorldModel` abstract base class.

## Interface

```python
class WorldModel(nn.Module, ABC):
    def encode(self, obs: Tensor | dict[str, Tensor], deterministic: bool = False) -> State: ...
    def transition(self, state: State, action: Tensor, deterministic: bool = False) -> State: ...
    def update(self, state: State, action: Tensor, obs: Tensor | dict[str, Tensor]) -> State: ...
    def decode(self, state: State) -> ModelOutput: ...
    def rollout(self, initial_state: State, actions: Tensor, deterministic: bool = False) -> Trajectory: ...
    def loss(self, batch: Batch) -> LossOutput: ...
```

## Key Methods

### encode

```python
state = model.encode(obs)
```

### transition

```python
next_state = model.transition(state, action)
```

### update

```python
next_state = model.update(state, action, obs)
```

### decode

```python
output = model.decode(state)
preds = output.preds
```

`ModelOutput` contains:

- `preds`: model predictions (e.g. `obs`, `reward`, `continue`, `q_values`)
- `state`: optional state
- `aux`: optional metadata

### rollout

```python
trajectory = model.rollout(initial_state, actions)
```

### loss

```python
loss_out = model.loss(batch)
```

`LossOutput` contains:

- `loss`: scalar training loss
- `components`: individual loss terms
- `metrics`: float metrics for logging

## Capabilities

Some models do not predict rewards or continuation probabilities. Use capability
helpers to branch safely:

```python
if model.supports_reward():
    ...
if model.supports_continue():
    ...
```

## Batch Format

Training uses the `Batch` container:

```python
batch = Batch(
    obs=...,
    actions=...,
    rewards=...,
    terminations=...,
    mask=...,
    context=...,
    target=...,
    extras=...,
)
```

## Trajectory

```python
trajectory.states    # list[State]
trajectory.rewards   # Tensor[T, B] | None
trajectory.continues # Tensor[T, B] | None
```
