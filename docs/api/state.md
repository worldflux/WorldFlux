# State

The core representation used by all world models.

## Overview

`State` is a lightweight container with two fields:

- `tensors`: `dict[str, Tensor]` holding model-specific latent tensors
- `meta`: `dict[str, Any]` for optional metadata

```python
from worldflux.core.state import State
```

## Creating a State

Most users get a `State` via `model.encode()` or `model.update()`:

```python
state = model.encode(obs)
```

## Accessing Tensors

`State` does not fix a schema. Each model defines its own tensor keys.

### DreamerV3

- `deter`: deterministic GRU state
- `stoch`: stochastic categorical samples
- `prior_logits` / `posterior_logits`: logits for KL

```python
features = torch.cat(
    [state.tensors["deter"], state.tensors["stoch"].flatten(1)],
    dim=-1,
)
```

### TD-MPC2

- `latent`: SimNorm embedding

```python
latent = state.tensors["latent"]
```

### JEPA

- `rep`: encoder representation

```python
rep = state.tensors["rep"]
```

## Metadata

Use `meta` for non-tensor bookkeeping:

```python
state.meta["latent_type"] = "simnorm"
```

## Implementation

```python
@dataclass
class State:
    tensors: dict[str, Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
```
