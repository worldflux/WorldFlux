# Component Override Matrix

This matrix documents which canonical component slots are declared as
runtime-effective by core model families.

## Canonical Slots

- `observation_encoder`
- `action_conditioner`
- `dynamics_model`
- `decoder`
- `rollout_executor`

Deprecated aliases such as `rollout_engine` and `decoder_module` are normalized
to the canonical slots above.

## Current Family Matrix

| Family | observation_encoder | action_conditioner | dynamics_model | decoder | rollout_executor |
| --- | --- | --- | --- | --- | --- |
| DreamerV3 | yes | yes | yes | yes | yes |
| TD-MPC2 | yes | yes | yes | yes | yes |
| JEPA | no | no | no | no | no |

## Source of Truth

The docs are backed by registry and model metadata:

- `WorldModelRegistry.list_component_slots()`
- `WorldModelRegistry.describe_composable_support(...)`
- model-level `composable_support`

Example:

```python
from worldflux.core.registry import WorldModelRegistry

support = WorldModelRegistry.describe_composable_support(
    "tdmpc2:ci",
    obs_shape=(4,),
    action_dim=2,
)
print(support["supports"])
```
