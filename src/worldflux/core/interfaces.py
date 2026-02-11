"""Pluggable component interfaces for universal world model composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from .payloads import ActionPayload, ActionSequence, ConditionPayload
from .state import State


@dataclass(frozen=True)
class ComponentSpec:
    """Lightweight metadata for component registration and introspection."""

    name: str
    component_type: str
    version: str = "v0"
    config: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservationEncoder(Protocol):
    """Observation encoder abstraction."""

    def encode(self, observations: dict[str, Tensor]) -> State: ...


@runtime_checkable
class AsyncObservationEncoder(Protocol):
    """Asynchronous observation encoder abstraction."""

    async def encode_async(self, observations: dict[str, Tensor]) -> State: ...


@runtime_checkable
class ActionConditioner(Protocol):
    """Action conditioning abstraction used by dynamics/generators."""

    def condition(
        self,
        state: State,
        action: ActionPayload | None,
        conditions: ConditionPayload | None = None,
    ) -> dict[str, Tensor]: ...


@runtime_checkable
class DynamicsModel(Protocol):
    """Dynamics transition abstraction."""

    def transition(
        self,
        state: State,
        conditioned: dict[str, Tensor],
        deterministic: bool = False,
    ) -> State: ...


@runtime_checkable
class AsyncDynamicsModel(Protocol):
    """Asynchronous dynamics transition abstraction."""

    async def transition_async(
        self,
        state: State,
        conditioned: dict[str, Tensor],
        deterministic: bool = False,
    ) -> State: ...


@runtime_checkable
class Decoder(Protocol):
    """Optional decoder abstraction."""

    def decode(
        self, state: State, conditions: ConditionPayload | None = None
    ) -> dict[str, Tensor]: ...


@runtime_checkable
class AsyncDecoder(Protocol):
    """Asynchronous decoder abstraction."""

    async def decode_async(
        self,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> dict[str, Tensor]: ...


@runtime_checkable
class RolloutExecutor(Protocol):
    """Open-loop rollout execution abstraction."""

    def rollout_open_loop(
        self,
        model: Any,
        initial_state: State,
        action_sequence: ActionSequence | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ): ...


@runtime_checkable
class AsyncRolloutExecutor(Protocol):
    """Asynchronous open-loop rollout execution abstraction."""

    async def rollout_open_loop_async(
        self,
        model: Any,
        initial_state: State,
        action_sequence: ActionSequence | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ): ...


@runtime_checkable
class RolloutEngine(Protocol):
    """
    Deprecated compatibility alias for rollout executors.

    Kept in v0.2 for legacy integrations. Use ``RolloutExecutor``.
    """

    def rollout(
        self,
        model: Any,
        initial_state: State,
        action_sequence: Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
        mode: str = "autoregressive",
    ): ...
