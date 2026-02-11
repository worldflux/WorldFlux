"""Pluggable component interfaces for universal world model composition."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

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


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await cast(Awaitable[Any], value)
    return value


class AsyncObservationEncoderAdapter:
    """Adapter that exposes ``encode_async`` for sync-only encoders."""

    def __init__(self, encoder: ObservationEncoder | AsyncObservationEncoder):
        self._encoder = encoder

    async def encode_async(self, observations: dict[str, Tensor]) -> State:
        async_fn = getattr(self._encoder, "encode_async", None)
        if callable(async_fn):
            return cast(State, await _await_if_needed(async_fn(observations)))

        sync_fn = getattr(self._encoder, "encode", None)
        if not callable(sync_fn):
            raise TypeError(
                f"Encoder {type(self._encoder).__name__} must implement encode(...) or encode_async(...)"
            )
        return cast(State, await asyncio.to_thread(sync_fn, observations))


class AsyncDynamicsModelAdapter:
    """Adapter that exposes ``transition_async`` for sync-only dynamics models."""

    def __init__(self, dynamics: DynamicsModel | AsyncDynamicsModel):
        self._dynamics = dynamics

    async def transition_async(
        self,
        state: State,
        conditioned: dict[str, Tensor],
        deterministic: bool = False,
    ) -> State:
        async_fn = getattr(self._dynamics, "transition_async", None)
        if callable(async_fn):
            return cast(
                State,
                await _await_if_needed(async_fn(state, conditioned, deterministic=deterministic)),
            )

        sync_fn = getattr(self._dynamics, "transition", None)
        if not callable(sync_fn):
            raise TypeError(
                "Dynamics model "
                f"{type(self._dynamics).__name__} must implement transition(...) "
                "or transition_async(...)"
            )
        return cast(State, await asyncio.to_thread(sync_fn, state, conditioned, deterministic))


class AsyncDecoderAdapter:
    """Adapter that exposes ``decode_async`` for sync-only decoders."""

    def __init__(self, decoder: Decoder | AsyncDecoder):
        self._decoder = decoder

    async def decode_async(
        self,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> dict[str, Tensor]:
        async_fn = getattr(self._decoder, "decode_async", None)
        if callable(async_fn):
            return cast(dict[str, Tensor], await _await_if_needed(async_fn(state, conditions)))

        sync_fn = getattr(self._decoder, "decode", None)
        if not callable(sync_fn):
            raise TypeError(
                f"Decoder {type(self._decoder).__name__} must implement decode(...) or decode_async(...)"
            )
        return cast(dict[str, Tensor], await asyncio.to_thread(sync_fn, state, conditions))


class AsyncRolloutExecutorAdapter:
    """Adapter that exposes ``rollout_open_loop_async`` for sync-only executors."""

    def __init__(self, executor: RolloutExecutor | AsyncRolloutExecutor):
        self._executor = executor

    async def rollout_open_loop_async(
        self,
        model: Any,
        initial_state: State,
        action_sequence: ActionSequence | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ):
        async_fn = getattr(self._executor, "rollout_open_loop_async", None)
        if callable(async_fn):
            return await _await_if_needed(
                async_fn(
                    model,
                    initial_state,
                    action_sequence,
                    conditions=conditions,
                    deterministic=deterministic,
                )
            )

        sync_fn = getattr(self._executor, "rollout_open_loop", None)
        if not callable(sync_fn):
            raise TypeError(
                "Rollout executor "
                f"{type(self._executor).__name__} must implement rollout_open_loop(...) "
                "or rollout_open_loop_async(...)"
            )
        return await asyncio.to_thread(
            sync_fn,
            model,
            initial_state,
            action_sequence,
            conditions,
            deterministic,
        )


def ensure_async_observation_encoder(
    encoder: ObservationEncoder | AsyncObservationEncoder,
) -> AsyncObservationEncoder:
    """Return an async-capable observation encoder."""
    if isinstance(encoder, AsyncObservationEncoder):
        return encoder
    return AsyncObservationEncoderAdapter(encoder)


def ensure_async_dynamics_model(
    dynamics: DynamicsModel | AsyncDynamicsModel,
) -> AsyncDynamicsModel:
    """Return an async-capable dynamics model."""
    if isinstance(dynamics, AsyncDynamicsModel):
        return dynamics
    return AsyncDynamicsModelAdapter(dynamics)


def ensure_async_decoder(
    decoder: Decoder | AsyncDecoder,
) -> AsyncDecoder:
    """Return an async-capable decoder."""
    if isinstance(decoder, AsyncDecoder):
        return decoder
    return AsyncDecoderAdapter(decoder)


def ensure_async_rollout_executor(
    executor: RolloutExecutor | AsyncRolloutExecutor,
) -> AsyncRolloutExecutor:
    """Return an async-capable rollout executor."""
    if isinstance(executor, AsyncRolloutExecutor):
        return executor
    return AsyncRolloutExecutorAdapter(executor)
