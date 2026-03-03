"""Async execution mixin for WorldModel."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ..output import ModelOutput
from ..state import State
from ..trajectory import Trajectory

if TYPE_CHECKING:
    from ..payloads import (
        ActionPayload,
        ActionSequence,
        ConditionPayload,
        WorldModelInput,
    )


class AsyncWorldModelMixin:
    """Mixin providing async variants of encode/transition/decode/rollout.

    Requires the host class to have:
    - ``observation_encoder``, ``dynamics_model``, ``decoder_module``,
      ``rollout_executor`` component slots (from :class:`ComponentHostMixin`).
    - ``io_contract()``, ``coerce_action_payload()``,
      ``coerce_condition_payload()``, ``_validate_action_payload()``,
      ``_validate_condition_payload()``, ``_validate_input_modalities()``,
      ``action_tensor_or_none()``, ``_coerce_world_input()``
      from the base :class:`WorldModel`.
    """

    @staticmethod
    async def _run_component_async(
        component: object,
        async_name: str,
        sync_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async_fn = getattr(component, async_name, None)
        if callable(async_fn):
            maybe_awaitable = async_fn(*args, **kwargs)
            if inspect.isawaitable(maybe_awaitable):
                return await maybe_awaitable
            return maybe_awaitable

        sync_fn = getattr(component, sync_name, None)
        if callable(sync_fn):
            return await asyncio.to_thread(sync_fn, *args, **kwargs)
        raise NotImplementedError(
            f"Component {type(component).__name__} must implement {async_name} or {sync_name}"
        )

    async def async_encode(
        self: Any,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        """Asynchronous non-blocking variant of ``encode``."""
        del deterministic
        encoder = getattr(self, "observation_encoder", None)
        if encoder is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.async_encode(...) requires observation_encoder"
            )
        world_input = self._coerce_world_input(obs)
        self._validate_input_modalities(world_input.observations)
        from ..interfaces import AsyncObservationEncoder

        if isinstance(encoder, AsyncObservationEncoder) or hasattr(encoder, "encode_async"):
            return await self._run_component_async(
                encoder, "encode_async", "encode", world_input.observations
            )
        return await asyncio.to_thread(encoder.encode, world_input.observations)

    async def async_transition(
        self: Any,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Asynchronous non-blocking variant of ``transition``."""
        dynamics = getattr(self, "dynamics_model", None)
        if dynamics is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.async_transition(...) requires dynamics_model"
            )

        contract = self.io_contract()
        action_spec = contract.action_spec
        action_payload = self.coerce_action_payload(action, kind=action_spec.kind)
        if action_payload is not None:
            self._validate_action_payload(action_payload)
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        conditioned: dict[str, Tensor] = {}

        ac = getattr(self, "action_conditioner", None)
        if ac is not None:
            conditioned.update(ac.condition(state, action_payload, condition_payload))

        action_tensor = self.action_tensor_or_none(action_payload, validate_contract=False)
        if action_tensor is not None and "action" not in conditioned:
            conditioned["action"] = action_tensor

        from ..interfaces import AsyncDynamicsModel

        if isinstance(dynamics, AsyncDynamicsModel) or hasattr(dynamics, "transition_async"):
            return await self._run_component_async(
                dynamics,
                "transition_async",
                "transition",
                state,
                conditioned,
                deterministic=deterministic,
            )
        return await asyncio.to_thread(dynamics.transition, state, conditioned, deterministic)

    async def async_decode(
        self: Any,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> ModelOutput:
        """Asynchronous non-blocking variant of ``decode``."""
        from ..exceptions import CapabilityError

        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        decoder = getattr(self, "decoder_module", None)
        if decoder is None:
            raise CapabilityError(f"{self.__class__.__name__} does not expose a decoder component")
        from ..interfaces import AsyncDecoder

        if isinstance(decoder, AsyncDecoder) or hasattr(decoder, "decode_async"):
            preds = await self._run_component_async(
                decoder,
                "decode_async",
                "decode",
                state,
                conditions=condition_payload,
            )
        else:
            preds = await asyncio.to_thread(decoder.decode, state, condition_payload)
        return ModelOutput(predictions=preds, state=state)

    async def async_rollout(
        self: Any,
        initial_state: State,
        action_sequence: ActionSequence | ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> Trajectory:
        """Asynchronous non-blocking variant of ``rollout``."""
        from ..payloads import ActionPayload as _ActionPayload
        from ..payloads import ActionSequence as _ActionSequence
        from ..payloads import normalize_planned_action

        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)

        api_version = self._get_api_version()

        if isinstance(action_sequence, _ActionPayload):
            action_sequence = normalize_planned_action(action_sequence, api_version=api_version)

        executor = getattr(self, "rollout_executor", None)
        if executor is not None:
            from ..interfaces import AsyncRolloutExecutor

            if isinstance(executor, AsyncRolloutExecutor) or hasattr(
                executor, "rollout_open_loop_async"
            ):
                return await self._run_component_async(
                    executor,
                    "rollout_open_loop_async",
                    "rollout_open_loop",
                    self,
                    initial_state,
                    action_sequence,
                    conditions=condition_payload,
                    deterministic=deterministic,
                )
            return await asyncio.to_thread(
                executor.rollout_open_loop,
                self,
                initial_state,
                action_sequence,
                condition_payload,
                deterministic,
            )

        if isinstance(action_sequence, Tensor):
            horizon = int(action_sequence.shape[0])
            actions_tensor: Tensor | None = action_sequence
        elif isinstance(action_sequence, _ActionSequence) and action_sequence.tensor is not None:
            horizon = int(action_sequence.tensor.shape[0])
            actions_tensor = action_sequence.tensor
        elif isinstance(action_sequence, _ActionSequence) and action_sequence.payloads is not None:
            horizon = len(action_sequence.payloads)
            actions_tensor = None
        else:
            horizon = 0
            actions_tensor = None

        states = [initial_state]
        rewards = []
        continues = []

        state = initial_state
        for t in range(horizon):
            action_t = self._action_from_sequence(action_sequence, t)
            state = await self.async_transition(
                state, action_t, conditions=condition_payload, deterministic=deterministic
            )
            states.append(state)
            decoded = await self.async_decode(state, conditions=condition_payload)
            if "reward" in decoded.predictions:
                rewards.append(decoded.predictions["reward"])
            if "continue" in decoded.predictions:
                continues.append(decoded.predictions["continue"])

        rewards_t = torch.stack(rewards, dim=0).squeeze(-1) if rewards else None
        continues_t = torch.stack(continues, dim=0).squeeze(-1) if continues else None

        if actions_tensor is None:
            batch = initial_state.batch_size
            actions_tensor = torch.zeros(horizon, batch, 0, device=initial_state.device)

        return Trajectory(
            states=states,
            actions=actions_tensor,
            rewards=rewards_t,
            continues=continues_t,
        )
