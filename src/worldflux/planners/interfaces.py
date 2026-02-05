"""Planner interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.payloads import ActionPayload, ConditionPayload
from ..core.state import State


@runtime_checkable
class Planner(Protocol):
    """Planning strategy interface."""

    def plan(
        self,
        model,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> ActionPayload: ...
