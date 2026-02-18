#!/usr/bin/env python3
"""Family plugin registry for parity suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class FamilyPlugin(Protocol):
    """Contract for family-specific parity behavior."""

    family: str

    def prepare_run(self, *, task: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]: ...

    def normalize_task(self, task_id: str) -> str: ...

    def collect_metrics(self, metrics: dict[str, Any]) -> dict[str, float]: ...

    def validate_runtime(self, *, record: dict[str, Any]) -> list[str]: ...


@dataclass
class BaseFamilyPlugin:
    """Default family plugin with no-op normalization and metric passthrough."""

    family: str

    def prepare_run(self, *, task: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        _ = task
        _ = context
        return {}

    def normalize_task(self, task_id: str) -> str:
        return str(task_id).strip()

    def collect_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in ("final_return_mean", "auc_return"):
            value = metrics.get(key)
            if isinstance(value, int | float):
                out[key] = float(value)
        return out

    def validate_runtime(self, *, record: dict[str, Any]) -> list[str]:
        _ = record
        return []


class DreamerV3FamilyPlugin(BaseFamilyPlugin):
    family = "dreamerv3"

    def __init__(self) -> None:
        super().__init__(family="dreamerv3")

    def normalize_task(self, task_id: str) -> str:
        task = str(task_id).strip().lower()
        if task.startswith("atari_"):
            return "atari100k_" + task[len("atari_") :]
        return task


class TDMPC2FamilyPlugin(BaseFamilyPlugin):
    family = "tdmpc2"

    def __init__(self) -> None:
        super().__init__(family="tdmpc2")

    def normalize_task(self, task_id: str) -> str:
        return str(task_id).strip().lower().replace("_", "-")


class FamilyPluginRegistry:
    """Registry of family plugins for parity orchestration/statistics."""

    def __init__(self) -> None:
        self._plugins: dict[str, FamilyPlugin] = {}

    def register(self, plugin: FamilyPlugin) -> None:
        family = str(plugin.family).strip().lower()
        if not family:
            raise ValueError("plugin.family must be a non-empty string")
        self._plugins[family] = plugin

    def get(self, family: str) -> FamilyPlugin | None:
        return self._plugins.get(str(family).strip().lower())

    def require(self, family: str) -> FamilyPlugin:
        normalized = str(family).strip().lower()
        plugin = self.get(normalized)
        if plugin is None:
            raise KeyError(
                f"No plugin registered for family '{family}'. Known families: {sorted(self._plugins)}"
            )
        return plugin

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


def build_default_registry() -> FamilyPluginRegistry:
    registry = FamilyPluginRegistry()
    registry.register(DreamerV3FamilyPlugin())
    registry.register(TDMPC2FamilyPlugin())
    return registry


__all__ = [
    "BaseFamilyPlugin",
    "DreamerV3FamilyPlugin",
    "FamilyPlugin",
    "FamilyPluginRegistry",
    "TDMPC2FamilyPlugin",
    "build_default_registry",
]
