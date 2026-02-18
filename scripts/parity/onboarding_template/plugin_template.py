"""Template plugin for new parity family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NewFamilyPlugin:
    family: str = "<family>"

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
