"""Reusable scheduling helpers for recipe-driven training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalClock:
    """Step-based clock that fires when local step crosses interval boundaries."""

    interval: int
    _next_fire: int = 0

    def __post_init__(self) -> None:
        self.interval = max(1, int(self.interval))
        self._next_fire = self.interval

    def ready(self, step: int) -> bool:
        return int(step) >= int(self._next_fire)

    def consume(self, step: int) -> int:
        fired = 0
        current = int(step)
        while current >= self._next_fire:
            fired += 1
            self._next_fire += self.interval
        return fired


@dataclass
class RatioUpdateScheduler:
    """Track update credit from env steps using official train_ratio semantics."""

    train_ratio: float
    batch_size: int
    batch_length: int
    credit: float = 0.0
    target_updates: int = 0

    def __post_init__(self) -> None:
        self.train_ratio = max(0.0, float(self.train_ratio))
        self.batch_size = max(1, int(self.batch_size))
        self.batch_length = max(1, int(self.batch_length))

    @property
    def updates_per_env_step(self) -> float:
        return float(self.train_ratio) / float(self.batch_size * self.batch_length)

    def on_env_step(self, env_steps: int = 1) -> int:
        self.credit += float(env_steps) * self.updates_per_env_step
        due = int(self.credit)
        if due > 0:
            self.credit -= float(due)
            self.target_updates += due
        return due


__all__ = ["LocalClock", "RatioUpdateScheduler"]
