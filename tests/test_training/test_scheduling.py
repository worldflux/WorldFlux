"""Tests for reusable scheduling helpers."""

from __future__ import annotations

from worldflux.training import LocalClock, RatioUpdateScheduler


def test_local_clock_fires_on_interval_boundaries() -> None:
    clock = LocalClock(3)
    assert clock.consume(1) == 0
    assert clock.consume(2) == 0
    assert clock.consume(3) == 1
    assert clock.consume(6) == 1
    assert clock.consume(9) == 1


def test_ratio_update_scheduler_matches_expected_credit() -> None:
    scheduler = RatioUpdateScheduler(train_ratio=256, batch_size=16, batch_length=64)
    due = 0
    for _ in range(4):
        due += scheduler.on_env_step()
    assert due == 1
    assert scheduler.target_updates == 1
    assert 0.0 <= scheduler.credit < 1.0
