# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Training infrastructure for WorldFlux."""

from __future__ import annotations

from .backend import ExecutionDelegatingBackend, JobHandle, JobStatus, LocalBackend, TrainingBackend
from .callbacks import (
    Callback,
    CheckpointCallback,
    DiagnosisCallback,
    HeartbeatCallback,
    LoggingCallback,
    TrainingReportCallback,
)
from .config import TrainingConfig
from .data import ReplayBuffer, TokenSequenceProvider, TrajectoryDataset, TransitionArrayProvider
from .distributed import build_launch_config
from .replay_backends import MemoryReplayBackend, ParquetReplayBackend, ReplayBackend
from .report import HealthSignal, LossCurveSummary, TrainingReport
from .scheduling import LocalClock, RatioUpdateScheduler
from .trainer import Trainer, train

__all__ = [
    # Main API
    "train",
    "Trainer",
    "TrainingConfig",
    "ReplayBuffer",
    "build_launch_config",
    "ReplayBackend",
    "MemoryReplayBackend",
    "ParquetReplayBackend",
    "TrajectoryDataset",
    "TokenSequenceProvider",
    "TransitionArrayProvider",
    "TrainingBackend",
    "ExecutionDelegatingBackend",
    "LocalBackend",
    "JobHandle",
    "JobStatus",
    # Callbacks
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "HeartbeatCallback",
    "DiagnosisCallback",
    "TrainingReportCallback",
    # Report
    "TrainingReport",
    "HealthSignal",
    "LossCurveSummary",
    "LocalClock",
    "RatioUpdateScheduler",
]
