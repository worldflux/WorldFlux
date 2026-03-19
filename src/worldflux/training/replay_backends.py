# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Replay backend abstractions for scaling training data paths."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from worldflux.core.batch import Batch

from .data import ReplayBuffer


@runtime_checkable
class ReplayBackend(Protocol):
    """Protocol shared by replay storage backends."""

    @property
    def capacity(self) -> int: ...

    @property
    def obs_shape(self) -> tuple[int, ...]: ...

    @property
    def action_dim(self) -> int: ...

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> Batch: ...


class MemoryReplayBackend:
    """Thin adapter over the existing in-memory ReplayBuffer."""

    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer

    @property
    def capacity(self) -> int:
        return self.buffer.capacity

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.buffer.obs_shape

    @property
    def action_dim(self) -> int:
        return self.buffer.action_dim

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> Batch:
        return self.buffer.sample(batch_size=batch_size, seq_len=seq_len, device=device)

    def to_parquet(self, path: str | Path, *, compression: str = "zstd") -> Path:
        output_path = Path(path)
        self.buffer.to_parquet(output_path, compression=compression)
        return output_path


class ParquetReplayBackend:
    """Replay backend reconstructed from a parquet export."""

    def __init__(self, buffer: ReplayBuffer, source_path: Path):
        self.buffer = buffer
        self.source_path = source_path

    @classmethod
    def from_parquet(cls, path: str | Path) -> ParquetReplayBackend:
        source_path = Path(path)
        buffer = ReplayBuffer.from_parquet(source_path)
        return cls(buffer=buffer, source_path=source_path)

    @property
    def capacity(self) -> int:
        return self.buffer.capacity

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.buffer.obs_shape

    @property
    def action_dim(self) -> int:
        return self.buffer.action_dim

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> Batch:
        return self.buffer.sample(batch_size=batch_size, seq_len=seq_len, device=device)
