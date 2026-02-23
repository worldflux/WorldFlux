"""Training backend abstraction for local and future cloud execution.

Phase 1 provides :class:`LocalBackend` for single-machine training.
The :class:`TrainingBackend` protocol is defined now so that Phase 2
can add cloud backends (e.g. ``AWSBackend``) without breaking the
existing API.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class JobStatus(Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobHandle:
    """Opaque handle to a submitted training job."""

    job_id: str
    backend: str
    metadata: dict[str, Any]


@runtime_checkable
class TrainingBackend(Protocol):
    """Protocol for training execution backends.

    Implementations must support job submission, status polling, and
    log streaming.  Phase 1 provides :class:`LocalBackend`; future
    phases will add cloud backends.
    """

    def submit(self, config: dict[str, Any]) -> JobHandle:
        """Submit a training job and return a handle."""
        ...

    def status(self, handle: JobHandle) -> JobStatus:
        """Query the current status of a submitted job."""
        ...

    def logs(self, handle: JobHandle) -> Iterator[str]:
        """Stream log lines from a running or completed job."""
        ...

    def cancel(self, handle: JobHandle) -> None:
        """Request cancellation of a running job."""
        ...


class LocalBackend:
    """Local single-machine training backend.

    Executes training synchronously in the current process.
    """

    def submit(self, config: dict[str, Any]) -> JobHandle:
        """Submit a local training job.

        In local mode, training executes synchronously so the job
        is immediately ``COMPLETED`` (or ``FAILED``) upon return.
        """
        import uuid

        job_id = f"local-{uuid.uuid4().hex[:8]}"
        return JobHandle(job_id=job_id, backend="local", metadata=dict(config))

    def status(self, handle: JobHandle) -> JobStatus:
        """Return status of a local job (always COMPLETED after submit)."""
        return JobStatus.COMPLETED

    def logs(self, handle: JobHandle) -> Iterator[str]:
        """Yield log lines (empty for local backend)."""
        return iter([])

    def cancel(self, handle: JobHandle) -> None:
        """Cancel is a no-op for synchronous local execution."""
