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
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from worldflux.core.backend_handle import OfficialBackendHandle
from worldflux.execution import BackendExecutionRequest, ParityBackedExecutor


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


def _infer_family_from_model_id(model_id: str) -> str:
    normalized = str(model_id).strip().lower()
    if normalized.startswith("dreamer") or normalized.startswith("dreamerv3"):
        return "dreamer"
    if normalized.startswith("tdmpc2"):
        return "tdmpc2"
    return normalized.split(":", 1)[0] if ":" in normalized else normalized


def _env_to_task_filter(env: str) -> str:
    value = str(env).strip().lower()
    if not value:
        return ""
    if value.startswith("atari/"):
        game = value.split("/", 1)[1].strip().replace("-", "_")
        return f"atari100k_{game}" if game else ""
    if value.startswith("dmcontrol/"):
        return value.split("/", 1)[1].strip().replace("/", "-")
    if value.startswith("mujoco/"):
        return value.split("/", 1)[1].strip().replace("_", "-")
    return value


def _job_status_from_execution_status(status: str) -> JobStatus:
    normalized = str(status).strip().lower()
    if normalized in {"queued", "pending"}:
        return JobStatus.PENDING
    if normalized == "running":
        return JobStatus.RUNNING
    if normalized == "succeeded":
        return JobStatus.COMPLETED
    if normalized == "cancelled":
        return JobStatus.CANCELLED
    return JobStatus.FAILED


class ExecutionDelegatingBackend(TrainingBackend):
    """TrainingBackend implementation backed by worldflux.execution."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        scripts_root: Path | None = None,
        executor: ParityBackedExecutor | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.scripts_root = (scripts_root or (self.repo_root / "scripts" / "parity")).resolve()
        self.executor = executor or ParityBackedExecutor(
            repo_root=self.repo_root,
            scripts_root=self.scripts_root,
        )

    def submit(self, config: dict[str, Any]) -> JobHandle:
        model_handle = config.get("model_handle")
        if not isinstance(model_handle, OfficialBackendHandle):
            raise RuntimeError(
                "ExecutionDelegatingBackend requires model_handle=OfficialBackendHandle."
            )
        training_config = config.get("training_config")
        if training_config is None:
            raise RuntimeError("ExecutionDelegatingBackend requires training_config.")

        env = str(
            config.get("env")
            or model_handle.metadata.get("env")
            or model_handle.metadata.get("verify_env")
            or ""
        ).strip()
        task_filter = str(
            config.get("task_filter") or model_handle.metadata.get("task_filter") or ""
        ).strip()
        if not task_filter and env:
            task_filter = _env_to_task_filter(env)
        if not task_filter:
            raise RuntimeError(
                "Delegated backend requires task metadata. Provide env or task_filter on the backend handle."
            )

        family = str(
            config.get("family") or _infer_family_from_model_id(model_handle.model_id)
        ).strip()
        requested_device = str(
            config.get("device")
            or model_handle.metadata.get("requested_device")
            or training_config.resolve_device()
        ).strip()
        run_id = str(config.get("run_id") or "").strip() or f"train_{family}_{training_config.seed}"
        proof_requirements = dict(config.get("proof_requirements") or {})
        if config.get("resume_from"):
            proof_requirements["resume_from"] = str(config["resume_from"])

        request = BackendExecutionRequest(
            backend=model_handle.backend,
            family=family,
            mode="train",
            target=model_handle.model_id,
            baseline=None,
            task_filter=task_filter or None,
            env=env or None,
            seed_list=[int(training_config.seed)],
            device=requested_device,
            profile=str(
                config.get("backend_profile") or training_config.backend_profile or ""
            ).strip()
            or None,
            run_id=run_id,
            output_root=str(Path(training_config.output_dir).expanduser().resolve()),
            proof_requirements=proof_requirements,
        )
        result = self.executor.execute(request)
        metadata = {
            "status": result.status,
            "reason_code": result.reason_code,
            "message": result.message,
            "backend": request.backend,
            "backend_profile": request.profile or "",
            "model_id": model_handle.model_id,
            "manifest_path": result.manifest_path,
            "proof_phase": result.proof_phase,
            "family": request.family,
            "run_id": result.run_id or run_id,
            "summary_path": result.summary_path,
            "execution_result": result.to_dict(),
        }
        return JobHandle(
            job_id=str(metadata["run_id"]),
            backend=model_handle.backend,
            metadata=metadata,
        )

    def status(self, handle: JobHandle) -> JobStatus:
        return _job_status_from_execution_status(str(handle.metadata.get("status", "failed")))

    def logs(self, handle: JobHandle) -> Iterator[str]:
        summary_path = str(handle.metadata.get("summary_path", "")).strip()
        if not summary_path:
            return iter(())
        path = Path(summary_path)
        if not path.exists() or not path.is_file():
            return iter(())
        return iter(path.read_text(encoding="utf-8").splitlines())

    def cancel(self, handle: JobHandle) -> None:
        _ = handle
        return None
