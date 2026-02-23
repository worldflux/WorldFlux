"""Cloud training backend adapters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from worldflux.training.backend import JobHandle, JobStatus, TrainingBackend

from .client import WorldFluxCloudClient


def _status_from_payload(value: str) -> JobStatus:
    normalized = str(value).strip().lower()
    if normalized in {"pending", "queued"}:
        return JobStatus.PENDING
    if normalized in {"running", "in_progress"}:
        return JobStatus.RUNNING
    if normalized in {"completed", "succeeded", "success"}:
        return JobStatus.COMPLETED
    if normalized in {"cancelled", "canceled"}:
        return JobStatus.CANCELLED
    return JobStatus.FAILED


class ModalBackend(TrainingBackend):
    """TrainingBackend implementation backed by WorldFlux cloud API."""

    def __init__(self, client: WorldFluxCloudClient):
        self.client = client

    def submit(self, config: dict[str, Any]) -> JobHandle:
        payload = self.client.create_training_job(config)
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise RuntimeError("Cloud training submit succeeded but returned no job_id.")
        return JobHandle(job_id=job_id, backend="modal", metadata=payload)

    def status(self, handle: JobHandle) -> JobStatus:
        payload = self.client.request_json("GET", f"/v1/jobs/{handle.job_id}")
        return _status_from_payload(str(payload.get("status", "failed")))

    def logs(self, handle: JobHandle) -> Iterator[str]:
        return iter(self.client.get_job_logs(handle.job_id))

    def cancel(self, handle: JobHandle) -> None:
        self.client.cancel_job(handle.job_id)
