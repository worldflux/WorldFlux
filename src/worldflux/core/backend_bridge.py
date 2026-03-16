"""Internal bridge helpers for future multi-backend model resolution."""

from __future__ import annotations

from typing import Any

from .backend_handle import OfficialBackendHandle

NATIVE_TORCH_BACKEND = "native_torch"


def normalize_backend_hint(backend_hint: str | None) -> str:
    normalized = str(backend_hint or "").strip()
    return normalized or NATIVE_TORCH_BACKEND


def resolve_backend_execution(model_id: str, backend_hint: str) -> str | OfficialBackendHandle:
    normalized = normalize_backend_hint(backend_hint)
    if normalized == NATIVE_TORCH_BACKEND:
        return NATIVE_TORCH_BACKEND
    return OfficialBackendHandle(
        backend=normalized,
        model_id=str(model_id),
        metadata={
            "execution_kind": "official_backend",
            "adapter_id": normalized,
        },
    )


def backend_name_for(value: str | OfficialBackendHandle | Any) -> str:
    if isinstance(value, OfficialBackendHandle):
        return value.backend
    if isinstance(value, str):
        return normalize_backend_hint(value)
    return NATIVE_TORCH_BACKEND


__all__ = [
    "NATIVE_TORCH_BACKEND",
    "backend_name_for",
    "normalize_backend_hint",
    "resolve_backend_execution",
]
