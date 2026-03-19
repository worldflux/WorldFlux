# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Internal bridge helpers for future multi-backend model resolution."""

from __future__ import annotations

from typing import Any

from .backend_handle import OfficialBackendHandle

NATIVE_TORCH_BACKEND = "native_torch"
_DREAMER_OFFICIAL_BACKEND = "official_dreamerv3_jax_subprocess"
_DREAMER_WORLDFLUX_JAX_BACKEND = "worldflux_dreamerv3_jax_subprocess"
_DREAMER_OFFICIAL_MODEL_ID = "dreamerv3:official_xl"
_TDMPC2_OFFICIAL_BACKEND = "official_tdmpc2_torch_subprocess"
_TDMPC2_OFFICIAL_MODEL_ID = "tdmpc2:proof_5m"


def normalize_backend_hint(backend_hint: str | None) -> str:
    normalized = str(backend_hint or "").strip()
    return normalized or NATIVE_TORCH_BACKEND


def canonical_backend_profile(model_id: str, backend_hint: str | None) -> str:
    normalized_backend = normalize_backend_hint(backend_hint)
    normalized_model = str(model_id).strip().lower()
    if normalized_backend in {_DREAMER_OFFICIAL_BACKEND, _DREAMER_WORLDFLUX_JAX_BACKEND}:
        return "official_xl"
    if normalized_backend == _TDMPC2_OFFICIAL_BACKEND:
        return "proof_5m"
    if normalized_backend == NATIVE_TORCH_BACKEND and normalized_model == _TDMPC2_OFFICIAL_MODEL_ID:
        return "proof_5m"
    return ""


def validate_backend_model_pair(model_id: str, backend_hint: str | None) -> None:
    normalized_backend = normalize_backend_hint(backend_hint)
    normalized_model = str(model_id).strip().lower()
    if (
        normalized_backend in {_DREAMER_OFFICIAL_BACKEND, _DREAMER_WORLDFLUX_JAX_BACKEND}
        and normalized_model != _DREAMER_OFFICIAL_MODEL_ID
    ):
        raise ValueError(f"{normalized_backend} requires model '{_DREAMER_OFFICIAL_MODEL_ID}'.")
    if (
        normalized_backend == _TDMPC2_OFFICIAL_BACKEND
        and normalized_model != _TDMPC2_OFFICIAL_MODEL_ID
    ):
        raise ValueError(f"{normalized_backend} requires model '{_TDMPC2_OFFICIAL_MODEL_ID}'.")


def resolve_backend_execution(model_id: str, backend_hint: str) -> str | OfficialBackendHandle:
    normalized = normalize_backend_hint(backend_hint)
    validate_backend_model_pair(model_id, normalized)
    if normalized == NATIVE_TORCH_BACKEND:
        return NATIVE_TORCH_BACKEND
    return OfficialBackendHandle(
        backend=normalized,
        model_id=str(model_id),
        metadata={
            "execution_kind": "official_backend",
            "adapter_id": normalized,
            "backend_profile": canonical_backend_profile(model_id, normalized),
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
    "canonical_backend_profile",
    "normalize_backend_hint",
    "resolve_backend_execution",
    "validate_backend_model_pair",
]
