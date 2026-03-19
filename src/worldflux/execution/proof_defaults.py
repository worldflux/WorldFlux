# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Proof-mode canonical backend defaults."""

from __future__ import annotations

from worldflux.core.backend_bridge import canonical_backend_profile

_PROOF_CANONICAL_DEFAULTS: dict[str, tuple[str, str]] = {
    "dreamer": ("official_dreamerv3_jax_subprocess", "dreamerv3:official_xl"),
    "tdmpc2": ("official_tdmpc2_torch_subprocess", "tdmpc2:proof_5m"),
}


def resolve_proof_backend_defaults(
    family: str,
    *,
    backend: str | None,
    backend_profile: str | None,
) -> tuple[str, str]:
    normalized_family = str(family).strip().lower()
    normalized_backend = str(backend or "").strip()
    normalized_profile = str(backend_profile or "").strip()

    default = _PROOF_CANONICAL_DEFAULTS.get(normalized_family)
    if default is not None and not normalized_backend:
        default_backend, default_model = default
        return default_backend, canonical_backend_profile(default_model, default_backend)

    if normalized_backend and not normalized_profile:
        for _, (default_backend, default_model) in _PROOF_CANONICAL_DEFAULTS.items():
            if normalized_backend == default_backend:
                return normalized_backend, canonical_backend_profile(
                    default_model, normalized_backend
                )

    return normalized_backend or "native_torch", normalized_profile


__all__ = ["resolve_proof_backend_defaults"]
