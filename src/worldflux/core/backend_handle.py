# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Metadata handle for official backend-native execution in parity/proof flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OfficialBackendHandle:
    """Minimal metadata carrier for backend-native execution."""

    backend: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_native_torch(self) -> bool:
        return self.backend == "native_torch"

    @property
    def adapter_id(self) -> str | None:
        value = self.metadata.get("adapter_id")
        return str(value) if value is not None else None

    def with_metadata(self, **metadata: Any) -> OfficialBackendHandle:
        merged = dict(self.metadata)
        merged.update(metadata)
        return OfficialBackendHandle(
            backend=self.backend,
            model_id=self.model_id,
            metadata=merged,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }


__all__ = ["OfficialBackendHandle"]
