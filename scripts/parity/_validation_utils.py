"""Shared validation helpers for parity script argument parsing."""

from __future__ import annotations

from typing import Any


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object.")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{name} must be a boolean.")
    return bool(value)


def _require_float(value: Any, *, name: str) -> float:
    if not isinstance(value, int | float):
        raise RuntimeError(f"{name} must be numeric.")
    return float(value)


def _coerce_string_list(value: Any, *, name: str, non_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RuntimeError(f"{name} must be list[str].")
    out = tuple(v.strip() for v in value if v.strip())
    if non_empty and not out:
        raise RuntimeError(f"{name} must include at least one value.")
    return out
