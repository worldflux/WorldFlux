"""Shared formatting utilities for parity reporting."""

from __future__ import annotations

from typing import Any


def fmt_bool(value: Any) -> str:
    """Format a boolean value as PASS/FAIL."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return "-"


def fmt_float(value: Any, digits: int = 4) -> str:
    """Format a numeric value with fixed decimal places."""
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "-"
