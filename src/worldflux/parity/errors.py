"""Custom exceptions for parity harness."""

from __future__ import annotations


class ParityError(RuntimeError):
    """Raised when parity input or evaluation cannot be completed."""


class ParitySampleSizeError(ParityError):
    """Raised when paired parity samples are insufficient for configured statistics."""

    error_code = "WF_PARITY_SAMPLE_SIZE"

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.error_code}: {message}")
