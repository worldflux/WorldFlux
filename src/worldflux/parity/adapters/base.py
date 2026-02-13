"""Base adapter utilities for parity campaign execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ParityCampaignAdapter:
    """Family adapter for campaign task naming normalization."""

    family: str

    def to_adapter_task(self, task: str) -> str:
        """Convert canonical suite task to adapter-specific task string."""
        return task

    def to_canonical_task(self, task: str) -> str:
        """Convert adapter-specific task string back to canonical suite task."""
        return task

    def default_result_format(self, source_name: str) -> str | None:
        """Return default artifact format for command-generated results."""
        return None


FAMILY_DREAMERV3: Final[str] = "dreamerv3"
FAMILY_TDMPC2: Final[str] = "tdmpc2"
