"""DreamerV3 adapter for parity campaign task normalization."""

from __future__ import annotations

from .base import ParityCampaignAdapter


class DreamerV3ParityAdapter(ParityCampaignAdapter):
    """Task conversion helper for DreamerV3 campaign integration."""

    def __init__(self) -> None:
        super().__init__(family="dreamerv3")

    def to_adapter_task(self, task: str) -> str:
        # Dreamer configs often use `atari100k_<game>` while suite keys are `atari_<game>`.
        if task.startswith("atari_"):
            return "atari100k_" + task[len("atari_") :]
        return task

    def to_canonical_task(self, task: str) -> str:
        if task.startswith("atari100k_"):
            return "atari_" + task[len("atari100k_") :]
        return task

    def default_result_format(self, source_name: str) -> str | None:
        if source_name == "oracle":
            return "dreamerv3_scores_json_gz"
        return "canonical_json"
