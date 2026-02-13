"""TD-MPC2 adapter for parity campaign task normalization."""

from __future__ import annotations

from .base import ParityCampaignAdapter


class TDMPC2ParityAdapter(ParityCampaignAdapter):
    """TD-MPC2 keeps canonical task naming as-is."""

    def __init__(self) -> None:
        super().__init__(family="tdmpc2")

    def default_result_format(self, source_name: str) -> str | None:
        if source_name == "oracle":
            return "tdmpc2_results_csv_dir"
        return "canonical_json"
