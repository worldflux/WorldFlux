"""Parity campaign adapters."""

from __future__ import annotations

from .base import ParityCampaignAdapter
from .dreamerv3 import DreamerV3ParityAdapter
from .tdmpc2 import TDMPC2ParityAdapter

_REGISTRY: dict[str, ParityCampaignAdapter] = {
    "dreamerv3": DreamerV3ParityAdapter(),
    "tdmpc2": TDMPC2ParityAdapter(),
}


def resolve_adapter(family: str, explicit: str | None = None) -> ParityCampaignAdapter:
    """Resolve adapter by explicit key first, then family key, else identity adapter."""
    for key in (explicit, family):
        if key is None:
            continue
        normalized = key.strip().lower()
        adapter = _REGISTRY.get(normalized)
        if adapter is not None:
            return adapter
    return ParityCampaignAdapter(family=family)


__all__ = ["ParityCampaignAdapter", "resolve_adapter"]
