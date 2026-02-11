"""Optional PyArrow Plasma adapter for State transport.

This module is import-safe without pyarrow installed. Users should guard usage via
`plasma_available()` or handle `ModuleNotFoundError`.
"""

from __future__ import annotations

from typing import Any

from .state import State


def _plasma_module():
    try:
        import pyarrow.plasma as plasma  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow[plasma] is required for Plasma-backed State transport. "
            "Install with: uv pip install pyarrow"
        ) from exc
    return plasma


def plasma_available() -> bool:
    """Return True when Plasma support is importable."""
    try:
        _plasma_module()
    except ModuleNotFoundError:
        return False
    return True


def connect(path: str, *, num_retries: int = 5) -> Any:
    """Connect to an existing Plasma store socket."""
    plasma = _plasma_module()
    return plasma.connect(path, num_retries=num_retries)


def put_state(client: Any, state: State) -> Any:
    """Store serialized State bytes in Plasma and return object id."""
    payload = state.serialize(version="v1", format="binary")
    return client.put(payload)


def get_state(client: Any, object_id: Any) -> State:
    """Fetch serialized State bytes from Plasma and deserialize to State."""
    payload = client.get(object_id)
    if payload is None:
        raise KeyError(f"Plasma object not found: {object_id!r}")

    if isinstance(payload, memoryview):
        data = payload.tobytes()
    elif isinstance(payload, bytearray):
        data = bytes(payload)
    elif isinstance(payload, bytes):
        data = payload
    else:
        raise TypeError(f"Unsupported Plasma payload type for State: {type(payload).__name__}")

    return State.deserialize(data)
