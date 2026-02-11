"""Tests for versioned State serialization and shared-memory transport."""

from __future__ import annotations

import pytest
import torch

from worldflux.core.exceptions import StateError
from worldflux.core.state import State


def test_state_binary_serialize_roundtrip() -> None:
    state = State(
        tensors={
            "latent": torch.randn(3, 4, dtype=torch.float32),
            "indices": torch.arange(12, dtype=torch.int64).reshape(3, 4),
        },
        meta={"scenario": "unit-test", "step": 7},
    )

    payload = state.serialize(version="v1", format="binary")
    restored = State.deserialize(payload)

    assert restored.meta == state.meta
    assert set(restored.tensors.keys()) == {"latent", "indices"}
    assert torch.allclose(restored.tensors["latent"], state.tensors["latent"])
    assert torch.equal(restored.tensors["indices"], state.tensors["indices"])


def test_state_deserialize_rejects_corrupted_magic() -> None:
    payload = bytearray(State(tensors={"x": torch.zeros(1, 2)}).serialize())
    payload[0:4] = b"BAD!"
    with pytest.raises(StateError, match="magic"):
        State.deserialize(bytes(payload))


def test_state_deserialize_rejects_unknown_version() -> None:
    payload = bytearray(State(tensors={"x": torch.zeros(1, 2)}).serialize())
    payload[4] = 99
    with pytest.raises(StateError, match="version"):
        State.deserialize(bytes(payload))


def test_state_deserialize_rejects_truncated_payload() -> None:
    payload = State(tensors={"x": torch.zeros(1, 2)}).serialize()
    with pytest.raises(StateError, match="truncated"):
        State.deserialize(payload[:-1])


def test_shared_memory_roundtrip_has_zero_copy_semantics() -> None:
    state = State(tensors={"latent": torch.arange(6, dtype=torch.float32).reshape(2, 3)})
    descriptor = state.to_shared_memory()
    s1 = State.from_shared_memory(descriptor, copy=False)
    s2 = State.from_shared_memory(descriptor, copy=False)

    try:
        s1.tensors["latent"][0, 0] = 123.0
        assert torch.isclose(s2.tensors["latent"][0, 0], torch.tensor(123.0))
    finally:
        s1.close_shared_memory()
        s2.close_shared_memory(unlink=True)


def test_shared_memory_copy_mode_breaks_aliasing() -> None:
    state = State(tensors={"latent": torch.arange(6, dtype=torch.float32).reshape(2, 3)})
    descriptor = state.to_shared_memory()
    aliased = State.from_shared_memory(descriptor, copy=False)
    copied = State.from_shared_memory(descriptor, copy=True)

    try:
        aliased.tensors["latent"][0, 1] = 77.0
        assert not torch.isclose(copied.tensors["latent"][0, 1], torch.tensor(77.0))
    finally:
        aliased.close_shared_memory(unlink=True)


def test_to_shared_memory_rejects_cuda_without_explicit_copy() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = State(tensors={"latent": torch.ones(2, 2, device="cuda")})
    with pytest.raises(StateError, match="allow_copy_from_cuda=True"):
        state.to_shared_memory()
