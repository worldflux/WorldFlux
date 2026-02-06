"""Tests for Batch container."""

import pytest
import torch

from worldflux.core.batch import Batch
from worldflux.core.spec import SequenceFieldSpec


class TestBatch:
    """Batch unit tests."""

    def test_batch_size_tensor(self):
        batch = Batch(obs=torch.randn(3, 4))
        assert batch.batch_size == 3

    def test_batch_size_dict(self):
        batch = Batch(obs={"image": torch.randn(5, 3, 8, 8)})
        assert batch.batch_size == 5

    def test_batch_size_empty_dict_raises(self):
        batch = Batch(obs={})
        with pytest.raises(ValueError, match="Cannot determine batch size"):
            _ = batch.batch_size

    def test_to_device(self):
        obs = torch.randn(2, 4)
        batch = Batch(obs=obs, actions=torch.randn(2, 3))
        moved = batch.to("cpu")
        assert moved.obs.device.type == "cpu"
        assert moved.actions.device.type == "cpu"

    def test_detach_removes_grad(self):
        obs = torch.randn(2, 4, requires_grad=True)
        batch = Batch(obs=obs)
        detached = batch.detach()
        assert not detached.obs.requires_grad

    def test_clone_independent(self):
        obs = torch.randn(2, 4)
        batch = Batch(obs=obs, actions=torch.randn(2, 3))
        cloned = batch.clone()
        batch.obs.fill_(0)
        assert not torch.allclose(cloned.obs, batch.obs)

    def test_to_dict_roundtrip(self):
        batch = Batch(
            obs=torch.randn(2, 4),
            actions=torch.randn(2, 3),
            rewards=torch.randn(2),
            extras={"id": "test"},
        )
        d = batch.to_dict()
        reconstructed = Batch.from_dict(d)
        assert torch.allclose(reconstructed.obs, batch.obs)
        assert torch.allclose(reconstructed.actions, batch.actions)
        assert torch.allclose(reconstructed.rewards, batch.rewards)
        assert reconstructed.extras["id"] == "test"

    def test_nested_dict_mapping(self):
        batch = Batch(obs={"a": {"b": torch.randn(2, 4)}})
        moved = batch.to("cpu")
        assert moved.obs["a"]["b"].device.type == "cpu"

    def test_validate_passes(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            actions=torch.randn(3, 5, 2),
            rewards=torch.randn(3, 5),
        )
        batch.validate(strict_time=True)

    def test_validate_batch_mismatch_raises(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            actions=torch.randn(2, 5, 2),
        )
        with pytest.raises(Exception, match="batch size mismatch"):
            batch.validate()

    def test_validate_time_mismatch_raises(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            actions=torch.randn(3, 4, 2),
        )
        with pytest.raises(Exception, match="time dimension mismatch"):
            batch.validate(strict_time=True)

    def test_validate_with_explicit_layout_for_tokens(self):
        batch = Batch(
            obs=torch.randint(0, 100, (3, 5)),
            target=torch.randint(0, 100, (3, 5)),
            layouts={"obs": "BT", "target": "BT"},
            strict_layout=True,
        )
        batch.validate(strict_time=True)

    def test_validate_layout_mismatch_raises(self):
        batch = Batch(
            obs=torch.randint(0, 100, (3, 5)),
            target=torch.randint(0, 100, (3, 4)),
            layouts={"obs": "BT", "target": "BT"},
            strict_layout=True,
        )
        with pytest.raises(Exception, match="time dimension mismatch"):
            batch.validate(strict_time=True)

    def test_with_layouts_merges_keys(self):
        batch = Batch(obs=torch.randn(2, 3, 4), layouts={"obs": "BT..."})
        updated = batch.with_layouts({"actions": "BT..."}, strict=True)
        assert updated.layouts["obs"] == "BT..."
        assert updated.layouts["actions"] == "BT..."
        assert updated.strict_layout is True

    def test_validate_strict_layout_rejects_unknown_layout_key(self):
        batch = Batch(
            obs=torch.randn(2, 3, 4),
            layouts={"imaginary": "BT..."},
            strict_layout=True,
        )
        with pytest.raises(Exception, match="Unknown layout field"):
            batch.validate(strict_time=True)

    def test_validate_allows_variable_length_when_lengths_present(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            actions=torch.randn(3, 4, 2),
            lengths={"actions": torch.tensor([4, 4, 4])},
        )
        batch.validate(strict_time=True)

    def test_validate_allows_variable_length_from_sequence_field_spec(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            actions=torch.randn(3, 4, 2),
            layouts={"obs": "BT...", "actions": "BT..."},
            strict_layout=True,
        )
        batch.validate(
            strict_time=True,
            sequence_field_spec={
                "actions": SequenceFieldSpec(layout="BT...", variable_length=True)
            },
        )

    def test_validate_rejects_invalid_lengths_batch_dimension(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            lengths={"obs": torch.tensor([5, 5])},
        )
        with pytest.raises(Exception, match="lengths\\['obs'\\] batch size mismatch"):
            batch.validate(strict_time=True)

    def test_validate_rejects_invalid_masks_batch_dimension(self):
        batch = Batch(
            obs=torch.randn(3, 5, 4),
            masks={"obs": torch.ones(2, 5)},
        )
        with pytest.raises(Exception, match="masks\\['obs'\\] batch size mismatch"):
            batch.validate(strict_time=True)
