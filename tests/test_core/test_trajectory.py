"""Tests for Trajectory container."""

import pytest
import torch

from worldflux.core.exceptions import StateError
from worldflux.core.state import State
from worldflux.core.trajectory import Trajectory


class TestTrajectory:
    """Trajectory unit tests."""

    def test_basic_properties(self):
        state0 = State(tensors={"latent": torch.randn(2, 8)})
        state1 = State(tensors={"latent": torch.randn(2, 8)})
        actions = torch.randn(1, 2, 3)
        rewards = torch.randn(1, 2)
        continues = torch.rand(1, 2)
        values = torch.randn(2, 2)

        traj = Trajectory(
            states=[state0, state1],
            actions=actions,
            rewards=rewards,
            continues=continues,
            values=values,
        )

        assert len(traj) == 2
        assert traj.horizon == 1
        assert traj.batch_size == 2
        stacked = traj.to_tensor("latent")
        assert stacked.shape == (2, 2, 8)

    def test_to_and_detach(self):
        state0 = State(tensors={"latent": torch.randn(2, 8, requires_grad=True)})
        state1 = State(tensors={"latent": torch.randn(2, 8, requires_grad=True)})
        actions = torch.randn(1, 2, 3, requires_grad=True)
        traj = Trajectory(states=[state0, state1], actions=actions)

        detached = traj.detach()
        assert not detached.states[0].tensors["latent"].requires_grad
        assert not detached.actions.requires_grad

        moved = traj.to(torch.device("cpu"))
        assert moved.states[0].tensors["latent"].device.type == "cpu"

    def test_state_action_length_mismatch(self):
        state0 = State(tensors={"latent": torch.randn(2, 8)})
        actions = torch.randn(2, 2, 3)
        with pytest.raises(StateError, match="Trajectory state/action mismatch"):
            Trajectory(states=[state0], actions=actions)

    def test_inconsistent_batch_size(self):
        state0 = State(tensors={"latent": torch.randn(2, 8)})
        state1 = State(tensors={"latent": torch.randn(3, 8)})
        actions = torch.randn(1, 2, 3)
        with pytest.raises(StateError, match="Inconsistent batch size"):
            Trajectory(states=[state0, state1], actions=actions)

    def test_rewards_length_mismatch(self):
        state0 = State(tensors={"latent": torch.randn(2, 8)})
        state1 = State(tensors={"latent": torch.randn(2, 8)})
        actions = torch.randn(1, 2, 3)
        rewards = torch.randn(2, 2)
        with pytest.raises(StateError, match="Rewards length"):
            Trajectory(states=[state0, state1], actions=actions, rewards=rewards)

    def test_continues_length_mismatch(self):
        state0 = State(tensors={"latent": torch.randn(2, 8)})
        state1 = State(tensors={"latent": torch.randn(2, 8)})
        actions = torch.randn(1, 2, 3)
        continues = torch.randn(2, 2)
        with pytest.raises(StateError, match="Continues length"):
            Trajectory(states=[state0, state1], actions=actions, continues=continues)
