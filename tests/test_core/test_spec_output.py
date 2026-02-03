"""Tests for core spec and output types."""

from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import (
    ActionSpec,
    ModalityKind,
    ModalitySpec,
    ObservationSpec,
    StateSpec,
    TokenSpec,
)


class TestSpecTypes:
    """Spec dataclass tests."""

    def test_modality_spec(self):
        spec = ModalitySpec(kind=ModalityKind.IMAGE, shape=(3, 64, 64), dtype="float32")
        assert spec.kind == ModalityKind.IMAGE
        assert spec.shape == (3, 64, 64)

    def test_observation_spec(self):
        mod = ModalitySpec(kind=ModalityKind.VECTOR, shape=(10,))
        obs_spec = ObservationSpec(modalities={"state": mod})
        assert obs_spec.modalities["state"] == mod

    def test_action_spec(self):
        action_spec = ActionSpec(kind="continuous", dim=6, discrete=False)
        assert action_spec.dim == 6
        assert not action_spec.discrete

    def test_state_spec(self):
        mod = ModalitySpec(kind=ModalityKind.VECTOR, shape=(128,))
        state_spec = StateSpec(tensors={"latent": mod})
        assert state_spec.tensors["latent"] == mod

    def test_token_spec(self):
        token_spec = TokenSpec(vocab_size=1024, seq_len=256)
        assert token_spec.vocab_size == 1024
        assert token_spec.seq_len == 256


class TestOutputTypes:
    """Output dataclass tests."""

    def test_model_output_defaults_are_isolated(self):
        out1 = ModelOutput()
        out2 = ModelOutput()
        out1.preds["x"] = 1
        assert "x" not in out2.preds

    def test_loss_output_defaults_are_isolated(self):
        import torch

        loss1 = LossOutput(loss=torch.tensor(0.0))
        loss2 = LossOutput(loss=torch.tensor(0.0))
        loss1.components["a"] = torch.tensor(1.0)
        assert "a" not in loss2.components
