"""Integration tests for world models."""

import torch

from worldflux import AutoConfig, AutoWorldModel


class TestModelSwitching:
    """Test unified interface across models."""

    def test_both_models_same_interface(self):
        """Both models implement the same interface."""
        dreamer = AutoWorldModel.from_pretrained(
            "dreamerv3:size12m",
            obs_shape=(3, 64, 64),
            action_dim=6,
        )
        tdmpc = AutoWorldModel.from_pretrained(
            "tdmpc2:5m",
            obs_shape=(39,),
            action_dim=6,
        )

        for model in [dreamer, tdmpc]:
            assert hasattr(model, "encode")
            assert hasattr(model, "predict")
            assert hasattr(model, "observe")
            assert hasattr(model, "decode")
            assert hasattr(model, "imagine")
            assert hasattr(model, "initial_state")
            assert hasattr(model, "compute_loss")

    def test_dreamer_encode_predict_cycle(self):
        """DreamerV3 encode/predict cycle."""
        model = AutoWorldModel.from_pretrained(
            "dreamerv3:size12m",
            obs_shape=(3, 64, 64),
            action_dim=6,
            deter_dim=256,
            stoch_discrete=8,
            stoch_classes=8,
            hidden_dim=128,
            cnn_depth=16,
        )

        obs = torch.randn(2, 3, 64, 64)
        actions = torch.randn(5, 2, 6)

        state = model.encode(obs)
        trajectory = model.imagine(state, actions)

        assert len(trajectory.states) == 6
        assert trajectory.rewards is not None

    def test_tdmpc_encode_predict_cycle(self):
        """TD-MPC2 encode/predict cycle."""
        model = AutoWorldModel.from_pretrained(
            "tdmpc2:5m",
            obs_shape=(39,),
            action_dim=6,
        )

        obs = torch.randn(2, 39)
        actions = torch.randn(5, 2, 6)

        state = model.encode(obs)
        trajectory = model.imagine(state, actions)

        assert len(trajectory.states) == 6
        assert trajectory.rewards is not None


class TestAutoConfig:
    """Test AutoConfig functionality."""

    def test_dreamer_config(self):
        config = AutoConfig.from_pretrained("dreamerv3:size12m")
        assert config.model_type == "dreamer"
        assert config.model_name == "size12m"

    def test_tdmpc_config(self):
        config = AutoConfig.from_pretrained("tdmpc2:5m")
        assert config.model_type == "tdmpc2"
        assert config.model_name == "5m"


class TestSaveLoad:
    """Test model save/load functionality."""

    def test_dreamer_save_load(self, tmp_path):
        """Save and load DreamerV3 model."""
        model = AutoWorldModel.from_pretrained(
            "dreamerv3:size12m",
            deter_dim=256,
            stoch_discrete=8,
            stoch_classes=8,
            hidden_dim=128,
            cnn_depth=16,
        )

        save_path = str(tmp_path / "dreamer_test")
        model.save_pretrained(save_path)

        loaded = AutoWorldModel.from_pretrained(save_path)

        # Should have same config
        assert loaded.config.model_type == model.config.model_type
        assert loaded.config.deter_dim == model.config.deter_dim

    def test_tdmpc_save_load(self, tmp_path):
        """Save and load TD-MPC2 model."""
        model = AutoWorldModel.from_pretrained(
            "tdmpc2:5m",
            obs_shape=(39,),
        )

        save_path = str(tmp_path / "tdmpc_test")
        model.save_pretrained(save_path)

        loaded = AutoWorldModel.from_pretrained(save_path)

        assert loaded.config.model_type == model.config.model_type
        assert loaded.config.latent_dim == model.config.latent_dim
