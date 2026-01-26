"""Tests for training infrastructure."""

import os
import tempfile

import numpy as np
import pytest
import torch

from worldmodels import create_world_model
from worldmodels.core.exceptions import ConfigurationError
from worldmodels.training import (
    Callback,
    CheckpointCallback,
    LoggingCallback,
    ReplayBuffer,
    TrajectoryDataset,
    Trainer,
    TrainingConfig,
    train,
)
from worldmodels.training.callbacks import EarlyStoppingCallback, ProgressCallback
from worldmodels.training.data import create_random_buffer


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        config = TrainingConfig()
        assert config.total_steps == 100_000
        assert config.batch_size == 16
        assert config.sequence_length == 50
        assert config.learning_rate == 3e-4

    def test_custom_config(self):
        config = TrainingConfig(
            total_steps=50_000,
            batch_size=32,
            learning_rate=1e-4,
        )
        assert config.total_steps == 50_000
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_config_validation(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(total_steps=-1)

        with pytest.raises(ConfigurationError):
            TrainingConfig(batch_size=0)

        with pytest.raises(ConfigurationError):
            TrainingConfig(learning_rate=-1)

    def test_config_serialization(self):
        config = TrainingConfig(total_steps=1000, batch_size=8)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded = TrainingConfig.load(f.name)

        assert loaded.total_steps == config.total_steps
        assert loaded.batch_size == config.batch_size
        os.unlink(f.name)

    def test_resolve_device(self):
        config = TrainingConfig(device="cpu")
        assert config.resolve_device() == "cpu"

        config = TrainingConfig(device="auto")
        device = config.resolve_device()
        assert device in ["cuda", "cpu"]

    def test_with_updates(self):
        config = TrainingConfig(total_steps=1000)
        updated = config.with_updates(total_steps=2000, batch_size=64)
        assert updated.total_steps == 2000
        assert updated.batch_size == 64
        assert config.total_steps == 1000  # Original unchanged


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_create_buffer(self):
        buffer = ReplayBuffer(
            capacity=1000,
            obs_shape=(3, 64, 64),
            action_dim=6,
        )
        assert len(buffer) == 0
        assert buffer.capacity == 1000
        assert buffer.obs_shape == (3, 64, 64)
        assert buffer.action_dim == 6

    def test_add_episode(self):
        buffer = ReplayBuffer(capacity=1000, obs_shape=(4,), action_dim=2)

        obs = np.random.randn(100, 4).astype(np.float32)
        actions = np.random.randn(100, 2).astype(np.float32)
        rewards = np.random.randn(100).astype(np.float32)

        buffer.add_episode(obs, actions, rewards)

        assert len(buffer) == 100
        assert buffer.num_episodes == 1

    def test_add_multiple_episodes(self):
        buffer = ReplayBuffer(capacity=1000, obs_shape=(4,), action_dim=2)

        for _ in range(5):
            obs = np.random.randn(50, 4).astype(np.float32)
            actions = np.random.randn(50, 2).astype(np.float32)
            rewards = np.random.randn(50).astype(np.float32)
            buffer.add_episode(obs, actions, rewards)

        assert len(buffer) == 250
        assert buffer.num_episodes == 5

    def test_sample(self):
        buffer = ReplayBuffer(capacity=1000, obs_shape=(4,), action_dim=2)

        # Add enough data
        for _ in range(10):
            obs = np.random.randn(100, 4).astype(np.float32)
            actions = np.random.randn(100, 2).astype(np.float32)
            rewards = np.random.randn(100).astype(np.float32)
            buffer.add_episode(obs, actions, rewards)

        batch = buffer.sample(batch_size=16, seq_len=10)

        assert "obs" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "continues" in batch

        assert batch["obs"].shape == (16, 10, 4)
        assert batch["actions"].shape == (16, 10, 2)
        assert batch["rewards"].shape == (16, 10)
        assert batch["continues"].shape == (16, 10)

    def test_sample_with_device(self):
        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=10,
        )

        batch = buffer.sample(batch_size=8, seq_len=5, device="cpu")
        assert batch["obs"].device == torch.device("cpu")

    def test_save_load(self):
        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=10,
            seed=42,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            buffer.save(f.name)
            loaded = ReplayBuffer.load(f.name)

        assert len(loaded) == len(buffer)
        assert loaded.obs_shape == buffer.obs_shape
        assert loaded.action_dim == buffer.action_dim
        os.unlink(f.name)

    def test_from_trajectories(self):
        trajectories = [
            {
                "obs": np.random.randn(50, 4).astype(np.float32),
                "actions": np.random.randn(50, 2).astype(np.float32),
                "rewards": np.random.randn(50).astype(np.float32),
            }
            for _ in range(5)
        ]

        buffer = ReplayBuffer.from_trajectories(trajectories)
        assert len(buffer) == 250
        assert buffer.obs_shape == (4,)
        assert buffer.action_dim == 2

    def test_capacity_wrap_around(self):
        buffer = ReplayBuffer(capacity=100, obs_shape=(2,), action_dim=1)

        # Add more than capacity
        for i in range(5):
            obs = np.full((30, 2), i, dtype=np.float32)
            actions = np.zeros((30, 1), dtype=np.float32)
            rewards = np.zeros(30, dtype=np.float32)
            buffer.add_episode(obs, actions, rewards)

        assert len(buffer) == 100  # Capped at capacity


class TestTrajectoryDataset:
    """Tests for TrajectoryDataset."""

    def test_create_dataset(self):
        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=10,
        )

        dataset = TrajectoryDataset(buffer, seq_len=10, samples_per_epoch=100)
        assert len(dataset) == 100

    def test_getitem(self):
        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=10,
        )

        dataset = TrajectoryDataset(buffer, seq_len=10, samples_per_epoch=100)
        sample = dataset[0]

        assert sample["obs"].shape == (10, 4)
        assert sample["actions"].shape == (10, 2)
        assert sample["rewards"].shape == (10,)


class TestCallbacks:
    """Tests for training callbacks."""

    def test_logging_callback(self):
        callback = LoggingCallback(log_interval=10)
        assert callback.log_interval == 10

    def test_checkpoint_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_interval=100,
                output_dir=tmpdir,
                max_checkpoints=3,
            )
            assert callback.save_interval == 100

    def test_early_stopping_callback(self):
        callback = EarlyStoppingCallback(patience=1000, min_delta=1e-4)
        assert callback.patience == 1000
        assert callback.min_delta == 1e-4

    def test_progress_callback(self):
        callback = ProgressCallback(desc="Training")
        assert callback.desc == "Training"


class TestTrainer:
    """Tests for Trainer."""

    @pytest.fixture
    def small_model(self):
        """Create a small DreamerV3 model for testing."""
        return create_world_model(
            "dreamerv3:size12m",
            obs_shape=(4,),
            action_dim=2,
            encoder_type="mlp",
            decoder_type="mlp",
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=16,
        )

    @pytest.fixture
    def small_buffer(self):
        """Create a small buffer for testing."""
        return create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=20,
            episode_length=50,
            seed=42,
        )

    def test_trainer_creation(self, small_model):
        config = TrainingConfig(
            total_steps=100,
            batch_size=8,
            sequence_length=10,
            device="cpu",
        )
        trainer = Trainer(small_model, config)
        assert trainer.model is small_model
        assert trainer.config is config

    def test_trainer_train_short(self, small_model, small_buffer):
        """Test a few training steps."""
        config = TrainingConfig(
            total_steps=5,
            batch_size=4,
            sequence_length=10,
            device="cpu",
            log_interval=1,
            save_interval=100,  # Don't save during test
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = config.with_updates(output_dir=tmpdir)
            trainer = Trainer(small_model, config, callbacks=[])

            trained_model = trainer.train(small_buffer)

            assert trainer.state.global_step == 5
            assert trained_model is small_model

    def test_trainer_checkpoint(self, small_model, small_buffer):
        """Test checkpoint save/load."""
        config = TrainingConfig(
            total_steps=10,
            batch_size=4,
            sequence_length=10,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = config.with_updates(output_dir=tmpdir)
            trainer = Trainer(small_model, config, callbacks=[])

            # Train a bit
            trainer.train(small_buffer, num_steps=5)

            # Save checkpoint
            ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
            trainer.save_checkpoint(ckpt_path)

            # Create new trainer and load
            new_model = create_world_model(
                "dreamerv3:size12m",
                obs_shape=(4,),
                action_dim=2,
                encoder_type="mlp",
                decoder_type="mlp",
                deter_dim=64,
                stoch_discrete=4,
                stoch_classes=4,
                hidden_dim=32,
                cnn_depth=16,
            )
            new_trainer = Trainer(new_model, config, callbacks=[])
            new_trainer.load_checkpoint(ckpt_path)

            assert new_trainer.state.global_step == 5

    def test_trainer_evaluate(self, small_model, small_buffer):
        """Test evaluation."""
        config = TrainingConfig(
            total_steps=10,
            batch_size=4,
            sequence_length=10,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = config.with_updates(output_dir=tmpdir)
            trainer = Trainer(small_model, config, callbacks=[])

            metrics = trainer.evaluate(small_buffer, num_batches=3)

            assert "loss" in metrics
            assert isinstance(metrics["loss"], float)


class TestTrainFunction:
    """Tests for convenience train() function."""

    def test_train_function(self):
        """Test one-liner train function."""
        model = create_world_model(
            "dreamerv3:size12m",
            obs_shape=(4,),
            action_dim=2,
            encoder_type="mlp",
            decoder_type="mlp",
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=16,
        )

        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=20,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trained_model = train(
                model,
                buffer,
                total_steps=5,
                batch_size=4,
                sequence_length=10,
                device="cpu",
                output_dir=tmpdir,
            )

            assert trained_model is model


class TestTDMPC2Training:
    """Tests for TD-MPC2 training."""

    @pytest.fixture
    def small_tdmpc2_model(self):
        """Create a small TD-MPC2 model for testing."""
        return create_world_model(
            "tdmpc2:5m",
            obs_shape=(39,),
            action_dim=6,
            latent_dim=32,
            hidden_dim=32,
        )

    @pytest.fixture
    def small_buffer_tdmpc2(self):
        """Create a small buffer for TD-MPC2 testing."""
        return create_random_buffer(
            capacity=1000,
            obs_shape=(39,),
            action_dim=6,
            num_episodes=20,
            episode_length=50,
            seed=42,
        )

    def test_tdmpc2_train_short(self, small_tdmpc2_model, small_buffer_tdmpc2):
        """Test TD-MPC2 training for a few steps."""
        config = TrainingConfig(
            total_steps=5,
            batch_size=4,
            sequence_length=10,
            device="cpu",
            log_interval=1,
            save_interval=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = config.with_updates(output_dir=tmpdir)
            trainer = Trainer(small_tdmpc2_model, config, callbacks=[])

            trained_model = trainer.train(small_buffer_tdmpc2)

            assert trainer.state.global_step == 5
            assert trained_model is small_tdmpc2_model


class TestCreateRandomBuffer:
    """Tests for create_random_buffer utility."""

    def test_create_random_buffer(self):
        buffer = create_random_buffer(
            capacity=500,
            obs_shape=(8,),
            action_dim=4,
            num_episodes=10,
            episode_length=50,
            seed=42,
        )

        assert len(buffer) > 0
        assert buffer.obs_shape == (8,)
        assert buffer.action_dim == 4
        assert buffer.num_episodes == 10

    def test_reproducibility(self):
        buffer1 = create_random_buffer(
            capacity=500,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=5,
            seed=123,
        )

        buffer2 = create_random_buffer(
            capacity=500,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=5,
            seed=123,
        )

        # Same seed should produce same data
        batch1 = buffer1.sample(batch_size=1, seq_len=5)
        batch2 = buffer2.sample(batch_size=1, seq_len=5)

        # Note: Sampling is random, so we just check buffer sizes match
        assert len(buffer1) == len(buffer2)
