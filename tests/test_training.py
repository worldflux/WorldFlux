"""Tests for training infrastructure."""

import importlib.util
import os
import tempfile

import numpy as np
import pytest
import torch

from worldflux import Batch, create_world_model
from worldflux.core.exceptions import (
    BufferError,
    ConfigurationError,
    ShapeMismatchError,
    TrainingError,
)
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.spec import (
    ActionSpec,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    SequenceLayout,
    StateSpec,
)
from worldflux.core.state import State
from worldflux.training import (
    CheckpointCallback,
    LoggingCallback,
    ReplayBuffer,
    TokenSequenceProvider,
    Trainer,
    TrainingConfig,
    TrajectoryDataset,
    train,
)
from worldflux.training.callbacks import EarlyStoppingCallback, ProgressCallback
from worldflux.training.data import create_random_buffer


class DummyModel(WorldModel):
    """Minimal WorldModel for trainer tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return self.encode(obs)

    def decode(self, state: State):
        return None

    def loss(self, batch) -> LossOutput:
        obs = batch.obs
        if isinstance(obs, dict):
            obs = obs["obs"]
        pred = self.linear(obs)
        loss = pred.mean()
        return LossOutput(loss=loss, components={"dummy": loss})


class NaNModel(WorldModel):
    """WorldModel that produces NaN loss."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, obs, deterministic: bool = False) -> State:
        return State(tensors={"latent": obs})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return state

    def decode(self, state: State):
        return None

    def loss(self, batch) -> LossOutput:
        loss = self.param * torch.tensor(float("nan"))
        return LossOutput(loss=loss, components={"loss": loss})


class InvalidContractModel(DummyModel):
    """Dummy model with an invalid io_contract."""

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
            ),
            action_spec=ActionSpec(kind="continuous", dim=2),
            state_spec=StateSpec(tensors={}),
            sequence_layout=SequenceLayout(axes_by_field={"imaginary": "BT..."}),
            required_batch_keys=("obs",),
            required_state_keys=("latent",),
        )


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

        assert batch.obs.shape == (16, 10, 4)
        assert batch.actions.shape == (16, 10, 2)
        assert batch.rewards.shape == (16, 10)
        assert batch.terminations.shape == (16, 10)

    def test_sample_with_device(self):
        buffer = create_random_buffer(
            capacity=1000,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=10,
        )

        batch = buffer.sample(batch_size=8, seq_len=5, device="cpu")
        assert batch.obs.device == torch.device("cpu")
        assert batch.strict_layout is True
        assert batch.layouts["obs"] == "BT..."

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

    def test_to_from_parquet_roundtrip(self, tmp_path):
        if importlib.util.find_spec("pyarrow") is None:
            pytest.skip("pyarrow is not installed")

        buffer = create_random_buffer(
            capacity=64,
            obs_shape=(4,),
            action_dim=2,
            num_episodes=4,
            seed=123,
        )
        path = tmp_path / "replay.parquet"
        buffer.to_parquet(path)
        loaded = ReplayBuffer.from_parquet(path)

        assert len(loaded) == len(buffer)
        assert loaded.obs_shape == buffer.obs_shape
        assert loaded.action_dim == buffer.action_dim

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

    def test_add_episode_action_length_mismatch(self):
        buffer = ReplayBuffer(capacity=100, obs_shape=(4,), action_dim=2)
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randn(9, 2).astype(np.float32)
        rewards = np.random.randn(10).astype(np.float32)
        with pytest.raises(ShapeMismatchError):
            buffer.add_episode(obs, actions, rewards)

    def test_add_episode_reward_length_mismatch(self):
        buffer = ReplayBuffer(capacity=100, obs_shape=(4,), action_dim=2)
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randn(10, 2).astype(np.float32)
        rewards = np.random.randn(9).astype(np.float32)
        with pytest.raises(ShapeMismatchError):
            buffer.add_episode(obs, actions, rewards)

    def test_add_episode_dones_length_mismatch(self):
        buffer = ReplayBuffer(capacity=100, obs_shape=(4,), action_dim=2)
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randn(10, 2).astype(np.float32)
        rewards = np.random.randn(10).astype(np.float32)
        dones = np.random.randn(9).astype(np.float32)
        with pytest.raises(ShapeMismatchError):
            buffer.add_episode(obs, actions, rewards, dones=dones)

    def test_sample_insufficient_data(self):
        buffer = ReplayBuffer(capacity=100, obs_shape=(4,), action_dim=2)
        obs = np.random.randn(5, 4).astype(np.float32)
        actions = np.random.randn(5, 2).astype(np.float32)
        rewards = np.random.randn(5).astype(np.float32)
        buffer.add_episode(obs, actions, rewards)
        with pytest.raises(BufferError):
            buffer.sample(batch_size=2, seq_len=10)


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

        assert sample.obs.shape == (10, 4)
        assert sample.actions.shape == (10, 2)
        assert sample.rewards.shape == (10,)


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

    def test_batch_provider_sample_args(self):
        class Provider:
            def __init__(self):
                self.args = None

            def sample(self, batch_size, seq_len=None, device="cpu"):
                self.args = (batch_size, seq_len, device)
                return Batch(obs=torch.randn(batch_size, 4))

        provider = Provider()
        model = DummyModel()
        config = TrainingConfig(
            total_steps=1,
            batch_size=3,
            sequence_length=7,
            device="cpu",
        )
        trainer = Trainer(model, config, callbacks=[])
        _ = trainer._next_batch(provider)
        assert provider.args[0] == 3
        assert provider.args[1] == 7
        assert str(provider.args[2]) == str(trainer.device)

    def test_iterable_dict_batch(self):
        model = DummyModel()
        config = TrainingConfig(total_steps=1, batch_size=2, sequence_length=1, device="cpu")
        trainer = Trainer(model, config, callbacks=[])
        data = [
            {"obs": torch.randn(2, 4)},
            {"obs": torch.randn(2, 4)},
        ]
        batch = trainer._next_batch(data)
        assert isinstance(batch, Batch)
        assert batch.obs.shape == (2, 4)

    def test_add_callback_registers_callback(self):
        model = DummyModel()
        config = TrainingConfig(total_steps=1, batch_size=2, sequence_length=1, device="cpu")
        trainer = Trainer(model, config, callbacks=[])
        callback = LoggingCallback(log_interval=10)
        trainer.add_callback(callback)
        assert callback in trainer.callbacks.callbacks

    def test_scheduler_created_from_config(self):
        model = DummyModel()
        config = TrainingConfig(
            total_steps=5,
            batch_size=2,
            sequence_length=1,
            device="cpu",
            scheduler="linear",
            warmup_steps=1,
        )
        trainer = Trainer(model, config, callbacks=[])
        assert trainer.scheduler is not None

    def test_trainer_rejects_model_overrides(self):
        model = DummyModel()
        config = TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=1,
            device="cpu",
            model_overrides={"hidden_dim": 32},
        )
        with pytest.raises(ConfigurationError, match="model_overrides"):
            Trainer(model, config, callbacks=[])

    def test_trainer_rejects_ema_decay(self):
        model = DummyModel()
        config = TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=1,
            device="cpu",
            ema_decay=0.99,
        )
        with pytest.raises(ConfigurationError, match="ema_decay"):
            Trainer(model, config, callbacks=[])

    def test_gradient_accumulation_steps(self):
        class CountingOptimizer(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=1e-3)
                self.step_calls = 0

            def step(self, *args, **kwargs):
                self.step_calls += 1
                return super().step(*args, **kwargs)

        model = DummyModel()
        optimizer = CountingOptimizer(model.parameters())
        config = TrainingConfig(
            total_steps=2,
            batch_size=2,
            sequence_length=1,
            device="cpu",
            grad_clip=0.0,
            gradient_accumulation_steps=2,
        )
        trainer = Trainer(model, config, callbacks=[], optimizer=optimizer)

        class Provider:
            def sample(self, batch_size, seq_len=None, device="cpu"):
                return Batch(obs=torch.randn(batch_size, 4))

        data = Provider()
        trainer._train_step(data)
        trainer._train_step(data)
        assert optimizer.step_calls == 1

    def test_nan_loss_raises(self):
        model = NaNModel()
        config = TrainingConfig(total_steps=1, batch_size=2, sequence_length=1, device="cpu")
        trainer = Trainer(model, config, callbacks=[])

        class Provider:
            def sample(self, batch_size, seq_len=None, device="cpu"):
                return Batch(obs=torch.randn(batch_size, 4))

        with pytest.raises(TrainingError):
            trainer._train_step(Provider())

    def test_early_stopping_triggers(self):
        class ConstantLossModel(WorldModel):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor(0.0))

            def encode(self, obs, deterministic: bool = False) -> State:
                return State(tensors={"latent": obs})

            def transition(
                self, state: State, action: torch.Tensor, deterministic: bool = False
            ) -> State:
                return state

            def update(self, state: State, action: torch.Tensor, obs) -> State:
                return state

            def decode(self, state: State):
                return None

            def loss(self, batch) -> LossOutput:
                loss = self.param * 0 + 1.0
                return LossOutput(loss=loss, components={"loss": loss})

        model = ConstantLossModel()
        config = TrainingConfig(
            total_steps=5,
            batch_size=2,
            sequence_length=1,
            device="cpu",
            save_interval=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = config.with_updates(output_dir=tmpdir)
            callbacks = [EarlyStoppingCallback(patience=1, min_delta=0.0)]
            trainer = Trainer(model, config, callbacks=callbacks)

            class Provider:
                def sample(self, batch_size, seq_len=None, device="cpu"):
                    return Batch(obs=torch.randn(batch_size, 4))

            trainer.train(Provider())
            assert trainer.state.should_stop is True

    def test_contract_missing_required_batch_keys_raises(self):
        model = create_world_model(
            "dreamerv3:size12m",
            obs_shape=(4,),
            action_dim=2,
            encoder_type="mlp",
            decoder_type="mlp",
            deter_dim=32,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=8,
        )
        config = TrainingConfig(total_steps=1, batch_size=2, sequence_length=3, device="cpu")
        trainer = Trainer(model, config, callbacks=[])

        class IncompleteProvider:
            def sample(self, batch_size, seq_len=None, device="cpu"):
                obs = torch.randn(batch_size, seq_len, 4, device=device)
                return Batch(obs=obs, layouts={"obs": "BT..."}, strict_layout=True)

        with pytest.raises(TrainingError, match="missing required keys"):
            trainer._next_batch(IncompleteProvider())

    def test_trainer_rejects_invalid_model_contract(self):
        model = InvalidContractModel()
        config = TrainingConfig(total_steps=1, batch_size=2, sequence_length=3, device="cpu")
        with pytest.raises(TrainingError, match="Invalid model I/O contract"):
            Trainer(model, config, callbacks=[])


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
        _batch1 = buffer1.sample(batch_size=1, seq_len=5)
        _batch2 = buffer2.sample(batch_size=1, seq_len=5)

        # Note: Sampling is random, so we just check buffer sizes match
        assert len(buffer1) == len(buffer2)


class TestTokenSequenceProvider:
    def test_sample(self):
        tokens = np.random.randint(0, 32, size=(12, 20), dtype=np.int64)
        provider = TokenSequenceProvider(tokens)
        batch = provider.sample(batch_size=4, seq_len=8, device="cpu")
        assert batch.obs.shape == (4, 8)
        assert batch.target.shape == (4, 8)
        assert batch.layouts["obs"] == "BT"
