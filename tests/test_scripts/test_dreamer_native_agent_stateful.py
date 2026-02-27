"""Unit tests for stateful Dreamer parity runtime behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_ROOT = REPO_ROOT / "scripts" / "parity"
if str(PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARITY_ROOT))

from runtime import dreamer_native_agent as dna  # noqa: E402

from worldflux.core.state import State  # noqa: E402


class _FakeEnv:
    def __init__(self, *, action_dim: int = 3, episode_len: int = 2):
        self.backend = "stub"
        self.task_id = "atari100k_pong"
        self.obs_shape = (3, 64, 64)
        self.action_dim = int(action_dim)
        self.max_episode_steps = int(episode_len)
        self._episode_len = int(episode_len)
        self._step = 0
        self.actions: list[int] = []

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        _ = seed
        self._step = 0
        return np.zeros(self.obs_shape, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.actions.append(int(action))
        self._step += 1
        done = self._step >= self._episode_len
        obs = np.zeros(self.obs_shape, dtype=np.float32)
        return obs, float(action), bool(done), False, {}

    def sample_action(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.action_dim))

    def to_model_action(self, action: int) -> np.ndarray:
        out = np.zeros((self.action_dim,), dtype=np.float32)
        out[int(action)] = 1.0
        return out

    def close(self) -> None:
        return


class _FiniteActorHead:
    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((features.shape[0], self.action_dim), device=features.device)
        logits[:, -1] = 1.0
        return logits

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = torch.zeros((features.shape[0], self.action_dim), device=features.device)
        onehot[:, -1] = 1.0
        log_prob = torch.zeros((features.shape[0],), device=features.device)
        return onehot, log_prob


class _NonFiniteActorHead(_FiniteActorHead):
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        logits = super().__call__(features)
        logits[:, 0] = torch.nan
        return logits


class _FakeModel(torch.nn.Module):
    def __init__(self, *, action_dim: int = 3, actor_mode: str = "finite"):
        super().__init__()
        self.action_dim = int(action_dim)
        self.config = type("Cfg", (), {"action_dim": self.action_dim})()
        self._dummy_param = torch.nn.Parameter(torch.zeros(()))
        self.initial_state_calls = 0
        self.update_actions: list[torch.Tensor] = []
        self.update_state_ids: list[int] = []
        self.encode_calls = 0

        if actor_mode == "finite":
            self.actor_head = _FiniteActorHead(self.action_dim)
        elif actor_mode == "nonfinite":
            self.actor_head = _NonFiniteActorHead(self.action_dim)

    def encode(self, obs: torch.Tensor) -> State:  # pragma: no cover - should not be used
        _ = obs
        self.encode_calls += 1
        raise AssertionError("encode() must not be called in stateful policy path.")

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> State:
        if device is None:
            device = torch.device("cpu")
        self.initial_state_calls += 1
        state_id = self.initial_state_calls
        deter = torch.zeros((batch_size, 4), device=device)
        stoch = torch.zeros((batch_size, 2, 2), device=device)
        return State(tensors={"deter": deter, "stoch": stoch}, meta={"state_id": state_id})

    def update(self, state: State, action: torch.Tensor | None, obs: torch.Tensor) -> State:
        if action is None:
            action_tensor = torch.zeros((obs.shape[0], self.action_dim), device=obs.device)
        else:
            action_tensor = action.detach().clone()
        self.update_actions.append(action_tensor.cpu())
        state_id = int(state.meta.get("state_id", -1))
        self.update_state_ids.append(state_id)
        deter = torch.full((obs.shape[0], 4), float(state_id), device=obs.device)
        stoch = torch.zeros((obs.shape[0], 2, 2), device=obs.device)
        return State(tensors={"deter": deter, "stoch": stoch}, meta={"state_id": state_id})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        _ = action
        _ = deterministic
        deter = state.tensors["deter"]
        stoch = state.tensors["stoch"]
        return State(tensors={"deter": deter, "stoch": stoch}, meta=dict(state.meta))

    def decode(self, state: State):
        batch_size = state.tensors["deter"].shape[0]
        reward = torch.ones((batch_size, 1), device=state.tensors["deter"].device)
        return type(
            "Decoded", (), {"predictions": {"reward": reward}, "preds": {"reward": reward}}
        )()


class _FakeTrainer:
    instances: list[_FakeTrainer] = []

    def __init__(self, model: _FakeModel, config: object):
        _ = config
        self.model = model
        self.calls: list[int] = []
        self.state = type("TrainerState", (), {"metrics": {}})()
        _FakeTrainer.instances.append(self)

    def train(self, data: object, num_steps: int | None = None):
        _ = data
        step = int(num_steps) if num_steps is not None else 0
        self.calls.append(step)
        self.state.metrics = {
            "loss": float(step),
            "reward": 1.5,
            "actor": 0.5,
            "critic": 0.25,
        }
        return self.model


def _stub_eval(**_: object) -> tuple[float, dict[str, int]]:
    return 0.0, {"actor_steps": 0, "shooting_steps": 0, "fallback_steps": 0}


def test_run_dreamer_native_stateful_policy_resets_episode_state_and_uses_zero_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv(action_dim=3, episode_len=2)
    fake_model = _FakeModel(action_dim=3, actor_mode="finite")
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(dna, "build_atari_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(dna, "create_world_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(dna, "Trainer", _FakeTrainer)
    monkeypatch.setattr(dna, "_evaluate_policy", _stub_eval)

    curve, metadata = dna.run_dreamer_native(
        dna.DreamerNativeRunConfig(
            task_id="atari100k_pong",
            seed=0,
            steps=4,
            eval_interval=100,
            eval_episodes=1,
            eval_window=2,
            env_backend="stub",
            device="cpu",
            run_dir=tmp_path / "run",
            batch_size=2,
            sequence_length=2,
            warmup_steps=0,
            replay_ratio=0.0,
            train_chunk_size=1,
            policy_mode="parity_candidate",
            policy_impl="actor",
            model_profile="ci",
        )
    )

    assert curve
    assert metadata["policy_impl"] == "candidate_actor_stateful"
    assert metadata["recurrent_policy_state"] is True
    assert fake_model.initial_state_calls == 2
    assert fake_model.update_state_ids == [1, 1, 2, 2]
    assert fake_env.actions == [2, 2, 2, 2]

    for state_id in {1, 2}:
        first_idx = fake_model.update_state_ids.index(state_id)
        assert torch.allclose(fake_model.update_actions[first_idx], torch.zeros((1, 3)))


def test_evaluate_policy_uses_stateful_update_without_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeModel(action_dim=3, actor_mode="finite")
    monkeypatch.setattr(
        dna, "build_atari_env", lambda **kwargs: _FakeEnv(action_dim=3, episode_len=2)
    )

    mean_return, stats = dna._evaluate_policy(
        task_id="atari100k_pong",
        backend="stub",
        seed=123,
        num_episodes=1,
        max_episode_steps=4,
        policy_mode="parity_candidate",
        policy_impl="actor",
        model=fake_model,
        action_dim=3,
        device="cpu",
        shooting_horizon=2,
        shooting_num_candidates=4,
    )

    assert isinstance(mean_return, float)
    assert stats["actor_steps"] > 0
    assert stats["shooting_steps"] == 0
    assert fake_model.encode_calls == 0


def test_policy_auto_falls_back_to_shooting_on_nonfinite_actor() -> None:
    fake_model = _FakeModel(action_dim=3, actor_mode="nonfinite")
    state = fake_model.initial_state(1, torch.device("cpu"))
    prev_action = torch.zeros((1, 3))
    obs = np.zeros((3, 64, 64), dtype=np.float32)
    rng = np.random.default_rng(7)

    action, onehot, next_state, used_impl, did_fallback = dna._select_dreamer_policy_action(
        model=fake_model,
        obs=obs,
        state=state,
        prev_action=prev_action,
        action_dim=3,
        rng=rng,
        device="cpu",
        policy_impl="auto",
        shooting_horizon=2,
        shooting_num_candidates=4,
    )
    assert isinstance(action, int)
    assert onehot.shape == (1, 3)
    assert next_state.tensors["deter"].shape == (1, 4)
    assert used_impl == "shooting"
    assert did_fallback is True

    with pytest.raises(dna.AtariEnvError):
        dna._select_dreamer_policy_action(
            model=fake_model,
            obs=obs,
            state=state,
            prev_action=prev_action,
            action_dim=3,
            rng=np.random.default_rng(8),
            device="cpu",
            policy_impl="actor",
            shooting_horizon=2,
            shooting_num_candidates=4,
        )


def test_replay_ratio_scheduler_matches_expected_update_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv(action_dim=3, episode_len=128)
    fake_model = _FakeModel(action_dim=3, actor_mode="finite")
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(dna, "build_atari_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(dna, "create_world_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(dna, "Trainer", _FakeTrainer)
    monkeypatch.setattr(dna, "_evaluate_policy", _stub_eval)
    monkeypatch.setattr(dna, "_train_ready", lambda **kwargs: True)

    _, metadata = dna.run_dreamer_native(
        dna.DreamerNativeRunConfig(
            task_id="atari100k_pong",
            seed=1,
            steps=128,
            eval_interval=1000,
            eval_episodes=1,
            eval_window=2,
            env_backend="stub",
            device="cpu",
            run_dir=tmp_path / "sched",
            batch_size=16,
            sequence_length=64,
            warmup_steps=0,
            replay_ratio=64.0,
            train_chunk_size=4,
            policy_mode="diagnostic_random",
            model_profile="ci",
        )
    )

    assert metadata["target_train_updates"] == 8
    assert metadata["train_updates_executed"] == 8
    assert _FakeTrainer.instances
    assert _FakeTrainer.instances[0].calls == [4, 8]


def test_long_episode_flushes_replay_and_executes_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv(action_dim=3, episode_len=10_000)
    fake_model = _FakeModel(action_dim=3, actor_mode="finite")
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(dna, "build_atari_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(dna, "create_world_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(dna, "Trainer", _FakeTrainer)
    monkeypatch.setattr(dna, "_evaluate_policy", _stub_eval)

    _, metadata = dna.run_dreamer_native(
        dna.DreamerNativeRunConfig(
            task_id="atari100k_pong",
            seed=11,
            steps=256,
            eval_interval=1000,
            eval_episodes=1,
            eval_window=2,
            env_backend="stub",
            device="cpu",
            run_dir=tmp_path / "long_ep_flush",
            batch_size=16,
            sequence_length=64,
            warmup_steps=0,
            replay_ratio=64.0,
            train_chunk_size=4,
            policy_mode="diagnostic_random",
            model_profile="ci",
        )
    )

    assert metadata["target_train_updates"] == 16
    assert metadata["train_updates_executed"] == 16
    assert _FakeTrainer.instances
    assert _FakeTrainer.instances[0].calls[-1] == 16


def test_build_dreamer_model_supports_official_xl_via_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_create_world_model(model_id: str, **kwargs: object) -> object:
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(dna, "create_world_model", _fake_create_world_model)

    model, model_id = dna._build_dreamer_model(
        profile="official_xl",
        obs_shape=(3, 64, 64),
        action_dim=18,
        device="cpu",
        learning_rate=4e-5,
    )

    assert model is sentinel
    assert model_id == "dreamerv3:official_xl"
    assert captured["model_id"] == "dreamerv3:official_xl"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["actor_critic"] is True
    assert kwargs["action_type"] == "discrete"


def test_normalize_model_profile_deprecates_official_like() -> None:
    with pytest.warns(DeprecationWarning, match="official_like"):
        normalized = dna._normalize_model_profile("official_like")
    assert normalized == "official_xl"


def test_run_dreamer_native_writes_diagnostics_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv(action_dim=3, episode_len=2)
    fake_model = _FakeModel(action_dim=3, actor_mode="finite")
    _FakeTrainer.instances.clear()

    def _stub_eval_with_std(**_: object) -> tuple[float, dict[str, int | float]]:
        return 1.25, {
            "actor_steps": 2,
            "shooting_steps": 0,
            "fallback_steps": 0,
            "return_std": 0.2,
        }

    monkeypatch.setattr(dna, "build_atari_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(dna, "create_world_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(dna, "Trainer", _FakeTrainer)
    monkeypatch.setattr(dna, "_evaluate_policy", _stub_eval_with_std)
    monkeypatch.setattr(dna, "_train_ready", lambda **kwargs: True)

    run_dir = tmp_path / "diag_run"
    dna.run_dreamer_native(
        dna.DreamerNativeRunConfig(
            task_id="atari100k_pong",
            seed=3,
            steps=8,
            eval_interval=4,
            eval_episodes=1,
            eval_window=2,
            env_backend="stub",
            device="cpu",
            run_dir=run_dir,
            batch_size=2,
            sequence_length=2,
            warmup_steps=0,
            replay_ratio=64.0,
            train_chunk_size=2,
            policy_mode="parity_candidate",
            policy_impl="actor",
            model_profile="ci",
            dreamer_diagnostic=True,
        )
    )

    diagnostics = run_dir / "diagnostics.jsonl"
    assert diagnostics.exists()
    rows = [
        json.loads(line)
        for line in diagnostics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows

    required_keys = {
        "env_step",
        "episode",
        "buffer_size",
        "update_credit",
        "train_backlog",
        "target_train_updates",
        "train_updates_executed",
        "policy_impl_effective",
        "actor_steps",
        "shooting_steps",
        "actor_fallback_steps",
        "eval_return_mean",
        "eval_return_std",
        "loss_total",
        "loss_reward",
        "loss_actor",
        "loss_critic",
    }
    for row in rows:
        assert required_keys.issubset(row.keys())

    eval_rows = [row for row in rows if row.get("event") == "eval"]
    assert eval_rows
    assert eval_rows[-1]["eval_return_mean"] == pytest.approx(1.25)
    assert eval_rows[-1]["eval_return_std"] == pytest.approx(0.2)


def test_policy_fallback_counter_increments_in_run_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv(action_dim=3, episode_len=2)
    fake_model = _FakeModel(action_dim=3, actor_mode="nonfinite")
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(dna, "build_atari_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(dna, "create_world_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(dna, "Trainer", _FakeTrainer)
    monkeypatch.setattr(dna, "_evaluate_policy", _stub_eval)

    _, metadata = dna.run_dreamer_native(
        dna.DreamerNativeRunConfig(
            task_id="atari100k_pong",
            seed=4,
            steps=4,
            eval_interval=100,
            eval_episodes=1,
            eval_window=2,
            env_backend="stub",
            device="cpu",
            run_dir=tmp_path / "fallback",
            batch_size=2,
            sequence_length=2,
            warmup_steps=0,
            replay_ratio=0.0,
            train_chunk_size=1,
            policy_mode="parity_candidate",
            policy_impl="auto",
            model_profile="ci",
        )
    )

    assert metadata["actor_fallback_steps"] > 0
