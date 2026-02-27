"""WorldFlux native Dreamer parity agent using online environment interaction."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from worldflux import create_world_model
from worldflux.core.state import State
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig

from .atari_env import AtariEnvError, build_atari_env

_VALID_POLICY_IMPLS = {"auto", "actor", "shooting"}
_VALID_MODEL_PROFILES = {
    "ci",
    "wf12m",
    "wf25m",
    "wf50m",
    "wf200m",
    "official_like",
    "official_xl",
}


@dataclass(frozen=True)
class DreamerNativeRunConfig:
    task_id: str
    seed: int
    steps: int
    eval_interval: int
    eval_episodes: int
    eval_window: int
    env_backend: str
    device: str
    run_dir: Path
    buffer_capacity: int = 200_000
    warmup_steps: int = 1024
    train_steps_per_eval: int = 64
    sequence_length: int = 64
    batch_size: int = 16
    max_episode_steps: int = 27_000
    policy_mode: str = "diagnostic_random"
    shooting_horizon: int = 5
    shooting_num_candidates: int = 128
    policy_impl: str = "auto"
    replay_ratio: float = 128.0
    train_chunk_size: int = 64
    model_profile: str = "wf25m"
    learning_rate_override: float = 4e-5
    dreamer_diagnostic: bool = False


def _normalize_policy_impl(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _VALID_POLICY_IMPLS:
        raise AtariEnvError(
            f"policy_impl must be one of {sorted(_VALID_POLICY_IMPLS)}, got {value!r}"
        )
    return normalized


def _normalize_model_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "official_like":
        warnings.warn(
            "model_profile='official_like' is deprecated, use 'official_xl'",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "official_xl"
    if normalized not in _VALID_MODEL_PROFILES:
        raise AtariEnvError(
            f"model_profile must be one of {sorted(_VALID_MODEL_PROFILES)}, got {value!r}"
        )
    return normalized


def _obs_to_tensor(obs: np.ndarray, *, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device=device)


def _expand_state(state: State, *, batch_size: int) -> State:
    expanded = {
        key: value.expand(batch_size, *value.shape[1:]).clone()
        for key, value in state.tensors.items()
    }
    return State(tensors=expanded, meta=dict(state.meta))


def _state_to_features(state: State) -> torch.Tensor:
    deter = state.tensors.get("deter")
    stoch = state.tensors.get("stoch")
    if deter is None or stoch is None:
        raise AtariEnvError("Dreamer state must contain both 'deter' and 'stoch' tensors.")
    if stoch.dim() == 3:
        stoch = stoch.flatten(start_dim=1)
    return torch.cat([deter, stoch], dim=-1)


def _action_index_to_onehot(
    *, action_index: int, action_dim: int, device: torch.device
) -> torch.Tensor:
    index = torch.tensor([int(action_index)], device=device, dtype=torch.long)
    return F.one_hot(index, num_classes=max(1, int(action_dim))).to(torch.float32)


def _extract_reward_prediction(decoded: Any) -> torch.Tensor:
    preds = getattr(decoded, "predictions", None)
    if not isinstance(preds, dict):
        preds = getattr(decoded, "preds", None)
    if not isinstance(preds, dict):
        raise AtariEnvError("Dreamer decode output is missing predictions dictionary.")
    reward = preds.get("reward")
    if reward is None:
        raise AtariEnvError("Dreamer shooting policy requires 'reward' prediction output.")
    return reward


def _build_dreamer_model(
    *,
    profile: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    device: str,
    learning_rate: float,
) -> tuple[Any, str]:
    common_overrides: dict[str, Any] = {
        "actor_critic": True,
        "action_type": "discrete",
        "learning_rate": float(learning_rate),
        "actor_lr": float(learning_rate),
        "critic_lr": float(learning_rate),
    }

    profile_to_model_id = {
        "ci": "dreamerv3:ci",
        "wf12m": "dreamerv3:size12m",
        "wf25m": "dreamerv3:size25m",
        "wf50m": "dreamerv3:size50m",
        "wf200m": "dreamerv3:size200m",
        "official_xl": "dreamerv3:official_xl",
    }
    model_id = profile_to_model_id[profile]
    model = create_world_model(
        model_id,
        obs_shape=tuple(obs_shape),
        action_dim=int(action_dim),
        device=device,
        **common_overrides,
    )
    return model, model_id


def _select_dreamer_shooting_action_from_state(
    *,
    model: Any,
    base_state: State,
    action_dim: int,
    rng: np.random.Generator,
    horizon: int,
    num_candidates: int,
) -> tuple[int, torch.Tensor]:
    with torch.no_grad():
        candidates = max(1, int(num_candidates))
        rollout_horizon = max(1, int(horizon))
        simulated_state = _expand_state(base_state, batch_size=candidates)
        device = simulated_state.tensors["deter"].device

        sampled = rng.integers(
            low=0,
            high=max(1, int(action_dim)),
            size=(candidates, rollout_horizon),
            endpoint=False,
        )
        reward_sum = torch.zeros((candidates,), device=device)

        for step in range(rollout_horizon):
            indices = torch.from_numpy(sampled[:, step]).to(device=device, dtype=torch.long)
            actions = F.one_hot(indices, num_classes=max(1, int(action_dim))).to(torch.float32)
            simulated_state = model.transition(simulated_state, actions, deterministic=True)
            rewards = _extract_reward_prediction(model.decode(simulated_state))
            reward_sum += rewards.reshape(candidates, -1)[:, 0]

        best = int(torch.argmax(reward_sum).item())
        action_index = int(sampled[best, 0])
        action_onehot = _action_index_to_onehot(
            action_index=action_index,
            action_dim=action_dim,
            device=device,
        )
    return action_index, action_onehot


def _select_dreamer_policy_action(
    *,
    model: Any,
    obs: np.ndarray,
    state: State,
    prev_action: torch.Tensor,
    action_dim: int,
    rng: np.random.Generator,
    device: str,
    policy_impl: str,
    shooting_horizon: int,
    shooting_num_candidates: int,
) -> tuple[int, torch.Tensor, State, str, bool]:
    with torch.no_grad():
        obs_tensor = _obs_to_tensor(obs, device=device)
        posterior = model.update(state, prev_action, obs_tensor)

    def _actor_action() -> tuple[int, torch.Tensor]:
        actor_head = getattr(model, "actor_head", None)
        if actor_head is None:
            raise AtariEnvError("Actor policy requested but model has no actor_head.")
        with torch.no_grad():
            features = _state_to_features(posterior)
            logits = actor_head(features)
            if not torch.isfinite(logits).all():
                raise AtariEnvError("Actor logits contain non-finite values.")
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                raise AtariEnvError("Actor probabilities contain non-finite values.")
            action_onehot, _ = actor_head.sample(features)
            if not torch.isfinite(action_onehot).all():
                raise AtariEnvError("Sampled actor action contains non-finite values.")
            action = int(torch.argmax(action_onehot, dim=-1).item())
        return action, action_onehot

    def _shooting_action() -> tuple[int, torch.Tensor]:
        return _select_dreamer_shooting_action_from_state(
            model=model,
            base_state=posterior,
            action_dim=action_dim,
            rng=rng,
            horizon=max(1, int(shooting_horizon)),
            num_candidates=max(1, int(shooting_num_candidates)),
        )

    normalized_impl = _normalize_policy_impl(policy_impl)
    if normalized_impl == "shooting":
        action, action_onehot = _shooting_action()
        return action, action_onehot.detach(), posterior, "shooting", False

    if normalized_impl == "actor":
        action, action_onehot = _actor_action()
        return action, action_onehot.detach(), posterior, "actor", False

    try:
        action, action_onehot = _actor_action()
        return action, action_onehot.detach(), posterior, "actor", False
    except Exception:
        action, action_onehot = _shooting_action()
        return action, action_onehot.detach(), posterior, "shooting", True


def _resolve_policy_impl_label(
    *,
    policy_mode: str,
    actor_steps: int,
    shooting_steps: int,
    is_eval: bool,
) -> str:
    if policy_mode != "parity_candidate":
        return "random_env_sampler_eval" if is_eval else "random_env_sampler"
    if actor_steps > 0 and shooting_steps > 0:
        base = "candidate_actor_stateful_with_shooting_fallback"
    elif shooting_steps > 0:
        base = "candidate_shooting_stateful"
    else:
        base = "candidate_actor_stateful"
    return f"{base}_eval" if is_eval else base


def _evaluate_policy(
    *,
    task_id: str,
    backend: str,
    seed: int,
    num_episodes: int,
    max_episode_steps: int,
    policy_mode: str,
    policy_impl: str,
    model: Any,
    action_dim: int,
    device: str,
    shooting_horizon: int,
    shooting_num_candidates: int,
) -> tuple[float, dict[str, int | float]]:
    """Evaluate policy on real environment episodes with deterministic seeds."""
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    normalized_policy_mode = str(policy_mode).strip().lower()
    actor_steps = 0
    shooting_steps = 0
    fallback_steps = 0

    was_training = bool(getattr(model, "training", False))
    if was_training:
        model.eval()

    try:
        for episode in range(max(1, int(num_episodes))):
            env = build_atari_env(
                task_id=task_id,
                seed=seed + episode,
                backend=backend,
                max_episode_steps=max_episode_steps,
            )
            try:
                obs = env.reset(seed=seed + episode)
                done = False
                ep_return = 0.0
                steps = 0

                policy_state: State | None = None
                prev_action: torch.Tensor | None = None
                if normalized_policy_mode == "parity_candidate":
                    policy_state = model.initial_state(1, torch.device(device))
                    prev_action = torch.zeros((1, int(action_dim)), device=torch.device(device))

                while not done and steps < max_episode_steps:
                    if normalized_policy_mode == "parity_candidate":
                        assert policy_state is not None
                        assert prev_action is not None
                        action, action_onehot, policy_state, used_impl, did_fallback = (
                            _select_dreamer_policy_action(
                                model=model,
                                obs=obs,
                                state=policy_state,
                                prev_action=prev_action,
                                action_dim=action_dim,
                                rng=rng,
                                device=device,
                                policy_impl=policy_impl,
                                shooting_horizon=shooting_horizon,
                                shooting_num_candidates=shooting_num_candidates,
                            )
                        )
                        prev_action = action_onehot
                        if used_impl == "actor":
                            actor_steps += 1
                        else:
                            shooting_steps += 1
                        if did_fallback:
                            fallback_steps += 1
                    else:
                        action = env.sample_action(rng)

                    _next_obs, reward, terminated, truncated, _info = env.step(action)
                    obs = _next_obs
                    ep_return += float(reward)
                    done = bool(terminated or truncated)
                    steps += 1
                returns.append(ep_return)
            finally:
                env.close()
    finally:
        if was_training:
            model.train()

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    return mean_return, {
        "actor_steps": int(actor_steps),
        "shooting_steps": int(shooting_steps),
        "fallback_steps": int(fallback_steps),
        "return_std": std_return,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _metric_lookup(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return _safe_float(metrics[key])
    return None


def _train_ready(
    *, buffer: ReplayBuffer, env_steps: int, warmup_steps: int, min_buffer_steps: int
) -> bool:
    return env_steps >= warmup_steps and len(buffer) >= min_buffer_steps


def _drain_backlog(
    *,
    trainer: Trainer,
    buffer: ReplayBuffer,
    backlog: int,
    chunk_size: int,
    train_target_steps: int,
    train_updates: int,
) -> tuple[int, int, int]:
    while backlog >= chunk_size:
        train_target_steps += chunk_size
        trainer.train(buffer, num_steps=train_target_steps)
        train_updates += chunk_size
        backlog -= chunk_size
    return backlog, train_target_steps, train_updates


def run_dreamer_native(
    config: DreamerNativeRunConfig,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Run native online interaction + training loop and return evaluation curve."""
    env = build_atari_env(
        task_id=config.task_id,
        seed=config.seed,
        backend=config.env_backend,
        max_episode_steps=config.max_episode_steps,
    )

    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    model_profile = _normalize_model_profile(config.model_profile)
    learning_rate = float(config.learning_rate_override)
    if learning_rate <= 0:
        raise AtariEnvError(f"learning_rate_override must be positive, got {learning_rate}")

    model, model_id = _build_dreamer_model(
        profile=model_profile,
        obs_shape=tuple(env.obs_shape),
        action_dim=int(env.action_dim),
        device=config.device,
        learning_rate=learning_rate,
    )

    replay_ratio = max(0.0, float(config.replay_ratio))
    max_env_steps = max(0, int(config.steps))
    sequence_length = max(2, int(config.sequence_length))
    batch_size = max(1, int(config.batch_size))
    train_chunk_size = max(1, int(config.train_chunk_size))
    updates_per_env_step = replay_ratio / float(batch_size * sequence_length)
    target_train_updates = int(math.floor(float(max_env_steps) * updates_per_env_step + 1e-12))

    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=max(1, train_chunk_size),
            batch_size=batch_size,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            output_dir=str((config.run_dir / "trainer").resolve()),
            device=config.device,
            seed=int(config.seed),
            log_interval=max(1, train_chunk_size // 4),
            save_interval=max(2, train_chunk_size),
        ),
    )

    buffer = ReplayBuffer(
        capacity=max(1, int(config.buffer_capacity)),
        obs_shape=tuple(env.obs_shape),
        action_dim=int(env.action_dim),
    )

    rng = np.random.default_rng(config.seed)
    curve: list[tuple[float, float]] = []
    policy_mode = str(config.policy_mode).strip().lower()
    if policy_mode not in {"diagnostic_random", "parity_candidate"}:
        raise AtariEnvError(
            "policy_mode must be either 'diagnostic_random' or 'parity_candidate', "
            f"got {config.policy_mode!r}"
        )
    policy_impl = _normalize_policy_impl(config.policy_impl)

    env_steps = 0
    train_target_steps = 0
    train_updates = 0
    episodes = 0
    next_eval = max(1, int(config.eval_interval))
    update_credit = 0.0
    train_backlog = 0
    # ReplayBuffer samples sequences with replacement, so readiness only needs one valid sequence.
    min_buffer_steps = sequence_length
    warmup_steps = max(0, int(config.warmup_steps))

    train_actor_steps = 0
    train_shooting_steps = 0
    eval_actor_steps = 0
    eval_shooting_steps = 0
    actor_fallback_steps = 0
    policy_impl_effective = "shooting" if policy_impl == "shooting" else "actor"
    last_eval_return_mean: float | None = None
    last_eval_return_std: float | None = None

    diagnostics_path = config.run_dir / "diagnostics.jsonl"
    diagnostics_enabled = bool(config.dreamer_diagnostic)
    next_diagnostic_step = 100
    if diagnostics_enabled:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text("", encoding="utf-8")

    def _append_diagnostic_record(
        *,
        event: str,
        eval_return_mean: float | None = None,
        eval_return_std: float | None = None,
    ) -> None:
        if not diagnostics_enabled:
            return
        trainer_metrics = getattr(getattr(trainer, "state", None), "metrics", {})
        if not isinstance(trainer_metrics, dict):
            trainer_metrics = {}
        record = {
            "event": str(event),
            "env_step": int(env_steps),
            "episode": int(episodes),
            "buffer_size": int(len(buffer)),
            "update_credit": float(update_credit),
            "train_backlog": int(train_backlog),
            "target_train_updates": int(target_train_updates),
            "train_updates_executed": int(train_updates),
            "policy_impl_effective": str(policy_impl_effective),
            "actor_steps": int(train_actor_steps),
            "shooting_steps": int(train_shooting_steps),
            "actor_fallback_steps": int(actor_fallback_steps),
            "eval_return_mean": _safe_float(eval_return_mean),
            "eval_return_std": _safe_float(eval_return_std),
            "loss_total": _metric_lookup(trainer_metrics, "loss_total", "total", "loss"),
            "loss_reward": _metric_lookup(trainer_metrics, "loss_reward", "reward"),
            "loss_actor": _metric_lookup(trainer_metrics, "loss_actor", "actor"),
            "loss_critic": _metric_lookup(trainer_metrics, "loss_critic", "critic"),
        }
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def _drain_if_ready() -> None:
        nonlocal train_backlog, train_target_steps, train_updates
        if _train_ready(
            buffer=buffer,
            env_steps=env_steps,
            warmup_steps=warmup_steps,
            min_buffer_steps=min_buffer_steps,
        ):
            train_backlog, train_target_steps, train_updates = _drain_backlog(
                trainer=trainer,
                buffer=buffer,
                backlog=train_backlog,
                chunk_size=train_chunk_size,
                train_target_steps=train_target_steps,
                train_updates=train_updates,
            )

    try:
        while env_steps < max_env_steps:
            obs = env.reset(seed=config.seed + episodes)
            ep_obs: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[float] = []
            ep_dones: list[float] = []

            done = False
            ep_len = 0

            policy_state: State | None = None
            prev_action: torch.Tensor | None = None
            if policy_mode == "parity_candidate":
                policy_state = model.initial_state(1, torch.device(config.device))
                prev_action = torch.zeros(
                    (1, int(env.action_dim)), device=torch.device(config.device)
                )

            def _flush_replay_chunk(*, force_terminal: bool) -> None:
                chunk_len = len(ep_obs)
                if chunk_len <= 0:
                    return
                if not force_terminal:
                    chunk_len = min(sequence_length, chunk_len)
                    if chunk_len < sequence_length:
                        return

                obs_chunk = np.stack(ep_obs[:chunk_len], axis=0).astype(np.float32)
                actions_chunk = np.stack(ep_actions[:chunk_len], axis=0).astype(np.float32)
                rewards_chunk = np.asarray(ep_rewards[:chunk_len], dtype=np.float32)
                dones_chunk = np.asarray(ep_dones[:chunk_len], dtype=np.float32)
                if dones_chunk.size > 0:
                    dones_chunk[-1] = 1.0 if force_terminal else 0.0

                buffer.add_episode(
                    obs=obs_chunk,
                    actions=actions_chunk,
                    rewards=rewards_chunk,
                    dones=dones_chunk,
                )

                del ep_obs[:chunk_len]
                del ep_actions[:chunk_len]
                del ep_rewards[:chunk_len]
                del ep_dones[:chunk_len]

            while not done and env_steps < max_env_steps:
                if policy_mode == "parity_candidate":
                    assert policy_state is not None
                    assert prev_action is not None
                    action, action_onehot, policy_state, used_impl, did_fallback = (
                        _select_dreamer_policy_action(
                            model=model,
                            obs=obs,
                            state=policy_state,
                            prev_action=prev_action,
                            action_dim=env.action_dim,
                            rng=rng,
                            device=config.device,
                            policy_impl=policy_impl,
                            shooting_horizon=max(1, int(config.shooting_horizon)),
                            shooting_num_candidates=max(1, int(config.shooting_num_candidates)),
                        )
                    )
                    prev_action = action_onehot
                    model_action = (
                        action_onehot.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    if used_impl == "actor":
                        train_actor_steps += 1
                    else:
                        train_shooting_steps += 1
                    policy_impl_effective = "actor" if used_impl == "actor" else "shooting"
                    if did_fallback:
                        actor_fallback_steps += 1
                else:
                    action = env.sample_action(rng)
                    model_action = env.to_model_action(action)

                next_obs, reward, terminated, truncated, _ = env.step(action)

                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_actions.append(np.asarray(model_action, dtype=np.float32))
                ep_rewards.append(float(reward))
                ep_dones.append(float(terminated or truncated))

                obs = next_obs
                env_steps += 1
                ep_len += 1
                done = bool(terminated or truncated)

                update_credit += updates_per_env_step
                newly_due = int(update_credit)
                if newly_due > 0:
                    train_backlog += newly_due
                    update_credit -= float(newly_due)

                if not done and env_steps < max_env_steps and len(ep_obs) >= sequence_length:
                    _flush_replay_chunk(force_terminal=False)
                _drain_if_ready()
                while env_steps >= next_diagnostic_step:
                    _append_diagnostic_record(event="periodic")
                    next_diagnostic_step += 100

                while env_steps >= next_eval:
                    eval_return, eval_stats = _evaluate_policy(
                        task_id=config.task_id,
                        backend=config.env_backend,
                        seed=config.seed + 10_000 + int(next_eval),
                        num_episodes=config.eval_episodes,
                        max_episode_steps=config.max_episode_steps,
                        policy_mode=policy_mode,
                        policy_impl=policy_impl,
                        model=model,
                        action_dim=env.action_dim,
                        device=config.device,
                        shooting_horizon=config.shooting_horizon,
                        shooting_num_candidates=config.shooting_num_candidates,
                    )
                    eval_actor_steps += int(eval_stats.get("actor_steps", 0))
                    eval_shooting_steps += int(eval_stats.get("shooting_steps", 0))
                    actor_fallback_steps += int(eval_stats.get("fallback_steps", 0))
                    last_eval_return_mean = float(eval_return)
                    last_eval_return_std = _safe_float(eval_stats.get("return_std"))
                    curve.append((float(env_steps), float(eval_return)))
                    _append_diagnostic_record(
                        event="eval",
                        eval_return_mean=last_eval_return_mean,
                        eval_return_std=last_eval_return_std,
                    )
                    next_eval += max(1, int(config.eval_interval))

                if ep_len >= int(config.max_episode_steps):
                    break

            if ep_obs:
                _flush_replay_chunk(force_terminal=True)
            if ep_len > 0:
                episodes += 1
            _drain_if_ready()
    finally:
        env.close()

    if _train_ready(
        buffer=buffer,
        env_steps=env_steps,
        warmup_steps=warmup_steps,
        min_buffer_steps=min_buffer_steps,
    ):
        while train_backlog > 0:
            chunk = min(train_chunk_size, train_backlog)
            train_target_steps += chunk
            trainer.train(buffer, num_steps=train_target_steps)
            train_updates += chunk
            train_backlog -= chunk

    if not curve:
        eval_return, eval_stats = _evaluate_policy(
            task_id=config.task_id,
            backend=config.env_backend,
            seed=config.seed + 20_000,
            num_episodes=config.eval_episodes,
            max_episode_steps=config.max_episode_steps,
            policy_mode=policy_mode,
            policy_impl=policy_impl,
            model=model,
            action_dim=env.action_dim,
            device=config.device,
            shooting_horizon=config.shooting_horizon,
            shooting_num_candidates=config.shooting_num_candidates,
        )
        eval_actor_steps += int(eval_stats.get("actor_steps", 0))
        eval_shooting_steps += int(eval_stats.get("shooting_steps", 0))
        actor_fallback_steps += int(eval_stats.get("fallback_steps", 0))
        last_eval_return_mean = float(eval_return)
        last_eval_return_std = _safe_float(eval_stats.get("return_std"))
        curve.append((float(config.steps), float(eval_return)))
        _append_diagnostic_record(
            event="eval",
            eval_return_mean=last_eval_return_mean,
            eval_return_std=last_eval_return_std,
        )

    _append_diagnostic_record(
        event="end",
        eval_return_mean=last_eval_return_mean,
        eval_return_std=last_eval_return_std,
    )

    policy_impl_effective = _resolve_policy_impl_label(
        policy_mode=policy_mode,
        actor_steps=train_actor_steps,
        shooting_steps=train_shooting_steps,
        is_eval=False,
    )
    eval_policy_impl_effective = _resolve_policy_impl_label(
        policy_mode=policy_mode,
        actor_steps=eval_actor_steps,
        shooting_steps=eval_shooting_steps,
        is_eval=True,
    )

    metadata: dict[str, Any] = {
        "mode": "native_real_env",
        "family": "dreamerv3",
        "task_id": config.task_id,
        "model_id": model_id,
        "model_profile": model_profile,
        "policy": "model_based" if policy_mode == "parity_candidate" else "random",
        "policy_mode": config.policy_mode,
        "policy_impl": policy_impl_effective,
        "policy_impl_requested": policy_impl,
        "policy_impl_effective": policy_impl_effective,
        "eval_policy": "model_based" if policy_mode == "parity_candidate" else "random",
        "eval_policy_impl": eval_policy_impl_effective,
        "env_backend": config.env_backend,
        "obs_shape": list(env.obs_shape),
        "action_dim": int(env.action_dim),
        "shooting_horizon": int(config.shooting_horizon),
        "shooting_num_candidates": int(config.shooting_num_candidates),
        "buffer_capacity": int(config.buffer_capacity),
        "warmup_steps": int(config.warmup_steps),
        "train_steps_per_eval": int(config.train_steps_per_eval),
        "train_steps_per_eval_status": "deprecated_ignored_for_dreamer",
        "replay_ratio": float(replay_ratio),
        "train_chunk_size": int(train_chunk_size),
        "target_train_updates": int(target_train_updates),
        "train_updates_executed": int(train_updates),
        "env_steps_collected": int(env_steps),
        "episodes_collected": int(episodes),
        "eval_interval": int(config.eval_interval),
        "eval_episodes": int(config.eval_episodes),
        "eval_window": int(config.eval_window),
        "sequence_length": int(sequence_length),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "recurrent_policy_state": bool(policy_mode == "parity_candidate"),
        "state_reset_on_episode": bool(policy_mode == "parity_candidate"),
        "actor_fallback_steps": int(actor_fallback_steps),
    }

    return curve, metadata


__all__ = ["DreamerNativeRunConfig", "run_dreamer_native", "AtariEnvError"]
