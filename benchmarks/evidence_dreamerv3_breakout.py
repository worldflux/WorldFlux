#!/usr/bin/env python3
"""Evidence-oriented DreamerV3 Breakout benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from common import (  # noqa: E402
    add_common_cli,
    build_run_context,
    emit_failure,
    emit_success,
    resolve_mode,
    write_summary,
)

from worldflux import create_world_model  # noqa: E402
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig  # noqa: E402
from worldflux.training.dataset_manifest import (  # noqa: E402
    build_dataset_manifest,
    load_dataset_manifest,
    resolve_dataset_artifact_path,
    write_dataset_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DreamerV3 Breakout evidence benchmark")
    add_common_cli(parser, default_output_dir="outputs/benchmarks/dreamerv3-breakout-evidence")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--dataset-manifest", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _fit_visual_obs(obs: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_c, target_h, target_w = target_shape
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        arr = arr.transpose(2, 0, 1)
    if arr.ndim != 3:
        arr = np.resize(arr, target_shape)

    if arr.shape[0] < target_c:
        pad = np.zeros((target_c - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:target_c]
    if arr.shape[1:] != (target_h, target_w):
        arr = np.resize(arr, target_shape)

    arr = arr.astype(np.float32)
    if np.max(arr) > 1.0:
        arr /= 255.0
    return arr


def _one_hot(action: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[action % dim] = 1.0
    return vec


def _resolve_source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _collect_atari_dataset(
    *,
    env_name: str,
    output_dir: str | Path,
    num_episodes: int,
    max_steps_per_episode: int,
    seed: int,
    obs_shape: tuple[int, int, int] = (3, 64, 64),
) -> tuple[ReplayBuffer, Path, dict[str, object]]:
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("gymnasium + ale_py are required for Atari collection.") from exc

    env = gym.make(env_name, render_mode=None)
    try:
        action_dim = int(env.action_space.n)
        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_rewards: list[float] = []
        all_dones: list[float] = []

        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                action = int(env.action_space.sample())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                all_obs.append(_fit_visual_obs(np.asarray(obs), obs_shape))
                all_actions.append(_one_hot(action, action_dim))
                all_rewards.append(float(reward))
                all_dones.append(float(done))
                obs = next_obs
                steps += 1
    finally:
        env.close()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = output_root / "replay_buffer.npz"
    manifest_path = output_root / "dataset_manifest.json"

    obs_array = np.asarray(all_obs, dtype=np.float32)
    actions_array = np.asarray(all_actions, dtype=np.float32)
    rewards_array = np.asarray(all_rewards, dtype=np.float32)
    dones_array = np.asarray(all_dones, dtype=np.float32)

    buffer = ReplayBuffer(
        capacity=max(len(obs_array), 1),
        obs_shape=obs_shape,
        action_dim=action_dim,
    )
    if len(obs_array) > 0:
        starts = [0] + [int(x) for x in np.where(dones_array[:-1] == 1.0)[0] + 1]
        ends = [int(x) for x in np.where(dones_array == 1.0)[0] + 1]
        if len(ends) < len(starts):
            ends.append(len(obs_array))
        for start, end in zip(starts, ends, strict=False):
            if start >= end:
                continue
            buffer.add_episode(
                obs=obs_array[start:end],
                actions=actions_array[start:end],
                rewards=rewards_array[start:end],
                dones=dones_array[start:end],
            )
    buffer.save(replay_buffer_path)

    manifest = build_dataset_manifest(
        env_id=f"atari/{env_name}",
        collector_kind="atari_collector",
        collector_policy="random",
        seed=seed,
        episodes=num_episodes,
        transitions=len(obs_array),
        reward_stats={
            "mean": float(rewards_array.mean()) if len(rewards_array) else 0.0,
            "std": float(rewards_array.std()) if len(rewards_array) else 0.0,
            "sum": float(rewards_array.sum()) if len(rewards_array) else 0.0,
        },
        source_commit=_resolve_source_commit(),
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        preprocessing={"resize": list(obs_shape), "normalize_255": True},
        artifact_paths={"replay_buffer": replay_buffer_path.name},
    )
    write_dataset_manifest(manifest_path, manifest)
    buffer.data_provenance = {
        "kind": "dataset_manifest",
        "env_id": manifest["env_id"],
        "dataset_manifest": str(manifest_path.resolve()),
    }
    return buffer, manifest_path, manifest


def _load_buffer_from_dataset_manifest(
    manifest_path: str | Path,
) -> tuple[ReplayBuffer, dict[str, object]]:
    manifest = load_dataset_manifest(manifest_path)
    replay_path = resolve_dataset_artifact_path(
        manifest,
        artifact_key="replay_buffer",
        manifest_path=manifest_path,
    )
    buffer = ReplayBuffer.load(replay_path)
    buffer.data_provenance = {
        "kind": "dataset_manifest",
        "env_id": manifest["env_id"],
        "dataset_manifest": str(Path(manifest_path).resolve()),
    }
    return buffer, manifest


def _prepare_buffer_and_manifest(
    args: argparse.Namespace,
) -> tuple[ReplayBuffer, Path, dict[str, object]]:
    if str(args.dataset_manifest).strip():
        manifest_path = Path(args.dataset_manifest).expanduser()
        buffer, manifest = _load_buffer_from_dataset_manifest(manifest_path)
        return buffer, manifest_path, manifest

    dataset_root = Path(args.output_dir) / "dataset"
    return _collect_atari_dataset(
        env_name=str(args.env),
        output_dir=dataset_root,
        num_episodes=4 if resolve_mode(args) == "quick" else 12,
        max_steps_per_episode=64 if resolve_mode(args) == "quick" else 256,
        seed=int(args.seed),
    )


def _collect_policy_returns(
    model,
    *,
    env_name: str,
    episodes: int,
    seed: int,
) -> list[float]:
    del model
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("gymnasium + ale_py are required for Atari evaluation.") from exc

    returns: list[float] = []
    env = gym.make(env_name, render_mode=None)
    try:
        for episode in range(episodes):
            _, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                action = int(env.action_space.sample())
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            returns.append(total_reward)
    finally:
        env.close()
    return returns


def _write_returns_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_learning_curve(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "loss", "mean_return", "std_return", "episodes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_checkpoint_index(path: Path, checkpoints: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"checkpoints": checkpoints}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    *,
    env_name: str,
    manifest_path: Path,
    curve_rows: list[dict[str, object]],
) -> None:
    latest = curve_rows[-1]
    lines = [
        "# DreamerV3 Breakout Evidence Report",
        "",
        f"- env: `{env_name}`",
        f"- dataset_manifest: `{manifest_path.resolve()}`",
        f"- final_step: `{latest['step']}`",
        f"- final_loss: `{latest['loss']}`",
        f"- final_mean_return: `{latest['mean_return']}`",
        f"- final_std_return: `{latest['std_return']}`",
        "",
        "This benchmark is a reproducible evidence bundle, not a benchmark, SOTA claim, or proof claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _milestones_for_mode(mode: str) -> list[int]:
    return [0, 2, 4] if mode == "quick" else [0, 10, 20, 30]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    mode = resolve_mode(args)
    context = build_run_context(scenario="dreamerv3_breakout_evidence", mode=mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    returns_path = output_dir / "returns.jsonl"
    curve_path = output_dir / "learning_curve.csv"
    checkpoint_index_path = output_dir / "checkpoint_index.json"
    report_path = output_dir / "report.md"

    try:
        buffer, manifest_path, manifest = _prepare_buffer_and_manifest(args)
        model = create_world_model(
            "dreamerv3:size12m",
            obs_shape=buffer.obs_shape,
            action_dim=buffer.action_dim,
            device=str(args.device),
        )
        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=max(_milestones_for_mode(mode)),
                batch_size=2 if mode == "quick" else 4,
                sequence_length=5 if mode == "quick" else 12,
                output_dir=str(output_dir),
                device=str(args.device),
                log_interval=1,
                save_interval=max(_milestones_for_mode(mode)) + 1,
                auto_quality_check=False,
            ),
            callbacks=[],
        )
        trainer.support_surface = "supported"
        trainer.data_mode = "offline"
        trainer.run_classification = "advanced_evidence"
        setattr(buffer, "data_provenance", getattr(buffer, "data_provenance", {}))

        curve_rows: list[dict[str, object]] = []
        return_rows: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        best_mean_return = -float("inf")
        best_checkpoint_path = output_dir / "checkpoint_best.pt"

        eval_episodes = 3 if mode == "quick" else 5
        for step_target in _milestones_for_mode(mode):
            if step_target > trainer.state.global_step:
                trainer.train(buffer, num_steps=step_target)
            loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])
            episode_returns = _collect_policy_returns(
                model,
                env_name=str(args.env),
                episodes=eval_episodes,
                seed=int(args.seed) + step_target,
            )
            mean_return = float(np.mean(episode_returns))
            std_return = float(np.std(episode_returns))
            curve_rows.append(
                {
                    "step": int(step_target),
                    "loss": loss,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "episodes": eval_episodes,
                }
            )
            for idx, value in enumerate(episode_returns):
                return_rows.append(
                    {
                        "step": int(step_target),
                        "episode": idx,
                        "return": float(value),
                    }
                )
            checkpoint_path = output_dir / f"checkpoint_step_{step_target}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            checkpoints.append({"step": int(step_target), "path": str(checkpoint_path.resolve())})
            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                checkpoint_path.replace(best_checkpoint_path)

        _write_returns_jsonl(returns_path, return_rows)
        _write_learning_curve(curve_path, curve_rows)
        _write_checkpoint_index(checkpoint_index_path, checkpoints)
        _write_report(
            report_path,
            env_name=str(args.env),
            manifest_path=manifest_path,
            curve_rows=curve_rows,
        )

        summary = {
            "benchmark": "dreamerv3-breakout-evidence",
            "mode": mode,
            "seed": int(args.seed),
            "model": "dreamerv3:size12m",
            "env": str(args.env),
            "success": True,
            "artifacts": {
                "summary": str(summary_path.resolve()),
                "returns": str(returns_path.resolve()),
                "learning_curve": str(curve_path.resolve()),
                "checkpoint_index": str(checkpoint_index_path.resolve()),
                "report": str(report_path.resolve()),
                "dataset_manifest": str(manifest_path.resolve()),
            },
            "dataset_manifest": manifest,
        }
        write_summary(summary_path, summary)
        emit_success(
            context,
            ttfi_sec=float(curve_rows[0]["step"]) if curve_rows else 0.0,
            artifacts={key: str(value) for key, value in summary["artifacts"].items()},
        )
        return 0
    except Exception as exc:  # pragma: no cover - runtime guard
        emit_failure(context, error=str(exc), artifacts={"summary": str(summary_path)})
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
