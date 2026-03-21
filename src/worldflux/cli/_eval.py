# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""The ``eval`` command — run evaluation metrics on a world model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer

from ._app import app, console
from ._rich_output import metric_table


@app.command("eval", rich_help_panel="Quality & Evaluation")
def eval_cmd(
    model_or_path: str = typer.Argument(..., help="Checkpoint path or model ID."),
    suite: str = typer.Option(
        "quick", "--suite", "-s", help="quick (~5s), standard (~30s), comprehensive (~5min)."
    ),
    mode: str = typer.Option(
        "synthetic",
        "--mode",
        help="synthetic (compatibility gate) or real (dataset/env-backed evaluation).",
    ),
    dataset_manifest: Path | None = typer.Option(
        None,
        "--dataset-manifest",
        help="Dataset manifest JSON for real evaluation mode.",
    ),
    env_id: str | None = typer.Option(
        None,
        "--env-id",
        help="Gymnasium environment id for real evaluation mode when no dataset manifest is provided.",
    ),
    device: str = typer.Option("cpu", "--device"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("rich", "--format", "-f"),
) -> None:
    """Run evaluation metrics on a world model.

    [dim]Examples:[/dim]
      worldflux eval dreamer:ci --suite quick
      worldflux eval ./outputs --suite standard --device cuda
      worldflux eval tdmpc2:ci --mode real --dataset-manifest data/halfcheetah.manifest.json -o results.json
    """
    from rich.status import Status

    from worldflux.evals import SUITE_CONFIGS, run_eval_suite

    eval_mode = str(mode).strip().lower() or "synthetic"
    if eval_mode not in {"synthetic", "real"}:
        console.print(f"[wf.fail]Unknown eval mode:[/wf.fail] {mode}. Expected synthetic or real.")
        raise typer.Exit(code=1)

    if suite not in SUITE_CONFIGS:
        console.print(
            f"[wf.fail]Unknown suite:[/wf.fail] {suite}. "
            f"Available: {', '.join(sorted(SUITE_CONFIGS))}."
        )
        raise typer.Exit(code=1)

    # Load or create model
    target = Path(model_or_path)
    model_id = model_or_path

    with Status("[wf.brand]Loading model...[/wf.brand]", console=console, spinner="dots"):
        if target.exists():
            from worldflux.verify.quick import _load_model_from_target

            try:
                model = _load_model_from_target(target, device=device)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                console.print(f"[wf.fail]Failed to load model:[/wf.fail] {exc}")
                raise typer.Exit(code=1) from None
            model_id = str(target)
        else:
            from worldflux.factory import create_world_model

            try:
                model = create_world_model(model_or_path, device=device)
            except (ValueError, RuntimeError) as exc:
                console.print(f"[wf.fail]Failed to create model:[/wf.fail] {exc}")
                raise typer.Exit(code=1) from None

    eval_data = None
    provenance: dict[str, Any] = {"kind": "synthetic", "label": "synthetic evaluation input"}
    if eval_mode == "real":
        if dataset_manifest is None and not str(env_id or "").strip():
            console.print(
                "[wf.fail]Real evaluation requires a dataset manifest or env id.[/wf.fail]"
            )
            raise typer.Exit(code=1)
        try:
            eval_data, provenance = _load_real_eval_data(
                model,  # type: ignore[arg-type]
                dataset_manifest=dataset_manifest,
                env_id=env_id,
                device=device,
                batch_size=4,
                horizon=10,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            console.print(f"[wf.fail]Failed to load real evaluation data:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

    with Status(
        f"[wf.brand]Running {eval_mode} {suite} evaluation...[/wf.brand]",
        console=console,
        spinner="dots",
    ):
        report = run_eval_suite(
            model,  # type: ignore[arg-type]
            suite=suite,
            device=device,
            model_id=model_id,
            output_path=output,
            mode=eval_mode,
            data=eval_data,
            provenance=provenance,
        )

    if format == "json":
        json_str = json.dumps(report.to_dict(), indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
        else:
            console.print(json_str)
        if report.all_passed is False:
            raise typer.Exit(code=1)
        return

    # Rich table output
    rows: list[tuple[str, str, str, bool | None]] = []
    for result in report.results:
        threshold_str = f"{result.threshold:.4f}" if result.threshold is not None else "-"
        rows.append((result.metric, f"{result.value:.4f}", threshold_str, result.passed))

    console.print(metric_table(rows, title=f"Eval: {suite} | {model_id}"))
    console.print(f"[wf.muted]Mode: {eval_mode}[/wf.muted]")
    console.print(f"[wf.muted]Wall time: {report.wall_time_sec:.2f}s[/wf.muted]")

    if report.all_passed is True:
        console.print("[wf.pass]All metrics passed.[/wf.pass]")
    elif report.all_passed is False:
        console.print("[wf.fail]Some metrics failed.[/wf.fail]")
        raise typer.Exit(code=1)


def _load_real_eval_data(
    model,
    *,
    dataset_manifest: Path | None,
    env_id: str | None,
    device: str,
    batch_size: int,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if dataset_manifest is not None:
        return _load_real_eval_data_from_manifest(
            model,
            manifest_path=dataset_manifest,
            device=device,
            batch_size=batch_size,
            horizon=horizon,
        )
    if str(env_id or "").strip():
        return _collect_real_eval_data_from_env(
            model,
            env_id=str(env_id).strip(),
            device=device,
            batch_size=batch_size,
            horizon=horizon,
        )
    raise ValueError("Real evaluation requires a dataset manifest or env id.")


def _load_real_eval_data_from_manifest(
    model,
    *,
    manifest_path: Path,
    device: str,
    batch_size: int,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from worldflux.training import ReplayBuffer
    from worldflux.training.dataset_manifest import (
        load_dataset_manifest,
        resolve_dataset_artifact_path,
    )

    manifest = load_dataset_manifest(manifest_path)
    replay_path = resolve_dataset_artifact_path(
        manifest,
        artifact_key="replay_buffer",
        manifest_path=manifest_path,
    )
    buffer = ReplayBuffer.load(replay_path)
    data = _buffer_to_eval_data(
        buffer=buffer, device=device, batch_size=batch_size, horizon=horizon
    )
    provenance = {
        "kind": "dataset_manifest",
        "env_id": manifest["env_id"],
        "dataset_manifest": str(manifest_path.resolve()),
        "collector_kind": manifest.get("collector_kind"),
        "collector_policy": manifest.get("collector_policy"),
        "source_commit": manifest.get("source_commit"),
    }
    return data, provenance


def _buffer_to_eval_data(
    *,
    buffer,
    device: str,
    batch_size: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    batch = buffer.sample(batch_size=batch_size, seq_len=horizon + 1, device=device)
    return {
        "obs": batch.obs[:, 0],
        "actions": batch.actions[:, :horizon].permute(1, 0, 2),
        "rewards": batch.rewards[:, :horizon].permute(1, 0),
    }


def _collect_real_eval_data_from_env(
    model,
    *,
    env_id: str,
    device: str,
    batch_size: int,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("gymnasium is required for env-backed real evaluation.") from exc

    config = getattr(model, "config", None)
    obs_shape = tuple(getattr(config, "obs_shape", (4,)))
    action_dim = int(getattr(config, "action_dim", 1))

    obs_batch = np.zeros((batch_size, *obs_shape), dtype=np.float32)
    actions = np.zeros((horizon, batch_size, action_dim), dtype=np.float32)
    rewards = np.zeros((horizon, batch_size), dtype=np.float32)

    env = gym.make(env_id)
    try:
        for batch_idx in range(batch_size):
            obs, _ = env.reset(seed=42 + batch_idx)
            obs_batch[batch_idx] = _fit_observation(obs, obs_shape)
            for step in range(horizon):
                action = env.action_space.sample()
                actions[step, batch_idx] = _encode_action(action, action_dim)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                rewards[step, batch_idx] = float(reward)
                if terminated or truncated:
                    obs, _ = env.reset(seed=42 + batch_idx + step + 1)
                else:
                    obs = next_obs
    finally:
        env.close()

    provenance = {
        "kind": "env_protocol",
        "env_id": env_id,
        "batch_size": batch_size,
        "horizon": horizon,
    }
    return {
        "obs": torch.as_tensor(obs_batch, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "rewards": torch.as_tensor(rewards, device=device),
    }, provenance


def _fit_observation(obs: object, obs_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if len(obs_shape) == 1:
        flat = arr.reshape(-1)
        out = np.zeros(obs_shape[0], dtype=np.float32)
        size = min(obs_shape[0], flat.size)
        out[:size] = flat[:size]
        return out

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        arr = arr.transpose(2, 0, 1)
    if arr.shape != obs_shape:
        arr = np.resize(arr, obs_shape)
    return arr.astype(np.float32, copy=False)


def _encode_action(action: object, action_dim: int) -> np.ndarray:
    if np.isscalar(action):
        out = np.zeros(action_dim, dtype=np.float32)
        index = int(action)  # type: ignore[arg-type]
        if 0 <= index < action_dim:
            out[index] = 1.0
        return out

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    out = np.zeros(action_dim, dtype=np.float32)
    size = min(action_dim, arr.size)
    out[:size] = arr[:size]
    return out
