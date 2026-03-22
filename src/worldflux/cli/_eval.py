# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""The ``eval`` command — run evaluation metrics on a world model."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer

from ._app import app, console
from ._rich_output import metric_table

_EVAL_MODES = {"synthetic", "dataset_replay", "env_policy"}


@app.command("eval", rich_help_panel="Quality & Evaluation")
def eval_cmd(
    model_or_path: str = typer.Argument(..., help="Checkpoint path or model ID."),
    suite: str = typer.Option(
        "quick", "--suite", "-s", help="quick (~5s), standard (~30s), comprehensive (~5min)."
    ),
    mode: str = typer.Option(
        "synthetic",
        "--mode",
        help=(
            "synthetic (compatibility gate), dataset_replay (replay-backed proxy eval), "
            "or env_policy (learned-policy env rollout)."
        ),
    ),
    dataset_manifest: Path | None = typer.Option(
        None,
        "--dataset-manifest",
        help="Dataset manifest JSON for dataset_replay mode.",
    ),
    env_id: str | None = typer.Option(
        None,
        "--env-id",
        help="Gymnasium environment id for env_policy mode.",
    ),
    device: str = typer.Option("cpu", "--device"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("rich", "--format", "-f"),
) -> None:
    """Run evaluation metrics on a world model.

    [dim]Examples:[/dim]
      worldflux eval dreamer:ci --suite quick
      worldflux eval ./outputs --suite standard --device cuda
      worldflux eval tdmpc2:ci --mode dataset_replay --dataset-manifest data/halfcheetah.manifest.json -o results.json
    """
    from rich.status import Status

    from worldflux.evals import SUITE_CONFIGS, run_eval_suite

    eval_mode = _normalize_eval_mode(mode, dataset_manifest=dataset_manifest, env_id=env_id)
    if eval_mode not in _EVAL_MODES:
        console.print(
            "[wf.fail]Unknown eval mode:[/wf.fail] "
            f"{mode}. Expected synthetic, dataset_replay, or env_policy."
        )
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
    if eval_mode == "dataset_replay":
        if dataset_manifest is None:
            console.print(
                "[wf.fail]dataset_replay evaluation requires a dataset manifest.[/wf.fail]"
            )
            raise typer.Exit(code=1)
        try:
            eval_data, provenance = _load_dataset_replay_eval_data(
                model,  # type: ignore[arg-type]
                manifest_path=dataset_manifest,
                device=device,
                batch_size=4,
                horizon=10,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            console.print(f"[wf.fail]Failed to load dataset_replay data:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None
    elif eval_mode == "env_policy":
        if not str(env_id or "").strip():
            console.print("[wf.fail]env_policy evaluation requires an env id.[/wf.fail]")
            raise typer.Exit(code=1)
        try:
            eval_data, provenance = _load_env_policy_eval_data(
                model,  # type: ignore[arg-type]
                env_id=str(env_id).strip(),
                device=device,
                batch_size=4,
                horizon=10,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            console.print(f"[wf.fail]Failed to load env_policy data:[/wf.fail] {exc}")
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


def _normalize_eval_mode(
    mode: str,
    *,
    dataset_manifest: Path | None,
    env_id: str | None,
) -> str:
    normalized = str(mode).strip().lower() or "synthetic"
    if normalized != "real":
        return normalized

    warnings.warn(
        "'real' eval mode is deprecated; use 'dataset_replay' or 'env_policy'.",
        DeprecationWarning,
        stacklevel=2,
    )
    console.print(
        "[wf.warn]The eval mode `real` is deprecated; use `dataset_replay` or `env_policy`.[/wf.warn]"
    )
    if dataset_manifest is not None:
        return "dataset_replay"
    if str(env_id or "").strip():
        return "env_policy"
    return "real"


def _load_dataset_replay_eval_data(
    model,
    *,
    manifest_path: Path,
    device: str,
    batch_size: int,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    del model
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


def _load_env_policy_eval_data(
    model,
    *,
    env_id: str,
    device: str,
    batch_size: int,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from worldflux.evals.env_policy import collect_env_policy_rollout

    rollout = collect_env_policy_rollout(
        model,
        env_id=env_id,
        episodes=batch_size,
        horizon=horizon,
        seed=42,
        device=device,
        allow_fallback=False,
    )
    assert rollout.obs is not None
    assert rollout.actions is not None
    assert rollout.rewards is not None
    provenance = dict(rollout.provenance)
    provenance["batch_size"] = batch_size
    provenance["horizon"] = horizon
    return {
        "obs": rollout.obs,
        "actions": rollout.actions,
        "rewards": rollout.rewards,
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
