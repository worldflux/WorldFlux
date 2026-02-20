#!/usr/bin/env python3
# ruff: noqa: E402
"""Schema-normalized suite contract for parity execution and statistics."""

from __future__ import annotations

import shlex
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metric_transforms import SUPPORTED_EFFECT_TRANSFORMS

SCHEMA_V1 = "parity.manifest.v1"
SCHEMA_V2 = "parity.suite.v2"


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object.")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{name} must be a boolean.")
    return bool(value)


def _require_float(value: Any, *, name: str) -> float:
    if not isinstance(value, int | float):
        raise RuntimeError(f"{name} must be numeric.")
    return float(value)


def _coerce_string_list(value: Any, *, name: str, non_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RuntimeError(f"{name} must be list[str].")
    out = tuple(v.strip() for v in value if v.strip())
    if non_empty and not out:
        raise RuntimeError(f"{name} must include at least one value.")
    return out


def _coerce_command(value: Any, *, name: str) -> str | list[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value):
        return [str(v).strip() for v in value]
    raise RuntimeError(f"{name} must be a non-empty string or non-empty list[str].")


def _tokenize_legacy_command(raw: str, *, name: str) -> list[str]:
    if any(pattern in raw for pattern in ("`", "$(", "${", "\n", "\r")):
        raise RuntimeError(
            f"{name} string command contains shell-special constructs that are forbidden. "
            "Use list[str] command form."
        )
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as exc:
        raise RuntimeError(f"{name} string command could not be tokenized: {exc}") from exc
    if not tokens:
        raise RuntimeError(f"{name} string command produced empty argv after tokenization.")
    dangerous_tokens = {";", "&&", "||", "|", "&"}
    if any(token in dangerous_tokens for token in tokens):
        raise RuntimeError(
            f"{name} string command contains control operators {sorted(dangerous_tokens)}. "
            "Use list[str] command form."
        )
    return [str(token) for token in tokens]


@dataclass(frozen=True)
class SourceReference:
    commit: str
    artifact_path: str


@dataclass(frozen=True)
class CommandContract:
    adapter: str
    cwd: str
    command: list[str]
    env: dict[str, str]
    timeout_sec: int | None
    source: SourceReference | None


@dataclass(frozen=True)
class SeedPolicyContract:
    mode: str
    values: tuple[int, ...]
    pilot_seeds: int
    min_seeds: int
    max_seeds: int
    power_target: float


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    family: str
    required_metrics: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    higher_is_better: bool
    effect_transform: str
    equivalence_margin: float
    noninferiority_margin: float
    alpha: float
    holm_scope: str
    train_budget: dict[str, Any]
    eval_protocol: dict[str, Any]
    validity_requirements: dict[str, Any]
    official: CommandContract
    worldflux: CommandContract


@dataclass(frozen=True)
class SuiteContract:
    schema_version: str
    suite_id: str
    family: str
    defaults: dict[str, Any]
    statistical: dict[str, Any]
    seed_policy: SeedPolicyContract
    train_budget: dict[str, Any]
    eval_protocol: dict[str, Any]
    validity_requirements: dict[str, Any]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    higher_is_better: bool
    effect_transform: str
    equivalence_margin: float
    noninferiority_margin: float
    alpha: float
    holm_scope: str
    tasks: tuple[TaskContract, ...]


def _parse_statistical_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    obj = _require_object(raw, name="statistical")
    out: dict[str, Any] = {}
    bayesian_raw = obj.get("bayesian")
    if bayesian_raw is None:
        return out

    bayesian = _require_object(bayesian_raw, name="statistical.bayesian")
    normalized: dict[str, Any] = {}
    if "enable" in bayesian:
        normalized["enable"] = _require_bool(bayesian["enable"], name="statistical.bayesian.enable")
    if "draws" in bayesian:
        draws = bayesian["draws"]
        if not isinstance(draws, int) or draws <= 0:
            raise RuntimeError("statistical.bayesian.draws must be a positive integer.")
        normalized["draws"] = draws
    if "seed" in bayesian:
        seed = bayesian["seed"]
        if not isinstance(seed, int) or seed < 0:
            raise RuntimeError("statistical.bayesian.seed must be a non-negative integer.")
        normalized["seed"] = seed
    if "probability_threshold_equivalence" in bayesian:
        threshold = _require_float(
            bayesian["probability_threshold_equivalence"],
            name="statistical.bayesian.probability_threshold_equivalence",
        )
        if not (0.0 < threshold <= 1.0):
            raise RuntimeError(
                "statistical.bayesian.probability_threshold_equivalence must be in (0, 1]."
            )
        normalized["probability_threshold_equivalence"] = threshold
    if "probability_threshold_noninferiority" in bayesian:
        threshold = _require_float(
            bayesian["probability_threshold_noninferiority"],
            name="statistical.bayesian.probability_threshold_noninferiority",
        )
        if not (0.0 < threshold <= 1.0):
            raise RuntimeError(
                "statistical.bayesian.probability_threshold_noninferiority must be in (0, 1]."
            )
        normalized["probability_threshold_noninferiority"] = threshold
    if "dual_pass_required" in bayesian:
        normalized["dual_pass_required"] = _require_bool(
            bayesian["dual_pass_required"],
            name="statistical.bayesian.dual_pass_required",
        )

    out["bayesian"] = normalized
    return out


def _parse_seed_policy(raw: Any) -> SeedPolicyContract:
    obj = _require_object(raw, name="seed_policy")
    mode = _require_string(obj.get("mode", "fixed"), name="seed_policy.mode")
    if mode not in {"fixed", "auto_power"}:
        raise RuntimeError("seed_policy.mode must be either 'fixed' or 'auto_power'.")

    values = obj.get("values", [])
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        raise RuntimeError("seed_policy.values must be list[int].")

    pilot_seeds = int(obj.get("pilot_seeds", 10))
    min_seeds = int(obj.get("min_seeds", 20))
    max_seeds = int(obj.get("max_seeds", 50))
    power_target = float(obj.get("power_target", 0.80))

    if pilot_seeds < 1:
        raise RuntimeError("seed_policy.pilot_seeds must be >= 1.")
    if not (1 <= min_seeds <= max_seeds):
        raise RuntimeError("seed_policy must satisfy 1 <= min_seeds <= max_seeds.")
    if not (0.5 <= power_target < 1.0):
        raise RuntimeError("seed_policy.power_target must be in [0.5, 1.0).")

    return SeedPolicyContract(
        mode=mode,
        values=tuple(int(v) for v in values),
        pilot_seeds=pilot_seeds,
        min_seeds=min_seeds,
        max_seeds=max_seeds,
        power_target=power_target,
    )


def _parse_source(raw: Any, *, name: str, required: bool) -> SourceReference | None:
    if raw is None:
        if required:
            raise RuntimeError(f"{name} is required.")
        return None

    obj = _require_object(raw, name=name)
    commit = _require_string(obj.get("commit"), name=f"{name}.commit")
    artifact_path = _require_string(obj.get("artifact_path"), name=f"{name}.artifact_path")
    return SourceReference(commit=commit, artifact_path=artifact_path)


def _parse_command_contract(
    raw: Any,
    *,
    name: str,
    supported_adapters: set[str] | None,
    require_source: bool,
    allow_legacy_string_command: bool,
) -> CommandContract:
    obj = _require_object(raw, name=name)

    adapter = _require_string(obj.get("adapter"), name=f"{name}.adapter")
    if supported_adapters is not None and adapter not in supported_adapters:
        raise RuntimeError(
            f"Unsupported adapter '{adapter}' in {name}.adapter. Supported: {sorted(supported_adapters)}"
        )

    cwd = _require_string(obj.get("cwd", "."), name=f"{name}.cwd")
    command_raw = _coerce_command(obj.get("command"), name=f"{name}.command")
    if isinstance(command_raw, list):
        command: list[str] = list(command_raw)
    else:
        if not allow_legacy_string_command:
            raise RuntimeError(
                f"{name}.command must be list[str] for schema '{SCHEMA_V2}'. "
                "String commands are only supported for legacy schema parity.manifest.v1."
            )
        command = _tokenize_legacy_command(command_raw, name=f"{name}.command")
        warnings.warn(
            f"{name}.command provided as legacy string and tokenized to argv list. "
            "Please migrate to list[str] command form.",
            stacklevel=2,
        )

    env_raw = obj.get("env", {})
    if not isinstance(env_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
    ):
        raise RuntimeError(f"{name}.env must be mapping[str, str].")

    timeout = obj.get("timeout_sec", None)
    if timeout is not None:
        if not isinstance(timeout, int) or timeout <= 0:
            raise RuntimeError(f"{name}.timeout_sec must be a positive integer when provided.")

    source = _parse_source(obj.get("source"), name=f"{name}.source", required=require_source)

    return CommandContract(
        adapter=adapter,
        cwd=cwd,
        command=command,
        env=dict(env_raw),
        timeout_sec=timeout,
        source=source,
    )


def _parse_metric_defaults(raw: dict[str, Any], *, require_all: bool) -> dict[str, Any]:
    if require_all and "primary_metric" not in raw:
        raise RuntimeError("primary_metric is required for parity.suite.v2")
    if require_all and "secondary_metrics" not in raw:
        raise RuntimeError("secondary_metrics is required for parity.suite.v2")
    if require_all and "higher_is_better" not in raw:
        raise RuntimeError("higher_is_better is required for parity.suite.v2")
    if require_all and "effect_transform" not in raw:
        raise RuntimeError("effect_transform is required for parity.suite.v2")
    if require_all and "equivalence_margin" not in raw:
        raise RuntimeError("equivalence_margin is required for parity.suite.v2")
    if require_all and "noninferiority_margin" not in raw:
        raise RuntimeError("noninferiority_margin is required for parity.suite.v2")
    if require_all and "alpha" not in raw:
        raise RuntimeError("alpha is required for parity.suite.v2")
    if require_all and "holm_scope" not in raw:
        raise RuntimeError("holm_scope is required for parity.suite.v2")

    primary_metric = _require_string(
        raw.get("primary_metric", "final_return_mean"),
        name="primary_metric",
    )
    secondary_metrics = _coerce_string_list(
        raw.get("secondary_metrics", ["auc_return"]),
        name="secondary_metrics",
    )
    higher_is_better = _require_bool(
        raw["higher_is_better"] if require_all else raw.get("higher_is_better", True),
        name="higher_is_better",
    )
    effect_transform = _require_string(
        raw.get("effect_transform", "paired_log_ratio"),
        name="effect_transform",
    ).lower()
    if effect_transform not in SUPPORTED_EFFECT_TRANSFORMS:
        raise RuntimeError(f"effect_transform must be one of {sorted(SUPPORTED_EFFECT_TRANSFORMS)}")

    return {
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics,
        "higher_is_better": higher_is_better,
        "effect_transform": effect_transform,
        "equivalence_margin": _require_float(
            raw.get("equivalence_margin", 0.05), name="equivalence_margin"
        ),
        "noninferiority_margin": _require_float(
            raw.get("noninferiority_margin", 0.05),
            name="noninferiority_margin",
        ),
        "alpha": _require_float(raw.get("alpha", 0.05), name="alpha"),
        "holm_scope": _require_string(raw.get("holm_scope", "all_metrics"), name="holm_scope"),
    }


def _parse_task_contract(
    raw: Any,
    *,
    name: str,
    root_family: str,
    metric_defaults: dict[str, Any],
    train_budget: dict[str, Any],
    eval_protocol: dict[str, Any],
    validity_requirements: dict[str, Any],
    supported_adapters: set[str] | None,
    require_source: bool,
    allow_legacy_string_command: bool,
) -> TaskContract:
    obj = _require_object(raw, name=name)
    task_id = _require_string(obj.get("task_id"), name=f"{name}.task_id")
    family = _require_string(obj.get("family", root_family), name=f"{name}.family")

    primary_metric = _require_string(
        obj.get("primary_metric", metric_defaults["primary_metric"]),
        name=f"{name}.primary_metric",
    )
    secondary_metrics = _coerce_string_list(
        obj.get("secondary_metrics", list(metric_defaults["secondary_metrics"])),
        name=f"{name}.secondary_metrics",
    )
    required_metrics_extra = _coerce_string_list(
        obj.get("required_metrics", []),
        name=f"{name}.required_metrics",
    )

    required_metrics_ordered: list[str] = []
    for metric in (primary_metric, *secondary_metrics, *required_metrics_extra):
        if metric not in required_metrics_ordered:
            required_metrics_ordered.append(metric)

    effect_transform = _require_string(
        obj.get("effect_transform", metric_defaults["effect_transform"]),
        name=f"{name}.effect_transform",
    ).lower()
    if effect_transform not in SUPPORTED_EFFECT_TRANSFORMS:
        raise RuntimeError(
            f"{name}.effect_transform must be one of {sorted(SUPPORTED_EFFECT_TRANSFORMS)}"
        )

    higher_is_better = bool(obj.get("higher_is_better", metric_defaults["higher_is_better"]))

    task_train_budget = _require_object(
        obj.get("train_budget", train_budget),
        name=f"{name}.train_budget",
    )
    task_eval_protocol = _require_object(
        obj.get("eval_protocol", eval_protocol),
        name=f"{name}.eval_protocol",
    )
    task_validity_requirements = _require_object(
        obj.get("validity_requirements", validity_requirements),
        name=f"{name}.validity_requirements",
    )

    official = _parse_command_contract(
        obj.get("official"),
        name=f"{name}.official",
        supported_adapters=supported_adapters,
        require_source=require_source,
        allow_legacy_string_command=allow_legacy_string_command,
    )
    worldflux = _parse_command_contract(
        obj.get("worldflux"),
        name=f"{name}.worldflux",
        supported_adapters=supported_adapters,
        require_source=require_source,
        allow_legacy_string_command=allow_legacy_string_command,
    )

    return TaskContract(
        task_id=task_id,
        family=family,
        required_metrics=tuple(required_metrics_ordered),
        primary_metric=primary_metric,
        secondary_metrics=tuple(secondary_metrics),
        higher_is_better=higher_is_better,
        effect_transform=effect_transform,
        equivalence_margin=_require_float(
            obj.get("equivalence_margin", metric_defaults["equivalence_margin"]),
            name=f"{name}.equivalence_margin",
        ),
        noninferiority_margin=_require_float(
            obj.get("noninferiority_margin", metric_defaults["noninferiority_margin"]),
            name=f"{name}.noninferiority_margin",
        ),
        alpha=_require_float(
            obj.get("alpha", metric_defaults["alpha"]),
            name=f"{name}.alpha",
        ),
        holm_scope=_require_string(
            obj.get("holm_scope", metric_defaults["holm_scope"]),
            name=f"{name}.holm_scope",
        ),
        train_budget=dict(task_train_budget),
        eval_protocol=dict(task_eval_protocol),
        validity_requirements=dict(task_validity_requirements),
        official=official,
        worldflux=worldflux,
    )


def _require_validity_requirements_for_v2(raw: dict[str, Any]) -> dict[str, Any]:
    requirements = _require_object(raw.get("validity_requirements"), name="validity_requirements")
    _require_string(requirements.get("policy_mode"), name="validity_requirements.policy_mode")
    _require_string(
        requirements.get("environment_backend"), name="validity_requirements.environment_backend"
    )
    forbidden = requirements.get("forbidden_shortcuts")
    _coerce_string_list(forbidden, name="validity_requirements.forbidden_shortcuts")
    return requirements


def load_suite_contract(
    raw: dict[str, Any],
    *,
    supported_adapters: set[str] | None = None,
) -> SuiteContract:
    schema_version = _require_string(raw.get("schema_version"), name="schema_version")
    statistical = _parse_statistical_config(raw.get("statistical"))

    if schema_version == SCHEMA_V1:
        defaults = _require_object(raw.get("defaults", {}), name="defaults")
        metric_defaults = _parse_metric_defaults(defaults, require_all=False)
        seed_policy = _parse_seed_policy(raw.get("seed_policy", {}))
        tasks_raw = raw.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise RuntimeError("tasks must be a non-empty list.")

        tasks: list[TaskContract] = []
        seen: set[str] = set()
        for index, item in enumerate(tasks_raw):
            name = f"tasks[{index}]"
            task = _parse_task_contract(
                item,
                name=name,
                root_family=_require_string(
                    _require_object(item, name=name).get("family"),
                    name=f"{name}.family",
                ),
                metric_defaults=metric_defaults,
                train_budget={},
                eval_protocol={},
                validity_requirements={
                    "policy_mode": "diagnostic_random",
                    "environment_backend": "auto",
                    "forbidden_shortcuts": [],
                },
                supported_adapters=supported_adapters,
                require_source=False,
                allow_legacy_string_command=True,
            )
            if task.task_id in seen:
                raise RuntimeError(f"Duplicate task_id: {task.task_id}")
            seen.add(task.task_id)
            tasks.append(task)

        family = "mixed" if len({task.family for task in tasks}) > 1 else tasks[0].family
        return SuiteContract(
            schema_version=schema_version,
            suite_id=_require_string(
                raw.get("suite_id", "legacy.parity.manifest.v1"), name="suite_id"
            ),
            family=family,
            defaults=dict(defaults),
            statistical=statistical,
            seed_policy=seed_policy,
            train_budget={},
            eval_protocol={},
            validity_requirements={
                "policy_mode": "diagnostic_random",
                "environment_backend": "auto",
                "forbidden_shortcuts": [],
            },
            primary_metric=str(metric_defaults["primary_metric"]),
            secondary_metrics=tuple(metric_defaults["secondary_metrics"]),
            higher_is_better=bool(metric_defaults["higher_is_better"]),
            effect_transform=str(metric_defaults["effect_transform"]),
            equivalence_margin=float(metric_defaults["equivalence_margin"]),
            noninferiority_margin=float(metric_defaults["noninferiority_margin"]),
            alpha=float(metric_defaults["alpha"]),
            holm_scope=str(metric_defaults["holm_scope"]),
            tasks=tuple(tasks),
        )

    if schema_version != SCHEMA_V2:
        raise RuntimeError(
            f"Unsupported schema_version '{schema_version}'. Expected '{SCHEMA_V1}' or '{SCHEMA_V2}'."
        )

    defaults = _require_object(raw.get("defaults", {}), name="defaults")
    family = _require_string(raw.get("family"), name="family")
    suite_id = _require_string(raw.get("suite_id"), name="suite_id")

    metric_defaults = _parse_metric_defaults(raw, require_all=True)
    seed_policy = _parse_seed_policy(raw.get("seed_policy"))
    train_budget = _require_object(raw.get("train_budget"), name="train_budget")
    eval_protocol = _require_object(raw.get("eval_protocol"), name="eval_protocol")
    validity_requirements = _require_validity_requirements_for_v2(raw)

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise RuntimeError("tasks must be a non-empty list.")

    tasks: list[TaskContract] = []
    seen: set[str] = set()
    for index, item in enumerate(tasks_raw):
        name = f"tasks[{index}]"
        task = _parse_task_contract(
            item,
            name=name,
            root_family=family,
            metric_defaults=metric_defaults,
            train_budget=train_budget,
            eval_protocol=eval_protocol,
            validity_requirements=validity_requirements,
            supported_adapters=supported_adapters,
            require_source=True,
            allow_legacy_string_command=False,
        )
        if task.task_id in seen:
            raise RuntimeError(f"Duplicate task_id: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)

    return SuiteContract(
        schema_version=schema_version,
        suite_id=suite_id,
        family=family,
        defaults=dict(defaults),
        statistical=statistical,
        seed_policy=seed_policy,
        train_budget=dict(train_budget),
        eval_protocol=dict(eval_protocol),
        validity_requirements=dict(validity_requirements),
        primary_metric=str(metric_defaults["primary_metric"]),
        secondary_metrics=tuple(metric_defaults["secondary_metrics"]),
        higher_is_better=bool(metric_defaults["higher_is_better"]),
        effect_transform=str(metric_defaults["effect_transform"]),
        equivalence_margin=float(metric_defaults["equivalence_margin"]),
        noninferiority_margin=float(metric_defaults["noninferiority_margin"]),
        alpha=float(metric_defaults["alpha"]),
        holm_scope=str(metric_defaults["holm_scope"]),
        tasks=tuple(tasks),
    )


def to_legacy_manifest_dict(contract: SuiteContract) -> dict[str, Any]:
    """Render suite contract into backward-compatible parity.manifest.v1-like shape."""
    out = {
        "schema_version": contract.schema_version,
        "suite_id": contract.suite_id,
        "family": contract.family,
        "defaults": {
            **contract.defaults,
            "alpha": contract.alpha,
            "equivalence_margin": contract.equivalence_margin,
            "noninferiority_margin": contract.noninferiority_margin,
            "effect_transform": contract.effect_transform,
            "primary_metric": contract.primary_metric,
            "secondary_metrics": list(contract.secondary_metrics),
            "higher_is_better": contract.higher_is_better,
            "holm_scope": contract.holm_scope,
        },
        "seed_policy": {
            "mode": contract.seed_policy.mode,
            "values": list(contract.seed_policy.values),
            "pilot_seeds": contract.seed_policy.pilot_seeds,
            "min_seeds": contract.seed_policy.min_seeds,
            "max_seeds": contract.seed_policy.max_seeds,
            "power_target": contract.seed_policy.power_target,
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "family": task.family,
                "required_metrics": list(task.required_metrics),
                "primary_metric": task.primary_metric,
                "secondary_metrics": list(task.secondary_metrics),
                "higher_is_better": task.higher_is_better,
                "effect_transform": task.effect_transform,
                "equivalence_margin": task.equivalence_margin,
                "noninferiority_margin": task.noninferiority_margin,
                "alpha": task.alpha,
                "holm_scope": task.holm_scope,
                "train_budget": task.train_budget,
                "eval_protocol": task.eval_protocol,
                "validity_requirements": task.validity_requirements,
                "official": {
                    "adapter": task.official.adapter,
                    "cwd": task.official.cwd,
                    "command": task.official.command,
                    "env": task.official.env,
                    "timeout_sec": task.official.timeout_sec,
                    "source": (
                        {
                            "commit": task.official.source.commit,
                            "artifact_path": task.official.source.artifact_path,
                        }
                        if task.official.source is not None
                        else None
                    ),
                },
                "worldflux": {
                    "adapter": task.worldflux.adapter,
                    "cwd": task.worldflux.cwd,
                    "command": task.worldflux.command,
                    "env": task.worldflux.env,
                    "timeout_sec": task.worldflux.timeout_sec,
                    "source": (
                        {
                            "commit": task.worldflux.source.commit,
                            "artifact_path": task.worldflux.source.artifact_path,
                        }
                        if task.worldflux.source is not None
                        else None
                    ),
                },
            }
            for task in contract.tasks
        ],
    }
    if contract.statistical:
        out["statistical"] = dict(contract.statistical)
    return out


__all__ = [
    "SCHEMA_V1",
    "SCHEMA_V2",
    "CommandContract",
    "SeedPolicyContract",
    "SourceReference",
    "SuiteContract",
    "TaskContract",
    "load_suite_contract",
    "to_legacy_manifest_dict",
]
