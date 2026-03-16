#!/usr/bin/env python3
# ruff: noqa: E402
"""Validity gate for parity proof runs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from suite_registry import FamilyPluginRegistry, build_default_registry


def _metrics_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    metadata = metrics.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _entry_recipe(entry: dict[str, Any], key: str) -> dict[str, Any]:
    value = entry.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _metadata_recipe(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _expected_model_identity(*, family: str, system: str) -> tuple[str, str]:
    normalized_family = str(family).strip().lower()
    if normalized_family == "dreamerv3":
        return "dreamerv3:official_xl", "official_xl"
    if normalized_family == "tdmpc2":
        if str(system).strip().lower() == "official":
            return "tdmpc2:5m", "5m"
        return "tdmpc2:proof_5m", "proof_5m"
    return "", ""


def _compare_recipe_fields(
    *,
    issues: list[dict[str, Any]],
    task_id: str,
    seed: int,
    system: str,
    category: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    for key, expected_value in expected.items():
        if key == "notes":
            continue
        if key not in actual:
            issues.append(
                {
                    "code": "missing_recipe_metadata",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": f"{category}.{key} missing from metrics metadata.",
                }
            )
            continue
        actual_value = actual.get(key)
        if actual_value != expected_value:
            issues.append(
                {
                    "code": "recipe_mismatch",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": (
                        f"{category}.{key} mismatch: expected {expected_value!r}, "
                        f"got {actual_value!r}"
                    ),
                }
            )


def _flatten(prefix: str, payload: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in payload.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        out[current] = value
        if isinstance(value, dict):
            _flatten(current, value, out)


def _value_from_flat(flat: dict[str, Any], key: str) -> Any:
    if key in flat:
        return flat[key]
    if key.startswith("metadata."):
        return flat.get(key)
    direct = flat.get(f"metadata.{key}")
    if direct is not None:
        return direct
    return flat.get(key)


def _match_forbidden(flat: dict[str, Any], rule: str) -> bool:
    token = str(rule).strip()
    if not token:
        return False

    if "=" in token:
        key, expected = token.split("=", 1)
        key = key.strip()
        expected_value = expected.strip().lower()
        value = _value_from_flat(flat, key)
        if value is None:
            return False
        return str(value).strip().lower() == expected_value

    value = _value_from_flat(flat, token)
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "mock", "random"}


def _infer_legacy_official_backend(*, task_id: str, family: str) -> str:
    """Infer env backend for legacy official records missing metadata.env_backend."""
    normalized_task = str(task_id).strip().lower()
    normalized_family = str(family).strip().lower()

    if normalized_family in {"dreamerv3", "dreamer"}:
        return "gymnasium"
    if normalized_family in {"tdmpc2", "tdmpc"}:
        return "dmcontrol"
    if normalized_task.startswith("atari100k_") or normalized_task.startswith("atari_"):
        return "gymnasium"
    if normalized_task in {"cheetah-run", "hopper-hop", "walker-run", "dog-run"}:
        return "dmcontrol"
    return ""


def evaluate_validity(
    entries: list[dict[str, Any]],
    *,
    proof_mode: bool,
    required_policy_mode: str,
    requirements: dict[str, Any] | None = None,
    registry: FamilyPluginRegistry | None = None,
) -> dict[str, Any]:
    """Evaluate parity-run validity conditions before statistical testing."""
    active_requirements = dict(requirements or {})
    default_forbidden = active_requirements.get("forbidden_shortcuts", [])
    if not isinstance(default_forbidden, list):
        default_forbidden = []
    default_backend = str(active_requirements.get("environment_backend", "auto")).strip().lower()
    default_policy_mode = str(active_requirements.get("policy_mode", required_policy_mode)).strip()

    plugin_registry = registry or build_default_registry()
    issues: list[dict[str, Any]] = []

    protocol_hashes: dict[tuple[str, int], dict[str, str]] = {}

    for entry in entries:
        if entry.get("status") != "success":
            continue

        task_id = str(entry.get("task_id", ""))
        seed = int(entry.get("seed", -1))
        system = str(entry.get("system", ""))
        family = str(entry.get("family", "")).strip().lower()
        metadata = _metrics_metadata(entry)
        record_requirements = dict(active_requirements)
        entry_requirements = entry.get("validity_requirements")
        if isinstance(entry_requirements, dict):
            record_requirements.update(entry_requirements)
        forbidden_rules = record_requirements.get("forbidden_shortcuts", default_forbidden)
        if not isinstance(forbidden_rules, list):
            forbidden_rules = list(default_forbidden)
        required_backend = (
            str(record_requirements.get("environment_backend", default_backend)).strip().lower()
        )
        policy_mode_expected = str(
            record_requirements.get("policy_mode", default_policy_mode)
        ).strip()

        flat: dict[str, Any] = {}
        _flatten("metadata", metadata, flat)

        plugin = plugin_registry.get(family)
        if plugin is None:
            issues.append(
                {
                    "code": "unknown_family",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": f"No family plugin registered for '{family}'.",
                }
            )
        else:
            for plugin_issue in plugin.validate_runtime(record=entry):
                issues.append(
                    {
                        "code": "plugin_validation",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": str(plugin_issue),
                    }
                )

        if proof_mode and str(metadata.get("mode", "")).strip().lower() == "mock":
            issues.append(
                {
                    "code": "mock_mode_forbidden",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": "Mock mode is forbidden in proof mode.",
                }
            )

        if system == "worldflux":
            policy_mode = str(metadata.get("policy_mode", "")).strip()
            policy_impl = str(metadata.get("policy_impl", "")).strip().lower()
            policy_name = str(metadata.get("policy", "")).strip().lower()

            if proof_mode and policy_mode != policy_mode_expected:
                issues.append(
                    {
                        "code": "policy_mode_mismatch",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": (
                            f"worldflux policy_mode='{policy_mode}' does not match expected "
                            f"'{policy_mode_expected}'"
                        ),
                    }
                )

            if proof_mode and (policy_name == "random" or "random" in policy_impl):
                issues.append(
                    {
                        "code": "random_policy_forbidden",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "Random policy is forbidden for proof parity.",
                    }
                )
            if (
                proof_mode
                and family == "dreamerv3"
                and not bool(metadata.get("strict_official_semantics", False))
            ):
                issues.append(
                    {
                        "code": "strict_semantics_required",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "Dreamer proof mode requires strict_official_semantics=true.",
                    }
                )
            if proof_mode and family == "dreamerv3" and policy_impl != "candidate_actor_stateful":
                issues.append(
                    {
                        "code": "policy_impl_mismatch",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": (
                            "Dreamer proof mode requires actor policy only; "
                            f"got policy_impl={policy_impl!r}"
                        ),
                    }
                )

        expected_model_id, expected_model_profile = _expected_model_identity(
            family=family,
            system=system,
        )
        if proof_mode:
            expected_adapter_id = str(
                entry.get("expected_adapter_id", entry.get("adapter", ""))
            ).strip()
            expected_backend_kind = str(entry.get("expected_backend_kind", "")).strip()
            actual_adapter_id = str(
                entry.get("adapter_id") or metadata.get("adapter_id", "")
            ).strip()
            actual_backend_kind = str(
                entry.get("backend_kind") or metadata.get("backend_kind", "")
            ).strip()
            recipe_hash = str(entry.get("recipe_hash") or metadata.get("recipe_hash", "")).strip()
            artifact_manifest = entry.get("artifact_manifest")
            if not isinstance(artifact_manifest, dict):
                artifact_manifest = metadata.get("artifact_manifest")

            if expected_adapter_id and actual_adapter_id != expected_adapter_id:
                issues.append(
                    {
                        "code": "adapter_id_mismatch",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": (
                            f"adapter_id mismatch: expected {expected_adapter_id!r}, got {actual_adapter_id!r}"
                        ),
                    }
                )
            if expected_backend_kind and actual_backend_kind != expected_backend_kind:
                issues.append(
                    {
                        "code": "backend_kind_mismatch",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": (
                            "backend_kind mismatch: "
                            f"expected {expected_backend_kind!r}, got {actual_backend_kind!r}"
                        ),
                    }
                )
            if not recipe_hash:
                issues.append(
                    {
                        "code": "missing_recipe_hash",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "recipe_hash missing from metrics metadata.",
                    }
                )
            if not isinstance(artifact_manifest, dict):
                issues.append(
                    {
                        "code": "missing_artifact_manifest",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "artifact_manifest missing from metrics metadata.",
                    }
                )

            expected_train_budget = _entry_recipe(entry, "train_budget")
            expected_eval_protocol = _entry_recipe(entry, "eval_protocol")
            actual_train_budget = _metadata_recipe(metadata, "effective_recipe") or metadata.get(
                "train_budget"
            )
            actual_eval_protocol = metadata.get("eval_protocol")

            if system == "worldflux":
                if not isinstance(actual_train_budget, dict):
                    issues.append(
                        {
                            "code": "missing_recipe_metadata",
                            "task_id": task_id,
                            "seed": seed,
                            "system": system,
                            "message": "train_budget missing from metrics metadata.",
                        }
                    )
                else:
                    _compare_recipe_fields(
                        issues=issues,
                        task_id=task_id,
                        seed=seed,
                        system=system,
                        category="train_budget",
                        expected=expected_train_budget,
                        actual=actual_train_budget,
                    )

                if not isinstance(actual_eval_protocol, dict):
                    issues.append(
                        {
                            "code": "missing_recipe_metadata",
                            "task_id": task_id,
                            "seed": seed,
                            "system": system,
                            "message": "eval_protocol missing from metrics metadata.",
                        }
                    )
                else:
                    filtered_expected_eval_protocol = {
                        key: value
                        for key, value in expected_eval_protocol.items()
                        if key not in {"policy_mode", "environment_backend"}
                    }
                    _compare_recipe_fields(
                        issues=issues,
                        task_id=task_id,
                        seed=seed,
                        system=system,
                        category="eval_protocol",
                        expected=filtered_expected_eval_protocol,
                        actual=actual_eval_protocol,
                    )

            if (
                not str(entry.get("source_commit", "")).strip()
                or not str(entry.get("source_artifact_path", "")).strip()
            ):
                issues.append(
                    {
                        "code": "missing_source_provenance",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "source_commit/source_artifact_path must be present in proof mode.",
                    }
                )

            model_id = str(metadata.get("model_id", "")).strip()
            model_profile = str(metadata.get("model_profile", "")).strip().lower()
            if not model_id or not model_profile:
                issues.append(
                    {
                        "code": "missing_recipe_metadata",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": "model_id/model_profile missing from metrics metadata.",
                    }
                )
            else:
                if expected_model_id and model_id != expected_model_id:
                    issues.append(
                        {
                            "code": "model_id_mismatch",
                            "task_id": task_id,
                            "seed": seed,
                            "system": system,
                            "message": (
                                f"model_id mismatch: expected {expected_model_id!r}, got {model_id!r}"
                            ),
                        }
                    )
                if expected_model_profile and model_profile != expected_model_profile:
                    issues.append(
                        {
                            "code": "model_profile_mismatch",
                            "task_id": task_id,
                            "seed": seed,
                            "system": system,
                            "message": (
                                "model_profile mismatch: "
                                f"expected {expected_model_profile!r}, got {model_profile!r}"
                            ),
                        }
                    )
                if family == "tdmpc2":
                    canonical_compare_profile = (
                        str(metadata.get("canonical_compare_profile", "")).strip().lower()
                    )
                    if canonical_compare_profile != "proof_5m":
                        issues.append(
                            {
                                "code": "canonical_profile_missing",
                                "task_id": task_id,
                                "seed": seed,
                                "system": system,
                                "message": (
                                    "TD-MPC2 proof compare requires canonical_compare_profile='proof_5m'."
                                ),
                            }
                        )
                    if system == "worldflux":
                        alignment_status = str(metadata.get("alignment_status", "")).strip().lower()
                        alignment_report_path = str(
                            metadata.get("alignment_report_path", "")
                        ).strip()
                        if not alignment_report_path:
                            issues.append(
                                {
                                    "code": "alignment_report_missing",
                                    "task_id": task_id,
                                    "seed": seed,
                                    "system": system,
                                    "message": "TD-MPC2 proof compare requires alignment_report_path metadata.",
                                }
                            )
                        if alignment_status != "aligned":
                            issues.append(
                                {
                                    "code": "alignment_status_mismatch",
                                    "task_id": task_id,
                                    "seed": seed,
                                    "system": system,
                                    "message": (
                                        "TD-MPC2 proof compare requires alignment_status='aligned'."
                                    ),
                                }
                            )

        env_backend = str(metadata.get("env_backend", "")).strip().lower()
        effective_env_backend = env_backend
        if proof_mode and system == "official" and not effective_env_backend:
            issues.append(
                {
                    "code": "missing_env_backend",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": "metadata.env_backend is required in proof mode.",
                }
            )
        elif not proof_mode and not effective_env_backend and system == "official":
            inferred_backend = _infer_legacy_official_backend(task_id=task_id, family=family)
            if inferred_backend:
                effective_env_backend = inferred_backend
                issues.append(
                    {
                        "code": "official_env_backend_inferred",
                        "severity": "info",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": (
                            "metadata.env_backend is missing on legacy official record; "
                            f"inferred '{inferred_backend}' from task/family."
                        ),
                    }
                )

        if (
            proof_mode
            and required_backend not in {"", "auto"}
            and effective_env_backend != required_backend
        ):
            issues.append(
                {
                    "code": "environment_backend_mismatch",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": (
                        "env_backend="
                        f"'{effective_env_backend}' does not match required '{required_backend}'"
                    ),
                }
            )

        for rule in forbidden_rules:
            if _match_forbidden(flat, str(rule)):
                issues.append(
                    {
                        "code": "forbidden_shortcut_detected",
                        "task_id": task_id,
                        "seed": seed,
                        "system": system,
                        "message": f"forbidden shortcut matched: {rule}",
                    }
                )

        protocol_hash = str(metadata.get("eval_protocol_hash", "")).strip()
        pair_key = (task_id, seed)
        if pair_key not in protocol_hashes:
            protocol_hashes[pair_key] = {}
        if protocol_hash:
            protocol_hashes[pair_key][system] = protocol_hash
        elif proof_mode:
            issues.append(
                {
                    "code": "missing_eval_protocol_hash",
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "message": "Missing eval_protocol_hash in successful run metadata.",
                }
            )

    for (task_id, seed), values in sorted(protocol_hashes.items()):
        off = values.get("official")
        wf = values.get("worldflux")
        if off and wf and off != wf:
            issues.append(
                {
                    "code": "eval_protocol_hash_mismatch",
                    "task_id": task_id,
                    "seed": seed,
                    "system": "both",
                    "message": f"eval protocol hash differs: official={off}, worldflux={wf}",
                }
            )

    error_issues = [issue for issue in issues if str(issue.get("severity", "error")) != "info"]
    payload = {
        "schema_version": "parity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proof_mode": bool(proof_mode),
        "required_policy_mode": str(required_policy_mode),
        "requirements": {
            "environment_backend": default_backend,
            "forbidden_shortcuts": [str(item) for item in default_forbidden],
        },
        "issue_count": len(error_issues),
        "info_count": len(issues) - len(error_issues),
        "issues": issues,
        "pass": len(error_issues) == 0,
    }
    return payload


__all__ = ["evaluate_validity"]
