"""Serialization mixin for WorldModel (save/load/push to hub)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import torch

from ..exceptions import CheckpointError, ValidationError


class SerializationMixin:
    """Mixin providing save_pretrained / push_to_hub / contract_fingerprint.

    Requires the host class to have:
    - ``config`` attribute with ``.save(path)`` method.
    - ``state_dict()`` from ``nn.Module``.
    - ``io_contract()`` from the base ``WorldModel``.
    - ``_get_api_version()`` from the base ``WorldModel``.
    """

    def save_pretrained(self: Any, path: str) -> None:
        """Save model weights and config using a unified directory layout."""
        os.makedirs(path, exist_ok=True)

        config = getattr(self, "config", None)
        if config is None or not hasattr(config, "save"):
            raise ValidationError(
                f"{self.__class__.__name__} cannot be saved because config.save(...) is unavailable."
            )
        config.save(os.path.join(path, "config.json"))

        weights_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), weights_path)

        # Compute SHA-256 hash of saved weights for integrity verification.
        weights_hash = _sha256_file(weights_path)

        metadata: dict[str, Any] = {
            "save_format_version": 1,
            "worldflux_version": _resolve_worldflux_version(),
            "api_version": self._get_api_version(),
            "model_type": str(getattr(config, "model_type", self.__class__.__name__)),
            "contract_fingerprint": self.contract_fingerprint(),
            "weights_sha256": weights_hash,
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        with open(os.path.join(path, "worldflux_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    def push_to_hub(
        self: Any,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        commit_message: str | None = None,
    ) -> str:
        """Upload this model to the Hugging Face Hub.

        Requires optional dependency group ``worldflux[hub]``.
        """
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise ValidationError(
                "Hugging Face Hub support requires optional dependency "
                '`huggingface_hub`. Install with: uv pip install "worldflux[hub]"'
            ) from exc

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="worldflux-hub-") as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmpdir,
                commit_message=commit_message or f"Upload {self.__class__.__name__} from WorldFlux",
                token=token,
            )

        return f"https://huggingface.co/{repo_id}"

    def contract_fingerprint(self: Any) -> str:
        """Return a stable fingerprint for this model's declared IO contract."""
        contract = _normalize_contract_value(self.io_contract())
        serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> Any:
        """Load a model from a pretrained checkpoint or registry name."""
        from ..registry import WorldModelRegistry

        model = WorldModelRegistry.from_pretrained(name_or_path, **kwargs)

        # Verify checkpoint integrity if loading from a directory.
        meta_path = os.path.join(name_or_path, "worldflux_meta.json")
        weights_path = os.path.join(name_or_path, "model.pt")
        if os.path.isfile(meta_path) and os.path.isfile(weights_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            expected_hash = meta.get("weights_sha256")
            if expected_hash:
                actual_hash = _sha256_file(weights_path)
                if actual_hash != expected_hash:
                    raise CheckpointError(
                        f"Checkpoint integrity check failed for {weights_path}: "
                        f"expected SHA-256 {expected_hash[:16]}..., "
                        f"got {actual_hash[:16]}..."
                    )

        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(model).__name__}. "
                f"Check that the model type in the config matches."
            )
        return model


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_contract_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _normalize_contract_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _normalize_contract_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize_contract_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_contract_value(v) for v in value]
    return value


def _resolve_worldflux_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("worldflux")
    except (PackageNotFoundError, ImportError, ValueError):
        return "0.1.1.dev0"
