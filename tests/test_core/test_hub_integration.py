"""Hub integration tests for WorldModel helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from worldflux import create_world_model


def test_world_model_push_to_hub_uploads_saved_artifacts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            calls["token"] = token

        def create_repo(
            self,
            *,
            repo_id: str,
            private: bool | None = None,
            repo_type: str = "model",
            exist_ok: bool = True,
        ) -> None:
            calls["create_repo"] = {
                "repo_id": repo_id,
                "private": private,
                "repo_type": repo_type,
                "exist_ok": exist_ok,
            }

        def upload_folder(
            self,
            *,
            repo_id: str,
            repo_type: str,
            folder_path: str,
            commit_message: str,
            token: str | None = None,
        ) -> None:
            path = Path(folder_path)
            calls["upload_folder"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "commit_message": commit_message,
                "token": token,
                "files": sorted(p.name for p in path.glob("*")),
            }

    fake_module = types.SimpleNamespace(HfApi=_FakeHfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    model = create_world_model("dreamer:ci", obs_shape=(3, 64, 64), action_dim=6)
    url = model.push_to_hub(
        "worldflux/test-model",
        token="hf_test_token",
        private=True,
        commit_message="test upload",
    )

    assert url == "https://huggingface.co/worldflux/test-model"
    assert calls["token"] == "hf_test_token"
    assert calls["create_repo"] == {
        "repo_id": "worldflux/test-model",
        "private": True,
        "repo_type": "model",
        "exist_ok": True,
    }
    uploaded = calls["upload_folder"]
    assert isinstance(uploaded, dict)
    assert uploaded["repo_id"] == "worldflux/test-model"
    assert uploaded["repo_type"] == "model"
    assert uploaded["commit_message"] == "test upload"
    assert uploaded["token"] == "hf_test_token"
    assert "config.json" in uploaded["files"]
    assert "model.pt" in uploaded["files"]
    assert "worldflux_meta.json" in uploaded["files"]
