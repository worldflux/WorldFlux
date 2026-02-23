"""Minimal HTTP client for WorldFlux Cloud APIs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from .auth import get_api_key, set_api_key


class WorldFluxCloudClient:
    """Small JSON-over-HTTP client for cloud training and verification."""

    def __init__(self, *, base_url: str, api_key: str | None = None, timeout_sec: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = int(timeout_sec)

    @classmethod
    def from_env(cls) -> WorldFluxCloudClient:
        """Create a cloud client from environment defaults and local credentials."""
        base_url = os.environ.get("WORLDFLUX_CLOUD_API_URL", "https://api.worldflux.ai")
        api_key = os.environ.get("WORLDFLUX_CLOUD_API_KEY") or get_api_key()
        timeout_sec = int(os.environ.get("WORLDFLUX_CLOUD_TIMEOUT_SEC", "30"))
        return cls(base_url=base_url, api_key=api_key, timeout_sec=timeout_sec)

    def login(self, *, api_key: str) -> None:
        """Store API key for future CLI sessions."""
        normalized = api_key.strip()
        if not normalized:
            raise RuntimeError("API key must not be empty.")
        set_api_key(normalized)
        self.api_key = normalized

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an HTTP request and decode a JSON object response."""
        if query:
            query_text = parse.urlencode(
                {k: v for k, v in query.items() if v is not None},
                doseq=True,
            )
            url = f"{self.base_url}{path}?{query_text}"
        else:
            url = f"{self.base_url}{path}"
        self._validate_http_url(url)

        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(url=url, data=body, method=method.upper(), headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:  # nosec B310
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(
                f"Cloud API request failed ({method.upper()} {path}, status={exc.code}): {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Cloud API request failed ({method.upper()} {path}): {exc}"
            ) from exc

        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Cloud API returned non-object payload for {method.upper()} {path}: {type(parsed)!r}"
            )
        return parsed

    @staticmethod
    def _validate_http_url(url: str) -> None:
        parsed = parse.urlparse(url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise RuntimeError(f"Invalid cloud API URL: {url!r}")

    def create_training_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", "/v1/jobs", payload=payload)

    def list_jobs(self) -> list[dict[str, Any]]:
        payload = self.request_json("GET", "/v1/jobs")
        jobs = payload.get("jobs", [])
        if isinstance(jobs, list):
            return [item for item in jobs if isinstance(item, dict)]
        return []

    def get_job_logs(self, job_id: str, *, limit: int = 200) -> list[str]:
        payload = self.request_json("GET", f"/v1/jobs/{job_id}/logs", query={"limit": limit})
        logs = payload.get("logs", [])
        if not isinstance(logs, list):
            return []
        return [str(line) for line in logs]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self.request_json("DELETE", f"/v1/jobs/{job_id}")

    def verify_cloud(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", "/v1/verify", payload=payload)

    def pull_job_artifacts(self, job_id: str, *, output_dir: str | Path) -> dict[str, Any]:
        """Fetch artifact manifest and save to local disk.

        Current MVP stores the JSON manifest for deterministic retrieval.
        """
        payload = self.request_json("GET", f"/v1/jobs/{job_id}/artifacts")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / f"{job_id}.artifacts.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"manifest": str(manifest_path.resolve()), "artifact_index": payload}
