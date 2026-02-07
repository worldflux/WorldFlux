from __future__ import annotations

import base64
import json
import threading
import time
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

import numpy as np

from worldflux.training.callbacks import Callback


class MetricPoint(TypedDict):
    step: int
    timestamp: float
    speed: float
    metrics: dict[str, float]


class MetricBuffer:
    """Thread-safe ring buffer for training metrics."""

    def __init__(self, max_points: int = 2000):
        self._points: deque[MetricPoint] = deque(maxlen=max(1, int(max_points)))
        self._lock = threading.Lock()
        self._status = "running"
        self._phase = "starting"
        self._phase_message: str | None = None
        self._gameplay_available = False
        self._error: str | None = None
        self._started_at = time.time()
        self._ended_at: float | None = None
        self._latest_step = 0
        self._latest_metrics: dict[str, float] = {}
        self._latest_speed = 0.0

    def append(
        self,
        *,
        step: int,
        timestamp: float,
        speed: float,
        metrics: dict[str, float],
    ) -> MetricPoint:
        point: MetricPoint = {
            "step": int(step),
            "timestamp": float(timestamp),
            "speed": float(speed),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        with self._lock:
            self._points.append(point)
            self._latest_step = point["step"]
            self._latest_metrics = dict(point["metrics"])
            self._latest_speed = point["speed"]
        return point

    def set_status(self, status: str, error: str | None = None) -> None:
        now = time.time()
        normalized = status if status in {"running", "finished", "error"} else "error"
        with self._lock:
            self._status = normalized
            if normalized == "error":
                self._phase = "error"
            if error:
                self._error = str(error)
            if normalized in {"finished", "error"}:
                self._ended_at = now

    def set_phase(self, phase: str, message: str | None = None) -> None:
        with self._lock:
            self._phase = phase
            self._phase_message = message

    def set_gameplay_available(self, available: bool) -> None:
        with self._lock:
            self._gameplay_available = bool(available)

    def metrics_payload(self, since_step: int = -1) -> dict[str, Any]:
        with self._lock:
            points = [p for p in self._points if p["step"] > since_step]
            payload_points = [
                {
                    "step": p["step"],
                    "timestamp": p["timestamp"],
                    "speed": p["speed"],
                    "metrics": dict(p["metrics"]),
                }
                for p in points
            ]
            return {
                "status": self._status,
                "phase": self._phase,
                "latest_step": self._latest_step,
                "points": payload_points,
            }

    def summary_payload(self, *, host: str, port: int) -> dict[str, Any]:
        with self._lock:
            ended_at = self._ended_at
            now = time.time()
            return {
                "status": self._status,
                "phase": self._phase,
                "phase_message": self._phase_message,
                "gameplay_available": self._gameplay_available,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "elapsed_seconds": (ended_at or now) - self._started_at,
                "latest_step": self._latest_step,
                "latest_metrics": dict(self._latest_metrics),
                "latest_speed": self._latest_speed,
                "error": self._error,
                "host": host,
                "port": int(port),
                "total_points": len(self._points),
            }


class GameplayFrame(TypedDict):
    seq: int
    timestamp: float
    width: int
    height: int
    rgb_b64: str
    episode: int
    episode_step: int
    reward: float
    done: bool


class GameplayBuffer:
    """Thread-safe ring buffer for gameplay frames."""

    def __init__(self, max_frames: int = 512, fps: int = 8):
        self._frames: deque[GameplayFrame] = deque(maxlen=max(1, int(max_frames)))
        self._fps = max(1, int(fps))
        self._lock = threading.Lock()
        self._latest_seq = 0
        self._status = "running"
        self._phase = "starting"
        self._detail: str | None = None

    @property
    def fps(self) -> int:
        return self._fps

    def set_phase(self, phase: str, detail: str | None = None) -> None:
        with self._lock:
            self._phase = phase
            self._detail = detail

    def set_status(self, status: str) -> None:
        normalized = (
            status if status in {"running", "finished", "error", "unavailable"} else "error"
        )
        with self._lock:
            self._status = normalized

    def append_frame(
        self,
        frame: Any,
        *,
        episode: int,
        episode_step: int,
        reward: float,
        done: bool,
    ) -> None:
        rgb = self._normalize_frame(frame)
        if rgb is None:
            return

        now = time.time()
        with self._lock:
            self._latest_seq += 1
            encoded = base64.b64encode(rgb.tobytes()).decode("ascii")
            self._frames.append(
                {
                    "seq": self._latest_seq,
                    "timestamp": now,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "rgb_b64": encoded,
                    "episode": int(episode),
                    "episode_step": int(episode_step),
                    "reward": float(reward),
                    "done": bool(done),
                }
            )

    def payload(self, since_seq: int = -1) -> dict[str, Any]:
        with self._lock:
            frames = [frame for frame in self._frames if frame["seq"] > since_seq]
            return {
                "status": self._status,
                "phase": self._phase,
                "detail": self._detail,
                "latest_seq": self._latest_seq,
                "fps": self._fps,
                "frames": [dict(frame) for frame in frames],
            }

    @staticmethod
    def _normalize_frame(frame: Any) -> np.ndarray | None:
        arr = np.asarray(frame)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[-1] >= 3:
                arr = arr[:, :, :3]
        else:
            return None

        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None

        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.max(arr))
            if max_value <= 1.0:
                arr = arr * 255.0

        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)


class DashboardCallback(Callback):
    """Trainer callback that streams metrics into a MetricBuffer and JSONL file."""

    def __init__(self, buffer: MetricBuffer, jsonl_path: Path):
        self.buffer = buffer
        self.jsonl_path = jsonl_path
        self._last_time: float | None = None
        self._last_step: int | None = None
        self._handle: Any = None

    def on_train_begin(self, trainer) -> None:  # type: ignore[override]
        del trainer
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._last_time = None
        self._last_step = None
        self.buffer.set_status("running")

    def on_step_end(self, trainer) -> None:  # type: ignore[override]
        step = int(trainer.state.global_step)
        now = time.time()
        metrics = {k: float(v) for k, v in trainer.state.metrics.items()}

        if self._last_time is None or self._last_step is None:
            speed = 0.0
        else:
            dt = now - self._last_time
            ds = step - self._last_step
            speed = float(ds / dt) if dt > 0 and ds > 0 else 0.0

        self._last_time = now
        self._last_step = step

        point = self.buffer.append(step=step, timestamp=now, speed=speed, metrics=metrics)
        if self._handle is not None:
            self._handle.write(json.dumps(point, separators=(",", ":")) + "\n")

    def on_train_end(self, trainer) -> None:  # type: ignore[override]
        del trainer
        self.buffer.set_status("finished")
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class _DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class,
        *,
        metric_buffer: MetricBuffer,
        gameplay_buffer: GameplayBuffer | None,
        dashboard_root: Path,
        bind_host: str,
        refresh_ms: int,
    ):
        super().__init__(server_address, request_handler_class)
        self.metric_buffer = metric_buffer
        self.gameplay_buffer = gameplay_buffer
        self.dashboard_root = dashboard_root
        self.bind_host = bind_host
        self.refresh_ms = int(refresh_ms)


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._serve_file(self.server.dashboard_root / "index.html", "text/html; charset=utf-8")
            return

        if path == "/api/metrics":
            query = parse_qs(parsed.query)
            raw = query.get("since_step", ["-1"])[0]
            try:
                since_step = int(raw)
            except (TypeError, ValueError):
                since_step = -1
            self._send_json(self.server.metric_buffer.metrics_payload(since_step=since_step))
            return

        if path == "/api/summary":
            payload = self.server.metric_buffer.summary_payload(
                host=self.server.bind_host,
                port=self.server.server_port,
            )
            if self.server.gameplay_buffer is not None:
                payload["gameplay_fps"] = self.server.gameplay_buffer.fps
            self._send_json(
                {
                    **payload,
                    "refresh_ms": self.server.refresh_ms,
                }
            )
            return

        if path == "/api/gameplay":
            query = parse_qs(parsed.query)
            raw = query.get("since_seq", ["-1"])[0]
            try:
                since_seq = int(raw)
            except (TypeError, ValueError):
                since_seq = -1

            if self.server.gameplay_buffer is None:
                self._send_json(
                    {
                        "status": "unavailable",
                        "phase": "unavailable",
                        "detail": "Gameplay stream is disabled.",
                        "latest_seq": 0,
                        "fps": 0,
                        "frames": [],
                    }
                )
                return

            self._send_json(self.server.gameplay_buffer.payload(since_seq=since_seq))
            return

        if path == "/healthz":
            self._send_text("ok\n", status=HTTPStatus.OK)
            return

        self._send_text("not found\n", status=HTTPStatus.NOT_FOUND)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self._send_text("not found\n", status=HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, status: HTTPStatus) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args


class MetricsDashboardServer:
    """Small local HTTP server for visualizing training metrics."""

    def __init__(
        self,
        *,
        metric_buffer: MetricBuffer,
        gameplay_buffer: GameplayBuffer | None = None,
        host: str,
        start_port: int,
        dashboard_root: Path,
        refresh_ms: int,
        max_port_tries: int = 100,
    ):
        self.metric_buffer = metric_buffer
        self.gameplay_buffer = gameplay_buffer
        self.host = host
        self.start_port = int(start_port)
        self.dashboard_root = dashboard_root
        self.refresh_ms = max(100, int(refresh_ms))
        self.max_port_tries = max(1, int(max_port_tries))

        self._server: _DashboardHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stopped_event = threading.Event()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("dashboard server is not started")
        return int(self._server.server_port)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> int:
        if self._server is not None:
            return self.port

        index_path = self.dashboard_root / "index.html"
        if not index_path.exists():
            raise RuntimeError(f"dashboard file not found: {index_path}")

        last_error: Exception | None = None
        for offset in range(self.max_port_tries):
            candidate = self.start_port + offset
            try:
                self._server = _DashboardHTTPServer(
                    (self.host, candidate),
                    _DashboardRequestHandler,
                    metric_buffer=self.metric_buffer,
                    gameplay_buffer=self.gameplay_buffer,
                    dashboard_root=self.dashboard_root,
                    bind_host=self.host,
                    refresh_ms=self.refresh_ms,
                )
                break
            except OSError as exc:
                last_error = exc

        if self._server is None:
            raise RuntimeError(
                f"failed to bind dashboard port after {self.max_port_tries} attempts"
            ) from last_error

        self._stopped_event.clear()
        self._thread = threading.Thread(target=self._serve, name="metrics-dashboard", daemon=False)
        self._thread.start()
        return self.port

    def _serve(self) -> None:
        assert self._server is not None
        try:
            self._server.serve_forever(poll_interval=0.5)
        finally:
            self._stopped_event.set()

    def open_browser(self) -> None:
        webbrowser.open(self.url)

    def stop(self) -> None:
        server = self._server
        if server is None:
            self._stopped_event.set()
            return
        server.shutdown()
        server.server_close()
        self._server = None
        self._stopped_event.set()

    def schedule_shutdown(self, delay_seconds: float) -> None:
        def _shutdown_later() -> None:
            time.sleep(max(0.0, float(delay_seconds)))
            self.stop()

        thread = threading.Thread(
            target=_shutdown_later, name="metrics-dashboard-shutdown", daemon=True
        )
        thread.start()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        return self._stopped_event.wait(timeout=timeout)
