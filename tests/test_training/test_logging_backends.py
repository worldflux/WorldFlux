# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for multi-backend logging."""

from __future__ import annotations

import csv
from pathlib import Path


class TestCSVBackend:
    def test_log_scalar_creates_file(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        backend.log_scalar("loss", 0.5, step=1)
        backend.log_scalar("loss", 0.4, step=2)
        backend.flush()
        backend.close()

        csv_file = tmp_path / "scalars.csv"
        assert csv_file.exists()
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["tag"] == "loss"
        assert float(rows[0]["value"]) == 0.5
        assert int(rows[0]["step"]) == 1

    def test_append_mode(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        backend.log_scalar("a", 1.0, step=0)
        backend.flush()
        backend.log_scalar("b", 2.0, step=1)
        backend.flush()
        backend.close()

        csv_file = tmp_path / "scalars.csv"
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_empty_flush(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        backend.flush()  # Should not create file
        assert not (tmp_path / "scalars.csv").exists()

    def test_histogram_noop(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        backend.log_histogram("h", [1, 2, 3], step=0)  # Should not raise

    def test_image_noop(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        backend.log_image("img", None, step=0)  # Should not raise


class TestCompositeBackend:
    def test_routes_to_all(self, tmp_path: Path):
        from worldflux.training.logging_backends import CompositeBackend, CSVBackend

        b1 = CSVBackend(log_dir=str(tmp_path / "a"))
        b2 = CSVBackend(log_dir=str(tmp_path / "b"))
        composite = CompositeBackend([b1, b2])
        composite.log_scalar("x", 1.0, step=0)
        composite.flush()
        composite.close()
        assert (tmp_path / "a" / "scalars.csv").exists()
        assert (tmp_path / "b" / "scalars.csv").exists()


class TestLoggingBackendProtocol:
    def test_csv_satisfies_protocol(self, tmp_path: Path):
        from worldflux.training.logging_backends import CSVBackend, LoggingBackend

        backend = CSVBackend(log_dir=str(tmp_path))
        assert isinstance(backend, LoggingBackend)

    def test_composite_satisfies_protocol(self, tmp_path: Path):
        from worldflux.training.logging_backends import CompositeBackend, LoggingBackend

        backend = CompositeBackend([])
        assert isinstance(backend, LoggingBackend)

    def test_tensorboard_class_exists(self):
        from worldflux.training.logging_backends import TensorBoardBackend

        assert TensorBoardBackend is not None

    def test_wandb_class_exists(self):
        from worldflux.training.logging_backends import WandbBackend

        assert WandbBackend is not None
