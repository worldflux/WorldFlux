"""Enforce per-file coverage thresholds for critical runtime modules."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CRITICAL_THRESHOLDS: dict[str, float] = {
    "src/worldflux/__main__.py": 90.0,
    # NOTE:
    # The CLI surface grew substantially with additional command groups.
    # Keep this gate strict-but-realistic for the current critical test suite.
    "src/worldflux/cli.py": 65.0,
    "src/worldflux/samplers/token.py": 95.0,
    "src/worldflux/samplers/diffusion.py": 90.0,
    "src/worldflux/training/callbacks.py": 85.0,
    "src/worldflux/training/trainer.py": 82.0,
    "src/worldflux/training/data.py": 77.5,
}


def _normalize_filename(raw: str) -> str:
    filename = raw.replace("\\", "/")
    if "/src/worldflux/" in filename:
        return filename[filename.index("src/worldflux/") :]
    if filename.startswith("src/worldflux/"):
        return filename
    if "/worldflux/" in filename:
        return "src/" + filename[filename.index("worldflux/") :]
    if filename.startswith("worldflux/"):
        return "src/" + filename
    return filename


def _collect_file_coverage(report_path: Path) -> dict[str, tuple[int, int]]:
    tree = ET.parse(report_path)
    root = tree.getroot()
    coverage: dict[str, tuple[int, int]] = {}

    for class_elem in root.findall(".//class"):
        filename = _normalize_filename(class_elem.get("filename", ""))
        if not filename:
            continue
        line_elems = class_elem.findall("./lines/line")
        if line_elems:
            total = len(line_elems)
            hits = sum(1 for line in line_elems if int(line.get("hits", "0")) > 0)
        else:
            rate = float(class_elem.get("line-rate", "0.0"))
            total = 100
            hits = int(rate * total)

        old_hits, old_total = coverage.get(filename, (0, 0))
        coverage[filename] = (old_hits + hits, old_total + total)

    return coverage


def _pct(hits: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (hits / total) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to coverage XML report (default: coverage.xml)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"[coverage] report not found: {args.report}")
        return 1

    coverage = _collect_file_coverage(args.report)

    failures: list[str] = []
    print("[coverage] critical module thresholds")
    for file_path, threshold in CRITICAL_THRESHOLDS.items():
        if file_path not in coverage:
            failures.append(f"{file_path}: missing from coverage report")
            print(f"- {file_path}: MISSING (required >= {threshold:.1f}%)")
            continue
        hits, total = coverage[file_path]
        actual = _pct(hits, total)
        status = "OK" if actual >= threshold else "FAIL"
        print(f"- {file_path}: {actual:.2f}% (required >= {threshold:.1f}%) [{status}]")
        if actual < threshold:
            failures.append(f"{file_path}: {actual:.2f}% below required threshold {threshold:.1f}%")

    if failures:
        print("\n[coverage] critical coverage check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\n[coverage] all critical thresholds satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
