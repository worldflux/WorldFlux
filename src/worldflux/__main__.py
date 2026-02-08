"""Module entry point for ``python -m worldflux``."""

from __future__ import annotations


def main() -> None:
    try:
        from .cli import app
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0].lower()
        if missing in {"typer", "rich", "inquirerpy"}:
            print("WorldFlux CLI dependencies are not installed. Run: uv pip install -e '.[cli]'")
            raise SystemExit(1) from exc
        raise

    app(prog_name="worldflux")


if __name__ == "__main__":
    main()
