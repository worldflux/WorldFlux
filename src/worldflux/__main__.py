"""Module entry point for ``python -m worldflux``."""

from __future__ import annotations


def main() -> None:
    try:
        from .cli import app
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0].lower()
        if missing in {"typer", "rich"}:
            print(
                "WorldFlux CLI dependencies are missing from this environment. "
                "Reinstall with: uv pip install -U worldflux"
            )
            raise SystemExit(1) from exc
        raise

    app(prog_name="worldflux")


if __name__ == "__main__":
    main()
