"""Tests for the CLI theme and rich_output helpers."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("rich") is None:
    pytest.skip("CLI dependencies are not installed", allow_module_level=True)

from worldflux.cli._theme import PALETTE, STATUS_ICONS, WF_THEME, ColorPalette

# ---------------------------------------------------------------------------
# _theme.py
# ---------------------------------------------------------------------------


class TestColorPalette:
    def test_palette_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            PALETTE.primary = "#000000"  # type: ignore[misc]

    def test_palette_defaults_are_hex(self) -> None:
        for field in ColorPalette.__dataclass_fields__:
            value = getattr(PALETTE, field)
            assert value.startswith("#"), f"{field} should be a hex color"

    def test_palette_is_singleton(self) -> None:
        from worldflux.cli._theme import PALETTE as P2

        assert PALETTE is P2


class TestWFTheme:
    def test_theme_has_required_styles(self) -> None:
        required = [
            "wf.header",
            "wf.label",
            "wf.value",
            "wf.muted",
            "wf.dim",
            "wf.pass",
            "wf.fail",
            "wf.warn",
            "wf.info",
            "wf.ok",
            "wf.err",
            "wf.caution",
            "wf.brand",
            "wf.accent",
            "wf.border",
            "wf.border.success",
            "wf.border.error",
            "wf.border.info",
        ]
        for name in required:
            assert name in WF_THEME.styles, f"Missing theme style: {name}"

    def test_console_uses_theme(self) -> None:
        from rich.console import Console

        c = Console(theme=WF_THEME)
        # Should not raise
        c.print("[wf.pass]ok[/wf.pass]", end="")


class TestStatusIcons:
    def test_has_all_keys(self) -> None:
        assert set(STATUS_ICONS) == {"pass", "fail", "warn", "info"}


# ---------------------------------------------------------------------------
# _rich_output.py
# ---------------------------------------------------------------------------


class TestRichOutput:
    def test_key_value_panel_renders(self) -> None:
        from worldflux.cli._rich_output import key_value_panel

        panel = key_value_panel({"Key1": "Val1", "Key2": "Val2"}, title="Test")
        assert panel.title is not None

    def test_status_table_renders(self) -> None:
        from worldflux.cli._rich_output import status_table

        rows = [("pass", "Comp", "v1"), ("warn", "Missing", "none")]
        table = status_table(rows, title="Status")
        assert table.title == "Status"
        assert table.row_count == 2

    def test_result_banner_pass(self) -> None:
        from worldflux.cli._rich_output import result_banner

        panel = result_banner(passed=True, lines=["line1"])
        assert panel.title is not None
        assert "\u2713" in str(panel.title)

    def test_result_banner_fail(self) -> None:
        from worldflux.cli._rich_output import result_banner

        panel = result_banner(passed=False, lines=["line1"])
        assert panel.title is not None
        assert "\u2717" in str(panel.title)

    def test_metric_table_renders(self) -> None:
        from worldflux.cli._rich_output import metric_table

        rows = [
            ("mse", "0.0012", "0.01", True),
            ("mae", "0.0500", "0.03", False),
            ("info_metric", "1.0", "-", None),
        ]
        table = metric_table(rows, title="Eval")
        assert table.row_count == 3

    def test_section_header_renders(self) -> None:
        from io import StringIO

        from rich.console import Console

        from worldflux.cli._rich_output import section_header
        from worldflux.cli._theme import WF_THEME

        buf = StringIO()
        test_console = Console(file=buf, theme=WF_THEME, width=60)

        import worldflux.cli._rich_output as _ro

        original_console = _ro.console
        _ro.console = test_console
        try:
            section_header("Test Section")
        finally:
            _ro.console = original_console

        output = buf.getvalue()
        assert "Test Section" in output
