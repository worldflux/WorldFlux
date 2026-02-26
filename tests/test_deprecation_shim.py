"""Tests for the top-level deprecation shim in worldflux.__init__."""

from __future__ import annotations

import warnings

import pytest

import worldflux
from worldflux import _DEPRECATED_IMPORTS

_ALL_DEPRECATED_NAMES: list[str] = sorted(_DEPRECATED_IMPORTS.keys())


class TestDeprecationShim:
    """Verify every deprecated symbol is importable with a DeprecationWarning."""

    @pytest.mark.parametrize("name", _ALL_DEPRECATED_NAMES)
    def test_deprecated_import_emits_warning(self, name: str) -> None:
        """Importing a deprecated name triggers DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = getattr(worldflux, name)
            assert obj is not None, f"{name} resolved to None"

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert (
            len(deprecation_warnings) == 1
        ), f"Expected exactly 1 DeprecationWarning for '{name}', got {len(deprecation_warnings)}"

    @pytest.mark.parametrize("name", _ALL_DEPRECATED_NAMES)
    def test_warning_message_contains_migration_path(self, name: str) -> None:
        """Warning message must include the fully-qualified replacement path."""
        _mod_path, full_path = _DEPRECATED_IMPORTS[name]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getattr(worldflux, name)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        msg = str(deprecation_warnings[0].message)
        assert full_path in msg, f"Expected '{full_path}' in warning message: {msg}"

    def test_nonexistent_attribute_raises_attribute_error(self) -> None:
        """Accessing a truly missing name must raise AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(worldflux, "__this_does_not_exist_at_all__")

    def test_dir_contains_deprecated_names(self) -> None:
        """dir(worldflux) must include all deprecated names for discoverability."""
        module_dir = dir(worldflux)
        for name in _ALL_DEPRECATED_NAMES:
            assert name in module_dir, f"'{name}' missing from dir(worldflux)"

    def test_dir_contains_all_public_names(self) -> None:
        """dir(worldflux) must include everything in __all__."""
        module_dir = dir(worldflux)
        for name in worldflux.__all__:
            assert name in module_dir, f"'{name}' from __all__ missing from dir(worldflux)"
