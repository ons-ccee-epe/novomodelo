"""Tests for the cobre.schema sub-module.

Verify that cobre.schema.export writes all generated JSON Schema files to the
target directory, returns the file count, and that every written file is valid
JSON.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_schema.py
"""

from __future__ import annotations

import json
import os
import pathlib


def test_export_returns_file_count(tmp_path: pathlib.Path) -> None:
    """export returns the number of files written, matching the directory."""
    import cobre.schema  # noqa: PLC0415

    count = cobre.schema.export(str(tmp_path))

    assert isinstance(count, int)
    assert count > 0
    assert count == len(os.listdir(tmp_path))


def test_export_writes_valid_json(tmp_path: pathlib.Path) -> None:
    """Every file written by export parses as JSON."""
    import cobre.schema  # noqa: PLC0415

    cobre.schema.export(str(tmp_path))

    written = list(tmp_path.iterdir())
    assert written, "export must write at least one schema file"
    for path in written:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)


def test_export_creates_missing_directory(tmp_path: pathlib.Path) -> None:
    """export creates the output directory if it does not yet exist."""
    import cobre.schema  # noqa: PLC0415

    target = tmp_path / "nested" / "schemas"
    count = cobre.schema.export(str(target))

    assert target.is_dir()
    assert count == len(os.listdir(target))


def test_export_accessible_as_submodule() -> None:
    """cobre.schema.export must be importable as a submodule function."""
    import cobre.schema  # noqa: PLC0415

    assert callable(cobre.schema.export)
