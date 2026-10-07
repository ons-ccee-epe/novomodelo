"""Tests for the top-level cobre.version_info function.

Verify the dict shape and the single-process invariant (comm is always
"local"), and that the solver string carries the active LP backend name
(HiGHS or CLP) followed by a live version token.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_version.py
"""

from __future__ import annotations


def test_version_info_returns_dict() -> None:
    """version_info returns a dict with the expected keys."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert isinstance(info, dict)
    for key in ("version", "solver", "comm", "zstd", "arch", "build"):
        assert key in info


def test_version_matches_module_version() -> None:
    """version_info()["version"] equals cobre.__version__."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["version"] == cobre.__version__


def test_comm_is_always_local() -> None:
    """The single-process binding always reports comm == "local"."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["comm"] == "local"


def test_solver_string_format() -> None:
    """The solver field names the active backend and carries a version token."""
    import cobre  # noqa: PLC0415

    solver = cobre.version_info()["solver"]
    parts = solver.split()

    assert parts[0] in ("HiGHS", "CLP")
    assert len(parts) >= 2  # noqa: PLR2004
    assert parts[1]  # a non-empty version token follows the backend name


def test_zstd_enabled_and_build_known() -> None:
    """zstd is enabled and build is one of debug/release."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["zstd"] == "enabled"
    assert info["build"] in ("debug", "release")
