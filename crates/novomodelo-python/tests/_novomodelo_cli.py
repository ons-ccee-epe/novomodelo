"""Shared CLI binary discovery and invocation for CLI-vs-Python parity tests.

Provides `resolve_cli_binary` (the skip-versus-fail policy when the binary is
absent) and `run_cli` (the subprocess invocation). Both are used across the ten
parity modules that compare CLI output against Python-binding output. The
resolver is exposed via a session-scoped pytest fixture in `conftest.py`; the
invocation is called directly with the resolved binary path.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest


def resolve_cli_binary(search_root: pathlib.Path, *, required: bool) -> pathlib.Path:
    """Return the compiled `cobre` CLI binary path, failing or skipping if absent.

    Searches `<search_root>/target/{release,debug}/cobre` in that order,
    returning the first that exists. When `required` is True, absence is a
    hard failure; when False, it raises pytest.Skipped with guidance to build
    the binary.
    """
    for profile in ("release", "debug"):
        candidate = search_root / "target" / profile / "cobre"
        if candidate.is_file():
            return candidate

    message = (
        "No compiled `cobre` binary found in target/release or target/debug. "
        "Run `cargo build --release -p cobre-cli` first."
    )
    if required:
        pytest.fail(message, pytrace=False)
    else:
        pytest.skip(message)
    raise RuntimeError("unreachable: pytest.skip raises Skipped")


def run_cli(
    case_dir: pathlib.Path, output_dir: pathlib.Path, binary: pathlib.Path
) -> None:
    """Run the cobre CLI for `case_dir`, writing outputs to `output_dir`."""
    result = subprocess.run(
        [str(binary), "run", str(case_dir), "--output", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"cobre CLI failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
