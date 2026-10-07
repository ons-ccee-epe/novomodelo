"""pytest configuration for the cobre-python test suite.

Registers the `--require-cli-binary` flag and provides a session-scoped
`cli_binary` fixture that wraps the binary-discovery policy from `_cobre_cli`.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from _cobre_cli import resolve_cli_binary

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser
    from _pytest.config import Config

_REPO_ROOT = pathlib.Path(__file__).parents[3]


def pytest_addoption(parser: Parser) -> None:
    """Register the `--require-cli-binary` flag for the parity suite."""
    parser.addoption(
        "--require-cli-binary",
        action="store_true",
        default=False,
        help="Make a missing CLI binary a hard failure instead of a skip",
    )


@pytest.fixture(scope="session")
def cli_binary(pytestconfig: Config) -> pathlib.Path:
    """Return the compiled `cobre` CLI binary path.

    Uses the `--require-cli-binary` flag to determine whether absence is a
    skip (default, for local development) or a failure (CI).
    """
    required = pytestconfig.getoption("--require-cli-binary")
    return resolve_cli_binary(_REPO_ROOT, required=required)
