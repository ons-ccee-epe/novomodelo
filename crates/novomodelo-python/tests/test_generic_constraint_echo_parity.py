"""CLI-vs-Python parity for the resolved generic-constraint echo.

The Python-parity hard rule (`CLAUDE.md`): every output file the CLI writes
must also be written by the Python bindings. This closes the loop for
`generic_constraints/resolved_echo.parquet`: it runs both write
paths end-to-end on a case that declares a generic constraint and asserts the
echo exists on BOTH sides with identical content. Both paths serialize the
identical `cobre_sddp::build_generic_constraint_echo_rows` output through the
identical `cobre_io::write_generic_constraint_echo` writer, so the two files
must match column-for-column.

`cobre-python` is excluded from the cargo workspace (it needs a Python
interpreter to build), so `cargo test --workspace` never runs this gate; it
runs in cobre-python's own test job:

    maturin develop --release --manifest-path crates/cobre-python/Cargo.toml
    pytest crates/cobre-python/tests/test_generic_constraint_echo_parity.py -v

The fixture `examples/deterministic/d13-generic-constraint` declares one
generic constraint (`thermal_generation(0) <= 10`), so a single run exercises
the echo on both write paths.
"""

from __future__ import annotations

import pathlib

import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
D13_CASE = _REPO_ROOT / "examples" / "deterministic" / "d13-generic-constraint"

_ECHO_REL = "generic_constraints/resolved_echo.parquet"


@pytest.fixture(scope="module")
def d13_cli_output(
    tmp_path_factory: pytest.TempPathFactory, cli_binary: pathlib.Path
) -> pathlib.Path:
    """Run D13 through the compiled CLI binary."""
    assert D13_CASE.is_dir(), f"the D13 fixture must exist at {D13_CASE}"
    output_dir = tmp_path_factory.mktemp("d13_cli_out")
    run_cli(D13_CASE, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d13_python_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D13 through the module-level Python bindings entry point."""
    import cobre.run  # noqa: PLC0415

    assert D13_CASE.is_dir(), f"the D13 fixture must exist at {D13_CASE}"
    output_dir = tmp_path_factory.mktemp("d13_python_out")
    cobre.run.run(str(D13_CASE), output_dir=str(output_dir))
    return output_dir


def test_cli_and_python_echo_are_identical(
    d13_cli_output: pathlib.Path, d13_python_output: pathlib.Path
) -> None:
    """Both write paths emit the echo, and its content is identical.

    Reads both `resolved_echo.parquet` files with pyarrow and compares the
    full table (13-column schema + every value). Both paths produce the rows
    through the same builder in canonical order, so the tables must be equal
    column-for-column.
    """
    cli_echo = d13_cli_output / _ECHO_REL
    py_echo = d13_python_output / _ECHO_REL
    assert cli_echo.is_file(), f"CLI must write the echo at {cli_echo}"
    assert py_echo.is_file(), f"Python must write the echo at {py_echo}"

    cli_table = pq.read_table(cli_echo)
    py_table = pq.read_table(py_echo)

    assert cli_table.num_columns == 13, "echo must carry the 13-column schema"
    assert cli_table.schema.names == py_table.schema.names, (
        "CLI and Python echo column names diverge"
    )
    assert cli_table.num_rows > 0, "D13 echo must contain at least one row"
    assert cli_table.to_pydict() == py_table.to_pydict(), (
        "CLI and Python echo contents diverge"
    )
