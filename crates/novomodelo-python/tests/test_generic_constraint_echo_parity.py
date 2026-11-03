"""CLI-vs-Python parity for the resolved generic-constraint echo.

The Python-parity hard rule (`CLAUDE.md`): every output file the CLI writes
must also be written by the Python bindings. This closes the loop for
`generic_constraints/resolved_echo.parquet`: it runs both write
paths end-to-end on a case that declares a generic constraint and asserts the
echo exists on BOTH sides with identical content. Both paths serialize the
identical `novomodelo_sddp::build_generic_constraint_echo_rows` output through the
identical `novomodelo_io::write_generic_constraint_echo` writer, so the two files
must match column-for-column.

`novomodelo-python` is excluded from the cargo workspace (it needs a Python
interpreter to build), so `cargo test --workspace` never runs this gate; it
runs in novomodelo-python's own test job:

    maturin develop --release --manifest-path crates/novomodelo-python/Cargo.toml
    pytest crates/novomodelo-python/tests/test_generic_constraint_echo_parity.py -v

The module fixture is parametrized over two cases, and each one runs both
write paths:

- `examples/deterministic/d13-generic-constraint` declares one thermal cap
  (`thermal_generation(0) <= 10`);
- `examples/deterministic/d59-parameter-sum-useful-volume-floor` puts a
  parameter-sum coefficient on a useful-volume term, under a `per_unit` slack.
"""

from __future__ import annotations

import pathlib

import pyarrow.parquet as pq
import pytest

from _novomodelo_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
D13_CASE = _REPO_ROOT / "examples" / "deterministic" / "d13-generic-constraint"
D59_CASE = (
    _REPO_ROOT / "examples" / "deterministic" / "d59-parameter-sum-useful-volume-floor"
)

_ECHO_REL = "generic_constraints/resolved_echo.parquet"


@pytest.fixture(scope="module", params=[D13_CASE, D59_CASE], ids=["d13", "d59"])
def case_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    """The case directory both write paths run."""
    case: pathlib.Path = request.param
    assert case.is_dir(), f"the {case.name} fixture must exist at {case}"
    return case


@pytest.fixture(scope="module")
def cli_output(
    case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run the case through the compiled CLI binary."""
    output_dir = tmp_path_factory.mktemp(f"{case_dir.name}_cli_out")
    run_cli(case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def python_output(
    case_dir: pathlib.Path, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Run the case through the module-level Python bindings entry point."""
    import novomodelo.run

    output_dir = tmp_path_factory.mktemp(f"{case_dir.name}_python_out")
    novomodelo.run.run(str(case_dir), output_dir=str(output_dir))
    return output_dir


def test_cli_and_python_echo_are_identical(
    case_dir: pathlib.Path, cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """Both write paths emit the echo, and its content is identical.

    Reads both `resolved_echo.parquet` files with pyarrow and compares the
    full table (14-column schema + every value). Both paths produce the rows
    through the same builder in canonical order, so the tables must be equal
    column-for-column.
    """
    cli_echo = cli_output / _ECHO_REL
    py_echo = python_output / _ECHO_REL
    assert cli_echo.is_file(), f"CLI must write the echo at {cli_echo}"
    assert py_echo.is_file(), f"Python must write the echo at {py_echo}"

    cli_table = pq.read_table(cli_echo)
    py_table = pq.read_table(py_echo)

    assert cli_table.num_columns == 14, "echo must carry the 14-column schema"
    assert cli_table.schema.names == py_table.schema.names, (
        f"CLI and Python echo column names diverge for {case_dir.name}"
    )
    assert cli_table.num_rows > 0, f"{case_dir.name} echo must contain at least one row"
    assert cli_table.to_pydict() == py_table.to_pydict(), (
        f"CLI and Python echo contents diverge for {case_dir.name}"
    )
