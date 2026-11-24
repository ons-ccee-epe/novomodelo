"""CLI-versus-Python parity on the stage-varying bus deficit deck.

Both front ends read `constraints/penalty_overrides_bus_deficit.parquet` through the
same case loader, so their `simulation/costs` files are equal. The Python run also
reports the deck's closed-form bounds and per-stage deficit costs.

Run with (from the repo root):

    pytest crates/novomodelo-python/tests/test_bus_deficit_override_parity.py -v \\
        --require-cli-binary
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import pyarrow.parquet as pq
import pytest

from _novomodelo_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
D63_CASE = _REPO_ROOT / "examples/deterministic/d63-stage-varying-deficit"

# derived: D63_EXPECTED_COST, D63_STAGE0_DEFICIT_COST and D63_STAGE1_DEFICIT_COST in
# crates/novomodelo-sddp/tests/deterministic.rs
D63_EXPECTED_COST = 94_148_000.0
D63_STAGE_DEFICIT_COSTS = [40_150_000.0, 53_560_000.0]
REL_TOL = 1e-9


def _costs_files(output_dir: pathlib.Path) -> list[str]:
    return sorted(
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "simulation" / "costs").rglob("*.parquet")
    )


def _assert_parquet_tree_equal(cli_dir: pathlib.Path, py_dir: pathlib.Path) -> None:
    cli_files = _costs_files(cli_dir)
    assert cli_files, "the CLI must write simulation/costs Parquet files"
    assert cli_files == _costs_files(py_dir)

    for rel_path in cli_files:
        cli_table = pq.ParquetFile(cli_dir / rel_path).read()
        py_table = pq.ParquetFile(py_dir / rel_path).read()
        assert cli_table.schema.equals(py_table.schema), rel_path
        assert cli_table.num_rows == py_table.num_rows, rel_path
        for name in cli_table.schema.names:
            assert cli_table.column(name).equals(py_table.column(name)), (
                f"{rel_path}: column {name}"
            )


@pytest.fixture(scope="module")
def d63_cli_output(
    tmp_path_factory: pytest.TempPathFactory, cli_binary: pathlib.Path
) -> pathlib.Path:
    output_dir = tmp_path_factory.mktemp("d63_cli_out")
    run_cli(D63_CASE, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d63_python_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, Mapping[str, Any]]:
    import novomodelo.run

    output_dir = tmp_path_factory.mktemp("d63_python_out")
    result = novomodelo.run.run(str(D63_CASE), output_dir=str(output_dir))
    return output_dir, result


def test_simulation_costs_match_cli(
    d63_cli_output: pathlib.Path,
    d63_python_run: tuple[pathlib.Path, Mapping[str, Any]],
) -> None:
    _assert_parquet_tree_equal(d63_cli_output, d63_python_run[0])


def test_python_bounds_match_closed_form(
    d63_python_run: tuple[pathlib.Path, Mapping[str, Any]],
) -> None:
    result = d63_python_run[1]
    assert result["lower_bound"] == pytest.approx(D63_EXPECTED_COST, rel=REL_TOL)
    assert result["upper_bound"] == pytest.approx(D63_EXPECTED_COST, rel=REL_TOL)


def test_python_stage_deficit_costs_match_closed_form(
    d63_python_run: tuple[pathlib.Path, Mapping[str, Any]],
) -> None:
    output_dir = d63_python_run[0]
    stage_rows = sorted(
        (
            row
            for rel_path in _costs_files(output_dir)
            for row in pq.ParquetFile(output_dir / rel_path).read().to_pylist()
            if row["block_id"] is None
        ),
        key=lambda row: row["stage_id"],
    )
    assert [row["deficit_cost"] for row in stage_rows] == pytest.approx(
        D63_STAGE_DEFICIT_COSTS, rel=REL_TOL
    )
