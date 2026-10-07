"""CLI / Python parity tests for the per-stage filling slack ``sigma_fill``.

The dead-volume-filling reformulation turned the filling-shortfall slack
``sigma_fill`` from a single terminal value into a **per-stage** floor: one
``sigma_fill[t]`` per Filling stage, anchored backward to the per-stage target
storage ``V_target[t]``. In the simulation output that slack is the
``filling_target_violation_hm3`` column of the per-``(stage, block, hydro)``
``simulation/hydros.parquet`` table.

A single shared ``SimulationParquetWriter`` and a single
``From<SimulationHydroResult> for HydroWriteRecord`` conversion are invoked
verbatim by both the CLI (``write_simulation_outputs``) and the Python bindings
(``run_simulation_phase_py``), so Python-parity is satisfied structurally. These
tests lock that guarantee with a regression on a real **filling** case (d38),
where the per-stage ``sigma_fill`` binds at more than one Filling stage:

- the column is present, ``Float64``, non-nullable in both paths;
- it carries at least two distinct non-zero values across stages for the filling
  hydro ``H2`` (genuinely per-stage, not a single terminal value broadcast);
- the CLI and Python columns agree bit-for-bit;
- the self-describing catalog (``training/dictionaries/variables.csv``) documents
  the column in both paths.

The existing ``test_parity_hydros.py`` D02 suite cannot catch a per-stage
``sigma_fill`` regression: D02 is non-filling, so its
``filling_target_violation_hm3`` is uniformly ``0.0``.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_parity_filling_sigma.py -v
"""

from __future__ import annotations

import csv
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli
from test_parity_hydros import (
    _collect_hydros_parquet,
    _make_case_with_simulation,
    _read_hydros,
)

# Filling case re-derived to a sufficiency-passing, binding per-stage sigma_fill.
D38_CASE = "examples/deterministic/d38-dead-volume-filling"

# The per-stage filling-shortfall slack column under test.
SIGMA_FILL_COLUMN = "filling_target_violation_hm3"

# Entity id of the filling hydro H2 in d38 (see the
# `d38_dead_volume_filling_simulation` mod in filling_commissioning.rs, `H2_ID`).
# The `hydro_id` column is Int32, so this is the integer id, not "H2".
H2_ID = 1

# Tolerances for the distinct-non-zero check: drop values within `ZERO_TOL` of 0,
# and round to `ROUND_DECIMALS` decimals before building the distinct set so
# floating-point noise does not manufacture spurious distinct values.
ZERO_TOL = 1e-9
ROUND_DECIMALS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filling_hydro_sigma_values(table: pa.Table, hydro_id: int) -> list[float]:
    """Return ``filling_target_violation_hm3`` for the given hydro's rows."""
    hydro_ids = table.column("hydro_id").to_pylist()
    sigma = table.column(SIGMA_FILL_COLUMN).to_pylist()
    return [s for hid, s in zip(hydro_ids, sigma, strict=True) if hid == hydro_id]


def _distinct_nonzero_rounded(values: list[float]) -> set[float]:
    """Distinct non-zero values, rounded to suppress FP noise."""
    return {round(v, ROUND_DECIMALS) for v in values if abs(v) > ZERO_TOL}


def _variables_catalog_row(
    output_dir: pathlib.Path,
    file_name: str,
    column: str,
) -> dict[str, str]:
    """Return the ``variables.csv`` row matching ``(file_name, column)``.

    Fails the test if no such row exists.
    """
    csv_path = output_dir / "training" / "dictionaries" / "variables.csv"
    assert csv_path.is_file(), (
        f"training/dictionaries/variables.csv must exist after run; checked {csv_path}"
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["file"] == file_name and row["column"] == column:
                return row
    pytest.fail(f"variables.csv has no row for (file={file_name!r}, column={column!r})")
    raise RuntimeError("unreachable: pytest.fail raises Failed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d38_case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a temp copy of d38 with simulation enabled (shared across tests)."""
    repo_root = pathlib.Path(__file__).parents[3]
    src = repo_root / D38_CASE
    dest = tmp_path_factory.mktemp("d38_case")
    _make_case_with_simulation(src, dest)
    return dest


@pytest.fixture(scope="module")
def d38_cli_output(
    d38_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run d38 via the CLI and return the output directory."""
    output_dir = tmp_path_factory.mktemp("d38_cli_out")
    run_cli(d38_case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d38_python_output(
    d38_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Run d38 via ``cobre.run.run()`` and return the output directory."""
    cobre_run = pytest.importorskip("cobre.run")
    output_dir = tmp_path_factory.mktemp("d38_python_out")
    cobre_run.run(str(d38_case_dir), output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# Schema: sigma_fill column present, Float64, non-nullable (1.i)
# ---------------------------------------------------------------------------


def test_python_filling_sigma_column_is_float64_nonnullable(
    d38_python_output: pathlib.Path,
) -> None:
    """Python run: ``filling_target_violation_hm3`` is Float64, non-nullable."""
    schema = pq.read_schema(_collect_hydros_parquet(d38_python_output)[0])
    assert SIGMA_FILL_COLUMN in schema.names, (
        f"hydros.parquet (Python) is missing '{SIGMA_FILL_COLUMN}'.\n"
        f"Actual columns: {schema.names}"
    )
    field = schema.field(SIGMA_FILL_COLUMN)
    assert pa.types.is_float64(field.type), (
        f"Column '{SIGMA_FILL_COLUMN}' must be Float64, got {field.type}"
    )
    assert not field.nullable, f"Column '{SIGMA_FILL_COLUMN}' must be non-nullable"


def test_cli_filling_sigma_column_is_float64_nonnullable(
    d38_cli_output: pathlib.Path,
) -> None:
    """CLI run: ``filling_target_violation_hm3`` is Float64, non-nullable."""
    schema = pq.read_schema(_collect_hydros_parquet(d38_cli_output)[0])
    assert SIGMA_FILL_COLUMN in schema.names, (
        f"hydros.parquet (CLI) is missing '{SIGMA_FILL_COLUMN}'.\n"
        f"Actual columns: {schema.names}"
    )
    field = schema.field(SIGMA_FILL_COLUMN)
    assert pa.types.is_float64(field.type), (
        f"Column '{SIGMA_FILL_COLUMN}' must be Float64, got {field.type}"
    )
    assert not field.nullable, f"Column '{SIGMA_FILL_COLUMN}' must be non-nullable"


# ---------------------------------------------------------------------------
# Per-stage binding: at least two distinct non-zero values (1.ii, 1.iii)
# ---------------------------------------------------------------------------


def test_python_filling_sigma_has_two_distinct_nonzero_values(
    d38_python_output: pathlib.Path,
) -> None:
    """Python run: H2's ``sigma_fill`` binds with >=2 distinct non-zero values.

    The dead-volume-filling reformulation gives every Filling stage its own
    ``sigma_fill[t]`` floor. A regression back to a single terminal value (or a
    constant broadcast) would collapse the distinct count to one; this asserts
    the slack is genuinely per-stage. Filtered to the filling hydro H2 so a
    non-filling hydro's uniform 0.0 cannot mask or satisfy the check.
    """
    table = _read_hydros(d38_python_output)
    values = _filling_hydro_sigma_values(table, H2_ID)
    distinct = _distinct_nonzero_rounded(values)
    assert len(distinct) >= 2, (
        f"Python: H2 (id {H2_ID}) '{SIGMA_FILL_COLUMN}' must take at least two "
        f"distinct non-zero values across Filling stages (genuine per-stage "
        f"sigma_fill); got distinct non-zero set {sorted(distinct)} from "
        f"values {values}"
    )


def test_cli_filling_sigma_has_two_distinct_nonzero_values(
    d38_cli_output: pathlib.Path,
) -> None:
    """CLI run: H2's ``sigma_fill`` binds with >=2 distinct non-zero values."""
    table = _read_hydros(d38_cli_output)
    values = _filling_hydro_sigma_values(table, H2_ID)
    distinct = _distinct_nonzero_rounded(values)
    assert len(distinct) >= 2, (
        f"CLI: H2 (id {H2_ID}) '{SIGMA_FILL_COLUMN}' must take at least two "
        f"distinct non-zero values across Filling stages (genuine per-stage "
        f"sigma_fill); got distinct non-zero set {sorted(distinct)} from "
        f"values {values}"
    )


# ---------------------------------------------------------------------------
# CLI vs Python bit-for-bit parity on the sigma_fill column (1.iv)
# ---------------------------------------------------------------------------


def test_cli_python_filling_sigma_column_bit_for_bit(
    d38_cli_output: pathlib.Path,
    d38_python_output: pathlib.Path,
) -> None:
    """CLI and Python ``filling_target_violation_hm3`` agree bit-for-bit.

    Same length, values, and order. A divergence here would mean the CLI and
    Python write paths produce different per-stage ``sigma_fill``, violating the
    workspace Python-parity rule for this column.
    """
    cli_table = _read_hydros(d38_cli_output)
    py_table = _read_hydros(d38_python_output)
    cli_col = cli_table.column(SIGMA_FILL_COLUMN)
    py_col = py_table.column(SIGMA_FILL_COLUMN)
    assert cli_col.equals(py_col), (
        f"'{SIGMA_FILL_COLUMN}' mismatch between CLI and Python hydros.parquet:\n"
        f"  CLI first 8: {cli_col[:8].to_pylist()}\n"
        f"  PY  first 8: {py_col[:8].to_pylist()}"
    )


# ---------------------------------------------------------------------------
# Self-describing catalog: variables.csv documents the column (req 2)
# ---------------------------------------------------------------------------


def test_python_filling_sigma_catalog_row_is_documented(
    d38_python_output: pathlib.Path,
) -> None:
    """Python run: variables.csv documents (hydros, sigma_fill) correctly."""
    row = _variables_catalog_row(d38_python_output, "hydros", SIGMA_FILL_COLUMN)
    assert row["type"] == "f64", f"type must be 'f64', got {row['type']!r}"
    assert row["unit"] == "hm3", f"unit must be 'hm3', got {row['unit']!r}"
    assert row["nullable"] == "false", (
        f"nullable must be 'false', got {row['nullable']!r}"
    )
    assert row["description"].strip(), "description must be non-empty"


def test_cli_filling_sigma_catalog_row_is_documented(
    d38_cli_output: pathlib.Path,
) -> None:
    """CLI run: variables.csv documents (hydros, sigma_fill) correctly."""
    row = _variables_catalog_row(d38_cli_output, "hydros", SIGMA_FILL_COLUMN)
    assert row["type"] == "f64", f"type must be 'f64', got {row['type']!r}"
    assert row["unit"] == "hm3", f"unit must be 'hm3', got {row['unit']!r}"
    assert row["nullable"] == "false", (
        f"nullable must be 'false', got {row['nullable']!r}"
    )
    assert row["description"].strip(), "description must be non-empty"
