"""CLI / Python parity tests for ``simulation/hydros.parquet``.

Verifies that ``cobre.run.run()`` and the ``cobre`` CLI binary produce
identical ``hydros.parquet`` output for the D02 single-hydro deterministic
case, with particular focus on the five energy-conversion columns:

- ``equivalent_productivity_mw_per_m3s``
- ``accumulated_productivity_mw_per_m3s``
- ``incremental_inflow_energy_mw``
- ``stored_energy_initial_mwh``
- ``stored_energy_final_mwh``

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_parity_hydros.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli

D02_CASE = "examples/deterministic/d02-single-hydro"

# The five energy-conversion columns written by simulation/hydros.parquet.
ENERGY_COLUMNS: list[str] = [
    "equivalent_productivity_mw_per_m3s",
    "accumulated_productivity_mw_per_m3s",
    "incremental_inflow_energy_mw",
    "stored_energy_initial_mwh",
    "stored_energy_final_mwh",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_with_simulation(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy a deterministic case to ``dest`` and enable the simulation pass.

    Several deterministic cases ship with ``simulation.enabled = false`` (the
    parity harness trains only). The parity tests need the simulation write path
    exercised, so the copied ``config.json`` is overwritten with
    ``simulation.enabled = true``. Case-agnostic so any deterministic case
    (D02, d38, ...) can reuse it.
    """
    for item in src.iterdir():
        dst_item = dest / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    config = json.loads((src / "config.json").read_text())
    config.setdefault("simulation", {})["enabled"] = True
    (dest / "config.json").write_text(json.dumps(config))


def _collect_hydros_parquet(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted parquet files under ``simulation/hydros/``."""
    hydros_dir = output_dir / "simulation" / "hydros"
    assert hydros_dir.is_dir(), (
        f"simulation/hydros/ must exist after run; checked {hydros_dir}"
    )
    parquets = sorted(hydros_dir.rglob("*.parquet"))
    assert len(parquets) > 0, f"no parquet files found under {hydros_dir}"
    return parquets


def _read_hydros(output_dir: pathlib.Path) -> pa.Table:
    """Concatenate all hydros parquet partitions into a single Arrow table."""
    parquets = _collect_hydros_parquet(output_dir)
    # ParquetFile (not read_table): scenario_id is both the Hive partition key and
    # an in-file column, so read_table's dataset path would try to merge them.
    tables = [pq.ParquetFile(p).read() for p in parquets]
    return pa.concat_tables(tables)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d02_case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a temp copy of D02 with simulation enabled (shared across tests)."""
    repo_root = pathlib.Path(__file__).parents[3]
    src = repo_root / D02_CASE
    dest = tmp_path_factory.mktemp("d02_case")
    _make_case_with_simulation(src, dest)
    return dest


@pytest.fixture(scope="module")
def d02_cli_output(
    d02_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run D02 via the CLI and return the output directory."""
    output_dir = tmp_path_factory.mktemp("d02_cli_out")
    run_cli(d02_case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d02_python_output(
    d02_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Run D02 via ``cobre.run.run()`` and return the output directory."""
    cobre_run = pytest.importorskip("cobre.run")
    output_dir = tmp_path_factory.mktemp("d02_python_out")
    cobre_run.run(str(d02_case_dir), output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# Schema and column-existence tests (Python-only — no CLI required)
# ---------------------------------------------------------------------------


def test_python_hydros_has_energy_columns(d02_python_output: pathlib.Path) -> None:
    """Python run: hydros.parquet contains all five energy-conversion columns."""
    schema = pq.read_schema(_collect_hydros_parquet(d02_python_output)[0])
    missing = [col for col in ENERGY_COLUMNS if col not in schema.names]
    assert not missing, (
        f"hydros.parquet (Python) is missing energy columns: {missing}\n"
        f"Actual columns: {schema.names}"
    )


def test_python_energy_columns_are_float64_nonnullable(
    d02_python_output: pathlib.Path,
) -> None:
    """Python run: each energy column must be Float64 and non-nullable."""
    schema = pq.read_schema(_collect_hydros_parquet(d02_python_output)[0])
    for col in ENERGY_COLUMNS:
        field = schema.field(col)
        assert pa.types.is_float64(field.type), (
            f"Column '{col}' must be Float64, got {field.type}"
        )
        assert not field.nullable, f"Column '{col}' must be non-nullable"


def test_python_energy_columns_nonzero(d02_python_output: pathlib.Path) -> None:
    """Python run: at least one row per energy column must have a non-zero value.

    D02 has a hydro plant with constant-productivity turbining.  The five
    derived energy columns must therefore be non-zero for at least one
    (stage, block, hydro) tuple in the simulation output.
    """
    table = _read_hydros(d02_python_output)
    for col in ENERGY_COLUMNS:
        values = table.column(col).to_pylist()
        assert any(v != 0.0 for v in values), (
            f"Column '{col}' must have at least one non-zero value in the "
            f"D02 simulation output (hydro with constant productivity)."
        )


# ---------------------------------------------------------------------------
# CLI vs Python parity tests (require a compiled CLI binary)
# ---------------------------------------------------------------------------


def test_cli_hydros_has_energy_columns(d02_cli_output: pathlib.Path) -> None:
    """CLI run: hydros.parquet contains all five energy-conversion columns."""
    schema = pq.read_schema(_collect_hydros_parquet(d02_cli_output)[0])
    missing = [col for col in ENERGY_COLUMNS if col not in schema.names]
    assert not missing, (
        f"hydros.parquet (CLI) is missing energy columns: {missing}\n"
        f"Actual columns: {schema.names}"
    )


def test_cli_python_hydros_schema_identical(
    d02_cli_output: pathlib.Path,
    d02_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce hydros.parquet with an identical Arrow schema.

    Identical schema = same field names, same dtypes, same nullability, same
    column order.
    """
    cli_schema = pq.read_schema(_collect_hydros_parquet(d02_cli_output)[0])
    py_schema = pq.read_schema(_collect_hydros_parquet(d02_python_output)[0])
    assert cli_schema.equals(py_schema), (
        "Schema mismatch between CLI and Python hydros.parquet:\n"
        f"  CLI: {cli_schema}\n"
        f"  PY:  {py_schema}"
    )


def test_cli_python_hydros_row_count_identical(
    d02_cli_output: pathlib.Path,
    d02_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce the same total number of hydro rows."""
    cli_table = _read_hydros(d02_cli_output)
    py_table = _read_hydros(d02_python_output)
    assert cli_table.num_rows == py_table.num_rows, (
        f"Row-count mismatch: CLI={cli_table.num_rows}, Python={py_table.num_rows}"
    )


def test_cli_python_energy_columns_bit_for_bit(
    d02_cli_output: pathlib.Path,
    d02_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce bit-for-bit identical values for each energy column.

    Because D02 is fully deterministic (constant load, no stochastic inflows,
    fixed seed) the five energy-conversion columns must be byte-for-byte equal.
    If they differ, the CLI and Python paths diverge at some point between
    the LP solve and the Parquet write — which violates the workspace parity rule.
    """
    cli_table = _read_hydros(d02_cli_output)
    py_table = _read_hydros(d02_python_output)

    mismatches: list[str] = []
    for col in ENERGY_COLUMNS:
        cli_col = cli_table.column(col)
        py_col = py_table.column(col)
        if not cli_col.equals(py_col):
            mismatches.append(
                f"  column '{col}':\n"
                f"    CLI first 5: {cli_col[:5].to_pylist()}\n"
                f"    PY  first 5: {py_col[:5].to_pylist()}"
            )

    assert not mismatches, (
        "Energy column value mismatch between CLI and Python hydros.parquet:\n"
        + "\n".join(mismatches)
    )
