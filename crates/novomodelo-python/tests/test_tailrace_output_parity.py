"""CLI / Python parity for the tailrace columns of ``simulation/hydros``.

A hydro that declares ``tailrace_flow`` and uses computed FPHA gets its tailrace
flow, tailrace level and net head written to ``simulation/hydros``. Every other
parity case leaves those columns null, so this module runs a deck that exercises
the declaration and pins that the CLI and ``novomodelo.run.run()`` write identical
values, with nulls in the same positions.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_tailrace_output_parity.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _novomodelo_cli import run_cli

SOURCE_CASE = "examples/deterministic/d58-stored-energy-coefficient"

TAILRACE_COLUMNS = ["tailrace_flow_m3s", "tailrace_level_m", "net_head_m"]

DECLARED_HYDRO_ID = 0

TAILRACE_FLOW = {
    "own_outflow_factor": 1.2,
    "outflow_influences": [{"hydro_id": 1, "factor": 0.5}],
    "incremental_inflow_influences": [{"hydro_id": 0, "factor": 0.3}],
}


def _make_case_with_tailrace(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy ``src`` to ``dest``, enable simulation and declare ``tailrace_flow``."""
    for item in src.iterdir():
        dst_item = dest / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config.setdefault("simulation", {})["enabled"] = True
    (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")

    hydros_path = dest / "system" / "hydros.json"
    hydros = json.loads(hydros_path.read_text(encoding="utf-8"))
    declared = next(h for h in hydros["hydros"] if h["id"] == DECLARED_HYDRO_ID)
    declared["tailrace_flow"] = TAILRACE_FLOW
    hydros_path.write_text(json.dumps(hydros), encoding="utf-8")


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
    """Concatenate the hydros partitions and sort rows by their entity key."""
    # ParquetFile (not read_table): scenario_id is both the Hive partition key and
    # an in-file column, so read_table's dataset path would try to merge them.
    table = pa.concat_tables(
        [pq.ParquetFile(p).read() for p in _collect_hydros_parquet(output_dir)]
    )
    keys = ["stage_id", "block_id", "hydro_id"]
    if "scenario_id" in table.schema.names:
        keys.insert(0, "scenario_id")
    return table.sort_by([(key, "ascending") for key in keys])


@pytest.fixture(scope="module")
def tailrace_case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a temp copy of the source case declaring ``tailrace_flow``."""
    src = pathlib.Path(__file__).parents[3] / SOURCE_CASE
    dest = tmp_path_factory.mktemp("tailrace_case")
    _make_case_with_tailrace(src, dest)
    return dest


@pytest.fixture(scope="module")
def cli_out(
    tailrace_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run the case via the CLI and return the output directory."""
    output_dir = tmp_path_factory.mktemp("tailrace_cli_out")
    run_cli(tailrace_case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def py_out(
    tailrace_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Run the case via ``novomodelo.run.run()`` and return the output directory."""
    novomodelo_run = pytest.importorskip("novomodelo.run")
    output_dir = tmp_path_factory.mktemp("tailrace_python_out")
    novomodelo_run.run(str(tailrace_case_dir), output_dir=str(output_dir))
    return output_dir


def test_tailrace_columns_present_and_typed(
    cli_out: pathlib.Path, py_out: pathlib.Path
) -> None:
    """Both outputs carry the tailrace columns as nullable Float64."""
    for label, output_dir in (("CLI", cli_out), ("Python", py_out)):
        schema = pq.read_schema(_collect_hydros_parquet(output_dir)[0])
        for col in TAILRACE_COLUMNS:
            assert col in schema.names, (
                f"{label} hydros is missing '{col}'; columns: {schema.names}"
            )
            field = schema.field(col)
            assert field.type == pa.float64(), (
                f"{label} '{col}' must be Float64, got {field.type}"
            )
            assert field.nullable, f"{label} '{col}' must be nullable"


def test_tailrace_columns_identical(
    cli_out: pathlib.Path, py_out: pathlib.Path
) -> None:
    """CLI and Python write identical tailrace values and null positions."""
    cli_table = _read_hydros(cli_out)
    py_table = _read_hydros(py_out)
    for col in TAILRACE_COLUMNS:
        cli_col = cli_table.column(col).combine_chunks()
        py_col = py_table.column(col).combine_chunks()
        assert cli_col.equals(py_col), (
            f"'{col}' differs between CLI and Python:\n"
            f"  CLI first 5: {cli_col[:5].to_pylist()}\n"
            f"  PY  first 5: {py_col[:5].to_pylist()}"
        )


def test_declared_hydro_columns_non_null(
    cli_out: pathlib.Path, py_out: pathlib.Path
) -> None:
    """The declaring hydro has every tailrace column set and a positive flow."""
    for label, output_dir in (("CLI", cli_out), ("Python", py_out)):
        rows = [
            row
            for row in _read_hydros(output_dir).to_pylist()
            if row["hydro_id"] == DECLARED_HYDRO_ID
        ]
        assert rows, f"{label} hydros has no rows for hydro {DECLARED_HYDRO_ID}"
        for col in TAILRACE_COLUMNS:
            nulls = sum(row[col] is None for row in rows)
            assert nulls == 0, (
                f"{label} '{col}' has {nulls} null rows for hydro {DECLARED_HYDRO_ID}"
            )
        flow_max = max(row["tailrace_flow_m3s"] for row in rows)
        assert flow_max > 0, (
            f"{label} tailrace_flow_m3s must be positive on at least one row for "
            f"hydro {DECLARED_HYDRO_ID}, max is {flow_max}"
        )
