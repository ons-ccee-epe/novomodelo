"""CLI / Python parity tests for scalar-parameter resolution.

Verifies that ``cobre.run.run()`` and the ``cobre`` CLI binary produce
identical ``simulation/buses.parquet`` output for a case that uses
``constraints/generic_parameters.json``.

The test fabricates a minimal fixture by copying D13 (``d13-generic-constraint``)
into a pytest tmp dir, adding a ``constraints/generic_parameters.json`` that defines a
single ``constant`` parameter named ``demand_scale``, and rewriting the existing
generic constraint to reference ``@demand_scale`` in its expression.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_parity_scalar_parameters.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli

D13_CASE = "examples/deterministic/d13-generic-constraint"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_case(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy D13 to ``dst``, inject a generic_parameters.json, and patch the constraint.

    Also enables the simulation pass so ``simulation/buses/`` is written.
    """
    for item in src.iterdir():
        dst_item = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    # Enable simulation so buses.parquet is produced.
    config = json.loads((src / "config.json").read_text())
    config.setdefault("simulation", {})["enabled"] = True
    (dst / "config.json").write_text(json.dumps(config))

    # Add constraints/generic_parameters.json with a single constant parameter.
    (dst / "constraints" / "generic_parameters.json").write_text(
        json.dumps(
            {
                "scalar_parameters": [
                    {
                        "id": 1,
                        "name": "demand_scale",
                        "kind": "constant",
                        "value": 1.05,
                    }
                ]
            }
        )
    )

    # Patch the first generic constraint to reference @demand_scale.
    constraints_path = dst / "constraints" / "generic_constraints.json"
    payload = json.loads(constraints_path.read_text())
    expr = payload["constraints"][0]["expression"]
    payload["constraints"][0]["expression"] = f"@demand_scale * {expr}"
    constraints_path.write_text(json.dumps(payload))


def _collect_buses_parquet(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted parquet files under ``simulation/buses/``."""
    buses_dir = output_dir / "simulation" / "buses"
    assert buses_dir.is_dir(), (
        f"simulation/buses/ must exist after run; checked {buses_dir}"
    )
    parquets = sorted(buses_dir.rglob("*.parquet"))
    assert len(parquets) > 0, f"no parquet files found under {buses_dir}"
    return parquets


def _read_buses(output_dir: pathlib.Path) -> pa.Table:
    """Concatenate all buses parquet partitions into a single Arrow table."""
    parquets = _collect_buses_parquet(output_dir)
    # ParquetFile (not read_table): scenario_id is both the Hive partition key and
    # an in-file column, so read_table's dataset path would try to merge them.
    tables = [pq.ParquetFile(p).read() for p in parquets]
    return pa.concat_tables(tables)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d13_scalar_case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a temp copy of D13 with generic_parameters.json injected."""
    repo_root = pathlib.Path(__file__).parents[3]
    src = repo_root / D13_CASE
    dest = tmp_path_factory.mktemp("d13_scalar_case")
    _seed_case(src, dest)
    return dest


@pytest.fixture(scope="module")
def d13_cli_output(
    d13_scalar_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run the scalar-parameter D13 fixture via the CLI and return the output dir."""
    output_dir = tmp_path_factory.mktemp("d13_cli_out")
    run_cli(d13_scalar_case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d13_python_output(
    d13_scalar_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Run the scalar-parameter D13 fixture via ``cobre.run.run()`` and return output dir."""
    cobre_run = pytest.importorskip("cobre.run")
    output_dir = tmp_path_factory.mktemp("d13_python_out")
    cobre_run.run(str(d13_scalar_case_dir), output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def test_parity_buses_parquet_matches_cli(
    d13_cli_output: pathlib.Path,
    d13_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce bit-for-bit identical buses.parquet values.

    Because D13 is fully deterministic and scalar parameters are constant,
    the CLI and Python runs must produce identical buses output.  Any
    divergence indicates that the Python path failed to load or thread
    scalar_parameters into the LP.
    """
    cli_table = _read_buses(d13_cli_output)
    py_table = _read_buses(d13_python_output)

    assert cli_table.schema.equals(py_table.schema), (
        "Schema mismatch between CLI and Python buses.parquet:\n"
        f"  CLI: {cli_table.schema}\n"
        f"  PY:  {py_table.schema}"
    )

    assert cli_table.num_rows == py_table.num_rows, (
        f"Row-count mismatch: CLI={cli_table.num_rows}, Python={py_table.num_rows}"
    )

    mismatches: list[str] = []
    for col in cli_table.schema.names:
        cli_col = cli_table.column(col)
        py_col = py_table.column(col)
        if not cli_col.equals(py_col):
            mismatches.append(
                f"  column '{col}':\n"
                f"    CLI first 5: {cli_col[:5].to_pylist()}\n"
                f"    PY  first 5: {py_col[:5].to_pylist()}"
            )

    assert not mismatches, (
        "Value mismatch between CLI and Python buses.parquet:\n" + "\n".join(mismatches)
    )
