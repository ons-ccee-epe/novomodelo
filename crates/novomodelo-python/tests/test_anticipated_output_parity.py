"""Python-parity guard for the anticipated-thermal simulation output.

The anticipated refactor does NOT add a new output file: the anticipated result
is two nullable columns on the existing `thermals` table
(`anticipated_committed_mw`, `anticipated_decision_mw`), owned by
`thermals_schema()` in cobre-io's schemas.rs. The Python-parity hard rule
(`CLAUDE.md`) requires every output the CLI writes to also be written by the
`cobre-python` bindings, so those two columns must be confirmed to flow through
the Python paths identically to the CLI.

cobre-python has no anticipated-specific output code: it writes the thermals
table through the same shared `SimulationParquetWriter` the CLI uses, so Python
parity holds by construction. This module is the guard that proves it
end-to-end: it runs an anticipated deterministic case through both the compiled
`cobre` CLI (`cobre run --output`) and `cobre.run.run`, reads both thermals
parquets, and asserts the two anticipated columns are element-wise equal
INCLUDING null positions.

The anticipated cases ship `simulation.enabled = false` (they are training-only
parity cases). The Python surfaces and the CLI honor that flag, so the tests run
against a temp copy with simulation enabled (`_make_case_with_simulation`), the
convention `test_contract_output_parity.py` established for deterministic cases
that ship simulation disabled.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_anticipated_output_parity.py -v
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil

import pyarrow.parquet as pq

from _cobre_cli import run_cli

# d37 (commissioning windows) is the anticipated deterministic case whose
# decision column carries a mix of null and non-null positions — commissioning
# deactivates the decision at dormant delivery stages, so the "including null
# positions" assertion is load-bearing. Resolved against the repo root so the
# test is independent of pytest's working directory (never a CWD-relative path).
_REPO_ROOT = pathlib.Path(__file__).parents[3]
ANTICIPATED_CASE = (
    _REPO_ROOT / "examples" / "deterministic" / "d37-anticipated-commissioning"
)

# The two anticipated columns under test, both nullable Float64 on the thermals
# schema. A schema drift (a new anticipated sub-table, or a renamed column) fails
# the `_thermals_parquets` / column-read below.
_ANTICIPATED_COLUMNS = ("anticipated_committed_mw", "anticipated_decision_mw")

# Columns the CLI/Python comparison sorts by, so two writers that emit the same
# rows in a different physical order compare equal. `block_id` is nullable
# (Int32, true), so the sort key maps None to a sentinel.
_SORT_FIELDS = ("stage_id", "block_id", "thermal_id")


def _thermals_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the thermals data.parquet partitions under an output dir."""
    thermals_dir = output_dir / "simulation" / "thermals"
    if not thermals_dir.is_dir():
        return []
    return sorted(thermals_dir.rglob("*.parquet"))


def _make_case_with_simulation(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy a deterministic case to ``dest`` and enable the simulation pass.

    The anticipated cases ship ``simulation.enabled = false`` (training-only
    parity cases), and the CLI and Python surfaces honor that flag. The parity
    test needs the simulation write path exercised, so the copied ``config.json``
    is overwritten with ``simulation.enabled = true``. Mirrors
    ``test_contract_output_parity.py``.
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


def _read_thermal_rows(parquet: pathlib.Path) -> list[dict[str, object]]:
    """Read a thermals parquet into rows sorted on ``(stage_id, block_id,
    thermal_id)`` with a None-block_id sentinel, so two writers that emit the
    same rows in a different physical order compare equal.
    """

    def sort_key(row: dict[str, object]) -> tuple[object, ...]:
        block = row[_SORT_FIELDS[1]]
        block_key = -1 if block is None else block
        return (row[_SORT_FIELDS[0]], block_key, row[_SORT_FIELDS[2]])

    # Read the single file directly: scenario_id is both the Hive partition key
    # and an in-file column, so pq.read_table's dataset path would try to merge
    # the two same-named fields and error. ParquetFile reads only the file.
    rows = pq.ParquetFile(parquet).read().to_pylist()
    return sorted(rows, key=sort_key)


def _anticipated_values_equal(left: object, right: object) -> bool:
    """Compare one anticipated cell null-inclusively.

    Both ``None`` compare equal; exactly one ``None`` is a mismatch (a null
    position drifted); two numbers compare with a float tolerance.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def test_cli_python_anticipated_column_parity(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """The CLI and the Python surface emit identical anticipated columns.

    Runs the anticipated case (simulation enabled) through both the compiled
    `cobre` CLI (`cobre run --output`) and `cobre.run.run`, reads both thermals
    parquets, sorts each by `(stage_id, block_id, thermal_id)`, and asserts the
    `anticipated_committed_mw` and `anticipated_decision_mw` columns are
    element-wise equal INCLUDING null positions. If the CLI and Python paths
    diverge between the LP solve and the Parquet write, this fails.
    """
    import cobre.run  # noqa: PLC0415

    assert ANTICIPATED_CASE.is_dir(), (
        f"the anticipated fixture must exist at {ANTICIPATED_CASE}"
    )

    case_dir = tmp_path / "anticipated_case"
    case_dir.mkdir()
    _make_case_with_simulation(ANTICIPATED_CASE, case_dir)

    out_cli = tmp_path / "cli_out"
    run_cli(case_dir, out_cli, cli_binary)

    out_py = tmp_path / "py_out"
    cobre.run.run(str(case_dir), output_dir=str(out_py))

    cli_parquets = _thermals_parquets(out_cli)
    py_parquets = _thermals_parquets(out_py)
    assert len(cli_parquets) > 0, "CLI must emit a thermals parquet for the case"
    assert len(py_parquets) > 0, "Python must emit a thermals parquet for the case"

    for column in _ANTICIPATED_COLUMNS:
        cli_schema = set(pq.read_schema(cli_parquets[0]).names)
        py_schema = set(pq.read_schema(py_parquets[0]).names)
        assert column in cli_schema, (
            f"CLI thermals schema must carry {column}; got {sorted(cli_schema)}"
        )
        assert column in py_schema, (
            f"Python thermals schema must carry {column}; got {sorted(py_schema)}"
        )

    cli_rows = _read_thermal_rows(cli_parquets[0])
    py_rows = _read_thermal_rows(py_parquets[0])
    assert len(cli_rows) == len(py_rows), (
        f"thermals row count mismatch: CLI={len(cli_rows)}, Python={len(py_rows)}"
    )

    # Non-vacuity: the case must genuinely exercise the anticipated columns, so at
    # least one committed value is non-null. Without this a study with no
    # anticipated plant (all-null both sides) would pass the equality vacuously.
    assert any(row["anticipated_committed_mw"] is not None for row in cli_rows), (
        "the anticipated case must carry at least one non-null anticipated_committed_mw"
    )

    mismatches = [
        f"  row {idx} col {column}: CLI={cli[column]!r} PY={py[column]!r}"
        for idx, (cli, py) in enumerate(zip(cli_rows, py_rows))
        for column in _ANTICIPATED_COLUMNS
        if not _anticipated_values_equal(cli[column], py[column])
    ]
    assert not mismatches, (
        "CLI and Python anticipated columns diverge (including null positions; "
        "Python-parity violation):\n" + "\n".join(mismatches)
    )
