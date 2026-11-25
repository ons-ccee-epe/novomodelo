"""Python-parity guard for the post-horizon anticipated-lane simulation output.

The `anticipated_lanes/` partition carries one row per anticipated thermal's
genuine decision whose delivery target lands past the study horizon (a
post-study delivery), read from the commitment-hold ring's carried state. Both
the CLI (`crates/novomodelo-cli/src/commands/run/simulation.rs`) and the Python
bindings (`crates/novomodelo-python/src/run.rs`) reach the partition through the
SAME shared path — `ScenarioWritePayload::from(scenario_result)` into
`SimulationParquetWriter::write_scenario`, where `scenario_result.
anticipated_lanes` is populated by the single shared `extract_anticipated_lanes`
in `novomodelo-sddp`. novomodelo-python has no anticipated-lane-specific output code, so
Python parity holds by construction. This module is the guard that proves it
end-to-end: it runs a post-study case through both the compiled `novomodelo` CLI
(`novomodelo run --output`) and `novomodelo.run.run`, reads both `simulation/
anticipated_lanes/**/data.parquet`, and asserts the rows are element-wise
equal. It also asserts the symmetric negative — a study with no post-study
continuation writes no `anticipated_lanes/` directory at all.

Post-study case
---------------
The row-parity guard runs `POST_STUDY_CASE`, the committed case whose
`stages.json` declares `post_study_thermal_stages` AND whose anticipated
thermal's lead reaches a post-study delivery target, so the partition is
genuinely populated (a study with no such reach writes no `anticipated_lanes/`
directory — see the symmetric-negative test below).

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_anticipated_lanes_output_parity.py -v
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil

import pyarrow.parquet as pq

from _novomodelo_cli import run_cli

# Resolved against the repo root so the tests are independent of pytest's
# working directory (never a CWD-relative path).
_REPO_ROOT = pathlib.Path(__file__).parents[3]

POST_STUDY_CASE = (
    _REPO_ROOT / "examples" / "deterministic" / "d55-post-study-anticipated-lanes"
)

# A known case with no post-study continuation, simulation already enabled:
# the partition must NOT appear for it (the symmetric negative).
NO_POST_STUDY_CASE = _REPO_ROOT / "examples" / "1dtoy"

# The four value columns compared row by row. `stage_id`/`node_id`
# are read too, but only as the row sort key, mirroring
# `test_anticipated_output_parity.py`'s `stage_id`/`block_id`/`thermal_id` key.
_LANE_VALUE_COLUMNS = (
    "thermal_id",
    "delivery_date",
    "deposited_decision_mw",
    "carried_committed_mw",
    "delivery_start_date",
    "delivery_end_date",
)
_SORT_FIELDS = ("stage_id", "node_id", "thermal_id", "delivery_date")


def _lanes_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the anticipated_lanes data.parquet partitions under an output dir."""
    lanes_dir = output_dir / "simulation" / "anticipated_lanes"
    if not lanes_dir.is_dir():
        return []
    return sorted(lanes_dir.rglob("*.parquet"))


def _make_case_with_simulation(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy a deterministic case to ``dest`` and enable the simulation pass.

    Deterministic cases that ship ``simulation.enabled = false`` (training-only
    parity cases) need this so the simulation write path is exercised. Mirrors
    ``test_anticipated_output_parity.py``.
    """
    shutil.copytree(src, dest, dirs_exist_ok=True)

    config = json.loads((src / "config.json").read_text())
    config.setdefault("simulation", {})["enabled"] = True
    (dest / "config.json").write_text(json.dumps(config))


def _read_lane_rows(parquet: pathlib.Path) -> list[dict[str, object]]:
    """Read an anticipated_lanes parquet into rows sorted on `_SORT_FIELDS`,
    so two writers that emit the same rows in a different physical order
    compare equal.
    """

    def sort_key(row: dict[str, object]) -> tuple[object, ...]:
        return tuple(row[field] for field in _SORT_FIELDS)

    # Read the single file directly: scenario_id is both the Hive partition key
    # and an in-file column, so pq.read_table's dataset path would try to merge
    # the two same-named fields and error. ParquetFile reads only the file.
    rows = pq.ParquetFile(parquet).read().to_pylist()
    return sorted(rows, key=sort_key)


def _lane_values_equal(left: object, right: object) -> bool:
    """Compare one lane cell null-inclusively.

    Every `anticipated_lanes` column is non-nullable by schema, but the
    comparator stays null-safe for the same reason
    `test_anticipated_output_parity.py`'s does: a null appearing on only one
    side is a schema/writer divergence, not something to crash the comparison
    loop on.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def test_cli_python_anticipated_lanes_row_parity(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """The CLI and the Python surface emit identical anticipated_lanes rows.

    Runs the post-study case through both the compiled `novomodelo` CLI (`novomodelo run
    --output`) and `novomodelo.run.run`, reads both anticipated_lanes parquets,
    sorts each by `_SORT_FIELDS`, and asserts `thermal_id`, `delivery_date`,
    `deposited_decision_mw`, and `carried_committed_mw` are element-wise equal.
    If the CLI and Python paths diverge between the LP solve and the Parquet
    write, this fails.
    """
    import novomodelo.run

    assert POST_STUDY_CASE.is_dir(), (
        f"the committed post-study case must exist at {POST_STUDY_CASE}"
    )

    case_dir = tmp_path / "post_study_case"
    case_dir.mkdir()
    _make_case_with_simulation(POST_STUDY_CASE, case_dir)

    out_cli = tmp_path / "cli_out"
    run_cli(case_dir, out_cli, cli_binary)

    out_py = tmp_path / "py_out"
    novomodelo.run.run(str(case_dir), output_dir=str(out_py))

    cli_parquets = _lanes_parquets(out_cli)
    py_parquets = _lanes_parquets(out_py)
    assert len(cli_parquets) > 0, (
        "CLI must emit an anticipated_lanes parquet for the post-study case"
    )
    assert len(py_parquets) > 0, (
        "Python must emit an anticipated_lanes parquet for the post-study case"
    )

    cli_schema = set(pq.read_schema(cli_parquets[0]).names)
    py_schema = set(pq.read_schema(py_parquets[0]).names)
    for column in _LANE_VALUE_COLUMNS:
        assert column in cli_schema, (
            f"CLI anticipated_lanes schema must carry {column}; got {sorted(cli_schema)}"
        )
        assert column in py_schema, (
            f"Python anticipated_lanes schema must carry {column}; got {sorted(py_schema)}"
        )

    cli_rows = _read_lane_rows(cli_parquets[0])
    py_rows = _read_lane_rows(py_parquets[0])
    assert len(cli_rows) == len(py_rows), (
        f"anticipated_lanes row count mismatch: CLI={len(cli_rows)}, Python={len(py_rows)}"
    )

    # Non-vacuity: the case must genuinely populate the partition, so the
    # element-wise comparison below cannot pass vacuously on zero rows.
    assert len(cli_rows) > 0, (
        "the post-study case must produce at least one anticipated_lanes row"
    )

    mismatches = [
        f"  row {idx} col {column}: CLI={cli[column]!r} PY={py[column]!r}"
        for idx, (cli, py) in enumerate(zip(cli_rows, py_rows, strict=True))
        for column in _LANE_VALUE_COLUMNS
        if not _lane_values_equal(cli[column], py[column])
    ]
    assert not mismatches, (
        "CLI and Python anticipated_lanes columns diverge (Python-parity "
        "violation):\n" + "\n".join(mismatches)
    )


def test_no_post_study_continuation_produces_no_anticipated_lanes_partition(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """A study with no post-study continuation writes no anticipated_lanes/ dir.

    The symmetric negative, checked on both surfaces: this guards against a
    future regression that emits the partition (empty or otherwise)
    unconditionally rather than only when a genuine post-study decision
    exists.
    """
    import novomodelo.run

    assert NO_POST_STUDY_CASE.is_dir(), (
        f"the no-post-study fixture must exist at {NO_POST_STUDY_CASE}"
    )

    out_py = tmp_path / "py_out"
    novomodelo.run.run(str(NO_POST_STUDY_CASE), output_dir=str(out_py))
    assert _lanes_parquets(out_py) == [], (
        "Python must not emit an anticipated_lanes partition for a study with "
        "no post-study continuation"
    )

    out_cli = tmp_path / "cli_out"
    run_cli(NO_POST_STUDY_CASE, out_cli, cli_binary)
    assert _lanes_parquets(out_cli) == [], (
        "CLI must not emit an anticipated_lanes partition for a study with no "
        "post-study continuation"
    )
