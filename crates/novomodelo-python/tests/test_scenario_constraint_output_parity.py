"""CLI-vs-Python parity for the generic-constraint outputs of a scenario case.

The Python-parity hard rule (`CLAUDE.md`): every output file the CLI writes
must also be written by the Python bindings. This runs both write paths
end-to-end on a temporary case with four generic constraints and asserts that
`simulation/violations/generic`, `simulation/generic_activation`, the
`generic_violation_cost` column of `simulation/costs` and the resolved echo
`generic_constraints/resolved_echo.parquet` match on both sides, and that
`novomodelo.results.load_simulation` exposes the two simulation families.

The case is `examples/deterministic/d02-single-hydro` (one 50 MW hydro, one
bus) copied to a temporary directory with simulation enabled for 2 sampled
scenarios. Each constraint has one bounds row per (stage, block) with null
endpoints and a soft slack at 10 times the largest deficit cost of the case.
The four constraints cover one case each:

- 701 `hydro_generation(0) >= 1000` is violated (the plant caps at 50 MW);
- 702 `hydro_generation(0) <= hydro_availability(0)` reads `hydro_availability`;
- 703 holds under an always-true `active_when`;
- 704 holds under an always-false `active_when`.

Only 703 and 704 have activation rows: a constraint with neither
`active_when` nor a per-solve side has none.

`novomodelo-python` is excluded from the cargo workspace (it needs a Python
interpreter to build), so `cargo test --workspace` never runs this gate; it
runs in novomodelo-python's own test job:

    maturin develop --release --manifest-path crates/novomodelo-python/Cargo.toml
    pytest crates/novomodelo-python/tests/test_scenario_constraint_output_parity.py -v

Partitions are read with `pq.ParquetFile` and the file set by a directory walk,
never through a Hive dataset: `scenario_id` is both the partition key and an
in-file column, so a dataset read would fail on the collision.
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _novomodelo_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
D02_CASE = _REPO_ROOT / "examples" / "deterministic" / "d02-single-hydro"

_VIOLATIONS_DIR = "simulation/violations/generic"
_ACTIVATION_DIR = "simulation/generic_activation"
_COSTS_DIR = "simulation/costs"
_ECHO_FILE = "generic_constraints/resolved_echo.parquet"
_CONSTRAINT_PREFIXES = (_VIOLATIONS_DIR, _ACTIVATION_DIR, "generic_constraints")

_SCENARIO_IDS = (0, 1)
_STAGE_IDS = (0, 1)
_BLOCK_IDS = (0,)
_CONSTRAINT_IDS = (701, 702, 703, 704)
_GATED_CONSTRAINT_IDS = (703, 704)


def _constraints(slack: dict) -> dict:
    return {
        "constraints": [
            {
                "id": 701,
                "name": "constraint_701",
                "expression": "hydro_generation(0) >= 1000",
                "slack": slack,
            },
            {
                "id": 702,
                "name": "constraint_702",
                "expression": "hydro_generation(0) <= hydro_availability(0)",
                "slack": slack,
            },
            {
                "id": 703,
                "name": "constraint_703",
                "expression": "hydro_generation(0) <= 1000000",
                "active_when": "bus_load_value(0) <= 1000000000",
                "slack": slack,
            },
            {
                "id": 704,
                "name": "constraint_704",
                "expression": "hydro_generation(0) <= 1000000",
                "active_when": "bus_load_value(0) <= 0",
                "slack": slack,
            },
        ]
    }


def _default_slack(case: pathlib.Path) -> dict:
    """Per hour, at 10 times the largest deficit cost of the case's buses."""
    buses = json.loads((case / "system" / "buses.json").read_text(encoding="utf-8"))
    max_deficit_cost = max(
        segment["cost"] for bus in buses["buses"] for segment in bus["deficit_segments"]
    )
    return {
        "enabled": True,
        "penalty": 10.0 * max_deficit_cost,
        "penalty_basis": "per_hour",
    }


def _write_bounds_rows_per_cell(case: pathlib.Path) -> None:
    """One bounds row with null endpoints per (stage, block) of each constraint."""
    rows = [
        (constraint_id, stage_id, block_id)
        for constraint_id in _CONSTRAINT_IDS
        for stage_id in _STAGE_IDS
        for block_id in _BLOCK_IDS
    ]
    schema = pa.schema(
        [
            pa.field("constraint_id", pa.int32(), nullable=False),
            pa.field("stage_id", pa.int32(), nullable=False),
            pa.field("block_id", pa.int32()),
            pa.field("bound_lower", pa.float64()),
            pa.field("bound_upper", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "constraint_id": [r[0] for r in rows],
            "stage_id": [r[1] for r in rows],
            "block_id": [r[2] for r in rows],
            "bound_lower": [None] * len(rows),
            "bound_upper": [None] * len(rows),
        },
        schema=schema,
    )
    pq.write_table(
        table,
        case / "constraints" / "generic_constraint_bounds.parquet",
        compression="none",
    )


def _relative_files(root: pathlib.Path) -> set[str]:
    """Return the set of file paths under `root`, POSIX-relative to `root`."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _constraint_files(root: pathlib.Path) -> set[str]:
    """Return the generic-constraint output files under `root`, POSIX-relative to `root`."""
    return {
        f
        for f in _relative_files(root)
        if f.startswith(tuple(f"{prefix}/" for prefix in _CONSTRAINT_PREFIXES))
    }


def _partition(directory: str, scenario_id: int) -> str:
    return f"{directory}/scenario_id={scenario_id:04d}/data.parquet"


def _partition_files(directory: str) -> list[str]:
    return [_partition(directory, scenario_id) for scenario_id in _SCENARIO_IDS]


def _read_rows(root: pathlib.Path, relative: str) -> list[dict]:
    assert (root / relative).is_file(), f"{root} is missing {relative}"
    return pq.ParquetFile(root / relative).read().to_pylist()


def _read_partitions(root: pathlib.Path, directory: str) -> list[dict]:
    return [
        row
        for relative in _partition_files(directory)
        for row in _read_rows(root, relative)
    ]


@pytest.fixture(scope="module")
def case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A temporary copy of d02 with four generic constraints and a 2-scenario simulation."""
    assert D02_CASE.is_dir(), f"the d02 fixture must exist at {D02_CASE}"
    case = tmp_path_factory.mktemp("scenario_constraint_case") / "d02-constraints"
    shutil.copytree(D02_CASE, case)

    config_path = case / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["simulation"] = {
        "enabled": True,
        "selection": {"method": "sampled", "num_scenarios": len(_SCENARIO_IDS)},
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    constraints_dir = case / "constraints"
    constraints_dir.mkdir(exist_ok=True)
    (constraints_dir / "generic_constraints.json").write_text(
        json.dumps(_constraints(_default_slack(case)), indent=2), encoding="utf-8"
    )
    _write_bounds_rows_per_cell(case)
    return case


@pytest.fixture(scope="module")
def cli_output(
    case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run the case through the compiled CLI binary."""
    output_dir = tmp_path_factory.mktemp("scenario_constraint_cli_out")
    run_cli(case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def python_output(
    case_dir: pathlib.Path, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Run the case through the module-level Python bindings entry point."""
    import novomodelo.run

    output_dir = tmp_path_factory.mktemp("scenario_constraint_python_out")
    novomodelo.run.run(str(case_dir), output_dir=str(output_dir))
    return output_dir


def test_constraint_file_sets_are_equal_with_one_partition_per_scenario(
    cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """Both write paths emit the same files: the echo and one partition per scenario."""
    cli_files = _constraint_files(cli_output)
    py_files = _constraint_files(python_output)
    assert cli_files == py_files, (
        f"CLI and Python generic-constraint file sets diverge: "
        f"only CLI {sorted(cli_files - py_files)}, only Python {sorted(py_files - cli_files)}"
    )
    expected = {
        _ECHO_FILE,
        *_partition_files(_VIOLATIONS_DIR),
        *_partition_files(_ACTIVATION_DIR),
    }
    assert cli_files == expected, (
        f"unexpected generic-constraint file set {sorted(cli_files)}; "
        f"expected {sorted(expected)}"
    )


def test_constraint_partitions_and_echo_are_identical(
    cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """Each generic-constraint partition and the echo have equal tables column by column."""
    for relative in (
        *_partition_files(_VIOLATIONS_DIR),
        *_partition_files(_ACTIVATION_DIR),
        _ECHO_FILE,
    ):
        for side, root in (("CLI", cli_output), ("Python", python_output)):
            assert (root / relative).is_file(), f"{side} is missing {relative}"
        cli_table = pq.ParquetFile(cli_output / relative).read()
        py_table = pq.ParquetFile(python_output / relative).read()
        assert cli_table.schema.names == py_table.schema.names, (
            f"CLI and Python column names diverge in {relative}"
        )
        assert cli_table.to_pydict() == py_table.to_pydict(), (
            f"CLI and Python contents diverge in {relative}"
        )


def test_activation_rows_follow_active_when(
    cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """703 is always active, 704 never is and carries no bounds, one row per key for 703-704."""
    for side, root in (("CLI", cli_output), ("Python", python_output)):
        rows = _read_partitions(root, _ACTIVATION_DIR)
        keys = [
            (r["scenario_id"], r["stage_id"], r["block_id"], r["constraint_id"])
            for r in rows
        ]
        expected = {
            (scenario_id, stage_id, block_id, constraint_id)
            for scenario_id in _SCENARIO_IDS
            for stage_id in _STAGE_IDS
            for block_id in _BLOCK_IDS
            for constraint_id in _GATED_CONSTRAINT_IDS
        }
        assert len(keys) == len(set(keys)), f"{side} activation has duplicate keys"
        assert set(keys) == expected, f"{side} activation keys diverge from the case"

        for row in rows:
            if row["constraint_id"] == 703:
                assert row["active"] is True, f"{side} 703 must always be active: {row}"
            if row["constraint_id"] == 704:
                assert row["active"] is False, f"{side} 704 must never be active: {row}"
                assert row["bound_lower"] is None, f"{side} 704 bound_lower: {row}"
                assert row["bound_upper"] is None, f"{side} 704 bound_upper: {row}"


def test_violations_and_generic_violation_cost(
    cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """701 is violated on every row and the costs column matches the slack cost sum, on both paths."""
    costs_by_side = {}
    for side, root in (("CLI", cli_output), ("Python", python_output)):
        violations = _read_partitions(root, _VIOLATIONS_DIR)
        slack_701 = [r for r in violations if r["constraint_id"] == 701]
        assert slack_701, f"{side} violations hold no row for constraint 701"
        assert all(r["slack_value"] > 0 for r in slack_701), (
            f"{side} 701 must have slack_value > 0 on every row"
        )

        slack_cost: dict[tuple[int, int], float] = defaultdict(float)
        for r in violations:
            slack_cost[(r["scenario_id"], r["stage_id"])] += r["slack_cost"]

        cost = {
            (r["scenario_id"], r["stage_id"]): r["generic_violation_cost"]
            for r in _read_partitions(root, _COSTS_DIR)
        }
        assert cost.keys() == slack_cost.keys(), (
            f"{side} costs and violations disagree on (scenario_id, stage_id)"
        )
        for key, expected in slack_cost.items():
            assert math.isclose(cost[key], expected, rel_tol=1e-9), (
                f"{side} generic_violation_cost {cost[key]} != "
                f"sum of slack_cost {expected} at (scenario_id, stage_id) {key}"
            )
        costs_by_side[side] = cost

    assert costs_by_side["CLI"] == costs_by_side["Python"], (
        "CLI and Python generic_violation_cost diverge"
    )


def test_echo_has_the_18_column_schema_and_the_availability_read(
    cli_output: pathlib.Path, python_output: pathlib.Path
) -> None:
    """The echo has 18 columns and 702 reads hydro_availability = 50.0 on every (stage_id, block_id)."""
    for side, root in (("CLI", cli_output), ("Python", python_output)):
        table = pq.ParquetFile(root / _ECHO_FILE).read()
        assert table.num_columns == 18, f"{side} echo must carry the 18-column schema"

        reads = {
            (r["stage_id"], r["block_id"]): r["value"]
            for r in table.to_pylist()
            if r["constraint_id"] == 702 and r["variable_kind"] == "hydro_availability"
        }
        expected = {
            (stage_id, block_id): 50.0
            for stage_id in _STAGE_IDS
            for block_id in _BLOCK_IDS
        }
        assert reads == expected, (
            f"{side} 702 hydro_availability reads diverge: {reads}"
        )


def test_load_simulation_exposes_the_constraint_families(
    python_output: pathlib.Path,
) -> None:
    """`load_simulation` returns as many rows as the partitions for both generic-constraint families."""
    import novomodelo.results

    for entity_type, directory in (
        ("violations/generic", _VIOLATIONS_DIR),
        ("generic_activation", _ACTIVATION_DIR),
    ):
        loaded = novomodelo.results.load_simulation(
            str(python_output), entity_type=entity_type
        )
        written = _read_partitions(python_output, directory)
        assert written, f"{entity_type} partitions must not be empty"
        assert len(loaded) == len(written), (
            f"load_simulation({entity_type!r}) returned {len(loaded)} rows, "
            f"the partitions hold {len(written)}"
        )
