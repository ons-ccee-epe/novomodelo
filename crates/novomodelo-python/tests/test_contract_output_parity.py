"""Python-parity guard for the energy-contract simulation output.

The contract LP adds a NEW simulation output file
(`simulation/contracts/scenario_id=NNNN/data.parquet`). The Python-parity hard
rule (`CLAUDE.md`) requires every output the CLI writes to also be written by the
`cobre-python` bindings, so this new file must be confirmed to flow through the
Python paths.

The committed architecture routes all three surfaces (the `cobre` CLI, the
`cobre.Study.simulate` binding, and the module-level `cobre.run.run`) through a
single shared `write_simulation_results` call, and the writer is schema-driven
and already emits the contract partition. This module is the guard that proves it
end-to-end: it loads the D41 energy-contracts study (the only case shipping
`system/energy_contracts.json`), runs it through both Python surfaces, and asserts
the partition appears with the exact 8-field schema. It also subprocesses the
`cobre` CLI binary and compares the contract rows column-by-column to the Python
output (CLI/Python row parity), and asserts the symmetric negative — a
zero-contract study produces no contract partition — so a future regression that
emits an empty partition (or drops the populated one) fails loudly.

The D41 case ships `simulation.enabled = false` (it is a training-only parity
case). The Python surfaces and the CLI honor that flag, so the tests run against a
temp copy of D41 with simulation enabled (`_make_case_with_simulation`), which is
the convention `test_parity_hydros.py` established for deterministic cases that
ship simulation disabled. The zero-contract case (`examples/1dtoy`) already ships
simulation enabled, so the symmetric-negative test uses it unchanged.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_contract_output_parity.py -v
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil

import pyarrow.parquet as pq

from _cobre_cli import run_cli

# The D41 energy-contracts case is the only example shipping
# system/energy_contracts.json, so the only case whose run produces a contracts/
# partition. Resolved against the repo root so the test is independent of pytest's
# working directory.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
CONTRACT_CASE = _REPO_ROOT / "examples" / "deterministic" / "d41-energy-contracts"

# A known zero-contract case with simulation enabled: the partition must NOT
# appear for it (the symmetric negative).
ZERO_CONTRACT_CASE = _REPO_ROOT / "examples" / "1dtoy"

# The exact 8 fields of the contracts output schema. The schema is owned by
# `contracts_schema()` in cobre-io's schemas.rs; this set mirrors its shape so a
# schema drift (added/removed/renamed column) fails this test. `energy_mwh` is the
# writer-derived column, present in the parquet even though it is not a
# `SimulationContractResult` struct field.
CONTRACT_SCHEMA_FIELDS = {
    "scenario_id",
    "stage_id",
    "node_id",
    "block_id",
    "contract_id",
    "power_mw",
    "energy_mwh",
    "price_per_mwh",
    "total_cost",
    "operative_state_code",
}

# Columns the CLI/Python row-parity comparison sorts by. `block_id` is nullable in
# contracts_schema (Int32, true), so the sort key maps None to a sentinel.
_SORT_FIELDS = ("stage_id", "block_id", "contract_id")


def _contract_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the contract data.parquet partitions under an output dir."""
    contract_dir = output_dir / "simulation" / "contracts"
    if not contract_dir.is_dir():
        return []
    return sorted(contract_dir.rglob("*.parquet"))


def _make_case_with_simulation(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy a deterministic case to ``dest`` and enable the simulation pass.

    D41 ships ``simulation.enabled = false`` (it is a training-only parity case),
    and the CLI and Python surfaces honor that flag. The parity tests need the
    simulation write path exercised, so the copied ``config.json`` is overwritten
    with ``simulation.enabled = true``. Mirrors ``test_parity_hydros.py``.
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


def _read_contract_rows(parquet: pathlib.Path) -> list[dict[str, object]]:
    """Read a contract parquet into sorted row dicts on the shared schema columns.

    Sorts by ``(stage_id, block_id, contract_id)`` with a None-block_id sentinel,
    so two writers that emit the same rows in a different physical order compare
    equal.
    """

    def sort_key(row: dict[str, object]) -> tuple[object, ...]:
        block = row[_SORT_FIELDS[1]]
        block_key = -1 if block is None else block
        return (row[_SORT_FIELDS[0]], block_key, row[_SORT_FIELDS[2]])

    # ParquetFile (not read_table): scenario_id is both the Hive partition key and
    # an in-file column, so read_table's dataset path would try to merge them.
    rows = pq.ParquetFile(parquet).read().to_pylist()
    return sorted(rows, key=sort_key)


def _rows_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    """Compare two contract rows on the 8 schema columns with float tolerance.

    Numeric columns (everything except a possibly-None ``block_id``) are compared
    with ``math.isclose``; the ``isinstance(..., (int, float))`` guards narrow both
    operands to a real number type for the tolerance comparison and route any
    non-numeric or None value (a nullable ``block_id``) to exact equality.
    """
    for field in CONTRACT_SCHEMA_FIELDS:
        left_value = left[field]
        right_value = right[field]
        if isinstance(left_value, (int, float)) and isinstance(
            right_value, (int, float)
        ):
            if not math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif left_value != right_value:
            return False
    return True


def test_study_simulate_emits_contract_output_with_schema(
    tmp_path: pathlib.Path,
) -> None:
    """Study.train().simulate() emits the contract partition with the 8-field
    schema, and cobre.results surfaces the rows.

    Covers the `cobre.Study` Python surface: a successful load + clean validate,
    a `simulation/contracts/` directory with at least one `data.parquet`, a
    Parquet schema equal to exactly the 8 contract fields, and a non-empty read
    through `cobre.results(entity_type="contracts")`.
    """
    import cobre  # noqa: PLC0415

    assert CONTRACT_CASE.is_dir(), (
        f"the D41 energy-contracts fixture must exist at {CONTRACT_CASE}"
    )

    case_dir = tmp_path / "d41_case"
    case_dir.mkdir()
    _make_case_with_simulation(CONTRACT_CASE, case_dir)

    out = tmp_path / "study_out"
    study = cobre.Study(str(case_dir), output_dir=str(out))

    report = study.validate()
    assert report["valid"] is True, (
        f"the D41 study must validate cleanly; got errors {report['errors']}"
    )
    assert report["errors"] == [], (
        f"D41 fixture must not produce load errors, got {report['errors']}"
    )

    policy = study.train()
    study.simulate(policy)

    parquets = _contract_parquets(out)
    assert len(parquets) > 0, (
        "Study.simulate must emit at least one simulation/contracts/.../data.parquet"
    )

    # Reading the Parquet directly yields exactly the 8 schema fields (the
    # cobre.results read adds a synthetic scenario_id partition column, so the
    # schema-equality assertion reads the file directly).
    schema_names = set(pq.read_schema(parquets[0]).names)
    assert schema_names == CONTRACT_SCHEMA_FIELDS, (
        "contract parquet schema must equal exactly the 8 contract fields; "
        f"got {sorted(schema_names)}"
    )

    # The read-side (cobre.results) surfaces the rows through the already-listed
    # "contracts" entity type. Each row carries the 8 schema fields plus the
    # partition-derived scenario_id.
    rows = cobre.results.load_simulation(str(out), entity_type="contracts")
    assert len(rows) > 0, "cobre.results must surface contract rows for the D41 study"
    assert CONTRACT_SCHEMA_FIELDS <= set(rows[0].keys()), (
        "each contract row must carry the 8 schema fields; "
        f"got {sorted(rows[0].keys())}"
    )


def test_run_via_study_emits_contract_output(tmp_path: pathlib.Path) -> None:
    """The module-level cobre.run.run entry point emits the contract partition for
    the same fixture.

    Covers the second Python surface: cobre.run.run runs train + simulate and must
    produce simulation/contracts/.../data.parquet identically to Study.simulate
    (both converge on the shared write_simulation_results call).
    """
    import cobre.run  # noqa: PLC0415

    assert CONTRACT_CASE.is_dir(), (
        f"the D41 energy-contracts fixture must exist at {CONTRACT_CASE}"
    )

    case_dir = tmp_path / "d41_case"
    case_dir.mkdir()
    _make_case_with_simulation(CONTRACT_CASE, case_dir)

    out = tmp_path / "run_out"
    cobre.run.run(str(case_dir), output_dir=str(out))

    parquets = _contract_parquets(out)
    assert len(parquets) > 0, (
        "cobre.run.run must emit at least one simulation/contracts/.../data.parquet"
    )

    schema_names = set(pq.read_schema(parquets[0]).names)
    assert schema_names == CONTRACT_SCHEMA_FIELDS, (
        "contract parquet schema from run.run must equal exactly the 8 "
        f"contract fields; got {sorted(schema_names)}"
    )


def test_cli_python_contract_row_parity(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """The CLI and the Python surface emit identical contract rows.

    Runs the D41 case (simulation enabled) through both the compiled `cobre` CLI
    (`cobre run --output`) and `cobre.run.run`, reads both contract parquets, sorts
    each by `(stage_id, block_id, contract_id)`, and asserts row-for-row value
    equality on the 8 shared schema columns with a float tolerance. This is the
    end-to-end Python-parity check: if the CLI and Python paths diverge between the
    LP solve and the Parquet write, this fails.
    """
    import cobre.run  # noqa: PLC0415

    assert CONTRACT_CASE.is_dir(), (
        f"the D41 energy-contracts fixture must exist at {CONTRACT_CASE}"
    )

    case_dir = tmp_path / "d41_case"
    case_dir.mkdir()
    _make_case_with_simulation(CONTRACT_CASE, case_dir)

    out_cli = tmp_path / "cli_out"
    run_cli(case_dir, out_cli, cli_binary)

    out_py = tmp_path / "py_out"
    cobre.run.run(str(case_dir), output_dir=str(out_py))

    cli_parquets = _contract_parquets(out_cli)
    py_parquets = _contract_parquets(out_py)
    assert len(cli_parquets) > 0, "CLI must emit a contract parquet for D41"
    assert len(py_parquets) > 0, "Python must emit a contract parquet for D41"

    cli_schema = set(pq.read_schema(cli_parquets[0]).names)
    py_schema = set(pq.read_schema(py_parquets[0]).names)
    assert cli_schema == CONTRACT_SCHEMA_FIELDS, (
        f"CLI contract schema must equal the 8 contract fields; got {sorted(cli_schema)}"
    )
    assert py_schema == CONTRACT_SCHEMA_FIELDS, (
        f"Python contract schema must equal the 8 contract fields; got {sorted(py_schema)}"
    )

    cli_rows = _read_contract_rows(cli_parquets[0])
    py_rows = _read_contract_rows(py_parquets[0])
    assert len(cli_rows) == len(py_rows), (
        f"contract row count mismatch: CLI={len(cli_rows)}, Python={len(py_rows)}"
    )

    mismatches = [
        f"  row {idx}:\n    CLI: {cli}\n    PY : {py}"
        for idx, (cli, py) in enumerate(zip(cli_rows, py_rows))
        if not _rows_equal(cli, py)
    ]
    assert not mismatches, (
        "CLI and Python contract rows diverge on the shared schema columns "
        "(Python-parity violation):\n" + "\n".join(mismatches)
    )


def test_zero_contract_study_emits_no_contract_output(
    tmp_path: pathlib.Path,
) -> None:
    """A study with zero contracts produces no contract partition.

    The writer gates the partition on `n_contracts() > 0`; this asserts the gate
    is symmetric, so a regression that always created the directory (an empty
    partition) would fail here.
    """
    import cobre.run  # noqa: PLC0415

    assert ZERO_CONTRACT_CASE.is_dir(), (
        f"the zero-contract case must exist at {ZERO_CONTRACT_CASE}"
    )

    out = tmp_path / "zero_out"
    cobre.run.run(str(ZERO_CONTRACT_CASE), output_dir=str(out))

    # The simulation must have run (so absence is meaningful, not a skipped sim).
    assert (out / "simulation").is_dir(), (
        "the zero-contract case must still produce a simulation/ directory"
    )
    assert not (out / "simulation" / "contracts").exists(), (
        "a study with zero contracts must NOT create simulation/contracts/"
    )
