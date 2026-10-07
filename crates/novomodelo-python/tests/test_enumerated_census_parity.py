"""CLI / Python parity for the enumerated (census) simulation.

`cargo test --workspace` never builds `cobre-python` (it is workspace-excluded —
see `CONTRIBUTING.md`), so the Rust-side census gates in
`crates/cobre-sddp/tests/simulation_integration.rs` and
`crates/cobre-sddp/tests/mpi_wire.rs` say nothing about the Python write path.
This module closes that gap for `simulation.selection = enumerated`: it drives
a small `K = 2` census through both the compiled CLI binary and
`cobre.Study.train()` / `cobre.Study.simulate()`, then asserts the two output
trees agree on `scenario_summary.parquet`'s `mean_cost`/`std_cost`/
`probability` columns and on the per-entity `thermals` simulation output.

No committed case declares a branching policy graph today, so the fixture is
built here: a copy of the `d01-thermal-dispatch` deterministic example (single
bus, two thermals, two stages, zero hydros — the smallest shipped case) with
its `stages.json` `policy_graph` replaced by an explicit two-leaf fan (root at
stage 0, two leaves at stage 1 under non-uniform declared probabilities) and
its `config.json` `training`/`simulation` selections switched to `enumerated`.
`num_openings: 1` on both stages is left untouched — the K-fan shape comes
entirely from the declared nodes/transitions, exactly mirroring
`cobre_sddp::test_support::k_fan_policy_graph`'s construction, never from a
within-node opening count (which the enumerated engine's admission gate
requires to stay `1`).

Run with (from the repo root, after `maturin develop --release
--manifest-path crates/cobre-python/Cargo.toml`):
    pytest crates/cobre-python/tests/test_enumerated_census_parity.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
_D01_CASE = _REPO_ROOT / "examples" / "deterministic" / "d01-thermal-dispatch"

# Non-uniform declared leaf probabilities for the two-leaf fan — never 0.5/0.5,
# so a weight/mean mismatch has power to fail.
_LEAF_0_PROBABILITY = 0.35
_LEAF_1_PROBABILITY = 0.65


def _make_census_case(dest: pathlib.Path) -> None:
    """Copy D01 to `dest` and rewrite it into a two-leaf enumerated census.

    `stages.json`'s `policy_graph` gains an explicit `nodes`/`transitions` fan
    (root at stage 0, two leaves at stage 1, non-uniform declared
    probabilities); `config.json`'s `training.selection` and
    `simulation.selection` both switch to `enumerated`, with simulation
    enabled.
    """
    for item in _D01_CASE.iterdir():
        dst_item = dest / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    stages_path = dest / "stages.json"
    stages = json.loads(stages_path.read_text())
    stages["policy_graph"] = {
        "type": "finite_horizon",
        "annual_discount_rate": 0.0,
        "nodes": [
            {"id": 0, "stage_id": 0},
            {"id": 1, "stage_id": 1},
            {"id": 2, "stage_id": 1},
        ],
        "transitions": [
            {"source_id": 0, "target_id": 1, "probability": _LEAF_0_PROBABILITY},
            {"source_id": 0, "target_id": 2, "probability": _LEAF_1_PROBABILITY},
        ],
    }
    stages_path.write_text(json.dumps(stages))

    config_path = dest / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["selection"] = {"method": "enumerated"}
    config["simulation"] = {
        "enabled": True,
        "selection": {"method": "enumerated"},
    }
    config_path.write_text(json.dumps(config))


def _scenario_summary_table(output_dir: pathlib.Path) -> pa.Table:
    path = output_dir / "simulation" / "scenario_summary.parquet"
    assert path.is_file(), f"simulation/scenario_summary.parquet must exist at {path}"
    return pq.read_table(path)


def _thermals_table(output_dir: pathlib.Path) -> pa.Table:
    thermals_dir = output_dir / "simulation" / "thermals"
    assert thermals_dir.is_dir(), f"simulation/thermals/ must exist at {thermals_dir}"
    parquets = sorted(thermals_dir.rglob("*.parquet"))
    assert parquets, f"no parquet files found under {thermals_dir}"
    # ParquetFile (not read_table): scenario_id is both the Hive partition key
    # and an in-file Int32 column, so read_table's dataset path would try to
    # merge them and raise on the collision.
    tables = [pq.ParquetFile(p).read() for p in parquets]
    return pa.concat_tables(tables).sort_by(
        [("scenario_id", "ascending"), ("stage_id", "ascending")]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def census_case_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A two-leaf enumerated-census case, built once and shared across tests."""
    dest = tmp_path_factory.mktemp("census_case")
    _make_census_case(dest)
    return dest


@pytest.fixture(scope="module")
def census_cli_output(
    census_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    cli_binary: pathlib.Path,
) -> pathlib.Path:
    """Run the census case via the CLI and return the output directory."""
    output_dir = tmp_path_factory.mktemp("census_cli_out")
    run_cli(census_case_dir, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def census_python_output(
    census_case_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Run the census case via `cobre.Study.train()` / `.simulate()`.

    Drives the lower-level `Study` API (not the `cobre.run.run()` convenience
    wrapper) so the test genuinely exercises `Study.simulate` against a
    trained `Policy`, per this gate's charter.
    """
    cobre = pytest.importorskip("cobre")
    output_dir = tmp_path_factory.mktemp("census_python_out")
    study = cobre.Study(str(census_case_dir), output_dir=str(output_dir))
    policy = study.train()
    study.simulate(policy)
    return output_dir


# ---------------------------------------------------------------------------
# Non-vacuity: the fixture genuinely exercises a K=2 census
# ---------------------------------------------------------------------------


def test_python_census_produces_two_scenarios_with_probability(
    census_python_output: pathlib.Path,
) -> None:
    """The Python run's `scenario_summary.parquet` has 2 rows, weights summing to 1."""
    table = _scenario_summary_table(census_python_output)
    assert table.num_rows == 2, f"expected 2 census scenarios, got {table.num_rows}"

    probabilities = table.column("probability").to_pylist()
    assert all(p is not None for p in probabilities), (
        f"every row's probability must be populated (Some) under a census: {probabilities}"
    )
    assert abs(sum(probabilities) - 1.0) < 1e-9, (
        f"census probabilities must sum to 1.0, got {probabilities}"
    )
    assert sorted(probabilities) == sorted(
        [_LEAF_0_PROBABILITY, _LEAF_1_PROBABILITY]
    ), (
        f"probabilities must equal the fixture's declared leaf edge weights: {probabilities}"
    )


# ---------------------------------------------------------------------------
# CLI vs Python parity
# ---------------------------------------------------------------------------


def test_cli_python_scenario_summary_bit_for_bit(
    census_cli_output: pathlib.Path,
    census_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce identical `scenario_summary.parquet` contents.

    `mean_cost`/`std_cost` are not columns of this table (they live in
    `metadata.json`); this table's per-scenario `discounted_immediate_cost`
    and `probability` columns are what R6 requires to match exactly, and both
    write paths converge on `cobre_io::write_scenario_summary`.
    """
    cli_table = _scenario_summary_table(census_cli_output).sort_by(
        [("scenario_id", "ascending")]
    )
    py_table = _scenario_summary_table(census_python_output).sort_by(
        [("scenario_id", "ascending")]
    )

    assert cli_table.schema.equals(py_table.schema), (
        f"schema mismatch:\n  CLI: {cli_table.schema}\n  PY:  {py_table.schema}"
    )
    for col in ("scenario_id", "probability", "discounted_immediate_cost"):
        cli_col = cli_table.column(col)
        py_col = py_table.column(col)
        assert cli_col.equals(py_col), (
            f"column '{col}' mismatch:\n"
            f"  CLI: {cli_col.to_pylist()}\n"
            f"  PY:  {py_col.to_pylist()}"
        )


def test_cli_python_metadata_cost_bit_for_bit(
    census_cli_output: pathlib.Path,
    census_python_output: pathlib.Path,
) -> None:
    """CLI and Python `simulation/metadata.json` report identical `mean_cost`/`std_cost`."""
    cli_metadata = json.loads(
        (census_cli_output / "simulation" / "metadata.json").read_text()
    )
    py_metadata = json.loads(
        (census_python_output / "simulation" / "metadata.json").read_text()
    )
    cli_cost = cli_metadata["cost"]
    py_cost = py_metadata["cost"]
    assert cli_cost == py_cost, (
        f"metadata.json 'cost' mismatch:\n  CLI: {cli_cost}\n  PY:  {py_cost}"
    )


def test_cli_python_thermals_bit_for_bit(
    census_cli_output: pathlib.Path,
    census_python_output: pathlib.Path,
) -> None:
    """CLI and Python produce identical per-entity `simulation/thermals/` output."""
    cli_table = _thermals_table(census_cli_output)
    py_table = _thermals_table(census_python_output)

    assert cli_table.schema.equals(py_table.schema), (
        f"schema mismatch:\n  CLI: {cli_table.schema}\n  PY:  {py_table.schema}"
    )
    assert cli_table.num_rows == py_table.num_rows, (
        f"row-count mismatch: CLI={cli_table.num_rows}, Python={py_table.num_rows}"
    )

    mismatches: list[str] = []
    for name in cli_table.schema.names:
        cli_col = cli_table.column(name)
        py_col = py_table.column(name)
        if not cli_col.equals(py_col):
            mismatches.append(
                f"  column '{name}':\n"
                f"    CLI: {cli_col.to_pylist()}\n"
                f"    PY:  {py_col.to_pylist()}"
            )
    assert not mismatches, (
        "thermals column mismatch between CLI and Python:\n" + "\n".join(mismatches)
    )
