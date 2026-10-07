"""Python-parity guard for the travel-time in-transit simulation output.

The travel-time water arc adds a NEW simulation output file
(`simulation/in_transit/scenario_id=NNNN/data.parquet`) carrying the per-arc
in-transit bucket volumes and the maturing-now delivery. The Python-parity hard
rule (`CLAUDE.md`) requires every output the CLI writes to also be written by the
`cobre-python` bindings, so this new file must be confirmed to flow through the
Python paths.

Both the CLI and every Python surface route through a single shared
`SimulationParquetWriter::write_scenario(ScenarioWritePayload::from(...))` call,
and the writer creates the `in_transit/` directory only when the system declares
a travel-time arc. The write payload carries the shared `cobre-sddp` bucket
extraction, so the Python surfaces emit the same table as the CLI. This module is
the guard that proves it end-to-end: it loads a real travel-time study, runs it
through both Python surfaces, and asserts the partition appears with the 5-column
schema and the canonical `(downstream hydro_id, lag)` order. It also asserts the
symmetric negative — a non-travel-time study produces no in-transit partition.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_in_transit_output_parity.py -v
"""

from __future__ import annotations

import pathlib
import tempfile

import pyarrow.parquet as pq

# The travel-time fixture is a two-hydro cascade (H0 -> H1) where H0 declares a
# `travel_time_hours` arc feeding H1, spanning two monthly stages so the arc
# sizes to two maturity buckets. It lives under the cobre-sddp test tree (the
# convention for topology fixtures) and is resolved against the repo root so the
# test is independent of pytest's working directory.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
TRAVEL_TIME_CASE = (
    _REPO_ROOT / "crates" / "cobre-sddp" / "tests" / "fixtures" / "travel_time_arc"
)

# A known non-travel-time case with simulation enabled: the partition must NOT
# appear for it (the symmetric negative).
NO_ARC_CASE = _REPO_ROOT / "examples" / "1dtoy"

# The exact 5 fields of the in_transit output schema, owned by
# `in_transit_schema()` in cobre-io's schemas.rs. A schema drift (added/removed/
# renamed column) fails this test.
IN_TRANSIT_SCHEMA_FIELDS = {
    "scenario_id",
    "stage_id",
    "node_id",
    "hydro_id",
    "lag",
    "in_transit_volume_hm3",
    "delayed_arrival_hm3",
}


def _in_transit_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the in_transit data.parquet partitions under an output dir."""
    in_transit_dir = output_dir / "simulation" / "in_transit"
    if not in_transit_dir.is_dir():
        return []
    return sorted(in_transit_dir.rglob("*.parquet"))


def _assert_in_transit_table_shape(parquet_path: pathlib.Path) -> None:
    """Assert one written in_transit partition has the parity schema, canonical
    `(stage_id, lag)` row order, and the lag-1-only delivery contract.

    The delivery (`delayed_arrival_hm3`) is the maturing incoming bucket, which is
    non-zero only at the maturing lag (`lag == 1`); every deeper-maturity row must
    carry `0.0`. This holds for any values, so the assertion is value-robust.
    """
    schema_names = set(pq.read_schema(parquet_path).names)
    assert schema_names == IN_TRANSIT_SCHEMA_FIELDS, (
        "in_transit parquet schema must equal exactly the 5 in_transit fields; "
        f"got {sorted(schema_names)}"
    )

    # Read the single file directly: scenario_id is now both the Hive partition
    # key and an in-file column, so pq.read_table's dataset path would try to
    # merge the two same-named fields and error. ParquetFile reads only the file.
    table = pq.ParquetFile(parquet_path).read(columns=sorted(IN_TRANSIT_SCHEMA_FIELDS))
    cols = table.to_pydict()
    assert table.num_rows > 0, "the travel-time partition must carry at least one row"

    stage_ids = cols["stage_id"]
    lags = cols["lag"]
    hydro_ids = cols["hydro_id"]
    deliveries = cols["delayed_arrival_hm3"]

    # Canonical order: rows are grouped by stage, then ascending lag within a
    # stage (the single-downstream-plant column order). File order must equal the
    # (stage_id, lag)-sorted order.
    order_keys = list(zip(stage_ids, lags, strict=True))
    assert order_keys == sorted(order_keys), (
        "in_transit rows must be written in canonical (stage_id, lag) order; "
        f"got {order_keys}"
    )

    # Every lag is a positive maturity index; the declared arc feeds one
    # downstream plant, so a stable hydro_id repeats across its buckets.
    assert all(lag >= 1 for lag in lags), f"lag must be 1-based, got {lags}"
    assert len(set(hydro_ids)) == 1, (
        f"the single declared arc must feed one downstream plant, got {set(hydro_ids)}"
    )

    # Delivery contract: non-zero only at the maturing lag (lag == 1).
    for lag, delivery in zip(lags, deliveries, strict=True):
        if lag != 1:
            assert delivery == 0.0, (
                f"delayed_arrival_hm3 must be 0.0 at lag {lag} (delivery matures "
                f"only at lag 1); got {delivery}"
            )


def test_study_simulate_emits_in_transit_output_with_schema() -> None:
    """Study.train().simulate() emits the in_transit partition with the 5-column
    schema and canonical order.

    Covers the `cobre.Study` Python surface: a clean validate, a
    `simulation/in_transit/` directory with at least one `data.parquet`, a schema
    equal to exactly the 5 in_transit fields, and canonical (stage_id, lag) row
    order with the lag-1-only delivery contract.
    """
    import cobre  # noqa: PLC0415

    assert TRAVEL_TIME_CASE.is_dir(), (
        f"the travel-time fixture must exist at {TRAVEL_TIME_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        study = cobre.Study(str(TRAVEL_TIME_CASE), output_dir=out_dir)

        assert study.system.n_hydros == 2, (
            f"travel-time fixture must load both hydros, got {study.system.n_hydros}"
        )

        report = study.validate()
        assert report["valid"] is True, (
            "the travel-time study must validate cleanly; "
            f"got errors {report['errors']}"
        )
        assert report["errors"] == [], (
            f"travel-time fixture must not produce load errors, got {report['errors']}"
        )

        study.simulate(study.train())

        parquets = _in_transit_parquets(out)
        assert len(parquets) > 0, (
            "Study.simulate must emit at least one "
            "simulation/in_transit/.../data.parquet"
        )
        for parquet_path in parquets:
            _assert_in_transit_table_shape(parquet_path)


def test_run_via_study_emits_in_transit_output() -> None:
    """The module-level cobre.run.run entry point (run_via_study) emits the
    in_transit partition for the same fixture.

    Covers the second Python surface: cobre.run.run runs train + simulate and
    must produce simulation/in_transit/.../data.parquet identically to
    Study.simulate (both converge on run_simulation_phase_py).
    """
    import cobre.run  # noqa: PLC0415

    assert TRAVEL_TIME_CASE.is_dir(), (
        f"the travel-time fixture must exist at {TRAVEL_TIME_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        cobre.run.run(str(TRAVEL_TIME_CASE), output_dir=out_dir)

        parquets = _in_transit_parquets(out)
        assert len(parquets) > 0, (
            "cobre.run.run must emit at least one "
            "simulation/in_transit/.../data.parquet"
        )
        for parquet_path in parquets:
            _assert_in_transit_table_shape(parquet_path)


def test_non_travel_time_study_emits_no_in_transit_output() -> None:
    """A study with no declared travel-time arc produces no in_transit partition.

    The writer gates the directory on a declared arc; this asserts the gate is
    symmetric, so a regression that always created the directory (byte-noise for
    existing studies) would fail here.
    """
    import cobre.run  # noqa: PLC0415

    assert NO_ARC_CASE.is_dir(), f"the non-travel-time case must exist at {NO_ARC_CASE}"

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        cobre.run.run(str(NO_ARC_CASE), output_dir=out_dir)

        # The simulation must have run (so absence is meaningful, not a skipped sim).
        assert (out / "simulation").is_dir(), (
            "the non-travel-time case must still produce a simulation/ directory"
        )
        assert not (out / "simulation" / "in_transit").exists(), (
            "a study with no travel-time arc must NOT create simulation/in_transit/"
        )
