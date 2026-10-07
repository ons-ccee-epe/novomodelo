"""Python-parity guard for the pumping-stations simulation output.

Unlike the `hydro_inflow` keyword (which adds no output file and is auto-parity),
the pumping LP adds a NEW simulation output file
(`simulation/pumping_stations/scenario_id=NNNN/data.parquet`). The Python-parity
hard rule (`CLAUDE.md`) requires every output the CLI writes to also be written
by the `cobre-python` bindings, so this new file must be confirmed to flow
through the Python paths.

The committed architecture routes all three Python surfaces through a single
shared `run_simulation_phase_py` -> `cobre_io::write_simulation_results` call
(`cobre.Study.simulate` and the module-level `cobre.run.run` both reach it), and
the writer is schema-driven and already emits the pumping partition. This module
is the guard that proves it end-to-end: it loads a real pumping study, runs it
through both Python surfaces, and asserts the partition appears with the 9-column
schema. It also asserts the symmetric negative — a zero-station study produces no
pumping partition — so a future regression that emits an empty partition (or
drops the populated one) fails loudly.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_pumping_output_parity.py -v
"""

from __future__ import annotations

import pathlib
import tempfile

import pyarrow.parquet as pq

# The pumping fixture lives under the cobre-sddp test tree (the convention the
# B6a parity test established for topology fixtures), not examples/, and is
# resolved against the repo root so the test is independent of pytest's working
# directory. It is a two-hydro transfer (H0 -> H1) with one pumping station whose
# `flow.min_m3s = 10` forces a strictly positive pumped flow, so the pumping
# column participates in the LP and the output rows are non-empty.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
PUMPING_CASE = (
    _REPO_ROOT / "crates" / "cobre-sddp" / "tests" / "fixtures" / "pumping_transfer"
)

# A known zero-pumping-station case with simulation enabled: the partition must
# NOT appear for it (the symmetric negative).
ZERO_PUMPING_CASE = _REPO_ROOT / "examples" / "1dtoy"

# The exact 9 fields of the pumping_stations output schema. The schema is owned
# by `pumping_stations_schema()` in cobre-io's simulation_writer; this set mirrors
# its shape so a schema drift (added/removed/renamed column) fails this test.
PUMPING_SCHEMA_FIELDS = {
    "scenario_id",
    "stage_id",
    "node_id",
    "block_id",
    "pumping_station_id",
    "pumped_flow_m3s",
    "pumped_volume_hm3",
    "power_consumption_mw",
    "energy_consumption_mwh",
    "pumping_cost",
    "operative_state_code",
}


def _pumping_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the pumping-station data.parquet partitions under an output dir."""
    pumping_dir = output_dir / "simulation" / "pumping_stations"
    if not pumping_dir.is_dir():
        return []
    return sorted(pumping_dir.rglob("*.parquet"))


def test_study_simulate_emits_pumping_output_with_schema() -> None:
    """Study.train().simulate() emits the pumping partition with the 9-column
    schema, and cobre.results surfaces the rows.

    Covers the `cobre.Study` Python surface: a successful load + clean validate,
    a `simulation/pumping_stations/` directory with at least one `data.parquet`,
    a Parquet schema equal to exactly the 9 pumping fields, and a non-empty read
    through `cobre.results(entity_type="pumping_stations")`.
    """
    import cobre  # noqa: PLC0415

    assert PUMPING_CASE.is_dir(), f"the pumping fixture must exist at {PUMPING_CASE}"

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        study = cobre.Study(str(PUMPING_CASE), output_dir=out_dir)

        # Sanity: the fixture loads both hydros (source H0 and destination H1).
        assert study.system.n_hydros == 2, (
            f"pumping fixture must load both hydros, got {study.system.n_hydros}"
        )

        report = study.validate()
        assert report["valid"] is True, (
            "the pumping study must validate cleanly (distinct source/destination "
            f"hydros, resolving FK refs); got errors {report['errors']}"
        )
        assert report["errors"] == [], (
            f"pumping fixture must not produce load errors, got {report['errors']}"
        )

        policy = study.train()
        study.simulate(policy)

        parquets = _pumping_parquets(out)
        assert len(parquets) > 0, (
            "Study.simulate must emit at least one "
            "simulation/pumping_stations/.../data.parquet"
        )

        # Reading the Parquet directly yields exactly the 9 schema fields (the
        # cobre.results read adds a synthetic scenario_id partition column, so the
        # schema-equality assertion reads the file directly).
        schema_names = set(pq.read_schema(parquets[0]).names)
        assert schema_names == PUMPING_SCHEMA_FIELDS, (
            "pumping parquet schema must equal exactly the 9 pumping fields; "
            f"got {sorted(schema_names)}"
        )

        # The read-side (cobre.results) surfaces the rows through the already-listed
        # "pumping_stations" entity type. Each row carries the 9 schema fields plus
        # the partition-derived scenario_id.
        rows = cobre.results.load_simulation(out_dir, entity_type="pumping_stations")
        assert len(rows) > 0, (
            "cobre.results must surface pumping_stations rows for the pumping study"
        )
        assert PUMPING_SCHEMA_FIELDS <= set(rows[0].keys()), (
            "each pumping row must carry the 9 schema fields; "
            f"got {sorted(rows[0].keys())}"
        )


def test_run_via_study_emits_pumping_output() -> None:
    """The module-level cobre.run.run entry point (run_via_study) emits the
    pumping partition for the same fixture.

    Covers the second Python surface: cobre.run.run runs train + simulate and
    must produce simulation/pumping_stations/.../data.parquet identically to
    Study.simulate (both converge on run_simulation_phase_py).
    """
    import cobre.run  # noqa: PLC0415

    assert PUMPING_CASE.is_dir(), f"the pumping fixture must exist at {PUMPING_CASE}"

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        cobre.run.run(str(PUMPING_CASE), output_dir=out_dir)

        parquets = _pumping_parquets(out)
        assert len(parquets) > 0, (
            "cobre.run.run must emit at least one "
            "simulation/pumping_stations/.../data.parquet"
        )

        schema_names = set(pq.read_schema(parquets[0]).names)
        assert schema_names == PUMPING_SCHEMA_FIELDS, (
            "pumping parquet schema from run_via_study must equal exactly the 9 "
            f"pumping fields; got {sorted(schema_names)}"
        )


def test_zero_pumping_study_emits_no_pumping_output() -> None:
    """A study with zero pumping stations produces no pumping partition.

    The writer gates the partition on `n_pumping_stations() > 0`; this asserts the
    gate is symmetric, so a regression that always created the directory (an empty
    partition) would fail here.
    """
    import cobre.run  # noqa: PLC0415

    assert ZERO_PUMPING_CASE.is_dir(), (
        f"the zero-pumping case must exist at {ZERO_PUMPING_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        cobre.run.run(str(ZERO_PUMPING_CASE), output_dir=out_dir)

        # The simulation must have run (so absence is meaningful, not a skipped sim).
        assert (out / "simulation").is_dir(), (
            "the zero-pumping case must still produce a simulation/ directory"
        )
        assert not (out / "simulation" / "pumping_stations").exists(), (
            "a study with zero pumping stations must NOT create "
            "simulation/pumping_stations/"
        )
