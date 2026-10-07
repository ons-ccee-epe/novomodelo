"""Auto-parity guard for the `hydro_inflow` generic-constraint keyword.

B6a adds the block-capable `hydro_inflow(h[, k])` generic-constraint keyword
(total realized inflow = incremental `z_inflow(h)` + upstream cascade
turbine/spillage + diverted inflow). It adds **no new output file and no new
schema**: the keyword only contributes columns to an existing generic-constraint
row, whose effect surfaces in already-mirrored outputs (constraint rows/duals,
primal values).

B6a adds no new output; Python parity is satisfied automatically via `cobre_io`.
`cobre-python` carries no independent `VariableRef`/keyword parser — it loads
every system through `cobre_io`, so a keyword that `cobre_io` learns to parse is
accepted by the Python `Study` path with no Python-side change. This test is the
guard for that fact: it loads a multi-plant cascade study whose generic
constraint references `hydro_inflow(1)` through the Python `Study` pyclass and
asserts the load succeeds and validates, proving the keyword reaches the Python
side automatically.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_hydro_inflow_auto_parity.py -v
"""

from __future__ import annotations

import pathlib
import tempfile

# The cascade fixture lives under the cobre-sddp test tree (reused from the B6a
# integration test) rather than examples/, because it is a topology fixture, not
# a shipped example. Resolve it against the repo root so the test is independent
# of pytest's working directory.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
CASCADE_CASE = (
    _REPO_ROOT
    / "crates"
    / "cobre-sddp"
    / "tests"
    / "fixtures"
    / "b6a_hydro_inflow_cascade"
)


def test_study_loads_hydro_inflow_cascade() -> None:
    """Study loads a cascade study whose constraint uses `hydro_inflow(1)`.

    The fixture is a two-plant cascade (H0 -> H1) with a generic constraint
    `hydro_inflow(1) >=` bounding H1's total realized inflow. A successful load
    proves `cobre_io` parses the keyword and the Python `Study` path accepts it
    with no independent Python-side parser — the auto-parity claim.
    """
    import cobre  # noqa: PLC0415

    assert CASCADE_CASE.is_dir(), (
        f"the B6a cascade fixture must exist at {CASCADE_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        study = cobre.Study(str(CASCADE_CASE), output_dir=out_dir)

        system = study.system
        assert isinstance(system, cobre.model.System), (
            "system getter must return a cobre.model.System"
        )
        # The cascade has two hydros (H0 upstream of H1); a load that dropped the
        # cascade topology would not surface both plants.
        assert system.n_hydros == 2, (
            f"the cascade fixture must load both plants, got {system.n_hydros}"
        )
        assert system.n_stages > 0, "loaded cascade must report stages"

        report = study.validate()
        assert report["valid"] is True, (
            "a study whose only generic constraint uses hydro_inflow must load "
            "and validate cleanly through the Python Study path"
        )
        assert report["errors"] == [], (
            f"hydro_inflow(1) must not produce load errors, got {report['errors']}"
        )
