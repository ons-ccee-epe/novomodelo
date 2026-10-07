"""Python-parity smoke test for the d48 windowed IC-defluence-seed case.

The d48 case (`examples/deterministic/d48-travel-time-ic-seed`) declares a
travel-time arc whose `past_defluences` carries a NON-ZERO windowed release
covering `[start_0 - t_v, start_0)`. This test confirms the shared loader
accepts that windowed shape from Python: `cobre.io.load_case` and
`cobre.io.validate` succeed (belt-and-suspenders for the Rust load path), and a
`cobre.Study` constructs — exercising the front-half lifecycle (load ->
stochastic preprocessing -> hydro models -> StudySetup, where the windowed seed
is unrolled) end to end from Python. The epic changes no output schema, so no
output assertion is needed.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_d48_ic_seed_load.py -v
"""

from __future__ import annotations

import pathlib

# The d48 case is resolved against the repo root so the test is independent of
# pytest's working directory.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
D48_CASE = _REPO_ROOT / "examples" / "deterministic" / "d48-travel-time-ic-seed"


def test_load_case_accepts_windowed_defluence_shape() -> None:
    """load_case succeeds on d48 and returns a two-hydro System."""
    import cobre.io  # noqa: PLC0415
    import cobre.model  # noqa: PLC0415

    system = cobre.io.load_case(str(D48_CASE))
    assert isinstance(system, cobre.model.System)
    assert system.n_hydros == 2, "d48 declares the U -> J cascade (two hydros)"
    assert system.n_stages == 3, "d48 runs three weekly stages"


def test_validate_reports_no_errors_for_windowed_defluence() -> None:
    """validate accepts the non-zero windowed past_defluences shape (no errors).

    The window covers `[start_0 - t_v, start_0)`, so the io coverage gate passes;
    the case ships only benign advisories (e.g. deterministic-inflow notices).
    """
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(str(D48_CASE))
    assert isinstance(result, dict)
    assert result["valid"] is True, f"unexpected errors: {result['errors']}"
    assert result["errors"] == []


def test_study_constructs_on_windowed_defluence_seed(tmp_path: pathlib.Path) -> None:
    """Study(d48) runs the front-half lifecycle, unrolling the windowed seed."""
    import cobre  # noqa: PLC0415

    study = cobre.Study(str(D48_CASE), output_dir=str(tmp_path))
    system = study.system
    assert isinstance(system, cobre.model.System)
    assert system.n_stages == 3
