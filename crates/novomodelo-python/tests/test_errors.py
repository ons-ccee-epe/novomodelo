"""Tests for error propagation across the FFI boundary.

These tests verify that Rust errors are correctly translated into appropriate
Python exceptions with meaningful error messages.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_errors.py
"""

import pathlib
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


MISSING_CASE = "/tmp/nonexistent_novomodelo_case_xzy123"


def test_errors_importable_and_subclass_builtins() -> None:
    """The eight classes import from novomodelo.errors and subclass the right builtins."""
    import novomodelo.errors as e  # noqa: PLC0415

    assert e.NovomodeloError is not None
    assert e.ValidationError is not None
    assert e.CaseIoError is not None
    assert e.PolicyIncompatibleError is not None
    assert e.SolverError is not None
    assert e.SimulationError is not None
    assert e.OutputError is not None
    assert e.InternalError is not None

    # NovomodeloError is the common base.
    assert issubclass(e.NovomodeloError, Exception)

    # Dual-base subclassing: the matching builtin AND NovomodeloError.
    assert issubclass(e.ValidationError, ValueError)
    assert issubclass(e.ValidationError, e.NovomodeloError)
    assert issubclass(e.PolicyIncompatibleError, ValueError)
    assert issubclass(e.PolicyIncompatibleError, e.NovomodeloError)
    assert issubclass(e.CaseIoError, OSError)
    assert issubclass(e.CaseIoError, e.NovomodeloError)
    assert issubclass(e.OutputError, OSError)
    assert issubclass(e.OutputError, e.NovomodeloError)
    assert issubclass(e.SolverError, RuntimeError)
    assert issubclass(e.SolverError, e.NovomodeloError)
    assert issubclass(e.SimulationError, RuntimeError)
    assert issubclass(e.SimulationError, e.NovomodeloError)
    assert issubclass(e.InternalError, RuntimeError)
    assert issubclass(e.InternalError, e.NovomodeloError)
    # InternalError is a sibling, not a subclass of SolverError.
    assert not issubclass(e.InternalError, e.SolverError)

    # Qualified names read novomodelo.errors.<Name> (so tracebacks are unambiguous).
    assert e.SolverError.__module__ == "novomodelo.errors"
    assert e.SolverError.__qualname__ == "SolverError"
    assert e.InternalError.__module__ == "novomodelo.errors"


def test_run_nonexistent_dir_raises_oserror(tmp_path: pathlib.Path) -> None:
    """run() raises OSError with a descriptive message for a non-existent directory."""
    import novomodelo.run  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        novomodelo.run.run(MISSING_CASE, output_dir=str(tmp_path))


def test_run_empty_dir_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """run() raises ValidationError for a directory missing required case files."""
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(novomodelo.errors.ValidationError, match="constraint violation"):
        novomodelo.run.run(str(empty_case), output_dir=str(output))

    # The same failure is catchable as the builtin ValueError (dual base intact).
    with pytest.raises(ValueError):
        novomodelo.run.run(str(empty_case), output_dir=str(output))


def test_run_empty_dir_error_mentions_missing_files(tmp_path: pathlib.Path) -> None:
    """The ValidationError for an empty case lists specific missing files."""
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(novomodelo.errors.ValidationError) as exc_info:
        novomodelo.run.run(str(empty_case), output_dir=str(output))

    msg = str(exc_info.value)
    assert "config.json" in msg, "error must mention config.json"
    assert "FileNotFound" in msg, "error must include FileNotFound kind"


def test_study_empty_dir_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """Study() construction raises ValidationError for a directory missing required case files.

    Proves that the constructor and run() now agree on the exception class.
    """
    import novomodelo.errors  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(novomodelo.errors.ValidationError, match="constraint violation"):
        novomodelo.Study(str(empty_case), output_dir=str(output))


def test_load_case_nonexistent_raises_oserror() -> None:
    """load_case raises OSError for a non-existent path."""
    import novomodelo.io  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        novomodelo.io.load_case(MISSING_CASE)


def test_load_results_empty_dir_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    """load_results raises FileNotFoundError for a directory without _SUCCESS."""
    import novomodelo.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        novomodelo.results.load_results(str(tmp_path))


def test_validation_failure_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """A schema/constraint load failure raises ValidationError, also a ValueError.

    An empty case directory fails the structural-validation layer with a
    constraint violation, which maps to ``novomodelo.errors.ValidationError``.
    """
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.io  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()

    with pytest.raises(novomodelo.errors.ValidationError, match="constraint violation"):
        novomodelo.io.load_case(str(empty_case))

    # The same failure is catchable as the builtin ValueError (dual base intact)
    # and as the common NovomodeloError base.
    with pytest.raises(ValueError):
        novomodelo.io.load_case(str(empty_case))
    with pytest.raises(novomodelo.errors.NovomodeloError):
        novomodelo.io.load_case(str(empty_case))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="forces the write failure via os.chmod(dir, 0o500), which does not make "
    "a directory read-only on Windows, so the output-write error path cannot be "
    "exercised here; novomodelo.run's CaseIoError behavior itself is platform-independent",
)
def test_io_failure_raises_caseio_error(tmp_path: pathlib.Path) -> None:
    """An output-write I/O failure raises CaseIoError, also catchable as OSError.

    Running 1dtoy against a read-only output directory makes the first sidecar
    write fail with an "output write error", which maps to
    ``novomodelo.errors.CaseIoError``.
    """
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    case_dir = pathlib.Path(__file__).resolve().parents[3] / "examples" / "1dtoy"
    if not case_dir.exists():
        pytest.skip(f"examples/1dtoy not found at {case_dir}")

    out = tmp_path / "ro_output"
    out.mkdir()
    import os  # noqa: PLC0415

    os.chmod(out, 0o500)  # read + execute, no write
    try:
        with pytest.raises(novomodelo.errors.CaseIoError, match="output write error"):
            novomodelo.run.run(str(case_dir), output_dir=str(out))

        # The same failure is catchable as the builtin OSError (dual base intact).
        with pytest.raises(OSError):
            novomodelo.run.run(str(case_dir), output_dir=str(out))
    finally:
        os.chmod(out, 0o700)


def test_setup_validation_failure_raises_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    """A gap stopping rule rejected under sampled forward selection is a setup-time
    validation failure and raises ValidationError, also a ValueError.
    """
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    case_dir = pathlib.Path(__file__).resolve().parents[3] / "examples" / "1dtoy"
    if not case_dir.exists():
        pytest.skip(f"examples/1dtoy not found at {case_dir}")

    overrides = {
        "training.stopping_rules": [
            {"type": "iteration_limit", "limit": 1},
            {"type": "gap", "tolerance": 1000.0},
        ]
    }

    with pytest.raises(
        novomodelo.errors.ValidationError, match="gap stopping rule is inadmissible"
    ):
        novomodelo.run.run(
            str(case_dir), output_dir=str(tmp_path), config_overrides=overrides
        )

    # The same failure is catchable as the builtin ValueError (dual base intact).
    with pytest.raises(ValueError):
        novomodelo.run.run(
            str(case_dir), output_dir=str(tmp_path), config_overrides=overrides
        )


def test_preprocessing_validation_failure_raises_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    """Every FPHA hyperplane of a stage with gamma_q = 0 is a preprocessing-phase
    validation failure and raises ValidationError, not SolverError.
    """
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "examples"
        / "deterministic"
        / "d06-fpha-variable-head"
    )
    if not src.exists():
        pytest.skip(f"{src} not found")

    case_dir = tmp_path / "case"
    shutil.copytree(src, case_dir)

    hyperplanes_path = case_dir / "system" / "fpha_hyperplanes.parquet"
    table = pq.read_table(hyperplanes_path)
    field_index = table.schema.get_field_index("gamma_q")
    table = table.set_column(
        field_index,
        "gamma_q",
        pa.array([0.0] * table.num_rows, type=table.schema.field(field_index).type),
    )
    pq.write_table(table, hyperplanes_path, compression="zstd")

    with pytest.raises(
        novomodelo.errors.ValidationError, match="no hyperplane has gamma_q > 0"
    ):
        novomodelo.run.run(str(case_dir), output_dir=str(tmp_path / "out"))
