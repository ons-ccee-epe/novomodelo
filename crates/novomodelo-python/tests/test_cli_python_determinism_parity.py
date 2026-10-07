"""Golden CLI-versus-Python determinism test on examples/1dtoy.

This module verifies the byte-identity claim the bindings make in the docstrings
of `Study.train` and `Study.simulate` (`crates/cobre-python/src/study.rs`): both
state that they invoke the same writers as `run_via_study` in
`crates/cobre-python/src/run.rs`, and that the resulting outputs are identical. This test asserts
that claim by running the same case (`examples/1dtoy`) through both the compiled
CLI and the Python `cobre.run.run()` entry point, then comparing the two output
trees.

The case is deterministic (seed-pinned, all `in_sample` schemes, 128-iteration
limit, simulation enabled with 100 scenarios), so both runs produce the same
logical outputs. The comparison is value-level (JSON objects via the `json`
module, Parquet columns via `pyarrow`) rather than byte-level, because three
independent reasons make byte-level comparison guaranteed to fail:

1. `training/metadata.json` and `simulation/metadata.json` carry wall-clock
   timestamps (`started_at`, `completed_at`, `duration_seconds`).
2. The solver-stats Parquet files carry wall-clock millisecond columns
   (`solve_time_ms`, `load_model_time_ms`, `set_bounds_time_ms`,
   `basis_set_time_ms`).
3. The policy checkpoint embeds `created_at: cobre_io::now_iso8601()`.

Every other field and column is compared exactly. The two module-level mask
constants name the excluded paths and columns, with the reason each is masked:

- `_MASKED_JSON_POINTERS`: maps each JSON file to the tuple of slash-separated
  key paths dropped before comparison. Every path is wall-clock-related,
  including the `setup` section (per-phase setup timings) that both the CLI
  and Python now write to `training/metadata.json`.
- `_MASKED_PARQUET_COLUMNS`: the frozen set of wall-clock column names dropped
  from every Parquet comparison.

The policy checkpoint is content-exempt: its presence is asserted by the
file-set equality, but its bytes are not compared, because the manifest embeds
`created_at`.

Run with (from the repo root):

    pytest crates/cobre-python/tests/test_cli_python_determinism_parity.py -v \\
        --require-cli-binary

Or as part of the full suite:

    pytest crates/cobre-python/tests -q --require-cli-binary
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from typing import Any

import pyarrow.parquet as pq
import pytest

from _cobre_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
TOY_CASE = _REPO_ROOT / "examples" / "1dtoy"

_COMPARED_JSON_FILES = (
    "training/metadata.json",
    "simulation/metadata.json",
    "training/hydro_models.json",
    "training/model_provenance.json",
)

# Masked JSON pointer paths (slash-separated key paths) per file.
# Every entry is wall-clock-related.
_MASKED_JSON_POINTERS: dict[str, tuple[str, ...]] = {
    "training/metadata.json": (
        "started_at",  # wall-clock
        "completed_at",  # wall-clock
        "duration_seconds",  # wall-clock
        "setup",  # wall-clock (setup-phase timings)
        "solve_stats/forward_solve_seconds",  # cumulative wall-clock
        "solve_stats/backward_solve_seconds",  # cumulative wall-clock
    ),
    "simulation/metadata.json": (
        "started_at",  # wall-clock
        "completed_at",  # wall-clock
        "duration_seconds",  # wall-clock
        "solve_stats/solve_seconds",  # cumulative wall-clock
    ),
    "training/hydro_models.json": (),  # deterministic report struct
    "training/model_provenance.json": (),  # deterministic report struct
}

# Masked Parquet column names (wall-clock timing columns).
_MASKED_PARQUET_COLUMNS = frozenset(
    {
        "solve_time_ms",  # wall-clock (per-LP timing in solver/iterations.parquet)
        "load_model_time_ms",  # wall-clock (per-LP timing)
        "set_bounds_time_ms",  # wall-clock (per-LP timing)
        "basis_set_time_ms",  # wall-clock (per-LP timing)
        "time_forward_ms",  # wall-clock (per-iteration aggregate in convergence.parquet)
        "time_backward_ms",  # wall-clock (per-iteration aggregate)
        "time_total_ms",  # wall-clock (per-iteration aggregate)
        "forward_wall_ms",  # wall-clock (per-iteration phase timing in timing/iterations.parquet)
        "backward_wall_ms",  # wall-clock (per-iteration phase timing)
        "cut_selection_ms",  # wall-clock (per-iteration phase timing)
        "mpi_allreduce_ms",  # wall-clock (per-iteration MPI timing)
        "cut_sync_ms",  # wall-clock (per-iteration MPI timing)
        "lower_bound_ms",  # wall-clock (per-iteration phase timing)
        "state_exchange_ms",  # wall-clock (per-iteration MPI timing)
        "cut_batch_build_ms",  # wall-clock (per-iteration phase timing)
        "bwd_setup_ms",  # wall-clock (per-iteration phase timing)
        "bwd_load_imbalance_ms",  # wall-clock (per-iteration phase timing)
        "bwd_scheduling_overhead_ms",  # wall-clock (per-iteration phase timing)
        "fwd_setup_ms",  # wall-clock (per-iteration phase timing)
        "fwd_load_imbalance_ms",  # wall-clock (per-iteration phase timing)
        "fwd_scheduling_overhead_ms",  # wall-clock (per-iteration phase timing)
        "overhead_ms",  # wall-clock (per-iteration phase timing)
        "lazy_scoring_ms",  # wall-clock (per-iteration phase timing)
    }
)

# Content-exempt relative path prefixes: files are asserted present by the
# file-set equality but not compared byte-wise, because the policy checkpoint
# embeds `created_at`.
_CONTENT_EXEMPT_PREFIXES = ("policy/",)


def _relative_files(root: pathlib.Path) -> set[str]:
    """Return the set of file paths under `root`, POSIX-relative to `root`.

    A plain recursive directory walk (`Path.rglob`), not a Parquet-dataset
    read, so the Hive-partitioned `scenario_id=NNNN/` directories are captured
    as ordinary paths with no hive-vs-column ambiguity.
    """
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _drop_masked(obj: Any, paths: tuple[str, ...]) -> Any:
    """Drop each slash-separated path from `obj` (mutates and returns)."""
    for path in paths:
        parts = path.split("/")
        target = obj
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                break
            target = target[part]
        else:
            if isinstance(target, dict) and parts[-1] in target:
                del target[parts[-1]]
    return obj


def _json_mismatches(
    cli_obj: Any, py_obj: Any, label: str, path: str = ""
) -> list[str]:
    """Return human-readable difference lines by recursive comparison."""
    diffs: list[str] = []
    if type(cli_obj) != type(py_obj):  # noqa: E721
        diffs.append(
            f"{label} key '{path}': type mismatch (CLI={type(cli_obj).__name__}, "
            f"Python={type(py_obj).__name__})"
        )
    elif isinstance(cli_obj, dict):
        all_keys = set(cli_obj.keys()) | set(py_obj.keys())
        for key in sorted(all_keys):
            key_path = f"{path}/{key}" if path else key
            if key not in cli_obj:
                diffs.append(f"{label} key '{key_path}': missing in CLI")
            elif key not in py_obj:
                diffs.append(f"{label} key '{key_path}': missing in Python")
            else:
                diffs.extend(
                    _json_mismatches(cli_obj[key], py_obj[key], label, key_path)
                )
    elif isinstance(cli_obj, list):
        if len(cli_obj) != len(py_obj):
            diffs.append(
                f"{label} key '{path}': length mismatch "
                f"(CLI={len(cli_obj)}, Python={len(py_obj)})"
            )
        else:
            for i, (cli_item, py_item) in enumerate(zip(cli_obj, py_obj)):
                diffs.extend(_json_mismatches(cli_item, py_item, label, f"{path}[{i}]"))
    elif cli_obj != py_obj:
        diffs.append(
            f"{label} key '{path}': value mismatch (CLI={cli_obj!r}, Python={py_obj!r})"
        )
    return diffs


@pytest.fixture(scope="module")
def toy_cli_output(
    tmp_path_factory: pytest.TempPathFactory, cli_binary: pathlib.Path
) -> pathlib.Path:
    """Run 1dtoy (training + simulation enabled) through the compiled CLI binary."""
    assert TOY_CASE.is_dir(), f"the 1dtoy fixture must exist at {TOY_CASE}"
    output_dir = tmp_path_factory.mktemp("toy_cli_out")
    run_cli(TOY_CASE, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def toy_python_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Run 1dtoy once through `cobre.run.run()`; return the output dir and result."""
    import cobre.run  # noqa: PLC0415

    assert TOY_CASE.is_dir(), f"the 1dtoy fixture must exist at {TOY_CASE}"
    output_dir = tmp_path_factory.mktemp("toy_python_out")
    result = cobre.run.run(str(TOY_CASE), output_dir=str(output_dir))
    return output_dir, result


@pytest.fixture(scope="module")
def toy_python_output(
    toy_python_run: tuple[pathlib.Path, dict[str, Any]],
) -> pathlib.Path:
    """The output directory of the single Python run."""
    return toy_python_run[0]


@pytest.fixture(scope="module")
def toy_python_result(
    toy_python_run: tuple[pathlib.Path, dict[str, Any]],
) -> dict[str, Any]:
    """The result dict of the single Python run (the run that wrote `toy_python_output`)."""
    return toy_python_run[1]


def test_cli_python_output_file_sets_are_equal(
    toy_cli_output: pathlib.Path, toy_python_output: pathlib.Path
) -> None:
    """The CLI and Python write paths produce the exact same set of output files.

    Compares only the SET of relative file paths. Content comparison is handled
    by the JSON and Parquet tests below. A file present in one tree and absent
    from the other fails here with the exact missing/extra filename named.
    """
    cli_files = _relative_files(toy_cli_output)
    py_files = _relative_files(toy_python_output)

    only_cli = sorted(cli_files - py_files)
    only_py = sorted(py_files - cli_files)

    if only_cli or only_py:
        lines = ["CLI and Python output file sets diverge:"]
        if only_cli:
            lines.append("  present in CLI output but missing from Python:")
            lines.extend(f"    {name}" for name in only_cli)
        if only_py:
            lines.append("  present in Python output but missing from CLI:")
            lines.extend(f"    {name}" for name in only_py)
        pytest.fail("\n".join(lines), pytrace=False)


def test_cli_python_json_files_match(
    toy_cli_output: pathlib.Path, toy_python_output: pathlib.Path
) -> None:
    """All JSON files match after masking wall-clock and setup fields.

    The four JSON files (`training/metadata.json`, `simulation/metadata.json`,
    `training/hydro_models.json`, `training/model_provenance.json`) are loaded,
    the paths listed in `_MASKED_JSON_POINTERS` for that file are removed, and
    the two objects are compared. A mismatch reports the file, the key path and
    both values.
    """
    for rel_path in _COMPARED_JSON_FILES:
        cli_path = toy_cli_output / rel_path
        py_path = toy_python_output / rel_path

        assert cli_path.exists(), f"CLI must write {rel_path}"
        assert py_path.exists(), f"Python must write {rel_path}"

        cli_obj = json.loads(cli_path.read_text(encoding="utf-8"))
        py_obj = json.loads(py_path.read_text(encoding="utf-8"))

        _drop_masked(cli_obj, _MASKED_JSON_POINTERS[rel_path])
        _drop_masked(py_obj, _MASKED_JSON_POINTERS[rel_path])

        diffs = _json_mismatches(cli_obj, py_obj, rel_path)
        if diffs:
            pytest.fail(
                f"{rel_path}: CLI and Python JSON objects diverge:\n"
                + "\n".join(f"  {d}" for d in diffs),
                pytrace=False,
            )


def test_cli_python_parquet_files_match(
    toy_cli_output: pathlib.Path, toy_python_output: pathlib.Path
) -> None:
    """All Parquet files match in schema, row count and non-timing columns.

    Every relative path ending `.parquet` in the shared file set is read from
    both trees with `pyarrow.parquet.ParquetFile(path).read()` (single-file
    read, not dataset, to avoid Hive-partition vs in-file column collision).
    Schemas must be equal, row counts must be equal, and every column whose name
    is not in `_MASKED_PARQUET_COLUMNS` must compare equal. A mismatch names the
    file, the column and the first five values from each side.
    """
    cli_files = _relative_files(toy_cli_output)
    py_files = _relative_files(toy_python_output)
    shared_files = cli_files & py_files

    parquet_files = sorted(f for f in shared_files if f.endswith(".parquet"))

    for rel_path in parquet_files:
        cli_path = toy_cli_output / rel_path
        py_path = toy_python_output / rel_path

        cli_table = pq.ParquetFile(cli_path).read()
        py_table = pq.ParquetFile(py_path).read()

        if not cli_table.schema.equals(py_table.schema):
            pytest.fail(
                f"{rel_path}: schema mismatch\n"
                f"  CLI: {cli_table.schema}\n"
                f"  Python: {py_table.schema}",
                pytrace=False,
            )

        if cli_table.num_rows != py_table.num_rows:
            pytest.fail(
                f"{rel_path}: row count mismatch "
                f"(CLI={cli_table.num_rows}, Python={py_table.num_rows})",
                pytrace=False,
            )

        mismatches: list[str] = []
        for col_name in cli_table.schema.names:
            if col_name in _MASKED_PARQUET_COLUMNS:
                continue
            cli_col = cli_table.column(col_name)
            py_col = py_table.column(col_name)
            if not cli_col.equals(py_col):
                mismatches.append(
                    f"  column '{col_name}':\n"
                    f"    CLI first 5: {cli_col[:5].to_pylist()}\n"
                    f"    Python first 5: {py_col[:5].to_pylist()}"
                )

        if mismatches:
            pytest.fail(
                f"{rel_path}: non-timing column mismatch:\n" + "\n".join(mismatches),
                pytrace=False,
            )


def test_compared_surface_is_non_vacuous(toy_python_output: pathlib.Path) -> None:
    """The compared surface exercises the drift-prone outputs.

    Non-vacuity check for the tests above: all four JSON files are present, at
    least 10 Parquet files are compared (training solver stats, hydro models,
    simulation per-entity outputs, and per-scenario Hive partitions), at least
    two `scenario_id=NNNN/` Hive partitions appear, and every name in
    `_MASKED_PARQUET_COLUMNS` appears in at least one compared Parquet schema,
    so neither the comparison nor the mask can pass vacuously.
    """
    files = _relative_files(toy_python_output)

    for rel_path in _COMPARED_JSON_FILES:
        assert rel_path in files, f"expected JSON file {rel_path}"

    parquet_files = [f for f in files if f.endswith(".parquet")]
    assert len(parquet_files) >= 10, (
        f"expected at least 10 Parquet files; got {len(parquet_files)}"
    )

    scenario_partition_re = re.compile(r"scenario_id=(\d{4})")
    partition_ids = {
        match.group(1) for f in files if (match := scenario_partition_re.search(f))
    }
    assert len(partition_ids) >= 2, (
        f"expected multiple scenario_id=NNNN/ partitions; got {sorted(partition_ids)}"
    )

    found_masked_columns: set[str] = set()
    for rel_path in parquet_files:
        if any(rel_path.startswith(prefix) for prefix in _CONTENT_EXEMPT_PREFIXES):
            continue
        schema = pq.read_schema(toy_python_output / rel_path)
        found_masked_columns.update(schema.names)

    missing_from_mask = _MASKED_PARQUET_COLUMNS - found_masked_columns
    assert not missing_from_mask, (
        f"masked columns not found in any compared Parquet file: {missing_from_mask}"
    )


def test_negative_self_test_json_comparison_fails_on_perturbation(
    toy_python_output: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Perturbing one unmasked metadata value makes the JSON comparison fail.

    Proves the failure signal this test exists to produce: copies the Python
    output tree to a scratch directory (never mutating the shared fixture),
    perturbs one arbitrary unmasked value in `training/metadata.json` (the
    `iterations.completed` field, standing in for any unmasked field), and
    asserts the SAME comparison helper the real test uses now raises with that
    key path in the message.
    """
    scratch = tmp_path / "python_output_perturbed"
    shutil.copytree(toy_python_output, scratch)

    metadata_path = scratch / "training" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_iterations = metadata["iterations"]["completed"]
    metadata["iterations"]["completed"] = original_iterations + 1
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cli_metadata_path = toy_python_output / "training" / "metadata.json"
    cli_obj = json.loads(cli_metadata_path.read_text(encoding="utf-8"))
    py_obj = metadata

    _drop_masked(cli_obj, _MASKED_JSON_POINTERS["training/metadata.json"])
    _drop_masked(py_obj, _MASKED_JSON_POINTERS["training/metadata.json"])

    diffs = _json_mismatches(cli_obj, py_obj, "training/metadata.json")
    with pytest.raises(AssertionError, match=r"iterations/completed"):
        if diffs:
            raise AssertionError(
                "training/metadata.json: CLI and Python JSON objects diverge:\n"
                + "\n".join(f"  {d}" for d in diffs)
            )


def test_python_result_dict_matches_written_metadata(
    toy_python_result: dict[str, Any], toy_python_output: pathlib.Path
) -> None:
    """The Python result dict agrees with the training metadata the same call wrote.

    The five fields returned by `cobre.run.run()` (`converged`, `iterations`,
    `lower_bound`, `upper_bound`, `gap_percent`) are compared against their
    mapped paths in `training/metadata.json`. All must equal exactly, because
    both sides serialise the same in-memory `result`.
    """
    metadata_path = toy_python_output / "training" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert toy_python_result["converged"] == metadata["convergence"]["achieved"], (
        f"result['converged'] mismatch: result={toy_python_result['converged']}, "
        f"metadata={metadata['convergence']['achieved']}"
    )

    assert toy_python_result["iterations"] == metadata["iterations"]["completed"], (
        f"result['iterations'] mismatch: result={toy_python_result['iterations']}, "
        f"metadata={metadata['iterations']['completed']}"
    )

    assert (
        toy_python_result["lower_bound"] == metadata["bounds"]["final_lower_bound"]
    ), (
        f"result['lower_bound'] mismatch: result={toy_python_result['lower_bound']}, "
        f"metadata={metadata['bounds']['final_lower_bound']}"
    )

    assert (
        toy_python_result["upper_bound"] == metadata["bounds"]["final_upper_bound"]
    ), (
        f"result['upper_bound'] mismatch: result={toy_python_result['upper_bound']}, "
        f"metadata={metadata['bounds']['final_upper_bound']}"
    )

    expected_gap = metadata["convergence"]["final_gap_percent"]
    assert expected_gap is not None, "final_gap_percent must not be None for 1dtoy"
    assert toy_python_result["gap_percent"] == expected_gap, (
        f"result['gap_percent'] mismatch: result={toy_python_result['gap_percent']}, "
        f"metadata={expected_gap}"
    )
