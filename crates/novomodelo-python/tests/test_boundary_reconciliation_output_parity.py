"""Python-parity guard for `training/boundary_reconciliation.json`.

Both the CLI (`crates/novomodelo-cli/src/commands/run/policy.rs`) and the Python
bindings (`crates/novomodelo-python/src/run.rs`) write the file when a deck's
`policy.boundary` loads, and neither writes it otherwise. This module runs one
`tmp_path` self-boundary deck through the compiled `novomodelo` CLI and
`novomodelo.run.run` and asserts the two files are byte-identical with a
non-empty transit bucket mapping, then runs the same deck without its boundary
and asserts neither surface writes the file.

`build_self_boundary_case` is shared with the file-set gate in
`test_cli_python_file_set_parity.py`.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_boundary_reconciliation_output_parity.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil

from test_post_study_provenance_parity import STAGES_JSON_CASE, _copy_trimmed

from _novomodelo_cli import run_cli

_RECONCILIATION = pathlib.Path("training") / "boundary_reconciliation.json"


def build_self_boundary_case(tmp: pathlib.Path) -> pathlib.Path:
    """Return a trimmed `dual_calendar_boundary` copy whose boundary is its own policy.

    The source run drops `policy.boundary`, which would otherwise look for the
    uncommitted `source_policy`; its `policy/` becomes `source_policy` beside
    the returned case's `config.json`.
    """
    import novomodelo.run

    source_case = tmp / "source_case"
    _copy_trimmed(STAGES_JSON_CASE, source_case, drop_boundary=True)
    source_output = tmp / "source_output"
    novomodelo.run.run(str(source_case), output_dir=str(source_output))

    target_case = tmp / "target_case"
    _copy_trimmed(STAGES_JSON_CASE, target_case)
    shutil.copytree(source_output / "policy", target_case / "source_policy")
    return target_case


def _run_both(
    case_dir: pathlib.Path, tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """Run ``case_dir`` through the CLI and Python; return both output directories."""
    import novomodelo.run

    out_cli = tmp_path / "cli_out"
    run_cli(case_dir, out_cli, cli_binary)

    out_py = tmp_path / "py_out"
    novomodelo.run.run(str(case_dir), output_dir=str(out_py))

    return out_cli, out_py


def test_boundary_reconciliation_file_is_identical_on_both_paths(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """A self-boundary deck gets the same non-empty reconciliation file from both surfaces."""
    case_dir = build_self_boundary_case(tmp_path)

    out_cli, out_py = _run_both(case_dir, tmp_path, cli_binary)

    for out in (out_cli, out_py):
        assert (out / _RECONCILIATION).is_file(), f"{out} is missing {_RECONCILIATION}"
    cli_bytes = (out_cli / _RECONCILIATION).read_bytes()
    py_bytes = (out_py / _RECONCILIATION).read_bytes()
    cli_doc = json.loads(cli_bytes.decode("utf-8"))
    py_doc = json.loads(py_bytes.decode("utf-8"))
    assert cli_bytes == py_bytes, (
        f"{_RECONCILIATION} differs between CLI ({out_cli}) and Python ({out_py}) "
        f"(Python-parity violation):\nCLI: {cli_doc}\nPY:  {py_doc}"
    )
    assert cli_doc["boundary_date"] == "2024-07-01", (
        f"{out_cli / _RECONCILIATION}: boundary_date is {cli_doc['boundary_date']!r}"
    )
    assert cli_doc["report"]["transit_bucket_mapping"], (
        f"{out_cli / _RECONCILIATION}: transit_bucket_mapping is empty: {cli_doc}"
    )


def test_boundary_reconciliation_file_is_absent_on_both_paths_without_a_boundary(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """A deck with no `policy.boundary` writes no reconciliation file on either surface."""
    case_dir = tmp_path / "case"
    _copy_trimmed(STAGES_JSON_CASE, case_dir, drop_boundary=True)

    out_cli, out_py = _run_both(case_dir, tmp_path, cli_binary)

    for out in (out_cli, out_py):
        assert not (out / _RECONCILIATION).exists(), (
            f"{out} wrote {_RECONCILIATION} for a deck with no boundary"
        )
