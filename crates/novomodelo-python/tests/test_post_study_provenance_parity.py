"""Python-parity guard for the `post_study` section of `model_provenance.json`.

Both the CLI (`crates/novomodelo-cli/src/commands/run/setup.rs`) and the Python
bindings (`crates/novomodelo-python/src/run.rs`) hand `build_provenance_report`
the section built by `build_post_study_provenance`. The parameter is an
`Option`, so a call site that passes `None` still compiles; this module is what
pins the value. It runs one `tmp_path` copy of each post-study deck shape
through the compiled `novomodelo` CLI and `novomodelo.run.run`, and asserts the
`post_study` sections of the two `training/model_provenance.json` files are
equal, with one non-vacuity fact per shape.

Deck shapes
-----------
- `dual_calendar_boundary`: calendars declared in `stages.json`.
- `travel_time_arc`: calendars declared in the deprecated
  `post_study_stages.json`.
- `travel_time_arc` without `post_study_stages.json`: no post-study input and a
  travel-time arc, so the water calendar is padded.
- `d01-thermal-dispatch`: neither; the section is absent from both files.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_post_study_provenance_parity.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

from _novomodelo_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
_SDDP_TESTS = _REPO_ROOT / "crates" / "novomodelo-sddp" / "tests"

STAGES_JSON_CASE = _SDDP_TESTS / "unswept_fixtures" / "dual_calendar_boundary"
TRAVEL_TIME_ARC_CASE = _SDDP_TESTS / "fixtures" / "travel_time_arc"
NO_POST_STUDY_CASE = _REPO_ROOT / "examples" / "deterministic" / "d01-thermal-dispatch"


def _copy_trimmed(
    src: pathlib.Path, dest: pathlib.Path, *, drop_boundary: bool = False
) -> None:
    """Copy a deck to ``dest``, trimmed to one training iteration with no simulation.

    Provenance is written at setup, so the trim only shortens the run. With
    ``drop_boundary`` the copy loses ``policy.boundary``, whose source policy
    ``dual_calendar_boundary`` does not commit.
    """
    shutil.copytree(src, dest, dirs_exist_ok=True)

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 1}]
    config.setdefault("simulation", {})["enabled"] = False
    if drop_boundary:
        del config["policy"]["boundary"]
    (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _provenance(output_dir: pathlib.Path) -> dict[str, Any]:
    path = output_dir / "training" / "model_provenance.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_both(
    case_dir: pathlib.Path, tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``case_dir`` through the CLI and Python; return both provenance files parsed."""
    import novomodelo.run

    out_cli = tmp_path / "cli_out"
    run_cli(case_dir, out_cli, cli_binary)

    out_py = tmp_path / "py_out"
    novomodelo.run.run(str(case_dir), output_dir=str(out_py))

    return _provenance(out_cli), _provenance(out_py)


def _shared_section(
    cli: dict[str, Any], py: dict[str, Any], deck: str
) -> dict[str, Any]:
    """Return the `post_study` section after asserting both surfaces wrote an equal one."""
    assert "post_study" in cli, f"{deck}: CLI model_provenance.json has no post_study"
    assert "post_study" in py, f"{deck}: Python model_provenance.json has no post_study"
    assert cli["post_study"] == py["post_study"], (
        f"{deck}: CLI and Python post_study diverge (Python-parity violation):\n"
        f"CLI: {cli['post_study']}\nPY:  {py['post_study']}"
    )
    return cli["post_study"]


def test_post_study_section_matches_on_declared_stages_json(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """Calendars declared in `stages.json` reach both surfaces unchanged."""
    case_dir = tmp_path / "case"
    _copy_trimmed(STAGES_JSON_CASE, case_dir, drop_boundary=True)

    cli, py = _run_both(case_dir, tmp_path, cli_binary)

    section = _shared_section(cli, py, "dual_calendar_boundary")
    assert section["water"]["origin"] == "declared", (
        f"dual_calendar_boundary: water origin is {section['water']['origin']!r}"
    )


def test_post_study_section_matches_on_deprecated_post_study_file(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """Calendars declared in the deprecated `post_study_stages.json` reach both surfaces."""
    case_dir = tmp_path / "case"
    _copy_trimmed(TRAVEL_TIME_ARC_CASE, case_dir)

    cli, py = _run_both(case_dir, tmp_path, cli_binary)

    section = _shared_section(cli, py, "travel_time_arc")
    assert section["water"]["origin"] == "declared", (
        f"travel_time_arc: water origin is {section['water']['origin']!r}"
    )


def test_post_study_section_matches_on_padded_water_calendar(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """A travel-time arc with no post-study input reports the padding on both surfaces."""
    case_dir = tmp_path / "case"
    _copy_trimmed(TRAVEL_TIME_ARC_CASE, case_dir)
    (case_dir / "post_study_stages.json").unlink()

    cli, py = _run_both(case_dir, tmp_path, cli_binary)

    section = _shared_section(cli, py, "travel_time_arc without post_study_stages.json")
    assert section["water"]["padding"]["padded_stage_count"] == 2, (
        f"padded travel_time_arc: padding is {section['water']['padding']!r}"
    )


def test_post_study_section_absent_without_post_study_input_or_arc(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    """A deck with no post-study input and no arc writes no `post_study` key."""
    case_dir = tmp_path / "case"
    _copy_trimmed(NO_POST_STUDY_CASE, case_dir)

    cli, py = _run_both(case_dir, tmp_path, cli_binary)

    assert "post_study" not in cli, "d01-thermal-dispatch: CLI wrote a post_study key"
    assert "post_study" not in py, "d01-thermal-dispatch: Python wrote a post_study key"
