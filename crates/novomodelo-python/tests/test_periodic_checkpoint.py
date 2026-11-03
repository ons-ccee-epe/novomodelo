"""Periodic training checkpoints written by the CLI and the Python bindings.

A run killed during training must leave the checkpoint of its last scheduled
write, loadable by `novomodelo.results.load_policy`, and both front ends must write
the same files when periodic checkpointing is on.

Run with (from the repo root, after `cargo build --release -p novomodelo-cli`):
    pytest crates/novomodelo-python/tests/test_periodic_checkpoint.py --require-cli-binary
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import pytest

from _novomodelo_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
_TOY_CASE = _REPO_ROOT / "examples" / "1dtoy"

_UNREACHABLE_ITERATION_LIMIT = 100_000
_CHECKPOINT_WAIT_SECONDS = 120.0

_PYTHON_RUN = "import sys, novomodelo.run; novomodelo.run.run(sys.argv[1], output_dir=sys.argv[2])"


def _periodic_case(root: pathlib.Path, iteration_limit: int) -> pathlib.Path:
    """Copy 1dtoy under `root` with a checkpoint every iteration and no simulation."""
    case = root / "case"
    shutil.copytree(_TOY_CASE, case, ignore=shutil.ignore_patterns("output"))
    config_path = case / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["stopping_rules"] = [
        {"type": "iteration_limit", "limit": iteration_limit}
    ]
    config["simulation"]["enabled"] = False
    config["policy"] = {"checkpointing": {"enabled": True, "interval_iterations": 1}}
    config_path.write_text(json.dumps(config, indent=2))
    return case


def _kill_after_first_checkpoint(
    command: list[str], output_dir: pathlib.Path, stderr_path: pathlib.Path
) -> None:
    """Start `command`, wait for `<output_dir>/policy/manifest.bin`, then SIGKILL it."""
    manifest = output_dir / "policy" / "manifest.bin"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            deadline = time.monotonic() + _CHECKPOINT_WAIT_SECONDS
            while not manifest.is_file():
                if process.poll() is not None:
                    pytest.fail(
                        f"the run exited with {process.returncode} before writing "
                        f"{manifest}:\n{stderr_path.read_text(encoding='utf-8')}"
                    )
                if time.monotonic() > deadline:
                    pytest.fail(
                        f"no checkpoint at {manifest} within "
                        f"{_CHECKPOINT_WAIT_SECONDS:.0f} s:\n{stderr_path.read_text(encoding='utf-8')}"
                    )
                time.sleep(0.05)
        finally:
            process.kill()
            process.wait(timeout=30)


def _assert_loadable_periodic_checkpoint(output_dir: pathlib.Path) -> None:
    import novomodelo.results

    producer = novomodelo.results.load_policy(output_dir)["metadata"]["producer"]
    assert 1 <= producer["completed_iterations"] < _UNREACHABLE_ITERATION_LIMIT, (
        producer
    )
    assert producer["max_iterations"] == _UNREACHABLE_ITERATION_LIMIT, producer


def _relative_files(root: pathlib.Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_killed_python_run_leaves_a_loadable_periodic_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    case = _periodic_case(tmp_path, _UNREACHABLE_ITERATION_LIMIT)
    output_dir = tmp_path / "out"
    _kill_after_first_checkpoint(
        [sys.executable, "-c", _PYTHON_RUN, str(case), str(output_dir)],
        output_dir,
        tmp_path / "stderr.txt",
    )
    _assert_loadable_periodic_checkpoint(output_dir)


def test_killed_cli_run_leaves_a_loadable_periodic_checkpoint(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    case = _periodic_case(tmp_path, _UNREACHABLE_ITERATION_LIMIT)
    output_dir = tmp_path / "out"
    _kill_after_first_checkpoint(
        [str(cli_binary), "run", str(case), "--output", str(output_dir), "--quiet"],
        output_dir,
        tmp_path / "stderr.txt",
    )
    _assert_loadable_periodic_checkpoint(output_dir)


def test_periodic_checkpoint_runs_write_the_same_file_set_on_cli_and_python(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    import novomodelo.run

    case = _periodic_case(tmp_path, iteration_limit=4)
    cli_output = tmp_path / "cli_out"
    python_output = tmp_path / "python_out"
    run_cli(case, cli_output, cli_binary)
    novomodelo.run.run(str(case), output_dir=str(python_output))

    cli_files = _relative_files(cli_output)
    python_files = _relative_files(python_output)
    assert cli_files == python_files, (
        f"only CLI: {sorted(cli_files - python_files)}; "
        f"only Python: {sorted(python_files - cli_files)}"
    )
    for output_dir in (cli_output, python_output):
        for sibling in ("policy.staging", "policy.previous"):
            assert not (output_dir / sibling).exists(), output_dir / sibling


_INFEASIBLE_CASE = _REPO_ROOT / "examples" / "deterministic" / "d13-generic-constraint"


def _rewrite_json(path: pathlib.Path, edit) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _file_bytes(root: pathlib.Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def _set_iteration_limit(case: pathlib.Path, limit: int, mode: str | None) -> None:
    def edit(config: dict) -> None:
        config["training"]["stopping_rules"] = [
            {"type": "iteration_limit", "limit": limit}
        ]
        policy: dict[str, Any] = {
            "checkpointing": {"enabled": True, "interval_iterations": 1}
        }
        if mode is not None:
            policy["mode"] = mode
        config["policy"] = policy

    _rewrite_json(case / "config.json", edit)


def _make_training_lp_infeasible(case: pathlib.Path) -> None:
    def disable_slack(constraints: dict) -> None:
        constraints["constraints"][0]["slack"] = {"enabled": False}

    def raise_thermal_minimum(thermals: dict) -> None:
        thermals["thermals"][0]["generation"]["min_mw"] = 20.0

    _rewrite_json(case / "constraints" / "generic_constraints.json", disable_slack)
    _rewrite_json(case / "system" / "thermals.json", raise_thermal_minimum)


def test_a_failed_run_keeps_the_last_whole_checkpoint(tmp_path: pathlib.Path) -> None:
    import novomodelo.errors
    import novomodelo.results
    import novomodelo.run

    case = tmp_path / "case"
    healthy = tmp_path / "healthy"
    shutil.copytree(_INFEASIBLE_CASE, case, ignore=shutil.ignore_patterns("output"))
    shutil.copytree(case, healthy)
    output_dir = tmp_path / "out"
    policy_dir = output_dir / "policy"

    _set_iteration_limit(case, 3, None)
    novomodelo.run.run(str(case), output_dir=str(output_dir))
    whole = _file_bytes(policy_dir)
    assert "manifest.bin" in whole

    _set_iteration_limit(case, 6, "resume")
    _make_training_lp_infeasible(case)
    with pytest.raises(novomodelo.errors.SolverError) as failure:
        novomodelo.run.run(str(case), output_dir=str(output_dir))

    assert "the partial policy is in" in str(failure.value)
    assert "policy.partial" in str(failure.value)
    assert _file_bytes(policy_dir) == whole
    assert (output_dir / "policy.partial" / "manifest.bin").is_file()
    assert (
        novomodelo.results.load_policy(output_dir)["metadata"]["producer"][
            "completed_iterations"
        ]
        == 3
    )

    shutil.copy(healthy / "system" / "thermals.json", case / "system" / "thermals.json")
    shutil.copy(
        healthy / "constraints" / "generic_constraints.json",
        case / "constraints" / "generic_constraints.json",
    )
    novomodelo.run.run(str(case), output_dir=str(output_dir))
    assert (
        novomodelo.results.load_policy(output_dir)["metadata"]["producer"][
            "completed_iterations"
        ]
        == 6
    )
