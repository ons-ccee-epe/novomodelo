#!/usr/bin/env python3
"""Measure `level1` state windows and simulated cut sets on a case.

Runs five variants of one case, one after the other, each in its own case copy
under `--work-dir`:

  all-states        `level1`, every trial state kept, then a simulation of the
                    active cuts
  window-<W>        the same with `lookback_iterations = W`, one per `--windows`
  simulate-all      simulation only, over the `all-states` policy, with
                    `simulation.cut_set = "all"`
  simulate-active   the same with `simulation.cut_set = "active"`

and reports, per variant, the final lower bound (training variants), the mean
and standard deviation of the simulated cost, the wall time and the peak
resident set of the largest process the run started (the largest rank under a
launcher). Every variant sets `simulation.cut_set` explicitly.

Each case copy links the entries of the source case and holds its own
`config.json`: the source config with the variant's settings merged in (dicts
merge recursively, lists and scalars are replaced). The source case is never
written to.

Example:

    python3 scripts/measure_cut_selection.py --case examples/1dtoy \\
        --work-dir /var/tmp/selection-1dtoy --check-frequency 5 --windows 10,20 \\
        --report selection-1dtoy.csv

    # four ranks, two threads each (the launcher is a prefix of the novomodelo command)
    python3 scripts/measure_cut_selection.py ... --threads 2 \\
        --launcher "mpiexec -n 4"

The report and the work directory describe one machine and one build: do not
commit them or quote them in shipped documentation.

Exit codes: 0 report written, 1 a variant failed, 2 bad arguments or platform.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
COLUMNS = (
    "variant",
    "lower_bound",
    "mean_simulated_cost",
    "std_simulated_cost",
    "wall_seconds",
    "peak_rss_mib",
)
STDERR_TAIL_LINES = 20


class VariantFailed(Exception):
    pass


@dataclass(frozen=True)
class Variant:
    name: str
    overlay: dict[str, Any]
    trains: bool


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge(base[key], value)
        else:
            merged[key] = value
    return merged


def exec_one(command: list[str]) -> int:
    """Hidden child mode: run `command`, print its peak resident set in KiB."""
    import resource

    returncode = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=sys.stderr
    ).returncode
    peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    sys.stdout.write(f"{peak_kib}\n")
    return returncode


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure level1 state windows and simulated cut sets on a case."
    )
    parser.add_argument(
        "--case", type=Path, required=True, help="source case directory"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="directory for the variants; must not exist or be empty",
    )
    parser.add_argument("--check-frequency", type=positive_int, required=True)
    parser.add_argument(
        "--windows",
        default="",
        help="comma-separated lookback_iterations values, each >= --check-frequency",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        help="replace the training stopping rules with this limit",
    )
    parser.add_argument(
        "--scenarios",
        type=positive_int,
        help="replace the simulation selection with this many sampled scenarios",
    )
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument(
        "--launcher", default="", help='command prefix, for example "mpiexec -n 4"'
    )
    parser.add_argument(
        "--novomodelo", type=Path, default=REPO_ROOT / "target" / "release" / "novomodelo"
    )
    parser.add_argument(
        "--report", type=Path, help="write the report as CSV to this file"
    )
    args = parser.parse_args(argv)

    try:
        args.windows = [int(w) for w in args.windows.split(",") if w]
    except ValueError:
        parser.error(
            f"--windows must be a comma-separated list of integers: {args.windows!r}"
        )
    for window in args.windows:
        if window < args.check_frequency:
            parser.error(
                f"window {window} is below --check-frequency {args.check_frequency}"
            )
    if len(set(args.windows)) != len(args.windows):
        parser.error("--windows repeats a value")
    if not (args.case / "config.json").is_file():
        parser.error(f"{args.case} holds no config.json")
    if not args.novomodelo.is_file():
        parser.error(
            f"{args.novomodelo} is not a file; build it with `cargo build --release --bin novomodelo`"
        )
    if args.work_dir.exists() and (
        not args.work_dir.is_dir() or any(args.work_dir.iterdir())
    ):
        parser.error(f"{args.work_dir} exists and is not an empty directory")
    try:
        args.launcher = shlex.split(args.launcher)
    except ValueError as error:
        parser.error(f"--launcher: {error}")
    return args


def build_variants(args: argparse.Namespace, work_dir: Path) -> list[Variant]:
    def trained(name: str, window: int | None) -> Variant:
        selection = {
            "method": "level1",
            "check_frequency": args.check_frequency,
            "lookback_iterations": window,
        }
        return Variant(
            name,
            {
                "training": {"cut_selection": {"selection": selection}},
                "simulation": {"enabled": True, "cut_set": "active"},
            },
            trains=True,
        )

    def simulated(cut_set: str) -> Variant:
        policy_dir = work_dir / "all-states" / "output" / "policy"
        return Variant(
            f"simulate-{cut_set}",
            {
                "training": {"enabled": False},
                "policy": {"path": str(policy_dir)},
                "simulation": {"enabled": True, "cut_set": cut_set},
            },
            trains=False,
        )

    return [
        trained("all-states", None),
        *(trained(f"window-{window}", window) for window in args.windows),
        simulated("all"),
        simulated("active"),
    ]


def read_quantity(path: Path, *keys: str) -> float:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VariantFailed(f"cannot read {path}: {error}") from error
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VariantFailed(f"{path} holds no number at {'.'.join(keys)}")
    return float(value)


def run_variant(
    args: argparse.Namespace,
    source_config: dict[str, Any],
    work_dir: Path,
    variant: Variant,
) -> dict[str, str]:
    root = work_dir / variant.name
    case_dir, output_dir = root / "case", root / "output"
    case_dir.mkdir(parents=True)
    for entry in args.case.iterdir():
        if entry.name != "config.json":
            (case_dir / entry.name).symlink_to(entry)
    config = merge(source_config, variant.overlay)
    (case_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    command = [
        *args.launcher,
        str(args.novomodelo),
        "run",
        "--output",
        str(output_dir),
        "--quiet",
        "--threads",
        str(args.threads),
        "--color",
        "never",
        str(case_dir),
    ]
    started = time.monotonic()
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--exec-one", *command],
        capture_output=True,
        text=True,
    )
    wall_seconds = time.monotonic() - started
    stderr_tail = "\n".join(child.stderr.splitlines()[-STDERR_TAIL_LINES:])
    try:
        if child.returncode != 0:
            raise VariantFailed(f"exit status {child.returncode}")
        row = {"variant": variant.name, "lower_bound": ""}
        if variant.trains:
            training = output_dir / "training" / "metadata.json"
            row["lower_bound"] = repr(
                read_quantity(training, "bounds", "final_lower_bound")
            )
        simulation = output_dir / "simulation" / "metadata.json"
        row["mean_simulated_cost"] = repr(
            read_quantity(simulation, "cost", "mean_cost")
        )
        row["std_simulated_cost"] = repr(read_quantity(simulation, "cost", "std_cost"))
    except VariantFailed as failure:
        raise VariantFailed(f"{failure}\n{stderr_tail}") from failure
    row["wall_seconds"] = f"{wall_seconds:.3f}"
    row["peak_rss_mib"] = f"{int(child.stdout) / 1024:.1f}"
    return row


def print_table(rows: list[dict[str, str]]) -> None:
    widths = [
        max(len(column), *(len(row[column]) for row in rows)) for column in COLUMNS
    ]
    for cells in ([*COLUMNS], *([row[c] for c in COLUMNS] for row in rows)):
        sys.stdout.write(
            "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
            + "\n"
        )


def main(argv: list[str]) -> int:
    if sys.platform != "linux":
        sys.stderr.write(
            "error: peak memory is read from ru_maxrss in KiB, which only Linux reports\n"
        )
        return 2
    if argv[:1] == ["--exec-one"]:
        return exec_one(argv[1:])

    args = parse_args(argv)
    args.case, args.work_dir, args.novomodelo = (
        p.resolve() for p in (args.case, args.work_dir, args.novomodelo)
    )
    source_config = json.loads((args.case / "config.json").read_text(encoding="utf-8"))
    shared: dict[str, Any] = {}
    if args.iterations is not None:
        shared["training"] = {
            "stopping_rules": [{"type": "iteration_limit", "limit": args.iterations}]
        }
    if args.scenarios is not None:
        shared["simulation"] = {
            "selection": {"method": "sampled", "num_scenarios": args.scenarios}
        }
    source_config = merge(source_config, shared)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for variant in build_variants(args, args.work_dir):
        sys.stderr.write(f"running {variant.name}\n")
        try:
            rows.append(run_variant(args, source_config, args.work_dir, variant))
        except VariantFailed as failure:
            sys.stderr.write(f"error: variant {variant.name} failed: {failure}\n")
            sys.stderr.write(
                f"the work directory {args.work_dir} is kept for inspection\n"
            )
            return 1

    print_table(rows)
    if args.report is not None:
        with args.report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
