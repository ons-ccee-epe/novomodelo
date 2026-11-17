#!/usr/bin/env python3
"""Compare the training and simulation wall-clock time of two `novomodelo` binaries.

Runs `novomodelo run` on every `--case` with the baseline and the candidate
binary, `--repeats` times, alternating the two binaries within each repeat, and
reports per case the median `duration_seconds` that the run writes to
`training/metadata.json` and `simulation/metadata.json`, for each binary, and the
candidate/baseline ratio of the two medians. A ratio above 1 means the candidate
is slower. The simulation columns are empty for a case whose runs write no
simulation metadata.

Each case is linked into `<work-dir>/<case name>/case`, so the source case is
never written to. The output of repeat `i` of a binary goes to
`<work-dir>/<case name>/<baseline|candidate>/<i>`. Case directories must have
distinct names.

Example:

    python3 scripts/measure_run_wallclock.py \\
        --baseline-bin /var/tmp/baseline/novomodelo \\
        --candidate-bin target/release/novomodelo \\
        --case examples/1dtoy --repeats 5 --work-dir /var/tmp/wallclock-1dtoy \\
        --report wallclock-1dtoy.csv

The report and the work directory describe one machine and one build: do not
commit them or quote them in shipped documentation.

Exit codes: 0 report written, 1 a run failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

LABELS: tuple[str, str] = ("baseline", "candidate")
PHASES: tuple[str, str] = ("training", "simulation")
COLUMNS: tuple[str, ...] = (
    "case",
    "baseline_training_s",
    "candidate_training_s",
    "training_ratio",
    "baseline_simulation_s",
    "candidate_simulation_s",
    "simulation_ratio",
)
STDERR_TAIL_LINES = 20


class RunFailed(Exception):
    pass


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the training and simulation wall-clock time of two novomodelo binaries."
    )
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument(
        "--case",
        type=Path,
        action="append",
        required=True,
        help="source case directory; repeat the option for several cases",
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="directory for the case links and run outputs; must not exist or be empty",
    )
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument(
        "--report", type=Path, help="write the report as CSV to this file"
    )
    args = parser.parse_args(argv)

    for binary in (args.baseline_bin, args.candidate_bin):
        if not binary.is_file():
            parser.error(f"{binary} is not a file")
    for case in args.case:
        if not (case / "config.json").is_file():
            parser.error(f"{case} holds no config.json")
    names = [case.resolve().name for case in args.case]
    if len(set(names)) != len(names):
        parser.error("--case directories must have distinct names")
    if args.work_dir.exists() and (
        not args.work_dir.is_dir() or any(args.work_dir.iterdir())
    ):
        parser.error(f"{args.work_dir} exists and is not an empty directory")
    return args


def read_duration(path: Path) -> float:
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["duration_seconds"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RunFailed(f"cannot read duration_seconds from {path}: {error!r}") from error


def run_once(
    binary: Path, case_copy: Path, output_dir: Path, threads: int
) -> dict[str, float]:
    child = subprocess.run(
        [
            str(binary),
            "run",
            str(case_copy),
            "--output",
            str(output_dir),
            "--threads",
            str(threads),
            "--comm-backend",
            "local",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    try:
        if child.returncode != 0:
            raise RunFailed(f"exit status {child.returncode}")
        durations = {"training": read_duration(output_dir / "training" / "metadata.json")}
        simulation = output_dir / "simulation" / "metadata.json"
        if simulation.is_file():
            durations["simulation"] = read_duration(simulation)
    except RunFailed as failure:
        stderr_tail = "\n".join(child.stderr.splitlines()[-STDERR_TAIL_LINES:])
        raise RunFailed(f"{failure}\n{stderr_tail}") from failure
    return durations


def format_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def summarize(case: str, samples: dict[tuple[str, str], list[float]]) -> dict[str, str]:
    row = {"case": case}
    for phase in PHASES:
        baseline, candidate = (
            statistics.median(samples[label, phase]) if samples[label, phase] else None
            for label in LABELS
        )
        row[f"baseline_{phase}_s"] = format_seconds(baseline)
        row[f"candidate_{phase}_s"] = format_seconds(candidate)
        row[f"{phase}_ratio"] = format_seconds(
            None if baseline is None or candidate is None else candidate / baseline
        )
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
    args = parse_args(argv)
    binaries = dict(
        zip(LABELS, (p.resolve() for p in (args.baseline_bin, args.candidate_bin)))
    )
    work_dir = args.work_dir.resolve()

    rows: list[dict[str, str]] = []
    for case_arg in args.case:
        case = case_arg.resolve()
        case_root = work_dir / case.name
        case_copy = case_root / "case"
        case_copy.mkdir(parents=True)
        for entry in case.iterdir():
            (case_copy / entry.name).symlink_to(entry)

        samples: dict[tuple[str, str], list[float]] = {
            (label, phase): [] for label in LABELS for phase in PHASES
        }
        for repeat in range(args.repeats):
            for label, binary in binaries.items():
                sys.stderr.write(f"running {case_arg} {label} {repeat + 1}/{args.repeats}\n")
                try:
                    durations = run_once(
                        binary, case_copy, case_root / label / str(repeat), args.threads
                    )
                except RunFailed as failure:
                    sys.stderr.write(
                        f"error: case {case_arg}, {label} binary {binary}, "
                        f"repeat {repeat + 1} failed: {failure}\n"
                        f"the work directory {work_dir} is kept for inspection\n"
                    )
                    return 1
                for phase, seconds in durations.items():
                    samples[label, phase].append(seconds)
        rows.append(summarize(str(case_arg), samples))

    print_table(rows)
    if args.report is not None:
        with args.report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
