#!/usr/bin/env bash
# SLURM MPI integration test for `novomodelo run`.
#
# Exercises up to 5 MPI execution modes against the 4ree example case and
# verifies rank-count invariance: every classified output parquet of each
# test run is compared against the T1 single-rank baseline, decoded-value
# bit-exact (wall-clock timing columns and per-rank/worker execution-topology
# tables are exempt by design — see the CLASSIFICATION_ROWS table in
# compare_outputs_bitwise below).
#
# T1: mpiexec -n 1         (single rank, baseline)
# T2: mpiexec -n 2         (multi-rank, single machine)
# T3: sbatch -N 1 -n 2     (SLURM single-node)
# T4: sbatch -N 2 -n 2     (SLURM multi-node — catches MPICH 4.3.0 deadlock)
# T5: sbatch -N 2 -n 4     (SLURM multi-node, multiple ranks per node)
#
# The L cases train a copy of the case whose config.json adds cut selection
# (level1 every second iteration, at most 100 active cuts per stage). Each L
# run is compared against L1 as above, and L1 must have deactivated,
# reactivated and evicted cuts, so the comparison covers a selection that ran.
#
# L1: mpiexec -n 1               (selection-on baseline)
# L2: mpiexec -n 2 --threads 2
# L3: sbatch -N 2 -n 4           (mpiexec -n 4 --threads 1)
#
# The R cases resume the selection run at another rank count. A partial run
# trains a copy of the selection case stopped at iteration RESUME_AT and is not
# compared; a resume run continues its final checkpoint, written at one rank
# count, at another rank count up to L1's iteration limit, and is compared
# against L1 by compare_resumed_outputs and by its per-scenario solver
# counters.
#
# R0: mpiexec -n 1               (partial run)
# R1: mpiexec -n 2 --threads 2   (resumes R0's checkpoint)
# R2: sbatch -N 2 -n 2           (mpiexec -n 2; partial run)
# R3: mpiexec -n 1 --threads 1   (resumes R2's checkpoint)
#
# The S cases run simulation only, from T1's policy, on a copy of the case
# whose stages 4 to 7 have their single block split in two. The stored bases
# of those stages no longer fit their LP and are not used; the others are.
# Each S run is compared against S1 as above, and its per-scenario solver
# counters are compared too, so the same stages run warm and cold in every
# layout.
#
# S1: mpiexec -n 1 --threads 1   (simulation-only baseline)
# S2: mpiexec -n 2 --threads 2
# S3: sbatch -N 2 -n 2           (mpiexec -n 2)
# S4: sbatch -N 2 -n 4           (mpiexec -n 4 --threads 1)
#
# Usage:
#   /shared/run-tests.sh [--mode reduced|full]
#
#   reduced   T1, T2, L1, L2, R0, R1, S1 and S2 only (mpiexec, no sbatch / SLURM
#             cluster required).
#   full      every T, L, R and S case (default — preserves prior behavior).
#
# Prerequisites (satisfied by the Docker image):
#   - /shared/novomodelo-mpi   (novomodelo binary built with --features mpi)
#   - /shared/4ree        (example case directory)
#   - /opt/mpich/bin      (MPICH installation)
#   - python3 with pyarrow
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly NOVOMODELO_BIN="/shared/novomodelo-mpi"
readonly CASE_DIR="/shared/4ree"
readonly TIMEOUT=120
readonly WORK_DIR="/shared/work"
# Iteration whose final checkpoint the R cases resume.
readonly RESUME_AT=128

# Set by parse_mode(); read by main() and print_summary().
MODE="full"

# Prepend MPICH to PATH so mpiexec and Hydra proxy are found.
export PATH="/opt/mpich/bin:${PATH}"

# ---------------------------------------------------------------------------
# Result tracking (associative array: test name -> PASS|FAIL|<message>)
# ---------------------------------------------------------------------------
declare -A RESULTS

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage: run-tests.sh [--mode reduced|full]

  reduced   Run T1, T2, L1, L2, R0, R1, S1 and S2 (mpiexec only); no sbatch.
  full      Run T1 through T5, L1 through L3, R0 through R3 and S1 through S4
            (default).

  The L cases train a copy of the case with cut selection added to its
  config.json (level1, check_frequency 2, max_active_per_stage 100).
  R0 and R2 stop that copy at an earlier iteration; R1 and R3 resume their
  checkpoint at another rank count and are compared against L1.
USAGE
}

# ---------------------------------------------------------------------------
# parse_mode <args...>
#
# Sets the global MODE from a "--mode reduced|full" argument (default
# "full", preserving prior behavior). Any other argument, a missing --mode
# value, or an unrecognized --mode value prints usage and exits 1.
# ---------------------------------------------------------------------------
parse_mode() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: --mode requires a value (reduced|full)" >&2
                    usage
                    exit 1
                fi
                MODE="$2"
                shift 2
                ;;
            *)
                echo "ERROR: unrecognized argument: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    case "${MODE}" in
        reduced | full) ;;
        *)
            echo "ERROR: unrecognized --mode value: ${MODE} (expected reduced|full)" >&2
            usage
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight_check() {
    local ok=1

    if [[ ! -x "${NOVOMODELO_BIN}" ]]; then
        echo "ERROR: novomodelo binary not found or not executable: ${NOVOMODELO_BIN}"
        ok=0
    fi

    if [[ ! -d "${CASE_DIR}" ]]; then
        echo "ERROR: case directory not found: ${CASE_DIR}"
        ok=0
    fi

    if ! command -v mpiexec &>/dev/null; then
        echo "ERROR: mpiexec not found in PATH (checked /opt/mpich/bin)"
        ok=0
    fi

    if ! python3 -c "import pyarrow.parquet" 2>/dev/null; then
        echo "ERROR: python3 with pyarrow is required for output comparison"
        ok=0
    fi

    if [[ "${ok}" -eq 0 ]]; then
        echo "ABORT: pre-flight checks failed — see errors above"
        exit 1
    fi

    mkdir -p "${WORK_DIR}"

    echo "INFO: novomodelo binary: ${NOVOMODELO_BIN}"
    echo "INFO: case directory: ${CASE_DIR}"
    echo "INFO: work directory: ${WORK_DIR}"
    echo "INFO: mpiexec: $(command -v mpiexec)"
    echo "INFO: python3: $(command -v python3)"
    echo "INFO: mode: ${MODE}"
    echo "INFO: timeout per test: ${TIMEOUT}s"
}

# ---------------------------------------------------------------------------
# compare_outputs_bitwise <baseline_dir> <test_dir>
#
# Classifies every relative *.parquet path found under either directory
# against the CLASSIFICATION_ROWS table (first-match-wins; a path matching
# no rule FAILs as "unclassified output file"), asserts the two non-exempt
# relative-path sets are equal, then for each classified file compares
# every non-exempt column index-aligned (rows are NEVER sorted) and
# bit-exact for floats (raw IEEE-754 bytes, so -0.0/NaN/last-bit differences
# are caught) or by value for other types. Exits 1 naming the file, column,
# and row index on the first mismatch; container-level parquet metadata
# (row groups, codec, created_by) is never inspected — only decoded values.
# ---------------------------------------------------------------------------
compare_outputs_bitwise() {
    local baseline_dir="$1"
    local test_dir="$2"
    python3 - "${baseline_dir}" "${test_dir}" <<'PYEOF'
import glob
import os
import re
import struct
import sys
from collections import namedtuple

import pyarrow as pa
import pyarrow.parquet as pq

baseline_dir = sys.argv[1]
test_dir = sys.argv[2]

# Ordered (glob_pattern, class, exempt_columns) rules -- first match wins.
# class is one of "compare", "compare-except", "exempt". A *.parquet path
# found under either directory that matches no rule FAILs as unclassified:
# a new output added later forces a conscious classification decision here,
# never a silent skip.
CLASSIFICATION_ROWS = [
    # `solver_iterations_schema()` stamps every backward row with the
    # producing `rank`/`worker_id`/`opening`; both the row count (every
    # (rank, worker) pair also emits an opening=0 sentinel row, so the count
    # scales with the rank/worker grid) and the row order (rank-major,
    # worker-minor traversal) legitimately vary with rank count, so this
    # table -- and `training/solver/iterations.parquet` below, which shares
    # the schema -- is exempt outright, never `compare`/`compare-except`
    # (the same rationale as the already-exempt `training/timing/**`). Must
    # stay ahead of the `simulation/**/*.parquet` catch-all: exempt wins only
    # on a first match.
    ("simulation/solver/iterations.parquet", "exempt", []),
    ("simulation/**/*.parquet", "compare", []),
    (
        "training/convergence.parquet",
        "compare-except",
        ["time_forward_ms", "time_backward_ms", "time_total_ms"],
    ),
    ("training/dictionaries/*.parquet", "compare", []),
    ("training/cut_selection/iterations.parquet", "compare-except", ["selection_time_ms"]),
    # Shares `solver_iterations_schema()` with `simulation/solver/iterations.parquet`
    # above -- same exempt rationale.
    ("training/solver/iterations.parquet", "exempt", []),
    ("training/solver/retry_histogram.parquet", "compare", []),
    ("training/timing/**", "exempt", []),
]

Rule = namedtuple("Rule", ["cls", "exempt_columns"])


def _segment_to_regex(segment):
    return re.escape(segment).replace(r"\*", "[^/]*")


def glob_to_regex(pattern):
    """Translate the small glob subset used above into a regex.

    Supports a trailing "/**" (zero or more path segments) and a middle
    "/**/" (zero or more whole path segments), plus "*" matching within a
    single path segment. This is deliberately not a general glob engine --
    it covers exactly the patterns in CLASSIFICATION_ROWS.
    """
    if pattern.endswith("/**"):
        prefix = _segment_to_regex(pattern[: -len("/**")])
        return re.compile(f"^{prefix}(?:/.*)?$")
    if "/**/" in pattern:
        prefix, suffix = pattern.split("/**/", 1)
        prefix_re = _segment_to_regex(prefix)
        suffix_re = _segment_to_regex(suffix)
        return re.compile(f"^{prefix_re}(?:/.*)?/{suffix_re}$")
    return re.compile(f"^{_segment_to_regex(pattern)}$")


COMPILED_RULES = [
    (glob_to_regex(pattern), Rule(cls, exempt_columns))
    for pattern, cls, exempt_columns in CLASSIFICATION_ROWS
]


def classify(rel_path):
    for regex, rule in COMPILED_RULES:
        if regex.match(rel_path):
            return rule
    return None


def relative_parquet_paths(root):
    return [
        os.path.relpath(abs_path, root).replace(os.sep, "/")
        for abs_path in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    ]


def bits_of(value, byte_width):
    if value is None:
        return None
    return struct.pack(">d" if byte_width == 8 else ">f", value)


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


baseline_paths = set(relative_parquet_paths(baseline_dir))
test_paths = set(relative_parquet_paths(test_dir))
all_paths = sorted(baseline_paths | test_paths)

classified = {}
for rel_path in all_paths:
    rule = classify(rel_path)
    if rule is None:
        fail(f"{rel_path}: unclassified output file")
    classified[rel_path] = rule

non_exempt = sorted(p for p in all_paths if classified[p].cls != "exempt")
missing_from_test = [p for p in non_exempt if p not in test_paths]
missing_from_baseline = [p for p in non_exempt if p not in baseline_paths]
if missing_from_test or missing_from_baseline:
    for p in missing_from_test:
        print(f"FAIL: {p}: present in baseline, missing from test output")
    for p in missing_from_baseline:
        print(f"FAIL: {p}: present in test, missing from baseline output")
    sys.exit(1)

for rel_path in non_exempt:
    rule = classified[rel_path]
    # ParquetFile(path).read() reads the file's own columns only; pq.read_table()
    # would add a dictionary<int32> scenario_id partition column from the
    # scenario_id=NNNN/ Hive path that collides with the in-file int32 column
    # (the same read contract novomodelo's Python tests and native loader follow).
    baseline_table = pq.ParquetFile(os.path.join(baseline_dir, rel_path)).read()
    test_table = pq.ParquetFile(os.path.join(test_dir, rel_path)).read()

    baseline_cols = set(baseline_table.column_names)
    test_cols = set(test_table.column_names)
    if baseline_cols != test_cols:
        fail(
            f"{rel_path}: column set mismatch "
            f"(missing={sorted(baseline_cols - test_cols)}, "
            f"extra={sorted(test_cols - baseline_cols)})"
        )

    if baseline_table.num_rows != test_table.num_rows:
        fail(
            f"{rel_path}: row count mismatch "
            f"(baseline={baseline_table.num_rows}, test={test_table.num_rows})"
        )

    exempt_columns = set(rule.exempt_columns)
    for col_name in sorted(baseline_cols):
        if col_name in exempt_columns:
            continue

        field_type = baseline_table.schema.field(col_name).type
        is_float = pa.types.is_floating(field_type)
        byte_width = 4 if pa.types.is_float32(field_type) else 8

        # Index-aligned, never sorted: a row-order difference in a
        # compare-class table is a real finding, not noise to canonicalize.
        baseline_values = baseline_table.column(col_name).to_pylist()
        test_values = test_table.column(col_name).to_pylist()

        for row_idx, (baseline_value, test_value) in enumerate(
            zip(baseline_values, test_values)
        ):
            if is_float:
                baseline_repr = bits_of(baseline_value, byte_width)
                test_repr = bits_of(test_value, byte_width)
            else:
                baseline_repr = baseline_value
                test_repr = test_value

            if baseline_repr != test_repr:
                fail(
                    f"{rel_path}: column '{col_name}' row {row_idx} differs "
                    f"(baseline={baseline_value!r}, test={test_value!r})"
                )

print(f"INFO: {len(non_exempt)} classified output file(s) bit-identical")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# compare_solver_counters <baseline_dir> <test_dir>
#
# Compares simulation/solver/iterations.parquet with the `rank` and
# `worker_id` columns and every `*_ms` column dropped. The remaining rows are
# sorted before comparison because their order follows the rank and worker
# that wrote them. Exits 1 naming the first differing row.
# ---------------------------------------------------------------------------
compare_solver_counters() {
    local baseline_dir="$1"
    local test_dir="$2"
    python3 - "${baseline_dir}" "${test_dir}" <<'PYEOF'
import os
import sys

import pyarrow.parquet as pq

REL_PATH = "simulation/solver/iterations.parquet"
TOPOLOGY_COLUMNS = ("rank", "worker_id")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def counter_rows(root):
    path = os.path.join(root, REL_PATH)
    if not os.path.isfile(path):
        fail(f"{path}: missing")
    table = pq.ParquetFile(path).read()
    for column in TOPOLOGY_COLUMNS:
        if column not in table.column_names:
            fail(f"{path}: column '{column}' missing")
    kept = [
        c for c in table.column_names if c not in TOPOLOGY_COLUMNS and not c.endswith("_ms")
    ]
    columns = [table.column(c).to_pylist() for c in kept]
    # A null sorts after every value, so a nullable column never compares None with a value.
    rows = sorted(zip(*columns), key=lambda row: [(v is None, v) for v in row])
    return kept, rows


baseline_columns, baseline_rows = counter_rows(sys.argv[1])
test_columns, test_rows = counter_rows(sys.argv[2])

if baseline_columns != test_columns:
    fail(f"{REL_PATH}: column mismatch (baseline={baseline_columns}, test={test_columns})")
if len(baseline_rows) != len(test_rows):
    fail(
        f"{REL_PATH}: row count mismatch "
        f"(baseline={len(baseline_rows)}, test={len(test_rows)})"
    )
for index, (baseline_row, test_row) in enumerate(zip(baseline_rows, test_rows)):
    if baseline_row != test_row:
        fail(
            f"{REL_PATH}: sorted row {index} differs "
            f"(baseline={dict(zip(baseline_columns, baseline_row))}, "
            f"test={dict(zip(test_columns, test_row))})"
        )

print(f"INFO: {len(baseline_rows)} solver counter row(s) equal")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# compare_resumed_outputs <baseline_dir> <test_dir> <k>
#
# Compares a run resumed from the checkpoint of iteration <k> against the
# uninterrupted run in <baseline_dir>. Every file under policy/ and training/
# is classified by the RESUMED_ROWS table (first match wins; a path matching
# no rule FAILs as unclassified). The policy payloads must have equal path
# sets and be byte-identical. The per-iteration tables are compared on their
# rows with iteration > <k>, index-aligned, bit-exact except `lower_bound` and
# `gap_percent`, which must agree within 1e-6 relative
# (|a - b| <= 1e-6 * max(1, |a|, |b|)); a row with iteration <= <k> is not
# compared, but must name a key the baseline holds. The retry histogram is
# compared the same way without its lower_bound rows, and may hold no row.
# metadata.json and codes.json are compared key by key, except the keys a
# resumed run writes from its own clock, host, rank layout and solves; the
# final lower bound and gap within the same tolerance. simulation/ is
# compared by compare_outputs_bitwise. Prints the largest relative
# lower-bound difference and exits 1 naming the first mismatch.
# ---------------------------------------------------------------------------
compare_resumed_outputs() {
    local baseline_dir="$1"
    local test_dir="$2"
    local k="$3"
    local exit_code=0
    python3 - "${baseline_dir}" "${test_dir}" "${k}" <<'PYEOF' || exit_code=$?
import fnmatch
import json
import os
import struct
import sys

import pyarrow as pa
import pyarrow.parquet as pq

baseline_dir = sys.argv[1]
test_dir = sys.argv[2]
k = int(sys.argv[3])

RELATIVE_TOLERANCE = 1e-6
TOLERANT_COLUMNS = {"lower_bound", "gap_percent"}

# Ordered (fnmatch pattern, class, detail) rules -- first match wins, and `*`
# also matches `/`. Classes:
#   bytes                byte-identical; detail unused
#   rows-after-k         rows with iteration > k compared; detail is
#                        (exempt columns, the key, iteration first, that an
#                        earlier row must share with a baseline row)
#   sparse-rows-after-k  rows-after-k for a table that may hold no row; detail
#                        adds the phases whose rows are not compared
#   table                every row compared; detail is the exempt columns
#   json                 compared key by key, a nested key named by its dotted
#                        path; detail is (exempt key patterns, keys compared
#                        within RELATIVE_TOLERANCE)
#   exempt               not compared; detail is the reason
RESUMED_ROWS = [
    ("policy/manifest.bin", "exempt", "holds created_at"),
    ("policy/cuts/*.bin", "bytes", None),
    ("policy/basis/*.bin", "bytes", None),
    ("policy/states/*.bin", "bytes", None),
    (
        "training/convergence.parquet",
        "rows-after-k",
        ({"time_forward_ms", "time_backward_ms", "time_total_ms"}, ("iteration",)),
    ),
    (
        "training/cut_selection/iterations.parquet",
        "rows-after-k",
        ({"selection_time_ms"}, ("iteration", "pool_id")),
    ),
    ("training/_SUCCESS", "bytes", None),
    (
        "training/metadata.json",
        "json",
        (
            # Wall clock, host, rank layout, policy mode, and the counts the
            # run takes over its own iterations only.
            (
                "hostname",
                "started_at",
                "completed_at",
                "duration_seconds",
                "configuration.policy_mode",
                "row_pool.peak_active",
                "row_pool.rows_in_lp_*",
                "solve_stats.*",
                "setup.*",
                "distribution.*",
            ),
            {"bounds.final_lower_bound", "convergence.final_gap_percent"},
        ),
    ),
    ("training/hydro_models.json", "bytes", None),
    ("training/model_provenance.json", "bytes", None),
    ("training/scaling_report.json", "bytes", None),
    ("training/dictionaries/codes.json", "json", (("generated_at",), set())),
    ("training/dictionaries/*.csv", "bytes", None),
    ("training/dictionaries/*.parquet", "table", set()),
    ("training/solver/iterations.parquet", "exempt", "rows stamped with rank and worker"),
    (
        "training/solver/retry_histogram.parquet",
        "sparse-rows-after-k",
        # A resume rebuilds the lower-bound LP cold, so its retries differ.
        (set(), ("iteration", "phase", "stage_id", "retry_level"), {"lower_bound"}),
    ),
    ("training/timing/*", "exempt", "wall clock"),
]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def classify(rel_path):
    for pattern, cls, detail in RESUMED_ROWS:
        if fnmatch.fnmatchcase(rel_path, pattern):
            return cls, detail
    return None


def files_under(root, top):
    found = set()
    for dirpath, _, filenames in os.walk(os.path.join(root, top)):
        for name in filenames:
            found.add(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/"))
    return found


def compare_bytes(rel_path):
    with open(os.path.join(baseline_dir, rel_path), "rb") as f:
        baseline_bytes = f.read()
    with open(os.path.join(test_dir, rel_path), "rb") as f:
        test_bytes = f.read()
    if baseline_bytes == test_bytes:
        return
    first = next(
        (i for i, (a, b) in enumerate(zip(baseline_bytes, test_bytes)) if a != b),
        min(len(baseline_bytes), len(test_bytes)),
    )
    fail(
        f"{rel_path}: differs at byte {first} "
        f"(baseline {len(baseline_bytes)} bytes, test {len(test_bytes)} bytes)"
    )


def read_table(root, rel_path):
    return pq.ParquetFile(os.path.join(root, rel_path)).read()


def bits_of(value, byte_width):
    return struct.pack(">d" if byte_width == 8 else ">f", value)


largest_lower_bound_difference = None


def compare_rows(rel_path, baseline_table, test_table, exempt_columns):
    global largest_lower_bound_difference
    baseline_cols = set(baseline_table.column_names)
    test_cols = set(test_table.column_names)
    if baseline_cols != test_cols:
        fail(
            f"{rel_path}: column set mismatch "
            f"(missing={sorted(baseline_cols - test_cols)}, "
            f"extra={sorted(test_cols - baseline_cols)})"
        )
    if baseline_table.num_rows != test_table.num_rows:
        fail(
            f"{rel_path}: compared row count mismatch "
            f"(baseline={baseline_table.num_rows}, test={test_table.num_rows})"
        )
    iterations = (
        baseline_table.column("iteration").to_pylist()
        if "iteration" in baseline_cols
        else [None] * baseline_table.num_rows
    )
    for col_name in sorted(baseline_cols - set(exempt_columns)):
        field_type = baseline_table.schema.field(col_name).type
        is_float = pa.types.is_floating(field_type)
        byte_width = 4 if pa.types.is_float32(field_type) else 8
        tolerant = is_float and col_name in TOLERANT_COLUMNS
        baseline_values = baseline_table.column(col_name).to_pylist()
        test_values = test_table.column(col_name).to_pylist()
        for row, (a, b) in enumerate(zip(baseline_values, test_values)):
            if a is None or b is None:
                same = a is None and b is None
            elif tolerant:
                scale = max(1.0, abs(a), abs(b))
                same = abs(a - b) <= RELATIVE_TOLERANCE * scale
                if same and col_name == "lower_bound":
                    relative = abs(a - b) / scale
                    if (
                        largest_lower_bound_difference is None
                        or relative > largest_lower_bound_difference[0]
                    ):
                        largest_lower_bound_difference = (relative, iterations[row])
            elif is_float:
                same = bits_of(a, byte_width) == bits_of(b, byte_width)
            else:
                same = a == b
            if not same:
                fail(
                    f"{rel_path}: column '{col_name}' differs at iteration "
                    f"{iterations[row]} (compared row {row}; baseline={a!r}, test={b!r})"
                )


def select_rows(table, keep):
    # Typed indices: pyarrow cannot take with an untyped empty list.
    return table.take(pa.array([i for i, kept in enumerate(keep) if kept], type=pa.int64()))


def rows_after_k(table):
    return select_rows(table, (iteration > k for iteration in table.column("iteration").to_pylist()))


def without_phases(table, phases):
    return select_rows(table, (phase not in phases for phase in table.column("phase").to_pylist()))


def compare_rows_after_k(
    rel_path, exempt_columns, key_columns, excluded_phases=frozenset(), may_be_empty=False
):
    required_columns = tuple(key_columns) + (("phase",) if excluded_phases else ())
    tables = []
    for name, root in (("baseline", baseline_dir), ("test", test_dir)):
        table = read_table(root, rel_path)
        for column in required_columns:
            if column not in table.column_names:
                fail(f"{rel_path}: column '{column}' missing from the {name} output")
        tables.append(without_phases(table, excluded_phases) if excluded_phases else table)
    baseline_table, test_table = tables
    baseline_keys = set(zip(*(baseline_table.column(c).to_pylist() for c in key_columns)))
    test_keys = zip(*(test_table.column(c).to_pylist() for c in key_columns))
    for key in test_keys:
        if key[0] <= k and key not in baseline_keys:
            fail(f"{rel_path}: row {dict(zip(key_columns, key))} is absent from the baseline")
    baseline_after = rows_after_k(baseline_table)
    if baseline_after.num_rows == 0 and not may_be_empty:
        fail(f"{rel_path}: the baseline holds no row after iteration {k}")
    compare_rows(rel_path, baseline_after, rows_after_k(test_table), exempt_columns)


def flatten_json(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    flat = {}
    for key, child in value.items():
        flat.update(flatten_json(child, f"{prefix}.{key}" if prefix else key))
    return flat


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json(rel_path, exempt_patterns, tolerant_keys):
    flats = []
    for root in (baseline_dir, test_dir):
        with open(os.path.join(root, rel_path)) as f:
            flat = flatten_json(json.load(f))
        flats.append(
            {
                key: value
                for key, value in flat.items()
                if not any(fnmatch.fnmatchcase(key, pattern) for pattern in exempt_patterns)
            }
        )
    baseline, test = flats
    if baseline.keys() != test.keys():
        fail(
            f"{rel_path}: key set mismatch "
            f"(missing={sorted(baseline.keys() - test.keys())}, "
            f"extra={sorted(test.keys() - baseline.keys())})"
        )
    for key in sorted(baseline):
        a, b = baseline[key], test[key]
        if key in tolerant_keys and is_number(a) and is_number(b):
            same = abs(a - b) <= RELATIVE_TOLERANCE * max(1.0, abs(a), abs(b))
        else:
            same = type(a) is type(b) and a == b
        if not same:
            fail(f"{rel_path}: key '{key}' differs (baseline={a!r}, test={b!r})")


all_paths = sorted(
    files_under(baseline_dir, "policy")
    | files_under(test_dir, "policy")
    | files_under(baseline_dir, "training")
    | files_under(test_dir, "training")
)
if not any(fnmatch.fnmatchcase(p, "policy/cuts/*.bin") for p in all_paths):
    fail("policy/cuts: no file in either output")

compared = 0
for rel_path in all_paths:
    rule = classify(rel_path)
    if rule is None:
        fail(f"{rel_path}: unclassified output file")
    cls, detail = rule
    if cls == "exempt":
        continue
    for name, root in (("baseline", baseline_dir), ("test", test_dir)):
        if not os.path.isfile(os.path.join(root, rel_path)):
            fail(f"{rel_path}: missing from the {name} output")
    if cls == "bytes":
        compare_bytes(rel_path)
    elif cls == "rows-after-k":
        compare_rows_after_k(rel_path, *detail)
    elif cls == "sparse-rows-after-k":
        compare_rows_after_k(rel_path, *detail, may_be_empty=True)
    elif cls == "json":
        compare_json(rel_path, *detail)
    else:
        compare_rows(
            rel_path,
            read_table(baseline_dir, rel_path),
            read_table(test_dir, rel_path),
            detail,
        )
    compared += 1

print(f"INFO: {compared} policy and training file(s) match after iteration {k}")
if largest_lower_bound_difference is not None:
    relative, iteration = largest_lower_bound_difference
    print(
        f"INFO: largest relative lower_bound difference after iteration {k}: "
        f"{relative:.3e} (iteration {iteration})"
    )
sys.exit(0)
PYEOF
    if [[ "${exit_code}" -ne 0 ]]; then
        return "${exit_code}"
    fi

    # compare_outputs_bitwise classifies paths relative to the directory it is
    # given, so it reads each run's simulation/ through a view holding only it.
    local views
    views="$(mktemp -d)"
    mkdir -p "${views}/baseline" "${views}/test"
    ln -s "${baseline_dir}/simulation" "${views}/baseline/simulation"
    ln -s "${test_dir}/simulation" "${views}/test/simulation"
    compare_outputs_bitwise "${views}/baseline" "${views}/test" || exit_code=$?
    rm -rf "${views}"
    return "${exit_code}"
}

# ---------------------------------------------------------------------------
# assert_selection_power <output_dir>
#
# Reads training/cut_selection/iterations.parquet under <output_dir> and
# exits 1 unless the sums of `cuts_deactivated`, `cuts_reactivated` and
# `budget_evicted` (nulls ignored) are each positive and some row whose
# `cuts_deactivated` is null (an iteration without a selection run) has
# `budget_evicted > 0`. Prints the three sums.
# ---------------------------------------------------------------------------
assert_selection_power() {
    local output_dir="$1"
    python3 - "${output_dir}" <<'PYEOF'
import os
import sys

import pyarrow.parquet as pq

REL_PATH = "training/cut_selection/iterations.parquet"
SUMMED_COLUMNS = ("cuts_deactivated", "cuts_reactivated", "budget_evicted")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


path = os.path.join(sys.argv[1], REL_PATH)
if not os.path.isfile(path):
    fail(f"{path}: missing")
table = pq.ParquetFile(path).read()
for column in SUMMED_COLUMNS:
    if column not in table.column_names:
        fail(f"{path}: column '{column}' missing")

values = {column: table.column(column).to_pylist() for column in SUMMED_COLUMNS}
sums = {column: sum(v for v in values[column] if v is not None) for column in SUMMED_COLUMNS}
print("INFO: selection power: " + ", ".join(f"{c}={sums[c]}" for c in SUMMED_COLUMNS))

for column in SUMMED_COLUMNS:
    if sums[column] <= 0:
        fail(f"{path}: '{column}' sums to {sums[column]}, expected > 0")

evicted_between_selection_runs = any(
    deactivated is None and evicted is not None and evicted > 0
    for deactivated, evicted in zip(values["cuts_deactivated"], values["budget_evicted"])
)
if not evicted_between_selection_runs:
    fail(f"{path}: no row with a null 'cuts_deactivated' has 'budget_evicted' > 0")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# assert_selection_power_of_case <baseline_dir> <case_dir>
#
# assert_selection_power on <case_dir>, with the two-directory signature
# assert_compare_exit calls its comparator with; <baseline_dir> is unused.
# ---------------------------------------------------------------------------
assert_selection_power_of_case() {
    assert_selection_power "$2"
}

# ---------------------------------------------------------------------------
# run_comparator <comparator> <baseline_dir> <test_dir>
#
# Runs <comparator>: a function name, then any arguments the function takes
# after the two directories (as in "compare_resumed_outputs 128").
# ---------------------------------------------------------------------------
run_comparator() {
    local -a words
    read -r -a words <<<"$1"
    "${words[0]}" "$2" "$3" "${words[@]:1}"
}

# ---------------------------------------------------------------------------
# assert_compare_exit <label> <baseline_dir> <case_dir> <expected_exit> [comparator]
#
# Runs <comparator> (default compare_outputs_bitwise) through run_comparator
# and asserts its exit code matches expectation, counting the case in
# SELF_CHECK_CASES and recording any mismatch in SELF_CHECK_FAILURES (both
# used by run_self_check).
# ---------------------------------------------------------------------------
assert_compare_exit() {
    local label="$1"
    local baseline_dir="$2"
    local case_dir="$3"
    local expected_exit="$4"
    local comparator="${5:-compare_outputs_bitwise}"
    local actual_exit=0

    SELF_CHECK_CASES=$((SELF_CHECK_CASES + 1))
    echo "--- self-check case: ${label} ---"
    run_comparator "${comparator}" "${baseline_dir}" "${case_dir}" || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        echo "[self-check:${label}] PASS (exit ${actual_exit}, expected ${expected_exit})"
    else
        echo "[self-check:${label}] FAIL: exit ${actual_exit}, expected ${expected_exit}"
        SELF_CHECK_FAILURES=1
    fi
}

# ---------------------------------------------------------------------------
# run_self_check
#
# Exercises compare_outputs_bitwise against tiny generated fixtures before
# trusting it on a live run: identical dirs PASS, a single differing f64 bit
# in a compare-class column FAILs, exempt-column/exempt-path-only
# differences PASS, an unclassified parquet path FAILs, and a
# training/solver/iterations.parquet differing in `rank` and a wall-clock
# column PASSes (the whole file is exempt). Then exercises
# compare_solver_counters: rows differing only in `rank`, `worker_id`, a
# `*_ms` value and row order PASS, and one differing `basis_offered` FAILs.
# Then exercises assert_selection_power: a selection file meeting all four
# conditions PASSes, and one whose `cuts_reactivated` sums to 0 FAILs. Then
# exercises compare_resumed_outputs with k = 2: identical runs PASS, one
# differing byte in a cuts file FAILs, a lower bound of iteration 3 differing
# by 1e-9 relative PASSes and by 1e-5 FAILs, a differing row at iteration 2
# PASSes, a missing basis file FAILs, metadata.json and codes.json differing
# only in the keys a resumed run writes for itself PASS, a differing
# completed-iteration count or dictionary code FAILs, retry-histogram
# lower_bound rows differing after iteration 2 PASS and a differing backward
# row FAILs, and an unclassified file under policy/ FAILs.
# Aborts (exit 1) if any case's exit code does not match its expectation.
# ---------------------------------------------------------------------------
run_self_check() {
    local root
    root="$(mktemp -d)"

    echo ""
    echo "=== Self-check: comparator fixtures ==="

    python3 - "${root}" <<'PYEOF'
import copy
import json
import math
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

root = sys.argv[1]


def write(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(columns), path)


def write_common(base):
    write(
        os.path.join(base, "simulation", "costs", "data.parquet"),
        {"scenario_id": [0, 0], "value": [1.0, 2.5]},
    )
    write(
        os.path.join(base, "training", "convergence.parquet"),
        {"lower_bound": [10.0, 10.0], "time_forward_ms": [5, 6]},
    )
    write(
        os.path.join(base, "training", "timing", "iterations.parquet"),
        {"iteration": [1, 2], "forward_wall_ms": [3, 4]},
    )


def write_solver_iterations(base, rank, solve_time_ms):
    # Mirrors the columns of solver_iterations_schema() relevant to the
    # exempt reclassification: `rank` (execution topology) and
    # `solve_time_ms` (wall-clock) are exactly the two column kinds that
    # made this table bitwise-incomparable across rank counts.
    write(
        os.path.join(base, "training", "solver", "iterations.parquet"),
        {
            "iteration": [1, 1],
            "phase": ["backward", "backward"],
            "stage": [0, 0],
            "opening": [0, 1],
            "rank": [rank, rank],
            "worker_id": [0, 0],
            "lp_solves": [1, 1],
            "solve_time_ms": [solve_time_ms, solve_time_ms],
        },
    )


# baseline: the reference every other case is compared against.
write_common(os.path.join(root, "baseline"))
write_solver_iterations(os.path.join(root, "baseline"), rank=0, solve_time_ms=1.0)

# case_identical: decoded values identical -> expect PASS.
write_common(os.path.join(root, "case_identical"))

# case_bitdiff: one f64 bit differs in a compare-class column -> expect FAIL.
write(
    os.path.join(root, "case_bitdiff", "simulation", "costs", "data.parquet"),
    {"scenario_id": [0, 0], "value": [1.0, math.nextafter(2.5, math.inf)]},
)
write(
    os.path.join(root, "case_bitdiff", "training", "convergence.parquet"),
    {"lower_bound": [10.0, 10.0], "time_forward_ms": [5, 6]},
)
write(
    os.path.join(root, "case_bitdiff", "training", "timing", "iterations.parquet"),
    {"iteration": [1, 2], "forward_wall_ms": [3, 4]},
)

# case_exempt_diff: differs only in exempt columns/paths -> expect PASS.
write(
    os.path.join(root, "case_exempt_diff", "simulation", "costs", "data.parquet"),
    {"scenario_id": [0, 0], "value": [1.0, 2.5]},
)
write(
    os.path.join(root, "case_exempt_diff", "training", "convergence.parquet"),
    {"lower_bound": [10.0, 10.0], "time_forward_ms": [999, 999]},
)
write(
    os.path.join(root, "case_exempt_diff", "training", "timing", "iterations.parquet"),
    {"iteration": [1, 2, 3], "forward_wall_ms": [100, 200, 300]},
)

# case_unclassified: identical to baseline plus one path matching no rule.
write_common(os.path.join(root, "case_unclassified"))
write(os.path.join(root, "case_unclassified", "random", "unclassified.parquet"), {"x": [1]})

# case_exempt_solver_iterations: training/solver/iterations.parquet differs
# in `rank` and in the wall-clock `solve_time_ms` column -> expect PASS
# (the whole file is exempt, so neither difference is ever compared).
write_common(os.path.join(root, "case_exempt_solver_iterations"))
write_solver_iterations(
    os.path.join(root, "case_exempt_solver_iterations"), rank=1, solve_time_ms=999.0
)


def write_simulation_solver_iterations(base, rows):
    names = ["scenario_id", "phase", "rank", "worker_id", "basis_offered", "solve_time_ms"]
    write(
        os.path.join(base, "simulation", "solver", "iterations.parquet"),
        {name: [row[i] for row in rows] for i, name in enumerate(names)},
    )


write_simulation_solver_iterations(
    os.path.join(root, "counters_baseline"),
    [(0, "simulation", 0, 0, 2, 1.0), (1, "simulation", 0, 1, 2, 1.5)],
)

# counters_layout_diff: rows differ only in rank, worker_id, a *_ms value
# and row order -> expect PASS.
write_simulation_solver_iterations(
    os.path.join(root, "counters_layout_diff"),
    [(1, "simulation", 1, 0, 2, 1.5), (0, "simulation", 0, 0, 2, 999.0)],
)

# counters_basis_diff: one basis_offered value differs -> expect FAIL.
write_simulation_solver_iterations(
    os.path.join(root, "counters_basis_diff"),
    [(0, "simulation", 0, 0, 2, 1.0), (1, "simulation", 0, 1, 0, 1.5)],
)


def write_selection_iterations(base, cuts_reactivated):
    write(
        os.path.join(base, "training", "cut_selection", "iterations.parquet"),
        {
            "iteration": pa.array([1, 2, 2], type=pa.int32()),
            "pool_id": pa.array([1, 1, 2], type=pa.int32()),
            "cuts_deactivated": pa.array([None, 2, 1], type=pa.int32()),
            "cuts_reactivated": pa.array(cuts_reactivated, type=pa.int32()),
            "budget_evicted": pa.array([3, None, 0], type=pa.int32()),
        },
    )


# selection_power: positive sums and an eviction on an iteration without a
# selection run -> expect PASS.
write_selection_iterations(os.path.join(root, "selection_power"), [None, 1, 0])

# selection_no_reactivation: `cuts_reactivated` sums to 0 -> expect FAIL.
write_selection_iterations(os.path.join(root, "selection_no_reactivation"), [None, 0, 0])


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


RESUME_LOWER_BOUNDS = [1.0e6, 2.0e6, 3.0e6, 4.0e6]
RESUME_METADATA = {
    "software": "novomodelo",
    "hostname": "node-a",
    "started_at": "2026-01-01T00:00:00Z",
    "duration_seconds": 1.5,
    "configuration": {"max_iterations": 4, "policy_mode": "fresh"},
    "iterations": {"completed": 4},
    "row_pool": {"total_generated": 8, "peak_active": 6},
    "bounds": {"final_lower_bound": 4.0e6},
    "solve_stats": {"total_lp_solves": 40, "parallelism": 1},
    "distribution": {"world_size": 1, "hosts": [{"hostname": "node-a", "ranks": [0]}]},
}
RESUME_CODES = {"generated_at": "2026-01-01T00:00:00Z", "entity_type": {"hydro": 0}}
# (iteration, phase, stage_id, retry_level, count)
RESUME_RETRIES = [
    (1, "lower_bound", None, 1, 1),
    (3, "backward", 2, 1, 1),
    (3, "lower_bound", None, 1, 2),
]


def with_values(document, values):
    document = copy.deepcopy(document)
    for path, value in values.items():
        *parents, leaf = path.split(".")
        node = document
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    return document


def write_json(path, document):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(document, f)


def write_resumed_run(
    base,
    iterations=(1, 2, 3, 4),
    lower_bounds=None,
    cuts=b"\x01\x02\x03",
    basis=True,
    metadata=RESUME_METADATA,
    codes=RESUME_CODES,
    retries=RESUME_RETRIES,
):
    # The comparator is run with k = 2, so only iterations 3 and 4 are compared.
    lower_bounds = lower_bounds or [RESUME_LOWER_BOUNDS[i - 1] for i in iterations]
    write_bytes(os.path.join(base, "policy", "manifest.bin"), b"manifest")
    write_bytes(os.path.join(base, "policy", "cuts", "000.bin"), cuts)
    if basis:
        write_bytes(os.path.join(base, "policy", "basis", "000.bin"), b"\x04\x05")
    write_bytes(os.path.join(base, "policy", "states", "001.bin"), b"\x06")
    write(
        os.path.join(base, "training", "convergence.parquet"),
        {
            "iteration": pa.array(iterations, type=pa.int32()),
            "lower_bound": lower_bounds,
            "upper_bound": [5.0e6] * len(iterations),
            "gap_percent": [1.0] * len(iterations),
            "time_total_ms": [7] * len(iterations),
        },
    )
    write(
        os.path.join(base, "training", "cut_selection", "iterations.parquet"),
        {
            "iteration": pa.array(iterations, type=pa.int32()),
            "pool_id": pa.array([1] * len(iterations), type=pa.int32()),
            "cuts_deactivated": pa.array([2] * len(iterations), type=pa.int32()),
            "selection_time_ms": [0.5] * len(iterations),
        },
    )
    write(
        os.path.join(base, "simulation", "costs", "data.parquet"),
        {"scenario_id": [0], "value": [1.0]},
    )
    write_json(os.path.join(base, "training", "metadata.json"), metadata)
    write_json(os.path.join(base, "training", "dictionaries", "codes.json"), codes)
    write(
        os.path.join(base, "training", "solver", "retry_histogram.parquet"),
        {
            "iteration": pa.array([r[0] for r in retries], type=pa.uint32()),
            "phase": pa.array([r[1] for r in retries], type=pa.string()),
            "stage_id": pa.array([r[2] for r in retries], type=pa.int32()),
            "retry_level": pa.array([r[3] for r in retries], type=pa.uint32()),
            "count": pa.array([r[4] for r in retries], type=pa.uint64()),
        },
    )


# resume_identical: the uninterrupted run itself -> expect PASS.
write_resumed_run(os.path.join(root, "resume_baseline"))
write_resumed_run(os.path.join(root, "resume_identical"))

# resume_cuts_byte: one byte of a cuts file differs -> expect FAIL.
write_resumed_run(os.path.join(root, "resume_cuts_byte"), cuts=b"\x01\x02\x04")

# resume_lower_bound_1e-9 / 1e-5: the lower bound of iteration 3 differs by
# that relative amount -> expect PASS / FAIL.
for label, relative in (("1e-9", 1e-9), ("1e-5", 1e-5)):
    bounds = list(RESUME_LOWER_BOUNDS)
    bounds[2] *= 1.0 + relative
    write_resumed_run(os.path.join(root, f"resume_lower_bound_{label}"), lower_bounds=bounds)

# resume_early_row: the file starts at iteration 2, whose lower bound differs
# -> expect PASS.
write_resumed_run(
    os.path.join(root, "resume_early_row"),
    iterations=(2, 3, 4),
    lower_bounds=[9.0e6] + RESUME_LOWER_BOUNDS[2:],
)

# resume_missing_basis: the basis file is missing -> expect FAIL.
write_resumed_run(os.path.join(root, "resume_missing_basis"), basis=False)

# resume_own_keys: metadata.json and codes.json differ only in the keys a
# resumed run writes for itself, and in a final lower bound 1e-9 relative
# away -> expect PASS.
write_resumed_run(
    os.path.join(root, "resume_own_keys"),
    metadata=with_values(
        RESUME_METADATA,
        {
            "hostname": "node-b",
            "started_at": "2026-01-02T00:00:00Z",
            "duration_seconds": 0.7,
            "configuration.policy_mode": "resume",
            "row_pool.peak_active": 5,
            "bounds.final_lower_bound": 4.0e6 * (1.0 + 1e-9),
            "solve_stats.total_lp_solves": 20,
            "solve_stats.parallelism": 2,
            "distribution.world_size": 2,
            "distribution.hosts": [{"hostname": "node-b", "ranks": [0, 1]}],
        },
    ),
    codes=with_values(RESUME_CODES, {"generated_at": "2026-01-02T00:00:00Z"}),
)

# resume_metadata_iterations: the completed-iteration count differs -> expect
# FAIL.
write_resumed_run(
    os.path.join(root, "resume_metadata_iterations"),
    metadata=with_values(RESUME_METADATA, {"iterations.completed": 3}),
)

# resume_codes_entry: a dictionary code differs -> expect FAIL.
write_resumed_run(
    os.path.join(root, "resume_codes_entry"),
    codes=with_values(RESUME_CODES, {"entity_type.hydro": 1}),
)

# resume_retry_lower_bound: the lower_bound rows differ after iteration 2 and
# the row of iteration 1 is absent -> expect PASS.
write_resumed_run(
    os.path.join(root, "resume_retry_lower_bound"),
    retries=[
        (3, "backward", 2, 1, 1),
        (3, "lower_bound", None, 1, 5),
        (4, "lower_bound", None, 2, 1),
    ],
)

# resume_retry_backward: a backward row of iteration 3 differs -> expect FAIL.
write_resumed_run(
    os.path.join(root, "resume_retry_backward"),
    retries=[
        (1, "lower_bound", None, 1, 1),
        (3, "backward", 2, 1, 2),
        (3, "lower_bound", None, 1, 2),
    ],
)

# resume_unclassified_policy: identical plus one policy/ file no rule names
# -> expect FAIL.
write_resumed_run(os.path.join(root, "resume_unclassified_policy"))
write_bytes(os.path.join(root, "resume_unclassified_policy", "policy", "selection", "000.bin"), b"\x07")
PYEOF

    SELF_CHECK_CASES=0
    SELF_CHECK_FAILURES=0
    assert_compare_exit "identical" "${root}/baseline" "${root}/case_identical" 0
    assert_compare_exit "bitdiff" "${root}/baseline" "${root}/case_bitdiff" 1
    assert_compare_exit "exempt_diff" "${root}/baseline" "${root}/case_exempt_diff" 0
    assert_compare_exit "unclassified" "${root}/baseline" "${root}/case_unclassified" 1
    assert_compare_exit "exempt_solver_iterations" "${root}/baseline" \
        "${root}/case_exempt_solver_iterations" 0
    assert_compare_exit "counters_layout_diff" "${root}/counters_baseline" \
        "${root}/counters_layout_diff" 0 compare_solver_counters
    assert_compare_exit "counters_basis_diff" "${root}/counters_baseline" \
        "${root}/counters_basis_diff" 1 compare_solver_counters
    assert_compare_exit "selection_power" "" "${root}/selection_power" 0 \
        assert_selection_power_of_case
    assert_compare_exit "selection_no_reactivation" "" "${root}/selection_no_reactivation" 1 \
        assert_selection_power_of_case
    local resume_case expected_exit
    for resume_case in identical:0 cuts_byte:1 lower_bound_1e-9:0 lower_bound_1e-5:1 \
        early_row:0 missing_basis:1 own_keys:0 metadata_iterations:1 codes_entry:1 \
        retry_lower_bound:0 retry_backward:1 unclassified_policy:1; do
        expected_exit="${resume_case##*:}"
        resume_case="${resume_case%:*}"
        assert_compare_exit "resume_${resume_case}" "${root}/resume_baseline" \
            "${root}/resume_${resume_case}" "${expected_exit}" "compare_resumed_outputs 2"
    done

    rm -rf "${root}"

    if [[ "${SELF_CHECK_FAILURES}" -ne 0 ]]; then
        echo "ABORT: comparator self-check failed — see above"
        exit 1
    fi

    echo "INFO: comparator self-check passed (${SELF_CHECK_CASES}/${SELF_CHECK_CASES} cases)"
}

# ---------------------------------------------------------------------------
# prepare_partial_misfit_case <src_case> <dst_case>
#
# Copies <src_case> to <dst_case>, disables training, and splits the single
# block of stages 4 to 7 into two blocks of half its hours. A policy trained
# on <src_case> then holds stored bases that fit every stage but 4 to 7.
# ---------------------------------------------------------------------------
prepare_partial_misfit_case() {
    local src_case="$1"
    local dst_case="$2"
    mkdir -p "${dst_case}"
    cp -R "${src_case}/." "${dst_case}/"
    python3 - "${dst_case}" <<'PYEOF'
import json
import os
import sys

SPLIT_STAGE_IDS = {4, 5, 6, 7}

case_dir = sys.argv[1]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def load(name):
    path = os.path.join(case_dir, name)
    if not os.path.isfile(path):
        fail(f"{path}: missing")
    with open(path) as f:
        return path, json.load(f)


def dump(path, document):
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


config_path, config = load("config.json")
config.setdefault("training", {})["enabled"] = False
dump(config_path, config)

stages_path, stages = load("stages.json")
split = set()
for stage in stages["stages"]:
    if stage["id"] not in SPLIT_STAGE_IDS:
        continue
    if len(stage["blocks"]) != 1:
        fail(f"{stages_path}: stage {stage['id']} has {len(stage['blocks'])} blocks, expected 1")
    hours = stage["blocks"][0]["hours"]
    stage["blocks"] = [
        {"id": 0, "name": "A", "hours": hours / 2},
        {"id": 1, "name": "B", "hours": hours / 2},
    ]
    split.add(stage["id"])
if split != SPLIT_STAGE_IDS:
    fail(f"{stages_path}: stages {sorted(SPLIT_STAGE_IDS - split)} not found")
dump(stages_path, stages)
PYEOF
}

# ---------------------------------------------------------------------------
# prepare_selection_case <src_case> <dst_case>
#
# Copies <src_case> to <dst_case> and sets training.cut_selection in the
# copy's config.json to level1 every second iteration with at most 100
# active cuts per stage. Exits 1 if <src_case> already sets it.
# ---------------------------------------------------------------------------
prepare_selection_case() {
    local src_case="$1"
    local dst_case="$2"
    mkdir -p "${dst_case}"
    cp -R "${src_case}/." "${dst_case}/"
    python3 - "${dst_case}" <<'PYEOF'
import json
import os
import sys

CUT_SELECTION = {
    "selection": {"method": "level1", "check_frequency": 2},
    "max_active_per_stage": 100,
}

config_path = os.path.join(sys.argv[1], "config.json")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


if not os.path.isfile(config_path):
    fail(f"{config_path}: missing")
with open(config_path) as f:
    config = json.load(f)
training = config.setdefault("training", {})
if "cut_selection" in training:
    fail(f"{config_path}: training.cut_selection already set")
training["cut_selection"] = CUT_SELECTION
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PYEOF
}

# ---------------------------------------------------------------------------
# prepare_resume_cases <selection_case> <partial_case> <resume_case>
#
# Copies <selection_case> to <partial_case> and <resume_case>. The partial
# copy's iteration limit becomes RESUME_AT and nothing else changes; the
# resume copy sets policy.mode to "resume" and keeps the limit, since
# iteration numbers are absolute across a resume. Exits 1 unless
# the selection case has exactly one iteration_limit rule with a limit above
# RESUME_AT and sets no policy.mode.
# ---------------------------------------------------------------------------
prepare_resume_cases() {
    local selection_case="$1"
    local partial_case="$2"
    local resume_case="$3"
    local case_dir
    for case_dir in "${partial_case}" "${resume_case}"; do
        mkdir -p "${case_dir}"
        cp -R "${selection_case}/." "${case_dir}/"
    done
    python3 - "${partial_case}" "${resume_case}" "${RESUME_AT}" <<'PYEOF'
import json
import os
import sys

partial_case = sys.argv[1]
resume_case = sys.argv[2]
resume_at = int(sys.argv[3])


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def load(case_dir):
    path = os.path.join(case_dir, "config.json")
    if not os.path.isfile(path):
        fail(f"{path}: missing")
    with open(path) as f:
        return path, json.load(f)


def dump(path, config):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def iteration_limit_rule(path, config):
    rules = [
        rule
        for rule in config.get("training", {}).get("stopping_rules", [])
        if rule.get("type") == "iteration_limit"
    ]
    if len(rules) != 1:
        fail(f"{path}: {len(rules)} iteration_limit rules in training.stopping_rules, expected 1")
    if not isinstance(rules[0].get("limit"), int) or rules[0]["limit"] <= resume_at:
        fail(f"{path}: iteration_limit {rules[0].get('limit')!r} is not above {resume_at}")
    return rules[0]


path, config = load(partial_case)
iteration_limit_rule(path, config)["limit"] = resume_at
dump(path, config)

path, config = load(resume_case)
iteration_limit_rule(path, config)
policy = config.setdefault("policy", {})
if "mode" in policy:
    fail(f"{path}: policy.mode already set")
policy["mode"] = "resume"
dump(path, config)
PYEOF
}

# ---------------------------------------------------------------------------
# seed_policy <policy_dir> <output_dir>
#
# Copies <policy_dir> into <output_dir>/policy, where a simulation-only run
# reads its policy.
# ---------------------------------------------------------------------------
seed_policy() {
    local policy_dir="$1"
    local output_dir="$2"
    mkdir -p "${output_dir}/policy"
    cp -R "${policy_dir}/." "${output_dir}/policy/"
}

# ---------------------------------------------------------------------------
# check_outputs <test_name> <exit_code> <output_dir> <baseline_dir> <marker> <seeded_policy>
#               [comparator]
#
# Verifies:
#   1. Exit code is 0 (or reports timeout on code 124)
#   2. Output directory exists
#   3. policy/manifest.bin exists (the self-describing checkpoint commit signal)
#   4. <marker> exists (training/convergence.parquet for a training run,
#      simulation/_SUCCESS for a simulation-only run)
#   5. When <baseline_dir> is non-empty, <comparator> (run by run_comparator,
#      default compare_outputs_bitwise) accepts the output against it
#   6. When <baseline_dir> and <seeded_policy> are non-empty, the per-scenario
#      solver counters equal the baseline's
#
# Returns 0 on full pass, 1 on any failure. Prints diagnostics on failure.
# ---------------------------------------------------------------------------
check_outputs() {
    local test_name="$1"
    local exit_code="$2"
    local output_dir="$3"
    local baseline_dir="$4"
    local marker="$5"
    local seeded_policy="$6"
    local comparator="${7:-compare_outputs_bitwise}"

    if [[ "${exit_code}" -eq 124 ]]; then
        echo "[${test_name}] FAIL: timed out after ${TIMEOUT}s (exit 124)"
        echo "[${test_name}] NOTE: exit code 124 is the primary indicator of the MPICH 4.3.0 PMI2 deadlock"
        return 1
    fi

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${test_name}] FAIL: command exited with code ${exit_code}"
        return 1
    fi

    if [[ ! -d "${output_dir}" ]]; then
        echo "[${test_name}] FAIL: output directory does not exist: ${output_dir}"
        return 1
    fi

    local policy_manifest="${output_dir}/policy/manifest.bin"
    if [[ ! -f "${policy_manifest}" ]]; then
        echo "[${test_name}] FAIL: policy/manifest.bin missing"
        echo "[${test_name}] Contents of ${output_dir}:"
        ls -la "${output_dir}" 2>&1 || true
        return 1
    fi

    local marker_path="${output_dir}/${marker}"
    if [[ ! -f "${marker_path}" ]]; then
        local marker_dir
        marker_dir="$(dirname "${marker_path}")"
        echo "[${test_name}] FAIL: ${marker} missing"
        echo "[${test_name}] Contents of ${marker_dir}:"
        ls -la "${marker_dir}" 2>&1 || true
        return 1
    fi

    if [[ -n "${baseline_dir}" ]]; then
        echo "[${test_name}] Comparing outputs against ${baseline_dir} (${comparator}) ..."
        if ! run_comparator "${comparator}" "${baseline_dir}" "${output_dir}"; then
            echo "[${test_name}] FAIL: output comparison failed (see above)"
            return 1
        fi
        if [[ -n "${seeded_policy}" ]]; then
            echo "[${test_name}] Comparing per-scenario solver counters against ${baseline_dir} ..."
            if ! compare_solver_counters "${baseline_dir}" "${output_dir}"; then
                echo "[${test_name}] FAIL: solver counter comparison failed (see above)"
                return 1
            fi
        fi
    fi

    return 0
}

# ---------------------------------------------------------------------------
# run_mpiexec_test <test_name> <case_dir> <baseline_tag> <marker> <policy_src>
#                  <comparator> <n_ranks> <extra_args...>
#
# Seeds $WORK_DIR/<test_name_lower> with <policy_src> when it is non-empty,
# then runs: timeout $TIMEOUT mpiexec -n <n_ranks> $NOVOMODELO_BIN run <case_dir>
#              --output $WORK_DIR/<test_name_lower> --quiet [extra_args]
# Calls check_outputs with <comparator> against $WORK_DIR/<baseline_tag>
# (none when the tag is empty or names the test itself) and updates RESULTS.
# ---------------------------------------------------------------------------
run_mpiexec_test() {
    local test_name="$1"
    local case_dir="$2"
    local baseline_tag="$3"
    local marker="$4"
    local policy_src="$5"
    local comparator="$6"
    local n_ranks="$7"
    shift 7
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"

    echo ""
    echo "=== ${test_name}: mpiexec -n ${n_ranks} ==="

    if [[ -n "${policy_src}" ]]; then
        seed_policy "${policy_src}" "${output_dir}"
    fi

    local exit_code=0
    timeout "${TIMEOUT}" mpiexec -n "${n_ranks}" \
        "${NOVOMODELO_BIN}" run "${case_dir}" --output "${output_dir}" --quiet \
        "${extra_args[@]+"${extra_args[@]}"}" || exit_code=$?

    local baseline=""
    if [[ -n "${baseline_tag}" && "${tag}" != "${baseline_tag}" ]]; then
        baseline="${WORK_DIR}/${baseline_tag}"
    fi

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}" \
        "${marker}" "${policy_src}" "${comparator}"; then
        echo "[${test_name}] PASS"
        RESULTS["${test_name}"]="PASS"
    else
        RESULTS["${test_name}"]="FAIL"
        print_summary
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# run_sbatch_test <test_name> <case_dir> <baseline_tag> <marker> <policy_src>
#                 <comparator> <sbatch_N> <sbatch_n> <mpiexec_n> <extra_args...>
#
# Seeds the output directory with <policy_src> when it is non-empty,
# generates a batch script as a heredoc, submits via sbatch --wait, then
# calls check_outputs with <comparator> against $WORK_DIR/<baseline_tag>
# (none when the tag is empty). On failure, prints SLURM .out/.err logs.
# ---------------------------------------------------------------------------
run_sbatch_test() {
    local test_name="$1"
    local case_dir="$2"
    local baseline_tag="$3"
    local marker="$4"
    local policy_src="$5"
    local comparator="$6"
    local sbatch_N="$7"
    local sbatch_n="$8"
    local mpiexec_n="$9"
    shift 9
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"
    local slurm_out="${WORK_DIR}/${tag}.slurm.out"
    local slurm_err="${WORK_DIR}/${tag}.slurm.err"
    local batch_script="${WORK_DIR}/${tag}.sbatch"

    echo ""
    echo "=== ${test_name}: sbatch -N ${sbatch_N} -n ${sbatch_n} (mpiexec -n ${mpiexec_n}) ==="

    if [[ -n "${policy_src}" ]]; then
        seed_policy "${policy_src}" "${output_dir}"
    fi

    # Build the extra args string for insertion into the batch script.
    local extra_str="${extra_args[*]+"${extra_args[*]}"}"

    # Generate the batch script with absolute paths.
    # The heredoc delimiter is unquoted so WORK_DIR, NOVOMODELO_BIN, case_dir,
    # output_dir, and mpiexec_n expand at generation time.
    cat > "${batch_script}" << SBEOF
#!/bin/bash
export PATH=/opt/mpich/bin:\$PATH
mpiexec -n ${mpiexec_n} ${NOVOMODELO_BIN} run ${case_dir} --output ${output_dir} --quiet ${extra_str}
SBEOF

    chmod +x "${batch_script}"

    local exit_code=0
    timeout "${TIMEOUT}" sbatch --wait \
        -N "${sbatch_N}" \
        -n "${sbatch_n}" \
        --output="${slurm_out}" \
        --error="${slurm_err}" \
        "${batch_script}" || exit_code=$?

    local baseline=""
    if [[ -n "${baseline_tag}" ]]; then
        baseline="${WORK_DIR}/${baseline_tag}"
    fi

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}" \
        "${marker}" "${policy_src}" "${comparator}"; then
        echo "[${test_name}] PASS"
        RESULTS["${test_name}"]="PASS"
    else
        RESULTS["${test_name}"]="FAIL"
        # Print SLURM logs for debugging before exiting.
        if [[ -f "${slurm_out}" ]]; then
            echo ""
            echo "--- SLURM stdout (${slurm_out}) ---"
            cat "${slurm_out}"
        fi
        if [[ -f "${slurm_err}" ]]; then
            echo ""
            echo "--- SLURM stderr (${slurm_err}) ---"
            cat "${slurm_err}"
        fi
        # For T4 multi-node failures, also print any Hydra proxy logs that
        # may have been captured in the SLURM output or work directory.
        if [[ "${test_name}" == "T4" ]]; then
            echo ""
            echo "--- T4 diagnostic: listing ${output_dir} (if it exists) ---"
            ls -la "${output_dir}" 2>&1 || echo "(directory missing)"
            echo "--- T4 diagnostic: Hydra proxy logs (if any in ${WORK_DIR}) ---"
            find "${WORK_DIR}" -name "hydra_pmi_proxy*" -o -name "*.hydra" 2>/dev/null \
                | while read -r f; do
                    echo "  ${f}:"
                    cat "${f}" 2>/dev/null || true
                done || true
        fi
        print_summary
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# print_summary
#
# Prints a pass/fail table of the tests attempted for the active MODE
# (T1, T2, L1, L2, R0, R1, S1 and S2 for "reduced", every T, L, R and S case
# for "full") so a "reduced" run is not marked FAILED merely for never
# attempting the sbatch cases.
# ---------------------------------------------------------------------------
print_summary() {
    local tests_to_check=("T1" "T2")
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=("T3" "T4" "T5")
    fi
    tests_to_check+=("L1" "L2")
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=("L3")
    fi
    tests_to_check+=("R0" "R1")
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=("R2" "R3")
    fi
    tests_to_check+=("S1" "S2")
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=("S3" "S4")
    fi

    echo ""
    echo "=============================="
    echo " SLURM MPI Test Suite Summary (mode: ${MODE})"
    echo "=============================="
    local all_pass=1
    for t in "${tests_to_check[@]}"; do
        local status="${RESULTS[${t}]:-NOT RUN}"
        printf "  %-4s  %s\n" "${t}" "${status}"
        if [[ "${status}" != "PASS" ]]; then
            all_pass=0
        fi
    done
    echo "=============================="
    if [[ "${all_pass}" -eq 1 ]]; then
        echo " Result: ALL PASS"
    else
        echo " Result: FAILED"
    fi
    echo "=============================="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_mode "$@"

    preflight_check
    run_self_check

    # Case directory, baseline tag, completion marker, policy to seed, comparator.
    local training_run=(
        "${CASE_DIR}" t1 training/convergence.parquet "" compare_outputs_bitwise
    )
    local selection_run=(
        "${WORK_DIR}/4ree-selection" l1 training/convergence.parquet "" compare_outputs_bitwise
    )
    local partial_run=(
        "${WORK_DIR}/4ree-selection-partial" "" training/convergence.parquet "" ""
    )
    local resume_case="${WORK_DIR}/4ree-selection-resume"
    local resumed_comparator="compare_resumed_outputs ${RESUME_AT}"
    local simulation_run=(
        "${WORK_DIR}/4ree-partial-misfit" s1 simulation/_SUCCESS "${WORK_DIR}/t1/policy"
        compare_outputs_bitwise
    )

    # T1: mpiexec single rank (baseline — no extra args)
    run_mpiexec_test "T1" "${training_run[@]}" 1

    # T2: mpiexec multi-rank single machine
    run_mpiexec_test "T2" "${training_run[@]}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # T3: sbatch single-node, 2 ranks
        run_sbatch_test "T3" "${training_run[@]}" 1 2 2

        # T4: sbatch multi-node, 1 rank per node (critical PMI2 deadlock test)
        run_sbatch_test "T4" "${training_run[@]}" 2 2 2

        # T5: sbatch multi-node, 2 ranks per node
        run_sbatch_test "T5" "${training_run[@]}" 2 4 4 --threads 1
    fi

    prepare_selection_case "${CASE_DIR}" "${selection_run[0]}"

    # L1: mpiexec single rank (selection-on baseline), then its power check
    run_mpiexec_test "L1" "${selection_run[@]}" 1
    if ! assert_selection_power "${WORK_DIR}/l1"; then
        echo "[L1] FAIL: the selection-on run has no selection power (see above)"
        RESULTS["L1"]="FAIL"
        print_summary
        exit 1
    fi

    # L2: mpiexec multi-rank single machine
    run_mpiexec_test "L2" "${selection_run[@]}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # L3: sbatch multi-node, 2 ranks per node
        run_sbatch_test "L3" "${selection_run[@]}" 2 4 4 --threads 1
    fi

    prepare_resume_cases "${selection_run[0]}" "${partial_run[0]}" "${resume_case}"

    # R0: mpiexec single rank, stopped at RESUME_AT (no comparison)
    run_mpiexec_test "R0" "${partial_run[@]}" 1

    # R1: resumes R0's checkpoint on two ranks, compared against L1
    run_mpiexec_test "R1" "${resume_case}" l1 training/convergence.parquet \
        "${WORK_DIR}/r0/policy" "${resumed_comparator}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # R2: sbatch -N 2 -n 2, mpiexec -n 2 in that allocation (placement of the two
        # ranks on separate nodes is not requested), stopped at RESUME_AT (no comparison)
        run_sbatch_test "R2" "${partial_run[@]}" 2 2 2

        # R3: resumes R2's two-rank checkpoint on one rank, compared against L1
        run_mpiexec_test "R3" "${resume_case}" l1 training/convergence.parquet \
            "${WORK_DIR}/r2/policy" "${resumed_comparator}" 1 --threads 1
    fi

    prepare_partial_misfit_case "${CASE_DIR}" "${simulation_run[0]}"

    # S1: mpiexec single rank, single thread (simulation-only baseline)
    run_mpiexec_test "S1" "${simulation_run[@]}" 1 --threads 1

    # S2: mpiexec multi-rank single machine
    run_mpiexec_test "S2" "${simulation_run[@]}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # S3: sbatch multi-node, 1 rank per node
        run_sbatch_test "S3" "${simulation_run[@]}" 2 2 2

        # S4: sbatch multi-node, 2 ranks per node
        run_sbatch_test "S4" "${simulation_run[@]}" 2 4 4 --threads 1
    fi

    print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
