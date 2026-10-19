#!/usr/bin/env bash
# Smoke-test scripts/measure_cut_selection.py end to end on examples/1dtoy, so
# the tool cannot rot unseen.
#
# What it asserts:
#   1. The script runs its five variants (all-states, two windows, simulate-all,
#      simulate-active) and writes a CSV with exactly those rows, in that order.
#   2. Every wall time, peak memory, mean and standard deviation of the simulated
#      cost parses as a finite number; the lower bound is finite on the three
#      training rows and empty on the two simulation-only rows; peak memory is
#      greater than zero.
#   3. The source case is left untouched (`git status` of examples/ is empty).
#
# It pins no measured value, only the shape and finiteness of the report.
#
# Usage:
#   scripts/ci/check-cut-selection-measurement.sh           — assumes ./target/release/novomodelo is built
#   scripts/ci/check-cut-selection-measurement.sh --build   — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

BIN="$REPO_ROOT/target/release/novomodelo"
if [[ $BUILD -eq 1 || ! -x "$BIN" ]]; then
  cargo build --release --bin novomodelo
fi

TMP_DIR="$(mktemp -d -t novomodelo-cut-selection-measurement-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "ERROR: cut-selection measurement smoke — $1" >&2
  exit 1
}

REPORT="$TMP_DIR/report.csv"
python3 scripts/measure_cut_selection.py \
  --case examples/1dtoy \
  --work-dir "$TMP_DIR/work" \
  --check-frequency 2 --windows 2,4 --iterations 6 --scenarios 10 \
  --novomodelo "$BIN" \
  --report "$REPORT" \
  || fail "the measurement script failed."

python3 - "$REPORT" <<'PY' || fail "the report is malformed."
import csv
import math
import sys

EXPECTED = ["all-states", "window-2", "window-4", "simulate-all", "simulate-active"]
TRAINING = EXPECTED[:3]

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

variants = [row["variant"] for row in rows]
if variants != EXPECTED:
    sys.exit(f"variants are {variants}, expected {EXPECTED}")


def finite(row, column):
    try:
        value = float(row[column])
    except ValueError:
        sys.exit(f"{row['variant']}: {column} is {row[column]!r}, not a number")
    if not math.isfinite(value):
        sys.exit(f"{row['variant']}: {column} is not finite")
    return value


for row in rows:
    for column in ("mean_simulated_cost", "std_simulated_cost", "wall_seconds"):
        finite(row, column)
    if finite(row, "peak_rss_mib") <= 0:
        sys.exit(f"{row['variant']}: peak_rss_mib is not positive")
    if row["variant"] in TRAINING:
        finite(row, "lower_bound")
    elif row["lower_bound"] != "":
        sys.exit(f"{row['variant']}: lower_bound is {row['lower_bound']!r}, expected empty")
PY

touched="$(git status --porcelain -- examples)"
[[ -z "$touched" ]] || fail "the source case was modified:
$touched"

echo "OK: measurement smoke (5 variants)"
