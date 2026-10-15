#!/usr/bin/env bash
#
# run-mpi-launcher-gates.sh — launch the MPI world-size gates under mpiexec.
#
# Each gate trains on the real MPI world and compares the result with the
# single-rank reference. Under plain `cargo test` the world has one rank and the
# comparison is an identity, so these tests prove something only when a launcher
# starts them on several ranks. Every gate in the table runs at 2 and 4 ranks.
#
# Usage: TEST_FEATURES="<cargo features>" [MPIEXEC=<launcher>] run-mpi-launcher-gates.sh
#        run-mpi-launcher-gates.sh --list
#
# The test executables come from
# `cargo test --workspace --features "$TEST_FEATURES" --no-run`, so a preceding
# workspace test step with the same features is reused, not rebuilt. Each launch
# runs from crates/novomodelo-sddp (where `cargo test` runs that package's tests) under
# a 600 s timeout, so a rank left waiting on a panicked peer fails the gate.
#
# Every rank prints its own libtest summary, and a filter that matches nothing or
# an ignored test also exits 0 with `0 passed`, so a launch passes only when its
# log holds exactly one `test result: ok. 1 passed;` line per rank.
#
# Needs cargo, jq, timeout and the launcher (MPIEXEC, default `mpiexec`).
#
# Exit codes:
#   0 — every launch passed (or --list).
#   1 — a build, executable resolution or launch failed (stops at the first).
#   2 — bad invocation: an unknown argument, TEST_FEATURES unset, or a missing
#       tool.

set -euo pipefail

readonly GATES=(
    "mpi_wire selection_on_determinism::level1_world_size_is_invariant_on_a_branching_fan"
    "mpi_wire by_node_k_fan_branching::by_node_k_fan_final_lb_bitwise_invariant_across_world_size"
    "test_mpi_sync_cuts_invariant k_fan_branching_rank_invariance::k_fan_final_lb_bitwise_invariant_across_world_size"
)
readonly RANK_COUNTS=(2 4)
readonly LAUNCH_TIMEOUT_S=600

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT
readonly PACKAGE_DIR="$ROOT/crates/novomodelo-sddp"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

refuse() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ $# -le 1 ]] || refuse "expected at most one argument, got $#"
case "${1:-}" in
    "") ;;
    --list)
        printf '%s\n' "${GATES[@]}"
        exit 0
        ;;
    *) refuse "unknown argument '$1' (usage: TEST_FEATURES=... $0 | $0 --list)" ;;
esac

[[ -n "${TEST_FEATURES:-}" ]] || refuse "TEST_FEATURES is not set (the cargo features of the workspace test build)"
MPIEXEC="${MPIEXEC:-mpiexec}"
for tool in cargo jq timeout "$MPIEXEC"; do
    command -v "$tool" >/dev/null 2>&1 || refuse "'$tool' not found"
done

WORK_DIR="$(mktemp -d -t mpi_launcher_gates.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
readonly BUILD_JSON="$WORK_DIR/build.json"
readonly LOG="$WORK_DIR/launch.log"

echo "Building the workspace test executables (features: $TEST_FEATURES)"
(cd "$ROOT" && cargo test --workspace --features "$TEST_FEATURES" --no-run --message-format=json) >"$BUILD_JSON" \
    || fail "cargo test --no-run failed"

declare -A EXECUTABLE=()
for gate in "${GATES[@]}"; do
    target="${gate%% *}"
    [[ -z "${EXECUTABLE[$target]:-}" ]] || continue
    mapfile -t found < <(jq -r --arg name "$target" \
        'select(.reason == "compiler-artifact" and .target.name == $name and .executable != null) | .executable' \
        "$BUILD_JSON")
    [[ ${#found[@]} -eq 1 ]] \
        || fail "expected one '$target' test executable in the workspace test build, found ${#found[@]}: ${found[*]:-none}"
    EXECUTABLE[$target]="${found[0]}"
done

cd "$PACKAGE_DIR"
for gate in "${GATES[@]}"; do
    target="${gate%% *}"
    test_name="${gate#* }"
    for ranks in "${RANK_COUNTS[@]}"; do
        echo "--- $test_name on $ranks ranks"
        started_us="${EPOCHREALTIME//[!0-9]/}"
        status=0
        timeout --kill-after=30 "$LAUNCH_TIMEOUT_S" \
            "$MPIEXEC" -n "$ranks" "${EXECUTABLE[$target]}" --exact "$test_name" --test-threads=1 \
            >"$LOG" 2>&1 || status=$?
        elapsed_us=$((${EPOCHREALTIME//[!0-9]/} - started_us))
        cat "$LOG"
        [[ $status -ne 124 && $status -ne 137 ]] \
            || fail "$test_name timed out on $ranks ranks after ${LAUNCH_TIMEOUT_S} s"
        [[ $status -eq 0 ]] || fail "$test_name failed on $ranks ranks (launcher exit $status)"
        passed="$(grep -c 'test result: ok\. 1 passed;' "$LOG" || true)"
        [[ "$passed" -eq "$ranks" ]] \
            || fail "$test_name: rank count mismatch, $ranks ranks launched but $passed reported '1 passed' (the filter matched nothing or the test is ignored)"
        printf 'OK: %s on %d ranks (%d.%02d s)\n' "$test_name" "$ranks" \
            $((elapsed_us / 1000000)) $((elapsed_us % 1000000 / 10000))
    done
done
