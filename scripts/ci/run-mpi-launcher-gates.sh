#!/usr/bin/env bash
#
# run-mpi-launcher-gates.sh — launch the MPI world-size gates under mpiexec.
#
# Each gate trains on the real MPI world. Most compare the result with the
# single-rank reference; the checkpoint gate makes every rank fail one periodic
# write in-band and checks what rank 0 leaves on disk. Under plain `cargo test`
# the world has one rank, so these tests prove something only when a launcher
# starts them on several ranks. Every gate in the table runs at 2 and 4 ranks.
#
# One probe runs the first gate on a world larger than its forward passes and
# requires the refusal that leaves a rank without a trial: a non-zero exit and
# one refusal line per rank in the log. A passing launch there would mean the
# gate no longer checks its world size.
#
# Usage: TEST_FEATURES="<cargo features>" [MPIEXEC=<launcher>] run-mpi-launcher-gates.sh
#        run-mpi-launcher-gates.sh --list      (the gates, then the probe)
#
# The test executables come from
# `cargo test --workspace --features "$TEST_FEATURES" --no-run`, so a preceding
# workspace test step with the same features is reused, not rebuilt. Each launch
# runs from crates/novomodelo-sddp (where `cargo test` runs that package's tests) under
# a 600 s timeout, so a rank left waiting on a panicked peer fails the gate.
#
# Each rank writes its own log under $TMPDIR, so every rank must see that
# directory: ranks sharing the launcher's output interleave libtest's summary
# line mid-line, so a merged log cannot be counted. A filter that matches nothing
# or an ignored test also exits 0 with `0 passed`, so a launch passes only when
# there is one log per rank and each holds `test result: ok. 1 passed;`.
#
# Needs cargo, jq, timeout and the launcher (MPIEXEC, default `mpiexec`).
#
# Exit codes:
#   0 — every launch passed and the probe was refused (or --list).
#   1 — a build, executable resolution, launch or the probe failed (stops at the
#       first).
#   2 — bad invocation: an unknown argument, TEST_FEATURES unset, or a missing
#       tool.

set -euo pipefail

readonly GATES=(
    "mpi_wire selection_on_determinism::level1_world_size_is_invariant_on_a_branching_fan"
    "mpi_wire by_node_k_fan_branching::by_node_k_fan_final_lb_bitwise_invariant_across_world_size"
    "test_mpi_sync_cuts_invariant k_fan_branching_rank_invariance::k_fan_final_lb_bitwise_invariant_across_world_size"
    "periodic_checkpoint a_failed_periodic_write_on_the_world_leaves_the_last_whole_checkpoint_and_a_partial_beside_it"
)
readonly RANK_COUNTS=(2 4)
readonly OVERSIZED_GATE="${GATES[0]}"
readonly OVERSIZED_RANKS=9
readonly OVERSIZED_REFUSAL="leaves a rank without a trial"
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
        printf '%s on %d ranks must be refused: %s\n' "${OVERSIZED_GATE#* }" "$OVERSIZED_RANKS" "$OVERSIZED_REFUSAL"
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
readonly RANK_LOG_DIR="$WORK_DIR/ranks"
readonly RANK_WRAPPER="$WORK_DIR/rank-wrapper.sh"

# shellcheck disable=SC2016 # expanded by each rank, not here
printf '%s\n' '#!/bin/sh' \
    'rank="${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-unknown}}}"' \
    'exec "$@" >"$(mktemp "$RANK_LOG_DIR/rank.$rank.XXXXXX")" 2>&1' >"$RANK_WRAPPER"
chmod +x "$RANK_WRAPPER"
export RANK_LOG_DIR

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

launch() {
    local target="$1" test_name="$2" ranks="$3"
    echo "--- $test_name on $ranks ranks"
    local started_us="${EPOCHREALTIME//[!0-9]/}"
    status=0
    rm -rf "$RANK_LOG_DIR"
    mkdir "$RANK_LOG_DIR"
    timeout --kill-after=30 "$LAUNCH_TIMEOUT_S" \
        "$MPIEXEC" -n "$ranks" "$RANK_WRAPPER" "${EXECUTABLE[$target]}" --exact "$test_name" --test-threads=1 \
        >"$LOG" 2>&1 || status=$?
    elapsed_us=$((${EPOCHREALTIME//[!0-9]/} - started_us))
    cat "$LOG"
    rank_logs=()
    local rank_log
    for rank_log in "$RANK_LOG_DIR"/rank.*; do
        [[ -e "$rank_log" ]] || continue
        rank_logs+=("$rank_log")
        echo "[${rank_log##*/}]"
        cat "$rank_log"
    done
    [[ $status -ne 124 && $status -ne 137 ]] \
        || fail "$test_name timed out on $ranks ranks after ${LAUNCH_TIMEOUT_S} s"
}

# Prints how many rank logs hold a line matching $1.
count_rank_logs_matching() {
    if [[ ${#rank_logs[@]} -eq 0 ]]; then
        echo 0
        return
    fi
    { grep -l -- "$1" "${rank_logs[@]}" || true; } | wc -l
}

report() {
    printf 'OK: %s on %d ranks (%d.%02d s)\n' "$1" "$2" $((elapsed_us / 1000000)) $((elapsed_us % 1000000 / 10000))
}

cd "$PACKAGE_DIR"
for gate in "${GATES[@]}"; do
    target="${gate%% *}"
    test_name="${gate#* }"
    for ranks in "${RANK_COUNTS[@]}"; do
        launch "$target" "$test_name" "$ranks"
        [[ $status -eq 0 ]] || fail "$test_name failed on $ranks ranks (launcher exit $status)"
        [[ ${#rank_logs[@]} -eq "$ranks" ]] \
            || fail "$test_name: $ranks ranks launched but ${#rank_logs[@]} wrote a log"
        passed="$(count_rank_logs_matching 'test result: ok\. 1 passed;')"
        [[ "$passed" -eq "$ranks" ]] \
            || fail "$test_name: rank count mismatch, $ranks ranks launched but $passed reported '1 passed' (the filter matched nothing or the test is ignored)"
        report "$test_name" "$ranks"
    done
done

launch "${OVERSIZED_GATE%% *}" "${OVERSIZED_GATE#* }" "$OVERSIZED_RANKS"
[[ $status -ne 0 ]] || fail "${OVERSIZED_GATE#* } passed on $OVERSIZED_RANKS ranks, a world larger than its forward passes"
refused="$(count_rank_logs_matching "$OVERSIZED_REFUSAL")"
[[ "$refused" -eq "$OVERSIZED_RANKS" ]] \
    || fail "${OVERSIZED_GATE#* } on $OVERSIZED_RANKS ranks: $refused ranks reported '$OVERSIZED_REFUSAL', expected every rank"
report "${OVERSIZED_GATE#* } refused" "$OVERSIZED_RANKS"
