#!/usr/bin/env bash
# Lock structural invariants of a fresh `novomodelo init` -> `run` against the live
# binary, so the CLI's on-disk output shape cannot silently drift.
#
# What it asserts (bump constants when CLI output shape intentionally changes):
#   1. `novomodelo init --template 1dtoy` materializes exactly EXPECTED_INPUT_FILES
#      regular files. Single source of truth below.
#   2. training/metadata.json EXISTS and carries every TRAINING_METADATA_KEYS
#      top-level key. Routed here per the file each key actually lives in:
#      warm_start_* are NOT here.
#   3. policy/manifest.bin EXISTS (the self-describing checkpoint's commit
#      signal; its binary FlatBuffers content — graph, warm-start counts,
#      provenance — is pinned by the Rust manifest conformance tests, not here).
#   4. simulation/metadata.json EXISTS and carries every SIMULATION_METADATA_KEYS
#      top-level key.
#
# This gate is scoped to structural invariants the assert_cmd integration suite
# does not already pin exactly (exact input-file count, exact top-level key SETS
# rather than individual key/value spot-checks); it deliberately does not
# re-cover command-execution behavior.
#
# Usage:
#   scripts/ci/check-docs-examples.sh           — assumes ./target/release/novomodelo is built
#   scripts/ci/check-docs-examples.sh --build   — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Single source of truth for the expected init file count.
# A future template addition must update this constant, or this gate fails.
readonly EXPECTED_INPUT_FILES=11
readonly TEMPLATE="1dtoy"

# Expected top-level keys of training/metadata.json. Per
# the routing decision, warm_start_* belong to the policy checkpoint's
# manifest.bin, NOT here.
readonly TRAINING_METADATA_KEYS=(
  software
  software_version
  hostname
  solver
  solver_version
  started_at
  completed_at
  duration_seconds
  status
  configuration
  problem_dimensions
  iterations
  convergence
  row_pool
  bounds
  solve_stats
  distribution
)

# Expected top-level keys of simulation/metadata.json. Source of truth is the
# SimulationMetadata struct's serde field names.
readonly SIMULATION_METADATA_KEYS=(
  software
  software_version
  hostname
  solver
  solver_version
  started_at
  completed_at
  duration_seconds
  status
  scenarios
  cost
  solve_stats
  distribution
)

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq is required but not found on PATH." >&2; exit 2; }

BIN="$REPO_ROOT/target/release/novomodelo"
if [[ $BUILD -eq 1 || ! -x "$BIN" ]]; then
  cargo build --release --bin novomodelo
fi

# Fresh mktemp tree: never reuse a pre-existing local output dir, so a stale
# artifact cannot mask a regression with a spurious pass.
TMP_DIR="$(mktemp -d -t novomodelo-docs-examples-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

CASE_DIR="$TMP_DIR/case"
OUT_DIR="$TMP_DIR/output"

fail() {
  echo "ERROR: structural invariant drifted — $1" >&2
  exit 1
}

check_keys() {
  local file="$1" label="$2"
  shift 2
  for key in "$@"; do
    jq -e "has(\"$key\")" "$file" >/dev/null \
      || fail "$label is missing expected top-level key \`$key\`."
  done
}

# Invariant 1: init materializes exactly the expected file count
"$BIN" init --template "$TEMPLATE" "$CASE_DIR" >/dev/null
actual_files="$(find "$CASE_DIR" -type f | wc -l | tr -d '[:space:]')"
if [[ "$actual_files" -ne "$EXPECTED_INPUT_FILES" ]]; then
  fail "\`novomodelo init --template $TEMPLATE\` wrote $actual_files input files, expected $EXPECTED_INPUT_FILES. Update EXPECTED_INPUT_FILES in this script if the template intentionally changed."
fi
echo "init file count: $actual_files == $EXPECTED_INPUT_FILES (expected) ✓"

# Drive a fresh run so the metadata assertions read live output.
"$BIN" run "$CASE_DIR" --output "$OUT_DIR" --quiet --color never >/dev/null

# Invariant 2: training/metadata.json exists + expected top-level keys
training_meta="$OUT_DIR/training/metadata.json"
[[ -f "$training_meta" ]] || fail "training/metadata.json was not written."
check_keys "$training_meta" "training/metadata.json" "${TRAINING_METADATA_KEYS[@]}"
echo "training/metadata.json: ${#TRAINING_METADATA_KEYS[@]} expected top-level keys present (incl. row_pool) ✓"

# Invariant 3: policy/manifest.bin exists (the checkpoint commit signal)
policy_manifest="$OUT_DIR/policy/manifest.bin"
[[ -f "$policy_manifest" ]] || fail "policy/manifest.bin was not written."
echo "policy/manifest.bin: present (self-describing checkpoint commit signal) ✓"

# Invariant 4: simulation/metadata.json exists + expected top-level keys
simulation_meta="$OUT_DIR/simulation/metadata.json"
[[ -f "$simulation_meta" ]] || fail "simulation/metadata.json was not written."
check_keys "$simulation_meta" "simulation/metadata.json" "${SIMULATION_METADATA_KEYS[@]}"
echo "simulation/metadata.json: ${#SIMULATION_METADATA_KEYS[@]} expected top-level keys present ✓"

echo "All init/run structural invariants hold. ✓"
