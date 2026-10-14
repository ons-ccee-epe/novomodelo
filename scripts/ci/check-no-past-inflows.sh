#!/usr/bin/env bash
#
# check-no-past-inflows.sh — past_inflows data-token regression gate.
#
# The legacy `past_inflows` / `HydroPastInflows` field was retired from every
# deterministic fixture in favor of the windowed `inflow_history` record and
# `recent_observations` conditioning layers. This gate pins that removal: no
# example case or test-data fixture may reintroduce a `past_inflows` data
# token.
#
# Scope: examples/ (all deterministic + other case directories) and
#   crates/*/tests/ (integration-test fixture trees, e.g.
#   crates/novomodelo-sddp/tests/fixtures/). This mirrors the scan scope of
#   `grep -rl past_inflows examples/ crates/*/tests/`.
#
# Deliberately NOT scanned: crates/*/src/ — the loader-rejection regression
# `test_legacy_past_inflows_field_is_rejected`
# (crates/novomodelo-io/src/initial_conditions.rs) legitimately constructs a JSON
# literal carrying the retired field name to assert the loader rejects it;
# this gate must never flag that literal. Excluding crates/*/src/ from the
# scan scope keeps the gate's own PATTERN out of the range it searches, so no
# self-referential-match workaround is needed here (contrast
# check-no-plan-leaks.sh, whose scan scope also excludes its own directory
# for the same reason).
#
# Exit codes:
#   0 — No past_inflows data token found.
#   1 — A past_inflows data token was found (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

readonly PATTERN='past_inflows'

SCAN_DIRS=(
    "${REPO_ROOT}/examples"
)

shopt -s nullglob
for dir in "${REPO_ROOT}"/crates/*/tests; do
    SCAN_DIRS+=("$dir")
done
shopt -u nullglob

readonly SCAN_DIRS

violations=""
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    matches=$(grep -rlI "$PATTERN" "$dir" 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        violations+="${matches}"$'\n'
    fi
done

violations="${violations%$'\n'}"

if [[ -n "$violations" ]]; then
    echo "FAIL: past_inflows data token found in examples/ or a test fixture."
    echo ""
    echo "$violations"
    echo ""
    echo "The past_inflows field is retired; seed historical inflow via"
    echo "windowed inflow_history record rows and/or recent_observations"
    echo "conditioning instead. If this hit is"
    echo "test_legacy_past_inflows_field_is_rejected's own JSON literal, that"
    echo "test lives under crates/*/src/ and is out of this gate's scan"
    echo "scope — check for a stray copy elsewhere."
    exit 1
fi

echo "OK: no past_inflows data token found in examples/ or test fixtures."
exit 0
