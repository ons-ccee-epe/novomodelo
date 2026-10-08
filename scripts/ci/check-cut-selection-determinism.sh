#!/usr/bin/env bash
#
# check-cut-selection-determinism.sh — Cut-selection determinism gate.
#
# Scans the workspace for tokens that would break the bit-for-bit determinism
# contract of the cut-selection GEMM call chain. The contract requires:
#
#   1. No external BLAS bindings (multiple BLAS implementations yield
#      slightly different results that break declaration-order invariance).
#   2. No runtime CPU dispatch (the active code path must be deterministic
#      regardless of host CPU features).
#   3. No fast-math reassociation (floating-point associativity violations
#      break bit-for-bit reproducibility across runs).
#   4. matrixmultiply must remain single-threaded (the workspace pins
#      matrixmultiply with default-features = false; enabling its threading
#      feature would non-deterministically interleave dgemm calls).
#
# Scope:
#   - Every Cargo.toml at the workspace root and under crates/**/.
#   - .rs files under crates/novomodelo-sddp/src/ and crates/novomodelo-solver/src/
#     (the two crates that participate in the cut-selection GEMM call
#     chain). Other crates' Rust files are out of scope.
#
# Exclusions:
#   - Files whose path contains target/ (build artefacts).
#   - Files whose basename contains "test" (test code may name forbidden
#     tokens for negative tests).
#   - Files under plans/, docs/ are not in scope by construction (the find
#     predicates only walk Cargo.toml and the two src/ subtrees).
#   - The crates/novomodelo-solver/examples/ tree is naturally excluded because
#     the .rs scan is scoped to crates/*/src/ only — the audit_mm_dispatch
#     example legitimately references matrixmultiply::dgemm.
#
# Implementation note on grep flavour:
#   This script uses `grep -E` (POSIX ERE) only. PCRE (`grep -P`) is not
#   available on every CI image. Word-boundary anchors (`\b`) ARE supported
#   by GNU grep -E and are used to avoid false positives on identifiers like
#   "fastmath_disabled" (no `\b` boundary on either side of "fast_math").
#
# Verification mode:
#   `bash check-cut-selection-determinism.sh --verify` writes a fixture
#   tree containing every forbidden pattern, runs the scan against it, and
#   asserts each pattern was reported at least once. The fixture is always
#   removed via a trap handler.
#
# Exit codes:
#   0 — Pass: no forbidden patterns found (or --verify succeeded).
#   1 — Fail: forbidden pattern detected (or --verify missed a pattern).
#   2 — Script error (bad invocation, missing prerequisite).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Parallel arrays of forbidden patterns (POSIX ERE) and human-readable
# messages. Using parallel arrays (rather than a single grouped ERE) keeps
# the FAIL output specific about which determinism rule was violated.
readonly PATTERNS=(
    '\bopenblas-src\b'
    '\bintel-mkl-src\b'
    '\bblas-src\b'
    '\bcblas-sys\b'
    '\bmultiversion\b'
    'is_x86_feature_detected!'
    '\bcpufeatures\b'
    'fast-math'
    'fast_math'
    'ffast-math'
    'matrixmultiply.*threading.*=.*true'
    'matrixmultiply.*features.*=.*\[.*threading'
)

readonly MESSAGES=(
    'External BLAS binding forbidden (design §5.4 item 1)'
    'External BLAS binding forbidden'
    'External BLAS binding forbidden'
    'External BLAS binding forbidden'
    'Runtime CPU dispatch forbidden (design §5.4 item 2)'
    'Runtime CPU dispatch forbidden'
    'Runtime CPU dispatch crate forbidden'
    'Fast-math reassociation forbidden (design §5.4 item 3)'
    'Fast-math attribute forbidden'
    'Fast-math flag forbidden'
    'matrixmultiply threading must remain disabled (design §5.4 item 4)'
    'matrixmultiply threading must remain disabled'
)

# scan_root <root>
#
# Run the full forbidden-pattern scan against the given root directory.
# Populates the global VIOLATION_LINES array with one entry per matched
# (pattern, file, line) tuple. Sets VIOLATION_PATTERNS to a list of the
# pattern indices that produced at least one match.
#
# Outputs to stdout: nothing (caller decides how to render).
# Returns: 0 if no violations, 1 if any pattern matched.
scan_root() {
    local root="$1"

    VIOLATION_LINES=()
    VIOLATION_PATTERNS=()

    local -a cargo_files=()
    local -a rs_files=()
    local file
    local basename

    # Enumerate Cargo.toml files at the workspace root and under every
    # crate root, skipping target/ build artefacts.
    while IFS= read -r -d '' file; do
        cargo_files+=("$file")
    done < <(find "$root" -name 'Cargo.toml' -not -path '*/target/*' -print0)

    # Enumerate .rs files under the two in-scope src/ subtrees, skipping
    # any file whose basename contains "test". Skip the search entirely
    # when the subtree does not exist (relevant for the --verify fixture
    # which only mocks crates/x/src/).
    local rs_dir
    for rs_dir in "$root/crates/novomodelo-sddp/src" "$root/crates/novomodelo-solver/src" "$root/crates/x/src"; do
        if [[ -d "$rs_dir" ]]; then
            while IFS= read -r -d '' file; do
                basename="${file##*/}"
                if [[ "$basename" == *test* ]]; then
                    continue
                fi
                rs_files+=("$file")
            done < <(find "$rs_dir" -name '*.rs' -print0)
        fi
    done

    # Apply every (pattern, message) pair to every in-scope file.
    local i
    local pat
    local msg
    local matches
    local f
    for i in "${!PATTERNS[@]}"; do
        pat="${PATTERNS[$i]}"
        msg="${MESSAGES[$i]}"

        local pattern_matched=0

        for f in "${cargo_files[@]}" "${rs_files[@]}"; do
            matches=$(grep -nE "$pat" "$f" 2>/dev/null) || true
            if [[ -n "$matches" ]]; then
                pattern_matched=1
                # Prepend file path and append the rule message to every
                # matched line so the failure report is self-explanatory.
                local line
                while IFS= read -r line; do
                    VIOLATION_LINES+=("${f}:${line}   [${msg}]")
                done <<<"$matches"
            fi
        done

        if [[ $pattern_matched -eq 1 ]]; then
            VIOLATION_PATTERNS+=("$i")
        fi
    done

    if [[ ${#VIOLATION_LINES[@]} -gt 0 ]]; then
        return 1
    fi
    return 0
}

# render_violations
#
# Print the accumulated VIOLATION_LINES with a FAIL header and a remediation
# hint. Caller is responsible for setting the exit code.
render_violations() {
    echo "FAIL: cut-selection determinism gate found forbidden patterns."
    echo ""
    local line
    for line in "${VIOLATION_LINES[@]}"; do
        echo "$line"
    done
    echo ""
    echo "These patterns would break the bit-for-bit determinism contract"
    echo "of the cut-selection GEMM call chain. Remove the offending"
    echo "dependency, attribute, or feature flag before committing."
}

# run_verify
#
# Self-test mode. Construct a fixture tree containing every forbidden
# pattern across both Cargo.toml and crates/x/src/lib.rs, then assert
# scan_root reports every pattern.
run_verify() {
    local fixture
    fixture="$(mktemp -d -t cut_sel_gate_fixture.XXXXXX)"

    # Always remove the fixture, even on failure.
    # shellcheck disable=SC2064
    trap "rm -rf '${fixture}'" EXIT

    mkdir -p "${fixture}/crates/x/src"

    # Cargo.toml carries the dependency-style patterns. The matrixmultiply
    # threading variants must live in a Cargo.toml because their ERE only
    # makes sense in TOML feature syntax.
    cat >"${fixture}/Cargo.toml" <<'TOML'
[package]
name = "fixture"
version = "0.0.0"

[dependencies]
openblas-src = "0.10"
intel-mkl-src = "0.8"
blas-src = "0.10"
cblas-sys = "0.1"
multiversion = "0.7"
cpufeatures = "0.2"
matrixmultiply = { version = "0.3", features = ["threading"] }
matrixmultiply-alt = { version = "0.3", threading = true }
TOML

    # lib.rs carries the code-style patterns. The fixture is placed under
    # crates/x/src/ which scan_root scans (alongside novomodelo-sddp and
    # novomodelo-solver) when present.
    cat >"${fixture}/crates/x/src/lib.rs" <<'RUST'
// Synthetic fixture exercising every forbidden code-style pattern.
fn check_avx() -> bool {
    is_x86_feature_detected!("avx2")
}

#[fast_math]
fn fast_math_attribute_marker() {}

// fast-math
// ffast-math
RUST

    if scan_root "$fixture"; then
        echo "FAIL: --verify scan reported zero matches against the fixture."
        echo "      Expected every forbidden pattern to be detected."
        return 1
    fi

    # Check that every pattern index appears in VIOLATION_PATTERNS.
    local missed=()
    local i
    local found
    local p
    for i in "${!PATTERNS[@]}"; do
        found=0
        for p in "${VIOLATION_PATTERNS[@]}"; do
            if [[ "$p" == "$i" ]]; then
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            missed+=("${PATTERNS[$i]}  (${MESSAGES[$i]})")
        fi
    done

    if [[ ${#missed[@]} -gt 0 ]]; then
        echo "FAIL: --verify missed the following forbidden patterns:"
        local m
        for m in "${missed[@]}"; do
            echo "  - $m"
        done
        return 1
    fi

    echo "OK: --verify passed (${#PATTERNS[@]}/${#PATTERNS[@]} patterns detected)."
    return 0
}

# Entry point.
case "${1:-}" in
    --verify)
        if run_verify; then
            exit 0
        else
            exit 1
        fi
        ;;
    '')
        if scan_root "$REPO_ROOT"; then
            echo "OK: no cut-selection determinism violations found."
            exit 0
        else
            render_violations
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [--verify]" >&2
        echo "" >&2
        echo "  (no args)  Scan the workspace for forbidden patterns." >&2
        echo "  --verify   Run the self-test against a synthetic fixture." >&2
        exit 2
        ;;
esac
