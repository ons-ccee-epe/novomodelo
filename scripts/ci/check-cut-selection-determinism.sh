#!/usr/bin/env bash
#
# check-cut-selection-determinism.sh — text guard for the cut-selection dgemm
# kernel.
#
# Four checks own the facts that keep the kernel deterministic:
#
#   - deny.toml (cargo-deny, run on the workspace and on
#     crates/novomodelo-python/Cargo.toml) owns the resolved dependency graph: no
#     external BLAS or runtime-dispatch crate, and no matrixmultiply feature
#     enabled by any crate.
#   - The witness test gemm::tests::resolved_kernel_is_the_unfused_baseline
#     owns the kernel a test build resolved.
#   - The release probe owns the kernel of the built artifacts.
#   - This script owns the text facts:
#       1. no runtime CPU detection (is_x86_feature_detected!) in the .rs
#          files of crates/novomodelo-sddp/src and crates/novomodelo-solver/src;
#       2. no fast-math spelling in those .rs files or in a build input;
#       3. no target-feature or target-cpu flag in a build input, and no
#          -march=, -mtune=, -mfma or -mavx C-compiler flag in a build.rs.
#
# Build inputs: every Cargo.toml, .cargo/config, .cargo/config.toml, build.rs
# and .github/workflows/*.yml, plus dist-workspace.toml and
# crates/novomodelo-python/pyproject.toml. Directories named target, vendor or .git
# are pruned. Rule 3 never reads a .rs file other than build.rs, because the
# witness test names target features on purpose.
#
# `--verify` writes a fixture with one hit per pattern in each file kind it
# applies to, plus lines that must not be reported (a workflow
# `RUSTFLAGS: -Dwarnings`, a target feature in a non-build .rs file, hits under
# target/ and vendor/), and fails if a pattern or a fixture file is missed or a
# negative is reported. The fixture lives under $TMPDIR and is always removed.
#
# The script uses `grep -E` (POSIX ERE) only and calls no cargo, so it runs on
# an image without a Rust toolchain.
#
# Exit codes:
#   0 — Pass: no violation found (or --verify succeeded).
#   1 — Fail: violation found (or --verify missed a pattern or reported a
#       negative).
#   2 — Script error (bad invocation).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Parallel arrays: pattern (POSIX ERE), message, and the file sets it applies
# to (space-separated names among: code, build, build_rs).
readonly PATTERNS=(
    'is_x86_feature_detected!'
    'fast-math'
    'fast_math'
    'ffast-math'
    'target[-_]feature'
    'target[-_]cpu'
    '-march='
    '-mtune='
    '-mfma'
    '-mavx'
)

readonly MESSAGES=(
    'Runtime CPU detection forbidden'
    'Fast-math reassociation forbidden'
    'Fast-math attribute forbidden'
    'Fast-math flag forbidden'
    'Target feature flag forbidden in a build input'
    'Target CPU flag forbidden in a build input'
    'C-compiler architecture flag forbidden in a build script'
    'C-compiler tuning flag forbidden in a build script'
    'C-compiler FMA flag forbidden in a build script'
    'C-compiler AVX flag forbidden in a build script'
)

readonly SCOPES=(
    'code'
    'code build'
    'code build'
    'code build'
    'build'
    'build'
    'build_rs'
    'build_rs'
    'build_rs'
    'build_rs'
)

# find_files <root> <find predicate...>
#
# Print (NUL-separated, absolute) every regular file under <root> that matches
# the predicate, pruning directories named target, vendor or .git. The walk is
# relative to <root> so a root that itself sits under a target/ directory (the
# --verify fixture under $TMPDIR) is still scanned.
find_files() {
    local root="$1"
    shift
    local rel
    while IFS= read -r -d '' rel; do
        printf '%s\0' "${root}/${rel#./}"
    done < <(cd "$root" && find . \( -name target -o -name vendor -o -name .git \) -type d -prune \
        -o -type f \( "$@" \) -print0)
}

# scan_root <root>
#
# Apply every pattern to the files of its scopes under <root>. Populates
# VIOLATION_LINES (one `file:line: text   [message]` entry per match) and
# VIOLATION_PATTERNS (the indices of the patterns that matched).
# Returns 0 if nothing matched, 1 otherwise.
scan_root() {
    local root="$1"

    VIOLATION_LINES=()
    VIOLATION_PATTERNS=()

    local -a code_files=()
    local -a build_files=()
    local -a build_rs_files=()
    local file

    local rs_dir
    for rs_dir in crates/novomodelo-sddp/src crates/novomodelo-solver/src; do
        if [[ -d "$root/$rs_dir" ]]; then
            while IFS= read -r -d '' file; do
                code_files+=("$file")
            done < <(find_files "$root/$rs_dir" -name '*.rs')
        fi
    done

    while IFS= read -r -d '' file; do
        build_files+=("$file")
    done < <(find_files "$root" \
        -name Cargo.toml -o -name build.rs \
        -o -path '*/.cargo/config' -o -path '*/.cargo/config.toml' \
        -o -path '*/.github/workflows/*.yml' \
        -o -path ./dist-workspace.toml -o -path ./crates/novomodelo-python/pyproject.toml)

    while IFS= read -r -d '' file; do
        build_rs_files+=("$file")
    done < <(find_files "$root" -name build.rs)

    local i pat msg scope f matches line
    local -a scopes files
    for i in "${!PATTERNS[@]}"; do
        pat="${PATTERNS[$i]}"
        msg="${MESSAGES[$i]}"
        read -ra scopes <<<"${SCOPES[$i]}"
        files=()
        for scope in "${scopes[@]}"; do
            case "$scope" in
                code) files+=("${code_files[@]}") ;;
                build) files+=("${build_files[@]}") ;;
                build_rs) files+=("${build_rs_files[@]}") ;;
            esac
        done

        local pattern_matched=0
        for f in "${files[@]}"; do
            matches=$(grep -nE -e "$pat" -- "$f" 2>/dev/null) || true
            if [[ -n "$matches" ]]; then
                pattern_matched=1
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

render_violations() {
    echo "FAIL: cut-selection determinism gate found forbidden patterns."
    echo ""
    local line
    for line in "${VIOLATION_LINES[@]}"; do
        echo "$line"
    done
    echo ""
    echo "These patterns would change the kernel the cut-selection determinism"
    echo "tests gate. Remove the offending call, attribute or flag before"
    echo "committing."
}

# write_fixture <dir>
#
# Lay out one file per kind. Every line holding a pattern is a positive; the
# files under target/ and vendor/, crates/novomodelo-sddp/src/simd.rs (a target
# feature in a non-build .rs file) and the `RUSTFLAGS: -Dwarnings` line are
# negatives.
write_fixture() {
    local d="$1"
    mkdir -p "$d/crates/novomodelo-sddp/src" "$d/crates/novomodelo-solver/src" "$d/crates/novomodelo-python" \
        "$d/.cargo" "$d/crates/novomodelo-cli/.cargo" "$d/.github/workflows" \
        "$d/target/release/build/x" "$d/crates/novomodelo-solver/vendor/HiGHS"

    cat >"$d/crates/novomodelo-sddp/src/lib.rs" <<'RUST'
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}
RUST
    cat >"$d/crates/novomodelo-solver/src/kernel_tests.rs" <<'RUST'
#[fast_math]
fn marker() {}
RUST
    cat >"$d/crates/novomodelo-sddp/src/simd.rs" <<'RUST'
const FUSED: bool = cfg!(target_feature = "fma"); // target-feature, target-cpu
RUST
    cat >"$d/crates/novomodelo-solver/build.rs" <<'RUST'
fn main() {
    cc::Build::new().flag("-march=native").flag("-mtune=native").flag("-mfma").flag("-mavx2");
}
RUST
    cat >"$d/Cargo.toml" <<'TOML'
[profile.release]
# fast-math
TOML
    cat >"$d/crates/novomodelo-python/pyproject.toml" <<'TOML'
[tool.maturin]
# -ffast-math
TOML
    cat >"$d/.cargo/config.toml" <<'TOML'
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+fma"]
TOML
    cat >"$d/crates/novomodelo-cli/.cargo/config" <<'TOML'
[build]
rustflags = ["-Ctarget_feature=+fma"]
TOML
    cat >"$d/.github/workflows/release.yml" <<'YAML'
env:
  RUSTFLAGS: -Dwarnings
  CARGO_ENCODED_RUSTFLAGS: -Ctarget-cpu=native
YAML
    cat >"$d/dist-workspace.toml" <<'TOML'
[dist]
rustflags = "--codegen target_cpu=haswell"
TOML
    cat >"$d/target/release/build/x/build.rs" <<'RUST'
// is_x86_feature_detected! fast-math target-feature -march=native
RUST
    cat >"$d/crates/novomodelo-solver/vendor/HiGHS/Cargo.toml" <<'TOML'
# fast-math target-cpu
TOML
}

run_verify() {
    local fixture
    fixture="$(mktemp -d -t cut_sel_gate_fixture.XXXXXX)"
    # shellcheck disable=SC2064
    trap "rm -rf '${fixture}'" EXIT

    write_fixture "$fixture"

    if scan_root "$fixture"; then
        echo "FAIL: --verify scan reported zero matches against the fixture."
        return 1
    fi

    local -a failures=()
    local i p found
    for i in "${!PATTERNS[@]}"; do
        found=0
        for p in "${VIOLATION_PATTERNS[@]}"; do
            if [[ "$p" == "$i" ]]; then
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            failures+=("missed pattern ${PATTERNS[$i]}  (${MESSAGES[$i]})")
        fi
    done

    local rel line
    for rel in crates/novomodelo-sddp/src/lib.rs crates/novomodelo-solver/src/kernel_tests.rs \
        crates/novomodelo-solver/build.rs Cargo.toml crates/novomodelo-python/pyproject.toml \
        .cargo/config.toml crates/novomodelo-cli/.cargo/config .github/workflows/release.yml \
        dist-workspace.toml; do
        found=0
        for line in "${VIOLATION_LINES[@]}"; do
            if [[ "$line" == "${fixture}/${rel}:"* ]]; then
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            failures+=("missed file ${rel}")
        fi
    done

    for line in "${VIOLATION_LINES[@]}"; do
        case "$line" in
            *-Dwarnings* | "${fixture}/target/"* | "${fixture}/crates/novomodelo-solver/vendor/"* | \
                "${fixture}/crates/novomodelo-sddp/src/simd.rs:"*)
                failures+=("reported a negative: ${line#"${fixture}/"}")
                ;;
        esac
    done

    if [[ ${#failures[@]} -gt 0 ]]; then
        echo "FAIL: --verify:"
        local m
        for m in "${failures[@]}"; do
            echo "  - $m"
        done
        return 1
    fi

    echo "OK: --verify passed (${#PATTERNS[@]}/${#PATTERNS[@]} patterns detected)."
    return 0
}

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
        echo "  (no args)  Scan the repository for forbidden patterns." >&2
        echo "  --verify   Run the self-test against a synthetic fixture." >&2
        exit 2
        ;;
esac
