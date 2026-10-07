#!/usr/bin/env bash
#
# check-lints-drift.sh — Per-crate lint-table drift gate.
#
# The workspace default forbids unsafe code, so the four crates that need it for
# FFI/PyO3 (novomodelo-solver, novomodelo-comm, novomodelo-sddp, novomodelo-python) cannot use
# `[lints] workspace = true` and instead hand-replicate the full
# `[workspace.lints.*]` tables with `unsafe_code = "allow"`. Those copies can
# silently drift from the workspace tables. This gate asserts each override
# crate's `[lints.rust]`/`[lints.clippy]` matches `[workspace.lints.*]`
# byte-for-byte, allowing exactly one difference — `unsafe_code` ("forbid" in
# the workspace vs "allow" in the crate). Any other delta exits non-zero naming
# the crate and lint. All other crates must use `[lints] workspace = true`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

readonly ROOT_CARGO="${REPO_ROOT}/Cargo.toml"
readonly CRATES=(novomodelo-solver novomodelo-comm novomodelo-sddp novomodelo-python)

build_table() {
    local file="$1"
    local header="$2"
    local prefix="$3"
    awk -v header="[$header]" -v prefix="$prefix" '
        $0 == header { intable = 1; next }
        intable && /^\[/ { intable = 0 }
        intable {
            line = $0
            sub(/#.*/, "", line)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line == "") next
            eq = index(line, "=")
            if (eq == 0) next
            key = substr(line, 1, eq - 1)
            val = substr(line, eq + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
            gsub(/[[:space:]]/, "", val)
            print prefix ":" key "\t" val
        }
    ' "$file"
}

declare -A WORKSPACE=()
while IFS=$'\t' read -r key val; do
    WORKSPACE["$key"]="$val"
done < <(
    build_table "$ROOT_CARGO" "workspace.lints.rust" "rust"
    build_table "$ROOT_CARGO" "workspace.lints.clippy" "clippy"
)

violations=""

for crate in "${CRATES[@]}"; do
    crate_file="${REPO_ROOT}/crates/${crate}/Cargo.toml"

    declare -A CRATE_LINTS=()
    while IFS=$'\t' read -r key val; do
        CRATE_LINTS["$key"]="$val"
    done < <(
        build_table "$crate_file" "lints.rust" "rust"
        build_table "$crate_file" "lints.clippy" "clippy"
    )

    declare -A ALL_KEYS=()
    for key in "${!WORKSPACE[@]}"; do
        ALL_KEYS["$key"]=1
    done
    for key in "${!CRATE_LINTS[@]}"; do
        ALL_KEYS["$key"]=1
    done

    for key in "${!ALL_KEYS[@]}"; do
        ws_val="${WORKSPACE[$key]:-<missing>}"
        crate_val="${CRATE_LINTS[$key]:-<missing>}"

        if [[ "$key" == "rust:unsafe_code" ]]; then
            if [[ "$ws_val" != '"forbid"' || "$crate_val" != '"allow"' ]]; then
                violations+="${crate}: ${key} is ${crate_val} (workspace: ${ws_val}, expected crate override \"allow\")"$'\n'
            fi
            continue
        fi

        if [[ "$ws_val" != "$crate_val" ]]; then
            violations+="${crate}: ${key} is ${crate_val} (workspace: ${ws_val})"$'\n'
        fi
    done
done

violations="${violations%$'\n'}"

if [[ -n "$violations" ]]; then
    echo "FAIL: per-crate [lints] table drifted from [workspace.lints]"
    echo ""
    echo "$violations"
    exit 1
fi

echo "OK: all per-crate lint overrides match the workspace lint tables."
exit 0
