#!/usr/bin/env bash
#
# comment_scan.sh — shared helpers for the comment/doc scanning gates.
#
# SOURCE this file; do not execute it. It is the single source of truth for the
# cfg(test) tail-block boundary — the line where a .rs file transitions from
# production to test scope. The boundary matches BOTH the bare `#[cfg(test)]`
# form and the `#[cfg(all(test, ...))]` form novomodelo uses to feature-gate test
# modules, and tolerates leading indentation.
#
# Two gates cannot use these streaming helpers and keep their own boundary copy:
#   * check-comment-banners.sh tracks `extern "C"` suppression state line-by-line.
#   * check-allow-rationale.sh reads the whole file into an array for symbol
#     resolution.
# Their boundary regex MUST stay identical to the one below — there is a
# meta-check at the bottom of this file that fails loudly if they drift.

# cs_emit_production_lines <file>
#   Emit "<file>:<lineno>:<line>" for every line BEFORE the first cfg(test)
#   boundary. This awk holds the one canonical boundary regex.
cs_emit_production_lines() {
    awk -v f="$1" '
        /^[[:space:]]*#\[cfg\((all\()?test[,)]/ { exit }
        { printf "%s:%d:%s\n", f, NR, $0 }
    ' "$1"
}

# cs_emit_comment_lines <file>
#   As cs_emit_production_lines, restricted to lines carrying a `//` marker in
#   their CONTENT (covers `//`, `///`, `//!`). Anchored past the `<file>:<n>:`
#   prefix so the location prefix itself never counts as a match.
cs_emit_comment_lines() {
    cs_emit_production_lines "$1" | grep -E '^[^:]*:[0-9]+:.*//' || true
}

# --- Boundary-drift guard --------------------------------------------------
# When EXECUTED directly (`bash scripts/ci/lib/comment_scan.sh`), assert that the
# two holdout gates still carry the canonical cfg(test) boundary, so an edit
# here cannot silently desync them. When SOURCED, this block is skipped, so the
# gates pay no runtime cost. Wire `bash scripts/ci/lib/comment_scan.sh` into CI to
# enforce the invariant.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    _lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    readonly _CANONICAL_BOUNDARY='#\[cfg\((all\()?test[,)]'
    readonly _HOLDOUTS=(
        "${_lib_dir}/../check-comment-banners.sh"
        "${_lib_dir}/../check-allow-rationale.sh"
        "${_lib_dir}/../quality-report.sh"
    )
    _drift=0
    for _h in "${_HOLDOUTS[@]}"; do
        if ! grep -qF "$_CANONICAL_BOUNDARY" "$_h"; then
            echo "DRIFT: $(basename "$_h") no longer carries the canonical" >&2
            echo "       cfg(test) boundary '${_CANONICAL_BOUNDARY}'." >&2
            _drift=1
        fi
    done
    if [[ "$_drift" -ne 0 ]]; then
        echo "Re-sync the holdout gate(s) with scripts/ci/lib/comment_scan.sh." >&2
        exit 1
    fi
    echo "OK: cfg(test) boundary in sync across comment_scan.sh and holdout gates."
    exit 0
fi
