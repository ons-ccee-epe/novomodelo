#!/usr/bin/env bash
#
# check-doc-placeholders.sh — E5 placeholder-text gate for shipped docs.
#
# Enforces the documentation-integrity contract (see
# .claude/rules/doc-integrity.md §5 "stale snapshots / executable claims" and
# .claude/rules/comments.md §5): shipped documentation must not contain an
# unfilled placeholder. A placeholder that ships is a promise the reader cannot
# cash — it reads as a fact while being a hole. One source class is scanned:
#
#   1. Production rustdoc — `///` and `//!` doc-comment lines in crates/*/src
#      .rs files, with the cfg(test) tail-block awk pre-filter (borrowed from
#      check-no-plan-leaks.sh / check-comment-refs.sh) dropping everything from
#      the first `#[cfg(test)]` line onward. A placeholder in a test-only
#      comment is out of the shipped-doc scope. Matching is restricted to
#      doc-comment lines (^[[:space:]]*(///|//!)) so a bare placeholder token
#      embedded in an identifier or string literal on a non-doc line is not
#      flagged.
#
#   Known limitation (same as the sibling gates): the cfg(test) exclusion
#   assumes the test module is a tail block. Files with a mid-file test module
#   followed by production code would incorrectly skip that trailing code. In
#   practice novomodelo files follow the tail-block convention.
#
#   (The former book-prose pass — every *.md under book/src/ — was retired with
#   book/ (mdBook decommission); the unified docs site at docs.novomodelo.invalid owns
#   placeholder hygiene for user-facing prose now.)
#
# Flagged patterns:
#   \bTBD\b         — the literal token TBD, word-boundary anchored so a
#                     substring (e.g. `aTBDb`) does not over-match.
#   to be inserted  — the phrase, case-insensitively.
#
# Exclusions (critical): any non-.rs file. The scan roots enforce this.
#
# Reporting: each hit is printed as `FILE:LINE: <matched token span>` — the
#   token span only (grep -noE style), not the whole line.
#
# Exit codes:
#   0 — No placeholder text in shipped docs.
#   1 — One or more placeholders found (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

# Placeholder token patterns. grep -noE extracts the matched span only.
readonly PLACEHOLDER_PATTERN='\bTBD\b|[Tt][Oo] [Bb][Ee] [Ii][Nn][Ss][Ee][Rr][Tt][Ee][Dd]'

# .rs source directories: scanned per-file with the cfg(test) tail-block
# exclusion (see header). Mirrors check-no-plan-leaks.sh SCAN_DIRS.
readonly SCAN_DIRS=(
    "${REPO_ROOT}/crates/novomodelo-core/src"
    "${REPO_ROOT}/crates/novomodelo-io/src"
    "${REPO_ROOT}/crates/novomodelo-solver/src"
    "${REPO_ROOT}/crates/novomodelo-comm/src"
    "${REPO_ROOT}/crates/novomodelo-stochastic/src"
    "${REPO_ROOT}/crates/novomodelo-sddp/src"
    "${REPO_ROOT}/crates/novomodelo-cli/src"
    "${REPO_ROOT}/crates/novomodelo-python/src"
    "${REPO_ROOT}/crates/novomodelo-mcp/src"
    "${REPO_ROOT}/crates/novomodelo-tui/src"
)

violations=""

# --- Production rustdoc pass ------------------------------------------------
# Stage 1 (awk): truncate each .rs file at the cfg(test) tail block and emit
#   "FILE:LINENO:CONTENT" so location is preserved downstream.
# Stage 2 (grep): keep doc-comment lines only (/// or //!), then extract the
#   matched token span (-oE) while keeping the FILE:LINE: prefix.
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' file; do
        # Pre-filter to doc-comment lines that ALSO carry a placeholder token
        # BEFORE the per-line span extraction below — otherwise every doc-comment
        # line (tens of thousands tree-wide) would spawn a subshell. Mirrors the
        # sibling check-comment-refs.sh, whose inner loop only sees real hits.
        doc_lines=$(cs_emit_production_lines "$file" | grep -E ':[[:space:]]*(///|//!)' \
            | grep -E "$PLACEHOLDER_PATTERN") || true
        [[ -n "$doc_lines" ]] || continue

        # Re-extract the token span from each candidate doc-comment line while
        # preserving FILE:LINE:. The content is everything after the second
        # colon; grep -oE on it yields the matched span only.
        while IFS= read -r line; do
            [[ -n "$line" ]] || continue
            file_part="${line%%:*}"
            rest="${line#*:}"
            line_part="${rest%%:*}"
            content="${rest#*:}"
            while IFS= read -r token; do
                [[ -n "$token" ]] || continue
                violations+="${file_part}:${line_part}:${token}"$'\n'
            done < <(printf '%s\n' "$content" \
                | grep -oE "$PLACEHOLDER_PATTERN" || true)
        done < <(printf '%s\n' "$doc_lines")
    done < <(find "$dir" -name "*.rs" -print0)
done

# Strip the trailing newline accumulated above.
violations="${violations%$'\n'}"

if [[ -n "$violations" ]]; then
    echo "FAIL: E5 placeholder text in shipped docs"
    echo ""
    echo "$violations"
    echo ""
    echo "Strip the rot, keep the invariant: fill the placeholder with the real"
    echo "value, or state the durable invariant (documentation-integrity §5)"
    echo "when no value can be pinned. A placeholder that ships reads as a fact"
    echo "while being a hole — the reader cannot cash it."
    exit 1
fi

echo "OK: no placeholder text in shipped docs."
exit 0
