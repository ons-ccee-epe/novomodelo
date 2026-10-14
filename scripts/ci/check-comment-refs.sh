#!/usr/bin/env bash
#
# check-comment-refs.sh — Un-rottable-reference gate for shipped .rs comments.
#
# Enforces the comment durability contract (see .claude/rules/comments.md §4
# "Durability" and N3): a shipped `//`/`///` comment must reference only things
# that cannot rot. Two sub-gates, scanned over production comment lines:
#
#   E3a (HARD, exit 1 on any hit) — a shipped comment must NOT cite:
#     * the literal token `MEMORY.md` (agent-memory file, not a shipped
#       contract; a reader cannot resolve it), OR
#     * a `.claude/` path that is NOT under `.claude/rules/` (private tooling
#       paths). The `.claude/rules/*` class IS allowed: those are path-scoped
#       contract-mirror files legitimately cited from source (the numerical /
#       architectural convention rule files). The allowlist is by PREFIX
#       (`.claude/rules/`), not a per-line exception, so future rule-file
#       citations (comments.md, doc-integrity.md) pass automatically.
#
#   E3b (ADVISORY, never changes the exit code) — a shipped comment that cites
#     a repo-relative path with prefix `docs/`, `artifacts/`, or `plans/`
#     (optionally with a leading `../`/`./` walk) that does NOT resolve against
#     the repository root. A dead design-doc pointer is drift, not a
#     shipped-contract lie, so it is reported as advisory only. External
#     citations (e.g. the sibling `novomodelo-docs` methodology repo, referenced via
#     `../../../novomodelo-docs/...`) carry a different prefix and are intentionally
#     NOT flagged.
#
# Scope: production source under crates/*/src/ (including stub crates). The
#   cfg(test) tail-block awk pre-filter (borrowed from check-no-plan-leaks.sh /
#   check-infra-genericity.sh) drops everything from the first `#[cfg(test)]`
#   line onward, so test-scope comments are out of scope.
#
#   Known limitation (same as the sibling gates): the exclusion assumes the
#   test module is a tail block. Files with a mid-file test module followed by
#   production code would incorrectly skip that trailing code. In practice
#   novomodelo files follow the tail-block convention.
#
# Reporting: each hit is printed as `FILE:LINE: <matched token span>` — the
#   token span only (grep -oE / -oP style), not the whole comment line.
#
# Exit codes:
#   0 — No E3a violations (E3b advisory block may still print).
#   1 — One or more E3a violations.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

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

# E3a token patterns (grep -oE extracts the matched span only):
#   1. The literal token `MEMORY.md`.
#   2. Any `.claude/` path; the `.claude/rules/` contract-mirror class is then
#      dropped by `grep -v` on the extracted token.
readonly E3A_MEMORY_PATTERN='MEMORY[.]md'
readonly E3A_CLAUDE_PATTERN='\.claude/[A-Za-z0-9_./-]+'
readonly E3A_CLAUDE_ALLOW='\.claude/rules/'

# E3b token extraction. A repo-relative doc-path token: an optional leading
# `../`/`./` walk followed by a `docs/`, `artifacts/`, or `plans/` prefix and
# the rest of the path. The negative lookbehind `(?<![A-Za-z0-9_./-])` ensures
# the prefix is not glued to a preceding word (so `novomodelo-docs/...` is NOT
# matched — its `docs` is preceded by `-`), while still allowing a `../`/`./`
# walk to lead the token.
readonly E3B_TOKEN_PATTERN='(?<![A-Za-z0-9_./-])((\.\./|\./)*(docs|artifacts|plans)/[A-Za-z0-9_./#-]+)'

# Emit production comment lines as `FILE:LINENO:CONTENT`, truncated at the
# cfg(test) tail block. Restricted to lines containing a `//` comment marker so
# the gate scopes to comments, not string literals or code.
emit_comment_lines() {
    cs_emit_comment_lines "$1"
}

# From a pre-filtered `FILE:LINE:CONTENT` stream (each line already known to
# match), re-extract each matched TOKEN with the given grep flag/pattern while
# preserving the `FILE:LINE:` location prefix. Emits `FILE:LINE: TOKEN`.
# $1 = grep mode flag set ("-oE" or "-oP"), $2 = pattern. Called only on the
# handful of candidate lines, so the per-line grep cost is negligible.
extract_tokens() {
    local mode="$1"
    local pattern="$2"
    local line file_part rest line_part content token
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        file_part="${line%%:*}"
        rest="${line#*:}"
        line_part="${rest%%:*}"
        content="${rest#*:}"
        while IFS= read -r token; do
            [[ -n "$token" ]] || continue
            printf '%s:%s: %s\n' "$file_part" "$line_part" "$token"
        done < <(printf '%s\n' "$content" | grep "$mode" "$pattern" || true)
    done
}

# Resolve a repo-relative doc-path token to a path relative to REPO_ROOT:
# strip every leading `../`/`./` walk, then drop any `#anchor` or ` §N`
# fragment suffix. The result is tested for existence under REPO_ROOT.
resolve_token() {
    local token="$1"
    # Strip leading ../ and ./ walk segments.
    while [[ "$token" == ../* || "$token" == ./* ]]; do
        token="${token#../}"
        token="${token#./}"
    done
    # Drop a trailing #anchor fragment.
    token="${token%%#*}"
    printf '%s' "$token"
}

# Print the E3b advisory block (always non-failing). Reads the accumulated
# advisory hits from the global e3b_hits.
print_e3b_advisory() {
    [[ -n "$e3b_hits" ]] || return 0
    echo ""
    echo "ADVISORY: E3b dead repo-relative doc-path references"
    echo ""
    echo "$e3b_hits"
    echo ""
    echo "Strip the rot, keep the invariant: repoint the dead design-doc link to"
    echo "the document's current location, or delete the link while keeping the"
    echo "surrounding prose. Advisory only — this does not fail the build."
}

# Build a single combined comment stream for every production .rs file across
# all SCAN_DIRS, in `FILE:LINE:CONTENT` form with the cfg(test) tail block
# already truncated. Whole-stream grep passes below then run once, not per file,
# keeping the gate fast in CI.
comment_stream=""
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' file; do
        file_stream="$(emit_comment_lines "$file")"
        [[ -n "$file_stream" ]] || continue
        comment_stream+="${file_stream}"$'\n'
    done < <(find "$dir" -name "*.rs" -print0)
done
comment_stream="${comment_stream%$'\n'}"

e3a_hits=""
e3b_hits=""

# --- E3a: MEMORY.md --------------------------------------------------------
# Pre-filter the whole stream for candidate lines (cheap), then extract the
# matched token span from each (few/none).
e3a_mem_lines="$(printf '%s\n' "$comment_stream" \
    | grep -E "$E3A_MEMORY_PATTERN" || true)"
if [[ -n "$e3a_mem_lines" ]]; then
    mem_hits="$(printf '%s\n' "$e3a_mem_lines" \
        | extract_tokens "-oE" "$E3A_MEMORY_PATTERN")"
    [[ -n "$mem_hits" ]] && e3a_hits+="${mem_hits}"$'\n'
fi

# --- E3a: .claude/ paths (excluding the .claude/rules/ contract-mirror) ----
e3a_claude_lines="$(printf '%s\n' "$comment_stream" \
    | grep -E "$E3A_CLAUDE_PATTERN" || true)"
if [[ -n "$e3a_claude_lines" ]]; then
    claude_hits="$(printf '%s\n' "$e3a_claude_lines" \
        | extract_tokens "-oE" "$E3A_CLAUDE_PATTERN" \
        | grep -vE "$E3A_CLAUDE_ALLOW" || true)"
    [[ -n "$claude_hits" ]] && e3a_hits+="${claude_hits}"$'\n'
fi

# --- E3b: repo-relative doc-path tokens that do not resolve ----------------
# Pre-filter to candidate lines carrying a docs/|artifacts/|plans/ token
# (-P lookbehind drops the novomodelo-docs glued prefix), then test each token's
# resolution against REPO_ROOT. Non-resolving tokens go to the advisory bucket.
e3b_lines="$(printf '%s\n' "$comment_stream" \
    | grep -P "$E3B_TOKEN_PATTERN" || true)"
if [[ -n "$e3b_lines" ]]; then
    while IFS= read -r prefixed; do
        [[ -n "$prefixed" ]] || continue
        file_part="${prefixed%%:*}"
        rest="${prefixed#*:}"
        line_part="${rest%%:*}"
        content="${rest#*:}"
        while IFS= read -r token; do
            [[ -n "$token" ]] || continue
            resolved="$(resolve_token "$token")"
            if [[ ! -e "${REPO_ROOT}/${resolved}" ]]; then
                e3b_hits+="${file_part}:${line_part}: ${token}"$'\n'
            fi
        done < <(printf '%s\n' "$content" | grep -oP "$E3B_TOKEN_PATTERN" || true)
    done < <(printf '%s\n' "$e3b_lines")
fi

# Strip trailing newlines accumulated above.
e3a_hits="${e3a_hits%$'\n'}"
e3b_hits="${e3b_hits%$'\n'}"

# ---------------------------------------------------------------------------
# E3a report (HARD).
# ---------------------------------------------------------------------------
if [[ -n "$e3a_hits" ]]; then
    echo "FAIL: E3a un-rottable-ref violations"
    echo ""
    echo "$e3a_hits"
    echo ""
    echo "Strip the rot, keep the invariant: a shipped comment must not cite a"
    echo "private memory file (MEMORY.md) or a .claude/ tooling path that a"
    echo "reader cannot resolve. Replace the citation with the durable contract"
    echo "it mirrors (a .claude/rules/*.md path-scoped rule) or with the living"
    echo "symbol / named test that pins the fact. The .claude/rules/"
    echo "contract-mirror class is allowed."
    print_e3b_advisory
    exit 1
fi

echo "OK: no E3a un-rottable-ref violations."
print_e3b_advisory
exit 0
