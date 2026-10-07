#!/usr/bin/env bash
#
# check-doc-paths.sh — Repo-relative path/link integrity gate for prose docs.
#
# Wired enforcement of .claude/rules/doc-integrity.md §6.4 ("Executable/
# resolvable claim with no compiler backstop"): a Markdown doc that cites a
# repo-relative tree path which silently rots when the tree moves underneath it
# has no compiler to catch the drift. This gate is that mechanical check.
#
# Scanned sources (exactly):
#   README.md, CONTRIBUTING.md, CLAUDE.md.
#
# One check:
#
#   1. Repo-relative tree-path tokens. A backtick-quoted or bare token that
#      begins with a RECOGNIZED top-level repo-relative prefix:
#
#          crates/  scripts/  examples/  docs/  .github/  .claude/
#
#      OR is one of the bare top-level repo filenames:
#
#          Cargo.toml  Cargo.lock  deny.toml  rust-toolchain.toml
#
#      Each token is stripped of a trailing :NNN line suffix, a #anchor, and a
#      trailing §N / ` §N` section suffix, then required to resolve under
#      REPO_ROOT via `[[ -e ]]`.
#
#      `src/` is intentionally NOT a prefix: there is no top-level src/ dir in
#      this repo (sources live under crates/*/src/). A token like `src/gemm.rs`
#      (CLAUDE.md) is intentional CRATE-RELATIVE shorthand that legitimately
#      does not resolve at repo root. Including `src/` would produce guaranteed
#      false positives on legitimate shorthand and violate §6.4's "never
#      require every cited filename resolves" bound.
#
#      Glob tokens (containing */** ) assert a directory class, not a literal
#      file. Resolve the longest wildcard-free ancestor (e.g. crates/novomodelo-sddp/
#      for `crates/novomodelo-sddp/**/*.rs`); a rotted directory prefix still fails.
#
#   (The former Check 2 — intra-book relative link resolution — was retired
#   with book/ (mdBook decommission); the unified docs site at
#   docs.novomodelo.invalid owns link integrity for user-facing prose now.)
#
# Exclusions (MUST NOT be flagged), per §6.4's scope bound:
#   - Absolute URLs: http://, https://, mailto:, ftp://.
#   - Any token containing the substring `novomodelo-docs` (the external methodology
#     repo, absent from this tree by design).
#   - Any token into an LP-solver submodule (`crates/novomodelo-solver/vendor/HiGHS`,
#     `.../Clp`, `.../CoinUtils`) — separately-versioned external repos, absent
#     without a recursive checkout. The authored `vendor/coin-build/` glue is in
#     this repo and stays validated.
#   - Any token NOT starting with a recognized prefix and not a recognized
#     top-level filename. User-data input-file names (config.json, thermals.json,
#     system/hydros.json, output/...) are out of scope BY CONSTRUCTION — they
#     lack a tree prefix and are never resolved.
#
# Reporting: each hit is printed as `FILE:LINE: <matched token span>` — the
#   token span only (grep -noE style), never the whole prose line.
#
# Exit codes:
#   0 — All repo-relative paths resolve.
#   1 — One or more do not resolve, OR a required scanned source is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# Explicit single-file sources (relative to REPO_ROOT).
readonly SCAN_FILES=(
    "README.md"
    "CONTRIBUTING.md"
    "CLAUDE.md"
)

# Recognized top-level repo-relative directory prefixes (a token must begin with
# one of `<prefix>/` to be in scope). `src/` is deliberately absent (see header).
readonly REPO_PREFIXES=(
    "crates"
    "scripts"
    "examples"
    "docs"
    ".github"
    ".claude"
)

# Recognized bare top-level repo filenames (a token may equal one of these).
readonly REPO_FILES=(
    "Cargo.toml"
    "Cargo.lock"
    "deny.toml"
    "rust-toolchain.toml"
)

# Substrings that disqualify a token outright (external repos).
readonly EXCLUDED_SUBSTRINGS=(
    "novomodelo-docs"
    # The LP-solver submodules are separately-versioned external repos, absent
    # unless checked out recursively (the lightweight script gates do not). Their
    # contents are external to this tree, so paths into them are excluded on the
    # same basis as novomodelo-docs. The authored `vendor/coin-build/` glue lives in
    # this repo and is NOT matched here, so it stays validated.
    "crates/novomodelo-solver/vendor/HiGHS"
    "crates/novomodelo-solver/vendor/Clp"
    "crates/novomodelo-solver/vendor/CoinUtils"
)

violations=""

# True if the token begins with a recognized `<prefix>/` or equals a recognized
# top-level filename; false otherwise (out of scope — never resolved).
is_in_scope() {
    local token="$1"
    local prefix file
    for prefix in "${REPO_PREFIXES[@]}"; do
        [[ "$token" == "${prefix}/"* ]] && return 0
    done
    for file in "${REPO_FILES[@]}"; do
        [[ "$token" == "$file" ]] && return 0
    done
    return 1
}

# True if the token carries an excluded substring or an absolute-URL scheme.
is_excluded() {
    local token="$1"
    local sub
    for sub in "${EXCLUDED_SUBSTRINGS[@]}"; do
        [[ "$token" == *"$sub"* ]] && return 0
    done
    case "$token" in
        http://*|https://*|mailto:*|ftp://*) return 0 ;;
    esac
    return 1
}

# Normalize a path token for resolution: drop a trailing :NNN line suffix, a
# #anchor fragment, and a trailing ` §N` / `§N` section suffix.
normalize_token() {
    local token="$1"
    token="${token%% §*}"   # drop ` §N` (space-separated) section suffix
    token="${token%%§*}"    # drop a glued §N section suffix
    token="${token%%#*}"    # drop a #anchor fragment
    token="${token%:}"      # drop a bare trailing colon (defensive)
    # Drop a trailing :NNN line suffix (e.g. book/src/crates/sddp.md:13).
    if [[ "$token" =~ ^(.*):[0-9]+$ ]]; then
        token="${BASH_REMATCH[1]}"
    fi
    printf '%s' "$token"
}

# Test whether a normalized token resolves under REPO_ROOT.
#   - For a literal token: `[[ -e REPO_ROOT/token ]]`.
#   - For a glob token (contains * or **): resolve the longest wildcard-free
#     ancestor (the path up to the first segment containing `*`) and test that.
#     A glob asserts a directory class, not a literal file; a rotted directory
#     prefix still fails.
token_resolves() {
    local token="$1"
    if [[ "$token" == *"*"* ]]; then
        # Longest wildcard-free ancestor: everything before the first `*`,
        # with the trailing slash dropped (e.g. crates/novomodelo-sddp/**/*.rs ->
        # crates/novomodelo-sddp). A glob asserts a directory class, not a literal
        # file; a rotted directory prefix still fails.
        local ancestor="${token%%"*"*}"
        ancestor="${ancestor%/}"
        # If the ancestor still contains no slash it is a bare wildcard segment;
        # nothing concrete to resolve, treat as resolved (out of scope).
        [[ -n "$ancestor" ]] || return 0
        [[ -e "${REPO_ROOT}/${ancestor}" ]]
        return
    fi
    [[ -e "${REPO_ROOT}/${token}" ]]
}

# --- Check 1: repo-relative tree-path tokens -------------------------------
# Extract candidate tokens from one Markdown file and accumulate any that are
# in scope, not excluded, and fail to resolve. Two extraction passes:
#   (a) backtick-quoted spans `...` (well-delimited),
#   (b) bare whitespace/paren/comma-delimited words.
# Each pass uses `grep -noE` so every hit carries its line number.
check_path_tokens() {
    local file="$1"
    local rel="$2"
    local ln raw inner token norm

    # (a) backtick-quoted tokens.
    while IFS=: read -r ln raw; do
        [[ -n "$raw" ]] || continue
        inner="${raw#\`}"
        inner="${inner%\`}"
        is_in_scope "$inner" || continue
        is_excluded "$inner" && continue
        norm="$(normalize_token "$inner")"
        if ! token_resolves "$norm"; then
            violations+="${rel}:${ln}: ${inner}"$'\n'
        fi
    done < <(grep -noE '`[^`]+`' "$file" 2>/dev/null || true)

    # (b) bare tokens: words that start with a recognized prefix or filename.
    # The character class covers path bytes plus `*` (globs), `#` (anchors),
    # `=` (Hive-style partition dirs); a `§N` suffix is handled via normalize.
    #
    # The `(?<![A-Za-z0-9_./(-])` negative lookbehind requires the prefix to sit
    # at a genuine left token boundary. It rejects the prefix when it is glued to
    # a preceding `/`, `.`, `-`, `(`, or word char — i.e. when it is really a URL
    # path segment (.../crates/...), a relative-link walk (../crates/...), or a
    # Markdown link target (](crates/...)). Those constructs are handled by the
    # URL exclusion and by Check 2 (intra-book links), not by this path pass.
    # This mirrors check-comment-refs.sh's -P lookbehind that drops the glued
    # `novomodelo-docs` case.
    while IFS=: read -r ln token; do
        [[ -n "$token" ]] || continue
        is_in_scope "$token" || continue
        is_excluded "$token" && continue
        norm="$(normalize_token "$token")"
        if ! token_resolves "$norm"; then
            violations+="${rel}:${ln}: ${token}"$'\n'
        fi
    done < <(grep -noP '(?<![A-Za-z0-9_./(-])(crates|scripts|examples|docs|\.github|\.claude)/[A-Za-z0-9_./*#=-]+|(?<![A-Za-z0-9_./(-])(Cargo\.toml|Cargo\.lock|deny\.toml|rust-toolchain\.toml)' "$file" 2>/dev/null || true)
}

# --- Drive the check over every scanned source ------------------------------
# Explicit single-file sources (hard error if a required source is gone).
for rel in "${SCAN_FILES[@]}"; do
    file="${REPO_ROOT}/${rel}"
    if [[ ! -f "$file" ]]; then
        echo "FAIL: required scanned source is missing: ${rel}"
        exit 1
    fi
    check_path_tokens "$file" "$rel"
done

# Strip the trailing newline accumulated above, then de-duplicate: a token may
# be reported by both the backtick pass and the bare pass (e.g. a backtick-quoted
# path whose `crates/` also sits at a bare left boundary). The dedupe is
# order-stable so output stays grouped by file/line.
violations="${violations%$'\n'}"
if [[ -n "$violations" ]]; then
    violations="$(printf '%s\n' "$violations" | awk '!seen[$0]++')"
fi

if [[ -n "$violations" ]]; then
    echo "FAIL: unresolved repo-relative doc paths/links."
    echo ""
    echo "$violations"
    echo ""
    echo "Strip the rot, keep the invariant: fix the path so it resolves against"
    echo "the live tree, or — if the reference points at the external novomodelo-docs"
    echo "spec/theory pages — rewrite it as an absolute URL or confirm it belongs"
    echo "to the novomodelo-docs exclusion. A repo-relative path with a recognized"
    echo "prefix must resolve; never freeze a path that the tree can move out from"
    echo "under."
    exit 1
fi

echo "OK: all repo-relative doc paths resolve."
exit 0
