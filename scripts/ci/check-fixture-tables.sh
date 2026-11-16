#!/usr/bin/env bash
#
# check-fixture-tables.sh — hand-built bounds/penalties tables in test code
# (ADVISORY, git-diff scoped).
#
# Tests build their ResolvedBounds / ResolvedPenalties tables from the entities
# with `build_with_resolved_tables` / `resolved_tables`, which call the
# production resolvers. A table written out by hand can drift from what the
# resolvers produce. This check flags NEW hand-built tables in test code, so the
# migration to the resolver-backed helpers does not erode. It NEVER fails the
# build.
#
# What it flags: an added line (diff against the merge base) that contains one
#   of the four tokens
#     ResolvedBounds::new(   ResolvedPenalties::new(
#     BoundsDefaults {       PenaltiesDefaults {
#   and is not a `//` comment line.
#
# Scope: added lines in crates/**/*.rs that are test scope:
#   * any file under crates/<crate>/tests/;
#   * any file named tests.rs or *_tests.rs (a `#[cfg(test)] mod ...;` module
#     file carries no cfg(test) line of its own);
#   * the cfg(test) tail of a crates/<crate>/src/**/*.rs file, i.e. the lines
#     after the first boundary found by cs_emit_production_lines in
#     scripts/ci/lib/comment_scan.sh. Known limitation (same as the sibling
#     gates): the boundary assumes the test module is a tail block.
#
# Homes: the shared helper modules crates/<crate>/src/test_support.rs and
#   crates/<crate>/src/test_support/ are excluded; they own the literal tables.
#
# Allowlist: scripts/ci/fixture-tables-allowlist.txt lists the sites whose table
#   must stay literal. Keys are `path::qualified_fn  # class`. An added line is
#   allowlisted when its path equals the key's path and the innermost fn that
#   encloses it is named like the last `::` segment of the key's fn. Keys carry
#   no line numbers or counts, which drift.
#
# Innermost enclosing fn: one awk pass over the HEAD file tracks brace depth and
#   a stack of `fn NAME` items, so a line inside a nested fn is attributed to
#   that fn and a line after it to the outer one. Strings, char braces and `//`
#   comments are stripped before counting braces.
#
# Base: BASE = `git merge-base HEAD "${BASE_REF:-origin/main}"`, falling back to
#   HEAD~1 with a warning. Added lines are read from
#   `git diff --unified=0 "$BASE"...HEAD -- 'crates/*.rs'`.
#
# Exit code: ALWAYS 0 (advisory — never fails the build).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT
cd "$REPO_ROOT"

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

readonly ALLOWLIST_REL="scripts/ci/fixture-tables-allowlist.txt"

BASE_REF="${BASE_REF:-origin/main}"
if ! BASE="$(git merge-base HEAD "$BASE_REF" 2>/dev/null)"; then
    echo "WARNING: git merge-base HEAD '$BASE_REF' failed; falling back to HEAD~1." >&2
    if ! BASE="$(git rev-parse HEAD~1 2>/dev/null)"; then
        echo "WARNING: no diff base; nothing checked." >&2
        exit 0
    fi
fi

declare -A ALLOWED=()
if [[ -f "$ALLOWLIST_REL" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        ALLOWED["${line%%::*}::${line##*::}"]=1
    done < "$ALLOWLIST_REL"
else
    echo "WARNING: allowlist missing: ${ALLOWLIST_REL}" >&2
fi

FN_AWK="$(cat <<'AWK'
{ s=$0
  gsub(/\\\\/,"",s); gsub(/\\"/,"",s); gsub(/"[^"]*"/,"",s); gsub(/'[{}]'/,"",s); sub(/\/\/.*/,"",s)
  if (match(s,/^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?((const|async|unsafe)[[:space:]]+)*fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
    nm=substr(s,RSTART,RLENGTH); sub(/.*fn[[:space:]]+/,"",nm); top++; name[top]=nm; d0[top]=depth; op[top]=0 }
  if (NR==t) { print (top>0 ? name[top] : "<top>"); exit }
  o=gsub(/[{]/,"{",s); c=gsub(/[}]/,"}",s); depth+=o-c
  if (top>0 && depth>d0[top]) op[top]=1
  while (top>0 && ((op[top] && depth<=d0[top]) || (!op[top] && (depth<d0[top] || (s ~ /;[[:space:]]*$/ && depth==d0[top]))))) top--
}
AWK
)"
readonly FN_AWK

candidates() {
    git diff --unified=0 --no-color "${BASE}"...HEAD -- 'crates/*.rs' | awk '
        /^\+\+\+ b\// { path = substr($0, 7); next }
        /^\+\+\+ \/dev\/null/ { path = ""; next }
        /^@@ / { n = $3; sub(/^\+/, "", n); sub(/,.*/, "", n); n += 0; next }
        /^\+/ {
            if (path != "" && path !~ /^crates\/[^\/]+\/src\/test_support(\.rs$|\/)/) {
                body = substr($0, 2)
                if (body !~ /^[[:space:]]*\/\// \
                    && match(body, /ResolvedBounds::new[(]|ResolvedPenalties::new[(]|BoundsDefaults [{]|PenaltiesDefaults [{]/))
                    printf "%s\t%d\t%s\n", path, n, substr(body, RSTART, RLENGTH)
            }
            n++
        }'
}

rows=""
while IFS=$'\t' read -r path lineno token; do
    [[ -f "$path" ]] || continue
    if [[ "$path" =~ ^crates/[^/]+/tests/ || "$path" =~ /(tests|[^/]*_tests)\.rs$ ]]; then
        :
    elif [[ "$path" =~ ^crates/[^/]+/src/ ]]; then
        prod_lines="$(cs_emit_production_lines "$path" | wc -l)"
        (( lineno > prod_lines )) || continue
    else
        continue
    fi
    fn_name="$(awk -v t="$lineno" "$FN_AWK" "$path")"
    [[ -z "${ALLOWED["${path}::${fn_name}"]:-}" ]] || continue
    rows+="${path}:${lineno}: ${token} (in fn ${fn_name})"$'\n'
done < <(candidates || true)

if [[ -n "$rows" ]]; then
    echo "ADVISORY: hand-built bounds/penalties tables added in test code"
    echo ""
    printf '%s' "$rows"
    echo ""
    echo "Build the table from its System with build_with_resolved_tables /"
    echo "resolved_tables (crates/novomodelo-sddp/src/test_support/table_resolution.rs)."
    echo "A table that must stay literal gets a 'path::fn  # class' row in"
    echo "${ALLOWLIST_REL}."
    echo "Advisory only — this does not fail the build."
    exit 0
fi

echo "OK: no hand-built bounds/penalties tables added in test code."
exit 0
