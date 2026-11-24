#!/usr/bin/env bash
#
# check-fixture-tables.sh — hand-built bounds/penalties tables in
# novomodelo-sddp test code.
#
# Tests build their ResolvedBounds / ResolvedPenalties tables from the entities
# with build_with_resolved_tables / resolved_tables
# (crates/novomodelo-sddp/src/test_support/table_resolution.rs), which call the
# production resolvers; a table written out by hand can drift from what the
# resolvers produce. The gate fails on every hand-built table in sddp test code
# that the allowlist does not exempt, and on every allowlist row that exempts
# none.
#
# A hand-built table: a line, other than a `//` comment line, holding one of
#     ResolvedBounds::new(   ResolvedPenalties::new(
#     BoundsDefaults {       PenaltiesDefaults {
#
# Test code in crates/novomodelo-sddp:
#   * every .rs file under tests/;
#   * every src/ file named tests.rs or *_tests.rs (a `#[cfg(test)] mod ...;`
#     module file carries no cfg(test) line of its own);
#   * the cfg(test) tail of any other src/ .rs file, i.e. the lines from the
#     first boundary cs_emit_production_lines (scripts/ci/lib/comment_scan.sh)
#     finds. Known limitation, shared with the sibling gates: the boundary
#     assumes the test module is a tail block.
# The homes, src/test_support.rs and src/test_support/, own the literal tables
# and are not scanned. Other crates are not scanned: io's resolver tests build
# the tables they assert on.
#
# Allowlist: scripts/ci/fixture-tables-allowlist.txt, one
#   `path::qualified_fn  # class` row per site whose table must stay literal. A
#   hit is exempt when its path equals the row's path and the innermost fn
#   enclosing it is named like the last `::` segment of the row's fn.
#
# Innermost enclosing fn: one awk pass over the file tracks brace depth and a
#   stack of `fn NAME` items, so a line inside a nested fn is attributed to that
#   fn and a line after it to the outer one. Strings, char braces and `//`
#   comments are stripped before counting braces.
#
# `--verify` writes a fixture tree with a hit for every token and test-file
# kind, lines that must not be reported (production code before the boundary,
# a comment line, the homes, another crate, an allowlisted fn) and a stale row
# whose fn only encloses a nested fn's hit, and fails unless exactly the
# expected hits and the stale row are reported. The fixture lives under
# $TMPDIR and is always removed.
#
# Exit codes:
#   0 — Pass (or --verify succeeded).
#   1 — Fail: an unexempted table or a stale row (or --verify misreported).
#   2 — Script error (bad invocation, scan library missing).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

readonly ALLOWLIST_REL="scripts/ci/fixture-tables-allowlist.txt"
readonly CRATE_REL="crates/novomodelo-sddp"
readonly TOKEN_ERE='(ResolvedBounds::new[(]|ResolvedPenalties::new[(]|BoundsDefaults [{]|PenaltiesDefaults [{])'

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

# scan_root <root>
#
# Populates HITS (`path:line: token (in fn NAME)`, path relative to <root>) and
# STALE (allowlist rows that exempt no hit). Returns 0 when both are empty.
scan_root() {
    local root="$1"
    HITS=()
    STALE=()

    local -A rows=() used=()
    local row key
    if [[ -f "$root/$ALLOWLIST_REL" ]]; then
        while IFS= read -r row; do
            row="${row%%#*}"
            row="${row#"${row%%[![:space:]]*}"}"
            row="${row%"${row##*[![:space:]]}"}"
            [[ -z "$row" ]] && continue
            key="${row%%::*}::${row##*::}"
            rows["$key"]+="${rows["$key"]:+$'\n'}${row}"
        done <"$root/$ALLOWLIST_REL"
    fi

    local rel path prod_lines lineno text token fn_name
    while IFS= read -r -d '' rel; do
        path="$root/$rel"
        prod_lines=0
        if [[ ! "$rel" =~ ^${CRATE_REL}/tests/ && ! "$rel" =~ /(tests|[^/]*_tests)\.rs$ ]]; then
            prod_lines="$(cs_emit_production_lines "$path" | wc -l)"
        fi
        while IFS=: read -r lineno text; do
            ((lineno > prod_lines)) || continue
            [[ "$text" =~ ^[[:space:]]*// ]] && continue
            [[ "$text" =~ $TOKEN_ERE ]] || continue
            token="${BASH_REMATCH[1]}"
            fn_name="$(awk -v t="$lineno" "$FN_AWK" "$path")"
            key="${rel}::${fn_name}"
            if [[ -n "${rows["$key"]:-}" ]]; then
                used["$key"]=1
                continue
            fi
            HITS+=("${rel}:${lineno}: ${token} (in fn ${fn_name})")
        done < <(grep -nE -e "$TOKEN_ERE" -- "$path" || true)
    done < <(cd "$root" && find "$CRATE_REL" \
        \( -type d \( -name target -o -path "$CRATE_REL/src/test_support" \) -prune \) \
        -o \( -type f -name '*.rs' ! -path "$CRATE_REL/src/test_support.rs" -print0 \) | sort -z)

    for key in "${!rows[@]}"; do
        [[ -n "${used["$key"]:-}" ]] && continue
        while IFS= read -r row; do
            STALE+=("$row")
        done <<<"${rows["$key"]}"
    done

    [[ ${#HITS[@]} -eq 0 && ${#STALE[@]} -eq 0 ]]
}

render_failures() {
    local line
    if [[ ${#HITS[@]} -gt 0 ]]; then
        echo "FAIL: hand-built bounds/penalties tables in novomodelo-sddp test code:"
        echo ""
        for line in "${HITS[@]}"; do
            echo "  $line"
        done
        echo ""
        echo "Build the table from its System with build_with_resolved_tables /"
        echo "resolved_tables (crates/novomodelo-sddp/src/test_support/table_resolution.rs)."
        echo "A table that must stay literal gets a 'path::fn  # class' row in"
        echo "${ALLOWLIST_REL}."
    fi
    if [[ ${#STALE[@]} -gt 0 ]]; then
        [[ ${#HITS[@]} -gt 0 ]] && echo ""
        echo "FAIL: ${ALLOWLIST_REL} rows that exempt no hand-built table:"
        echo ""
        while IFS= read -r line; do
            echo "  $line"
        done < <(printf '%s\n' "${STALE[@]}" | sort)
        echo ""
        echo "Delete them."
    fi
}

# write_fixture <dir>
write_fixture() {
    local d="$1"
    local c="$d/$CRATE_REL"
    mkdir -p "$c/tests" "$c/src/lp" "$c/src/test_support" "$d/crates/novomodelo-io/tests" \
        "$d/$(dirname "$ALLOWLIST_REL")"

    cat >"$c/tests/integration.rs" <<'RUST'
fn bounds_by_hand() {
    let bounds = ResolvedBounds::new(1);
}
RUST
    cat >"$c/src/lp/tests.rs" <<'RUST'
fn penalties_by_hand() {
    let penalties = ResolvedPenalties::new(1);
}

fn pinned_geometry() {
    let defaults = BoundsDefaults { storage: 1.0 };
}
RUST
    cat >"$c/src/lp/layout_tests.rs" <<'RUST'
fn defaults_by_hand() {
    let defaults = BoundsDefaults { storage: 1.0 };
}
RUST
    cat >"$c/src/stage.rs" <<'RUST'
pub fn production_table() -> ResolvedBounds {
    ResolvedBounds::new(0)
}

#[cfg(test)]
mod tests {
    fn outer() {
        fn inner() {
            let defaults = PenaltiesDefaults { spillage: 1.0 };
        }
        // ResolvedBounds::new( in a comment
    }
}
RUST
    cat >"$c/src/test_support.rs" <<'RUST'
pub fn home() -> ResolvedBounds { ResolvedBounds::new(0) }
RUST
    cat >"$c/src/test_support/tables.rs" <<'RUST'
pub fn home() -> ResolvedPenalties { ResolvedPenalties::new(0) }
RUST
    cat >"$d/crates/novomodelo-io/tests/resolution.rs" <<'RUST'
fn resolver_oracle() { let bounds = ResolvedBounds::new(1); }
RUST
    cat >"$d/$ALLOWLIST_REL" <<'TXT'
# fixture allowlist
crates/novomodelo-sddp/src/lp/tests.rs::pinned_geometry  # geometry-pinning
crates/novomodelo-sddp/src/stage.rs::tests::outer  # geometry-pinning
TXT
}

run_verify() {
    local fixture
    fixture="$(mktemp -d -t fixture_tables_gate.XXXXXX)"
    # shellcheck disable=SC2064
    trap "rm -rf '${fixture}'" EXIT

    write_fixture "$fixture"
    if scan_root "$fixture"; then
        echo "FAIL: --verify scan reported nothing against the fixture."
        return 1
    fi

    local expected_hits actual_hits expected_stale actual_stale
    expected_hits="$(printf '%s\n' \
        "$CRATE_REL/src/lp/layout_tests.rs:2: BoundsDefaults { (in fn defaults_by_hand)" \
        "$CRATE_REL/src/lp/tests.rs:2: ResolvedPenalties::new( (in fn penalties_by_hand)" \
        "$CRATE_REL/src/stage.rs:9: PenaltiesDefaults { (in fn inner)" \
        "$CRATE_REL/tests/integration.rs:2: ResolvedBounds::new( (in fn bounds_by_hand)" | sort)"
    actual_hits="$(printf '%s\n' "${HITS[@]}" | sort)"
    expected_stale="$CRATE_REL/src/stage.rs::tests::outer"
    actual_stale="$(printf '%s\n' "${STALE[@]}" | sort)"

    if [[ "$actual_hits" != "$expected_hits" || "$actual_stale" != "$expected_stale" ]]; then
        echo "FAIL: --verify:"
        diff <(printf 'hit %s\n' "$expected_hits"; printf 'stale %s\n' "$expected_stale") \
            <(printf 'hit %s\n' "$actual_hits"; printf 'stale %s\n' "$actual_stale") || true
        return 1
    fi

    echo "OK: --verify passed (4 tables and 1 stale row reported, no negative)."
    return 0
}

case "${1:-}" in
    --verify)
        run_verify
        ;;
    '')
        if scan_root "$REPO_ROOT"; then
            echo "OK: no unexempted hand-built bounds/penalties table in novomodelo-sddp test code."
            exit 0
        fi
        render_failures
        exit 1
        ;;
    *)
        echo "Usage: $0 [--verify]" >&2
        echo "" >&2
        echo "  (no args)  Scan novomodelo-sddp test code for hand-built tables." >&2
        echo "  --verify   Run the self-test against a synthetic fixture." >&2
        exit 2
        ;;
esac
