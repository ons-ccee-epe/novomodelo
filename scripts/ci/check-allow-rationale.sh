#!/usr/bin/env bash
#
# check-allow-rationale.sh — E4 rationale-on-suppression gate (git-diff scoped).
#
# Enforces comments.md directive D4: a `#[allow(...)]` that suppresses a
# shape/dead-code lint in PRODUCTION code is a standing judgment — "this
# length/complexity is load-bearing" — and must carry a justifying comment.
# An un-rationaled suppression is an unexplained exception that silently
# accretes during review.
#
# Enforcing D4 on the whole tree at once is infeasible: dozens of production
# suppressions predate this gate, almost none carry a rationale, and burning
# them down is a separate remediation track. So this gate enforces only on
# NEW or CHANGED `#[allow(...)]` attributes (diffed against the merge base),
# with a tracked allowlist of the pre-existing un-rationaled sites
# (scripts/ci/allow-rationale-allowlist.txt). New code must justify its
# suppressions; the legacy backlog is grandfathered until it is burned down.
#
# In-scope lints (the shape/dead-code lints D4 names):
#   clippy::too_many_lines, clippy::too_many_arguments, clippy::type_complexity,
#   dead_code, and the unused_* family.
# Out of scope: unwrap_used / expect_used / panic / float_cmp test-relaxation
#   allows, cast_* numeric-conversion allows, and everything else.
#
# Production scope only: a `#[allow(...)]` at or below the first test-module
#   boundary is test scope and is not evaluated. The boundary matches both the
#   bare `#[cfg(test)]` form AND the `#[cfg(all(test, ...))]` form novomodelo uses to
#   feature-gate test modules — a strict superset of the bare-`#[cfg(test)]`
#   tail-block filter in check-infra-genericity.sh / check-no-plan-leaks.sh.
#   Known limitation (same as those gates): the filter assumes the test module
#   is a tail block; a mid-file test module followed by production code would
#   incorrectly exempt that trailing code. novomodelo files follow the tail-block
#   convention.
#
# Rationale recognition: a suppression is justified if there is a non-empty,
#   non-doc `//` comment EITHER trailing inline on the attribute line OR
#   CONTIGUOUS with the attribute group above the opener — reachable by crossing
#   only `///` / `//!` doc lines and attribute / attribute-continuation lines. A
#   blank or code line ends the group, so a `//` belonging to the PRECEDING item
#   is not mis-credited as this site's rationale. Any non-empty comment counts
#   (a `RATIONALE:` token is conventional but not required); the rule wants a
#   justification, not a magic keyword, and free-form justifications stay valid.
#
# Test-scope exclusion: besides the in-file `#[cfg(test)]` tail-block filter,
#   modules whose `mod NAME;` declaration is itself `#[cfg(test)]`-gated in a
#   parent `mod.rs` / `lib.rs` are test scope even though the module file carries
#   no `#[cfg(test)]` line. Such files are dropped from the scan (see
#   TEST_ONLY_FILES below).
#
# Allowlist: scripts/ci/allow-rationale-allowlist.txt, keyed by `path::symbol`
#   (file path + the enclosing fn/impl/mod item name resolved as the next item
#   below the attribute), NOT by path:line — line numbers drift, the symbol is
#   stable. A site whose `path::symbol` key is in the allowlist is exempt even
#   if changed, UNLESS the change introduces a lint not already covered by its
#   pre-existing suppression (a new lint on an existing site is NEW).
#
# Scope to NEW/CHANGED: BASE = `git merge-base HEAD "${BASE_REF:-origin/main}"`,
#   with a HEAD~1 fallback + warning if merge-base fails. Added in-scope
#   `#[allow(...)]` openers are read from
#   `git diff --unified=0 "$BASE"...HEAD -- 'crates/*/src/*.rs'` (three-dot, so
#   only changes on HEAD's side of the merge base are considered).
#
# Still-true line-count mode (WHOLE-TREE, additive):
#   The presence check above is diff-scoped (it grandfathers the legacy backlog).
#   A separate "still-true" pass re-measures any production
#   `clippy::too_many_lines` rationale that cites a numeric line count and fails
#   if the cited count has drifted from the function body. A rationale that
#   states the function's *shape* ("single linear pipeline") rather than a count
#   is durable and is SKIPPED — only a parseable `<N> lines` cite is checked.
#
#   Why whole-tree and not diff-scoped: unlike presence, the still-true check has
#   no legacy backlog to grandfather — every cited count must be true NOW. A
#   diff-scoped variant would never re-examine an unchanged-but-drifted site,
#   which is the exact failure mode (a rationale citing "214 lines" stays green
#   forever after the body shrinks to 84). This is the prose-drift class the
#   doc-integrity rules target (stale snapshot presented as standing truth).
#
#   Counting convention (DOCUMENTED so a future reader does not file a false
#   drift finding from a different convention): the gate counts PHYSICAL lines —
#   every line from the `fn`/`pub fn`/`pub(crate) fn` signature line that follows
#   the attribute group, down to and INCLUDING its matching closing brace, with
#   no exclusion of blank or comment lines. This is intentionally NOT clippy's
#   own metric (clippy counts non-blank, non-comment lines and is typically lower
#   by 10-25%). The TOLERANCE band below is wide enough to absorb that
#   convention gap plus minor edits, while still catching gross drift (214-vs-84).
#
#   Count parsing: the first `<N> lines` token in the recognised rationale text,
#   via the COUNT_RE ERE below. A trailing `each` / `per` qualifier marks a RATE
#   ("~10 lines each") not a body total, and is NOT treated as a cited body
#   count — such a rationale is qualitative for this mode and is skipped.
#
#   Unmeasurable body (no matching closing brace located) is reported as an
#   explicit WARNING, never silently skipped (false-confidence guard).
#
# Exit codes:
#   0 — every new/changed in-scope suppression carries a rationale or is
#       allowlisted, AND every whole-tree numeric too_many_lines cite is
#       still true within TOLERANCE.
#   1 — at least one new/changed in-scope suppression lacks a rationale and is
#       not allowlisted, OR at least one numeric too_many_lines cite has drifted
#       beyond TOLERANCE (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

readonly ALLOWLIST_FILE="${REPO_ROOT}/scripts/ci/allow-rationale-allowlist.txt"

# In-scope lint tokens (ERE alternation, used to test an extracted lint list).
readonly IN_SCOPE_LINTS='too_many_lines|too_many_arguments|type_complexity|dead_code|unused_'

# Still-true mode: how far a cited line count may drift from the measured
# physical body before it is a violation. Wide enough to absorb the gap between
# the gate's physical-line convention and clippy's non-blank-non-comment metric
# (typically 10-25% lower) plus minor edits; narrow enough to catch gross drift
# such as a "214 lines" cite on an 84-line body.
readonly TOLERANCE=10

# Cited-count ERE: a number immediately bound to a `lines`/`ln`/`lns` token.
# `awk` ERE has no `\b`; a `[^A-Za-z0-9_]` (or end-of-string) trailing boundary
# is the POSIX-portable equivalent and avoids matching `linesXYZ`. A trailing
# `each`/`per` qualifier (handled separately) marks a RATE, not a body total.
readonly COUNT_RE='[0-9]+[[:space:]]*(lines|lns|ln|line)([^A-Za-z0-9_]|$)'

# --------------------------------------------------------------------------
# Allowlist membership: load `path::symbol` keys (and the lints each key
# already covers) from the tracked allowlist. Lines are `path::symbol` with an
# optional trailing `  # lint1,lint2` comment naming the pre-existing lints;
# blank lines and `#`-prefixed preamble lines are ignored.
# --------------------------------------------------------------------------
ALLOWED_KEYS=()
ALLOWED_LINTS=()
if [[ -f "$ALLOWLIST_FILE" ]]; then
    while IFS= read -r line; do
        # Strip leading/trailing whitespace.
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue
        # Split off an inline `# lints` annotation, if present.
        key="${line%%#*}"
        key="${key%"${key##*[![:space:]]}"}"
        local_lints=""
        if [[ "$line" == *"#"* ]]; then
            local_lints="${line#*#}"
        fi
        [[ -z "$key" ]] && continue
        ALLOWED_KEYS+=("$key")
        ALLOWED_LINTS+=("$local_lints")
    done <"$ALLOWLIST_FILE"
fi
readonly ALLOWED_KEYS ALLOWED_LINTS

# is_allowlisted <path::symbol> <comma-lints>
# Exempt when the key is present AND every offending lint is already covered by
# the key's recorded lint set. An empty recorded set (key with no `# lints`
# annotation) exempts the key for any in-scope lint — the conservative reading
# for legacy seeds whose lint set is the whole site.
is_allowlisted() {
    local query_key="$1"
    local offending="$2"
    local i
    for ((i = 0; i < ${#ALLOWED_KEYS[@]}; i++)); do
        if [[ "${ALLOWED_KEYS[$i]}" == "$query_key" ]]; then
            local recorded="${ALLOWED_LINTS[$i]}"
            # No recorded lint set → blanket exemption for this key.
            if [[ -z "${recorded//[[:space:]]/}" ]]; then
                return 0
            fi
            # Every offending lint must be among the recorded lints.
            local lint
            local all_covered=1
            local IFS=','
            for lint in $offending; do
                lint="${lint//[[:space:]]/}"
                [[ -z "$lint" ]] && continue
                if [[ ",${recorded//[[:space:]]/}," != *",$lint,"* ]]; then
                    all_covered=0
                    break
                fi
            done
            [[ $all_covered -eq 1 ]] && return 0
            return 1
        fi
    done
    return 1
}

# --------------------------------------------------------------------------
# extract_sites <file>
# Emit one record per production-scope `#[allow(...)]` site in <file>:
#   LINE<TAB>has_rationale(0|1)<TAB>symbol<TAB>in_scope_lints_csv
# Scanning stops at the first `#[cfg(test)]` line (tail-block filter), so test
# scope is never emitted. Handles single-line and multi-line `#[allow(...)]`.
# --------------------------------------------------------------------------
extract_sites() {
    local file="$1"
    awk '
        # Tail-block filter: stop at the first test-module boundary. Matches
        # both the bare `#[cfg(test)]` form and the `#[cfg(all(test, ...))]`
        # form novomodelo uses to feature-gate test modules — a strict superset of
        # the bare-`#[cfg(test)]` idiom in the sibling gates, so test-scope
        # suppressions under either form are correctly out of scope.
        /^[[:space:]]*#\[cfg\((all\()?test[,)]/ { exit }
        { lines[NR] = $0 }
        END {
            n = NR
            for (i = 1; i <= n; i++) {
                line = lines[i]
                # Detect an `#[allow(` opener (single- or multi-line).
                if (line !~ /#\[allow\(/) continue
                open_line = i
                # Accumulate the full attribute text from the opener until the
                # matching `)]`. A single-line attribute closes on open_line.
                attr = line
                close_line = i
                while (attr !~ /\)\]/ && close_line < n) {
                    close_line++
                    attr = attr " " lines[close_line]
                }
                # Restrict to the lint list inside the outermost allow(...).
                inner = attr
                sub(/^.*#\[allow\(/, "", inner)
                sub(/\)\].*$/, "", inner)
                # Collect in-scope lints present in this attribute.
                csv = ""
                split("too_many_lines too_many_arguments type_complexity dead_code", L, " ")
                for (k in L) {
                    if (index(inner, L[k]) > 0) {
                        csv = (csv == "" ? L[k] : csv "," L[k])
                    }
                }
                # unused_* family (token-prefix match).
                if (inner ~ /unused_[A-Za-z_]*/) {
                    m = inner
                    while (match(m, /unused_[A-Za-z_]+/)) {
                        tok = substr(m, RSTART, RLENGTH)
                        if (("," csv ",") !~ ("," tok ",")) {
                            csv = (csv == "" ? tok : csv "," tok)
                        }
                        m = substr(m, RSTART + RLENGTH)
                    }
                }
                if (csv == "") continue

                # Rationale recognition: a trailing inline // on the closing line,
                # or a // comment CONTIGUOUS with the attribute group above the
                # opener. "Contiguous" means reachable by crossing only doc-comment
                # and attribute / attribute-continuation lines — a blank or code
                # line ends the group. This is deliberately stricter than a fixed
                # N-line window: a window that crosses blank lines mis-credits a
                # comment belonging to the PRECEDING item as the rationale here.
                has_rat = 0
                # (a) trailing inline comment on the closing attribute line.
                cl = lines[close_line]
                if (cl ~ /\/\//) {
                    ct = cl
                    sub(/^.*\/\//, "", ct)
                    if (is_rationale(ct)) has_rat = 1
                }
                # (b) contiguous upward walk above the opener.
                if (!has_rat) {
                    j = open_line - 1
                    while (j >= 1) {
                        prev = lines[j]
                        # Cross doc-comment lines (do not match, do not stop).
                        if (prev ~ /^[[:space:]]*\/\/\// || prev ~ /^[[:space:]]*\/\/!/) {
                            j--
                            continue
                        }
                        # Cross attribute / attribute-continuation lines (other
                        # attrs, or the bare lint-list continuation of a multi-line
                        # allow, or its closing paren).
                        if (prev ~ /^[[:space:]]*#\[/ || prev ~ /^[[:space:]]*[A-Za-z_:]+,?[[:space:]]*$/ || prev ~ /^[[:space:]]*\)/ ) {
                            j--
                            continue
                        }
                        # First plain // comment contiguous with the group → rationale.
                        if (prev ~ /^[[:space:]]*\/\//) {
                            ct = prev
                            sub(/^[[:space:]]*\/\//, "", ct)
                            if (is_rationale(ct)) has_rat = 1
                            break
                        }
                        # Blank or code line: end of the attribute group, no
                        # contiguous rationale.
                        break
                    }
                }

                # Enclosing symbol: the next item below the attribute, resolved
                # as the declared name. Handles fn/struct/enum/trait/mod/const/
                # static/type/union/macro item names, impl-block target types,
                # and the item-that-is-not-a-keyword cases (struct fields, enum
                # variants, `use` re-exports) by falling back to the leading
                # identifier on the declaring line.
                sym = "?"
                for (j = close_line + 1; j <= n; j++) {
                    s = lines[j]
                    if (s ~ /^[[:space:]]*$/) continue
                    if (s ~ /^[[:space:]]*\/\//) continue
                    if (s ~ /^[[:space:]]*#\[/ || s ~ /^[[:space:]]*#!/) continue
                    sym = resolve_symbol(s)
                    break
                }

                printf "%d\t%d\t%s\t%s\n", open_line, has_rat, sym, csv
            }
        }
        # is_rationale(text): accepts case-insensitive RATIONALE: token OR any
        # non-empty comment text. A `//`-prefixed (doc-style triple-slash already
        # filtered upstream) non-empty comment counts as a free-form rationale.
        function is_rationale(text,   t) {
            t = text
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
            if (t == "") return 0
            return 1
        }
        # resolve_symbol(decl_line): the declared name of the item on the line
        # immediately below the attribute. Tries keyword items first (fn, type,
        # impl, ...), then falls back to a leading identifier (field/variant/use).
        function resolve_symbol(s,   rest, t, name, sp) {
            # Pad both ends so a leading/trailing keyword always has a non-word
            # neighbour. The `[^A-Za-z0-9_]` boundary is POSIX-portable; the
            # `\<`/`\>` GNU-awk word-boundary escapes are NOT (they silently
            # never match under mawk, the default awk on Ubuntu CI runners).
            sp = " " s " "
            # fn NAME — after any qualifiers (pub, const, async, unsafe, ...).
            if (match(sp, /[^A-Za-z0-9_]fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_]fn[[:space:]]+/, "", name); return name
            }
            # struct/enum/trait/mod/const/static/type/union NAME.
            if (match(sp, /[^A-Za-z0-9_](struct|enum|trait|mod|const|static|type|union)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_][A-Za-z_]+[[:space:]]+/, "", name); return name
            }
            # impl [<...>] [Trait for] Type — use the target type identifier.
            if (match(sp, /[^A-Za-z0-9_]impl[^A-Za-z0-9_]/)) {
                rest = substr(sp, RSTART); t = rest
                sub(/^[^A-Za-z0-9_]impl[[:space:]]*(<[^>]*>)?[[:space:]]*/, "", t)
                sub(/[[:space:]]*for[[:space:]]+/, " ", t)
                if (match(t, /[A-Za-z_][A-Za-z0-9_]*/)) return substr(t, RSTART, RLENGTH)
                return "impl"
            }
            # macro_rules! NAME.
            if (match(sp, /[^A-Za-z0-9_]macro_rules![[:space:]]*[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_]macro_rules![[:space:]]*/, "", name); return name
            }
            # use re-export: key on the first path identifier after `use`.
            if (match(sp, /[^A-Za-z0-9_]use[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_]use[[:space:]]+/, "", name); return name
            }
            # Fallback: struct field / enum variant — the leading identifier
            # (after any pub(...) / pub visibility qualifier).
            t = s
            sub(/^[[:space:]]*/, "", t)
            sub(/^pub(\([^)]*\))?[[:space:]]+/, "", t)
            if (match(t, /^[A-Za-z_][A-Za-z0-9_]*/)) return substr(t, RSTART, RLENGTH)
            return "?"
        }
    ' "$file"
}

# --------------------------------------------------------------------------
# Resolve BASE: merge-base of HEAD and BASE_REF (default origin/main), with a
# HEAD~1 fallback + warning if merge-base fails.
# --------------------------------------------------------------------------
BASE_REF="${BASE_REF:-origin/main}"
if ! BASE="$(git -C "$REPO_ROOT" merge-base HEAD "$BASE_REF" 2>/dev/null)"; then
    echo "WARNING: git merge-base HEAD '$BASE_REF' failed; falling back to HEAD~1." >&2
    if ! BASE="$(git -C "$REPO_ROOT" rev-parse HEAD~1 2>/dev/null)"; then
        echo "WARNING: HEAD~1 unavailable; treating BASE as empty (no diff)." >&2
        BASE=""
    fi
fi

# Collect the set of files with added in-scope `#[allow(...)]` lines, and the
# added line numbers per file. We parse the unified=0 diff: `+++ b/<path>`
# headers and `@@ ... +start[,count] @@` hunk headers give us added line spans;
# `+`-prefixed body lines containing `#[allow(` are candidate openers.
declare -A ADDED_OPENERS  # "path<TAB>line" -> 1

if [[ -n "$BASE" ]]; then
    cur_file=""
    add_line=0
    while IFS= read -r dline; do
        case "$dline" in
            "+++ b/"*)
                cur_file="${dline#+++ b/}"
                ;;
            "+++ /dev/null")
                cur_file=""
                ;;
            "@@ "*)
                # @@ -a,b +c,d @@  → next added line number is c.
                hunk="${dline#@@ }"
                hunk="${hunk%% @@*}"
                plus="${hunk#*+}"
                plus="${plus%% *}"
                add_line="${plus%%,*}"
                ;;
            "+"*)
                # Added body line (not the +++ header, handled above).
                if [[ -n "$cur_file" ]]; then
                    body="${dline#+}"
                    if [[ "$body" == *'#[allow('* ]]; then
                        ADDED_OPENERS["${cur_file}"$'\t'"${add_line}"]=1
                    fi
                    add_line=$((add_line + 1))
                fi
                ;;
            "-"*)
                : # removed line: does not advance the added-line counter.
                ;;
            *)
                : # context (unified=0 emits none) / metadata lines.
                ;;
        esac
    # git pathspec `*` crosses directory separators (unlike shell globbing), so
    # 'crates/*/src/*.rs' also matches nested sub-modules such as
    # crates/novomodelo-sddp/src/lp_builder/template.rs — they ARE in scope.
    done < <(git -C "$REPO_ROOT" diff --unified=0 "${BASE}"...HEAD -- 'crates/*/src/*.rs')
fi

# Test-only module files: a `mod NAME;` declaration gated by `#[cfg(test)]` in a
# parent `mod.rs` / `lib.rs` makes NAME's source file test scope, even though the
# file itself carries no `#[cfg(test)]` line for the tail-block filter to catch.
# Collect those files (both `NAME.rs` and `NAME/mod.rs` forms) and drop them from
# the scan. An optional blank or attribute line between the cfg attr and the
# `mod` declaration is tolerated.
declare -A TEST_ONLY_FILES
while IFS= read -r -d '' modfile; do
    moddir="$(dirname "$modfile")"
    while IFS= read -r modname; do
        [[ -n "$modname" ]] || continue
        for cand in "${moddir}/${modname}.rs" "${moddir}/${modname}/mod.rs"; do
            TEST_ONLY_FILES["${cand#"${REPO_ROOT}/"}"]=1
        done
    done < <(awk '
        /#\[cfg\(test\)\]/ { pend = 1; next }
        pend == 1 && /^[[:space:]]*(pub[[:space:]]+)?mod[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*;/ {
            name = $0
            sub(/^[[:space:]]*(pub[[:space:]]+)?mod[[:space:]]+/, "", name)
            sub(/[[:space:]]*;.*/, "", name)
            print name; pend = 0; next
        }
        pend == 1 && (/^[[:space:]]*$/ || /^[[:space:]]*#\[/) { next }
        { pend = 0 }
    ' "$modfile")
done < <(find "${REPO_ROOT}/crates" -type f \( -name "mod.rs" -o -name "lib.rs" \) -print0 2>/dev/null)
readonly TEST_ONLY_FILES

# For every file that has at least one added `#[allow(` opener, extract its
# production-scope sites and flag any whose opener line was added, is in scope,
# lacks a rationale, and is not allowlisted.
declare -A SEEN_FILES
violations=""

for key in "${!ADDED_OPENERS[@]}"; do
    file="${key%%$'\t'*}"
    SEEN_FILES["$file"]=1
done

for relfile in "${!SEEN_FILES[@]}"; do
    # Skip files that are test scope by virtue of a parent #[cfg(test)] mod gate.
    [[ -n "${TEST_ONLY_FILES["$relfile"]:-}" ]] && continue
    absfile="${REPO_ROOT}/${relfile}"
    [[ -f "$absfile" ]] || continue
    while IFS=$'\t' read -r site_line has_rat sym csv; do
        # Only consider sites whose opener line was an added line in the diff.
        [[ -n "${ADDED_OPENERS["${relfile}"$'\t'"${site_line}"]:-}" ]] || continue
        # Already rationaled → fine.
        [[ "$has_rat" == "1" ]] && continue
        # Allowlisted (and no new lint) → exempt.
        path_symbol="${relfile}::${sym}"
        if is_allowlisted "$path_symbol" "$csv"; then
            continue
        fi
        # Offending: report the matched token span (the #[allow] attribute).
        offending_lint="${csv%%,*}"
        violations+="${relfile}:${site_line}: #[allow(${csv})]  (lint: ${offending_lint}; symbol: ${sym})"$'\n'
    done < <(extract_sites "$absfile")
done

violations="${violations%$'\n'}"

presence_failed=0
if [[ -n "$violations" ]]; then
    echo "FAIL: E4 un-rationaled suppression"
    echo ""
    echo "$violations"
    echo ""
    echo "Each new/changed production #[allow(...)] for a shape/dead-code lint"
    echo "must carry a justification. Add a '// RATIONALE: ...' comment above"
    echo "the attribute (or a trailing inline // comment) explaining WHY the"
    echo "length/complexity/dead item is load-bearing and the refactor that"
    echo "would remove the lint is inappropriate."
    presence_failed=1
else
    echo "OK: all new/changed suppressions carry a rationale."
fi

# --------------------------------------------------------------------------
# extract_too_many_lines_cites <file>
# Emit one record per production-scope `#[allow(clippy::too_many_lines)]` site
# whose recognised rationale text cites a numeric line count:
#   SIG_LINE<TAB>SYMBOL<TAB>CITED_COUNT
# A site with no `too_many_lines` lint, or whose rationale is qualitative (no
# parseable `<N> lines` token), or whose count is a RATE (`<N> lines each/per`),
# emits nothing. The recognised rationale is the same contiguous-group / inline
# comment the presence pass credits, so the count parsed here is exactly the
# text the gate treats as the suppression's justification.
# The tail-block `#[cfg(test)]` filter mirrors extract_sites (test scope is out).
# --------------------------------------------------------------------------
extract_too_many_lines_cites() {
    local file="$1"
    awk -v count_re="$COUNT_RE" '
        /^[[:space:]]*#\[cfg\((all\()?test[,)]/ { exit }
        { lines[NR] = $0 }
        END {
            n = NR
            for (i = 1; i <= n; i++) {
                line = lines[i]
                if (line !~ /#\[allow\(/) continue
                open_line = i
                # Accumulate the full attribute text to its closing `)]`.
                attr = line
                close_line = i
                while (attr !~ /\)\]/ && close_line < n) {
                    close_line++
                    attr = attr " " lines[close_line]
                }
                inner = attr
                sub(/^.*#\[allow\(/, "", inner)
                sub(/\)\].*$/, "", inner)
                if (index(inner, "too_many_lines") == 0) continue

                # Recover the recognised rationale text (mirror extract_sites):
                # (a) trailing inline comment on the closing attribute line, else
                # (b) first plain // comment contiguous with the attribute group.
                rat = ""
                cl = lines[close_line]
                if (cl ~ /\/\//) {
                    ct = cl; sub(/^.*\/\//, "", ct)
                    if (is_rationale(ct)) rat = ct
                }
                if (rat == "") {
                    j = open_line - 1
                    while (j >= 1) {
                        prev = lines[j]
                        if (prev ~ /^[[:space:]]*\/\/\// || prev ~ /^[[:space:]]*\/\/!/) { j--; continue }
                        if (prev ~ /^[[:space:]]*#\[/ || prev ~ /^[[:space:]]*[A-Za-z_:]+,?[[:space:]]*$/ || prev ~ /^[[:space:]]*\)/) { j--; continue }
                        if (prev ~ /^[[:space:]]*\/\//) {
                            ct = prev; sub(/^[[:space:]]*\/\//, "", ct)
                            if (is_rationale(ct)) rat = ct
                            break
                        }
                        break
                    }
                }
                if (rat == "") continue

                # Parse the cited count. A RATE form (`<N> lines each|per`) is not
                # a body total → skip. Otherwise take the first `<N> lines` token.
                if (!match(rat, count_re)) continue
                tok = substr(rat, RSTART, RLENGTH)
                tail = substr(rat, RSTART + RLENGTH)
                if (tail ~ /^[[:space:]]*(each|per)([^A-Za-z0-9_]|$)/) continue
                cited = tok
                gsub(/[^0-9].*$/, "", cited)
                if (cited == "") continue

                # Enclosing symbol + its signature line: the first non-blank,
                # non-comment, non-attribute item line below the attribute group.
                sym = "?"
                sig = 0
                for (j = close_line + 1; j <= n; j++) {
                    s = lines[j]
                    if (s ~ /^[[:space:]]*$/) continue
                    if (s ~ /^[[:space:]]*\/\//) continue
                    if (s ~ /^[[:space:]]*#\[/ || s ~ /^[[:space:]]*#!/) continue
                    sym = resolve_symbol(s)
                    sig = j
                    break
                }
                if (sig == 0) continue
                printf "%d\t%s\t%d\n", sig, sym, cited
            }
        }
        function is_rationale(text,   t) {
            t = text
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
            if (t == "") return 0
            return 1
        }
        function resolve_symbol(s,   rest, t, name, sp) {
            sp = " " s " "
            if (match(sp, /[^A-Za-z0-9_]fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_]fn[[:space:]]+/, "", name); return name
            }
            if (match(sp, /[^A-Za-z0-9_](struct|enum|trait|mod|const|static|type|union)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
                name = substr(sp, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_][A-Za-z_]+[[:space:]]+/, "", name); return name
            }
            if (match(sp, /[^A-Za-z0-9_]impl[^A-Za-z0-9_]/)) {
                rest = substr(sp, RSTART); t = rest
                sub(/^[^A-Za-z0-9_]impl[[:space:]]*(<[^>]*>)?[[:space:]]*/, "", t)
                sub(/[[:space:]]*for[[:space:]]+/, " ", t)
                if (match(t, /[A-Za-z_][A-Za-z0-9_]*/)) return substr(t, RSTART, RLENGTH)
                return "impl"
            }
            t = s
            sub(/^[[:space:]]*/, "", t)
            sub(/^pub(\([^)]*\))?[[:space:]]+/, "", t)
            if (match(t, /^[A-Za-z_][A-Za-z0-9_]*/)) return substr(t, RSTART, RLENGTH)
            return "?"
        }
    ' "$file"
}

# --------------------------------------------------------------------------
# measure_body_lines <file> <sig_line>
# Count PHYSICAL lines from the signature line (inclusive) down to the matching
# closing brace (inclusive), tracking `{`/`}` balance. The body is considered to
# open on the first line at or after <sig_line> that contains a `{`. Emits the
# integer line span on success; emits the literal `UNMEASURABLE` when no
# matching close brace is found (false-confidence guard). No exclusion of blank
# or comment lines — this is the PHYSICAL-line convention documented in the
# header. Brace counting is line-granular and does not strip braces inside
# string/char literals or comments; the TOLERANCE band absorbs the rare skew
# this introduces for a drift check.
# --------------------------------------------------------------------------
measure_body_lines() {
    local file="$1"
    local sig="$2"
    awk -v sig="$sig" '
        NR < sig { next }
        {
            opens = gsub(/{/, "{")
            closes = gsub(/}/, "}")
            if (started == 0) {
                if (opens == 0) { count++; next }   # signature spread before the `{`
                started = 1
            }
            depth += opens - closes
            count++
            if (started == 1 && depth <= 0) { print count; found = 1; exit }
        }
        END { if (found != 1) print "UNMEASURABLE" }
    ' "$file"
}

# --------------------------------------------------------------------------
# check_still_true: whole-tree re-measurement of every production
# `too_many_lines` rationale that cites a numeric line count. Sets
# still_true_failed=1 on drift beyond TOLERANCE; emits an explicit WARNING (does
# NOT fail) when a body cannot be measured. Test-only module files (parent
# `#[cfg(test)] mod`) and the in-file tail-block filter are honoured exactly as
# the presence pass does.
# --------------------------------------------------------------------------
still_true_failed=0
still_true_violations=""
still_true_warnings=""

while IFS= read -r -d '' absfile; do
    relfile="${absfile#"${REPO_ROOT}/"}"
    [[ -n "${TEST_ONLY_FILES["$relfile"]:-}" ]] && continue
    while IFS=$'\t' read -r sig sym cited; do
        [[ -n "$sig" ]] || continue
        measured="$(measure_body_lines "$absfile" "$sig")"
        if [[ "$measured" == "UNMEASURABLE" ]]; then
            still_true_warnings+="${relfile}::${sym}: cited ${cited} lines, but the function body could not be measured (no matching closing brace located)."$'\n'
            continue
        fi
        local_diff=$((cited - measured))
        [[ $local_diff -lt 0 ]] && local_diff=$((-local_diff))
        if [[ $local_diff -gt $TOLERANCE ]]; then
            still_true_violations+="${relfile}::${sym}: rationale cites ${cited} lines, measured body is ${measured} physical lines (drift ${local_diff} > tolerance ${TOLERANCE})."$'\n'
        fi
    done < <(extract_too_many_lines_cites "$absfile")
done < <(find "${REPO_ROOT}/crates" -type f -name '*.rs' -path '*/src/*' -print0 2>/dev/null)

still_true_violations="${still_true_violations%$'\n'}"
still_true_warnings="${still_true_warnings%$'\n'}"

if [[ -n "$still_true_warnings" ]]; then
    echo ""
    echo "WARNING: E4 still-true could not measure these function bodies:"
    echo "$still_true_warnings"
fi

if [[ -n "$still_true_violations" ]]; then
    still_true_failed=1
    echo ""
    echo "FAIL: E4 stale line-count rationale"
    echo ""
    echo "$still_true_violations"
    echo ""
    echo "A too_many_lines rationale that cites a line count must stay true. Either"
    echo "correct the cited number to the current body length, or — preferred —"
    echo "replace it with a qualitative shape invariant (no number), mirroring the"
    echo "cleaned exemplars (run_forward_stage: 'single linear pipeline'; "
    echo "run_cut_management: '5 interleaved mutation phases')."
else
    echo "OK: every numeric too_many_lines rationale is still true (within ${TOLERANCE} lines)."
fi

if [[ $presence_failed -ne 0 || $still_true_failed -ne 0 ]]; then
    exit 1
fi
exit 0
