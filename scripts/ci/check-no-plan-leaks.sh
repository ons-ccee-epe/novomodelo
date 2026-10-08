#!/usr/bin/env bash
#
# check-no-plan-leaks.sh — Plan-structure leak gate.
#
# Scans shipped artefacts (production source, CHANGELOG, README) for
# plan-structure tokens that must not appear in user-facing
# content per CLAUDE.md hard rule:
#
#   "No plan-structure references in user-facing artifacts"
#
# Forbidden patterns:
#   [Ee]pic[ -][0-9]+   — "Epic 06", "epic-03", "Epic-12", etc.
#   [Tt]icket[ -][0-9]+ — "ticket-001", "Ticket 42", etc.
#   T0[0-9][0-9]     — "T002", "T007", "T015", etc.
#   \bsprint\b       — sprint planning vocabulary
#   \bF[0-9]?-[0-9]{2,}\b — workstream feature IDs: "F2-002", "F3-006",
#                          "F1-007", etc. (mandatory >=2 trailing digits so
#                          single-digit tokens do not collide).
#   \bW[0-9]+ (reset|rebake|workstream|phase)\b — context-anchored workstream
#                          phase language: "W2 reset", "W3 rebake", etc. Bare
#                          "W2" / "W3" without the anchored phrase is allowed.
#   \bW-[0-9]+\b      — hyphenated workstream ID form: "W-1", "W-5", etc.,
#                          distinct from the "W2 reset" phase form above. Added
#                          after a "W-5" tag slipped past the phase-anchored form.
#
# Informal plan vocabulary (added after a standards review found these shapes
# shipped while evading the formal-token patterns above):
#   \b[Tt]he (ticket|campaign)\b — prose references to the driving plan artefact.
#   \b[Dd]ecision [A-Z][0-9]+\b  — anchored plan-decision ids ("decision D11").
#                          The "decision" anchor is what disambiguates from the
#                          bare-D<n> collision documented below.
#   \bOption [A-Z]\b     — plan-alternative labels ("Option A"). Never collides
#                          with Rust `Option<T>` (no space before `<`). Name the
#                          chosen design by its durable invariant instead.
#   \bFix [0-9]\b        — plan fix-numbering ("Fix 1 — ..."). Test *names*
#                          carry no such tokens; banners/docs must not either.
#   \bAC: |\bAC[0-9]\b|\bAC-[0-9] — acceptance-criteria tags ("AC:", "AC1",
#                          "AC-3"). Deliberately colon/digit-anchored: the bare
#                          word "AC" stays free for the electrical domain sense
#                          (AC power flow) that novomodelo-flow will legitimately use.
#
# Section-reference pass (separate stage below, not part of PATTERN):
#   `§<n>` references are gated UNLESS the same line anchors them to a named
#   source-of-truth document — an `.md` filename, "spec"/"Spec", or a spec-doc
#   title word (Structures/Scenarios/Entities) immediately before the `§`.
#   Anchored spec refs ("internal-structures.md §15", "the spec (§15)",
#   "Internal Structures §1.9.4", a markdown link whose .md target follows the
#   §, "the paper's §3.4") are the N3-sanctioned durable form; a bare
#   "§5" ("design §5", "the §5 rescale") anchors into a gitignored plan doc and
#   is unrecoverable for a code-alone reader. A new legitimately-anchored doc
#   title extends SECTION_REF_ALLOW below.
#
# NOT gated (lexical collision — caught by review + the epic/plan hygiene sweep,
# not a regex): BARE owner-decision tokens "D<n>" (D0-D8) are lexically identical
# to the comment-rule directives D1-D5 (.claude/rules/comments.md), which ARE
# legitimately cited inline ("Determinism (D5)"). A bare \bD[0-9]\b gate would
# false-positive on every such directive citation, so bare plan D-decisions are
# stripped by the comment/plan hygiene sweep and review, not this script (the
# "decision D<n>"-anchored form IS gated above). The 2-digit deterministic case
# ids ("D06", "D15") never collide (single-digit gate) and are always allowed.
#
# Scope: production source under crates/*/src/ (including the umbrella
#   crates/novomodelo/src and the reserved stub crates), test/bench source under
#   crates/*/tests/ and crates/*/benches/, CHANGELOG.md, and README.md.
#   Vendored *.min.js files are excluded from every pass.
#
# Excluded: plans/ (gitignored), .github/, target/, .git/, Cargo.lock,
#   and the script itself.
#
# cfg(test) tail-block exclusion (borrowed from check-infra-genericity.sh):
#   For each .rs file under crates/*/src/, scanning stops at the first line
#   matching `#[cfg(test)]`; all subsequent lines until end-of-file are
#   considered test scope and skipped. This mirrors the awk pre-filter mechanism
#   in check-infra-genericity.sh.
#
#   Sibling test files (`tests.rs`, the `#[cfg(test)] mod tests;` convention)
#   carry no in-file `#[cfg(test)]` line and are therefore scanned in FULL. They
#   were once excluded by filename, which made the gate self-inconsistent — an
#   integration binary under crates/*/tests/ was scanned while the unit-test
#   sibling next to the code it tests was not — and let three leaks ship. Test
#   NAMES stay allowed without needing a carve-out: `\b` in PATTERN does not
#   match across the `_` that Rust identifiers join with.
#
#   Known limitation (same as check-infra-genericity.sh): the exclusion
#   assumes the test module is a tail block. Files with mid-file test modules
#   followed by production code would incorrectly skip that trailing code. In
#   practice, novomodelo files follow the tail-block convention.
#
#   Non-.rs targets (CHANGELOG.md, README.md) have no cfg(test) concept
#   and stay on the plain whole-file grep path.
#
# Exit codes:
#   0 — No leaks found.
#   1 — Leaks found (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

# The epic/ticket/sprint words are matched BARE, with no digit required, because a
# number is not what makes them leaks — "the epic's safety mechanism", "untouched by
# this ticket" and the plural "Tickets 016-020" all named the plan and all evaded the
# earlier digit-anchored forms (`[Ee]pic[ -][0-9]+` cannot match "epic's", and the "s"
# in "Tickets" displaces the required separator). `\b` keeps Rust test names allowed
# without a carve-out: identifiers join with `_`, which is a word character, so
# `ticket_016_regression` does not match while the prose "ticket 016" does.
readonly PATTERN='\b[Ee]pics?\b|\b[Tt]ickets?\b|\b[Ss]prints?\b|T0[0-9][0-9]|\bF[0-9]?-[0-9]{2,}\b|\bW[0-9]+ (reset|rebake|workstream|phase)\b|\bW-[0-9]+\b|\b[Tt]he campaign\b|\b[Rr]equirement [0-9]+[a-z]?\b|\b[Rr]equirements [0-9]+\b|\b[Dd]ecision [A-Z][0-9]+\b|\bOption [A-Z]\b|\bFix [0-9]\b|\bAC: |\bAC[0-9]\b|\bAC-[0-9]'

# Section-reference pass: flag `§<n>` unless the line anchors it to a named
# source-of-truth document (see header). Two-stage because "allowed unless
# anchored" is a negative match grep -E alone cannot express in PATTERN.
readonly SECTION_REF='§[0-9]'
readonly SECTION_REF_ALLOW='\.md|([Ss]pec|Structures|Scenarios|Entities|Principles|Model|paper.s)[] )]*[ (]?§'

# .rs source directories: scanned per-file with the cfg(test) tail-block
# exclusion (see header). The umbrella crates/novomodelo/src and the reserved stub
# crates are scanned too. The crates/*/tests and crates/*/benches dirs are
# glob-expanded below and appended; the cfg(test) awk filter is a harmless
# no-op there (test/bench files carry no #[cfg(test)] tail boundary).
SCAN_DIRS=(
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
    "${REPO_ROOT}/crates/novomodelo/src"
    "${REPO_ROOT}/crates/novomodelo-flow/src"
    "${REPO_ROOT}/crates/novomodelo-uc/src"
    "${REPO_ROOT}/crates/novomodelo-emt/src"
)

# Append the test/bench source dirs. nullglob keeps the loop a no-op when a
# glob matches nothing (e.g. a crate without a benches/ dir), so unexpanded
# literal globs never enter SCAN_DIRS. The [[ -d ]] guard in the scan loop is
# the second line of defence.
shopt -s nullglob
for dir in "${REPO_ROOT}"/crates/*/tests "${REPO_ROOT}"/crates/*/benches; do
    SCAN_DIRS+=("$dir")
done
shopt -u nullglob

readonly SCAN_DIRS

# Non-.rs targets: scanned whole-file with plain grep (no cfg(test) concept).
readonly SCAN_FILES=(
    "${REPO_ROOT}/CHANGELOG.md"
    "${REPO_ROOT}/README.md"
)

violations=""

# Stage 1 (awk): structural pre-filter — stop emitting lines once a bare
#   `#[cfg(test)]` line is encountered (tail test-module exclusion). For each
#   production line, emit "FILE:LINENO:CONTENT" so grep output retains location.
# Stage 2 (grep -E): apply PATTERN. awk performs no regex matching here; it only
#   handles line truncation + prefix.
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' file; do
        matches=$(cs_emit_production_lines "$file" | grep -E "$PATTERN") || true
        if [[ -n "$matches" ]]; then
            violations+="${matches}"$'\n'
        fi
        section_matches=$(cs_emit_production_lines "$file" \
            | grep -E "$SECTION_REF" | grep -vE "$SECTION_REF_ALLOW") || true
        if [[ -n "$section_matches" ]]; then
            violations+="${section_matches}"$'\n'
        fi
    done < <(find "$dir" -name "*.rs" -print0)
done

# Non-.rs targets: plain whole-file recursive grep. *.min.js files are
# excluded — the widened PATTERN could otherwise match minified JS tokens.
file_violations=$(grep -rnE "$PATTERN" \
    --exclude='*.min.js' "${SCAN_FILES[@]}" 2>/dev/null \
    || true)
if [[ -n "$file_violations" ]]; then
    violations+="${file_violations}"$'\n'
fi
file_section_violations=$(grep -rnE "$SECTION_REF" \
    --exclude='*.min.js' "${SCAN_FILES[@]}" 2>/dev/null \
    | grep -vE "$SECTION_REF_ALLOW" || true)
if [[ -n "$file_section_violations" ]]; then
    violations+="${file_section_violations}"$'\n'
fi

# Strip the trailing newline accumulated above.
violations="${violations%$'\n'}"

if [[ -n "$violations" ]]; then
    echo "FAIL: plan-structure leaks found in shipped artefacts."
    echo ""
    echo "$violations"
    echo ""
    echo "Per CLAUDE.md, plan-structure references must not appear"
    echo "in shipped source, README, or the CHANGELOG. Rewrite"
    echo "in behavioural terms or move to plans/ (gitignored)."
    echo ""
    echo "For workstream feature IDs (F2-002, F3-006, ...) and anchored"
    echo "workstream phase language (W3 rebake, W2 reset, ...), strip the"
    echo "rot, keep the invariant: replace the plan ID with the behavioural"
    echo "phrase it tags and leave the surrounding invariant intact — never"
    echo "delete the comment line."
    echo ""
    echo "For informal plan vocabulary (Option A, decision D11, AC:, Fix 1,"
    echo "the ticket, bare §5): name the design by its durable invariant"
    echo "instead of the plan label. A §-reference is allowed only when the"
    echo "same line names its source-of-truth document (an .md filename or"
    echo "spec-doc title); extend SECTION_REF_ALLOW for a new legitimate one."
    exit 1
fi

echo "OK: no plan-structure leaks found in shipped artefacts."
exit 0
