#!/usr/bin/env bash
#
# check-comment-bloat.sh — comment-bloat advisory (E8, ADVISORY).
#
# Surfaces the two comment-bloat shapes that are mechanically detectable without
# judgment, as candidates for the comment-skeptic pass (the nuanced restatement /
# drift-copy / contract-vs-bloat calls are NOT mechanical and are left to that
# pass and to .claude/rules/comments.md review). This gate is ADVISORY — it ranks
# candidates and NEVER fails the build.
#
# What it flags, over production `//` / `///` / `//!` comment lines:
#
#   1. LONG COMMENT BLOCK — a run of >= COMMENT_BLOCK_THRESHOLD (default 8)
#      consecutive comment lines. A long block is almost always verbose
#      load-bearing prose that wants cutting to one clause (comments.md §6), an
#      inline narration block, or a re-derived contract. Runs that contain a
#      rustdoc section header (`# Errors`, `# Panics`, `# Safety`, `# Examples`,
#      etc.) are NOT flagged — those are the legitimately structured API-doc
#      survivors comments.md keeps.
#
#   2. REPEATED COMMENT LINE — the same normalized comment text (prefix and
#      surrounding whitespace stripped, length >= 20) appearing
#      >= COMMENT_REPEAT_THRESHOLD (default 3) times in one file. This is the
#      "stated once, echoed across siblings" pattern (comments.md §3) in its
#      whole-line form — hoist it to the enclosing scope and delete the copies.
#
# Scope: production source under crates/*/src/ (matching the sibling comment
#   gates). The cfg(test) tail (everything from the first `#[cfg(test)]` line
#   onward) is dropped, as in check-comment-banners.sh — test modules carry
#   legitimately denser explanation (expected-value derivations, fixture
#   contracts) and get N5 leniency.
#
#   Known limitation (shared with the sibling gates): the cfg(test) exclusion
#   assumes the test module is a tail block.
#
# Reporting: each hit is printed as `FILE:LINE(S): <description>` under an
#   `ADVISORY:` banner, followed by the hit count and a footer.
#
# Exit code: ALWAYS 0 (advisory — never fails the build).

set -uo pipefail

BLOCK_THRESHOLD="${COMMENT_BLOCK_THRESHOLD:-8}"
REPEAT_THRESHOLD="${COMMENT_REPEAT_THRESHOLD:-3}"
ROOT="${1:-.}"

mapfile -t files < <(find "$ROOT/crates" -type f -name '*.rs' -path '*/src/*' 2>/dev/null | sort)

out=""
for f in "${files[@]}"; do
  [ -f "$f" ] || continue
  file_out="$(
    awk -v TB="$BLOCK_THRESHOLD" -v TR="$REPEAT_THRESHOLD" '
      # cfg(test) tail: stop scanning at the first test-module marker. Reset any
      # open comment run so it is not reported with a stale end line at EOF.
      /^[[:space:]]*#\[cfg\(test\)\]/ { intest = 1; runlen = 0; runhassection = 0 }
      intest { next }

      {
        is_comment = ($0 ~ /^[[:space:]]*\/\//)
        is_section = ($0 ~ /^[[:space:]]*\/\/[\/!]?[[:space:]]*#[[:space:]]/)
      }

      is_comment {
        if (runlen == 0) { runstart = FNR }
        runlen++
        if (is_section) { runhassection = 1 }

        norm = $0
        sub(/^[[:space:]]*\/\/[\/!]?[[:space:]]*/, "", norm)
        sub(/[[:space:]]+$/, "", norm)
        # Count only prose-like clauses: at least 20 chars, containing letters,
        # and not a code statement (drops `// ----` dividers and doctest `use ...;`).
        if (length(norm) >= 20 && norm ~ /[a-zA-Z]/ && norm !~ /;/) { rc[norm]++ }
        next
      }

      {
        if (runlen >= TB && runhassection == 0) {
          printf "%s:%d-%d: %d-line comment block (cut to the load-bearing clause)\n", FILENAME, runstart, FNR - 1, runlen
        }
        runlen = 0; runhassection = 0
      }

      END {
        if (runlen >= TB && runhassection == 0) {
          printf "%s:%d-%d: %d-line comment block (cut to the load-bearing clause)\n", FILENAME, runstart, FNR - 1, runlen
        }
        for (n in rc) {
          if (rc[n] >= TR) {
            printf "%s: comment repeated %dx (hoist to the enclosing scope): \"%s\"\n", FILENAME, rc[n], n
          }
        }
      }
    ' "$f"
  )"
  if [ -n "$file_out" ]; then
    out="${out}${file_out}"$'\n'
  fi
done

count="$(printf '%s' "$out" | grep -c . || true)"

echo "ADVISORY: comment-bloat candidates (E8) — block>=${BLOCK_THRESHOLD} lines, repeat>=${REPEAT_THRESHOLD}x"
echo "------------------------------------------------------------------------"
if [ "$count" -gt 0 ]; then
  printf '%s' "$out"
  echo "------------------------------------------------------------------------"
  echo "${count} candidate(s). Advisory only — run the comment-skeptic pass; see .claude/rules/comments.md."
else
  echo "No comment-bloat candidates found."
fi

exit 0
