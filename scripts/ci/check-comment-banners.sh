#!/usr/bin/env bash
#
# check-comment-banners.sh — N5 in-function banner advisory (ADVISORY).
#
# Box-drawing banner glyphs (U+2500..U+257F) that fence groups INSIDE a function
# body are the specific thing N5 forbids ("no banners fencing groups inside one
# long production function — extract a function instead"; see
# .claude/rules/comments.md). Dividers between top-level items, inside `impl` /
# `mod` blocks, and inside `extern "C"` blocks are explicitly N5-OK. This gate is
# ADVISORY — it surfaces the in-fn banners (each a "this function wants
# splitting" marker) so they can be ranked and remedied, but NEVER fails the
# build.
#
# What it flags, over production `//`/`///`/`//!` comment lines:
#   A box-drawing glyph in the Unicode range U+2500..U+257F on a comment line
#   that sits inside a `fn` body. Matching uses grep -P with the
#   `\x{2500}-\x{257F}` UTF-8-aware class — a byte class would mis-handle these
#   multibyte glyphs. Only the token span is reported. Top-level / impl / mod
#   dividers are deliberately NOT flagged (they are N5-OK).
#
# Scope: production source under crates/*/src/ (including stub crates), with
#   three pre-filter exclusions applied in the awk stage:
#     1. cfg(test) tail block — everything from the first `#[cfg(test)]` line
#        onward is dropped (borrowed from the sibling gates).
#     2. `extern "C"` blocks — from an `extern "C"` opener line to its matching
#        column-0 closing `}` (inclusive), all lines are suppressed. Dividers
#        that mirror a C header inside an `extern "C"` block are an allowed N5
#        carve-out, so they are out of scope here.
#     3. non-fn-body scope — only comment lines whose enclosing scope includes a
#        `fn` body are considered; top-level and impl/mod dividers are skipped.
#
#   Known limitation (same as the sibling gates): the cfg(test) exclusion
#   assumes the test module is a tail block. The `extern "C"` skip assumes the
#   block's closing brace is the first column-0 `}` after the opener (true for
#   the novomodelo FFI blocks, whose items are indented).
#
# Reporting: each hit is printed as `FILE:LINE: <matched glyph span>` under an
#   `ADVISORY:` banner, followed by a non-zero hit count and a footer.
#
# Exit code: ALWAYS 0 (advisory — never fails the build).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# Production .rs source directories, scanned per-file with the cfg(test)
# tail-block + extern "C" block exclusions. Mirrors check-comment-refs.sh.
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

# Box-drawing glyph range U+2500..U+257F. `\x{...}` requires grep -P (PCRE);
# the class is UTF-8-aware so it matches the whole multibyte glyph, not a byte.
# The `+` collapses a contiguous divider run (`────────`) into one matched span
# so a banner reports as a single span, not one entry per glyph.
readonly GLYPH_PATTERN='[\x{2500}-\x{257F}]+'

# Emit production comment lines as `FILE:LINENO:CONTENT`, restricted to lines
# carrying a `//` comment marker (covers `//`, `///`, `//!`) that sit INSIDE a
# `fn` body — the only place N5 forbids a banner ("no banners fencing groups
# inside one long production function"). Dividers between top-level items and
# inside `impl` / `mod` blocks are N5-OK and are NOT emitted. Three structural
# pre-filters are applied as the file is walked:
#   * exit once a bare `#[cfg(test)]` line is reached (tail test-module skip).
#   * suppress every line from an `extern "C"` opener through its matching
#     column-0 closing `}` (inclusive) via the in_extern_c state flag.
#   * track brace depth and an fn-body stack so only lines whose enclosing
#     scope includes a `fn` body are emitted.
#
# fn-context tracking is a heuristic: a `fn NAME(` definition arms `pending_fn`,
# and the next `{` that opens marks that brace level as a fn body; the level is
# popped when its matching `}` is seen. Closures inside a fn stay in-fn (the
# enclosing fn level is still on the stack). A `fn` pointer type (`fn(` with no
# following name) does not arm the flag. Good enough for an advisory gate.
emit_comment_lines() {
    local file="$1"
    awk -v f="$file" '
        /^[[:space:]]*#\[cfg\((all\()?test[,)]/ { exit }
        # Enter extern "C" block: suppress the opener and set state.
        /extern[[:space:]]+"C"/ { in_extern_c = 1; next }
        # Inside extern "C": suppress; the first column-0 `}` closes the block.
        in_extern_c == 1 {
            if ($0 ~ /^\}/) { in_extern_c = 0 }
            next
        }
        {
            # Emit a comment line only when inside a fn body. The in-fn test uses
            # state from PRECEDING lines, so a comment on the fn-signature line
            # (not yet inside the body) is correctly excluded.
            if ($0 ~ /\/\// && fn_depth_count > 0) printf "%s:%d:%s\n", f, NR, $0
            # Update brace / fn-context state for subsequent lines.
            if ($0 ~ /(^|[^A-Za-z0-9_])fn[[:space:]]+[A-Za-z_]/) pending_fn = 1
            rest = $0
            while (match(rest, /[{}]/)) {
                ch = substr(rest, RSTART, 1)
                if (ch == "{") {
                    depth++
                    if (pending_fn) { fn_open[++fn_depth_count] = depth; pending_fn = 0 }
                } else {
                    if (fn_depth_count > 0 && depth == fn_open[fn_depth_count]) fn_depth_count--
                    if (depth > 0) depth--
                }
                rest = substr(rest, RSTART + 1)
            }
        }
    ' "$file"
}

# From a pre-filtered `FILE:LINE:CONTENT` stream, emit ONE entry per flagged
# comment line — the divider line is the unit of finding — preserving the
# `FILE:LINE:` location prefix and showing the first matched glyph-run span as
# the representative token. Emits `FILE:LINE: SPAN`. One-per-line keeps the hit
# count at divider granularity (a banner is one finding, not one-per-glyph).
extract_tokens() {
    local line file_part rest line_part content token
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        file_part="${line%%:*}"
        rest="${line#*:}"
        line_part="${rest%%:*}"
        content="${rest#*:}"
        token="$(printf '%s\n' "$content" | grep -oP "$GLYPH_PATTERN" | head -n 1 || true)"
        [[ -n "$token" ]] || continue
        printf '%s:%s: %s\n' "$file_part" "$line_part" "$token"
    done
}

# Build a single combined comment stream across every production .rs file, in
# `FILE:LINE:CONTENT` form with the cfg(test) tail block and extern "C" blocks
# already removed.
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

# Pre-filter to candidate lines (cheap), then extract the matched glyph spans.
hits=""
candidate_lines="$(printf '%s\n' "$comment_stream" \
    | grep -P "$GLYPH_PATTERN" || true)"
if [[ -n "$candidate_lines" ]]; then
    hits="$(printf '%s\n' "$candidate_lines" | extract_tokens || true)"
fi
hits="${hits%$'\n'}"

if [[ -n "$hits" ]]; then
    hit_count="$(printf '%s\n' "$hits" | grep -c '' || true)"
    echo "ADVISORY: box-drawing banners inside fn bodies (${hit_count} N5 candidates)"
    echo ""
    echo "$hits"
    echo ""
    echo "Each banner fences groups inside a function — an N5 signal that the"
    echo "function wants splitting. The remedy is to extract a function, not to"
    echo "delete the divider. Advisory only — this does not fail the build."
    exit 0
fi

echo "OK: no in-fn box-drawing banners found in shipped comments."
exit 0
