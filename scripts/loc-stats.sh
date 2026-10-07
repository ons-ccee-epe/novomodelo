#!/usr/bin/env bash
# loc-stats.sh — Rust line & test-count statistics for the novomodelo workspace.
#
# Splits every tracked .rs line into PRODUCTION vs TEST, per crate and global,
# and counts test functions. Refreshable: run it any time to get current numbers.
#
# What counts as TEST code:
#   - any file under a tests/ or benches/ directory (integration tests, benches)
#   - sibling test modules: files named tests.rs or test_support.rs
#     (the `#[cfg(test)] mod tests;` / `mod test_support;` pattern)
#   - inline `#[cfg(test)] mod … { … }` blocks inside source files
# Everything else is PRODUCTION. Inline-block detection trusts cargo fmt's
# canonical indentation: a #[cfg(test)] item's closing brace is a `}` at the
# item's own indent. The tree is always fmt-clean (CI enforces it), so this is
# reliable.
#
# "code" = physical lines minus blank and //-comment lines; "all" = physical.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/loc-stats.sh [options]

Rust line counts (production vs test) and test-function counts, per crate and
global, for the novomodelo workspace.

Options:
  --worktree   Count working-tree files (find) instead of git-tracked files.
  --nextest    Also print the runtime test count via `cargo nextest list`
               (compiles the workspace; authoritative, excludes doctests).
  --csv        Emit CSV instead of formatted tables.
  -h, --help   Show this help.

Notes:
  * benches/ files are bucketed as TEST (non-production tooling).
  * TEST_FNS counts static `#[test]`/`#[tokio::test]`/`#[rstest]` annotations;
    it excludes proptest cases and cfg-gated expansion — use --nextest for the
    authoritative runtime count.
EOF
}

WORKTREE=0
NEXTEST=0
CSV=0
for a in "$@"; do
  case "$a" in
    --worktree) WORKTREE=1 ;;
    --nextest)  NEXTEST=1 ;;
    --csv)      CSV=1 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "loc-stats: unknown option: $a" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ "$WORKTREE" -eq 1 ]; then
  mapfile -t FILES < <(find . -name '*.rs' -not -path '*/target/*' -not -path './.git/*' | sed 's#^\./##' | sort)
else
  mapfile -t FILES < <(git ls-files '*.rs' | sort)
fi
[ "${#FILES[@]}" -gt 0 ] || { echo "loc-stats: no .rs files found under $ROOT" >&2; exit 1; }

awk -v csv="$CSV" '
function crate_of(p,   c) {
  if (p ~ /^crates\//) { c = substr(p, 8); sub(/\/.*/, "", c); return c }
  c = p; sub(/\/.*/, "", c); return "(" c ")"
}
function is_testfile(p,   b) {
  if (p ~ /(^|\/)tests\//)   return 1
  if (p ~ /(^|\/)benches\//) return 1
  b = p; sub(/.*\//, "", b)
  if (b == "tests.rs")        return 1
  if (b == "test_support.rs") return 1
  return 0
}
function indent_of(s) { match(s, /^ */); return RLENGTH }

FNR == 1 {
  crate = crate_of(FILENAME)
  ftype = is_testfile(FILENAME)   # 1 = whole-file test, 0 = source file
  in_test = 0; pend = 0; tind = -1
  if (!(crate in seen)) { seen[crate] = 1; keys[++nk] = crate }
  files[crate]++; gfiles++
}
{
  t = $0; sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
  blank = (t == "")
  comment = (t ~ /^\/\//)

  tl = 0
  if (ftype == 1) {
    tl = 1
  } else if (in_test) {
    tl = 1
    if (t ~ /^}/ && indent_of($0) == tind) in_test = 0
  } else if (pend) {
    tl = 1
    if ($0 ~ /{/)      { in_test = 1; pend = 0 }
    else if (t ~ /;$/) { pend = 0 }
  } else if (t ~ /^#\[cfg/ && t ~ /\(test[,)]/ && t !~ /not[ ]*\(test/) {
    tl = 1; tind = indent_of($0)
    if ($0 ~ /{/)      in_test = 1
    else if (t ~ /;$/) { }        # inline `#[cfg(test)] mod x;` — one line
    else               pend = 1
  }

  if (tl) {
    ta[crate]++; gta++
    if (!blank && !comment) { tc[crate]++; gtc++ }
  } else {
    pa[crate]++; gpa++
    if (!blank && !comment) { pc[crate]++; gpc++ }
  }

  if (t ~ /^#\[test\]/ || t ~ /^#\[tokio::test/ || t ~ /^#\[rstest/) { tf[crate]++; gtf++ }
}
END {
  for (i = 2; i <= nk; i++) {           # insertion sort crate names
    x = keys[i]; j = i - 1
    while (j >= 1 && keys[j] > x) { keys[j+1] = keys[j]; j-- }
    keys[j+1] = x
  }
  if (csv == "1") {
    print "crate,files,prod_code,prod_all,test_code,test_all,test_fns"
    for (i = 1; i <= nk; i++) { c = keys[i];
      printf "%s,%d,%d,%d,%d,%d,%d\n", c, files[c]+0, pc[c]+0, pa[c]+0, tc[c]+0, ta[c]+0, tf[c]+0 }
    printf "TOTAL,%d,%d,%d,%d,%d,%d\n", gfiles, gpc, gpa, gtc, gta, gtf
    exit
  }
  printf "%-16s %6s %11s %10s %11s %10s %9s\n", "CRATE", "FILES", "PROD_CODE", "PROD_ALL", "TEST_CODE", "TEST_ALL", "TEST_FNS"
  printf "%-16s %6s %11s %10s %11s %10s %9s\n", "----------------", "-----", "----------", "---------", "----------", "---------", "--------"
  for (i = 1; i <= nk; i++) { c = keys[i];
    printf "%-16s %6d %11d %10d %11d %10d %9d\n", c, files[c]+0, pc[c]+0, pa[c]+0, tc[c]+0, ta[c]+0, tf[c]+0 }
  printf "%-16s %6s %11s %10s %11s %10s %9s\n", "----------------", "-----", "----------", "---------", "----------", "---------", "--------"
  printf "%-16s %6d %11d %10d %11d %10d %9d\n", "TOTAL", gfiles, gpc, gpa, gtc, gta, gtf
  gall = gpa + gta
  gcode = gpc + gtc
  printf "\nTotals: %d physical lines (%d code) across %d files.\n", gall, gcode, gfiles
  if (gall > 0)  printf "  production %d (%.1f%%) | test %d (%.1f%%) — physical lines\n", gpa, 100*gpa/gall, gta, 100*gta/gall
  if (gcode > 0) printf "  production %d (%.1f%%) | test %d (%.1f%%) — code lines\n", gpc, 100*gpc/gcode, gtc, 100*gtc/gcode
}
' "${FILES[@]}"

if [ "$NEXTEST" -eq 1 ]; then
  echo
  echo "Runtime test count (cargo nextest list; HiGHS backend, doctests excluded):"
  if ! command -v cargo-nextest >/dev/null 2>&1 && ! cargo nextest --version >/dev/null 2>&1; then
    echo "  cargo-nextest not installed — skipping (install: cargo install cargo-nextest)" >&2
  else
    NEXTEST_FEATURES="mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"
    cargo nextest list --workspace --features "$NEXTEST_FEATURES" 2>/dev/null \
      | awk '
          { split($1, a, "::"); crate = a[1]; n[crate]++; total++
            if (!(crate in seen)) { seen[crate] = 1; keys[++nk] = crate } }
          END {
            for (i = 2; i <= nk; i++) { x = keys[i]; j = i-1;
              while (j >= 1 && keys[j] > x) { keys[j+1] = keys[j]; j-- }; keys[j+1] = x }
            for (i = 1; i <= nk; i++) printf "  %-18s %6d\n", keys[i], n[keys[i]]
            printf "  %-18s %6d\n", "TOTAL", total
          }'
  fi
fi
