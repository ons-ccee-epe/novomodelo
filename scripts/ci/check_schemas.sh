#!/usr/bin/env bash
# Verify that schemas/ is in sync with what `novomodelo schema export` produces.
# Exits non-zero on any drift, printing the diff for the failing files.
#
# Usage:
#   scripts/ci/check_schemas.sh        — assumes ./target/release/novomodelo is built
#   scripts/ci/check_schemas.sh --build  — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

BIN="$REPO_ROOT/target/release/novomodelo"
if [[ $BUILD -eq 1 || ! -x "$BIN" ]]; then
  cargo build --release --bin novomodelo
fi

TMP_DIR="$(mktemp -d -t novomodelo-schemas-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

"$BIN" schema export --output-dir "$TMP_DIR" > /dev/null

if diff -ruN "$REPO_ROOT/schemas/" "$TMP_DIR/" > "$TMP_DIR/drift.diff"; then
  echo "Schemas in schemas/ match \`novomodelo schema export\` output. ✓"
  exit 0
fi

echo "ERROR: schemas/ drifts from \`novomodelo schema export\` output."
echo "Run \`scripts/ci/check_schemas.sh --build\` locally, then regenerate with:"
echo "  cargo build --release --bin novomodelo"
echo "  ./target/release/novomodelo schema export --output-dir schemas"
echo ""
echo "--- Drift diff ---"
cat "$TMP_DIR/drift.diff"
exit 1
