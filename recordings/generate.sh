#!/usr/bin/env bash
#
# generate.sh — (re)generate the terminal-recording GIFs in this directory.
#
# Each VHS tape is run from WITHIN recordings/, so `Output <name>.gif` lands
# here and the `demo/` case that the tape's `novomodelo init demo/` creates lands in
# recordings/demo/ (which is gitignored). That temporary demo/ — and any stray
# tmp.json a tape's jq step leaves behind — is removed BEFORE and AFTER every
# tape. Cleaning outside the tape is deliberate: it keeps every helper command
# out of the recording, so a GIF shows only the demo it is meant to show.
#
# Usage:
#   ./recordings/generate.sh              # regenerate every tape
#   ./recordings/generate.sh quickstart   # just quickstart.tape (name or path)
set -euo pipefail

# CDPATH can hijack a bare `cd recordings` if the user exports it; neutralise it,
# then resolve to this script's own directory so paths (Output, demo/) are
# correct no matter where the script is invoked from.
unset CDPATH
cd -- "$(dirname -- "$0")"

log() { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# The transient artifacts a tape produces while recording. Removed around every
# run so the working tree keeps only the committed .tape / .gif files.
clean_temp() { rm -rf demo/ tmp.json; }

# --- preflight ---------------------------------------------------------------

have vhs || die "vhs not found — run ./recordings/setup.sh first"
have novomodelo || die "novomodelo not on PATH — 'cargo install --path crates/novomodelo-cli' (or add target/release to PATH)"
have jq || die "jq not found — validation-error.tape and multithreading.tape need it (install via your package manager)"

# --- select tapes ------------------------------------------------------------

tapes=()
if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    tape="$(basename "${name%.tape}").tape"
    [ -f "$tape" ] || die "no such tape: $tape"
    tapes+=("$tape")
  done
else
  for tape in *.tape; do
    tapes+=("$tape")
  done
fi
[ "${#tapes[@]}" -gt 0 ] || die "no .tape files found in $(pwd)"

# --- generate ----------------------------------------------------------------

trap clean_temp EXIT
for tape in "${tapes[@]}"; do
  log "recording ${tape%.tape}"
  clean_temp # clean slate so `novomodelo init demo/` starts fresh, never prompting
  vhs "$tape"
done
clean_temp

log "done — regenerated ${#tapes[@]} recording(s)"
log "if a GIF changed, vendor it into novomodelo-docs:  npm run refresh:recordings -- --ref <tag>"
