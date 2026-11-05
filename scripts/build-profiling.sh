#!/usr/bin/env bash
#
# build-profiling.sh — build with the `profiling` profile into target/perf/,
# linked so that `perf record --call-graph dwarf` can unwind the binary.
#
# Usage:
#   scripts/build-profiling.sh                     # the CLI: target/perf/profiling/novomodelo
#   scripts/build-profiling.sh <cargo build args>  # e.g. -p novomodelo-sddp --example dhat_baseline
#
# Runs from anywhere: it changes to the repo root so rustup picks the toolchain
# pinned in rust-toolchain.toml. See docs/profiling.md.

set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -eq 0 ]]; then
    set -- --bin novomodelo
fi

# perf's libdw unwinder rejects rust-lld's segment layout as an overlapping
# module and truncates every stack; GNU ld's layout unwinds.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-Clinker-features=-lld"
# Its own target dir, so the RUSTFLAGS change does not rebuild target/profiling.
export CARGO_TARGET_DIR="$repo_root/target/perf"

exec cargo build --profile profiling "$@"
