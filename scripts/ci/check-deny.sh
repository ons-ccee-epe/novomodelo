#!/usr/bin/env bash
#
# check-deny.sh — cargo-deny over both dependency graphs, at the version CI runs.
#
# Runs every check (advisories, bans, licenses, sources) on the workspace, then
# every check on crates/novomodelo-python against the root deny.toml: that crate is
# outside the workspace and has its own Cargo.lock, so the workspace check never
# sees its graph.
#
# cargo-deny's command line changes between releases (0.20 moved --config from
# `check` to the top level), so the script refuses any version other than
# CARGO_DENY_VERSION. CI installs the version `--print-version` prints, so a
# local run executes the same commands CI does.
#
# Usage: check-deny.sh
#        check-deny.sh --print-version
#
# Exit codes:
#   0 — Pass: both graphs passed.
#   1 — Fail: a check failed. Both graphs are checked either way.
#   2 — Refused: bad invocation, or cargo-deny is missing or at another version.

set -euo pipefail

readonly CARGO_DENY_VERSION="0.20.2"

if [[ $# -eq 1 && "$1" == "--print-version" ]]; then
    echo "${CARGO_DENY_VERSION}"
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "usage: $0 [--print-version]" >&2
    exit 2
fi

readonly INSTALL_HINT="install it with: cargo install cargo-deny --version ${CARGO_DENY_VERSION} --locked"

if ! command -v cargo-deny >/dev/null 2>&1; then
    echo "error: cargo-deny not found; ${INSTALL_HINT}" >&2
    exit 2
fi

installed="$(cargo-deny --version | awk '{print $2}')"
if [[ "${installed}" != "${CARGO_DENY_VERSION}" ]]; then
    echo "error: cargo-deny ${installed} found, ${CARGO_DENY_VERSION} required; ${INSTALL_HINT}" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

status=0

echo "==> cargo-deny: workspace (all checks)"
cargo-deny --all-features check || status=1

echo "==> cargo-deny: crates/novomodelo-python (all checks)"
cargo-deny --manifest-path crates/novomodelo-python/Cargo.toml --all-features \
    --config deny.toml check || status=1

exit "${status}"
