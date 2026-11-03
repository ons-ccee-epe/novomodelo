#!/usr/bin/env python3
"""Check that the Python binding crate is versioned in lockstep with the workspace.

`crates/novomodelo-python` is outside the Cargo workspace, so it cannot inherit
`[workspace.package] version` and carries a `version` of its own, bumped by hand at
each release. maturin takes the wheel's version from that field, so a missed bump
publishes a wheel whose version differs from the release.

Compares the workspace version with:
  1. `[package] version` in `crates/novomodelo-python/Cargo.toml`;
  2. the version of every first-party package (name starting `novomodelo`, no
     `source`) in `crates/novomodelo-python/Cargo.lock`, which `cargo audit` reads
     as committed.

Usage:
    python3 scripts/ci/check_binding_version.py
    python3 scripts/ci/check_binding_version.py --root /path/to/repo

Exit code 0 if every version equals the workspace version, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

BINDING = "crates/novomodelo-python"


def load(root: Path, relative: str) -> dict:
    with (root / relative).open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    root = parser.parse_args().root

    expected = load(root, "Cargo.toml")["workspace"]["package"]["version"]
    manifest = load(root, f"{BINDING}/Cargo.toml")
    found = {f"{BINDING}/Cargo.toml": manifest["package"]["version"]}
    for package in load(root, f"{BINDING}/Cargo.lock")["package"]:
        if package["name"].startswith("novomodelo") and "source" not in package:
            found[f"{BINDING}/Cargo.lock ({package['name']})"] = package["version"]

    mismatches = {w: v for w, v in found.items() if v != expected}
    if mismatches:
        print(f"FAIL: workspace version is {expected}; these differ:")
        for where, version in mismatches.items():
            print(f"  {where}: {version}")
        return 1

    print(f"OK: {len(found)} binding versions equal the workspace version {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
