#!/usr/bin/env python3
"""Inject repo-root license/notice files into each built wheel.

maturin builds the `novomodelo-python` wheels from `crates/novomodelo-python`, but the
canonical license texts (LICENSE, NOTICE, THIRD_PARTY_NOTICES.md,
THIRD_PARTY_LICENSES.md) live at the repository root. PEP 639 `license-files`
globs cannot reference paths above the project directory, so this post-build
step drops the files into each wheel's `<dist-info>/licenses/` directory — the
PEP 639 location pip installs to — instead of duplicating them into the crate.

Each wheel is unpacked and repacked with `python -m wheel`, which recomputes
`RECORD` so the wheel stays valid.

Usage:
    inject_wheel_licenses.py <dist-dir> <license-file>...
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    dist_dir = Path(sys.argv[1])
    license_files = [Path(p) for p in sys.argv[2:]]

    missing = [str(p) for p in license_files if not p.is_file()]
    if missing:
        print(f"error: license files not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"error: no wheels found in {dist_dir}", file=sys.stderr)
        return 1

    for wheel in wheels:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", tmp],
                check=True,
            )
            unpacked = next(p for p in Path(tmp).iterdir() if p.is_dir())
            dist_info = next(unpacked.glob("*.dist-info"))
            licenses_dir = dist_info / "licenses"
            licenses_dir.mkdir(exist_ok=True)
            for lf in license_files:
                shutil.copy2(lf, licenses_dir / lf.name)
            # Remove the original so `wheel pack` writes a fresh, RECORD-correct
            # archive into the same directory without colliding.
            wheel.unlink()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wheel",
                    "pack",
                    str(unpacked),
                    "-d",
                    str(dist_dir),
                ],
                check=True,
            )
        print(f"injected {len(license_files)} license files into {wheel.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
