#!/usr/bin/env python3
"""Inject repo-root license/notice files into each built wheel and sdist.

maturin builds the `novomodelo-python` wheels and sdist from `crates/novomodelo-python`,
but the canonical license texts (LICENSE, NOTICE, THIRD_PARTY_NOTICES.md) live at the
repository root. Neither PEP 639 `license-files` globs nor maturin `include` patterns
can reference paths above the project directory, so this post-build step adds the
files to each artifact instead of duplicating them into the crate:

- a wheel gets them in `<dist-info>/licenses/`, the PEP 639 location pip installs to.
  It is unpacked and repacked with `python -m wheel`, which recomputes `RECORD` so the
  wheel stays valid;
- an sdist gets them at its root, next to `PKG-INFO`.

Usage:
    inject_dist_licenses.py <dist-dir> <license-file>...
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def inject_wheel(wheel: Path, license_files: list[Path]) -> None:
    dist_dir = wheel.parent
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
            [sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(dist_dir)],
            check=True,
        )


def inject_sdist(sdist: Path, license_files: list[Path]) -> None:
    repacked = sdist.with_name(sdist.name + ".tmp")
    with tarfile.open(sdist, "r:gz") as src:
        members = src.getmembers()
        root = members[0].name.split("/", 1)[0]
        injected = {f"{root}/{lf.name}" for lf in license_files}
        with tarfile.open(repacked, "w:gz", format=tarfile.PAX_FORMAT) as dst:
            for member in members:
                if member.name not in injected:
                    dst.addfile(
                        member, src.extractfile(member) if member.isfile() else None
                    )
            for lf in license_files:
                dst.add(lf, arcname=f"{root}/{lf.name}")
    repacked.replace(sdist)


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
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if not wheels and not sdists:
        print(f"error: no wheels or sdists found in {dist_dir}", file=sys.stderr)
        return 1

    for wheel in wheels:
        inject_wheel(wheel, license_files)
        print(f"injected {len(license_files)} license files into {wheel.name}")
    for sdist in sdists:
        inject_sdist(sdist, license_files)
        print(f"injected {len(license_files)} license files into {sdist.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
