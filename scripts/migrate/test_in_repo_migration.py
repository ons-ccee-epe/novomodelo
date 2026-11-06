"""The committed migrated decks are exactly the script's output on their legacy inputs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import post_study_stages as pss

_ROOT = Path(__file__).resolve().parents[2]
_OVERLAY_ROOT = (
    _ROOT / "crates" / "novomodelo-sddp" / "tests" / "gate_fixtures" / "legacy_inputs"
)
_ABSENT = "absent.json"

_OVERLAYS = sorted(path for path in _OVERLAY_ROOT.iterdir() if path.is_dir())
assert _OVERLAYS, f"no legacy-input overlay under {_OVERLAY_ROOT}"


def materialize(overlay: Path, deck: Path) -> None:
    absent: list[str] = json.loads((overlay / _ABSENT).read_text(encoding="utf-8"))
    for file in sorted(overlay.rglob("*")):
        relative = file.relative_to(overlay)
        if file.is_file() and relative != Path(_ABSENT):
            (deck / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, deck / relative)
    for relative_path in absent:
        (deck / relative_path).unlink(missing_ok=True)


def is_unmigrated(deck: Path) -> bool:
    path = deck / pss.FILE_NAME
    if not path.exists():
        return True
    doc = json.loads(path.read_text(encoding="utf-8"))
    return pss.classify_file(doc) is pss.FileFormat.LEGACY


@pytest.mark.parametrize("overlay", _OVERLAYS, ids=lambda path: path.name)
def test_main_on_the_legacy_inputs_reproduces_the_committed_migrated_decks(
    tmp_path: Path, overlay: Path
) -> None:
    committed = _ROOT / overlay.name.replace("__", "/")
    legacy = tmp_path / "deck"
    shutil.copytree(committed, legacy, ignore=shutil.ignore_patterns("scenarios"))
    materialize(overlay, legacy)
    assert is_unmigrated(legacy)

    assert pss.main([str(legacy)]) == 0

    migrated = committed / pss.FILE_NAME
    assert (legacy / pss.FILE_NAME).read_bytes() == migrated.read_bytes()
