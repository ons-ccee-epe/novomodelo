"""Unmigrated-input overlays: patch a copy of a migrated committed deck back to its legacy twin.

Python counterpart of `crates/novomodelo-sddp/src/test_support/unmigrated_inputs.rs`; both apply
generations in the same order. An overlay is a directory holding the files the legacy deck had
plus `absent.json`, a JSON array of the relative paths it did not have. Overlays live under
`<OVERLAY_ROOT>/<generation>/<flat deck key>/`, where the flat key replaces `/` with `__`, and
patch the deck as the next-newer generation left it. Materializing generation `g` therefore
applies every newer generation first, newest to oldest. A missing generation directory or deck
overlay means "no overlay".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

GENERATIONS: tuple[str, ...] = ("initial_conditions", "post_study_stages_key")
OVERLAY_ROOT = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "novomodelo-sddp"
    / "tests"
    / "unswept_fixtures"
    / "unmigrated_inputs"
)
ABSENT_FILE = "absent.json"
LEGACY_MARKER = "initial_conditions.json"
MIGRATED_MARKER = "boundary_conditions.json"


def _overlay(root: Path, generation: str, deck_key: str) -> Path:
    return root / generation / deck_key.replace("/", "__")


def _read_absent(overlay: Path) -> list[str]:
    absent = json.loads((overlay / ABSENT_FILE).read_text(encoding="utf-8"))
    if not isinstance(absent, list) or not all(
        isinstance(path, str) for path in absent
    ):
        raise ValueError(f"{overlay / ABSENT_FILE} is not a JSON array of strings")
    return absent


def _materialize(overlay: Path, deck: Path) -> None:
    absent = _read_absent(overlay)
    for file in sorted(overlay.rglob("*")):
        relative = file.relative_to(overlay)
        if file.is_file() and relative != Path(ABSENT_FILE):
            (deck / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, deck / relative)
    for relative_path in absent:
        (deck / relative_path).unlink(missing_ok=True)


def apply_newer(
    generation: str, deck_key: str, deck: Path, root: Path = OVERLAY_ROOT
) -> None:
    """Apply, newest first, each generation newer than `generation` that has an overlay for the deck.

    Raises `ValueError` for an unknown generation or a malformed `absent.json`.
    """
    for newer in GENERATIONS[: GENERATIONS.index(generation)]:
        overlay = _overlay(root, newer, deck_key)
        if overlay.is_dir():
            _materialize(overlay, deck)


def check_legacy_twin(deck: Path) -> None:
    """Raise `ValueError` naming the offending file unless `deck` is a legacy twin."""
    if not (deck / LEGACY_MARKER).exists():
        raise ValueError(f"{deck} is not a legacy twin: {LEGACY_MARKER} is missing")
    if (deck / MIGRATED_MARKER).exists():
        raise ValueError(f"{deck} is not a legacy twin: {MIGRATED_MARKER} is present")


def materialize_twin(
    generation: str, deck_key: str, deck: Path, root: Path = OVERLAY_ROOT
) -> None:
    """Patch `deck`, a copy of the committed deck, into its legacy twin at `generation`.

    Nothing is written when `generation` has no overlay for the deck: `FileNotFoundError`.
    A malformed `absent.json` raises `ValueError`; a result that is not a legacy twin raises
    `ValueError` through `check_legacy_twin`.
    """
    own = _overlay(root, generation, deck_key)
    if not own.is_dir():
        raise FileNotFoundError(f"no {generation} overlay for {deck_key} under {root}")
    apply_newer(generation, deck_key, deck, root)
    _materialize(own, deck)
    check_legacy_twin(deck)


def deck_keys(generation: str, root: Path = OVERLAY_ROOT) -> list[str]:
    """The sorted, unflattened deck keys that have an overlay in `generation`."""
    directory = root / generation
    if not directory.is_dir():
        return []
    return sorted(
        path.name.replace("__", "/") for path in directory.iterdir() if path.is_dir()
    )
