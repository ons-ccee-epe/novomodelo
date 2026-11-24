"""The committed migrated decks are exactly the script's output on their unmigrated inputs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import legacy_overlays as lo
import post_study_stages as pss

_ROOT = Path(__file__).resolve().parents[2]
_GENERATION = "post_study_stages_key"

_DECK_KEYS = lo.deck_keys(_GENERATION)
assert _DECK_KEYS, f"no {_GENERATION} unmigrated-input overlay under {lo.OVERLAY_ROOT}"


def is_unmigrated(deck: Path) -> bool:
    path = deck / pss.FILE_NAME
    if not path.exists():
        return True
    doc = json.loads(path.read_text(encoding="utf-8"))
    return pss.classify_file(doc) is pss.FileFormat.LEGACY


@pytest.mark.parametrize("deck_key", _DECK_KEYS, ids=lambda key: key.replace("/", "__"))
def test_main_on_the_unmigrated_inputs_reproduces_the_committed_migrated_decks(
    tmp_path: Path, deck_key: str
) -> None:
    committed = _ROOT / deck_key
    expected_deck = tmp_path / "expected"
    shutil.copytree(
        committed, expected_deck, ignore=shutil.ignore_patterns("scenarios")
    )
    lo.apply_newer(_GENERATION, deck_key, expected_deck)
    expected = (expected_deck / pss.FILE_NAME).read_bytes()

    legacy = tmp_path / "deck"
    shutil.copytree(committed, legacy, ignore=shutil.ignore_patterns("scenarios"))
    lo.materialize_twin(_GENERATION, deck_key, legacy)
    assert is_unmigrated(legacy)

    assert pss.main([str(legacy)]) == 0

    assert (legacy / pss.FILE_NAME).read_bytes() == expected
