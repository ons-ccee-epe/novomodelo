"""Tests for legacy_overlays.py over overlay roots built in a temporary directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import legacy_overlays as lo

DECK = "examples/deck"
NEWER = "initial_conditions"
OLDER = "post_study_stages_key"


def write(root: Path, relative: str, content: str | bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def write_overlay(
    root: Path,
    generation: str,
    deck: str,
    absent: list[str],
    files: dict[str, str | bytes],
) -> None:
    flat = deck.replace("/", "__")
    write(root, f"{generation}/{flat}/{lo.ABSENT_FILE}", json.dumps(absent))
    for relative, content in files.items():
        write(root, f"{generation}/{flat}/{relative}", content)


def migrated_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "copy"
    write(copy, "boundary_conditions.json", "migrated")
    write(copy, "post_study_stages.json", "migrated")
    return copy


def test_newer_generation_is_applied_first(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = migrated_copy(tmp_path)
    write_overlay(
        root,
        NEWER,
        DECK,
        ["boundary_conditions.json"],
        {
            "initial_conditions.json": "ic",
            "stages.json": "newer stages",
            "post_study_stages.json": "newer",
        },
    )
    write_overlay(
        root, OLDER, DECK, ["stages.json"], {"post_study_stages.json": "older"}
    )

    lo.materialize_twin(OLDER, DECK, copy, root)

    assert (copy / "post_study_stages.json").read_text(encoding="utf-8") == "older"
    assert (copy / "initial_conditions.json").read_text(encoding="utf-8") == "ic"
    assert not (copy / "stages.json").exists()
    assert not (copy / "boundary_conditions.json").exists()


def test_missing_newer_overlay_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = tmp_path / "copy"
    write(copy, "initial_conditions.json", "ic")
    write(copy, "post_study_stages.json", "migrated")
    write_overlay(root, NEWER, "examples/other", [], {"other.json": "other"})
    write_overlay(root, OLDER, DECK, [], {"post_study_stages.json": "older"})

    lo.materialize_twin(OLDER, DECK, copy, root)

    assert (copy / "post_study_stages.json").read_text(encoding="utf-8") == "older"
    assert not (copy / "other.json").exists()


def test_missing_requested_overlay_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = migrated_copy(tmp_path)
    write_overlay(root, NEWER, DECK, [], {"initial_conditions.json": "ic"})

    with pytest.raises(FileNotFoundError, match=OLDER):
        lo.materialize_twin(OLDER, DECK, copy, root)

    assert not (copy / "initial_conditions.json").exists()


def test_missing_generation_directory_is_no_overlay(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = migrated_copy(tmp_path)

    lo.apply_newer(OLDER, DECK, copy, root)

    assert (copy / "post_study_stages.json").read_text(encoding="utf-8") == "migrated"
    assert lo.deck_keys(NEWER, root) == []
    with pytest.raises(FileNotFoundError):
        lo.materialize_twin(NEWER, DECK, copy, root)


def test_twin_holding_boundary_conditions_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = migrated_copy(tmp_path)
    write(copy, "initial_conditions.json", "ic")
    write_overlay(root, OLDER, DECK, [], {})

    with pytest.raises(ValueError, match="boundary_conditions.json"):
        lo.materialize_twin(OLDER, DECK, copy, root)


def test_twin_without_initial_conditions_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = tmp_path / "copy"
    write(copy, "post_study_stages.json", "migrated")
    write_overlay(root, OLDER, DECK, [], {})

    with pytest.raises(ValueError, match="initial_conditions.json"):
        lo.materialize_twin(OLDER, DECK, copy, root)


def test_expected_file_comes_from_the_newer_generation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = migrated_copy(tmp_path)
    newer_bytes = b'{"stages": []}\r\n\x00newer'
    write_overlay(
        root,
        NEWER,
        DECK,
        ["boundary_conditions.json"],
        {"initial_conditions.json": "ic", "post_study_stages.json": newer_bytes},
    )
    write_overlay(root, OLDER, DECK, [], {"post_study_stages.json": b"older"})

    lo.apply_newer(OLDER, DECK, copy, root)

    assert (copy / "post_study_stages.json").read_bytes() == newer_bytes
    assert not (copy / "boundary_conditions.json").exists()


def test_absent_paths_are_deleted_and_a_missing_one_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    copy = tmp_path / "copy"
    write(copy, "post_study_stages.json", "migrated")
    write(copy, "source/post_study_stages.json", "migrated")
    write(copy, "source/config.json", "kept")
    write_overlay(
        root,
        NEWER,
        DECK,
        ["post_study_stages.json", "source/post_study_stages.json", "missing.json"],
        {"sub/new.json": "legacy new"},
    )

    lo.apply_newer(OLDER, DECK, copy, root)

    assert not (copy / "post_study_stages.json").exists()
    assert not (copy / "source/post_study_stages.json").exists()
    assert (copy / "source/config.json").read_text(encoding="utf-8") == "kept"
    assert (copy / "sub/new.json").read_text(encoding="utf-8") == "legacy new"
    assert not (copy / lo.ABSENT_FILE).exists()


def test_malformed_absent_list_raises_before_any_file_is_written(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    copy = tmp_path / "copy"
    write(copy, "a.json", "migrated")
    write(
        root,
        f"{NEWER}/{DECK.replace('/', '__')}/{lo.ABSENT_FILE}",
        '{"not": "an array"}',
    )
    write(root, f"{NEWER}/{DECK.replace('/', '__')}/a.json", "legacy")

    with pytest.raises(ValueError, match="JSON array of strings"):
        lo.apply_newer(OLDER, DECK, copy, root)

    assert (copy / "a.json").read_text(encoding="utf-8") == "migrated"


def test_deck_keys_are_sorted_and_unflattened(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for deck in ["examples/b", "crates/a/tests/c"]:
        write_overlay(root, OLDER, deck, [], {})

    assert lo.deck_keys(OLDER, root) == ["crates/a/tests/c", "examples/b"]
