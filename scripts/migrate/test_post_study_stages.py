"""Tests for post_study_stages.py: classification, inference, planning and the CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import post_study_stages as pss

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "deterministic"
_SCHEMA = "schema-uri"
_ARC_DECK = "d45-travel-time-mixed-calendar"
_NO_ARC_DECK = "d01-thermal-dispatch"
_POST_STUDY_WARNING = "warning: post_study_stages.json"

_BOUNDS = [
    {
        "thermal_id": 1,
        "post_study_stage_index": 0,
        "cost_per_mwh": 5.0,
        "min_mw": 0.0,
        "max_mw": 150.0,
    }
]
_LEGACY: dict[str, Any] = {
    "$schema": _SCHEMA,
    "stages": [{"start_date": "2024-03-01", "duration_hours": 744.0}],
    "thermal_bounds": _BOUNDS,
}
_ANTICIPATED: list[dict[str, Any]] = [
    {"id": 0},
    {"id": 1, "anticipated_config": {"lead_time_hours": 1000.0}},
]
_NOT_ANTICIPATED: list[dict[str, Any]] = [{"id": 0, "anticipated_config": None}]


def stage(sid: int, start: str, end: str, hours: float) -> dict[str, Any]:
    blocks = [{"id": 0, "name": "SINGLE", "hours": hours}]
    return {"id": sid, "start_date": start, "end_date": end, "blocks": blocks}


_WEEKLY = [
    stage(0, "2024-01-01", "2024-01-08", 168),
    stage(1, "2024-01-08", "2024-01-15", 168),
]
_MONTHLY = [
    stage(0, "2024-01-01", "2024-02-01", 744),
    stage(1, "2024-02-01", "2024-03-01", 696),
]


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def make_deck(
    root: Path,
    name: str = "deck",
    *,
    stages: list[dict[str, Any]],
    travel_time_hours: float | None = None,
    thermals: list[dict[str, Any]] | None = None,
    post_study: dict[str, Any] | str | None = None,
) -> Path:
    deck = root / name
    write_json(deck / "stages.json", {"stages": stages})
    upstream: dict[str, Any] = {"id": 0, "downstream_id": 1}
    if travel_time_hours is not None:
        upstream["travel_time_hours"] = travel_time_hours
    write_json(
        deck / "system" / "hydros.json",
        {"hydros": [upstream, {"id": 1, "downstream_id": None}]},
    )
    write_json(
        deck / "system" / "thermals.json",
        {"thermals": _NOT_ANTICIPATED if thermals is None else thermals},
    )
    if isinstance(post_study, str):
        (deck / pss.FILE_NAME).write_text(post_study, encoding="utf-8")
    elif post_study is not None:
        write_json(deck / pss.FILE_NAME, post_study)
    return deck


def snapshot(deck: Path) -> dict[str, bytes]:
    return {
        path.relative_to(deck).as_posix(): path.read_bytes()
        for path in sorted(deck.rglob("*"))
        if path.is_file()
    }


def run(capsys: pytest.CaptureFixture[str], *decks: Path) -> tuple[int, str, str]:
    code = pss.main([str(deck) for deck in decks])
    out, err = capsys.readouterr()
    return code, out, err


def migrated_doc(deck: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((deck / pss.FILE_NAME).read_text(encoding="utf-8"))
    return doc


def last_stage(start: str, end: str, hours: float) -> pss.LastStage:
    return pss.LastStage(
        date.fromisoformat(start), date.fromisoformat(end), float(hours)
    )


def water(*stages: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"start_date": start, "duration_hours": hours} for start, hours in stages]


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (None, pss.FileFormat.ABSENT),
        ({"stages": [], "thermal_bounds": []}, pss.FileFormat.LEGACY),
        ({"stages": []}, pss.FileFormat.LEGACY),
        ({"thermal_stages": [], "thermal_bounds": []}, pss.FileFormat.CURRENT),
        ({"water_bucket_stages": []}, pss.FileFormat.CURRENT),
        ({"$schema": _SCHEMA}, pss.FileFormat.CURRENT),
        ({}, pss.FileFormat.CURRENT),
        ({"stages": [], "thermal_stages": []}, pss.FileFormat.MIXED),
        ({"stages": [], "water_bucket_stages": []}, pss.FileFormat.MIXED),
    ],
)
def test_classify_file_distinguishes_the_four_formats(
    doc: dict[str, Any] | None, expected: pss.FileFormat
) -> None:
    assert pss.classify_file(doc) is expected


def test_infer_water_stages_week_stage_steps_168_hours_from_the_study_end() -> None:
    got = pss.infer_water_stages(last_stage("2024-02-14", "2024-02-21", 168), 360.0)
    assert got == water(
        ("2024-02-21", 168.0), ("2024-02-28", 168.0), ("2024-03-06", 168.0)
    )


@pytest.mark.parametrize(
    ("last", "deepest", "expected"),
    [
        (
            ("2024-03-01", "2024-04-01", 720),
            1080.0,
            water(("2024-04-01", 720.0), ("2024-05-01", 744.0)),
        ),
        (
            ("2024-03-01", "2024-04-01", 730),
            1460.0,
            water(("2024-04-01", 720.0), ("2024-05-01", 744.0)),
        ),
        (
            ("2024-01-01", "2024-02-01", 744),
            1000.0,
            water(("2024-02-01", 696.0), ("2024-03-01", 744.0)),
        ),
        (
            ("2024-11-01", "2024-12-01", 720),
            1400.0,
            water(("2024-12-01", 744.0), ("2025-01-01", 744.0)),
        ),
        (
            ("2024-02-01", "2024-03-01", 720),
            360.0,
            water(("2024-03-01", 744.0)),
        ),
    ],
    ids=[
        "apr-may",
        "month-with-730h-blocks",
        "leap-february",
        "year-end",
        "single-month",
    ],
)
def test_infer_water_stages_month_stage_steps_by_calendar_months(
    last: tuple[str, str, float], deepest: float, expected: list[dict[str, Any]]
) -> None:
    assert pss.infer_water_stages(last_stage(*last), deepest) == expected


@pytest.mark.parametrize(
    ("last", "deepest", "expected"),
    [
        (
            ("2024-01-15", "2024-02-14", 720),
            200.0,
            water(("2024-02-14", 720.0)),
        ),
        (
            ("2024-01-15", "2024-02-14", 720),
            1000.0,
            water(("2024-02-14", 720.0), ("2024-03-15", 720.0)),
        ),
        (
            ("2024-01-01", "2024-01-04", 72),
            100.0,
            water(("2024-01-04", 72.0), ("2024-01-07", 72.0)),
        ),
        (
            ("2024-01-01", "2024-03-01", 1440),
            100.0,
            water(("2024-03-01", 1440.0)),
        ),
    ],
    ids=["one-repeat", "two-repeats", "three-days", "two-months-from-the-first"],
)
def test_infer_water_stages_other_whole_day_length_repeats_its_exact_length(
    last: tuple[str, str, float], deepest: float, expected: list[dict[str, Any]]
) -> None:
    assert pss.infer_water_stages(last_stage(*last), deepest) == expected


@pytest.mark.parametrize(
    ("deepest", "count"), [(336.0, 2), (336.5, 3), (168.0, 1), (0.5, 1)]
)
def test_infer_water_stages_stops_once_the_total_reaches_the_deepest_arc(
    deepest: float, count: int
) -> None:
    got = pss.infer_water_stages(last_stage("2024-01-08", "2024-01-15", 168), deepest)
    assert len(got) == count


@pytest.mark.parametrize("hours", [700, 0, -24])
def test_infer_water_stages_refuses_a_length_that_is_not_a_whole_number_of_days(
    hours: float,
) -> None:
    with pytest.raises(pss.Refusal, match="whole number of days"):
        pss.infer_water_stages(last_stage("2024-01-15", "2024-02-14", hours), 100.0)


def test_main_arc_deck_without_a_file_gets_only_water_bucket_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(tmp_path, stages=_WEEKLY, travel_time_hours=336.0)

    code, out, err = run(capsys, deck)

    assert (code, out, err) == (0, f"{deck}: created\n", "")
    assert migrated_doc(deck) == {
        "water_bucket_stages": water(("2024-01-15", 168.0), ("2024-01-22", 168.0))
    }


def test_main_arc_deck_with_an_anticipated_thermal_writes_the_layout_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    legacy = {**_LEGACY, "thermal_bounds": []}
    deck = make_deck(
        tmp_path,
        stages=_WEEKLY,
        travel_time_hours=336.0,
        thermals=_ANTICIPATED,
        post_study=legacy,
    )

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: migrated\n")
    assert (deck / pss.FILE_NAME).read_text(encoding="utf-8") == (
        "{\n"
        '  "$schema": "schema-uri",\n'
        '  "thermal_stages": [\n'
        "    {\n"
        '      "start_date": "2024-03-01",\n'
        '      "duration_hours": 744.0\n'
        "    }\n"
        "  ],\n"
        '  "thermal_bounds": [],\n'
        '  "water_bucket_stages": [\n'
        "    {\n"
        '      "start_date": "2024-01-15",\n'
        '      "duration_hours": 168.0\n'
        "    },\n"
        "    {\n"
        '      "start_date": "2024-01-22",\n'
        '      "duration_hours": 168.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def test_main_arc_deck_keeps_the_thermal_pair_verbatim_in_key_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path,
        stages=_MONTHLY,
        travel_time_hours=360.0,
        thermals=_ANTICIPATED,
        post_study=_LEGACY,
    )

    code, out, _ = run(capsys, deck)

    doc = migrated_doc(deck)
    assert (code, out) == (0, f"{deck}: migrated\n")
    assert list(doc) == [
        "$schema",
        "thermal_stages",
        "thermal_bounds",
        "water_bucket_stages",
    ]
    assert doc["thermal_stages"] == _LEGACY["stages"]
    assert doc["thermal_bounds"] == _BOUNDS
    assert doc["water_bucket_stages"] == water(("2024-03-01", 744.0))


def test_main_no_arc_legacy_deck_with_an_anticipated_thermal_is_renamed_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path, stages=_MONTHLY, thermals=_ANTICIPATED, post_study=_LEGACY
    )

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: renamed\n")
    assert migrated_doc(deck) == {
        "$schema": _SCHEMA,
        "thermal_stages": _LEGACY["stages"],
        "thermal_bounds": _BOUNDS,
    }


@pytest.mark.parametrize("schema", [True, False], ids=["with-schema", "without-schema"])
def test_arc_deck_without_anticipated_thermal_drops_thermal_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], schema: bool
) -> None:
    legacy = _LEGACY if schema else {k: v for k, v in _LEGACY.items() if k != "$schema"}
    deck = make_deck(
        tmp_path, stages=_MONTHLY, travel_time_hours=360.0, post_study=legacy
    )

    code, out, _ = run(capsys, deck)

    doc = migrated_doc(deck)
    assert (code, out) == (0, f"{deck}: migrated\n")
    assert list(doc) == (["$schema"] if schema else []) + ["water_bucket_stages"]


@pytest.mark.parametrize("schema", [True, False], ids=["with-schema", "without-schema"])
def test_no_arc_deck_without_anticipated_thermal_drops_thermal_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], schema: bool
) -> None:
    legacy = _LEGACY if schema else {k: v for k, v in _LEGACY.items() if k != "$schema"}
    deck = make_deck(tmp_path, stages=_MONTHLY, post_study=legacy)

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: cleared\n")
    assert migrated_doc(deck) == ({"$schema": _SCHEMA} if schema else {})

    before = snapshot(deck)
    code, out, _ = run(capsys, deck)
    assert (code, out) == (0, f"{deck}: already current\n")
    assert snapshot(deck) == before


@pytest.mark.parametrize(
    "thermals",
    [[{"id": 0}], [{"id": 0, "anticipated_config": None}]],
    ids=["absent", "null"],
)
def test_main_thermal_without_an_anticipated_config_is_not_anticipated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], thermals: list[dict[str, Any]]
) -> None:
    deck = make_deck(tmp_path, stages=_MONTHLY, thermals=thermals, post_study=_LEGACY)

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: cleared\n")


def test_main_deck_already_in_the_current_format_is_left_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = (
        '{"water_bucket_stages":[{"start_date":"2024-03-01","duration_hours":744}]}'
    )
    deck = make_deck(
        tmp_path, stages=_MONTHLY, travel_time_hours=360.0, post_study=current
    )
    before = snapshot(deck)

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: already current\n")
    assert snapshot(deck) == before


def test_main_no_arc_deck_without_a_file_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(tmp_path, stages=_MONTHLY)
    before = snapshot(deck)

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: no change\n")
    assert snapshot(deck) == before


def test_main_non_whole_day_last_stage_without_an_arc_is_not_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path, stages=[stage(0, "2024-01-15", "2024-02-14", 700)], post_study=_LEGACY
    )

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: cleared\n")


@pytest.mark.parametrize("kind", ["migrated", "created", "renamed", "cleared"])
def test_main_second_run_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    arc = kind in {"migrated", "created"}
    deck = make_deck(
        tmp_path,
        stages=_MONTHLY,
        travel_time_hours=360.0 if arc else None,
        thermals=_ANTICIPATED if kind in {"migrated", "renamed"} else None,
        post_study=None if kind == "created" else _LEGACY,
    )
    assert run(capsys, deck)[:2] == (0, f"{deck}: {kind}\n")
    first = snapshot(deck)

    code, out, err = run(capsys, deck)

    assert (code, out, err) == (0, f"{deck}: already current\n", "")
    assert snapshot(deck) == first


def test_main_mixed_file_is_refused_and_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path,
        stages=_MONTHLY,
        travel_time_hours=360.0,
        post_study={**_LEGACY, "water_bucket_stages": []},
    )
    before = snapshot(deck)

    code, out, err = run(capsys, deck)

    assert (code, out) == (1, "")
    assert err.startswith(f"{deck}: ")
    assert "`stages` together with" in err
    assert snapshot(deck) == before


def test_main_non_whole_day_last_stage_with_an_arc_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path,
        stages=[stage(0, "2024-01-15", "2024-02-14", 700)],
        travel_time_hours=360.0,
        post_study=_LEGACY,
    )
    before = snapshot(deck)

    code, out, err = run(capsys, deck)

    assert (code, out) == (1, "")
    assert err.startswith(f"{deck}: ")
    assert "whole number of days" in err
    assert snapshot(deck) == before


def test_main_arc_deck_without_an_in_study_stage_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path,
        stages=[stage(-1, "2023-12-01", "2024-01-01", 744)],
        travel_time_hours=360.0,
    )
    before = snapshot(deck)

    code, out, err = run(capsys, deck)

    assert (code, out) == (1, "")
    assert err.startswith(f"{deck}: ")
    assert "no in-study stage" in err
    assert snapshot(deck) == before


def test_main_legacy_file_without_thermal_bounds_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_deck(
        tmp_path, stages=_MONTHLY, post_study={"stages": _LEGACY["stages"]}
    )

    code, _, err = run(capsys, deck)

    assert code == 1
    assert "without `thermal_bounds`" in err


@pytest.mark.parametrize("broken", [pss.FILE_NAME, "system/hydros.json", "stages.json"])
def test_main_unparsable_input_is_a_refusal_naming_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], broken: str
) -> None:
    deck = make_deck(
        tmp_path, stages=_MONTHLY, travel_time_hours=360.0, post_study=_LEGACY
    )
    (deck / broken).write_text("{", encoding="utf-8")

    code, out, err = run(capsys, deck)

    assert (code, out) == (1, "")
    assert err.startswith(f"{deck}: {broken}: ")
    assert "Traceback" not in err


def test_main_refused_deck_stops_the_other_decks_from_being_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    migratable = make_deck(
        tmp_path, "first", stages=_MONTHLY, travel_time_hours=360.0, post_study=_LEGACY
    )
    refused = make_deck(
        tmp_path,
        "second",
        stages=_MONTHLY,
        travel_time_hours=360.0,
        post_study={**_LEGACY, "thermal_stages": []},
    )
    before = (snapshot(migratable), snapshot(refused))

    code, out, err = run(capsys, migratable, refused)

    assert (code, out) == (1, "")
    assert err.count("\n") == 1
    assert err.startswith(f"{refused}: ")
    assert (snapshot(migratable), snapshot(refused)) == before


def test_main_every_deck_is_migrated_when_none_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = make_deck(tmp_path, "first", stages=_MONTHLY, travel_time_hours=360.0)
    second = make_deck(tmp_path, "second", stages=_MONTHLY, post_study=_LEGACY)

    code, out, _ = run(capsys, first, second)

    assert (code, out) == (0, f"{first}: created\n{second}: cleared\n")


def test_main_without_an_argument_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run(capsys)

    assert (code, out) == (2, "")
    assert "DECK_DIR" in err


def test_main_path_that_is_not_a_directory_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_deck = tmp_path / "stages.json"
    not_a_deck.write_text("{}", encoding="utf-8")

    code, out, err = run(capsys, not_a_deck)

    assert (code, out) == (2, "")
    assert err == f"{not_a_deck}: not a directory\n"


@pytest.mark.parametrize(
    "missing", ["stages.json", "system/hydros.json", "system/thermals.json"]
)
def test_main_deck_missing_an_input_file_is_a_usage_error_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    incomplete = make_deck(tmp_path, "incomplete", stages=_MONTHLY, post_study=_LEGACY)
    (incomplete / missing).unlink()
    valid = make_deck(tmp_path, "valid", stages=_MONTHLY, post_study=_LEGACY)
    before = snapshot(valid)

    code, out, err = run(capsys, valid, incomplete)

    assert (code, out) == (2, "")
    assert err == f"{incomplete}: missing {missing}\n"
    assert snapshot(valid) == before


def test_write_atomic_replaces_the_file_through_a_sibling_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / pss.FILE_NAME
    target.write_text("old", encoding="utf-8")
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record(src: Path, dst: Path) -> None:
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", record)

    pss.write_atomic(target, "new\n")

    assert replaced == [(target.with_name(pss.FILE_NAME + ".tmp"), target)]
    assert snapshot(tmp_path) == {pss.FILE_NAME: b"new\n"}


def copy_committed_deck(name: str, destination: Path, *, with_scenarios: bool) -> Path:
    ignore = None if with_scenarios else shutil.ignore_patterns("scenarios")
    shutil.copytree(_EXAMPLES / name, destination, ignore=ignore)
    (destination / pss.FILE_NAME).unlink(missing_ok=True)
    return destination


def study_end(deck: Path) -> str:
    stages = json.loads((deck / "stages.json").read_text(encoding="utf-8"))["stages"]
    last = max((s for s in stages if s["id"] >= 0), key=lambda s: s["id"])
    end: str = last["end_date"]
    return end


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("d44-travel-time-substage", water(("2024-03-01", 744.0))),
        (
            "d45-travel-time-mixed-calendar",
            water(("2024-02-21", 168.0), ("2024-02-28", 168.0), ("2024-03-06", 168.0)),
        ),
        ("d46-travel-time-chronological", water(("2024-03-01", 744.0))),
        (
            "d47-travel-time-confluence",
            water(("2024-04-01", 720.0), ("2024-05-01", 744.0)),
        ),
        (
            "d48-travel-time-ic-seed",
            water(("2024-01-22", 168.0), ("2024-01-29", 168.0)),
        ),
        ("d49-travel-time-chronological-arrival", water(("2024-02-14", 720.0))),
        ("d50-travel-time-plain-tributary-confluence", water(("2024-02-14", 720.0))),
    ],
)
def test_main_committed_arc_deck_gets_the_expected_water_calendar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    expected: list[dict[str, Any]],
) -> None:
    deck = copy_committed_deck(name, tmp_path / name, with_scenarios=False)

    code, out, _ = run(capsys, deck)

    assert (code, out) == (0, f"{deck}: created\n")
    assert migrated_doc(deck) == {"water_bucket_stages": expected}


@pytest.mark.parametrize("name", [_ARC_DECK, _NO_ARC_DECK], ids=["arc", "no-arc"])
def test_migrated_decks_raise_no_post_study_warning(tmp_path: Path, name: str) -> None:
    binary = os.environ.get("NOVOMODELO_BIN")
    if not binary:
        pytest.skip("NOVOMODELO_BIN is unset; point it at a built novomodelo binary")

    unmigrated = copy_committed_deck(name, tmp_path / "unmigrated", with_scenarios=True)
    legacy = {
        "stages": [{"start_date": study_end(unmigrated), "duration_hours": 168.0}],
        "thermal_bounds": [],
    }
    write_json(unmigrated / pss.FILE_NAME, legacy)
    migrated = tmp_path / "migrated"
    shutil.copytree(unmigrated, migrated)

    def validate(deck: Path) -> tuple[int, list[str], str]:
        proc = subprocess.run(
            [binary, "validate", "--color", "never", str(deck)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        warnings = [
            line for line in output.splitlines() if line.startswith(_POST_STUDY_WARNING)
        ]
        return proc.returncode, warnings, output

    assert validate(unmigrated)[1]
    assert pss.main([str(migrated)]) == 0
    code, warnings, output = validate(migrated)
    assert code == 0, output
    assert not warnings, output
