"""Tests for boundary_conditions.py: refusals, calendars, boundary entries and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import boundary_conditions as bc
import post_study_stages as pss

_SCRIPT = Path(__file__).with_name("boundary_conditions.py")
_SCHEMA_URL = (
    "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/"
    "schemas/boundary_conditions.schema.json"
)
_ARC_HOURS = 1000.0
THERMAL_KEY = "post_study_thermal_stages"
WATER_KEY = "post_study_water_stages"


def stage(sid: int, start: str, end: str, hours: float) -> dict[str, Any]:
    blocks = [{"id": 0, "name": "SINGLE", "hours": hours}]
    return {"id": sid, "start_date": start, "end_date": end, "blocks": blocks}


_MONTHLY = [
    stage(0, "2024-01-01", "2024-02-01", 744),
    stage(1, "2024-02-01", "2024-03-01", 696),
]
_STAGES_730 = [
    stage(0, "2024-01-01", "2024-02-01", 730),
    stage(1, "2024-02-01", "2024-03-01", 730),
]
_INITIAL: dict[str, Any] = {
    "storage": [{"hydro_id": 0, "value_hm3": 5.0}],
    "filling_storage": [],
}
_BOUND = {
    "thermal_id": 1,
    "post_study_stage_index": 0,
    "cost_per_mwh": 5.0,
    "min_mw": 0.0,
    "max_mw": 150.0,
}
_MARCH = {"start_date": "2024-03-01", "duration_hours": 744.0}
_APRIL = {"start_date": "2024-04-01", "duration_hours": 720.0}
_CURRENT: dict[str, Any] = {"thermal_stages": [_MARCH], "thermal_bounds": [_BOUND]}


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_legacy_deck(
    root: Path,
    name: str = "deck",
    *,
    initial: dict[str, Any] | None = None,
    post_study: dict[str, Any] | None = None,
    stages: list[dict[str, Any]] = _MONTHLY,
    travel_time_hours: float | None = None,
    filling: tuple[int, ...] = (),
    anticipated: bool = True,
    stages_extra: dict[str, Any] | None = None,
) -> Path:
    deck = root / name
    write_json(
        deck / "stages.json",
        {"$schema": "stages-schema", "stages": stages, **(stages_extra or {})},
    )
    hydros: list[dict[str, Any]] = [
        {"id": 0, "downstream_id": 1},
        {"id": 1, "downstream_id": None},
    ]
    if travel_time_hours is not None:
        hydros[0]["travel_time_hours"] = travel_time_hours
    for hydro in hydros:
        if hydro["id"] in filling:
            hydro["filling"] = {"min_rate_m3s": 1.0}
    write_json(deck / "system" / "hydros.json", {"hydros": hydros})
    write_json(
        deck / "system" / "thermals.json",
        {
            "thermals": [
                {"id": 0},
                {
                    "id": 1,
                    "anticipated_config": {"lead_time_hours": 1000.0}
                    if anticipated
                    else None,
                },
            ]
        },
    )
    write_json(deck / bc.INITIAL_FILE, _INITIAL if initial is None else initial)
    if post_study is not None:
        write_json(deck / pss.FILE_NAME, post_study)
    return deck


def snapshot(deck: Path) -> dict[str, bytes]:
    return {
        path.relative_to(deck).as_posix(): path.read_bytes()
        for path in sorted(deck.rglob("*"))
        if path.is_file()
    }


def run(capsys: pytest.CaptureFixture[str], *decks: Path) -> tuple[int, str, str]:
    code = bc.main([str(deck) for deck in decks])
    out, err = capsys.readouterr()
    return code, out, err


def refused(capsys: pytest.CaptureFixture[str], deck: Path, *needles: str) -> None:
    before = snapshot(deck)
    code, out, err = run(capsys, deck)
    assert (code, out) == (1, "")
    assert err.startswith(f"{deck}: ")
    assert err.count("\n") == 1
    for needle in needles:
        assert needle in err
    assert snapshot(deck) == before


def rekey(rows: list[dict[str, Any]], first_id: int) -> list[dict[str, Any]]:
    return [
        {
            "id": first_id + k,
            "start_date": row["start_date"],
            "end_date": (
                date.fromisoformat(row["start_date"])
                + timedelta(days=row["duration_hours"] / 24)
            ).isoformat(),
        }
        for k, row in enumerate(rows)
    ]


def dumped(doc: Any) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


@pytest.mark.parametrize(
    "legacy", [bc.INITIAL_FILE, pss.FILE_NAME], ids=["initial", "post-study"]
)
def test_main_refuses_both_file_sets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], legacy: str
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=_CURRENT)
    (deck / bc.BOUNDARY_FILE).write_text('{"hydros": [], "thermals": []}\n')
    for name in {bc.INITIAL_FILE, pss.FILE_NAME} - {legacy}:
        (deck / name).unlink()
    refused(capsys, deck, "structural.3", legacy)


@pytest.mark.parametrize("key", [THERMAL_KEY, WATER_KEY])
def test_main_refuses_calendars_with_legacy_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    deck = make_legacy_deck(
        tmp_path,
        stages_extra={
            key: [{"id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01"}]
        },
    )
    refused(capsys, deck, "semantic.5a.71")


def test_main_null_post_study_list_beside_legacy_files_is_not_a_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path, stages_extra={THERMAL_KEY: None})
    assert run(capsys, deck) == (0, f"{deck}: migrated\n", "")
    assert read_json(deck / "stages.json")[THERMAL_KEY] is None


@pytest.mark.parametrize(
    ("array", "filling", "hydro_id"),
    [("storage", (0,), 0), ("filling_storage", (), 0)],
    ids=["storage-of-a-filling-hydro", "filling-storage-of-an-operating-hydro"],
)
def test_main_refuses_wrong_array_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    array: str,
    filling: tuple[int, ...],
    hydro_id: int,
) -> None:
    other = "filling_storage" if array == "storage" else "storage"
    initial = {array: [{"hydro_id": hydro_id, "value_hm3": 5.0}], other: []}
    deck = make_legacy_deck(tmp_path, initial=initial, filling=filling)
    refused(capsys, deck, "semantic.5a.54", f"hydro {hydro_id}", f"not `{array}`")


@pytest.mark.parametrize(
    "extra",
    [{"thermal_stages": []}, {"water_bucket_stages": []}],
    ids=["thermal_stages", "water_bucket_stages"],
)
def test_main_refuses_mixed_post_study_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: dict[str, Any]
) -> None:
    mixed = {"stages": [_MARCH], "thermal_bounds": [], **extra}
    deck = make_legacy_deck(tmp_path, post_study=mixed)
    refused(
        capsys,
        deck,
        "declares `stages` together with `thermal_stages` or `water_bucket_stages`",
    )


@pytest.mark.parametrize(
    ("post_study", "present", "missing"),
    [
        ({"thermal_stages": [_MARCH]}, "thermal_stages", "thermal_bounds"),
        ({"thermal_bounds": [_BOUND]}, "thermal_bounds", "thermal_stages"),
        ({"stages": [_MARCH]}, "stages", "thermal_bounds"),
    ],
    ids=["current-stages-only", "current-bounds-only", "legacy"],
)
def test_main_refuses_thermal_stages_without_bounds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    post_study: dict[str, Any],
    present: str,
    missing: str,
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    refused(capsys, deck, "semantic.5a.63", f"`{present}` without `{missing}`")


def test_main_null_thermal_pair_counts_as_undeclared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post_study = {"thermal_stages": None, "thermal_bounds": None}
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    assert run(capsys, deck) == (0, f"{deck}: migrated\n", "")
    assert THERMAL_KEY not in read_json(deck / "stages.json")


@pytest.mark.parametrize("index", [1, 7, -1])
def test_main_refuses_out_of_range_bound_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], index: int
) -> None:
    post_study = {
        **_CURRENT,
        "thermal_bounds": [{**_BOUND, "post_study_stage_index": index}],
    }
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    refused(capsys, deck, f"`post_study_stage_index` {index} is out of range for 1")


@pytest.mark.parametrize(
    "post_study", [_CURRENT, {}], ids=["thermal-pair-only", "empty-file"]
)
def test_main_refuses_arc_without_water_calendar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], post_study: dict[str, Any]
) -> None:
    deck = make_legacy_deck(
        tmp_path, post_study=post_study, travel_time_hours=_ARC_HOURS
    )
    refused(capsys, deck, "semantic.5a.61", "never padded")


@pytest.mark.parametrize(
    ("post_study", "arc", "start", "hours"),
    [
        (
            {
                "thermal_stages": [{**_MARCH, "duration_hours": 730.0}],
                "thermal_bounds": [],
            },
            None,
            "2024-03-01",
            "730",
        ),
        (
            {"stages": [{**_MARCH, "duration_hours": 744.5}], "thermal_bounds": []},
            None,
            "2024-03-01",
            "744.5",
        ),
        (
            {"water_bucket_stages": [{**_MARCH, "duration_hours": 730.0}]},
            _ARC_HOURS,
            "2024-03-01",
            "730",
        ),
        (
            {"thermal_stages": [{**_MARCH, "duration_hours": 0}], "thermal_bounds": []},
            None,
            "2024-03-01",
            "lasts 0 h",
        ),
    ],
    ids=["thermal-stages", "legacy-stages", "water-bucket-stages", "zero-duration"],
)
def test_main_refuses_non_whole_day_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    post_study: dict[str, Any],
    arc: float | None,
    start: str,
    hours: str,
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=post_study, travel_time_hours=arc)
    refused(capsys, deck, start, hours, "24")


@pytest.mark.parametrize(
    "post_study",
    [None, {"stages": [], "thermal_bounds": []}],
    ids=["no-file", "empty-legacy-stages"],
)
def test_main_refuses_non_whole_day_after_padding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    post_study: dict[str, Any] | None,
) -> None:
    deck = make_legacy_deck(
        tmp_path,
        post_study=post_study,
        stages=_STAGES_730,
        travel_time_hours=_ARC_HOURS,
    )
    refused(capsys, deck, "2024-03-01", "730", "24")


def test_main_refuses_a_malformed_start_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post_study = {
        "thermal_stages": [{**_MARCH, "start_date": "March"}],
        "thermal_bounds": [],
    }
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    refused(capsys, deck, "`thermal_stages[0].start_date` is malformed")


def test_main_refuses_null_legacy_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(
        tmp_path,
        post_study={"stages": None, "thermal_bounds": []},
        travel_time_hours=_ARC_HOURS,
    )
    refused(capsys, deck, "`stages` must be a list of objects")


def test_main_refused_deck_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = make_legacy_deck(tmp_path, "good", post_study=_CURRENT)
    bad = make_legacy_deck(tmp_path, "bad", post_study={"stages": [_MARCH]})
    before = (snapshot(good), snapshot(bad))

    code, out, err = run(capsys, good, bad)

    assert (code, out) == (1, "")
    assert err.splitlines() == [
        f"{bad}: {pss.FILE_NAME} declares `stages` without `thermal_bounds` (semantic.5a.63)"
    ]
    assert (snapshot(good), snapshot(bad)) == before


def test_created_path_writes_the_padded_water_calendar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path, travel_time_hours=_ARC_HOURS)
    padded = pss.padded_water_calendar(deck, None)
    assert padded is not None and len(padded) == 2

    assert run(capsys, deck) == (0, f"{deck}: migrated\n", "")

    stages = read_json(deck / "stages.json")
    assert stages["post_study_water_stages"] == rekey(padded, 2)
    assert THERMAL_KEY not in stages
    assert sorted(snapshot(deck)) == [
        "boundary_conditions.json",
        "stages.json",
        "system/hydros.json",
        "system/thermals.json",
    ]


def test_migrated_path_maps_the_stages_key_to_the_thermal_calendar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    legacy = {"stages": [_MARCH], "thermal_bounds": [_BOUND]}
    deck = make_legacy_deck(tmp_path, post_study=legacy, travel_time_hours=_ARC_HOURS)
    padded = pss.padded_water_calendar(deck, legacy["stages"])
    assert padded is not None and len(padded) == 2

    assert run(capsys, deck) == (0, f"{deck}: migrated\n", "")

    stages = read_json(deck / "stages.json")
    assert stages[THERMAL_KEY] == rekey([_MARCH], 2)
    assert stages[WATER_KEY] == rekey(padded, 3)
    boundary = read_json(deck / bc.BOUNDARY_FILE)
    assert boundary["thermals"] == [
        {
            "thermal_id": 1,
            "post_study_bounds": [
                {"stage_id": 2, "cost_per_mwh": 5.0, "min_mw": 0.0, "max_mw": 150.0}
            ],
        }
    ]


def test_main_rekeys_bounds_through_the_start_date_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post_study = {
        "thermal_stages": [_APRIL, _MARCH],
        "thermal_bounds": [
            {**_BOUND, "post_study_stage_index": 1, "cost_per_mwh": 9.0},
            {**_BOUND, "post_study_stage_index": 0, "cost_per_mwh": 4.0},
        ],
    }
    deck = make_legacy_deck(tmp_path, post_study=post_study)

    assert run(capsys, deck)[0] == 0

    stages = read_json(deck / "stages.json")
    assert stages[THERMAL_KEY] == rekey([_MARCH, _APRIL], 2)
    bounds = read_json(deck / bc.BOUNDARY_FILE)["thermals"][0]["post_study_bounds"]
    assert [(b["stage_id"], b["cost_per_mwh"]) for b in bounds] == [(2, 4.0), (3, 9.0)]


def test_main_assigns_water_ids_after_the_thermal_ids_in_start_date_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    water = [_APRIL, _MARCH]
    post_study = {
        "thermal_stages": [_MARCH, _APRIL],
        "thermal_bounds": [_BOUND],
        "water_bucket_stages": water,
    }
    deck = make_legacy_deck(
        tmp_path, post_study=post_study, travel_time_hours=_ARC_HOURS
    )

    assert run(capsys, deck)[0] == 0

    stages = read_json(deck / "stages.json")
    assert stages[THERMAL_KEY] == rekey([_MARCH, _APRIL], 2)
    assert stages[WATER_KEY] == rekey([_MARCH, _APRIL], 4)


def test_main_writes_a_declared_empty_list_as_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post_study: dict[str, Any] = {"thermal_stages": [], "thermal_bounds": []}
    deck = make_legacy_deck(tmp_path, post_study=post_study)

    assert run(capsys, deck)[0] == 0

    stages = read_json(deck / "stages.json")
    assert stages[THERMAL_KEY] == []
    assert WATER_KEY not in stages


def test_main_keeps_the_thermal_pair_of_a_deck_without_an_anticipated_thermal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=_CURRENT, anticipated=False)

    assert run(capsys, deck)[0] == 0

    assert read_json(deck / "stages.json")[THERMAL_KEY] == rekey([_MARCH], 2)
    assert read_json(deck / bc.BOUNDARY_FILE)["thermals"][0]["post_study_bounds"]


def test_main_first_post_study_id_follows_the_largest_study_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pre_study = stage(-1, "2023-12-01", "2024-01-01", 744)
    gapped = [pre_study, *_MONTHLY[:1], {**_MONTHLY[1], "id": 5}]
    deck = make_legacy_deck(tmp_path, post_study=_CURRENT, stages=gapped)

    assert run(capsys, deck)[0] == 0

    assert read_json(deck / "stages.json")[THERMAL_KEY][0]["id"] == 6


def test_main_builds_entries_sorted_by_id_with_windows_in_declaration_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def flow(hydro_id: int, start: str, end: str, value: float) -> dict[str, Any]:
        return {
            "value_m3s": value,
            "end_date": end,
            "start_date": start,
            "hydro_id": hydro_id,
        }

    initial = {
        "$schema": "initial-schema",
        "storage": [
            {"value_hm3": 10.0, "hydro_id": 1},
            {"hydro_id": 0, "value_hm3": 5.5},
        ],
        "filling_storage": [],
        "recent_observations": [
            flow(1, "2023-12-15", "2024-01-01", -1.0),
            flow(1, "2023-12-01", "2023-12-15", 3.5),
        ],
        "past_defluences": [flow(1, "2023-11-01", "2024-01-01", 20.0)],
        "past_anticipated_commitments": [
            {
                "thermal_id": 1,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "value_mw": 0.0,
            },
            {
                "thermal_id": 1,
                "start_date": "2024-02-01",
                "end_date": "2024-03-01",
                "value_mw": 75.0,
            },
        ],
    }
    post_study = {
        "thermal_stages": [_MARCH, _APRIL],
        "thermal_bounds": [
            {**_BOUND, "post_study_stage_index": 1, "max_mw": 90.0},
            {**_BOUND, "post_study_stage_index": 0},
            {**_BOUND, "thermal_id": 0, "post_study_stage_index": 1, "max_mw": 20.0},
        ],
    }
    deck = make_legacy_deck(tmp_path, initial=initial, post_study=post_study)

    assert run(capsys, deck)[0] == 0

    def bound(stage_id: int, max_mw: float) -> dict[str, Any]:
        return {
            "stage_id": stage_id,
            "cost_per_mwh": 5.0,
            "min_mw": 0.0,
            "max_mw": max_mw,
        }

    expected = {
        "$schema": _SCHEMA_URL,
        "hydros": [
            {"hydro_id": 0, "storage_hm3": 5.5},
            {
                "hydro_id": 1,
                "storage_hm3": 10.0,
                "recent_inflows": [
                    {
                        "start_date": "2023-12-15",
                        "end_date": "2024-01-01",
                        "value_m3s": -1.0,
                    },
                    {
                        "start_date": "2023-12-01",
                        "end_date": "2023-12-15",
                        "value_m3s": 3.5,
                    },
                ],
                "past_defluences": [
                    {
                        "start_date": "2023-11-01",
                        "end_date": "2024-01-01",
                        "value_m3s": 20.0,
                    }
                ],
            },
        ],
        "thermals": [
            {"thermal_id": 0, "post_study_bounds": [bound(3, 20.0)]},
            {
                "thermal_id": 1,
                "commitments": [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-02-01",
                        "value_mw": 0.0,
                    },
                    {
                        "start_date": "2024-02-01",
                        "end_date": "2024-03-01",
                        "value_mw": 75.0,
                    },
                ],
                "post_study_bounds": [bound(2, 150.0), bound(3, 90.0)],
            },
        ],
    }
    assert (deck / bc.BOUNDARY_FILE).read_text(encoding="utf-8") == dumped(expected)


def test_main_routes_filling_storage_to_storage_hm3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    initial = {
        "storage": [{"hydro_id": 0, "value_hm3": 5.0}],
        "filling_storage": [{"hydro_id": 1, "value_hm3": 7.5}],
    }
    deck = make_legacy_deck(tmp_path, initial=initial, filling=(1,))

    assert run(capsys, deck)[0] == 0

    assert read_json(deck / bc.BOUNDARY_FILE) == {
        "hydros": [
            {"hydro_id": 0, "storage_hm3": 5.0},
            {"hydro_id": 1, "storage_hm3": 7.5},
        ],
        "thermals": [],
    }


@pytest.mark.parametrize("schema", [True, False], ids=["with-schema", "without-schema"])
def test_main_schema_key_follows_the_legacy_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], schema: bool
) -> None:
    initial = {**_INITIAL, **({"$schema": "initial-schema"} if schema else {})}
    deck = make_legacy_deck(tmp_path, initial=initial)

    assert run(capsys, deck)[0] == 0

    boundary = read_json(deck / bc.BOUNDARY_FILE)
    assert boundary.get("$schema") == (_SCHEMA_URL if schema else None)
    assert list(boundary)[:1] == (["$schema"] if schema else ["hydros"])


def test_main_without_a_post_study_list_leaves_stages_json_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path)
    raw = '{"stages":[{"id":0,"name":"é"}]}'
    (deck / "stages.json").write_text(raw, encoding="utf-8")

    assert run(capsys, deck) == (0, f"{deck}: migrated\n", "")

    assert (deck / "stages.json").read_text(encoding="utf-8") == raw
    assert not (deck / bc.INITIAL_FILE).exists()


def test_main_rewrites_stages_json_keeping_its_keys_and_appending_the_lists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post_study = {**_CURRENT, "water_bucket_stages": [_MARCH]}
    deck = make_legacy_deck(
        tmp_path,
        post_study=post_study,
        travel_time_hours=_ARC_HOURS,
        stages_extra={"name": "é"},
    )

    assert run(capsys, deck)[0] == 0

    written = read_json(deck / "stages.json")
    assert list(written) == ["$schema", "stages", "name", THERMAL_KEY, WATER_KEY]
    assert (deck / "stages.json").read_text(encoding="utf-8") == dumped(written)


@pytest.mark.parametrize(
    "post_study", [None, _CURRENT], ids=["initial-only", "with-post-study"]
)
def test_main_second_run_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    post_study: dict[str, Any] | None,
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    assert run(capsys, deck)[0] == 0
    migrated = snapshot(deck)

    assert run(capsys, deck) == (0, f"{deck}: already current\n", "")

    assert snapshot(deck) == migrated


def test_main_every_deck_is_migrated_when_none_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = make_legacy_deck(tmp_path, "first")
    second = make_legacy_deck(tmp_path, "second", post_study=_CURRENT)

    code, out, err = run(capsys, first, second)

    assert (code, err) == (0, "")
    assert out.splitlines() == [f"{first}: migrated", f"{second}: migrated"]


@pytest.mark.parametrize(
    "broken",
    [bc.INITIAL_FILE, pss.FILE_NAME, "stages.json", "system/hydros.json"],
)
def test_main_unparsable_input_is_a_refusal_naming_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], broken: str
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=_CURRENT)
    (deck / broken).write_text("{not json", encoding="utf-8")
    refused(capsys, deck, f"{broken}: ")


def test_main_non_standard_json_constant_is_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path)
    (deck / bc.INITIAL_FILE).write_text(
        '{"storage": [{"hydro_id": 0, "value_hm3": NaN}], "filling_storage": []}',
        encoding="utf-8",
    )
    refused(capsys, deck, f"{bc.INITIAL_FILE}: ", "NaN")


@pytest.mark.parametrize(
    ("initial", "needle"),
    [
        ({**_INITIAL, "recent_observation": []}, "unknown key(s) recent_observation"),
        ({"storage": [], "filling_storage": None}, "`filling_storage` must be a list"),
        (
            {"storage": [{"hydro_id": 0}], "filling_storage": []},
            "`storage` must be a list of objects with exactly the keys hydro_id, value_hm3",
        ),
        (
            {"storage": [{"hydro_id": "0", "value_hm3": 5.0}], "filling_storage": []},
            "`storage` must be a list",
        ),
        (
            {**_INITIAL, "past_defluences": [{"hydro_id": 0, "value_m3s": 1.0}]},
            "`past_defluences` must be a list",
        ),
    ],
    ids=["unknown-key", "null-array", "missing-field", "string-id", "short-window"],
)
def test_main_refuses_an_initial_conditions_file_the_new_format_cannot_hold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    initial: dict[str, Any],
    needle: str,
) -> None:
    refused(capsys, make_legacy_deck(tmp_path, initial=initial), needle)


@pytest.mark.parametrize(
    ("arrays", "filling"),
    [
        (
            {
                "storage": [
                    {"hydro_id": 0, "value_hm3": 1.0},
                    {"hydro_id": 0, "value_hm3": 2.0},
                ]
            },
            (),
        ),
        (
            {
                "storage": [{"hydro_id": 7, "value_hm3": 1.0}],
                "filling_storage": [{"hydro_id": 7, "value_hm3": 2.0}],
            },
            (),
        ),
    ],
    ids=["twice-in-storage", "in-both-arrays"],
)
def test_main_refuses_a_second_storage_value_for_one_hydro(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arrays: dict[str, Any],
    filling: tuple[int, ...],
) -> None:
    initial = {"storage": [], "filling_storage": [], **arrays}
    deck = make_legacy_deck(tmp_path, initial=initial, filling=filling)
    refused(capsys, deck, "more than one storage value")


def test_main_deck_with_only_the_boundary_file_is_already_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deck = make_legacy_deck(tmp_path)
    (deck / bc.INITIAL_FILE).unlink()
    (deck / bc.BOUNDARY_FILE).write_text('{"hydros": [], "thermals": []}\n')
    before = snapshot(deck)

    assert run(capsys, deck) == (0, f"{deck}: already current\n", "")

    assert snapshot(deck) == before


def test_main_without_an_argument_is_a_usage_error() -> None:
    assert bc.main([]) == 2


def test_main_path_that_is_not_a_directory_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing"
    assert run(capsys, missing) == (2, "", f"{missing}: not a directory\n")


@pytest.mark.parametrize(
    "missing", ["stages.json", "system/hydros.json", "system/thermals.json"]
)
def test_main_deck_missing_an_input_file_is_a_usage_error_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    good = make_legacy_deck(tmp_path, "good")
    bad = make_legacy_deck(tmp_path, "bad")
    (bad / missing).unlink()
    before = (snapshot(good), snapshot(bad))

    code, out, err = run(capsys, good, bad)

    assert (code, out) == (2, "")
    assert err == f"{bad}: missing {missing}\n"
    assert (snapshot(good), snapshot(bad)) == before


@pytest.mark.parametrize(
    "post_study", [None, _CURRENT], ids=["bare", "post-study-only"]
)
def test_main_deck_without_a_boundary_file_is_a_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    post_study: dict[str, Any] | None,
) -> None:
    deck = make_legacy_deck(tmp_path, post_study=post_study)
    (deck / bc.INITIAL_FILE).unlink()
    before = snapshot(deck)

    code, out, err = run(capsys, deck)

    assert (code, out) == (2, "")
    assert err == f"{deck}: missing {bc.BOUNDARY_FILE} or {bc.INITIAL_FILE}\n"
    assert snapshot(deck) == before


def test_main_runs_as_a_script_from_any_directory(tmp_path: Path) -> None:
    deck = make_legacy_deck(tmp_path / "decks", post_study=_CURRENT)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(deck)],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        f"{deck}: migrated\n",
        "",
    )
    assert (deck / bc.BOUNDARY_FILE).is_file()
