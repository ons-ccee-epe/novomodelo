#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Migrate decks from `initial_conditions.json` to `boundary_conditions.json`.

Usage:
    python3 scripts/migrate/boundary_conditions.py DECK_DIR [DECK_DIR ...]

Every deck is planned before any file is written: one refused deck writes
nothing, and each refusal is one stderr line `DECK: reason` naming the rule it
mirrors. Each deck prints one stdout line, `DECK: OUTCOME`:

  migrated         `initial_conditions.json` (and `post_study_stages.json`) became
                   `boundary_conditions.json`; both legacy files are deleted
  already current  `boundary_conditions.json` and no legacy file

The post-study calendars move into `stages.json` as `post_study_thermal_stages`
and `post_study_water_stages`, ids continuing the study sequence: thermal stages
first, then water stages, each in `start_date` order. The deprecated `stages` key
of `post_study_stages.json` is read as `thermal_stages`. A deck with a travel-time
arc and no `water_bucket_stages` gets the calendar the legacy reader resolves to
(`post_study_stages.padded_water_calendar`), so migrating a deck does not change
its results. A list is written only when the legacy file declared it or padding
produced it; `stages.json` is rewritten only when a list is written.

A deck is refused when it mixes both file sets (structural.3), when `stages.json`
declares a post-study list beside the legacy files (semantic.5a.71), when the legacy
file or storage arrays break semantic.5a.54, 5a.61 or 5a.63, when
`post_study_stages.json` declares `stages` together with `thermal_stages` or
`water_bucket_stages`, when a bound's `post_study_stage_index` is out of range, when a
storage value cannot be held by one `storage_hm3`, or when a stage lasts a duration
that is not a whole number of days (never rounded; checked on the final calendar,
after padding).

Exit code 0 when every deck is migrated or already current, 1 when any deck is
refused, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, NoReturn

import post_study_stages as pss
from post_study_stages import FileFormat, Refusal, write_atomic

BOUNDARY_FILE = "boundary_conditions.json"
INITIAL_FILE = "initial_conditions.json"
STAGES_FILE = "stages.json"
_THERMAL_KEY = "post_study_thermal_stages"
_WATER_KEY = "post_study_water_stages"
_SCHEMA_URL = (
    "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/"
    "schemas/boundary_conditions.schema.json"
)
_DAY_HOURS = 24

Check = Callable[[object], bool]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_str(value: object) -> bool:
    return isinstance(value, str)


_STORAGE_ROW: dict[str, Check] = {"hydro_id": _is_int, "value_hm3": _is_number}
_FLOW_ROW: dict[str, Check] = {
    "hydro_id": _is_int,
    "start_date": _is_str,
    "end_date": _is_str,
    "value_m3s": _is_number,
}
_COMMITMENT_ROW: dict[str, Check] = {
    "thermal_id": _is_int,
    "start_date": _is_str,
    "end_date": _is_str,
    "value_mw": _is_number,
}
_STAGE_ROW: dict[str, Check] = {"start_date": _is_str, "duration_hours": _is_number}
_BOUND_ROW: dict[str, Check] = {
    "thermal_id": _is_int,
    "post_study_stage_index": _is_int,
    "cost_per_mwh": _is_number,
    "min_mw": _is_number,
    "max_mw": _is_number,
}
_INITIAL_KEYS = frozenset(
    {
        "$schema",
        "storage",
        "filling_storage",
        "recent_observations",
        "past_defluences",
        "past_anticipated_commitments",
    }
)


@dataclass(frozen=True)
class DeckPlan:
    deck: Path
    outcome: str
    writes: dict[str, str]
    deletes: tuple[str, ...]


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {name}")


def _read_object(path: Path, deck: Path) -> dict[str, Any]:
    name = path.relative_to(deck)
    try:
        doc = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, ValueError) as exc:
        raise Refusal(f"{name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise Refusal(f"{name}: expected a JSON object")
    return doc


def _declared(doc: Mapping[str, Any], key: str) -> bool:
    return doc.get(key) is not None


def _rows(
    doc: Mapping[str, Any], key: str, file: str, fields: Mapping[str, Check]
) -> list[dict[str, Any]]:
    """The list `key` of `doc`, each row an object with exactly `fields` of the expected types."""
    rows = doc.get(key, [])
    if not isinstance(rows, list) or not all(
        isinstance(row, dict)
        and row.keys() == fields.keys()
        and all(check(row[name]) for name, check in fields.items())
        for row in rows
    ):
        raise Refusal(
            f"{file}: `{key}` must be a list of objects with exactly the keys "
            f"{', '.join(fields)}, each of the expected JSON type"
        )
    return rows


def _filling_hydros(deck: Path) -> dict[int, bool]:
    hydros = _read_object(deck / "system" / "hydros.json", deck).get("hydros")
    if not isinstance(hydros, list) or not all(isinstance(h, dict) for h in hydros):
        raise Refusal("system/hydros.json: `hydros` must be a list of objects")
    return {h["id"]: _declared(h, "filling") for h in hydros if _is_int(h.get("id"))}


def _first_post_study_id(stages_doc: Mapping[str, Any]) -> int:
    stages = stages_doc.get("stages")
    if not isinstance(stages, list) or not all(isinstance(s, dict) for s in stages):
        raise Refusal(f"{STAGES_FILE}: `stages` must be a list of objects")
    return 1 + max(
        (s["id"] for s in stages if _is_int(s.get("id")) and s["id"] >= 0), default=-1
    )


def _storage_seeds(deck: Path, initial: Mapping[str, Any]) -> dict[int, int | float]:
    """`storage_hm3` per hydro; refuses a seed in the wrong array (semantic.5a.54)."""
    filling = _filling_hydros(deck)
    seeds: dict[int, int | float] = {}
    for array, for_filling in (("storage", False), ("filling_storage", True)):
        for row in _rows(initial, array, INITIAL_FILE, _STORAGE_ROW):
            hydro_id = row["hydro_id"]
            if filling.get(hydro_id, for_filling) != for_filling:
                reason, expected = (
                    ("has a filling configuration", "filling_storage")
                    if filling[hydro_id]
                    else ("has no filling configuration", "storage")
                )
                raise Refusal(
                    f"{INITIAL_FILE}: hydro {hydro_id} {reason}, so its initial "
                    f"storage belongs in `{expected}`, not `{array}` (semantic.5a.54)"
                )
            if hydro_id in seeds:
                raise Refusal(
                    f"{INITIAL_FILE}: hydro {hydro_id} has more than one storage "
                    "value, which one `storage_hm3` cannot hold"
                )
            seeds[hydro_id] = row["value_hm3"]
    return seeds


def _calendar(rows: Sequence[Mapping[str, Any]], key: str) -> list[tuple[date, date]]:
    """`(start, end)` of each stage in `start_date` order; a stage must last whole days."""
    stages: list[tuple[date, Any]] = []
    for i, row in enumerate(rows):
        try:
            stages.append(
                (date.fromisoformat(row["start_date"]), row["duration_hours"])
            )
        except ValueError as exc:
            raise Refusal(
                f"{pss.FILE_NAME}: `{key}[{i}].start_date` is malformed ({exc})"
            ) from exc
    calendar: list[tuple[date, date]] = []
    for start, hours in sorted(stages, key=lambda stage: stage[0]):
        if hours <= 0 or hours % _DAY_HOURS != 0:
            raise Refusal(
                f"`{key}` stage starting {start} lasts {hours:g} h; the new format "
                f"writes {{start_date, end_date}}, so a stage must last a positive "
                f"multiple of {_DAY_HOURS} h, and a duration is never rounded"
            )
        calendar.append((start, start + timedelta(days=int(hours) // _DAY_HOURS)))
    return calendar


def _with_ids(
    calendar: Sequence[tuple[date, date]], first_id: int
) -> list[dict[str, Any]]:
    return [
        {
            "id": first_id + k,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        for k, (start, end) in enumerate(calendar)
    ]


def _group(
    rows: Iterable[Mapping[str, Any]], id_field: str, fields: Sequence[str]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[id_field], []).append(
            {name: row[name] for name in fields}
        )
    return grouped


def _entries(
    id_field: str, sources: Mapping[str, Mapping[int, Any]]
) -> list[dict[str, Any]]:
    """One entry per plant id found in any source, ascending; a key is omitted without data."""
    ids = sorted(set[int]().union(*sources.values()))
    return [
        {
            id_field: plant_id,
            **{k: src[plant_id] for k, src in sources.items() if plant_id in src},
        }
        for plant_id in ids
    ]


def _dump(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def plan_deck(deck: Path) -> DeckPlan:
    """Decide what migrating `deck` writes and deletes, without touching it.

    Raises `Refusal` when the deck cannot be migrated.
    """
    legacy = tuple(
        name for name in (INITIAL_FILE, pss.FILE_NAME) if (deck / name).exists()
    )
    if (deck / BOUNDARY_FILE).exists():
        if legacy:
            raise Refusal(
                f"{BOUNDARY_FILE} is present together with the deprecated "
                f"{' and '.join(legacy)}; keep one file set (structural.3)"
            )
        return DeckPlan(deck, "already current", {}, ())

    stages_doc = _read_object(deck / STAGES_FILE, deck)
    if _declared(stages_doc, _THERMAL_KEY) or _declared(stages_doc, _WATER_KEY):
        raise Refusal(
            f"{STAGES_FILE} declares a post_study_* list beside the deprecated "
            f"{' and '.join(legacy)}, which ignores it (semantic.5a.71)"
        )

    initial = _read_object(deck / INITIAL_FILE, deck)
    if unknown := sorted(initial.keys() - _INITIAL_KEYS):
        raise Refusal(f"{INITIAL_FILE}: unknown key(s) {', '.join(unknown)}")
    seeds = _storage_seeds(deck, initial)

    has_post_study = (deck / pss.FILE_NAME).exists()
    post_doc = _read_object(deck / pss.FILE_NAME, deck) if has_post_study else {}
    file_format = pss.classify_file(post_doc if has_post_study else None)
    if file_format is FileFormat.MIXED:
        raise Refusal(
            f"{pss.FILE_NAME} declares `stages` together with `thermal_stages` or "
            "`water_bucket_stages`"
        )
    is_legacy = file_format is FileFormat.LEGACY
    thermal_key = "stages" if is_legacy else "thermal_stages"
    thermal_declared = is_legacy or _declared(post_doc, thermal_key)
    bounds_declared = _declared(post_doc, "thermal_bounds")
    if thermal_declared != bounds_declared:
        present, missing = (
            (thermal_key, "thermal_bounds")
            if thermal_declared
            else ("thermal_bounds", thermal_key)
        )
        raise Refusal(
            f"{pss.FILE_NAME} declares `{present}` without `{missing}` (semantic.5a.63)"
        )

    thermal_rows = (
        _rows(post_doc, thermal_key, pss.FILE_NAME, _STAGE_ROW)
        if thermal_declared
        else []
    )
    bound_rows = (
        _rows(post_doc, "thermal_bounds", pss.FILE_NAME, _BOUND_ROW)
        if bounds_declared
        else []
    )
    for row in bound_rows:
        index = row["post_study_stage_index"]
        if not 0 <= index < len(thermal_rows):
            raise Refusal(
                f"{pss.FILE_NAME}: `post_study_stage_index` {index} is out of range "
                f"for {len(thermal_rows)} thermal stage(s), as the legacy reader rejects"
            )

    water_declared = _declared(post_doc, "water_bucket_stages")
    if file_format is FileFormat.CURRENT:
        if not water_declared and pss.deepest_arc_hours(deck) > 0:
            raise Refusal(
                f"{pss.FILE_NAME} declares no `water_bucket_stages` on a deck with a "
                "travel-time arc, and a current file is never padded (semantic.5a.61)"
            )
        water_rows = (
            _rows(post_doc, "water_bucket_stages", pss.FILE_NAME, _STAGE_ROW)
            if water_declared
            else None
        )
    else:
        water_rows = pss.padded_water_calendar(
            deck, post_doc["stages"] if is_legacy else None
        )

    thermal = _calendar(thermal_rows, thermal_key)
    water = (
        _calendar(water_rows, "water_bucket_stages") if water_rows is not None else None
    )

    first_id = _first_post_study_id(stages_doc)
    lists: dict[str, list[dict[str, Any]]] = {}
    if thermal_declared:
        lists[_THERMAL_KEY] = _with_ids(thermal, first_id)
    if water is not None:
        lists[_WATER_KEY] = _with_ids(water, first_id + len(thermal))

    bounds = _group(
        sorted(
            (
                {**row, "stage_id": first_id + row["post_study_stage_index"]}
                for row in bound_rows
            ),
            key=lambda row: row["stage_id"],
        ),
        "thermal_id",
        ("stage_id", "cost_per_mwh", "min_mw", "max_mw"),
    )
    window = ("start_date", "end_date")
    boundary: dict[str, Any] = {"$schema": _SCHEMA_URL} if "$schema" in initial else {}
    boundary["hydros"] = _entries(
        "hydro_id",
        {
            "storage_hm3": seeds,
            "recent_inflows": _group(
                _rows(initial, "recent_observations", INITIAL_FILE, _FLOW_ROW),
                "hydro_id",
                (*window, "value_m3s"),
            ),
            "past_defluences": _group(
                _rows(initial, "past_defluences", INITIAL_FILE, _FLOW_ROW),
                "hydro_id",
                (*window, "value_m3s"),
            ),
        },
    )
    boundary["thermals"] = _entries(
        "thermal_id",
        {
            "commitments": _group(
                _rows(
                    initial,
                    "past_anticipated_commitments",
                    INITIAL_FILE,
                    _COMMITMENT_ROW,
                ),
                "thermal_id",
                (*window, "value_mw"),
            ),
            "post_study_bounds": bounds,
        },
    )

    writes = {BOUNDARY_FILE: _dump(boundary)}
    if lists:
        writes[STAGES_FILE] = _dump({**stages_doc, **lists})
    return DeckPlan(deck, "migrated", writes, legacy)


def layout_error(deck: Path) -> str | None:
    """Usage error for a path that is not a migratable deck directory, else `None`."""
    error = pss.layout_error(deck)
    if error or any((deck / name).is_file() for name in (BOUNDARY_FILE, INITIAL_FILE)):
        return error
    return f"{deck}: missing {BOUNDARY_FILE} or {INITIAL_FILE}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="boundary_conditions.py",
        description="Migrate decks' initial_conditions.json to boundary_conditions.json.",
    )
    parser.add_argument("decks", nargs="+", type=Path, metavar="DECK_DIR")
    try:
        decks: list[Path] = parser.parse_args(argv).decks
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    usage_errors = [error for deck in decks if (error := layout_error(deck))]
    if usage_errors:
        sys.stderr.write("".join(f"{error}\n" for error in usage_errors))
        return 2

    plans: list[DeckPlan] = []
    refusals: list[str] = []
    for deck in decks:
        try:
            plans.append(plan_deck(deck))
        except Refusal as refusal:
            refusals.append(f"{deck}: {refusal}")
    if refusals:
        sys.stderr.write("".join(f"{refusal}\n" for refusal in refusals))
        return 1

    for plan in plans:
        for name, text in plan.writes.items():
            write_atomic(plan.deck / name, text)
        for name in plan.deletes:
            (plan.deck / name).unlink(missing_ok=True)
        sys.stdout.write(f"{plan.deck}: {plan.outcome}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
