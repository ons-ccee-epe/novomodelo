#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Migrate decks' `post_study_stages.json` from the deprecated `stages` format.

Usage:
    python3 scripts/migrate/post_study_stages.py DECK_DIR [DECK_DIR ...]

Every deck is planned before any file is written: one refused deck writes
nothing, and each refusal is one stderr line naming the deck and the reason.
A deck already in the current format is left untouched, so a second run
changes nothing. Each deck prints one stdout line, `DECK: OUTCOME`:

  migrated         legacy file, travel-time arc: `water_bucket_stages` is inferred
  created          no file, travel-time arc: a file holding only `water_bucket_stages`
  renamed          legacy file, no arc, an anticipated thermal: `stages` becomes
                   `thermal_stages`, with `thermal_bounds`, verbatim
  cleared          legacy file, no arc, no anticipated thermal: the thermal pair is
                   dropped (a `thermal_stages` key there would only warn)
  already current  the file has no `stages` key
  no change        no file and no arc

With an anticipated thermal (an entry of `thermals[]` in `system/thermals.json`
whose `anticipated_config` is not null) the thermal pair is copied verbatim;
otherwise it is dropped. The water calendar starts at the study end and steps from
the last in-study stage (highest `id >= 0`) until it covers the deepest
`travel_time_hours` of a hydro with a downstream:

  - a 7-day stage steps 168 h;
  - a stage spanning one whole calendar month steps by successive calendar months;
  - any other whole-day length (sum of block hours) repeats itself;
  - a length that is not a whole number of days is refused.

Output layout: `$schema` (when present), `thermal_stages`, `thermal_bounds`,
`water_bucket_stages`; 2-space indentation; trailing newline.

Exit code 0 when every deck is migrated or already current, 1 when any deck is
refused, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

FILE_NAME = "post_study_stages.json"
_REQUIRED_FILES = ("stages.json", "system/hydros.json", "system/thermals.json")
_WEEK_DAYS = 7
_WEEK_HOURS = 168.0


class Refusal(Exception):
    """A deck cannot be migrated; the message is the reason."""


class FileFormat(Enum):
    ABSENT = "absent"
    LEGACY = "legacy"
    CURRENT = "current"
    MIXED = "mixed"


@dataclass(frozen=True)
class LastStage:
    start: date
    end: date
    hours: float


@dataclass(frozen=True)
class DeckPlan:
    deck: Path
    outcome: str
    text: str | None


def classify_file(doc: dict[str, Any] | None) -> FileFormat:
    """Classify a parsed `post_study_stages.json`; `None` is an absent file."""
    if doc is None:
        return FileFormat.ABSENT
    if "stages" not in doc:
        return FileFormat.CURRENT
    if "thermal_stages" in doc or "water_bucket_stages" in doc:
        return FileFormat.MIXED
    return FileFormat.LEGACY


def _spans_one_month(stage: LastStage) -> bool:
    days = calendar.monthrange(stage.start.year, stage.start.month)[1]
    return stage.start.day == 1 and stage.end == stage.start + timedelta(days=days)


def _fixed_step_hours(stage: LastStage) -> float:
    if (stage.end - stage.start).days == _WEEK_DAYS:
        return _WEEK_HOURS
    if stage.hours <= 0 or stage.hours % 24:
        raise Refusal(
            f"the last in-study stage lasts {stage.hours:g} h, not a positive whole number of days"
        )
    return stage.hours


def infer_water_stages(last: LastStage, deepest_hours: float) -> list[dict[str, Any]]:
    """Water calendar from the study end, covering `deepest_hours` (equality covers).

    Raises `Refusal` when the last in-study stage has no whole-day length to repeat.
    """
    fixed = None if _spans_one_month(last) else _fixed_step_hours(last)
    stages: list[dict[str, Any]] = []
    start, covered = last.end, 0.0
    while covered < deepest_hours:
        hours = (
            fixed
            if fixed is not None
            else calendar.monthrange(start.year, start.month)[1] * 24.0
        )
        stages.append({"start_date": start.isoformat(), "duration_hours": hours})
        start += timedelta(hours=hours)
        covered += hours
    return stages


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


def _entries(deck: Path, relative: str, key: str) -> list[dict[str, Any]]:
    entries = _read_object(deck / relative, deck).get(key)
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise Refusal(f"{relative}: `{key}` must be a list of objects")
    return entries


def _declares_arc(hydro: dict[str, Any]) -> bool:
    hours = hydro.get("travel_time_hours")
    return (
        isinstance(hours, int | float)
        and hours > 0
        and hydro.get("downstream_id") is not None
    )


def _last_study_stage(deck: Path) -> LastStage:
    in_study = [
        stage
        for stage in _entries(deck, "stages.json", "stages")
        if isinstance(stage.get("id"), int) and stage["id"] >= 0
    ]
    if not in_study:
        raise Refusal("stages.json has no in-study stage (id >= 0)")
    stage = max(in_study, key=lambda s: s["id"])
    try:
        return LastStage(
            date.fromisoformat(stage["start_date"]),
            date.fromisoformat(stage["end_date"]),
            float(sum(block["hours"] for block in stage["blocks"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Refusal(
            f"stages.json: stage {stage['id']} is malformed ({exc!r})"
        ) from exc


def layout_error(deck: Path) -> str | None:
    """Usage error for a path that is not a deck directory, else `None`."""
    if not deck.is_dir():
        return f"{deck}: not a directory"
    missing = [name for name in _REQUIRED_FILES if not (deck / name).is_file()]
    return f"{deck}: missing {', '.join(missing)}" if missing else None


def plan_deck(deck: Path) -> DeckPlan:
    """Decide what migrating `deck` writes, without writing it.

    Raises `Refusal` when the deck cannot be migrated.
    """
    path = deck / FILE_NAME
    doc = _read_object(path, deck) if path.exists() else None
    file_format = classify_file(doc)
    if file_format is FileFormat.MIXED:
        raise Refusal(
            f"{FILE_NAME} declares `stages` together with `thermal_stages` or `water_bucket_stages`"
        )
    if file_format is FileFormat.CURRENT:
        return DeckPlan(deck, "already current", None)

    deepest_hours = max(
        (
            hydro["travel_time_hours"]
            for hydro in _entries(deck, "system/hydros.json", "hydros")
            if _declares_arc(hydro)
        ),
        default=0.0,
    )
    has_arc = deepest_hours > 0
    anticipates_thermal = any(
        thermal.get("anticipated_config") is not None
        for thermal in _entries(deck, "system/thermals.json", "thermals")
    )

    migrated: dict[str, Any] = {}
    if doc is None:
        outcome = "created" if has_arc else "no change"
    else:
        if "thermal_bounds" not in doc:
            raise Refusal(f"{FILE_NAME} declares `stages` without `thermal_bounds`")
        if "$schema" in doc:
            migrated["$schema"] = doc["$schema"]
        if anticipates_thermal:
            migrated["thermal_stages"] = doc["stages"]
            migrated["thermal_bounds"] = doc["thermal_bounds"]
        outcome = (
            "migrated" if has_arc else "renamed" if anticipates_thermal else "cleared"
        )
    if has_arc:
        migrated["water_bucket_stages"] = infer_water_stages(
            _last_study_stage(deck), deepest_hours
        )
    if outcome == "no change":
        return DeckPlan(deck, outcome, None)
    return DeckPlan(
        deck, outcome, json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
    )


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` through a sibling temp file and `os.replace`."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="post_study_stages.py",
        description="Migrate decks' post_study_stages.json to the current format.",
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
        if plan.text is not None:
            write_atomic(plan.deck / FILE_NAME, plan.text)
        sys.stdout.write(f"{plan.deck}: {plan.outcome}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
