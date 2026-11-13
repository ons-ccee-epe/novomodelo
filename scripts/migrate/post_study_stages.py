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

  migrated         legacy file, travel-time arc: `water_bucket_stages` starts from `stages`
  created          no file, travel-time arc: a file holding only `water_bucket_stages`
  renamed          legacy file, no arc, an anticipated thermal: `stages` becomes
                   `thermal_stages`, with `thermal_bounds`, verbatim
  cleared          legacy file, no arc, no anticipated thermal: the thermal pair is
                   dropped (a `thermal_stages` key there would only warn)
  already current  the file has no `stages` key
  no change        no file and no arc

With an anticipated thermal (an entry of `thermals[]` in `system/thermals.json`
whose `anticipated_config` is not null) the thermal pair is copied verbatim;
otherwise it is dropped. The water calendar is the one the deprecated format
resolves to, so migrating a deck does not change its results: the legacy `stages`
(none without a file), then copies of the last stage's length until the calendar
covers the deepest `travel_time_hours` of a hydro with a downstream. Without a
legacy stage the last stage is the last in-study stage (highest `id >= 0`, sum of
block hours). Each copy starts where the previous stage ends, its length rounded to
whole days; a copy shorter than half a day is refused.

Output layout: `$schema` (when present), `thermal_stages`, `thermal_bounds`,
`water_bucket_stages`; 2-space indentation; trailing newline.

Exit code 0 when every deck is migrated or already current, 1 when any deck is
refused, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

FILE_NAME = "post_study_stages.json"
_REQUIRED_FILES = ("stages.json", "system/hydros.json", "system/thermals.json")
_DAY_HOURS = 24


class Refusal(Exception):
    """A deck cannot be migrated; the message is the reason."""


class FileFormat(Enum):
    ABSENT = "absent"
    LEGACY = "legacy"
    CURRENT = "current"
    MIXED = "mixed"


@dataclass(frozen=True)
class LastStage:
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


def _whole_days(hours: float) -> int:
    """`hours` in days rounded half away from zero, as the validator dates a stage end."""
    return int(Decimal(hours / _DAY_HOURS).to_integral_value(rounding=ROUND_HALF_UP))


def legacy_water_seed(stages: object) -> list[dict[str, Any]]:
    """The legacy `stages` in `start_date` order, the order the runtime reads them in.

    Raises `Refusal` when an entry has no ISO `start_date` or no finite positive
    `duration_hours`.
    """
    if not isinstance(stages, list) or not all(isinstance(s, dict) for s in stages):
        raise Refusal(f"{FILE_NAME}: `stages` must be a list of objects")
    starts: list[date] = []
    for i, stage in enumerate(stages):
        hours = stage.get("duration_hours")
        try:
            starts.append(date.fromisoformat(stage["start_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise Refusal(
                f"{FILE_NAME}: `stages[{i}].start_date` is malformed ({exc!r})"
            ) from exc
        if (
            isinstance(hours, bool)
            or not isinstance(hours, int | float)
            or not math.isfinite(hours)
            or hours <= 0
        ):
            raise Refusal(
                f"{FILE_NAME}: `stages[{i}].duration_hours` must be a finite number > 0"
            )
    return [
        stage
        for _, stage in sorted(zip(starts, stages, strict=True), key=lambda p: p[0])
    ]


def water_stages(
    seed: Sequence[dict[str, Any]], last: LastStage, deepest_hours: float
) -> list[dict[str, Any]]:
    """The water calendar the deprecated format resolves to, covering `deepest_hours`.

    `seed` (the legacy `stages`, in `start_date` order) is kept verbatim and extended
    with copies of its last length, or of `last`'s when it is empty, until the total
    reaches `deepest_hours` (equality covers).

    Raises `Refusal` when a copy is needed and its length rounds to no whole day.
    """
    stages = list(seed)
    if stages:
        hours = float(stages[-1]["duration_hours"])
        start = date.fromisoformat(stages[-1]["start_date"]) + timedelta(
            days=_whole_days(hours)
        )
    else:
        hours, start = last.hours, last.end
    covered = 0.0
    for stage in stages:
        covered += stage["duration_hours"]
    if covered < deepest_hours and _whole_days(hours) < 1:
        raise Refusal(
            f"the last stage lasts {hours:g} h, under the whole day a water stage needs"
        )
    while covered < deepest_hours:
        stages.append({"start_date": start.isoformat(), "duration_hours": hours})
        start += timedelta(days=_whole_days(hours))
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
        seed = legacy_water_seed(doc["stages"]) if doc is not None else []
        migrated["water_bucket_stages"] = water_stages(
            seed, _last_study_stage(deck), deepest_hours
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
