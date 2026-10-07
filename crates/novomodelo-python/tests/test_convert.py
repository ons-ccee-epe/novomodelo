"""Tests for the GIL-bound Python -> serde_json conversion helpers.

`cobre_python::convert::py_to_json_value` and `pydict_to_json_map` translate a
`config_overrides` mapping (Python objects) into a `serde_json::Map` under the
GIL, before the solver lifecycle releases it. Because those helpers require a
live interpreter, they cannot be unit-tested from a `#[cfg(test)]` Rust module
linked against the extension-module build. Instead they are exercised here
through `cobre.io.validate(..., config_overrides=...)`, mirroring how
`results::json_value_to_py` is covered from Python.

The conversion is verified per Python type by routing each value onto a real
config-schema field and asserting the merged config validates: a mis-shaped
conversion would fail `serde` deserialization and surface as a SchemaError, so
``valid is True`` confirms ``py_to_json_value`` produced exactly the right JSON
shape. Rejection of unsupported types (and non-str dict keys) is asserted via
``pytest.raises(ValueError)`` — those are malformed call payloads raised under
the GIL before the merge, not case-validation failures returned as data.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_convert.py -v
"""

from __future__ import annotations

import pytest

VALID_CASE_1DTOY = "examples/1dtoy"


def _validate(overrides: dict[str, object]) -> dict[str, object]:
    """Run validate() with the given override map and return the result dict."""
    import cobre.io  # noqa: PLC0415

    return cobre.io.validate(VALID_CASE_1DTOY, config_overrides=overrides)


# ── per-type round-trip (py_to_json_value) ────────────────────────────────────


def test_py_to_json_value_int_round_trips() -> None:
    """A Python int converts to a JSON number accepted by an i64 field."""
    result = _validate({"training.tree_seed": 7})
    assert result["valid"] is True, (
        f"int override must validate, errors: {result['errors']}"
    )


def test_py_to_json_value_bool_round_trips() -> None:
    """A Python bool converts to a JSON bool (not coerced to 0/1) for a bool field."""
    result = _validate({"simulation.enabled": False})
    assert result["valid"] is True, (
        f"bool override must validate, errors: {result['errors']}"
    )


def test_py_to_json_value_float_round_trips() -> None:
    """A Python float converts to a JSON number accepted by a float field."""
    result = _validate({"training.cut_selection.row_activity_tolerance": 1e-6})
    assert result["valid"] is True, (
        f"float override must validate, errors: {result['errors']}"
    )


def test_py_to_json_value_str_round_trips() -> None:
    """A Python str converts to a JSON string accepted by an enum field."""
    result = _validate({"modeling.inflow_non_negativity.method": "penalty"})
    assert result["valid"] is True, (
        f"str override must validate, errors: {result['errors']}"
    )


def test_py_to_json_value_none_round_trips() -> None:
    """A Python None converts to JSON null, accepted by an Option<i64> field."""
    result = _validate({"training.tree_seed": None})
    assert result["valid"] is True, (
        f"None override must validate as null, errors: {result['errors']}"
    )


def test_py_to_json_value_list_round_trips() -> None:
    """A Python list of dicts converts to a JSON array of objects (recursive).

    Exercises both the list branch and the nested-dict branch of
    py_to_json_value in one shot: stopping_rules is a list whose elements are
    objects, so a mis-shaped conversion of either would fail deserialization.
    """
    result = _validate(
        {"training.stopping_rules": [{"type": "iteration_limit", "limit": 5}]}
    )
    assert result["valid"] is True, (
        f"list override must validate, errors: {result['errors']}"
    )


def test_py_to_json_value_dict_round_trips() -> None:
    """A Python dict converts to a JSON object (recursive) for an object field."""
    result = _validate(
        {
            "training.scenario_source": {
                "seed": 1,
                "inflow": {"scheme": "in_sample"},
                "load": {"scheme": "in_sample"},
                "ncs": {"scheme": "in_sample"},
            }
        }
    )
    assert result["valid"] is True, (
        f"dict override must validate, errors: {result['errors']}"
    )


# ── rejection paths (py_to_json_value / pydict_to_json_map) ────────────────────


def test_py_to_json_value_rejects_unsupported_type() -> None:
    """An override value of an unsupported type (set) raises ValueError.

    A set has no JSON representation in the supported-types table, so the
    conversion raises under the GIL before any merge or validation runs.
    """
    with pytest.raises(ValueError):
        _validate({"training.tree_seed": {1, 2, 3}})


def test_py_to_json_value_rejects_nan_float() -> None:
    """A NaN float has no JSON representation and raises ValueError."""
    with pytest.raises(ValueError):
        _validate({"training.cut_selection.row_activity_tolerance": float("nan")})


def test_py_to_json_value_rejects_infinite_float() -> None:
    """An infinite float has no JSON representation and raises ValueError."""
    with pytest.raises(ValueError):
        _validate({"training.cut_selection.row_activity_tolerance": float("inf")})


def test_pydict_to_json_map_rejects_nested_non_str_key() -> None:
    """A non-str key inside a nested dict value raises ValueError.

    py_to_json_value recurses into dict values via pydict_to_json_map, which
    rejects non-str keys. The top-level override map already has str keys, so
    this exercises the recursive key-type check.
    """
    with pytest.raises(ValueError):
        _validate({"training.scenario_source": {1: "in_sample"}})
