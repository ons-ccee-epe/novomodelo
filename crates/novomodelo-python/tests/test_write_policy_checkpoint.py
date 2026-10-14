"""Integration tests for novomodelo.write_policy_checkpoint.

Verifies that a policy checkpoint authored from plain Python dicts round-trips
through novomodelo.results.load_policy — the read path whose emitted dict shapes
write_policy_checkpoint's input mirrors — and that malformed input raises a
clear error naming the offending stage/cut.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_write_policy_checkpoint.py
"""

import json
import math
import pathlib
import shutil
from typing import Any, Optional

import pyarrow.parquet as pq
import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[3]
_TOY_CASE = _REPO_ROOT / "examples" / "1dtoy"

_POOL_COST_SCALE_FACTOR = 2_500_000.0


def _make_metadata(
    cost_scale_factor: Optional[float] = 2_500_000.0,
    season_manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    producer: dict[str, Any] = {
        "completed_iterations": 5,
        "final_lower_bound": 123.45,
        "best_upper_bound": 130.0,
        "max_iterations": 10,
        "forward_passes": 4,
        "warm_start_cuts": 0,
        "warm_start_counts": [0],
        "rng_seed": 42,
        "total_visited_states": 0,
        "training_block_mode": "parallel",
        "training_block_mode_per_stage": [],
    }
    if cost_scale_factor is not None:
        producer["cost_scale_factor"] = cost_scale_factor
    metadata = {
        "created_at": "2026-07-30T00:00:00Z",
        "num_stages": 1,
        "producer": producer,
    }
    if season_manifest is not None:
        metadata["season_manifest"] = season_manifest
    return metadata


def _make_stage_cuts(
    cost_scale_factor: float = _POOL_COST_SCALE_FACTOR,
) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": 0,
            "state_dimension": 3,
            "capacity": 10,
            "cost_scale_factor": cost_scale_factor,
            "cuts": [
                {
                    "slot_index": 0,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 42.0,
                    "coefficients": [1.0, 2.0, 3.0],
                    "is_active": True,
                },
                {
                    "slot_index": 1,
                    "iteration": 1,
                    "forward_pass_index": 1,
                    "intercept": 10.5,
                    "coefficients": [0.5, -1.5, 2.5],
                    "is_active": True,
                },
            ],
        }
    ]


def test_write_policy_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    """A synthetic checkpoint written from dicts reads back with matching cuts
    and metadata, including the cost_scale_factor provenance marker.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), _make_metadata()
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))

    assert loaded["metadata"]["format_version"] == 4
    assert loaded["metadata"]["producer"]["cost_scale_factor"] == pytest.approx(
        2_500_000.0
    )
    assert loaded["metadata"]["producer"]["completed_iterations"] == 5

    assert len(loaded["stage_cuts"]) == 1
    stage = loaded["stage_cuts"][0]
    assert stage["stage_id"] == 0
    assert stage["state_dimension"] == 3
    assert stage["capacity"] == 10
    assert stage["cost_scale_factor"] == pytest.approx(_POOL_COST_SCALE_FACTOR)
    assert stage["populated_count"] == 2, "populated_count must default to len(cuts)"

    cuts = stage["cuts"]
    assert len(cuts) == 2
    assert [cut["slot_index"] for cut in cuts] == [0, 1]
    assert "cut_id" not in cuts[0]
    assert cuts[0]["intercept"] == pytest.approx(42.0)
    assert cuts[0]["coefficients"] == pytest.approx([1.0, 2.0, 3.0])
    assert cuts[0]["scored"] is False
    assert "active_count" not in cuts[0]
    assert cuts[1]["coefficients"] == pytest.approx([0.5, -1.5, 2.5])


def test_write_policy_checkpoint_stamps_the_running_software(
    tmp_path: pathlib.Path,
) -> None:
    """A caller-supplied software identity is ignored; the checkpoint always
    records the running software and version.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    metadata = _make_metadata()
    metadata["software"] = "another-program"
    metadata["software_version"] = "0.13.0"
    metadata["novomodelo_version"] = "0.13.0"

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), metadata
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["software"] == "novomodelo"
    assert loaded["metadata"]["software_version"] == novomodelo.__version__


def test_write_policy_checkpoint_cost_scale_factor_omitted_reads_as_none(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting cost_scale_factor from the producer block reads back as None: it
    is provenance only, and the pool's own factor is what the values are in.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(cost_scale_factor=None),
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["producer"]["cost_scale_factor"] is None
    assert loaded["stage_cuts"][0]["cost_scale_factor"] == pytest.approx(
        _POOL_COST_SCALE_FACTOR
    )


def test_write_policy_checkpoint_lower_bound_history_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    """The producer's lower_bound_history reads back bit for bit, -0.0 included."""
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    history = [130.0, -0.0, 2.2250738585072014e-308, 123.45]
    metadata = _make_metadata()
    metadata["producer"]["lower_bound_history"] = history

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), metadata
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    recorded = loaded["metadata"]["producer"]["lower_bound_history"]
    assert [v.hex() for v in recorded] == [v.hex() for v in history]


def test_write_policy_checkpoint_lower_bound_history_omitted_reads_as_empty(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting lower_bound_history from the producer dict writes an empty series."""
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    metadata = _make_metadata()
    assert "lower_bound_history" not in metadata["producer"]

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), metadata
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["producer"]["lower_bound_history"] == []


def test_write_policy_checkpoint_coefficient_length_mismatch_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A cut whose coefficients length disagrees with its stage's
    state_dimension raises ValueError naming the stage and cut.
    """
    import novomodelo  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["cuts"][0]["coefficients"] = [1.0, 2.0]  # state_dimension is 3

    with pytest.raises(ValueError, match=r"stage 0 cut at slot 0.*coefficients"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )


def test_write_policy_checkpoint_defaults_apply_when_keys_omitted(
    tmp_path: pathlib.Path,
) -> None:
    """warm_start_count and entity_manifest default when their keys are absent
    from the stage_cuts dict.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    assert "warm_start_count" not in stage_cuts[0]
    assert "entity_manifest" not in stage_cuts[0]

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["stage_cuts"][0]["warm_start_count"] == 0


# EntityType discriminants from schemas/policy.fbs (owned by novomodelo-sddp, mirrored
# here for the manifest a bridge-style caller supplies).
_ENTITY_TYPE_HYDRO_STORAGE = 0
_ENTITY_TYPE_HYDRO_INFLOW_LAG = 1


def _storage_only_stage_cuts(
    hydro_ids: list[int], lag_coefficients: dict[int, list[float]]
) -> list[dict[str, Any]]:
    """A single stage whose manifest is one HydroStorage slot per hydro (no lag
    slots — the DECOMP bootstrap shape), one cut carrying storage-aligned
    coefficients plus keyed inflow_lag_coefficients.
    """
    manifest = [
        {
            "entity_type": _ENTITY_TYPE_HYDRO_STORAGE,
            "entity_id": hid,
            "subindex": 0,
            "was_active": True,
        }
        for hid in hydro_ids
    ]
    return [
        {
            "stage_id": 0,
            "state_dimension": len(hydro_ids),
            "capacity": 4,
            "cost_scale_factor": _POOL_COST_SCALE_FACTOR,
            "entity_manifest": manifest,
            "cuts": [
                {
                    "slot_index": 0,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 7.0,
                    "coefficients": [float(hid) for hid in hydro_ids],
                    "inflow_lag_coefficients": lag_coefficients,
                    "is_active": True,
                }
            ],
        }
    ]


def test_write_policy_checkpoint_reserves_inflow_lag_slots(
    tmp_path: pathlib.Path,
) -> None:
    """inflow_lag_depth=N widens the written manifest with N canonical
    HydroInflowLag slots per storage hydro, self-describing depth N, and places
    each keyed pi_qafl coefficient at its (hydro, depth) position.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    hydro_ids = [1, 2]
    # hydro 1: depth1=1.1, depth2=1.2; hydro 2: depth1=2.1, depth2 defaults 0.0.
    lag_coefficients = {1: [1.1, 1.2], 2: [2.1]}
    stage_cuts = _storage_only_stage_cuts(hydro_ids, lag_coefficients)

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"),
        stage_cuts,
        _make_metadata(),
        inflow_lag_depth=2,
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    stage = loaded["stage_cuts"][0]

    # 2 storage + 2 hydros × 2 depths = 6 slots.
    assert stage["state_dimension"] == 6
    manifest = stage["entity_manifest"]
    assert len(manifest) == 6

    lag_slots = [
        s for s in manifest if s["entity_type"] == _ENTITY_TYPE_HYDRO_INFLOW_LAG
    ]
    assert len(lag_slots) == 4
    # Self-describes depth 2: deepest 1-based lag subindex is 2.
    assert max(s["subindex"] for s in lag_slots) == 2

    # Coefficients: storage[1.0, 2.0] ++ lag-major (h1,d1),(h2,d1),(h1,d2),(h2,d2).
    coeffs = stage["cuts"][0]["coefficients"]
    assert coeffs == pytest.approx([1.0, 2.0, 1.1, 2.1, 1.2, 0.0])


def test_write_policy_checkpoint_inflow_lag_depth_absent_is_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting inflow_lag_depth (or passing 0) writes the exact same bytes as
    not passing it — the reservation path is inert by default.
    """
    import novomodelo  # noqa: PLC0415

    def _digest(policy_dir: pathlib.Path) -> dict[str, bytes]:
        return {
            p.name: p.read_bytes() for p in sorted((policy_dir / "cuts").glob("*.bin"))
        }

    stage_cuts = _storage_only_stage_cuts([1, 2], {})

    default_dir = tmp_path / "default"
    novomodelo.write_policy_checkpoint(str(default_dir), stage_cuts, _make_metadata())

    zero_dir = tmp_path / "zero"
    novomodelo.write_policy_checkpoint(
        str(zero_dir), stage_cuts, _make_metadata(), inflow_lag_depth=0
    )

    assert _digest(default_dir) == _digest(zero_dir)


def test_write_policy_checkpoint_unplaceable_lag_coefficient_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A keyed inflow-lag coefficient for a hydro with no storage slot is
    unplaceable and raises ValueError naming the hydro — never silently dropped.
    """
    import novomodelo  # noqa: PLC0415

    # hydro 99 is not among the storage hydros [1, 2].
    stage_cuts = _storage_only_stage_cuts([1, 2], {99: [0.5]})

    with pytest.raises(ValueError, match=r"hydro 99"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"),
            stage_cuts,
            _make_metadata(),
            inflow_lag_depth=2,
        )


def test_write_policy_checkpoint_accepts_self_describing_stage_fields(
    tmp_path: pathlib.Path,
) -> None:
    """The binding accepts the per-stage self-describing facts (cost_scale_factor,
    node_id, graph_stage_id) that the CLI writer emits — Python write-parity — and
    produces a checkpoint whose read surfaces those facts again.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["cost_scale_factor"] = 2_500_000.0
    stage_cuts[0]["node_id"] = 3
    stage_cuts[0]["graph_stage_id"] = 7

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    stage = loaded["stage_cuts"][0]
    assert stage["state_dimension"] == 3
    assert len(stage["cuts"]) == 2
    assert stage["node_id"] == 3
    assert stage["graph_stage_id"] == 7
    assert stage["cost_scale_factor"] == pytest.approx(2_500_000.0)
    assert loaded["metadata"]["producer"]["cost_scale_factor"] == pytest.approx(
        2_500_000.0
    )


def test_load_policy_self_describing_fields_survive_rewrite(
    tmp_path: pathlib.Path,
) -> None:
    """A non-sentinel node_id/graph_stage_id and a stage-level cost_scale_factor
    survive a load_policy -> write_policy_checkpoint -> reload cycle — the
    load -> edit -> write round-trip the binding advertises.

    Regression: load_policy previously omitted these per-stage fields, so the
    rewrite defaulted the missing keys back to the -1 node/graph sentinel (and
    cost_scale_factor to None), collapsing a genuine single-node pool and
    breaking boundary-cut load. This test fails unless load_policy surfaces them.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["node_id"] = 3
    stage_cuts[0]["graph_stage_id"] = 7
    stage_cuts[0]["cost_scale_factor"] = 7_777.0

    # metadata carries no cost_scale_factor, so only the surfaced stage-level
    # value can keep it non-None through the rewrite — isolating it from the
    # producer-block fallback that would otherwise mask a dropped field.
    metadata = _make_metadata(cost_scale_factor=None)

    first_dir = tmp_path / "first"
    novomodelo.write_policy_checkpoint(str(first_dir / "policy"), stage_cuts, metadata)

    loaded = novomodelo.results.load_policy(str(first_dir))

    second_dir = tmp_path / "second"
    novomodelo.write_policy_checkpoint(
        str(second_dir / "policy"), loaded["stage_cuts"], metadata
    )
    reloaded = novomodelo.results.load_policy(str(second_dir))

    stage = reloaded["stage_cuts"][0]
    assert stage["node_id"] == 3
    assert stage["graph_stage_id"] == 7
    assert stage["cost_scale_factor"] == pytest.approx(7_777.0)


def test_write_policy_checkpoint_lag_coefficients_without_depth_raises(
    tmp_path: pathlib.Path,
) -> None:
    """inflow_lag_coefficients supplied without inflow_lag_depth raises
    ValueError naming the stage and cut — never silently dropped.
    """
    import novomodelo  # noqa: PLC0415

    stage_cuts = _storage_only_stage_cuts([1, 2], {1: [0.5]})

    with pytest.raises(ValueError, match=r"stage 0 cut .*inflow_lag_depth"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )


def test_write_policy_checkpoint_season_manifest_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    """A checkpoint authored with season_manifest reads back with the same
    descriptor, and writing the loaded metadata produces a byte-identical
    checkpoint file.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 3,
        "hydro_orders": [
            {"hydro_id": 1, "orders": [1, 2, 1]},
            {"hydro_id": 6, "orders": [2, 2, 1]},
        ],
    }

    first_dir = tmp_path / "first"
    novomodelo.write_policy_checkpoint(
        str(first_dir / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    loaded = novomodelo.results.load_policy(str(first_dir))
    assert loaded["metadata"]["season_manifest"] == season_manifest

    second_dir = tmp_path / "second"
    novomodelo.write_policy_checkpoint(
        str(second_dir / "policy"),
        loaded["stage_cuts"],
        loaded["metadata"],
    )

    first_manifest = (first_dir / "policy" / "manifest.bin").read_bytes()
    second_manifest = (second_dir / "policy" / "manifest.bin").read_bytes()
    assert first_manifest == second_manifest


def test_write_policy_checkpoint_season_manifest_omitted_is_absent(
    tmp_path: pathlib.Path,
) -> None:
    """A checkpoint authored with today's _make_metadata() (no season_manifest
    key) reads back with the absent descriptor.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), _make_metadata()
    )

    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["season_manifest"] == {
        "cycle_code": 255,
        "n_seasons": 0,
        "hydro_orders": [],
    }


def test_write_policy_checkpoint_season_manifest_unsorted_hydros_rejected_on_load(
    tmp_path: pathlib.Path,
) -> None:
    """hydro_orders with non-ascending hydro_id values: the write succeeds, and
    load_policy raises OutputError with 'not ascending by hydro_id'.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 2,
        "hydro_orders": [
            {"hydro_id": 6, "orders": [1, 2]},
            {"hydro_id": 1, "orders": [2, 1]},
        ],
    }

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    with pytest.raises(
        novomodelo.errors.OutputError, match=r"not ascending by hydro_id"
    ):
        novomodelo.results.load_policy(str(tmp_path))


def test_write_policy_checkpoint_season_manifest_order_length_rejected_on_load(
    tmp_path: pathlib.Path,
) -> None:
    """n_seasons=3 but one hydro with two orders: the write succeeds, and
    load_policy raises OutputError with 'expected n_seasons=3'.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 3,
        "hydro_orders": [
            {"hydro_id": 1, "orders": [1, 2]},
        ],
    }

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    with pytest.raises(novomodelo.errors.OutputError, match=r"expected n_seasons=3"):
        novomodelo.results.load_policy(str(tmp_path))


def test_write_policy_checkpoint_refuses_a_directory_holding_other_files(
    tmp_path: pathlib.Path,
) -> None:
    """A policy directory holding a file no checkpoint writer leaves there is
    refused with ValidationError, and the file survives.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.errors  # noqa: PLC0415

    policy = tmp_path / "policy"
    policy.mkdir()
    notes = policy / "notes.txt"
    notes.write_text("keep me")

    with pytest.raises(
        novomodelo.errors.ValidationError,
        match=r"notes\.txt, found in .*, is not part of a checkpoint",
    ):
        novomodelo.write_policy_checkpoint(
            str(policy), _make_stage_cuts(), _make_metadata()
        )

    assert notes.read_text() == "keep me"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy"]
    assert sorted(p.name for p in policy.iterdir()) == ["notes.txt"]


def _file_bytes(root: pathlib.Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _states_checkpoint(
    slots: tuple[int, ...] = (0, 2),
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """One pool of three scored cuts, with a stored trial state at each of `slots`."""
    manifest = [
        {
            "entity_type": _ENTITY_TYPE_HYDRO_STORAGE,
            "entity_id": hid,
            "subindex": 0,
            "was_active": True,
        }
        for hid in (1, 2, 3)
    ]
    stage_cuts = [
        {
            "stage_id": 0,
            "state_dimension": 3,
            "capacity": 4,
            "cost_scale_factor": 1.0,
            "entity_manifest": manifest,
            "cuts": [
                {
                    "slot_index": slot,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 1.0,
                    "coefficients": [1.0, 2.0, 3.0],
                    "is_active": True,
                    "scored": True,
                }
                for slot in range(3)
            ],
        }
    ]
    stage_states = [
        {
            "stage_id": 0,
            "state_dimension": 3,
            "count": len(slots),
            "slots": list(slots),
            "data": [0.5] * (3 * len(slots)),
        }
    ]
    metadata = _make_metadata()
    metadata["producer"]["total_visited_states"] = len(slots)
    metadata["producer"]["state_history"] = "whole"
    return stage_cuts, metadata, stage_states


_RETIRED_KEYS = [
    pytest.param("active_cut_indices", "stage_cuts[0]", id="active_cut_indices"),
    pytest.param("cut_id", "stage_cuts[0].cuts[1]", id="cut_id"),
    pytest.param("num_cut_rows", "stage_bases[0]", id="num_cut_rows"),
    pytest.param("num_columns", "stage_bases[0]", id="num_columns"),
    pytest.param("num_rows", "stage_bases[0]", id="num_rows"),
]


@pytest.mark.parametrize(("key", "position"), _RETIRED_KEYS)
def test_write_policy_checkpoint_retired_key_raises_naming_key_and_position(
    tmp_path: pathlib.Path, key: str, position: str
) -> None:
    import novomodelo  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_bases: list[dict[str, Any]] = [{"stage_id": 0, "iteration": 1, "entries": []}]
    holders = {
        "stage_cuts[0]": stage_cuts[0],
        "stage_cuts[0].cuts[1]": stage_cuts[0]["cuts"][1],
        "stage_bases[0]": stage_bases[0],
    }
    holders[position][key] = 0

    with pytest.raises(ValueError) as failure:
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata(), stage_bases
        )

    assert str(failure.value) == (
        f"{position}: key '{key}' is not part of a policy checkpoint "
        "(format 4); remove it"
    )
    assert not (tmp_path / "policy").exists()


def test_write_policy_checkpoint_ignores_unknown_keys_a_loaded_result_carries(
    tmp_path: pathlib.Path,
) -> None:
    import novomodelo  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["unrelated"] = 1
    stage_cuts[0]["cuts"][0]["unrelated"] = 1

    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    assert (tmp_path / "policy" / "manifest.bin").is_file()


@pytest.mark.parametrize(
    "factor",
    [None, 0.0, -1.0, math.nan, math.inf],
    ids=["missing", "zero", "negative", "nan", "inf"],
)
def test_write_policy_checkpoint_pool_cost_scale_factor_must_be_finite_and_positive(
    tmp_path: pathlib.Path, factor: Optional[float]
) -> None:
    """The producer block's factor (set by _make_metadata) is never a fallback."""
    import novomodelo  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    if factor is None:
        del stage_cuts[0]["cost_scale_factor"]
    else:
        stage_cuts[0]["cost_scale_factor"] = factor
    assert _make_metadata()["producer"]["cost_scale_factor"] > 0.0

    with pytest.raises(
        ValueError,
        match=r"stage_cuts\[0\] \(stage_id 0\): cost_scale_factor .*"
        r"units of this factor \(pass 1\.0 for values in currency units\)",
    ):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )

    assert not (tmp_path / "policy").exists()


@pytest.mark.parametrize("history", [True, False])
def test_write_policy_checkpoint_state_history_rejects_a_bool(
    tmp_path: pathlib.Path, history: bool
) -> None:
    import novomodelo  # noqa: PLC0415

    metadata = _make_metadata()
    metadata["producer"]["state_history"] = history

    with pytest.raises(ValueError, match=r"state_history (True|False) is a bool"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), _make_stage_cuts(), metadata
        )


def test_write_policy_checkpoint_whole_basis_store_needs_the_bases(
    tmp_path: pathlib.Path,
) -> None:
    """A checkpoint never claims a whole basis store it did not receive."""
    import novomodelo  # noqa: PLC0415
    import novomodelo.results  # noqa: PLC0415

    metadata = _make_metadata()
    metadata["producer"]["basis_store_complete"] = True

    with pytest.raises(ValueError, match=r"basis_store_complete.*no basis"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), _make_stage_cuts(), metadata
        )
    assert not (tmp_path / "policy").exists()

    stage_bases = [
        {
            "stage_id": 0,
            "iteration": 1,
            "entries": [
                {
                    "scenario": 0,
                    "column_status": [0],
                    "row_status": [1],
                    "piece_row_slots": [],
                }
            ],
        }
    ]
    novomodelo.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), metadata, stage_bases
    )
    loaded = novomodelo.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["producer"]["basis_store_complete"] is True


def test_write_policy_checkpoint_counter_flag_must_match_the_cuts(
    tmp_path: pathlib.Path,
) -> None:
    """activity_counters is a claim about every cut, and a cut without its
    counters breaks it.
    """
    import novomodelo  # noqa: PLC0415

    metadata = _make_metadata()
    metadata["producer"]["activity_counters"] = True

    with pytest.raises(ValueError, match=r"activity_counters is true"):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), _make_stage_cuts(), metadata
        )


_DATE_INCONSISTENT_MANIFEST = [
    {
        "entity_type": _ENTITY_TYPE_HYDRO_STORAGE,
        "entity_id": 1,
        "subindex": 0,
        "was_active": True,
        "reference_date": 20_311_201,
    }
]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(
            lambda pools: pools[0].update(entity_manifest=_DATE_INCONSISTENT_MANIFEST),
            r"pool 0 entity 1 subindex 0 is a storage slot, which carries no per-slot date",
            id="date_inconsistent_manifest",
        ),
        pytest.param(
            lambda pools: pools.append(dict(pools[0])),
            r"cuts/000\.bin: more than one payload carries this id",
            id="repeated_stage_id",
        ),
    ],
)
def test_write_policy_checkpoint_refuses_what_the_reader_refuses(
    tmp_path: pathlib.Path, edit: Any, message: str
) -> None:
    import novomodelo  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    edit(stage_cuts)

    with pytest.raises(ValueError, match=message):
        novomodelo.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )

    assert not (tmp_path / "policy").exists()


def test_write_policy_checkpoint_structural_error_raises_value_error_and_keeps_the_target(
    tmp_path: pathlib.Path,
) -> None:
    """A payload rule broken by the caller is a ValueError, not an OutputError
    (an OSError), and a refused write leaves the target directory as it was.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo.errors  # noqa: PLC0415

    stage_cuts, metadata, stage_states = _states_checkpoint()
    policy = tmp_path / "policy"
    novomodelo.write_policy_checkpoint(
        str(policy), stage_cuts, metadata, stage_states=stage_states
    )
    before = _file_bytes(tmp_path)
    assert any(name.startswith("policy/states/") for name in before)

    stage_states[0]["slots"] = [2, 0]
    for target in (policy, tmp_path / "fresh"):
        with pytest.raises(
            ValueError, match=r"states/000\.bin: slots not strictly ascending"
        ) as failure:
            novomodelo.write_policy_checkpoint(
                str(target), stage_cuts, metadata, stage_states=stage_states
            )
        assert not isinstance(failure.value, novomodelo.errors.OutputError)

    assert _file_bytes(tmp_path) == before


def _level1_case(
    root: pathlib.Path, *, iteration_limit: int, lookback: Optional[int]
) -> pathlib.Path:
    """1dtoy under `level1` with an active-cut budget, so a checkpoint holds
    trial states, scored flags and binding counters; simulation is off.
    """
    case = root / "case"
    shutil.copytree(_TOY_CASE, case, ignore=shutil.ignore_patterns("output"))
    selection: dict[str, Any] = {"method": "level1", "check_frequency": 1}
    if lookback is not None:
        selection["lookback_iterations"] = lookback
    config_path = case / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["stopping_rules"] = [
        {"type": "iteration_limit", "limit": iteration_limit}
    ]
    config["training"]["cut_selection"] = {
        "selection": selection,
        "max_active_per_stage": 4,
    }
    config["simulation"]["enabled"] = False
    config_path.write_text(json.dumps(config, indent=2))
    return case


def _write_back(loaded: dict[str, Any], policy: pathlib.Path) -> None:
    import novomodelo  # noqa: PLC0415

    novomodelo.write_policy_checkpoint(
        str(policy),
        loaded["stage_cuts"],
        loaded["metadata"],
        loaded["stage_bases"],
        loaded["stage_states"],
    )


def test_load_policy_round_trips_every_version_4_field(tmp_path: pathlib.Path) -> None:
    """Load -> write -> load returns equal dicts and rewrites every file of a
    CLI-shaped checkpoint byte for byte: the trial states, the scored flags, the
    binding counters, the basis entries and the manifest flags included.
    """
    import novomodelo.results  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    case = _level1_case(tmp_path, iteration_limit=6, lookback=4)
    original = tmp_path / "original"
    novomodelo.run.run(str(case), output_dir=str(original))

    loaded = novomodelo.results.load_policy(str(original))
    producer = loaded["metadata"]["producer"]
    cuts = [cut for pool in loaded["stage_cuts"] for cut in pool["cuts"]]
    assert loaded["stage_states"], "level1 must store trial states"
    assert producer["state_history"] == 4
    assert producer["activity_counters"] is True
    assert producer["basis_store_complete"] is True
    assert producer["total_visited_states"] == sum(
        states["count"] for states in loaded["stage_states"]
    )
    assert any(cut["scored"] for cut in cuts)
    assert any(not cut["is_active"] for cut in cuts), "the budget must evict"
    assert all("active_count" in cut and "last_active_iteration" in cut for cut in cuts)
    assert any(basis["entries"] for basis in loaded["stage_bases"])

    rewritten = tmp_path / "rewritten"
    _write_back(loaded, rewritten / "policy")

    assert novomodelo.results.load_policy(str(rewritten)) == loaded
    assert _file_bytes(rewritten / "policy") == _file_bytes(original / "policy")


def _convergence(output_dir: pathlib.Path) -> dict[str, list[Any]]:
    table = pq.read_table(output_dir / "training" / "convergence.parquet")
    return {
        name: table.column(name).to_pylist()
        for name in table.column_names
        if not name.endswith("_ms")
    }


def test_a_checkpoint_written_back_from_python_resumes(tmp_path: pathlib.Path) -> None:
    """A `level1` checkpoint loaded and written back by Python resumes as the
    original does: the same iterations and convergence output, and the same
    final policy files.
    """
    import novomodelo.results  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    case = _level1_case(tmp_path, iteration_limit=3, lookback=None)
    original = tmp_path / "original"
    novomodelo.run.run(str(case), output_dir=str(original))

    config_path = case / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 6}]
    config["policy"] = {"mode": "resume"}
    config_path.write_text(json.dumps(config, indent=2))

    from_original = tmp_path / "from_original"
    shutil.copytree(original, from_original)
    from_rewrite = tmp_path / "from_rewrite"
    _write_back(novomodelo.results.load_policy(str(original)), from_rewrite / "policy")

    results = {
        leg: novomodelo.run.run(str(case), output_dir=str(leg))
        for leg in (from_original, from_rewrite)
    }

    keys = ("iterations", "converged", "lower_bound", "upper_bound", "gap_percent")
    assert [results[from_rewrite][key] for key in keys] == [
        results[from_original][key] for key in keys
    ]
    assert results[from_original]["iterations"] == 6
    assert _convergence(from_rewrite) == _convergence(from_original)

    def policy_files(output_dir: pathlib.Path) -> dict[str, bytes]:
        return {
            name: data
            for name, data in _file_bytes(output_dir / "policy").items()
            if name != "manifest.bin"
        }

    assert policy_files(from_rewrite) == policy_files(from_original)
    assert any(name.startswith("states/") for name in policy_files(from_original))

    def manifest(output_dir: pathlib.Path) -> dict[str, Any]:
        metadata = novomodelo.results.load_policy(str(output_dir))["metadata"]
        del metadata["created_at"]
        return metadata

    assert manifest(from_rewrite) == manifest(from_original)
