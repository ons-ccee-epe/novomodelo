"""Table-driven parity between `novomodelo.io.validate` and `novomodelo.run.run` refusals.

One table, mirrored from `crates/novomodelo-cli/tests/refusal_parity.rs`: rows are
added in both files together, each with one of the three outcomes
(`BracketedRefusal`, `PlainRefusal`, `Warning`). The checker is not changed by
row additions.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_refusal_parity.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[3]


def _row(
    *,
    name: str,
    base_case: str,
    mutate: Callable[[pathlib.Path], None],
    outcome: str,
    fragment: str,
    kind: str | None = None,
    error_class_name: str | None = None,
) -> Any:
    return pytest.param(
        base_case, mutate, outcome, kind, fragment, error_class_name, id=name
    )


def _edit_json(path: pathlib.Path, edit: Callable[[Any], None]) -> None:
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value, indent=2))


def _negative_travel_time(case: pathlib.Path) -> None:
    def edit(hydros: Any) -> None:
        hydros["hydros"][0]["travel_time_hours"] = -1.0

    _edit_json(case / "system" / "hydros.json", edit)


def _release_before_downstream_entry(case: pathlib.Path) -> None:
    def edit(hydros: Any) -> None:
        hydros["hydros"][1]["entry_stage_id"] = 1

    _edit_json(case / "system" / "hydros.json", edit)


def _pumping_endpoint_exits_while_station_active(case: pathlib.Path) -> None:
    def edit(hydros: Any) -> None:
        hydros["hydros"][1]["exit_stage_id"] = 1

    _edit_json(case / "system" / "hydros.json", edit)


def _duplicate_january_season(case: pathlib.Path) -> None:
    def edit(stages: Any) -> None:
        stages["season_definitions"]["seasons"].append(
            {
                "id": 16,
                "label": "January bis",
                "month_start": 1,
                "day_start": 1,
                "month_end": 1,
                "day_end": 31,
            }
        )

    _edit_json(case / "stages.json", edit)


def _repeat_generic_constraint_block_argument(case: pathlib.Path) -> None:
    def edit(constraints: Any) -> None:
        constraints["constraints"][0]["expression"] = "thermal_generation(0, 0, 0)"

    _edit_json(case / "constraints" / "generic_constraints.json", edit)


def _set_policy_path(case: pathlib.Path, policy_path: str) -> None:
    def edit(config: Any) -> None:
        config.setdefault("policy", {})["path"] = policy_path

    _edit_json(case / "config.json", edit)


def _empty_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "")


def _current_directory_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, ".")


def _parent_directory_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "..")


def _output_directory_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, str(case / "output"))


def _climbing_back_into_the_output_directory_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "../output")


def _unrecognized_file_in_the_policy_directory(case: pathlib.Path) -> None:
    policy_dir = case / "output" / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "notes.txt").write_text("kept by the user")


def _simulation_family_child_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "simulation/costs/policy")


def _simulation_solver_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "simulation/solver")


def _simulation_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "simulation")


def _training_policy_path(case: pathlib.Path) -> None:
    _set_policy_path(case, "training")


def _historical_forward_scheme(case: pathlib.Path) -> None:
    def edit(config: Any) -> None:
        config["training"]["scenario_source"]["inflow"] = {"scheme": "historical"}

    _edit_json(case / "config.json", edit)


def _write_inflow_seasonal_stats_with_zero_deviation_at_stage_3(
    path: pathlib.Path,
) -> None:
    stage_ids = list(range(-2, 12))
    table = pa.table(
        {
            "hydro_id": pa.array([0] * len(stage_ids), type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "mean_m3s": pa.array([200.0] * len(stage_ids), type=pa.float64()),
            "std_m3s": pa.array(
                [0.0 if stage == 3 else 50.0 for stage in stage_ids],
                type=pa.float64(),
            ),
        }
    )
    pq.write_table(table, path, compression="none")


def _historical_forward_scheme_with_a_zero_deviation_season(
    case: pathlib.Path,
) -> None:
    def edit(config: Any) -> None:
        config["training"]["scenario_source"] = {
            "seed": 1,
            "inflow": {"scheme": "historical"},
        }
        config["estimation"] = {"max_order": 0}

    _edit_json(case / "config.json", edit)
    _write_inflow_seasonal_stats_with_zero_deviation_at_stage_3(
        case / "scenarios" / "inflow_seasonal_stats.parquet"
    )


def _time_limit_only_stopping_rules(case: pathlib.Path) -> None:
    def edit(config: Any) -> None:
        config["training"]["stopping_rules"] = [{"type": "time_limit", "seconds": 600}]

    _edit_json(case / "config.json", edit)


def _train_one_iteration(case: pathlib.Path) -> None:
    import novomodelo.run  # noqa: PLC0415

    def edit(config: Any) -> None:
        config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 1}]
        config["simulation"]["enabled"] = False

    _edit_json(case / "config.json", edit)
    novomodelo.run.run(str(case), output_dir=str(case / "output"))


def _restamp_policy_version(policy_dir: pathlib.Path) -> None:
    import novomodelo  # noqa: PLC0415

    manifest = policy_dir / "manifest.bin"
    data = manifest.read_bytes()
    running = novomodelo.__version__.encode()
    assert data.count(running) == 1, "the running version must occur once"
    other = (b"8" if running.startswith(b"9") else b"9") + running[1:]
    manifest.write_bytes(data.replace(running, other))


def _set_policy_mode(case: pathlib.Path, mode: str) -> None:
    def edit(config: Any) -> None:
        config.setdefault("policy", {})["mode"] = mode

    _edit_json(case / "config.json", edit)


def _empty_stopping_rules(case: pathlib.Path) -> None:
    def edit(config: Any) -> None:
        config["training"]["stopping_rules"] = []

    _edit_json(case / "config.json", edit)


def _select_simulation_only(case: pathlib.Path) -> None:
    def edit(config: Any) -> None:
        config["training"]["enabled"] = False
        config["simulation"]["enabled"] = True
        config["simulation"]["selection"] = {"method": "sampled", "num_scenarios": 1}

    _edit_json(case / "config.json", edit)


def _warm_start_policy_from_another_version(case: pathlib.Path) -> None:
    _train_one_iteration(case)
    _restamp_policy_version(case / "output" / "policy")
    _set_policy_mode(case, "warm_start")


def _resume_policy_from_another_version(case: pathlib.Path) -> None:
    _train_one_iteration(case)
    _restamp_policy_version(case / "output" / "policy")
    _set_policy_mode(case, "resume")


def _resume_without_the_trial_state_history(case: pathlib.Path) -> None:
    _train_one_iteration(case)
    _set_policy_mode(case, "resume")
    _set_cut_selection(case, {"selection": {"method": "level1", "check_frequency": 1}})


def _resume_from_a_partial_policy(case: pathlib.Path) -> None:
    """A resume that fails in-band leaves the partial policy in `policy.partial`;
    the healthy deck then resumes from it."""
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    healthy = {
        file: (case / file).read_bytes()
        for file in ("system/thermals.json", "constraints/generic_constraints.json")
    }
    _train_one_iteration(case)

    def resume_two_iterations(config: Any) -> None:
        config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 2}]

    def disable_slack(constraints: Any) -> None:
        constraints["constraints"][0]["slack"] = {"enabled": False}

    def raise_thermal_minimum(thermals: Any) -> None:
        thermals["thermals"][0]["generation"]["min_mw"] = 20.0

    _edit_json(case / "config.json", resume_two_iterations)
    _set_policy_mode(case, "resume")
    _edit_json(case / "constraints" / "generic_constraints.json", disable_slack)
    _edit_json(case / "system" / "thermals.json", raise_thermal_minimum)
    with pytest.raises(novomodelo.errors.SolverError):
        novomodelo.run.run(str(case), output_dir=str(case / "output"))
    for file, data in healthy.items():
        (case / file).write_bytes(data)
    _set_policy_path(case, "policy.partial")


def _simulation_only_policy_from_another_version(case: pathlib.Path) -> None:
    _train_one_iteration(case)
    _restamp_policy_version(case / "output" / "policy")
    _select_simulation_only(case)


def _warm_start_without_a_policy_directory(case: pathlib.Path) -> None:
    _set_policy_mode(case, "warm_start")


def _simulation_only_policy_with_unused_stored_bases(case: pathlib.Path) -> None:
    thermals = case / "system" / "thermals.json"

    def add_third_thermal(document: Any) -> None:
        document["thermals"].append(
            {
                "id": 2,
                "name": "UTE3",
                "operational_start_date": "2020-01-01",
                "bus_id": 0,
                "generation": {"min_mw": 0.0, "max_mw": 15.0},
                "cost_per_mwh": 20.0,
            }
        )

    _edit_json(thermals, add_third_thermal)
    _train_one_iteration(case)
    original = _REPO_ROOT / "examples" / "1dtoy" / "system" / "thermals.json"
    shutil.copy(original, thermals)
    _select_simulation_only(case)


def _set_cut_selection(case: pathlib.Path, cut_selection: dict[str, Any]) -> None:
    def edit(config: Any) -> None:
        config["training"]["cut_selection"] = cut_selection

    _edit_json(case / "config.json", edit)


def _removed_selection_method_lml1(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"selection": {"method": "lml1"}})


def _removed_selection_method_domination(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"selection": {"method": "domination"}})


def _tie_tolerance_under_level1(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"selection": {"method": "level1", "tie_tolerance": 1e-9}})


def _zero_selection_check_frequency(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"selection": {"method": "level1", "check_frequency": 0}})


def _lookback_window_below_the_check_frequency(case: pathlib.Path) -> None:
    _set_cut_selection(
        case,
        {
            "selection": {
                "method": "level1",
                "check_frequency": 2,
                "lookback_iterations": 1,
            }
        },
    )


def _negative_row_activity_tolerance(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"row_activity_tolerance": -1})


def _null_row_activity_tolerance(case: pathlib.Path) -> None:
    _set_cut_selection(case, {"row_activity_tolerance": None})


def _violation_tolerance_below_the_solver_tolerance(case: pathlib.Path) -> None:
    _set_cut_selection(
        case, {"selection": {"method": "dynamic", "violation_tolerance": 1e-10}}
    )


def _violation_tolerance_below_the_simulation_solver_tolerance(
    case: pathlib.Path,
) -> None:
    def edit(config: Any) -> None:
        config["simulation"]["solver"] = {"primal_feasibility_tolerance": 1e-7}

    _edit_json(case / "config.json", edit)
    _set_cut_selection(
        case, {"selection": {"method": "dynamic", "violation_tolerance": 1e-8}}
    )


def _active_row_cap_with_dynamic_selection(case: pathlib.Path) -> None:
    _set_cut_selection(
        case, {"selection": {"method": "dynamic"}, "max_active_per_stage": 8}
    )


def _active_row_cap_with_enumerated_training(case: pathlib.Path) -> None:
    def one_opening_per_stage(stages: Any) -> None:
        for stage in stages["stages"]:
            stage["num_openings"] = 1

    def enumerated(config: Any) -> None:
        config["training"]["selection"] = {"method": "enumerated"}

    _edit_json(case / "stages.json", one_opening_per_stage)
    _edit_json(case / "config.json", enumerated)
    _set_cut_selection(case, {"max_active_per_stage": 8})


def _unknown_simulation_cut_set(case: pathlib.Path) -> None:
    def edit(config: Any) -> None:
        config["simulation"]["cut_set"] = "every"

    _edit_json(case / "config.json", edit)


ROWS = [
    _row(
        name="travel_time_negative",
        base_case="deterministic/d44-travel-time-substage",
        mutate=_negative_travel_time,
        outcome="BracketedRefusal",
        kind="InvalidValue",
        fragment="travel_time_hours must be finite and >= 0.0",
        error_class_name="ValidationError",
    ),
    _row(
        name="travel_time_release_before_downstream_entry",
        base_case="deterministic/d44-travel-time-substage",
        mutate=_release_before_downstream_entry,
        outcome="BracketedRefusal",
        kind="BusinessRuleViolation",
        fragment="has not reached Operating status there",
        error_class_name="ValidationError",
    ),
    _row(
        name="pumping_station_active_after_endpoint_exit",
        base_case="deterministic/d35-pumping-commissioning",
        mutate=_pumping_endpoint_exits_while_station_active,
        outcome="BracketedRefusal",
        kind="BusinessRuleViolation",
        fragment="is not Operating there",
        error_class_name="ValidationError",
    ),
    _row(
        name="season_overlap_within_one_level",
        base_case="deterministic/d30-multi-resolution-monthly-quarterly",
        mutate=_duplicate_january_season,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="overlap within one resolution level",
        error_class_name="ValidationError",
    ),
    _row(
        name="generic_constraint_repeated_block_argument",
        base_case="deterministic/d13-generic-constraint",
        mutate=_repeat_generic_constraint_block_argument,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="repeated block argument in variable",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_empty",
        base_case="1dtoy",
        mutate=_empty_policy_path,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="names the output directory or one of its ancestors",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_current_directory",
        base_case="1dtoy",
        mutate=_current_directory_policy_path,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="names the output directory or one of its ancestors",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_parent_directory",
        base_case="1dtoy",
        mutate=_parent_directory_policy_path,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="names the output directory or one of its ancestors",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_naming_the_output_directory",
        base_case="1dtoy",
        mutate=_output_directory_policy_path,
        outcome="PlainRefusal",
        fragment="names the output directory or one of its ancestors",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_climbing_back_into_the_output_directory",
        base_case="1dtoy",
        mutate=_climbing_back_into_the_output_directory_policy_path,
        outcome="PlainRefusal",
        fragment="names the output directory or one of its ancestors",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_directory_holding_an_unrecognized_file",
        base_case="1dtoy",
        mutate=_unrecognized_file_in_the_policy_directory,
        outcome="PlainRefusal",
        fragment="is not part of a checkpoint",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_inside_a_simulation_family_directory",
        base_case="1dtoy",
        mutate=_simulation_family_child_policy_path,
        outcome="PlainRefusal",
        fragment="lies inside simulation/costs, which a run removes whole",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_naming_the_simulation_solver_directory",
        base_case="1dtoy",
        mutate=_simulation_solver_policy_path,
        outcome="PlainRefusal",
        fragment="names simulation/solver, which holds files a run writes",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_containing_a_cleared_directory",
        base_case="1dtoy",
        mutate=_simulation_policy_path,
        outcome="PlainRefusal",
        fragment="contains simulation/costs, which a run removes whole",
        error_class_name="ValidationError",
    ),
    _row(
        name="policy_path_containing_the_training_solver_directory",
        base_case="1dtoy",
        mutate=_training_policy_path,
        outcome="PlainRefusal",
        fragment="contains training/solver, which holds files a run writes",
        error_class_name="ValidationError",
    ),
    _row(
        name="historical_forward_scheme_without_inflow_history",
        base_case="1dtoy",
        mutate=_historical_forward_scheme,
        outcome="PlainRefusal",
        fragment="no valid historical windows found",
        error_class_name="ValidationError",
    ),
    _row(
        name="historical_forward_scheme_with_a_zero_deviation_season",
        base_case="deterministic/d26-estimated-par2",
        mutate=_historical_forward_scheme_with_a_zero_deviation_season,
        outcome="PlainRefusal",
        fragment="V2.3: historical library contains non-finite eta",
        error_class_name="ValidationError",
    ),
    _row(
        name="stopping_rules_without_iteration_limit_rule",
        base_case="deterministic/d01-thermal-dispatch",
        mutate=_time_limit_only_stopping_rules,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="field training.stopping_rules: must contain an iteration_limit rule",
        error_class_name="ValidationError",
    ),
    _row(
        name="empty_stopping_rules",
        base_case="deterministic/d01-thermal-dispatch",
        mutate=_empty_stopping_rules,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="field training.stopping_rules: must contain an iteration_limit rule",
        error_class_name="ValidationError",
    ),
    _row(
        name="warm_start_policy_from_another_version",
        base_case="1dtoy",
        mutate=_warm_start_policy_from_another_version,
        outcome="PlainRefusal",
        fragment="policy was written by",
        error_class_name="PolicyIncompatibleError",
    ),
    _row(
        name="resume_policy_from_another_version",
        base_case="1dtoy",
        mutate=_resume_policy_from_another_version,
        outcome="PlainRefusal",
        fragment="policy was written by",
        error_class_name="PolicyIncompatibleError",
    ),
    _row(
        name="resume_without_the_trial_state_history",
        base_case="1dtoy",
        mutate=_resume_without_the_trial_state_history,
        outcome="PlainRefusal",
        fragment=(
            "resume needs the trial-state history of "
            "training.cut_selection.selection.method = level1"
        ),
        error_class_name="PolicyIncompatibleError",
    ),
    _row(
        name="resume_from_a_partial_policy",
        base_case="deterministic/d13-generic-constraint",
        mutate=_resume_from_a_partial_policy,
        outcome="PlainRefusal",
        fragment="resume needs a whole checkpoint",
        error_class_name="PolicyIncompatibleError",
    ),
    _row(
        name="simulation_only_policy_from_another_version",
        base_case="1dtoy",
        mutate=_simulation_only_policy_from_another_version,
        outcome="PlainRefusal",
        fragment="policy was written by",
        error_class_name="PolicyIncompatibleError",
    ),
    _row(
        name="warm_start_without_a_policy_directory",
        base_case="1dtoy",
        mutate=_warm_start_without_a_policy_directory,
        outcome="PlainRefusal",
        fragment="Policy directory not found",
        error_class_name="ValidationError",
    ),
    _row(
        name="simulation_only_policy_with_unused_stored_bases",
        base_case="1dtoy",
        mutate=_simulation_only_policy_with_unused_stored_bases,
        outcome="Warning",
        fragment="stored bases not used",
    ),
    _row(
        name="removed_selection_method_lml1",
        base_case="1dtoy",
        mutate=_removed_selection_method_lml1,
        outcome="BracketedRefusal",
        kind="ParseError",
        fragment=(
            "the row-selection method `lml1` was removed: use `level1` "
            "(key `training.cut_selection.selection.method`)"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="removed_selection_method_domination",
        base_case="1dtoy",
        mutate=_removed_selection_method_domination,
        outcome="BracketedRefusal",
        kind="ParseError",
        fragment=(
            "the row-selection method `domination` was removed: use `level1` "
            "(key `training.cut_selection.selection.method`)"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="removed_tie_tolerance_key",
        base_case="1dtoy",
        mutate=_tie_tolerance_under_level1,
        outcome="BracketedRefusal",
        kind="ParseError",
        fragment=(
            "the key `tie_tolerance` was removed: `level1` compares row values "
            "exactly; delete the key (`training.cut_selection.selection`)"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="zero_selection_check_frequency",
        base_case="1dtoy",
        mutate=_zero_selection_check_frequency,
        outcome="BracketedRefusal",
        kind="ParseError",
        fragment="training.cut_selection.selection.check_frequency must be >= 1 (got 0)",
        error_class_name="ValidationError",
    ),
    _row(
        name="lookback_window_below_the_check_frequency",
        base_case="1dtoy",
        mutate=_lookback_window_below_the_check_frequency,
        outcome="PlainRefusal",
        fragment=(
            "cut_selection config error: cut_selection.lookback_iterations (1) "
            "must be >= cut_selection.check_frequency (2)"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="negative_row_activity_tolerance",
        base_case="1dtoy",
        mutate=_negative_row_activity_tolerance,
        outcome="PlainRefusal",
        fragment=(
            "cut_selection config error: cut_selection.row_activity_tolerance "
            "must be finite and >= 0 (got -1)"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="violation_tolerance_below_the_solver_tolerance",
        base_case="1dtoy",
        mutate=_violation_tolerance_below_the_solver_tolerance,
        outcome="PlainRefusal",
        fragment=(
            "training.cut_selection.selection.violation_tolerance (0.0000000001) "
            "is a relative factor and must be at least the primal feasibility "
            "tolerance of the backward solver profile"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="active_row_cap_with_dynamic_selection",
        base_case="1dtoy",
        mutate=_active_row_cap_with_dynamic_selection,
        outcome="PlainRefusal",
        fragment=(
            "training.cut_selection.max_active_per_stage is inadmissible with the "
            "dynamic cut selection method; a lazy solve takes its candidates from "
            "the active rows and dynamic runs no selection pass, so an evicted row "
            "never returns — remove the cap, or use the `level1` method"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="active_row_cap_with_enumerated_training",
        base_case="1dtoy",
        mutate=_active_row_cap_with_enumerated_training,
        outcome="PlainRefusal",
        fragment=(
            "training.cut_selection.max_active_per_stage is inadmissible under "
            "enumerated forward traversal; enumerated traversal records no binding "
            "activity, so the cap would evict by generation order alone — remove "
            "the cap, or use sampled forward selection"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="unknown_simulation_cut_set",
        base_case="1dtoy",
        mutate=_unknown_simulation_cut_set,
        outcome="BracketedRefusal",
        kind="SchemaViolation",
        fragment="`simulation.cut_set`: unknown variant `every`, expected `all` or `active`",
        error_class_name="ValidationError",
    ),
    _row(
        name="violation_tolerance_below_the_simulation_solver_tolerance",
        base_case="1dtoy",
        mutate=_violation_tolerance_below_the_simulation_solver_tolerance,
        outcome="PlainRefusal",
        fragment=(
            "training.cut_selection.selection.violation_tolerance (0.00000001) "
            "is a relative factor and must be at least the primal feasibility "
            "tolerance of the simulation solver profile (0.0000001); raise it, "
            "or omit it to use each phase's solver tolerance"
        ),
        error_class_name="ValidationError",
    ),
    _row(
        name="null_row_activity_tolerance",
        base_case="1dtoy",
        mutate=_null_row_activity_tolerance,
        outcome="BracketedRefusal",
        kind="ParseError",
        fragment="invalid type: null, expected f64",
        error_class_name="ValidationError",
    ),
]


def _reported_tail(
    text: str, fragment: str, anchor: str, marker: str | None = None
) -> str | None:
    for line in text.splitlines():
        if fragment in line and (marker is None or marker in line):
            start = line.find(anchor)
            return None if start < 0 else line[start:].rstrip()
    return None


def _parity_violations(
    outcome: str,
    kind: str | None,
    fragment: str,
    error_class: type[BaseException] | None,
    validate_result: dict[str, Any],
    run_error: BaseException | None,
    run_stderr: str,
) -> list[str]:
    anchor = f"[{kind}]" if outcome == "BracketedRefusal" else fragment
    violations: list[str] = []
    if outcome == "Warning":
        if not validate_result["valid"]:
            violations.append("validate reported the case invalid, expected valid")
        if run_error is not None:
            violations.append(f"run raised {run_error!r}, expected no error")
        validate_text = "\n".join(w["message"] for w in validate_result["warnings"])
        run_text = run_stderr
        run_marker: str | None = "warning:"
    else:
        if validate_result["valid"]:
            violations.append("validate reported the case valid, expected invalid")
        if error_class is None or not isinstance(run_error, error_class):
            violations.append(f"run raised {run_error!r}, expected {error_class}")
        validate_text = "\n".join(e["message"] for e in validate_result["errors"])
        run_text = "" if run_error is None else str(run_error)
        run_marker = None

    validate_tail = _reported_tail(validate_text, fragment, anchor)
    run_tail = _reported_tail(run_text, fragment, anchor, run_marker)
    if validate_tail is None:
        violations.append(
            f"validate reported no line containing {fragment!r} from {anchor!r}"
        )
    if run_tail is None:
        violations.append(
            f"run reported no line containing {fragment!r} from {anchor!r}"
        )
    if validate_tail is not None and run_tail is not None and validate_tail != run_tail:
        violations.append(
            "validate and run reported different lines: "
            f"validate {validate_tail!r}, run {run_tail!r}"
        )
    return violations


@pytest.mark.parametrize(
    ("base_case", "mutate", "outcome", "kind", "fragment", "error_class_name"), ROWS
)
def test_validate_and_run_report_identically(
    tmp_path: pathlib.Path,
    capfd: pytest.CaptureFixture[str],
    base_case: str,
    mutate: Callable[[pathlib.Path], None],
    outcome: str,
    kind: str | None,
    fragment: str,
    error_class_name: str | None,
) -> None:
    """`validate` and `run` report the row's refusal or warning with the same line."""
    import novomodelo.errors  # noqa: PLC0415
    import novomodelo.io  # noqa: PLC0415
    import novomodelo.run  # noqa: PLC0415

    case = tmp_path / "case"
    shutil.copytree(_REPO_ROOT / "examples" / base_case, case)
    mutate(case)

    out = case / "output"

    validate_result = novomodelo.io.validate(str(case), output_dir=str(out))
    run_error: BaseException | None
    try:
        novomodelo.run.run(str(case), output_dir=str(out))
        run_error = None
    except Exception as exc:
        run_error = exc
    run_stderr = capfd.readouterr().err
    error_class = (
        getattr(novomodelo.errors, error_class_name) if error_class_name else None
    )

    assert (
        _parity_violations(
            outcome,
            kind,
            fragment,
            error_class,
            validate_result,
            run_error,
            run_stderr,
        )
        == []
    )


_BRACKETED_LINE = (
    "[InvalidValue] system/hydros.json (Hydro 0): "
    "Hydro 0: travel_time_hours must be finite and >= 0.0, got -1"
)
_PLAIN_LINE = "V2.1: seasonless historical stages for hydro 3"
_WARNING_LINE = "stored basis is stale"


def _invalid(*messages: str) -> dict[str, Any]:
    return {
        "valid": False,
        "errors": [{"kind": "ConstraintError", "message": m} for m in messages],
        "warnings": [],
    }


def _valid(*messages: str) -> dict[str, Any]:
    return {
        "valid": True,
        "errors": [],
        "warnings": [
            {
                "kind": "ScenarioStatistics",
                "message": m,
                "file": "policy/manifest.bin",
                "entity": "basis 2",
            }
            for m in messages
        ],
    }


_HOLDS = [
    pytest.param(
        "BracketedRefusal",
        "InvalidValue",
        "travel_time_hours must be finite",
        _invalid(f"constraint violation: {_BRACKETED_LINE}"),
        ValueError(f"constraint violation: {_BRACKETED_LINE}"),
        "",
        id="BracketedRefusal",
    ),
    pytest.param(
        "PlainRefusal",
        None,
        "V2.1: seasonless historical stage",
        _invalid(f"stages.json: stochastic error: insufficient data: {_PLAIN_LINE}"),
        ValueError(f"stochastic error: insufficient data: {_PLAIN_LINE}"),
        "",
        id="PlainRefusal",
    ),
    pytest.param(
        "Warning",
        None,
        _WARNING_LINE,
        _valid(_WARNING_LINE),
        None,
        f"novomodelo-python: policy validation warning: {_WARNING_LINE}\n",
        id="Warning",
    ),
]

_MISMATCHES = [
    pytest.param(
        "BracketedRefusal",
        "InvalidValue",
        "travel_time_hours must be finite",
        _invalid(f"constraint violation: {_BRACKETED_LINE}"),
        ValueError("constraint violation: Hydro 0: travel_time_hours must be finite"),
        "",
        ("run reported no line",),
        id="BracketedRefusal",
    ),
    pytest.param(
        "PlainRefusal",
        None,
        "V2.1: seasonless historical stage",
        _invalid(f"stages.json: stochastic error: insufficient data: {_PLAIN_LINE}"),
        ValueError(
            "stochastic error: insufficient data: "
            "V2.1: seasonless historical stages for hydro 4"
        ),
        "",
        ("reported different lines",),
        id="PlainRefusal",
    ),
    pytest.param(
        "Warning",
        None,
        _WARNING_LINE,
        _valid(),
        ValueError("boom"),
        f"novomodelo-python: policy validation warning: {_WARNING_LINE}\n",
        ("run raised", "validate reported no line"),
        id="Warning",
    ),
]


@pytest.mark.parametrize(
    ("outcome", "kind", "fragment", "validate_result", "run_error", "run_stderr"),
    _HOLDS,
)
def test_parity_check_holds_for_each_outcome_shape(
    outcome: str,
    kind: str | None,
    fragment: str,
    validate_result: dict[str, Any],
    run_error: BaseException | None,
    run_stderr: str,
) -> None:
    """A matching validate/run pair of each shape yields no violation."""
    assert (
        _parity_violations(
            outcome,
            kind,
            fragment,
            ValueError,
            validate_result,
            run_error,
            run_stderr,
        )
        == []
    )


@pytest.mark.parametrize(
    (
        "outcome",
        "kind",
        "fragment",
        "validate_result",
        "run_error",
        "run_stderr",
        "expected",
    ),
    _MISMATCHES,
)
def test_parity_check_flags_each_outcome_shape_mismatch(
    outcome: str,
    kind: str | None,
    fragment: str,
    validate_result: dict[str, Any],
    run_error: BaseException | None,
    run_stderr: str,
    expected: tuple[str, ...],
) -> None:
    """A mismatched validate/run pair of each shape names the broken rule."""
    violations = _parity_violations(
        outcome,
        kind,
        fragment,
        ValueError,
        validate_result,
        run_error,
        run_stderr,
    )
    for needle in expected:
        assert any(needle in v for v in violations), violations
