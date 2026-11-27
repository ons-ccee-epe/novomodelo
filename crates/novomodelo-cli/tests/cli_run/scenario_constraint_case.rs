//! Readers and run invariants of the generic-constraint outputs (violations, activation, resolved
//! echo and violation cost): a small synthetic case runs in CI, a converted case is run by hand
//! through `NOVOMODELO_CONVERTED_CASE`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, ArrayAccessor, BooleanArray, Float64Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};
use tempfile::TempDir;

use super::common::{case_dir, copy_dir_recursive, write_file};
use super::run_case;
use super::scenario_constraint_case_golden;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ActivationRow {
    pub(crate) scenario_id: i32,
    pub(crate) stage_id: i32,
    pub(crate) node_id: i32,
    pub(crate) block_id: Option<i32>,
    pub(crate) constraint_id: i32,
    pub(crate) active: bool,
    pub(crate) bound_lower: Option<f64>,
    pub(crate) bound_upper: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ViolationRow {
    pub(crate) scenario_id: i32,
    pub(crate) stage_id: i32,
    pub(crate) node_id: i32,
    pub(crate) block_id: Option<i32>,
    pub(crate) constraint_id: i32,
    pub(crate) slack_value: f64,
    pub(crate) slack_cost: f64,
}

// Rationale (struct_excessive_bools): each flag is a Boolean column of the echo.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EchoRow {
    pub(crate) stage_id: i32,
    pub(crate) block_id: Option<i32>,
    pub(crate) constraint_id: i32,
    pub(crate) variable_kind: Option<String>,
    pub(crate) value: Option<f64>,
    pub(crate) bound_lower: Option<f64>,
    pub(crate) bound_upper: Option<f64>,
    pub(crate) has_active_when: bool,
    pub(crate) slack_enabled: bool,
    pub(crate) slack_penalty: Option<f64>,
    pub(crate) slack_penalty_basis: Option<String>,
    pub(crate) lower_per_solve: bool,
    pub(crate) upper_per_solve: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CostRow {
    pub(crate) scenario_id: i32,
    pub(crate) stage_id: i32,
    pub(crate) block_id: Option<i32>,
    pub(crate) generic_violation_cost: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConstraintOutputs {
    pub(crate) activation: Vec<ActivationRow>,
    pub(crate) violations: Vec<ViolationRow>,
    pub(crate) echo: Vec<EchoRow>,
    pub(crate) generic_violation_cost: Vec<CostRow>,
}

/// `(stage_id, block_id)`.
type Cell = (i32, Option<i32>);

fn column<'a, A: Array + 'static>(batch: &'a RecordBatch, name: &str) -> &'a A {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column `{name}` is missing"))
        .as_any()
        .downcast_ref::<A>()
        .unwrap_or_else(|| panic!("column `{name}` has an unexpected type"))
}

fn optional<A: ArrayAccessor>(array: &A, row: usize) -> Option<A::Item> {
    array.is_valid(row).then(|| array.value(row))
}

fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        let entries =
            fs::read_dir(dir).unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()));
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
}

fn read_rows<T>(files: &[PathBuf], rows_of: impl Fn(&RecordBatch) -> Vec<T>) -> Vec<T> {
    let mut rows = Vec::new();
    for path in files {
        let file = fs::File::open(path)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("{} must be a parquet file: {e}", path.display()))
            .build()
            .unwrap();
        for batch in reader {
            rows.extend(rows_of(&batch.unwrap()));
        }
    }
    rows
}

fn activation_rows(batch: &RecordBatch) -> Vec<ActivationRow> {
    let scenario_id = column::<Int32Array>(batch, "scenario_id");
    let stage_id = column::<Int32Array>(batch, "stage_id");
    let node_id = column::<Int32Array>(batch, "node_id");
    let block_id = column::<Int32Array>(batch, "block_id");
    let constraint_id = column::<Int32Array>(batch, "constraint_id");
    let active = column::<BooleanArray>(batch, "active");
    let bound_lower = column::<Float64Array>(batch, "bound_lower");
    let bound_upper = column::<Float64Array>(batch, "bound_upper");
    (0..batch.num_rows())
        .map(|i| ActivationRow {
            scenario_id: scenario_id.value(i),
            stage_id: stage_id.value(i),
            node_id: node_id.value(i),
            block_id: optional(&block_id, i),
            constraint_id: constraint_id.value(i),
            active: active.value(i),
            bound_lower: optional(&bound_lower, i),
            bound_upper: optional(&bound_upper, i),
        })
        .collect()
}

fn violation_rows(batch: &RecordBatch) -> Vec<ViolationRow> {
    let scenario_id = column::<Int32Array>(batch, "scenario_id");
    let stage_id = column::<Int32Array>(batch, "stage_id");
    let node_id = column::<Int32Array>(batch, "node_id");
    let block_id = column::<Int32Array>(batch, "block_id");
    let constraint_id = column::<Int32Array>(batch, "constraint_id");
    let slack_value = column::<Float64Array>(batch, "slack_value");
    let slack_cost = column::<Float64Array>(batch, "slack_cost");
    (0..batch.num_rows())
        .map(|i| ViolationRow {
            scenario_id: scenario_id.value(i),
            stage_id: stage_id.value(i),
            node_id: node_id.value(i),
            block_id: optional(&block_id, i),
            constraint_id: constraint_id.value(i),
            slack_value: slack_value.value(i),
            slack_cost: slack_cost.value(i),
        })
        .collect()
}

fn echo_rows(batch: &RecordBatch) -> Vec<EchoRow> {
    let stage_id = column::<Int32Array>(batch, "stage_id");
    let block_id = column::<Int32Array>(batch, "block_id");
    let constraint_id = column::<Int32Array>(batch, "constraint_id");
    let variable_kind = column::<StringArray>(batch, "variable_kind");
    let value = column::<Float64Array>(batch, "value");
    let bound_lower = column::<Float64Array>(batch, "bound_lower");
    let bound_upper = column::<Float64Array>(batch, "bound_upper");
    let has_active_when = column::<BooleanArray>(batch, "has_active_when");
    let slack_enabled = column::<BooleanArray>(batch, "slack_enabled");
    let slack_penalty = column::<Float64Array>(batch, "slack_penalty");
    let slack_penalty_basis = column::<StringArray>(batch, "slack_penalty_basis");
    let lower_per_solve = column::<BooleanArray>(batch, "lower_per_solve");
    let upper_per_solve = column::<BooleanArray>(batch, "upper_per_solve");
    (0..batch.num_rows())
        .map(|i| EchoRow {
            stage_id: stage_id.value(i),
            block_id: optional(&block_id, i),
            constraint_id: constraint_id.value(i),
            variable_kind: optional(&variable_kind, i).map(str::to_owned),
            value: optional(&value, i),
            bound_lower: optional(&bound_lower, i),
            bound_upper: optional(&bound_upper, i),
            has_active_when: has_active_when.value(i),
            slack_enabled: slack_enabled.value(i),
            slack_penalty: optional(&slack_penalty, i),
            slack_penalty_basis: optional(&slack_penalty_basis, i).map(str::to_owned),
            lower_per_solve: lower_per_solve.value(i),
            upper_per_solve: upper_per_solve.value(i),
        })
        .collect()
}

fn cost_rows(batch: &RecordBatch) -> Vec<CostRow> {
    let scenario_id = column::<Int32Array>(batch, "scenario_id");
    let stage_id = column::<Int32Array>(batch, "stage_id");
    let block_id = column::<Int32Array>(batch, "block_id");
    let generic_violation_cost = column::<Float64Array>(batch, "generic_violation_cost");
    (0..batch.num_rows())
        .map(|i| CostRow {
            scenario_id: scenario_id.value(i),
            stage_id: stage_id.value(i),
            block_id: optional(&block_id, i),
            generic_violation_cost: generic_violation_cost.value(i),
        })
        .collect()
}

/// Reads the generic-constraint output families of a run under `out`, each partition file in
/// path order, and sorts every vector by `(scenario_id, stage_id, block_id, constraint_id)` (the
/// echo has no scenario). `scenario_id` is a column and a directory name, so the partitions are
/// read file by file, not as a Hive dataset.
pub(crate) fn read_constraint_outputs(out: &Path) -> ConstraintOutputs {
    let simulation = out.join("simulation");
    let mut activation = read_rows(
        &parquet_files(&simulation.join("generic_activation")),
        activation_rows,
    );
    let mut violations = read_rows(
        &parquet_files(&simulation.join("violations/generic")),
        violation_rows,
    );
    let mut echo = read_rows(
        &[out.join("generic_constraints/resolved_echo.parquet")],
        echo_rows,
    );
    let mut generic_violation_cost =
        read_rows(&parquet_files(&simulation.join("costs")), cost_rows);

    activation.sort_by_key(|r| (r.scenario_id, r.stage_id, r.block_id, r.constraint_id));
    violations.sort_by_key(|r| (r.scenario_id, r.stage_id, r.block_id, r.constraint_id));
    echo.sort_by_key(|r| (r.stage_id, r.block_id, r.constraint_id));
    generic_violation_cost.sort_by_key(|r| (r.scenario_id, r.stage_id, r.block_id));
    ConstraintOutputs {
        activation,
        violations,
        echo,
        generic_violation_cost,
    }
}

fn training_convergence(out: &Path) -> (bool, String) {
    let metadata: Value =
        serde_json::from_str(&fs::read_to_string(out.join("training/metadata.json")).unwrap())
            .unwrap();
    let convergence = &metadata["convergence"];
    (
        convergence["achieved"]
            .as_bool()
            .expect("convergence.achieved must be a bool"),
        convergence["termination_reason"]
            .as_str()
            .expect("convergence.termination_reason must be a string")
            .to_owned(),
    )
}

fn assert_run_completed(out: &Path) {
    for marker in ["training/_SUCCESS", "simulation/_SUCCESS"] {
        assert!(out.join(marker).is_file(), "{marker} must exist");
    }
    let (_, termination_reason) = training_convergence(out);
    assert!(
        !matches!(termination_reason.as_str(), "error" | "graceful_shutdown"),
        "training ended with `{termination_reason}`"
    );
}

/// The constraints a solve records the activation of, under each `(stage_id, block_id)` the echo
/// holds them at: those with `active_when` or a per-solve side in some echo row.
fn activation_cells(echo: &[EchoRow]) -> BTreeMap<Cell, Vec<i32>> {
    let recorded: BTreeSet<i32> = echo
        .iter()
        .filter(|r| r.has_active_when || r.lower_per_solve || r.upper_per_solve)
        .map(|r| r.constraint_id)
        .collect();
    let mut cells: BTreeMap<Cell, BTreeSet<i32>> = BTreeMap::new();
    for row in echo.iter().filter(|r| recorded.contains(&r.constraint_id)) {
        cells
            .entry((row.stage_id, row.block_id))
            .or_default()
            .insert(row.constraint_id);
    }
    cells
        .into_iter()
        .map(|(cell, constraints)| (cell, constraints.into_iter().collect()))
        .collect()
}

fn assert_activation_follows_the_echo(outputs: &ConstraintOutputs) {
    let expected = activation_cells(&outputs.echo);
    let mut scenarios: BTreeMap<i32, BTreeMap<Cell, Vec<i32>>> = outputs
        .generic_violation_cost
        .iter()
        .map(|r| (r.scenario_id, BTreeMap::new()))
        .collect();
    for row in &outputs.activation {
        scenarios
            .entry(row.scenario_id)
            .or_default()
            .entry((row.stage_id, row.block_id))
            .or_default()
            .push(row.constraint_id);
    }
    for (scenario_id, cells) in scenarios {
        assert_eq!(
            cells, expected,
            "scenario {scenario_id}: the activation constraints per (stage, block) differ from \
             one row each for the echo's constraints with `active_when` or a per-solve side"
        );
    }
}

fn assert_inactive_rows_carry_no_bounds(activation: &[ActivationRow]) {
    for row in activation.iter().filter(|r| !r.active) {
        assert!(
            row.bound_lower.is_none() && row.bound_upper.is_none(),
            "an inactive row carries bounds: {row:?}"
        );
    }
}

fn assert_every_value_is_finite(outputs: &ConstraintOutputs) {
    let finite = |value: f64, family: &str, field: &str| {
        assert!(value.is_finite(), "{family}.{field} holds {value}");
    };
    for row in &outputs.activation {
        for value in [row.bound_lower, row.bound_upper].into_iter().flatten() {
            finite(value, "activation", "bound");
        }
    }
    for row in &outputs.violations {
        finite(row.slack_value, "violations", "slack_value");
        finite(row.slack_cost, "violations", "slack_cost");
    }
    for row in &outputs.echo {
        for value in [
            row.value,
            row.bound_lower,
            row.bound_upper,
            row.slack_penalty,
        ]
        .into_iter()
        .flatten()
        {
            finite(value, "echo", "f64 column");
        }
    }
    for row in &outputs.generic_violation_cost {
        finite(
            row.generic_violation_cost,
            "costs",
            "generic_violation_cost",
        );
    }
}

fn assert_slack_cost_closes_the_violation_cost(outputs: &ConstraintOutputs) {
    let mut cost: BTreeMap<(i32, i32, Option<i32>), f64> = BTreeMap::new();
    for row in &outputs.generic_violation_cost {
        *cost
            .entry((row.scenario_id, row.stage_id, row.block_id))
            .or_default() += row.generic_violation_cost;
    }
    let mut slack_cost: BTreeMap<(i32, i32, Option<i32>), f64> = BTreeMap::new();
    for row in &outputs.violations {
        let block = (row.scenario_id, row.stage_id, row.block_id);
        // A cost row with a null block totals its stage.
        let key = if cost.contains_key(&block) {
            block
        } else {
            (row.scenario_id, row.stage_id, None)
        };
        assert!(
            cost.contains_key(&key),
            "costs hold no row at (scenario_id, stage_id, block_id) {block:?} nor for its stage, \
             which has violations"
        );
        *slack_cost.entry(key).or_default() += row.slack_cost;
    }
    for (key, cost) in cost {
        let slack_cost = slack_cost.get(&key).copied().unwrap_or(0.0);
        let tolerance = (1e-9 * cost.abs().max(slack_cost.abs())).max(1e-6);
        assert!(
            (cost - slack_cost).abs() <= tolerance,
            "generic_violation_cost {cost} != sum of slack_cost {slack_cost} at \
             (scenario_id, stage_id, block_id) {key:?}"
        );
    }
}

/// Asserts what every run with generic constraints must satisfy: both phase markers exist and
/// training did not end in `error` or `graceful_shutdown`; in each simulated scenario a constraint
/// with `active_when` or a per-solve side in the echo has exactly one activation row per
/// `(stage_id, block_id)` the echo holds it at, and any other constraint has none; an inactive
/// row has no bounds; every `f64` read is finite; and per `(scenario_id, stage_id, block_id)` of
/// the costs the sum of `slack_cost` equals `generic_violation_cost` (relative 1e-9, absolute
/// floor 1e-6).
pub(crate) fn assert_constraint_run_invariants(out: &Path, outputs: &ConstraintOutputs) {
    assert_run_completed(out);
    assert!(
        !outputs.echo.is_empty() && !outputs.generic_violation_cost.is_empty(),
        "the run resolved no generic constraint or simulated no scenario"
    );
    assert_activation_follows_the_echo(outputs);
    assert_inactive_rows_carry_no_bounds(&outputs.activation);
    assert_every_value_is_finite(outputs);
    assert_slack_cost_closes_the_violation_cost(outputs);
}

/// The `(stage_id, block_id)` of every study stage of `case`.
fn cells(case: &Path) -> Vec<(i32, i32)> {
    let stages: Value =
        serde_json::from_str(&fs::read_to_string(case.join("stages.json")).unwrap()).unwrap();
    let id = |value: &Value| i32::try_from(value["id"].as_i64().unwrap()).unwrap();
    stages["stages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|stage| id(stage) >= 0)
        .flat_map(|stage| {
            let stage_id = id(stage);
            stage["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .map(move |block| (stage_id, id(block)))
        })
        .collect()
}

/// Per hour, at 10 times the largest deficit cost of the case's buses.
fn default_slack(case: &Path) -> Value {
    let buses: Value =
        serde_json::from_str(&fs::read_to_string(case.join("system/buses.json")).unwrap()).unwrap();
    let max_deficit_cost = buses["buses"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|bus| bus["deficit_segments"].as_array().unwrap())
        .map(|segment| segment["cost"].as_f64().unwrap())
        .fold(0.0, f64::max);
    json!({ "enabled": true, "penalty": 10.0 * max_deficit_cost, "penalty_basis": "per_hour" })
}

/// One bounds row with null endpoints per `(stage_id, block_id)` of `case` for each constraint
/// of `constraint_ids`, whose expressions carry their operators.
fn write_bounds_rows_per_cell(case: &Path, constraint_ids: &[i32]) {
    let rows: Vec<(i32, i32, i32)> = constraint_ids
        .iter()
        .flat_map(|&id| {
            cells(case)
                .into_iter()
                .map(move |(stage_id, block_id)| (id, stage_id, block_id))
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_lower", DataType::Float64, true),
        Field::new("bound_upper", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2))),
            Arc::new(Float64Array::from(vec![None; rows.len()])),
            Arc::new(Float64Array::from(vec![None; rows.len()])),
        ],
    )
    .unwrap();
    let path = case.join("constraints/generic_constraint_bounds.parquet");
    let mut writer = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// The synthetic case: `d02-single-hydro` (one 50 MW hydro, one bus) with a simulation of 2
/// sampled scenarios and four constraints, one per case: 701 is violated (the plant caps at
/// 50 MW), 702 reads `hydro_availability`, 703 holds under an always-true `active_when` and 704
/// under an always-false one.
pub(crate) fn write_scenario_constraint_case(dst: &Path) {
    copy_dir_recursive(&case_dir("deterministic/d02-single-hydro"), dst);
    let mut config: Value =
        serde_json::from_str(&fs::read_to_string(dst.join("config.json")).unwrap()).unwrap();
    config["simulation"] = json!({
        "enabled": true,
        "selection": { "method": "sampled", "num_scenarios": 2 }
    });
    write_file(dst, "config.json", &config.to_string());

    let slack = default_slack(dst);
    let constraints = json!({
        "constraints": [
            {
                "id": 701,
                "name": "constraint_701",
                "expression": "hydro_generation(0) >= 1000",
                "slack": slack
            },
            {
                "id": 702,
                "name": "constraint_702",
                "expression": "hydro_generation(0) <= hydro_availability(0)",
                "slack": slack
            },
            {
                "id": 703,
                "name": "constraint_703",
                "expression": "hydro_generation(0) <= 1000000",
                "active_when": "bus_load_value(0) <= 1000000000",
                "slack": slack
            },
            {
                "id": 704,
                "name": "constraint_704",
                "expression": "hydro_generation(0) <= 1000000",
                "active_when": "bus_load_value(0) <= 0",
                "slack": slack
            }
        ]
    });
    write_file(
        dst,
        "constraints/generic_constraints.json",
        &constraints.to_string(),
    );
    write_bounds_rows_per_cell(dst, &[701, 702, 703, 704]);
}

fn activation_of(outputs: &ConstraintOutputs, constraint_id: i32) -> Vec<&ActivationRow> {
    let rows: Vec<_> = outputs
        .activation
        .iter()
        .filter(|r| r.constraint_id == constraint_id)
        .collect();
    assert!(
        !rows.is_empty(),
        "no activation row for constraint {constraint_id}"
    );
    rows
}

#[test]
fn the_constraint_output_checks_hold_on_a_small_scenario_constraint_case() {
    let case = TempDir::new().unwrap();
    write_scenario_constraint_case(case.path());
    let out = TempDir::new().unwrap();
    run_case(case.path(), out.path());

    let outputs = read_constraint_outputs(out.path());
    assert_constraint_run_invariants(out.path(), &outputs);

    assert!(
        activation_of(&outputs, 703).iter().all(|r| r.active),
        "703 holds under an always-true `active_when` and must be active in every row"
    );
    assert!(
        activation_of(&outputs, 704).iter().all(|r| !r.active),
        "704 holds under an always-false `active_when` and must be inactive in every row"
    );

    let slack_701: Vec<_> = outputs
        .violations
        .iter()
        .filter(|r| r.constraint_id == 701)
        .collect();
    assert!(!slack_701.is_empty(), "violations hold no row for 701");
    assert!(
        slack_701.iter().all(|r| r.slack_value != 0.0),
        "701 asks for 1000 MW from a 50 MW plant and must be violated in every row"
    );

    let reads_702: Vec<_> = outputs
        .echo
        .iter()
        .filter(|r| {
            r.constraint_id == 702 && r.variable_kind.as_deref() == Some("hydro_availability")
        })
        .collect();
    assert!(
        !reads_702.is_empty(),
        "the echo holds no hydro_availability read of 702"
    );
    assert!(
        reads_702.iter().all(|r| r.value == Some(50.0)),
        "702 must read the 50 MW plant availability: {reads_702:?}"
    );
}

fn assert_echo_follows_the_input_file(case: &Path, echo: &[EchoRow]) {
    let input: Value = serde_json::from_str(
        &fs::read_to_string(case.join("constraints/generic_constraints.json")).unwrap(),
    )
    .unwrap();
    for constraint in input["constraints"].as_array().into_iter().flatten() {
        let constraint_id = constraint["id"].as_i64().unwrap();
        let has_active_when = constraint
            .get("active_when")
            .is_some_and(|condition| !condition.is_null());
        let rows: Vec<_> = echo
            .iter()
            .filter(|r| i64::from(r.constraint_id) == constraint_id)
            .collect();
        assert!(
            !has_active_when || !rows.is_empty(),
            "`active_when` of constraint {constraint_id} has no echo row"
        );
        assert!(
            rows.iter().all(|r| r.has_active_when == has_active_when),
            "constraint {constraint_id} must carry has_active_when = {has_active_when} in the echo"
        );
    }
    for constraint_id in [402, 403] {
        assert!(
            echo.iter().any(|r| r.constraint_id == constraint_id),
            "constraint {constraint_id} must be in the echo"
        );
    }
    assert!(
        echo.iter().any(|r| r.constraint_id == 402
            && r.variable_kind.as_deref() == Some("hydro_availability")
            && r.value.is_some_and(|v| v.is_finite() && v >= 0.0)),
        "constraint 402 must carry a hydro_availability read with a finite non-negative value"
    );
}

fn assert_activation_is_the_same_in_every_scenario(activation: &[ActivationRow]) {
    type Outcome = (bool, Option<u64>, Option<u64>);
    let mut first: BTreeMap<(i32, Option<i32>, i32), Outcome> = BTreeMap::new();
    for row in activation {
        let outcome = (
            row.active,
            row.bound_lower.map(f64::to_bits),
            row.bound_upper.map(f64::to_bits),
        );
        let cell = (row.stage_id, row.block_id, row.constraint_id);
        assert_eq!(
            *first.entry(cell).or_insert(outcome),
            outcome,
            "(stage, block, constraint) {cell:?} differs in scenario {}",
            row.scenario_id
        );
    }
}

#[test]
#[ignore = "needs a converted case; set NOVOMODELO_CONVERTED_CASE to its directory"]
fn the_converted_case_trains_simulates_and_writes_its_constraint_outputs() {
    let case = PathBuf::from(
        std::env::var("NOVOMODELO_CONVERTED_CASE")
            .expect("NOVOMODELO_CONVERTED_CASE: set it to the directory of a converted case"),
    );
    let out = TempDir::new().unwrap();
    run_case(&case, out.path());

    let outputs = read_constraint_outputs(out.path());
    assert_constraint_run_invariants(out.path(), &outputs);
    assert_echo_follows_the_input_file(&case, &outputs.echo);
    assert_activation_is_the_same_in_every_scenario(&outputs.activation);

    let scenarios: BTreeSet<i32> = outputs.activation.iter().map(|r| r.scenario_id).collect();
    let activation_cells: usize = activation_cells(&outputs.echo).values().map(Vec::len).sum();
    assert!(
        scenarios.len() >= 2,
        "the stochastic part needs at least 2 simulated scenarios, got {}",
        scenarios.len()
    );
    assert_eq!(
        outputs.activation.len(),
        scenarios.len() * activation_cells,
        "activation must hold one row per scenario and (stage, block, constraint) of a \
         constraint with `active_when` or a per-solve side"
    );

    let constraints: BTreeSet<i32> = outputs.echo.iter().map(|r| r.constraint_id).collect();
    let gated: BTreeSet<i32> = outputs
        .echo
        .iter()
        .filter(|r| r.has_active_when)
        .map(|r| r.constraint_id)
        .collect();
    let (achieved, termination_reason) = training_convergence(out.path());
    let active = outputs.activation.iter().filter(|r| r.active).count();
    eprintln!(
        "converted case: scenarios={} constraints={} constraints_with_active_when={} \
         activation_rows_active={} activation_rows_inactive={} \
         convergence.achieved={achieved} termination_reason={termination_reason}",
        scenarios.len(),
        constraints.len(),
        gated.len(),
        active,
        outputs.activation.len() - active,
    );

    scenario_constraint_case_golden::assert_matches_baseline(&outputs);
}
