//! The synthetic cases of the constraint-row golden, written as generic constraints, with the load
//! probes each is read under, and the facts of their simulation outputs.

#![allow(clippy::panic)]

use std::collections::BTreeSet;
use std::ops::RangeBounds;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use novomodelo_io::GenericConstraintEchoRow;
use novomodelo_io::config::SimulationSelection;
use novomodelo_sddp::{SimulationScenarioResult, build_generic_constraint_echo_rows};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::TempDir;

use super::parquet_fixtures::{
    GenericBoundsRow, write_generic_constraint_bounds, write_seasonal_stats,
};
use super::{
    copy_case_with_stage_field, copy_toy_case, fresh_system_and_setup_with, run_simulation,
};

const GENERIC_FILE: &str = "constraints/generic_constraints.json";
const GENERIC_BOUNDS_FILE: &str = "constraints/generic_constraint_bounds.parquet";
const PARAMETERS_FILE: &str = "constraints/generic_parameters.json";
const LOAD_STATS_FILE: &str = "scenarios/load_seasonal_stats.parquet";

pub struct OracleCase {
    pub name: &'static str,
    pub dir: TempDir,
    /// `probes[p][stage_idx]`: member-major and block-minor over the load-noise buses.
    pub probes: Vec<Vec<Vec<f64>>>,
}

pub fn oracle_cases() -> Vec<OracleCase> {
    let builders = [
        (
            "formula_with_bounds",
            formula_with_bounds as fn() -> TempDir,
        ),
        ("inequalities_and_horizon", inequalities_and_horizon),
        ("range_overrides", range_overrides),
        ("gated_and_selected", gated_and_selected),
        ("unit_group", unit_group),
        ("capacity_as_data", capacity_as_data),
    ];
    builders
        .into_iter()
        .map(|(name, build)| {
            naming_case(name, || {
                let dir = build();
                let probes = probes(dir.path());
                OracleCase { name, dir, probes }
            })
        })
        .collect()
}

/// The simulation outputs of a case, every float as its `{:?}` rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputFacts {
    pub echo: Vec<EchoFacts>,
    pub stages: Vec<ScenarioStageFacts>,
}

/// One row of the generic constraint echo, column for column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
// Rationale (struct_excessive_bools): each flag is a Boolean column of the echo.
#[allow(clippy::struct_excessive_bools)]
pub struct EchoFacts {
    pub stage_id: i32,
    pub block_id: Option<i32>,
    pub constraint_id: i32,
    pub constraint_name: String,
    pub term_index: Option<i32>,
    pub variable_kind: Option<String>,
    pub variable: Option<String>,
    pub coefficient: Option<String>,
    pub bound_lower: Option<String>,
    pub bound_upper: Option<String>,
    pub derived_shape: String,
    pub slack_enabled: bool,
    pub slack_penalty: Option<String>,
    pub slack_penalty_basis: Option<String>,
    pub value: Option<String>,
    pub lower_per_solve: bool,
    pub upper_per_solve: bool,
    pub has_active_when: bool,
}

/// `(constraint_id, block_id, active, bound_lower, bound_upper)`.
pub type ActivationFacts = (i32, Option<u32>, bool, Option<String>, Option<String>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioStageFacts {
    pub scenario_id: u32,
    pub stage_id: u32,
    /// `(constraint_id, block_id, slack_value, slack_cost)`.
    pub violations: Vec<(i32, Option<u32>, String, String)>,
    pub activation: Vec<ActivationFacts>,
    /// `(block_id, generic_violation_cost)`.
    pub violation_cost: Vec<(Option<u32>, String)>,
    /// `(bus_id, block_id, spot_price)`.
    pub spot_price: Vec<(i32, Option<u32>, String)>,
}

const SIMULATED_SCENARIOS: u32 = 8;

/// Trains `case` with its deck's training config, simulates [`SIMULATED_SCENARIOS`] sampled
/// scenarios on one thread and records the generic outputs.
pub fn output_facts(case: &OracleCase) -> OutputFacts {
    naming_case(case.name, || {
        let (system, mut setup) = fresh_system_and_setup_with(case.dir.path(), |config| {
            config.simulation.enabled = true;
            config.simulation.selection = Some(SimulationSelection::Sampled {
                num_scenarios: SIMULATED_SCENARIOS,
            });
        });
        let echo = build_generic_constraint_echo_rows(&setup, &system)
            .into_iter()
            .map(echo_facts)
            .collect();
        let mut stages: Vec<_> = run_simulation(&mut setup, 1)
            .iter()
            .flat_map(scenario_stage_facts)
            .collect();
        stages.sort_by_key(|stage| (stage.scenario_id, stage.stage_id));
        OutputFacts { echo, stages }
    })
}

fn text(value: f64) -> String {
    format!("{value:?}")
}

fn echo_facts(row: GenericConstraintEchoRow) -> EchoFacts {
    EchoFacts {
        stage_id: row.stage_id,
        block_id: row.block_id,
        constraint_id: row.constraint_id,
        constraint_name: row.constraint_name,
        term_index: row.term_index,
        variable_kind: row.variable_kind,
        variable: row.variable,
        coefficient: row.coefficient.map(text),
        bound_lower: row.bound_lower.map(text),
        bound_upper: row.bound_upper.map(text),
        derived_shape: row.derived_shape,
        slack_enabled: row.slack_enabled,
        slack_penalty: row.slack_penalty.map(text),
        slack_penalty_basis: row.slack_penalty_basis,
        value: row.value.map(text),
        lower_per_solve: row.lower_per_solve,
        upper_per_solve: row.upper_per_solve,
        has_active_when: row.has_active_when,
    }
}

fn scenario_stage_facts(
    scenario: &SimulationScenarioResult,
) -> impl Iterator<Item = ScenarioStageFacts> + '_ {
    scenario.stages.iter().map(|stage| {
        let mut violations: Vec<_> = stage
            .generic_violations
            .iter()
            .map(|v| {
                (
                    v.constraint_id,
                    v.block_id,
                    text(v.slack_value),
                    text(v.slack_cost),
                )
            })
            .collect();
        violations.sort_by_key(|&(id, block, ..)| (id, block));
        let mut activation: Vec<_> = stage
            .generic_activation
            .iter()
            .map(|a| {
                (
                    a.constraint_id,
                    a.block_id,
                    a.active,
                    a.bound_lower.map(text),
                    a.bound_upper.map(text),
                )
            })
            .collect();
        activation.sort_by_key(|&(id, block, ..)| (id, block));
        let mut violation_cost: Vec<_> = stage
            .costs
            .iter()
            .map(|c| (c.block_id, text(c.generic_violation_cost)))
            .collect();
        violation_cost.sort_by_key(|&(block, _)| block);
        let mut spot_price: Vec<_> = stage
            .buses
            .iter()
            .map(|b| (b.bus_id, b.block_id, text(b.spot_price)))
            .collect();
        spot_price.sort_by_key(|&(bus, block, _)| (bus, block));
        ScenarioStageFacts {
            scenario_id: scenario.scenario_id,
            stage_id: stage.stage_id,
            violations,
            activation,
            violation_cost,
            spot_price,
        }
    })
}

pub fn naming_case<T>(name: &str, run: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("a panic without a message");
        panic!("case {name}: {message}")
    })
}

fn formula_with_bounds() -> TempDir {
    let case = copy_deck("d04-transmission");
    let slack = default_slack(case.path());
    write_generic(
        &case,
        &json!({
            "constraints": [
                { "id": 401, "name": "constraint_401", "expression": "@FLOW", "slack": slack },
                { "id": 411, "name": "constraint_411", "expression": "@FLOW + bus_load_value(1)",
                  "slack": slack },
            ],
            "expressions": [{
                "name": "FLOW",
                "expression":
                    "line_flow(source_bus=0, target_bus=1) - line_flow(source_bus=1, target_bus=0)",
            }],
        }),
        &[
            cells(case.path(), 401, 0..=0, Some(-50.0), Some(80.0)),
            cells(case.path(), 401, 1..=1, None, Some(60.0)),
            cells(case.path(), 411, 0..=0, Some(-500.0), Some(500.0)),
            cells(case.path(), 411, 1..=1, Some(-500.0), Some(500.0)),
        ]
        .concat(),
    );
    case
}

/// 503 stays below the 50 MW plant cap: the cases are also solved, and a hard row must be feasible.
fn inequalities_and_horizon() -> TempDir {
    let case = copy_deck("d02-single-hydro");
    write_generic(
        &case,
        &json!({
            "constraints": [
                { "id": 501, "name": "constraint_501", "expression": "hydro_generation(0) >= 1000",
                  "slack": soft(2000.0) },
                { "id": 502, "name": "constraint_502",
                  "expression": "hydro_generation(0) + thermal_generation(0) <= @cap",
                  "slack": default_slack(case.path()) },
                { "id": 503, "name": "constraint_503", "expression": "hydro_generation(0) == 25",
                  "slack": hard() },
            ],
        }),
        &[
            cells(case.path(), 501, 0..=0, None, None),
            cells(case.path(), 502, .., None, None),
            cells(case.path(), 503, .., None, None),
        ]
        .concat(),
    );
    let block_values: Vec<Value> = stage_blocks(case.path())
        .into_iter()
        .flat_map(|(stage_id, n_blocks)| {
            (0..n_blocks).map(move |block| json!([stage_id, block, cap(stage_id)]))
        })
        .collect();
    write_json(
        &case,
        PARAMETERS_FILE,
        &json!({
            "scalar_parameters": [
                { "id": 0, "name": "cap", "kind": "per_stage_block", "block_values": block_values },
            ],
        }),
    );
    case
}

fn cap(stage_id: i32) -> f64 {
    120.0 + 10.0 * f64::from(stage_id)
}

/// 601 and 602 split by hand into one constraint per stretch of constant relation and treatment.
fn range_overrides() -> TempDir {
    let case = copy_deck("d14-block-factors");
    let default = default_slack(case.path());
    let segments = [
        (
            6011,
            "constraint_601_1",
            "thermal_generation(0) + thermal_generation(1) <= 400",
            soft(1500.0),
            vec![(0, 0), (0, 1)],
        ),
        (
            6012,
            "constraint_601_2",
            "thermal_generation(0) <= 200",
            hard(),
            vec![(1, 0)],
        ),
        (
            6013,
            "constraint_601_3",
            "thermal_generation(0) <= 300",
            hard(),
            vec![(1, 1)],
        ),
        (
            6021,
            "constraint_602_1",
            "thermal_generation(1) <= 250",
            default.clone(),
            vec![(0, 0), (0, 1)],
        ),
        (
            6022,
            "constraint_602_2",
            "thermal_generation(1) <= 350",
            default,
            vec![(1, 0), (1, 1)],
        ),
    ];
    let constraints: Vec<Value> = segments
        .iter()
        .map(|(id, name, expression, slack, _)| {
            json!({ "id": id, "name": name, "expression": expression, "slack": slack })
        })
        .collect();
    let bounds: Vec<GenericBoundsRow> = segments
        .iter()
        .flat_map(|(id, .., cells)| {
            cells
                .iter()
                .map(|&(stage_id, block)| (*id, stage_id, Some(block_id(block)), None, None))
        })
        .collect();
    write_generic(&case, &json!({ "constraints": constraints }), &bounds);
    case
}

fn gated_and_selected() -> TempDir {
    let (case, mean) = drawn_load_case("d02-single-hydro");
    let slack = default_slack(case.path());
    write_generic(
        &case,
        &json!({
            "constraints": [
                { "id": 701, "name": "constraint_701",
                  "expression": "hydro_generation(0) <= 1000000",
                  "active_when": format!("bus_load_value(0) <= {mean}"), "slack": slack },
                { "id": 702, "name": "constraint_702",
                  "expression":
                      format!("hydro_generation(0) <= if(bus_load_value(0) > {mean}, 800, 600)"),
                  "slack": slack },
                { "id": 703, "name": "constraint_703", "expression": "hydro_generation(0) >= @MARGIN",
                  "slack": slack },
            ],
            "expressions": [{ "name": "MARGIN", "expression": "0.05 * bus_load_value(0)" }],
        }),
        &[701, 702, 703]
            .map(|id| cells(case.path(), id, .., None, None))
            .concat(),
    );
    case
}

/// Group 1 is the only generator on bus 1.
fn unit_group() -> TempDir {
    let case = copy_deck("d51-split-plant-two-bus");
    write_generic(
        &case,
        &json!({
            "constraints": [
                { "id": 901, "name": "constraint_901",
                  "expression": "hydro_generation(0, unit_group=1) <= 50",
                  "slack": default_slack(case.path()) },
            ],
        }),
        &cells(case.path(), 901, .., None, None),
    );
    case
}

fn capacity_as_data() -> TempDir {
    let case = copy_deck("d02-single-hydro");
    write_generic(
        &case,
        &json!({
            "constraints": [
                { "id": 951, "name": "constraint_951",
                  "expression": "hydro_generation(0) <= hydro_availability(0)",
                  "slack": default_slack(case.path()) },
            ],
        }),
        &cells(case.path(), 951, .., None, None),
    );
    case
}

fn soft(penalty: f64) -> Value {
    json!({ "enabled": true, "penalty": penalty, "penalty_basis": "per_hour" })
}

fn hard() -> Value {
    json!({ "enabled": false })
}

/// The slack of a constraint without a violation treatment: 10 times the largest deficit cost of
/// the case's buses.
pub fn default_slack(case: &Path) -> Value {
    let text = std::fs::read_to_string(case.join("system/buses.json")).expect("read buses.json");
    let document: Value = serde_json::from_str(&text).expect("parse buses.json");
    let max_deficit_cost = document["buses"]
        .as_array()
        .expect("a buses array")
        .iter()
        .flat_map(|bus| {
            bus["deficit_segments"]
                .as_array()
                .expect("deficit segments")
        })
        .map(|segment| segment["cost"].as_f64().expect("a deficit cost"))
        .fold(0.0, f64::max);
    soft(10.0 * max_deficit_cost)
}

/// `document` as the case's generic constraints, bounded by `bounds`.
pub fn write_generic(case: &TempDir, document: &Value, bounds: &[GenericBoundsRow]) {
    write_json(case, GENERIC_FILE, document);
    write_generic_constraint_bounds(&case.path().join(GENERIC_BOUNDS_FILE), bounds);
}

/// A bounds row of constraint `id` at `(lower, upper)` per block of each study stage of `case` in
/// `stages`.
pub fn cells(
    case: &Path,
    id: i32,
    stages: impl RangeBounds<i32>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Vec<GenericBoundsRow> {
    stage_blocks(case)
        .into_iter()
        .filter(|(stage_id, _)| stages.contains(stage_id))
        .flat_map(|(stage_id, n_blocks)| {
            (0..n_blocks).map(move |block| (id, stage_id, Some(block_id(block)), lower, upper))
        })
        .collect()
}

fn block_id(block: usize) -> i32 {
    i32::try_from(block).expect("a block id fits i32")
}

fn deck(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(name)
}

fn copy_deck(name: &str) -> TempDir {
    let case = TempDir::new().expect("create the case directory");
    copy_toy_case(&deck(name), case.path());
    case
}

/// `name` with 4 openings per stage and every load drawn with a deviation of 20% of its mean, and
/// the mean of its first load-stats row.
fn drawn_load_case(name: &str) -> (TempDir, f64) {
    let case = copy_case_with_stage_field(&deck(name), "num_openings", &json!(4));
    let load_stats = case.path().join(LOAD_STATS_FILE);
    let rows: Vec<_> = novomodelo_io::parse_load_seasonal_stats(&load_stats)
        .expect("the load stats must parse")
        .into_iter()
        .map(|row| (row.bus_id.0, row.stage_id, row.mean_mw, 0.2 * row.mean_mw))
        .collect();
    write_seasonal_stats(&load_stats, "bus_id", "mean_mw", "std_mw", &rows);
    let mean = rows[0].2;
    (case, mean)
}

fn write_json(case: &TempDir, relative: &str, value: &Value) {
    let path = case.path().join(relative);
    std::fs::create_dir_all(path.parent().expect("a case file has a parent"))
        .expect("create the case file's directory");
    std::fs::write(&path, value.to_string()).expect("write a case file");
}

fn stage_blocks(case: &Path) -> Vec<(i32, usize)> {
    let text = std::fs::read_to_string(case.join("stages.json")).expect("read stages.json");
    let document: Value = serde_json::from_str(&text).expect("parse stages.json");
    let mut stages: Vec<(i32, usize)> = document["stages"]
        .as_array()
        .expect("a stages array")
        .iter()
        .map(|stage| {
            let id = stage["id"].as_i64().expect("an integer stage id");
            let blocks = stage["blocks"].as_array().expect("a blocks array").len();
            (i32::try_from(id).expect("an i32 stage id"), blocks)
        })
        .filter(|&(id, _)| id >= 0)
        .collect();
    stages.sort_unstable();
    stages
}

fn probes(case: &Path) -> Vec<Vec<Vec<f64>>> {
    let stats = novomodelo_io::parse_load_seasonal_stats(&case.join(LOAD_STATS_FILE))
        .expect("the load stats must parse");
    let stages = stage_blocks(case);
    let members: BTreeSet<i32> = stats
        .iter()
        .filter(|row| row.std_mw > 0.0)
        .map(|row| row.bus_id.0)
        .collect();
    if members.is_empty() {
        return vec![vec![Vec::new(); stages.len()]];
    }
    let mean = |bus_id: i32, stage_id: i32| {
        stats
            .iter()
            .find(|row| row.bus_id.0 == bus_id && row.stage_id == stage_id)
            .unwrap_or_else(|| panic!("bus {bus_id} has no load stats at stage {stage_id}"))
            .mean_mw
    };
    [0.5, 1.0, 1.5]
        .into_iter()
        .map(|factor| {
            stages
                .iter()
                .map(|&(stage_id, n_blocks)| {
                    members
                        .iter()
                        .flat_map(|&bus_id| {
                            std::iter::repeat_n(factor * mean(bus_id, stage_id), n_blocks)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}
