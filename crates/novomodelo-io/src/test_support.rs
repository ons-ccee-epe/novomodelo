//! Crate-internal test fixture builders shared across this crate's unit tests
//! and reachable from `tests/` integration binaries and downstream crates'
//! tests via the `test-support` feature. Compiles only under `cfg(test)` or
//! that feature; must never be enabled in a production build.
//!
//! Entity builders, `ParsedData` skeletons, `Config` variants, input-file
//! writers, the minimal-case corpus, output fixtures and stats-parser
//! templates. Single-use helpers stay in their own module's `mod tests` block
//! to keep the blast radius small.
//!
//! Builders returning the crate-private `ParsedData` (`base_parsed_data`,
//! `make_data`, `make_data_5b`, `make_data_estimation`) are additionally
//! gated `#[cfg(test)]`, since no `test-support`-feature-only caller outside
//! this crate's own `#[cfg(test)]` modules can name that type.

#![allow(
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::NaiveDate;
use cobre_core::{
    CorrelationGroup, CorrelationModel, EntityId, HorizonGraph, SeasonMap,
    entities::{
        Bus, DeficitSegment, Hydro, HydroGenerationModel, HydroPenalties, HydroUnitGroup, Thermal,
    },
    penalty::GlobalPenaltyDefaults,
    temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    },
};
#[cfg(test)]
use cobre_core::{entities::Line, initial_conditions::InitialConditions};

use crate::{
    InflowArCoefficientRow, InflowHistoryRow, LoadError,
    config::Config,
    extensions::{FphaHyperplaneRow, HydroGeometryRow},
    parse_config,
    stages::StagesData,
};
#[cfg(test)]
use crate::{InflowSeasonalStatsRow, validation::schema::ParsedData};
use parquet::arrow::ArrowWriter;
use std::fmt::Debug;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::{NamedTempFile, TempDir};

/// The `NaiveDate` for a calendar-valid `(year, month, day)` triple.
///
/// # Panics
///
/// Never for the literal triples the builders below pass.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: every call site below passes a calendar-valid literal triple, so
// `from_ymd_opt` cannot return `None`.
pub fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("caller passes a calendar-valid triple")
}

// ── Parquet fixture writers ───────────────────────────────────────────────────

/// A temporary parquet file holding `batch`.
///
/// # Panics
///
/// If the test host cannot create, reopen, or write to a temporary file.
#[must_use]
pub fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
    write_parquet_batches(std::slice::from_ref(batch))
}

/// A temporary parquet file holding `batches` as consecutive row groups.
///
/// # Panics
///
/// If `batches` is empty, or if the test host cannot create, reopen, or write
/// to a temporary file.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: a failure here means the test host cannot create a temporary
// file — an environment fault, not a fixture condition.
pub fn write_parquet_batches(batches: &[RecordBatch]) -> NamedTempFile {
    assert!(!batches.is_empty(), "must provide at least one batch");
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batches[0].schema(), None)
        .expect("ArrowWriter");
    for batch in batches {
        writer.write(batch).expect("write batch");
    }
    writer.close().expect("close writer");
    tmp
}

// ── Stats-parser fixtures ─────────────────────────────────────────────────────

/// A four-column seasonal-stats record batch: an id column, `stage_id`, a
/// mean column and a std column, with the id/mean/std column names supplied
/// by the caller so one builder serves the inflow, load and NCS parsers.
///
/// # Panics
///
/// If `ids`, `stage_ids`, `means` and `stds` do not share one length — a
/// defect in the calling test.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: the arrays are built from the caller's own slices and the
// schema from the caller's own names, so a `try_new` failure means the
// slices have mismatched lengths — a defect in the calling test.
pub fn make_stats_batch(
    id_column: &str,
    mean_column: &str,
    std_column: &str,
    ids: &[i32],
    stage_ids: &[i32],
    means: &[f64],
    stds: &[f64],
) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(id_column, DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new(mean_column, DataType::Float64, false),
        Field::new(std_column, DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(Int32Array::from(stage_ids.to_vec())),
            Arc::new(Float64Array::from(means.to_vec())),
            Arc::new(Float64Array::from(stds.to_vec())),
        ],
    )
    .expect("caller's slices share one length")
}

/// Assert that `err` is a [`LoadError::SchemaError`] whose `field` contains
/// `field_substr`, and — when given — whose `message` contains
/// `message_substr`.
fn assert_schema_error_shape(err: &LoadError, field_substr: &str, message_substr: Option<&str>) {
    assert!(
        matches!(err, LoadError::SchemaError { .. }),
        "expected SchemaError, got: {err:?}"
    );
    if let LoadError::SchemaError { field, message, .. } = err {
        assert!(
            field.contains(field_substr),
            "field should contain '{field_substr}', got: {field}"
        );
        if let Some(expected) = message_substr {
            assert!(
                message.contains(expected),
                "message should mention '{expected}', got: {message}"
            );
        }
    }
}

/// Assert that `parse` accepts a valid four-row batch built from
/// `id_column`/`mean_column`/`std_column` and sorts it by `(id, stage)`,
/// extracting `(id, stage_id, mean, std)` from each parsed row via
/// `extract`.
///
/// # Panics
///
/// If `parse` rejects the batch, or the parsed rows do not match the
/// expected sort order or the first row's mean/std.
#[allow(clippy::expect_used)]
// Rationale: `parse` is handed a batch this helper just built and wrote
// itself, so a rejection means the parser under test is broken, not a
// fixture condition the caller can supply differently.
pub fn assert_stats_happy_path<T>(
    parse: impl Fn(&Path) -> Result<Vec<T>, LoadError>,
    id_column: &str,
    mean_column: &str,
    std_column: &str,
    extract: impl Fn(&T) -> (i32, i32, f64, f64),
) {
    let batch = make_stats_batch(
        id_column,
        mean_column,
        std_column,
        &[3, 1, 3, 1],
        &[1, 0, 0, 1],
        &[0.5, 0.3, 0.45, 0.35],
        &[0.05, 0.03, 0.045, 0.035],
    );
    let tmp = write_parquet(&batch);
    let rows = parse(tmp.path()).expect("fixture batch built above must parse");

    assert_eq!(rows.len(), 4);
    let keys: Vec<(i32, i32)> = rows
        .iter()
        .map(|row| {
            let (id, stage, ..) = extract(row);
            (id, stage)
        })
        .collect();
    assert_eq!(keys, vec![(1, 0), (1, 1), (3, 0), (3, 1)]);

    let (_, _, mean, std) = extract(&rows[0]);
    assert!((mean - 0.3).abs() < 1e-10);
    assert!((std - 0.03).abs() < 1e-10);
}

/// Assert that `parse` rejects a batch missing `mean_column`: a
/// [`LoadError::SchemaError`] whose `field` names it and whose `message`
/// says the column is missing.
///
/// # Panics
///
/// If `parse` accepts the batch, or the rejection is not the expected
/// [`LoadError::SchemaError`] shape.
#[allow(clippy::expect_used)]
// Rationale: the batch this helper builds always omits `mean_column`, so a
// non-error result means the parser under test is broken, not a fixture
// condition the caller can supply differently.
pub fn assert_stats_missing_column<T: Debug>(
    parse: impl Fn(&Path) -> Result<Vec<T>, LoadError>,
    id_column: &str,
    mean_column: &str,
    std_column: &str,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(id_column, DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new(std_column, DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1_i32])),
            Arc::new(Int32Array::from(vec![0_i32])),
            Arc::new(Float64Array::from(vec![0.5])),
        ],
    )
    .expect("literal single-row arrays share one length");
    let tmp = write_parquet(&batch);
    let err = parse(tmp.path()).expect_err("batch omitting the mean column must be rejected");
    assert_schema_error_shape(&err, mean_column, Some("missing required column"));
}

/// Assert that `parse` rejects a single row whose std is negative: a
/// [`LoadError::SchemaError`] whose `field` names `std_column`.
///
/// # Panics
///
/// If `parse` accepts the batch, or the rejection is not the expected
/// [`LoadError::SchemaError`] shape.
#[allow(clippy::expect_used)]
// Rationale: the batch this helper builds always carries a negative std, so
// a non-error result means the parser under test is broken, not a fixture
// condition the caller can supply differently.
pub fn assert_stats_negative_std<T: Debug>(
    parse: impl Fn(&Path) -> Result<Vec<T>, LoadError>,
    id_column: &str,
    mean_column: &str,
    std_column: &str,
) {
    let batch = make_stats_batch(
        id_column,
        mean_column,
        std_column,
        &[1],
        &[0],
        &[0.5],
        &[-0.5],
    );
    let tmp = write_parquet(&batch);
    let err = parse(tmp.path()).expect_err("negative std must be rejected");
    assert_schema_error_shape(&err, std_column, None);
}

/// Assert that `parse` rejects a single row whose mean is NaN: a
/// [`LoadError::SchemaError`] whose `field` names `mean_column`.
///
/// # Panics
///
/// If `parse` accepts the batch, or the rejection is not the expected
/// [`LoadError::SchemaError`] shape.
#[allow(clippy::expect_used)]
// Rationale: the batch this helper builds always carries a NaN mean, so a
// non-error result means the parser under test is broken, not a fixture
// condition the caller can supply differently.
pub fn assert_stats_nan_mean<T: Debug>(
    parse: impl Fn(&Path) -> Result<Vec<T>, LoadError>,
    id_column: &str,
    mean_column: &str,
    std_column: &str,
) {
    let batch = make_stats_batch(
        id_column,
        mean_column,
        std_column,
        &[1],
        &[0],
        &[f64::NAN],
        &[0.5],
    );
    let tmp = write_parquet(&batch);
    let err = parse(tmp.path()).expect_err("NaN mean must be rejected");
    assert_schema_error_shape(&err, mean_column, None);
}

/// Assert that `parse` accepts an empty batch and returns an empty vector.
///
/// # Panics
///
/// If `parse` rejects the empty batch, or returns a non-empty vector.
#[allow(clippy::expect_used)]
// Rationale: the batch this helper builds carries zero rows over a valid
// schema, so a rejection means the parser under test is broken, not a
// fixture condition the caller can supply differently.
pub fn assert_stats_empty_file<T>(
    parse: impl Fn(&Path) -> Result<Vec<T>, LoadError>,
    id_column: &str,
    mean_column: &str,
    std_column: &str,
) {
    let batch = make_stats_batch(id_column, mean_column, std_column, &[], &[], &[], &[]);
    let tmp = write_parquet(&batch);
    let rows = parse(tmp.path()).expect("empty batch must parse");
    assert!(rows.is_empty());
}

// ── JSON fixture writer ───────────────────────────────────────────────────────

/// A temporary file holding `content`.
///
/// # Panics
///
/// If the test host cannot create or write to a temporary file.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: a failure here means the test host cannot write a temporary
// file — an environment fault, not a fixture condition.
pub fn write_json(content: &str) -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("tempfile");
    tmp.write_all(content.as_bytes()).expect("write JSON");
    tmp
}

// ── Minimal case corpus ───────────────────────────────────────────────────────

/// config.json: `training.stopping_rules[].type` = "iteration_limit",
/// field name is "limit" (not "max_iterations").
pub const VALID_CONFIG_JSON: &str = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;

/// penalties.json: top-level keys are "bus", "line", "hydro",
/// "non_controllable_source". Under "bus": "deficit_segments" (not
/// "segments") and "excess_cost". Under each segment: "cost" (not
/// "cost_per_mwh"). Under "line": "exchange_cost". Under "hydro": plain
/// field names without unit suffixes. Under "non_controllable_source":
/// "curtailment_cost". See src/penalties.rs for the raw serde types.
pub const VALID_PENALTIES_JSON: &str = r#"{
        "bus": {
            "deficit_segments": [
                { "depth_mw": 500.0, "cost": 1000.0 },
                { "depth_mw": null,  "cost": 5000.0 }
            ],
            "excess_cost": 100.0
        },
        "line": { "exchange_cost": 2.0 },
        "hydro": {
            "spillage_cost": 0.01,
            "turbined_cost": 0.05,
            "diversion_cost": 0.1,
            "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
        },
        "non_controllable_source": { "curtailment_cost": 0.005 }
    }"#;

/// stages.json: `target_id` in transitions must be an integer (not null).
/// For a single-stage finite horizon we omit transitions entirely.
/// Only mandatory per-stage fields: id, start_date, end_date, blocks,
/// num_openings. season_id, block_mode, state_variables, risk_measure,
/// sampling_method all have serde defaults and are optional.
pub const VALID_STAGES_JSON: &str = r#"{
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": []
        },
        "stages": [
            {
                "id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
                "num_openings": 50
            }
        ]
    }"#;

/// Minimal valid `initial_conditions.json`.
pub const VALID_INITIAL_CONDITIONS_JSON: &str = r#"{
        "storage": [],
        "filling_storage": []
    }"#;

/// buses.json: mandatory fields are "id" and "name" only.
/// "base_kv" does not exist in the actual Bus raw type.
pub const VALID_BUSES_JSON: &str =
    r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#;

/// Empty lines array.
pub const VALID_LINES_JSON: &str = r#"{ "lines": [] }"#;

/// Empty hydros array.
pub const VALID_HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

/// Empty thermals array.
pub const VALID_THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

/// Write `content` to `root.join(relative)`, creating all parent directories.
///
/// # Panics
///
/// If the filesystem refuses to create the parent directories or write the
/// file.
#[allow(clippy::expect_used)]
// Rationale: a failure here means the filesystem refused a create/write —
// an environment fault, not a fixture condition.
pub fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).expect("create parent directories");
    }
    fs::write(&full, content).expect("write fixture file");
}

/// Populate `dir` with the eight required JSON files for a minimal valid
/// case: 1 bus, 0 lines/hydros/thermals, 1 finite-horizon stage with no
/// transitions.
///
/// # Panics
///
/// Propagates any panic from [`write_file`].
pub fn make_minimal_case(dir: &TempDir) {
    let root = dir.path();
    write_file(root, "config.json", VALID_CONFIG_JSON);
    write_file(root, "penalties.json", VALID_PENALTIES_JSON);
    write_file(root, "stages.json", VALID_STAGES_JSON);
    write_file(
        root,
        "initial_conditions.json",
        VALID_INITIAL_CONDITIONS_JSON,
    );
    write_file(root, "system/buses.json", VALID_BUSES_JSON);
    write_file(root, "system/lines.json", VALID_LINES_JSON);
    write_file(root, "system/hydros.json", VALID_HYDROS_JSON);
    write_file(root, "system/thermals.json", VALID_THERMALS_JSON);
}

// ── Penalty helpers ───────────────────────────────────────────────────────────

/// Build `HydroPenalties` with every field set to `v` except
/// `inflow_nonnegativity_cost`.
#[must_use]
pub fn penalties_all(v: f64) -> HydroPenalties {
    HydroPenalties {
        spillage_cost: v,
        diversion_cost: v,
        turbined_cost: v,
        storage_violation_below_cost: v,
        filling_target_violation_cost: v,
        turbined_violation_below_cost: v,
        outflow_violation_below_cost: v,
        outflow_violation_above_cost: v,
        generation_violation_below_cost: v,
        evaporation_violation_cost: v,
        water_withdrawal_violation_cost: v,
        water_withdrawal_violation_pos_cost: v,
        water_withdrawal_violation_neg_cost: v,
        evaporation_violation_pos_cost: v,
        evaporation_violation_neg_cost: v,
        inflow_nonnegativity_cost: 1000.0,
    }
}

/// A `GlobalPenaltyDefaults` with a two-segment deficit curve and a
/// non-uniform hydro penalty set.
#[must_use]
pub fn make_global() -> GlobalPenaltyDefaults {
    GlobalPenaltyDefaults {
        bus_deficit_segments: vec![
            DeficitSegment {
                depth_mw: Some(500.0),
                cost_per_mwh: 1000.0,
            },
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            },
        ],
        bus_excess_cost: 100.0,
        line_exchange_cost: 2.0,
        hydro: HydroPenalties {
            spillage_cost: 0.01,
            turbined_cost: 0.05,
            diversion_cost: 0.1,
            storage_violation_below_cost: 10_000.0,
            filling_target_violation_cost: 50_000.0,
            turbined_violation_below_cost: 500.0,
            outflow_violation_below_cost: 500.0,
            outflow_violation_above_cost: 500.0,
            generation_violation_below_cost: 1_000.0,
            evaporation_violation_cost: 5_000.0,
            water_withdrawal_violation_cost: 1_000.0,
            water_withdrawal_violation_pos_cost: 1_000.0,
            water_withdrawal_violation_neg_cost: 1_000.0,
            evaporation_violation_pos_cost: 5_000.0,
            evaporation_violation_neg_cost: 5_000.0,
            inflow_nonnegativity_cost: 1000.0,
        },
        ncs_curtailment_cost: 0.005,
    }
}

// ── Entity builders ───────────────────────────────────────────────────────────

/// Build a minimal valid `Hydro` using default sensible values, with an empty
/// `unit_groups` — callers that need groups declare them directly, or route
/// through `make_data` / `make_data_5b` / `make_data_estimation`, which
/// sort at the same boundary `convert_hydros` and `SystemBuilder::build` do
/// (after the hydros' own field values are final).
#[must_use]
pub fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
    Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(id),
        name: format!("Hydro {id}"),
        operational_start_date: date(2024, 1, 1),
        downstream_id: downstream_id.map(EntityId::from),
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 1000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1000.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: penalties_all(1.0),
    }
}

/// Build a `HydroUnitGroup` with the given id, bus, and four bounds.
#[must_use]
pub fn make_unit_group(
    id: i32,
    bus_id: i32,
    min_generation_mw: f64,
    max_generation_mw: f64,
    min_turbined_m3s: f64,
    max_turbined_m3s: f64,
) -> HydroUnitGroup {
    HydroUnitGroup {
        id: EntityId::from(id),
        name: format!("Group {id}"),
        bus_id: EntityId::from(bus_id),
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }
}

/// Build a minimal valid `Thermal`.
#[must_use]
pub fn make_thermal(id: i32, min_mw: f64, max_mw: f64) -> Thermal {
    Thermal {
        id: EntityId::from(id),
        name: format!("Thermal {id}"),
        operational_start_date: date(2024, 1, 1),
        bus_id: EntityId::from(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 100.0,
        min_generation_mw: min_mw,
        max_generation_mw: max_mw,
        anticipated_config: None,
    }
}

/// `n` equal-duration blocks spanning a 720-hour stage, shared by
/// [`make_stage`] (`n = 1`) and [`make_stage_with_blocks`] so the two
/// builders cannot drift apart.
fn stage_blocks(n: usize) -> Vec<Block> {
    (0..n)
        .map(|index| Block {
            index,
            name: format!("B{index}"),
            duration_hours: 720.0 / n as f64,
        })
        .collect()
}

/// Build one study stage with the given `id`, carrying a single block
/// spanning the full stage.
#[must_use]
pub fn make_stage(id: i32) -> Stage {
    Stage {
        id,
        index: 0,
        start_date: date(2024, 1, 1),
        end_date: date(2024, 2, 1),
        season_id: None,
        blocks: stage_blocks(1),
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// Build a stage with `id` and `n` equal-duration blocks; every other field
/// comes from [`make_stage`].
#[must_use]
pub fn make_stage_with_blocks(id: i32, n: usize) -> Stage {
    let mut stage = make_stage(id);
    stage.blocks = stage_blocks(n);
    stage
}

/// `HorizonGraph` shared by [`make_stages`] and [`make_stages_with_seasons`]:
/// a finite-horizon graph with no discount-rate overrides or transitions.
fn make_horizon_graph(season_map: Option<SeasonMap>) -> HorizonGraph {
    HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.06,
        transitions: vec![],
        nodes: Vec::new(),
        season_map,
    }
}

/// Build a minimal valid `StagesData` with the given stage IDs.
#[must_use]
pub fn make_stages(ids: Vec<i32>) -> StagesData {
    StagesData {
        openings_declared: std::collections::HashSet::new(),
        stages: ids.into_iter().map(make_stage).collect(),
        policy_graph: make_horizon_graph(None),
    }
}

// ── Layer 5a data builder (hydro + thermal) ───────────────────────────────────

/// `ParsedData` skeleton with the caller-supplied stages and buses; every
/// other field empty or `None`.
#[cfg(test)]
pub(crate) fn base_parsed_data(stages: StagesData, buses: Vec<Bus>) -> ParsedData {
    ParsedData {
        config: minimal_config(),
        stages,
        initial_conditions: InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        },
        post_study_stages: None,
        buses,
        thermals: vec![],
        hydros: vec![],
        lines: vec![],
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry: vec![],
        production_models: vec![],
        plane_reduction: None,
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes: vec![],
        scalar_parameters: vec![],
        inflow_history: vec![],
        inflow_seasonal_stats: vec![],
        inflow_ar_coefficients: vec![],
        inflow_annual_components: vec![],
        external_scenarios: vec![],
        external_load_scenarios: vec![],
        external_ncs_scenarios: vec![],
        load_seasonal_stats: vec![],
        load_factors: vec![],
        correlation: None,
        non_controllable_factors: vec![],
        ncs_models: vec![],
        thermal_bounds: vec![],
        hydro_bounds: vec![],
        line_bounds: vec![],
        pumping_bounds: vec![],
        contract_bounds: vec![],
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
        hydro_unit_group_bounds: vec![],
    }
}

/// The one-`BUS_1` vector `base_parsed_data`'s in-module callers pass when
/// they have no bus fixture of their own to supply.
#[cfg(test)]
fn base_bus() -> Vec<Bus> {
    vec![Bus {
        id: EntityId::from(1),
        name: "BUS_1".to_string(),
        operational_start_date: date(2024, 1, 1),
        deficit_segments: vec![],
        excess_cost: 100.0,
    }]
}

/// Sort every hydro's `unit_groups`, at the boundary each `make_data*` builder
/// below sorts at (see `make_data`'s doc for why not inside `make_hydro`).
#[cfg(test)]
fn sort_hydros(hydros: &mut [Hydro]) {
    for hydro in hydros {
        hydro.sort_unit_groups();
    }
}

/// Build a minimal `ParsedData` with the provided hydros, thermals, stages,
/// geometry, and FPHA rows.  All other fields are empty/minimal.
///
/// Sorts each hydro's `unit_groups` here, at the boundary — mirroring where
/// `convert_hydros` and `SystemBuilder::build` sort in production, after the
/// hydros' own field values are final. Sorting inside `make_hydro` instead
/// would snapshot a stale group order for any caller that mutates
/// `unit_groups` afterward.
#[cfg(test)]
pub(crate) fn make_data(
    mut hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    lines: Vec<Line>,
    stages: StagesData,
    hydro_geometry: Vec<HydroGeometryRow>,
    fpha_hyperplanes: Vec<FphaHyperplaneRow>,
) -> ParsedData {
    sort_hydros(&mut hydros);
    ParsedData {
        thermals,
        hydros,
        lines,
        hydro_geometry,
        fpha_hyperplanes,
        ..base_parsed_data(stages, base_bus())
    }
}

// ── Layer 5b data builders (stages + penalties + scenarios) ──────────────────

/// Build a minimal valid `ParsedData` for Layer 5b tests.
/// All hydro penalties satisfy the ordering hierarchy by default.
///
/// Sorts `hydros` here, at the boundary — see `make_data`'s doc for why
/// this must not happen inside `make_hydro`.
#[cfg(test)]
pub(crate) fn make_data_5b(
    mut hydros: Vec<Hydro>,
    stages: StagesData,
    buses: Vec<Bus>,
    inflow_stats: Vec<InflowSeasonalStatsRow>,
    inflow_ar: Vec<InflowArCoefficientRow>,
    correlation: Option<CorrelationModel>,
) -> ParsedData {
    sort_hydros(&mut hydros);
    ParsedData {
        buses,
        hydros,
        inflow_seasonal_stats: inflow_stats,
        inflow_ar_coefficients: inflow_ar,
        correlation,
        ..base_parsed_data(stages, base_bus())
    }
}

/// Build a hydro with penalties satisfying the ordering hierarchy.
#[must_use]
pub fn make_hydro_ordered_penalties(id: i32) -> Hydro {
    let mut h = make_hydro(id, None);
    h.penalties = HydroPenalties {
        filling_target_violation_cost: 1000.0,
        storage_violation_below_cost: 500.0,
        turbined_violation_below_cost: 50.0,
        outflow_violation_below_cost: 50.0,
        outflow_violation_above_cost: 50.0,
        generation_violation_below_cost: 50.0,
        evaporation_violation_cost: 50.0,
        water_withdrawal_violation_cost: 50.0,
        water_withdrawal_violation_pos_cost: 50.0,
        water_withdrawal_violation_neg_cost: 50.0,
        evaporation_violation_pos_cost: 50.0,
        evaporation_violation_neg_cost: 50.0,
        spillage_cost: 1.0,
        diversion_cost: 1.0,
        turbined_cost: 2.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    h
}

/// Build a minimal valid `StagesData` for Layer 5b tests, delegating to
/// [`make_stages`].
#[must_use]
pub fn make_stages_5b(ids: Vec<i32>) -> StagesData {
    make_stages(ids)
}

/// Build a bus with a single deficit segment at the given cost.
#[must_use]
pub fn make_bus_with_deficit(id: i32, cost_per_mwh: f64) -> Bus {
    Bus {
        id: EntityId::from(id),
        name: format!("Bus {id}"),
        operational_start_date: date(2024, 1, 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh,
        }],
        excess_cost: 100.0,
    }
}

// ── Geometry and FPHA row builders ────────────────────────────────────────────

/// Build a minimal `FphaHyperplaneRow` with the given parameters.
#[must_use]
pub fn make_fpha_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
    FphaHyperplaneRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        plane_id,
        gamma_0: 100.0,
        gamma_v: 0.5, // valid: > 0
        gamma_q: 0.8,
        gamma_s: -0.02, // valid: <= 0
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    }
}

/// Build a minimal `HydroGeometryRow`.
#[must_use]
pub fn make_geom_row(
    hydro_id: i32,
    volume_hm3: f64,
    height_m: f64,
    area_km2: f64,
) -> HydroGeometryRow {
    HydroGeometryRow {
        hydro_id: EntityId::from(hydro_id),
        volume_hm3,
        height_m,
        area_km2,
    }
}

// ── Correlation helpers ───────────────────────────────────────────────────────

/// Build a valid 2x2 symmetric correlation group.
#[must_use]
pub fn make_corr_group(name: &str, matrix: Vec<Vec<f64>>) -> CorrelationGroup {
    use cobre_core::scenario::CorrelationEntity;
    CorrelationGroup {
        name: name.to_string(),
        entities: vec![
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(1),
            },
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(2),
            },
        ],
        matrix,
    }
}

/// Build a `CorrelationModel` with a single "default" profile containing the
/// given group.
#[must_use]
pub fn make_correlation(group: CorrelationGroup) -> CorrelationModel {
    use cobre_core::scenario::CorrelationProfile;
    use std::collections::BTreeMap;
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![group],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

// ── Config helpers ────────────────────────────────────────────────────────────

/// Parse `json` into a `Config` via a scratch temp file — the shared
/// write-then-parse path every fixture builder below uses.
#[allow(clippy::expect_used)]
// Rationale: the JSON literals are the fixtures' own, parsed by the production
// `parse_config` the crate ships, so a failure here is a fixture defect, not a
// runtime condition.
pub(crate) fn config_from_json(json: &str) -> Config {
    let tmp = tempfile::NamedTempFile::new().expect("scratch temp file creation cannot fail");
    std::fs::write(tmp.path(), json).expect("write to a fresh scratch temp file cannot fail");
    parse_config(tmp.path()).expect("fixture JSON literal is valid Config input")
}

/// Minimal `Config` required to fill `ParsedData`.
#[must_use]
pub fn minimal_config() -> Config {
    let json = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;
    config_from_json(json)
}

/// Build a `Config` with `training.scenario_source.inflow.scheme = "external"`.
#[must_use]
pub fn config_with_training_external_inflow() -> Config {
    let json = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "scenario_source": {
                "seed": 42,
                "inflow": { "scheme": "external" }
            }
        }
    }"#;
    config_from_json(json)
}

/// Build a `Config` with `training.selection = enumerated` and
/// `training.scenario_source.inflow.scheme = "external"` — the enumerated
/// external-openings case rule 36/37 governs.
#[must_use]
pub fn config_enumerated_external_inflow() -> Config {
    let json = r#"{
        "training": {
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "selection": { "method": "enumerated" },
            "scenario_source": {
                "seed": 42,
                "inflow": { "scheme": "external" }
            }
        }
    }"#;
    config_from_json(json)
}

/// Build a `Config` with `training.selection = enumerated` and every class at
/// its default (in-sample) scheme — an enumerated study carrying no external
/// class, where a node `scenario_id` is meaningless.
#[must_use]
pub fn config_enumerated() -> Config {
    let json = r#"{
        "training": {
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "selection": { "method": "enumerated" }
        }
    }"#;
    config_from_json(json)
}

/// Build a `Config` whose training scenario source sets the `external` scheme for
/// each requested class, leaving the rest at their default (in-sample) scheme.
#[must_use]
pub fn config_with_training_external(inflow: bool, load: bool, ncs: bool) -> Config {
    let mut classes: Vec<&str> = Vec::new();
    if inflow {
        classes.push(r#""inflow": { "scheme": "external" }"#);
    }
    if load {
        classes.push(r#""load": { "scheme": "external" }"#);
    }
    if ncs {
        classes.push(r#""ncs": { "scheme": "external" }"#);
    }
    let json = format!(
        r#"{{
            "training": {{
                "selection": {{"method": "sampled", "forward_passes": 10}},
                "stopping_rules": [{{ "type": "iteration_limit", "limit": 100 }}],
                "scenario_source": {{ "seed": 42, {} }}
            }}
        }}"#,
        classes.join(", ")
    );
    config_from_json(&json)
}

/// Build a sampled `Config` declaring `training.scenario_source.openings =
/// {source: file}` — the user-supplied opening-tree file arm.
#[must_use]
pub fn config_sampled_file_openings() -> Config {
    let json = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "scenario_source": {
                "openings": { "source": "file" }
            }
        }
    }"#;
    config_from_json(json)
}

/// Build an enumerated `Config` declaring `training.scenario_source.openings =
/// {source: file}` — the file arm is rejected under enumerated selection.
#[must_use]
pub fn config_enumerated_file_openings() -> Config {
    let json = r#"{
        "training": {
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "selection": { "method": "enumerated" },
            "scenario_source": {
                "openings": { "source": "file" }
            }
        }
    }"#;
    config_from_json(json)
}

/// Build a `Config` with `simulation.scenario_source.load.scheme = "external"`.
#[must_use]
pub fn config_with_simulation_external_load() -> Config {
    let json = r#"{
        "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        },
        "simulation": {
            "scenario_source": {
                "seed": 7,
                "load": { "scheme": "external" }
            }
        }
    }"#;
    config_from_json(json)
}

// ── Season / estimation data builders ────────────────────────────────────────

/// Build a monthly `SeasonMap` with 12 seasons (January=0 .. December=11).
#[must_use]
pub fn make_monthly_season_map() -> SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition};
    let seasons = (0..12u32)
        .map(|m| SeasonDefinition {
            id: m as usize,
            label: format!("Month{m}"),
            month_start: m + 1,
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

/// Build `n_obs` `InflowHistoryRow` records for `hydro_id`, one per calendar
/// month starting from January 2000.
#[must_use]
pub fn make_history_rows(hydro_id: i32, n_obs: usize) -> Vec<InflowHistoryRow> {
    let mut rows = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        rows.push(InflowHistoryRow {
            hydro_id: EntityId::from(hydro_id),
            start_date: date(year, month, 15),
            end_date: date(year, month, 16),
            value_m3s: 100.0,
        });
    }
    rows
}

/// Build a `StagesData` whose stages cover `n_months` monthly periods
/// starting from January 2000, each with `season_id = month_index % 12`.
/// The policy graph includes a `SeasonMap` when `with_season_map` is `true`.
#[must_use]
pub fn make_stages_with_seasons(n_months: usize, with_season_map: bool) -> StagesData {
    let mut stages = Vec::with_capacity(n_months);
    for i in 0..n_months {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        let (end_year, end_month) = if month == 12 {
            (year + 1, 1u32)
        } else {
            (year, month + 1)
        };
        let mut stage = make_stage(i as i32);
        stage.index = i;
        stage.start_date = date(year, month, 1);
        stage.end_date = date(end_year, end_month, 1);
        stage.season_id = Some(i % 12);
        stages.push(stage);
    }
    StagesData {
        openings_declared: std::collections::HashSet::new(),
        stages,
        policy_graph: make_horizon_graph(with_season_map.then(make_monthly_season_map)),
    }
}

/// Build `ParsedData` for estimation prerequisite tests.
///
/// `inflow_history` rows are provided directly; `inflow_seasonal_stats` is
/// empty (triggering the estimation path when history is non-empty).
/// Sorts `hydros` here, at the boundary — see `make_data`'s doc for why
/// this must not happen inside `make_hydro`.
#[cfg(test)]
pub(crate) fn make_data_estimation(
    mut hydros: Vec<Hydro>,
    stages: StagesData,
    inflow_history: Vec<InflowHistoryRow>,
) -> ParsedData {
    sort_hydros(&mut hydros);
    ParsedData {
        hydros,
        inflow_history,
        ..base_parsed_data(stages, base_bus())
    }
}

/// Build an `InflowArCoefficientRow` with the given hydro_id, stage_id, and lag.
#[must_use]
pub fn make_ar_row(hydro_id: i32, stage_id: i32, lag: i32) -> InflowArCoefficientRow {
    InflowArCoefficientRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        lag,
        coefficient: 0.5,
    }
}

// ── Output fixtures ───────────────────────────────────────────────────────────

/// Fixtures for the output writers and readers.
pub mod output {
    use arrow::record_batch::RecordBatch;
    use cobre_core::{System, SystemBuilder};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::path::Path;

    use crate::Config;
    use crate::output::{DistributionInfo, OutputContext};

    /// An empty but valid `System`.
    ///
    /// # Panics
    ///
    /// Never for an empty system.
    #[must_use]
    #[allow(clippy::expect_used)]
    // Rationale: `SystemBuilder::new().build()` cannot fail on an empty
    // builder; a panic here means the invariant broke, not a fixture condition.
    pub fn make_system() -> System {
        SystemBuilder::new()
            .build()
            .expect("empty system must be valid")
    }

    /// The `Config` the output writers' tests run against.
    #[must_use]
    pub fn make_config() -> Config {
        use crate::config::{
            CheckpointingConfig, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
            ModelingConfig, ParallelismConfig, PolicyConfig, PolicyMode, RowSelectionConfig,
            SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
            TrainingSolverConfig, UpperBoundEvaluationConfig,
        };
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig::default(),
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 4 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                path: "./policy".to_string(),
                mode: PolicyMode::Fresh,
                checkpointing: CheckpointingConfig::default(),
                boundary: None,
            },
            simulation: SimulationConfig {
                enabled: false,
                io_channel_capacity: 64,
                scenario_source: None,
                solver: None,
                selection: None,
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The `OutputContext` the output writers' tests run against.
    #[must_use]
    pub fn make_output_context() -> OutputContext {
        OutputContext {
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: None,
            started_at: "2026-01-17T08:00:00Z".to_string(),
            completed_at: "2026-01-17T12:30:00Z".to_string(),
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_hosts: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
                hosts: Vec::new(),
            },
            setup: None,
            production_fit_deviation: None,
        }
    }

    /// The first record batch of the parquet file at `path`.
    ///
    /// # Panics
    ///
    /// If the file cannot be opened, its parquet reader cannot be built or
    /// run, or the file carries no record batch.
    #[must_use]
    #[allow(clippy::expect_used)]
    // Rationale: each caller supplies a parquet file it just wrote itself, so
    // a failure here means the test host or the writer under test is broken,
    // not a fixture condition.
    pub fn read_first_batch(path: &Path) -> RecordBatch {
        let file = std::fs::File::open(path).expect("open parquet");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
        let mut reader = builder.build().expect("build reader");
        reader.next().expect("first batch").expect("first batch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        EstimationConfig, ExportsConfig, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };

    #[test]
    fn test_minimal_config_equals_validation_phase_struct_literal() -> serde_json::Result<()> {
        let literal = Config {
            schema: None,
            modeling: ModelingConfig::default(),
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 100 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 10 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        };

        assert_eq!(
            serde_json::to_value(&literal)?,
            serde_json::to_value(minimal_config())?,
        );
        Ok(())
    }
}
