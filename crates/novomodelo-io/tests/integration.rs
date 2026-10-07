//! End-to-end integration tests for `cobre_io::load_case`.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_wrap
)]

mod helpers;

use std::collections::HashSet;
use std::path::Path;

use arrow::array::{Array, Int32Array};
use arrow::record_batch::RecordBatch;
use cobre_core::AnticipatedConfig;
use cobre_core::EntityId;
use cobre_core::System;
use cobre_io::constraints::ThermalBoundsRow;
use cobre_io::output::simulation_writer::{
    HydroBusWriteRecord, HydroWriteRecord, ScenarioWritePayload, SimulationParquetWriter,
    StageWritePayload, write_paths,
};
use cobre_io::{load_case, validate_case};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

// ── test_minimal_valid_case ────────────────────────────────────────────────────

#[test]
fn test_minimal_valid_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("expected Ok(System) for minimal case, got Err: {e}"));

    assert_eq!(
        system.n_buses(),
        1,
        "minimal case should have exactly 1 bus"
    );
    assert_eq!(system.n_hydros(), 0, "minimal case should have 0 hydros");
    assert_eq!(
        system.n_thermals(),
        0,
        "minimal case should have 0 thermals"
    );
    assert_eq!(system.n_lines(), 0, "minimal case should have 0 lines");
    assert_eq!(
        system.n_stages(),
        1,
        "minimal case should have exactly 1 stage"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be found by O(1) lookup"
    );
}

// ── test_multi_entity_case ────────────────────────────────────────────────────

#[test]
fn test_multi_entity_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("expected Ok(System) for multi-entity case, got Err: {e}"));

    assert_eq!(
        system.n_buses(),
        2,
        "multi-entity case should have exactly 2 buses"
    );
    assert_eq!(
        system.n_hydros(),
        1,
        "multi-entity case should have exactly 1 hydro"
    );
    assert_eq!(
        system.n_thermals(),
        1,
        "multi-entity case should have exactly 1 thermal"
    );
    assert_eq!(
        system.n_lines(),
        1,
        "multi-entity case should have exactly 1 line"
    );
    assert_eq!(
        system.n_stages(),
        2,
        "multi-entity case should have exactly 2 stages"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be accessible via O(1) lookup"
    );
    assert!(
        system.bus(EntityId(2)).is_some(),
        "bus with id=2 should be accessible via O(1) lookup"
    );
}

// ── test_missing_required_file ────────────────────────────────────────────────

#[test]
fn test_missing_required_file() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    std::fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    match load_case(dir.path()) {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("buses"),
                "error display should mention 'buses', got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when buses.json is missing, got Ok"),
    }
}

// ── test_malformed_json ───────────────────────────────────────────────────────

#[test]
fn test_malformed_json() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    helpers::write_file(
        dir.path(),
        "system/hydros.json",
        "{ this is not valid json }",
    );

    match load_case(dir.path()) {
        Err(err) => {
            // Variant is implementation-defined (ParseError or ConstraintError wrapping the parse); assert only Err.
            let display = err.to_string();
            assert!(
                !display.is_empty(),
                "error display should be non-empty for malformed JSON"
            );
        }
        Ok(_) => panic!("expected Err for malformed hydros.json, got Ok"),
    }
}

// ── test_referential_integrity_violation ─────────────────────────────────────

#[test]
fn test_referential_integrity_violation() {
    let dir = TempDir::new().unwrap();
    helpers::make_referential_violation_case(&dir);

    match load_case(dir.path()) {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("999") || display.contains("bus") || display.contains("Bus"),
                "error display should mention the invalid bus reference (999), got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when hydro references non-existent bus_id=999, got Ok"),
    }
}

// ── test_inflow_history_wired_into_system ─────────────────────────────────────

/// The seasonal-stats + AR-coefficient parquets bypass the estimation path, but
/// `stages.json` still needs `season_definitions`: the loading pipeline derives
/// every order-bearing model's `residual_std_ratio` from the periodic-ACF
/// closure (`populate_derived_residual_ratios`), which requires a resolvable
/// season for any AR order > 0, regardless of whether estimation ran.
#[test]
fn test_inflow_history_wired_into_system() {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // `make_multi_entity_case`'s default `stages.json` carries no
    // `season_definitions`; this case's `inflow_ar_coefficients.parquet` below
    // has real AR order > 0, so it needs a resolvable season per stage.
    std::fs::write(
        dir.path().join("stages.json"),
        r#"{
    "season_definitions": {
        "cycle_type": "monthly",
        "seasons": [
            { "id": 0, "month_start": 1, "label": "January" },
            { "id": 1, "month_start": 2, "label": "February" },
            { "id": 2, "month_start": 3, "label": "March" },
            { "id": 3, "month_start": 4, "label": "April" },
            { "id": 4, "month_start": 5, "label": "May" },
            { "id": 5, "month_start": 6, "label": "June" },
            { "id": 6, "month_start": 7, "label": "July" },
            { "id": 7, "month_start": 8, "label": "August" },
            { "id": 8, "month_start": 9, "label": "September" },
            { "id": 9, "month_start": 10, "label": "October" },
            { "id": 10, "month_start": 11, "label": "November" },
            { "id": 11, "month_start": 12, "label": "December" }
        ]
    },
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 }
        ]
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 10
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 10
        }
    ]
}"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        r#"{ "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 2, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.85 }] },
            { "hydro_id": 3, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.8 }] }
        ] }"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 },
              "unit_groups": [
                { "id": 0, "name": "H1", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 200.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0 }
              ] },
            { "id": 2, "name": "H2", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 },
              "unit_groups": [
                { "id": 0, "name": "H2", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 100.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0 }
              ] },
            { "id": 3, "name": "H3", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 },
              "unit_groups": [
                { "id": 0, "name": "H3", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 80.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0 }
              ] }
        ] }"#,
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();

    {
        let stats_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean_m3s", DataType::Float64, false),
            Field::new("std_m3s", DataType::Float64, false),
        ]));
        let stats_batch = RecordBatch::try_new(
            Arc::clone(&stats_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Float64Array::from(vec![
                    150.0, 120.0, 80.0, 70.0, 50.0, 45.0,
                ])),
                Arc::new(Float64Array::from(vec![20.0, 15.0, 10.0, 8.0, 6.0, 5.0])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_seasonal_stats.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, stats_batch.schema(), None).unwrap();
        writer.write(&stats_batch).unwrap();
        writer.close().unwrap();
    }

    // Both stats AND coefficients must be present to skip estimation.
    {
        let ar_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("residual_std_ratio", DataType::Float64, false),
        ]));
        let ar_batch = RecordBatch::try_new(
            Arc::clone(&ar_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1, 1])),
                Arc::new(Float64Array::from(vec![0.3, 0.25, 0.4, 0.35, 0.2, 0.15])),
                Arc::new(Float64Array::from(vec![0.95, 0.92, 0.90, 0.88, 0.93, 0.91])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_ar_coefficients.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, ar_batch.schema(), None).unwrap();
        writer.write(&ar_batch).unwrap();
        writer.close().unwrap();
    }

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(864);
    let mut start_dates: Vec<i32> = Vec::with_capacity(864);
    let mut end_dates: Vec<i32> = Vec::with_capacity(864);
    let mut values: Vec<f64> = Vec::with_capacity(864);

    for hid in 1_i32..=3 {
        for year in 2000_i32..=2023 {
            for month in 1_u32..=12 {
                let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
                let end = start.checked_add_months(chrono::Months::new(1)).unwrap();
                hydro_ids.push(hid);
                start_dates.push(i32::try_from((start - epoch).num_days()).unwrap());
                end_dates.push(i32::try_from((end - epoch).num_days()).unwrap());
                values.push(f64::from(hid) * 100.0 + f64::from(month));
            }
        }
    }

    let history_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let history_batch = RecordBatch::try_new(
        Arc::clone(&history_schema),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(dir.path().join("scenarios/inflow_history.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, history_batch.schema(), None).unwrap();
    writer.write(&history_batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for inflow_history case: {e}"));

    assert_eq!(
        system.inflow_history().len(),
        864,
        "system.inflow_history() must have 864 rows (3 hydros × 24 years × 12 months)"
    );
    for row in system.inflow_history() {
        assert!(
            row.value_m3s.is_finite(),
            "every inflow_history row must have a finite value_m3s"
        );
    }

    assert!(
        system
            .inflow_models()
            .iter()
            .any(|m| !m.ar_coefficients.is_empty()),
        "the deck must produce at least one order-bearing inflow model, or the \
         residual_std_ratio check below passes vacuously"
    );
    for model in system.inflow_models() {
        if model.ar_coefficients.is_empty() {
            continue;
        }
        assert!(
            model.residual_std_ratio > 0.0 && model.residual_std_ratio < 1.0,
            "order-bearing model for hydro {:?} stage {} must have a closure-derived \
             residual_std_ratio in (0, 1), not the assemble_inflow_models placeholder \
             1.0; got {}",
            model.hydro_id,
            model.stage_id,
            model.residual_std_ratio
        );
    }
}

// ── test_external_scenarios_wired_into_system ─────────────────────────────────

#[test]
fn test_external_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        r#"{ "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 2, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.85 }] },
            { "hydro_id": 3, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.8 }] }
        ] }"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 },
              "unit_groups": [
                { "id": 0, "name": "H1", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 200.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0 }
              ] },
            { "id": 2, "name": "H2", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 },
              "unit_groups": [
                { "id": 0, "name": "H2", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 100.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0 }
              ] },
            { "id": 3, "name": "H3", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 },
              "unit_groups": [
                { "id": 0, "name": "H3", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 80.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0 }
              ] }
        ] }"#,
    )
    .unwrap();

    let mut stage_ids: Vec<i32> = Vec::with_capacity(30);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(30);
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(30);
    let mut values: Vec<f64> = Vec::with_capacity(30);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..5 {
            for hid in 1_i32..=3 {
                stage_ids.push(stage_id);
                scenario_ids.push(scenario_id);
                hydro_ids.push(hid);
                values.push(
                    f64::from(stage_id) * 1000.0 + f64::from(scenario_id) * 10.0 + f64::from(hid),
                );
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file = std::fs::File::create(
        dir.path()
            .join("scenarios/external_inflow_scenarios.parquet"),
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_inflow_scenarios case: {e}"));

    assert_eq!(
        system.external_scenarios().len(),
        30,
        "system.external_scenarios() must have 30 rows (2 stages × 5 scenarios × 3 hydros)"
    );
    for row in system.external_scenarios() {
        assert!(
            row.value_m3s.is_finite(),
            "every external_scenarios row must have a finite value_m3s"
        );
    }
}

// ── test_inflow_history_absent_returns_empty ──────────────────────────────────

#[test]
fn test_inflow_history_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.inflow_history().is_empty(),
        "system.inflow_history() must be empty when inflow_history.parquet is absent"
    );
}

// ── test_external_scenarios_absent_returns_empty ──────────────────────────────

#[test]
fn test_external_scenarios_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.external_scenarios().is_empty(),
        "system.external_scenarios() must be empty when external_inflow_scenarios.parquet is absent"
    );
}

// ── test_external_load_scenarios_wired_into_system ───────────────────────────

#[test]
fn test_external_load_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let mut stage_ids: Vec<i32> = Vec::with_capacity(6);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(6);
    let mut bus_ids: Vec<i32> = Vec::with_capacity(6);
    let mut values: Vec<f64> = Vec::with_capacity(6);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..3 {
            stage_ids.push(stage_id);
            scenario_ids.push(scenario_id);
            bus_ids.push(1);
            values.push(100.0 + f64::from(stage_id) * 50.0 + f64::from(scenario_id) * 10.0);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("bus_id", DataType::Int32, false),
        Field::new("value_mw", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(bus_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file = std::fs::File::create(dir.path().join("scenarios/external_load_scenarios.parquet"))
        .unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_load_scenarios case: {e}"));

    assert_eq!(
        system.external_load_scenarios().len(),
        6,
        "system.external_load_scenarios() must have 6 rows (2 stages × 3 scenarios × 1 bus)"
    );
    for row in system.external_load_scenarios() {
        assert!(
            row.value_mw.is_finite(),
            "every external_load_scenarios row must have a finite value_mw"
        );
    }
}

// ── test_external_ncs_scenarios_wired_into_system ─────────────────────────────

#[test]
fn test_external_ncs_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // The referential validator rejects any ncs_id absent from the NCS registry, so register 1 and 2.
    helpers::write_file(
        dir.path(),
        "system/non_controllable_sources.json",
        r#"{
    "non_controllable_sources": [
        { "id": 1, "name": "NCS_A", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 50.0 },
        { "id": 2, "name": "NCS_B", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 30.0 }
    ]
}"#,
    );

    let mut stage_ids: Vec<i32> = Vec::with_capacity(16);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(16);
    let mut ncs_ids: Vec<i32> = Vec::with_capacity(16);
    let mut values: Vec<f64> = Vec::with_capacity(16);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..4 {
            for ncs_id in 1_i32..=2 {
                stage_ids.push(stage_id);
                scenario_ids.push(scenario_id);
                ncs_ids.push(ncs_id);
                // availability factor in (0, 1]
                values.push(0.5 + 0.1 * f64::from(scenario_id) + 0.05 * f64::from(ncs_id - 1));
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("ncs_id", DataType::Int32, false),
        Field::new("availability_factor", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(ncs_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file =
        std::fs::File::create(dir.path().join("scenarios/external_ncs_scenarios.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_ncs_scenarios case: {e}"));

    assert_eq!(
        system.external_ncs_scenarios().len(),
        16,
        "system.external_ncs_scenarios() must have 16 rows (2 stages × 4 scenarios × 2 NCS)"
    );
    for row in system.external_ncs_scenarios() {
        assert!(
            row.value.is_finite(),
            "every external_ncs_scenarios row must have a finite value"
        );
    }
}

// ── test_lead_time_single_decider_on_disk_load ────────────────────────────────

/// Single-decider `LeadTime` thermal loads and produces correct resolved bounds at every study stage.
#[test]
fn test_lead_time_single_decider_on_disk_load() {
    // SystemBuilder sorts thermals by (operational_start_date, id); both
    // declared thermals share the same date, so ascending id puts T_ANT
    // (id=2) at index 0.
    const THERMAL_IDX_ANT: usize = 0;

    let dir = TempDir::new().unwrap();
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "stages.json",
        r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.0,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 },
            { "source_id": 1, "target_id": 2, "probability": 1.0 }
        ]
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_openings": 1
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_openings": 1
        },
        {
            "id": 2,
            "start_date": "2024-03-01",
            "end_date": "2024-04-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_openings": 1
        }
    ]
}"#,
    );
    helpers::write_file(
        root,
        "initial_conditions.json",
        r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }
    ]
}"#,
    );
    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(
        root,
        "system/thermals.json",
        r#"{
    "thermals": [
        {
            "id": 2,
            "name": "T_ANT",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 10.0,
            "generation": { "min_mw": 0.0, "max_mw": 100.0 },
            "anticipated_config": { "lead_time_hours": 744.0 }
        },
        {
            "id": 3,
            "name": "T_BACKUP",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 100.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        }
    ]
}"#,
    );

    let system = load_case(root).unwrap_or_else(|e| {
        panic!("load_case must succeed for a valid single-decider LeadTime case, got: {e}")
    });

    assert_eq!(system.n_stages(), 3, "case declares 3 study stages");
    assert_eq!(system.n_thermals(), 2, "case declares 2 thermals");

    let ant = &system.thermals()[THERMAL_IDX_ANT];
    assert_eq!(
        ant.anticipated_config,
        Some(AnticipatedConfig::LeadTime(744.0)),
        "loaded thermal must carry the LeadTime(744.0) config"
    );

    // Every study stage is a legitimate delivery-stage index for the
    // delivery-anchored ring: the resolved bounds must be correct there, not
    // merely absent a panic. The k_max=0 contribution this LeadTime plant
    // makes to the padded thermal-bounds axis pads no region a real study
    // ever reads.
    for stage in 0..system.n_stages() {
        let b = system.bounds().thermal_bounds(THERMAL_IDX_ANT, stage);
        let bb = system.bounds().thermal_block_base(THERMAL_IDX_ANT, stage);
        assert!(
            (bb.min_generation_mw - 0.0).abs() < f64::EPSILON,
            "stage {stage}: min_generation_mw mismatch, got {}",
            bb.min_generation_mw
        );
        assert!(
            (bb.max_generation_mw - 100.0).abs() < f64::EPSILON,
            "stage {stage}: max_generation_mw mismatch, got {}",
            bb.max_generation_mw
        );
        assert!(
            (b.cost_per_mwh - 10.0).abs() < f64::EPSILON,
            "stage {stage}: cost_per_mwh mismatch, got {}",
            b.cost_per_mwh
        );
    }
}

// ── LeadStages full-anticipation: load_case ↔ validate_case agreement ────────

/// Write `system/thermals.json` with a single `LeadStages` anticipated thermal.
fn write_lead_stages_anticipated_thermal(root: &Path, lead_stages: u32) {
    helpers::write_file(
        root,
        "system/thermals.json",
        &format!(
            r#"{{
    "thermals": [
        {{
            "id": 1,
            "name": "T_ANT",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 10.0,
            "generation": {{ "min_mw": 0.0, "max_mw": 300.0 }},
            "anticipated_config": {{ "lead_stages": {lead_stages} }}
        }}
    ]
}}"#
        ),
    );
}

/// Write `initial_conditions.json` with past commitments for the study stage and optionally the post-study window.
fn write_ic_for_full_anticipation(root: &Path, with_post_study_window: bool) {
    let mut windows = vec![
        r#"{ "thermal_id": 1, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }"#
            .to_string(),
    ];
    if with_post_study_window {
        windows.push(
            r#"{ "thermal_id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01", "value_mw": 0.0 }"#
                .to_string(),
        );
    }
    helpers::write_file(
        root,
        "initial_conditions.json",
        &format!(
            r#"{{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [{}]
}}"#,
            windows.join(",\n")
        ),
    );
}

/// `LeadStages(2)` thermal reaching only a tiled post-study stage is accepted by both `load_case` and `validate_case`.
#[test]
fn test_full_anticipation_lead_stages_reaching_only_post_study_loads() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    let root = dir.path();

    write_lead_stages_anticipated_thermal(root, 2);
    write_ic_for_full_anticipation(root, true);
    helpers::write_file(
        root,
        "post_study_stages.json",
        r#"{
    "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
    "thermal_bounds": []
}"#,
    );

    load_case(root).unwrap_or_else(|e| {
        panic!("load_case must accept a full-anticipation LeadStages thermal, got: {e}")
    });
    validate_case(root).unwrap_or_else(|e| {
        panic!("validate_case must accept a full-anticipation LeadStages thermal, got: {e}")
    });
}

/// `LeadStages` exceeding horizon without post-study stages is rejected by both entry points.
#[test]
fn test_lead_stages_exceeding_horizon_without_post_study_stages_rejected_by_both_entry_points() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    let root = dir.path();

    write_lead_stages_anticipated_thermal(root, 2);
    write_ic_for_full_anticipation(root, false);

    let load_msg = load_case(root).unwrap_err().to_string();
    assert!(
        load_msg.contains("lead_stages exceeds study horizon") && load_msg.contains("Thermal 1"),
        "load_case should reject the unreachable LeadStages thermal, got: {load_msg}"
    );

    let validate_msg = validate_case(root).unwrap_err().to_string();
    assert!(
        validate_msg.contains("lead_stages exceeds study horizon")
            && validate_msg.contains("Thermal 1"),
        "validate_case should reject the unreachable LeadStages thermal identically, got: {validate_msg}"
    );
}

// ── test_postcard_round_trip ──────────────────────────────────────────────────

#[test]
fn test_postcard_round_trip() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let original = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case should succeed for minimal case, got: {e}"));

    let bytes = postcard::to_allocvec(&original)
        .unwrap_or_else(|e| panic!("postcard serialization should succeed, got: {e}"));

    assert!(!bytes.is_empty(), "serialized bytes should be non-empty");

    let deserialized = postcard::from_bytes::<System>(&bytes)
        .unwrap_or_else(|e| panic!("postcard deserialization should succeed, got: {e}"));

    assert_eq!(
        deserialized.n_buses(),
        original.n_buses(),
        "bus count must match after postcard round-trip"
    );
    assert!(
        deserialized.bus(EntityId(1)).is_some(),
        "O(1) bus lookup must work after index rebuild on deserialized System"
    );

    assert_eq!(
        deserialized.n_hydros(),
        original.n_hydros(),
        "hydro count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_thermals(),
        original.n_thermals(),
        "thermal count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_lines(),
        original.n_lines(),
        "line count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_stages(),
        original.n_stages(),
        "stage count must match after round-trip"
    );
}

// ── hydro_bus_generation output shape ─────────────────────────────────────────

/// Every non-essential column at a plausible value; only the caller-chosen
/// `turbined_m3s`/`generation_mw` are load-bearing for the shape assertions
/// these tests make.
fn make_hydro_write_record(
    stage_id: u32,
    block_id: Option<u32>,
    hydro_id: i32,
    turbined_m3s: f64,
    generation_mw: f64,
) -> HydroWriteRecord {
    HydroWriteRecord {
        stage_id,
        node_id: stage_id as i32,
        block_id,
        hydro_id,
        turbined_m3s,
        spillage_m3s: 0.0,
        evaporation_m3s: None,
        diverted_inflow_m3s: None,
        diverted_outflow_m3s: None,
        incremental_inflow_m3s: 0.0,
        inflow_m3s: 0.0,
        storage_initial_hm3: 0.0,
        storage_final_hm3: 0.0,
        generation_mw,
        equivalent_productivity_mw_per_m3s: 0.0,
        accumulated_productivity_mw_per_m3s: 0.0,
        incremental_inflow_energy_mw: 0.0,
        stored_energy_initial_mwh: 0.0,
        stored_energy_final_mwh: 0.0,
        spillage_cost: 0.0,
        water_value_per_hm3: 0.0,
        storage_binding_code: 0,
        operative_state_code: 0,
        turbined_slack_m3s: 0.0,
        outflow_slack_below_m3s: 0.0,
        outflow_slack_above_m3s: 0.0,
        generation_slack_mw: 0.0,
        storage_violation_below_hm3: 0.0,
        filling_target_violation_hm3: 0.0,
        evaporation_violation_pos_m3s: 0.0,
        evaporation_violation_neg_m3s: 0.0,
        inflow_nonnegativity_slack_m3s: 0.0,
        water_withdrawal_violation_pos_m3s: 0.0,
        water_withdrawal_violation_neg_m3s: 0.0,
        integrated_equivalent_productivity_mw_per_m3s: 0.0,
        integrated_accumulated_productivity_mw_per_m3s: 0.0,
        stored_energy_initial_mw: 0.0,
        stored_energy_final_mw: 0.0,
    }
}

fn empty_stage_write_payload(
    stage_id: u32,
    hydros: Vec<HydroWriteRecord>,
    hydro_bus_generation: Vec<HydroBusWriteRecord>,
) -> StageWritePayload {
    StageWritePayload {
        stage_id,
        node_id: stage_id as i32,
        costs: vec![],
        hydros,
        hydro_bus_generation,
        thermals: vec![],
        exchanges: vec![],
        buses: vec![],
        pumping_stations: vec![],
        contracts: vec![],
        non_controllables: vec![],
        inflow_lags: vec![],
        transit_buckets: vec![],
        generic_violations: vec![],
        anticipated_lanes: vec![],
    }
}

fn read_single_batch(path: &Path) -> RecordBatch {
    let file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{} must open: {e}", path.display()));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap_or_else(|e| panic!("reader builder must succeed: {e}"));
    let mut reader = builder
        .build()
        .unwrap_or_else(|e| panic!("reader must build: {e}"));
    let batch = reader
        .next()
        .unwrap_or_else(|| panic!("batch must have rows"));
    batch.unwrap_or_else(|e| panic!("batch must be Ok: {e}"))
}

/// Writes `stages` through a fresh [`SimulationParquetWriter`] and reads back
/// the written `hydros`/`hydro_bus_generation` batches for scenario 0 —
/// shared by the split-plant and single-bus output-shape tests below.
fn write_and_read_hydro_batches(
    system: &System,
    stages: Vec<StageWritePayload>,
) -> (RecordBatch, RecordBatch) {
    let tmp = TempDir::new().unwrap();
    let mut writer = SimulationParquetWriter::new(tmp.path(), system)
        .unwrap_or_else(|e| panic!("SimulationParquetWriter::new must succeed: {e}"));
    writer
        .write_scenario(ScenarioWritePayload {
            scenario_id: 0,
            stages,
            transit_seed: vec![],
        })
        .unwrap_or_else(|e| panic!("write_scenario must succeed: {e}"));

    let hydros_batch = read_single_batch(
        &tmp.path()
            .join("simulation/hydros/scenario_id=0000/data.parquet"),
    );
    let bus_batch = read_single_batch(
        &tmp.path()
            .join("simulation/hydro_bus_generation/scenario_id=0000/data.parquet"),
    );
    (hydros_batch, bus_batch)
}

fn i32_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int32Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} must exist"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("column {name} must be Int32Array"))
}

/// `(stage_id, block_id, hydro_id)` key set of a `hydros`- or
/// `hydro_bus_generation`-shaped batch — deduplicating over `bus_id` when
/// applied to the latter is exactly what lets the two be compared directly.
fn hydro_key_set(batch: &RecordBatch) -> HashSet<(i32, Option<i32>, i32)> {
    let stage = i32_column(batch, "stage_id");
    let block = i32_column(batch, "block_id");
    let hydro = i32_column(batch, "hydro_id");
    (0..batch.num_rows())
        .map(|i| {
            let block_id = if block.is_null(i) {
                None
            } else {
                Some(block.value(i))
            };
            (stage.value(i), block_id, hydro.value(i))
        })
        .collect()
}

fn hydro_bus_key_set(batch: &RecordBatch) -> HashSet<(i32, Option<i32>, i32, i32)> {
    let stage = i32_column(batch, "stage_id");
    let block = i32_column(batch, "block_id");
    let hydro = i32_column(batch, "hydro_id");
    let bus = i32_column(batch, "bus_id");
    (0..batch.num_rows())
        .map(|i| {
            let block_id = if block.is_null(i) {
                None
            } else {
                Some(block.value(i))
            };
            (stage.value(i), block_id, hydro.value(i), bus.value(i))
        })
        .collect()
}

/// End-to-end output shape: the split plant's
/// `hydro_bus_generation` rows have a non-null `bus_id` on every row, exactly
/// one row per `(stage, block, hydro, bus)`, `bus_id` in `{0, 1}` (the
/// fixture's two declared buses), and their `(stage, block, hydro)` key set
/// matches the plant's own rows in `simulation/hydros/`.
#[test]
fn test_split_plant_hydro_bus_generation_output_shape() {
    let case_dir = Path::new("../../examples/deterministic/d51-split-plant-two-bus");
    let system = load_case(case_dir).unwrap_or_else(|e| panic!("d51 load_case must succeed: {e}"));
    assert_eq!(system.n_hydros(), 1, "d51 declares exactly 1 hydro plant");

    // d51's own topology: stage 0 has 2 blocks, stage 1 has 1 block; H0 (id 0)
    // splits across bus 0 and bus 1.
    let stage_block_counts = [(0_u32, 2_u32), (1_u32, 1_u32)];
    let mut stages = Vec::new();
    for (stage_id, n_blks) in stage_block_counts {
        let mut hydros = Vec::new();
        let mut hydro_bus_generation = Vec::new();
        for block_id in 0..n_blks {
            hydros.push(make_hydro_write_record(
                stage_id,
                Some(block_id),
                0,
                70.0,
                40.0,
            ));
            hydro_bus_generation.push(HydroBusWriteRecord {
                stage_id,
                node_id: stage_id as i32,
                block_id: Some(block_id),
                hydro_id: 0,
                bus_id: 0,
                turbined_m3s: 42.0,
                generation_mw: 25.0,
            });
            hydro_bus_generation.push(HydroBusWriteRecord {
                stage_id,
                node_id: stage_id as i32,
                block_id: Some(block_id),
                hydro_id: 0,
                bus_id: 1,
                turbined_m3s: 28.0,
                generation_mw: 15.0,
            });
        }
        stages.push(empty_stage_write_payload(
            stage_id,
            hydros,
            hydro_bus_generation,
        ));
    }
    let (hydros_batch, bus_batch) = write_and_read_hydro_batches(&system, stages);

    let bus_id_col = i32_column(&bus_batch, "bus_id");
    assert_eq!(
        bus_id_col.null_count(),
        0,
        "every hydro_bus_generation row's bus_id must be non-null"
    );
    let bus_ids: HashSet<i32> = (0..bus_batch.num_rows())
        .map(|i| bus_id_col.value(i))
        .collect();
    assert_eq!(
        bus_ids,
        HashSet::from([0, 1]),
        "bus_id must range over exactly the fixture's two declared buses"
    );

    assert_eq!(
        bus_batch.num_rows(),
        2 * hydros_batch.num_rows(),
        "exactly one hydro_bus_generation row per (stage, block, hydro, bus) -- \
         2 buses per plant row, no more, no fewer"
    );
    let full_keys = hydro_bus_key_set(&bus_batch);
    assert_eq!(
        full_keys.len(),
        bus_batch.num_rows(),
        "no duplicate (stage, block, hydro, bus) row"
    );

    let hydros_keys = hydro_key_set(&hydros_batch);
    let bus_plant_keys = hydro_key_set(&bus_batch);
    assert_eq!(
        bus_plant_keys, hydros_keys,
        "hydro_bus_generation's (stage, block, hydro) key set must match hydros'"
    );
}

/// A fan run (two scenarios visiting distinct node ids per stage) writes entity
/// files carrying `scenario_id`/`node_id` columns and a run-level `paths.parquet`;
/// joining `paths` to an entity file on `(scenario_id, stage_id)` returns every
/// row matched on both sides — the join that is impossible when `scenario_id`
/// lives only in the Hive path. (The `scenario_summary` leg of the three-way join
/// is deferred until the run-level `scenario_summary.parquet` output exists.)
#[test]
fn test_paths_join_to_entity_file_on_scenario_and_stage() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let system = load_case(case_dir).unwrap_or_else(|e| panic!("d02 load_case must succeed: {e}"));

    let tmp = TempDir::new().unwrap();
    let mut writer = SimulationParquetWriter::new(tmp.path(), &system)
        .unwrap_or_else(|e| panic!("SimulationParquetWriter::new must succeed: {e}"));

    // Two scenarios; the node id visited at stage 1 differs across scenarios (a
    // fan), and differs from the stage index — so a value that merely echoed
    // stage_id would not pass the node_id join check below.
    let node_ids: [[i32; 2]; 2] = [[10, 21], [10, 32]];
    for (scenario_id, nodes) in node_ids.iter().enumerate() {
        let stages: Vec<StageWritePayload> = (0..2u32)
            .map(|stage_id| {
                let mut stage = empty_stage_write_payload(
                    stage_id,
                    vec![make_hydro_write_record(stage_id, Some(0), 0, 40.0, 20.0)],
                    vec![],
                );
                stage.node_id = nodes[stage_id as usize];
                stage.hydros[0].node_id = nodes[stage_id as usize];
                stage
            })
            .collect();
        #[allow(clippy::cast_possible_truncation)]
        writer
            .write_scenario(ScenarioWritePayload {
                scenario_id: scenario_id as u32,
                stages,
                transit_seed: vec![],
            })
            .unwrap_or_else(|e| panic!("write_scenario must succeed: {e}"));
    }
    let path_rows = writer.path_rows().to_vec();
    write_paths(tmp.path(), path_rows).unwrap_or_else(|e| panic!("write_paths must succeed: {e}"));

    // paths.parquet: run-level, unpartitioned, exactly (scenario_id, stage_id, node_id).
    let paths = read_single_batch(&tmp.path().join("simulation/paths.parquet"));
    let paths_schema = paths.schema();
    let paths_names: Vec<&str> = paths_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(paths_names, vec!["scenario_id", "stage_id", "node_id"]);
    assert_eq!(paths.num_rows(), 4, "2 scenarios × 2 stages");

    // (scenario_id, stage_id) -> node_id from paths.parquet.
    let p_scenario = i32_column(&paths, "scenario_id");
    let p_stage = i32_column(&paths, "stage_id");
    let p_node = i32_column(&paths, "node_id");
    let mut paths_map: std::collections::HashMap<(i32, i32), i32> =
        std::collections::HashMap::new();
    for i in 0..paths.num_rows() {
        paths_map.insert((p_scenario.value(i), p_stage.value(i)), p_node.value(i));
    }

    // Join each scenario's entity file to paths on (scenario_id, stage_id).
    let mut matched_paths_keys: HashSet<(i32, i32)> = HashSet::new();
    for scenario_id in 0..2 {
        let entity = read_single_batch(&tmp.path().join(format!(
            "simulation/hydros/scenario_id={scenario_id:04}/data.parquet"
        )));
        let e_scenario = i32_column(&entity, "scenario_id");
        let e_stage = i32_column(&entity, "stage_id");
        let e_node = i32_column(&entity, "node_id");
        for i in 0..entity.num_rows() {
            let key = (e_scenario.value(i), e_stage.value(i));
            let paths_node = paths_map
                .get(&key)
                .unwrap_or_else(|| panic!("entity row {key:?} has no matching paths.parquet row"));
            assert_eq!(
                *paths_node,
                e_node.value(i),
                "entity node_id must equal the paths node_id for {key:?}"
            );
            matched_paths_keys.insert(key);
        }
    }
    // No unmatched paths rows: every (scenario, stage) in paths appears in the
    // entity data (d02 has one hydro at every stage).
    assert_eq!(
        matched_paths_keys.len(),
        paths.num_rows(),
        "every paths.parquet row must match an entity row"
    );
}

/// Single-bus control (recruiting `d02-single-hydro` rather than
/// authoring a new case): a single-group plant's `hydro_bus_generation`
/// still emits exactly one row per `(stage, block, hydro)`, with a non-null
/// `bus_id` equal to that plant's one declared bus -- proving the output
/// shape is uniform, not a split-plant special case.
#[test]
fn test_single_bus_hydro_bus_generation_output_shape() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let system = load_case(case_dir).unwrap_or_else(|e| panic!("d02 load_case must succeed: {e}"));
    assert_eq!(system.n_hydros(), 1, "d02 declares exactly 1 hydro plant");
    assert_eq!(system.n_buses(), 1, "d02 declares exactly 1 bus");

    // d02's own topology: 2 stages, 1 block each, H0 (id 0) on bus 0.
    let mut stages = Vec::new();
    for stage_id in 0_u32..2 {
        let hydros = vec![make_hydro_write_record(stage_id, Some(0), 0, 40.0, 20.0)];
        let hydro_bus_generation = vec![HydroBusWriteRecord {
            stage_id,
            node_id: stage_id as i32,
            block_id: Some(0),
            hydro_id: 0,
            bus_id: 0,
            turbined_m3s: 40.0,
            generation_mw: 20.0,
        }];
        stages.push(empty_stage_write_payload(
            stage_id,
            hydros,
            hydro_bus_generation,
        ));
    }
    let (hydros_batch, bus_batch) = write_and_read_hydro_batches(&system, stages);

    assert_eq!(
        bus_batch.num_rows(),
        hydros_batch.num_rows(),
        "single-bus plant: exactly 1 hydro_bus_generation row per hydros row"
    );

    let bus_id_col = i32_column(&bus_batch, "bus_id");
    assert_eq!(
        bus_id_col.null_count(),
        0,
        "every hydro_bus_generation row's bus_id must be non-null"
    );
    let bus_ids: HashSet<i32> = (0..bus_batch.num_rows())
        .map(|i| bus_id_col.value(i))
        .collect();
    assert_eq!(
        bus_ids,
        HashSet::from([0]),
        "the single-group plant's cell must be pinned to its one declared bus"
    );

    assert_eq!(
        hydro_key_set(&bus_batch),
        hydro_key_set(&hydros_batch),
        "hydro_bus_generation's (stage, block, hydro) key set must match hydros'"
    );
}

// ── block-axis bound-row validation message shape ────────────────────────────

/// Writes `constraints/thermal_bounds.parquet` from explicit rows.
fn write_thermal_bounds_parquet(dir: &Path, rows: &[ThermalBoundsRow]) {
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("min_generation_mw", DataType::Float64, true),
        Field::new("max_generation_mw", DataType::Float64, true),
        Field::new("cost_per_mwh", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.thermal_id.0).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.stage_id).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.min_generation_mw).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.max_generation_mw).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.cost_per_mwh).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.block_id).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.join("constraints")).unwrap();
    let file = std::fs::File::create(dir.join("constraints/thermal_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Out-of-range `block_id` in thermal bounds is rejected naming the entity, stage, and block.
#[test]
fn test_thermal_bounds_block_id_out_of_range_rejected_by_load_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    write_thermal_bounds_parquet(
        dir.path(),
        &[ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(100.0),
            cost_per_mwh: None,
            block_id: Some(5),
        }],
    );

    match load_case(dir.path()) {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("thermal_id=1"),
                "message must name the entity: {display}"
            );
            assert!(
                display.contains("stage_id=0"),
                "message must name the stage: {display}"
            );
            assert!(
                display.contains("block_id=5"),
                "message must name the offending block_id: {display}"
            );
        }
        Ok(_) => panic!("expected Err for an out-of-range thermal_bounds block_id, got Ok"),
    }
}

/// Duplicate thermal bound row is rejected naming the entity, stage, and colliding column.
#[test]
fn test_thermal_bounds_duplicate_row_rejected_by_load_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    write_thermal_bounds_parquet(
        dir.path(),
        &[
            ThermalBoundsRow {
                thermal_id: EntityId::from(1),
                stage_id: 0,
                min_generation_mw: None,
                max_generation_mw: Some(100.0),
                cost_per_mwh: None,
                block_id: None,
            },
            ThermalBoundsRow {
                thermal_id: EntityId::from(1),
                stage_id: 0,
                min_generation_mw: None,
                max_generation_mw: Some(200.0),
                cost_per_mwh: None,
                block_id: None,
            },
        ],
    );

    match load_case(dir.path()) {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("thermal_id=1"),
                "message must name the entity: {display}"
            );
            assert!(
                display.contains("stage_id=0"),
                "message must name the stage: {display}"
            );
            assert!(
                display.contains("max_generation_mw"),
                "message must name the colliding column: {display}"
            );
        }
        Ok(_) => panic!("expected Err for a duplicate thermal_bounds row, got Ok"),
    }
}

/// `block_id` on stage-level-only `cost_per_mwh` column is rejected naming the entity and stage.
#[test]
fn test_thermal_bounds_block_id_on_stage_level_cost_column_rejected_by_load_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    write_thermal_bounds_parquet(
        dir.path(),
        &[ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: Some(999.0),
            block_id: Some(0),
        }],
    );

    match load_case(dir.path()) {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("thermal_id=1"),
                "message must name the entity: {display}"
            );
            assert!(
                display.contains("stage_id=0"),
                "message must name the stage: {display}"
            );
            assert!(
                display.contains("cost_per_mwh"),
                "message must name the offending stage-level-only column: {display}"
            );
        }
        Ok(_) => panic!(
            "expected Err for a block_id on the stage-level-only cost_per_mwh column, got Ok"
        ),
    }
}

/// Recursively copies `src` into `dst`, skipping `output` directories.
fn copy_case_dir_into(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let file_type = entry.file_type().unwrap();
        let src_path = entry.path();
        if file_type.is_dir() {
            if entry.file_name() == "output" {
                continue;
            }
            copy_case_dir_into(&src_path, &dst.join(entry.file_name()));
        } else {
            std::fs::copy(&src_path, dst.join(entry.file_name())).unwrap();
        }
    }
}

/// AR(1) model with sigma=0 external inflow is rejected naming the deterministic PAR output reason.
#[test]
fn test_d56_ar1_sigma_zero_inflow_rejected_by_validate_case() {
    let src = Path::new("../../examples/deterministic/d56-external-authoritative");
    let dir = TempDir::new().unwrap();
    copy_case_dir_into(src, dir.path());
    std::fs::copy(
        Path::new("tests/fixtures/d56_reject_ar_coefficients.parquet"),
        dir.path().join("scenarios/inflow_ar_coefficients.parquet"),
    )
    .unwrap();

    match validate_case(dir.path()) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("deterministic PAR output"),
                "message must name the deterministic PAR output reason: {msg}"
            );
            assert!(
                !msg.contains("inversion is undefined"),
                "message must not repeat the retired 'inversion is undefined' phrasing: {msg}"
            );
        }
        Ok(_) => panic!("expected the AR(1) + sigma=0 external inflow deck to be rejected, got Ok"),
    }
}
