//! Integration tests for the cross-file productivity-resolution validator.
//!
//! Exercise [`cobre_io::load_case`] against tempfile-based case directories
//! that conflict, leave coverage gaps, or are authored entirely through
//! the parquet override.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod helpers;

use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cobre_io::{LoadError, load_case};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

const PRODUCTION_MODEL_WITH_PRODUCTIVITY: &str = r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                {
                    "start_stage_id": 0,
                    "end_stage_id": null,
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": 0.9
                }
            ]
        }
    ]
}"#;

const PRODUCTION_MODEL_WITHOUT_PRODUCTIVITY: &str = r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                {
                    "start_stage_id": 0,
                    "end_stage_id": null,
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": null
                }
            ]
        }
    ]
}"#;

/// Writes a `system/hydro_energy_productivity.parquet` with the given rows.
///
/// Each row is `(hydro_id, stage_id, equivalent_productivity_mw_per_m3s)`.
fn write_productivity_parquet(dir: &TempDir, rows: &[(i32, Option<i32>, Option<f64>)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new(
            "equivalent_productivity_mw_per_m3s",
            DataType::Float64,
            true,
        ),
        Field::new("reference_volume_hm3", DataType::Float64, true),
        Field::new("reference_outflow_m3s", DataType::Float64, true),
        Field::new(
            "specific_productivity_mw_per_m3s_per_m",
            DataType::Float64,
            true,
        ),
    ]));

    let hydro_ids = Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let stage_ids = Int32Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>());
    let rho_eqs = Float64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>());
    let nulls = Float64Array::from(vec![None::<f64>; rows.len()]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(hydro_ids),
            Arc::new(stage_ids),
            Arc::new(rho_eqs),
            Arc::new(nulls.clone()),
            Arc::new(nulls.clone()),
            Arc::new(nulls),
        ],
    )
    .unwrap();

    let file =
        std::fs::File::create(dir.path().join("system/hydro_energy_productivity.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn test_load_case_rejects_conflict() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // The fixture JSON already supplies productivity for hydro 1; this parquet row duplicates it, which must be rejected as a conflict.
    write_productivity_parquet(&dir, &[(1, Some(0), Some(1.1))]);

    let err = load_case(dir.path()).expect_err("expected conflict to be rejected");
    let LoadError::ConstraintError { description } = err else {
        panic!("expected LoadError::ConstraintError, got {err}");
    };
    assert!(
        description.contains("conflict for (hydro_id=1, stage_id=0)"),
        "description should identify the offending pair, got: {description}"
    );
    assert!(
        description.contains("system/hydro_energy_productivity.parquet"),
        "description should reference the parquet path, got: {description}"
    );
    assert!(
        description.contains("system/hydro_production_models.json"),
        "description should reference the JSON path, got: {description}"
    );
}

#[test]
fn test_load_case_rejects_gap() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // No JSON productivity and no parquet leaves (hydro 1, stage 0) uncovered.
    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        PRODUCTION_MODEL_WITHOUT_PRODUCTIVITY,
    )
    .unwrap();

    let err = load_case(dir.path()).expect_err("expected coverage gap to be rejected");
    let LoadError::ConstraintError { description } = err else {
        panic!("expected LoadError::ConstraintError, got {err}");
    };
    assert!(
        description.contains("no productivity_mw_per_m3s available for (hydro_id=1, stage_id=0)"),
        "description should identify the missing pair, got: {description}"
    );
}

#[test]
fn test_load_case_accepts_parquet_only_authoring() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // JSON selector present but supplies no productivity; the parquet covers every (hydro, stage) pair.
    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        PRODUCTION_MODEL_WITHOUT_PRODUCTIVITY,
    )
    .unwrap();
    write_productivity_parquet(&dir, &[(1, Some(0), Some(0.7)), (1, Some(1), Some(0.8))]);

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("expected parquet-only authoring to load, got: {e}"));
    assert!(system.n_hydros() > 0);
}

#[test]
fn test_load_case_accepts_per_hydro_default_row() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // A parquet row with stage_id = NULL is a per-hydro default covering every stage.
    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        PRODUCTION_MODEL_WITHOUT_PRODUCTIVITY,
    )
    .unwrap();
    write_productivity_parquet(&dir, &[(1, None, Some(0.75))]);

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("expected per-hydro default row to cover, got: {e}"));
    assert!(system.n_hydros() > 0);
}

#[test]
fn test_load_case_accepts_json_only_when_parquet_absent() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // JSON supplies productivity and no parquet exists — single-source authoring is accepted.
    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        PRODUCTION_MODEL_WITH_PRODUCTIVITY,
    )
    .unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("expected JSON-only authoring to load, got: {e}"));
    assert!(system.n_hydros() > 0);
}
