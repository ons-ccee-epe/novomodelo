//! Integration coverage for the load-time resolved-generation-box validator
//! (`check_committed_value_bounds` in `validation/semantic/thermal.rs`): a
//! `past_anticipated_commitments` value that reaches an in-study delivery
//! stage is checked against that stage's RESOLVED box (the per-stage
//! `thermal_bounds.parquet` override folded in), not the plant's static
//! `[min_generation_mw, max_generation_mw]` — driven through the public
//! `cobre_io::load_case` pipeline end to end.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cobre_io::load_case;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

mod helpers;
use helpers::{make_minimal_case, write_file};

/// Declare thermal `thermal_id` on bus 1 as a `LeadStages(1)` anticipated
/// plant with static bounds `[0, 500]` — the leading study stage (id 0,
/// `make_minimal_case`'s single stage) is its one pre-study-committed
/// delivery stage.
fn write_anticipated_thermal(root: &Path, thermal_id: i32) {
    write_file(
        root,
        "system/thermals.json",
        &format!(
            r#"{{
          "thermals": [
            {{
                "id": {thermal_id},
                "name": "T_ANT",
                "operational_start_date": "2024-01-01",
                "bus_id": 1,
                "cost_per_mwh": 10.0,
                "generation": {{ "min_mw": 0.0, "max_mw": 500.0 }},
                "anticipated_config": {{ "lead_stages": 1 }}
            }}
          ]
        }}"#
        ),
    );
}

/// Write `initial_conditions.json` with a single `past_anticipated_commitments`
/// window tiling the leading study stage (id 0, `[2024-01-01, 2024-02-01)`)
/// at `value_mw`.
fn write_commitment(root: &Path, thermal_id: i32, value_mw: f64) {
    write_file(
        root,
        "initial_conditions.json",
        &format!(
            r#"{{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            {{ "thermal_id": {thermal_id}, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": {value_mw} }}
          ]
        }}"#
        ),
    );
}

/// Write `constraints/thermal_bounds.parquet` with a single stage-level
/// (`block_id = null`) row overriding `thermal_id`'s bounds at `stage_id`,
/// following the arrow `RecordBatch` + `ArrowWriter` pattern in
/// `resolver_builder_index_alignment.rs`.
fn write_thermal_bounds_override(
    root: &Path,
    thermal_id: i32,
    stage_id: i32,
    min_generation_mw: Option<f64>,
    max_generation_mw: Option<f64>,
) {
    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("min_generation_mw", DataType::Float64, true),
        Field::new("max_generation_mw", DataType::Float64, true),
        Field::new("cost_per_mwh", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![thermal_id])),
            Arc::new(Int32Array::from(vec![stage_id])),
            Arc::new(Float64Array::from(vec![min_generation_mw])),
            Arc::new(Float64Array::from(vec![max_generation_mw])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/thermal_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A per-stage override tightens `max_generation_mw` to 400 at delivery stage
/// 0, below the plant's static max of 500. The committed value 450 is INSIDE
/// the static bound `[0, 500]` (a static-bound check would pass it silently)
/// but OUTSIDE the resolved box `[0, 400]` — discriminating the resolved-box
/// check from the static one it replaced.
#[test]
fn over_commitment_above_resolved_box_is_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal(dir.path(), 31);
    write_commitment(dir.path(), 31, 450.0);
    write_thermal_bounds_override(dir.path(), 31, 0, None, Some(400.0));

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("BusinessRuleViolation"),
        "expected a BusinessRuleViolation, got: {msg}"
    );
    assert!(
        msg.contains("outside the resolved generation box"),
        "expected the resolved-box rejection message, got: {msg}"
    );
    assert!(
        msg.contains("delivery stage id 0"),
        "expected the message to name delivery stage 0, got: {msg}"
    );
}

/// A per-stage override sets `min_generation_mw = 600` at delivery stage 0
/// with no `max_generation_mw` override, so the resolved upper bound falls
/// back to the plant's static max of 500: the resolved box `[600, 500]` is
/// empty (`lower > upper`) regardless of the committed value.
#[test]
fn empty_resolved_box_is_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal(dir.path(), 32);
    write_commitment(dir.path(), 32, 0.0);
    write_thermal_bounds_override(dir.path(), 32, 0, Some(600.0), None);

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("BusinessRuleViolation"),
        "expected a BusinessRuleViolation, got: {msg}"
    );
    assert!(
        msg.contains("empty resolved generation box"),
        "expected the empty-box rejection message, got: {msg}"
    );
    assert!(
        msg.contains("delivery stage id 0"),
        "expected the message to name delivery stage 0, got: {msg}"
    );
}
