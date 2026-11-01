//! Rewrites, in place, a directory holding a copy of
//! `examples/deterministic/d56-external-authoritative` into a deck whose
//! external inflow and load libraries hold 3 scenarios per stage against 5
//! openings, so the external clamp leaves 3 openings per stage, trained with 3
//! forward passes under the given `openings.moment_matching` spelling.

use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};

const STAGE_IDS: [i32; 6] = [0, 0, 0, 1, 1, 1];
const SCENARIO_IDS: [i32; 6] = [0, 1, 2, 0, 1, 2];
const INFLOW_M3S: [f64; 6] = [60.0, 80.0, 100.0, 70.0, 90.0, 110.0];
const LOAD_MW: [f64; 6] = [110.0, 123.0, 136.0, 115.0, 123.0, 131.0];

pub fn write_varying_external_deck(case: &Path, moment_matching: &str) {
    edit_json(&case.join("stages.json"), |stages| {
        for stage in stages["stages"]
            .as_array_mut()
            .expect("stages.json must hold a stages array")
        {
            stage["num_openings"] = json!(5);
        }
    });
    edit_json(&case.join("config.json"), |config| {
        config["training"]["selection"]["forward_passes"] = json!(3);
        config["training"]["scenario_source"]["openings"] =
            json!({"source": "generated", "moment_matching": moment_matching});
    });
    let scenarios = case.join("scenarios");
    write_external_library(
        &scenarios.join("external_inflow_scenarios.parquet"),
        "hydro_id",
        "value_m3s",
        &INFLOW_M3S,
    );
    write_external_library(
        &scenarios.join("external_load_scenarios.parquet"),
        "bus_id",
        "value_mw",
        &LOAD_MW,
    );
}

fn edit_json(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("deck JSON file must be readable"))
            .expect("deck JSON file must parse");
    edit(&mut value);
    fs::write(
        path,
        serde_json::to_string_pretty(&value).expect("deck JSON file must serialize"),
    )
    .expect("deck JSON file must be writable");
}

fn write_external_library(path: &Path, entity_column: &str, value_column: &str, values: &[f64]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new(entity_column, DataType::Int32, false),
        Field::new(value_column, DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(STAGE_IDS.to_vec())),
            Arc::new(Int32Array::from(SCENARIO_IDS.to_vec())),
            Arc::new(Int32Array::from(vec![1; values.len()])),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .expect("external library batch must build");
    let mut writer = ArrowWriter::try_new(
        File::create(path).expect("external library file must be creatable"),
        schema,
        None,
    )
    .expect("external library writer must open");
    writer
        .write(&batch)
        .expect("external library batch must be written");
    writer.close().expect("external library writer must close");
}
