//! Smoke test for the dual-calendar boundary gate fixture under
//! `crates/novomodelo-sddp/tests/gate_fixtures/`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;

use arrow::array::{Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

mod common;
use common::{novomodelo, prepare_dual_calendar_boundary_case};

const FIRST_POST_STUDY_DELIVERY_DATE: i32 = 20_240_701;

#[test]
fn dual_calendar_boundary_fixture_trains_and_simulates() {
    let case = TempDir::new().unwrap();
    prepare_dual_calendar_boundary_case(case.path());
    let out = TempDir::new().unwrap();

    let run = novomodelo()
        .args(["run", case.path().to_str().unwrap()])
        .args(["--output", out.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "the target run must succeed: {stderr}"
    );
    assert!(
        !stderr.contains("prices at zero terminal"),
        "the boundary policy must price the post-study deliveries: {stderr}"
    );

    assert!(
        fs::read_dir(out.path().join("policy"))
            .unwrap()
            .next()
            .is_some(),
        "the run must write a non-empty policy/"
    );

    let mut delivery_dates = Vec::new();
    for scenario in fs::read_dir(out.path().join("simulation/anticipated_lanes")).unwrap() {
        let file = fs::File::open(scenario.unwrap().path().join("data.parquet")).unwrap();
        for batch in ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
        {
            let batch = batch.unwrap();
            let dates = batch
                .column_by_name("delivery_date")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            delivery_dates.extend(dates.iter().flatten());
        }
    }
    assert!(
        delivery_dates
            .iter()
            .any(|&date| date >= FIRST_POST_STUDY_DELIVERY_DATE),
        "anticipated_lanes must carry a delivery on or after the first post-study date: {delivery_dates:?}"
    );
}
