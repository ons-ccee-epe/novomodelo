//! Shared parquet input-fixture writers for in-code deck construction.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

/// Writes a `hydro_geometry.parquet` VHA table: `(hydro_id, volume_hm3, height_m, area_km2)`.
pub fn write_hydro_geometry(dest: &Path, rows: &[(i32, f64, f64, f64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("volume_hm3", DataType::Float64, false),
        Field::new("height_m", DataType::Float64, false),
        Field::new("area_km2", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("valid RecordBatch for hydro_geometry");
    let file = std::fs::File::create(dest).expect("create hydro_geometry.parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for geometry");
    writer.write(&batch).expect("write geometry batch");
    writer.close().expect("close geometry writer");
}

/// Writes a seasonal-stats table (`inflow_seasonal_stats.parquet` /
/// `load_seasonal_stats.parquet` shape): `(id, stage_id, mean, std)`, with the
/// id and mean/std column names supplied by the caller.
pub fn write_seasonal_stats(
    dest: &Path,
    id_col: &str,
    mean_col: &str,
    std_col: &str,
    rows: &[(i32, i32, f64, f64)],
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(id_col, DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new(mean_col, DataType::Float64, false),
        Field::new(std_col, DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("valid RecordBatch for seasonal stats");
    let file = std::fs::File::create(dest).expect("create seasonal stats parquet");
    let mut writer =
        ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for seasonal stats");
    writer.write(&batch).expect("write seasonal stats batch");
    writer.close().expect("close seasonal stats writer");
}
