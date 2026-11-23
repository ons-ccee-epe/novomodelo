//! Parquet parser for `constraints/penalty_overrides_bus_deficit.parquet` —
//! per-stage, optionally per-block depth overrides for a bus's deficit curve.
//!
//! [`parse_penalty_overrides_bus_deficit`] reads a Parquet file in which each row
//! sets the depth of one segment of one bus's deficit curve for one stage,
//! optionally narrowed to a single block. Segment costs always come from the
//! bus's base curve; this file carries depths only.
//!
//! ## Parquet schema
//!
//! | Column          | Type          | Required | Description                                                                      |
//! | --------------- | ------------- | -------- | -------------------------------------------------------------------------------- |
//! | `bus_id`        | INT32         | Yes      | Bus ID                                                                           |
//! | `stage_id`      | INT32         | Yes      | Study stage ID                                                                   |
//! | `block_id`      | INT32 (null)  | No       | 0-based block index within the stage; absent or null applies to the whole stage |
//! | `segment_index` | INT32         | Yes      | 0-based index into the bus's deficit curve                                       |
//! | `depth_mw`      | DOUBLE (null) | Yes      | Segment depth (MW); null leaves the segment uncapped                             |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(bus_id, stage_id, block_id, segment_index)` ascending,
//! with `block_id` `None` (stage-wide) before `Some(b)`.
//!
//! ## Validation
//!
//! The parser checks:
//!
//! - the column contract: no column outside the schema above; `bus_id`,
//!   `stage_id`, `segment_index` and `depth_mw` present with their declared
//!   types; `block_id`, when present, of type INT32;
//! - non-null keys: a null `bus_id`, `stage_id` or `segment_index` is refused.
//!
//! Later layers check that the bus exists, that the stage and block are in
//! range, that no key is duplicated, that each overridden `(bus, stage, block)`
//! carries a complete curve, and the depth values themselves.

use arrow::array::Array;
use novomodelo_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{ParquetColumns, open_parquet_input};

use super::bounds::{optional_f64, optional_i32};

/// A single row from `constraints/penalty_overrides_bus_deficit.parquet`.
///
/// Sets the depth of segment `segment_index` of bus `bus_id`'s deficit curve at
/// stage `stage_id`, for the whole stage (`block_id: None`) or for one block.
///
/// # Examples
///
/// ```
/// use novomodelo_io::constraints::BusDeficitOverrideRow;
/// use novomodelo_core::EntityId;
///
/// let row = BusDeficitOverrideRow {
///     bus_id: EntityId::from(1),
///     stage_id: 0,
///     block_id: Some(2),
///     segment_index: 0,
///     depth_mw: Some(150.0),
/// };
/// assert_eq!(row.bus_id, EntityId::from(1));
/// assert_eq!(row.depth_mw, Some(150.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BusDeficitOverrideRow {
    /// Bus ID.
    pub bus_id: EntityId,
    /// Study stage ID.
    pub stage_id: i32,
    /// `None` applies to the whole stage; `Some(b)` applies to block `b` only.
    pub block_id: Option<i32>,
    /// 0-based index into the bus's deficit curve.
    pub segment_index: i32,
    /// Segment depth (MW); `None` leaves the segment uncapped.
    pub depth_mw: Option<f64>,
}

static PENALTY_OVERRIDES_BUS_DEFICIT_COLUMNS: ParquetColumns = ParquetColumns {
    known: &[
        "bus_id",
        "stage_id",
        "block_id",
        "segment_index",
        "depth_mw",
    ],
    retired: &[],
};

/// Parse `constraints/penalty_overrides_bus_deficit.parquet`, returning rows
/// sorted by `(bus_id, stage_id, block_id, segment_index)` ascending (`None`
/// before `Some(b)`).
///
/// Depth values are returned as read; they are not range-checked here.
///
/// # Errors
///
/// | Condition                                                          | Error variant              |
/// |------------------------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                                | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)                           | [`LoadError::ParseError`]  |
/// | Unknown column                                                     | [`LoadError::SchemaError`] |
/// | Required column missing, or any column of the wrong type           | [`LoadError::SchemaError`] |
/// | Null `bus_id`, `stage_id` or `segment_index`                       | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use novomodelo_io::constraints::parse_penalty_overrides_bus_deficit;
/// use std::path::Path;
///
/// let rows = parse_penalty_overrides_bus_deficit(Path::new(
///     "constraints/penalty_overrides_bus_deficit.parquet",
/// ))
/// .expect("valid bus deficit override file");
/// println!("loaded {} bus deficit override rows", rows.len());
/// ```
pub fn parse_penalty_overrides_bus_deficit(
    path: &Path,
) -> Result<Vec<BusDeficitOverrideRow>, LoadError> {
    let mut rows = Vec::new();

    for batch in open_parquet_input(path, &PENALTY_OVERRIDES_BUS_DEFICIT_COLUMNS)? {
        let batch = batch?;

        let bus_col = batch.required_int32("bus_id")?;
        let stage_col = batch.required_int32("stage_id")?;
        let block_col = batch.optional_int32("block_id")?;
        let seg_col = batch.required_int32("segment_index")?;
        let depth_col = batch.required_float64("depth_mw")?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            for (col, column) in [
                (bus_col, "bus_id"),
                (stage_col, "stage_id"),
                (seg_col, "segment_index"),
            ] {
                if col.is_null(i) {
                    return Err(LoadError::SchemaError {
                        path: path.to_path_buf(),
                        field: format!("penalty_overrides_bus_deficit[{row_idx}].{column}"),
                        message: "value must not be null".to_string(),
                    });
                }
            }

            rows.push(BusDeficitOverrideRow {
                bus_id: EntityId::from(bus_col.value(i)),
                stage_id: stage_col.value(i),
                block_id: optional_i32(block_col, i),
                segment_index: seg_col.value(i),
                depth_mw: optional_f64(Some(depth_col), i),
            });
        }
    }

    rows.sort_by_key(|r| (r.bus_id.0, r.stage_id, r.block_id, r.segment_index));

    Ok(rows)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn int32_field(name: &str) -> Field {
        Field::new(name, DataType::Int32, true)
    }

    fn depth_field() -> Field {
        Field::new("depth_mw", DataType::Float64, true)
    }

    fn write_batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> NamedTempFile {
        let batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("valid batch");
        write_parquet(&batch)
    }

    fn full_fields() -> Vec<Field> {
        vec![
            int32_field("bus_id"),
            int32_field("stage_id"),
            int32_field("block_id"),
            int32_field("segment_index"),
            depth_field(),
        ]
    }

    fn assert_null_key_refused(err: &LoadError, expected_field: &str) {
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, expected_field);
                assert_eq!(message, "value must not be null");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn module_header_schema_table_matches_known_columns() {
        let source = include_str!("penalty_overrides_bus_deficit.rs");
        let table_names: Vec<&str> = source
            .lines()
            .skip_while(|line| line.trim() != "//! ## Parquet schema")
            .skip(1)
            .take_while(|line| !line.starts_with("//! ##"))
            .filter(|line| line.starts_with("//! | `"))
            .map(|line| {
                let rest = line.split_once('`').expect("opening backtick").1;
                rest.split_once('`').expect("closing backtick").0
            })
            .collect();

        assert_eq!(table_names, PENALTY_OVERRIDES_BUS_DEFICIT_COLUMNS.known);
    }

    #[test]
    fn parse_sorts_by_bus_stage_block_segment() {
        let tmp = write_batch(
            full_fields(),
            vec![
                Arc::new(Int32Array::from(vec![2, 1, 1, 1])),
                Arc::new(Int32Array::from(vec![0, 1, 1, 1])),
                Arc::new(Int32Array::from(vec![None, Some(0), None, None])),
                Arc::new(Int32Array::from(vec![0, 1, 1, 0])),
                Arc::new(Float64Array::from(vec![
                    Some(10.0),
                    Some(20.0),
                    Some(30.0),
                    None,
                ])),
            ],
        );

        let rows = parse_penalty_overrides_bus_deficit(tmp.path()).expect("valid file");
        let keys: Vec<(i32, i32, Option<i32>, i32)> = rows
            .iter()
            .map(|r| (r.bus_id.0, r.stage_id, r.block_id, r.segment_index))
            .collect();

        assert_eq!(
            keys,
            vec![
                (1, 1, None, 0),
                (1, 1, None, 1),
                (1, 1, Some(0), 1),
                (2, 0, None, 0),
            ]
        );
    }

    #[test]
    fn null_bus_id_is_refused() {
        let tmp = write_batch(
            full_fields(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None])),
                Arc::new(Int32Array::from(vec![0, 0])),
                Arc::new(Int32Array::from(vec![None::<i32>, None])),
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![Some(10.0), None])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert_null_key_refused(&err, "penalty_overrides_bus_deficit[1].bus_id");
    }

    #[test]
    fn null_stage_id_is_refused() {
        let tmp = write_batch(
            full_fields(),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![Some(0), None])),
                Arc::new(Int32Array::from(vec![None::<i32>, None])),
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![Some(10.0), None])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert_null_key_refused(&err, "penalty_overrides_bus_deficit[1].stage_id");
    }

    #[test]
    fn null_segment_index_is_refused() {
        let tmp = write_batch(
            full_fields(),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![0, 0])),
                Arc::new(Int32Array::from(vec![None::<i32>, None])),
                Arc::new(Int32Array::from(vec![Some(0), None])),
                Arc::new(Float64Array::from(vec![Some(10.0), None])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert_null_key_refused(&err, "penalty_overrides_bus_deficit[1].segment_index");
    }

    #[test]
    fn absent_block_id_column_reads_as_stage_wide() {
        let tmp = write_batch(
            vec![
                int32_field("bus_id"),
                int32_field("stage_id"),
                int32_field("segment_index"),
                depth_field(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![0, 0])),
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![Some(10.0), None])),
            ],
        );

        let rows = parse_penalty_overrides_bus_deficit(tmp.path()).expect("valid file");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.block_id.is_none()));
    }

    #[test]
    fn null_depth_reads_as_none() {
        let tmp = write_batch(
            full_fields(),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![0, 0])),
                Arc::new(Int32Array::from(vec![None::<i32>, None])),
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![Some(0.0), None])),
            ],
        );

        let rows = parse_penalty_overrides_bus_deficit(tmp.path()).expect("valid file");
        assert_eq!(rows[0].depth_mw, Some(0.0));
        assert_eq!(rows[1].depth_mw, None);
    }

    #[test]
    fn missing_depth_column_is_refused() {
        let tmp = write_batch(
            vec![
                int32_field("bus_id"),
                int32_field("stage_id"),
                int32_field("segment_index"),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![0])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    #[test]
    fn wrong_type_stage_id_is_refused() {
        let tmp = write_batch(
            vec![
                int32_field("bus_id"),
                Field::new("stage_id", DataType::Int64, true),
                int32_field("segment_index"),
                depth_field(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![0_i64])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![Some(10.0)])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    #[test]
    fn unknown_cost_column_is_refused() {
        let mut fields = full_fields();
        fields.push(Field::new("cost", DataType::Float64, true));
        let tmp = write_batch(
            fields,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![Some(10.0)])),
                Arc::new(Float64Array::from(vec![Some(500.0)])),
            ],
        );

        let err = parse_penalty_overrides_bus_deficit(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("cost"), "error: {err}");
    }
}
