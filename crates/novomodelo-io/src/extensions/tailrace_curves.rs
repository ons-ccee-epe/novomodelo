//! Parsing for `system/tailrace_curves.parquet` — piecewise-quartic
//! tailrace-level curves `tailrace_level(outflow_m3s)`.
//!
//! [`parse_tailrace_curves`] reads `system/tailrace_curves.parquet` from the
//! case directory and returns a flat, sorted `Vec<TailraceCurveRow>`.
//!
//! ## Parquet schema
//!
//! | Column          | Parquet type    | Description                                          |
//! |---------------- |---------------- |------------------------------------------------------|
//! | `hydro_id`      | INT32           | Plant whose tailrace this describes                  |
//! | `family_id`     | INT32           | Family index within the plant (sequential)           |
//! | `downstream_reference_level_m`    | DOUBLE nullable | Downstream reference level keying the family (m)      |
//! | `segment_id`    | INT32           | Piece index within the family                         |
//! | `outflow_min_m3s` | DOUBLE          | Segment lower validity bound (m³/s)                  |
//! | `outflow_max_m3s` | DOUBLE          | Segment upper validity bound (m³/s)                  |
//! | `coefficient_0`..`coefficient_4`| DOUBLE          | Degree-4 polynomial coefficients (ascending power)   |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, family_id, segment_id)` ascending using the
//! integer keys only. No floating-point column participates in the sort key, so
//! the order is exact and declaration-order-invariant.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All eleven columns must be present with the correct Arrow types
//!   (`downstream_reference_level_m` is nullable Float64; all other columns are non-nullable).
//! - `outflow_min_m3s` and `outflow_max_m3s` must be non-negative and finite.
//! - `outflow_max_m3s >= outflow_min_m3s` (a segment window is non-inverted).
//! - Each of `coefficient_0`..`coefficient_4` must be finite; any sign is accepted, because the
//!   higher-degree coefficients are routinely negative in the source data.
//! - `downstream_reference_level_m`, when non-null, must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - Cross-segment contiguity / C⁰-continuity within a family — a later layer.
//! - Family selection and downstream-level resolution — a later layer.
//! - `hydro_id` existence in the hydro registry — Layer 3.

use arrow::array::Array;
use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// A single row from `system/tailrace_curves.parquet`.
///
/// Each row is one segment of the piecewise-quartic tailrace curve for the plant
/// identified by `hydro_id`. A complete curve for one family consists of multiple
/// rows sharing `(hydro_id, family_id)`, sorted by ascending `segment_id`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::TailraceCurveRow;
/// use cobre_core::EntityId;
///
/// let row = TailraceCurveRow {
///     hydro_id: EntityId::from(42),
///     family_id: 1,
///     downstream_reference_level_m: Some(885.3),
///     segment_id: 1,
///     outflow_min_m3s: 0.0,
///     outflow_max_m3s: 1500.0,
///     coefficient_0: 320.0,
///     coefficient_1: 1.0e-3,
///     coefficient_2: -3.1521e-17,
///     coefficient_3: 0.0,
///     coefficient_4: 0.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(42));
/// assert_eq!(row.segment_id, 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TailraceCurveRow {
    /// Plant this tailrace curve belongs to.
    pub hydro_id: EntityId,
    /// Family index within the plant (sequential grouping key).
    pub family_id: i32,
    /// Downstream reference level keying the family (m). `None` when the source
    /// carries no reference level (the plant has a single family).
    pub downstream_reference_level_m: Option<f64>,
    /// Piece index within the family.
    pub segment_id: i32,
    /// Segment lower validity bound (m³/s). Non-negative.
    pub outflow_min_m3s: f64,
    /// Segment upper validity bound (m³/s). Non-negative, `>= outflow_min_m3s`.
    pub outflow_max_m3s: f64,
    /// Degree-0 polynomial coefficient.
    pub coefficient_0: f64,
    /// Degree-1 polynomial coefficient.
    pub coefficient_1: f64,
    /// Degree-2 polynomial coefficient.
    pub coefficient_2: f64,
    /// Degree-3 polynomial coefficient.
    pub coefficient_3: f64,
    /// Degree-4 polynomial coefficient.
    pub coefficient_4: f64,
}

/// Parse `system/tailrace_curves.parquet` and return a sorted segment table.
///
/// # Errors
///
/// | Condition                                          | Error variant              |
/// |--------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)           | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type              | [`LoadError::SchemaError`] |
/// | Negative / non-finite bound, inverted window, or non-finite coefficient | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::extensions::parse_tailrace_curves;
/// use std::path::Path;
///
/// let rows = parse_tailrace_curves(Path::new("system/tailrace_curves.parquet"))
///     .expect("valid tailrace file");
/// println!("loaded {} tailrace segments", rows.len());
/// ```
pub fn parse_tailrace_curves(path: &Path) -> Result<Vec<TailraceCurveRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<TailraceCurveRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let family_id_col = extract_required_int32(&batch, "family_id", path)?;
        let segment_id_col = extract_required_int32(&batch, "segment_id", path)?;
        let reference_level_col =
            extract_required_float64(&batch, "downstream_reference_level_m", path)?;
        let q_inf_col = extract_required_float64(&batch, "outflow_min_m3s", path)?;
        let q_sup_col = extract_required_float64(&batch, "outflow_max_m3s", path)?;
        let coefficient_0_col = extract_required_float64(&batch, "coefficient_0", path)?;
        let coefficient_1_col = extract_required_float64(&batch, "coefficient_1", path)?;
        let coefficient_2_col = extract_required_float64(&batch, "coefficient_2", path)?;
        let coefficient_3_col = extract_required_float64(&batch, "coefficient_3", path)?;
        let coefficient_4_col = extract_required_float64(&batch, "coefficient_4", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let family_id = family_id_col.value(i);
            let segment_id = segment_id_col.value(i);

            let downstream_reference_level_m = if reference_level_col.is_null(i) {
                None
            } else {
                Some(validate_non_negative(
                    reference_level_col.value(i),
                    row_idx,
                    "downstream_reference_level_m",
                    path,
                )?)
            };

            let outflow_min_m3s =
                validate_non_negative(q_inf_col.value(i), row_idx, "outflow_min_m3s", path)?;
            let outflow_max_m3s =
                validate_non_negative(q_sup_col.value(i), row_idx, "outflow_max_m3s", path)?;

            if outflow_max_m3s < outflow_min_m3s {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("tailrace_curves[{row_idx}].outflow_max_m3s"),
                    message: format!(
                        "upper bound must be >= lower bound, got outflow_max_m3s={outflow_max_m3s} < outflow_min_m3s={outflow_min_m3s}"
                    ),
                });
            }

            let coefficient_0 =
                validate_finite(coefficient_0_col.value(i), row_idx, "coefficient_0", path)?;
            let coefficient_1 =
                validate_finite(coefficient_1_col.value(i), row_idx, "coefficient_1", path)?;
            let coefficient_2 =
                validate_finite(coefficient_2_col.value(i), row_idx, "coefficient_2", path)?;
            let coefficient_3 =
                validate_finite(coefficient_3_col.value(i), row_idx, "coefficient_3", path)?;
            let coefficient_4 =
                validate_finite(coefficient_4_col.value(i), row_idx, "coefficient_4", path)?;

            rows.push(TailraceCurveRow {
                hydro_id,
                family_id,
                downstream_reference_level_m,
                segment_id,
                outflow_min_m3s,
                outflow_max_m3s,
                coefficient_0,
                coefficient_1,
                coefficient_2,
                coefficient_3,
                coefficient_4,
            });
        }
    }

    // Integer keys only — a float key (`total_cmp` on a level or coefficient) would
    // make tie-ordering depend on float values and break declaration-order-invariance.
    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.family_id.cmp(&b.family_id))
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    Ok(rows)
}

fn validate_non_negative(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("tailrace_curves[{row_idx}].{column}"),
            message: format!("value must be non-negative and finite, got {value}"),
        })
    }
}

/// Sign is unconstrained: the higher-degree tailrace coefficients are routinely
/// negative, so this gate rejects only NaN and the infinities.
fn validate_finite(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("tailrace_curves[{row_idx}].{column}"),
            message: format!("coefficient must be finite, got {value}"),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use crate::test_support::write_parquet_batches;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// One column's worth of test data, packed positionally so `make_batch` can
    /// assemble the twelve-column schema from per-row tuples.
    struct RowSpec {
        hydro_id: i32,
        family_id: i32,
        downstream_reference_level_m: Option<f64>,
        segment_id: i32,
        outflow_min_m3s: f64,
        outflow_max_m3s: f64,
        a_cf: [f64; 5],
    }

    fn make_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            // The reference level is the only nullable column; a non-nullable
            // Float64 array cannot carry the NULL that "single family" needs.
            Field::new("downstream_reference_level_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Int32, false),
            Field::new("outflow_min_m3s", DataType::Float64, false),
            Field::new("outflow_max_m3s", DataType::Float64, false),
            Field::new("coefficient_0", DataType::Float64, false),
            Field::new("coefficient_1", DataType::Float64, false),
            Field::new("coefficient_2", DataType::Float64, false),
            Field::new("coefficient_3", DataType::Float64, false),
            Field::new("coefficient_4", DataType::Float64, false),
        ]))
    }

    fn make_batch(rows: &[RowSpec]) -> RecordBatch {
        let hydro_ids: Vec<i32> = rows.iter().map(|r| r.hydro_id).collect();
        let family_ids: Vec<i32> = rows.iter().map(|r| r.family_id).collect();
        let reference_levels: Vec<Option<f64>> = rows
            .iter()
            .map(|r| r.downstream_reference_level_m)
            .collect();
        let segment_ids: Vec<i32> = rows.iter().map(|r| r.segment_id).collect();
        let q_infs: Vec<f64> = rows.iter().map(|r| r.outflow_min_m3s).collect();
        let q_sups: Vec<f64> = rows.iter().map(|r| r.outflow_max_m3s).collect();
        let a0: Vec<f64> = rows.iter().map(|r| r.a_cf[0]).collect();
        let a1: Vec<f64> = rows.iter().map(|r| r.a_cf[1]).collect();
        let a2: Vec<f64> = rows.iter().map(|r| r.a_cf[2]).collect();
        let a3: Vec<f64> = rows.iter().map(|r| r.a_cf[3]).collect();
        let a4: Vec<f64> = rows.iter().map(|r| r.a_cf[4]).collect();

        RecordBatch::try_new(
            make_schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids)),
                Arc::new(Int32Array::from(family_ids)),
                Arc::new(Float64Array::from(reference_levels)),
                Arc::new(Int32Array::from(segment_ids)),
                Arc::new(Float64Array::from(q_infs)),
                Arc::new(Float64Array::from(q_sups)),
                Arc::new(Float64Array::from(a0)),
                Arc::new(Float64Array::from(a1)),
                Arc::new(Float64Array::from(a2)),
                Arc::new(Float64Array::from(a3)),
                Arc::new(Float64Array::from(a4)),
            ],
        )
        .expect("valid batch construction")
    }

    /// Compact constructor for a valid segment with a fixed coefficient set.
    fn seg(
        hydro_id: i32,
        family_id: i32,
        reference_level: Option<f64>,
        segment_id: i32,
    ) -> RowSpec {
        RowSpec {
            hydro_id,
            family_id,
            downstream_reference_level_m: reference_level,
            segment_id,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 1500.0,
            a_cf: [320.0, 1.0e-3, -3.1521e-17, 0.0, 0.0],
        }
    }

    // ── AC: multi-segment family sorted ascending by segment_id ────────────────

    /// Two segments for hydro 1 / family 1 written in segment order `[2, 1]`
    /// come back sorted ascending: `rows[0].segment_id == 1`, `rows[1] == 2`.
    #[test]
    fn test_segments_sorted_ascending() {
        let batch = make_batch(&[seg(1, 1, Some(885.3), 2), seg(1, 1, Some(885.3), 1)]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].segment_id, 1);
        assert_eq!(rows[1].segment_id, 2);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].downstream_reference_level_m, Some(885.3));
    }

    // ── AC: negative quartic coefficient accepted ──────────────────────────────

    /// A negative `coefficient_2` is preserved to bit precision (no sign constraint).
    #[test]
    fn test_negative_coefficient_accepted() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.a_cf[2] = -3.1521e-17;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].coefficient_2.to_bits(), (-3.1521e-17_f64).to_bits());
    }

    // ── AC: NULL downstream_reference_level_m maps to None ───────────────────────────────────────

    /// A NULL `downstream_reference_level_m` parses to `None` and the row is otherwise accepted.
    #[test]
    fn test_null_reference_level_maps_to_none() {
        let batch = make_batch(&[seg(1, 1, None, 1)]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].downstream_reference_level_m, None);
    }

    // ── AC: inverted window rejected ───────────────────────────────────────────

    /// `outflow_max_m3s < outflow_min_m3s` -> SchemaError naming the column and row.
    #[test]
    fn test_inverted_window_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.outflow_min_m3s = 1000.0;
        spec.outflow_max_m3s = 500.0;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("outflow_max_m3s"),
                    "field should name outflow_max_m3s, got: {field}"
                );
                assert!(
                    field.contains("[0]"),
                    "field should name row 0, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ───────────────────────────────────────

    /// Two clouds of identical rows supplied in different record-batch orders
    /// parse to element-for-element equal `Vec`s (integer-key sort is total).
    #[test]
    fn test_declaration_order_invariance() {
        let order_one = make_batch(&[
            seg(7, 2, Some(900.0), 3),
            seg(7, 1, Some(885.0), 1),
            seg(3, 1, None, 2),
        ]);
        let order_two = make_batch(&[
            seg(3, 1, None, 2),
            seg(7, 2, Some(900.0), 3),
            seg(7, 1, Some(885.0), 1),
        ]);

        let rows_one = parse_tailrace_curves(write_parquet(&order_one).path()).unwrap();
        let rows_two = parse_tailrace_curves(write_parquet(&order_two).path()).unwrap();

        assert_eq!(rows_one, rows_two);
        // Coefficient fields match to bit precision under reordering.
        for (x, y) in rows_one.iter().zip(rows_two.iter()) {
            assert_eq!(x.coefficient_0.to_bits(), y.coefficient_0.to_bits());
            assert_eq!(x.coefficient_2.to_bits(), y.coefficient_2.to_bits());
        }
    }

    // ── Missing column ─────────────────────────────────────────────────────────

    /// A Parquet file missing `coefficient_4` -> SchemaError naming the column.
    #[test]
    fn test_missing_coefficient_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            Field::new("downstream_reference_level_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Int32, false),
            Field::new("outflow_min_m3s", DataType::Float64, false),
            Field::new("outflow_max_m3s", DataType::Float64, false),
            Field::new("coefficient_0", DataType::Float64, false),
            Field::new("coefficient_1", DataType::Float64, false),
            Field::new("coefficient_2", DataType::Float64, false),
            Field::new("coefficient_3", DataType::Float64, false),
            // coefficient_4 deliberately omitted
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![Some(885.3)])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1500.0])),
                Arc::new(Float64Array::from(vec![320.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "coefficient_4");
                assert!(message.contains("missing required column"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Wrong column type ──────────────────────────────────────────────────────

    /// `segment_id` provided as Float64 instead of Int32 -> SchemaError.
    #[test]
    fn test_wrong_type_segment_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("family_id", DataType::Int32, false),
            Field::new("downstream_reference_level_m", DataType::Float64, true),
            Field::new("segment_id", DataType::Float64, false), // wrong type
            Field::new("outflow_min_m3s", DataType::Float64, false),
            Field::new("outflow_max_m3s", DataType::Float64, false),
            Field::new("coefficient_0", DataType::Float64, false),
            Field::new("coefficient_1", DataType::Float64, false),
            Field::new("coefficient_2", DataType::Float64, false),
            Field::new("coefficient_3", DataType::Float64, false),
            Field::new("coefficient_4", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![Some(885.3)])),
                Arc::new(Float64Array::from(vec![1.0_f64])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![1500.0])),
                Arc::new(Float64Array::from(vec![320.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert_eq!(field, "segment_id"),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Non-finite coefficient rejected ────────────────────────────────────────

    /// A NaN coefficient -> SchemaError naming the coefficient column.
    #[test]
    fn test_nan_coefficient_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.a_cf[3] = f64::NAN;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("coefficient_3")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An infinite `outflow_min_m3s` -> SchemaError naming the bound column.
    #[test]
    fn test_infinite_bound_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.outflow_min_m3s = f64::INFINITY;
        spec.outflow_max_m3s = f64::INFINITY;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("outflow_min_m3s")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A negative `outflow_min_m3s` -> SchemaError naming the bound column.
    #[test]
    fn test_negative_bound_rejected() {
        let mut spec = seg(1, 1, Some(885.3), 1);
        spec.outflow_min_m3s = -10.0;
        let batch = make_batch(&[spec]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("outflow_min_m3s")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A negative non-null `downstream_reference_level_m` -> SchemaError naming the column.
    #[test]
    fn test_negative_reference_level_rejected() {
        let batch = make_batch(&[seg(1, 1, Some(-5.0), 1)]);
        let tmp = write_parquet(&batch);
        let err = parse_tailrace_curves(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("downstream_reference_level_m"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Empty / not-found ──────────────────────────────────────────────────────

    /// Empty Parquet (zero rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// Non-existent path -> IoError with the matching path.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/tailrace_curves.parquet");
        let err = parse_tailrace_curves(path).unwrap_err();

        match err {
            LoadError::IoError { path: err_path, .. } => assert_eq!(err_path, path),
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── Multi-batch read ───────────────────────────────────────────────────────

    /// A file with multiple record batches is fully read and globally sorted.
    #[test]
    fn test_multiple_record_batches() {
        let batch1 = make_batch(&[seg(5, 1, None, 2), seg(5, 1, None, 1)]);
        let batch2 = make_batch(&[seg(5, 1, None, 3)]);
        let tmp = write_parquet_batches(&[batch1, batch2]);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].segment_id, 1);
        assert_eq!(rows[1].segment_id, 2);
        assert_eq!(rows[2].segment_id, 3);
    }

    /// Multiple families of one plant sort by `(family_id, segment_id)`.
    #[test]
    fn test_multiple_families_sorted() {
        let batch = make_batch(&[
            seg(1, 2, Some(900.0), 1),
            seg(1, 1, Some(885.0), 2),
            seg(1, 1, Some(885.0), 1),
        ]);
        let tmp = write_parquet(&batch);
        let rows = parse_tailrace_curves(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].family_id, rows[0].segment_id), (1, 1));
        assert_eq!((rows[1].family_id, rows[1].segment_id), (1, 2));
        assert_eq!((rows[2].family_id, rows[2].segment_id), (2, 1));
    }
}
