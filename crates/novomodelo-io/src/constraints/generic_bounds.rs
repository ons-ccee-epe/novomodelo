//! Parsing for `constraints/generic_constraint_bounds.parquet` — stage/block-varying
//! RHS values for user-defined generic constraints.
//!
//! [`parse_generic_constraint_bounds`] reads
//! `constraints/generic_constraint_bounds.parquet` and returns a sorted
//! `Vec<GenericConstraintBoundsRow>`.
//!
//! ## Parquet schema
//!
//! | Column          | Type          | Required | Description                       |
//! | --------------- | ------------- | -------- | --------------------------------- |
//! | `constraint_id` | INT32         | Yes      | References constraint definition  |
//! | `stage_id`      | INT32         | Yes      | Stage index                       |
//! | `block_id`      | INT32 (null)  | No       | Block index (`null` = all blocks) |
//! | `bound_lower`   | DOUBLE (null) | No       | Lower RHS endpoint                |
//! | `bound_upper`   | DOUBLE (null) | No       | Upper RHS endpoint                |
//!
//! Shape is derived from which endpoints a row carries: lower-only, upper-only, or
//! both (a band); referential validation rejects a row with neither.
//!
//! ## Output ordering
//!
//! Rows are sorted by `(constraint_id, stage_id, block_id)` ascending.
//! `None` block sorts before `Some(i)`.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - `constraint_id`, `stage_id` must be present with the correct types.
//! - `bound_lower`, when present, must be finite (NaN and ±Inf are rejected).
//! - `bound_upper`, when present, must be finite (NaN and ±Inf are rejected).
//!
//! Deferred validations (not performed here):
//!
//! - `constraint_id` existence in the generic constraint registry.
//! - `block_id` validity for the referenced stage.
//! - Duplicate `(constraint_id, stage_id, block_id)` key detection.
//! - At-least-one-endpoint-present and `bound_upper >= bound_lower` (referential).

use arrow::array::Array;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_float64, extract_optional_int32, extract_required_int32,
    open_record_batch_reader,
};

/// A single row from `constraints/generic_constraint_bounds.parquet`.
///
/// Carries a stage/block-varying RHS interval for a generic constraint. When
/// `block_id` is `None`, the bound applies to all blocks of the given stage. Shape
/// (lower-only, upper-only, or a two-sided band) is derived from which of
/// `bound_lower`/`bound_upper` are present.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::GenericConstraintBoundsRow;
///
/// let row = GenericConstraintBoundsRow {
///     constraint_id: 0,
///     stage_id: 5,
///     block_id: None,
///     bound_lower: None,
///     bound_upper: Some(1500.0),
/// };
/// assert_eq!(row.constraint_id, 0);
/// assert!(row.block_id.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GenericConstraintBoundsRow {
    /// Constraint ID.
    pub constraint_id: i32,
    /// Stage ID.
    pub stage_id: i32,
    /// Block ID (None = all blocks).
    pub block_id: Option<i32>,
    /// Lower RHS endpoint; `None` when the row is upper-only.
    pub bound_lower: Option<f64>,
    /// Upper RHS endpoint; `None` when the row is lower-only.
    pub bound_upper: Option<f64>,
}

/// Parse `constraints/generic_constraint_bounds.parquet`, returning rows sorted by
/// `(constraint_id, stage_id, block_id)` ascending (`None` block before `Some(i)`).
///
/// # Errors
///
/// | Condition                                 | Error variant              |
/// | ----------------------------------------- | -------------------------- |
/// | File not found or permission denied       | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)  | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type     | [`LoadError::SchemaError`] |
/// | `bound_lower` or `bound_upper` is NaN or infinite | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_generic_constraint_bounds;
/// use std::path::Path;
///
/// let rows = parse_generic_constraint_bounds(
///     Path::new("case/constraints/generic_constraint_bounds.parquet")
/// ).expect("valid bounds file");
/// println!("loaded {} bound rows", rows.len());
/// ```
pub fn parse_generic_constraint_bounds(
    path: &Path,
) -> Result<Vec<GenericConstraintBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<GenericConstraintBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let constraint_id_col = extract_required_int32(&batch, "constraint_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;
        let bound_lower_col = extract_optional_float64(&batch, "bound_lower", path)?;
        let bound_upper_col = extract_optional_float64(&batch, "bound_upper", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let constraint_id = constraint_id_col.value(i);
            let stage_id = stage_id_col.value(i);
            let block_id = block_id_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let bound_lower = bound_lower_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let bound_upper = bound_upper_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            if let Some(bl) = bound_lower
                && !bl.is_finite()
            {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("generic_constraint_bounds[{row_idx}].bound_lower"),
                    message: format!("value must be finite, got {bl}"),
                });
            }

            if let Some(bu) = bound_upper
                && !bu.is_finite()
            {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("generic_constraint_bounds[{row_idx}].bound_upper"),
                    message: format!("value must be finite, got {bu}"),
                });
            }

            rows.push(GenericConstraintBoundsRow {
                constraint_id,
                stage_id,
                block_id,
                bound_lower,
                bound_upper,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.constraint_id
            .cmp(&b.constraint_id)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
            .then_with(|| match (a.block_id, b.block_id) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(a), Some(b)) => a.cmp(&b),
            })
    });

    Ok(rows)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema_full() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("block_id", DataType::Int32, true), // nullable
            Field::new("bound_lower", DataType::Float64, true), // nullable
            Field::new("bound_upper", DataType::Float64, true), // nullable
        ]))
    }

    fn schema_no_endpoints() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]))
    }

    /// Build a full 5-column batch: `constraint_id`, `stage_id`, `block_id`,
    /// `bound_lower`, `bound_upper` — every endpoint column nullable.
    fn make_batch(
        constraint_ids: &[i32],
        stage_ids: &[i32],
        block_ids: Vec<Option<i32>>,
        bound_lowers: Vec<Option<f64>>,
        bound_uppers: Vec<Option<f64>>,
    ) -> RecordBatch {
        let block_arr: Int32Array = block_ids.into_iter().collect();
        let lower_arr: Float64Array = bound_lowers.into_iter().collect();
        let upper_arr: Float64Array = bound_uppers.into_iter().collect();
        RecordBatch::try_new(
            schema_full(),
            vec![
                Arc::new(Int32Array::from(constraint_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(block_arr),
                Arc::new(lower_arr),
                Arc::new(upper_arr),
            ],
        )
        .expect("valid batch")
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Valid rows with nullable block_id and a lower endpoint on every row.
    #[test]
    fn test_parse_with_nullable_block_id() {
        let batch = make_batch(
            &[0, 0, 1],
            &[0, 0, 2],
            vec![Some(0), None, Some(1)],
            vec![Some(100.0), Some(200.0), Some(300.0)],
            vec![None, None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);

        assert_eq!(rows[0].constraint_id, 0);
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].block_id, None);
        assert_eq!(rows[0].bound_lower, Some(200.0));

        assert_eq!(rows[1].constraint_id, 0);
        assert_eq!(rows[1].stage_id, 0);
        assert_eq!(rows[1].block_id, Some(0));
        assert_eq!(rows[1].bound_lower, Some(100.0));

        assert_eq!(rows[2].constraint_id, 1);
        assert_eq!(rows[2].stage_id, 2);
        assert_eq!(rows[2].block_id, Some(1));
        assert_eq!(rows[2].bound_lower, Some(300.0));
    }

    /// When block_id column is absent, all block_id values are None.
    #[test]
    fn test_parse_without_block_id_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("bound_lower", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Int32Array::from(vec![3, 4])),
                Arc::new(Float64Array::from(vec![50.0, 75.0])),
            ],
        )
        .expect("valid batch");
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].block_id.is_none());
        assert!(rows[1].block_id.is_none());
    }

    /// Non-finite `bound_lower` → `SchemaError` naming the `bound_lower` field.
    #[test]
    fn test_parse_non_finite_bound_lower_returns_schema_error() {
        let batch = make_batch(&[0], &[0], vec![None], vec![Some(f64::NAN)], vec![None]);
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "generic_constraint_bounds[0].bound_lower");
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Infinite `bound_lower` → `SchemaError` naming the `bound_lower` field.
    #[test]
    fn test_parse_infinite_bound_lower_returns_schema_error() {
        let batch = make_batch(
            &[0],
            &[0],
            vec![None],
            vec![Some(f64::INFINITY)],
            vec![None],
        );
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "generic_constraint_bounds[0].bound_lower");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Non-finite `bound_upper` → `SchemaError` naming the `bound_upper` field.
    #[test]
    fn test_parse_non_finite_bound_upper_returns_schema_error() {
        let batch = make_batch(
            &[0],
            &[0],
            vec![None],
            vec![Some(100.0)],
            vec![Some(f64::NAN)],
        );
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "generic_constraint_bounds[0].bound_upper");
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A file with neither endpoint column present is not a reader error: every
    /// row's endpoints read `None` (referential validation is the layer that
    /// rejects the both-absent shape).
    #[test]
    fn test_parse_missing_endpoint_columns_yields_none_endpoints() {
        let batch = RecordBatch::try_new(
            schema_no_endpoints(),
            vec![
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![0])),
            ],
        )
        .expect("valid batch");
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, None);
        assert_eq!(rows[0].bound_upper, None);
    }

    /// Empty Parquet file → Ok(Vec::new()).
    #[test]
    fn test_parse_empty_parquet() {
        let batch = make_batch(&[], &[], vec![], vec![], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// Declaration-order invariance: scrambled input → sorted output.
    #[test]
    fn test_parse_sort_order_invariance() {
        let batch = make_batch(
            &[2, 0, 1, 0],
            &[0, 1, 0, 0],
            vec![None, Some(0), None, None],
            vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
            vec![None, None, None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(
            (rows[0].constraint_id, rows[0].stage_id, rows[0].block_id),
            (0, 0, None)
        );
        assert_eq!(
            (rows[1].constraint_id, rows[1].stage_id, rows[1].block_id),
            (0, 1, Some(0))
        );
        assert_eq!(
            (rows[2].constraint_id, rows[2].stage_id, rows[2].block_id),
            (1, 0, None)
        );
        assert_eq!(
            (rows[3].constraint_id, rows[3].stage_id, rows[3].block_id),
            (2, 0, None)
        );
    }

    /// Lower-only row: `bound_lower` present, `bound_upper` absent.
    #[test]
    fn test_parse_lower_only_row() {
        let batch = make_batch(&[0], &[0], vec![None], vec![Some(5.0)], vec![None]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, Some(5.0));
        assert_eq!(rows[0].bound_upper, None);
    }

    /// Upper-only row: `bound_upper` present, `bound_lower` absent.
    #[test]
    fn test_parse_upper_only_row() {
        let batch = make_batch(&[0], &[0], vec![None], vec![None], vec![Some(10.0)]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, None);
        assert_eq!(rows[0].bound_upper, Some(10.0));
    }

    /// Two-sided row: both `bound_lower` and `bound_upper` present.
    #[test]
    fn test_parse_two_sided_row() {
        let batch = make_batch(&[0], &[0], vec![None], vec![Some(35.0)], vec![Some(100.0)]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, Some(35.0));
        assert_eq!(rows[0].bound_upper, Some(100.0));
    }

    /// Neither endpoint present on a row whose endpoint columns exist but are
    /// null — accepted by the reader; referential validation rejects the shape.
    #[test]
    fn test_parse_both_endpoints_null_row() {
        let batch = make_batch(&[0], &[0], vec![None], vec![None], vec![None]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, None);
        assert_eq!(rows[0].bound_upper, None);
    }

    /// Endpoint values carry no sort weight: scrambled input still sorts by
    /// `(constraint_id, stage_id, block_id)` with both endpoints present.
    #[test]
    fn test_parse_sort_order_preserved_with_both_endpoints_present() {
        let batch = make_batch(
            &[2, 0, 1, 0],
            &[0, 1, 0, 0],
            vec![None, Some(0), None, None],
            vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
            vec![Some(15.0), None, Some(35.0), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        // Proves endpoint values do not perturb the sort (order identical to
        // test_parse_sort_order_invariance).
        assert_eq!(
            (rows[0].constraint_id, rows[0].stage_id, rows[0].block_id),
            (0, 0, None)
        );
        assert_eq!(rows[0].bound_upper, None);
        assert_eq!(
            (rows[1].constraint_id, rows[1].stage_id, rows[1].block_id),
            (0, 1, Some(0))
        );
        assert_eq!(rows[1].bound_upper, None);
        assert_eq!(
            (rows[2].constraint_id, rows[2].stage_id, rows[2].block_id),
            (1, 0, None)
        );
        assert_eq!(rows[2].bound_upper, Some(35.0));
        assert_eq!(
            (rows[3].constraint_id, rows[3].stage_id, rows[3].block_id),
            (2, 0, None)
        );
        assert_eq!(rows[3].bound_upper, Some(15.0));
    }
}
