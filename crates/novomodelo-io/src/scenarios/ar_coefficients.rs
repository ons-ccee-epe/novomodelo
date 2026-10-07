//! Parsing for `scenarios/inflow_ar_coefficients.parquet` — AR lag coefficients
//! per (hydro, stage, lag).
//!
//! [`parse_inflow_ar_coefficients`] reads `scenarios/inflow_ar_coefficients.parquet`
//! and returns a sorted `Vec<InflowArCoefficientRow>`.
//!
//! ## Parquet schema (spec SS3.2)
//!
//! | Column               | Type   | Required | Description                                  |
//! | -------------------- | ------ | -------- | -------------------------------------------- |
//! | `hydro_id`           | INT32  | Yes      | Hydro plant ID                               |
//! | `stage_id`           | INT32  | Yes      | Stage ID                                     |
//! | `lag`                | INT32  | Yes      | Lag index (1-based)                          |
//! | `coefficient`        | DOUBLE | Yes      | AR coeff ψ*, standardized by the seasonal std sₘ (dimensionless) |
//!
//! The model's innovation scale is not part of the schema: it is derived at
//! load from the AR coefficients (see
//! [`crate::scenarios::populate_derived_residual_ratios`]). Columns beyond the
//! schema are ignored.
//!
//! ## Standardization basis (external fits)
//!
//! `coefficient` is the AR coefficient of the process normalized by the seasonal
//! **sample std** sₘ (the `std_m3s` column of `inflow_seasonal_stats.parquet`),
//! not by the innovation std σₘ. A model fitted outside Cobre must store
//! `coefficient = ψ · s_{m-ℓ}/s_m` against the same sₘ it reports in `std_m3s`;
//! runtime reconstructs original-unit ψ and σ from the stored sₘ
//! (`cobre-stochastic::par::precompute`), so an inconsistent sₘ silently
//! rescales the model. See the PAR(p) methodology, "Two planes".
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id, lag)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `lag` must be ≥ 1 (lags are 1-based per spec).
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3.
//! - `stage_id` existence in the stages registry — Layer 3.
//! - Lag contiguity (1, 2, …, p for each (hydro, stage)) — Layer 3/5.

use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// A single row from `scenarios/inflow_ar_coefficients.parquet`.
///
/// Each row defines one lag coefficient for the PAR(p) model of a
/// (hydro, stage) pair. Multiple rows with the same `(hydro_id, stage_id)`
/// cover lags 1 through p, where p (the AR order) is derived from the
/// count of rows in the group.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowArCoefficientRow;
/// use cobre_core::EntityId;
///
/// let row = InflowArCoefficientRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 0,
///     lag: 1,
///     coefficient: 0.45,
/// };
/// assert_eq!(row.lag, 1);
/// assert!((row.coefficient - 0.45).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowArCoefficientRow {
    /// Hydro plant this coefficient belongs to.
    pub hydro_id: EntityId,
    /// Stage (0-based index within `System::stages`) this coefficient applies to.
    pub stage_id: i32,
    /// Lag index, 1-based (ψ₁ = lag 1, ψ₂ = lag 2, …).
    pub lag: i32,
    /// AR coefficient `ψ*_lag`, standardized by seasonal std (dimensionless).
    pub coefficient: f64,
}

/// Parse `scenarios/inflow_ar_coefficients.parquet` and return a sorted row table.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `lag` < 1                                     | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_ar_coefficients;
/// use std::path::Path;
///
/// let rows = parse_inflow_ar_coefficients(Path::new("scenarios/inflow_ar_coefficients.parquet"))
///     .expect("valid AR coefficients file");
/// println!("loaded {} AR coefficient rows", rows.len());
/// ```
pub fn parse_inflow_ar_coefficients(path: &Path) -> Result<Vec<InflowArCoefficientRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<InflowArCoefficientRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let lag_col = extract_required_int32(&batch, "lag", path)?;
        let coefficient_col = extract_required_float64(&batch, "coefficient", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let lag = lag_col.value(i);
            let coefficient = coefficient_col.value(i);

            if lag < 1 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_ar_coefficients[{row_idx}].lag"),
                    message: format!("lag must be >= 1 (1-based), got {lag}"),
                });
            }

            rows.push(InflowArCoefficientRow {
                hydro_id,
                stage_id,
                lag,
                coefficient,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
            .then_with(|| a.lag.cmp(&b.lag))
    });

    Ok(rows)
}

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

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
        ]))
    }

    fn schema_with_extra_column() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("extra", DataType::Float64, false),
        ]))
    }

    fn make_batch(
        hydro_ids: &[i32],
        stage_ids: &[i32],
        lags: &[i32],
        coefficients: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(lags.to_vec())),
                Arc::new(Float64Array::from(coefficients.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    fn make_batch_with_extra(
        hydro_ids: &[i32],
        stage_ids: &[i32],
        lags: &[i32],
        coefficients: &[f64],
        extras: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema_with_extra_column(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(lags.to_vec())),
                Arc::new(Float64Array::from(coefficients.to_vec())),
                Arc::new(Float64Array::from(extras.to_vec())),
            ],
        )
        .expect("valid batch with extra column")
    }

    #[test]
    fn test_valid_6_rows_sorted_by_hydro_stage_lag() {
        let batch = make_batch(
            &[2, 2, 2, 1, 1, 1],
            &[0, 0, 0, 0, 0, 0],
            &[3, 2, 1, 2, 3, 1],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();

        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].lag, 1);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].lag, 2);
        assert_eq!(rows[2].hydro_id, EntityId::from(1));
        assert_eq!(rows[2].lag, 3);
        assert_eq!(rows[3].hydro_id, EntityId::from(2));
        assert_eq!(rows[3].lag, 1);
        assert_eq!(rows[4].hydro_id, EntityId::from(2));
        assert_eq!(rows[4].lag, 2);
        assert_eq!(rows[5].hydro_id, EntityId::from(2));
        assert_eq!(rows[5].lag, 3);
    }

    #[test]
    fn test_lag_zero_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[0], &[0.45]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("lag"),
                    "field should contain 'lag', got: {field}"
                );
                assert!(
                    message.contains('1'),
                    "message should mention >= 1, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_missing_coefficient_column() {
        let schema_no_coeff = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_coeff,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Int32Array::from(vec![1_i32])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("coefficient"),
                    "field should contain 'coefficient', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_coefficient_values_preserved() {
        let batch = make_batch(&[42], &[3], &[1], &[0.12345]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(42));
        assert_eq!(row.stage_id, 3);
        assert_eq!(row.lag, 1);
        assert!((row.coefficient - 0.12345).abs() < 1e-10);
    }

    #[test]
    fn extra_columns_are_ignored() {
        let batch = make_batch_with_extra(
            &[1, 1, 2],
            &[0, 0, 0],
            &[1, 2, 1],
            &[0.5, 0.2, 0.3],
            &[0.85, 0.85, 0.9],
        );
        let tmp = write_parquet(&batch);

        let rows =
            parse_inflow_ar_coefficients(tmp.path()).expect("a file with extra columns must parse");
        assert_eq!(rows.len(), 3, "all rows must still parse");
    }
}
