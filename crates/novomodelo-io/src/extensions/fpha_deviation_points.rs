//! Row type and reader for per-sampled-point computed-FPHA fit deviations.
//!
//! [`FphaDeviationPointRow`] describes one `(V, Q)` grid point at spillage = 0
//! for a `(hydro, stage)` fit: the exact production-function value `fph_exact`,
//! the fitted min-envelope value `fpha_fitted`, their signed difference
//! `deviation`, and `relative` (the absolute deviation against the grid peak).
//! [`parse_fpha_deviation_points`] reads a Parquet file written by
//! `crate::output::write_fpha_deviation_points` back into a sorted
//! `Vec<FphaDeviationPointRow>`.
//!
//! ## Parquet schema
//!
//! | Column        | Type    | Required | Description                                    |
//! | ------------- | ------- | -------- | ---------------------------------------------- |
//! | `hydro_id`    | INT32   | Yes      | Hydro plant identifier                         |
//! | `stage_id`    | INT32?  | No       | Stage (`null` = single fit covering all stages)|
//! | `v`           | DOUBLE  | Yes      | Volume grid coordinate (hm³)                   |
//! | `q`           | DOUBLE  | Yes      | Turbined-flow grid coordinate (m³/s)           |
//! | `fph_exact`   | DOUBLE  | Yes      | Exact production-function value (MW)            |
//! | `fpha_fitted` | DOUBLE  | Yes      | Fitted min-envelope value (MW)                 |
//! | `deviation`   | DOUBLE  | Yes      | Signed residual `fpha_fitted − fph_exact` (MW) |
//! | `relative`    | DOUBLE  | Yes      | `|deviation|` relative to the grid peak        |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id)` ascending. Null `stage_id` sorts
//! before any non-null value. Within a `(hydro_id, stage_id)` block the grid
//! order is preserved as written (the canonical `(V, Q)` walk).

use arrow::array::Array;
use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_int32, extract_required_float64, extract_required_int32,
    open_record_batch_reader,
};

/// A single per-sampled-point computed-FPHA fit-deviation row at spillage = 0.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::FphaDeviationPointRow;
/// use cobre_core::EntityId;
///
/// let row = FphaDeviationPointRow {
///     hydro_id: EntityId::from(66),
///     stage_id: Some(3),
///     v: 14_500.0,
///     q: 1_200.0,
///     fph_exact: 980.5,
///     fpha_fitted: 985.0,
///     deviation: 4.5,
///     relative: 0.0046,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(66));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FphaDeviationPointRow {
    /// Hydro plant this deviation point belongs to.
    pub hydro_id: EntityId,
    /// Stage the covering fit applies to. `None` means a single fit for all stages.
    pub stage_id: Option<i32>,
    /// Volume grid coordinate (hm³).
    pub v: f64,
    /// Turbined-flow grid coordinate (m³/s).
    pub q: f64,
    /// Exact production-function value at `(v, q, 0.0)` (MW).
    pub fph_exact: f64,
    /// Fitted min-envelope value at `(v, q, 0.0)` (MW).
    pub fpha_fitted: f64,
    /// Signed residual `fpha_fitted − fph_exact` (MW).
    pub deviation: f64,
    /// `|deviation|` relative to the grid's peak exact generation (dimensionless).
    pub relative: f64,
}

/// Parse an FPHA-deviation-points Parquet file into a table sorted by `(hydro_id,
/// stage_id)` ascending (NULL `stage_id` first; within-block grid order preserved).
///
/// # Errors
///
/// | Condition                                | Error variant              |
/// |------------------------------------------|----------------------------|
/// | File not found or permission denied      | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.) | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type    | [`LoadError::SchemaError`] |
pub fn parse_fpha_deviation_points(path: &Path) -> Result<Vec<FphaDeviationPointRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<FphaDeviationPointRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let v_col = extract_required_float64(&batch, "v", path)?;
        let q_col = extract_required_float64(&batch, "q", path)?;
        let fph_exact_col = extract_required_float64(&batch, "fph_exact", path)?;
        let fpha_fitted_col = extract_required_float64(&batch, "fpha_fitted", path)?;
        let deviation_col = extract_required_float64(&batch, "deviation", path)?;
        let relative_col = extract_required_float64(&batch, "relative", path)?;

        let stage_id_col = extract_optional_int32(&batch, "stage_id", path)?;

        let n = batch.num_rows();
        rows.reserve(n);

        for i in 0..n {
            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let v = v_col.value(i);
            let q = q_col.value(i);
            let fph_exact = fph_exact_col.value(i);
            let fpha_fitted = fpha_fitted_col.value(i);
            let deviation = deviation_col.value(i);
            let relative = relative_col.value(i);

            let stage_id = stage_id_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            rows.push(FphaDeviationPointRow {
                hydro_id,
                stage_id,
                v,
                q,
                fph_exact,
                fpha_fitted,
                deviation,
                relative,
            });
        }
    }

    // Stable sort: preserves the within-block canonical grid order.
    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}
