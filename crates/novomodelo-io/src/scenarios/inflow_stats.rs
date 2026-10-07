//! Parsing for `scenarios/inflow_seasonal_stats.parquet` — PAR(p) seasonal
//! mean and standard deviation per (hydro, stage).
//!
//! [`parse_inflow_seasonal_stats`] reads `scenarios/inflow_seasonal_stats.parquet`
//! and returns a sorted `Vec<InflowSeasonalStatsRow>`.
//!
//! ## Parquet schema (spec SS3.1)
//!
//! | Column     | Type   | Required | Description                          |
//! | ---------- | ------ | -------- | ------------------------------------ |
//! | `hydro_id` | INT32  | Yes      | Hydro plant ID                       |
//! | `stage_id` | INT32  | Yes      | Stage ID                             |
//! | `mean_m3s` | DOUBLE | Yes      | Seasonal mean inflow (m³/s)          |
//! | `std_m3s`  | DOUBLE | Yes      | Seasonal standard deviation (m³/s)   |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `mean_m3s` must be finite (NaN and ±inf are rejected).
//! - `std_m3s` must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3.
//! - `stage_id` existence in the stages registry — Layer 3.
//! - Coverage: every (hydro, stage) with AR coefficients has a stats row — Layer 4/5.

use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// A single row from `scenarios/inflow_seasonal_stats.parquet`.
///
/// Carries the PAR(p) seasonal statistics for a (hydro, stage) pair loaded
/// from the `inflow_seasonal_stats.parquet` file. These rows are later joined
/// with [`InflowArCoefficientRow`](super::InflowArCoefficientRow) by
/// [`super::assemble_inflow_models`] to produce [`cobre_core::scenario::InflowModel`] entries.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowSeasonalStatsRow;
/// use cobre_core::EntityId;
///
/// let row = InflowSeasonalStatsRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 3,
///     mean_m3s: 150.0,
///     std_m3s: 30.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(1));
/// assert_eq!(row.stage_id, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowSeasonalStatsRow {
    /// Hydro plant this model belongs to.
    pub hydro_id: EntityId,
    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,
    /// Seasonal mean inflow μ in m³/s. Must be finite.
    pub mean_m3s: f64,
    /// Seasonal sample standard deviation `s_m` in m³/s. Must be non-negative and finite.
    pub std_m3s: f64,
}

/// Parse `scenarios/inflow_seasonal_stats.parquet` and return a sorted row table.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean_m3s` is NaN or infinite                 | [`LoadError::SchemaError`] |
/// | `std_m3s` is negative or not finite           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_seasonal_stats;
/// use std::path::Path;
///
/// let rows = parse_inflow_seasonal_stats(Path::new("scenarios/inflow_seasonal_stats.parquet"))
///     .expect("valid inflow seasonal stats file");
/// println!("loaded {} inflow seasonal stats rows", rows.len());
/// ```
pub fn parse_inflow_seasonal_stats(path: &Path) -> Result<Vec<InflowSeasonalStatsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<InflowSeasonalStatsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_m3s_col = extract_required_float64(&batch, "mean_m3s", path)?;
        let std_m3s_col = extract_required_float64(&batch, "std_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean_m3s = mean_m3s_col.value(i);
            let std_m3s = std_m3s_col.value(i);

            if !mean_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_seasonal_stats[{row_idx}].mean_m3s"),
                    message: format!("value must be finite, got {mean_m3s}"),
                });
            }

            if !std_m3s.is_finite() || std_m3s < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_seasonal_stats[{row_idx}].std_m3s"),
                    message: format!("value must be non-negative and finite, got {std_m3s}"),
                });
            }

            rows.push(InflowSeasonalStatsRow {
                hydro_id,
                stage_id,
                mean_m3s,
                std_m3s,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
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
    use crate::test_support::{
        assert_stats_empty_file, assert_stats_happy_path, assert_stats_missing_column,
        assert_stats_nan_mean, assert_stats_negative_std, make_stats_batch, write_parquet,
    };

    #[test]
    fn test_valid_4_rows_sorted_by_hydro_stage() {
        assert_stats_happy_path(
            parse_inflow_seasonal_stats,
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            |row| (row.hydro_id.0, row.stage_id, row.mean_m3s, row.std_m3s),
        );
    }

    #[test]
    fn test_missing_mean_m3s_column() {
        assert_stats_missing_column(
            parse_inflow_seasonal_stats,
            "hydro_id",
            "mean_m3s",
            "std_m3s",
        );
    }

    #[test]
    fn test_negative_std_m3s() {
        assert_stats_negative_std(
            parse_inflow_seasonal_stats,
            "hydro_id",
            "mean_m3s",
            "std_m3s",
        );
    }

    #[test]
    fn test_nan_mean_m3s() {
        assert_stats_nan_mean(
            parse_inflow_seasonal_stats,
            "hydro_id",
            "mean_m3s",
            "std_m3s",
        );
    }

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        assert_stats_empty_file(
            parse_inflow_seasonal_stats,
            "hydro_id",
            "mean_m3s",
            "std_m3s",
        );
    }

    #[test]
    fn test_zero_std_m3s_is_accepted() {
        let batch = make_stats_batch(
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            &[1],
            &[0],
            &[150.0],
            &[0.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_seasonal_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].std_m3s.abs() < f64::EPSILON);
    }

    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_stats_batch(
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[100.0, 110.0, 200.0, 210.0],
            &[10.0, 11.0, 20.0, 21.0],
        );
        let batch_desc = make_stats_batch(
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[210.0, 200.0, 110.0, 100.0],
            &[21.0, 20.0, 11.0, 10.0],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_inflow_seasonal_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_inflow_seasonal_stats(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }
}
