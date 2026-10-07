//! Parsing for `scenarios/non_controllable_stats.parquet` — per-NCS-per-stage
//! mean and standard deviation of the stochastic availability factor.
//!
//! ## Parquet schema
//!
//! | Column     | Type   | Required | Description                                            |
//! | ---------- | ------ | -------- | ------------------------------------------------------ |
//! | `ncs_id`   | INT32  | Yes      | Non-controllable source entity ID                      |
//! | `stage_id` | INT32  | Yes      | Stage ID                                               |
//! | `mean`     | DOUBLE | Yes      | Mean availability factor [0, 1]                        |
//! | `std`      | DOUBLE | Yes      | Std dev of availability factor (>= 0), 0 = deterministic |
//!
//! Entity-ID existence is deferred to Layer 3 referential validation.

use cobre_core::EntityId;
use cobre_core::scenario::NcsModel;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// Parse `scenarios/non_controllable_stats.parquet` and return rows sorted by
/// `(ncs_id, stage_id)` ascending.
///
/// The `mean` column's `[0, 1]` bound is the availability-factor domain
/// specific to this parser; inflow and load means carry no equivalent bound.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean` is NaN, infinite, or outside `[0, 1]` | [`LoadError::SchemaError`] |
/// | `std` is negative or not finite              | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_ncs_stats;
/// use std::path::Path;
///
/// let models = parse_ncs_stats(Path::new("scenarios/non_controllable_stats.parquet"))
///     .expect("valid NCS models file");
/// println!("loaded {} NCS model rows", models.len());
/// ```
pub fn parse_ncs_stats(path: &Path) -> Result<Vec<NcsModel>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<NcsModel> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let ncs_id_col = extract_required_int32(&batch, "ncs_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_col = extract_required_float64(&batch, "mean", path)?;
        let std_col = extract_required_float64(&batch, "std", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let ncs_id = EntityId::from(ncs_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean = mean_col.value(i);
            let std = std_col.value(i);

            if !mean.is_finite() || !(0.0..=1.0).contains(&mean) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("non_controllable_stats[{row_idx}].mean"),
                    message: format!("value must be finite and in [0, 1], got {mean}"),
                });
            }

            if !std.is_finite() || std < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("non_controllable_stats[{row_idx}].std"),
                    message: format!("value must be non-negative and finite, got {std}"),
                });
            }

            rows.push(NcsModel {
                ncs_id,
                stage_id,
                mean,
                std,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.ncs_id
            .0
            .cmp(&b.ncs_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
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
    use crate::test_support::{
        assert_stats_empty_file, assert_stats_happy_path, assert_stats_missing_column,
        assert_stats_nan_mean, assert_stats_negative_std, make_stats_batch, write_parquet,
    };

    // ── AC: valid file with 4 rows, verify sort order and field values ───────

    #[test]
    fn test_valid_4_rows_sorted_by_ncs_stage() {
        assert_stats_happy_path(parse_ncs_stats, "ncs_id", "mean", "std", |row| {
            (row.ncs_id.0, row.stage_id, row.mean, row.std)
        });
    }

    // ── AC: std = 0.0 (deterministic) is accepted ───────────────────────────

    #[test]
    fn test_zero_std_is_accepted() {
        let batch = make_stats_batch("ncs_id", "mean", "std", &[1], &[0], &[0.5], &[0.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_ncs_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].std.abs() < f64::EPSILON);
    }

    // ── AC: std negative -> SchemaError ─────────────────────────────────────

    #[test]
    fn test_negative_std() {
        assert_stats_negative_std(parse_ncs_stats, "ncs_id", "mean", "std");
    }

    // ── AC: mean NaN -> SchemaError ──────────────────────────────────────────

    #[test]
    fn test_nan_mean() {
        assert_stats_nan_mean(parse_ncs_stats, "ncs_id", "mean", "std");
    }

    // ── AC: mean > 1.0 -> SchemaError ───────────────────────────────────────

    #[test]
    fn test_mean_out_of_range() {
        let batch = make_stats_batch("ncs_id", "mean", "std", &[1], &[0], &[1.5], &[0.0]);
        let tmp = write_parquet(&batch);
        let err = parse_ncs_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("mean"),
                    "field should contain 'mean', got: {field}"
                );
                assert!(
                    message.contains("[0, 1]"),
                    "message should mention [0, 1] range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: missing required column -> SchemaError ──────────────────────────

    #[test]
    fn test_missing_mean_column() {
        assert_stats_missing_column(parse_ncs_stats, "ncs_id", "mean", "std");
    }

    // ── AC: empty file -> Ok(vec![]) ────────────────────────────────────────

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        assert_stats_empty_file(parse_ncs_stats, "ncs_id", "mean", "std");
    }

    // ── AC: declaration-order invariance ────────────────────────────────────

    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_stats_batch(
            "ncs_id",
            "mean",
            "std",
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[0.30, 0.35, 0.50, 0.55],
            &[0.03, 0.035, 0.05, 0.055],
        );
        let batch_desc = make_stats_batch(
            "ncs_id",
            "mean",
            "std",
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[0.55, 0.50, 0.35, 0.30],
            &[0.055, 0.05, 0.035, 0.03],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_ncs_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_ncs_stats(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc.iter().map(|r| (r.ncs_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> =
            rows_desc.iter().map(|r| (r.ncs_id.0, r.stage_id)).collect();
        assert_eq!(keys_asc, keys_desc);
    }
}
