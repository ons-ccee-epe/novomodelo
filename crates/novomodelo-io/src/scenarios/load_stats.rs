//! Parsing for `scenarios/load_seasonal_stats.parquet` (spec SS3.3) —
//! per-bus-per-stage mean and standard deviation of load demand.
//!
//! ## Parquet schema
//!
//! | Column     | Type   | Required | Description                          |
//! | ---------- | ------ | -------- | ------------------------------------ |
//! | `bus_id`   | INT32  | Yes      | Bus ID                               |
//! | `stage_id` | INT32  | Yes      | Stage ID                             |
//! | `mean_mw`  | DOUBLE | Yes      | Mean load demand (MW)                |
//! | `std_mw`   | DOUBLE | Yes      | Standard deviation (MW), 0 = deterministic |
//!
//! Entity-ID existence is deferred to Layer 3 referential validation.

use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// A single row from `scenarios/load_seasonal_stats.parquet`.
///
/// Assembled by [`crate::scenarios::assembly::assemble_load_models`] into
/// [`cobre_core::scenario::LoadModel`] entries.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::LoadSeasonalStatsRow;
/// use cobre_core::EntityId;
///
/// let row = LoadSeasonalStatsRow {
///     bus_id: EntityId::from(1),
///     stage_id: 3,
///     mean_mw: 500.0,
///     std_mw: 50.0,
/// };
/// assert_eq!(row.bus_id, EntityId::from(1));
/// assert_eq!(row.stage_id, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LoadSeasonalStatsRow {
    /// Bus this load model belongs to.
    pub bus_id: EntityId,
    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,
    /// Seasonal mean load demand μ in MW. Must be finite.
    pub mean_mw: f64,
    /// Seasonal standard deviation σ in MW. Must be non-negative and finite.
    /// A value of 0.0 indicates deterministic load.
    pub std_mw: f64,
}

/// Parse `scenarios/load_seasonal_stats.parquet` and return rows sorted by
/// `(bus_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean_mw` is NaN or infinite                 | [`LoadError::SchemaError`] |
/// | `std_mw` is negative or not finite            | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_load_seasonal_stats;
/// use std::path::Path;
///
/// let rows = parse_load_seasonal_stats(Path::new("scenarios/load_seasonal_stats.parquet"))
///     .expect("valid load seasonal stats file");
/// println!("loaded {} load seasonal stats rows", rows.len());
/// ```
pub fn parse_load_seasonal_stats(path: &Path) -> Result<Vec<LoadSeasonalStatsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<LoadSeasonalStatsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let bus_id_col = extract_required_int32(&batch, "bus_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_mw_col = extract_required_float64(&batch, "mean_mw", path)?;
        let std_mw_col = extract_required_float64(&batch, "std_mw", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let bus_id = EntityId::from(bus_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean_mw = mean_mw_col.value(i);
            let std_mw = std_mw_col.value(i);

            if !mean_mw.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("load_seasonal_stats[{row_idx}].mean_mw"),
                    message: format!("value must be finite, got {mean_mw}"),
                });
            }

            if !std_mw.is_finite() || std_mw < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("load_seasonal_stats[{row_idx}].std_mw"),
                    message: format!("value must be non-negative and finite, got {std_mw}"),
                });
            }

            rows.push(LoadSeasonalStatsRow {
                bus_id,
                stage_id,
                mean_mw,
                std_mw,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.bus_id
            .0
            .cmp(&b.bus_id.0)
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

    // ── AC: valid file with 4 rows, verify sort order and field values ─────────

    #[test]
    fn test_valid_4_rows_sorted_by_bus_stage() {
        assert_stats_happy_path(
            parse_load_seasonal_stats,
            "bus_id",
            "mean_mw",
            "std_mw",
            |row| (row.bus_id.0, row.stage_id, row.mean_mw, row.std_mw),
        );
    }

    // ── AC: std_mw = 0.0 (deterministic) is accepted ─────────────────────────

    #[test]
    fn test_zero_std_mw_is_accepted() {
        let batch = make_stats_batch("bus_id", "mean_mw", "std_mw", &[1], &[0], &[500.0], &[0.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_load_seasonal_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].std_mw.abs() < f64::EPSILON);
    }

    // ── AC: std_mw negative -> SchemaError ───────────────────────────────────

    #[test]
    fn test_negative_std_mw() {
        assert_stats_negative_std(parse_load_seasonal_stats, "bus_id", "mean_mw", "std_mw");
    }

    // ── AC: mean_mw NaN -> SchemaError ───────────────────────────────────────

    #[test]
    fn test_nan_mean_mw() {
        assert_stats_nan_mean(parse_load_seasonal_stats, "bus_id", "mean_mw", "std_mw");
    }

    // ── AC: missing required column -> SchemaError ────────────────────────────

    #[test]
    fn test_missing_mean_mw_column() {
        assert_stats_missing_column(parse_load_seasonal_stats, "bus_id", "mean_mw", "std_mw");
    }

    // ── AC: empty file -> Ok(vec![]) ──────────────────────────────────────────

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        assert_stats_empty_file(parse_load_seasonal_stats, "bus_id", "mean_mw", "std_mw");
    }

    // ── AC: declaration-order invariance ─────────────────────────────────────

    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_stats_batch(
            "bus_id",
            "mean_mw",
            "std_mw",
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[100.0, 110.0, 200.0, 210.0],
            &[10.0, 11.0, 20.0, 21.0],
        );
        let batch_desc = make_stats_batch(
            "bus_id",
            "mean_mw",
            "std_mw",
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[210.0, 200.0, 110.0, 100.0],
            &[21.0, 20.0, 11.0, 10.0],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_load_seasonal_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_load_seasonal_stats(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc.iter().map(|r| (r.bus_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> =
            rows_desc.iter().map(|r| (r.bus_id.0, r.stage_id)).collect();
        assert_eq!(keys_asc, keys_desc);
    }
}
