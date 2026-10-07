//! Parquet parser for `constraints/hydro_unit_group_bounds.parquet` — per-stage,
//! optionally per-block bound overrides for a hydro unit group.
//!
//! [`parse_hydro_unit_group_bounds`] reads a Parquet file carrying stage-varying
//! overrides for a hydro unit group's four declared bounds (`min_turbined_m3s`,
//! `max_turbined_m3s`, `min_generation_mw`, `max_generation_mw`). The group's
//! declared values on `system/hydros.json` remain the base; a row here overrides
//! one bound for one `(hydro_id, hydro_unit_group_id, stage_id)`, optionally
//! narrowed to a single block.
//!
//! ## Parquet schema
//!
//! | Column                | Type         | Required | Description                                        |
//! | ---------------------- | ------------ | -------- | --------------------------------------------------- |
//! | `hydro_id`             | INT32        | Yes      | Hydro plant ID                                      |
//! | `hydro_unit_group_id`  | INT32        | Yes      | Hydro unit group ID                                 |
//! | `stage_id`             | INT32        | Yes      | Stage ID                                            |
//! | `min_turbined_m3s`     | DOUBLE       | No       | Min turbined flow override (m3/s)                   |
//! | `max_turbined_m3s`     | DOUBLE       | No       | Max turbined flow override (m3/s)                   |
//! | `min_generation_mw`    | DOUBLE       | No       | Min generation override (MW)                        |
//! | `max_generation_mw`    | DOUBLE       | No       | Max generation override (MW)                        |
//! | `block_id`             | INT32 (null) | No       | 0-based block index within the stage (matches `Block::index`) selecting one block's override |
//!
//! ## Block eligibility
//!
//! | Column                | Block-eligible |
//! | ---------------------- | :------------: |
//! | `min_turbined_m3s`     | yes            |
//! | `max_turbined_m3s`     | yes            |
//! | `min_generation_mw`    | yes            |
//! | `max_generation_mw`    | yes            |
//!
//! All four are block-eligible; the `block_id` range and duplicate-row rules are
//! enforced elsewhere. `bounds.rs`'s `## Block eligibility` table governs the five
//! PLANT-level bound families only — it does not extend to this group family.
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, hydro_unit_group_id, stage_id, block_id)`
//! ascending (`None` before `Some(i)`), since a stage-wide row and a block row may
//! legitimately share `(hydro_id, hydro_unit_group_id, stage_id)`.
//!
//! ## Validation
//!
//! - Required columns (`hydro_id`, `hydro_unit_group_id`, `stage_id`) must be
//!   present with Int32 type.
//! - A bound column or `block_id` absent from the file schema yields `None` in
//!   every row for that column (sparse override, not a schema error).
//! - Any present (non-null) bound value must be finite — NaN and ±Inf are rejected.
//!
//! Deferred to later layers: entity-id existence in registries, and `block_id`
//! range and duplicate-row checks.

use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_float64, extract_optional_int32, extract_required_int32,
    open_record_batch_reader,
};

use super::bounds::{optional_f64, optional_i32, validate_optional_finite};

// ── Row type ─────────────────────────────────────────────────────────────────

/// A single row from `constraints/hydro_unit_group_bounds.parquet`.
///
/// Carries a stage-varying, optionally block-scoped override for one of a hydro
/// unit group's four declared bounds. All four bound columns are block-eligible.
/// Fields are `None` when the corresponding column is absent or null in the
/// Parquet file (sparse storage: absent means "use the group's declared value").
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::HydroUnitGroupBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = HydroUnitGroupBoundsRow {
///     hydro_id: EntityId::from(1),
///     hydro_unit_group_id: EntityId::from(2),
///     stage_id: 3,
///     min_turbined_m3s: Some(10.0),
///     max_turbined_m3s: None,
///     min_generation_mw: None,
///     max_generation_mw: Some(50.0),
///     block_id: None,
/// };
/// assert_eq!(row.hydro_unit_group_id, EntityId::from(2));
/// assert_eq!(row.max_generation_mw, Some(50.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HydroUnitGroupBoundsRow {
    /// Hydro plant ID.
    pub hydro_id: EntityId,
    /// Hydro unit group ID.
    pub hydro_unit_group_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Minimum turbined flow (m³/s).
    pub min_turbined_m3s: Option<f64>,
    /// Maximum turbined flow (m³/s).
    pub max_turbined_m3s: Option<f64>,
    /// Minimum generation (MW).
    pub min_generation_mw: Option<f64>,
    /// Maximum generation (MW).
    pub max_generation_mw: Option<f64>,
    /// `None` applies at the stage level; `Some(b)` applies to block `b` only.
    pub block_id: Option<i32>,
}

// ── Parser ───────────────────────────────────────────────────────────────────

/// Parse `constraints/hydro_unit_group_bounds.parquet`, returning rows sorted by
/// `(hydro_id, hydro_unit_group_id, stage_id, block_id)` ascending (`None` before
/// `Some(i)`).
///
/// The file may contain any subset of the four optional bound columns; columns
/// absent from the file schema produce `None` in all rows (the sparse-override
/// design, not a schema error).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |----------------------------------------------- |--------------------------- |
/// | File not found or permission denied            | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)       | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type          | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column   | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_hydro_unit_group_bounds;
/// use std::path::Path;
///
/// let rows =
///     parse_hydro_unit_group_bounds(Path::new("constraints/hydro_unit_group_bounds.parquet"))
///         .expect("valid hydro unit group bounds file");
/// println!("loaded {} hydro unit group bounds rows", rows.len());
/// ```
pub fn parse_hydro_unit_group_bounds(
    path: &Path,
) -> Result<Vec<HydroUnitGroupBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let group_id_col = extract_required_int32(&batch, "hydro_unit_group_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let min_turbined_col = extract_optional_float64(&batch, "min_turbined_m3s", path)?;
        let max_turbined_col = extract_optional_float64(&batch, "max_turbined_m3s", path)?;
        let min_gen_col = extract_optional_float64(&batch, "min_generation_mw", path)?;
        let max_gen_col = extract_optional_float64(&batch, "max_generation_mw", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let hydro_unit_group_id = EntityId::from(group_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_turbined_m3s = optional_f64(min_turbined_col, i);
            let max_turbined_m3s = optional_f64(max_turbined_col, i);
            let min_generation_mw = optional_f64(min_gen_col, i);
            let max_generation_mw = optional_f64(max_gen_col, i);
            let block_id = optional_i32(block_id_col, i);

            for (value, column) in [
                (min_turbined_m3s, "min_turbined_m3s"),
                (max_turbined_m3s, "max_turbined_m3s"),
                (min_generation_mw, "min_generation_mw"),
                (max_generation_mw, "max_generation_mw"),
            ] {
                validate_optional_finite(value, "hydro_unit_group_bounds", row_idx, column, path)?;
            }

            rows.push(HydroUnitGroupBoundsRow {
                hydro_id,
                hydro_unit_group_id,
                stage_id,
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| {
        (
            r.hydro_id.0,
            r.hydro_unit_group_id.0,
            r.stage_id,
            r.block_id,
        )
    });

    Ok(rows)
}

// ── Tests ────────────────────────────────────────────────────────────────────

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
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// AC: scrambled declaration order — hydro=1 declares group 5 before group 2
    /// at the same stage, and a `block_id = None` row precedes its
    /// `block_id = Some(3)` sibling for `(hydro=1, group=5, stage=2)` — sorts to
    /// `(hydro_id, hydro_unit_group_id, stage_id, block_id)` ascending.
    #[test]
    fn parse_hydro_unit_group_bounds_sorts_by_full_key() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("hydro_unit_group_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
            Field::new("max_turbined_m3s", DataType::Float64, true),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 0])),
                Arc::new(Int32Array::from(vec![5, 2, 5, 9])),
                Arc::new(Int32Array::from(vec![2, 2, 2, 5])),
                Arc::new(Float64Array::from(vec![
                    Some(10.0),
                    Some(20.0),
                    None,
                    Some(1.0),
                ])),
                Arc::new(Float64Array::from(vec![None, None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, Some(99.0), None])),
                Arc::new(Int32Array::from(vec![None, None, Some(3), None])),
            ],
        )
        .expect("valid batch");

        let tmp = write_parquet(&batch);
        let rows = parse_hydro_unit_group_bounds(tmp.path()).expect("valid file");

        assert_eq!(rows.len(), 4);

        assert_eq!(rows[0].hydro_id, EntityId::from(0));
        assert_eq!(rows[0].hydro_unit_group_id, EntityId::from(9));
        assert_eq!(rows[0].stage_id, 5);

        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].hydro_unit_group_id, EntityId::from(2));
        assert_eq!(rows[1].stage_id, 2);
        assert!(rows[1].block_id.is_none());

        assert_eq!(rows[2].hydro_id, EntityId::from(1));
        assert_eq!(rows[2].hydro_unit_group_id, EntityId::from(5));
        assert_eq!(rows[2].stage_id, 2);
        assert!(rows[2].block_id.is_none());

        assert_eq!(rows[3].hydro_id, EntityId::from(1));
        assert_eq!(rows[3].hydro_unit_group_id, EntityId::from(5));
        assert_eq!(rows[3].stage_id, 2);
        assert_eq!(rows[3].block_id, Some(3));
        assert_eq!(rows[3].max_generation_mw, Some(99.0));
    }

    /// AC: NaN in `max_generation_mw` of row index 1 -> `SchemaError` whose
    /// `field` is exactly `hydro_unit_group_bounds[1].max_generation_mw`.
    #[test]
    fn parse_hydro_unit_group_bounds_rejects_non_finite() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("hydro_unit_group_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Float64Array::from(vec![Some(10.0), Some(f64::NAN)])),
            ],
        )
        .expect("valid batch");
        let tmp = write_parquet(&batch);
        let err = parse_hydro_unit_group_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_unit_group_bounds[1].max_generation_mw");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: a file carrying only the three required columns yields four `None`
    /// bounds and `block_id: None` in every row — a sparse absent column, not a
    /// schema error.
    #[test]
    fn parse_hydro_unit_group_bounds_treats_absent_columns_as_none() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("hydro_unit_group_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(Int32Array::from(vec![0])),
            ],
        )
        .expect("valid batch");
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_unit_group_bounds(tmp.path()).expect("valid file");

        assert_eq!(rows.len(), 1);
        assert!(rows[0].min_turbined_m3s.is_none());
        assert!(rows[0].max_turbined_m3s.is_none());
        assert!(rows[0].min_generation_mw.is_none());
        assert!(rows[0].max_generation_mw.is_none());
        assert!(rows[0].block_id.is_none());
    }
}
