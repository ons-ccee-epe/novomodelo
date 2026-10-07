//! Parser for `system/hydro_energy_productivity.parquet` — per-plant per-stage
//! override values used by the energy-conversion preprocessing layer.
//!
//! ## Applicability
//!
//! The `equivalent_productivity_mw_per_m3s` column applies to **all** hydro
//! generation models. For FPHA hydros it overrides the `ρ_eq` value otherwise
//! derived from VHA geometry and `ρ_esp`. For non-FPHA hydros
//! (`constant_productivity`, `linearized_head`) it supplies `ρ_eq` directly
//! when `productivity_mw_per_m3s` is omitted from
//! `system/hydro_production_models.json`. Load-time validation enforces that
//! exactly one source supplies the value for each non-FPHA `(hydro, stage)`
//! pair — see [`crate::validation::productivity_resolution`].
//!
//! The `specific_productivity_mw_per_m3s_per_m` (`ρ_esp`) override feeds the
//! head-derived productivity of both the reference-point and mean
//! energy-conversion evaluators, and the `specific_productivity` computed tag
//! — the same override-then-entity precedence in both. The
//! `reference_outflow_m3s` (`Q_ref`) override reaches only the
//! `reference_turbine` computed tag; the energy-conversion head evaluation
//! always reads `max_turbined_m3s`, never this column.
//!
//! The reference operating volume is declared in `hydro_production_models.json`
//! (`reference_volume`), the single source of truth, not here. A stale
//! `reference_volume_hm3` column is warned-and-ignored rather than erroring, so
//! an old parquet still loads while the inert column is surfaced.
//!
//! ## Parquet schema
//!
//! | Column                                    | Parquet type | Nullable | Description                                     |
//! |-------------------------------------------|--------------|----------|-------------------------------------------------|
//! | `hydro_id`                                | INT32        | no       | Hydro plant identifier                          |
//! | `stage_id`                                | INT32        | yes      | Stage; NULL means "applies to all stages"       |
//! | `equivalent_productivity_mw_per_m3s`      | DOUBLE       | yes      | Direct `ρ_eq` override; finite and `>= 0.0`     |
//! | `reference_outflow_m3s`                   | DOUBLE       | yes      | `Q_ref` override; finite and `>= 0.0`           |
//! | `specific_productivity_mw_per_m3s_per_m`  | DOUBLE       | yes      | `ρ_esp` override; finite and `>= 0.0`           |
//!
//! ## Validation
//!
//! Each override column, when set, must be finite and `>= 0.0`; `hydro_id` must
//! not be null. A `0.0` `equivalent_productivity_mw_per_m3s` is accepted as a
//! planned-outage marker — the LP treats `ρ_eq` as a multiplier, so zero
//! generation carries no division-by-zero hazard. An all-NULL override row is
//! accepted.
//!
//! Duplicate `(hydro_id, stage_id)` detection is performed at build time by
//! the consumer that assembles the loaded rows into the override table.

use std::path::Path;

use arrow::array::{Array, Float64Array};
use cobre_core::EntityId;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, open_record_batch_reader,
};

/// A single row of the `system/hydro_energy_productivity.parquet` override table.
///
/// An all-`None` override row is accepted as a duplicate-detection key.
#[derive(Debug, Clone, PartialEq)]
pub struct HydroEnergyProductivityRow {
    /// Hydro plant this override applies to.
    pub hydro_id: EntityId,
    /// Stage the override applies to. `None` is a per-hydro default for all stages.
    pub stage_id: Option<i32>,
    /// Direct `ρ_eq` override \[MW/(m³/s)\]. `0.0` is a planned-outage marker.
    pub equivalent_productivity_mw_per_m3s: Option<f64>,
    /// `Q_ref` override \[m³/s\].
    pub reference_outflow_m3s: Option<f64>,
    /// `ρ_esp` override \[MW/(m³/s)/m\].
    pub specific_productivity_mw_per_m3s_per_m: Option<f64>,
}

/// Parse `system/hydro_energy_productivity.parquet`, sorted by `(hydro_id,
/// stage_id)` with NULL `stage_id` (per-hydro default) before any concrete stage.
///
/// # Errors
///
/// Returns [`LoadError::IoError`] when the file cannot be opened,
/// [`LoadError::ParseError`] for malformed Parquet, and
/// [`LoadError::SchemaError`] for missing/wrong-typed columns, null `hydro_id`,
/// or out-of-range override values.
pub fn parse_hydro_energy_productivity(
    path: &Path,
) -> Result<Vec<HydroEnergyProductivityRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<HydroEnergyProductivityRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        warn_on_stale_reference_volume_column(&batch);

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let rho_eq_col =
            extract_required_float64(&batch, "equivalent_productivity_mw_per_m3s", path)?;
        let q_ref_col = extract_required_float64(&batch, "reference_outflow_m3s", path)?;
        let rho_esp_col =
            extract_required_float64(&batch, "specific_productivity_mw_per_m3s_per_m", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            if hydro_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_energy_productivity[{row_idx}].hydro_id"),
                    message: "value must not be null".to_string(),
                });
            }

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = if stage_id_col.is_null(i) {
                None
            } else {
                Some(stage_id_col.value(i))
            };

            let equivalent_productivity_mw_per_m3s = extract_nonnegative_override(
                rho_eq_col,
                i,
                row_idx,
                "equivalent_productivity_mw_per_m3s",
                path,
            )?;

            let reference_outflow_m3s =
                extract_nonnegative_override(q_ref_col, i, row_idx, "reference_outflow_m3s", path)?;

            let specific_productivity_mw_per_m3s_per_m = extract_nonnegative_override(
                rho_esp_col,
                i,
                row_idx,
                "specific_productivity_mw_per_m3s_per_m",
                path,
            )?;

            rows.push(HydroEnergyProductivityRow {
                hydro_id,
                stage_id,
                equivalent_productivity_mw_per_m3s,
                reference_outflow_m3s,
                specific_productivity_mw_per_m3s_per_m,
            });
        }
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id.unwrap_or(-1)));
    Ok(rows)
}

// ── stale-column deprecation notice ─────────────────────────────────────────────

/// Process-wide guard so the stale-column deprecation notice is emitted at most
/// once, no matter how many files or batches carry the column.
static STALE_REFERENCE_VOLUME_NOTICE: std::sync::Once = std::sync::Once::new();

fn warn_on_stale_reference_volume_column(batch: &arrow::record_batch::RecordBatch) {
    if batch
        .schema()
        .column_with_name("reference_volume_hm3")
        .is_some()
    {
        STALE_REFERENCE_VOLUME_NOTICE.call_once(|| {
            tracing::warn!(
                "reference_volume_hm3 in hydro_energy_productivity.parquet is no longer read; \
                 declare reference_volume in hydro_production_models.json instead"
            );
        });
    }
}

// ── per-value validation helpers ───────────────────────────────────────────────

fn extract_nonnegative_override(
    col: &Float64Array,
    i: usize,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<Option<f64>, LoadError> {
    if col.is_null(i) {
        Ok(None)
    } else {
        Ok(Some(validate_nonnegative(
            col.value(i),
            row_idx,
            column,
            path,
        )?))
    }
}

fn validate_nonnegative(
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
            field: format!("hydro_energy_productivity[{row_idx}].{column}"),
            message: format!("value must be finite and non-negative (>= 0.0), got {value}"),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::test_support::write_parquet;

    fn make_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]))
    }

    fn make_batch(
        hydro_ids: &[i32],
        stage_ids: &[Option<i32>],
        rho_eqs: &[Option<f64>],
        q_refs: &[Option<f64>],
        rho_esps: &[Option<f64>],
    ) -> RecordBatch {
        let schema = make_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(rho_eqs.to_vec())),
                Arc::new(Float64Array::from(q_refs.to_vec())),
                Arc::new(Float64Array::from(rho_esps.to_vec())),
            ],
        )
        .expect("valid batch construction")
    }

    /// Round-trip: three rows matching the acceptance criterion fixture.
    ///
    /// Fixture:
    /// - row 0: `(hydro=1, stage=0, rho_eq=3.6, Q_ref=NULL, rho_esp=NULL)`
    /// - row 1: `(hydro=1, stage=NULL, rho_eq=4.0, Q_ref=NULL, rho_esp=0.009)`
    /// - row 2: `(hydro=2, stage=NULL, rho_eq=5.0, Q_ref=200.0, rho_esp=NULL)`
    ///
    /// After sort the expected order is:
    /// `(hydro=1, NULL)` → `(hydro=1, stage=0)` → `(hydro=2, NULL)`.
    #[test]
    fn test_round_trip_three_rows() {
        // Write in non-sorted order to verify the parser sorts the output.
        let batch = make_batch(
            &[1, 1, 2],
            &[Some(0), None, None],
            &[Some(3.6), Some(4.0), Some(5.0)],
            &[None, None, Some(200.0)],
            &[None, Some(0.009), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3, "expected 3 rows");
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, None);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, Some(0));
        assert_eq!(rows[2].hydro_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, None);
    }

    /// `equivalent_productivity_mw_per_m3s = 0.0` is accepted as a planned-outage marker.
    #[test]
    fn test_zero_rho_eq_accepted() {
        let batch = make_batch(&[1], &[Some(0)], &[Some(0.0)], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path())
            .expect("zero ρ_eq must be accepted as a planned-outage marker");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].equivalent_productivity_mw_per_m3s, Some(0.0));
    }

    /// Negative `equivalent_productivity_mw_per_m3s` is still rejected.
    #[test]
    fn test_negative_rho_eq_rejected() {
        let batch = make_batch(&[1], &[Some(0)], &[Some(-0.1)], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("equivalent_productivity_mw_per_m3s"),
                    "field should name the column, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Builds a batch that still physically carries the retired
    /// `reference_volume_hm3` column (6 columns) to exercise the
    /// warn-and-ignore forward-compat path.
    fn make_stale_batch(
        hydro_ids: &[i32],
        stage_ids: &[Option<i32>],
        rho_eqs: &[Option<f64>],
        v_refs: &[Option<f64>],
        q_refs: &[Option<f64>],
        rho_esps: &[Option<f64>],
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_volume_hm3", DataType::Float64, true),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(rho_eqs.to_vec())),
                Arc::new(Float64Array::from(v_refs.to_vec())),
                Arc::new(Float64Array::from(q_refs.to_vec())),
                Arc::new(Float64Array::from(rho_esps.to_vec())),
            ],
        )
        .expect("valid stale batch construction")
    }

    /// A parquet that still physically carries a stale `reference_volume_hm3`
    /// column parses `Ok` (warn-and-ignore forward-compat), the column is
    /// ignored, and the other override values survive. The one-time
    /// deprecation notice goes to `tracing::warn!`; no tracing capture is wired
    /// in this crate's tests, so this asserts the tolerate-and-ignore behavior
    /// (the parser must NOT error and the row carries the kept overrides).
    #[test]
    fn parser_warns_and_ignores_stale_reference_volume_column() {
        let batch = make_stale_batch(
            &[1],
            &[Some(0)],
            &[Some(3.6)],
            // A populated stale column must not error and must be ignored.
            &[Some(120.0)],
            &[Some(200.0)],
            &[Some(0.009)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path())
            .expect("a stale reference_volume_hm3 column must be ignored, not rejected");

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(1));
        assert_eq!(row.stage_id, Some(0));
        assert_eq!(row.equivalent_productivity_mw_per_m3s, Some(3.6));
        assert_eq!(row.reference_outflow_m3s, Some(200.0));
        assert_eq!(row.specific_productivity_mw_per_m3s_per_m, Some(0.009));
    }

    /// `reference_outflow_m3s = NaN` must be rejected.
    #[test]
    fn test_nan_q_ref_rejected() {
        let batch = make_batch(&[1], &[None], &[None], &[Some(f64::NAN)], &[None]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// A row where all three override columns are NULL is accepted.
    #[test]
    fn test_all_overrides_null_accepted() {
        let batch = make_batch(&[1], &[Some(0)], &[None], &[None], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(1));
        assert_eq!(row.stage_id, Some(0));
        assert!(row.equivalent_productivity_mw_per_m3s.is_none());
        assert!(row.reference_outflow_m3s.is_none());
        assert!(row.specific_productivity_mw_per_m3s_per_m.is_none());
    }

    /// A null `hydro_id` must be rejected.
    #[test]
    fn test_null_hydro_id_rejected() {
        // Build a batch with a nullable hydro_id column that has a null value.
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, true), // nullable for this test
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
            ],
        )
        .expect("valid batch");
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let err = parse_hydro_energy_productivity(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should mention hydro_id, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `reference_outflow_m3s = 0.0` must be accepted (zero outflow is valid).
    #[test]
    fn test_zero_q_ref_accepted() {
        let batch = make_batch(&[1], &[None], &[None], &[Some(0.0)], &[None]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].reference_outflow_m3s, Some(0.0));
    }

    /// Duplicate `(hydro_id, stage_id)` keys produce two distinct rows — the
    /// parser does not detect duplicates (the builder test covers that).
    #[test]
    fn test_duplicate_keys_not_rejected_by_parser() {
        let batch = make_batch(
            &[1, 1],
            &[Some(0), Some(0)],
            &[Some(3.6), Some(4.0)],
            &[None, None],
            &[None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_sort_order_null_stage_before_concrete() {
        let batch = make_batch(
            &[2, 1, 1],
            &[None, Some(5), None],
            &[Some(1.0), Some(2.0), Some(3.0)],
            &[None, None, None],
            &[None, None, None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_energy_productivity(tmp.path()).unwrap();

        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, None);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, Some(5));
        assert_eq!(rows[2].hydro_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, None);
    }
}
