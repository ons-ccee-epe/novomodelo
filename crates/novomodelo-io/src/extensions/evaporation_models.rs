//! Row type and reader for fitted evaporation-model coefficients.
//!
//! [`EvaporationModelRow`] describes one linearized evaporation model
//! `evaporation_outflow = intercept_m3s + volume_slope_m3s_per_hm3·v` fitted
//! around a reference volume `reference_volume_hm3` for a `(hydro, stage)` pair.
//! [`parse_evaporation_models`] reads a Parquet file written by
//! `crate::output::write_evaporation_models` back into a sorted
//! `Vec<EvaporationModelRow>`.
//!
//! ## Parquet schema
//!
//! | Column                     | Type    | Required | Description                                    |
//! | -------------------------- | ------- | -------- | ---------------------------------------------- |
//! | `hydro_id`                 | INT32   | Yes      | Hydro plant identifier                         |
//! | `stage_id`                 | INT32?  | No       | Stage (`null` = single coefficient all stages) |
//! | `intercept_m3s`            | DOUBLE  | Yes      | Constant evaporation-outflow term (m³/s)       |
//! | `volume_slope_m3s_per_hm3` | DOUBLE  | Yes      | Volume slope of the linearized model           |
//! | `reference_volume_hm3`     | DOUBLE  | Yes      | Reference volume the model is fitted around    |
//! | `source`                   | UTF8    | Yes      | Provenance tag of the reference volume         |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id)` ascending. Null `stage_id` sorts
//! before any non-null value.

use arrow::array::Array;
use novomodelo_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{ParquetColumns, open_parquet_input};

/// A single fitted evaporation-model coefficient row.
///
/// # Examples
///
/// ```
/// use novomodelo_io::extensions::EvaporationModelRow;
/// use novomodelo_core::EntityId;
///
/// let row = EvaporationModelRow {
///     hydro_id: EntityId::from(66),
///     stage_id: None,
///     intercept_m3s: 12.5,
///     volume_slope_m3s_per_hm3: 0.0031,
///     reference_volume_hm3: 14_500.0,
///     source: "default_midpoint".to_string(),
/// };
/// assert_eq!(row.hydro_id, EntityId::from(66));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EvaporationModelRow {
    /// Hydro plant this evaporation model belongs to.
    pub hydro_id: EntityId,
    /// Stage this model applies to. `None` means a single coefficient for all stages.
    pub stage_id: Option<i32>,
    /// Constant evaporation-outflow term of the linearized model (m³/s).
    pub intercept_m3s: f64,
    /// Volume slope of the linearized model ((m³/s)/hm³).
    pub volume_slope_m3s_per_hm3: f64,
    /// Reference volume the model is fitted around (hm³).
    pub reference_volume_hm3: f64,
    /// Provenance tag of the reference volume (e.g. `"user_supplied"` or `"default_midpoint"`).
    pub source: String,
}

static EVAPORATION_MODELS_COLUMNS: ParquetColumns = ParquetColumns {
    known: &[
        "hydro_id",
        "intercept_m3s",
        "volume_slope_m3s_per_hm3",
        "reference_volume_hm3",
        "source",
        "stage_id",
    ],
    retired: &[],
};

/// Parse an evaporation-models Parquet file into a coefficient table sorted by
/// `(hydro_id, stage_id)` ascending (NULL `stage_id` before any non-null stage).
///
/// # Errors
///
/// | Condition                                | Error variant              |
/// |------------------------------------------|----------------------------|
/// | File not found or permission denied      | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.) | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type    | [`LoadError::SchemaError`] |
/// | Unknown or retired column                | [`LoadError::SchemaError`] |
/// | `source` column missing or wrong type    | [`LoadError::SchemaError`] |
pub fn parse_evaporation_models(path: &Path) -> Result<Vec<EvaporationModelRow>, LoadError> {
    let mut rows: Vec<EvaporationModelRow> = Vec::new();

    for batch in open_parquet_input(path, &EVAPORATION_MODELS_COLUMNS)? {
        let batch = batch?;

        let hydro_id_col = batch.required_int32("hydro_id")?;
        let intercept_col = batch.required_float64("intercept_m3s")?;
        let volume_slope_col = batch.required_float64("volume_slope_m3s_per_hm3")?;
        let reference_volume_col = batch.required_float64("reference_volume_hm3")?;
        let source_col = batch.required_string("source")?;

        let stage_id_col = batch.optional_int32("stage_id")?;

        let n = batch.num_rows();
        rows.reserve(n);

        for i in 0..n {
            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let intercept_m3s = intercept_col.value(i);
            let volume_slope_m3s_per_hm3 = volume_slope_col.value(i);
            let reference_volume_hm3 = reference_volume_col.value(i);

            // A NaN/±Inf in any coefficient is corrupt input, not a model.
            for (value, column) in [
                (intercept_m3s, "intercept_m3s"),
                (volume_slope_m3s_per_hm3, "volume_slope_m3s_per_hm3"),
                (reference_volume_hm3, "reference_volume_hm3"),
            ] {
                if !value.is_finite() {
                    return Err(LoadError::SchemaError {
                        path: path.to_path_buf(),
                        field: format!("evaporation_models[{i}].{column}"),
                        message: format!("value must be finite, got {value}"),
                    });
                }
            }

            let source = source_col.value(i).to_string();

            let stage_id = stage_id_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            rows.push(EvaporationModelRow {
                hydro_id,
                stage_id,
                intercept_m3s,
                volume_slope_m3s_per_hm3,
                reference_volume_hm3,
                source,
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
