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

use arrow::array::{Array, StringArray};
use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_int32, extract_required_float64, extract_required_int32,
    open_record_batch_reader,
};

/// A single fitted evaporation-model coefficient row.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::EvaporationModelRow;
/// use cobre_core::EntityId;
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
/// | `source` column missing or wrong type    | [`LoadError::SchemaError`] |
pub fn parse_evaporation_models(path: &Path) -> Result<Vec<EvaporationModelRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<EvaporationModelRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let intercept_col = extract_required_float64(&batch, "intercept_m3s", path)?;
        let volume_slope_col = extract_required_float64(&batch, "volume_slope_m3s_per_hm3", path)?;
        let reference_volume_col = extract_required_float64(&batch, "reference_volume_hm3", path)?;
        let source_col = extract_required_string(&batch, "source", path)?;

        let stage_id_col = extract_optional_int32(&batch, "stage_id", path)?;

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

/// Local to this module because `source` is the only Utf8 column the extension
/// readers consume; the shared `parquet_helpers` cover only the numeric types.
fn extract_required_string<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a StringArray, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Utf8 is required",
                col.data_type()
            ),
        })
}
