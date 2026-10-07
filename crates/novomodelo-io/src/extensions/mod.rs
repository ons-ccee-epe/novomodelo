//! Parsers for extension data files in the `system/` subdirectory.
//!
//! Extension files carry tabular data that augments the core entity registries
//! but is stored separately because it is multi-row per entity (e.g., VHA
//! curves for hydro plants, FPHA hyperplanes).
//!
//! ## Parsing convention
//!
//! Parquet parsers follow the canonical pattern:
//!
//! 1. Obtain the batch reader from `open_record_batch_reader` and consume all record batches.
//! 2. Extract typed columns by name; return `SchemaError` for missing or wrong-type columns.
//! 3. Validate per-row constraints; return `SchemaError` on violation.
//! 4. Sort the output by the documented sort key and return.
//!
//! JSON parsers follow the four-step pipeline:
//!
//! 1. Read the file with `std::fs::read_to_string`.
//! 2. Deserialize with `serde_json::from_str` through intermediate raw types.
//! 3. Validate with a `validate_raw` function.
//! 4. Convert raw types to public types and sort the output.
//!
//! Cross-reference validation (checking that referenced entity IDs exist in
//! their registries) is deferred to Layer 3. Monotonicity and other
//! multi-row semantic constraints are deferred to Layer 5.
//!
//! ## `load_*` wrappers
//!
//! Each `load_*(path: Option<&Path>)` is the pipeline entry point for an optional
//! file: `None` returns the empty result without touching the filesystem; `Some`
//! delegates to the matching `parse_*` and propagates its [`LoadError`].

pub mod evaporation_models;
pub mod fpha_deviation_points;
pub mod fpha_hyperplanes;
pub mod hydro_energy_productivity;
pub mod hydro_geometry;
pub mod hydro_reference_volumes;
pub mod production_models;
pub mod scalar_parameters;
pub mod tailrace_curves;

pub use evaporation_models::{EvaporationModelRow, parse_evaporation_models};
pub use fpha_deviation_points::{FphaDeviationPointRow, parse_fpha_deviation_points};
pub use fpha_hyperplanes::{FphaHyperplaneRow, parse_fpha_hyperplanes};
pub use hydro_energy_productivity::{HydroEnergyProductivityRow, parse_hydro_energy_productivity};
pub use hydro_geometry::{HydroGeometryRow, parse_hydro_geometry};
pub use hydro_reference_volumes::{
    HydroReferenceVolumeFractions, build_hydro_reference_volumes_resolved,
};
pub use production_models::{
    FittingWindow, FphaColumnLayout, PlaneReductionConfig, ProductionModelConfig,
    ProductionModelFile, ReferenceVolume, SeasonConfig, SelectionMode, StageRange,
    parse_production_models,
};
pub use scalar_parameters::parse_scalar_parameters_json;
pub use tailrace_curves::{TailraceCurveRow, parse_tailrace_curves};

use crate::LoadError;
use std::path::Path;

/// Load `system/hydro_geometry.parquet`, or an empty `Vec` when `path` is `None`.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_hydro_geometry`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::load_hydro_geometry;
///
/// let rows = load_hydro_geometry(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_hydro_geometry(path: Option<&Path>) -> Result<Vec<HydroGeometryRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_hydro_geometry(p),
    }
}

/// Load `system/hydro_production_models.json`, or an empty
/// [`ProductionModelFile`] (no configs, no reduction) when `path` is `None`.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_production_models`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::load_production_models;
///
/// let file = load_production_models(None).expect("no file is fine");
/// assert!(file.configs.is_empty());
/// assert!(file.plane_reduction.is_none());
/// ```
pub fn load_production_models(path: Option<&Path>) -> Result<ProductionModelFile, LoadError> {
    match path {
        None => Ok(ProductionModelFile::default()),
        Some(p) => parse_production_models(p),
    }
}

/// Load `system/fpha_hyperplanes.parquet`, or an empty `Vec` when `path` is `None`.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_fpha_hyperplanes`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::load_fpha_hyperplanes;
///
/// let rows = load_fpha_hyperplanes(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_fpha_hyperplanes(path: Option<&Path>) -> Result<Vec<FphaHyperplaneRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_fpha_hyperplanes(p),
    }
}

/// Load `system/hydro_energy_productivity.parquet`, or an empty `Vec` when
/// `path` is `None`.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_hydro_energy_productivity`] when
/// `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::load_hydro_energy_productivity;
///
/// let rows = load_hydro_energy_productivity(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_hydro_energy_productivity(
    path: Option<&Path>,
) -> Result<Vec<HydroEnergyProductivityRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_hydro_energy_productivity(p),
    }
}

/// Load `system/tailrace_curves.parquet`, or an empty `Vec` when `path` is `None`.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_tailrace_curves`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::load_tailrace_curves;
///
/// let rows = load_tailrace_curves(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_tailrace_curves(path: Option<&Path>) -> Result<Vec<TailraceCurveRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_tailrace_curves(p),
    }
}

#[cfg(test)]
#[allow(clippy::doc_markdown, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_load_production_models_none_returns_empty() {
        let file = load_production_models(None).unwrap();
        assert!(
            file.configs.is_empty(),
            "expected empty configs for None path"
        );
        assert!(
            file.plane_reduction.is_none(),
            "expected no plane reduction for None path"
        );
    }

    #[test]
    fn test_load_fpha_hyperplanes_none_returns_empty() {
        let result = load_fpha_hyperplanes(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    #[test]
    fn test_load_hydro_geometry_none_returns_empty() {
        let result = load_hydro_geometry(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    #[test]
    fn test_load_hydro_energy_productivity_none_returns_empty() {
        let result = load_hydro_energy_productivity(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    #[test]
    fn test_load_tailrace_curves_none_returns_empty() {
        let result = load_tailrace_curves(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }
}
