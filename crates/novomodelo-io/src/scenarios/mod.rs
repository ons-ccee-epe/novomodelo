//! Parsers for scenario data files in the `scenarios/` subdirectory.
//!
//! Scenario files carry the stochastic inputs used by multi-stage optimization
//! solvers. All files are optional; when absent the `load_*` wrapper returns an
//! empty `Vec` (or default type).
//!
//! Cross-reference validation (entity-ID existence) and dimensional/semantic
//! constraints (lag contiguity, AR coefficient count, block count matching) are
//! deferred to Layer 3/5.

pub mod annual_component;
pub mod ar_coefficients;
pub mod assembly;
pub mod correlation;
pub mod estimation;
pub mod external;
pub mod inflow_history;
pub mod inflow_stats;
pub mod load_factors;
pub mod load_stats;
pub mod noise_openings;
pub mod non_controllable_factors;
pub mod non_controllable_stats;
pub mod residual_derivation;

pub use annual_component::{InflowAnnualComponentRow, parse_inflow_annual_component};
pub use ar_coefficients::{InflowArCoefficientRow, parse_inflow_ar_coefficients};
pub use assembly::{assemble_inflow_models, assemble_load_models};
pub use correlation::parse_correlation;
pub use estimation::{
    EstimationError, EstimationPath, estimate_from_history, resolve_model_stage_seasons,
};
pub use external::{
    ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, parse_external_inflow_scenarios,
    parse_external_load_scenarios, parse_external_ncs_scenarios,
};
pub use inflow_history::{InflowHistoryRow, parse_inflow_history};
pub use inflow_stats::{InflowSeasonalStatsRow, parse_inflow_seasonal_stats};
pub use load_factors::{BlockFactor, LoadFactorEntry, parse_load_factors};
pub use load_stats::{LoadSeasonalStatsRow, parse_load_seasonal_stats};
pub use noise_openings::{
    NoiseOpeningRow, assemble_opening_tree, parse_noise_openings, validate_noise_openings,
};
pub use non_controllable_factors::{NcsFactorEntry, parse_non_controllable_factors};
pub use non_controllable_stats::parse_ncs_stats;
pub use residual_derivation::{populate_derived_residual_ratios, resolve_stage_seasons};

use cobre_core::scenario::{CorrelationModel, NcsModel};

use crate::LoadError;
use std::path::Path;

/// Load `scenarios/inflow_seasonal_stats.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<InflowSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_seasonal_stats(p),
    }
}

/// Load `scenarios/inflow_ar_coefficients.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_ar_coefficients(
    path: Option<&Path>,
) -> Result<Vec<InflowArCoefficientRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_ar_coefficients(p),
    }
}

/// Load `scenarios/inflow_annual_component.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_annual_component(
    path: Option<&Path>,
) -> Result<Vec<InflowAnnualComponentRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_annual_component(p),
    }
}

/// Load `scenarios/inflow_history.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_history(path: Option<&Path>) -> Result<Vec<InflowHistoryRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_history(p),
    }
}

/// Load `scenarios/load_seasonal_stats.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_load_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<LoadSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_seasonal_stats(p),
    }
}

/// Load `scenarios/load_factors.json`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_load_factors(path: Option<&Path>) -> Result<Vec<LoadFactorEntry>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_factors(p),
    }
}

/// Load `scenarios/non_controllable_factors.json`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_non_controllable_factors(
    path: Option<&Path>,
) -> Result<Vec<NcsFactorEntry>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_non_controllable_factors(p),
    }
}

/// Load `scenarios/non_controllable_stats.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_ncs_stats(path: Option<&Path>) -> Result<Vec<NcsModel>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_ncs_stats(p),
    }
}

/// Load `scenarios/correlation.json`, returning a default model when absent.
///
/// Unlike the Parquet loaders, this returns `Ok(CorrelationModel::default())` for
/// `None` rather than an empty collection, since the target type is a structured model.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_correlation(path: Option<&Path>) -> Result<CorrelationModel, LoadError> {
    match path {
        None => Ok(CorrelationModel::default()),
        Some(p) => parse_correlation(p),
    }
}

/// Load `scenarios/external_inflow_scenarios.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_external_inflow_scenarios(
    path: Option<&Path>,
) -> Result<Vec<ExternalScenarioRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_external_inflow_scenarios(p),
    }
}

/// Load `scenarios/external_load_scenarios.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_external_load_scenarios(
    path: Option<&Path>,
) -> Result<Vec<ExternalLoadRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_external_load_scenarios(p),
    }
}

/// Load `scenarios/external_ncs_scenarios.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_external_ncs_scenarios(path: Option<&Path>) -> Result<Vec<ExternalNcsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_external_ncs_scenarios(p),
    }
}

/// Load `scenarios/noise_openings.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_noise_openings(path: Option<&Path>) -> Result<Vec<NoiseOpeningRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_noise_openings(p),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
mod tests {
    use super::*;

    #[test]
    fn test_load_inflow_seasonal_stats_none_returns_empty() {
        let result = load_inflow_seasonal_stats(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_inflow_ar_coefficients_none_returns_empty() {
        let result = load_inflow_ar_coefficients(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_inflow_history_none_returns_empty() {
        let result = load_inflow_history(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_load_seasonal_stats_none_returns_empty() {
        let result = load_load_seasonal_stats(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_load_factors_none_returns_empty() {
        let result = load_load_factors(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_correlation_none_returns_default() {
        let result = load_correlation(None).unwrap();
        assert!(result.profiles.is_empty());
        assert!(result.schedule.is_empty());
    }

    #[test]
    fn test_load_external_inflow_scenarios_none_returns_empty() {
        let result = load_external_inflow_scenarios(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_external_load_scenarios_none_returns_empty() {
        let result = load_external_load_scenarios(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_external_ncs_scenarios_none_returns_empty() {
        let result = load_external_ncs_scenarios(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_noise_openings_none_returns_empty() {
        let result = load_noise_openings(None).unwrap();
        assert!(result.is_empty());
    }
}
