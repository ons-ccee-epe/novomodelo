//! Time series estimation configuration types for `config.json → estimation`.

use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};

use crate::LoadError;
use crate::config::HistoricalYearRange;

/// Order selection criterion for autoregressive model fitting.
///
/// Controls how the lag order is chosen when fitting a time series model.
/// Two variants are accepted:
///
/// - `"pacf"` — classical periodic Yule-Walker with PACF-based order
///   selection. Default.
/// - `"pacf_annual"` — extends `"pacf"` with an annual component (PAR(p)-A),
///   adding one extra coefficient ψ per (entity, season) that multiplies
///   the rolling 12-month average of past observations.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OrderSelectionMethod {
    /// Periodic Yule-Walker partial autocorrelation method (PACF).
    #[default]
    Pacf,
    /// Periodic Yule-Walker order selection augmented with an annual component.
    ///
    /// When selected, the estimation pipeline performs four steps beyond the
    /// classical `pacf` path:
    ///
    /// 1. **Extended Yule-Walker fitting** — the system is augmented with a
    ///    cross-correlation term between the current-season inflow and the
    ///    rolling 12-month average, yielding the annual coefficient ψ
    ///    alongside the classical AR coefficients.
    /// 2. **Annual-stats computation** — per-season sample mean μ^A and
    ///    population standard deviation σ^A of the rolling 12-month
    ///    average are computed for each hydro plant.
    /// 3. **Parquet emission** — the triple (ψ, μ^A, σ^A) is written to
    ///    `inflow_annual_component.parquet` in the output directory.
    /// 4. **Widened LP lag stride** — the noise-column layout in the LP is
    ///    extended to accommodate the annual term alongside the classical lags.
    PacfAnnual,
}

impl<'de> serde::Deserialize<'de> for OrderSelectionMethod {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "pacf" => Ok(Self::Pacf),
            "pacf_annual" => Ok(Self::PacfAnnual),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["pacf", "pacf_annual"],
            )),
        }
    }
}

/// Series the estimated spatial inflow correlation is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum CorrelationTarget {
    /// Residuals of the fitted PAR lags.
    Residuals,
    /// Standardized historical inflows.
    #[default]
    Inflows,
}

impl CorrelationTarget {
    /// The `config.json` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Residuals => "residuals",
            Self::Inflows => "inflows",
        }
    }
}

impl From<CorrelationTarget> for novomodelo_stochastic::par::fitting::CorrelationTarget {
    fn from(target: CorrelationTarget) -> Self {
        match target {
            CorrelationTarget::Residuals => Self::Residuals,
            CorrelationTarget::Inflows => Self::Inflows,
        }
    }
}

/// Time series estimation settings (`config.json → estimation`).
///
/// Controls automatic parameter estimation when historical inflow data is
/// provided without explicit model statistics or coefficients.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EstimationConfig {
    /// Maximum lag order considered during autoregressive model fitting.
    pub max_order: u32,

    /// Order selection criterion. Accepts `"pacf"` (classical PACF, default)
    /// or `"pacf_annual"` (PACF augmented with an annual component, PAR(p)-A).
    pub order_selection: OrderSelectionMethod,

    /// Minimum number of observations required per (entity, season) group
    /// to proceed with estimation. Groups below this threshold are skipped.
    pub min_observations_per_season: u32,

    /// Maximum allowed absolute magnitude for any AR coefficient.
    ///
    /// When set, any (entity, season) pair with `|coefficient| > threshold`
    /// is immediately reduced to order 0 before the contribution analysis
    /// runs. This acts as a fast-path safety net for the most extreme
    /// explosive models. Absent by default (disabled; contribution analysis
    /// is the primary guard).
    #[serde(default)]
    pub max_coefficient_magnitude: Option<f64>,

    /// Series the spatial inflow correlation is estimated from when no
    /// `scenarios/correlation.json` is supplied: `"inflows"` (default) or `"residuals"`.
    ///
    /// Evaluation option, expected to be removed once evaluated.
    pub correlation_target: CorrelationTarget,

    /// Years of `scenarios/inflow_history.parquet` the inflow model is estimated
    /// from, as an inclusive `{"from", "to"}` range of the years season
    /// occurrences start in. Absent: every year. A range reaching outside the
    /// file's years is refused. The historical scenario scheme and the initial
    /// conditions still read the whole file.
    pub history_years: Option<HistoricalYearRange>,
}

impl Default for EstimationConfig {
    fn default() -> Self {
        Self {
            max_order: 6,
            order_selection: OrderSelectionMethod::default(),
            min_observations_per_season: 30,
            max_coefficient_magnitude: None,
            correlation_target: CorrelationTarget::default(),
            history_years: None,
        }
    }
}

impl EstimationConfig {
    pub(crate) fn check_history_years(&self, config_path: &Path) -> Result<(), LoadError> {
        match self.history_years {
            Some(HistoricalYearRange { from, to }) if from > to => Err(LoadError::SchemaError {
                path: config_path.to_path_buf(),
                field: "estimation.history_years".to_string(),
                message: format!("range 'from' ({from}) must be <= 'to' ({to})"),
            }),
            _ => Ok(()),
        }
    }
}
