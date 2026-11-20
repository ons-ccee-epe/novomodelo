//! Modeling option types for `config.json → modeling`.

use serde::{Deserialize, Serialize};

/// Method string for inflow non-negativity enforcement.
///
/// Accepted values in `config.json → modeling.inflow_non_negativity.method`:
///
/// - `"none"` — no enforcement; the PAR(p) inflow may be negative.
/// - `"truncation"` — the PAR(p) value stays raw: it is the incremental inflow, and the
///   inflow lags of later stages are built from it (aggregated over the lag's period when
///   the period spans several stages). A per-hydro correction fixed at `max(−r, 0)`, at no
///   cost, lifts the local inflow to zero.
/// - `"penalty"` — an unbounded correction priced by
///   `penalties.json → hydro.inflow_nonnegativity_cost`.
/// - `"truncation_with_penalty"` — the PAR(p) value stays raw; a correction between 0 and
///   `max(−r, 0)`, priced by `penalties.json → hydro.inflow_nonnegativity_cost`.
///
/// Under the two truncation methods the correction depends on the incoming state, so the
/// lower bound can exceed the upper bound, and a `gap` stopping rule can then report
/// convergence that has not occurred (Larroyd et al. (2022), the paper's §2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum InflowNonNegativityMethod {
    /// No enforcement; the inflow may be negative.
    None,
    /// Raw PAR(p) value; correction fixed at the truncation amount, at no cost.
    Truncation,
    /// Unbounded correction priced by `penalties.json → hydro.inflow_nonnegativity_cost`.
    Penalty,
    /// Raw PAR(p) value; priced correction capped at the truncation amount.
    #[default]
    TruncationWithPenalty,
}

/// Modeling options (`config.json → modeling`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelingConfig {
    /// Strategy for handling non-negative inflow constraints.
    #[serde(default)]
    pub inflow_non_negativity: InflowNonNegativityConfig,

    /// Distribution of the inflow residual.
    #[serde(default)]
    pub inflow_residual: InflowResidualConfig,

    /// Divisor applied to every non-theta objective coefficient at template
    /// build time, multiplied back at every cost-domain reporting boundary.
    /// Default `1_000_000.0` — the value every golden parity baseline is pinned at.
    ///
    /// Objective conditioning only — results are identical in exact arithmetic;
    /// this does not alter the model, unlike `modeling`'s other fields. The
    /// effective dual tolerance in currency units is
    /// `dual_feasibility_tolerance × this factor`: raising the factor without
    /// lowering `dual_feasibility_tolerance` proportionally loosens optimality
    /// in currency terms even though the configured tolerance value is
    /// unchanged — the inverse-direction trap a name cannot carry.
    ///
    /// Absent uses the default. Must be finite and `> 0`; a value outside
    /// `[1.0, 1e12]` is accepted but logs an advisory warning.
    #[serde(default)]
    pub cost_scale_factor: Option<f64>,
}

/// Inflow non-negativity treatment settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InflowNonNegativityConfig {
    /// Method: `"none"`, `"truncation"`, `"penalty"`, or `"truncation_with_penalty"`.
    ///
    /// Default: `"truncation_with_penalty"`. Under `"penalty"` and
    /// `"truncation_with_penalty"` the correction's cost is
    /// `penalties.json → hydro.inflow_nonnegativity_cost` (default 1000.0 when absent);
    /// under `"truncation"` it has no cost.
    pub method: InflowNonNegativityMethod,
}

impl Default for InflowNonNegativityConfig {
    fn default() -> Self {
        Self {
            method: InflowNonNegativityMethod::TruncationWithPenalty,
        }
    }
}

/// Distribution of the PAR(p) inflow residual.
///
/// Accepted values in `config.json → modeling.inflow_residual.distribution`:
///
/// - `"normal"` — the inflow is the PAR(p) conditional mean plus `σ·η`.
/// - `"lognormal_3p"` — a three-parameter lognormal residual whose lower bound is zero
///   inflow. The inflow keeps the conditional mean and the standard deviation `σ`, and is
///   never negative. Where a solve's conditional mean is at or below zero, or `(σ/mean)²`
///   is not finite, that hydro takes the `"normal"` residual, and
///   `modeling.inflow_non_negativity.method` treats a negative inflow as it does under
///   `"normal"`.
///
/// `"lognormal_3p"` is accepted only with the `in_sample` and `out_of_sample` inflow
/// schemes; it is refused with `enumerated` selection, `historical_residuals` stages and
/// nodes that declare `scenario_id`.
///
/// Under `"lognormal_3p"` the residual depends on the incoming state, so the lower bound
/// can exceed the upper bound, and a `gap` stopping rule can then report convergence that
/// has not occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum InflowResidualDistribution {
    /// PAR(p) conditional mean plus `σ·η`.
    #[default]
    Normal,
    /// Three-parameter lognormal residual bounded below at zero inflow; the normal residual
    /// where the conditional mean is not positive.
    #[serde(rename = "lognormal_3p")]
    Lognormal3p,
}

impl InflowResidualDistribution {
    /// The `config.json` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lognormal3p => "lognormal_3p",
        }
    }
}

/// Inflow residual settings (`config.json → modeling.inflow_residual`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InflowResidualConfig {
    /// Residual distribution: `"normal"` or `"lognormal_3p"`.
    ///
    /// Default: `"normal"`. Combines with every `modeling.inflow_non_negativity.method`. Two
    /// warnings show only in `novomodelo validate` and Python `novomodelo.io.validate`, not in
    /// `novomodelo run` or Python `run`: seasons whose fitted mean inflow is at or below
    /// zero, and `"lognormal_3p"` with `modeling.inflow_non_negativity.method = "none"`.
    pub distribution: InflowResidualDistribution,
}
