//! Modeling option types for `config.json → modeling`.

use serde::{Deserialize, Serialize};

/// Method string for inflow non-negativity enforcement.
///
/// Accepted values in `config.json → modeling.inflow_non_negativity.method`.
/// Under every method a hydro's incremental inflow keeps the value the inflow
/// model gives it, which may be negative, and the next stage's inflow lags carry
/// that value. The methods differ in the water that may be added to it in the
/// hydro's water balance:
///
/// - `"none"` — none; a negative inflow can make a stage infeasible.
/// - `"truncation"` — exactly `max(-inflow, 0)`, at no cost, so the water
///   balance receives `max(inflow, 0)`.
/// - `"penalty"` — any non-negative amount, priced at the hydro's
///   `inflow_nonnegativity_cost`.
/// - `"truncation_with_penalty"` — any amount from 0 to `max(-inflow, 0)`,
///   priced at the hydro's `inflow_nonnegativity_cost`.
///
/// Known limitation of `"truncation"` and `"truncation_with_penalty"`
/// (Larroyd et al. (2022), the paper's §2.5): the lower bound can exceed the
/// upper bound, and a `gap` stopping rule (admissible only under enumerated
/// forward selection) can then report convergence that has not occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum InflowNonNegativityMethod {
    /// No water is added; a negative incremental inflow can make a stage infeasible.
    None,
    /// Adds exactly `max(-inflow, 0)` to the hydro's water balance, at no cost.
    Truncation,
    /// Adds any non-negative amount of water to the hydro's water balance, priced
    /// at the hydro's `inflow_nonnegativity_cost`.
    #[default]
    Penalty,
    /// Adds at most `max(-inflow, 0)` to the hydro's water balance, priced at the
    /// hydro's `inflow_nonnegativity_cost`.
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
    /// Default: `"penalty"`. Under `"penalty"` and `"truncation_with_penalty"` the
    /// added water is priced at the hydro's `inflow_nonnegativity_cost`, whose
    /// study-wide default is `penalties.json → hydro.inflow_nonnegativity_cost`;
    /// the cost has no effect under `"none"` and `"truncation"`.
    pub method: InflowNonNegativityMethod,
}

impl Default for InflowNonNegativityConfig {
    fn default() -> Self {
        Self {
            method: InflowNonNegativityMethod::Penalty,
        }
    }
}
