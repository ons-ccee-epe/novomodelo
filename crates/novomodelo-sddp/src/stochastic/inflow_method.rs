//! Inflow non-negativity treatment method for SDDP subproblems.
//!
//! [`InflowNonNegativityMethod`] is a flat enum of the strategies for handling
//! negative PAR(p) inflow realisations, dispatched via `match` when constructing
//! LP templates and extracting simulation results.

use cobre_io::config::InflowNonNegativityConfig;
use cobre_io::config::InflowNonNegativityMethod::Penalty;
use cobre_io::config::InflowNonNegativityMethod::Truncation;
use cobre_io::config::InflowNonNegativityMethod::TruncationWithPenalty;
/// Inflow non-negativity treatment method.
///
/// The variant must be the same across all stages (set once at solver
/// initialisation from the loaded case config).
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::InflowNonNegativityMethod;
///
/// let penalty = InflowNonNegativityMethod::Penalty;
/// assert!(penalty.has_slack_columns());
///
/// let none = InflowNonNegativityMethod::None;
/// assert!(!none.has_slack_columns());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InflowNonNegativityMethod {
    /// No enforcement: no slack columns; a sufficiently negative PAR(p)
    /// realisation may make the LP infeasible.
    None,

    /// Clamps negative PAR(p) inflows to zero by adjusting the noise vector
    /// before LP patching — no slack columns, no objective perturbation.
    Truncation,

    /// Appends `N` slack columns (`sigma_inf_h >= 0`); each enters hydro `h`'s
    /// water-balance row with coefficient `tau_total * M3S_TO_HM3`
    /// (`tau_total` = total stage hours). Objective coefficient from
    /// `penalties.json → hydro.inflow_nonnegativity_cost` (default 1000.0).
    Penalty,

    /// Both `Truncation` and `Penalty`: the noise is clamped and slack columns
    /// let the solver undo part of the clamping if cost-effective. Matches
    /// `SPTcpp`'s `truncamento_penalizacao` mode.
    TruncationWithPenalty,
}

impl InflowNonNegativityMethod {
    /// Returns `true` when slack columns are appended to the LP.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::InflowNonNegativityMethod;
    ///
    /// assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    /// assert!(InflowNonNegativityMethod::Penalty.has_slack_columns());
    /// ```
    #[must_use]
    pub fn has_slack_columns(&self) -> bool {
        matches!(
            self,
            InflowNonNegativityMethod::Penalty | InflowNonNegativityMethod::TruncationWithPenalty
        )
    }
}

impl From<&InflowNonNegativityConfig> for InflowNonNegativityMethod {
    /// The config method is a typed enum, so typos are rejected at parse time
    /// before this total conversion runs.
    fn from(cfg: &InflowNonNegativityConfig) -> Self {
        match cfg.method {
            cobre_io::config::InflowNonNegativityMethod::None => InflowNonNegativityMethod::None,
            Truncation => InflowNonNegativityMethod::Truncation,
            Penalty => InflowNonNegativityMethod::Penalty,
            TruncationWithPenalty => InflowNonNegativityMethod::TruncationWithPenalty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InflowNonNegativityMethod;
    use cobre_io::config::{InflowNonNegativityConfig, InflowNonNegativityMethod as CfgMethod};

    // ── has_slack_columns ────────────────────────────────────────────────────

    #[test]
    fn none_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    }

    #[test]
    fn truncation_has_no_slack_columns() {
        assert!(!InflowNonNegativityMethod::Truncation.has_slack_columns());
    }

    #[test]
    fn penalty_has_slack_columns() {
        assert!(InflowNonNegativityMethod::Penalty.has_slack_columns());
    }

    // ── conversion from config ───────────────────────────────────────────────

    #[test]
    fn test_inflow_method_conversion_none() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::None,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::None
        );
    }

    #[test]
    fn test_inflow_method_conversion_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::Penalty,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Penalty
        );
    }

    #[test]
    fn test_inflow_method_conversion_truncation() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::Truncation,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::Truncation
        );
    }

    // ── TruncationWithPenalty ───────────────────────────────────────────────

    #[test]
    fn truncation_with_penalty_has_slack_columns() {
        assert!(InflowNonNegativityMethod::TruncationWithPenalty.has_slack_columns());
    }

    #[test]
    fn test_inflow_method_conversion_truncation_with_penalty() {
        let cfg = InflowNonNegativityConfig {
            method: CfgMethod::TruncationWithPenalty,
        };
        assert_eq!(
            InflowNonNegativityMethod::from(&cfg),
            InflowNonNegativityMethod::TruncationWithPenalty
        );
    }
}
