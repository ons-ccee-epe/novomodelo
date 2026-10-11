//! Inflow non-negativity treatment method for SDDP subproblems.
//!
//! [`InflowNonNegativityMethod`] is a flat enum of the strategies for handling
//! negative PAR(p) inflow realisations, dispatched via `match` when constructing
//! LP templates and extracting simulation results.

use novomodelo_io::config::InflowNonNegativityConfig;
use novomodelo_io::config::InflowNonNegativityMethod::Penalty;
use novomodelo_io::config::InflowNonNegativityMethod::Truncation;
use novomodelo_io::config::InflowNonNegativityMethod::TruncationWithPenalty;
/// Inflow non-negativity treatment method.
///
/// The variant must be the same across all stages (set once at solver
/// initialisation from the loaded case config).
///
/// # Examples
///
/// ```rust
/// use novomodelo_sddp::InflowNonNegativityMethod;
///
/// let penalty = InflowNonNegativityMethod::Penalty;
/// assert!(penalty.has_slack_columns());
///
/// let none = InflowNonNegativityMethod::None;
/// assert!(!none.has_slack_columns());
///
/// assert!(InflowNonNegativityMethod::Truncation.has_slack_columns());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InflowNonNegativityMethod {
    /// No enforcement: no slack columns; a sufficiently negative PAR(p)
    /// realisation may make the LP infeasible.
    None,

    /// The PAR value stays raw, and the hydro's inflow-slack column is fixed per
    /// solve at the truncation amount (`max(-r, 0)` for the raw PAR value `r`), at
    /// cost 0.
    Truncation,

    /// Appends `N` slack columns (`sigma_inf_h >= 0`); each enters hydro `h`'s
    /// water-balance row with coefficient `tau_total * M3S_TO_HM3`
    /// (`tau_total` = total stage hours). Objective coefficient from
    /// `penalties.json → hydro.inflow_nonnegativity_cost` (default 1000.0).
    Penalty,

    /// The PAR value stays raw, and the hydro's slack column, bounded per solve to
    /// `[0, max(-r, 0)]` for that solve's raw PAR value `r`, is priced at
    /// `inflow_nonnegativity_cost`.
    TruncationWithPenalty,
}

impl InflowNonNegativityMethod {
    /// Returns `true` when slack columns are appended to the LP.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use novomodelo_sddp::InflowNonNegativityMethod;
    ///
    /// assert!(!InflowNonNegativityMethod::None.has_slack_columns());
    /// assert!(InflowNonNegativityMethod::Penalty.has_slack_columns());
    /// assert!(InflowNonNegativityMethod::Truncation.has_slack_columns());
    /// ```
    #[must_use]
    pub fn has_slack_columns(&self) -> bool {
        self.truncates() || self.slack_is_priced()
    }

    #[must_use]
    pub(crate) fn truncates(self) -> bool {
        matches!(self, Self::Truncation | Self::TruncationWithPenalty)
    }

    #[must_use]
    pub(crate) fn slack_is_priced(self) -> bool {
        matches!(self, Self::Penalty | Self::TruncationWithPenalty)
    }
}

impl From<&InflowNonNegativityConfig> for InflowNonNegativityMethod {
    /// The config method is a typed enum, so typos are rejected at parse time
    /// before this total conversion runs.
    fn from(cfg: &InflowNonNegativityConfig) -> Self {
        match cfg.method {
            novomodelo_io::config::InflowNonNegativityMethod::None => {
                InflowNonNegativityMethod::None
            }
            Truncation => InflowNonNegativityMethod::Truncation,
            Penalty => InflowNonNegativityMethod::Penalty,
            TruncationWithPenalty => InflowNonNegativityMethod::TruncationWithPenalty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InflowNonNegativityMethod;
    use novomodelo_io::config::{
        InflowNonNegativityConfig, InflowNonNegativityMethod as CfgMethod,
    };

    #[test]
    fn predicates_follow_the_method_truth_table() {
        use InflowNonNegativityMethod::{None, Penalty, Truncation, TruncationWithPenalty};
        let table = [
            (None, false, false, false),
            (Truncation, true, false, true),
            (Penalty, false, true, true),
            (TruncationWithPenalty, true, true, true),
        ];
        for (method, truncates, priced, has_slack) in table {
            assert_eq!(method.truncates(), truncates, "{method:?} truncates");
            assert_eq!(
                method.slack_is_priced(),
                priced,
                "{method:?} slack_is_priced"
            );
            assert_eq!(
                method.has_slack_columns(),
                has_slack,
                "{method:?} has_slack_columns"
            );
        }
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
