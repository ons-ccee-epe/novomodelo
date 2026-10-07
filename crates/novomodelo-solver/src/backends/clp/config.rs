//! CLP simplex-algorithm selection and tuning profile.

use crate::DEFAULT_PROFILE_HEURISTIC_SENTINEL;

/// Simplex algorithm `solve` runs: dual (`Clp_dual`) or primal (`Clp_primal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClpAlgorithm {
    /// Dual simplex (`Clp_dual`).
    #[default]
    Dual,
    /// Primal simplex (`Clp_primal`).
    Primal,
}

/// CLP-specific solver profile carrying the tunable option surface.
///
/// The field defaults are tuned for deterministic, warm-started repeated
/// re-solves: perturbation off, scaling off (the cobre prescaler conditions the
/// matrix), feasibility tolerances matching `HighsProfile` bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClpProfile {
    /// CLP perturbation mode (`Clp_setPerturbation`). `102` disables
    /// perturbation (required for deterministic re-solves); CLP's own default
    /// `100` requests automatic perturbation.
    pub perturbation: i32,
    /// CLP scaling mode (`Clp_scaling`). `0` disables scaling.
    pub scaling: i32,
    /// Primal feasibility tolerance (`Clp_setPrimalTolerance`).
    pub primal_feasibility_tolerance: f64,
    /// Dual feasibility tolerance (`Clp_setDualTolerance`).
    pub dual_feasibility_tolerance: f64,
    /// Per-attempt simplex iteration cap (`Clp_setMaximumIterations`).
    /// `DEFAULT_PROFILE_HEURISTIC_SENTINEL` (0) selects the per-call heuristic.
    pub simplex_iteration_limit: u32,
    /// Simplex algorithm `solve` dispatches on.
    pub algorithm: ClpAlgorithm,
    /// Dual-simplex row-pricing mode (drives `cobre_clp_set_dual_row_steepest`).
    /// `1` pins full dual steepest-edge pricing; the default `3` is CLP's own
    /// steepest-edge constructor default and is the "issue no shim call"
    /// sentinel, keeping the default profile byte-identical to a build that
    /// never set pricing.
    pub dual_pricing_mode: i32,
    /// Refactorization cadence (drives `cobre_clp_set_factorization_frequency`).
    /// The sentinel `0` leaves CLP's internal default in place — do not
    /// override; any non-zero value sets the cadence through the shim.
    pub factorization_frequency: i32,
}

impl Default for ClpProfile {
    fn default() -> Self {
        Self {
            perturbation: 102,
            scaling: 0,
            primal_feasibility_tolerance: 1e-9,
            dual_feasibility_tolerance: 1e-9,
            simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
            algorithm: ClpAlgorithm::Dual,
            dual_pricing_mode: 3,
            factorization_frequency: 0,
        }
    }
}
