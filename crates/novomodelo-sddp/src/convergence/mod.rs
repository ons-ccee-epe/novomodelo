//! Convergence monitoring for the SDDP training loop: bound tracking, stopping
//! rules, and risk-measure cut aggregation.

// Rationale: renaming the `convergence` submodule to avoid the inception lint
// would break the `cobre_sddp::convergence::convergence` re-export path.
#[allow(clippy::module_inception)]
pub mod convergence;
pub mod risk_measure;
pub mod stopping_rule;
