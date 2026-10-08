//! Shared LP-solve seam used by the training passes (`forward`, `backward`) and
//! by `simulation`.
//!
//! - [`solver_phase`] — the SDDP phase enum, the backend-agnostic profile trait,
//!   and the per-phase solver-profile constants.
//! - `stage_solve` — the single unified per-stage LP-solve entry point
//!   (`run_stage_solve`). Kept as ONE file: fragmenting the
//!   invariant-enforcement seam would let a driver bypass it.
//! - `partition` — the deterministic static work partition every driver uses
//!   to split scenarios across workers.

pub(crate) mod partition;
pub mod solver_phase;
pub(crate) mod stage_solve;

pub(crate) use partition::partition;
pub use solver_phase::Phase;
#[cfg(feature = "highs")]
pub use solver_phase::{BACKWARD_PROFILE, FORWARD_PROFILE, SIMULATION_PROFILE};
