//! `HiGHS` LP solver backend implementing [`SolverInterface`](crate::SolverInterface).
//!
//! # Thread Safety
//!
//! [`HighsSolver`] is `Send` but not `Sync`. The underlying `HiGHS` handle is
//! exclusively owned; transferring ownership to a worker thread is safe.
//! Concurrent access from multiple threads is not permitted (`HiGHS`
//! Implementation SS6.3).
//!
//! # Configuration
//!
//! The constructor applies performance-tuned defaults (`HiGHS` Implementation
//! SS4.1). Per-run parameters (time limit, iteration limit) are not set here --
//! those are applied by the caller before each solve.

mod config;
mod interface;
mod retry;
mod solver;
#[cfg(test)]
mod tests;

pub use config::{HighsProfile, PresolveKind};
pub use solver::{HighsSolver, highs_version};
