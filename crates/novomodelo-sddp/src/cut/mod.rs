//! Cut management data structures for the SDDP Future Cost Function (FCF).
//!
//! This module provides the per-stage cut pool, the all-stages FCF container,
//! the wire format for MPI exchange, cut-row construction for the LP, and
//! supporting types used to store, query, and prune Benders cuts during the
//! SDDP training loop.
//!
//! ## Contents
//!
//! - [`pool`] — [`CutPool`]: pre-allocated per-stage cut storage with
//!   deterministic slot assignment and activity tracking.
//! - [`fcf`] — [`FutureCostFunction`]: all-stages container wrapping one
//!   [`CutPool`] per stage; the high-level API for the training loop.
//! - [`wire`] — [`CutWireHeader`] and serialization for the MPI cut-exchange
//!   wire format.
//! - [`row`] — cut-row construction; owns the cut-sign convention
//!   (`push_scaled_coefficient` negates the raw subgradient).
//! - [`row_map`] — [`CutRowMap`]: slot-to-LP-row mapping that preserves cut-pool
//!   slot identity for warm-start basis reconstruction.
//! - [`cut_selection`] — [`CutSelectionStrategy`]: Level-1 / LML1 / domination
//!   periodic cut-selection strategies.
//! - [`dcs`] — Dynamic Cut Selection: scores all resident cuts per stage.
//! - [`cut_sync`] — [`CutSyncBuffers`]: MPI cut-synchronization scratch space.
//! - [`basis_reconstruct`] — warm-start basis reconstruction for the frozen
//!   hot path and the DCS path.

pub mod basis_reconstruct;
pub mod cut_selection;
pub mod cut_sync;
pub mod dcs;
pub mod fcf;
pub mod pool;
pub mod row;
pub mod row_map;
pub mod wire;

pub use cut_selection::CutSelectionStrategy;
pub use cut_sync::CutSyncBuffers;
pub use fcf::FutureCostFunction;
pub use pool::{CutPool, SparsityReport};
pub use row_map::CutRowMap;
pub use wire::CutWireHeader;

/// Sentinel value stored in [`crate::cut_selection::CutMetadata::iteration_generated`]
/// for warm-start cuts loaded from a policy checkpoint.
///
/// Set to [`u64::MAX`] so that `WARM_START_ITERATION != current_iteration` is
/// always true for any valid training iteration, allowing cut selection
/// strategies to distinguish warm-start cuts from training-generated cuts
/// without special casing.
pub const WARM_START_ITERATION: u64 = u64::MAX;
