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
//! - [`cut_selection`] — [`CutSelectionStrategy`]: the `Level1` periodic
//!   selection pass and the `Dynamic` lazy-row method.
//! - [`dcs`] — Dynamic Cut Selection: scores all resident cuts per stage.
//! - [`cut_sync`] — [`CutSyncBuffers`]: MPI cut-synchronization scratch space.
//! - [`basis_reconstruct`] — warm-start basis reconstruction for the frozen
//!   hot path and the DCS path.
//! - `trial_states` — the per-slot rows of the state each scored cut was
//!   computed at, reached through [`CutPool`].
//! - `ownership` — the per-state record of the Level 1 owner, a cache of the
//!   ownership function, reached through [`CutPool`].

pub mod basis_reconstruct;
pub mod cut_selection;
pub mod cut_sync;
pub mod dcs;
pub mod fcf;
mod ownership;
pub mod pool;
pub mod row;
pub mod row_map;
#[cfg(test)]
pub(crate) mod selection_reference;
mod trial_states;
pub mod wire;

pub use cut_selection::CutSelectionStrategy;
pub use cut_sync::CutSyncBuffers;
pub use fcf::FutureCostFunction;
pub use pool::{CutPool, SlotOrigin, SlotRole, SparsityReport};
pub use row_map::CutRowMap;
pub use wire::CutWireHeader;
