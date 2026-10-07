//! Pre-resolved penalty and bound containers for O(1) solver lookup.
//!
//! The three-tier cascade (global defaults → entity overrides → stage overrides)
//! is evaluated once during input loading; these containers hold the result.
//! Penalties resolve in exactly these three tiers; bounds add a per-block
//! overlay ([`ResolvedBlockBounds`]) that wins over the stage cell. Populated
//! by `cobre-io`; never modified after construction.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_core::resolved::Symbol` module path resolve to the
//! same item regardless of which submodule owns it.

mod block_bounds;
mod bounds;
mod factors;
mod generic;
mod group_bounds;
mod penalties;

pub use block_bounds::{
    BlockBoundsCountsSpec, ContractBlockOverride, HydroBlockOverride, LineBlockOverride,
    PumpingBlockOverride, ResolvedBlockBounds, ThermalBlockOverride,
};
pub use bounds::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds, ThermalStageBounds,
};
pub use factors::{ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors};
pub use generic::{GenericConstraintBoundEntry, ResolvedGenericConstraintBounds};
pub use group_bounds::{
    HydroUnitGroupBoundsCountsSpec, HydroUnitGroupOverride, ResolvedHydroUnitGroupBounds,
};
pub use penalties::{
    BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults, ResolvedPenalties,
};
