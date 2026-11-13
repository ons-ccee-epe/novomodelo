//! Pre-resolved penalty and bound containers for O(1) solver lookup.
//!
//! The three-tier cascade (global defaults → entity overrides → stage overrides)
//! is evaluated once during input loading; these containers hold the result.
//! Penalties resolve in exactly these three tiers; bounds add three overlays:
//! a per-block overlay ([`ResolvedBlockBounds`]) that wins over the stage cell,
//! a hydro unit group overlay ([`ResolvedHydroUnitGroupBounds`]), and a
//! stage-volume overlay of stage totals ([`ResolvedStageVolumeBounds`]).
//! Populated by `novomodelo-io`; never modified after construction.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `novomodelo_core::resolved::Symbol` module path resolve to the
//! same item regardless of which submodule owns it.

mod block_bounds;
mod bounds;
mod effective;
mod factors;
mod generic;
mod group_bounds;
mod penalties;
mod stage_volume_bounds;

pub use block_bounds::{
    BlockBoundsCountsSpec, ContractBlockOverride, HydroBlockOverride, LineBlockOverride,
    PumpingBlockOverride, ResolvedBlockBounds, ThermalBlockOverride,
};
pub use bounds::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds, ThermalStageBounds,
};
pub use effective::{effective_diversion_m3s, effective_pumping_m3s};
pub use factors::{ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors};
pub use generic::{GenericConstraintBoundEntry, ResolvedGenericConstraintBounds};
pub use group_bounds::{
    HydroUnitGroupBoundsCountsSpec, HydroUnitGroupOverride, ResolvedHydroUnitGroupBounds,
};
pub use penalties::{
    BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults, ResolvedPenalties,
};
pub use stage_volume_bounds::{
    HydroStageVolumeBounds, PumpingStageVolumeBounds, ResolvedStageVolumeBounds,
    StageVolumeBoundsCountsSpec,
};
