//! Resolution functions that apply the penalty and bound cascades.
//!
//! Each takes parsed entity data plus sparse stage-varying override rows and
//! produces a fully pre-resolved [`cobre_core::resolved`] table for O(1) lookup.

pub mod bounds;
pub mod factors;
pub mod generic_bounds;
pub mod group_bounds;
pub mod ncs_bounds;
pub mod penalties;

pub use bounds::{BoundsEntitySlices, BoundsOverrides, resolve_bounds};
pub use factors::{resolve_load_factors, resolve_ncs_factors};
pub use generic_bounds::resolve_generic_constraint_bounds;
pub use group_bounds::resolve_hydro_unit_group_bounds;
pub use ncs_bounds::resolve_ncs_bounds;
pub use penalties::{PenaltiesEntitySlices, PenaltiesOverrides, resolve_penalties};
