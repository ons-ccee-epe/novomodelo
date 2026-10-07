//! Production-modeling cluster for the SDDP solver.
//!
//! This directory groups the preprocessing that turns reservoir geometry and
//! plant data into the per-`(hydro, stage)` production representations the LP
//! consumes — FPHA hyperplane fitting, resolved production/evaporation models,
//! energy-conversion scalars, and the simulation write-payload conversions.

pub(crate) mod conversion;
pub mod energy_conversion;
pub(crate) mod fpha_fitting;
pub mod hydro_models;
