//! Energy-conversion preprocessing: derives the per-`(hydro, stage)` scalars
//! `ρ_eq` (equivalent productivity, MW per m³/s), `V_ref` (reference reservoir
//! volume, hm³), `Q_ref` (reference turbined flow, m³/s), and `ρ_acum`
//! (accumulated cascade productivity, MW per m³/s).
//!
//! The output [`EnergyConversionSet`] is consumed by simulation extraction and
//! by the energy-balance constraints (natural inflow energy and stored energy).

mod builder;
mod productivity_override;
mod types;

pub use builder::build_energy_conversion_set;
pub use productivity_override::{
    HydroEnergyProductivityOverride, build_hydro_energy_productivity_override,
};
pub use types::{EnergyConversion, EnergyConversionError, EnergyConversionSet};
