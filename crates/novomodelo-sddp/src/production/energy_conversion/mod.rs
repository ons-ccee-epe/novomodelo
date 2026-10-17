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

/// Conversion factor from `hm³ · MW/(m³/s)` to `MWh`.
///
/// Unit cancellation: `hm³ × 10⁶ m³/hm³ ÷ 3600 s/h × MW/(m³/s) = MWh`.
pub const ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S: f64 = 1.0e6 / 3600.0;
