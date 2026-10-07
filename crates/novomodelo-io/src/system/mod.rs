//! Entity registry parsers for the `system/` subdirectory.
//!
//! Each sub-module implements a `parse_*` function that reads one entity registry
//! file from the case `system/` directory, validates it, and returns a sorted
//! `Vec` of core entity types.
//!
//! Cross-reference validation (e.g., checking that `bus_id` exists in the bus
//! registry) is deferred to Layer 3. Only schema-level invariants
//! are checked here.
//!
//! ## Optional files
//!
//! Three entity registries are optional: `non_controllable_sources.json`,
//! `pumping_stations.json`, and `energy_contracts.json`. When the file is absent
//! from the case directory the corresponding `load_*` wrapper (which accepts
//! `Option<&Path>`) returns `Ok(Vec::new())` without error.

pub mod buses;
pub mod energy_contracts;
pub mod hydros;
pub mod lines;
pub mod non_controllable;
pub mod pumping_stations;
pub mod thermals;

pub use buses::parse_buses;
pub use energy_contracts::parse_energy_contracts;
pub use hydros::parse_hydros;
pub use lines::parse_lines;
pub use non_controllable::parse_non_controllable_sources;
pub use pumping_stations::parse_pumping_stations;
pub use thermals::parse_thermals;

use chrono::NaiveDate;
use cobre_core::{
    entities::{EnergyContract, NonControllableSource, PumpingStation},
    penalty::GlobalPenaltyDefaults,
};
use std::path::Path;

use crate::LoadError;

/// Parse an `operational_start_date` string into a [`NaiveDate`].
///
/// The single owner of the `operational_start_date` date format for every
/// `system/*.json` entity registry. Accepts an ISO-8601 calendar date
/// (`YYYY-MM-DD`).
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] naming `path`, `field`, and the offending
/// value when `raw` is not a valid ISO-8601 date.
pub(crate) fn parse_operational_start_date(
    raw: &str,
    path: &Path,
    field: &str,
) -> Result<NaiveDate, LoadError> {
    NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|_| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: format!("'{raw}' is not a valid ISO-8601 date (YYYY-MM-DD)"),
    })
}

/// Load `system/non_controllable_sources.json`, or return an empty vec when absent.
///
/// # Errors
///
/// Propagates errors from [`parse_non_controllable_sources`] when `path` is `Some`.
pub fn load_non_controllable_sources(
    path: Option<&Path>,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<NonControllableSource>, LoadError> {
    match path {
        Some(p) => non_controllable::parse_non_controllable_sources(p, global_penalties),
        None => Ok(Vec::new()),
    }
}

/// Load `system/pumping_stations.json`, or return an empty vec when absent.
///
/// # Errors
///
/// Propagates errors from [`parse_pumping_stations`] when `path` is `Some`.
pub fn load_pumping_stations(path: Option<&Path>) -> Result<Vec<PumpingStation>, LoadError> {
    match path {
        Some(p) => pumping_stations::parse_pumping_stations(p),
        None => Ok(Vec::new()),
    }
}

/// Load `system/energy_contracts.json`, or return an empty vec when absent.
///
/// # Errors
///
/// Propagates errors from [`parse_energy_contracts`] when `path` is `Some`.
pub fn load_energy_contracts(path: Option<&Path>) -> Result<Vec<EnergyContract>, LoadError> {
    match path {
        Some(p) => energy_contracts::parse_energy_contracts(p),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::make_global;

    // ── AC: optional file wrapper returns empty vec for None ──────────────────

    #[test]
    fn test_load_ncs_none_returns_empty() {
        let global = make_global();
        let result = load_non_controllable_sources(None, &global).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }

    #[test]
    fn test_load_pumping_stations_none_returns_empty() {
        let result = load_pumping_stations(None).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }

    #[test]
    fn test_load_energy_contracts_none_returns_empty() {
        let result = load_energy_contracts(None).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }
}
