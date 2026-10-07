//! Parsers for constraint files in the `constraints/` subdirectory.
//!
//! Constraint files provide stage-varying bound overrides for entity types and
//! user-defined generic linear constraints.
//! All files are optional — when absent, the `load_*` wrappers return
//! `Ok(Vec::new())` without touching the filesystem.
//!
//! ## Parsing convention
//!
//! Parquet parsers follow the canonical pattern:
//!
//! 1. Obtain the batch reader from `open_record_batch_reader` and consume all record batches.
//! 2. Extract typed columns by name; return `SchemaError` for missing or wrong-type columns.
//! 3. Validate per-row constraints; return `SchemaError` on violation.
//! 4. Sort the output by the documented sort key and return.
//!
//! JSON parsers follow the 4-step pipeline:
//! `fs::read_to_string` → `serde_json::from_str` → `validate_raw` → `convert`.
//!
//! Cross-reference validation (checking that entity IDs exist in registries)
//! is deferred to Layer 3.
//! Semantic bound validation (e.g., min < max) is deferred.

pub mod bounds;
pub mod generic;
pub mod generic_bounds;
pub mod hydro_unit_group_bounds;
pub(crate) mod named_expression_inline;
pub mod ncs_bounds;
pub mod penalty_overrides;
pub(crate) mod rhs_normalize;

pub use bounds::{
    ContractBoundsRow, HydroBoundsRow, LineBoundsRow, PumpingBoundsRow, ThermalBoundsRow,
    parse_contract_bounds, parse_hydro_bounds, parse_line_bounds, parse_pumping_bounds,
    parse_thermal_bounds,
};
pub use generic::{LineBusPairIndex, build_line_bus_pair_index, parse_generic_constraints};
pub use generic_bounds::{GenericConstraintBoundsRow, parse_generic_constraint_bounds};
pub use hydro_unit_group_bounds::{HydroUnitGroupBoundsRow, parse_hydro_unit_group_bounds};
pub use ncs_bounds::{NcsBoundsRow, parse_ncs_bounds};
pub use penalty_overrides::{
    BusPenaltyOverrideRow, HydroPenaltyOverrideRow, LinePenaltyOverrideRow, NcsPenaltyOverrideRow,
    parse_penalty_overrides_bus, parse_penalty_overrides_hydro, parse_penalty_overrides_line,
    parse_penalty_overrides_ncs,
};

use cobre_core::{EntityId, GenericConstraint};
use std::collections::HashMap;
use std::path::Path;

use crate::LoadError;

fn load_optional<T>(
    path: Option<&Path>,
    parse: impl FnOnce(&Path) -> Result<Vec<T>, LoadError>,
) -> Result<Vec<T>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse(p),
    }
}

/// Load `constraints/thermal_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_thermal_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_thermal_bounds;
///
/// let rows = load_thermal_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_thermal_bounds(path: Option<&Path>) -> Result<Vec<ThermalBoundsRow>, LoadError> {
    load_optional(path, parse_thermal_bounds)
}

/// Load `constraints/hydro_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_hydro_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_hydro_bounds;
///
/// let rows = load_hydro_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_hydro_bounds(path: Option<&Path>) -> Result<Vec<HydroBoundsRow>, LoadError> {
    load_optional(path, parse_hydro_bounds)
}

/// Load `constraints/line_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_line_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_line_bounds;
///
/// let rows = load_line_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_line_bounds(path: Option<&Path>) -> Result<Vec<LineBoundsRow>, LoadError> {
    load_optional(path, parse_line_bounds)
}

/// Load `constraints/pumping_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_pumping_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_pumping_bounds;
///
/// let rows = load_pumping_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_pumping_bounds(path: Option<&Path>) -> Result<Vec<PumpingBoundsRow>, LoadError> {
    load_optional(path, parse_pumping_bounds)
}

/// Load `constraints/contract_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_contract_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_contract_bounds;
///
/// let rows = load_contract_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_contract_bounds(path: Option<&Path>) -> Result<Vec<ContractBoundsRow>, LoadError> {
    load_optional(path, parse_contract_bounds)
}

/// Load `constraints/penalty_overrides_bus.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_bus`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_bus;
///
/// let rows = load_penalty_overrides_bus(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_bus(
    path: Option<&Path>,
) -> Result<Vec<BusPenaltyOverrideRow>, LoadError> {
    load_optional(path, parse_penalty_overrides_bus)
}

/// Load `constraints/penalty_overrides_line.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_line`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_line;
///
/// let rows = load_penalty_overrides_line(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_line(
    path: Option<&Path>,
) -> Result<Vec<LinePenaltyOverrideRow>, LoadError> {
    load_optional(path, parse_penalty_overrides_line)
}

/// Load `constraints/penalty_overrides_hydro.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_hydro`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_hydro;
///
/// let rows = load_penalty_overrides_hydro(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_hydro(
    path: Option<&Path>,
) -> Result<Vec<HydroPenaltyOverrideRow>, LoadError> {
    load_optional(path, parse_penalty_overrides_hydro)
}

/// Load `constraints/penalty_overrides_ncs.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_penalty_overrides_ncs`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_penalty_overrides_ncs;
///
/// let rows = load_penalty_overrides_ncs(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_penalty_overrides_ncs(
    path: Option<&Path>,
) -> Result<Vec<NcsPenaltyOverrideRow>, LoadError> {
    load_optional(path, parse_penalty_overrides_ncs)
}

/// Load `constraints/generic_constraints.json` (optional file; see the module doc).
///
/// `name_to_id` maps parameter definition names to their [`EntityId`].
/// Pass `&HashMap::new()` when no parameters have been loaded; expressions that
/// contain `@name` tokens will then fail with a schema error. The real mapping
/// is wired in by the caller once the parameter loader output is available.
///
/// `line_index` resolves the `line_exchange(source_bus=X, target_bus=Y)` addressing
/// form; pass [`LineBusPairIndex::default`] when no line topology is available.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_generic_constraints`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::{LineBusPairIndex, load_generic_constraints};
/// use std::collections::HashMap;
///
/// let constraints =
///     load_generic_constraints(None, &HashMap::new(), &LineBusPairIndex::default())
///         .expect("no file is fine");
/// assert!(constraints.is_empty());
/// ```
#[allow(clippy::implicit_hasher)]
pub fn load_generic_constraints(
    path: Option<&Path>,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<Vec<GenericConstraint>, LoadError> {
    load_optional(path, |p| {
        parse_generic_constraints(p, name_to_id, line_index)
    })
}

/// Load `constraints/generic_constraint_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_generic_constraint_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_generic_constraint_bounds;
///
/// let rows = load_generic_constraint_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_generic_constraint_bounds(
    path: Option<&Path>,
) -> Result<Vec<GenericConstraintBoundsRow>, LoadError> {
    load_optional(path, parse_generic_constraint_bounds)
}

/// Load `constraints/ncs_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_ncs_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_ncs_bounds;
///
/// let rows = load_ncs_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_ncs_bounds(path: Option<&Path>) -> Result<Vec<NcsBoundsRow>, LoadError> {
    load_optional(path, parse_ncs_bounds)
}

/// Load `constraints/hydro_unit_group_bounds.parquet` (optional file; see the module doc).
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_hydro_unit_group_bounds`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::load_hydro_unit_group_bounds;
///
/// let rows = load_hydro_unit_group_bounds(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_hydro_unit_group_bounds(
    path: Option<&Path>,
) -> Result<Vec<HydroUnitGroupBoundsRow>, LoadError> {
    load_optional(path, parse_hydro_unit_group_bounds)
}
