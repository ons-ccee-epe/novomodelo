//! Parquet parsers for entity bounds override files in the `constraints/` subdirectory.
//!
//! Each parser reads a sparse Parquet file containing stage-varying bound overrides
//! for a specific entity type. Sparse storage means only `(entity_id, stage_id)` pairs
//! that differ from base values need rows.
//!
//! ## Parquet schemas
//!
//! ### `thermal_bounds`
//!
//! | Column             | Type   | Required | Description                                          |
//! | ------------------ | ------ | -------- | ---------------------------------------------------- |
//! | `thermal_id`       | INT32  | Yes      | Thermal plant ID                                     |
//! | `stage_id`         | INT32  | Yes      | Stage ID                                             |
//! | `min_generation_mw`| DOUBLE | No       | Minimum generation (MW)                              |
//! | `max_generation_mw`| DOUBLE | No       | Maximum generation (MW)                              |
//! | `cost_per_mwh`     | DOUBLE | No       | Dispatch cost override (`$/MWh`); rejected at validation if `block_id` is also non-null |
//! | `block_id`         | INT32 (null) | No | 0-based block index within the stage (matches `Block::index`) selecting one block's `min_generation_mw`/`max_generation_mw` override; a value outside `[0, n_blocks)` for the referenced stage, a non-null `cost_per_mwh` on the same row, or any `block_id` on a thermal declaring `anticipated_config` is rejected at validation |
//!
//! ### `hydro_bounds`
//!
//! | Column                | Type   | Required | Description                        |
//! | --------------------- | ------ | -------- | ---------------------------------- |
//! | `hydro_id`            | INT32  | Yes      | Hydro plant ID                     |
//! | `stage_id`            | INT32  | Yes      | Stage ID                           |
//! | `min_turbined_m3s`    | DOUBLE | No       | Min turbined flow (m3/s)           |
//! | `max_turbined_m3s`    | DOUBLE | No       | Max turbined flow (m3/s)           |
//! | `min_storage_hm3`     | DOUBLE | No       | Min reservoir storage (hm3)        |
//! | `max_storage_hm3`     | DOUBLE | No       | Max reservoir storage (hm3)        |
//! | `min_outflow_m3s`     | DOUBLE | No       | Min non-diverted outflow (m3/s)    |
//! | `max_outflow_m3s`     | DOUBLE | No       | Max non-diverted outflow (m3/s)    |
//! | `min_generation_mw`   | DOUBLE | No       | Min generation (MW)                |
//! | `max_generation_mw`   | DOUBLE | No       | Max generation (MW)                |
//! | `min_diversion_m3s`   | DOUBLE | No       | Min diversion flow (m3/s)          |
//! | `max_diversion_m3s`   | DOUBLE | No       | Max diversion flow (m3/s)          |
//! | `min_spillage_m3s`    | DOUBLE | No       | Min spillage flow (m3/s)           |
//! | `max_spillage_m3s`    | DOUBLE | No       | Max spillage flow (m3/s)           |
//! | `filling_min_rate_m3s`| DOUBLE | No       | Filling min-rate override (m3/s)   |
//! | `water_withdrawal_m3s`| DOUBLE | No       | Water withdrawal (m3/s)            |
//! | `block_id`            | INT32 (null) | No | 0-based block index within the stage (matches `Block::index`) selecting one block's turbined/outflow/generation/diversion/spillage override; a value outside `[0, n_blocks)` for the referenced stage is rejected at validation; a non-null storage, filling-rate, or withdrawal value on the same row is rejected at validation (stage-level only, no per-block variant) |
//!
//! ### `line_bounds`
//!
//! | Column       | Type   | Required | Description                        |
//! | ------------ | ------ | -------- | ---------------------------------- |
//! | `line_id`    | INT32  | Yes      | Transmission line ID               |
//! | `stage_id`   | INT32  | Yes      | Stage ID                           |
//! | `direct_mw`  | DOUBLE | No       | Direct-flow capacity (MW)          |
//! | `reverse_mw` | DOUBLE | No       | Reverse-flow capacity (MW)         |
//! | `block_id`   | INT32 (null) | No | 0-based block index within the stage (matches `Block::index`) selecting one block's `direct_mw`/`reverse_mw` override; a value outside `[0, n_blocks)` for the referenced stage is rejected at validation |
//!
//! ### `pumping_bounds`
//!
//! | Column               | Type   | Required | Description                        |
//! | -------------------- | ------ | -------- | ---------------------------------- |
//! | `pumping_station_id` | INT32  | Yes      | Pumping station ID                               |
//! | `stage_id`           | INT32  | Yes      | Stage ID                           |
//! | `min_m3s`            | DOUBLE | No       | Minimum pumping flow (m3/s)        |
//! | `max_m3s`            | DOUBLE | No       | Maximum pumping flow (m3/s)        |
//! | `block_id`           | INT32 (null) | No | 0-based block index within the stage (matches `Block::index`) selecting one block's `min_m3s`/`max_m3s` override; a value outside `[0, n_blocks)` for the referenced stage is rejected at validation |
//!
//! ### `contract_bounds`
//!
//! | Column           | Type   | Required | Description                    |
//! | ---------------- | ------ | -------- | ------------------------------ |
//! | `contract_id`    | INT32  | Yes      | Energy contract ID             |
//! | `stage_id`       | INT32  | Yes      | Stage ID                       |
//! | `min_mw`         | DOUBLE | No       | Minimum power (MW)             |
//! | `max_mw`         | DOUBLE | No       | Maximum power (MW)             |
//! | `price_per_mwh`  | DOUBLE | No       | Price override ($/`MWh`)       |
//! | `block_id`       | INT32 (null) | No | 0-based block index within the stage (matches `Block::index`) selecting one block's `min_mw`/`max_mw`/`price_per_mwh` override; a value outside `[0, n_blocks)` for the referenced stage is rejected at validation |
//!
//! ## Block eligibility
//!
//! A bound column is block-eligible exactly when its family's
//! `<Family>BlockOverride` struct (`cobre_core::resolved`) carries a field for
//! it — the struct's field set is the check, not this table. A column marked
//! "no" below has no per-block variant: a row combining a non-null value here
//! with a non-null `block_id` is rejected at validation (see the per-file
//! tables above) and never reaches
//! [`resolve_bounds`](crate::resolution::resolve_bounds), which documents the
//! precedence law the block-eligible columns feed.
//!
//! | Family   | Column                 | Block-eligible | Note                                                                                          |
//! |----------|------------------------|:--------------:|------------------------------------------------------------------------------------------------|
//! | hydro    | `min_turbined_m3s`     | yes            | —                                                                                                |
//! | hydro    | `max_turbined_m3s`     | yes            | —                                                                                                |
//! | hydro    | `min_outflow_m3s`      | yes            | —                                                                                                |
//! | hydro    | `max_outflow_m3s`      | yes            | —                                                                                                |
//! | hydro    | `min_generation_mw`    | yes            | —                                                                                                |
//! | hydro    | `max_generation_mw`    | yes            | —                                                                                                |
//! | hydro    | `min_diversion_m3s`    | yes            | —                                                                                                |
//! | hydro    | `max_diversion_m3s`    | yes            | —                                                                                                |
//! | hydro    | `min_spillage_m3s`     | yes            | —                                                                                                |
//! | hydro    | `max_spillage_m3s`     | yes            | —                                                                                                |
//! | hydro    | `min_storage_hm3`      | no             | storage bound; stage-level only                                                                 |
//! | hydro    | `max_storage_hm3`      | no             | storage bound; stage-level only                                                                 |
//! | hydro    | `filling_min_rate_m3s` | no             | filling schedule; stage-level only                                                              |
//! | hydro    | `water_withdrawal_m3s` | no             | stage-level only                                                                                 |
//! | thermal  | `min_generation_mw`    | yes            | —                                                                                                |
//! | thermal  | `max_generation_mw`    | yes            | —                                                                                                |
//! | thermal  | `cost_per_mwh`         | no             | per-block thermal cost is out of scope; deliberately asymmetric with `contract_bounds.price_per_mwh` below |
//! | line     | `direct_mw`            | yes            | —                                                                                                |
//! | line     | `reverse_mw`           | yes            | —                                                                                                |
//! | contract | `min_mw`               | yes            | —                                                                                                |
//! | contract | `max_mw`               | yes            | —                                                                                                |
//! | contract | `price_per_mwh`        | yes            | deliberately asymmetric with `thermal_bounds.cost_per_mwh` above, which is NOT block-eligible    |
//! | pumping  | `min_m3s`              | yes            | —                                                                                                |
//! | pumping  | `max_m3s`              | yes            | —                                                                                                |
//!
//! ## Output ordering
//!
//! All parsers return rows sorted by `(entity_id, stage_id, block_id)`
//! ascending (`None` before `Some(i)`), since a stage-wide row and a block row
//! may legitimately share `(entity_id, stage_id)`.
//!
//! ## Validation
//!
//! Per-row constraints enforced by these parsers:
//!
//! - Required key columns (`*_id`, `stage_id`) must be present with Int32 type.
//! - Any provided (non-null) optional Float64 value must be finite — NaN and ±Inf are rejected.
//! - Some parsers additionally enforce domain constraints on their own values:
//!   `pumping_bounds` rejects negative flows and `min_m3s > max_m3s` (matching the
//!   `pumping_stations.json` entity-reader checks); `thermal_bounds` rejects a
//!   negative `cost_per_mwh`; `hydro_bounds` rejects negative `min_diversion_m3s`,
//!   `min_spillage_m3s`, and `max_spillage_m3s`, and a same-row
//!   `min_spillage_m3s > max_spillage_m3s`.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in registries — Layer 3.
//! - Duplicate `(entity_id, stage_id, block_id, column)` keys — Layer 5a
//!   (`check_duplicate_bound_rows`); a stage-wide row and a block row sharing
//!   `(entity_id, stage_id)` is designed usage, not a duplicate, so `block_id`
//!   is part of the key, and each bound column is keyed independently since
//!   two rows setting disjoint columns for the same `(entity, stage, block)`
//!   are legitimate sparse input.
//! - `block_id` validity for the referenced stage — Layer 5a
//!   (`check_bound_block_id_range`).
//! - Cross-field validation for the remaining bounds types — deferred.

use arrow::array::{Array, Float64Array, Int32Array};
use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_float64, extract_optional_int32, extract_required_int32,
    open_record_batch_reader,
};

// ── Row types ─────────────────────────────────────────────────────────────────

/// A single row from `constraints/thermal_bounds.parquet`.
///
/// Carries stage-varying bound overrides for a thermal generation plant.
/// Fields are `None` when the corresponding column is absent or null in the
/// Parquet file (sparse storage: absent means "use base value").
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::ThermalBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = ThermalBoundsRow {
///     thermal_id: EntityId::from(2),
///     stage_id: 5,
///     min_generation_mw: Some(10.0),
///     max_generation_mw: None,
///     cost_per_mwh: Some(75.0),
///     block_id: None,
/// };
/// assert_eq!(row.thermal_id, EntityId::from(2));
/// assert_eq!(row.min_generation_mw, Some(10.0));
/// assert!(row.max_generation_mw.is_none());
/// assert_eq!(row.cost_per_mwh, Some(75.0));
/// assert!(row.block_id.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalBoundsRow {
    /// Thermal plant ID.
    pub thermal_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Minimum generation override (MW).
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override (MW).
    pub max_generation_mw: Option<f64>,
    /// Dispatch cost override (`$/MWh`). Overrides `Thermal.cost_per_mwh` when `Some` and
    /// `block_id` is `None`. Rejected at validation when `block_id` is also `Some`
    /// (see the `block_id` field).
    pub cost_per_mwh: Option<f64>,
    /// Selects one block within the stage for `min_generation_mw` /
    /// `max_generation_mw`; `None` applies the row at the stage level. A
    /// 0-based index matching `Block::index`; a value outside `[0, n_blocks)`
    /// for the referenced stage is rejected at validation. Does not create a
    /// per-block cost — a row with a non-null `block_id` and a non-null
    /// `cost_per_mwh` is rejected at validation (see the `cost_per_mwh` field),
    /// as is any `block_id` on a thermal declaring `anticipated_config`
    /// (commitment is stage-level).
    pub block_id: Option<i32>,
}

/// A single row from `constraints/hydro_bounds.parquet`.
///
/// Carries stage-varying bound overrides for a hydro plant. Every bound
/// column is optional; absent or null means "use base value".
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::HydroBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = HydroBoundsRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 3,
///     min_turbined_m3s: Some(50.0),
///     max_turbined_m3s: None,
///     min_storage_hm3: None,
///     max_storage_hm3: Some(200.0),
///     min_outflow_m3s: None,
///     max_outflow_m3s: None,
///     min_generation_mw: None,
///     max_generation_mw: None,
///     min_diversion_m3s: None,
///     max_diversion_m3s: None,
///     min_spillage_m3s: None,
///     max_spillage_m3s: None,
///     filling_min_rate_m3s: None,
///     water_withdrawal_m3s: None,
///     block_id: None,
/// };
/// assert_eq!(row.min_turbined_m3s, Some(50.0));
/// assert!(row.max_turbined_m3s.is_none());
/// ```
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq)]
pub struct HydroBoundsRow {
    /// Hydro plant ID.
    pub hydro_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Minimum turbined flow override (m³/s).
    pub min_turbined_m3s: Option<f64>,
    /// Maximum turbined flow override (m³/s).
    pub max_turbined_m3s: Option<f64>,
    /// Minimum storage override (hm³).
    pub min_storage_hm3: Option<f64>,
    /// Maximum storage override (hm³).
    pub max_storage_hm3: Option<f64>,
    /// Minimum outflow override (m³/s).
    pub min_outflow_m3s: Option<f64>,
    /// Maximum outflow override (m³/s).
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation override (MW).
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override (MW).
    pub max_generation_mw: Option<f64>,
    /// Minimum diversion override (m³/s).
    pub min_diversion_m3s: Option<f64>,
    /// Maximum diversion override (m³/s).
    pub max_diversion_m3s: Option<f64>,
    /// Minimum spillage override (m³/s).
    pub min_spillage_m3s: Option<f64>,
    /// Maximum spillage override (m³/s).
    pub max_spillage_m3s: Option<f64>,
    /// Filling minimum accumulation rate override (m³/s).
    pub filling_min_rate_m3s: Option<f64>,
    /// Water withdrawal override (m³/s).
    pub water_withdrawal_m3s: Option<f64>,
    /// `None` applies at the stage level; `Some(b)` applies to block `b` only and is
    /// valid on the turbined/outflow/generation/diversion/spillage columns. A row
    /// combining `Some(b)` with a non-null `min_storage_hm3`/`max_storage_hm3`/
    /// `filling_min_rate_m3s`/`water_withdrawal_m3s` is rejected at validation —
    /// those four are stage-level with no per-block variant.
    pub block_id: Option<i32>,
}

/// Neutral test scaffold — every optional column `None`, `hydro_id`/`stage_id`
/// zeroed — never a bound source; production sites construct `HydroBoundsRow`
/// exhaustively.
impl Default for HydroBoundsRow {
    fn default() -> Self {
        Self {
            hydro_id: EntityId::from(0),
            stage_id: 0,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_storage_hm3: None,
            max_storage_hm3: None,
            min_outflow_m3s: None,
            max_outflow_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            min_diversion_m3s: None,
            max_diversion_m3s: None,
            min_spillage_m3s: None,
            max_spillage_m3s: None,
            filling_min_rate_m3s: None,
            water_withdrawal_m3s: None,
            block_id: None,
        }
    }
}

/// A single row from `constraints/line_bounds.parquet`.
///
/// Carries stage-varying capacity overrides for a transmission line.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::LineBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = LineBoundsRow {
///     line_id: EntityId::from(10),
///     stage_id: 0,
///     direct_mw: Some(500.0),
///     reverse_mw: Some(500.0),
///     block_id: None,
/// };
/// assert_eq!(row.line_id, EntityId::from(10));
/// assert_eq!(row.direct_mw, Some(500.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LineBoundsRow {
    /// Transmission line this override applies to.
    pub line_id: EntityId,
    /// Declared study-stage id this override applies to (not a 0-based index).
    pub stage_id: i32,
    /// Override for direct-flow capacity (MW). `None` means use base value.
    pub direct_mw: Option<f64>,
    /// Override for reverse-flow capacity (MW). `None` means use base value.
    pub reverse_mw: Option<f64>,
    /// `None` applies at the stage level; `Some(b)` applies to block `b` only and is
    /// valid on both `direct_mw` and `reverse_mw`.
    pub block_id: Option<i32>,
}

/// A single row from `constraints/pumping_bounds.parquet`.
///
/// Carries stage-varying flow bound overrides for a pumping station.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::PumpingBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = PumpingBoundsRow {
///     station_id: EntityId::from(3),
///     stage_id: 2,
///     min_m3s: Some(0.0),
///     max_m3s: Some(100.0),
///     block_id: None,
/// };
/// assert_eq!(row.station_id, EntityId::from(3));
/// assert_eq!(row.max_m3s, Some(100.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PumpingBoundsRow {
    /// Pumping station this override applies to.
    pub station_id: EntityId,
    /// Declared study-stage id this override applies to (not a 0-based index).
    pub stage_id: i32,
    /// Override for minimum pumping flow (m³/s). `None` means use base value.
    pub min_m3s: Option<f64>,
    /// Override for maximum pumping flow (m³/s). `None` means use base value.
    pub max_m3s: Option<f64>,
    /// `None` applies at the stage level; `Some(b)` applies to block `b` only and is
    /// valid on both `min_m3s` and `max_m3s`.
    pub block_id: Option<i32>,
}

/// A single row from `constraints/contract_bounds.parquet`.
///
/// Carries stage-varying bound and price overrides for an energy contract.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::ContractBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = ContractBoundsRow {
///     contract_id: EntityId::from(7),
///     stage_id: 1,
///     min_mw: Some(0.0),
///     max_mw: Some(200.0),
///     price_per_mwh: None,
///     block_id: None,
/// };
/// assert_eq!(row.contract_id, EntityId::from(7));
/// assert!(row.price_per_mwh.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ContractBoundsRow {
    /// Energy contract this override applies to.
    pub contract_id: EntityId,
    /// Declared study-stage id this override applies to (not a 0-based index).
    pub stage_id: i32,
    /// Override for minimum power (MW). `None` means use base value.
    pub min_mw: Option<f64>,
    /// Override for maximum power (MW). `None` means use base value.
    pub max_mw: Option<f64>,
    /// Override for contract price (USD/`MWh`). `None` means use base value.
    pub price_per_mwh: Option<f64>,
    /// `None` applies at the stage level; `Some(b)` applies to block `b` only and is
    /// valid on `min_mw`, `max_mw`, and `price_per_mwh` — block-eligible here, asymmetric
    /// with `ThermalBoundsRow.cost_per_mwh`, which stays stage-level.
    pub block_id: Option<i32>,
}

// ── Parsers ───────────────────────────────────────────────────────────────────

/// Row `i` of an optional `Float64` column; absent column or null cell yields `None`.
pub(super) fn optional_f64(col: Option<&Float64Array>, i: usize) -> Option<f64> {
    col.filter(|c| !c.is_null(i)).map(|c| c.value(i))
}

/// Row `i` of an optional `Int32` column; absent column or null cell yields `None`.
pub(super) fn optional_i32(col: Option<&Int32Array>, i: usize) -> Option<i32> {
    col.filter(|c| !c.is_null(i)).map(|c| c.value(i))
}

/// Validate that a present (non-null) optional float value is finite.
pub(super) fn validate_optional_finite(
    value: Option<f64>,
    file_label: &str,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<(), LoadError> {
    if let Some(v) = value
        && !v.is_finite()
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{file_label}[{row_idx}].{column}"),
            message: format!("value must be finite, got {v}"),
        });
    }
    Ok(())
}

/// Validate that a present (non-null) optional value is non-negative.
pub(super) fn validate_optional_nonneg(
    value: Option<f64>,
    file_label: &str,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<(), LoadError> {
    if let Some(v) = value
        && v < 0.0
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{file_label}[{row_idx}].{column}"),
            message: format!("value must be >= 0.0, got {v}"),
        });
    }
    Ok(())
}

/// Parse `constraints/thermal_bounds.parquet`, returning rows sorted by
/// `(thermal_id, stage_id, block_id)` ascending (`None` before `Some(i)`).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_thermal_bounds;
/// use std::path::Path;
///
/// let rows = parse_thermal_bounds(Path::new("constraints/thermal_bounds.parquet"))
///     .expect("valid thermal bounds file");
/// println!("loaded {} thermal bounds rows", rows.len());
/// ```
pub fn parse_thermal_bounds(path: &Path) -> Result<Vec<ThermalBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<ThermalBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let thermal_id_col = extract_required_int32(&batch, "thermal_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let min_gen_col = extract_optional_float64(&batch, "min_generation_mw", path)?;
        let max_gen_col = extract_optional_float64(&batch, "max_generation_mw", path)?;
        let cost_col = extract_optional_float64(&batch, "cost_per_mwh", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let thermal_id = EntityId::from(thermal_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_generation_mw = optional_f64(min_gen_col, i);
            let max_generation_mw = optional_f64(max_gen_col, i);
            let cost_per_mwh = optional_f64(cost_col, i);
            let block_id = optional_i32(block_id_col, i);

            for (value, column) in [
                (min_generation_mw, "min_generation_mw"),
                (max_generation_mw, "max_generation_mw"),
                (cost_per_mwh, "cost_per_mwh"),
            ] {
                validate_optional_finite(value, "thermal_bounds", row_idx, column, path)?;
            }
            validate_optional_nonneg(
                cost_per_mwh,
                "thermal_bounds",
                row_idx,
                "cost_per_mwh",
                path,
            )?;

            rows.push(ThermalBoundsRow {
                thermal_id,
                stage_id,
                min_generation_mw,
                max_generation_mw,
                cost_per_mwh,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| (r.thermal_id.0, r.stage_id, r.block_id));

    Ok(rows)
}

/// Parse `constraints/hydro_bounds.parquet`, returning rows sorted by
/// `(hydro_id, stage_id, block_id)` ascending (`None` before `Some(i)`).
///
/// The file may contain any subset of the optional bound columns; columns
/// absent from the file schema produce `None` in all rows (the sparse-override design,
/// not a schema error).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_hydro_bounds;
/// use std::path::Path;
///
/// let rows = parse_hydro_bounds(Path::new("constraints/hydro_bounds.parquet"))
///     .expect("valid hydro bounds file");
/// println!("loaded {} hydro bounds rows", rows.len());
/// ```
// Rationale: a single sequential pass validates every column for every row before returning;
// splitting would require multiple passes over the batch.
#[allow(clippy::too_many_lines)]
pub fn parse_hydro_bounds(path: &Path) -> Result<Vec<HydroBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<HydroBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let min_turbined_col = extract_optional_float64(&batch, "min_turbined_m3s", path)?;
        let max_turbined_col = extract_optional_float64(&batch, "max_turbined_m3s", path)?;
        let min_storage_col = extract_optional_float64(&batch, "min_storage_hm3", path)?;
        let max_storage_col = extract_optional_float64(&batch, "max_storage_hm3", path)?;
        let min_outflow_col = extract_optional_float64(&batch, "min_outflow_m3s", path)?;
        let max_outflow_col = extract_optional_float64(&batch, "max_outflow_m3s", path)?;
        let min_gen_col = extract_optional_float64(&batch, "min_generation_mw", path)?;
        let max_gen_col = extract_optional_float64(&batch, "max_generation_mw", path)?;
        let min_diversion_col = extract_optional_float64(&batch, "min_diversion_m3s", path)?;
        let max_diversion_col = extract_optional_float64(&batch, "max_diversion_m3s", path)?;
        let min_spillage_col = extract_optional_float64(&batch, "min_spillage_m3s", path)?;
        let max_spillage_col = extract_optional_float64(&batch, "max_spillage_m3s", path)?;
        let filling_min_rate_col = extract_optional_float64(&batch, "filling_min_rate_m3s", path)?;
        let water_withdrawal_col = extract_optional_float64(&batch, "water_withdrawal_m3s", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_turbined_m3s = optional_f64(min_turbined_col, i);
            let max_turbined_m3s = optional_f64(max_turbined_col, i);
            let min_storage_hm3 = optional_f64(min_storage_col, i);
            let max_storage_hm3 = optional_f64(max_storage_col, i);
            let min_outflow_m3s = optional_f64(min_outflow_col, i);
            let max_outflow_m3s = optional_f64(max_outflow_col, i);
            let min_generation_mw = optional_f64(min_gen_col, i);
            let max_generation_mw = optional_f64(max_gen_col, i);
            let min_diversion_m3s = optional_f64(min_diversion_col, i);
            let max_diversion_m3s = optional_f64(max_diversion_col, i);
            let min_spillage_m3s = optional_f64(min_spillage_col, i);
            let max_spillage_m3s = optional_f64(max_spillage_col, i);
            let filling_min_rate_m3s = optional_f64(filling_min_rate_col, i);
            let water_withdrawal_m3s = optional_f64(water_withdrawal_col, i);
            let block_id = optional_i32(block_id_col, i);

            for (value, column) in [
                (min_turbined_m3s, "min_turbined_m3s"),
                (max_turbined_m3s, "max_turbined_m3s"),
                (min_storage_hm3, "min_storage_hm3"),
                (max_storage_hm3, "max_storage_hm3"),
                (min_outflow_m3s, "min_outflow_m3s"),
                (max_outflow_m3s, "max_outflow_m3s"),
                (min_generation_mw, "min_generation_mw"),
                (max_generation_mw, "max_generation_mw"),
                (min_diversion_m3s, "min_diversion_m3s"),
                (max_diversion_m3s, "max_diversion_m3s"),
                (min_spillage_m3s, "min_spillage_m3s"),
                (max_spillage_m3s, "max_spillage_m3s"),
                (filling_min_rate_m3s, "filling_min_rate_m3s"),
                (water_withdrawal_m3s, "water_withdrawal_m3s"),
            ] {
                validate_optional_finite(value, "hydro_bounds", row_idx, column, path)?;
            }

            // build_filling_v_target and check_filling_sufficiency assume rate ≥ 0;
            // a negative override silently inverts the V_target floor (validate_filling_configs
            // enforces this for the entity; the finiteness gate above does not).
            validate_optional_nonneg(
                filling_min_rate_m3s,
                "hydro_bounds",
                row_idx,
                "filling_min_rate_m3s",
                path,
            )?;

            // A diversion/spillage override is non-negative and, for spillage,
            // `min <= max`; a negative or inverted row otherwise yields an
            // infeasible per-stage bound (the finiteness gate above catches
            // neither sign nor ordering).
            validate_optional_nonneg(
                min_diversion_m3s,
                "hydro_bounds",
                row_idx,
                "min_diversion_m3s",
                path,
            )?;
            validate_optional_nonneg(
                min_spillage_m3s,
                "hydro_bounds",
                row_idx,
                "min_spillage_m3s",
                path,
            )?;
            validate_optional_nonneg(
                max_spillage_m3s,
                "hydro_bounds",
                row_idx,
                "max_spillage_m3s",
                path,
            )?;
            if let (Some(min), Some(max)) = (min_spillage_m3s, max_spillage_m3s)
                && min > max
            {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("hydro_bounds[{row_idx}].max_spillage_m3s"),
                    message: format!(
                        "max_spillage_m3s ({max}) must be >= min_spillage_m3s ({min})"
                    ),
                });
            }

            rows.push(HydroBoundsRow {
                hydro_id,
                stage_id,
                min_turbined_m3s,
                max_turbined_m3s,
                min_storage_hm3,
                max_storage_hm3,
                min_outflow_m3s,
                max_outflow_m3s,
                min_generation_mw,
                max_generation_mw,
                min_diversion_m3s,
                max_diversion_m3s,
                min_spillage_m3s,
                max_spillage_m3s,
                filling_min_rate_m3s,
                water_withdrawal_m3s,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| (r.hydro_id.0, r.stage_id, r.block_id));

    Ok(rows)
}

/// Parse `constraints/line_bounds.parquet`, returning rows sorted by
/// `(line_id, stage_id, block_id)` ascending (`None` before `Some(i)`).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_line_bounds;
/// use std::path::Path;
///
/// let rows = parse_line_bounds(Path::new("constraints/line_bounds.parquet"))
///     .expect("valid line bounds file");
/// println!("loaded {} line bounds rows", rows.len());
/// ```
pub fn parse_line_bounds(path: &Path) -> Result<Vec<LineBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<LineBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let line_id_col = extract_required_int32(&batch, "line_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let direct_col = extract_optional_float64(&batch, "direct_mw", path)?;
        let reverse_col = extract_optional_float64(&batch, "reverse_mw", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let line_id = EntityId::from(line_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let direct_mw = optional_f64(direct_col, i);
            let reverse_mw = optional_f64(reverse_col, i);
            let block_id = optional_i32(block_id_col, i);

            validate_optional_finite(direct_mw, "line_bounds", row_idx, "direct_mw", path)?;
            validate_optional_finite(reverse_mw, "line_bounds", row_idx, "reverse_mw", path)?;

            rows.push(LineBoundsRow {
                line_id,
                stage_id,
                direct_mw,
                reverse_mw,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| (r.line_id.0, r.stage_id, r.block_id));

    Ok(rows)
}

/// Parse `constraints/pumping_bounds.parquet`, returning rows sorted by
/// `(station_id, stage_id, block_id)` ascending (`None` before `Some(i)`).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_pumping_bounds;
/// use std::path::Path;
///
/// let rows = parse_pumping_bounds(Path::new("constraints/pumping_bounds.parquet"))
///     .expect("valid pumping bounds file");
/// println!("loaded {} pumping bounds rows", rows.len());
/// ```
pub fn parse_pumping_bounds(path: &Path) -> Result<Vec<PumpingBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<PumpingBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let station_id_col = extract_required_int32(&batch, "pumping_station_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let min_col = extract_optional_float64(&batch, "min_m3s", path)?;
        let max_col = extract_optional_float64(&batch, "max_m3s", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let station_id = EntityId::from(station_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_m3s = optional_f64(min_col, i);
            let max_m3s = optional_f64(max_col, i);
            let block_id = optional_i32(block_id_col, i);

            validate_optional_finite(min_m3s, "pumping_bounds", row_idx, "min_m3s", path)?;
            validate_optional_finite(max_m3s, "pumping_bounds", row_idx, "max_m3s", path)?;

            // A pumped-flow override is non-negative and `min <= max`; a negative or
            // inverted row otherwise yields an infeasible per-stage bound (the
            // `pumping_stations.json` entity-reader enforces this; the finiteness gate
            // above does not).
            validate_optional_nonneg(min_m3s, "pumping_bounds", row_idx, "min_m3s", path)?;
            validate_optional_nonneg(max_m3s, "pumping_bounds", row_idx, "max_m3s", path)?;
            if let (Some(min), Some(max)) = (min_m3s, max_m3s)
                && min > max
            {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("pumping_bounds[{row_idx}].max_m3s"),
                    message: format!("max_m3s ({max}) must be >= min_m3s ({min})"),
                });
            }

            rows.push(PumpingBoundsRow {
                station_id,
                stage_id,
                min_m3s,
                max_m3s,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| (r.station_id.0, r.stage_id, r.block_id));

    Ok(rows)
}

/// Parse `constraints/contract_bounds.parquet`, returning rows sorted by
/// `(contract_id, stage_id, block_id)` ascending (`None` before `Some(i)`).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_contract_bounds;
/// use std::path::Path;
///
/// let rows = parse_contract_bounds(Path::new("constraints/contract_bounds.parquet"))
///     .expect("valid contract bounds file");
/// println!("loaded {} contract bounds rows", rows.len());
/// ```
pub fn parse_contract_bounds(path: &Path) -> Result<Vec<ContractBoundsRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<ContractBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let contract_id_col = extract_required_int32(&batch, "contract_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let min_col = extract_optional_float64(&batch, "min_mw", path)?;
        let max_col = extract_optional_float64(&batch, "max_mw", path)?;
        let price_col = extract_optional_float64(&batch, "price_per_mwh", path)?;
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let contract_id = EntityId::from(contract_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_mw = optional_f64(min_col, i);
            let max_mw = optional_f64(max_col, i);
            let price_per_mwh = optional_f64(price_col, i);
            let block_id = optional_i32(block_id_col, i);

            validate_optional_finite(min_mw, "contract_bounds", row_idx, "min_mw", path)?;
            validate_optional_finite(max_mw, "contract_bounds", row_idx, "max_mw", path)?;
            validate_optional_finite(
                price_per_mwh,
                "contract_bounds",
                row_idx,
                "price_per_mwh",
                path,
            )?;

            rows.push(ContractBoundsRow {
                contract_id,
                stage_id,
                min_mw,
                max_mw,
                price_per_mwh,
                block_id,
            });
        }
    }

    rows.sort_by_key(|r| (r.contract_id.0, r.stage_id, r.block_id));

    Ok(rows)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // ── ThermalBoundsRow tests ────────────────────────────────────────────────

    fn thermal_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]))
    }

    fn make_thermal_batch(
        thermal_ids: &[i32],
        stage_ids: &[i32],
        min_gen: Vec<Option<f64>>,
        max_gen: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            thermal_schema(),
            vec![
                Arc::new(Int32Array::from(thermal_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_gen)),
                Arc::new(Float64Array::from(max_gen)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: 3 rows for 2 thermals across 2 stages, valid happy path, correct sort order.
    #[test]
    fn test_thermal_valid_3_rows_sorted() {
        // Scrambled: (3,0), (1,1), (1,0) — result must be sorted (1,0),(1,1),(3,0).
        let batch = make_thermal_batch(
            &[3, 1, 1],
            &[0, 1, 0],
            vec![Some(10.0), Some(5.0), Some(8.0)],
            vec![Some(100.0), Some(80.0), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].thermal_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].min_generation_mw.unwrap() - 8.0).abs() < f64::EPSILON);
        assert!(rows[0].max_generation_mw.is_none());
        assert_eq!(rows[1].thermal_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[2].thermal_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert!((rows[2].min_generation_mw.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((rows[2].max_generation_mw.unwrap() - 100.0).abs() < f64::EPSILON);
    }

    /// AC: missing `stage_id` column -> SchemaError with field "stage_id".
    #[test]
    fn test_thermal_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stage_id"),
                    "field should contain 'stage_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `max_generation_mw` -> SchemaError mentioning "max_generation_mw" and "finite".
    #[test]
    fn test_thermal_nan_max_generation() {
        let batch = make_thermal_batch(&[1], &[0], vec![Some(10.0)], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_generation_mw"),
                    "field should contain 'max_generation_mw', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_thermal_empty_parquet() {
        let batch = make_thermal_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: scrambled input order -> output sorted by (thermal_id, stage_id).
    #[test]
    fn test_thermal_declaration_order_invariance() {
        let batch_asc = make_thermal_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            vec![Some(10.0), Some(11.0), Some(50.0), Some(51.0)],
            vec![Some(100.0), Some(110.0), Some(500.0), Some(510.0)],
        );
        let batch_desc = make_thermal_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(51.0), Some(50.0), Some(11.0), Some(10.0)],
            vec![Some(510.0), Some(500.0), Some(110.0), Some(100.0)],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_thermal_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_thermal_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.thermal_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.thermal_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_thermal_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_thermal_bounds_none() {
        let rows = super::super::load_thermal_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    /// Parquet with cost_per_mwh and block_id columns — values read correctly.
    #[test]
    fn test_thermal_cost_and_block_id_columns_read() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("cost_per_mwh", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 1_i32])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64), Some(100.0_f64)])),
                Arc::new(Int32Array::from(vec![None::<i32>, None::<i32>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].thermal_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].cost_per_mwh, Some(50.0));
        assert!(rows[0].block_id.is_none());
        assert_eq!(rows[1].thermal_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[1].cost_per_mwh, Some(100.0));
        assert!(rows[1].block_id.is_none());
    }

    /// Parquet without cost_per_mwh and block_id columns — all rows have None.
    #[test]
    fn test_thermal_missing_cost_and_block_id_columns_are_none() {
        let batch = make_thermal_batch(
            &[1, 2],
            &[0, 0],
            vec![Some(10.0), Some(20.0)],
            vec![Some(100.0), Some(200.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].cost_per_mwh.is_none());
        assert!(rows[0].block_id.is_none());
        assert!(rows[1].cost_per_mwh.is_none());
        assert!(rows[1].block_id.is_none());
    }

    /// Parquet row with non-null block_id is parsed with block_id = Some(_).
    #[test]
    fn test_thermal_non_null_block_id_is_parsed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("cost_per_mwh", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(75.0_f64)])),
                Arc::new(Int32Array::from(vec![Some(2_i32)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].block_id, Some(2));
        assert_eq!(rows[0].cost_per_mwh, Some(75.0));
    }

    /// NaN in cost_per_mwh -> SchemaError mentioning "cost_per_mwh" and "finite".
    #[test]
    fn test_thermal_nan_cost_per_mwh() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("cost_per_mwh", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(f64::NAN)])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("cost_per_mwh"),
                    "field should contain 'cost_per_mwh', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `cost_per_mwh` in thermal_bounds.parquet → `SchemaError`.
    #[test]
    fn test_thermal_negative_cost_per_mwh() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("cost_per_mwh", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(-10.0_f64)])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("cost_per_mwh"),
                    "field should contain 'cost_per_mwh', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should contain '>= 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError for negative cost, got: {other:?}"),
        }
    }

    // ── HydroBoundsRow tests ──────────────────────────────────────────────────

    fn hydro_full_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
            Field::new("max_turbined_m3s", DataType::Float64, true),
            Field::new("min_storage_hm3", DataType::Float64, true),
            Field::new("max_storage_hm3", DataType::Float64, true),
            Field::new("min_outflow_m3s", DataType::Float64, true),
            Field::new("max_outflow_m3s", DataType::Float64, true),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("min_diversion_m3s", DataType::Float64, true),
            Field::new("max_diversion_m3s", DataType::Float64, true),
            Field::new("min_spillage_m3s", DataType::Float64, true),
            Field::new("max_spillage_m3s", DataType::Float64, true),
            Field::new("filling_min_rate_m3s", DataType::Float64, true),
            Field::new("water_withdrawal_m3s", DataType::Float64, true),
        ]))
    }

    /// AC: valid file with all 14 optional columns, some null and some filled.
    #[test]
    fn test_hydro_all_14_columns_mixed_null() {
        let batch = RecordBatch::try_new(
            hydro_full_schema(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])), // min_turbined
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_turbined: null
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])), // min_storage
                Arc::new(Float64Array::from(vec![Some(500.0_f64)])), // max_storage
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_outflow: null
                Arc::new(Float64Array::from(vec![Some(200.0_f64)])), // max_outflow
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_gen: null
                Arc::new(Float64Array::from(vec![Some(80.0_f64)])), // max_gen
                Arc::new(Float64Array::from(vec![Some(2.0_f64)])),  // min_diversion
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_diversion: null
                Arc::new(Float64Array::from(vec![Some(3.0_f64)])),  // min_spillage
                Arc::new(Float64Array::from(vec![Some(9.0_f64)])),  // max_spillage
                Arc::new(Float64Array::from(vec![Some(10.0_f64)])), // filling_min_rate
                Arc::new(Float64Array::from(vec![None::<f64>])),    // water_withdrawal: null
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(1));
        assert_eq!(r.stage_id, 0);
        assert!((r.min_turbined_m3s.unwrap() - 50.0).abs() < f64::EPSILON);
        assert!(r.max_turbined_m3s.is_none());
        assert!((r.min_storage_hm3.unwrap() - 100.0).abs() < f64::EPSILON);
        assert!((r.max_storage_hm3.unwrap() - 500.0).abs() < f64::EPSILON);
        assert!(r.min_outflow_m3s.is_none());
        assert!((r.max_outflow_m3s.unwrap() - 200.0).abs() < f64::EPSILON);
        assert!(r.min_generation_mw.is_none());
        assert!((r.max_generation_mw.unwrap() - 80.0).abs() < f64::EPSILON);
        assert!((r.min_diversion_m3s.unwrap() - 2.0).abs() < f64::EPSILON);
        assert!(r.max_diversion_m3s.is_none());
        assert!((r.min_spillage_m3s.unwrap() - 3.0).abs() < f64::EPSILON);
        assert!((r.max_spillage_m3s.unwrap() - 9.0).abs() < f64::EPSILON);
        assert!((r.filling_min_rate_m3s.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!(r.water_withdrawal_m3s.is_none());
    }

    /// AC: only a subset of optional columns present in the schema.
    #[test]
    fn test_hydro_subset_of_optional_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![2_i32])),
                Arc::new(Int32Array::from(vec![3_i32])),
                Arc::new(Float64Array::from(vec![Some(75.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(150.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(2));
        assert_eq!(r.stage_id, 3);
        assert!((r.min_turbined_m3s.unwrap() - 75.0).abs() < f64::EPSILON);
        assert!(r.max_turbined_m3s.is_none());
        assert!(r.min_storage_hm3.is_none());
        assert!(r.max_storage_hm3.is_none());
        assert!(r.min_outflow_m3s.is_none());
        assert!(r.max_outflow_m3s.is_none());
        assert!(r.min_generation_mw.is_none());
        assert!((r.max_generation_mw.unwrap() - 150.0).abs() < f64::EPSILON);
        assert!(r.min_diversion_m3s.is_none());
        assert!(r.max_diversion_m3s.is_none());
        assert!(r.min_spillage_m3s.is_none());
        assert!(r.max_spillage_m3s.is_none());
        assert!(r.filling_min_rate_m3s.is_none());
        assert!(r.water_withdrawal_m3s.is_none());
    }

    /// AC: missing `stage_id` column -> SchemaError with field "stage_id".
    #[test]
    fn test_hydro_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stage_id"),
                    "field should contain 'stage_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in a present optional column -> SchemaError mentioning column and "finite".
    #[test]
    fn test_hydro_nan_in_optional_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_storage_hm3", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(f64::NAN)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_storage_hm3"),
                    "field should contain 'min_storage_hm3', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_hydro_negative_filling_min_rate() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("filling_min_rate_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-5.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("filling_min_rate_m3s"),
                    "field should contain 'filling_min_rate_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should contain '>= 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError for negative filling rate, got: {other:?}"),
        }
    }

    /// AC: negative `min_diversion_m3s` -> SchemaError mentioning the column and ">= 0.0".
    #[test]
    fn test_hydro_negative_min_diversion() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_diversion_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-1.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_diversion_m3s"),
                    "field should contain 'min_diversion_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should contain '>= 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError for negative min_diversion_m3s, got: {other:?}"),
        }
    }

    /// AC: negative `min_spillage_m3s` -> SchemaError mentioning the column and ">= 0.0".
    #[test]
    fn test_hydro_negative_min_spillage() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_spillage_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-2.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_spillage_m3s"),
                    "field should contain 'min_spillage_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should contain '>= 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError for negative min_spillage_m3s, got: {other:?}"),
        }
    }

    /// AC: negative `max_spillage_m3s` -> SchemaError mentioning the column and ">= 0.0".
    #[test]
    fn test_hydro_negative_max_spillage() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_spillage_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-3.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_spillage_m3s"),
                    "field should contain 'max_spillage_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should contain '>= 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError for negative max_spillage_m3s, got: {other:?}"),
        }
    }

    /// AC: `min_spillage_m3s > max_spillage_m3s` (both present, same row) -> SchemaError
    /// whose `field` is `hydro_bounds[0].max_spillage_m3s`.
    #[test]
    fn test_hydro_min_spillage_exceeds_max_spillage() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_spillage_m3s", DataType::Float64, true),
            Field::new("max_spillage_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(9.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(3.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "hydro_bounds[0].max_spillage_m3s");
                assert!(
                    message.contains("must be >= min_spillage_m3s"),
                    "got: {message}"
                );
            }
            other => panic!("expected SchemaError for inverted spillage bounds, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_hydro_empty_parquet() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: scrambled input order -> output sorted by (hydro_id, stage_id).
    #[test]
    fn test_hydro_declaration_order_invariance() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_turbined_m3s", DataType::Float64, true),
        ]));

        let make = |hydro_ids: &[i32], stage_ids: &[i32]| -> RecordBatch {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(hydro_ids.to_vec())),
                    Arc::new(Int32Array::from(stage_ids.to_vec())),
                    Arc::new(Float64Array::from(
                        hydro_ids
                            .iter()
                            .map(|_| Some(100.0_f64))
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()
        };

        let tmp_asc = write_parquet(&make(&[1, 1, 3, 3], &[0, 1, 0, 1]));
        let tmp_desc = write_parquet(&make(&[3, 3, 1, 1], &[1, 0, 1, 0]));
        let rows_asc = parse_hydro_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_hydro_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_hydro_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_hydro_bounds_none() {
        let rows = super::super::load_hydro_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: no `block_id` column in the file — every row's `block_id` is `None`
    /// and every present bound column parses unchanged.
    #[test]
    fn test_hydro_bounds_missing_block_id_column_is_none() {
        let batch = RecordBatch::try_new(
            hydro_full_schema(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])), // min_turbined
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_turbined: null
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])), // min_storage
                Arc::new(Float64Array::from(vec![Some(500.0_f64)])), // max_storage
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_outflow: null
                Arc::new(Float64Array::from(vec![Some(200.0_f64)])), // max_outflow
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_gen: null
                Arc::new(Float64Array::from(vec![Some(80.0_f64)])), // max_gen
                Arc::new(Float64Array::from(vec![Some(2.0_f64)])),  // min_diversion
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_diversion: null
                Arc::new(Float64Array::from(vec![Some(3.0_f64)])),  // min_spillage
                Arc::new(Float64Array::from(vec![Some(9.0_f64)])),  // max_spillage
                Arc::new(Float64Array::from(vec![Some(10.0_f64)])), // filling_min_rate
                Arc::new(Float64Array::from(vec![None::<f64>])),    // water_withdrawal: null
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert!(r.block_id.is_none());
        assert!((r.min_turbined_m3s.unwrap() - 50.0).abs() < f64::EPSILON);
        assert!(r.max_turbined_m3s.is_none());
        assert!((r.min_storage_hm3.unwrap() - 100.0).abs() < f64::EPSILON);
        assert!((r.max_storage_hm3.unwrap() - 500.0).abs() < f64::EPSILON);
        assert!(r.min_outflow_m3s.is_none());
        assert!((r.max_outflow_m3s.unwrap() - 200.0).abs() < f64::EPSILON);
        assert!(r.min_generation_mw.is_none());
        assert!((r.max_generation_mw.unwrap() - 80.0).abs() < f64::EPSILON);
        assert!((r.min_diversion_m3s.unwrap() - 2.0).abs() < f64::EPSILON);
        assert!(r.max_diversion_m3s.is_none());
        assert!((r.min_spillage_m3s.unwrap() - 3.0).abs() < f64::EPSILON);
        assert!((r.max_spillage_m3s.unwrap() - 9.0).abs() < f64::EPSILON);
        assert!((r.filling_min_rate_m3s.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!(r.water_withdrawal_m3s.is_none());
    }

    /// AC: rows with `block_id` set to `0`, `2`, and null parse to `Some(0)`,
    /// `Some(2)`, and `None` respectively.
    #[test]
    fn test_hydro_bounds_block_id_column_is_parsed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_turbined_m3s", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 0_i32, 0_i32])),
                Arc::new(Float64Array::from(vec![
                    Some(100.0_f64),
                    Some(50.0_f64),
                    Some(400.0_f64),
                ])),
                Arc::new(Int32Array::from(vec![Some(0_i32), Some(2_i32), None])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        // Sorted by (hydro_id, stage_id, block_id); all three rows tie on
        // (hydro_id, stage_id), so `None` (the stage-wide row) sorts first,
        // ascending `Some(i)` after it.
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].block_id, None);
        assert_eq!(rows[1].block_id, Some(0));
        assert_eq!(rows[2].block_id, Some(2));
    }

    // ── LineBoundsRow tests ───────────────────────────────────────────────────

    fn line_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("direct_mw", DataType::Float64, true),
            Field::new("reverse_mw", DataType::Float64, true),
        ]))
    }

    fn make_line_batch(
        line_ids: &[i32],
        stage_ids: &[i32],
        direct: Vec<Option<f64>>,
        reverse: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            line_schema(),
            vec![
                Arc::new(Int32Array::from(line_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(direct)),
                Arc::new(Float64Array::from(reverse)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid line bounds, correct sort order.
    #[test]
    fn test_line_valid_rows_sorted() {
        let batch = make_line_batch(
            &[5, 2, 2],
            &[0, 1, 0],
            vec![Some(400.0), Some(300.0), Some(320.0)],
            vec![Some(400.0), None, Some(320.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_line_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].line_id, EntityId::from(2));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].direct_mw.unwrap() - 320.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].line_id, EntityId::from(2));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].reverse_mw.is_none());
        assert_eq!(rows[2].line_id, EntityId::from(5));
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_line_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("direct_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_line_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `direct_mw` -> SchemaError mentioning "direct_mw" and "finite".
    #[test]
    fn test_line_nan_direct_mw() {
        let batch = make_line_batch(&[1], &[0], vec![Some(f64::NAN)], vec![Some(100.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_line_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("direct_mw"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_line_empty_parquet() {
        let batch = make_line_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_line_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_line_declaration_order_invariance() {
        let asc = make_line_batch(
            &[1, 1, 4, 4],
            &[0, 1, 0, 1],
            vec![Some(100.0), Some(110.0), Some(400.0), Some(410.0)],
            vec![Some(100.0), Some(110.0), Some(400.0), Some(410.0)],
        );
        let desc = make_line_batch(
            &[4, 4, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(410.0), Some(400.0), Some(110.0), Some(100.0)],
            vec![Some(410.0), Some(400.0), Some(110.0), Some(100.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_line_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_line_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> =
            rows_asc.iter().map(|r| (r.line_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.line_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_line_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_line_bounds_none() {
        let rows = super::super::load_line_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── PumpingBoundsRow tests ────────────────────────────────────────────────

    fn pumping_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("pumping_station_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_m3s", DataType::Float64, true),
            Field::new("max_m3s", DataType::Float64, true),
        ]))
    }

    fn make_pumping_batch(
        station_ids: &[i32],
        stage_ids: &[i32],
        min_m3s: Vec<Option<f64>>,
        max_m3s: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            pumping_schema(),
            vec![
                Arc::new(Int32Array::from(station_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_m3s)),
                Arc::new(Float64Array::from(max_m3s)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid pumping bounds, correct sort order.
    #[test]
    fn test_pumping_valid_rows_sorted() {
        let batch = make_pumping_batch(
            &[3, 1, 1],
            &[0, 1, 0],
            vec![Some(0.0), None, Some(5.0)],
            vec![Some(80.0), Some(90.0), Some(85.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_pumping_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].station_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].min_m3s.unwrap() - 5.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].station_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].min_m3s.is_none());
        assert_eq!(rows[2].station_id, EntityId::from(3));
    }

    /// Missing id column -> error names the `pumping_station_id` spelling.
    #[test]
    fn test_pumping_missing_id_column_errors_new_spelling() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(5.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(
                    field, "pumping_station_id",
                    "field should be 'pumping_station_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_pumping_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("pumping_station_id", DataType::Int32, false),
            Field::new("min_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(5.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `max_m3s` -> SchemaError mentioning "max_m3s" and "finite".
    #[test]
    fn test_pumping_nan_max_m3s() {
        let batch = make_pumping_batch(&[1], &[0], vec![Some(0.0)], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("max_m3s"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: negative `min_m3s` -> SchemaError mentioning "min_m3s" and ">= 0.0".
    #[test]
    fn test_pumping_negative_min_m3s() {
        let batch = make_pumping_batch(&[1], &[0], vec![Some(-1.0)], vec![Some(10.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("min_m3s"), "got: {field}");
                assert!(message.contains(">= 0.0"), "got: {message}");
            }
            other => panic!("expected SchemaError for negative min, got: {other:?}"),
        }
    }

    /// AC: negative `max_m3s` -> SchemaError mentioning "max_m3s" and ">= 0.0".
    #[test]
    fn test_pumping_negative_max_m3s() {
        let batch = make_pumping_batch(&[1], &[0], vec![None], vec![Some(-5.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("max_m3s"), "got: {field}");
                assert!(message.contains(">= 0.0"), "got: {message}");
            }
            other => panic!("expected SchemaError for negative max, got: {other:?}"),
        }
    }

    /// AC: `min_m3s > max_m3s` (both present) -> SchemaError mentioning "max_m3s".
    #[test]
    fn test_pumping_min_exceeds_max() {
        let batch = make_pumping_batch(&[1], &[0], vec![Some(80.0)], vec![Some(10.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("max_m3s"), "got: {field}");
                assert!(message.contains("must be >= min_m3s"), "got: {message}");
            }
            other => panic!("expected SchemaError for inverted bounds, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_pumping_empty_parquet() {
        let batch = make_pumping_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_pumping_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_pumping_declaration_order_invariance() {
        let asc = make_pumping_batch(
            &[1, 1, 2, 2],
            &[0, 1, 0, 1],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)],
            vec![Some(10.0), Some(10.0), Some(20.0), Some(20.0)],
        );
        let desc = make_pumping_batch(
            &[2, 2, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(1.0), Some(1.0), Some(0.0), Some(0.0)],
            vec![Some(20.0), Some(20.0), Some(10.0), Some(10.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_pumping_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_pumping_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.station_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.station_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_pumping_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_pumping_bounds_none() {
        let rows = super::super::load_pumping_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── ContractBoundsRow tests ───────────────────────────────────────────────

    fn contract_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("contract_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_mw", DataType::Float64, true),
            Field::new("max_mw", DataType::Float64, true),
            Field::new("price_per_mwh", DataType::Float64, true),
        ]))
    }

    fn make_contract_batch(
        contract_ids: &[i32],
        stage_ids: &[i32],
        min_mw: Vec<Option<f64>>,
        max_mw: Vec<Option<f64>>,
        price: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            contract_schema(),
            vec![
                Arc::new(Int32Array::from(contract_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_mw)),
                Arc::new(Float64Array::from(max_mw)),
                Arc::new(Float64Array::from(price)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid contract bounds, correct sort order.
    #[test]
    fn test_contract_valid_rows_sorted() {
        let batch = make_contract_batch(
            &[7, 3, 3],
            &[0, 1, 0],
            vec![Some(0.0), None, Some(0.0)],
            vec![Some(200.0), Some(150.0), Some(180.0)],
            vec![Some(50.0), None, Some(45.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_contract_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].contract_id, EntityId::from(3));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].max_mw.unwrap() - 180.0).abs() < f64::EPSILON);
        assert!((rows[0].price_per_mwh.unwrap() - 45.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].contract_id, EntityId::from(3));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].min_mw.is_none());
        assert!(rows[1].price_per_mwh.is_none());
        assert_eq!(rows[2].contract_id, EntityId::from(7));
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_contract_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("contract_id", DataType::Int32, false),
            Field::new("max_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_contract_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `price_per_mwh` -> SchemaError mentioning "price_per_mwh" and "finite".
    #[test]
    fn test_contract_nan_price() {
        let batch = make_contract_batch(
            &[1],
            &[0],
            vec![Some(0.0)],
            vec![Some(100.0)],
            vec![Some(f64::NAN)],
        );
        let tmp = write_parquet(&batch);
        let err = parse_contract_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("price_per_mwh"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_contract_empty_parquet() {
        let batch = make_contract_batch(&[], &[], vec![], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_contract_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_contract_declaration_order_invariance() {
        let asc = make_contract_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![Some(100.0), Some(100.0), Some(500.0), Some(500.0)],
            vec![Some(10.0), Some(10.0), Some(50.0), Some(50.0)],
        );
        let desc = make_contract_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![Some(500.0), Some(500.0), Some(100.0), Some(100.0)],
            vec![Some(50.0), Some(50.0), Some(10.0), Some(10.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_contract_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_contract_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.contract_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.contract_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_contract_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_contract_bounds_none() {
        let rows = super::super::load_contract_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── Combined line/contract/pumping block_id tests ────────────────────────

    /// AC: `line_bounds.parquet`, `contract_bounds.parquet`, and
    /// `pumping_bounds.parquet` each written without a `block_id` column — every
    /// row's `block_id` is `None` and every other field parses unchanged.
    #[test]
    fn test_line_contract_pumping_missing_block_id_column_is_none() {
        let line_batch = make_line_batch(&[3], &[0], vec![Some(500.0)], vec![Some(400.0)]);
        let line_rows = parse_line_bounds(write_parquet(&line_batch).path()).unwrap();
        assert_eq!(line_rows.len(), 1);
        assert!(line_rows[0].block_id.is_none());
        assert_eq!(line_rows[0].direct_mw, Some(500.0));
        assert_eq!(line_rows[0].reverse_mw, Some(400.0));

        let contract_batch = make_contract_batch(
            &[1],
            &[0],
            vec![Some(0.0)],
            vec![Some(200.0)],
            vec![Some(80.0)],
        );
        let contract_rows = parse_contract_bounds(write_parquet(&contract_batch).path()).unwrap();
        assert_eq!(contract_rows.len(), 1);
        assert!(contract_rows[0].block_id.is_none());
        assert_eq!(contract_rows[0].max_mw, Some(200.0));
        assert_eq!(contract_rows[0].price_per_mwh, Some(80.0));

        let pumping_batch = make_pumping_batch(&[1], &[0], vec![Some(0.0)], vec![Some(100.0)]);
        let pumping_rows = parse_pumping_bounds(write_parquet(&pumping_batch).path()).unwrap();
        assert_eq!(pumping_rows.len(), 1);
        assert!(pumping_rows[0].block_id.is_none());
        assert_eq!(pumping_rows[0].max_m3s, Some(100.0));
    }

    /// AC: each of the three files written with a `block_id` column holding
    /// `Some(2)` and null — the rows carry `Some(2)` and `None` respectively.
    #[test]
    fn test_line_contract_pumping_block_id_column_is_parsed() {
        let line_schema = Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("direct_mw", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let line_batch = RecordBatch::try_new(
            line_schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 0_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64), Some(200.0_f64)])),
                Arc::new(Int32Array::from(vec![Some(2_i32), None])),
            ],
        )
        .unwrap();
        let line_rows = parse_line_bounds(write_parquet(&line_batch).path()).unwrap();
        assert_eq!(line_rows.len(), 2);
        // Sorted by (line_id, stage_id, block_id); both rows tie on (line_id,
        // stage_id), so `None` sorts before `Some(2)`.
        assert_eq!(line_rows[0].block_id, None);
        assert_eq!(line_rows[1].block_id, Some(2));

        let contract_schema = Arc::new(Schema::new(vec![
            Field::new("contract_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_mw", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let contract_batch = RecordBatch::try_new(
            contract_schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 0_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64), Some(60.0_f64)])),
                Arc::new(Int32Array::from(vec![Some(2_i32), None])),
            ],
        )
        .unwrap();
        let contract_rows = parse_contract_bounds(write_parquet(&contract_batch).path()).unwrap();
        assert_eq!(contract_rows.len(), 2);
        assert_eq!(contract_rows[0].block_id, None);
        assert_eq!(contract_rows[1].block_id, Some(2));

        let pumping_schema = Arc::new(Schema::new(vec![
            Field::new("pumping_station_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_m3s", DataType::Float64, true),
            Field::new("block_id", DataType::Int32, true),
        ]));
        let pumping_batch = RecordBatch::try_new(
            pumping_schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 0_i32])),
                Arc::new(Float64Array::from(vec![Some(10.0_f64), Some(20.0_f64)])),
                Arc::new(Int32Array::from(vec![Some(2_i32), None])),
            ],
        )
        .unwrap();
        let pumping_rows = parse_pumping_bounds(write_parquet(&pumping_batch).path()).unwrap();
        assert_eq!(pumping_rows.len(), 2);
        assert_eq!(pumping_rows[0].block_id, None);
        assert_eq!(pumping_rows[1].block_id, Some(2));
    }
}
