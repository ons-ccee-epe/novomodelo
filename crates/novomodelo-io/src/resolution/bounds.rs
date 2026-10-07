//! Bound resolution from base entity values plus stage-varying overrides.
//!
//! [`resolve_bounds`] produces a [`ResolvedBounds`] table for O(1) lookup by
//! `(entity_index, stage_index, block_index)` during LP construction; see its
//! own doc for the four-layer bound-precedence law it populates.

use std::collections::HashMap;

use cobre_core::{
    EntityId,
    entities::{EnergyContract, Hydro, Line, PumpingStation, Thermal},
    resolved::{
        ContractBlockBounds, ContractBlockOverride, HydroBlockBounds, HydroBlockOverride,
        HydroStageBounds, LineBlockBounds, LineBlockOverride, PumpingBlockBounds,
        PumpingBlockOverride, ResolvedBounds, ThermalBlockBounds, ThermalBlockOverride,
        ThermalStageBounds,
    },
};

use crate::constraints::{
    ContractBoundsRow, HydroBoundsRow, LineBoundsRow, PumpingBoundsRow, ThermalBoundsRow,
};

/// Entity slices for bounds resolution. Each must be in the order
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) establishes;
/// slice position becomes the entity index.
pub struct BoundsEntitySlices<'a> {
    /// Hydro plants.
    pub hydros: &'a [Hydro],
    /// Thermal units.
    pub thermals: &'a [Thermal],
    /// Transmission lines.
    pub lines: &'a [Line],
    /// Pumping stations.
    pub pumping_stations: &'a [PumpingStation],
    /// Energy contracts.
    pub contracts: &'a [EnergyContract],
}

/// Per-entity-type stage-varying override rows for bounds resolution.
pub struct BoundsOverrides<'a> {
    /// Hydro bound overrides.
    pub hydro: &'a [HydroBoundsRow],
    /// Thermal bound overrides.
    pub thermal: &'a [ThermalBoundsRow],
    /// Line bound overrides.
    pub line: &'a [LineBoundsRow],
    /// Pumping bound overrides.
    pub pumping: &'a [PumpingBoundsRow],
    /// Contract bound overrides.
    pub contract: &'a [ContractBoundsRow],
}

/// Pre-compute the full bound table from entity base values and stage-varying overrides.
///
/// Resolves every block-eligible bound column via a four-layer precedence law,
/// most specific first:
///
/// 1. per-block override — an override row with `block_id = Some(b)`; replaces
///    layer 2 for block `b` only, never merges with it
/// 2. stage-wide override — an override row with `block_id = None`
/// 3. base value — the entity's own field
/// 4. *(reserved, not yet implemented)* a computed capability derived from unit
///    nominals
///
/// Both override layers are sparse: an absent row falls through to the next
/// layer, never zero. The law is uniform across all five entity families —
/// no per-family exception.
///
/// This function only **populates** the two storages the law reads from — a
/// dense per-`(entity, stage)` table (layers 2/3, pre-merged: base values fill
/// first, then stage-wide overrides patch the columns a row supplies) and,
/// once any row carries a `block_id`, a sparse per-`(entity, stage, block)`
/// overlay (layer 1). It does not itself enforce the law: the two passes
/// write disjoint storage, so their relative order here is immaterial to the
/// result. Precedence is enforced at **read time** by the `*_bounds_at_block`
/// accessors (e.g. [`ResolvedBounds::thermal_bounds_at_block`]):
/// `over.<field>.unwrap_or(cell.<field>)` — the block overlay wins when
/// present, else the pre-merged stage cell. Reversing that fallback, or
/// returning the stage cell without consulting the overlay at all, still
/// type-checks and silently discards every block-specific override.
///
/// Override rows referencing unknown entity IDs, out-of-range stage IDs, or an
/// out-of-range `block_id` are silently skipped (referential integrity is
/// deferred to validation).
///
/// # Arguments
///
/// * `k_max` — maximum lead-stages across anticipated thermals; extends the
///   thermal stage axis to `n_stages + k_max`. Pass `0` when no anticipated
///   thermals are present (the padded region is then empty)
/// * `blocks_per_stage` — per-stage block count, study-stage-ordered; sizes the
///   per-block overlay's block axis and bounds `block_id` validity
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::EntityId;
/// use cobre_core::entities::{
///     EnergyContract, ContractType, Hydro, HydroGenerationModel, HydroPenalties,
///     Line, PumpingStation, Thermal,
/// };
/// use cobre_io::constraints::HydroBoundsRow;
/// use cobre_io::resolution::{resolve_bounds, BoundsEntitySlices, BoundsOverrides};
///
/// let penalties = HydroPenalties {
///     spillage_cost: 0.01,
///     diversion_cost: 0.02,
///     turbined_cost: 0.03,
///     storage_violation_below_cost: 1000.0,
///     filling_target_violation_cost: 5000.0,
///     turbined_violation_below_cost: 500.0,
///     outflow_violation_below_cost: 500.0,
///     outflow_violation_above_cost: 500.0,
///     generation_violation_below_cost: 500.0,
///     evaporation_violation_cost: 500.0,
///     water_withdrawal_violation_cost: 500.0,
///     water_withdrawal_violation_pos_cost: 500.0,
///     water_withdrawal_violation_neg_cost: 500.0,
///     evaporation_violation_pos_cost: 500.0,
///     evaporation_violation_neg_cost: 500.0,
///     inflow_nonnegativity_cost: 1000.0,
/// };
///
/// let hydro = Hydro {
///     id: EntityId::from(0),
///     name: "H0".to_string(),
///     operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     downstream_id: None,
///     travel_time_hours: None,
///     entry_stage_id: None,
///     exit_stage_id: None,
///     min_storage_hm3: 10.0,
///     max_storage_hm3: 200.0,
///     min_outflow_m3s: 0.0,
///     max_outflow_m3s: None,
///     generation_model: HydroGenerationModel::ConstantProductivity,
///     specific_productivity_mw_per_m3s_per_m: None,
///     min_turbined_m3s: 0.0,
///     max_turbined_m3s: 50.0,
///     min_generation_mw: 0.0,
///     max_generation_mw: 50.0,
///     unit_groups: Vec::new(),
///     tailrace: None,
///     hydraulic_losses: None,
///     efficiency: None,
///     evaporation_coefficients_mm: None,
///     evaporation_reference_volumes_hm3: None,
///     diversion: None,
///     filling: None,
///     penalties,
/// };
///
/// let override_row = HydroBoundsRow {
///     hydro_id: EntityId::from(0),
///     stage_id: 1,
///     min_storage_hm3: Some(20.0),
///     max_storage_hm3: None,
///     min_turbined_m3s: None,
///     max_turbined_m3s: None,
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
///
/// let stage_index: std::collections::HashMap<i32, usize> =
///     [(0, 0), (1, 1), (2, 2)].into_iter().collect();
/// let result = resolve_bounds(
///     &BoundsEntitySlices {
///         hydros: &[hydro],
///         thermals: &[],
///         lines: &[],
///         pumping_stations: &[],
///         contracts: &[],
///     },
///     3,
///     0,
///     &stage_index,
///     &BoundsOverrides {
///         hydro: &[override_row],
///         thermal: &[],
///         line: &[],
///         pumping: &[],
///         contract: &[],
///     },
///     &[1, 1, 1],
/// );
///
/// // Stage 0: base value.
/// assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// // Stage 1: overridden.
/// assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
/// // Stage 2: base value.
/// assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// // max_storage_hm3 is unchanged across all stages.
/// assert!((result.hydro_bounds(0, 0).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// assert!((result.hydro_bounds(0, 1).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// assert!((result.hydro_bounds(0, 2).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
/// ```
// too_many_lines: the five per-entity override passes share one stage-index map; splitting
// them would duplicate the lookup and hide the common cascade.
// implicit_hasher: callers pass a concrete `HashMap`; a `BuildHasher` generic buys nothing.
#[must_use]
#[allow(clippy::too_many_lines, clippy::implicit_hasher)]
pub fn resolve_bounds(
    entities: &BoundsEntitySlices<'_>,
    n_stages: usize,
    k_max: usize,
    stage_index: &HashMap<i32, usize>,
    overrides: &BoundsOverrides<'_>,
    blocks_per_stage: &[usize],
) -> ResolvedBounds {
    let hydro_overrides = overrides.hydro;
    let thermal_overrides = overrides.thermal;
    let line_overrides = overrides.line;
    let pumping_overrides = overrides.pumping;
    let contract_overrides = overrides.contract;

    let hydro_index: HashMap<EntityId, usize> = entities
        .hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();
    let thermal_index: HashMap<EntityId, usize> = entities
        .thermals
        .iter()
        .enumerate()
        .map(|(idx, t)| (t.id, idx))
        .collect();
    let line_index: HashMap<EntityId, usize> = entities
        .lines
        .iter()
        .enumerate()
        .map(|(idx, l)| (l.id, idx))
        .collect();
    let pumping_index: HashMap<EntityId, usize> = entities
        .pumping_stations
        .iter()
        .enumerate()
        .map(|(idx, p)| (p.id, idx))
        .collect();
    let contract_index: HashMap<EntityId, usize> = entities
        .contracts
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.id, idx))
        .collect();

    // ResolvedBounds::new fills every cell with one repeated default; the per-entity
    // fill loop below overwrites them all, so this default is allocation-only.
    let hydro_default = entities
        .hydros
        .first()
        .map_or(zero_hydro_stage_bounds(), hydro_base_stage_bounds);
    let hydro_block_default = entities
        .hydros
        .first()
        .map_or(zero_hydro_block_bounds(), hydro_base_block_bounds);
    let thermal_default =
        entities
            .thermals
            .first()
            .map_or(ThermalStageBounds { cost_per_mwh: 0.0 }, |t| {
                ThermalStageBounds {
                    cost_per_mwh: t.cost_per_mwh,
                }
            });
    let thermal_block_default = entities.thermals.first().map_or(
        ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
        },
        |t| ThermalBlockBounds {
            min_generation_mw: t.min_generation_mw,
            max_generation_mw: t.max_generation_mw,
        },
    );
    let line_default = entities.lines.first().map_or(
        LineBlockBounds {
            direct_mw: 0.0,
            reverse_mw: 0.0,
        },
        |l| LineBlockBounds {
            direct_mw: l.direct_capacity_mw,
            reverse_mw: l.reverse_capacity_mw,
        },
    );
    let pumping_default = entities.pumping_stations.first().map_or(
        PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        },
        |p| PumpingBlockBounds {
            min_flow_m3s: p.min_flow_m3s,
            max_flow_m3s: p.max_flow_m3s,
        },
    );
    let contract_default = entities.contracts.first().map_or(
        ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        },
        |c| ContractBlockBounds {
            min_mw: c.min_mw,
            max_mw: c.max_mw,
            price_per_mwh: c.price_per_mwh,
        },
    );

    // n_stages == 0: allocate with 1 (the API needs a non-zero stride) then return
    // the empty table without filling.
    let alloc_stages = n_stages.max(1);

    let mut table = ResolvedBounds::new(
        &cobre_core::BoundsCountsSpec {
            n_hydros: entities.hydros.len(),
            n_thermals: entities.thermals.len(),
            n_lines: entities.lines.len(),
            n_pumping: entities.pumping_stations.len(),
            n_contracts: entities.contracts.len(),
            n_stages: alloc_stages,
            k_max,
        },
        &cobre_core::BoundsDefaults {
            hydro: hydro_default,
            hydro_block: hydro_block_default,
            thermal: thermal_default,
            thermal_block: thermal_block_default,
            line_block: line_default,
            pumping_block: pumping_default,
            contract_block: contract_default,
        },
    );

    if n_stages == 0 {
        return table;
    }

    for (entity_idx, hydro) in entities.hydros.iter().enumerate() {
        let base_stage = hydro_base_stage_bounds(hydro);
        let base_block = hydro_base_block_bounds(hydro);
        for stage_idx in 0..n_stages {
            *table.hydro_bounds_mut(entity_idx, stage_idx) = base_stage;
            *table.hydro_block_base_mut(entity_idx, stage_idx) = base_block;
        }
    }

    // Padded cells [n_stages, n_stages + k_max) carry each thermal's own base values
    // so LP lookups at delivery stage t + K_i (anticipated thermals) stay in range.
    let thermal_axis = n_stages + k_max;
    for (entity_idx, thermal) in entities.thermals.iter().enumerate() {
        let base_stage = ThermalStageBounds {
            cost_per_mwh: thermal.cost_per_mwh,
        };
        let base_block = ThermalBlockBounds {
            min_generation_mw: thermal.min_generation_mw,
            max_generation_mw: thermal.max_generation_mw,
        };
        for stage_idx in 0..thermal_axis {
            *table.thermal_bounds_mut(entity_idx, stage_idx) = base_stage;
            *table.thermal_block_base_mut(entity_idx, stage_idx) = base_block;
        }
    }

    for (entity_idx, line) in entities.lines.iter().enumerate() {
        let base = LineBlockBounds {
            direct_mw: line.direct_capacity_mw,
            reverse_mw: line.reverse_capacity_mw,
        };
        for stage_idx in 0..n_stages {
            *table.line_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, pumping) in entities.pumping_stations.iter().enumerate() {
        let base = PumpingBlockBounds {
            min_flow_m3s: pumping.min_flow_m3s,
            max_flow_m3s: pumping.max_flow_m3s,
        };
        for stage_idx in 0..n_stages {
            *table.pumping_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    for (entity_idx, contract) in entities.contracts.iter().enumerate() {
        let base = ContractBlockBounds {
            min_mw: contract.min_mw,
            max_mw: contract.max_mw,
            price_per_mwh: contract.price_per_mwh,
        };
        for stage_idx in 0..n_stages {
            *table.contract_bounds_mut(entity_idx, stage_idx) = base;
        }
    }

    let max_blocks = blocks_per_stage.iter().copied().max().unwrap_or(0);
    let any_block_row = thermal_overrides.iter().any(|row| row.block_id.is_some())
        || hydro_overrides.iter().any(|row| row.block_id.is_some())
        || line_overrides.iter().any(|row| row.block_id.is_some())
        || pumping_overrides.iter().any(|row| row.block_id.is_some())
        || contract_overrides.iter().any(|row| row.block_id.is_some());
    let mut block_overlay = (any_block_row && max_blocks > 0).then(|| {
        cobre_core::ResolvedBlockBounds::new(&cobre_core::BlockBoundsCountsSpec {
            n_hydros: entities.hydros.len(),
            n_thermals: entities.thermals.len(),
            n_lines: entities.lines.len(),
            n_pumping: entities.pumping_stations.len(),
            n_contracts: entities.contracts.len(),
            n_stages,
            max_blocks,
        })
    });

    for row in hydro_overrides.iter().filter(|row| row.block_id.is_none()) {
        let Some(&entity_idx) = hydro_index.get(&row.hydro_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.hydro_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_storage_hm3 {
            cell.min_storage_hm3 = v;
        }
        if let Some(v) = row.max_storage_hm3 {
            cell.max_storage_hm3 = v;
        }
        if let Some(v) = row.filling_min_rate_m3s {
            cell.filling_min_rate_m3s = v;
        }
        if let Some(v) = row.water_withdrawal_m3s {
            cell.water_withdrawal_m3s = v;
        }

        let block_cell = table.hydro_block_base_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_turbined_m3s {
            block_cell.min_turbined_m3s = v;
        }
        if let Some(v) = row.max_turbined_m3s {
            block_cell.max_turbined_m3s = v;
        }
        if let Some(v) = row.min_outflow_m3s {
            block_cell.min_outflow_m3s = v;
        }
        if let Some(v) = row.max_outflow_m3s {
            block_cell.max_outflow_m3s = Some(v);
        }
        if let Some(v) = row.min_generation_mw {
            block_cell.min_generation_mw = v;
        }
        if let Some(v) = row.max_generation_mw {
            block_cell.max_generation_mw = v;
        }
        if let Some(v) = row.min_diversion_m3s {
            block_cell.min_diversion_m3s = Some(v);
        }
        if let Some(v) = row.max_diversion_m3s {
            block_cell.max_diversion_m3s = Some(v);
        }
        if let Some(v) = row.min_spillage_m3s {
            block_cell.min_spillage_m3s = Some(v);
        }
        if let Some(v) = row.max_spillage_m3s {
            block_cell.max_spillage_m3s = Some(v);
        }
    }

    if let Some(overlay) = block_overlay.as_mut() {
        for row in hydro_overrides {
            let Some((entity_idx, stage_idx, block_idx)) = block_slot(
                row.block_id,
                row.hydro_id,
                row.stage_id,
                &hydro_index,
                stage_index,
                blocks_per_stage,
            ) else {
                continue;
            };
            let Some(cell) = overlay.hydro_override_mut(entity_idx, stage_idx, block_idx) else {
                continue;
            };
            apply_hydro_block_row(row, cell);
        }
    }

    for row in thermal_overrides
        .iter()
        .filter(|row| row.block_id.is_none())
    {
        let Some(&entity_idx) = thermal_index.get(&row.thermal_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        if let Some(v) = row.cost_per_mwh {
            table.thermal_bounds_mut(entity_idx, stage_idx).cost_per_mwh = v;
        }
        if let Some(v) = row.min_generation_mw {
            table
                .thermal_block_base_mut(entity_idx, stage_idx)
                .min_generation_mw = v;
        }
        if let Some(v) = row.max_generation_mw {
            table
                .thermal_block_base_mut(entity_idx, stage_idx)
                .max_generation_mw = v;
        }
    }

    if let Some(overlay) = block_overlay.as_mut() {
        for row in thermal_overrides {
            let Some((entity_idx, stage_idx, block_idx)) = block_slot(
                row.block_id,
                row.thermal_id,
                row.stage_id,
                &thermal_index,
                stage_index,
                blocks_per_stage,
            ) else {
                continue;
            };
            let Some(cell) = overlay.thermal_override_mut(entity_idx, stage_idx, block_idx) else {
                continue;
            };
            apply_thermal_block_row(row, cell);
        }
    }

    for row in line_overrides.iter().filter(|row| row.block_id.is_none()) {
        let Some(&entity_idx) = line_index.get(&row.line_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.line_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.direct_mw {
            cell.direct_mw = v;
        }
        if let Some(v) = row.reverse_mw {
            cell.reverse_mw = v;
        }
    }

    if let Some(overlay) = block_overlay.as_mut() {
        for row in line_overrides {
            let Some((entity_idx, stage_idx, block_idx)) = block_slot(
                row.block_id,
                row.line_id,
                row.stage_id,
                &line_index,
                stage_index,
                blocks_per_stage,
            ) else {
                continue;
            };
            let Some(cell) = overlay.line_override_mut(entity_idx, stage_idx, block_idx) else {
                continue;
            };
            apply_line_block_row(row, cell);
        }
    }

    for row in pumping_overrides
        .iter()
        .filter(|row| row.block_id.is_none())
    {
        let Some(&entity_idx) = pumping_index.get(&row.station_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.pumping_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_m3s {
            cell.min_flow_m3s = v;
        }
        if let Some(v) = row.max_m3s {
            cell.max_flow_m3s = v;
        }
    }

    if let Some(overlay) = block_overlay.as_mut() {
        for row in pumping_overrides {
            let Some((entity_idx, stage_idx, block_idx)) = block_slot(
                row.block_id,
                row.station_id,
                row.stage_id,
                &pumping_index,
                stage_index,
                blocks_per_stage,
            ) else {
                continue;
            };
            let Some(cell) = overlay.pumping_override_mut(entity_idx, stage_idx, block_idx) else {
                continue;
            };
            apply_pumping_block_row(row, cell);
        }
    }

    for row in contract_overrides
        .iter()
        .filter(|row| row.block_id.is_none())
    {
        let Some(&entity_idx) = contract_index.get(&row.contract_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let cell = table.contract_bounds_mut(entity_idx, stage_idx);
        if let Some(v) = row.min_mw {
            cell.min_mw = v;
        }
        if let Some(v) = row.max_mw {
            cell.max_mw = v;
        }
        if let Some(v) = row.price_per_mwh {
            cell.price_per_mwh = v;
        }
    }

    if let Some(overlay) = block_overlay.as_mut() {
        for row in contract_overrides {
            let Some((entity_idx, stage_idx, block_idx)) = block_slot(
                row.block_id,
                row.contract_id,
                row.stage_id,
                &contract_index,
                stage_index,
                blocks_per_stage,
            ) else {
                continue;
            };
            let Some(cell) = overlay.contract_override_mut(entity_idx, stage_idx, block_idx) else {
                continue;
            };
            apply_contract_block_row(row, cell);
        }
    }

    if let Some(overlay) = block_overlay {
        table.set_block_overlay(overlay);
    }

    table
}

/// Resolve an override row's `(entity_idx, stage_idx, block_idx)` overlay slot;
/// `None` skips the row (no `block_id`, unknown entity or stage, or a `block_id`
/// outside the stage's block count).
fn block_slot(
    block_id: Option<i32>,
    entity_id: EntityId,
    stage_id: i32,
    entity_index: &HashMap<EntityId, usize>,
    stage_index: &HashMap<i32, usize>,
    blocks_per_stage: &[usize],
) -> Option<(usize, usize, usize)> {
    let block_id = block_id?;
    let &entity_idx = entity_index.get(&entity_id)?;
    let &stage_idx = stage_index.get(&stage_id)?;
    let block_idx = usize::try_from(block_id).ok()?;
    if block_idx >= *blocks_per_stage.get(stage_idx)? {
        return None;
    }
    Some((entity_idx, stage_idx, block_idx))
}

/// Write a hydro block row's present columns into a per-block overlay cell.
/// Only the block-eligible columns are read from `row` — every field on
/// [`HydroBlockOverride`]; the four stage-level columns (`min`/
/// `max_storage_hm3`, `filling_min_rate_m3s`, `water_withdrawal_m3s`) have no
/// field there and are never written here.
fn apply_hydro_block_row(row: &HydroBoundsRow, over: &mut HydroBlockOverride) {
    if let Some(v) = row.min_turbined_m3s {
        over.min_turbined_m3s = Some(v);
    }
    if let Some(v) = row.max_turbined_m3s {
        over.max_turbined_m3s = Some(v);
    }
    if let Some(v) = row.min_outflow_m3s {
        over.min_outflow_m3s = Some(v);
    }
    if let Some(v) = row.max_outflow_m3s {
        over.max_outflow_m3s = Some(v);
    }
    if let Some(v) = row.min_generation_mw {
        over.min_generation_mw = Some(v);
    }
    if let Some(v) = row.max_generation_mw {
        over.max_generation_mw = Some(v);
    }
    if let Some(v) = row.min_diversion_m3s {
        over.min_diversion_m3s = Some(v);
    }
    if let Some(v) = row.max_diversion_m3s {
        over.max_diversion_m3s = Some(v);
    }
    if let Some(v) = row.min_spillage_m3s {
        over.min_spillage_m3s = Some(v);
    }
    if let Some(v) = row.max_spillage_m3s {
        over.max_spillage_m3s = Some(v);
    }
}

fn apply_thermal_block_row(row: &ThermalBoundsRow, over: &mut ThermalBlockOverride) {
    if let Some(v) = row.min_generation_mw {
        over.min_generation_mw = Some(v);
    }
    if let Some(v) = row.max_generation_mw {
        over.max_generation_mw = Some(v);
    }
}

fn apply_line_block_row(row: &LineBoundsRow, over: &mut LineBlockOverride) {
    if let Some(v) = row.direct_mw {
        over.direct_mw = Some(v);
    }
    if let Some(v) = row.reverse_mw {
        over.reverse_mw = Some(v);
    }
}

fn apply_pumping_block_row(row: &PumpingBoundsRow, over: &mut PumpingBlockOverride) {
    if let Some(v) = row.min_m3s {
        over.min_flow_m3s = Some(v);
    }
    if let Some(v) = row.max_m3s {
        over.max_flow_m3s = Some(v);
    }
}

/// `price_per_mwh` is block-eligible here, asymmetric with
/// `ThermalBoundsRow.cost_per_mwh`, which stays stage-level.
fn apply_contract_block_row(row: &ContractBoundsRow, over: &mut ContractBlockOverride) {
    if let Some(v) = row.min_mw {
        over.min_mw = Some(v);
    }
    if let Some(v) = row.max_mw {
        over.max_mw = Some(v);
    }
    if let Some(v) = row.price_per_mwh {
        over.price_per_mwh = Some(v);
    }
}

/// Derive the base stage-level [`HydroStageBounds`] from a `Hydro` entity's fields.
///
/// `filling_min_rate_m3s` is `0.0` without a filling config; `water_withdrawal_m3s`
/// has no entity default (`0.0`, set only via overrides).
#[inline]
fn hydro_base_stage_bounds(hydro: &Hydro) -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: hydro.min_storage_hm3,
        max_storage_hm3: hydro.max_storage_hm3,
        filling_min_rate_m3s: hydro.filling.map_or(0.0, |f| f.filling_min_rate_m3s),
        water_withdrawal_m3s: 0.0,
    }
}

/// Derive the base block-eligible [`HydroBlockBounds`] from a `Hydro` entity's
/// fields. `max_diversion_m3s` is `None` without a diversion channel;
/// `min_diversion_m3s`, `min_spillage_m3s`, and `max_spillage_m3s` have no
/// entity-level declaration source at all, so they are always `None` here —
/// `None` on that axis means "unbounded on that side".
#[inline]
fn hydro_base_block_bounds(hydro: &Hydro) -> HydroBlockBounds {
    HydroBlockBounds {
        min_turbined_m3s: hydro.min_turbined_m3s,
        max_turbined_m3s: hydro.max_turbined_m3s,
        min_outflow_m3s: hydro.min_outflow_m3s,
        max_outflow_m3s: hydro.max_outflow_m3s,
        min_generation_mw: hydro.min_generation_mw,
        max_generation_mw: hydro.max_generation_mw,
        min_diversion_m3s: None,
        max_diversion_m3s: hydro.diversion.as_ref().map(|d| d.max_flow_m3s),
        min_spillage_m3s: None,
        max_spillage_m3s: None,
    }
}

/// Return a zero-valued [`HydroStageBounds`] sentinel used when no hydro entities exist.
#[inline]
fn zero_hydro_stage_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 0.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

/// Return a zero-valued [`HydroBlockBounds`] sentinel used when no hydro entities exist.
#[inline]
fn zero_hydro_block_bounds() -> HydroBlockBounds {
    HydroBlockBounds {
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 0.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 0.0,
        min_diversion_m3s: None,
        max_diversion_m3s: None,
        min_spillage_m3s: None,
        max_spillage_m3s: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use chrono::NaiveDate;
    use std::collections::HashMap;

    use cobre_core::{
        EntityId,
        entities::{
            ContractType, DiversionChannel, EnergyContract, FillingConfig, Hydro,
            HydroGenerationModel, HydroPenalties, Line, PumpingStation, Thermal,
        },
        resolved::ResolvedBounds,
    };

    use crate::constraints::{
        ContractBoundsRow, HydroBoundsRow, LineBoundsRow, PumpingBoundsRow, ThermalBoundsRow,
    };

    /// Build a 0-based consecutive stage_index map: {0→0, 1→1, …, (n-1)→(n-1)}.
    fn si(n: usize) -> HashMap<i32, usize> {
        (0..n).map(|i| (i32::try_from(i).unwrap(), i)).collect()
    }

    /// Test wrapper that passes a 0-based consecutive `stage_index` to
    /// [`super::resolve_bounds`].
    #[allow(clippy::too_many_arguments)]
    fn resolve_bounds(
        hydros: &[Hydro],
        thermals: &[Thermal],
        lines: &[Line],
        pumping_stations: &[PumpingStation],
        contracts: &[EnergyContract],
        n_stages: usize,
        k_max: usize,
        hydro_overrides: &[HydroBoundsRow],
        thermal_overrides: &[ThermalBoundsRow],
        line_overrides: &[LineBoundsRow],
        pumping_overrides: &[PumpingBoundsRow],
        contract_overrides: &[ContractBoundsRow],
        blocks_per_stage: &[usize],
    ) -> ResolvedBounds {
        super::resolve_bounds(
            &super::BoundsEntitySlices {
                hydros,
                thermals,
                lines,
                pumping_stations,
                contracts,
            },
            n_stages,
            k_max,
            &si(n_stages),
            &super::BoundsOverrides {
                hydro: hydro_overrides,
                thermal: thermal_overrides,
                line: line_overrides,
                pumping: pumping_overrides,
                contract: contract_overrides,
            },
            blocks_per_stage,
        )
    }

    // ── Entity construction helpers ───────────────────────────────────────────

    fn make_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.02,
            turbined_cost: 0.03,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 5000.0,
            turbined_violation_below_cost: 500.0,
            outflow_violation_below_cost: 500.0,
            outflow_violation_above_cost: 500.0,
            generation_violation_below_cost: 500.0,
            evaporation_violation_cost: 500.0,
            water_withdrawal_violation_cost: 500.0,
            water_withdrawal_violation_pos_cost: 500.0,
            water_withdrawal_violation_neg_cost: 500.0,
            evaporation_violation_pos_cost: 500.0,
            evaporation_violation_neg_cost: 500.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(
        id: i32,
        min_storage_hm3: f64,
        max_storage_hm3: f64,
        max_outflow_m3s: Option<f64>,
        diversion: Option<DiversionChannel>,
        filling: Option<FillingConfig>,
    ) -> Hydro {
        Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3,
            max_storage_hm3,
            min_outflow_m3s: 0.0,
            max_outflow_m3s,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion,
            filling,
            penalties: make_penalties(),
        }
    }

    fn make_thermal(id: i32, min_generation_mw: f64, max_generation_mw: f64) -> Thermal {
        Thermal {
            id: EntityId::from(id),
            name: format!("Thermal {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw,
            max_generation_mw,
            anticipated_config: None,
        }
    }

    fn make_line(id: i32, direct_mw: f64, reverse_mw: f64) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: direct_mw,
            reverse_capacity_mw: reverse_mw,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_pumping(id: i32, min_flow_m3s: f64, max_flow_m3s: f64) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pumping {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            source_hydro_id: EntityId::from(1),
            destination_hydro_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s,
            max_flow_m3s,
        }
    }

    fn make_contract(id: i32, min_mw: f64, max_mw: f64, price_per_mwh: f64) -> EnergyContract {
        EnergyContract {
            id: EntityId::from(id),
            name: format!("Contract {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh,
            min_mw,
            max_mw,
        }
    }

    fn all_none_hydro_row(hydro_id: i32, stage_id: i32) -> HydroBoundsRow {
        HydroBoundsRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            ..Default::default()
        }
    }

    // ── Test: base values only (no overrides) ──────────────────────────────────

    /// Given 2 hydros, 2 thermals, 1 line, 1 pumping, 1 contract, 3 stages, and no
    /// overrides, all cells equal each entity's base values.
    #[test]
    fn test_base_values_no_overrides() {
        let hydros = vec![
            make_hydro(0, 10.0, 100.0, None, None, None),
            make_hydro(1, 20.0, 200.0, None, None, None),
        ];
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 50.0, 600.0)];
        let lines = vec![make_line(0, 1000.0, 800.0)];
        let pumpings = vec![make_pumping(0, 0.0, 50.0)];
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];

        let result = resolve_bounds(
            &hydros,
            &thermals,
            &lines,
            &pumpings,
            &contracts,
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..3 {
            assert!(
                (result.hydro_bounds(0, stage).min_storage_hm3 - 10.0).abs() < f64::EPSILON,
                "hydro 0 stage {stage}: expected min_storage_hm3=10.0"
            );
            assert!(
                (result.hydro_bounds(0, stage).max_storage_hm3 - 100.0).abs() < f64::EPSILON,
                "hydro 0 stage {stage}: expected max_storage_hm3=100.0"
            );
            assert!(
                (result.hydro_bounds(1, stage).min_storage_hm3 - 20.0).abs() < f64::EPSILON,
                "hydro 1 stage {stage}: expected min_storage_hm3=20.0"
            );
            assert!(
                (result.hydro_bounds(1, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON,
                "hydro 1 stage {stage}: expected max_storage_hm3=200.0"
            );
            assert!(
                (result.thermal_block_base(0, stage).max_generation_mw - 400.0).abs()
                    < f64::EPSILON,
                "thermal 0 stage {stage}: expected max_generation_mw=400.0"
            );
            assert!(
                (result.thermal_block_base(1, stage).min_generation_mw - 50.0).abs() < f64::EPSILON,
                "thermal 1 stage {stage}: expected min_generation_mw=50.0"
            );
            assert!(
                (result.thermal_block_base(1, stage).max_generation_mw - 600.0).abs()
                    < f64::EPSILON,
                "thermal 1 stage {stage}: expected max_generation_mw=600.0"
            );
            assert!(
                (result.line_block_base(0, stage).direct_mw - 1000.0).abs() < f64::EPSILON,
                "line 0 stage {stage}: expected direct_mw=1000.0"
            );
            assert!(
                (result.line_block_base(0, stage).reverse_mw - 800.0).abs() < f64::EPSILON,
                "line 0 stage {stage}: expected reverse_mw=800.0"
            );
            assert!(
                (result.pumping_block_base(0, stage).max_flow_m3s - 50.0).abs() < f64::EPSILON,
                "pumping 0 stage {stage}: expected max_flow_m3s=50.0"
            );
            assert!(
                (result.contract_block_base(0, stage).max_mw - 200.0).abs() < f64::EPSILON,
                "contract 0 stage {stage}: expected max_mw=200.0"
            );
            assert!(
                (result.contract_block_base(0, stage).price_per_mwh - 80.0).abs() < f64::EPSILON,
                "contract 0 stage {stage}: expected price_per_mwh=80.0"
            );
        }
    }

    // ── Test: single hydro override ────────────────────────────────────────────

    /// Given 1 hydro, 3 stages, override min_storage_hm3 at stage 1 only.
    /// Only that cell changes; others retain base value.
    #[test]
    fn test_single_hydro_override() {
        let hydros = vec![make_hydro(0, 10.0, 200.0, None, None, None)];
        let override_row = HydroBoundsRow {
            min_storage_hm3: Some(20.0),
            ..all_none_hydro_row(0, 1)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            3,
            0,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        for stage in 0..3 {
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: hydro with diversion and filling ─────────────────────────────────

    /// Given a hydro with a diversion channel and filling config, base bounds correctly
    /// derive max_diversion_m3s and filling_min_rate_m3s.
    #[test]
    fn test_hydro_with_diversion_and_filling() {
        let diversion = DiversionChannel {
            downstream_id: EntityId::from(2),
            max_flow_m3s: 50.0,
        };
        let filling = FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 30.0,
        };
        let hydros = vec![make_hydro(
            0,
            10.0,
            100.0,
            None,
            Some(diversion),
            Some(filling),
        )];

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        let b = result.hydro_bounds(0, 0);
        assert_eq!(result.hydro_block_base(0, 0).max_diversion_m3s, Some(50.0));
        assert!((b.filling_min_rate_m3s - 30.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: hydro without diversion ─────────────────────────────────────────

    /// Given a hydro with no diversion channel and no filling config, base bounds
    /// have max_diversion_m3s = None and filling_min_rate_m3s = 0.0.
    #[test]
    fn test_hydro_without_diversion() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        let b = result.hydro_bounds(0, 0);
        assert!(result.hydro_block_base(0, 0).max_diversion_m3s.is_none());
        assert!((b.filling_min_rate_m3s - 0.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: thermal override ─────────────────────────────────────────────────

    /// Given 2 thermals, 2 stages, override max_generation_mw for thermal_id=1 at stage=0.
    /// Only that cell changes; thermal_id=0 at all stages and thermal_id=1 at stage=1
    /// retain base values.
    #[test]
    fn test_thermal_override() {
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 50.0, 600.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(500.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.thermal_block_base(1, 0).max_generation_mw - 500.0).abs() < f64::EPSILON);
        assert!((result.thermal_block_base(0, 0).max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!((result.thermal_block_base(1, 1).max_generation_mw - 600.0).abs() < f64::EPSILON);
    }

    // ── Test: line override ────────────────────────────────────────────────────

    /// Given 1 line, 3 stages, override direct_mw at stage=1.
    #[test]
    fn test_line_override() {
        let lines = vec![make_line(0, 1000.0, 800.0)];
        let override_row = LineBoundsRow {
            line_id: EntityId::from(0),
            stage_id: 1,
            direct_mw: Some(750.0),
            reverse_mw: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &lines,
            &[],
            &[],
            3,
            0,
            &[],
            &[],
            &[override_row],
            &[],
            &[],
            &[],
        );

        assert!((result.line_block_base(0, 0).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((result.line_block_base(0, 1).direct_mw - 750.0).abs() < f64::EPSILON);
        assert!((result.line_block_base(0, 2).direct_mw - 1000.0).abs() < f64::EPSILON);
        for stage in 0..3 {
            assert!((result.line_block_base(0, stage).reverse_mw - 800.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: pumping override ─────────────────────────────────────────────────

    /// Given 1 pumping station, 2 stages, override max_m3s (→ max_flow_m3s) at stage=0.
    /// Note: PumpingBoundsRow uses min_m3s / max_m3s column names which map to
    /// PumpingBlockBounds.min_flow_m3s / max_flow_m3s respectively.
    #[test]
    fn test_pumping_override() {
        let pumpings = vec![make_pumping(0, 0.0, 50.0)];
        let override_row = PumpingBoundsRow {
            station_id: EntityId::from(0),
            stage_id: 0,
            min_m3s: None,
            max_m3s: Some(100.0),
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &pumpings,
            &[],
            2,
            0,
            &[],
            &[],
            &[],
            &[override_row],
            &[],
            &[],
        );

        assert!((result.pumping_block_base(0, 0).max_flow_m3s - 100.0).abs() < f64::EPSILON);
        assert!((result.pumping_block_base(0, 1).max_flow_m3s - 50.0).abs() < f64::EPSILON);
        for stage in 0..2 {
            assert!((result.pumping_block_base(0, stage).min_flow_m3s - 0.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: contract override with price ─────────────────────────────────────

    /// Given 1 contract, 3 stages, override price_per_mwh at stage=1.
    #[test]
    fn test_contract_override_with_price() {
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let override_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 1,
            min_mw: None,
            max_mw: None,
            price_per_mwh: Some(90.0),
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &contracts,
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[override_row],
            &[],
        );

        assert!((result.contract_block_base(0, 0).price_per_mwh - 80.0).abs() < f64::EPSILON);
        assert!((result.contract_block_base(0, 1).price_per_mwh - 90.0).abs() < f64::EPSILON);
        assert!((result.contract_block_base(0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON);
        for stage in 0..3 {
            assert!((result.contract_block_base(0, stage).min_mw - 0.0).abs() < f64::EPSILON);
            assert!((result.contract_block_base(0, stage).max_mw - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: unknown entity_id silently skipped ───────────────────────────────

    /// Given an override row with entity_id not in any registry, no panic and
    /// the result is unchanged for all known entities.
    #[test]
    fn test_unknown_entity_id_skipped() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];
        let override_row = HydroBoundsRow {
            hydro_id: EntityId::from(999), // Not in registry.
            min_storage_hm3: Some(9999.0),
            ..all_none_hydro_row(999, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            0,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        // Hydro 0 is unchanged.
        assert!(
            (result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON,
            "unknown entity ID must not affect known entities"
        );
    }

    // ── Test: empty entities with non-empty overrides ─────────────────────────

    /// Given 0 thermals and non-empty thermal overrides, no panic occurs and the
    /// result has an empty thermal dimension.
    #[test]
    fn test_empty_entities_no_panic() {
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: Some(10.0),
            max_generation_mw: Some(200.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &[],
            3,
            0,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert_eq!(result.n_stages(), 3);
        // `k_max == 0` parity at the resolver entry point: thermal stage axis
        // matches `n_stages` when no anticipated thermals are present.
        assert_eq!(result.thermal_stage_axis_len(), 3);
        // No thermal dimension to query — verifying no panic above is sufficient.
    }

    // ── Test: acceptance criterion — AC1: hydro min_storage_hm3 override ───────

    /// AC1: 1 hydro, min_storage_hm3=10.0, max_storage_hm3=200.0, 3 stages,
    /// override at stage_id=1 with min_storage_hm3=Some(20.0), all other fields None.
    #[test]
    fn test_ac1_hydro_storage_override() {
        let hydros = vec![make_hydro(0, 10.0, 200.0, None, None, None)];
        let override_row = HydroBoundsRow {
            min_storage_hm3: Some(20.0),
            ..all_none_hydro_row(0, 1)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            3,
            0,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 1).min_storage_hm3 - 20.0).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 2).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        for stage in 0..3 {
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        }
    }

    // ── Test: acceptance criterion — AC2: thermal max_generation_mw override ───

    /// AC2: 2 thermals (max=400.0 and 600.0), 2 stages,
    /// override thermal_id=1 at stage_id=0, max_generation_mw=Some(500.0).
    #[test]
    fn test_ac2_thermal_max_generation_override() {
        let thermals = vec![make_thermal(0, 0.0, 400.0), make_thermal(1, 0.0, 600.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(1),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(500.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.thermal_block_base(1, 0).max_generation_mw - 500.0).abs() < f64::EPSILON);
        assert!((result.thermal_block_base(0, 0).max_generation_mw - 400.0).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — AC3: hydro with diversion no filling ──────

    /// AC3: hydro with diversion channel (max_flow_m3s=50.0) and no filling config.
    #[test]
    fn test_ac3_hydro_diversion_no_filling() {
        let diversion = DiversionChannel {
            downstream_id: EntityId::from(2),
            max_flow_m3s: 50.0,
        };
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, Some(diversion), None)];

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        let b = result.hydro_bounds(0, 0);
        assert_eq!(result.hydro_block_base(0, 0).max_diversion_m3s, Some(50.0));
        assert!((b.filling_min_rate_m3s - 0.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Test: acceptance criterion — AC4: empty overrides, base values ─────────

    /// AC4: all five override slices empty, every cell equals entity base value.
    #[test]
    fn test_ac4_empty_overrides_base_values() {
        let hydros = vec![make_hydro(0, 5.0, 50.0, None, None, None)];
        let thermals = vec![make_thermal(0, 10.0, 200.0)];
        let lines = vec![make_line(0, 500.0, 300.0)];
        let pumpings = vec![make_pumping(0, 0.0, 75.0)];
        let contracts = vec![make_contract(0, 0.0, 100.0, 60.0)];

        let result = resolve_bounds(
            &hydros,
            &thermals,
            &lines,
            &pumpings,
            &contracts,
            2,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..2 {
            assert!((result.hydro_bounds(0, stage).min_storage_hm3 - 5.0).abs() < f64::EPSILON);
            assert!((result.hydro_bounds(0, stage).max_storage_hm3 - 50.0).abs() < f64::EPSILON);
            assert!(
                (result.thermal_block_base(0, stage).min_generation_mw - 10.0).abs() < f64::EPSILON
            );
            assert!(
                (result.thermal_block_base(0, stage).max_generation_mw - 200.0).abs()
                    < f64::EPSILON
            );
            assert!((result.line_block_base(0, stage).direct_mw - 500.0).abs() < f64::EPSILON);
            assert!((result.line_block_base(0, stage).reverse_mw - 300.0).abs() < f64::EPSILON);
            assert!((result.pumping_block_base(0, stage).max_flow_m3s - 75.0).abs() < f64::EPSILON);
            assert!(
                (result.contract_block_base(0, stage).price_per_mwh - 60.0).abs() < f64::EPSILON
            );
        }
    }

    // ── Test: acceptance criterion — AC5: contract price_per_mwh override ──────

    /// AC5: contract override with price_per_mwh=Some(90.0) at stage 1.
    #[test]
    fn test_ac5_contract_price_override() {
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let override_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 1,
            min_mw: None,
            max_mw: None,
            price_per_mwh: Some(90.0),
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &contracts,
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[override_row],
            &[],
        );

        assert!((result.contract_block_base(0, 1).price_per_mwh - 90.0).abs() < f64::EPSILON);
        assert!((result.contract_block_base(0, 0).price_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    // ── Test: water_withdrawal_m3s override can be negative ───────────────────

    /// Given a water_withdrawal override with a negative value (water addition),
    /// the resolver accepts it without validation.
    #[test]
    fn test_water_withdrawal_negative_accepted() {
        let hydros = vec![make_hydro(0, 10.0, 100.0, None, None, None)];
        let override_row = HydroBoundsRow {
            water_withdrawal_m3s: Some(-5.0),
            ..all_none_hydro_row(0, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            2,
            0,
            &[override_row],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        assert!((result.hydro_bounds(0, 0).water_withdrawal_m3s - (-5.0)).abs() < f64::EPSILON);
        assert!((result.hydro_bounds(0, 1).water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);
    }

    // ── Tests: thermal cost override acceptance criteria ─────────────────────

    /// cost_per_mwh override with null block_id — applied.
    /// thermal T1, stages 0 and 1, cost overrides 50.0 and 100.0.
    #[test]
    fn test_thermal_cost_override_null_block_id() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let overrides = vec![
            ThermalBoundsRow {
                thermal_id: EntityId::from(0),
                stage_id: 0,
                min_generation_mw: None,
                max_generation_mw: None,
                cost_per_mwh: Some(50.0),
                block_id: None,
            },
            ThermalBoundsRow {
                thermal_id: EntityId::from(0),
                stage_id: 1,
                min_generation_mw: None,
                max_generation_mw: None,
                cost_per_mwh: Some(100.0),
                block_id: None,
            },
        ];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &overrides,
            &[],
            &[],
            &[],
            &[],
        );

        assert!(
            (result.thermal_bounds(0, 0).cost_per_mwh - 50.0).abs() < f64::EPSILON,
            "expected cost_per_mwh=50.0 at stage 0, got {}",
            result.thermal_bounds(0, 0).cost_per_mwh
        );
        assert!(
            (result.thermal_bounds(0, 1).cost_per_mwh - 100.0).abs() < f64::EPSILON,
            "expected cost_per_mwh=100.0 at stage 1, got {}",
            result.thermal_bounds(0, 1).cost_per_mwh
        );
    }

    /// No parquet cost override — falls back to base Thermal.cost_per_mwh.
    #[test]
    fn test_thermal_cost_fallback_to_base() {
        // make_thermal uses cost_per_mwh: 50.0 in the helper.
        let thermals = vec![make_thermal(0, 0.0, 400.0)];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..3 {
            assert!(
                (result.thermal_bounds(0, stage).cost_per_mwh - 50.0).abs() < f64::EPSILON,
                "expected base cost_per_mwh=50.0 at stage {stage}"
            );
        }
    }

    /// A block row's `cost_per_mwh` is never applied — `ThermalBlockOverride`
    /// carries no cost field (per-block cost is out of scope); `min`/`max_generation_mw`
    /// on a block row ARE applied (see `test_thermal_block_id_is_no_longer_skipped`).
    #[test]
    fn test_thermal_cost_block_id_row_ignored() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: Some(999.0),
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        assert!(
            (result.thermal_bounds(0, 0).cost_per_mwh - 50.0).abs() < f64::EPSILON,
            "a block row's cost_per_mwh must remain ignored; expected 50.0, got {}",
            result.thermal_bounds(0, 0).cost_per_mwh
        );
    }

    /// Thermal with cost_per_mwh == 0.0, no override — resolves to 0.0.
    #[test]
    fn test_thermal_zero_base_cost_no_override() {
        let thermals = vec![Thermal {
            id: EntityId::from(0),
            name: "ZeroCost".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            2,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..2 {
            assert!(
                result.thermal_bounds(0, stage).cost_per_mwh.abs() < f64::EPSILON,
                "expected cost_per_mwh=0.0 at stage {stage}"
            );
        }
    }

    // ── Tests: padded thermal region (k_max > 0) ──────────────────────────────

    /// Given two thermals with distinct base values, `n_stages = 3`, and
    /// `k_max = 2`, every padded cell `[3, 5)` for each thermal contains
    /// that thermal's *own* base values — not a uniform default and not
    /// thermal 0's values overwriting thermal 1's.
    #[test]
    fn test_padded_region_uses_per_thermal_base_values() {
        let mut t0 = make_thermal(0, 0.0, 100.0);
        t0.cost_per_mwh = 50.0;
        let mut t1 = make_thermal(1, 10.0, 200.0);
        t1.cost_per_mwh = 80.0;
        let thermals = vec![t0, t1];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            3,
            2,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        // Study-horizon cells [0, 3): each thermal's own base values.
        for stage in 0..3 {
            let b0 = result.thermal_block_base(0, stage);
            assert!((b0.min_generation_mw - 0.0).abs() < f64::EPSILON);
            assert!((b0.max_generation_mw - 100.0).abs() < f64::EPSILON);
            assert!((result.thermal_bounds(0, stage).cost_per_mwh - 50.0).abs() < f64::EPSILON);
            let b1 = result.thermal_block_base(1, stage);
            assert!((b1.min_generation_mw - 10.0).abs() < f64::EPSILON);
            assert!((b1.max_generation_mw - 200.0).abs() < f64::EPSILON);
            assert!((result.thermal_bounds(1, stage).cost_per_mwh - 80.0).abs() < f64::EPSILON);
        }

        // Padded cells [3, 5): each thermal still carries its own base values.
        for stage in 3..5 {
            let b0 = result.thermal_block_base(0, stage);
            assert!((b0.min_generation_mw - 0.0).abs() < f64::EPSILON);
            assert!((b0.max_generation_mw - 100.0).abs() < f64::EPSILON);
            assert!((result.thermal_bounds(0, stage).cost_per_mwh - 50.0).abs() < f64::EPSILON);
            let b1 = result.thermal_block_base(1, stage);
            assert!((b1.min_generation_mw - 10.0).abs() < f64::EPSILON);
            assert!((b1.max_generation_mw - 200.0).abs() < f64::EPSILON);
            assert!((result.thermal_bounds(1, stage).cost_per_mwh - 80.0).abs() < f64::EPSILON);
        }
    }

    /// Given `k_max == 0`, behavior is bit-identical to the baseline before
    /// padding support was added: the fill loop iterates `0..n_stages` and no
    /// padding cells exist.
    #[test]
    fn test_k_max_zero_preserves_existing_behavior() {
        let mut t0 = make_thermal(0, 0.0, 100.0);
        t0.cost_per_mwh = 50.0;
        let thermals = vec![t0];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        for stage in 0..3 {
            let b = result.thermal_block_base(0, stage);
            assert!((b.min_generation_mw - 0.0).abs() < f64::EPSILON);
            assert!((b.max_generation_mw - 100.0).abs() < f64::EPSILON);
            assert!((result.thermal_bounds(0, stage).cost_per_mwh - 50.0).abs() < f64::EPSILON);
        }
    }

    /// Given zero thermals and `k_max > 0`, `resolve_bounds` does not panic
    /// and the thermal stage axis is `n_stages + k_max`.
    #[test]
    fn test_padded_region_with_zero_thermals() {
        let result = resolve_bounds(&[], &[], &[], &[], &[], 4, 3, &[], &[], &[], &[], &[], &[]);
        assert_eq!(result.thermal_stage_axis_len(), 4 + 3);
    }

    // ── Thermal padding boundary tests (resolver) ─────────────────────────────
    //
    // These tests pin down the per-thermal base-fill semantics of the resolver
    // at the four boundary stage indices that LP-template consumers exercise.
    // The accessor-only invariants (uniform-default fill, `n_stages()`
    // unchanged, `thermal_stage_axis_len()` extended) live in `cobre-core`'s
    // `ResolvedBounds` tests.

    /// Two thermals with distinct `(min, max, cost)` bases, `n_stages = 4`,
    /// `k_max = 2`, no overrides. The last study stage and both padded stages
    /// must return each thermal's *own* base values — never the uniform
    /// `BoundsDefaults.thermal` fallback and never thermal 0's values
    /// overwriting thermal 1's.
    #[test]
    fn test_resolve_bounds_thermal_padded_uses_plant_base() {
        let mut t0 = make_thermal(0, 0.0, 100.0);
        t0.cost_per_mwh = 50.0;
        let mut t1 = make_thermal(1, 10.0, 200.0);
        t1.cost_per_mwh = 80.0;
        let thermals = vec![t0, t1];

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            4,
            2,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        // Last study stage (T - 1 == 3): no override → thermal 0 base.
        let b0_last_study = result.thermal_block_base(0, 3);
        assert!((b0_last_study.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((b0_last_study.max_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 3).cost_per_mwh - 50.0).abs() < f64::EPSILON);

        // First padded stage (T == 4): thermal 0 base.
        let b0_first_pad = result.thermal_block_base(0, 4);
        assert!((b0_first_pad.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((b0_first_pad.max_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 4).cost_per_mwh - 50.0).abs() < f64::EPSILON);

        // Last padded stage (T + K - 1 == 5): thermal 0 base.
        let b0_last_pad = result.thermal_block_base(0, 5);
        assert!((b0_last_pad.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((b0_last_pad.max_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 5).cost_per_mwh - 50.0).abs() < f64::EPSILON);

        // First padded stage for thermal 1 must NOT pick up thermal 0's base.
        let b1_first_pad = result.thermal_block_base(1, 4);
        assert!(
            (b1_first_pad.min_generation_mw - 10.0).abs() < f64::EPSILON,
            "thermal 1 padded min must be 10.0, not 0.0 (thermal 0) or 0.0 (uniform default)"
        );
        assert!((b1_first_pad.max_generation_mw - 200.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(1, 4).cost_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    /// One thermal, `n_stages = 4`, `k_max = 2`, override row at the *last*
    /// study stage (`stage_id = 3`). The override applies at that stage only;
    /// the padded cells at `T == 4` and `T + 1 == 5` must contain the entity
    /// base value, not the override (overrides must not leak into padding).
    #[test]
    fn test_resolve_bounds_override_at_t_minus_1_applied() {
        let thermals = vec![make_thermal(0, 0.0, 100.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 3,
            min_generation_mw: None,
            max_generation_mw: Some(999.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            4,
            2,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[],
        );

        // Override applies at T - 1 == 3.
        assert!(
            (result.thermal_block_base(0, 3).max_generation_mw - 999.0).abs() < f64::EPSILON,
            "override at stage_id=3 must be applied"
        );
        // Padded cells at T == 4 and T + 1 == 5 carry the entity base value.
        assert!(
            (result.thermal_block_base(0, 4).max_generation_mw - 100.0).abs() < f64::EPSILON,
            "override must NOT leak into first padded stage (T == 4)"
        );
        assert!(
            (result.thermal_block_base(0, 5).max_generation_mw - 100.0).abs() < f64::EPSILON,
            "override must NOT leak into last padded stage (T + 1 == 5)"
        );
    }

    /// `k_max == 0`: the thermal stage axis equals `n_stages` exactly and
    /// there are no padded cells. Reading at `stage_index == n_stages` panics
    /// in debug builds (gated by `#[cfg(debug_assertions)]` because release
    /// builds may silently read adjacent memory via `Vec` indexing).
    #[test]
    fn test_resolve_bounds_k_max_zero_no_padded_cells() {
        let thermals = vec![make_thermal(0, 0.0, 100.0)];
        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            3,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );

        assert_eq!(result.thermal_stage_axis_len(), 3);

        #[cfg(debug_assertions)]
        {
            let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = result.thermal_bounds(0, 3);
            }));
            assert!(
                panic_result.is_err(),
                "thermal_bounds(0, 3) must panic in debug builds when n_stages=3, k_max=0"
            );
        }
    }

    // ── Tests: hydro per-block overlay activation ─────────────────────────────

    /// Precedence: block (layer 1) beats stage-wide (layer 2) beats base (layer 3).
    /// Hydro base `max_turbined_m3s = 500.0`, a stage-wide row sets `400.0`, and a
    /// block-1 row (of 3 blocks) sets `100.0`.
    #[test]
    fn test_hydro_precedence_block_over_stage_over_base() {
        let mut hydro = make_hydro(0, 10.0, 200.0, None, None, None);
        hydro.max_turbined_m3s = 500.0;
        let hydros = vec![hydro];
        let stage_wide_row = HydroBoundsRow {
            max_turbined_m3s: Some(400.0),
            ..all_none_hydro_row(0, 0)
        };
        let block_row = HydroBoundsRow {
            max_turbined_m3s: Some(100.0),
            block_id: Some(1),
            ..all_none_hydro_row(0, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[stage_wide_row, block_row],
            &[],
            &[],
            &[],
            &[],
            &[3],
        );

        assert!(
            (result.hydro_bounds_at_block(0, 0, 0).max_turbined_m3s - 400.0).abs() < f64::EPSILON,
            "block 0 has no block override: falls through to the stage-wide value"
        );
        assert!(
            (result.hydro_bounds_at_block(0, 0, 1).max_turbined_m3s - 100.0).abs() < f64::EPSILON,
            "block 1 carries the block override"
        );
        assert!(
            (result.hydro_bounds_at_block(0, 0, 2).max_turbined_m3s - 400.0).abs() < f64::EPSILON,
            "block 2 has no block override: falls through to the stage-wide value"
        );
    }

    /// A block row that sets only `max_storage_hm3` and `water_withdrawal_m3s`
    /// (both stage-level, not block-eligible — `HydroBlockOverride` has no
    /// field for them) leaves every field of `hydro_bounds_at_block` identical
    /// to `hydro_block_base` — the row's stage-level values do not leak into
    /// the stage-wide cell either.
    #[test]
    fn test_hydro_block_row_cannot_override_stage_level_columns() {
        let hydros = vec![make_hydro(0, 10.0, 200.0, None, None, None)];
        let block_row = HydroBoundsRow {
            max_storage_hm3: Some(999.0),
            water_withdrawal_m3s: Some(42.0),
            block_id: Some(1),
            ..all_none_hydro_row(0, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[block_row],
            &[],
            &[],
            &[],
            &[],
            &[3],
        );

        let stage_wide = *result.hydro_bounds(0, 0);
        let block_base = result.hydro_block_base(0, 0);
        let at_block = result.hydro_bounds_at_block(0, 0, 1);

        assert!((stage_wide.max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        assert!((stage_wide.water_withdrawal_m3s - 0.0).abs() < f64::EPSILON);

        assert!((at_block.min_turbined_m3s - block_base.min_turbined_m3s).abs() < f64::EPSILON);
        assert!((at_block.max_turbined_m3s - block_base.max_turbined_m3s).abs() < f64::EPSILON);
        assert!((at_block.min_outflow_m3s - block_base.min_outflow_m3s).abs() < f64::EPSILON);
        assert_eq!(at_block.max_outflow_m3s, block_base.max_outflow_m3s);
        assert!((at_block.min_generation_mw - block_base.min_generation_mw).abs() < f64::EPSILON);
        assert!((at_block.max_generation_mw - block_base.max_generation_mw).abs() < f64::EPSILON);
        assert_eq!(at_block.max_diversion_m3s, block_base.max_diversion_m3s);
    }

    /// A block row that sets only `max_turbined_m3s` leaves `min_turbined_m3s` at
    /// the stage-wide (here: base, since no stage-wide override is present) value —
    /// a layer-1 row replaces only the columns it names.
    #[test]
    fn test_hydro_block_row_is_sparse_per_column() {
        let mut hydro = make_hydro(0, 10.0, 200.0, None, None, None);
        hydro.min_turbined_m3s = 5.0;
        let hydros = vec![hydro];
        let block_row = HydroBoundsRow {
            max_turbined_m3s: Some(999.0),
            block_id: Some(1),
            ..all_none_hydro_row(0, 0)
        };

        let result = resolve_bounds(
            &hydros,
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[block_row],
            &[],
            &[],
            &[],
            &[],
            &[2],
        );

        let b = result.hydro_bounds_at_block(0, 0, 1);
        assert!((b.max_turbined_m3s - 999.0).abs() < f64::EPSILON);
        assert!((b.min_turbined_m3s - 5.0).abs() < f64::EPSILON);
    }

    // ── Tests: thermal per-block overlay activation ───────────────────────────

    /// A thermal row with `block_id = Some(1)` and `max_generation_mw = Some(100.0)`
    /// is no longer skipped: the value reaches the per-block overlay.
    #[test]
    fn test_thermal_block_id_is_no_longer_skipped() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let override_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(100.0),
            cost_per_mwh: None,
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[override_row],
            &[],
            &[],
            &[],
            &[2],
        );

        assert!(
            (result.thermal_bounds_at_block(0, 0, 1).max_generation_mw - 100.0).abs()
                < f64::EPSILON
        );
    }

    /// Precedence: block (layer 1) beats stage-wide (layer 2) beats base (layer 3).
    /// Thermal base `max_generation_mw = 500.0`, a stage-wide row sets `400.0`, and
    /// a block-1 row (of 3 blocks) sets `100.0`.
    #[test]
    fn test_thermal_precedence_block_over_stage_over_base() {
        let thermals = vec![make_thermal(0, 0.0, 500.0)];
        let stage_wide_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(400.0),
            cost_per_mwh: None,
            block_id: None,
        };
        let block_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(100.0),
            cost_per_mwh: None,
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[stage_wide_row, block_row],
            &[],
            &[],
            &[],
            &[3],
        );

        assert!(
            (result.thermal_bounds_at_block(0, 0, 0).max_generation_mw - 400.0).abs()
                < f64::EPSILON,
            "block 0 has no block override: falls through to the stage-wide value"
        );
        assert!(
            (result.thermal_bounds_at_block(0, 0, 1).max_generation_mw - 100.0).abs()
                < f64::EPSILON,
            "block 1 carries the block override"
        );
        assert!(
            (result.thermal_bounds_at_block(0, 0, 2).max_generation_mw - 400.0).abs()
                < f64::EPSILON,
            "block 2 has no block override: falls through to the stage-wide value"
        );
    }

    /// A block row that sets only `max_generation_mw` leaves `min_generation_mw`
    /// and `cost_per_mwh` at the stage-wide (here: base, since no stage-wide
    /// override is present) value — a layer-1 row replaces only the columns it names.
    #[test]
    fn test_thermal_block_row_is_sparse_per_column() {
        let thermals = vec![make_thermal(0, 10.0, 400.0)];
        let block_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(999.0),
            cost_per_mwh: None,
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[block_row],
            &[],
            &[],
            &[],
            &[2],
        );

        let b = result.thermal_bounds_at_block(0, 0, 1);
        assert!((b.max_generation_mw - 999.0).abs() < f64::EPSILON);
        assert!((b.min_generation_mw - 10.0).abs() < f64::EPSILON);
        assert!((result.thermal_bounds(0, 0).cost_per_mwh - 50.0).abs() < f64::EPSILON);
    }

    /// A `block_id` of `7` on a stage declaring 3 blocks, and separately a
    /// `block_id` of `-1`, are both skipped: no resolved value changes and
    /// `resolve_bounds` returns normally (no panic).
    #[test]
    fn test_out_of_range_thermal_block_id_is_skipped_not_panicked() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let too_large_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(999.0),
            cost_per_mwh: None,
            block_id: Some(7),
        };
        let negative_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(888.0),
            cost_per_mwh: None,
            block_id: Some(-1),
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[too_large_row, negative_row],
            &[],
            &[],
            &[],
            &[3],
        );

        for block_idx in 0..3 {
            assert!(
                (result
                    .thermal_bounds_at_block(0, 0, block_idx)
                    .max_generation_mw
                    - 400.0)
                    .abs()
                    < f64::EPSILON,
                "block {block_idx}: out-of-range and negative block_id rows must not apply"
            );
        }
    }

    /// With no `block_id` rows in any family, the overlay stays empty — it is
    /// never allocated unconditionally.
    #[test]
    fn test_no_block_rows_leaves_overlay_empty() {
        let thermals = vec![make_thermal(0, 0.0, 400.0)];
        let stage_wide_row = ThermalBoundsRow {
            thermal_id: EntityId::from(0),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(300.0),
            cost_per_mwh: None,
            block_id: None,
        };

        let result = resolve_bounds(
            &[],
            &thermals,
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[stage_wide_row],
            &[],
            &[],
            &[],
            &[3],
        );

        assert!(result.block_overlay().is_empty());
    }

    // ── Tests: line/contract/pumping per-block overlay activation ────────────

    /// A line `direct_mw` block-2 override, a contract `max_mw` block-0 override,
    /// and a pumping `max_flow_m3s` block-1 override each apply at their own block
    /// only; every other block on the 3-block stage falls through to the
    /// stage-wide (here: base, since no stage-wide override row is present) value.
    #[test]
    fn test_line_contract_pumping_block_precedence() {
        let lines = vec![make_line(0, 1000.0, 800.0)];
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let pumpings = vec![make_pumping(0, 0.0, 100.0)];

        let line_block_row = LineBoundsRow {
            line_id: EntityId::from(0),
            stage_id: 0,
            direct_mw: Some(800.0),
            reverse_mw: None,
            block_id: Some(2),
        };
        let contract_block_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 0,
            min_mw: None,
            max_mw: Some(50.0),
            price_per_mwh: None,
            block_id: Some(0),
        };
        let pumping_block_row = PumpingBoundsRow {
            station_id: EntityId::from(0),
            stage_id: 0,
            min_m3s: None,
            max_m3s: Some(20.0),
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &[],
            &lines,
            &pumpings,
            &contracts,
            1,
            0,
            &[],
            &[],
            &[line_block_row],
            &[pumping_block_row],
            &[contract_block_row],
            &[3],
        );

        assert!((result.line_bounds_at_block(0, 0, 0).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((result.line_bounds_at_block(0, 0, 1).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!(
            (result.line_bounds_at_block(0, 0, 2).direct_mw - 800.0).abs() < f64::EPSILON,
            "block 2 carries the line block override"
        );

        assert!(
            (result.contract_bounds_at_block(0, 0, 0).max_mw - 50.0).abs() < f64::EPSILON,
            "block 0 carries the contract block override"
        );
        assert!((result.contract_bounds_at_block(0, 0, 1).max_mw - 200.0).abs() < f64::EPSILON);
        assert!((result.contract_bounds_at_block(0, 0, 2).max_mw - 200.0).abs() < f64::EPSILON);

        assert!(
            (result.pumping_bounds_at_block(0, 0, 0).max_flow_m3s - 100.0).abs() < f64::EPSILON
        );
        assert!(
            (result.pumping_bounds_at_block(0, 0, 1).max_flow_m3s - 20.0).abs() < f64::EPSILON,
            "block 1 carries the pumping block override"
        );
        assert!(
            (result.pumping_bounds_at_block(0, 0, 2).max_flow_m3s - 100.0).abs() < f64::EPSILON
        );
    }

    /// A contract block row that sets only `price_per_mwh` overrides price at its
    /// own block while `min_mw`/`max_mw` at that same block fall through to the
    /// stage-wide value — `price_per_mwh` is block-eligible on its own, independent
    /// of the other two columns.
    #[test]
    fn test_contract_block_row_overrides_price_per_block() {
        let contracts = vec![make_contract(0, 0.0, 200.0, 80.0)];
        let block_row = ContractBoundsRow {
            contract_id: EntityId::from(0),
            stage_id: 0,
            min_mw: None,
            max_mw: None,
            price_per_mwh: Some(120.0),
            block_id: Some(1),
        };

        let result = resolve_bounds(
            &[],
            &[],
            &[],
            &[],
            &contracts,
            1,
            0,
            &[],
            &[],
            &[],
            &[],
            &[block_row],
            &[3],
        );

        assert!(
            (result.contract_bounds_at_block(0, 0, 1).price_per_mwh - 120.0).abs() < f64::EPSILON,
            "block 1 carries the price override"
        );
        assert!(
            (result.contract_bounds_at_block(0, 0, 0).price_per_mwh - 80.0).abs() < f64::EPSILON,
            "block 0 falls through to the stage-wide price"
        );
        assert!(
            (result.contract_bounds_at_block(0, 0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON,
            "block 2 falls through to the stage-wide price"
        );

        let at_block = result.contract_bounds_at_block(0, 0, 1);
        assert!((at_block.min_mw - 0.0).abs() < f64::EPSILON);
        assert!((at_block.max_mw - 200.0).abs() < f64::EPSILON);
    }
}
