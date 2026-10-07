//! Resolution of `hydro_unit_group_bounds` rows into a per-`(unit group, stage,
//! block)` override table — layers 1 and 2 of the bound-precedence law
//! [`resolve_bounds`](super::resolve_bounds) documents, applied to the hydro
//! unit group axis instead of the plant axis. Layer 3 (a group's own declared
//! value on `system/hydros.json`) is left for the consumer's `unwrap_or`; this
//! resolver never reads it.
//!
//! Referential integrity (unknown `(hydro_id, hydro_unit_group_id)`, an
//! out-of-range `block_id`, duplicate rows) is enforced upstream during
//! validation; [`resolve_hydro_unit_group_bounds`] silently skips whatever
//! still fails to resolve here, mirroring [`resolve_bounds`](super::resolve_bounds).

use std::collections::HashMap;

use cobre_core::{
    EntityId,
    entities::Hydro,
    resolved::{HydroUnitGroupOverride, ResolvedHydroUnitGroupBounds},
};

use crate::constraints::HydroUnitGroupBoundsRow;

/// Build a resolved hydro unit group bounds table from parsed override rows.
///
/// `hydros` must be in the order
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) establishes;
/// slice position becomes the entity index. Each plant's `unit_groups` are
/// sorted by id by `Hydro::sort_unit_groups`, applied identically at parse
/// time and by `SystemBuilder::build`, so that axis has no equivalent
/// divergence; the regression test
/// `hydro_unit_group_bound_override_follows_declared_id_through_canonical_resort`
/// falsifies a resolve-before-resort regression on the `hydros` ordering.
/// `blocks_per_stage` and `stage_index` are the same tables
/// [`resolve_bounds`](super::resolve_bounds) uses — reused, never rebuilt.
///
/// A row is skipped, silently, when its `(hydro_id, hydro_unit_group_id)`
/// names no group, its `stage_id` is not a study stage, or (for a block row)
/// `block_id` falls outside `[0, blocks_per_stage[stage_idx])`.
// implicit_hasher: callers pass a concrete `HashMap`; a `BuildHasher` generic buys nothing.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn resolve_hydro_unit_group_bounds(
    hydros: &[Hydro],
    n_stages: usize,
    stage_index: &HashMap<i32, usize>,
    rows: &[HydroUnitGroupBoundsRow],
    blocks_per_stage: &[usize],
) -> ResolvedHydroUnitGroupBounds {
    let mut groups_per_plant: Vec<usize> = Vec::with_capacity(hydros.len());
    let mut group_position: HashMap<(EntityId, EntityId), (usize, usize)> = HashMap::new();
    for (hydro_idx, hydro) in hydros.iter().enumerate() {
        groups_per_plant.push(hydro.unit_groups.len());
        for (group_pos, group) in hydro.unit_groups.iter().enumerate() {
            group_position.insert((hydro.id, group.id), (hydro_idx, group_pos));
        }
    }

    let max_blocks = blocks_per_stage.iter().copied().max().unwrap_or(0);

    let mut table =
        ResolvedHydroUnitGroupBounds::new(&cobre_core::HydroUnitGroupBoundsCountsSpec {
            groups_per_plant: &groups_per_plant,
            n_stages,
            max_blocks,
        });

    for row in rows.iter().filter(|row| row.block_id.is_none()) {
        let Some(&(hydro_idx, group_pos)) =
            group_position.get(&(row.hydro_id, row.hydro_unit_group_id))
        else {
            continue;
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        let Some(cell) = table.stage_override_mut(hydro_idx, group_pos, stage_idx) else {
            continue;
        };
        apply_group_bounds_row(row, cell);
    }

    for row in rows {
        let Some((hydro_idx, group_pos, stage_idx, block_idx)) =
            group_block_slot(row, &group_position, stage_index, blocks_per_stage)
        else {
            continue;
        };
        let Some(cell) = table.block_override_mut(hydro_idx, group_pos, stage_idx, block_idx)
        else {
            continue;
        };
        apply_group_bounds_row(row, cell);
    }

    table
}

/// Resolve a row's `(hydro_idx, group_pos, stage_idx, block_idx)` block-layer
/// target; `None` skips the row (no `block_id`, unresolvable group or stage,
/// or a `block_id` outside the stage's block count).
fn group_block_slot(
    row: &HydroUnitGroupBoundsRow,
    group_position: &HashMap<(EntityId, EntityId), (usize, usize)>,
    stage_index: &HashMap<i32, usize>,
    blocks_per_stage: &[usize],
) -> Option<(usize, usize, usize, usize)> {
    let block_id = row.block_id?;
    let &(hydro_idx, group_pos) = group_position.get(&(row.hydro_id, row.hydro_unit_group_id))?;
    let &stage_idx = stage_index.get(&row.stage_id)?;
    let block_idx = usize::try_from(block_id).ok()?;
    if block_idx >= *blocks_per_stage.get(stage_idx)? {
        return None;
    }
    Some((hydro_idx, group_pos, stage_idx, block_idx))
}

/// Write a row's present columns into an override cell. All four columns on
/// [`HydroUnitGroupBoundsRow`] are block-eligible, so this applies unchanged
/// to both the stage-wide and the per-block pass.
fn apply_group_bounds_row(row: &HydroUnitGroupBoundsRow, over: &mut HydroUnitGroupOverride) {
    if let Some(v) = row.min_turbined_m3s {
        over.min_turbined_m3s = Some(v);
    }
    if let Some(v) = row.max_turbined_m3s {
        over.max_turbined_m3s = Some(v);
    }
    if let Some(v) = row.min_generation_mw {
        over.min_generation_mw = Some(v);
    }
    if let Some(v) = row.max_generation_mw {
        over.max_generation_mw = Some(v);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::entities::{HydroGenerationModel, HydroPenalties, HydroUnitGroup};

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

    fn make_group(id: i32) -> HydroUnitGroup {
        HydroUnitGroup {
            id: EntityId::from(id),
            name: format!("Group {id}"),
            bus_id: EntityId::from(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
        }
    }

    fn make_hydro(id: i32, group_ids: &[i32]) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            specific_productivity_mw_per_m3s_per_m: None,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            unit_groups: group_ids.iter().copied().map(make_group).collect(),
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: make_penalties(),
        }
    }

    fn all_none_row(hydro_id: i32, group_id: i32, stage_id: i32) -> HydroUnitGroupBoundsRow {
        HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(hydro_id),
            hydro_unit_group_id: EntityId::from(group_id),
            stage_id,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            block_id: None,
        }
    }

    /// 0-based consecutive stage index map: `{0 -> 0, 1 -> 1, ..., (n-1) -> (n-1)}`.
    fn si(n: usize) -> HashMap<i32, usize> {
        (0..n).map(|i| (i32::try_from(i).unwrap(), i)).collect()
    }

    /// AC1: three plants declaring 1, 3, and 2 groups and an empty row slice
    /// leave the table empty, and `override_at_block` returns the all-`None`
    /// default for every in-range and out-of-range index without panicking.
    #[test]
    fn resolve_group_bounds_empty_yields_no_overrides() {
        let hydros = vec![
            make_hydro(0, &[10]),
            make_hydro(1, &[10, 11, 12]),
            make_hydro(2, &[10, 11]),
        ];
        let stage_index = si(3);

        let table = resolve_hydro_unit_group_bounds(&hydros, 3, &stage_index, &[], &[2, 2, 2]);

        assert!(table.is_empty());
        for (h, g, t, b) in [
            (0, 0, 0, 0),
            (1, 2, 2, 1),
            (2, 1, 1, 0),
            (99, 0, 0, 0),
            (0, 99, 0, 0),
            (0, 0, 99, 0),
            (0, 0, 0, 99),
        ] {
            assert_eq!(
                table.override_at_block(h, g, t, b),
                HydroUnitGroupOverride::default(),
                "expected default override at (h={h}, g={g}, t={t}, b={b})"
            );
        }
    }

    /// AC2: a stage-wide row on `(hydro_id=1, group=11, stage=2)` sets
    /// `max_turbined_m3s` and `min_generation_mw`; a block row on the same
    /// group at `block=1` sets only `max_turbined_m3s`. Column-independent
    /// resolution must keep `min_generation_mw` from the stage-wide row at
    /// block 1, while block 0 falls through to the stage-wide value entirely.
    #[test]
    fn resolve_group_bounds_block_row_beats_stage_row_per_column() {
        let hydros = vec![make_hydro(0, &[5]), make_hydro(1, &[10, 11, 12])];
        let stage_index = si(3);
        let rows = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(40.0),
                min_generation_mw: Some(5.0),
                ..all_none_row(1, 11, 2)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(12.0),
                block_id: Some(1),
                ..all_none_row(1, 11, 2)
            },
        ];

        let table = resolve_hydro_unit_group_bounds(&hydros, 3, &stage_index, &rows, &[2, 2, 2]);

        let at_block_1 = table.override_at_block(1, 1, 2, 1);
        assert_eq!(
            at_block_1.max_turbined_m3s.map(f64::to_bits),
            Some(12.0f64.to_bits())
        );
        assert_eq!(
            at_block_1.min_generation_mw.map(f64::to_bits),
            Some(5.0f64.to_bits())
        );

        let at_block_0 = table.override_at_block(1, 1, 2, 0);
        assert_eq!(
            at_block_0.max_turbined_m3s.map(f64::to_bits),
            Some(40.0f64.to_bits())
        );
    }

    /// AC3: plants declare `[1, 3, 2]` groups; a row names plant index 2's
    /// group at position 1. The written slot is `plant_group_start[2] + 1 ==
    /// 5`, and plant 1's group position 1 (slot 2) stays untouched — the
    /// ragged CSR axis must not alias across plants.
    #[test]
    fn resolve_group_bounds_ragged_group_axis_does_not_alias_across_plants() {
        let hydros = vec![
            make_hydro(0, &[10]),
            make_hydro(1, &[10, 11, 12]),
            make_hydro(2, &[10, 11]),
        ];
        let stage_index = si(2);
        let rows = vec![HydroUnitGroupBoundsRow {
            max_turbined_m3s: Some(77.0),
            ..all_none_row(2, 11, 0)
        }];

        let table = resolve_hydro_unit_group_bounds(&hydros, 2, &stage_index, &rows, &[1, 1]);

        assert_eq!(
            table
                .override_at_block(2, 1, 0, 0)
                .max_turbined_m3s
                .map(f64::to_bits),
            Some(77.0f64.to_bits())
        );
        assert_eq!(
            table.override_at_block(1, 1, 0, 0),
            HydroUnitGroupOverride::default(),
            "plant 1's group position 1 (slot 2) must not alias plant 2's write (slot 5)"
        );
    }

    /// AC4: a row naming an unknown group, a row with an out-of-study
    /// `stage_id`, and a row whose `block_id` equals its stage's block count
    /// are all skipped; the table stays empty and nothing panics.
    ///
    /// `blocks_per_stage` is deliberately non-uniform (`[2, 5]`): the third
    /// row's `block_id = 2` is in range against the global `max_blocks = 5`
    /// but out of range for stage 0's own 2 blocks, so only the per-stage
    /// guard — not the table's global `max_blocks` bound — can catch it.
    #[test]
    fn resolve_group_bounds_skips_unresolvable_rows() {
        let hydros = vec![make_hydro(0, &[10])];
        let stage_index = si(2);
        let rows = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(1.0),
                ..all_none_row(0, 999, 0)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(2.0),
                ..all_none_row(0, 10, 999)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(3.0),
                block_id: Some(2),
                ..all_none_row(0, 10, 0)
            },
        ];

        let table = resolve_hydro_unit_group_bounds(&hydros, 2, &stage_index, &rows, &[2, 5]);

        assert!(table.is_empty());
    }
}
