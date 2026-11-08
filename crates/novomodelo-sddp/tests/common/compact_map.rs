//! Order-preserving map from a stage's logical LP indices to its compact ones.

use std::ops::Range;

use novomodelo_core::temporal::BlockMode;
use novomodelo_sddp::indexer::{BlockRowFamily, EvaporationIndices, HydroSys, StateSpace};
use novomodelo_sddp::lp::builder::{GenericConstraintRowEntry, StageGeometry};
use novomodelo_solver::StageTemplate;

pub struct CompactMap {
    cols: Vec<Option<usize>>,
    rows: Vec<Option<usize>>,
}

fn kept_indices(len: usize, is_deleted: impl Fn(usize) -> bool) -> Vec<Option<usize>> {
    let mut next = 0;
    (0..len)
        .map(|index| {
            (!is_deleted(index)).then(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

fn kept_count(map: &[Option<usize>]) -> usize {
    map.iter().flatten().count()
}

fn compact_range(map: &[Option<usize>], range: Range<usize>, kind: &str) -> Range<usize> {
    assert!(
        range.end <= map.len() && map[range.clone()].iter().all(Option::is_some),
        "{kind} range {range:?} is out of bounds or contains a deleted index"
    );
    kept_count(&map[..range.start])..kept_count(&map[..range.end])
}

impl CompactMap {
    /// Deleted columns are the logical lag and z columns, deleted rows the z rows.
    ///
    /// # Panics
    ///
    /// Panics unless the deleted counts, the compact dimensions and every
    /// state range follow from the logical layout.
    pub fn of_stage(
        logical_state: &StateSpace,
        compact_state: &StateSpace,
        logical: &StageTemplate,
        compact: &StageTemplate,
    ) -> Self {
        let z_rows = logical_state.z_inflow_rows();
        let cols = kept_indices(logical.num_cols, |c| {
            logical_state.inflow_lags.contains(&c) || logical_state.z_inflow.contains(&c)
        });
        let rows = kept_indices(logical.num_rows, |r| z_rows.contains(&r));

        let deleted_cols = logical.num_cols - kept_count(&cols);
        let deleted_rows = logical.num_rows - kept_count(&rows);
        assert_eq!(
            deleted_cols,
            logical_state.hydro_count * (logical_state.max_par_order + 1),
            "deleted columns: hydro_count * (max_par_order + 1)"
        );
        assert_eq!(
            deleted_rows, logical_state.hydro_count,
            "deleted rows: hydro_count"
        );
        assert_eq!(compact.num_cols, kept_count(&cols), "compact column count");
        assert_eq!(compact.num_rows, kept_count(&rows), "compact row count");

        let map = Self { cols, rows };
        let ranges: [(&str, &Range<usize>, &Range<usize>); 6] = [
            ("storage", &logical_state.storage, &compact_state.storage),
            (
                "transit_buckets_out",
                &logical_state.transit_buckets_out,
                &compact_state.transit_buckets_out,
            ),
            (
                "commit_out",
                &logical_state.commit_out,
                &compact_state.commit_out,
            ),
            (
                "storage_in",
                &logical_state.storage_in,
                &compact_state.storage_in,
            ),
            (
                "transit_buckets_in",
                &logical_state.transit_buckets_in,
                &compact_state.transit_buckets_in,
            ),
            (
                "commit_in",
                &logical_state.commit_in,
                &compact_state.commit_in,
            ),
        ];
        for (name, logical_range, compact_range) in ranges {
            if logical_range.is_empty() {
                continue;
            }
            assert_eq!(
                map.col(logical_range.start),
                Some(compact_range.start),
                "{name} start"
            );
            assert_eq!(logical_range.len(), compact_range.len(), "{name} length");
        }
        assert_eq!(
            map.col(logical_state.theta),
            Some(compact_state.theta),
            "theta"
        );
        map
    }

    pub fn col(&self, c: usize) -> Option<usize> {
        self.cols[c]
    }

    pub fn row(&self, r: usize) -> Option<usize> {
        self.rows[r]
    }

    /// Maps a logical column range by position; an empty range is an anchor.
    ///
    /// # Panics
    ///
    /// Panics when `range` contains a deleted column.
    pub fn cols(&self, range: Range<usize>) -> Range<usize> {
        compact_range(&self.cols, range, "column")
    }

    /// Maps a logical row range by position; an empty range is an anchor.
    ///
    /// # Panics
    ///
    /// Panics when `range` contains a deleted row.
    pub fn rows(&self, range: Range<usize>) -> Range<usize> {
        compact_range(&self.rows, range, "row")
    }
}

/// Every kept column and every kept row outside `coupled_rows` (compact row
/// indices) has bit-identical scales in both forms; a coupled row's compact
/// scale is finite and positive.
///
/// # Panics
///
/// Panics naming `key`, `stage` and the index on the first violation.
pub fn assert_compact_scales_match(
    key: &str,
    stage: usize,
    map: &CompactMap,
    logical: &StageTemplate,
    compact: &StageTemplate,
    coupled_rows: &[u32],
) {
    for (c, logical_scale) in logical.col_scale.iter().enumerate() {
        if let Some(k) = map.col(c) {
            assert_eq!(
                logical_scale.to_bits(),
                compact.col_scale[k].to_bits(),
                "{key} stage {stage}: col_scale of logical col {c} (compact col {k})"
            );
        }
    }

    let mut coupled = vec![false; compact.num_rows];
    for &row in coupled_rows {
        coupled[row as usize] = true;
    }
    for (r, logical_scale) in logical.row_scale.iter().enumerate() {
        if let Some(k) = map.row(r).filter(|&k| !coupled[k]) {
            assert_eq!(
                logical_scale.to_bits(),
                compact.row_scale[k].to_bits(),
                "{key} stage {stage}: row_scale of logical row {r} (compact row {k})"
            );
        }
    }
    for &row in coupled_rows {
        let scale = compact.row_scale[row as usize];
        assert!(
            scale.is_finite() && scale > 0.0,
            "{key} stage {stage}: coupled compact row {row} row_scale {scale} must be finite and positive"
        );
    }
}

fn column_entries(template: &StageTemplate, c: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
    let start = usize::try_from(template.col_starts[c]).expect("column start is non-negative");
    let end = usize::try_from(template.col_starts[c + 1]).expect("column end is non-negative");
    (start..end).map(move |k| {
        let row = usize::try_from(template.row_indices[k]).expect("row index is non-negative");
        (row, template.values[k])
    })
}

/// On the pre-scaling templates, the compact template is the logical one with
/// the deleted columns and rows removed: kept columns carry the same entries in
/// the same order, and every bound and objective coefficient matches bit for bit.
///
/// # Panics
///
/// Panics naming `key`, `stage` and the index on the first difference.
pub fn assert_compact_template_matches(
    key: &str,
    stage: usize,
    map: &CompactMap,
    logical_state: &StateSpace,
    logical: &StageTemplate,
    compact: &StageTemplate,
) {
    assert_eq!(
        logical.n_state, compact.n_state,
        "{key} stage {stage}: n_state"
    );
    for (side, template) in [("logical", logical), ("compact", compact)] {
        assert!(
            template.col_scale.is_empty() && template.row_scale.is_empty(),
            "{key} stage {stage}: {side} template carries scales"
        );
    }

    let same_bits = |field: &str, kind: &str, index: usize, mapped: usize, l: f64, c: f64| {
        assert_eq!(
            l.to_bits(),
            c.to_bits(),
            "{key} stage {stage}: {field} of logical {kind} {index} (compact {kind} {mapped})"
        );
    };

    let mut kept_entries = 0;
    for c in 0..logical.num_cols {
        let kept = map.col(c);
        let is_lag = logical_state.inflow_lags.contains(&c);
        let mut logical_entries = Vec::new();
        for (r, value) in column_entries(logical, c) {
            let row = map.row(r);
            assert!(
                kept.is_none() || row.is_some(),
                "{key} stage {stage}: kept logical col {c} has an entry in deleted row {r}"
            );
            assert!(
                !is_lag || row.is_none(),
                "{key} stage {stage}: lag col {c} has an entry in kept row {r}"
            );
            logical_entries.push((row, value.to_bits()));
        }
        let Some(k) = kept else { continue };

        let compact_entries: Vec<_> = column_entries(compact, k)
            .map(|(r, value)| (Some(r), value.to_bits()))
            .collect();
        assert_eq!(
            logical_entries, compact_entries,
            "{key} stage {stage}: entries of logical col {c} (compact col {k})"
        );
        kept_entries += compact_entries.len();

        same_bits(
            "col_lower",
            "col",
            c,
            k,
            logical.col_lower[c],
            compact.col_lower[k],
        );
        same_bits(
            "col_upper",
            "col",
            c,
            k,
            logical.col_upper[c],
            compact.col_upper[k],
        );
        same_bits(
            "objective",
            "col",
            c,
            k,
            logical.objective[c],
            compact.objective[k],
        );
    }
    assert_eq!(
        compact.num_nz, kept_entries,
        "{key} stage {stage}: num_nz is the kept columns' entry total"
    );

    for r in 0..logical.num_rows {
        let Some(k) = map.row(r) else { continue };
        same_bits(
            "row_lower",
            "row",
            r,
            k,
            logical.row_lower[r],
            compact.row_lower[k],
        );
        same_bits(
            "row_upper",
            "row",
            r,
            k,
            logical.row_upper[r],
            compact.row_upper[k],
        );
    }
}

struct GeometryFields<'a> {
    col_ranges: Vec<(&'static str, &'a Range<usize>)>,
    row_ranges: Vec<(&'static str, &'a Range<usize>)>,
    row_families: [(&'static str, BlockRowFamily); 2],
    evap_indices: &'a [EvaporationIndices],
    storage_internal_start: usize,
    n_blks: usize,
    block_mode: BlockMode,
    hydro_lists: [(&'static str, &'a [HydroSys]); 6],
}

fn classify_geometry(geometry: &StageGeometry) -> GeometryFields<'_> {
    let StageGeometry {
        turbine,
        spillage,
        diversion,
        thermal,
        anticipated_decision,
        line_fwd,
        line_rev,
        deficit,
        excess,
        generation,
        ncs_generation,
        pumping_flow,
        evap_indices,
        inflow_slack,
        withdrawal_slack_neg,
        withdrawal_slack_pos,
        outflow_below_slack,
        outflow_above_slack,
        turbine_below_slack,
        generation_below_slack,
        plant_turbine_below_slack,
        plant_generation_below_slack,
        contract_import,
        contract_export,
        water_balance,
        load_balance,
        fpha,
        filling_target,
        filling_target_col,
        filled_min_storage_floor,
        filled_min_storage_floor_col,
        plant_max_turbine,
        plant_max_generation,
        n_blks,
        storage_internal_start,
        block_mode,
        fpha_hydro_indices,
        evap_hydro_indices,
        filling_target_hydro_indices,
        filled_min_storage_floor_hydro_indices,
        plant_floor_hydro_indices,
        plant_cap_hydro_indices,
    } = geometry;
    GeometryFields {
        col_ranges: vec![
            ("turbine", turbine),
            ("spillage", spillage),
            ("diversion", diversion),
            ("thermal", thermal),
            ("anticipated_decision", anticipated_decision),
            ("line_fwd", line_fwd),
            ("line_rev", line_rev),
            ("deficit", deficit),
            ("excess", excess),
            ("generation", generation),
            ("ncs_generation", ncs_generation),
            ("pumping_flow", pumping_flow),
            ("inflow_slack", inflow_slack),
            ("withdrawal_slack_neg", withdrawal_slack_neg),
            ("withdrawal_slack_pos", withdrawal_slack_pos),
            ("outflow_below_slack", outflow_below_slack),
            ("outflow_above_slack", outflow_above_slack),
            ("turbine_below_slack", turbine_below_slack),
            ("generation_below_slack", generation_below_slack),
            ("plant_turbine_below_slack", plant_turbine_below_slack),
            ("plant_generation_below_slack", plant_generation_below_slack),
            ("contract_import", contract_import),
            ("contract_export", contract_export),
            ("filling_target_col", filling_target_col),
            ("filled_min_storage_floor_col", filled_min_storage_floor_col),
        ],
        row_ranges: vec![
            ("fpha", fpha),
            ("filling_target", filling_target),
            ("filled_min_storage_floor", filled_min_storage_floor),
            ("plant_max_turbine", plant_max_turbine),
            ("plant_max_generation", plant_max_generation),
        ],
        row_families: [
            ("water_balance", *water_balance),
            ("load_balance", *load_balance),
        ],
        evap_indices,
        storage_internal_start: *storage_internal_start,
        n_blks: *n_blks,
        block_mode: *block_mode,
        hydro_lists: [
            ("fpha_hydro_indices", fpha_hydro_indices),
            ("evap_hydro_indices", evap_hydro_indices),
            ("filling_target_hydro_indices", filling_target_hydro_indices),
            (
                "filled_min_storage_floor_hydro_indices",
                filled_min_storage_floor_hydro_indices,
            ),
            ("plant_floor_hydro_indices", plant_floor_hydro_indices),
            ("plant_cap_hydro_indices", plant_cap_hydro_indices),
        ],
    }
}

/// Every geometry address of the compact stage is its logical address mapped by
/// position; the block count, block mode and hydro index lists are equal.
///
/// # Panics
///
/// Panics naming `key`, `stage` and the field on the first difference.
pub fn assert_compact_geometry_matches(
    key: &str,
    stage: usize,
    map: &CompactMap,
    logical: &StageGeometry,
    compact: &StageGeometry,
) {
    let l = classify_geometry(logical);
    let c = classify_geometry(compact);
    assert_eq!(l.n_blks, c.n_blks, "{key} stage {stage}: geometry n_blks");
    assert_eq!(
        l.block_mode, c.block_mode,
        "{key} stage {stage}: geometry block_mode"
    );

    for ((name, logical_range), (_, compact_range)) in l.col_ranges.iter().zip(&c.col_ranges) {
        assert_eq!(
            map.cols((*logical_range).clone()),
            **compact_range,
            "{key} stage {stage}: geometry column range {name}"
        );
    }
    for ((name, logical_range), (_, compact_range)) in l.row_ranges.iter().zip(&c.row_ranges) {
        assert_eq!(
            map.rows((*logical_range).clone()),
            **compact_range,
            "{key} stage {stage}: geometry row range {name}"
        );
    }
    for ((name, logical_family), (_, compact_family)) in l.row_families.iter().zip(&c.row_families)
    {
        assert_eq!(
            map.rows(logical_family.range()),
            compact_family.range(),
            "{key} stage {stage}: geometry row family {name}"
        );
        assert_eq!(
            logical_family.rows_per_entity(l.n_blks),
            compact_family.rows_per_entity(c.n_blks),
            "{key} stage {stage}: geometry row family {name} rows_per_entity"
        );
    }

    assert_eq!(
        l.evap_indices.len(),
        c.evap_indices.len(),
        "{key} stage {stage}: geometry evap_indices length"
    );
    for (i, (logical_evap, compact_evap)) in l.evap_indices.iter().zip(c.evap_indices).enumerate() {
        let columns = [
            (
                "evaporation_flow_col",
                logical_evap.evaporation_flow_col,
                compact_evap.evaporation_flow_col,
            ),
            (
                "f_evap_plus_col",
                logical_evap.f_evap_plus_col,
                compact_evap.f_evap_plus_col,
            ),
            (
                "f_evap_minus_col",
                logical_evap.f_evap_minus_col,
                compact_evap.f_evap_minus_col,
            ),
        ];
        for (field, logical_col, compact_col) in columns {
            assert_eq!(
                map.cols(logical_col..logical_col + 1),
                compact_col..compact_col + 1,
                "{key} stage {stage}: geometry evap_indices[{i}].{field}"
            );
        }
        assert_eq!(
            map.rows(logical_evap.evap_row..logical_evap.evap_row + 1),
            compact_evap.evap_row..compact_evap.evap_row + 1,
            "{key} stage {stage}: geometry evap_indices[{i}].evap_row"
        );
    }

    let anchor = l.storage_internal_start;
    assert_eq!(
        map.cols(anchor..anchor).start,
        c.storage_internal_start,
        "{key} stage {stage}: geometry storage_internal_start"
    );
    for ((name, logical_list), (_, compact_list)) in l.hydro_lists.iter().zip(&c.hydro_lists) {
        assert_eq!(
            logical_list, compact_list,
            "{key} stage {stage}: geometry {name}"
        );
    }
}

/// Generic-constraint row entries are equal field by field, with the slack
/// columns mapped by position.
///
/// # Panics
///
/// Panics naming `key`, `stage` and the entry index on the first difference.
pub fn assert_compact_generic_rows_match(
    key: &str,
    stage: usize,
    map: &CompactMap,
    logical: &[GenericConstraintRowEntry],
    compact: &[GenericConstraintRowEntry],
) {
    assert_eq!(
        logical.len(),
        compact.len(),
        "{key} stage {stage}: generic row count"
    );
    for (i, (logical_entry, compact_entry)) in logical.iter().zip(compact).enumerate() {
        let GenericConstraintRowEntry {
            constraint_idx,
            entity_id,
            block_idx,
            is_stage_level,
            bound_lower,
            bound_upper,
            slack,
            slack_plus_col,
            slack_minus_col,
        } = logical_entry;
        let GenericConstraintRowEntry {
            constraint_idx: compact_constraint_idx,
            entity_id: compact_entity_id,
            block_idx: compact_block_idx,
            is_stage_level: compact_is_stage_level,
            bound_lower: compact_bound_lower,
            bound_upper: compact_bound_upper,
            slack: compact_slack,
            slack_plus_col: compact_slack_plus_col,
            slack_minus_col: compact_slack_minus_col,
        } = compact_entry;
        let at = format!("{key} stage {stage}: generic row {i}");

        assert_eq!(
            constraint_idx, compact_constraint_idx,
            "{at} constraint_idx"
        );
        assert_eq!(entity_id, compact_entity_id, "{at} entity_id");
        assert_eq!(block_idx, compact_block_idx, "{at} block_idx");
        assert_eq!(
            is_stage_level, compact_is_stage_level,
            "{at} is_stage_level"
        );
        assert_eq!(
            bound_lower.map(f64::to_bits),
            compact_bound_lower.map(f64::to_bits),
            "{at} bound_lower"
        );
        assert_eq!(
            bound_upper.map(f64::to_bits),
            compact_bound_upper.map(f64::to_bits),
            "{at} bound_upper"
        );
        assert_eq!(slack, compact_slack, "{at} slack");
        assert_eq!(
            slack_plus_col.map(|col| map.col(col)),
            compact_slack_plus_col.map(Some),
            "{at} slack_plus_col"
        );
        assert_eq!(
            slack_minus_col.map(|col| map.col(col)),
            compact_slack_minus_col.map(Some),
            "{at} slack_minus_col"
        );
    }
}
