//! Benders subgradient extraction from a solved backward LP.
//!
//! The subgradient is `rc_scaled / col_scale[col]` — **divided, not multiplied** —
//! because the incoming-state column pin sets `v_scaled = v_orig / col_scale`
//! (sddp.md "Benders cut sign & subgradient extraction").

use novomodelo_solver::SolutionView;

use crate::{
    cut::{CutPool, CutRowMap},
    indexer::{CutSlot, CutStateProjection},
};

use super::SuccessorChild;

fn fill_state_duals(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) {
    // Divided, not multiplied (module-level contract). Empty col_scale ⇒ raw rc.
    state_duals.clear();
    for j in 0..cut_state.n_slots() {
        let col = cut_state.incoming_column(CutSlot::new(j)).get();
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
}

/// Extract state and cut duals from the live solver view into pre-warmed scratch
/// buffers, returning the LP objective. `state_duals` holds the unscaled
/// incoming-state reduced costs of the stage's enabled cut-state dimensions (the
/// module-level `col_scale`-division contract), iterated over `cut_state`; a
/// disabled dimension's reduced cost is computed in the LP but not extracted.
/// `cut_dual_slots` holds one (pool slot, raw cut-row dual) pair per active cut
/// row when binding is tracked (`succ.cut_activity_tolerance` is `Some`), and is
/// left empty otherwise. A zero dual is still a pair.
pub(crate) fn extract_duals_from_view(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    succ: &SuccessorChild<'_>,
    state_duals: &mut Vec<f64>,
    cut_dual_slots: &mut Vec<(usize, f64)>,
) -> f64 {
    fill_state_duals(view, cut_state, col_scale, state_duals);

    cut_dual_slots.clear();
    if succ.cut_activity_tolerance.is_some() {
        cut_dual_slots.extend(
            succ.successor_active_slots
                .iter()
                .enumerate()
                .take(succ.num_cuts_at_successor)
                .map(|(cut_idx, &slot)| (slot, view.dual[succ.template_num_rows + cut_idx])),
        );
    }

    view.objective
}

/// Fills `out` with (pool slot, raw cut-row dual) pairs for the lazy solve's
/// resident cuts, in ascending slot order.
///
/// Under a lazy solve the resident rows are a row-map-ordered subset, not the
/// frozen cut-row order, so a resident slot's dual is
/// `dual[row_map.lp_row_for_slot(slot)]`. A non-resident slot did not bind (by
/// exactness; else the lazy loop would have added it), so leaving it out is
/// correct.
///
/// `dual` must be the FINAL all-satisfied solve's dual vector and `row_map` the
/// residency that produced it.
pub(crate) fn fill_resident_cut_dual_slots(
    dual: &[f64],
    row_map: &CutRowMap,
    pool: &CutPool,
    out: &mut Vec<(usize, f64)>,
) {
    out.clear();
    out.extend((0..pool.populated()).filter_map(|slot| {
        let lp_row = row_map.lp_row_for_slot(slot)?;
        dual.get(lp_row).map(|&d| (slot, d))
    }));
}

/// State-dual half of [`extract_duals_from_view`] for the lazy-solve path: fills
/// `state_duals` with the unscaled incoming-state reduced costs of the enabled
/// cut-state dimensions, iterated over `cut_state`.
///
/// Unlike [`extract_duals_from_view`] it does NOT read cut-row duals — under
/// lazy-solve the resident cut rows are an insertion-order subset, so the
/// cut-row→slot mapping does not apply ([`fill_resident_cut_dual_slots`] reads
/// them through the row map).
pub(crate) fn extract_state_duals_only(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) -> f64 {
    fill_state_duals(view, cut_state, col_scale, state_duals);

    view.objective
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use novomodelo_solver::SolutionView;

    use super::{extract_state_duals_only, fill_resident_cut_dual_slots};
    use crate::cut::{CutPool, CutRowMap};
    use crate::indexer::{CutStateProjection, StateDim, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::setup::NodeId;
    use novomodelo_core::temporal::StageStateConfig;

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &lag_counts,
        )
    }

    /// Build a `SolutionView` whose `reduced_costs[col]` equals `col as f64`, so
    /// an extracted subgradient at LP column `col` is exactly `col as f64`
    /// (`col_scale` empty ⇒ raw rc).
    fn indexed_view(reduced_costs: &[f64]) -> SolutionView<'_> {
        SolutionView {
            objective: 0.0,
            primal: &[],
            dual: &[],
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Storage-only `CutStateProjection` (`N=3, L=2`): extraction yields an `N`-length
    /// subgradient reading exactly the storage columns `[0, N)`, no lag entries.
    #[test]
    fn storage_only_yields_n_length_subgradient_with_storage_columns() {
        let global = state_layout(3, 2);
        let cut_state = CutStateProjection::new(&global, STORAGE_ONLY);
        assert_eq!(cut_state.n_slots(), 3);

        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let view = indexed_view(&rc);
        let mut state_duals = Vec::new();

        let _ = extract_state_duals_only(&view, &cut_state, &[], &mut state_duals);

        assert_eq!(state_duals.len(), 3);
        for (j, &dual) in state_duals.iter().enumerate() {
            assert_eq!(
                dual,
                global.state_to_lp_incoming_column(StateDim::new(j)).get() as f64,
                "storage slot {j} reads its storage_in column reduced cost"
            );
        }
    }

    /// All-enabled `CutStateProjection` reproduces the pre-change global-loop result:
    /// `n_slots()` equals the global `n_state` and every slot reads the same LP
    /// column the global `StateSpace` resolver would.
    #[test]
    fn all_enabled_reproduces_global_loop_result() {
        let global = state_layout(3, 2);
        let cut_state = CutStateProjection::new(&global, ALL_ENABLED);
        assert_eq!(cut_state.n_slots(), global.n_state);

        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let view = indexed_view(&rc);

        let mut via_cut_state = Vec::new();
        let _ = extract_state_duals_only(&view, &cut_state, &[], &mut via_cut_state);

        let global_loop: Vec<f64> = (0..global.n_state)
            .map(|j| rc[global.state_to_lp_incoming_column(StateDim::new(j)).get()])
            .collect();

        assert_eq!(via_cut_state, global_loop);
    }

    /// Slots 3 then 1 are resident, so the row map's order (3, 1) is not slot
    /// order; the pairs follow ascending slot and carry a zero dual.
    #[test]
    fn fill_resident_cut_dual_slots_skips_non_resident_slots() {
        let mut pool = CutPool::new(8, 1, 8, 0);
        for forward_pass_index in 0..4 {
            pool.add_cut(NodeId(0), 0, forward_pass_index, 0.0, &[0.0]);
        }
        let mut row_map = CutRowMap::new(8, 2);
        row_map.insert(3);
        row_map.insert(1);
        let dual = [9.0, 9.0, 0.5, 0.0];
        let mut out = vec![(99, 99.0)];

        fill_resident_cut_dual_slots(&dual, &row_map, &pool, &mut out);

        assert_eq!(out, [(1, 0.0), (3, 0.5)]);
    }
}
