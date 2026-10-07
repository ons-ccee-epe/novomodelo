//! Per-opening outcome, binding-count, and basis-capture accumulators for the
//! backward pass: write the per-opening stats delta and aggregated outcome, bump
//! binding-cut slot increments (frozen-order or, for the DCS lazy layout,
//! row-map-correct), and capture the first-solved opening's basis.

use cobre_solver::{SolverInterface, SolverStatistics};

use crate::{
    cut::{CutRowMap, pool::CutPool},
    forward::write_capture_metadata,
    indexer::CutStateProjection,
    solver_stats::SolverStatsDelta,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorChild;

/// Accumulate one opening's solve result (stats delta, outcome, and binding-cut
/// slot increments) into the workspace accumulators. Call after `view` is dropped.
pub(crate) fn accumulate_opening_outcome<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorChild<'_>,
    cut_state: &CutStateProjection,
    omega: usize,
    objective: f64,
    x_hat: &[f64],
    stats_before: &SolverStatistics,
    stats_after: &SolverStatistics,
) {
    write_opening_outcome(
        ws,
        cut_state,
        omega,
        objective,
        x_hat,
        stats_before,
        stats_after,
    );

    // Bump the child's OWN pool region (`metadata_offset + slot`), so a fan's
    // sibling pools never collide on a shared slot index.
    for (cut_idx, &slot) in succ.successor_active_slots.iter().enumerate() {
        if ws
            .backward_accum
            .cut_duals_buf
            .get(cut_idx)
            .is_some_and(|&d| d > succ.cut_activity_tolerance)
        {
            ws.backward_accum.slot_increments[succ.metadata_offset + slot] += 1;
        }
    }
}

/// Accumulate binding-cut slot increments from the DCS lazy solve's final
/// all-satisfied LP, slot-correct under the resident [`CutRowMap`] layout.
///
/// The frozen path bumps by **frozen cut-row order**; under DCS the resident rows
/// are a row-map-ordered subset, so a resident slot's dual is
/// `dual[row_map.lp_row_for_slot(slot)]`. Bumps `slot_increments[slot]` when that
/// dual exceeds `cut_activity_tolerance` — the same binding criterion as the frozen
/// path (raw dual, not magnitude). A non-resident slot did not bind (by exactness;
/// else the lazy loop would have added it), so leaving it uncounted is correct.
///
/// `dual` must be the FINAL all-satisfied solve's dual vector and `row_map` the
/// residency that produced it. The bump is a deterministic function of the
/// resident map and cut-row duals only — no worker id, rank, or trace — so the
/// order-insensitive metadata allreduce stays rank-invariant.
///
/// `slot_increments` accumulates (summed) across the trial point's openings; the
/// per-trial-point reset happens in the stage loop before the openings run.
pub(crate) fn accumulate_dcs_binding_counts(
    dual: &[f64],
    row_map: &CutRowMap,
    pool: &CutPool,
    cut_activity_tolerance: f64,
    slot_increments: &mut [u64],
) {
    for (slot, increment) in slot_increments
        .iter_mut()
        .enumerate()
        .take(pool.populated())
    {
        let Some(lp_row) = row_map.lp_row_for_slot(slot) else {
            continue;
        };
        if dual
            .get(lp_row)
            .is_some_and(|&d| d > cut_activity_tolerance)
        {
            *increment += 1;
        }
    }
}

/// Fold this trial point's binding-cut `slot_increments` into the accumulating
/// `metadata_sync_contribution` over the successor pools' populated region only.
///
/// `populated_len` is `SuccessorOutcomes::total_metadata_len`, never the buffers'
/// capacity: only `[0, populated_len)` is zeroed each stage and only each pool's own
/// populated slice is reduced by `sync_stage_metadata`, so a `slot_increments.len()`
/// bound would fold never-zeroed, never-read tail slots into the contribution.
pub(crate) fn fold_slot_increments_into_sync(
    slot_increments: &[u64],
    metadata_sync_contribution: &mut [u64],
    populated_len: usize,
) {
    for slot in 0..populated_len {
        let increment = slot_increments[slot];
        if increment > 0 {
            metadata_sync_contribution[slot] += increment;
        }
    }
}

/// Write one opening's stats delta and outcome (coefficients + intercept) into
/// the workspace accumulators, without touching binding-count metadata.
///
/// Shared by the all-cuts path (which adds the `slot_increments` update) and the
/// lazy-solve path (which skips it because its resident cut rows are a
/// row-map-ordered subset whose cut-row→slot mapping differs). The gradient and
/// intercept come from the state duals and are identical either way.
pub(crate) fn write_opening_outcome<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    cut_state: &CutStateProjection,
    omega: usize,
    objective: f64,
    x_hat: &[f64],
    stats_before: &SolverStatistics,
    stats_after: &SolverStatistics,
) {
    let opening_delta = SolverStatsDelta::from_snapshots(stats_before, stats_after);
    SolverStatsDelta::accumulate_into(
        &mut ws.backward_accum.per_opening_stats[omega],
        &opening_delta,
    );

    let out = &mut ws.backward_accum.outcomes[omega];
    out.coefficients
        .copy_from_slice(&ws.backward_accum.state_duals_buf);
    out.objective_value = objective;
    // Intercept and coefficients are in scaled cost units (LP duals inherit cost scaling).
    out.intercept = objective - cut_state.dot_trial_state(&out.coefficients, x_hat);
}

/// Capture the post-solve basis at the first-solved opening into
/// `basis_slice[m, successor_node]`.
///
/// Only the first-solved opening (`solve_order[0]`, = canonical ω=0 under the
/// identity order) may capture: a later capture would store a basis whose retained
/// LU factorization has been overwritten by subsequent opening solves, leaving it
/// stale and potentially infeasible when reloaded.
pub(crate) fn save_basis_at_omega_zero<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorChild<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    m: usize,
    x_hat: &[f64],
) {
    let successor_node = succ.successor_node;
    let base_row_count = succ.template_num_rows;
    let cut_row_count = succ.num_cuts_at_successor;
    let captured = basis_slice
        .get_mut(m, successor_node)
        .get_or_insert_with(|| {
            CapturedBasis::new(
                succ.frozen_template.num_cols,
                base_row_count + cut_row_count,
                base_row_count,
                cut_row_count,
                x_hat.len(),
                succ.successor_node_id,
            )
        });
    ws.solver.get_basis(&mut captured.basis);
    write_capture_metadata(
        captured,
        succ.successor_pool,
        base_row_count,
        cut_row_count,
        x_hat,
        succ.successor_node_id,
    );
}
