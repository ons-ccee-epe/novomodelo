//! Per-opening outcome, binding-count, and basis-capture accumulators for the
//! backward pass: write the per-opening stats delta and aggregated outcome, bump
//! binding-cut slot increments from (slot, y) pairs, and capture the
//! first-solved opening's basis.

use novomodelo_solver::{SolverInterface, SolverStatistics};

use crate::{
    forward::write_capture_metadata,
    indexer::CutStateProjection,
    solver_stats::SolverStatsDelta,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorChild;

/// Bumps `slot_increments[metadata_offset + slot]` for each (slot, y) pair whose
/// raw cut-row dual exceeds `threshold`.
///
/// The bump lands in the child's OWN pool region (`metadata_offset + slot`), so a
/// fan's sibling pools never collide on a shared slot index. The criterion is the
/// raw dual, strictly, not its magnitude. The bump is a deterministic function of
/// the pairs only — no worker id, rank, or trace — so the order-insensitive
/// metadata allreduce stays rank-invariant.
///
/// `slot_increments` accumulates (summed) across the trial point's openings; the
/// per-trial-point reset happens in the stage loop before the openings run.
pub(crate) fn count_binding_slots(
    cut_dual_slots: &[(usize, f64)],
    threshold: f64,
    slot_increments: &mut [u64],
    metadata_offset: usize,
) {
    for &(slot, y) in cut_dual_slots {
        if y > threshold {
            slot_increments[metadata_offset + slot] += 1;
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
/// the workspace accumulators, without touching binding-count metadata. The
/// gradient and intercept come from the state duals alone.
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

#[cfg(test)]
mod tests {
    use super::count_binding_slots;

    #[test]
    fn count_binding_slots_bumps_only_strictly_binding_pairs_in_the_child_region() {
        let mut slot_increments = vec![0_u64; 10];

        count_binding_slots(
            &[(0, 0.0), (2, 1e-9), (3, 5.0)],
            1e-9,
            &mut slot_increments,
            4,
        );

        assert_eq!(slot_increments, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
    }
}
