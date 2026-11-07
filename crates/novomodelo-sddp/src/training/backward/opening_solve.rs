//! The per-opening solve both backward schedulers run, in either opening-solve mode.

use novomodelo_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    setup::node_graph::StageIdx,
    solve::{LazyInputs, StageOpeningSolver},
    stage_solve::StageInputs,
    workspace::{CapturedBasis, SolverWorkspace},
};

use super::{
    SuccessorChild, SuccessorSpec,
    duals_extraction::{
        extract_duals_from_view, extract_state_duals_only, fill_resident_cut_dual_slots,
    },
    lp_setup::patch_opening_bounds,
    outcome_aggregation::{count_binding_slots, write_opening_outcome},
};

/// Solves one opening of `child`'s already-loaded LP at `x_hat`, extracts its
/// duals and accumulates its outcome, returning the opening's simplex-iteration
/// delta.
///
/// `stored_basis` warm-starts a frozen solve; basis capture stays with the caller
/// (sddp.md "By-node scheduler is warm-start-only"). A lazy solve ignores it and
/// carries the previous opening's LP when `continue_carry` is `true`, which is
/// `!is_first` (sddp.md "Backward opening order is warm-start-only").
// Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ, child,
// stored_basis, opening_solver) and per-opening scalars (raw_noise, x_hat, s,
// scenario, iteration, omega, continue_carry); no natural grouping reduces
// caller-side borrows.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_backward_opening<S: SolverInterface + Send>(
    opening_solver: &StageOpeningSolver,
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    succ: &SuccessorSpec<'_>,
    child: &SuccessorChild<'_>,
    stored_basis: Option<&CapturedBasis>,
    raw_noise: &[f64],
    x_hat: &[f64],
    s: StageIdx,
    scenario: usize,
    iteration: u64,
    omega: usize,
    continue_carry: bool,
) -> Result<u64, SddpError> {
    patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

    // Moved out before the solve to avoid a borrow conflict with `view`'s
    // lifetime; pre-warmed capacity is reused across openings.
    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
    let mut cut_dual_slots = std::mem::take(&mut ws.backward_accum.cut_dual_slots);

    let mut stats_before = std::mem::take(&mut ws.backward_accum.stats_before_buf);
    ws.solver.statistics_into(&mut stats_before);

    let inputs = StageInputs {
        stage_context: ctx,
        pool: child.successor_pool,
        stored_basis,
        stage_index: s,
        scenario_index: scenario,
        iteration: Some(iteration),
        node_id: child.successor_node_id,
    };

    // The child's pool's projection, resolved by the child's node position
    // (`child.pool_id`), never `nodes[stage]`, which mis-indexes a branching
    // graph; `succ.cut_state` serves only the incoming extraction below.
    let lazy = LazyInputs {
        state: training_ctx.state,
        cut_layout: &training_ctx.cut_state_layouts[child.pool_id],
        continue_carry,
    };
    let view = opening_solver.solve_over(ws, &inputs, child.successor_pool.active_cuts(), lazy)?;

    let col_scale = &ctx.template(s).col_scale;
    let objective = match opening_solver {
        StageOpeningSolver::Frozen => extract_duals_from_view(
            &view,
            succ.cut_state,
            col_scale,
            child,
            &mut state_duals,
            &mut cut_dual_slots,
        ),
        StageOpeningSolver::Lazy(_) => {
            let objective =
                extract_state_duals_only(&view, succ.cut_state, col_scale, &mut state_duals);
            if child.cut_activity_tolerance.is_some() {
                let dcs_solve = &ws.backward_accum.dcs_solve;
                fill_resident_cut_dual_slots(
                    dcs_solve.result_view().dual,
                    &dcs_solve.row_map,
                    child.successor_pool,
                    &mut cut_dual_slots,
                );
            }
            objective
        }
    };

    ws.backward_accum.state_duals_buf = state_duals;
    ws.backward_accum.cut_dual_slots = cut_dual_slots;

    let mut stats_after = std::mem::take(&mut ws.backward_accum.stats_after_buf);
    ws.solver.statistics_into(&mut stats_after);

    // Counters are monotonically increasing (mirrors `SolverStatsDelta::from_snapshots`).
    let simplex_iterations = stats_after.total_iterations - stats_before.total_iterations;

    write_opening_outcome(
        ws,
        succ.cut_state,
        omega,
        objective,
        x_hat,
        &stats_before,
        &stats_after,
    );
    if let Some(threshold) = child.cut_activity_tolerance {
        count_binding_slots(
            &ws.backward_accum.cut_dual_slots,
            threshold,
            &mut ws.backward_accum.slot_increments,
            child.metadata_offset,
        );
    }

    ws.backward_accum.stats_before_buf = stats_before;
    ws.backward_accum.stats_after_buf = stats_after;

    Ok(simplex_iterations)
}
