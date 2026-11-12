//! The per-opening solve both backward schedulers run, in either opening-solve mode.

use novomodelo_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    setup::node_graph::StageIdx,
    solve::{LazyInputs, StageOpeningSolver},
    solver_stats::SolverStatsDelta,
    stage_solve::StageInputs,
    training::stage_solve_prep::{NoiseTermSource, StageSolvePrep},
    workspace::{CapturedBasis, SolverWorkspace},
};

use super::{
    SuccessorChild, SuccessorSpec,
    duals_extraction::{
        DualExtractionSpec, OpeningDualBuffers, extract_opening_duals, fill_resident_cut_dual_slots,
    },
    lp_setup::child_cut_rows,
    outcome_aggregation::count_binding_slots,
};

/// Solves one opening of `child`'s already-loaded LP, which the caller has run
/// `pin_child_lp` on, extracts its duals, records the opening under
/// `(trial_pos, omega)` and returns its simplex-iteration delta.
///
/// `stored_basis` warm-starts a frozen solve; basis capture stays with the caller
/// (sddp.md "By-node scheduler is warm-start-only"). A lazy solve ignores it and
/// carries the previous opening's LP when `continue_carry` is `true`, which is
/// `!is_first` (sddp.md "Backward opening order is warm-start-only").
///
/// A lazy solve reads its cut-row duals through the row map of its resident set,
/// since its cut rows are not in the order of the frozen LP's.
// Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ, child,
// stored_basis, opening_solver) and per-opening scalars (raw_noise, s,
// scenario, iteration, trial_pos, omega, continue_carry); no natural grouping
// reduces caller-side borrows.
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
    s: StageIdx,
    scenario: usize,
    iteration: u64,
    trial_pos: usize,
    omega: usize,
    continue_carry: bool,
) -> Result<u64, SddpError> {
    let noise = match succ.stage_noise {
        Some(table) => NoiseTermSource::Stage {
            noise_term: table.noise_term(omega),
            cut_noise: table.cut_noise(omega, child.successor_active_slots.len()),
        },
        None => NoiseTermSource::OnTheFly,
    };
    StageSolvePrep::patch_opening(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        s,
        raw_noise,
        &child_cut_rows(training_ctx, child, child.successor_active_slots),
        noise,
        opening_solver,
    );

    // Moved out before the solve to avoid a borrow conflict with `view`'s
    // lifetime; pre-warmed capacity is reused across openings.
    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
    let mut coupled_duals = std::mem::take(&mut ws.backward_accum.coupled_duals_buf);
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

    let template = ctx.template(s);
    let col_scale = &template.col_scale;
    let read_cut_duals = training_ctx.cut_state_layouts[child.pool_id].has_lag_slots()
        || child.cut_activity_tolerance.is_some();
    let frozen = matches!(opening_solver, StageOpeningSolver::Frozen);
    let objective = extract_opening_duals(
        &view,
        &DualExtractionSpec {
            extracted: succ.cut_state,
            col_scale,
            row_scale: &template.row_scale,
            coupling: &ctx.inflow_coupling[s.0],
            first_cut_row: child.template_num_rows,
            cut_row_slots: child.successor_active_slots,
            read_cut_duals: read_cut_duals && frozen,
        },
        &mut OpeningDualBuffers {
            state_duals: &mut state_duals,
            coupled_duals: &mut coupled_duals,
            cut_dual_slots: &mut cut_dual_slots,
        },
    );
    if read_cut_duals && !frozen {
        let dcs_solve = &ws.backward_accum.dcs_solve;
        fill_resident_cut_dual_slots(
            dcs_solve.result_view().dual,
            &dcs_solve.row_map,
            child.successor_pool,
            &mut cut_dual_slots,
        );
    }

    ws.backward_accum.state_duals_buf = state_duals;
    ws.backward_accum.coupled_duals_buf = coupled_duals;
    ws.backward_accum.cut_dual_slots = cut_dual_slots;

    let mut stats_after = std::mem::take(&mut ws.backward_accum.stats_after_buf);
    ws.solver.statistics_into(&mut stats_after);

    let accum = &mut ws.backward_accum;
    SolverStatsDelta::from_snapshots_into(
        &mut accum.opening_delta_buf,
        &stats_before,
        &stats_after,
    );
    let simplex_iterations = accum.opening_delta_buf.simplex_iterations;
    SolverStatsDelta::accumulate_into(
        &mut accum.per_opening_stats[omega],
        &accum.opening_delta_buf,
    );
    accum.opening_records.push(
        (trial_pos, omega),
        objective,
        &accum.state_duals_buf,
        &accum.coupled_duals_buf,
        &accum.cut_dual_slots,
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
