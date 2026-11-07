//! The forward node-solve both forward drivers run: prep, solve, stage cost,
//! outgoing state.

use novomodelo_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    cut::CutPool,
    error::SddpError,
    indexer::CutStateProjection,
    noise::{DownstreamAccumState, LagAccumState},
    setup::node_graph::{NodePos, StageIdx},
    solve::{LazyInputs, StageOpeningSolver},
    stage_solve::{
        StageInputs, assemble_outgoing_state, fill_unscaled, run_stage_solve_terminal_static,
    },
    training::stage_solve_prep::{StageSolvePrep, StageSolvePrepParams, StateSource},
    workspace::{CapturedBasis, SolverWorkspace},
};

use super::enumerated::{FusedTerminalSlice, capture_fused_terminal_slice};

/// How a frozen solve applies its stored basis. `TerminalOneToOne` is the sole
/// licensed bypass of `reconstruct_basis`: forward-only, frozen-only, and only
/// where the pool's cut set is provably invariant (the terminal leaf).
#[derive(Clone, Copy)]
pub(super) enum ForwardBasisApply {
    SlotReconstruct,
    TerminalOneToOne,
}

pub(super) struct ForwardNodeSolve<'a> {
    pub(super) t: StageIdx,
    pub(super) node: NodePos,
    pub(super) scenario_index: usize,
    pub(super) iteration: u64,
    pub(super) raw_noise: &'a [f64],
    pub(super) pool: &'a CutPool,
    pub(super) stored_basis: Option<&'a CapturedBasis>,
    pub(super) opening_solver: &'a StageOpeningSolver,
    pub(super) basis_apply: ForwardBasisApply,
    pub(super) fusion_cut_state: Option<&'a CutStateProjection>,
}

/// Solves one forward node at the incoming state on `ws.current_state`, leaves
/// the outgoing state there and returns the raw stage cost. The caller loads the
/// model and owns the basis capture and the trajectory record.
///
/// `fused_out` is filled from the solved view in either solve mode.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from the LP solve.
pub(super) fn solve_forward_node_core<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    solve: &ForwardNodeSolve<'_>,
    fused_out: Option<&mut FusedTerminalSlice>,
) -> Result<f64, SddpError> {
    let ForwardNodeSolve {
        t,
        node,
        scenario_index,
        iteration,
        raw_noise,
        pool,
        stored_basis,
        opening_solver,
        basis_apply,
        fusion_cut_state,
    } = *solve;
    let node_graph = training_ctx.node_graph;
    let pool_id = node_graph.nodes[node].pool_id;
    let state = training_ctx.state;
    let is_terminal = training_ctx.horizon.is_terminal(t.next().0);

    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &prep_params,
    );
    // Zero theta at the terminal stage (no successor to penalise), but NOT when
    // boundary cuts are loaded: those constrain theta from below and must stay
    // visible in the objective.
    if is_terminal && !pool.has_prefix_cuts() {
        ws.solver.set_col_bounds(&[state.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.template(t).col_scale;

    // Taken out before the solve so it can be filled from `view` slices tied to
    // `ws`; restored after the last read so the next stage reuses the allocation.
    let mut unscaled_primal = std::mem::take(&mut ws.scratch.unscaled_primal);

    opening_solver.seed_resident_set(ws, pool, iteration);
    let inputs = StageInputs {
        stage_context: ctx,
        pool,
        stored_basis,
        stage_index: t,
        scenario_index,
        iteration: Some(iteration),
        node_id: node_graph.node_ids[node],
    };
    let view = match (opening_solver, basis_apply) {
        (StageOpeningSolver::Frozen, ForwardBasisApply::TerminalOneToOne) => {
            run_stage_solve_terminal_static(ws, &inputs)?
        }
        _ => opening_solver.solve_over(
            ws,
            &inputs,
            pool.active_cuts(),
            LazyInputs {
                state,
                cut_layout: &training_ctx.cut_state_layouts[pool_id],
                continue_carry: false,
            },
        )?,
    };
    fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
    if let Some(out) = fused_out {
        capture_fused_terminal_slice(&view, col_scale, fusion_cut_state, out);
    }

    // The terminal boundary theta prices the post-horizon value-to-go and stays in
    // the cost; the interior form would drop it from the UB alone.
    let stage_cost = if is_terminal && pool.has_prefix_cuts() {
        view.objective * ctx.cost_scale_factor
    } else {
        (view.objective - ctx.discount_factor(t) * unscaled_primal[state.theta])
            * ctx.cost_scale_factor
    };

    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[state.state_dim_lag_range()]);

    assemble_outgoing_state(
        &mut ws.current_state,
        &unscaled_primal,
        &ws.scratch.lag_matrix_buf,
        state,
        ctx.state_box(t),
        ctx.stage_lag(t),
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: training_ctx.study_dims.downstream_par_order,
        },
    );
    ws.scratch.unscaled_primal = unscaled_primal;

    Ok(stage_cost)
}
