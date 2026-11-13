//! The forward node-solve both forward drivers run: prep, solve, stage cost,
//! outgoing state.

use novomodelo_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    cut::{CutPool, chain_rule::ChainRuleScratch},
    error::SddpError,
    indexer::{ChainRuleSlots, CutStateProjection},
    setup::node_graph::{NodePos, StageIdx},
    solve::{LazyInputs, StageOpeningSolver},
    stage_solve::{
        StageInputs, advance_worker_state, fill_unscaled, run_stage_solve_terminal_static,
    },
    training::{
        backward::{
            ChainRuleStageInputs, DualExtractionSpec, OpeningDualBuffers, extract_opening_cut,
        },
        stage_solve_prep::{SingleSolvePrepParams, StageSolvePrep, StateSource},
    },
    workspace::{CapturedBasis, SolverWorkspace},
};

/// A node's terminal-leaf subgradient, captured on the forward for the
/// backward to consume instead of re-solving (`NodeGraph::is_external_terminal_leaf`).
/// `captured` is `false` for every non-eligible node; `objective`/`subgradient`
/// are meaningful only when it is `true`.
#[derive(Clone, Default)]
pub(super) struct FusedTerminalSlice {
    pub(super) captured: bool,
    pub(super) objective: f64,
    /// The opening's cut coefficients in the parent projection: the chain-rule result.
    pub(super) subgradient: Vec<f64>,
}

impl FusedTerminalSlice {
    /// An uncaptured slice whose subgradient never grows past `n_state` entries.
    pub(super) fn with_capacity(n_state: usize) -> Self {
        Self {
            subgradient: Vec::with_capacity(n_state),
            ..Self::default()
        }
    }
}

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
/// `fused_out` is filled from the solved view in either solve mode, or marked
/// uncaptured when `fusion_cut_state` is `None`.
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
        pool,
        stored_basis,
        opening_solver,
        basis_apply,
        fusion_cut_state,
        ..
    } = *solve;
    let node_graph = training_ctx.node_graph;
    let pool_id = node_graph.nodes[node].pool_id;
    let resident = &training_ctx.cut_state_layouts[pool_id];
    let state = training_ctx.state;
    let is_terminal = training_ctx.horizon.is_terminal(t.next().0);

    prepare_stage_solve(ws, ctx, training_ctx, solve, resident);
    // Zero theta at the terminal stage (no successor to penalise), but NOT when
    // boundary cuts are loaded: those constrain theta from below and must stay
    // visible in the objective.
    if is_terminal && !pool.has_prefix_cuts() {
        ws.solver.set_col_bounds(&[state.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.template(t).col_scale;

    let fuses = fused_out.is_some() && fusion_cut_state.is_some();

    // Taken out before the solve so they can be filled from `view` slices tied to
    // `ws`; restored after the last read so the next stage reuses the allocation.
    let mut unscaled_primal = std::mem::take(&mut ws.scratch.unscaled_primal);
    let (cut_row_slots, mut state_duals, mut coupled_duals, mut cut_dual_slots) = if fuses {
        (
            std::mem::take(&mut ws.scratch.cut_slots_buf),
            std::mem::take(&mut ws.backward_accum.state_duals_buf),
            std::mem::take(&mut ws.backward_accum.coupled_duals_buf),
            std::mem::take(&mut ws.backward_accum.cut_dual_slots),
        )
    } else {
        Default::default()
    };
    let mut opening_duals = OpeningDualBuffers {
        state_duals: &mut state_duals,
        coupled_duals: &mut coupled_duals,
        cut_dual_slots: &mut cut_dual_slots,
    };
    let mut chain_scratch = if fuses {
        std::mem::take(&mut ws.backward_accum.chain_scratch)
    } else {
        ChainRuleScratch::default()
    };

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
                cut_layout: resident,
                continue_carry: false,
            },
        )?,
    };
    fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
    // The terminal boundary theta prices the post-horizon value-to-go and stays in
    // the cost; the interior form would drop it from the UB alone.
    let stage_cost = if is_terminal && pool.has_prefix_cuts() {
        view.objective * ctx.cost_scale_factor
    } else {
        (view.objective - ctx.discount_factor(t) * unscaled_primal[state.theta])
            * ctx.cost_scale_factor
    };

    match (fused_out, fusion_cut_state) {
        (Some(out), Some(parent)) => {
            let template = ctx.template(t);
            out.subgradient.resize(parent.n_slots(), 0.0);
            // `resident` is the solved leaf's own projection and `parent` the extracted one;
            // swapping them compiles.
            out.objective = extract_opening_cut(
                &view,
                &DualExtractionSpec {
                    extracted: parent,
                    col_scale: &template.col_scale,
                    row_scale: &template.row_scale,
                    coupling: &ctx.inflow_coupling[t.0],
                    first_cut_row: template.num_rows,
                    cut_row_slots: &cut_row_slots,
                    read_cut_duals: resident.has_lag_slots(),
                },
                &ChainRuleStageInputs {
                    slots: &ChainRuleSlots::compose(training_ctx.state, resident, parent),
                    resident_pool: pool,
                    par: training_ctx.stochastic.par(),
                    stage: t.0,
                },
                &mut opening_duals,
                &mut chain_scratch,
                &mut out.subgradient,
            );
            out.captured = true;
            ws.backward_accum.chain_scratch = chain_scratch;
            ws.scratch.cut_slots_buf = cut_row_slots;
            ws.backward_accum.state_duals_buf = state_duals;
            ws.backward_accum.coupled_duals_buf = coupled_duals;
            ws.backward_accum.cut_dual_slots = cut_dual_slots;
        }
        (Some(out), None) => out.captured = false,
        (None, _) => {}
    }

    ws.scratch.unscaled_primal = unscaled_primal;
    advance_worker_state(
        ws,
        state,
        ctx.state_box(t),
        ctx.stage_lag(t),
        training_ctx.study_dims.downstream_par_order,
    );

    Ok(stage_cost)
}

fn prepare_stage_solve<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    solve: &ForwardNodeSolve<'_>,
    resident: &CutStateProjection,
) {
    let ForwardNodeSolve {
        t,
        raw_noise,
        pool,
        opening_solver,
        ..
    } = *solve;
    StageSolvePrep::prepare_single(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &SingleSolvePrepParams {
            state_source: StateSource(&ws.current_state),
            raw_noise,
            pool,
            projection: resident,
            opening_solver,
        },
        pool.active_cuts().map(|(slot, _, _)| slot),
    );
}
