//! The forward node-solve both forward drivers run: prep, solve, stage cost,
//! outgoing state.

use novomodelo_solver::{SolutionView, SolverInterface};

use crate::{
    context::{StageContext, TrainingContext},
    cut::{
        CutPool,
        chain_rule::{ChainRuleInputs, OpeningDuals, per_opening_cut_into},
    },
    error::SddpError,
    indexer::{ChainRuleSlots, CutStateProjection, InflowForm},
    noise::{DownstreamAccumState, LagAccumState},
    setup::node_graph::{NodePos, StageIdx},
    solve::{LazyInputs, StageOpeningSolver},
    stage_solve::{
        StageInputs, assemble_outgoing_state, fill_unscaled, run_stage_solve_terminal_static,
    },
    training::{
        backward::{FoldedDuals, FoldedExtraction, extract_folded_duals, extract_state_duals_only},
        stage_solve_prep::{
            BracketSource, FoldedCutRows, NoiseTermSource, StageSolvePrep, StageSolvePrepParams,
            StateSource,
        },
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
    /// The opening's cut coefficients in the parent projection: the state duals on the logical LP, the chain-rule result on the compact LP.
    pub(super) subgradient: Vec<f64>,
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
    mut fused_out: Option<&mut FusedTerminalSlice>,
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

    let folded_parent = fusion_cut_state
        .filter(|_| fused_out.is_some() && state.inflow_form == InflowForm::Compact);

    // Taken out before the solve so they can be filled from `view` slices tied to
    // `ws`; restored after the last read so the next stage reuses the allocation.
    let mut unscaled_primal = std::mem::take(&mut ws.scratch.unscaled_primal);
    let (cut_row_slots, mut state_duals, mut coupled_duals, mut cut_dual_slots) =
        if folded_parent.is_some() {
            (
                std::mem::take(&mut ws.scratch.cut_slots_buf),
                std::mem::take(&mut ws.backward_accum.state_duals_buf),
                std::mem::take(&mut ws.backward_accum.coupled_duals_buf),
                std::mem::take(&mut ws.backward_accum.cut_dual_slots),
            )
        } else {
            Default::default()
        };
    let mut folded_duals = FoldedDuals {
        state_duals: &mut state_duals,
        coupled_duals: &mut coupled_duals,
        cut_dual_slots: &mut cut_dual_slots,
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
    if let Some(out) = fused_out.as_deref_mut() {
        match folded_parent {
            Some(parent) => {
                out.objective = extract_compact_slice(
                    &view,
                    ctx,
                    t,
                    parent,
                    resident,
                    &cut_row_slots,
                    &mut folded_duals,
                );
            }
            None => capture_fused_terminal_slice(&view, col_scale, fusion_cut_state, out),
        }
    }

    // The terminal boundary theta prices the post-horizon value-to-go and stays in
    // the cost; the interior form would drop it from the UB alone.
    let stage_cost = if is_terminal && pool.has_prefix_cuts() {
        view.objective * ctx.cost_scale_factor
    } else {
        (view.objective - ctx.discount_factor(t) * unscaled_primal[state.theta])
            * ctx.cost_scale_factor
    };

    // x̂ is `ws.current_state` until `assemble_outgoing_state` overwrites it.
    if let (Some(parent), Some(out)) = (folded_parent, fused_out) {
        chain_rule_compact_slice(
            ws,
            training_ctx,
            solve,
            parent,
            resident,
            &folded_duals,
            out,
        );
        ws.scratch.cut_slots_buf = cut_row_slots;
        ws.backward_accum.state_duals_buf = state_duals;
        ws.backward_accum.coupled_duals_buf = coupled_duals;
        ws.backward_accum.cut_dual_slots = cut_dual_slots;
    }

    ws.scratch.incoming_lags_buf.clear();
    ws.scratch
        .incoming_lags_buf
        .extend_from_slice(&ws.current_state[state.state_dim_lag_range()]);

    assemble_outgoing_state(
        &mut ws.current_state,
        &unscaled_primal,
        &ws.scratch.incoming_lags_buf,
        &ws.scratch.realized_inflow_buf,
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
    match training_ctx.state.inflow_form {
        InflowForm::Logical => {
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
        }
        InflowForm::Compact => {
            let mut slots = std::mem::take(&mut ws.scratch.cut_slots_buf);
            slots.clear();
            slots.extend(pool.active_cuts().map(|(slot, _, _)| slot));
            let rows = FoldedCutRows {
                pool,
                projection: resident,
                slots: &slots,
                first_row: ctx.template(t).num_rows,
            };
            StageSolvePrep::pin_folded(
                &mut ws.solver,
                &mut ws.patch_buf,
                &mut ws.scratch,
                ctx,
                training_ctx,
                t,
                StateSource(&ws.current_state),
                &rows,
                BracketSource::OnTheFly,
            );
            let open = match opening_solver {
                StageOpeningSolver::Frozen => StageSolvePrep::open_folded,
                StageOpeningSolver::Lazy(_) => StageSolvePrep::open_folded_lazy,
            };
            open(
                &mut ws.solver,
                &mut ws.patch_buf,
                &mut ws.scratch,
                ctx,
                training_ctx,
                t,
                raw_noise,
                &rows,
                NoiseTermSource::OnTheFly,
            );
            ws.scratch.cut_slots_buf = slots;
        }
    }
}

/// For an eligible node (`cut_state.is_some()`), extract this node's own
/// incoming-state subgradient via [`extract_state_duals_only`] — reused
/// verbatim, never re-derived (`rc / col_scale`, divided; sddp.md "Benders cut
/// sign & subgradient extraction") — before `view` is dropped; otherwise mark
/// `fused_out` uncaptured.
pub(super) fn capture_fused_terminal_slice(
    view: &SolutionView<'_>,
    col_scale: &[f64],
    cut_state: Option<&CutStateProjection>,
    fused_out: &mut FusedTerminalSlice,
) {
    match cut_state {
        Some(cut_state) => {
            fused_out.objective =
                extract_state_duals_only(view, cut_state, col_scale, &mut fused_out.subgradient);
            fused_out.captured = true;
        }
        None => fused_out.captured = false,
    }
}

fn extract_compact_slice(
    view: &SolutionView<'_>,
    ctx: &StageContext<'_>,
    t: StageIdx,
    parent: &CutStateProjection,
    resident: &CutStateProjection,
    cut_row_slots: &[usize],
    duals: &mut FoldedDuals<'_>,
) -> f64 {
    let template = ctx.template(t);
    extract_folded_duals(
        view,
        &FoldedExtraction {
            extracted: parent,
            col_scale: &template.col_scale,
            row_scale: &template.row_scale,
            coupling: &ctx.inflow_coupling[t.0],
            first_cut_row: template.num_rows,
            cut_row_slots,
            read_cut_duals: resident.has_lag_slots(),
        },
        duals,
    )
}

/// The solved leaf's own projection is the resident one and the parent's the
/// extracted one; swapping them compiles.
fn chain_rule_compact_slice<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    training_ctx: &TrainingContext<'_>,
    solve: &ForwardNodeSolve<'_>,
    parent: &CutStateProjection,
    resident: &CutStateProjection,
    duals: &FoldedDuals<'_>,
    out: &mut FusedTerminalSlice,
) {
    out.subgradient.resize(parent.n_slots(), 0.0);
    per_opening_cut_into(
        &ChainRuleInputs {
            slots: &ChainRuleSlots::compose(training_ctx.state, resident, parent),
            resident_pool: solve.pool,
            extracted: parent,
            par: training_ctx.stochastic.par(),
            stage: solve.t.0,
        },
        &OpeningDuals {
            objective: out.objective,
            state_duals: duals.state_duals,
            coupled_duals: duals.coupled_duals,
            cut_duals: duals.cut_dual_slots,
        },
        &ws.current_state,
        &mut ws.backward_accum.chain_scratch,
        &mut out.subgradient,
    );
    out.captured = true;
}
