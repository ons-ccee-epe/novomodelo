//! The deterministic by-scenario kernel.
//!
//! `process_by_scenario_backward` drives a [`StageOpeningSolver`]: it solves a trial
//! point's openings in a run-constant, rank-invariant `solve_order` permutation but
//! writes and aggregates outcomes by canonical ω, so the generated cut is
//! bit-identical regardless of solve order.

use novomodelo_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    cut::{
        FutureCostFunction,
        chain_rule::{ChainRuleInputs, ChildOpenings, OpeningDuals, aggregate_first_cut_into},
    },
    risk_measure::{RiskMeasure, RiskMeasureScratch},
    setup::node_graph::NodeId,
    solve::StageOpeningSolver,
    state_exchange::ExchangeBuffers,
    workspace::{
        BasisStoreSliceMut, SolverWorkspace,
        workspace::{SuccessorAggregateScratch, recycle},
    },
};

use super::{
    StagedCut, SuccessorChild, SuccessorOutcomes, SuccessorSpec,
    lp_setup::{
        OpeningNoise, load_backward_lp, pin_child_lp, resolve_backward_basis, resolve_opening_noise,
    },
    opening_solve::solve_backward_opening,
    outcome_aggregation::{fold_slot_increments_into_sync, save_basis_at_omega_zero},
};

/// Per-CHILD LP load, issued once after `reset_solver_state()` and before that
/// child's opening solves. Each child loads ITS OWN pool's LP, so a fan never
/// reuses child 0's LP (the child-0 collapse). One child ⟹ one load per trial
/// point ⟹ chain byte-parity.
///
/// The lazy mode loads the cut-free core, never the child's frozen template.
pub(super) fn prepare_child_lp<S: SolverInterface + Send>(
    opening_solver: &StageOpeningSolver,
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    succ: &SuccessorSpec<'_>,
    child: &SuccessorChild<'_>,
    iteration: u64,
) {
    match opening_solver {
        StageOpeningSolver::Frozen => load_backward_lp(ws, child),
        StageOpeningSolver::Lazy(_) => {
            ws.solver.load_model(ctx.template(succ.successor));
            opening_solver.seed_resident_set(ws, child.successor_pool, iteration);
        }
    }
}

/// Writes the aggregate-first cut over `succ`'s whole opening set into `coefficients` and returns
/// its intercept: one joint weight call over the canonical-ω costs, then one
/// [`aggregate_first_cut_into`] over the children. `view_of` maps a canonical ω to its duals.
pub(super) fn aggregate_successor_cut<'r>(
    training_ctx: &TrainingContext<'_>,
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    risk_measure: &RiskMeasure,
    x_hat: &[f64],
    view_of: impl Fn(usize) -> OpeningDuals<'r>,
    scratch: &mut SuccessorAggregateScratch,
    risk_scratch: &mut RiskMeasureScratch,
    coefficients: &mut [f64],
) -> f64 {
    let SuccessorAggregateScratch {
        aggregate,
        costs,
        views,
        children,
        ..
    } = scratch;
    let n_openings = succ.probabilities.len();
    let mut opening_views: Vec<OpeningDuals<'r>> = recycle(std::mem::take(views));
    costs.clear();
    for omega in 0..n_openings {
        let view = view_of(omega);
        costs.push(view.objective);
        opening_views.push(view);
    }
    let weights = risk_measure.opening_weights_into(costs, succ.probabilities, risk_scratch);

    let mut child_list: Vec<ChildOpenings<'_>> = recycle(std::mem::take(children));
    for ci in 0..outcomes.n_children() {
        let child = outcomes.child(ci);
        child_list.push(ChildOpenings {
            inputs: ChainRuleInputs {
                slots: child.chain_rule_slots,
                resident_pool: child.successor_pool,
                extracted: succ.cut_state,
                par: training_ctx.stochastic.par(),
                stage: succ.successor.0,
            },
            openings: &opening_views[child.outcome_range.clone()],
        });
    }
    let intercept = aggregate_first_cut_into(&child_list, weights, x_hat, aggregate, coefficients);
    *children = recycle(child_list);
    *views = recycle(opening_views);
    intercept
}

/// The unit's cut: the unit's records located by canonical ω, then
/// [`aggregate_successor_cut`] into `agg_coefficients`.
fn aggregate_unit_records<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    training_ctx: &TrainingContext<'_>,
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    risk_measure: &RiskMeasure,
    x_hat: &[f64],
) -> f64 {
    let accum = &mut ws.backward_accum;
    let n_openings = succ.probabilities.len();
    debug_assert_eq!(
        accum.opening_records.len(),
        n_openings,
        "every opening of the unit records exactly once"
    );
    let mut location = std::mem::take(&mut accum.successor_aggregate.location);
    location.clear();
    location.resize(n_openings, usize::MAX);
    for i in 0..accum.opening_records.len() {
        location[accum.opening_records.key(i).1] = i;
    }
    let records = &accum.opening_records;
    let intercept = aggregate_successor_cut(
        training_ctx,
        succ,
        outcomes,
        risk_measure,
        x_hat,
        |omega| records.view(location[omega]),
        &mut accum.successor_aggregate,
        &mut accum.risk_scratch,
        &mut accum.agg_coefficients,
    );
    accum.successor_aggregate.location = location;
    intercept
}

/// Process one trial point `m` in the backward pass, iterating over the node's
/// successor outcome set child-by-child.
///
/// Each child loads ITS OWN LP (frozen template, delta cut batch, pool, basis key,
/// External column) and solves its own openings, so pricing every child against
/// child 0's LP — the child-0 collapse, which silently misprices a non-interchangeable
/// fan — is unrepresentable. A chain is the one-element case: one child, one LP load
/// per trial point, byte-identical. The single joint `aggregate_cut_into` over the
/// flattened outcome arena is unchanged (joint risk applied once, not nested).
///
/// On the frozen path the post-solve basis is written into `basis_slice` only at each
/// child's first-solved opening; later writes are forbidden (retained-LU corruption).
/// The lazy mode skips capture (see [`StageOpeningSolver::captures_basis`]).
// RATIONALE: each parameter is a disjoint borrow (ws, ctx, training_ctx, exchange, succ,
// outcomes, basis_slice, opening_solver), a risk slice, or a plain scalar. Merging them into a
// struct would add indirection without reducing the caller's borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_by_scenario_backward<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    node_visit_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    opening_solver: &StageOpeningSolver,
    m: usize,
    compacted: usize,
    arena_offset: usize,
) -> Result<StagedCut, SddpError> {
    let tree_view = training_ctx.stochastic.tree_view();
    let x_hat = exchange.state_at(succ.my_rank, m);
    let scenario = fwd_offset + m;
    let s = succ.successor;

    debug_assert_eq!(
        ws.backward_accum.per_opening_stats.len(),
        succ.probabilities.len(),
        "per_opening_stats must be initialised to n_openings before each stage's trial-point loop"
    );
    debug_assert_eq!(
        outcomes.total_outcomes(),
        succ.probabilities.len(),
        "the reified outcome set's total outcomes must equal the flattened weight vector length"
    );

    // Openings within a child are SOLVED in `solve_order(s)` (a run-constant,
    // rank-invariant permutation) but WRITTEN and AGGREGATED by CANONICAL ω (the
    // child's `outcome_range` offset plus the local ω), so the generated cut is
    // bit-identical regardless of solve order: the reorder changes only the
    // warm-start chain, never which cuts are produced.
    let solve_order = tree_view.solve_order_data(s.0);
    let captures_basis = opening_solver.captures_basis();
    ws.backward_accum.opening_records.clear(
        succ.cut_state.columned_len(),
        training_ctx.state.hydro_count,
    );

    for ci in 0..outcomes.n_children() {
        let child = outcomes.child(ci);
        // Fresh cold head per child: each child loads a different LP, so its
        // warm-start chain across its own openings starts clean (CLP determinism).
        // One child ⟹ once per trial point ⟹ chain byte-parity.
        ws.solver.reset_solver_state();
        prepare_child_lp(opening_solver, ws, ctx, succ, &child, iteration);
        let source = succ.bracket_source(m, child.successor_node_id);
        pin_child_lp(ws, ctx, training_ctx, &child, x_hat, s, source);

        // Resolved per child, never once from child 0: each child's External
        // column is its own.
        let mut buf = std::mem::take(&mut ws.backward_accum.external_noise_buf);
        match resolve_opening_noise(
            training_ctx,
            s,
            &child.openings,
            child.successor_node_id,
            &mut buf,
        )? {
            OpeningNoise::External(noise) => {
                let omega = child.outcome_range.start;
                let stored = if captures_basis {
                    resolve_backward_basis(basis_slice, m, child.successor_node)
                } else {
                    None
                };
                solve_backward_opening(
                    opening_solver,
                    ws,
                    ctx,
                    training_ctx,
                    succ,
                    &child,
                    stored,
                    noise,
                    s,
                    scenario,
                    iteration,
                    0,
                    omega,
                    false,
                )?;
                if captures_basis {
                    save_basis_at_omega_zero(ws, &child, basis_slice, m, x_hat);
                }
            }
            OpeningNoise::Generated => {
                debug_assert_eq!(
                    solve_order.len(),
                    child.openings.len,
                    "a Generated child's opening count must equal stage {s}'s own opening count"
                );
                for (solve_pos, &local_omega_u32) in solve_order.iter().enumerate() {
                    let local_omega = local_omega_u32 as usize;
                    let raw_noise = tree_view.opening(s.0, local_omega);
                    let omega = child.outcome_range.start + local_omega;
                    // First-solved opening of THIS child: owns the per-(m, child node)
                    // basis load/capture and the fresh (non-warm-carry) solve.
                    let is_first = solve_pos == 0;
                    let stored = if is_first && captures_basis {
                        resolve_backward_basis(basis_slice, m, child.successor_node)
                    } else {
                        None
                    };
                    solve_backward_opening(
                        opening_solver,
                        ws,
                        ctx,
                        training_ctx,
                        succ,
                        &child,
                        stored,
                        raw_noise,
                        s,
                        scenario,
                        iteration,
                        0,
                        omega,
                        !is_first,
                    )?;
                    if is_first && captures_basis {
                        save_basis_at_omega_zero(ws, &child, basis_slice, m, x_hat);
                    }
                }
            }
        }
        ws.backward_accum.external_noise_buf = buf;
    }

    // Aggregate into `agg_coefficients`, then copy into this trial point's arena
    // slot so the bytes outlive the parallel closure without a per-cut allocation.
    let risk_measure = &risk_measures[succ.t.0];
    let agg_intercept =
        aggregate_unit_records(ws, training_ctx, succ, outcomes, risk_measure, x_hat);
    let n_state = ws.backward_accum.agg_coefficients.len();
    let coefficients_range = arena_offset..arena_offset + n_state;
    debug_assert!(
        coefficients_range.end <= ws.backward_accum.agg_arena.len(),
        "agg_arena must be sized to cover this trial point's slot before the solve"
    );
    ws.backward_accum.agg_arena[coefficients_range.clone()]
        .copy_from_slice(&ws.backward_accum.agg_coefficients);
    // The cut's `forward_pass_index` addresses `CutPool`'s per-iteration slot
    // block at stride `visit_bound[pool]` (the node's own routed visit count),
    // NOT `forward_passes`, so it must be the node-relative position within THIS
    // node's GLOBAL routed subset: `node_visit_offset + compacted`, where
    // `node_visit_offset` is the visits to this node from strictly-lower ranks.
    // On a single rank the offset is `0`, leaving `compacted`; on a single-node
    // level it is `fwd_offset` and `compacted == m`, reducing to the global
    // `scenario` (byte-neutral, cross-rank-disjoint on a chain). The GLOBAL
    // `scenario` (`fwd_offset + m`) is the wrong-but-compiling alternative: it
    // indexes `[0, forward_passes)`, overshooting the smaller `visit_bound` stride
    // on a fan; the bare `fwd_offset + compacted` is the OTHER wrong alternative —
    // `fwd_offset` is the global forward-pass offset, not the node's own lower-rank
    // visit prefix, so across ranks it overshoots the stride and collides.
    let node_relative_index = node_visit_offset + compacted;
    debug_assert!(
        u32::try_from(node_relative_index).is_ok(),
        "node-relative forward-pass index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = node_relative_index as u32;
    if outcomes.cut_activity_tolerance.is_some() {
        fold_slot_increments_into_sync(
            &ws.backward_accum.slot_increments,
            &mut ws.backward_accum.metadata_sync_contribution,
            outcomes.total_metadata_len(),
        );
    }
    Ok(StagedCut {
        trial_state_idx: m,
        intercept: agg_intercept,
        coefficients_range,
        forward_pass_index,
    })
}

/// Merge each worker's staged cuts into `staged_cuts_buf`, order them by trial
/// state, and commit one cut per trial point to the FCF. `trial_state_idx` is
/// globally unique across workers, so the committed order — and thus the cut
/// set — is independent of worker count and completion order. `staged_cuts_buf`
/// is the caller-owned scratch (`BackwardPassState::staged_cuts_buf`), cleared
/// and reused, never reallocated on the hot path.
///
/// # Errors
/// Returns the first failing worker's [`SddpError`]; on error no cut is committed.
// Rationale: disjoint borrows (worker_staged, workspaces, fcf, the staged-cut
// scratch) plus the node/pool/iteration commit scalars; a struct would add
// indirection without reducing the caller's borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn by_scenario_finish<S: SolverInterface>(
    worker_staged: Vec<Result<(usize, Vec<StagedCut>), SddpError>>,
    workspaces: &[SolverWorkspace<S>],
    trial_points: &[usize],
    cut_n_state: usize,
    fcf: &mut FutureCostFunction,
    node_id: NodeId,
    pool: usize,
    iteration: u64,
    staged_cuts_buf: &mut Vec<(usize, StagedCut)>,
) -> Result<usize, SddpError> {
    staged_cuts_buf.clear();
    for worker_result in worker_staged {
        let (w, cuts) = worker_result?;
        staged_cuts_buf.extend(cuts.into_iter().map(|cut| (w, cut)));
    }
    // `trial_state_idx` is the SOLE sort key: globally unique across workers
    // (disjoint contiguous partitions), so the merge order is identical regardless
    // of worker index.
    staged_cuts_buf.sort_by_key(|(_, cut)| cut.trial_state_idx);
    debug_assert_eq!(staged_cuts_buf.len(), trial_points.len());
    for (w, cut) in &*staged_cuts_buf {
        let range = cut.coefficients_range.clone();
        let arena = &workspaces[*w].backward_accum.agg_arena;
        debug_assert!(
            range.len() == cut_n_state && range.end <= arena.len(),
            "coefficients_range must span exactly the pool's cut n_state and lie within the worker arena"
        );
        fcf.add_cut(
            node_id,
            pool,
            iteration,
            cut.forward_pass_index,
            cut.intercept,
            &arena[range],
        );
    }
    Ok(staged_cuts_buf.len())
}
