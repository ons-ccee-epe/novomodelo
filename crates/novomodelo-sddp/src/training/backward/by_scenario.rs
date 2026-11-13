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
    cut::FutureCostFunction,
    risk_measure::RiskMeasure,
    setup::node_graph::NodeId,
    solve::StageOpeningSolver,
    workspace::{BasisStoreSliceMut, SolverWorkspace},
};

use super::{
    BackwardNodeParams, StagedCut, SuccessorChild, SuccessorOutcomes, SuccessorSpec,
    binding_count::fold_slot_increments_into_sync,
    lp_setup::{
        load_backward_lp, pin_child_lp, resolve_backward_basis, resolve_opening_noise,
        save_basis_at_omega_zero,
    },
    opening_solve::{OpeningRequest, solve_backward_opening},
    successor_cut::aggregate_successor_cut,
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
) {
    match opening_solver {
        StageOpeningSolver::Frozen => load_backward_lp(ws, child),
        StageOpeningSolver::Lazy(_) => {
            ws.solver.load_model(ctx.template(succ.successor));
            opening_solver.seed_resident_set(ws, child.successor_pool, succ.iteration);
        }
    }
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
/// per trial point, byte-identical. The node's cut is one joint
/// [`aggregate_successor_cut`] over the flattened outcome set (joint risk applied once,
/// not nested).
///
/// On the frozen path the post-solve basis is written into `basis_slice` only at each
/// child's first-solved opening; later writes are forbidden (retained-LU corruption).
/// The lazy mode skips capture (see [`StageOpeningSolver::captures_basis`]).
pub(crate) fn process_by_scenario_backward<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    params: &BackwardNodeParams<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    opening_solver: &StageOpeningSolver,
    m: usize,
    compacted: usize,
    arena_offset: usize,
) -> Result<StagedCut, SddpError> {
    let BackwardNodeParams {
        exchange,
        fwd_offset,
        node_visit_offset,
        risk_measure,
        succ,
        outcomes,
        ..
    } = *params;
    let x_hat = exchange.state_at(succ.my_rank, m);
    let scenario = fwd_offset + m;
    let s = succ.successor;

    debug_assert!(
        ws.backward_accum.per_opening_stats.len() >= succ.probabilities.len(),
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
        prepare_child_lp(opening_solver, ws, ctx, succ, &child);
        let source = succ.bracket_source(m, &child);
        pin_child_lp(ws, ctx, training_ctx, &child, x_hat, s, source);

        // Resolved per child, never once from child 0: each child's External
        // column is its own.
        let mut buf = std::mem::take(&mut ws.backward_accum.external_noise_buf);
        let noise = resolve_opening_noise(
            training_ctx,
            s,
            &child.openings,
            child.successor_node_id,
            &mut buf,
        )?;
        for solve_pos in 0..child.outcome_range.len() {
            let (omega, raw_noise) = noise.solved_opening(training_ctx, s, &child, solve_pos);
            // First-solved opening of THIS child: owns the per-(m, child node)
            // basis load/capture and the fresh (non-warm-carry) solve.
            let is_first = solve_pos == 0;
            let captures_here = is_first && captures_basis;
            let request = OpeningRequest {
                stored_basis: if captures_here {
                    resolve_backward_basis(basis_slice, m, child.successor_node)
                } else {
                    None
                },
                raw_noise,
                scenario,
                trial_pos: 0,
                omega,
                continue_carry: !is_first,
            };
            solve_backward_opening(
                opening_solver,
                ws,
                ctx,
                training_ctx,
                succ,
                &child,
                &request,
            )?;
            if captures_here {
                save_basis_at_omega_zero(ws, &child, basis_slice, m);
            }
        }
        ws.backward_accum.external_noise_buf = buf;
    }

    // Aggregate into `agg_coefficients`, then copy into this trial point's arena
    // slot so the bytes outlive the parallel closure without a per-cut allocation.
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

/// Drains each worker's staged cuts, in worker order, into `staged_cuts_buf`, orders them
/// by trial state, and commits one cut per trial point to the FCF, returning the count.
/// `trial_state_idx` is globally unique across workers, so the committed order, and thus
/// the cut set, is independent of worker count and completion order. `staged_cuts_buf` is
/// the caller-owned scratch (`BackwardPassState::staged_cuts_buf`), cleared and reused,
/// never reallocated on the hot path.
///
/// The caller runs it only after [`super::take_first_error`] returned `Ok` over the same
/// workspaces, so a worker's error commits no cut.
pub(crate) fn by_scenario_finish<S: SolverInterface>(
    workspaces: &mut [SolverWorkspace<S>],
    trial_points: &[usize],
    cut_n_state: usize,
    fcf: &mut FutureCostFunction,
    node_id: NodeId,
    pool: usize,
    iteration: u64,
    staged_cuts_buf: &mut Vec<(usize, StagedCut)>,
) -> usize {
    staged_cuts_buf.clear();
    for (w, ws) in workspaces.iter_mut().enumerate() {
        staged_cuts_buf.extend(
            ws.backward_accum
                .staged_cuts_buf
                .drain(..)
                .map(|cut| (w, cut)),
        );
    }
    // `trial_state_idx` is the SOLE sort key: globally unique across workers
    // (disjoint contiguous partitions), so the merge order is identical regardless
    // of worker index.
    staged_cuts_buf.sort_unstable_by_key(|(_, cut)| cut.trial_state_idx);
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
    staged_cuts_buf.len()
}
