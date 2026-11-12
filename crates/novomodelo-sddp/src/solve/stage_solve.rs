//! LP-solve entry points shared by the hot-path drivers (`forward.rs`,
//! `backward.rs`, `simulation/pipeline.rs`): [`run_stage_solve`] (the default,
//! slot-identity warm-start via `reconstruct_basis`) and
//! [`run_stage_solve_terminal_static`] (terminal-forward-only, a 1:1 basis
//! apply that skips slot-identity reconciliation because the terminal
//! template's shape never changes after priming).

use novomodelo_core::temporal::StageLagTransition;
use novomodelo_solver::{SolutionView, SolverError, SolverInterface};
use novomodelo_stochastic::par::{
    DownstreamLagAccum, LagMajor, PrimaryLagAccum, advance_lag_chain,
};

use crate::{
    basis_reconstruct::{ReconstructionTarget, enforce_basic_count_invariant, reconstruct_basis},
    context::StageContext,
    cut::pool::CutPool,
    error::SddpError,
    lp::builder::StateBox,
    lp::indexer::{REGION_ORDER, StateRegion, StateSpace},
    setup::{NodeId, StageIdx},
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Read-only inputs for one LP solve at stage `t`, scenario `m`.
///
/// Intentionally has no `Default`: callers must supply every field, so the
/// compiler rejects any construction that omits one — preventing the "new flag
/// wired three places, missed the fourth" bug class.
pub struct StageInputs<'a> {
    /// Per-stage LP layout and noise scaling parameters.
    pub stage_context: &'a StageContext<'a>,
    /// Active cut pool for stage `t`.
    pub pool: &'a CutPool,
    /// Stored basis from the previous iteration, if any.
    pub stored_basis: Option<&'a CapturedBasis>,
    /// Stage index `t`.
    pub stage_index: StageIdx,
    /// Scenario index `m`.
    pub scenario_index: usize,
    /// Training iteration (1-based) on the forward and backward passes; `None`
    /// on simulation and on the test-support probes. A training solve supplies
    /// it so `SddpError::Infeasible { iteration }` is populated without the
    /// caller re-wrapping the error.
    pub iteration: Option<u64>,
    /// Declared id of the node being solved (`NodeGraph::node_ids[node]`); a
    /// `stored_basis` whose `node_id` differs is treated as cold.
    pub node_id: NodeId,
}

/// Execute one LP solve at stage `t` for scenario `m`, owning basis
/// reconstruction, invariant enforcement, and the solve call.
///
/// Load-and-bounds setup is the caller's responsibility (via `StageInputs`).
/// Returns the solver's solution view on success.
///
/// # Errors
///
/// `SolverError::Infeasible` bubbles as `SddpError::Infeasible { ... }` with
/// stage/scenario context filled in. Other solver errors propagate as
/// `SddpError::Solver`.
pub fn run_stage_solve<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    run_stage_solve_over(ws, inputs, inputs.pool.active_cuts())
}

/// [`run_stage_solve`] for an LP whose cut rows are `cut_rows`, in row order, in
/// place of the pool's active cuts.
///
/// # Errors
///
/// Same as [`run_stage_solve`].
pub(crate) fn run_stage_solve_over<'ws, 'c, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    inputs: &StageInputs<'_>,
    cut_rows: impl Iterator<Item = (usize, f64, &'c [f64])>,
) -> Result<SolutionView<'ws>, SddpError> {
    // `pool.populated()` only grows, so this resize is a no-op after the
    // first few iterations.
    if ws.scratch.recon_slot_lookup.len() < inputs.pool.populated() {
        ws.scratch
            .recon_slot_lookup
            .resize(inputs.pool.populated(), None);
    }

    let stored_basis = filtered_stored_basis(ws, inputs);

    let solved = if let Some(captured) = stored_basis {
        let template = inputs.stage_context.template(inputs.stage_index);
        // `base_row_count` is the non-frozen template row count so cut rows are
        // matched by slot identity, not positional copy from the stored basis.
        let target = ReconstructionTarget::from_template(template);

        let _ = reconstruct_basis(
            captured,
            target,
            cut_rows,
            &mut ws.scratch_basis,
            &mut ws.scratch.recon_slot_lookup,
        );
        // `num_row_for_invariant` is the reconstructed length — `target.base_row_count`
        // plus one row per cut row, which is this LP's row count on every frozen
        // path (forward and simulation load `frozen[t]`; backward loads `frozen[t]`
        // then adds the delta cuts the pool has meanwhile gained). `frozen.num_rows`
        // is the wrong-but-plausible alternative: it omits those delta cuts.
        //
        // `base_row_for_invariant = 0` is safe: the loop demotes only currently-
        // BASIC rows, and equality rows are never BASIC by LP duality.
        let num_row_for_invariant = ws.scratch_basis.row_status.len();
        let base_row_for_invariant = 0;

        enforce_basic_count_invariant(
            &mut ws.scratch_basis,
            num_row_for_invariant,
            base_row_for_invariant,
        )?;

        ws.solver.record_reconstruction_stats();

        ws.solver.solve(Some(&ws.scratch_basis))
    } else {
        ws.solver.solve(None)
    };

    finish_solve(solved, inputs)
}

/// Terminal-forward-only 1:1 basis apply, bypassing [`reconstruct_basis`]'s
/// slot-identity reconciliation.
///
/// The terminal leaf's active-cut set is fixed once primed (a leaf never
/// gains or loses a cut), so the terminal template's shape never changes
/// across iterations: a node-matching stored basis maps onto the current LP
/// by plain position. A stored basis that fails the node-tag filter, or whose
/// shape does not match the current template (should be impossible given
/// that invariance), is treated as cold — never applied — rather than
/// risking `solve`'s own hard assertion on a column-count mismatch.
///
/// # Errors
///
/// Same as [`run_stage_solve`]: `SolverError::Infeasible` bubbles as
/// `SddpError::Infeasible { .. }`; other solver errors propagate as
/// `SddpError::Solver`.
pub fn run_stage_solve_terminal_static<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    let stored_basis = filtered_stored_basis(ws, inputs)
        .filter(|captured| terminal_basis_shape_matches(captured, inputs));

    let solved = if let Some(captured) = stored_basis {
        ws.scratch_basis.col_status.clear();
        ws.scratch_basis
            .col_status
            .extend_from_slice(&captured.basis.col_status);
        ws.scratch_basis.row_status.clear();
        ws.scratch_basis
            .row_status
            .extend_from_slice(&captured.basis.row_status);

        ws.solver.solve(Some(&ws.scratch_basis))
    } else {
        ws.solver.solve(None)
    };

    finish_solve(solved, inputs)
}

/// The stored basis to offer the next solve, if any.
///
/// A solver that will run cold gets none, sparing a basis it would discard.
/// A stored basis whose `node_id` differs from the node being solved is
/// treated as cold: a resampled path may visit a different node than the
/// one that captured it, and CLP accepts a shape-mismatched basis silently,
/// so the check must live here, never delegated to a solver-specific path.
fn filtered_stored_basis<'a, S: SolverInterface>(
    ws: &SolverWorkspace<S>,
    inputs: &StageInputs<'a>,
) -> Option<&'a CapturedBasis> {
    if !ws.solver.warm_starts() {
        return None;
    }
    inputs
        .stored_basis
        .filter(|captured| captured.node_id == inputs.node_id)
}

/// Whether `captured`'s column/row status vectors already match the terminal
/// template's current shape (base rows plus active cuts) — the only shape a
/// genuine same-node terminal capture can have.
fn terminal_basis_shape_matches(captured: &CapturedBasis, inputs: &StageInputs<'_>) -> bool {
    let template = inputs.stage_context.template(inputs.stage_index);
    captured.basis.col_status.len() == template.num_cols
        && captured.basis.row_status.len() == template.num_rows + inputs.pool.active_count()
}

/// Map a raw solve `Result` to the `SddpError` the caller returns, filling in
/// stage/scenario/iteration context on the `Infeasible` path.
fn finish_solve<'ws>(
    solved: Result<SolutionView<'ws>, SolverError>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    solved.map_err(|e| {
        map_solver_error(
            e,
            inputs.stage_index.0,
            inputs.scenario_index,
            inputs.iteration,
        )
    })
}

/// Map a [`SolverError`] to a [`SddpError`] with stage/scenario context.
///
/// `Infeasible` is wrapped with the full context fields so the caller can
/// report which stage/scenario triggered the infeasibility.  All other
/// variants are wrapped by the `From<SolverError>` impl (i.e. `SddpError::Solver`).
fn map_solver_error(
    e: SolverError,
    stage: usize,
    scenario: usize,
    iteration: Option<u64>,
) -> SddpError {
    match e {
        SolverError::Infeasible => SddpError::Infeasible {
            stage,
            iteration: iteration.unwrap_or(0),
            scenario,
        },
        other => SddpError::Solver(other),
    }
}

/// Unscale a primal vector into `out`: `x_original[i] = col_scale[i] *
/// x_scaled[i]` (or the raw values when `col_scale` is empty). `out` retains its
/// pre-warmed capacity (`clear` + `extend`).
pub(crate) fn fill_unscaled(out: &mut Vec<f64>, scaled: &[f64], col_scale: &[f64]) {
    debug_assert!(
        col_scale.is_empty() || col_scale.len() == scaled.len(),
        "col_scale length {} != primal length {}",
        col_scale.len(),
        scaled.len()
    );
    out.clear();
    if col_scale.is_empty() {
        out.extend_from_slice(scaled);
    } else {
        out.extend(scaled.iter().zip(col_scale.iter()).map(|(&xp, &d)| d * xp));
    }
}

/// Unscale a dual vector into `out`: `dual_original[i] = row_scale[i] *
/// dual_scaled[i]` for structural rows; rows beyond the template length (cut
/// rows) have implicit `row_scale = 1.0`. Empty `row_scale` means no scaling.
pub(crate) fn fill_unscaled_dual(out: &mut Vec<f64>, scaled: &[f64], row_scale: &[f64]) {
    // Unlike `fill_unscaled`, `scaled` may be *longer* than `row_scale`: cut
    // rows extend past the template-row count and take an implicit scale of
    // 1.0. The invariant is therefore `<=`, not `==`.
    debug_assert!(
        row_scale.len() <= scaled.len(),
        "row_scale length {} exceeds dual length {}",
        row_scale.len(),
        scaled.len()
    );
    out.clear();
    if row_scale.is_empty() {
        out.extend_from_slice(scaled);
    } else {
        out.extend(scaled.iter().enumerate().map(|(i, &d)| {
            let scale = row_scale.get(i).copied().unwrap_or(1.0);
            d * scale
        }));
    }
}

/// Canonicalize the outgoing state vector: copy each region's outgoing block
/// of the LP primal, run the lag accumulation/shift, verify the
/// bucket/commitment-hold copy-gap invariant, then clamp every dimension onto
/// its admissible box.
/// The clamp is the identity for an in-box value — a solve whose state stays
/// in box is byte-for-byte unaffected by it. Every consumer of the outgoing
/// state (the trajectory, the backward pass's `x_hat`, the cut intercept, the
/// simulation output) reads this same canonical vector; there is no second
/// canonicalization point at LP pin time.
///
/// `realized_inflow` is ẑ, the stage's realized inflow, which feeds lag 0.
#[allow(clippy::float_cmp)]
pub(crate) fn assemble_outgoing_state(
    current_state: &mut Vec<f64>,
    unscaled_primal: &[f64],
    incoming_lags: &[f64],
    realized_inflow: &[f64],
    layout: &StateSpace,
    state_box: &StateBox,
    stage_lag: StageLagTransition,
    lag: &mut PrimaryLagAccum<'_>,
    ds: &mut DownstreamLagAccum<'_>,
) {
    current_state.clear();
    for region in REGION_ORDER {
        if region.has_lp_columns() {
            current_state.extend_from_slice(&unscaled_primal[layout.outgoing_block_range(region)]);
        } else {
            debug_assert_eq!(incoming_lags.len(), layout.state_dim_lag_range().len());
            current_state.extend_from_slice(incoming_lags);
        }
    }
    debug_assert_eq!(realized_inflow.len(), layout.hydro_count);
    advance_lag_chain(
        LagMajor {
            entity_count: layout.hydro_count,
            max_order: layout.max_par_order,
        },
        &mut current_state[layout.state_dim_lag_range().start..],
        incoming_lags,
        realized_inflow,
        &stage_lag,
        lag,
        ds,
    );
    debug_assert_bucket_copy_gap_intact(current_state.as_slice(), unscaled_primal, layout);

    for (j, value) in current_state.iter_mut().take(layout.n_state).enumerate() {
        let clamped = value.clamp(state_box.lower[j], state_box.upper[j]);
        if clamped == *value {
            continue;
        }
        *value = clamped;
        debug_assert!(
            !layout.state_dim_lag_range().contains(&j),
            "inflow lags are unbounded and never clamp"
        );
    }
}

/// Advances the worker's `current_state` through [`assemble_outgoing_state`] from the
/// primal in `scratch.unscaled_primal` and the realized inflow in
/// `scratch.realized_inflow_buf`, leaving the incoming lag slice of the state it advanced
/// from in `scratch.incoming_lags_buf` for the caller.
pub(crate) fn advance_worker_state<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    layout: &StateSpace,
    state_box: &StateBox,
    stage_lag: StageLagTransition,
    downstream_par_order: usize,
) {
    let scratch = &mut ws.scratch;
    scratch.incoming_lags_buf.clear();
    scratch
        .incoming_lags_buf
        .extend_from_slice(&ws.current_state[layout.state_dim_lag_range()]);
    assemble_outgoing_state(
        &mut ws.current_state,
        &scratch.unscaled_primal,
        &scratch.incoming_lags_buf,
        &scratch.realized_inflow_buf,
        layout,
        state_box,
        stage_lag,
        &mut PrimaryLagAccum {
            accumulator: &mut scratch.lag_accumulator,
            weight_accum: &mut scratch.lag_weight_accum,
        },
        &mut DownstreamLagAccum {
            accumulator: &mut scratch.downstream_accumulator,
            weight_accum: &mut scratch.downstream_weight_accum,
            completed_lags: &mut scratch.downstream_completed_lags,
            n_completed: &mut scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
}

/// Assert the bucket and commitment-hold state rode the state-assembly plain
/// copy untouched.
///
/// The forward/simulation assembly plain-copies each region's outgoing block
/// into the advanced state, then overwrites only the lag region in place;
/// the bucket and commitment-hold regions sit in the shift-gap that overwrite
/// never reaches. Call after the lag overwrite, before the clamp — the clamp
/// legitimately moves bucket/commitment-hold values off the primal, which
/// would trip this assert if it ran after.
// Rationale: this checks a verbatim copy invariant (`extend_from_slice`), not
// a numerical result, so exact equality is the correct comparison — a
// tolerance would mask the one bug this guards against (an overwrite landing
// on the bucket/commitment-hold range).
#[allow(clippy::float_cmp)]
fn debug_assert_bucket_copy_gap_intact(
    assembled_state: &[f64],
    unscaled_primal: &[f64],
    layout: &StateSpace,
) {
    debug_assert!(
        [StateRegion::Buckets, StateRegion::CommitmentHold]
            .into_iter()
            .all(|region| {
                layout
                    .state_dim_range(region)
                    .zip(layout.outgoing_block_range(region))
                    .all(|(j, column)| assembled_state[j] == unscaled_primal[column])
            }),
        "bucket/commitment-hold state must equal the LP primal's outgoing \
         block: the lag overwrite must never touch the bucket/commitment-hold \
         shift-gap"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use novomodelo_core::temporal::StageLagTransition;
    use novomodelo_solver::BasisStatus::{Basic as B, Lower as L};
    use novomodelo_solver::{
        ActiveSolver, Basis, RowBatch, SolutionView, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };
    use novomodelo_stochastic::par::{
        DownstreamLagAccum, LagMajor, PrimaryLagAccum, advance_lag_chain,
    };

    use super::{
        REGION_ORDER, StageInputs, StateRegion, assemble_outgoing_state, run_stage_solve,
        run_stage_solve_over, run_stage_solve_terminal_static,
    };
    use crate::{
        SddpError,
        cut::pool::CutPool,
        lp::builder::{PatchBuffer, StageGeometry, StateBox},
        lp::indexer::{HydroSys, StateSpace},
        setup::{NodeId, StageIdx},
        test_support::{
            StageContextFixture, TrainingContextFixture, bits, constant_lead_resolution,
            equipment_free_geometry, permissive_state_boxes, state_layout,
            state_layout_with_transit_buckets,
        },
        workspace::{CapturedBasis, SolverWorkspace, WorkspaceSizing},
    };

    // -----------------------------------------------------------------------
    // Shared fixtures
    // -----------------------------------------------------------------------

    /// Minimal LP: 3 columns, 2 rows (same fixture used in novomodelo-solver tests).
    ///
    ///   min  0*x0 + 1*x1 + 50*x2
    ///   s.t. x0            = 6   (pins x0)
    ///        2*x0 + x2     = 14  (power balance)
    ///   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
    fn make_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 2, 2, 3],
            row_indices: vec![0_i32, 1, 1],
            values: vec![1.0, 2.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![10.0, f64::INFINITY, 8.0],
            objective: vec![0.0, 1.0, 50.0],
            row_lower: vec![6.0, 14.0],
            row_upper: vec![6.0, 14.0],
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Infeasible LP: 1 column with x0 >= 5 AND x0 <= 2.
    fn make_infeasible_template() -> StageTemplate {
        StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![5.0],
            col_upper: vec![2.0], // infeasible: lower > upper
            objective: vec![1.0],
            row_lower: vec![],
            row_upper: vec![],
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a fresh `SolverWorkspace<ActiveSolver>` with an LP loaded.
    fn make_workspace(template: &StageTemplate) -> SolverWorkspace<ActiveSolver> {
        make_workspace_with(template, ActiveSolver::new().expect("ActiveSolver::new()"))
    }

    fn make_workspace_with<S: SolverInterface>(
        template: &StageTemplate,
        mut solver: S,
    ) -> SolverWorkspace<S> {
        solver.load_model(template);
        let training_fixture = TrainingContextFixture::new(state_layout(0, 0));
        let training_ctx = training_fixture.training_ctx();
        let stage_ctx_fixture = StageContextFixture::new(&[], &[], &[]);
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(training_ctx.state, &[], &[], &[]),
            &training_ctx,
            &stage_ctx_fixture.ctx(),
            WorkspaceSizing::default(),
        )
    }

    /// Build a minimal `StageContext` fixture wrapping a single template.
    fn make_context<'a>(
        templates: &'a [StageTemplate],
        geometry_per_stage: &'a [StageGeometry],
    ) -> StageContextFixture<'a> {
        StageContextFixture::new(templates, &[], geometry_per_stage).without_inflow_entries(1)
    }

    /// Build an empty `CutPool` (no active cuts, `populated_count = 0`).
    fn make_empty_pool() -> CutPool {
        CutPool::new(16, 1, 1, 0)
    }

    // -----------------------------------------------------------------------
    // Test 1: cold start — `stored_basis: None` solves from scratch
    // -----------------------------------------------------------------------

    #[test]
    fn run_stage_solve_cold_start_returns_view() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: None,
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: Some(1),
            node_id: NodeId(0),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        let view = result.expect("cold start should succeed");
        // Optimal objective for the fixture LP: x1 = 14 - 2*6 = 2, cost = 2.
        assert!(view.objective.is_finite());
    }

    // -----------------------------------------------------------------------
    // Test 2: warm start on the frozen path — successful solve returns outcome
    // -----------------------------------------------------------------------

    /// Verifies that a warm-start with a valid `CapturedBasis` on the frozen
    /// path completes successfully.  On the frozen path all cut rows are
    /// structural, so `reconstruct_basis` uses an empty iterator and
    /// `enforce_basic_count_invariant` is a no-op (no excess BASIC rows).
    ///
    /// Basis accounting: `CapturedBasis` is built with `base_row_count=2` and
    /// `cut_row_slots=[]` (no stored cut rows).  `reconstruct_basis` with
    /// `target.base_row_count=2` copies the 2 stored row statuses.
    /// With 2 BASIC col statuses + 0 BASIC row statuses, `total_basic=2 == num_row=2`.
    #[test]
    fn run_stage_solve_warm_start_frozen_path_succeeds() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // Build a CapturedBasis with base_row_count=2, 0 cut rows.
        // col_status: first 2 cols BASIC (needed to match num_row=2),
        // row_status: 2 LOWER entries (rows are at bound in optimal solution).
        // total_basic = col_basic(2) + row_basic(0) = 2 == num_row(2). Valid.
        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(5), // node_id — matches inputs.node_id below (the match/warm case)
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B); // x0 BASIC
        captured.basis.col_status.push(B); // x1 BASIC
        captured.basis.col_status.push(L); // x2 LOWER
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L); // row 0 at bound
        captured.basis.row_status.push(L); // row 1 at bound

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "warm start on frozen path should succeed: {result:?}"
        );
        // No cut rows → no demotions. Row statuses are all LOWER (non-basic).
        let lower_count = ws
            .scratch_basis
            .row_status
            .iter()
            .filter(|&&s| s == L)
            .count();
        assert_eq!(
            lower_count, 2,
            "both rows should be LOWER after reconstruction"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: infeasible LP propagates as SddpError::Infeasible
    // -----------------------------------------------------------------------

    #[test]
    fn run_stage_solve_propagates_infeasible() {
        let template = make_infeasible_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = CutPool::new(16, 0, 1, 0);
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: None,
            stage_index: StageIdx(0),
            scenario_index: 7,
            iteration: Some(42),
            node_id: NodeId(0),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        match result {
            Err(SddpError::Infeasible {
                stage,
                scenario,
                iteration,
            }) => {
                assert_eq!(stage, 0, "stage must match inputs.stage_index");
                assert_eq!(scenario, 7, "scenario must match inputs.scenario_index");
                assert_eq!(iteration, 42, "iteration must match inputs.iteration");
            }
            other => panic!("expected SddpError::Infeasible, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: a basic-count deficit is rejected before the solver sees it
    // -----------------------------------------------------------------------

    /// A stored basis with zero basic variables (all LOWER) against `num_row = 2`
    /// is a deficit, which only a shape mismatch can produce. `run_stage_solve`
    /// must reject it with `SddpError::BasisShapeMismatch` naming the counters —
    /// never pass it to the solver, whose own `isBasisConsistent` rejection
    /// (`SolverError::BasisInconsistent` under `HiGHS`; accepted outright under CLP)
    /// cannot say which LP the basis came from.
    #[test]
    fn basis_deficit_rejected_before_solver_sees_it() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // All col_status and row_status LOWER (the zero-fill default from
        // Basis::new) → col_basic = 0, row_basic = 0, total_basic = 0 against
        // num_row = 2.
        let all_lower = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(9), // node_id — matches inputs.node_id below (the match/warm case)
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&all_lower),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(9),
        };

        match run_stage_solve(&mut ws, &inputs) {
            Err(SddpError::BasisShapeMismatch {
                num_row,
                total_basic,
                col_basic,
                row_basic,
            }) => {
                assert_eq!(num_row, template.num_rows);
                assert_eq!(total_basic, 0);
                assert_eq!(col_basic, 0);
                assert_eq!(row_basic, 0);
            }
            other => panic!("expected Err(SddpError::BasisShapeMismatch {{ .. }}), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Node-tag mismatch: cross-node reuse is treated as cold
    // -----------------------------------------------------------------------

    /// The same deficit-shaped `CapturedBasis` `basis_deficit_rejected_before_solver_sees_it`
    /// uses to prove the warm path surfaces `BasisShapeMismatch` — but tagged with a
    /// node id that does NOT match `inputs.node_id`. The mismatch must drop to the
    /// cold path before `reconstruct_basis`/`enforce_basic_count_invariant` ever see
    /// the deficit-shaped basis, so the solve succeeds instead of erroring.
    #[test]
    fn run_stage_solve_cross_node_stored_basis_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // Deficit shape identical to basis_deficit_rejected_before_solver_sees_it —
        // would surface as BasisShapeMismatch if the warm path were taken.
        let stored_at_node_a = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(1), // node A
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&stored_at_node_a),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(2), // node B — mismatches the stored basis's node A
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a node-id mismatch must drop to the cold path, never attempting the \
             deficit-shaped stored basis: {result:?}"
        );
    }

    #[test]
    fn stage_solve_applies_the_basis_over_the_given_cut_rows() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];
        ws.solver.add_rows(&RowBatch {
            num_rows: 2,
            row_starts: vec![0, 1, 2],
            col_indices: vec![1, 1],
            values: vec![1.0, 1.0],
            row_lower: vec![-10.0, -20.0],
            row_upper: vec![f64::INFINITY; 2],
        });

        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows + 1,
            template.num_rows,
            1,
            NodeId(5),
        );
        captured.basis.col_status.fill(B);
        captured.basis.row_status.fill(L);
        captured.cut_row_slots.push(0);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };
        let given = [(0, 0.0, &[][..]), (1, 0.0, &[][..])];

        let result = run_stage_solve_over(&mut ws, &inputs, given.into_iter());

        assert!(result.is_ok(), "{result:?}");
        let rows = &ws.scratch_basis.row_status;
        assert_eq!(rows.len(), template.num_rows + given.len());
        assert_eq!(rows[template.num_rows], L, "slot 0 keeps its stored status");
        assert_eq!(rows[template.num_rows + 1], B, "slot 1 starts basic");
    }

    // -----------------------------------------------------------------------
    // Terminal-static entry: 1:1 basis apply, no reconstruct_basis
    // -----------------------------------------------------------------------

    /// The terminal-static entry applies a node-matching stored basis
    /// verbatim — `ws.scratch_basis` ends up byte-identical to the stored
    /// `col_status`/`row_status` — and never calls `reconstruct_basis`:
    /// `basis_reconstructions` (incremented only by `record_reconstruction_stats`,
    /// which `run_stage_solve`'s reconstruct branch calls and
    /// `run_stage_solve_terminal_static` does not) stays at zero.
    #[test]
    fn run_stage_solve_terminal_static_applies_basis_1to1_without_reconstruct_basis() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(5),
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(L);
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L);
        captured.basis.row_status.push(L);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        // `basis_reconstructions` is HiGHS-only telemetry (CLP's
        // `SolverInterface` never overrides `record_reconstruction_stats`), so
        // it is meaningful only under the `highs` backend; the backend-agnostic
        // proof that no reconstruction happened is the byte-for-byte copy
        // assertion below, which runs under both backends.
        #[cfg(feature = "highs")]
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            0,
            "sanity: a fresh workspace starts with no reconstructions"
        );

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "terminal-static warm start should succeed: {result:?}"
        );
        assert_eq!(
            ws.scratch_basis.col_status, captured.basis.col_status,
            "terminal-static must copy col_status verbatim, never reconstruct it"
        );
        assert_eq!(
            ws.scratch_basis.row_status, captured.basis.row_status,
            "terminal-static must copy row_status verbatim, never reconstruct it"
        );
        #[cfg(feature = "highs")]
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            0,
            "run_stage_solve_terminal_static must never invoke reconstruct_basis"
        );
    }

    /// Interior (non-terminal) warm starts still go through `reconstruct_basis`:
    /// `basis_reconstructions` increments by exactly one, unlike the
    /// terminal-static entry above. HiGHS-only: CLP's `SolverInterface` never
    /// overrides `record_reconstruction_stats`, so the counter would stay `0`
    /// there regardless of whether `reconstruct_basis` ran.
    #[cfg(feature = "highs")]
    #[test]
    fn run_stage_solve_interior_warm_start_invokes_reconstruct_basis() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(5),
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(L);
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L);
        captured.basis.row_status.push(L);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "interior warm start should succeed: {result:?}"
        );
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            1,
            "run_stage_solve's slot-identity path must invoke reconstruct_basis"
        );
    }

    /// Wraps [`ActiveSolver`] with a fixed `warm_starts()` answer, recording each
    /// `solve`'s basis argument and each `record_reconstruction_stats` call.
    struct WarmStartProbe {
        inner: ActiveSolver,
        warm: bool,
        offered: Vec<bool>,
        reconstructions_recorded: usize,
    }

    impl WarmStartProbe {
        fn new(warm: bool) -> Self {
            Self {
                inner: ActiveSolver::new().expect("ActiveSolver::new()"),
                warm,
                offered: Vec::new(),
                reconstructions_recorded: 0,
            }
        }
    }

    impl SolverInterface for WarmStartProbe {
        type Profile = <ActiveSolver as SolverInterface>::Profile;
        fn apply_profile(&mut self, profile: &Self::Profile) {
            self.inner.apply_profile(profile);
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.inner.load_model(template);
        }
        fn add_rows(&mut self, rows: &RowBatch) {
            self.inner.add_rows(rows);
        }
        fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
            self.inner.set_row_bounds(indices, lower, upper);
        }
        fn set_row_lower_bounds_range(&mut self, first_row: usize, lower: &[f64]) {
            self.inner.set_row_lower_bounds_range(first_row, lower);
        }
        fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
            self.inner.set_col_bounds(indices, lower, upper);
        }
        fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            self.offered.push(basis.is_some());
            self.inner.solve(basis)
        }
        fn warm_starts(&self) -> bool {
            self.warm
        }
        fn get_basis(&mut self, out: &mut Basis) {
            self.inner.get_basis(out);
        }
        fn statistics(&self) -> SolverStatistics {
            self.inner.statistics()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            self.inner.statistics_into(out);
        }
        fn name(&self) -> &'static str {
            "WarmStartProbe"
        }
        fn solver_name_version(&self) -> String {
            self.inner.solver_name_version()
        }
        fn record_reconstruction_stats(&mut self) {
            self.reconstructions_recorded += 1;
            self.inner.record_reconstruction_stats();
        }
    }

    /// A node-matching, well-formed stored basis for `make_template()`.
    fn valid_stored_basis(template: &StageTemplate, node_id: NodeId) -> CapturedBasis {
        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            node_id,
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.extend_from_slice(&[B, B, L]);
        captured.basis.row_status.clear();
        captured.basis.row_status.extend_from_slice(&[L, L]);
        captured
    }

    /// Solves once through the interior entry and once through the
    /// terminal-static entry with a valid same-node stored basis, returning
    /// the probe's recorded `(offered, reconstructions_recorded)`.
    fn probe_both_entries(warm: bool) -> (Vec<bool>, usize, SolverWorkspace<WarmStartProbe>) {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace_with(&template, WarmStartProbe::new(warm));
        ws.scratch.recon_slot_lookup = vec![None; 16];
        let captured = valid_stored_basis(&template, NodeId(5));
        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        run_stage_solve(&mut ws, &inputs).expect("interior solve must succeed");
        ws.scratch_basis.col_status.clear();
        ws.scratch_basis.row_status.clear();
        run_stage_solve_terminal_static(&mut ws, &inputs)
            .expect("terminal-static solve must succeed");

        let probe = ws.solver.inner();
        (probe.offered.clone(), probe.reconstructions_recorded, ws)
    }

    #[test]
    fn run_stage_solve_cold_profile_skips_reconstruction() {
        let (offered, reconstructions, ws) = probe_both_entries(false);
        assert_eq!(
            offered,
            vec![false, false],
            "a cold solver must receive solve(None) on both entries"
        );
        assert_eq!(
            reconstructions, 0,
            "a cold solver must never record a reconstruction"
        );
        assert!(
            ws.scratch_basis.col_status.is_empty() && ws.scratch_basis.row_status.is_empty(),
            "terminal-static must not copy the stored basis for a cold solver"
        );
    }

    #[test]
    fn run_stage_solve_warm_profile_still_offers_the_stored_basis() {
        let (offered, reconstructions, _) = probe_both_entries(true);
        assert_eq!(offered, vec![true, true]);
        assert_eq!(
            reconstructions, 1,
            "only the interior entry reconstructs; terminal-static copies 1:1"
        );
    }

    /// The terminal-static entry preserves the node-tag filter: a stored basis
    /// tagged at a different node must drop to cold before
    /// `terminal_basis_shape_matches` ever inspects the shape — mirrors
    /// `run_stage_solve_cross_node_stored_basis_is_treated_as_cold` above.
    #[test]
    fn run_stage_solve_terminal_static_cross_node_stored_basis_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        // Deficit shape (all LOWER) — would be nonsense to apply even if the
        // node tag matched; the point here is that the node mismatch alone
        // must drop to cold before this shape is ever inspected.
        let stored_at_node_a = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(1),
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&stored_at_node_a),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(2), // node B — mismatches the stored basis's node A
        };

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a node-id mismatch must drop to the cold path on the terminal-static \
             entry too: {result:?}"
        );
    }

    /// A shape mismatch on the terminal-static path — genuinely impossible
    /// given the terminal template's structural invariance, but defended
    /// against rather than risking `solve`'s own hard assertion on a
    /// column-count mismatch — must also drop to cold, never applying the
    /// wrongly-shaped basis.
    #[test]
    fn run_stage_solve_terminal_static_shape_mismatch_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        // col_status.len() == template.num_cols + 1: node id matches, so only
        // the shape check can stop this from being applied.
        let wrong_shape = CapturedBasis::new(
            template.num_cols + 1,
            template.num_rows,
            template.num_rows,
            0,
            NodeId(5),
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&wrong_shape),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a col/row shape mismatch must drop to the cold path, never applying \
             the wrongly-shaped basis: {result:?}"
        );
    }

    /// `SolverError::BasisInconsistent` must route to `SddpError::Solver`, not
    /// `SddpError::Infeasible`: it is a basis defect, not a proof of
    /// infeasibility, and conflating them would abort a recoverable run.
    #[test]
    fn basis_inconsistent_maps_to_sddp_solver_error() {
        let mapped = super::map_solver_error(
            SolverError::BasisInconsistent {
                num_row: 4385,
                total_basic: 4383,
                col_basic: 4000,
                row_basic: 383,
            },
            0,
            3,
            Some(5),
        );
        assert!(
            matches!(
                mapped,
                SddpError::Solver(SolverError::BasisInconsistent { .. })
            ),
            "expected SddpError::Solver(BasisInconsistent), got {mapped:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: bucket copy-gap assertion helper
    // -----------------------------------------------------------------------

    #[test]
    fn debug_assert_bucket_copy_gap_intact_passes_when_bucket_matches_primal() {
        let layout = state_layout_with_transit_buckets(
            0,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let primal = vec![7.0, 11.0];
        let assembled = primal.clone();
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    #[should_panic(expected = "bucket/commitment-hold state must equal the LP primal's outgoing")]
    fn debug_assert_bucket_copy_gap_intact_panics_when_bucket_diverges() {
        let layout = state_layout_with_transit_buckets(
            0,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let primal = vec![7.0, 11.0];
        let mut assembled = primal.clone();
        assembled[1] = 999.0; // simulate an accidental overwrite of the bucket block
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    fn debug_assert_bucket_copy_gap_intact_passes_when_commitment_hold_matches_primal() {
        let layout = state_layout_with_transit_buckets(0, 0, vec![], vec![1, 1]);
        let primal = vec![3.0, 5.0];
        let assembled = primal.clone();
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    #[should_panic(expected = "bucket/commitment-hold state must equal the LP primal's outgoing")]
    fn debug_assert_bucket_copy_gap_intact_panics_when_commitment_hold_diverges() {
        let layout = state_layout_with_transit_buckets(0, 0, vec![], vec![1, 1]);
        let primal = vec![3.0, 5.0];
        let mut assembled = primal.clone();
        assembled[0] = 999.0; // simulate an accidental overwrite of the commitment-hold slot
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    // -----------------------------------------------------------------------
    // Test 8: assemble_outgoing_state — the canonicalization seam
    // -----------------------------------------------------------------------

    /// `N=1, L=1, B=1, A=1, K_max=1` layout: `n_state = 4`
    /// (storage, lag0, `bucket_out`, `commit_out`), mirroring the
    /// `transit_bucket_copy_gap` fixture shape.
    fn seam_layout() -> StateSpace {
        state_layout_with_transit_buckets(1, 1, vec![(HydroSys::new(0), 0)], vec![1])
    }

    fn identity_stage_lag() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    /// `unscaled_primal` for `seam_layout`'s 7-column LP: `[storage,
    /// bucket_out, commit_out, storage_in, bucket_in, commit_in, theta]`.
    fn seam_primal(storage: f64, bucket: f64, commit: f64) -> Vec<f64> {
        vec![storage, bucket, commit, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn assemble_outgoing_state_in_box_is_byte_identical() {
        let layout = seam_layout();
        let unscaled_primal = seam_primal(10.0, 30.0, 20.0);
        let incoming_lags = vec![2.0_f64];
        let stage_lag = identity_stage_lag();
        let state_box = StateBox {
            lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            upper: vec![100.0, f64::INFINITY, 100.0, 50.0],
        };

        // Reference: the outgoing-block copy + lag advance, with no clamp step.
        let mut expected = Vec::new();
        for region in REGION_ORDER {
            if region.has_lp_columns() {
                expected.extend_from_slice(&unscaled_primal[layout.outgoing_block_range(region)]);
            } else {
                expected.extend_from_slice(&incoming_lags);
            }
        }
        let mut exp_lag_accumulator = vec![0.0_f64; 1];
        let mut exp_lag_weight_accum = vec![0.0_f64; 1];
        let mut exp_ds_accum: Vec<f64> = vec![];
        let mut exp_ds_weight = 0.0_f64;
        let mut exp_ds_completed: Vec<f64> = vec![];
        let mut exp_ds_n_completed = 0_usize;
        advance_lag_chain(
            LagMajor {
                entity_count: layout.hydro_count,
                max_order: layout.max_par_order,
            },
            &mut expected[layout.state_dim_lag_range().start..],
            &incoming_lags,
            &[5.0],
            &stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut exp_lag_accumulator,
                weight_accum: &mut exp_lag_weight_accum,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut exp_ds_accum,
                weight_accum: &mut exp_ds_weight,
                completed_lags: &mut exp_ds_completed,
                n_completed: &mut exp_ds_n_completed,
                par_order: 0,
            },
        );

        let mut current_state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_outgoing_state(
            &mut current_state,
            &unscaled_primal,
            &incoming_lags,
            &[5.0],
            &layout,
            &state_box,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_accum,
                weight_accum: &mut ds_weight,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n_completed,
                par_order: 0,
            },
        );

        assert_eq!(
            current_state, expected,
            "an in-box clamp must be byte-identical to the pre-clamp assembled state"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn assemble_outgoing_state_clamps_out_of_box() {
        let layout = seam_layout();
        let commit_col = layout.commit_out.start;
        let commit_dim = layout.state_dim_commitment_hold_range().start;
        let mut unscaled_primal = seam_primal(10.0, 30.0, 20.0);
        unscaled_primal[commit_col] = -0.5; // 0.5 below the box lower of 0.0
        let incoming_lags = vec![2.0_f64];
        let stage_lag = identity_stage_lag();
        let state_box = StateBox {
            lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            upper: vec![100.0, f64::INFINITY, 100.0, 50.0],
        };

        let mut current_state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_outgoing_state(
            &mut current_state,
            &unscaled_primal,
            &incoming_lags,
            &[5.0],
            &layout,
            &state_box,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_accum,
                weight_accum: &mut ds_weight,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n_completed,
                par_order: 0,
            },
        );

        assert_eq!(
            current_state[commit_dim], 0.0,
            "must equal the clamped bound"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn assemble_outgoing_state_copies_each_region_from_its_outgoing_block() {
        // Displace the bucket and commitment outgoing blocks past the prefix so a
        // `[..n_state]` copy cannot satisfy the assertions.
        let mut layout = seam_layout();
        layout.transit_buckets_out = 9..10;
        layout.commit_out = 10..11;
        let unscaled_primal: Vec<f64> = (0..11_u32).map(|i| 100.0 + f64::from(i)).collect();
        let state_box = StateBox {
            lower: vec![f64::NEG_INFINITY; layout.n_state],
            upper: vec![f64::INFINITY; layout.n_state],
        };

        let mut current_state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_outgoing_state(
            &mut current_state,
            &unscaled_primal,
            &[2.0_f64],
            &[5.0],
            &layout,
            &state_box,
            identity_stage_lag(),
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_accum,
                weight_accum: &mut ds_weight,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n_completed,
                par_order: 0,
            },
        );

        for region in [
            StateRegion::Storage,
            StateRegion::Buckets,
            StateRegion::CommitmentHold,
        ] {
            assert_eq!(
                current_state[layout.state_dim_range(region)],
                unscaled_primal[layout.outgoing_block_range(region)],
                "{region:?} must be copied from its outgoing block"
            );
        }
    }

    const COMPACT_INCOMING_LAGS: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    const COMPACT_REALIZED_INFLOW: [f64; 2] = [5.5, 6.25];

    struct CompactAssembly {
        layout: StateSpace,
        unscaled_primal: Vec<f64>,
        state: Vec<f64>,
        lag_accumulator: Vec<f64>,
    }

    /// Assembles a two-hydro compact layout of `max_order` lags from `incoming_lags` and
    /// `COMPACT_REALIZED_INFLOW`.
    fn assemble_compact(
        max_order: usize,
        incoming_lags: &[f64],
        stage_lag: StageLagTransition,
    ) -> CompactAssembly {
        let layout = StateSpace::new(
            2,
            max_order,
            Vec::new(),
            Vec::new(),
            constant_lead_resolution(&[], 2),
            &[max_order, max_order],
        );
        let mut state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; max_order];
        let mut lag_weight_accum = vec![0.0_f64; max_order];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_stage(
            &mut state,
            incoming_lags,
            &COMPACT_REALIZED_INFLOW,
            &layout,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        CompactAssembly {
            unscaled_primal: distinct_primal(&layout),
            layout,
            state,
            lag_accumulator,
        }
    }

    #[test]
    fn assemble_outgoing_state_seeds_the_compact_lag_slice_from_the_incoming_lags() {
        let non_finalizing = StageLagTransition {
            accumulate_weight: 0.25,
            finalize_period: false,
            ..identity_stage_lag()
        };
        let out = assemble_compact(2, &COMPACT_INCOMING_LAGS, non_finalizing);

        assert_eq!(
            bits(&out.state[out.layout.state_dim_lag_range()]),
            bits(&COMPACT_INCOMING_LAGS),
            "a non-finalizing stage keeps the incoming lags bit for bit"
        );
        for (h, realized) in COMPACT_REALIZED_INFLOW.iter().enumerate() {
            assert_eq!(
                out.lag_accumulator[h].to_bits(),
                (0.0 + realized * 0.25).to_bits(),
                "hydro {h}: the accumulator gains the realized inflow times the weight"
            );
        }
    }

    #[test]
    fn assemble_outgoing_state_shifts_the_compact_lag_slice_onto_the_realized_inflow() {
        let out = assemble_compact(2, &COMPACT_INCOMING_LAGS, identity_stage_lag());
        let lags = &out.state[out.layout.state_dim_lag_range()];

        assert_eq!(
            bits(&lags[..2]),
            bits(&COMPACT_REALIZED_INFLOW),
            "lag 0 is the realized inflow"
        );
        assert_eq!(
            bits(&lags[2..]),
            bits(&COMPACT_INCOMING_LAGS[..2]),
            "lag 1 is the incoming lag 0"
        );
    }

    #[test]
    fn assemble_outgoing_state_shifts_a_compact_lag_slice_deeper_than_the_hydro_count() {
        let incoming_lags = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let out = assemble_compact(3, &incoming_lags, identity_stage_lag());

        let expected: Vec<f64> = COMPACT_REALIZED_INFLOW
            .iter()
            .chain(&incoming_lags[..4])
            .copied()
            .collect();
        assert_eq!(
            bits(&out.state[out.layout.state_dim_lag_range()]),
            bits(&expected),
            "lag 0 is the realized inflow and lag l the incoming lag l - 1"
        );
    }

    #[test]
    fn assemble_outgoing_state_copies_compact_storage_from_its_outgoing_block() {
        let out = assemble_compact(2, &COMPACT_INCOMING_LAGS, identity_stage_lag());

        assert_eq!(
            bits(&out.state[out.layout.state_dim_range(StateRegion::Storage)]),
            bits(&out.unscaled_primal[out.layout.outgoing_block_range(StateRegion::Storage)]),
        );
    }

    // Convenience helper: build a no-op DownstreamLagAccum for tests that
    // exercise only primary accumulation (uniform-resolution path).
    fn noop_ds<'a>(
        accumulator: &'a mut Vec<f64>,
        weight_accum: &'a mut f64,
        completed_lags: &'a mut Vec<f64>,
        n_completed: &'a mut usize,
    ) -> DownstreamLagAccum<'a> {
        DownstreamLagAccum {
            accumulator: accumulator.as_mut_slice(),
            weight_accum,
            completed_lags: completed_lags.as_mut_slice(),
            n_completed,
            par_order: 0,
        }
    }

    fn rebuild_stage_lag() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        }
    }

    /// Four weekly stages each contributing weight=0.25, finalize only on stage 3.
    ///
    /// After processing all four stages the lag[0] must equal the weighted
    /// average: (500 + 480 + 520 + 510) / 4 = 502.5.
    #[test]
    fn test_accumulate_four_weeks_then_finalize() {
        let layout = state_layout(1, 1);
        let mut state = Vec::new();
        let incoming_lags = vec![0.0]; // lag-major: lag0 for hydro 0
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];

        let z_inflows = [500.0_f64, 480.0, 520.0, 510.0];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;

        for (week, &z) in z_inflows.iter().enumerate() {
            let finalize = week == 3;
            let stage_lag = StageLagTransition {
                accumulate_weight: 0.25,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };
            assemble_stage(
                &mut state,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_accumulator,
                    weight_accum: &mut lag_weight_accum,
                },
                &mut noop_ds(
                    &mut ds_accum,
                    &mut ds_weight,
                    &mut ds_completed,
                    &mut ds_n_completed,
                ),
            );
        }

        let lag_start = layout.state_dim_lag_range().start;
        let expected = (500.0 + 480.0 + 520.0 + 510.0) / 4.0;
        assert!(
            (state[lag_start] - expected).abs() < 1e-12,
            "lag[0] must equal weighted average {expected}, got {}",
            state[lag_start]
        );
        // Accumulator reset after finalization.
        assert_eq!(lag_accumulator[0], 0.0);
        assert_eq!(lag_weight_accum[0], 0.0);
    }

    /// Spillover seeds the next lag period with the raw realized inflow times `spillover_weight`.
    #[test]
    fn test_accumulate_spillover_seeds_next_period() {
        let layout = state_layout(1, 1);
        let mut state = Vec::new();
        let incoming_lags = vec![0.0];
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];

        let stage_lag = StageLagTransition {
            accumulate_weight: 0.968, // 1.0 - 0.032 = days in period / days in month
            spillover_weight: 0.032,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        };
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_stage(
            &mut state,
            &incoming_lags,
            &[200.0],
            &layout,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );

        // After finalization, accumulator seeded with the raw realized inflow * spillover_weight.
        let expected_seed = 200.0 * 0.032;
        assert!(
            (lag_accumulator[0] - expected_seed).abs() < 1e-12,
            "accumulator must be seeded with realized inflow * spillover_weight = {expected_seed}, got {}",
            lag_accumulator[0]
        );
        assert!(
            (lag_weight_accum[0] - 0.032).abs() < 1e-12,
            "lag_weight_accum must equal spillover_weight = 0.032, got {}",
            lag_weight_accum[0]
        );
    }

    /// `max_par_order == 0`: the assembled state has no lag region and no accumulator is touched.
    #[test]
    fn test_accumulate_noop_for_par0() {
        let layout = state_layout(2, 0);
        let mut state = Vec::new();
        let incoming_lags: Vec<f64> = vec![];
        let mut lag_accumulator: Vec<f64> = vec![]; // empty — should never be accessed
        let mut lag_weight_accum: Vec<f64> = vec![]; // empty — should never be accessed
        let stage_lag = identity_stage_lag();
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_stage(
            &mut state,
            &incoming_lags,
            &vec![0.0; layout.hydro_count],
            &layout,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut noop_ds(
                &mut ds_accum,
                &mut ds_weight,
                &mut ds_completed,
                &mut ds_n_completed,
            ),
        );
        assert_eq!(state.len(), layout.n_state, "no lag region for PAR(0)");
        assert_eq!(
            lag_weight_accum,
            Vec::<f64>::new(),
            "weight must be unchanged for PAR(0)"
        );
    }

    // ── downstream accumulation tests ────────────────────────────────────────

    /// Build a `StageLagTransition` for a standard monthly stage that also
    /// accumulates into the downstream (quarterly) ring buffer.
    ///
    /// weight = 1/3 per month (3 months per quarter, no spillover).
    fn monthly_with_downstream(
        finalize_primary: bool,
        downstream_finalize: bool,
    ) -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: finalize_primary,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize,
            rebuild_from_downstream: false,
        }
    }

    /// Drives one stage through `assemble_outgoing_state` with the given realized inflow per
    /// hydro; the primal holds no lag columns and distinct values in every other slot.
    fn assemble_stage(
        state: &mut Vec<f64>,
        incoming_lags: &[f64],
        realized_inflow: &[f64],
        layout: &StateSpace,
        stage_lag: StageLagTransition,
        lag: &mut PrimaryLagAccum<'_>,
        ds: &mut DownstreamLagAccum<'_>,
    ) {
        let state_box = permissive_state_boxes(layout.n_state, 1).remove(0);
        assemble_outgoing_state(
            state,
            &distinct_primal(layout),
            incoming_lags,
            realized_inflow,
            layout,
            &state_box,
            stage_lag,
            lag,
            ds,
        );
    }

    fn distinct_primal(layout: &StateSpace) -> Vec<f64> {
        (100_u32..).map(f64::from).take(layout.theta + 1).collect()
    }

    /// Test 1: PAR(1) downstream accumulation with a 3-stage quarterly window.
    ///
    /// 3 monthly stages (each weight=1/3, no primary finalize, no primary
    /// spillover, `downstream_finalize` on last month) populate the downstream
    /// ring buffer.  After all 3 stages, `downstream_completed_lags[0]` must
    /// equal `(90.0 + 100.0 + 110.0) / 3.0 = 100.0`.  Then calling with
    /// `rebuild_from_downstream = true` on the first quarterly stage overwrites
    /// `state[lag_start]` with `100.0`.
    #[test]
    fn test_downstream_par1_accumulation_and_rebuild() {
        // N=1 hydro, L=1 lag (primary monthly PAR(1) order).
        let layout = state_layout(1, 1);
        let lag_start = layout.state_dim_lag_range().start;

        let mut state = Vec::new();
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        // downstream: par_order=1, ring buf capacity n_h * 1 = 1.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        let z_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in z_vals.iter().enumerate() {
            let ds_finalize = i == 2; // last month of the quarter
            let stage_lag = monthly_with_downstream(false, ds_finalize);
            assemble_stage(
                &mut state,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamLagAccum {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        // After 3 monthly stages the ring buffer should hold the quarterly average.
        let expected_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_avg).abs() < 1e-12,
            "ring buf slot 0 should be {expected_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1, "n_completed must be 1 after one quarter");

        // Now simulate the transition stage (first quarterly stage).
        // rebuild_from_downstream=true; primary accumulation is quarterly.
        let rebuild_lag = rebuild_stage_lag();
        assemble_stage(
            &mut state,
            &incoming_lags,
            &[999.0],
            &layout,
            rebuild_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] must be rebuilt to the quarterly average.
        assert!(
            (state[lag_start] - expected_avg).abs() < 1e-12,
            "state[lag_start] must be rebuilt to {expected_avg}, got {}",
            state[lag_start]
        );
        // Downstream state must be fully reset.
        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(ds_w, 0.0, "downstream weight must reset after rebuild");
    }

    /// Test 2: PAR(2) downstream accumulation with two consecutive quarters.
    ///
    /// 6 monthly stages (Q3: stages 0-2, Q4: stages 3-5) with `downstream_par_order=2`.
    /// After Q3, `completed_lags[slot=0] == avg(60,70,80) == 70.0`, `n_completed==1`.
    /// After Q4, `completed_lags[slot=1] == avg(90,100,110) == 100.0`, `n_completed==2`.
    /// At rebuild: `state[lag_start] == 100.0` (newest Q4), `state[lag_start+1] == 70.0` (Q3).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_downstream_par2_two_quarters() {
        let layout = state_layout(1, 2);
        let lag_start = layout.state_dim_lag_range().start;

        let mut state = Vec::new();
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag1 h0
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        // par_order=2: ring buf capacity n_h * 2 = 2.
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // Q3: z_inflows 60, 70, 80 — no primary finalize, downstream finalize on month 3.
        let q3_vals = [60.0_f64, 70.0, 80.0];
        for (i, &z) in q3_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            assemble_stage(
                &mut state,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamLagAccum {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q3_avg = (60.0 + 70.0 + 80.0) / 3.0;
        assert!(
            (ds_completed[0] - q3_avg).abs() < 1e-12,
            "slot 0 should be Q3 avg {q3_avg}, got {}",
            ds_completed[0]
        );
        assert_eq!(ds_n, 1);

        // Q4: z_inflows 90, 100, 110 — downstream finalize on month 3.
        let q4_vals = [90.0_f64, 100.0, 110.0];
        for (i, &z) in q4_vals.iter().enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            assemble_stage(
                &mut state,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamLagAccum {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 2,
                },
            );
        }
        let q4_avg = (90.0 + 100.0 + 110.0) / 3.0;
        assert!(
            (ds_completed[1] - q4_avg).abs() < 1e-12,
            "slot 1 should be Q4 avg {q4_avg}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 2);

        // Rebuild stage: lag[0] <- newest (Q4), lag[1] <- second-newest (Q3).
        let rebuild_lag = rebuild_stage_lag();
        assemble_stage(
            &mut state,
            &incoming_lags,
            &[999.0],
            &layout,
            rebuild_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 2,
            },
        );

        // lag[0] = newest = Q4 avg; lag[1] = Q3 avg.
        assert!(
            (state[lag_start] - q4_avg).abs() < 1e-12,
            "lag[0] should be newest Q4 avg {q4_avg}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - q3_avg).abs() < 1e-12,
            "lag[1] should be Q3 avg {q3_avg}, got {}",
            state[lag_start + 1]
        );
    }

    /// Test 3: Uniform monthly study — empty downstream buffers, zero overhead.
    ///
    /// Runs with empty downstream buffers (`par_order = 0`) against the same stages with a no-op downstream.
    /// The function must produce exactly the same result,
    /// with no downstream fields accessed.
    #[test]
    fn test_no_downstream_for_uniform_monthly() {
        let layout = state_layout(1, 1);
        let mut state_ds = Vec::new();
        let mut state_ref = Vec::new();
        let incoming_lags = vec![0.0];
        let z_inflows = [100.0_f64, 110.0, 120.0];

        let mut lag_acc_ref = vec![0.0_f64; 1];
        let mut lag_w_ref = vec![0.0_f64; 1];
        let mut lag_acc_ds = vec![0.0_f64; 1];
        let mut lag_w_ds = vec![0.0_f64; 1];

        for (i, &z) in z_inflows.iter().enumerate() {
            let finalize = i == 2;
            let stage_lag = StageLagTransition {
                accumulate_weight: 1.0 / 3.0,
                spillover_weight: 0.0,
                finalize_period: finalize,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };

            // Reference: empty downstream (noop path).
            let mut ds_accum_ref: Vec<f64> = vec![];
            let mut ds_weight_ref = 0.0_f64;
            let mut ds_completed_ref: Vec<f64> = vec![];
            let mut ds_n_completed_ref = 0_usize;
            assemble_stage(
                &mut state_ref,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc_ref,
                    weight_accum: &mut lag_w_ref,
                },
                &mut noop_ds(
                    &mut ds_accum_ref,
                    &mut ds_weight_ref,
                    &mut ds_completed_ref,
                    &mut ds_n_completed_ref,
                ),
            );

            // Test: inline empty downstream (par_order=0).
            let mut ds_accum_ds: Vec<f64> = vec![];
            let mut ds_weight_ds = 0.0_f64;
            let mut ds_completed_ds: Vec<f64> = vec![];
            let mut ds_n_completed_ds = 0_usize;
            assemble_stage(
                &mut state_ds,
                &incoming_lags,
                &[z],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc_ds,
                    weight_accum: &mut lag_w_ds,
                },
                &mut DownstreamLagAccum {
                    accumulator: &mut ds_accum_ds,
                    weight_accum: &mut ds_weight_ds,
                    completed_lags: &mut ds_completed_ds,
                    n_completed: &mut ds_n_completed_ds,
                    par_order: 0,
                },
            );
        }

        assert_eq!(
            state_ds, state_ref,
            "uniform monthly study must be identical with or without downstream buffers"
        );
    }

    /// Test 4: `rebuild_from_downstream` resets all downstream state.
    ///
    /// After rebuild, `n_completed == 0`, `completed_lags` all zero, and
    /// `downstream_weight_accum == 0.0`.
    #[test]
    fn test_rebuild_resets_downstream_state() {
        let layout = state_layout(1, 1);
        let mut state = Vec::new();
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_w = 0.5_f64; // non-zero before rebuild
        let mut ds_completed = vec![77.0_f64; 1]; // non-zero before rebuild
        let mut ds_n = 1_usize; // pretend one quarter was completed

        let rebuild_lag = rebuild_stage_lag();

        assemble_stage(
            &mut state,
            &incoming_lags,
            &[0.0],
            &layout,
            rebuild_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        assert_eq!(ds_n, 0, "n_completed must reset to 0 after rebuild");
        assert_eq!(
            ds_completed[0], 0.0,
            "completed_lags must be zeroed after rebuild"
        );
        assert_eq!(
            ds_w, 0.0,
            "downstream weight_accum must reset after rebuild"
        );
        assert_eq!(
            ds_acc[0], 0.0,
            "downstream accumulator must be zeroed after rebuild"
        );
    }

    /// Test 5: Downstream spillover seeds the next quarterly accumulation.
    ///
    /// A monthly stage with `downstream_spillover_weight = 0.1` and
    /// `downstream_finalize = true` should: (a) finalize the current quarter,
    /// (b) seed the next quarter's accumulator with the realized inflow * 0.1.
    #[test]
    fn test_downstream_spillover_seeds_next_quarter() {
        let layout = state_layout(1, 1);
        let mut state = Vec::new();
        let incoming_lags = vec![0.0];
        let mut lag_acc = vec![0.0_f64; 1];
        let mut lag_w = vec![0.0_f64; 1];
        let mut ds_acc = vec![0.0_f64; 1];
        let mut ds_completed = vec![0.0_f64; 1];
        let mut ds_n = 0_usize;

        // Single monthly stage that finalizes the quarter with spillover.
        // Pre-load the accumulator to simulate prior months already accumulated.
        ds_acc[0] = 200.0; // months 1+2 already accumulated
        let mut ds_w = 2.0 / 3.0_f64; // two months of weight 1/3 each

        let spillover_weight = 0.1;
        let stage_lag = StageLagTransition {
            accumulate_weight: 1.0 / 3.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: spillover_weight,
            downstream_finalize: true,
            rebuild_from_downstream: false,
        };

        let z = 120.0_f64;
        assemble_stage(
            &mut state,
            &incoming_lags,
            &[z],
            &layout,
            stage_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // Quarter should be finalized and the ring buffer filled.
        assert_eq!(ds_n, 1, "one quarter must be finalized");
        // Downstream accumulator must be seeded with z * spillover_weight.
        let expected_seed = z * spillover_weight;
        assert!(
            (ds_acc[0] - expected_seed).abs() < 1e-12,
            "accumulator should be seeded with {expected_seed}, got {}",
            ds_acc[0]
        );
        assert!(
            (ds_w - spillover_weight).abs() < 1e-12,
            "weight_accum should be {spillover_weight}, got {ds_w}"
        );
    }

    /// Test 6: Multi-hydro downstream — 2 hydros, PAR(1).
    ///
    /// Each hydro has its own realized inflow values.  After 3 monthly stages,
    /// `downstream_completed_lags[0]` (hydro 0) and `[1]` (hydro 1) must
    /// each equal the independently computed quarterly average for that hydro.
    #[test]
    fn test_downstream_multi_hydro() {
        // N=2 hydros, L=1 lag order.
        let layout = state_layout(2, 1);
        let lag_start = layout.state_dim_lag_range().start;

        let mut state = Vec::new();
        let incoming_lags = vec![0.0, 0.0]; // lag-major: lag0 h0, lag0 h1
        let mut lag_acc = vec![0.0_f64; 2];
        let mut lag_w = vec![0.0_f64; 2];
        // ring buf capacity: n_h * par_order = 2 * 1 = 2
        let mut ds_acc = vec![0.0_f64; 2];
        let mut ds_w = 0.0_f64;
        let mut ds_completed = vec![0.0_f64; 2];
        let mut ds_n = 0_usize;

        // 3 monthly stages: hydro 0 inflows [10, 20, 30], hydro 1 inflows [40, 50, 60].
        let h0_vals = [10.0_f64, 20.0, 30.0];
        let h1_vals = [40.0_f64, 50.0, 60.0];

        for (i, (&z0, &z1)) in h0_vals.iter().zip(h1_vals.iter()).enumerate() {
            let stage_lag = monthly_with_downstream(false, i == 2);
            assemble_stage(
                &mut state,
                &incoming_lags,
                &[z0, z1],
                &layout,
                stage_lag,
                &mut PrimaryLagAccum {
                    accumulator: &mut lag_acc,
                    weight_accum: &mut lag_w,
                },
                &mut DownstreamLagAccum {
                    accumulator: &mut ds_acc,
                    weight_accum: &mut ds_w,
                    completed_lags: &mut ds_completed,
                    n_completed: &mut ds_n,
                    par_order: 1,
                },
            );
        }

        let expected_h0 = (10.0 + 20.0 + 30.0) / 3.0;
        let expected_h1 = (40.0 + 50.0 + 60.0) / 3.0;
        assert!(
            (ds_completed[0] - expected_h0).abs() < 1e-12,
            "hydro 0 quarterly avg should be {expected_h0}, got {}",
            ds_completed[0]
        );
        assert!(
            (ds_completed[1] - expected_h1).abs() < 1e-12,
            "hydro 1 quarterly avg should be {expected_h1}, got {}",
            ds_completed[1]
        );
        assert_eq!(ds_n, 1);

        // Rebuild: both hydros rebuilt independently.
        let rebuild_lag = rebuild_stage_lag();
        assemble_stage(
            &mut state,
            &incoming_lags,
            &[999.0, 888.0],
            &layout,
            rebuild_lag,
            &mut PrimaryLagAccum {
                accumulator: &mut lag_acc,
                weight_accum: &mut lag_w,
            },
            &mut DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: 1,
            },
        );

        // lag[0] for hydro 0 and hydro 1 must be rebuilt independently.
        assert!(
            (state[lag_start] - expected_h0).abs() < 1e-12,
            "rebuilt lag[0] hydro 0 should be {expected_h0}, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - expected_h1).abs() < 1e-12,
            "rebuilt lag[0] hydro 1 should be {expected_h1}, got {}",
            state[lag_start + 1]
        );
    }
}
