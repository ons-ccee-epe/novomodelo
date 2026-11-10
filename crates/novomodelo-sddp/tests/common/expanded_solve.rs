//! One compact folded solve and the expanded LP of the same solve: the compact LP is prepared by
//! the production folded prep and solved once, and every per-solve bound it wrote is carried onto
//! the expanded template, so that mapping has this one owner.

use std::collections::{HashMap, HashSet};

use novomodelo_sddp::indexer::{CutLagPoint, CutSlot, CutStateProjection, HydroSys, StateDim};
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::test_support::bound_recording::{
    BoundRecordingSolver, BoundWrite, RecordedSolve,
};
use novomodelo_sddp::test_support::expander::{
    ExpandedLayout, ExpandedStudy, expanded_cut_rows_into, expanded_solve_patch,
};
use novomodelo_sddp::test_support::template_structure::column_entries;
use novomodelo_sddp::test_support::{
    FoldedProbeCut, FoldedProbeCuts, FoldedProbeOpening, cut_rows_for_slots_into, empty_row_batch,
    folded_opening_cut_for_probe,
};
use novomodelo_sddp::{CutPool, FutureCostFunction, StudySetup};
use novomodelo_solver::{
    ActiveSolver, Basis, BasisStatus, FreezeScratch, LpSolution, ProfiledSolver, RowBatch,
    SolverInterface, StageTemplate, freeze_rows_into_template,
};

use super::StubComm;
use super::oracle_tolerance::{UnscaledTolerance, assert_first_try, value_tolerance};
use super::power_guards::{Guard, PowerEvidence};
use super::row_families::{self, classify_expanded_rows};

/// The cut rows of one solve and the pools their projections come from.
#[derive(Clone)]
pub struct CutRowSet {
    /// The pool whose cuts are the rows; its projection renders them.
    pub resident_pool: usize,
    /// The pool whose projection the fold's cut is written in.
    pub extracted_pool: usize,
    /// `slots[k]` is the resident pool slot of cut row `k`, in both LPs.
    pub slots: Vec<usize>,
    /// The first `frozen` slots are frozen into the compact template; the rest are appended to it.
    pub frozen: usize,
}

impl CutRowSet {
    /// The active cuts of `resident_pool`, in slot order.
    pub fn active(fcf: &FutureCostFunction, resident_pool: usize, extracted_pool: usize) -> Self {
        Self {
            resident_pool,
            extracted_pool,
            slots: fcf
                .active_cuts(resident_pool)
                .map(|(slot, _, _)| slot)
                .collect(),
            frozen: 0,
        }
    }
}

/// The compact solve and the expanded LP assembled from it.
pub struct ExpandedSolve {
    /// The compact LP as solved.
    pub compact: RecordedSolve,
    /// The compact LP's optimal basis.
    pub compact_basis: Basis,
    /// The compact solution and the cut the fold extracts from it.
    pub compact_cut: FoldedProbeCut,
    /// The cut rows of both LPs.
    pub rows: CutRowSet,
    /// The scaled expanded template with every per-solve bound applied.
    pub template: StageTemplate,
    /// The expanded cut rows, to append to `template`.
    pub cut_rows: RowBatch,
    /// Cut `k`'s lag coefficients at offset `k * N * L`, lag-major: offset `l * N + h` within a cut.
    pub lag_coefficients: Vec<f64>,
}

fn scale_at(scale: &[f64], index: usize) -> f64 {
    if scale.is_empty() { 1.0 } else { scale[index] }
}

/// The `(hydro, lag)` each global state dimension holds, or `None` for a dimension that is no lag.
pub fn lag_dims(layout: &ExpandedLayout) -> Vec<Option<(usize, usize)>> {
    let state = layout.state();
    (0..state.n_state)
        .map(|d| {
            let column = layout.incoming_state_col(StateDim::new(d));
            (0..state.hydro_count).find_map(|h| {
                (0..state.max_par_order).find_map(|l| {
                    (layout.incoming_lag_col(HydroSys::new(h), l) == column).then_some((h, l))
                })
            })
        })
        .collect()
}

/// Every trial state stored in pool `stage - 1`, in slot order, as a whole state; `None` when that
/// pool's projection drops a state dimension.
pub fn stored_incoming_states(trained: &StudySetup, stage: usize) -> Option<Vec<Vec<f64>>> {
    assert!(
        stage >= 1,
        "stored_incoming_states: stage 0 has no producer pool"
    );
    let producer = stage - 1;
    let n_state = trained.stage_state().n_state;
    let training_ctx = trained.training_ctx();
    let projection = &training_ctx.cut_state_layouts[producer];
    if projection.n_slots() != n_state {
        return None;
    }
    let pool = &trained.fcf.pools[producer];
    assert!(
        pool.stores_trial_states(),
        "pool {producer} stores no trial state"
    );
    Some(
        (0..pool.populated())
            .filter_map(|slot| pool.trial_state(slot))
            .map(|row| {
                let mut whole = vec![0.0; n_state];
                for (j, &value) in row.iter().enumerate() {
                    whole[projection.global_state_index(CutSlot::new(j)).get()] = value;
                }
                whole
            })
            .collect(),
    )
}

/// The stages `s >= 1` with an active cut whose producer pool `s - 1` stores fewer dimensions than
/// the whole state, so [`stored_incoming_states`] has no whole incoming state for them. Reads the
/// pool's own dimension, not the projection [`stored_incoming_states`] tests.
pub fn stages_whose_producer_drops_a_state_dimension(trained: &StudySetup) -> Vec<usize> {
    let n_state = trained.stage_state().n_state;
    (1..trained.fcf.pools.len())
        .filter(|&stage| trained.fcf.active_cuts(stage).next().is_some())
        .filter(|&stage| trained.fcf.pools[stage - 1].state_dimension < n_state)
        .collect()
}

/// `(Gᵀπ)_h` for every hydro from the expanded LP's duals `dual`.
pub fn coupled_duals(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    dual: &[f64],
) -> Vec<f64> {
    let ctx = compact.stage_ctx();
    let coupling = &ctx.inflow_coupling[stage];
    let row_scale = &expanded.scaled[stage].row_scale;
    (0..coupling.n_hydros())
        .map(|h| {
            coupling
                .hydro_entries(HydroSys::new(h))
                .fold(-0.0_f64, |acc, (r, g)| {
                    let row = expanded.layout.row(r);
                    acc + g * (scale_at(row_scale, row) * dual[row])
                })
        })
        .collect()
}

/// `Σ |g·π|` over the coupled rows of every hydro, the absolute magnitudes of the terms
/// [`coupled_duals`] sums.
pub fn coupled_dual_magnitudes(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    dual: &[f64],
) -> Vec<f64> {
    let ctx = compact.stage_ctx();
    let coupling = &ctx.inflow_coupling[stage];
    let row_scale = &expanded.scaled[stage].row_scale;
    (0..coupling.n_hydros())
        .map(|h| {
            coupling
                .hydro_entries(HydroSys::new(h))
                .fold(0.0_f64, |acc, (r, g)| {
                    let row = expanded.layout.row(r);
                    acc + (g * (scale_at(row_scale, row) * dual[row])).abs()
                })
        })
        .collect()
}

/// Writes every compact column and untouched row bound of `writes` onto `template` at its expanded
/// index; a touched row keeps its own bounds, since the expanded LP carries that row's noise in the
/// z rows.
fn carry_compact_bounds(
    template: &mut StageTemplate,
    layout: &ExpandedLayout,
    writes: &[BoundWrite],
    touched: &[u32],
    stage: usize,
) {
    for write in writes {
        match write {
            BoundWrite::ColBounds {
                indices,
                lower,
                upper,
            } => {
                for ((&c, &lo), &up) in indices.iter().zip(lower).zip(upper) {
                    let col = layout.col(c);
                    template.col_lower[col] = lo;
                    template.col_upper[col] = up;
                }
            }
            BoundWrite::RowBounds {
                indices,
                lower,
                upper,
            } => {
                let mut coupled = HashSet::new();
                for ((&r, &lo), &up) in indices.iter().zip(lower).zip(upper) {
                    if touched
                        .binary_search(&u32::try_from(r).expect("row fits u32"))
                        .is_ok()
                    {
                        assert!(
                            coupled.insert(r),
                            "stage {stage}: coupled row {r} appears twice in one row-bounds write"
                        );
                    } else {
                        let row = layout.row(r);
                        template.row_lower[row] = lo;
                        template.row_upper[row] = up;
                    }
                }
            }
            BoundWrite::AddRows { .. } | BoundWrite::RowLowerRange { .. } => {}
        }
    }
}

/// Loads the compact template into `solver` with the first `rows.frozen` cut rows frozen into it
/// and the rest appended.
fn load_compact_cut_rows(
    solver: &mut ProfiledSolver<BoundRecordingSolver<ActiveSolver>>,
    compact: &StudySetup,
    compact_template: &StageTemplate,
    cut_pool: &CutPool,
    projection: &CutStateProjection,
    rows: &CutRowSet,
) {
    assert!(
        rows.frozen <= rows.slots.len(),
        "pool {}: {} frozen rows of {} cut rows",
        rows.resident_pool,
        rows.frozen,
        rows.slots.len()
    );
    let (frozen_slots, delta_slots) = rows.slots.split_at(rows.frozen);
    let batch = |slots: &[usize]| {
        let mut batch = empty_row_batch();
        cut_rows_for_slots_into(
            &mut batch,
            cut_pool,
            slots,
            compact.stage_state(),
            projection,
            &compact_template.col_scale,
        );
        batch
    };
    if frozen_slots.is_empty() {
        solver.load_model(compact_template);
    } else {
        let mut frozen_template = StageTemplate::empty();
        freeze_rows_into_template(
            compact_template,
            &batch(frozen_slots),
            &mut frozen_template,
            &mut FreezeScratch::new(),
        );
        solver.load_model(&frozen_template);
    }
    if !delta_slots.is_empty() {
        solver.add_rows(&batch(delta_slots));
    }
}

/// Solves the compact LP of `stage` over the cut rows of `rows` once, through the production folded
/// prep and extraction, and assembles the expanded LP of the same solve.
pub fn assemble_expanded_solve(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    fcf: &FutureCostFunction,
    rows: &CutRowSet,
    stage: usize,
    incoming_state: &[f64],
    raw_noise: &[f64],
) -> ExpandedSolve {
    let layout = &expanded.layout;
    let pool = rows.resident_pool;
    let slots = &rows.slots;
    assert!(!slots.is_empty(), "pool {pool} holds no cut row");
    let cut_pool = &fcf.pools[pool];

    let ctx = compact.stage_ctx();
    let training_ctx = compact.training_ctx();
    let projection = &training_ctx.cut_state_layouts[pool];
    let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];

    let mut workspaces = compact
        .create_workspace_pool(&StubComm, 1, || {
            ActiveSolver::new().map(BoundRecordingSolver::new)
        })
        .expect("the workspace pool must build");
    let ws = &mut workspaces.workspaces[0];
    load_compact_cut_rows(
        &mut ws.solver,
        compact,
        compact_template,
        cut_pool,
        projection,
        rows,
    );

    let mut compact_cut = FoldedProbeCut::new(&training_ctx);
    let before = ws.solver.statistics();
    let solved = folded_opening_cut_for_probe(
        ws,
        &ctx,
        &training_ctx,
        &FoldedProbeOpening {
            stage: StageIdx(stage),
            pinned_state: incoming_state,
            raw_noise,
            cuts: FoldedProbeCuts {
                pool: cut_pool,
                pool_id: pool,
                slots,
            },
            extracted_pool_id: rows.extracted_pool,
        },
        &mut compact_cut,
    );
    let after = ws.solver.statistics();
    assert!(
        solved.is_ok(),
        "stage {stage} pool {pool}: the compact solve failed: {:?}",
        solved.as_ref().err()
    );
    assert_first_try(
        &format!("stage {stage} pool {pool}: compact solve"),
        &before,
        &after,
    );
    let mut compact_basis = Basis::new(
        compact_template.num_cols,
        compact_template.num_rows + slots.len(),
    );
    ws.solver.get_basis(&mut compact_basis);

    let compact_solve = ws
        .solver
        .inner()
        .recorded()
        .last()
        .cloned()
        .expect("the compact solve must be recorded");
    let cut_ranges: Vec<(usize, usize)> = compact_solve
        .writes
        .iter()
        .filter_map(|write| match write {
            BoundWrite::RowLowerRange { first_row, lower } => Some((*first_row, lower.len())),
            _ => None,
        })
        .collect();
    assert_eq!(
        cut_ranges,
        [(compact_template.num_rows, slots.len())],
        "stage {stage} pool {pool}: the folded prep must push one cut-row range over the appended rows"
    );

    let mut template = expanded.scaled[stage].clone();
    carry_compact_bounds(
        &mut template,
        layout,
        &compact_solve.writes,
        ctx.inflow_coupling[stage].touched_rows(),
        stage,
    );

    let patch = expanded_solve_patch(
        compact,
        expanded,
        stage,
        &raw_noise[compact.inputs.stochastic.class_dimensions().hydro_range()],
        incoming_state,
    );
    for &(row, rhs) in &patch.z_rows {
        template.row_lower[row] = rhs;
        template.row_upper[row] = rhs;
    }
    for &(col, pin) in &patch.incoming_cols {
        template.col_lower[col] = pin;
        template.col_upper[col] = pin;
    }

    let mut cut_rows = empty_row_batch();
    expanded_cut_rows_into(
        &mut cut_rows,
        cut_pool,
        slots,
        projection,
        layout,
        &expanded.scaled[stage].col_scale,
    );
    assert_eq!(
        cut_rows.num_rows,
        slots.len(),
        "stage {stage} pool {pool}: expanded cut-row count"
    );

    let lag_dims = lag_dims(layout);
    let dims = layout.state();
    let width = dims.hydro_count * dims.max_par_order;
    let mut lag_coefficients = vec![0.0; slots.len() * width];
    for (k, &slot) in slots.iter().enumerate() {
        for (j, &coefficient) in cut_pool.coefficient_row(slot).iter().enumerate() {
            let d = projection.global_state_index(CutSlot::new(j)).get();
            if let Some((h, l)) = lag_dims[d] {
                lag_coefficients[k * width + l * dims.hydro_count + h] = coefficient;
            }
        }
    }

    ExpandedSolve {
        compact: compact_solve,
        compact_basis,
        compact_cut,
        rows: rows.clone(),
        template,
        cut_rows,
        lag_coefficients,
    }
}

/// Loads the expanded LP of `solve` on `solver`, cut rows included.
pub fn load_expanded<S: SolverInterface>(solver: &mut S, solve: &ExpandedSolve) {
    solver.load_model(&solve.template);
    solver.add_rows(&solve.cut_rows);
}

fn write_once(slots: &mut [Option<BasisStatus>], what: &str, index: usize, status: BasisStatus) {
    assert!(
        slots[index].replace(status).is_none(),
        "map_compact_basis: {what} {index} is written twice"
    );
}

fn written(what: &str, slots: Vec<Option<BasisStatus>>) -> Vec<BasisStatus> {
    assert_eq!(
        slots.iter().position(Option::is_none),
        None,
        "map_compact_basis: a {what} is never written"
    );
    slots.into_iter().flatten().collect()
}

/// Maps the optimal basis of a compact LP onto the expanded LP of the same solve: every kept
/// column, kept row and cut row keeps its compact status, each z column is basic because its z row
/// determines it, and each z row and incoming lag column is nonbasic at its bound, so the basic
/// count still equals the row count. Panics unless the shapes agree, every expanded entry is
/// written exactly once and the basic count equals `expanded_rows + n_cut_rows`.
#[must_use]
fn map_compact_basis(
    compact: &Basis,
    layout: &ExpandedLayout,
    expanded_cols: usize,
    expanded_rows: usize,
    n_cut_rows: usize,
) -> Basis {
    let (n, lags) = (layout.state().hydro_count, layout.state().max_par_order);
    assert!(
        compact.row_status.len() >= n_cut_rows,
        "map_compact_basis: {} compact rows hold fewer than the {n_cut_rows} cut rows",
        compact.row_status.len()
    );
    let compact_rows = compact.row_status.len() - n_cut_rows;
    assert_eq!(
        compact.col_status.len() + n * (lags + 1),
        expanded_cols,
        "map_compact_basis: {} compact columns plus {n} hydros x {} columns vs {expanded_cols} expanded columns",
        compact.col_status.len(),
        lags + 1
    );
    assert_eq!(
        compact_rows + n,
        expanded_rows,
        "map_compact_basis: {compact_rows} compact rows plus {n} z rows vs {expanded_rows} expanded rows"
    );

    let mut cols = vec![None; expanded_cols];
    let mut rows = vec![None; expanded_rows + n_cut_rows];
    for (c, &status) in compact.col_status.iter().enumerate() {
        write_once(&mut cols, "column", layout.col(c), status);
    }
    for h in (0..n).map(HydroSys::new) {
        write_once(&mut cols, "column", layout.z_col(h), BasisStatus::Basic);
        for l in 0..lags {
            write_once(
                &mut cols,
                "column",
                layout.incoming_lag_col(h, l),
                BasisStatus::Lower,
            );
        }
    }
    for (r, &status) in compact.row_status[..compact_rows].iter().enumerate() {
        write_once(&mut rows, "row", layout.row(r), status);
    }
    for h in (0..n).map(HydroSys::new) {
        write_once(&mut rows, "row", layout.z_row(h), BasisStatus::Lower);
    }
    for (k, &status) in compact.row_status[compact_rows..].iter().enumerate() {
        write_once(&mut rows, "row", expanded_rows + k, status);
    }

    let basis = Basis {
        col_status: written("column", cols),
        row_status: written("row", rows),
    };
    let basic_count = basis
        .col_status
        .iter()
        .chain(&basis.row_status)
        .filter(|&&status| status == BasisStatus::Basic)
        .count();
    assert_eq!(
        basic_count,
        expanded_rows + n_cut_rows,
        "map_compact_basis: basic count over {expanded_cols} columns and {} rows",
        expanded_rows + n_cut_rows
    );
    basis
}

/// Solves the expanded LP of `solve` from the compact optimal basis mapped onto it, on a fresh
/// solver that must accept the offered basis, succeed on the first attempt and take no simplex
/// iteration; any other outcome fails under `label`, and nothing falls back to a cold solve.
pub fn solve_from_mapped_basis(
    label: &str,
    expanded: &ExpandedStudy,
    solve: &ExpandedSolve,
) -> LpSolution {
    let basis = map_compact_basis(
        &solve.compact_basis,
        &expanded.layout,
        solve.template.num_cols,
        solve.template.num_rows,
        solve.cut_rows.num_rows,
    );
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    assert!(
        solver.warm_starts(),
        "{label}: the solver profile ignores an offered basis"
    );
    load_expanded(&mut solver, solve);

    let before = solver.statistics();
    let outcome = solver.solve(Some(&basis)).map(|view| view.to_owned());
    assert!(
        outcome.is_ok(),
        "{label}: the warm solve failed: {:?}",
        outcome.as_ref().err()
    );
    let after = solver.statistics();

    assert_first_try(label, &before, &after);
    assert_eq!(
        after.basis_offered - before.basis_offered,
        1,
        "{label}: the solver did not count the offered basis"
    );
    assert_eq!(
        after.basis_consistency_failures - before.basis_consistency_failures,
        0,
        "{label}: the solver rejected the mapped basis as inconsistent"
    );
    let iterations = after.total_iterations - before.total_iterations;
    assert_eq!(
        iterations, 0,
        "{label}: the warm solve took {iterations} simplex iterations"
    );
    outcome.expect("checked above")
}

/// Records what an optimal solution `dual` of the expanded LP of `solve` shows about the fixture.
pub fn record_expanded_evidence(
    evidence: &mut PowerEvidence,
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    dual: &[f64],
) {
    let layout = &expanded.layout;
    let dims = layout.state();
    let (n, lags) = (dims.hydro_count, dims.max_par_order);
    let width = n * lags;
    let scaled = &expanded.scaled[stage];

    for k in 0..solve.rows.slots.len() {
        evidence.record_cut(
            dual[solve.template.num_rows + k],
            &solve.lag_coefficients[k * width..(k + 1) * width],
        );
    }

    let z_scales: Vec<f64> = (0..n)
        .map(|h| scale_at(&scaled.col_scale, layout.z_col(HydroSys::new(h))))
        .collect();
    let lag_scales: Vec<f64> = (0..lags)
        .flat_map(|l| (0..n).map(move |h| (h, l)))
        .map(|(h, l)| {
            scale_at(
                &scaled.col_scale,
                layout.incoming_lag_col(HydroSys::new(h), l),
            )
        })
        .collect();
    evidence.record_expanded_scales(&z_scales, &lag_scales);

    let ctx = compact.stage_ctx();
    let compact_row_scale = &ctx.templates[stage].row_scale;
    for &r in ctx.inflow_coupling[stage].touched_rows() {
        let r = usize::try_from(r).expect("row fits usize");
        let row = layout.row(r);
        let expanded_scale = scale_at(&scaled.row_scale, row);
        evidence.record_coupled_row(
            expanded_scale * dual[row],
            scale_at(compact_row_scale, r),
            expanded_scale,
        );
    }

    for (h, gt_pi) in coupled_duals(compact, expanded, stage, dual)
        .into_iter()
        .enumerate()
    {
        evidence.record_coupling_dual(h, gt_pi);
    }
}

/// What [`check_lifted_compact_solution`] examined, for the caller's power assertions.
#[derive(Clone, Copy, Default)]
pub struct LiftedSolutionReport {
    /// Rows of the expanded LP checked, cut rows included.
    pub rows_checked: usize,
    /// The cut rows among them.
    pub cut_rows_checked: usize,
    /// Rows within tolerance of a bound whose warm dual is beyond its dual tolerance.
    pub binding_rows_with_dual: usize,
    /// Rows farther than tolerance from both bounds, whose warm dual is within its dual tolerance of 0.
    pub slack_rows: usize,
    /// Rows judged against the compact solve's own residual on their compact counterpart.
    pub compact_referenced_rows: usize,
    /// The largest violation the compact solve left on a row of the compact LP, unscaled.
    pub largest_compact_violation: f64,
    /// The largest slack at the nearer bound of a row the compact solve treats as binding, unscaled.
    pub largest_compact_slack: f64,
    /// The largest `n` of the rounding bounds of the rows checked.
    pub largest_order: usize,
}

impl LiftedSolutionReport {
    /// Adds `other`'s counts and keeps the larger violation, slack and order.
    pub fn absorb(&mut self, other: Self) {
        self.rows_checked += other.rows_checked;
        self.cut_rows_checked += other.cut_rows_checked;
        self.binding_rows_with_dual += other.binding_rows_with_dual;
        self.slack_rows += other.slack_rows;
        self.compact_referenced_rows += other.compact_referenced_rows;
        self.largest_compact_violation = self
            .largest_compact_violation
            .max(other.largest_compact_violation);
        self.largest_compact_slack = self.largest_compact_slack.max(other.largest_compact_slack);
        self.largest_order = self.largest_order.max(other.largest_order);
    }
}

#[derive(Clone, Copy, Default)]
struct RowSum {
    activity: f64,
    magnitude: f64,
    nonzeros: usize,
}

impl RowSum {
    fn add(&mut self, coefficient: f64, x: f64) {
        if coefficient != 0.0 {
            let term = coefficient * x;
            self.activity += term;
            self.magnitude += term.abs();
            self.nonzeros += 1;
        }
    }
}

fn csr_offset(index: i32) -> usize {
    usize::try_from(index).expect("CSR offsets and indices are >= 0")
}

fn rounding_factor(n: usize) -> f64 {
    let n_u = f64::from(u32::try_from(n).expect("n fits u32")) * (f64::EPSILON / 2.0);
    n_u / (1.0 - n_u)
}

fn lifted_primal(
    label: &str,
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    incoming_state: &[f64],
) -> Vec<f64> {
    let layout = &expanded.layout;
    let dims = layout.state();
    let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];
    let cut = &solve.compact_cut;
    assert_eq!(
        cut.primal.len(),
        compact_template.num_cols,
        "{label} stage {stage}: compact primal length vs compact column count"
    );
    assert_eq!(
        cut.realized_inflow.len(),
        dims.hydro_count,
        "{label} stage {stage}: realized inflow length vs hydro count"
    );
    assert_eq!(
        incoming_state.len(),
        dims.n_state,
        "{label} stage {stage}: incoming state length vs state dimension count"
    );

    let mut x = vec![0.0; solve.template.num_cols];
    for (c, &value) in cut.primal.iter().enumerate() {
        x[layout.col(c)] = value * scale_at(&compact_template.col_scale, c);
    }
    for (h, &z) in cut.realized_inflow.iter().enumerate() {
        x[layout.z_col(HydroSys::new(h))] = z;
    }
    for (d, lag) in lag_dims(layout).into_iter().enumerate() {
        if lag.is_some() {
            x[layout.incoming_state_col(StateDim::new(d))] = incoming_state[d];
        }
    }
    if let Some(col) = x.iter().position(|value| !value.is_finite()) {
        panic!(
            "{label} stage {stage}: lifted value of expanded column {col} is {}",
            x[col]
        );
    }
    x
}

fn lifted_row_sums(solve: &ExpandedSolve, x: &[f64]) -> Vec<RowSum> {
    let template = &solve.template;
    let cuts = &solve.cut_rows;
    let mut sums = vec![RowSum::default(); template.num_rows + cuts.num_rows];
    for (col, &x_col) in x.iter().enumerate().take(template.num_cols) {
        let col_scale = scale_at(&template.col_scale, col);
        for (row, value) in column_entries(template, col) {
            sums[row].add(
                value / (scale_at(&template.row_scale, row) * col_scale),
                x_col,
            );
        }
    }
    for k in 0..cuts.num_rows {
        for p in csr_offset(cuts.row_starts[k])..csr_offset(cuts.row_starts[k + 1]) {
            let col = csr_offset(cuts.col_indices[p]);
            sums[template.num_rows + k]
                .add(cuts.values[p] / scale_at(&template.col_scale, col), x[col]);
        }
    }
    sums
}

/// The roundings of the test's own evaluation of a compact row after its `nnz` products and adds:
/// the division of the activity by the row scale and the closing subtraction from a bound.
const COMPACT_ACTIVITY_CLOSING_ROUNDINGS: usize = 2;

/// What the compact solve left on one row of the compact LP, in unscaled units.
#[derive(Clone, Copy)]
struct CompactResidual {
    activity: f64,
    violation: f64,
    slack: f64,
    dual: f64,
    binding: bool,
    rounding: f64,
    order: usize,
}

/// The residual of each row of the compact LP as `solve.compact` solved it, from its recorded
/// bounds, its scaled primal and its scaled dual.
///
/// `rounding` bounds the test's own evaluation of a row: `nnz` products and adds chain
/// `nnz` deep (one product, then `nnz - 1` adds), the closing roundings follow, so
/// `n = nnz + COMPACT_ACTIVITY_CLOSING_ROUNDINGS` over the row's term magnitudes and its larger
/// finite bound. No coefficient is unscaled, since the LP as solved is evaluated in its own units.
fn compact_residuals(
    label: &str,
    stage: usize,
    solve: &ExpandedSolve,
    tolerance: UnscaledTolerance,
) -> Vec<CompactResidual> {
    let loaded = &solve.compact.template;
    let appended = &solve.compact.appended;
    let cut = &solve.compact_cut;
    let n_rows = loaded.num_rows + appended.num_rows;
    assert_eq!(
        cut.dual.len(),
        n_rows,
        "{label} stage {stage}: compact dual length vs the compact LP's row count"
    );
    assert_eq!(
        cut.primal.len(),
        loaded.num_cols,
        "{label} stage {stage}: compact primal length vs the compact LP's column count"
    );
    let mut sums = vec![RowSum::default(); n_rows];
    for col in 0..loaded.num_cols {
        for (row, value) in column_entries(loaded, col) {
            sums[row].add(value, cut.primal[col]);
        }
    }
    for k in 0..appended.num_rows {
        for p in csr_offset(appended.row_starts[k])..csr_offset(appended.row_starts[k + 1]) {
            sums[loaded.num_rows + k].add(
                appended.values[p],
                cut.primal[csr_offset(appended.col_indices[p])],
            );
        }
    }
    (0..n_rows)
        .map(|row| {
            let (scale, raw_lower, raw_upper) = if row < loaded.num_rows {
                (
                    scale_at(&loaded.row_scale, row),
                    loaded.row_lower[row],
                    loaded.row_upper[row],
                )
            } else {
                let k = row - loaded.num_rows;
                (1.0, appended.row_lower[k], appended.row_upper[k])
            };
            let (lower, upper) = (raw_lower / scale, raw_upper / scale);
            let activity = sums[row].activity / scale;
            let bound_magnitude = [lower, upper]
                .into_iter()
                .filter(|bound| bound.is_finite())
                .fold(0.0, |largest, bound| f64::max(largest, bound.abs()));
            let dual = scale * cut.dual[row];
            let order = sums[row].nonzeros + COMPACT_ACTIVITY_CLOSING_ROUNDINGS;
            CompactResidual {
                activity,
                violation: (lower - activity).max(activity - upper).max(0.0),
                slack: [activity - lower, upper - activity]
                    .into_iter()
                    .filter(|distance| distance.is_finite())
                    .fold(f64::INFINITY, f64::min)
                    .max(0.0),
                dual,
                binding: dual.abs() > tolerance.row_dual(scale),
                rounding: rounding_factor(order) * (sums[row].magnitude / scale + bound_magnitude),
                order,
            }
        })
        .collect()
}

/// Checks the compact optimal solution of `solve`, lifted onto the expanded LP, against every
/// expanded row, the warm duals and the objectives, in unscaled units.
///
/// A row with a compact counterpart, the same base row or the cut row of the same slot, is judged
/// against the compact solve's own residual there: the feasibility bound grows by its measured
/// violation, and a row the compact solve treats as binding (its dual beyond the dual tolerance)
/// is interior only beyond its measured slack. A z row has no counterpart and keeps the
/// tolerance and rounding alone.
///
/// # Panics
///
/// Panics naming the first row or objective pair that fails, on a length mismatch and on a
/// non-finite lifted value.
pub fn check_lifted_compact_solution(
    label: &str,
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    incoming_state: &[f64],
    warm: &LpSolution,
) -> LiftedSolutionReport {
    let layout = &expanded.layout;
    let template = &solve.template;
    let cuts = &solve.cut_rows;
    let n_rows = template.num_rows + cuts.num_rows;
    assert_eq!(
        warm.dual.len(),
        n_rows,
        "{label} stage {stage}: warm dual length vs expanded row count"
    );
    let x = lifted_primal(label, compact, expanded, stage, solve, incoming_state);
    let sums = lifted_row_sums(solve, &x);
    let classes = classify_expanded_rows(compact, layout, stage, n_rows);
    let n_hydros = layout.state().hydro_count;
    let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];
    let tolerance = UnscaledTolerance::from_active_profile();
    let residuals = compact_residuals(label, stage, solve, tolerance);

    let mut report = LiftedSolutionReport::default();
    for (row, (sum, &class)) in sums.iter().zip(&classes).enumerate() {
        let is_cut = row >= template.num_rows;
        let (scale, raw_lower, raw_upper) = if is_cut {
            let k = row - template.num_rows;
            (1.0, cuts.row_lower[k], cuts.row_upper[k])
        } else {
            (
                scale_at(&template.row_scale, row),
                template.row_lower[row],
                template.row_upper[row],
            )
        };
        let (lower, upper) = (raw_lower / scale, raw_upper / scale);
        let counterpart = if is_cut {
            Some(compact_template.num_rows + row - template.num_rows)
        } else {
            row.checked_sub(n_hydros)
        };
        let reference = counterpart.map(|r| (r, residuals[r]));
        let solver_tolerance = match counterpart {
            _ if is_cut => tolerance.row_residual(1.0),
            Some(r) => tolerance.row_residual(scale_at(&compact_template.row_scale, r)),
            None => 0.0,
        };
        // n: the row's `nnz - 1` adds, one product and two roundings of the unscaled coefficient;
        // the factor 2 is the same row evaluated by the compact solver in its scaled form.
        let order = sum.nonzeros + 2;
        let rounding = 2.0 * rounding_factor(order) * sum.magnitude;
        let bound = solver_tolerance + rounding;
        let (feasibility_bound, interior_bound) = match reference {
            Some((_, c)) => (
                bound + c.violation + c.rounding,
                if c.binding {
                    bound + c.slack + c.rounding
                } else {
                    bound
                },
            ),
            None => (bound, bound),
        };
        let name = row_families::name(class);
        let compact_note = reference.map_or_else(String::new, |(r, c)| {
            format!(
                "; compact row {r}: activity {:e}, violation {:e}, slack {:e}, dual {:e}, \
                 rounding {:e}, n = {}",
                c.activity, c.violation, c.slack, c.dual, c.rounding, c.order
            )
        });
        assert!(
            sum.activity >= lower - feasibility_bound && sum.activity <= upper + feasibility_bound,
            "{label} stage {stage}: row {row} ({name}): lifted activity {:e} outside [{lower:e}, \
             {upper:e}] by more than {feasibility_bound:e} (solver {solver_tolerance:e} + \
             rounding {rounding:e}, n = {order}){compact_note}",
            sum.activity
        );

        let dual = scale * warm.dual[row];
        let dual_bound = tolerance.row_dual(scale);
        if sum.activity - lower > interior_bound && upper - sum.activity > interior_bound {
            assert!(
                dual.abs() <= dual_bound,
                "{label} stage {stage}: row {row} ({name}): lifted activity {:e} is interior to \
                 [{lower:e}, {upper:e}] by more than {interior_bound:e}, so its dual {dual:e} \
                 must be within {dual_bound:e} of 0{compact_note}",
                sum.activity
            );
            report.slack_rows += 1;
        } else if dual.abs() > dual_bound {
            report.binding_rows_with_dual += 1;
        }
        report.largest_order = report.largest_order.max(order);
        if let Some((_, c)) = reference {
            report.largest_order = report.largest_order.max(c.order);
            report.compact_referenced_rows += 1;
            report.largest_compact_violation = report.largest_compact_violation.max(c.violation);
            if c.binding {
                report.largest_compact_slack = report.largest_compact_slack.max(c.slack);
            }
        }
        report.rows_checked += 1;
        report.cut_rows_checked += usize::from(is_cut);
    }

    let lifted_objective: f64 = (0..template.num_cols)
        .map(|col| template.objective[col] / scale_at(&template.col_scale, col) * x[col])
        .sum();
    let compact_objective = solve.compact_cut.objective;
    for (pair, a, b) in [
        ("lifted vs compact", lifted_objective, compact_objective),
        ("compact vs expanded", compact_objective, warm.objective),
        ("lifted vs expanded", lifted_objective, warm.objective),
    ] {
        let bound = value_tolerance(a, b);
        assert!(
            (a - b).abs() <= bound,
            "{label} stage {stage}: objective {pair}: {a:e} vs {b:e}, tolerance {bound:e}"
        );
    }
    report
}

/// What [`check_compact_cut_rhs`] examined, for the caller's power assertions.
#[derive(Clone, Copy, Default)]
pub struct CutRhsReport {
    /// Cut rows checked.
    pub rows_checked: usize,
    /// The cut rows among them with a nonzero coefficient on a lag slot.
    pub rows_with_lag_terms: usize,
    /// The largest `|RHS_cmp - RHS_exp| / (2·γ_n·S)` over the rows checked.
    pub largest_bound_fraction: f64,
    /// The largest `n` of the bounds.
    pub largest_chain: usize,
}

impl CutRhsReport {
    /// Adds `other`'s counts and keeps the larger bound fraction and chain.
    pub fn absorb(&mut self, other: Self) {
        self.rows_checked += other.rows_checked;
        self.rows_with_lag_terms += other.rows_with_lag_terms;
        self.largest_bound_fraction = self
            .largest_bound_fraction
            .max(other.largest_bound_fraction);
        self.largest_chain = self.largest_chain.max(other.largest_chain);
    }
}

/// The compact RHS closes with three adds: α + S, then + P, then + W.
const COMPACT_CLOSING_ADDS: usize = 3;
/// `b_h` is the z row's RHS after one add, one row-scale product and its division.
const Z_RHS_ROUNDINGS: usize = 3;
/// `ẑ_h` takes one subtraction and one division after `b_h` and the column sum.
const Z_CLOSING_ROUNDINGS: usize = 2;

/// Longest chain of the compact RHS over its terms: S's dot and the three adds; p's accumulation,
/// P's dot and the last two adds; W's dot and the last add; α's three adds.
fn compact_chain_length(lag_point: &CutLagPoint, n_hydros: usize, lag_depth: usize) -> usize {
    let s_chain = lag_point.shifted().len() + COMPACT_CLOSING_ADDS;
    let p_chain = (lag_depth + 1) + n_hydros + (COMPACT_CLOSING_ADDS - 1);
    let w_chain = n_hydros + 1;
    s_chain.max(p_chain).max(w_chain).max(COMPACT_CLOSING_ADDS)
}

/// Longest chain of `ẑ_h` in the expanded LP: from `b_h`, or from the column sum of `lag_depth`
/// products.
fn z_chain_length(lag_depth: usize) -> usize {
    (Z_RHS_ROUNDINGS + Z_CLOSING_ROUNDINGS).max(lag_depth + Z_CLOSING_ROUNDINGS)
}

/// Longest chain of the expanded RHS: `ẑ_h`, then the dot over the lag slots and α's add.
fn expanded_chain_length(lag_point: &CutLagPoint, lag_depth: usize) -> usize {
    z_chain_length(lag_depth) + lag_point.full().len() + 1
}

struct ZDefinition {
    z_hat: f64,
    /// `(|b_h| + Σ_c |A[z_h, c]·x_c|) / |A[z_h, z_col(h)]|`, with `|b_h|` read as `|base| + |w|`
    /// because the compact chain rounds `base` and `w` apart, so cancellation in `b_h` does not
    /// shrink its error.
    magnitude: f64,
}

fn lag_state_dims(layout: &ExpandedLayout) -> Vec<usize> {
    let state = layout.state();
    let mut dims = vec![usize::MAX; state.hydro_count * state.max_par_order];
    for (d, lag) in lag_dims(layout).into_iter().enumerate() {
        if let Some((h, l)) = lag {
            dims[l * state.hydro_count + h] = d;
        }
    }
    assert!(
        dims.iter().all(|&d| d != usize::MAX),
        "some (hydro, lag) has no state dimension"
    );
    dims
}

fn z_definitions(
    label: &str,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    incoming_state: &[f64],
    lag_state_dim: &[usize],
) -> Vec<ZDefinition> {
    let layout = &expanded.layout;
    let n = layout.state().hydro_count;
    let unscaled = &expanded.unscaled[stage];
    let scaled = &expanded.scaled[stage];
    let pinned: HashMap<usize, usize> = lag_state_dim
        .iter()
        .map(|&d| (layout.incoming_state_col(StateDim::new(d)), d))
        .collect();
    let hydro_of_z_row: HashMap<usize, usize> = (0..n)
        .map(|h| (layout.z_row(HydroSys::new(h)), h))
        .collect();
    let mut z_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for col in 0..unscaled.num_cols {
        for (row, value) in column_entries(unscaled, col) {
            if let Some(&h) = hydro_of_z_row.get(&row) {
                z_rows[h].push((col, value));
            }
        }
    }

    (0..n)
        .map(|h| {
            let hydro = HydroSys::new(h);
            let (row, z_col) = (layout.z_row(hydro), layout.z_col(hydro));
            let mut a_z = None;
            let (mut column_sum, mut column_magnitude) = (0.0, 0.0);
            for &(col, a) in &z_rows[h] {
                if col == z_col {
                    a_z = Some(a);
                    continue;
                }
                let d = *pinned.get(&col).unwrap_or_else(|| {
                    panic!(
                        "{label} stage {stage}: z row {row} of hydro {h} holds column {col}, which \
                         is neither its z column {z_col} nor an incoming lag column"
                    )
                });
                let x = incoming_state[d];
                let (pin_lower, pin_upper) =
                    (solve.template.col_lower[col], solve.template.col_upper[col]);
                let pinned_value = pin_lower * scale_at(&scaled.col_scale, col);
                assert!(
                    pin_lower.to_bits() == pin_upper.to_bits()
                        && (pinned_value - x).abs() <= rounding_factor(2) * x.abs(),
                    "{label} stage {stage}: z row {row} of hydro {h}: column {col} (state \
                     dimension {d}) is bounded [{pin_lower:e}, {pin_upper:e}], which unscales to \
                     {pinned_value:e}, not pinned at the incoming state {x:e}"
                );
                column_sum += a * x;
                column_magnitude += (a * x).abs();
            }
            let a_z = a_z.unwrap_or_else(|| {
                panic!(
                    "{label} stage {stage}: z row {row} of hydro {h} has no entry at its z column \
                     {z_col}"
                )
            });
            assert!(
                a_z != 0.0,
                "{label} stage {stage}: z row {row} of hydro {h}: the z column's coefficient is 0"
            );

            let patched = solve.template.row_lower[row];
            assert!(
                patched.to_bits() == solve.template.row_upper[row].to_bits(),
                "{label} stage {stage}: z row {row} of hydro {h} is bounded [{patched:e}, {:e}], \
                 not fixed",
                solve.template.row_upper[row]
            );
            let rhs = patched / scale_at(&scaled.row_scale, row);
            let base = unscaled.row_lower[row];
            let rhs_magnitude =
                (base.abs() + (rhs - base).abs()) / (1.0 - rounding_factor(Z_RHS_ROUNDINGS));
            ZDefinition {
                z_hat: (rhs - column_sum) / a_z,
                magnitude: (rhs_magnitude + column_magnitude) / a_z.abs(),
            }
        })
        .collect()
}

fn compact_cut_rhs<'a>(label: &str, stage: usize, solve: &'a ExpandedSolve) -> &'a [f64] {
    let ranges: Vec<&[f64]> = solve
        .compact
        .writes
        .iter()
        .filter_map(|write| match write {
            BoundWrite::RowLowerRange { lower, .. } => Some(lower.as_slice()),
            _ => None,
        })
        .collect();
    assert_eq!(
        ranges.len(),
        1,
        "{label} stage {stage}: the compact solve wrote {} cut-row RHS ranges, not one",
        ranges.len()
    );
    assert_eq!(
        ranges[0].len(),
        solve.rows.slots.len(),
        "{label} stage {stage}: compact RHS count vs cut-row count"
    );
    ranges[0]
}

fn assert_compact_cut_rows_have_unit_row_scale(label: &str, stage: usize, solve: &ExpandedSolve) {
    let batch = &solve.compact.appended;
    let col_scale = &solve.compact.template.col_scale;
    assert_eq!(
        batch.num_rows + solve.rows.frozen,
        solve.rows.slots.len(),
        "{label} stage {stage}: appended compact cut rows plus frozen ones vs cut-row count"
    );
    for k in 0..batch.num_rows {
        let theta = csr_offset(batch.row_starts[k + 1]) - 1;
        let col = csr_offset(batch.col_indices[theta]);
        let expected = scale_at(col_scale, col);
        assert!(
            batch.values[theta].to_bits() == expected.to_bits(),
            "{label} stage {stage}: compact cut row {k}: its theta entry {:e} differs from the \
             column scale {expected:e}, so the row carries a row scale",
            batch.values[theta]
        );
    }
}

fn cut_row_entries(cut_rows: &RowBatch, k: usize) -> HashMap<usize, f64> {
    (csr_offset(cut_rows.row_starts[k])..csr_offset(cut_rows.row_starts[k + 1]))
        .map(|p| (csr_offset(cut_rows.col_indices[p]), cut_rows.values[p]))
        .collect()
}

/// Checks the RHS the compact solve wrote for each cut row of `solve` against the RHS the expanded
/// LP's own z rows give at the same pinned incoming state and noise, within `2·γ_n·S`.
///
/// # Panics
///
/// Panics naming the first cut row whose two RHS differ by more than the bound, a z row holding a
/// column other than its z column and pinned incoming lag columns, a lag slot whose expanded cut
/// row attaches to another column than the one carrying its value, and a compact cut row with a
/// row scale.
// Rationale (too_many_arguments): the arguments are what every cross-formulation call site
// already holds for the assembly, so bundling them would add a one-use struct.
#[allow(clippy::too_many_arguments)]
pub fn check_compact_cut_rhs(
    label: &str,
    expanded: &ExpandedStudy,
    fcf: &FutureCostFunction,
    cut_state: &CutStateProjection,
    stage: usize,
    solve: &ExpandedSolve,
    incoming_state: &[f64],
    raw_noise_hydro: &[f64],
) -> CutRhsReport {
    let layout = &expanded.layout;
    let dims = layout.state();
    let (n_hydros, depth) = (dims.hydro_count, dims.max_par_order);
    assert_eq!(
        raw_noise_hydro.len(),
        n_hydros,
        "{label} stage {stage}: hydro noise length vs hydro count"
    );
    assert_eq!(
        incoming_state.len(),
        dims.n_state,
        "{label} stage {stage}: incoming state length vs state dimension count"
    );
    let lag_point = cut_state.lag_point();
    let lag_slots = lag_point.full();
    assert!(
        lag_slots.is_empty() || lag_slots.len() == n_hydros * depth,
        "{label} stage {stage}: {} lag slots vs {n_hydros} hydros x {depth} lags",
        lag_slots.len()
    );
    assert_eq!(
        solve.cut_rows.num_rows,
        solve.rows.slots.len(),
        "{label} stage {stage}: expanded cut-row count"
    );
    assert_compact_cut_rows_have_unit_row_scale(label, stage, solve);

    let compact_rhs = compact_cut_rhs(label, stage, solve);
    let lag_state_dim = lag_state_dims(layout);
    let z_defs = z_definitions(
        label,
        expanded,
        stage,
        solve,
        incoming_state,
        &lag_state_dim,
    );
    let chain = compact_chain_length(lag_point, n_hydros, depth)
        .max(expanded_chain_length(lag_point, depth));
    let pool = &fcf.pools[solve.rows.resident_pool];
    let col_scale = &expanded.scaled[stage].col_scale;

    let mut report = CutRhsReport::default();
    for (row, &slot) in solve.rows.slots.iter().enumerate() {
        let (alpha, beta) = (pool.intercept(slot), pool.coefficient_row(slot));
        let attached = cut_row_entries(&solve.cut_rows, row);
        let (mut lag_sum, mut magnitude, mut has_lag_terms) = (0.0, alpha.abs(), false);
        for lag_slot in lag_slots.clone() {
            let offset = lag_slot - lag_slots.start;
            let (lag, hydro_index) = (offset / n_hydros, offset % n_hydros);
            let hydro = HydroSys::new(hydro_index);
            let coefficient = beta[lag_slot];
            let (column, value) = if lag == 0 {
                (layout.z_col(hydro), z_defs[hydro_index].z_hat)
            } else {
                let dim = lag_state_dim[(lag - 1) * n_hydros + hydro_index];
                (
                    layout.incoming_state_col(StateDim::new(dim)),
                    incoming_state[dim],
                )
            };
            assert_eq!(
                layout.outgoing_lag_col(hydro, lag),
                column,
                "{label} stage {stage} cut row {row} (slot {slot}): hydro {hydro_index} lag \
                 {lag}: outgoing_lag_col vs the column that carries the value"
            );
            assert_eq!(
                cut_state.global_state_index(CutSlot::new(lag_slot)).get(),
                lag_state_dim[lag * n_hydros + hydro_index],
                "{label} stage {stage} cut row {row} (slot {slot}): slot {lag_slot} vs the state \
                 dimension of hydro {hydro_index} lag {lag}"
            );
            match attached.get(&column) {
                Some(&entry) => assert!(
                    entry.to_bits() == (-coefficient * col_scale[column]).to_bits(),
                    "{label} stage {stage} cut row {row} (slot {slot}): hydro {hydro_index} lag \
                     {lag}: the expanded cut row holds {entry:e} at column {column}, expected {:e}",
                    -coefficient * col_scale[column]
                ),
                None => assert!(
                    coefficient == 0.0,
                    "{label} stage {stage} cut row {row} (slot {slot}): hydro {hydro_index} lag \
                     {lag} has coefficient {coefficient:e} and no entry at column {column}"
                ),
            }
            let term = coefficient * value;
            lag_sum += term;
            magnitude += term.abs();
            if lag == 0 {
                magnitude += coefficient.abs() * z_defs[hydro_index].magnitude;
            }
            has_lag_terms |= coefficient != 0.0;
        }

        let (rhs_cmp, rhs_exp) = (compact_rhs[row], alpha + lag_sum);
        let (gap, bound) = (
            (rhs_cmp - rhs_exp).abs(),
            2.0 * rounding_factor(chain) * magnitude,
        );
        assert!(
            gap <= bound,
            "{label} stage {stage} cut row {row} (slot {slot}): compact RHS {rhs_cmp:e}, expanded \
             RHS {rhs_exp:e} (alpha {alpha:e}, lag terms {lag_sum:e}), gap {gap:e} > 2*gamma_n*S \
             {bound:e} (S = {magnitude:e}, n = {chain}: compact {}, expanded {}; hydro noise \
             {raw_noise_hydro:?})",
            compact_chain_length(lag_point, n_hydros, depth),
            expanded_chain_length(lag_point, depth)
        );
        report.rows_checked += 1;
        report.largest_chain = chain;
        report.rows_with_lag_terms += usize::from(has_lag_terms);
        if bound > 0.0 {
            report.largest_bound_fraction = report.largest_bound_fraction.max(gap / bound);
        }
    }
    report
}

/// What [`check_compact_shifted_base_rows`] examined, for the caller's power assertions.
#[derive(Clone, Copy, Default)]
pub struct ShiftedBaseRowReport {
    /// Expanded base rows holding a z or incoming lag column.
    pub rows_checked: usize,
    /// The largest `|L_cmp - L_exp| / (2·γ_n·S)` over the finite bounds checked.
    pub largest_bound_fraction: f64,
    /// The largest `n` of the bounds.
    pub largest_chain: usize,
}

impl ShiftedBaseRowReport {
    /// Adds `other`'s count and keeps the larger bound fraction and chain.
    pub fn absorb(&mut self, other: Self) {
        self.rows_checked += other.rows_checked;
        self.largest_bound_fraction = self
            .largest_bound_fraction
            .max(other.largest_bound_fraction);
        self.largest_chain = self.largest_chain.max(other.largest_chain);
    }
}

/// A compact shifted bound closes with the subtraction from the unscaled bound, the product with
/// the row scale, and the division by it that unscales the written bound.
const SHIFTED_BOUND_CLOSING_ROUNDINGS: usize = 3;

/// Longest chain of a compact shifted bound over a row with `columns` inflow columns: `ẑ_h` is
/// the conditional mean (one product, then `lag_depth` adds) plus the noise term, then the shift
/// is one product and `columns - 1` adds from +0.0, then the closing.
fn compact_shifted_bound_chain_length(lag_depth: usize, columns: usize) -> usize {
    (lag_depth + 2) + columns + SHIFTED_BOUND_CLOSING_ROUNDINGS
}

/// Longest chain of the expanded shifted bound over a row with `columns` inflow columns: `ẑ_h`,
/// then one product, `columns - 1` adds and the subtraction from the bound. The coefficients are
/// read unscaled, so no unscaling rounds.
fn expanded_shifted_bound_chain_length(lag_depth: usize, columns: usize) -> usize {
    z_chain_length(lag_depth) + columns + 1
}

/// The last `(lower, upper)` of compact base row `row` written by `set_row_bounds` in `writes`.
fn last_row_bounds_write(writes: &[BoundWrite], row: usize) -> Option<(f64, f64)> {
    writes
        .iter()
        .filter_map(|write| match write {
            BoundWrite::RowBounds {
                indices,
                lower,
                upper,
            } => indices
                .iter()
                .rposition(|&r| r == row)
                .map(|i| (lower[i], upper[i])),
            _ => None,
        })
        .next_back()
}

/// Checks each base row of the expanded LP that holds a z or an incoming lag column: the bound the
/// compact solve saw on its counterpart row equals the unscaled expanded bound minus the row's
/// inflow columns evaluated at the z-row definition of `ẑ` and the pinned incoming lags, within
/// `2·γ_n·S`.
///
/// # Panics
///
/// Panics naming the first row and bound that differ by more than the bound, with the row family
/// and whether no write covers a row whose bound shifts; a column of the expanded LP with no
/// compact counterpart that is neither a z column nor an incoming lag column; and a z row that
/// does not define `ẑ`.
pub fn check_compact_shifted_base_rows(
    label: &str,
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    incoming_state: &[f64],
) -> ShiftedBaseRowReport {
    let layout = &expanded.layout;
    let dims = layout.state();
    let (n_hydros, depth) = (dims.hydro_count, dims.max_par_order);
    assert_eq!(
        incoming_state.len(),
        dims.n_state,
        "{label} stage {stage}: incoming state length vs state dimension count"
    );
    let unscaled = &expanded.unscaled[stage];
    let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];
    let lag_state_dim = lag_state_dims(layout);
    let z_defs = z_definitions(
        label,
        expanded,
        stage,
        solve,
        incoming_state,
        &lag_state_dim,
    );

    let compact_columns: HashSet<usize> = (0..compact_template.num_cols)
        .map(|c| layout.col(c))
        .collect();
    let mut inflow_columns: HashMap<usize, (f64, Option<usize>)> = HashMap::new();
    for h in 0..n_hydros {
        let hydro = HydroSys::new(h);
        inflow_columns.insert(layout.z_col(hydro), (z_defs[h].z_hat, Some(h)));
        for l in 0..depth {
            let dim = lag_state_dim[l * n_hydros + h];
            inflow_columns.insert(
                layout.incoming_lag_col(hydro, l),
                (incoming_state[dim], None),
            );
        }
    }
    let base_rows: HashMap<usize, usize> = (0..compact_template.num_rows)
        .map(|r| (layout.row(r), r))
        .collect();
    let mut by_row: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for col in (0..unscaled.num_cols).filter(|col| !compact_columns.contains(col)) {
        assert!(
            inflow_columns.contains_key(&col),
            "{label} stage {stage}: expanded column {col} has no compact counterpart and is \
             neither a z column nor an incoming lag column"
        );
        for (row, value) in column_entries(unscaled, col) {
            if base_rows.contains_key(&row) {
                by_row.entry(row).or_default().push((col, value));
            }
        }
    }
    let mut shifted: Vec<_> = by_row.into_iter().collect();
    shifted.sort_unstable_by_key(|&(row, _)| row);

    let n_rows = unscaled.num_rows + solve.cut_rows.num_rows;
    let classes = classify_expanded_rows(compact, layout, stage, n_rows);
    let writes = &solve.compact.writes;
    let mut report = ShiftedBaseRowReport::default();
    for (row, entries) in &shifted {
        let compact_row = base_rows[row];
        let name = row_families::name(classes[*row]);
        let (mut shift, mut magnitude) = (0.0, 0.0);
        for &(col, a) in entries {
            let (value, z_hydro) = inflow_columns[&col];
            shift += a * value;
            magnitude += (a * value).abs();
            if let Some(h) = z_hydro {
                magnitude += a.abs() * z_defs[h].magnitude;
            }
        }
        let written = last_row_bounds_write(writes, compact_row);
        let row_scale = scale_at(&compact_template.row_scale, compact_row);
        let (scaled_lower, scaled_upper) = written.unwrap_or((
            compact_template.row_lower[compact_row],
            compact_template.row_upper[compact_row],
        ));
        let (cmp_lower, cmp_upper) = (scaled_lower / row_scale, scaled_upper / row_scale);
        let columns = entries.len();
        let chain = compact_shifted_bound_chain_length(depth, columns)
            .max(expanded_shifted_bound_chain_length(depth, columns));
        for (side, bound, cmp) in [
            ("lower", unscaled.row_lower[*row], cmp_lower),
            ("upper", unscaled.row_upper[*row], cmp_upper),
        ] {
            let note = if written.is_none() && shift != 0.0 {
                "; no write covers the row, so its inflow shift was dropped"
            } else {
                ""
            };
            if !bound.is_finite() {
                assert!(
                    bound.to_bits() == cmp.to_bits(),
                    "{label} stage {stage}: compact row {compact_row} / expanded row {row} \
                     ({name}): {side} bound {bound:e} is not finite but the compact bound is \
                     {cmp:e}{note}"
                );
                continue;
            }
            let expected = bound - shift;
            let s = bound.abs() + magnitude;
            let (gap, limit) = ((cmp - expected).abs(), 2.0 * rounding_factor(chain) * s);
            assert!(
                gap <= limit,
                "{label} stage {stage}: compact row {compact_row} / expanded row {row} ({name}): \
                 {side} bound {cmp:e} the compact solve saw vs {expected:e} from the expanded row \
                 (bound {bound:e}, inflow columns {shift:e}), gap {gap:e} > 2*gamma_n*S {limit:e} \
                 (S = {s:e}, n = {chain}: compact {}, expanded {}){note}",
                compact_shifted_bound_chain_length(depth, columns),
                expanded_shifted_bound_chain_length(depth, columns)
            );
            if limit > 0.0 {
                report.largest_bound_fraction = report.largest_bound_fraction.max(gap / limit);
            }
        }
        report.rows_checked += 1;
        report.largest_chain = report.largest_chain.max(chain);
    }
    report
}

/// The cross-formulation checks of one trained study at identical pinned states, with their totals.
pub struct PinnedStateRun<'a> {
    label: &'a str,
    logical: &'a StudySetup,
    compact: &'a StudySetup,
    expanded: &'a ExpandedStudy,
    evidence: PowerEvidence,
    /// The cut-row RHS reports summed over every solve.
    pub rhs: CutRhsReport,
    /// The shifted base-row reports summed over every solve.
    pub shifted: ShiftedBaseRowReport,
    /// The lifted-solution reports summed over every solve.
    pub lifted: LiftedSolutionReport,
    stages: usize,
    solves: usize,
    non_unit_row_scale: Option<(usize, usize, f64)>,
}

impl<'a> PinnedStateRun<'a> {
    /// A run over `logical`'s trained cuts and trial states, solved in `compact` and `expanded`.
    pub fn new(
        label: &'a str,
        logical: &'a StudySetup,
        compact: &'a StudySetup,
        expanded: &'a ExpandedStudy,
    ) -> Self {
        let state = compact.stage_state();
        Self {
            label,
            logical,
            compact,
            expanded,
            evidence: PowerEvidence::new(state.hydro_count, state.max_par_order),
            rhs: CutRhsReport::default(),
            shifted: ShiftedBaseRowReport::default(),
            lifted: LiftedSolutionReport::default(),
            stages: 0,
            solves: 0,
            non_unit_row_scale: None,
        }
    }

    /// Checks every opening of `stage` at every state of `states`: the cut-row RHS and the shifted
    /// base-row bounds first, then the lifted compact solution on the expanded LP warm-started
    /// from the mapped basis.
    pub fn check_stage(&mut self, stage: usize, states: &[Vec<f64>]) -> CutRhsReport {
        let training_ctx = self.compact.training_ctx();
        let tree = training_ctx.stochastic.tree_view();
        let hydro_range = self
            .compact
            .inputs
            .stochastic
            .class_dimensions()
            .hydro_range();
        let rows = CutRowSet::active(&self.logical.fcf, stage, stage - 1);
        let n_openings = tree.n_openings(stage);
        let mut stage_rhs = CutRhsReport::default();
        let mut stage_shifted = ShiftedBaseRowReport::default();
        let mut stage_lifted = LiftedSolutionReport::default();
        for opening in 0..n_openings {
            let raw_noise = tree.opening(stage, opening);
            for x_hat in states {
                let solve = assemble_expanded_solve(
                    self.compact,
                    self.expanded,
                    &self.logical.fcf,
                    &rows,
                    stage,
                    x_hat,
                    raw_noise,
                );
                stage_rhs.absorb(check_compact_cut_rhs(
                    self.label,
                    self.expanded,
                    &self.logical.fcf,
                    &training_ctx.cut_state_layouts[rows.resident_pool],
                    stage,
                    &solve,
                    x_hat,
                    &raw_noise[hydro_range.clone()],
                ));
                stage_shifted.absorb(check_compact_shifted_base_rows(
                    self.label,
                    self.compact,
                    self.expanded,
                    stage,
                    &solve,
                    x_hat,
                ));
                let warm = solve_from_mapped_basis(self.label, self.expanded, &solve);
                stage_lifted.absorb(check_lifted_compact_solution(
                    self.label,
                    self.compact,
                    self.expanded,
                    stage,
                    &solve,
                    x_hat,
                    &warm,
                ));
                record_expanded_evidence(
                    &mut self.evidence,
                    self.compact,
                    self.expanded,
                    stage,
                    &solve,
                    &warm.dual,
                );
                self.solves += 1;
            }
        }

        self.rhs.absorb(stage_rhs);
        self.shifted.absorb(stage_shifted);
        self.lifted.absorb(stage_lifted);
        self.stages += 1;
        if self.non_unit_row_scale.is_none() {
            let ctx = self.compact.stage_ctx();
            let row_scale = &ctx.templates[stage].row_scale;
            self.non_unit_row_scale = ctx.inflow_coupling[stage]
                .touched_rows()
                .iter()
                .map(|&r| usize::try_from(r).expect("row fits usize"))
                .find_map(|r| {
                    let scale = scale_at(row_scale, r);
                    (scale.to_bits() != 1.0_f64.to_bits()).then_some((stage, r, scale))
                });
        }
        eprintln!(
            "{}: stage {stage}: {} states x {n_openings} openings, {} cut rows ({} with lag \
             terms), largest RHS gap {:.3e} of its bound (n {}); {} shifted base rows, largest \
             bound gap {:.3e} of its bound (n {}); {} lifted rows ({} cut rows, {} binding with a \
             dual, {} slack, n up to {}), {} judged against a compact row, largest violation \
             {:.3e}, largest slack {:.3e}",
            self.label,
            states.len(),
            stage_rhs.rows_checked,
            stage_rhs.rows_with_lag_terms,
            stage_rhs.largest_bound_fraction,
            stage_rhs.largest_chain,
            stage_shifted.rows_checked,
            stage_shifted.largest_bound_fraction,
            stage_shifted.largest_chain,
            stage_lifted.rows_checked,
            stage_lifted.cut_rows_checked,
            stage_lifted.binding_rows_with_dual,
            stage_lifted.slack_rows,
            stage_lifted.largest_order,
            stage_lifted.compact_referenced_rows,
            stage_lifted.largest_compact_violation,
            stage_lifted.largest_compact_slack
        );
        stage_rhs
    }

    /// The totals of the run as one line.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} stages, {} solves, {} cut rows ({} with lag terms), largest RHS gap {:.3e} of \
             its bound (n {}), {} shifted base rows, largest bound gap {:.3e} of its bound (n {}), \
             {} lifted rows ({} cut rows, {} binding with a dual, {} slack, n up to {}), {} judged \
             against a compact row, largest violation {:.3e}, largest slack {:.3e}, first coupled \
             row with a row scale other than 1 (stage, row, scale): {:?}",
            self.label,
            self.stages,
            self.solves,
            self.rhs.rows_checked,
            self.rhs.rows_with_lag_terms,
            self.rhs.largest_bound_fraction,
            self.rhs.largest_chain,
            self.shifted.rows_checked,
            self.shifted.largest_bound_fraction,
            self.shifted.largest_chain,
            self.lifted.rows_checked,
            self.lifted.cut_rows_checked,
            self.lifted.binding_rows_with_dual,
            self.lifted.slack_rows,
            self.lifted.largest_order,
            self.lifted.compact_referenced_rows,
            self.lifted.largest_compact_violation,
            self.lifted.largest_compact_slack,
            self.non_unit_row_scale
        )
    }

    /// Asserts the run had power: a lag-bearing cut row, a shifted base row, a lifted cut row, a
    /// binding row with a dual, and a coupled row with a non-unit compact row scale and a dual.
    pub fn assert_powers(&self) {
        let label = self.label;
        assert!(
            self.rhs.rows_with_lag_terms >= 1,
            "{label}: no checked cut row has a nonzero lag coefficient among {} rows",
            self.rhs.rows_checked
        );
        assert!(
            self.shifted.rows_checked >= 1,
            "{label}: no base row holding a z or incoming lag column was checked"
        );
        assert!(
            self.lifted.cut_rows_checked >= 1,
            "{label}: no cut row was checked among {} lifted rows",
            self.lifted.rows_checked
        );
        assert!(
            self.lifted.binding_rows_with_dual >= 1,
            "{label}: no row within tolerance of a bound carried a dual"
        );
        self.evidence
            .assert_holds(label, &[Guard::NonUnitCompactCoupledRowScaleWithDual]);
    }
}
