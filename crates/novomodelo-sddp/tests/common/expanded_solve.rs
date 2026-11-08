//! One compact folded solve and the expanded LP of the same solve: the compact LP is prepared by
//! the production folded prep and solved once, and every per-solve bound it wrote is carried onto
//! the expanded template, so that mapping has this one owner.

use std::collections::HashSet;

use novomodelo_sddp::indexer::{CutSlot, HydroSys, StateDim};
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::test_support::bound_recording::{
    BoundRecordingSolver, BoundWrite, RecordedSolve,
};
use novomodelo_sddp::test_support::expander::{
    ExpandedLayout, ExpandedStudy, expanded_cut_rows_into, expanded_solve_patch,
};
use novomodelo_sddp::test_support::{FoldedProbeCuts, folded_prep_for_probe};
use novomodelo_sddp::{FutureCostFunction, StudySetup, build_cut_row_batch_into};
use novomodelo_solver::{
    ActiveSolver, Basis, BasisStatus, LpSolution, RowBatch, SolverInterface, StageTemplate,
};

use super::StubComm;
use super::oracle_tolerance::assert_first_try;
use super::power_guards::PowerEvidence;

/// The compact solve and the expanded LP assembled from it.
pub struct ExpandedSolve {
    /// The compact LP as solved.
    pub compact: RecordedSolve,
    /// The compact LP's optimal basis.
    pub compact_basis: Basis,
    /// `slots[k]` is the resident pool slot of cut row `k`, in both LPs.
    pub slots: Vec<usize>,
    /// The scaled expanded template with every per-solve bound applied.
    pub template: StageTemplate,
    /// The expanded cut rows, to append to `template`.
    pub cut_rows: RowBatch,
    /// Cut `k`'s lag coefficients at offset `k * N * L`, lag-major: offset `l * N + h` within a cut.
    pub lag_coefficients: Vec<f64>,
}

pub fn empty_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: vec![0],
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    }
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

/// Prepares the compact LP of `stage` through the production folded prep, solves it once, and
/// assembles the expanded LP of the same solve.
pub fn assemble_expanded_solve(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    fcf: &FutureCostFunction,
    pool: usize,
    stage: usize,
    incoming_state: &[f64],
    raw_noise: &[f64],
) -> ExpandedSolve {
    let layout = &expanded.layout;
    let slots: Vec<usize> = fcf.active_cuts(pool).map(|(slot, _, _)| slot).collect();
    assert!(!slots.is_empty(), "pool {pool} holds no active cut");

    let ctx = compact.stage_ctx();
    let training_ctx = compact.training_ctx();
    let projection = &training_ctx.cut_state_layouts[pool];
    let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];

    let mut compact_cuts = empty_batch();
    build_cut_row_batch_into(
        &mut compact_cuts,
        fcf,
        pool,
        compact.stage_state(),
        projection,
        &compact_template.col_scale,
    );
    let mut workspaces = compact
        .create_workspace_pool(&StubComm, 1, || {
            ActiveSolver::new().map(BoundRecordingSolver::new)
        })
        .expect("the workspace pool must build");
    let ws = &mut workspaces.workspaces[0];
    ws.solver.load_model(compact_template);
    ws.solver.add_rows(&compact_cuts);

    folded_prep_for_probe(
        ws,
        &ctx,
        &training_ctx,
        StageIdx(stage),
        incoming_state,
        raw_noise,
        &FoldedProbeCuts {
            pool: &fcf.pools[pool],
            pool_id: pool,
            slots: &slots,
        },
    );
    let before = ws.solver.statistics();
    let solved = ws.solver.solve(None).map(|_| ());
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
        [(compact_solve.base_rows, slots.len())],
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

    let mut cut_rows = empty_batch();
    expanded_cut_rows_into(
        &mut cut_rows,
        fcf,
        pool,
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
    for (k, (_, _, coefficients)) in fcf.active_cuts(pool).enumerate() {
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let d = projection.global_state_index(CutSlot::new(j)).get();
            if let Some((h, l)) = lag_dims[d] {
                lag_coefficients[k * width + l * dims.hydro_count + h] = coefficient;
            }
        }
    }

    ExpandedSolve {
        compact: compact_solve,
        compact_basis,
        slots,
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
pub fn map_compact_basis(
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

    for k in 0..solve.slots.len() {
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
