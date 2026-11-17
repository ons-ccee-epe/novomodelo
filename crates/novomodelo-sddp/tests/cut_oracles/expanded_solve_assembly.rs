use std::collections::HashSet;

use novomodelo_io::config::InflowNonNegativityMethod;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::test_support::bits;
use novomodelo_sddp::test_support::bound_recording::BoundWrite;
use novomodelo_sddp::test_support::expander::{
    ExpandedPatch, ExpandedStudy, expand_stage_templates, expanded_solve_patch,
};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use crate::common::expanded_solve::{
    CutRowSet, ExpandedSolve, assemble_expanded_solve, check_lifted_compact_solution,
    load_expanded, record_expanded_evidence, solve_from_mapped_basis,
};
use crate::common::in_code_studies::FPHA_LAG_STUDY_KEY as LAG_STUDY;
use crate::common::in_code_studies::fpha_lag_study;
use crate::common::oracle_tolerance::assert_first_try;
use crate::common::power_guards::{Guard, PowerEvidence};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    build_setup_in_code_with_models, train_storing_trial_states, try_stored_trial_states,
};

const STAGE: usize = 1;

struct Case {
    compact: StudySetup,
    expanded: ExpandedStudy,
    incoming_state: Vec<f64>,
    raw_noise: Vec<f64>,
}

impl Case {
    fn new() -> Self {
        let (system, mut config, hydro_models) = fpha_lag_study();
        apply_oracle_config(&mut config);
        config.modeling.inflow_non_negativity.method = InflowNonNegativityMethod::Truncation;
        let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
        train_storing_trial_states(LAG_STUDY, &mut compact);
        let (system, _, _) = fpha_lag_study();
        let expanded = expand_stage_templates(&system, &compact);
        let incoming_state = try_stored_trial_states(&compact, NodePos(STAGE - 1))
            .expect("the lag study's pools project every state dimension")
            .into_iter()
            .next()
            .expect("pool 0 stores a trial state");
        let raw_noise = compact
            .training_ctx()
            .stochastic
            .tree_view()
            .opening(STAGE, 0)
            .to_vec();
        Self {
            compact,
            expanded,
            incoming_state,
            raw_noise,
        }
    }

    fn assemble(&self) -> ExpandedSolve {
        assemble_expanded_solve(
            &self.compact,
            &self.expanded,
            &self.compact.fcf,
            &CutRowSet::active(&self.compact.fcf, STAGE, STAGE - 1),
            STAGE,
            &self.incoming_state,
            &self.raw_noise,
        )
    }
}

fn written_bounds(solve: &ExpandedSolve) -> (HashSet<usize>, HashSet<usize>) {
    let mut cols = HashSet::new();
    let mut rows = HashSet::new();
    for write in &solve.compact.writes {
        match write {
            BoundWrite::ColBounds { indices, .. } => cols.extend(indices.iter().copied()),
            BoundWrite::RowBounds { indices, .. } => rows.extend(indices.iter().copied()),
            BoundWrite::AddRows { .. } | BoundWrite::RowLowerRange { .. } => {}
        }
    }
    (cols, rows)
}

fn assert_columns_carry_the_compact_bounds(
    case: &Case,
    solve: &ExpandedSolve,
    patch: &ExpandedPatch,
    written: &HashSet<usize>,
) {
    let layout = &case.expanded.layout;
    let base = &case.expanded.scaled[STAGE];
    let compact_final = &solve.compact.template;
    let pinned: HashSet<usize> = patch.incoming_cols.iter().map(|&(col, _)| col).collect();
    let mapped: HashSet<usize> = written.iter().map(|&c| layout.col(c)).collect();
    for &c in written {
        let col = layout.col(c);
        if pinned.contains(&col) {
            continue;
        }
        assert_eq!(
            (
                solve.template.col_lower[col].to_bits(),
                solve.template.col_upper[col].to_bits()
            ),
            (
                compact_final.col_lower[c].to_bits(),
                compact_final.col_upper[c].to_bits()
            ),
            "stage {STAGE}: compact column {c} (expanded column {col}) lost its written bounds"
        );
    }
    for col in 0..base.num_cols {
        if pinned.contains(&col) || mapped.contains(&col) {
            continue;
        }
        assert_eq!(
            (
                solve.template.col_lower[col].to_bits(),
                solve.template.col_upper[col].to_bits()
            ),
            (base.col_lower[col].to_bits(), base.col_upper[col].to_bits()),
            "stage {STAGE}: expanded column {col} is neither written nor pinned and moved"
        );
    }
    for &(col, pin) in &patch.incoming_cols {
        assert_eq!(
            (
                solve.template.col_lower[col].to_bits(),
                solve.template.col_upper[col].to_bits()
            ),
            (pin.to_bits(), pin.to_bits()),
            "stage {STAGE}: incoming column {col} does not hold its pin"
        );
    }
}

fn assert_a_misplaced_column_index_would_move_an_unwritten_column(
    case: &Case,
    solve: &ExpandedSolve,
    patch: &ExpandedPatch,
) {
    let layout = &case.expanded.layout;
    let base = &case.expanded.scaled[STAGE];
    let entries: Vec<(usize, u64, u64)> = solve
        .compact
        .writes
        .iter()
        .filter_map(|write| match write {
            BoundWrite::ColBounds {
                indices,
                lower,
                upper,
            } => Some((indices, lower, upper)),
            _ => None,
        })
        .flat_map(|(indices, lower, upper)| {
            indices
                .iter()
                .zip(lower)
                .zip(upper)
                .map(|((&c, &lo), &up)| (c, lo.to_bits(), up.to_bits()))
        })
        .collect();
    let accounted: HashSet<usize> = patch
        .incoming_cols
        .iter()
        .map(|&(col, _)| col)
        .chain(entries.iter().map(|&(c, _, _)| layout.col(c)))
        .collect();
    let powered = entries.iter().any(|&(c, lo, up)| {
        layout.col(c) != c
            && !accounted.contains(&c)
            && (lo, up) != (base.col_lower[c].to_bits(), base.col_upper[c].to_bits())
    });
    assert!(
        powered,
        "stage {STAGE}: no written column bound has a relocated index whose own expanded column is \
         unwritten, unpinned and has other bounds, so a misplaced column index would not move an \
         unwritten column"
    );
}

fn assert_rows_carry_the_compact_bounds(
    case: &Case,
    solve: &ExpandedSolve,
    patch: &ExpandedPatch,
    written: &HashSet<usize>,
) {
    let layout = &case.expanded.layout;
    let base = &case.expanded.scaled[STAGE];
    let compact_final = &solve.compact.template;
    let touched: HashSet<usize> = case.compact.stage_ctx().inflow_coupling[STAGE]
        .touched_rows()
        .iter()
        .map(|&r| usize::try_from(r).expect("row fits usize"))
        .collect();
    let mapped: HashSet<usize> = written
        .iter()
        .filter(|&&r| !touched.contains(&r))
        .map(|&r| layout.row(r))
        .collect();
    for &r in written {
        if touched.contains(&r) {
            continue;
        }
        let row = layout.row(r);
        assert_eq!(
            (
                solve.template.row_lower[row].to_bits(),
                solve.template.row_upper[row].to_bits()
            ),
            (
                compact_final.row_lower[r].to_bits(),
                compact_final.row_upper[r].to_bits()
            ),
            "stage {STAGE}: compact row {r} (expanded row {row}) lost its written bounds"
        );
    }
    let z_rows: HashSet<usize> = patch.z_rows.iter().map(|&(row, _)| row).collect();
    let coupled: HashSet<usize> = touched.iter().map(|&r| layout.row(r)).collect();
    for row in 0..base.num_rows {
        if z_rows.contains(&row) || mapped.contains(&row) || coupled.contains(&row) {
            continue;
        }
        assert_eq!(
            (
                solve.template.row_lower[row].to_bits(),
                solve.template.row_upper[row].to_bits()
            ),
            (base.row_lower[row].to_bits(), base.row_upper[row].to_bits()),
            "stage {STAGE}: expanded row {row} is neither written nor a z row and moved"
        );
    }
    for &r in &touched {
        let row = layout.row(r);
        assert_eq!(
            bits(&[solve.template.row_lower[row], solve.template.row_upper[row]]),
            bits(&[base.row_lower[row], base.row_upper[row]]),
            "stage {STAGE}: coupled row {r} (expanded row {row}) lost its own bounds"
        );
    }
    for &(row, rhs) in &patch.z_rows {
        assert_eq!(
            (
                solve.template.row_lower[row].to_bits(),
                solve.template.row_upper[row].to_bits()
            ),
            (rhs.to_bits(), rhs.to_bits()),
            "stage {STAGE}: z row {row} does not hold its right-hand side"
        );
    }
}

#[test]
fn the_expanded_solve_carries_every_compact_per_solve_bound() {
    let case = Case::new();
    let solve = case.assemble();
    let hydro_range = case
        .compact
        .inputs
        .stochastic
        .class_dimensions()
        .hydro_range();
    let patch = expanded_solve_patch(
        &case.compact,
        &case.expanded,
        STAGE,
        &case.raw_noise[hydro_range],
        &case.incoming_state,
    );

    let cut_ranges: Vec<(usize, usize)> = solve
        .compact
        .writes
        .iter()
        .filter_map(|write| match write {
            BoundWrite::RowLowerRange { first_row, lower } => Some((*first_row, lower.len())),
            _ => None,
        })
        .collect();
    assert_eq!(
        cut_ranges,
        [(solve.compact.base_rows, solve.rows.slots.len())],
        "stage {STAGE}: the compact solve must push one cut-row range over the appended rows"
    );

    let (written_cols, written_rows) = written_bounds(&solve);
    assert!(
        !written_cols.is_empty() && !written_rows.is_empty(),
        "stage {STAGE}: the compact solve wrote no column or no row bound"
    );
    assert_columns_carry_the_compact_bounds(&case, &solve, &patch, &written_cols);
    assert_a_misplaced_column_index_would_move_an_unwritten_column(&case, &solve, &patch);
    assert_rows_carry_the_compact_bounds(&case, &solve, &patch, &written_rows);

    assert_eq!(
        solve.cut_rows.num_rows,
        solve.rows.slots.len(),
        "stage {STAGE}: expanded cut-row count"
    );
    assert_eq!(
        solve.rows.slots.len(),
        solve.compact.appended.num_rows,
        "stage {STAGE}: compact cut-row count"
    );
}

#[test]
fn the_assembled_expanded_lp_solves_on_the_first_attempt() {
    let case = Case::new();
    let solve = case.assemble();
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    load_expanded(&mut solver, &solve);

    let before = solver.statistics();
    let outcome = solver.solve(None).map(|view| view.dual.to_vec());
    let after = solver.statistics();
    assert_first_try(LAG_STUDY, &before, &after);
    let dual = outcome.expect("the expanded LP must solve");

    let state = case.compact.stage_state();
    let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
    record_expanded_evidence(
        &mut evidence,
        &case.compact,
        &case.expanded,
        STAGE,
        &solve,
        &dual,
    );
    evidence.assert_holds(LAG_STUDY, &[Guard::NonUnitZScale, Guard::NonUnitLagScale]);
}

#[test]
fn the_lifted_compact_solution_meets_every_expanded_row() {
    let (system, mut config, hydro_models) = fpha_lag_study();
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    train_storing_trial_states(LAG_STUDY, &mut compact);
    let (system, _, _) = fpha_lag_study();
    let expanded = expand_stage_templates(&system, &compact);
    let state = compact.stage_state();
    let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
    let training_ctx = compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();

    let (mut rows, mut cut_rows, mut binding_with_dual, mut slack) = (0, 0, 0, 0);
    for stage in 1..compact.fcf.pools.len() {
        if compact.fcf.active_cuts(stage).next().is_none() {
            continue;
        }
        let states = try_stored_trial_states(&compact, NodePos(stage - 1))
            .expect("the lag study's pools project every state dimension");
        for opening in 0..tree.n_openings(stage) {
            let raw_noise = tree.opening(stage, opening);
            for x_hat in &states {
                let solve = assemble_expanded_solve(
                    &compact,
                    &expanded,
                    &compact.fcf,
                    &CutRowSet::active(&compact.fcf, stage, stage - 1),
                    stage,
                    x_hat,
                    raw_noise,
                );
                let warm = solve_from_mapped_basis(LAG_STUDY, &expanded, &solve);
                let report = check_lifted_compact_solution(
                    LAG_STUDY, &compact, &expanded, stage, &solve, x_hat, &warm,
                );
                record_expanded_evidence(
                    &mut evidence,
                    &compact,
                    &expanded,
                    stage,
                    &solve,
                    &warm.dual,
                );
                rows += report.rows_checked;
                cut_rows += report.cut_rows_checked;
                binding_with_dual += report.binding_rows_with_dual;
                slack += report.slack_rows;
            }
        }
    }

    assert!(
        cut_rows >= 1,
        "{LAG_STUDY}: no cut row was checked among {rows} rows"
    );
    assert!(
        binding_with_dual >= 1,
        "{LAG_STUDY}: no row within tolerance of a bound carried a dual"
    );
    assert!(
        slack >= 1,
        "{LAG_STUDY}: no row was interior, so complementarity was never exercised"
    );
    evidence.assert_holds(
        LAG_STUDY,
        &[
            Guard::BindingLagCut,
            Guard::BindingShiftedLagCut,
            Guard::NonUnitZScale,
            Guard::NonUnitLagScale,
            Guard::RescaledCoupledRowWithDual,
            Guard::NonUnitCompactCoupledRowScaleWithDual,
        ],
    );
}
