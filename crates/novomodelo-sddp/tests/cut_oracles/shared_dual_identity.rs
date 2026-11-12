use std::path::Path;

use novomodelo_core::System;
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::cut::chain_rule::{
    ChainRuleInputs, ChainRuleScratch, OpeningDuals, per_opening_cut_into,
};
use novomodelo_sddp::indexer::{ChainRuleSlots, CutSlot, HydroSys};
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::setup::{NodePos, StudyParams};
use novomodelo_sddp::test_support::expander::{ExpandedStudy, expand_stage_templates};
use novomodelo_sddp::test_support::template_structure::column_entries;
use novomodelo_sddp::test_support::{bits, fixture_priced_date, write_synthetic_boundary, ymd};
use novomodelo_sddp::{
    BoundaryLoadRequest, BoundaryStateRequirements, StudySetup, inject_boundary_cuts,
    load_boundary_cuts,
};
use novomodelo_solver::{ActiveSolver, LpSolution, RowBatch, SolverInterface};

use crate::common::expanded_solve::{
    CutRowSet, ExpandedSolve, assemble_expanded_solve, coupled_dual_magnitudes, coupled_duals,
    lag_dims, load_expanded, record_expanded_evidence, solve_from_mapped_basis,
    stages_whose_producer_drops_a_state_dimension,
};
use crate::common::in_code_studies::FPHA_LAG_STUDY_KEY as LAG_STUDY;
use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study, fpha_lag_study_with};
use crate::common::oracle_tolerance::{UnscaledTolerance, assert_first_try};
use crate::common::power_guards::{D43_UNMET_SCALE_GUARDS, DualFamily, Guard, PowerEvidence};
use crate::common::row_families::{RowClass, classify_expanded_rows};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    build_setup_in_code_with_models, fresh_system_and_setup_with, stochastic_in_code,
    train_storing_trial_states, try_stored_trial_states,
};
use crate::{D19_DIR, D43_DIR};

const WIDENED_LAG_DEPTH: u32 = 4;
const BOUNDARY_INTERCEPT: f64 = 5.0;
const DEEP_LAG_SLOPE: f64 = 0.5;

pub fn dual_family(class: RowClass) -> Option<DualFamily> {
    match class {
        RowClass::Base(RowFamily::Fpha) => Some(DualFamily::Fpha),
        RowClass::Base(RowFamily::Evaporation) => Some(DualFamily::Evaporation),
        RowClass::Base(RowFamily::GenericConstraint) => Some(DualFamily::Generic),
        _ => None,
    }
}

fn offset(index: i32) -> usize {
    usize::try_from(index).expect("CSC and CSR offsets and indices are >= 0")
}

/// The cut rows by column: each column's `(k, value)` entries in increasing row `k`, the order a
/// row-by-row scan of the batch visits them.
struct CutColumns {
    starts: Vec<usize>,
    entries: Vec<(usize, f64)>,
}

impl CutColumns {
    fn new(cut_rows: &RowBatch, num_cols: usize) -> Self {
        let positions =
            |k: usize| offset(cut_rows.row_starts[k])..offset(cut_rows.row_starts[k + 1]);
        let mut starts = vec![0; num_cols + 1];
        for p in (0..cut_rows.num_rows).flat_map(positions) {
            starts[offset(cut_rows.col_indices[p]) + 1] += 1;
        }
        for col in 0..num_cols {
            starts[col + 1] += starts[col];
        }
        let mut next = starts.clone();
        let mut entries = vec![(0, 0.0); starts[num_cols]];
        for k in 0..cut_rows.num_rows {
            for p in positions(k) {
                let slot = &mut next[offset(cut_rows.col_indices[p])];
                entries[*slot] = (k, cut_rows.values[p]);
                *slot += 1;
            }
        }
        Self { starts, entries }
    }

    fn column(&self, col: usize) -> &[(usize, f64)] {
        &self.entries[self.starts[col]..self.starts[col + 1]]
    }
}

/// `(ρ_j, Σ|terms|)` of column `col` in unscaled units, from the scaled expanded LP's own
/// entries and the reported duals `dual`: `(c̃ − Σ Ã·π̃) / col_scale`.
fn reduced_cost_of(
    solve: &ExpandedSolve,
    cut_columns: &CutColumns,
    dual: &[f64],
    col: usize,
) -> (f64, f64) {
    let template = &solve.template;
    let mut rho = template.objective[col];
    let mut abs_sum = rho.abs();
    let base = column_entries(template, col).map(|(row, a)| a * dual[row]);
    let cuts = cut_columns
        .column(col)
        .iter()
        .map(|&(k, a)| a * dual[template.num_rows + k]);
    for product in base.chain(cuts) {
        rho -= product;
        abs_sum += product.abs();
    }
    let col_scale = template.col_scale[col];
    (rho / col_scale, abs_sum / col_scale)
}

fn solve_cold(label: &str, solve: &ExpandedSolve) -> LpSolution {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    load_expanded(&mut solver, solve);
    let before = solver.statistics();
    let outcome = solver.solve(None).map(|view| view.to_owned());
    let after = solver.statistics();
    assert!(
        outcome.is_ok(),
        "{label}: the cold expanded solve failed: {:?}",
        outcome.as_ref().err()
    );
    assert_first_try(label, &before, &after);
    outcome.expect("checked above")
}

/// The kernel's cut coefficients, in the extracted projection's slot space, and the absolute
/// magnitudes of the terms it sums, `g_abs[h]` for `g_h` and `w_abs[l * N + h]` for `w` of
/// lag `l`.
struct KernelCut {
    coefficients: Vec<f64>,
    g_abs: Vec<f64>,
    w_abs: Vec<f64>,
}

fn nonzero_cut_duals(solve: &ExpandedSolve, solution: &LpSolution) -> Vec<(usize, f64)> {
    solve
        .rows
        .slots
        .iter()
        .enumerate()
        .filter_map(|(k, &slot)| {
            let y = solution.dual[solve.template.num_rows + k];
            (y != 0.0).then_some((slot, y))
        })
        .collect()
}

/// `(g_abs, w_abs)` of a [`KernelCut`] for the duals `solution` of the expanded LP.
fn term_magnitudes(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    solution: &LpSolution,
) -> (Vec<f64>, Vec<f64>) {
    let training_ctx = compact.training_ctx();
    let space = compact.stage_state();
    let slots = ChainRuleSlots::compose(
        space,
        &training_ctx.cut_state_layouts[stage],
        &training_ctx.cut_state_layouts[stage - 1],
    );
    let resident_pool = &compact.fcf.pools[stage];
    let mut g_abs = coupled_dual_magnitudes(compact, expanded, stage, &solution.dual);
    let mut w_abs = vec![0.0; space.hydro_count * space.max_par_order.saturating_sub(1)];
    let (lag0, shifted) = (slots.resident_lag0(), slots.resident_shifted());
    for (slot, y) in nonzero_cut_duals(solve, solution) {
        let row = resident_pool.coefficient_row(slot);
        for (acc, &beta) in g_abs.iter_mut().zip(&row[lag0.clone()]) {
            *acc += (y * beta).abs();
        }
        for (acc, &beta) in w_abs.iter_mut().zip(&row[shifted.clone()]) {
            *acc += (y * beta).abs();
        }
    }
    (g_abs, w_abs)
}

/// The production compact extraction's cut, with the term magnitudes of the expanded LP's duals
/// at the compact vertex.
fn compact_cut(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    solution: &LpSolution,
) -> KernelCut {
    let (g_abs, w_abs) = term_magnitudes(compact, expanded, stage, solve, solution);
    KernelCut {
        coefficients: solve.compact_cut.coefficients.clone(),
        g_abs,
        w_abs,
    }
}

/// The kernel's cut, from the duals of the expanded LP itself.
fn kernel_cut(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    resident_pool: &CutPool,
    stage: usize,
    x_hat: &[f64],
    solve: &ExpandedSolve,
    solution: &LpSolution,
) -> KernelCut {
    let training_ctx = compact.training_ctx();
    let (resident, extracted) = (
        &training_ctx.cut_state_layouts[stage],
        &training_ctx.cut_state_layouts[stage - 1],
    );
    let col_scale = &solve.template.col_scale;
    let state_duals: Vec<f64> = extracted
        .incoming_pairs()
        .map(|(slot, _)| {
            let col = expanded
                .layout
                .incoming_state_col(extracted.global_state_index(slot));
            solution.reduced_costs[col] / col_scale[col]
        })
        .collect();
    let coupled = coupled_duals(compact, expanded, stage, &solution.dual);
    let cut_duals = nonzero_cut_duals(solve, solution);

    let space = compact.stage_state();
    let slots = ChainRuleSlots::compose(space, resident, extracted);
    let (g_abs, w_abs) = term_magnitudes(compact, expanded, stage, solve, solution);
    let mut coefficients = vec![0.0; extracted.n_slots()];
    per_opening_cut_into(
        &ChainRuleInputs {
            slots: &slots,
            resident_pool,
            extracted,
            par: training_ctx.stochastic.par(),
            stage,
        },
        &OpeningDuals {
            objective: solution.objective,
            state_duals: &state_duals,
            coupled_duals: &coupled,
            cut_duals: &cut_duals,
        },
        x_hat,
        &mut ChainRuleScratch::new(space.hydro_count, space.max_par_order),
        &mut coefficients,
    );
    KernelCut {
        coefficients,
        g_abs,
        w_abs,
    }
}

/// Checks the identity and the reported tie of every `(hydro, lag)` of one solve and returns
/// how many pairs it checked and whether some lag at or past the PAR order has a nonzero
/// `β_fold`.
fn check_lag_pairs(
    label: &str,
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    solve: &ExpandedSolve,
    solution: &LpSolution,
    kernel: &KernelCut,
    lag_dims: &[Option<(usize, usize)>],
) -> (usize, bool) {
    let layout = &expanded.layout;
    let space = compact.stage_state();
    let (n, depth) = (space.hydro_count, space.max_par_order);
    let extracted = &compact.training_ctx().cut_state_layouts[stage - 1];
    let cut_columns = CutColumns::new(&solve.cut_rows, solve.template.num_cols);
    let mut lag_slot = vec![None; n * depth];
    for j in 0..extracted.n_slots() {
        let d = extracted.global_state_index(CutSlot::new(j));
        if let Some((h, l)) = lag_dims[d.get()] {
            lag_slot[l * n + h] = Some(j);
        }
    }

    let par = compact.training_ctx().stochastic.par();
    let coupling = &compact.stage_ctx().inflow_coupling[stage];
    let cut_dual_count = solution.dual[solve.template.num_rows..]
        .iter()
        .filter(|&&y| y != 0.0)
        .count();
    let tolerance = UnscaledTolerance::from_active_profile();
    let unit_roundoff = f64::EPSILON / 2.0;
    // The identity is invariant under any shift of the z-row duals (the kernel eliminates z);
    // the shifted leg makes rho_z nonzero, so the sign of psi is observable.
    let mut shifted_dual = solution.dual.clone();
    for h in 0..n {
        shifted_dual[layout.z_row(HydroSys::new(h))] += 1.0;
    }
    let legs = [
        (solution.dual.as_slice(), ""),
        (shifted_dual.as_slice(), ", z-row duals shifted by 1"),
    ];
    let mut checked = 0;
    let mut deep_nonzero = false;
    for h in 0..n {
        let hydro = HydroSys::new(h);
        let z_costs =
            legs.map(|(duals, _)| reduced_cost_of(solve, &cut_columns, duals, layout.z_col(hydro)));
        // Longest rounding chain over both sides. Recomputation: two template scalings, one
        // product, `m_h + K` column-sum adds, one division by the column scale, one `ψ·ρ_z`
        // and the final add. Kernel: the two scalings and `m_h` adds of (Gᵀπ)_h, `K`
        // product-and-add steps into g, the `ψ·g` product and the final add of w, whose own
        // chain is `K` product-and-add steps.
        let order = coupling.hydro_entries(hydro).count() + 2 * cut_dual_count + 6;
        let order_u = f64::from(u32::try_from(order).expect("n fits u32")) * unit_roundoff;
        let gamma = order_u / (1.0 - order_u);
        for l in 0..depth {
            let col = layout.incoming_lag_col(hydro, l);
            let a_costs = legs.map(|(duals, _)| reduced_cost_of(solve, &cut_columns, duals, col));
            let beta = kernel.coefficients[lag_slot[l * n + h].expect("every lag has a slot")];
            let psi = par.psi_slice(stage, h).get(l).copied().unwrap_or(0.0);
            let kernel_abs =
                psi.abs() * kernel.g_abs[h] + kernel.w_abs.get(l * n + h).copied().unwrap_or(0.0);
            deep_nonzero |= l >= par.max_order() && beta != 0.0;
            for (i, (_, leg)) in legs.iter().enumerate() {
                let ((rho_a, abs_a), (rho_z, abs_z)) = (a_costs[i], z_costs[i]);
                let gap = (beta - (rho_a + psi * rho_z)).abs();
                let bound = 2.0 * gamma * (psi.abs() * abs_z + abs_a + kernel_abs);
                assert!(
                    gap <= bound,
                    "{label} stage {stage} hydro {h} lag {l}{leg}: beta_fold {beta:e}, \
                     rho_a {rho_a:e}, psi {psi:e}, rho_z {rho_z:e}, gap {gap:e} > bound \
                     {bound:e} (n = {order})"
                );
            }

            let [(rho_a, _), _] = a_costs;
            let col_scale = solve.template.col_scale[col];
            let reported = solution.reduced_costs[col] / col_scale;
            let tie = tolerance.reduced_cost(col_scale);
            assert!(
                (rho_a - reported).abs() <= tie,
                "{label} stage {stage} hydro {h} lag {l}: recomputed rho_a {rho_a:e} vs reported \
                 {reported:e}, tolerance {tie:e}"
            );
            checked += 1;
        }
    }
    (checked, deep_nonzero)
}

#[derive(Clone, Copy)]
enum Leg {
    Expanded,
    Compact,
}

struct SharedDualRun {
    evidence: PowerEvidence,
    skipped_stages: Vec<usize>,
    solves_per_stage: Vec<(usize, usize)>,
}

fn check_shared_dual_identity_on(
    label: &str,
    system: &System,
    compact: &StudySetup,
    leg: Leg,
) -> SharedDualRun {
    let expanded = expand_stage_templates(system, compact);
    let lag_dims = lag_dims(&expanded.layout);
    let state = compact.stage_state();
    let (n, depth) = (state.hydro_count, state.max_par_order);
    assert!(n >= 1, "{label}: no hydro, so no coupling dual can bind");
    let mut evidence = PowerEvidence::new(n, depth);
    let mut skipped_stages = Vec::new();
    let mut solves_per_stage = Vec::new();
    let mut deep_nonzero = false;
    let training_ctx = compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();

    for stage in 1..compact.fcf.pools.len() {
        if compact.fcf.active_cuts(stage).next().is_none() {
            continue;
        }
        let Some(states) = try_stored_trial_states(compact, NodePos(stage - 1)) else {
            skipped_stages.push(stage);
            continue;
        };
        let mut solves = 0;
        for opening in 0..tree.n_openings(stage) {
            let raw_noise = tree.opening(stage, opening);
            for x_hat in &states {
                let solve = assemble_expanded_solve(
                    compact,
                    &expanded,
                    &compact.fcf,
                    &CutRowSet::active(&compact.fcf, stage, stage - 1),
                    stage,
                    x_hat,
                    raw_noise,
                );
                let (solution, kernel) = match leg {
                    Leg::Expanded => {
                        let solution = solve_cold(label, &solve);
                        let kernel = kernel_cut(
                            compact,
                            &expanded,
                            &compact.fcf.pools[stage],
                            stage,
                            x_hat,
                            &solve,
                            &solution,
                        );
                        (solution, kernel)
                    }
                    Leg::Compact => {
                        let solution = solve_from_mapped_basis(label, &expanded, &solve);
                        let kernel = compact_cut(compact, &expanded, stage, &solve, &solution);
                        (solution, kernel)
                    }
                };
                let (checked, stage_deep_nonzero) = check_lag_pairs(
                    label, compact, &expanded, stage, &solve, &solution, &kernel, &lag_dims,
                );
                deep_nonzero |= stage_deep_nonzero;
                assert_eq!(
                    checked,
                    n * depth,
                    "{label} stage {stage}: every (hydro, lag) pair must be checked"
                );

                record_expanded_evidence(
                    &mut evidence,
                    compact,
                    &expanded,
                    stage,
                    &solve,
                    &solution.dual,
                );
                let classes = classify_expanded_rows(
                    compact,
                    &expanded.layout,
                    stage,
                    solve.template.num_rows + solve.cut_rows.num_rows,
                );
                for (row, &class) in classes.iter().enumerate() {
                    if let Some(family) = dual_family(class) {
                        evidence.record_family_dual(
                            family,
                            solve.template.row_scale[row] * solution.dual[row],
                        );
                    }
                }
                solves += 1;
            }
        }
        solves_per_stage.push((stage, solves));
    }
    let par_order = training_ctx.stochastic.par().max_order();
    assert!(
        depth <= par_order || deep_nonzero,
        "{label}: no lag at or past the PAR order {par_order} carries a nonzero beta_fold"
    );
    SharedDualRun {
        evidence,
        skipped_stages,
        solves_per_stage,
    }
}

fn check_shared_dual_identity(
    label: &str,
    system: &System,
    compact: &mut StudySetup,
) -> SharedDualRun {
    train_storing_trial_states(label, compact);
    check_shared_dual_identity_on(label, system, compact, Leg::Expanded)
}

const D19_UNMET_SCALE_GUARDS: &[(Guard, &str)] = &[(
    Guard::RescaledCoupledRowWithDual,
    "the coupled rows are water-balance rows, whose compact and expanded scales are bit-identical; only a generic-constraint inflow row is rescaled",
)];

const FPHA_LAG_STUDY_GUARDS: [Guard; 6] = [
    Guard::FamilyDual(DualFamily::Fpha),
    Guard::FamilyDual(DualFamily::Evaporation),
    Guard::FamilyDual(DualFamily::Generic),
    Guard::CouplingDualEveryHydro,
    Guard::BindingLagCut,
    Guard::BindingShiftedLagCut,
];

/// Asserts the skipped stages are the reduced producers' successors and a stage was checked, then
/// `guards` and each scale guard not in `unmet`, and that every guard of `unmet` is still unmet.
fn assert_run(
    label: &str,
    compact: &StudySetup,
    run: &SharedDualRun,
    guards: &[Guard],
    unmet: &[(Guard, &str)],
) {
    assert_eq!(
        run.skipped_stages,
        stages_whose_producer_drops_a_state_dimension(compact),
        "{label}: the skipped stages must be the stages whose producer pool drops a state dimension"
    );
    assert!(
        !run.solves_per_stage.is_empty(),
        "{label}: no stage was checked"
    );
    run.evidence.assert_scale_guards(label, guards, unmet);
}

#[test]
fn the_shared_dual_identity_holds_on_the_fpha_lag_study() {
    let (system, mut config, hydro_models) = fpha_lag_study();
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    let (system, _, _) = fpha_lag_study();

    let run = check_shared_dual_identity(LAG_STUDY, &system, &mut compact);
    assert!(
        run.skipped_stages.is_empty(),
        "{LAG_STUDY}: a stage's producer pool drops a state dimension"
    );
    assert_run(LAG_STUDY, &compact, &run, &FPHA_LAG_STUDY_GUARDS, &[]);
}

#[test]
fn the_compact_extraction_meets_the_shared_dual_identity_on_the_fpha_lag_study() {
    let (system, mut config, hydro_models) = fpha_lag_study();
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    let (system, _, _) = fpha_lag_study();

    train_storing_trial_states(LAG_STUDY, &mut compact);
    let run = check_shared_dual_identity_on(LAG_STUDY, &system, &compact, Leg::Compact);
    assert!(
        run.skipped_stages.is_empty(),
        "{LAG_STUDY}: a stage's producer pool drops a state dimension"
    );
    assert_run(LAG_STUDY, &compact, &run, &FPHA_LAG_STUDY_GUARDS, &[]);
}

#[test]
fn the_shared_dual_identity_holds_at_every_opening_of_a_three_opening_lag_study() {
    const OPENINGS: usize = 3;
    let options = FphaLagOptions {
        openings: OPENINGS,
        ..FphaLagOptions::default()
    };
    let (system, mut config, hydro_models) = fpha_lag_study_with(options);
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    let (system, _, _) = fpha_lag_study_with(options);

    let training_ctx = compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();
    for stage in 1..compact.fcf.pools.len() - 1 {
        assert_eq!(
            tree.n_openings(stage),
            OPENINGS,
            "stage {stage}: the study must have {OPENINGS} openings"
        );
        for first in 0..OPENINGS {
            for second in first + 1..OPENINGS {
                assert_ne!(
                    bits(tree.opening(stage, first)),
                    bits(tree.opening(stage, second)),
                    "stage {stage}: openings {first} and {second} must differ in bits"
                );
            }
        }
    }

    let label = "three-opening lag study";
    let run = check_shared_dual_identity(label, &system, &mut compact);
    assert!(
        run.skipped_stages.is_empty(),
        "{label}: a stage's producer pool drops a state dimension"
    );
    for &(stage, solves) in &run.solves_per_stage {
        let states = try_stored_trial_states(&compact, NodePos(stage - 1))
            .expect("a checked stage has states");
        assert_eq!(
            solves,
            states.len() * OPENINGS,
            "{label} stage {stage}: one solve per incoming state and opening"
        );
    }
    assert_run(label, &compact, &run, &FPHA_LAG_STUDY_GUARDS, &[]);
}

#[test]
fn the_shared_dual_identity_holds_on_d19() {
    let (system, mut compact) =
        fresh_system_and_setup_with(Path::new(D19_DIR), apply_oracle_config);
    let depth = compact.stage_state().max_par_order;
    println!("d19 lag depth: {depth}");

    let run = check_shared_dual_identity("d19", &system, &mut compact);
    assert!(
        run.skipped_stages.is_empty(),
        "d19: a stage's producer pool drops a state dimension"
    );
    let mut guards = vec![Guard::CouplingDualEveryHydro, Guard::BindingLagCut];
    if depth >= 2 {
        guards.push(Guard::BindingShiftedLagCut);
    }
    assert_run("d19", &compact, &run, &guards, D19_UNMET_SCALE_GUARDS);
}

#[test]
fn the_shared_dual_identity_holds_on_d43_outside_its_lag_dropping_stage() {
    let (system, mut compact) =
        fresh_system_and_setup_with(Path::new(D43_DIR), apply_oracle_config);

    let run = check_shared_dual_identity("d43", &system, &mut compact);
    assert!(
        !run.skipped_stages.is_empty(),
        "d43: no stage reads the reduced pool"
    );
    assert!(
        run.solves_per_stage.iter().any(|&(stage, _)| stage == 1),
        "d43: stage 1, whose resident pool is the reduced pool, was not checked"
    );
    assert_run(
        "d43",
        &compact,
        &run,
        &[Guard::CouplingDualEveryHydro],
        D43_UNMET_SCALE_GUARDS,
    );
}

/// The lag study with its lag state widened to `WIDENED_LAG_DEPTH` by a boundary requirement,
/// and a one-cut boundary whose slope on every lag at or past the PAR order is
/// `DEEP_LAG_SLOPE` injected into the compact form's terminal pool.
fn widened_lag_study(boundary_dir: &Path) -> (System, StudySetup) {
    let (system, mut config, hydro_models) = fpha_lag_study();
    apply_oracle_config(&mut config);
    let sentinel = Path::new("config.json");
    let training_source = config
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");
    let mut params = StudyParams::from_config(&config, Vec::new())
        .expect("StudyParams::from_config must succeed");
    params.boundary = BoundaryStateRequirements::present(WIDENED_LAG_DEPTH);
    let mut compact = StudySetup::from_broadcast_params(
        &system,
        stochastic_in_code(&system),
        params,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("the widened-lag study must build");

    let par_order = compact.training_ctx().stochastic.par().max_order();
    let coefficients: Vec<f64> = lag_dims(&expand_stage_templates(&system, &compact).layout)
        .iter()
        .map(|lag| match lag {
            Some((_, l)) if *l >= par_order => DEEP_LAG_SLOPE,
            _ => 0.0,
        })
        .collect();
    let state_dimension = u32::try_from(compact.fcf.state_dimension).expect("fits u32");
    assert_eq!(
        coefficients.len(),
        compact.fcf.state_dimension,
        "the boundary's slopes must cover the whole state"
    );
    let date = fixture_priced_date(ymd(2030, 1, 1), 0);
    write_synthetic_boundary(
        boundary_dir,
        state_dimension,
        BOUNDARY_INTERCEPT,
        &coefficients,
        date,
    );
    let boundary = load_boundary_cuts(&BoundaryLoadRequest::new(
        boundary_dir,
        date,
        state_dimension,
        &[],
        1.0,
    ))
    .expect("the boundary cut must load");
    inject_boundary_cuts(&mut compact, &boundary).expect("the boundary cut must inject");
    (system, compact)
}

#[test]
fn the_shared_dual_identity_holds_on_a_lag_state_deeper_than_the_par_order() {
    let boundary_dir = tempfile::tempdir().expect("tempdir");
    let (system, mut compact) = widened_lag_study(boundary_dir.path());
    let par_order = compact.training_ctx().stochastic.par().max_order();
    let depth = compact.stage_state().max_par_order;
    assert!(
        depth > par_order,
        "the lag state ({depth}) must be deeper than the PAR order ({par_order})"
    );

    let run = check_shared_dual_identity("widened lag", &system, &mut compact);
    assert!(
        run.skipped_stages.is_empty(),
        "widened lag: a stage's producer pool drops a state dimension"
    );
    assert_run(
        "widened lag",
        &compact,
        &run,
        &[
            Guard::CouplingDualEveryHydro,
            Guard::BindingLagCut,
            Guard::BindingShiftedLagCut,
        ],
        &[],
    );
}
