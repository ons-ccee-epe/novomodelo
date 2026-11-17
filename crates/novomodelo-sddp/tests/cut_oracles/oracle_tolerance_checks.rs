use std::panic::{AssertUnwindSafe, catch_unwind};

use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::capture_patched_node_template_with_inflow_noise;
use novomodelo_solver::{ActiveProfile, ActiveSolver, SolverInterface, SolverStatistics};

use crate::common::build_setup_in_code;
use crate::common::in_code_studies::{
    CHRONOLOGICAL_NOISE_INFLOW_STD_M3S, ChronologicalNoiseSpec, chronological_noise_study,
};
use crate::common::oracle_tolerance::{UnscaledTolerance, assert_first_try};

const CALLER: &str = "root stage oracle";

fn assert_close(quantity: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-15 * want.abs(),
        "{quantity}: expected {want}, got {got}"
    );
}

fn failure_message(before: &SolverStatistics, after: &SolverStatistics) -> Option<String> {
    let payload = catch_unwind(AssertUnwindSafe(|| {
        assert_first_try(CALLER, before, after);
    }))
    .err()?;
    Some(
        *payload
            .downcast::<String>()
            .expect("an assert message is a String"),
    )
}

#[test]
fn conversions_scale_by_the_row_or_column_they_bound() {
    let tolerance = UnscaledTolerance::new(1e-9, 1e-7);
    let (row_scale, col_scale) = (4.0, 0.5);
    assert_close("row residual", tolerance.row_residual(row_scale), 2.5e-10);
    assert_close("reduced cost", tolerance.reduced_cost(col_scale), 2e-7);
    assert_close("row dual", tolerance.row_dual(row_scale), 4e-7);
}

#[test]
fn from_active_profile_reads_the_backend_defaults() {
    let tolerance = UnscaledTolerance::from_active_profile();
    let profile = ActiveProfile::default();
    assert_eq!(
        tolerance.primal().to_bits(),
        profile.primal_feasibility_tolerance.to_bits()
    );
    assert_eq!(
        tolerance.dual().to_bits(),
        profile.dual_feasibility_tolerance.to_bits()
    );
}

#[test]
fn a_first_try_solve_passes() {
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec::default());
    let setup = build_setup_in_code(system, &config);
    let graph = &setup.inputs.node_graph;
    let root = (0..graph.nodes.len())
        .map(NodePos)
        .find(|&pos| graph.nodes[pos].stage == StageIdx(0))
        .expect("study must have a stage-0 node");
    let zero_draw = [0.0; CHRONOLOGICAL_NOISE_INFLOW_STD_M3S.len()];
    let template = capture_patched_node_template_with_inflow_noise(&setup, root, &zero_draw);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    solver.load_model(&template);

    let before = solver.statistics();
    solver.solve(None).expect("root stage LP must solve");
    let after = solver.statistics();

    assert_first_try(CALLER, &before, &after);
}

#[test]
fn a_retried_solve_fails() {
    let before = SolverStatistics {
        solve_count: 1,
        first_try_successes: 1,
        ..Default::default()
    };
    let after = SolverStatistics {
        solve_count: 2,
        first_try_successes: 1,
        retry_count: 3,
        ..Default::default()
    };
    let message = failure_message(&before, &after).expect("a retried solve must fail");
    assert!(
        message.contains(CALLER) && message.contains("retried"),
        "unexpected message: {message}"
    );
}

#[test]
fn no_new_solve_fails() {
    let snapshot = SolverStatistics::default();
    let message = failure_message(&snapshot, &snapshot).expect("no new solve must fail");
    assert!(
        message.contains(CALLER) && message.contains("no solve ran"),
        "unexpected message: {message}"
    );
}
