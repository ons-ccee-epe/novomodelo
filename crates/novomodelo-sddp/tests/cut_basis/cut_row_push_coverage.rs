use std::fmt::Debug;
use std::sync::mpsc;

use novomodelo_io::Config;
use novomodelo_io::config::{
    BackwardScheduler, InflowResidualDistribution, SelectionMethod, SimulationSelection,
    TrainingSelection,
};
use novomodelo_sddp::cut::SlotOrigin;
use novomodelo_sddp::indexer::{LagPointMode, StateDim};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::bound_recording::{
    AppendedCutRowBound, CutRowPushLog, CutRowPushRecord, CutRowPushRecorder,
};
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::external_distinct_fan_setup_heterogeneous_cut_state;
use novomodelo_sddp::test_support::node_opening_noise;
use novomodelo_sddp::test_support::pool_cut_state_dimensions;
use novomodelo_sddp::test_support::solve_sites::SolveSiteKind;
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
use novomodelo_sddp::{
    CutPool, SimulationScenarioResult, SolverProfiles, StudySetup, TrainingResult,
};
use novomodelo_solver::ActiveSolver;
use novomodelo_stochastic::{
    add_noise_term_batch, evaluate_lognormal_noise_term_batch, evaluate_par_noise_term_batch,
};

use crate::common::expanded_solve::{OracleLagPoint, realized_inflow, realized_inflow_with};
use crate::common::in_code_studies::{
    FphaLagOptions, SAME_PATH_LAG_SEED_M3S, WEEKLY_PARTIAL_SUM_INITIAL_STORAGE_HM3,
    fpha_lag_study_with, weekly_partial_sum_study,
};
use crate::common::lognormal_branches::{
    assert_the_stored_pins_reach_both_branches, conditional_means,
};
use crate::common::oracle_tolerance::rhs_tolerance;
use crate::common::stored_cut_oracles::{
    PartialSums, apply_oracle_config, apply_oracle_config_with_selection, replay_partial_sums,
};
use crate::common::{
    StubComm, build_setup_in_code_with_models, fresh_system_and_setup_with,
    inject_constant_terminal_boundary_cuts, lift_to_global_state, positive,
};

const THREADS: usize = 2;
const SIMULATION_SCENARIOS: u32 = 3;
const FIRST_SOLVING_ITERATION: u64 = 2;
const FAN_K: usize = 2;
const FAN_ITERATIONS: u32 = 3;
const BOUNDARY_CUTS: usize = FAN_ITERATIONS as usize + 2;
const BOUNDARY_INTERCEPT: f64 = 100.0;
// Largest window at which this run's lazy solves still add a cut in a selection round.
const SEED_WINDOW: u32 = 1;
const FORWARD: &str = "forward";
const BACKWARD: &str = "backward";
const LOWER_BOUND: &str = "lower_bound";

fn enable_simulation(config: &mut Config) {
    config.simulation.enabled = true;
    config.simulation.selection = Some(SimulationSelection::Sampled {
        num_scenarios: SIMULATION_SCENARIOS,
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fixture {
    LagStudy,
    Weekly,
    TerminalFan,
}

#[derive(Clone, Copy, PartialEq)]
struct RunSpec {
    fixture: Fixture,
    openings: usize,
    selection: TrainingSelection,
    dynamic_cut_selection: bool,
    seed_window: u32,
    backward_scheduler: BackwardScheduler,
    simulate: bool,
    distribution: InflowResidualDistribution,
    reach_the_fallback: bool,
}

const SAMPLED: RunSpec = RunSpec {
    fixture: Fixture::LagStudy,
    openings: 3,
    selection: TrainingSelection::sampled(1),
    dynamic_cut_selection: false,
    seed_window: SEED_WINDOW,
    backward_scheduler: BackwardScheduler::ByScenario {},
    simulate: true,
    distribution: InflowResidualDistribution::Normal,
    reach_the_fallback: false,
};

const SAMPLED_BY_NODE: RunSpec = RunSpec {
    backward_scheduler: BackwardScheduler::ByNode { block_size: None },
    simulate: false,
    ..SAMPLED
};

const ENUMERATED: RunSpec = RunSpec {
    openings: 1,
    selection: TrainingSelection::Enumerated {},
    simulate: false,
    ..SAMPLED
};

const TERMINAL_FAN: RunSpec = RunSpec {
    fixture: Fixture::TerminalFan,
    ..ENUMERATED
};

const DYNAMIC: RunSpec = RunSpec {
    dynamic_cut_selection: true,
    ..SAMPLED
};

const WEEKLY_SAMPLED: RunSpec = RunSpec {
    fixture: Fixture::Weekly,
    selection: TrainingSelection::sampled(4),
    ..SAMPLED
};

const WEEKLY_BY_NODE: RunSpec = RunSpec {
    backward_scheduler: BackwardScheduler::ByNode { block_size: None },
    simulate: false,
    ..WEEKLY_SAMPLED
};

const WEEKLY_ENUMERATED: RunSpec = RunSpec {
    openings: 1,
    selection: TrainingSelection::Enumerated {},
    simulate: false,
    ..WEEKLY_SAMPLED
};

const WEEKLY_DYNAMIC: RunSpec = RunSpec {
    dynamic_cut_selection: true,
    seed_window: 0,
    ..WEEKLY_SAMPLED
};

const LOGNORMAL_SAMPLED: RunSpec = RunSpec {
    selection: TrainingSelection::sampled(4),
    distribution: InflowResidualDistribution::Lognormal3p,
    reach_the_fallback: true,
    ..SAMPLED
};

const LOGNORMAL_BY_NODE: RunSpec = RunSpec {
    backward_scheduler: BackwardScheduler::ByNode { block_size: None },
    simulate: false,
    ..LOGNORMAL_SAMPLED
};

const LOGNORMAL_DYNAMIC: RunSpec = RunSpec {
    dynamic_cut_selection: true,
    seed_window: 0,
    ..LOGNORMAL_SAMPLED
};

const LOGNORMAL_WEEKLY_SAMPLED: RunSpec = RunSpec {
    distribution: InflowResidualDistribution::Lognormal3p,
    ..WEEKLY_SAMPLED
};

const LOGNORMAL_WEEKLY_BY_NODE: RunSpec = RunSpec {
    distribution: InflowResidualDistribution::Lognormal3p,
    ..WEEKLY_BY_NODE
};

const LOGNORMAL_WEEKLY_DYNAMIC: RunSpec = RunSpec {
    distribution: InflowResidualDistribution::Lognormal3p,
    ..WEEKLY_DYNAMIC
};

#[derive(Clone, Copy)]
enum Witness {
    PhaseSolves(&'static str),
    WorkerPhaseSolves(&'static str),
    LowerBoundSolves,
    SimulationCutRows,
    BoundaryRows,
    SelectionRound,
}

fn recorder_config(kind: SolveSiteKind) -> (RunSpec, Witness) {
    match kind {
        SolveSiteKind::SampledForward => (SAMPLED, Witness::WorkerPhaseSolves(FORWARD)),
        SolveSiteKind::BackwardByScenario => (SAMPLED, Witness::PhaseSolves(BACKWARD)),
        SolveSiteKind::LowerBound => (SAMPLED, Witness::LowerBoundSolves),
        SolveSiteKind::Simulation => (SAMPLED, Witness::SimulationCutRows),
        SolveSiteKind::BackwardByNode => (SAMPLED_BY_NODE, Witness::PhaseSolves(BACKWARD)),
        SolveSiteKind::EnumeratedForward => (ENUMERATED, Witness::WorkerPhaseSolves(FORWARD)),
        SolveSiteKind::BackwardReplicated => (ENUMERATED, Witness::PhaseSolves(BACKWARD)),
        SolveSiteKind::FusedTerminal => (TERMINAL_FAN, Witness::BoundaryRows),
        SolveSiteKind::DynamicCutSelection => (DYNAMIC, Witness::SelectionRound),
    }
}

/// `None` where the setup refuses the site's configuration under the option: enumerated training
/// selection and external inflow.
fn lognormal_recorder_config(kind: SolveSiteKind) -> Option<(RunSpec, Witness)> {
    match kind {
        SolveSiteKind::SampledForward => {
            Some((LOGNORMAL_SAMPLED, Witness::WorkerPhaseSolves(FORWARD)))
        }
        SolveSiteKind::BackwardByScenario => {
            Some((LOGNORMAL_SAMPLED, Witness::PhaseSolves(BACKWARD)))
        }
        SolveSiteKind::LowerBound => Some((LOGNORMAL_SAMPLED, Witness::LowerBoundSolves)),
        SolveSiteKind::Simulation => Some((LOGNORMAL_SAMPLED, Witness::SimulationCutRows)),
        SolveSiteKind::BackwardByNode => Some((LOGNORMAL_BY_NODE, Witness::PhaseSolves(BACKWARD))),
        SolveSiteKind::DynamicCutSelection => Some((LOGNORMAL_DYNAMIC, Witness::SelectionRound)),
        SolveSiteKind::EnumeratedForward
        | SolveSiteKind::BackwardReplicated
        | SolveSiteKind::FusedTerminal => None,
    }
}

struct RecordedRun {
    setup: StudySetup,
    result: TrainingResult,
    worker: Vec<CutRowPushRecord>,
    lower_bound: Vec<CutRowPushRecord>,
    simulation: Vec<CutRowPushRecord>,
    simulated: Vec<SimulationScenarioResult>,
    partial_sums: Option<PartialSums>,
}

impl RecordedRun {
    fn logs(&self) -> [(&'static str, &[CutRowPushRecord]); 3] {
        [
            ("worker", &self.worker),
            ("lower bound", &self.lower_bound),
            ("simulation", &self.simulation),
        ]
    }
}

fn configure(config: &mut Config, spec: &RunSpec) {
    if spec.dynamic_cut_selection {
        apply_oracle_config_with_selection(
            config,
            SelectionMethod::Dynamic {
                start_iteration: positive(1),
                seed_window: spec.seed_window,
                candidate_recency: None,
                max_added_per_round: positive(10),
                violation_tolerance: None,
            },
        );
    } else {
        apply_oracle_config(config);
    }
    config.training.selection = Some(spec.selection);
    config.modeling.inflow_residual.distribution = spec.distribution;
    if spec.simulate {
        enable_simulation(config);
    }
}

fn build_setup(spec: &RunSpec) -> StudySetup {
    match spec.fixture {
        Fixture::LagStudy => {
            let (system, mut config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
                openings: spec.openings,
                reach_the_fallback: spec.reach_the_fallback,
                ..FphaLagOptions::default()
            });
            configure(&mut config, spec);
            let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);
            if spec.distribution == InflowResidualDistribution::Lognormal3p {
                setup.set_export_states(true);
            }
            setup
        }
        Fixture::Weekly => {
            let (system, mut config, hydro_models) = weekly_partial_sum_study(spec.openings);
            configure(&mut config, spec);
            let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);
            setup.set_export_states(true);
            setup
        }
        Fixture::TerminalFan => {
            assert!(
                spec.selection == TrainingSelection::Enumerated {},
                "the terminal fan runs enumerated"
            );
            let mut setup =
                external_distinct_fan_setup_heterogeneous_cut_state(FAN_K, FAN_ITERATIONS);
            setup.set_export_states(true);
            inject_constant_terminal_boundary_cuts(&mut setup, &boundary_intercepts(), true);
            setup
        }
    }
}

fn boundary_intercepts() -> Vec<f64> {
    (0..BOUNDARY_CUTS)
        .map(|cut| BOUNDARY_INTERCEPT + f64::from(u32::try_from(cut).expect("few cuts")))
        .collect()
}

fn incoming_storage_column(setup: &StudySetup) -> usize {
    setup
        .stage_state()
        .state_to_lp_incoming_column(StateDim::new(0))
        .get()
}

fn record_run(
    label: &str,
    mut setup: StudySetup,
    backward_scheduler: BackwardScheduler,
    simulate: bool,
    appended_bound: AppendedCutRowBound,
    pinned_columns: &[usize],
) -> RecordedRun {
    let theta = setup.training_ctx().state.theta;
    let lower_bound_log = CutRowPushLog::new();
    let worker_log = CutRowPushLog::new();
    let simulation_log = CutRowPushLog::new();
    let recording = |log: &CutRowPushLog, appended_bound| {
        ActiveSolver::new().map(|solver| {
            CutRowPushRecorder::new(solver, theta, log.clone(), appended_bound)
                .with_pinned_columns(pinned_columns.to_vec())
        })
    };

    let mut solver = recording(&lower_bound_log, AppendedCutRowBound::Patched)
        .expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train_with_solver_profiles(
            &mut solver,
            &StubComm,
            THREADS,
            || recording(&worker_log, appended_bound),
            SolverProfiles {
                backward_scheduler,
                ..SolverProfiles::default()
            },
        )
        .unwrap_or_else(|error| panic!("{label}: training must return Ok: {error:?}"));
    assert!(
        outcome.error.is_none(),
        "{label}: training must not error: {:?}",
        outcome.error
    );

    let mut simulated = Vec::new();
    if simulate {
        let mut pool = setup
            .create_workspace_pool(&StubComm, THREADS, || {
                recording(&simulation_log, appended_bound)
            })
            .unwrap_or_else(|error| {
                panic!("{label}: create_workspace_pool must succeed: {error:?}")
            });
        let capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(capacity);
        let drain = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
        setup
            .simulate(
                &mut pool.workspaces,
                &StubComm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .unwrap_or_else(|error| panic!("{label}: simulate must return Ok: {error:?}"));
        drop(result_tx);
        simulated = drain.join().expect("drain thread must not panic");
    }

    RecordedRun {
        setup,
        result: outcome.result,
        worker: worker_log.records(),
        lower_bound: lower_bound_log.records(),
        simulation: simulation_log.records(),
        simulated,
        partial_sums: None,
    }
}

fn run_recorded(spec: &RunSpec) -> RecordedRun {
    let label = format!("{:?}", spec.fixture);
    let appended_bound = if spec.dynamic_cut_selection {
        AppendedCutRowBound::Final
    } else {
        AppendedCutRowBound::Patched
    };
    let setup = build_setup(spec);
    let partial_sums = (spec.fixture == Fixture::Weekly
        && matches!(spec.selection, TrainingSelection::Sampled { .. }))
    .then(|| replay_partial_sums(&setup));
    let pinned_columns = [incoming_storage_column(&setup)];
    let mut run = record_run(
        &label,
        setup,
        spec.backward_scheduler,
        spec.simulate,
        appended_bound,
        &pinned_columns,
    );
    run.partial_sums = partial_sums;
    run
}

fn has_cut_rows(record: &CutRowPushRecord) -> bool {
    record.frozen_rows + record.appended_rows >= 1
}

fn assert_phase_solves(kind: SolveSiteKind, run: &RecordedRun, phase: &str) {
    let solving = run
        .result
        .solver_stats_log
        .iter()
        .filter(|entry| {
            entry.phase == phase
                && entry.iteration >= FIRST_SOLVING_ITERATION
                && entry.delta.lp_solves > 0
        })
        .count();
    assert!(
        solving > 0,
        "{kind:?}: no {phase} stats entry at iteration >= {FIRST_SOLVING_ITERATION} solved an LP"
    );
}

fn assert_witness(kind: SolveSiteKind, witness: Witness, run: &RecordedRun) {
    match witness {
        Witness::PhaseSolves(phase) => assert_phase_solves(kind, run, phase),
        Witness::WorkerPhaseSolves(phase) => {
            assert_phase_solves(kind, run, phase);
            assert!(
                run.worker.iter().any(has_cut_rows),
                "{kind:?}: no worker record has cut rows"
            );
        }
        Witness::LowerBoundSolves => {
            assert_phase_solves(kind, run, LOWER_BOUND);
            assert!(
                run.lower_bound
                    .iter()
                    .any(|record| record.appended_rows >= 1),
                "{kind:?}: no lower bound record has an appended row"
            );
        }
        Witness::SimulationCutRows => assert!(
            run.simulation.iter().any(has_cut_rows),
            "{kind:?}: no simulation record has cut rows"
        ),
        Witness::BoundaryRows => assert_boundary_rows(kind, run),
        Witness::SelectionRound => assert!(
            run.worker
                .iter()
                .any(|record| !record.structural_window && record.added_in_window > 0),
            "{kind:?}: no worker record is a selection round"
        ),
    }
}

fn fan_pools(setup: &StudySetup) -> (usize, usize) {
    let graph = &setup.inputs.node_graph;
    let leaf_pool = graph
        .terminal_pool(setup.num_stages())
        .expect("the fan has a terminal pool");
    let root_pool = graph
        .nodes
        .iter()
        .find(|node| node.stage == StageIdx(0))
        .expect("the fan has a stage-0 node")
        .pool_id;
    (leaf_pool, root_pool)
}

fn assert_boundary_rows(kind: SolveSiteKind, run: &RecordedRun) {
    let setup = &run.setup;
    let (leaf_pool, root_pool) = fan_pools(setup);
    for (pool_id, pool) in setup.fcf.pools.iter().enumerate() {
        assert!(
            pool_id == leaf_pool || pool.populated() < BOUNDARY_CUTS,
            "{kind:?}: pool {pool_id} holds {} populated slots, so {BOUNDARY_CUTS} rows do not \
             identify the terminal pool",
            pool.populated()
        );
    }
    let dimensions = pool_cut_state_dimensions(setup);
    assert!(
        dimensions[leaf_pool] != dimensions[root_pool],
        "{kind:?}: the leaf and root pools project the same {} slots",
        dimensions[leaf_pool]
    );
    assert!(
        run.worker
            .iter()
            .any(|record| record.frozen_rows + record.appended_rows == BOUNDARY_CUTS),
        "{kind:?}: no worker record has the {BOUNDARY_CUTS} boundary rows"
    );
}

fn assert_no_stale_rows(label: &impl Debug, run: &RecordedRun) {
    for (log, records) in run.logs() {
        for (index, record) in records.iter().enumerate() {
            assert!(
                record.stale_rows == 0,
                "{label:?}: {log} log record {index} has stale cut rows: {record:?}"
            );
        }
    }
}

const CLOSING_STAGE: usize = 3;
const OPEN_WEEKS: usize = 3;
const WEEK_HOURS: f64 = 168.0;
const JANUARY_HOURS: f64 = 744.0;
const CLOSING_STAGE_HOURS: f64 = 240.0;
const STATE_MATCH_TOLERANCE: f64 = 1.0e-9;
const SLOPE_FLOOR: f64 = 1.0e-9;

enum PartialSumArm {
    Weekly(RunSpec, Witness),
    Fan,
}

use PartialSumArm::{Fan, Weekly};

fn partial_sum_arm(kind: SolveSiteKind) -> PartialSumArm {
    match kind {
        SolveSiteKind::SampledForward => {
            Weekly(WEEKLY_SAMPLED, Witness::WorkerPhaseSolves(FORWARD))
        }
        SolveSiteKind::BackwardByScenario => Weekly(WEEKLY_SAMPLED, Witness::PhaseSolves(BACKWARD)),
        SolveSiteKind::LowerBound => Weekly(WEEKLY_SAMPLED, Witness::LowerBoundSolves),
        SolveSiteKind::Simulation => Weekly(WEEKLY_SAMPLED, Witness::SimulationCutRows),
        SolveSiteKind::BackwardByNode => Weekly(WEEKLY_BY_NODE, Witness::PhaseSolves(BACKWARD)),
        SolveSiteKind::EnumeratedForward => {
            Weekly(WEEKLY_ENUMERATED, Witness::WorkerPhaseSolves(FORWARD))
        }
        SolveSiteKind::BackwardReplicated => {
            Weekly(WEEKLY_ENUMERATED, Witness::PhaseSolves(BACKWARD))
        }
        SolveSiteKind::DynamicCutSelection => Weekly(WEEKLY_DYNAMIC, Witness::SelectionRound),
        SolveSiteKind::FusedTerminal => Fan,
    }
}

/// `None` where the setup refuses the site's configuration under the option: enumerated training
/// selection and external inflow.
fn lognormal_partial_sum_arm(kind: SolveSiteKind) -> Option<(RunSpec, Witness)> {
    match kind {
        SolveSiteKind::SampledForward => Some((
            LOGNORMAL_WEEKLY_SAMPLED,
            Witness::WorkerPhaseSolves(FORWARD),
        )),
        SolveSiteKind::BackwardByScenario => {
            Some((LOGNORMAL_WEEKLY_SAMPLED, Witness::PhaseSolves(BACKWARD)))
        }
        SolveSiteKind::LowerBound => Some((LOGNORMAL_WEEKLY_SAMPLED, Witness::LowerBoundSolves)),
        SolveSiteKind::Simulation => Some((LOGNORMAL_WEEKLY_SAMPLED, Witness::SimulationCutRows)),
        SolveSiteKind::BackwardByNode => {
            Some((LOGNORMAL_WEEKLY_BY_NODE, Witness::PhaseSolves(BACKWARD)))
        }
        SolveSiteKind::DynamicCutSelection => {
            Some((LOGNORMAL_WEEKLY_DYNAMIC, Witness::SelectionRound))
        }
        SolveSiteKind::EnumeratedForward
        | SolveSiteKind::BackwardReplicated
        | SolveSiteKind::FusedTerminal => None,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= STATE_MATCH_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

fn written_slots(pool: &CutPool) -> impl Iterator<Item = usize> + '_ {
    (0..pool.populated()).filter(|&slot| !matches!(pool.role(slot).origin, SlotOrigin::Vacant))
}

/// Whether `pushed` is a sub-multiset of `values`, each element matched once within `tolerance`.
fn is_sub_multiset(pushed: &[f64], values: &[f64], tolerance: f64) -> bool {
    let mut free = values.to_vec();
    pushed.iter().all(|&value| {
        free.iter()
            .position(|&candidate| (candidate - value).abs() <= tolerance)
            .map(|index| free.swap_remove(index))
            .is_some()
    })
}

/// One solve's own inputs: the state it pins, the realized inflows it can draw and the partial
/// sum of the open lag period it reads.
struct Visit {
    stage: usize,
    storage: f64,
    lag: f64,
    partial_sum: f64,
    inflows: Vec<f64>,
}

struct StageOracle {
    point: OracleLagPoint,
    cuts: Vec<(f64, f64)>,
}

impl Visit {
    fn lag_point(&self, point: &OracleLagPoint, inflow: f64) -> f64 {
        point.lag0(0, &[self.partial_sum], inflow, self.lag)
    }

    fn lag_points(&self, point: &OracleLagPoint) -> Vec<f64> {
        let mut points: Vec<f64> = self
            .inflows
            .iter()
            .map(|&inflow| self.lag_point(point, inflow))
            .collect();
        points.sort_by(f64::total_cmp);
        points.dedup_by(|a, b| close(*a, *b));
        points
    }

    fn magnitude(&self) -> f64 {
        self.inflows
            .iter()
            .fold(self.lag.abs().max(self.partial_sum.abs()), |top, z| {
                top.max(z.abs())
            })
    }

    fn shares_point_inputs(&self, other: &Self) -> bool {
        self.stage == other.stage
            && close(self.partial_sum, other.partial_sum)
            && close(self.lag, other.lag)
    }
}

fn cut_values(cuts: &[(f64, f64)], point: f64) -> Vec<f64> {
    cuts.iter()
        .map(|&(intercept, slope)| intercept + slope * point)
        .collect()
}

fn tolerance(cuts: &[(f64, f64)], magnitude: f64) -> f64 {
    let top = |select: fn(&(f64, f64)) -> f64| cuts.iter().map(select).fold(0.0_f64, f64::max);
    rhs_tolerance(top(|cut| cut.0.abs()) + top(|cut| cut.1.abs()) * 2.0 * magnitude)
}

fn stage_cuts(setup: &StudySetup, stage: usize) -> Vec<(f64, f64)> {
    let pool = &setup.fcf.pools[stage];
    let lag_slots = setup.training_ctx().cut_state_layouts[stage]
        .lag_point()
        .full();
    assert_eq!(lag_slots.len(), 1, "pool {stage} holds one lag slot");
    written_slots(pool)
        .map(|slot| {
            (
                pool.intercept(slot),
                pool.coefficient_row(slot)[lag_slots.start],
            )
        })
        .collect()
}

fn stage_oracles(setup: &StudySetup) -> Vec<StageOracle> {
    let training_ctx = setup.training_ctx();
    let state = training_ctx.state;
    assert_eq!(
        (state.hydro_count, state.max_par_order),
        (1, 1),
        "one hydro with one lag"
    );
    (0..=CLOSING_STAGE)
        .map(|stage| {
            assert_eq!(
                training_ctx.node_graph.pool_stage[stage].0, stage,
                "pool {stage} is stage {stage}'s"
            );
            let point = OracleLagPoint::of(setup, stage);
            match &point {
                OracleLagPoint::Carry if stage < CLOSING_STAGE => {}
                OracleLagPoint::Closing {
                    own_weight,
                    period_weight,
                } if stage == CLOSING_STAGE => assert!(
                    close(*own_weight, CLOSING_STAGE_HOURS / JANUARY_HOURS)
                        && close(period_weight[0], 1.0),
                    "stage {stage} closes January at w = {own_weight} over {}, derived \
                     {CLOSING_STAGE_HOURS}/{JANUARY_HOURS} over 1",
                    period_weight[0]
                ),
                other => panic!("stage {stage} has the wrong lag point {other:?}"),
            }
            StageOracle {
                point,
                cuts: stage_cuts(setup, stage),
            }
        })
        .collect()
}

fn weekly_inflows(setup: &StudySetup, stage: usize, lag: f64) -> Vec<f64> {
    let openings = setup
        .training_ctx()
        .stochastic
        .tree_view()
        .n_openings(stage);
    (0..openings)
        .map(|opening| realized_inflow(setup, stage, opening, &[lag]))
        .collect()
}

/// derived: every January sum a path of the opening tree reaches at the closing stage, the weeks'
/// inflows read at the carried lag.
fn independent_partial_sums(setup: &StudySetup) -> Vec<f64> {
    let mut sums = vec![0.0];
    for stage in 0..OPEN_WEEKS {
        let inflows = weekly_inflows(setup, stage, SAME_PATH_LAG_SEED_M3S);
        sums = sums
            .iter()
            .flat_map(|sum| {
                inflows
                    .iter()
                    .map(move |inflow| sum + WEEK_HOURS / JANUARY_HOURS * inflow)
            })
            .collect();
    }
    sums.sort_by(f64::total_cmp);
    sums.dedup_by(|a, b| close(*a, *b));
    sums
}

fn training_visits(run: &RecordedRun, oracles: &[StageOracle]) -> Vec<Visit> {
    let setup = &run.setup;
    let training_ctx = setup.training_ctx();
    let lag_dim = state_lag_dims(training_ctx.state).start;
    let zeros = vec![0.0; training_ctx.state.n_state];
    let independent = independent_partial_sums(setup);
    let mut visits = vec![Visit {
        stage: 0,
        storage: WEEKLY_PARTIAL_SUM_INITIAL_STORAGE_HM3,
        lag: SAME_PATH_LAG_SEED_M3S,
        partial_sum: 0.0,
        inflows: weekly_inflows(setup, 0, SAME_PATH_LAG_SEED_M3S),
    }];
    for (stage, oracle) in oracles.iter().enumerate().skip(1) {
        let pool = &setup.fcf.pools[stage - 1];
        let layout = &training_ctx.cut_state_layouts[stage - 1];
        for slot in written_slots(pool) {
            let row = pool
                .trial_state(slot)
                .expect("every written slot stores its trial state");
            let state = lift_to_global_state(layout, row, &zeros);
            let lag = state[lag_dim];
            assert!(
                close(lag, SAME_PATH_LAG_SEED_M3S),
                "stage {stage}: the lag carried through the open weeks is {lag}, not the seed"
            );
            let partial_sum = match (&oracle.point, &run.partial_sums) {
                (OracleLagPoint::Closing { .. }, Some(sums)) => {
                    let sum = sums.of_cut(pool, slot, stage)[0];
                    assert!(
                        independent.iter().any(|&path| close(path, sum)),
                        "the replayed partial sum {sum} is no January path's {independent:?}"
                    );
                    sum
                }
                (OracleLagPoint::Closing { .. }, None) => {
                    assert_eq!(independent.len(), 1, "an enumerated chain walks one path");
                    independent[0]
                }
                _ => 0.0,
            };
            visits.push(Visit {
                stage,
                storage: state[StateDim::new(0).get()],
                lag,
                partial_sum,
                inflows: weekly_inflows(setup, stage, lag),
            });
        }
    }
    visits
}

fn simulation_visits(simulated: &[SimulationScenarioResult]) -> Vec<Visit> {
    let mut visits = Vec::new();
    for scenario in simulated {
        assert!(
            scenario.stages.len() > CLOSING_STAGE,
            "scenario {} holds no stage detail",
            scenario.scenario_id
        );
        let inflows: Vec<f64> = scenario
            .stages
            .iter()
            .map(|stage| {
                assert_eq!(stage.hydros.len(), 1, "one hydro over one block");
                stage.hydros[0].incremental_inflow_m3s
            })
            .collect();
        let january_sum: f64 = inflows[..OPEN_WEEKS]
            .iter()
            .map(|inflow| WEEK_HOURS / JANUARY_HOURS * inflow)
            .sum();
        for (stage, result) in scenario.stages.iter().enumerate().take(CLOSING_STAGE + 1) {
            visits.push(Visit {
                stage,
                storage: result.hydros[0].storage_initial_hm3,
                lag: SAME_PATH_LAG_SEED_M3S,
                partial_sum: if stage == CLOSING_STAGE {
                    january_sum
                } else {
                    0.0
                },
                inflows: vec![inflows[stage]],
            });
        }
    }
    visits
}

/// Matches each record to the visits whose incoming state its solve pinned, and checks its cut
/// rows against `α + β·Φ` at that visit's own point; returns each matched record's stage and
/// partial sum. Visits that share a state share their stage, lag and partial sum and differ in
/// the inflow they drew, so the match never depends on the order solves ran in.
fn assert_pushes_at_their_own_visit(
    label: &str,
    records: &[CutRowPushRecord],
    visits: &[Visit],
    oracles: &[StageOracle],
) -> Vec<(usize, f64)> {
    let mut matched = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if record.cut_row_rhs.is_empty() {
            continue;
        }
        let pinned = record.pinned[0];
        let mut own = visits.iter().filter(|visit| close(visit.storage, pinned));
        let first = own.next().unwrap_or_else(|| {
            panic!("{label}: record {index} pins storage {pinned}, the incoming state of no visit")
        });
        let mut visit = Visit {
            inflows: first.inflows.clone(),
            ..*first
        };
        for other in own {
            assert!(
                first.shares_point_inputs(other),
                "{label}: record {index} pins storage {pinned}, which visits at other stages or \
                 partial sums share, so the solve is not told apart by its state"
            );
            visit.inflows.extend_from_slice(&other.inflows);
        }
        let oracle = &oracles[visit.stage];
        let tolerance = tolerance(&oracle.cuts, visit.magnitude());
        let fits: Vec<f64> = visit
            .lag_points(&oracle.point)
            .into_iter()
            .filter(|&point| {
                is_sub_multiset(
                    &record.cut_row_rhs,
                    &cut_values(&oracle.cuts, point),
                    tolerance,
                )
            })
            .collect();
        assert_eq!(
            fits.len(),
            1,
            "{label}: record {index} at stage {} (partial sum {}) pushes {:?}, which is α + β·Φ \
             at {} of its visit's points {:?} (tolerance {tolerance:e}); the shifted alias \
             pushes {:?}",
            visit.stage,
            visit.partial_sum,
            record.cut_row_rhs,
            fits.len(),
            visit.lag_points(&oracle.point),
            visit
                .inflows
                .iter()
                .map(|&inflow| cut_values(&oracle.cuts, inflow))
                .collect::<Vec<_>>()
        );
        matched.push((visit.stage, visit.partial_sum));
    }
    matched
}

fn assert_alias_is_separated(label: &str, visits: &[Visit], oracles: &[StageOracle]) {
    for visit in visits {
        let oracle = &oracles[visit.stage];
        let tolerance = tolerance(&oracle.cuts, visit.magnitude());
        for &inflow in &visit.inflows {
            let wanted = cut_values(&oracle.cuts, visit.lag_point(&oracle.point, inflow));
            for aliased in cut_values(&oracle.cuts, inflow) {
                for &want in &wanted {
                    assert!(
                        (aliased - want).abs() > 2.0 * tolerance,
                        "{label}: the shifted alias {aliased} at stage {} is within the \
                         tolerance of α + β·Φ = {want}, so the check could hold vacuously",
                        visit.stage
                    );
                }
            }
        }
    }
}

fn assert_partial_sums(label: &str, multi_path: bool, sums: &[f64], oracles: &[StageOracle]) {
    let mut distinct = sums.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup_by(|a, b| close(*a, *b));
    if !multi_path {
        assert_eq!(
            distinct.len(),
            1,
            "{label}: a single-path run reads one partial sum, got {distinct:?}"
        );
        return;
    }
    assert!(
        distinct.len() >= 2,
        "{label}: every closing-stage solve read the partial sum {distinct:?}, so no trial \
         told its own sum from another's"
    );
    let cuts = &oracles[CLOSING_STAGE].cuts;
    let gap = distinct
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min);
    let slope = cuts
        .iter()
        .map(|cut| cut.1.abs())
        .fold(f64::INFINITY, f64::min);
    let magnitude = distinct.iter().fold(0.0_f64, |top, sum| top.max(sum.abs()));
    assert!(
        slope * gap > 2.0 * tolerance(cuts, magnitude),
        "{label}: partial sums {gap} apart move a cut row by {} at its smallest slope, within \
         the tolerance",
        slope * gap
    );
}

fn assert_weekly_pushes(kinds: &[SolveSiteKind], spec: &RunSpec, run: &RecordedRun) {
    let oracles = stage_oracles(&run.setup);
    for (stage, oracle) in oracles.iter().enumerate() {
        for &(_, slope) in &oracle.cuts {
            assert!(
                slope.abs() > SLOPE_FLOOR,
                "{kinds:?}: a stage {stage} cut has lag slope {slope:e}, so its lag term could \
                 hold vacuously"
            );
        }
    }
    let training = training_visits(run, &oracles);
    let simulated = simulation_visits(&run.simulated);
    assert_alias_is_separated(&format!("{kinds:?}"), &training, &oracles);
    assert_alias_is_separated(&format!("{kinds:?} simulation"), &simulated, &oracles);

    let mut logs = vec![
        ("worker", &run.worker, &training, CLOSING_STAGE),
        ("lower bound", &run.lower_bound, &training, 0),
    ];
    if spec.simulate {
        logs.push(("simulation", &run.simulation, &simulated, CLOSING_STAGE));
    }
    for (log, records, visits, required_stage) in logs {
        let label = format!("{kinds:?} {log}");
        let matched = assert_pushes_at_their_own_visit(&label, records, visits, &oracles);
        let sums: Vec<f64> = matched
            .iter()
            .filter(|&&(stage, _)| stage == required_stage)
            .map(|&(_, sum)| sum)
            .collect();
        assert!(
            !sums.is_empty(),
            "{label}: no record with cut rows is a stage {required_stage} solve"
        );
        if required_stage == CLOSING_STAGE {
            assert_partial_sums(&label, spec.openings > 1, &sums, &oracles);
        }
    }
}

fn assert_fan_boundary_values(kinds: &[SolveSiteKind], run: &RecordedRun) {
    let setup = &run.setup;
    let (_, root_pool) = fan_pools(setup);
    let root = &setup.fcf.pools[root_pool];
    let layout = &setup.training_ctx().cut_state_layouts[root_pool];
    let zeros = vec![0.0; setup.training_ctx().state.n_state];
    let root_storages: Vec<f64> = written_slots(root)
        .map(|slot| {
            let row = root
                .trial_state(slot)
                .expect("the root pool stores its states");
            lift_to_global_state(layout, row, &zeros)[StateDim::new(0).get()]
        })
        .collect();
    let intercepts = boundary_intercepts();
    let tolerance = rhs_tolerance(intercepts.iter().fold(0.0_f64, |top, v| top.max(v.abs())));
    let leaf_solves: Vec<&CutRowPushRecord> = run
        .worker
        .iter()
        .filter(|record| record.frozen_rows + record.appended_rows == BOUNDARY_CUTS)
        .collect();
    assert!(!leaf_solves.is_empty(), "{kinds:?}: no leaf solve recorded");
    for record in leaf_solves {
        assert!(
            root_storages
                .iter()
                .any(|&storage| close(storage, record.pinned[0])),
            "{kinds:?}: a leaf solve pins storage {}, no state the root pool exported {root_storages:?}",
            record.pinned[0]
        );
        assert!(
            is_sub_multiset(&record.cut_row_rhs, &intercepts, tolerance),
            "{kinds:?}: a leaf solve pushes {:?}, not the boundary intercepts {intercepts:?}",
            record.cut_row_rhs
        );
    }
}

fn stores_a_lag_cut(setup: &StudySetup) -> bool {
    let training_ctx = setup.training_ctx();
    setup.fcf.pools.iter().enumerate().any(|(pool_id, pool)| {
        let lag_slots = training_ctx.cut_state_layouts[pool_id].lag_point().full();
        (0..pool.populated()).any(|slot| {
            pool.coefficient_row(slot)[lag_slots.clone()]
                .iter()
                .any(|&c| c != 0.0)
        })
    })
}

/// derived: a root solve of the lower bound pushes the cut row of pool 0's slot `k` at
/// `α_k + Σ_h β_k(h, 0)·ẑ_h(o) + Σ_h Σ_{l ≥ 1} β_k(h, l)·â(h, l − 1)` for a root opening `o`, the lag
/// slots read lag-major (slot `full().start + l·N + h` holds `(h, l)`) and `â` the initial lags, with
/// `ẑ(o) = p + w` for the conditional mean `p` at the initial state and the noise term `w` of
/// `distribution`. Returns each opening's value per written slot and their tolerance.
fn root_candidates(
    setup: &StudySetup,
    distribution: InflowResidualDistribution,
) -> (Vec<Vec<f64>>, f64) {
    let training_ctx = setup.training_ctx();
    let par = training_ctx.stochastic.par();
    let state = training_ctx.state;
    let n_hydros = state.hydro_count;
    let layout = &training_ctx.cut_state_layouts[0];
    let lag_slots = layout.lag_point().full();
    assert_eq!(
        layout.lag_point().mode(),
        LagPointMode::Shifted,
        "pool 0 reads the shifted lag point"
    );
    assert_eq!(
        layout.n_slots(),
        state.n_state,
        "pool 0 projects every state dimension"
    );
    assert_eq!(
        lag_slots.len(),
        n_hydros * state.max_par_order,
        "pool 0 holds every lag slot"
    );
    let initial = training_ctx.initial_state;
    let lags = &initial[state_lag_dims(state)];
    let means = conditional_means(setup, 0, initial);
    let pool = &setup.fcf.pools[0];
    let slots: Vec<usize> = written_slots(pool).collect();

    let mut magnitude = lags.iter().fold(0.0_f64, |top, lag| top.max(lag.abs()));
    let mut candidates = Vec::new();
    for opening in 0..setup.inputs.node_graph.nodes[NodePos(0)].openings.len {
        let noise = node_opening_noise(setup, NodePos(0), opening);
        let eta = &noise[..n_hydros];
        let mut term = vec![0.0; n_hydros];
        match distribution {
            InflowResidualDistribution::Normal => {
                evaluate_par_noise_term_batch(par, 0, eta, &mut term);
            }
            InflowResidualDistribution::Lognormal3p => {
                evaluate_lognormal_noise_term_batch(par, 0, &means, eta, &mut term);
            }
        }
        let mut inflow = vec![0.0; n_hydros];
        add_noise_term_batch(&means, &term, &mut inflow);
        magnitude = inflow.iter().fold(magnitude, |top, z| top.max(z.abs()));
        candidates.push(
            slots
                .iter()
                .map(|&slot| {
                    let row = pool.coefficient_row(slot);
                    let lag_terms: f64 = lag_slots
                        .clone()
                        .enumerate()
                        .map(|(j, at)| {
                            let phi = if j < n_hydros {
                                inflow[j]
                            } else {
                                lags[j - n_hydros]
                            };
                            row[at] * phi
                        })
                        .sum();
                    pool.intercept(slot) + lag_terms
                })
                .collect(),
        );
    }
    let cuts: Vec<(f64, f64)> = slots
        .iter()
        .map(|&slot| {
            let lag_row = &pool.coefficient_row(slot)[lag_slots.clone()];
            (pool.intercept(slot), lag_row.iter().map(|b| b.abs()).sum())
        })
        .collect();
    (candidates, tolerance(&cuts, magnitude))
}

fn assert_lower_bound_pushes_at_the_lognormal_inflow(label: &str, run: &RecordedRun) {
    let (lognormal, tolerance) =
        root_candidates(&run.setup, InflowResidualDistribution::Lognormal3p);
    let (normal, normal_tolerance) =
        root_candidates(&run.setup, InflowResidualDistribution::Normal);
    let fits = |pushed: &[f64], candidates: &[Vec<f64>], tolerance: f64| {
        candidates
            .iter()
            .any(|values| is_sub_multiset(pushed, values, tolerance))
    };
    let pushing: Vec<&CutRowPushRecord> = run
        .lower_bound
        .iter()
        .filter(|record| !record.cut_row_rhs.is_empty())
        .collect();
    assert!(!pushing.is_empty(), "{label}: no record has cut rows");
    for (index, record) in pushing.iter().enumerate() {
        assert!(
            fits(&record.cut_row_rhs, &lognormal, tolerance),
            "{label}: record {index} pushes {:?}, which is no root opening's α + β·Φ at the \
             lognormal inflow {lognormal:?} (tolerance {tolerance:e})",
            record.cut_row_rhs
        );
    }
    assert!(
        pushing
            .iter()
            .any(|record| !fits(&record.cut_row_rhs, &normal, normal_tolerance)),
        "{label}: every record also fits the normal candidates {normal:?}, so the check could \
         hold vacuously"
    );
}

fn assert_realized_inflows_are_lognormal(
    kinds: &[SolveSiteKind],
    spec: &RunSpec,
    run: &RecordedRun,
) {
    let setup = &run.setup;
    let differs = (0..=CLOSING_STAGE).any(|stage| {
        weekly_inflows(setup, stage, SAME_PATH_LAG_SEED_M3S)
            .iter()
            .enumerate()
            .any(|(opening, &lognormal)| {
                let normal = realized_inflow_with(
                    setup,
                    stage,
                    opening,
                    &[SAME_PATH_LAG_SEED_M3S],
                    InflowResidualDistribution::Normal,
                );
                !close(lognormal, normal)
            })
    });
    assert!(
        differs,
        "{kinds:?}: the lognormal and normal inflows agree at every opening of stages \
         0..={CLOSING_STAGE}, so the check could hold vacuously"
    );
    assert_eq!(
        run.simulated.is_empty(),
        !spec.simulate,
        "{kinds:?} simulation: the run simulates exactly when its spec does"
    );
    for scenario in &run.simulated {
        for (stage, result) in scenario.stages.iter().enumerate().take(CLOSING_STAGE + 1) {
            let inflow = result.hydros[0].incremental_inflow_m3s;
            let drawn = weekly_inflows(setup, stage, SAME_PATH_LAG_SEED_M3S);
            assert!(
                drawn.iter().any(|&lognormal| close(lognormal, inflow)),
                "{kinds:?} simulation: scenario {} realizes {inflow} at stage {stage}, none of \
                 the lognormal inflows {drawn:?}",
                scenario.scenario_id
            );
        }
    }
}

type SitesByRun = Vec<(RunSpec, Vec<(SolveSiteKind, Witness)>)>;

fn push_site(runs: &mut SitesByRun, spec: RunSpec, kind: SolveSiteKind, witness: Witness) {
    match runs.iter_mut().find(|(seen, _)| *seen == spec) {
        Some((_, sites)) => sites.push((kind, witness)),
        None => runs.push((spec, vec![(kind, witness)])),
    }
}

#[test]
fn every_solve_site_patches_every_resident_cut_row_at_every_solve() {
    let mut runs = SitesByRun::new();
    for kind in SolveSiteKind::ALL {
        let (spec, witness) = recorder_config(kind);
        push_site(&mut runs, spec, kind, witness);
    }

    for (spec, sites) in &runs {
        let kinds: Vec<SolveSiteKind> = sites.iter().map(|&(kind, _)| kind).collect();
        let run = run_recorded(spec);
        let records: usize = run.logs().iter().map(|(_, records)| records.len()).sum();
        assert!(records >= 1, "{kinds:?}: the run recorded no solve");
        for &(kind, witness) in sites {
            assert_witness(kind, witness, &run);
        }
        assert_no_stale_rows(&kinds, &run);
    }
}

/// derived: the weekly deck's stages 0 to 2 are one-week `Carry` stages and stage 3 closes
/// January, so a stage-3 cut row is pushed at `Φ = s + (240/744)·ẑ_3`, with
/// `s = (168/744)·(ẑ_0 + ẑ_1 + ẑ_2)` and `ẑ_t = b_t + ψ·200 + σ_t·η_t`, and a row of stages 0 to
/// 2 at the carried lag 200; every row's RHS is `α + β·Φ`. A solve is tied to its visit by the
/// storage it pins, against the incoming states the run exports (the simulation's from its own
/// results), never by solve order. A state fixes the January path and so `s`, which the test
/// asserts for visits that share one. The sampled sites read other `s` on other trials; the
/// enumerated chain has one opening per stage, hence one path and one `s`. `FusedTerminal`
/// needs an external terminal leaf, which no weekly deck has: it is checked on the terminal fan,
/// where no lag period is open and the realized inflow and the lags are 0, so the check pins
/// each boundary row at its intercept and cannot carry a differing `s`.
#[test]
fn every_solve_site_pushes_cut_rows_at_its_own_partial_sum() {
    let mut weekly = SitesByRun::new();
    let mut fan = Vec::new();
    for kind in SolveSiteKind::ALL {
        match partial_sum_arm(kind) {
            Weekly(spec, witness) => push_site(&mut weekly, spec, kind, witness),
            Fan => fan.push(kind),
        }
    }

    for (spec, sites) in &weekly {
        let kinds: Vec<SolveSiteKind> = sites.iter().map(|&(kind, _)| kind).collect();
        let run = run_recorded(spec);
        for &(kind, witness) in sites {
            assert_witness(kind, witness, &run);
        }
        assert_weekly_pushes(&kinds, spec, &run);
    }

    let run = run_recorded(&TERMINAL_FAN);
    for &kind in &fan {
        assert_witness(kind, Witness::BoundaryRows, &run);
    }
    assert_fan_boundary_values(&fan, &run);
}

#[test]
fn every_solve_site_patches_every_resident_cut_row_at_every_solve_under_lognormal() {
    let mut runs = SitesByRun::new();
    for kind in SolveSiteKind::ALL {
        if let Some((spec, witness)) = lognormal_recorder_config(kind) {
            push_site(&mut runs, spec, kind, witness);
        }
    }

    for (spec, sites) in &runs {
        let kinds: Vec<SolveSiteKind> = sites.iter().map(|&(kind, _)| kind).collect();
        let run = run_recorded(spec);
        let records: usize = run.logs().iter().map(|(_, records)| records.len()).sum();
        assert!(records >= 1, "{kinds:?}: the run recorded no solve");
        for &(kind, witness) in sites {
            assert_witness(kind, witness, &run);
        }
        assert_no_stale_rows(&kinds, &run);
        assert_the_stored_pins_reach_both_branches(&format!("{kinds:?}"), &run.setup);
        if *spec == LOGNORMAL_SAMPLED {
            let label = format!("{kinds:?} {:?} lower bound", spec.fixture);
            assert_lower_bound_pushes_at_the_lognormal_inflow(&label, &run);
        }
    }
}

#[test]
fn every_solve_site_pushes_cut_rows_at_its_own_partial_sum_under_lognormal() {
    let mut weekly = SitesByRun::new();
    for kind in SolveSiteKind::ALL {
        if let Some((spec, witness)) = lognormal_partial_sum_arm(kind) {
            push_site(&mut weekly, spec, kind, witness);
        }
    }

    for (spec, sites) in &weekly {
        let kinds: Vec<SolveSiteKind> = sites.iter().map(|&(kind, _)| kind).collect();
        let run = run_recorded(spec);
        for &(kind, witness) in sites {
            assert_witness(kind, witness, &run);
        }
        assert_weekly_pushes(&kinds, spec, &run);
        assert_realized_inflows_are_lognormal(&kinds, spec, &run);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains and simulates every committed deck"
)]
fn every_committed_deck_patches_every_resident_cut_row_at_every_solve() {
    let mut visited = 0;
    let mut lag_bearing = false;
    for deck in committed_decks() {
        if SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let (_, setup) = fresh_system_and_setup_with(&deck.dir, |config| {
            apply_oracle_config(config);
            enable_simulation(config);
        });
        let run = record_run(
            &deck.key,
            setup,
            BackwardScheduler::ByScenario {},
            true,
            AppendedCutRowBound::Patched,
            &[],
        );
        assert_no_stale_rows(&deck.key, &run);
        lag_bearing |= stores_a_lag_cut(&run.setup);
        visited += 1;
    }
    assert_eq!(
        visited,
        committed_decks().len() - SLOW_DECKS.len(),
        "the sweep skipped a deck"
    );
    assert!(lag_bearing, "no deck stored a cut with a lag coefficient");
}
