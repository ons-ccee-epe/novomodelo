use std::fmt::Debug;
use std::sync::mpsc;

use novomodelo_io::Config;
use novomodelo_io::config::{
    BackwardScheduler, SelectionMethod, SimulationSelection, TrainingSelection,
};
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::test_support::bound_recording::{
    AppendedCutRowBound, BoundRecordingSolver, CutRowPushLog, CutRowPushRecord,
};
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::external_distinct_fan_setup_heterogeneous_cut_state;
use novomodelo_sddp::test_support::pool_cut_state_dimensions;
use novomodelo_sddp::test_support::solve_sites::SolveSiteKind;
use novomodelo_sddp::{SolverProfiles, StudySetup, TrainingResult};
use novomodelo_solver::ActiveSolver;

use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use crate::common::stored_cut_oracles::{apply_oracle_config, apply_oracle_config_with_selection};
use crate::common::{
    StubComm, build_setup_in_code_with_models, fresh_system_and_setup_with,
    inject_constant_terminal_boundary_cuts, positive,
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
    TerminalFan,
}

#[derive(Clone, Copy, PartialEq)]
struct RunSpec {
    fixture: Fixture,
    openings: usize,
    selection: TrainingSelection,
    dynamic_cut_selection: bool,
    backward_scheduler: BackwardScheduler,
    simulate: bool,
}

const SAMPLED: RunSpec = RunSpec {
    fixture: Fixture::LagStudy,
    openings: 3,
    selection: TrainingSelection::Sampled { forward_passes: 1 },
    dynamic_cut_selection: false,
    backward_scheduler: BackwardScheduler::ByScenario {},
    simulate: true,
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

struct RecordedRun {
    setup: StudySetup,
    result: TrainingResult,
    worker: Vec<CutRowPushRecord>,
    lower_bound: Vec<CutRowPushRecord>,
    simulation: Vec<CutRowPushRecord>,
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

fn build_setup(spec: &RunSpec) -> StudySetup {
    match spec.fixture {
        Fixture::LagStudy => {
            let (system, mut config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
                openings: spec.openings,
                ..FphaLagOptions::default()
            });
            if spec.dynamic_cut_selection {
                apply_oracle_config_with_selection(
                    &mut config,
                    SelectionMethod::Dynamic {
                        start_iteration: positive(1),
                        seed_window: SEED_WINDOW,
                        candidate_recency: None,
                        max_added_per_round: positive(10),
                        violation_tolerance: None,
                    },
                );
            } else {
                apply_oracle_config(&mut config);
            }
            config.training.selection = Some(spec.selection);
            if spec.simulate {
                enable_simulation(&mut config);
            }
            build_setup_in_code_with_models(system, &config, hydro_models)
        }
        Fixture::TerminalFan => {
            assert!(
                spec.selection == TrainingSelection::Enumerated {},
                "the terminal fan runs enumerated"
            );
            let mut setup =
                external_distinct_fan_setup_heterogeneous_cut_state(FAN_K, FAN_ITERATIONS);
            inject_constant_terminal_boundary_cuts(
                &mut setup,
                &[BOUNDARY_INTERCEPT; BOUNDARY_CUTS],
                true,
            );
            setup
        }
    }
}

fn record_run(
    label: &str,
    mut setup: StudySetup,
    backward_scheduler: BackwardScheduler,
    simulate: bool,
    appended_bound: AppendedCutRowBound,
) -> RecordedRun {
    let theta = setup.training_ctx().state.theta;
    let lower_bound_log = CutRowPushLog::new();
    let worker_log = CutRowPushLog::new();
    let simulation_log = CutRowPushLog::new();
    let recording = |log: &CutRowPushLog, appended_bound| {
        ActiveSolver::new().map(|solver| {
            BoundRecordingSolver::logging_cut_row_pushes(solver, theta, log.clone(), appended_bound)
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
        let drain = std::thread::spawn(move || result_rx.into_iter().for_each(drop));
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
        drain.join().expect("drain thread must not panic");
    }

    RecordedRun {
        setup,
        result: outcome.result,
        worker: worker_log.records(),
        lower_bound: lower_bound_log.records(),
        simulation: simulation_log.records(),
    }
}

fn run_recorded(spec: &RunSpec) -> RecordedRun {
    let label = format!("{:?}", spec.fixture);
    let appended_bound = if spec.dynamic_cut_selection {
        AppendedCutRowBound::Final
    } else {
        AppendedCutRowBound::Patched
    };
    record_run(
        &label,
        build_setup(spec),
        spec.backward_scheduler,
        spec.simulate,
        appended_bound,
    )
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

fn assert_boundary_rows(kind: SolveSiteKind, run: &RecordedRun) {
    let setup = &run.setup;
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

fn stores_a_lag_cut(setup: &StudySetup) -> bool {
    let training_ctx = setup.training_ctx();
    setup.fcf.pools.iter().enumerate().any(|(pool_id, pool)| {
        let lag0 = training_ctx.cut_state_layouts[pool_id].lag_point().lag0();
        (0..pool.populated()).any(|slot| {
            pool.coefficient_row(slot)[lag0.clone()]
                .iter()
                .any(|&c| c != 0.0)
        })
    })
}

#[test]
fn every_solve_site_patches_every_resident_cut_row_at_every_solve() {
    let mut runs: Vec<(RunSpec, Vec<(SolveSiteKind, Witness)>)> = Vec::new();
    for kind in SolveSiteKind::ALL {
        let (spec, witness) = recorder_config(kind);
        match runs.iter_mut().find(|(seen, _)| *seen == spec) {
            Some((_, sites)) => sites.push((kind, witness)),
            None => runs.push((spec, vec![(kind, witness)])),
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
