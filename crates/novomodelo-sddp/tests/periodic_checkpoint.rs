//! Checkpoints written by the training loop on the `policy.checkpointing`
//! schedule.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

mod common;

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use novomodelo_core::{System, TrainingEvent};
use novomodelo_io::Config;
use novomodelo_io::config::{
    CheckpointingConfig, SelectionMethod, StoppingRuleConfig, TrainingSelection,
};
use novomodelo_io::output::policy::{PolicyCheckpoint, read_policy_checkpoint};
use novomodelo_sddp::cut::SlotOrigin;
use novomodelo_sddp::policy::full_fcf_load::{FullFcfLoadKind, check_full_fcf_load};
use novomodelo_sddp::policy::orchestration::{
    CheckpointParams, partial_policy_dir, write_checkpoint,
};
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::{
    CapturedBasis, DEFAULT_COST_SCALE_FACTOR, SddpError, StudySetup, TrainingResult,
};
use novomodelo_solver::{ActiveSolver, BasisStatus};
use tempfile::TempDir;

use common::selection_fingerprint::{
    OwnershipPower, SelectionFingerprint, SelectionPower, row_management_events,
    selection_ran_at_the_last_iteration,
};
use common::{StubComm, fresh_system_and_setup_with, positive};

fn toy_case() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy")
}

fn checkpointing_setup(
    iteration_limit: u32,
    initial_iteration: Option<u32>,
    interval_iterations: u32,
    forward_passes: u32,
) -> (System, StudySetup) {
    fresh_system_and_setup_with(&toy_case(), |config| {
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: iteration_limit,
        }]);
        config.training.cut_selection.selection = Some(SelectionMethod::Level1 {
            check_frequency: positive(2),
            lookback_iterations: None,
        });
        config.policy.checkpointing = CheckpointingConfig {
            enabled: Some(true),
            initial_iteration,
            interval_iterations: Some(interval_iterations),
            ..CheckpointingConfig::default()
        };
    })
}

fn train(
    setup: &mut StudySetup,
    event_sender: Option<mpsc::Sender<TrainingEvent>>,
) -> TrainingResult {
    train_on(setup, 1, event_sender)
}

fn train_on(
    setup: &mut StudySetup,
    n_threads: usize,
    event_sender: Option<mpsc::Sender<TrainingEvent>>,
) -> TrainingResult {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(
            &mut solver,
            &StubComm,
            n_threads,
            ActiveSolver::new,
            event_sender,
            None,
        )
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );
    outcome.result
}

/// Trains inside a rayon pool of `n_threads` threads: the ownership scan runs on
/// the ambient pool, so without it the thread count would not vary it.
fn train_in_pool(
    setup: &mut StudySetup,
    n_threads: usize,
    event_sender: Option<mpsc::Sender<TrainingEvent>>,
) -> TrainingResult {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("the rayon pool must build")
        .install(|| train_on(setup, n_threads, event_sender))
}

fn train_collecting(
    setup: &mut StudySetup,
    n_threads: usize,
) -> (TrainingResult, Vec<TrainingEvent>) {
    let (sender, receiver) = mpsc::channel();
    let result = train_in_pool(setup, n_threads, Some(sender));
    (result, receiver.try_iter().collect())
}

/// The N-run (`iteration_limit: 6`) leaves only its iteration-3 periodic
/// checkpoint in its policy directory; the k-run (`iteration_limit: 3`) is
/// written by the final-checkpoint path.
struct PeriodicAndFinal {
    periodic: PolicyCheckpoint,
    final_at_k: PolicyCheckpoint,
    n_run_pool_capacities: Vec<u32>,
}

fn periodic_at_3_and_final_at_3(forward_passes: u32) -> PeriodicAndFinal {
    let n_out = tempfile::tempdir().expect("tempdir");
    let (system, mut n_setup) = checkpointing_setup(6, Some(3), 6, forward_passes);
    let n_run_pool_capacities = n_setup
        .fcf
        .pools
        .iter()
        .map(|p| u32::try_from(p.capacity).unwrap())
        .collect();
    n_setup.enable_periodic_checkpoints(&system, n_out.path());
    assert_eq!(train(&mut n_setup, None).iterations, 6);
    let periodic = read_policy_checkpoint(&n_out.path().join(&n_setup.policy_path))
        .expect("the periodic checkpoint must be readable");

    let (k_system, mut k_setup) = checkpointing_setup(3, Some(3), 6, forward_passes);
    let k_result = train(&mut k_setup, None);
    assert_eq!(k_result.iterations, 3);
    let k_out = tempfile::tempdir().expect("tempdir");
    let k_policy_dir = k_out.path().join(&k_setup.policy_path);
    write_checkpoint(
        &k_policy_dir,
        &k_setup,
        &k_system,
        &k_result,
        &CheckpointParams {
            max_iterations: k_setup.loop_params.max_iterations,
            forward_passes: k_setup.loop_params.forward_passes,
            seed: k_setup.loop_params.seed,
        },
    )
    .expect("write_checkpoint must succeed");
    let final_at_k = read_policy_checkpoint(&k_policy_dir).expect("read_policy_checkpoint");

    PeriodicAndFinal {
        periodic,
        final_at_k,
        n_run_pool_capacities,
    }
}

fn assert_periodic_equals_final(periodic: &PolicyCheckpoint, final_at_k: &PolicyCheckpoint) {
    assert_eq!(periodic.metadata.producer.completed_iterations, 3);
    assert_eq!(periodic.stage_cuts.len(), final_at_k.stage_cuts.len());
    assert!(
        !periodic.stage_states.is_empty(),
        "the case's pools must store trial states"
    );

    let mut aligned = periodic.clone();
    aligned
        .metadata
        .created_at
        .clone_from(&final_at_k.metadata.created_at);
    aligned.metadata.producer.max_iterations = final_at_k.metadata.producer.max_iterations;
    for (pool, final_pool) in aligned.stage_cuts.iter_mut().zip(&final_at_k.stage_cuts) {
        pool.capacity = final_pool.capacity;
    }

    assert_eq!(format!("{aligned:?}"), format!("{final_at_k:?}"));
}

#[test]
fn periodic_checkpoint_equals_the_final_checkpoint_of_a_run_stopped_at_that_iteration() {
    let PeriodicAndFinal {
        periodic,
        final_at_k,
        ..
    } = periodic_at_3_and_final_at_3(1);
    assert_periodic_equals_final(&periodic, &final_at_k);
}

#[test]
fn periodic_checkpoint_with_several_forward_passes_holds_every_scenario_basis_and_equals_the_final_one()
 {
    let PeriodicAndFinal {
        periodic,
        final_at_k,
        ..
    } = periodic_at_3_and_final_at_3(3);

    for (which, checkpoint) in [("periodic", &periodic), ("final", &final_at_k)] {
        assert!(
            checkpoint.metadata.producer.basis_store_complete,
            "the {which} checkpoint holds the whole basis store"
        );
        assert!(
            !checkpoint.stage_bases.is_empty(),
            "the {which} checkpoint holds bases"
        );
        for record in &checkpoint.stage_bases {
            assert_eq!(
                record
                    .entries
                    .iter()
                    .map(|e| e.scenario)
                    .collect::<Vec<_>>(),
                [0, 1, 2],
                "{which} checkpoint, node {}: one basis per forward pass",
                record.stage_id
            );
        }
    }
    assert!(
        periodic.stage_bases.iter().any(|record| {
            record.entries.iter().any(|a| {
                record.entries.iter().any(|b| {
                    (&a.column_status, &a.row_status, &a.piece_row_slots)
                        != (&b.column_status, &b.row_status, &b.piece_row_slots)
                })
            })
        }),
        "power: at some node two forward passes store different bases"
    );

    assert_periodic_equals_final(&periodic, &final_at_k);
}

#[test]
fn periodic_checkpoint_records_the_running_configuration_as_provenance() {
    let PeriodicAndFinal {
        periodic,
        final_at_k,
        n_run_pool_capacities,
    } = periodic_at_3_and_final_at_3(1);

    assert_eq!(periodic.metadata.producer.max_iterations, 6);
    assert_eq!(final_at_k.metadata.producer.max_iterations, 3);

    let periodic_capacities: Vec<u32> = periodic.stage_cuts.iter().map(|p| p.capacity).collect();
    assert_eq!(periodic_capacities, n_run_pool_capacities);
    assert!(
        periodic
            .stage_cuts
            .iter()
            .zip(&final_at_k.stage_cuts)
            .any(|(p, k)| p.capacity != k.capacity),
        "the iteration limit must size at least one pool differently"
    );
}

#[test]
fn periodic_checkpoints_follow_the_schedule_and_skip_the_stop_iteration() {
    let out = tempfile::tempdir().expect("tempdir");
    let (system, mut setup) = checkpointing_setup(6, Some(2), 2, 1);
    setup.enable_periodic_checkpoints(&system, out.path());
    let policy_dir = out.path().join(&setup.policy_path);

    let (tx, rx) = mpsc::channel();
    assert_eq!(train(&mut setup, Some(tx)).iterations, 6);

    let written: Vec<(u64, String)> = rx
        .try_iter()
        .filter_map(|event| match event {
            TrainingEvent::CheckpointComplete {
                iteration,
                checkpoint_path,
                ..
            } => Some((iteration, checkpoint_path)),
            _ => None,
        })
        .collect();
    let expected_path = policy_dir.display().to_string();
    assert_eq!(
        written,
        vec![(2, expected_path.clone()), (4, expected_path)]
    );

    let latest = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
    assert_eq!(latest.metadata.producer.completed_iterations, 4);
}

#[test]
fn failed_periodic_write_ends_training_with_the_failing_iteration() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let blocker = tmp.path().join("blocker");
    let (system, mut setup) = checkpointing_setup(6, None, 2, 1);
    setup.enable_periodic_checkpoints(&system, &blocker);
    std::fs::write(&blocker, b"a regular file, not a directory").expect("write blocker");

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");

    assert!(
        matches!(
            outcome.error,
            Some(SddpError::CheckpointWrite { iteration: 2, .. })
        ),
        "{:?}",
        outcome.error
    );
    assert_eq!(outcome.result.iterations, 2);
}

fn policy_bytes(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, dir: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).expect("read"),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(dir, dir, &mut files);
    files
}

#[test]
fn a_failed_periodic_write_leaves_the_last_whole_checkpoint_and_a_partial_beside_it() {
    let out = tempfile::tempdir().expect("tempdir");
    let (system, mut first) = checkpointing_setup(3, None, 1, 1);
    let first_result = train(&mut first, None);
    let policy_dir = out.path().join(&first.policy_path);
    let params = CheckpointParams {
        max_iterations: first.loop_params.max_iterations,
        forward_passes: first.loop_params.forward_passes,
        seed: first.loop_params.seed,
    };
    assert_eq!(
        write_checkpoint(&policy_dir, &first, &system, &first_result, &params)
            .expect("write_checkpoint must succeed"),
        policy_dir
    );
    let whole = policy_bytes(&policy_dir);
    std::fs::write(
        policy_dir.join("foreign"),
        b"no checkpoint writer leaves this",
    )
    .unwrap();

    let (system, mut resumed) = checkpointing_setup(6, None, 1, 1);
    check_full_fcf_load(
        FullFcfLoadKind::Resume,
        &policy_dir,
        &system,
        &resumed,
        &mut |_msg: &str| {},
    )
    .expect("the checkpoint must load beside the foreign file")
    .apply_to_training(&mut resumed);
    resumed.enable_periodic_checkpoints(&system, out.path());
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = resumed
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        matches!(
            outcome.error,
            Some(SddpError::CheckpointWrite { iteration: 4, .. })
        ),
        "{:?}",
        outcome.error
    );

    let partial = write_checkpoint(&policy_dir, &resumed, &system, &outcome.result, &params)
        .expect("the failure exit writes beside the policy directory");

    assert_eq!(Some(partial.clone()), partial_policy_dir(&policy_dir));
    let partial_checkpoint = read_policy_checkpoint(&partial).expect("the partial policy reads");
    assert!(!partial_checkpoint.metadata.producer.basis_store_complete);
    let mut kept = policy_bytes(&policy_dir);
    assert!(kept.remove(Path::new("foreign")).is_some());
    assert_eq!(
        kept, whole,
        "the whole checkpoint is byte for byte the one the first run wrote"
    );

    std::fs::remove_file(policy_dir.join("foreign")).unwrap();
    let (system, third) = checkpointing_setup(6, None, 1, 1);
    let load = check_full_fcf_load(
        FullFcfLoadKind::Resume,
        &policy_dir,
        &system,
        &third,
        &mut |_msg: &str| {},
    )
    .expect("the policy directory still resumes");
    assert_eq!(load.completed_iterations(), 3);
}

/// Bound on two solves of one LP that start from different solver states.
const LOWER_BOUND_REL_TOL: f64 = 1e-6;

struct RunOutcome {
    fingerprint: SelectionFingerprint,
    resolved_cost_scale_factor: f64,
    simplex_iterations: BTreeMap<(u64, &'static str, Option<i32>), u64>,
    final_checkpoint: PolicyCheckpoint,
    events: Vec<TrainingEvent>,
    selection_ran_at_the_last_iteration: bool,
}

fn checkpoint_params(setup: &StudySetup) -> CheckpointParams {
    CheckpointParams {
        max_iterations: setup.loop_params.max_iterations,
        forward_passes: setup.loop_params.forward_passes,
        seed: setup.loop_params.seed,
    }
}

fn write_and_read_checkpoint(
    system: &System,
    setup: &StudySetup,
    result: &TrainingResult,
) -> PolicyCheckpoint {
    let out = tempfile::tempdir().expect("tempdir");
    let policy_dir = out.path().join(&setup.policy_path);
    write_checkpoint(
        &policy_dir,
        setup,
        system,
        result,
        &checkpoint_params(setup),
    )
    .expect("write_checkpoint must succeed");
    read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint")
}

impl RunOutcome {
    /// Reads the run off its setup and result; `compared_iterations` are the
    /// iterations whose simplex counts the run reports.
    fn capture(
        system: &System,
        setup: &StudySetup,
        result: &TrainingResult,
        events: Vec<TrainingEvent>,
        compared_iterations: &RangeInclusive<u64>,
    ) -> Self {
        let mut simplex_iterations = BTreeMap::new();
        for entry in result
            .solver_stats_log
            .iter()
            .filter(|e| compared_iterations.contains(&e.iteration))
            .filter(|e| matches!(e.phase, "forward" | "backward"))
        {
            *simplex_iterations
                .entry((entry.iteration, entry.phase, entry.stage_id))
                .or_insert(0) += entry.delta.simplex_iterations;
        }
        assert!(
            !simplex_iterations.is_empty(),
            "no forward or backward solver statistics were logged for the compared iterations"
        );
        Self {
            fingerprint: SelectionFingerprint::capture(setup, result),
            resolved_cost_scale_factor: setup.inputs.stage_data.stage_templates.cost_scale_factor,
            simplex_iterations,
            final_checkpoint: write_and_read_checkpoint(system, setup, result),
            selection_ran_at_the_last_iteration: selection_ran_at_the_last_iteration(
                &events, result,
            ),
            events,
        }
    }
}

/// One basis as the checkpoint writes it: column and row status codes and the
/// slot of each cut row.
type StoredBasisView = (Vec<u8>, Vec<u8>, Vec<u32>);

fn stored_view(basis: &CapturedBasis) -> StoredBasisView {
    let codes = |statuses: &[BasisStatus]| {
        statuses
            .iter()
            .map(|status| status.to_discriminant_code())
            .collect()
    };
    (
        codes(&basis.basis.col_status),
        codes(&basis.basis.row_status),
        basis.cut_row_slots.clone(),
    )
}

/// The basis store a resume restores, against the one the interrupted run
/// wrote: per `(node, scenario)`, the basis the resumed setup seeds and the
/// checkpoint's entry.
struct RestoredStore {
    seeded: BTreeMap<(usize, u32), Option<StoredBasisView>>,
    written: BTreeMap<(usize, u32), Option<StoredBasisView>>,
}

impl RestoredStore {
    fn capture(setup: &StudySetup, interrupted: &PolicyCheckpoint, forward_passes: u32) -> Self {
        let mut seeded = BTreeMap::new();
        let mut written = BTreeMap::new();
        for node in 0..setup.inputs.node_graph.nodes.len() {
            for scenario in 0..forward_passes {
                let seed = setup
                    .loaded_policy()
                    .seed_basis(NodePos(node), scenario)
                    .map(stored_view);
                seeded.insert((node, scenario), seed);
                written.insert((node, scenario), None);
            }
        }
        for record in &interrupted.stage_bases {
            for entry in &record.entries {
                written.insert(
                    (record.stage_id as usize, entry.scenario),
                    Some((
                        entry.column_status.clone(),
                        entry.row_status.clone(),
                        entry.piece_row_slots.clone(),
                    )),
                );
            }
        }
        Self { seeded, written }
    }
}

/// A resume leg on `1dtoy`: the config edit every run of the leg takes, the
/// iteration `k` the interrupted run stops at, the iteration `n` the other
/// runs stop at, the thread counts before and after the resume, and the policy
/// the runs warm-start from. Each run stops on its iteration limit.
#[derive(Clone, Copy)]
struct Leg {
    forward_passes: u32,
    configure: fn(&mut Config),
    k: u32,
    n: u32,
    threads_before: usize,
    threads_after: usize,
    source: Option<SourceRun>,
}

/// The run whose checkpoint a leg warm-starts from.
#[derive(Clone, Copy)]
struct SourceRun {
    iterations: u32,
    configure: fn(&mut Config),
}

impl Leg {
    fn resumed_iterations(&self) -> RangeInclusive<u64> {
        u64::from(self.k) + 1..=u64::from(self.n)
    }

    fn setup(&self, configure: fn(&mut Config), iteration_limit: u32) -> (System, StudySetup) {
        fresh_system_and_setup_with(&toy_case(), |config| {
            config.training.selection = Some(TrainingSelection::Sampled {
                forward_passes: self.forward_passes,
            });
            config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
                limit: iteration_limit,
            }]);
            configure(config);
        })
    }
}

fn no_edit(_: &mut Config) {}

const SELECTION_OFF: Leg = Leg {
    forward_passes: 1,
    configure: no_edit,
    k: 3,
    n: 6,
    threads_before: 1,
    threads_after: 1,
    source: None,
};

/// A source's checkpoint and the rows each of its pools holds.
struct TrainedSource {
    _output: TempDir,
    policy_dir: PathBuf,
    rows: Vec<u32>,
}

fn train_source(leg: &Leg, source: SourceRun) -> TrainedSource {
    let (system, mut setup) = leg.setup(source.configure, source.iterations);
    let result = train(&mut setup, None);
    assert_eq!(result.iterations, u64::from(source.iterations));
    let output = tempfile::tempdir().expect("tempdir");
    let policy_dir = output.path().join(&setup.policy_path);
    write_checkpoint(
        &policy_dir,
        &setup,
        &system,
        &result,
        &checkpoint_params(&setup),
    )
    .expect("write_checkpoint must succeed");
    TrainedSource {
        rows: setup
            .fcf
            .pools
            .iter()
            .map(|pool| u32::try_from(pool.generated()).unwrap())
            .collect(),
        _output: output,
        policy_dir,
    }
}

fn load(kind: FullFcfLoadKind, policy_dir: &Path, system: &System, setup: &mut StudySetup) {
    check_full_fcf_load(kind, policy_dir, system, setup, &mut |_msg: &str| {})
        .expect("the checkpoint must load")
        .apply_to_training(setup);
}

/// A leg's setup stopping at `iteration_limit`, warm-started from `source`.
fn leg_setup(
    leg: &Leg,
    iteration_limit: u32,
    source: Option<&TrainedSource>,
) -> (System, StudySetup) {
    let (system, mut setup) = leg.setup(leg.configure, iteration_limit);
    if let Some(source) = source {
        load(
            FullFcfLoadKind::WarmStart,
            &source.policy_dir,
            &system,
            &mut setup,
        );
    }
    (system, setup)
}

/// What a leg's three runs leave: the uninterrupted run, the run resumed from
/// the interrupted one's checkpoint, and the interrupted run itself at `k`.
struct LegRuns {
    leg: Leg,
    resumed: RunOutcome,
    uninterrupted: RunOutcome,
    restored: RestoredStore,
    at_k: SelectionFingerprint,
    ownership_at_k: OwnershipPower,
    checkpoint_prefix: Vec<u32>,
    expected_prefix: Vec<u32>,
}

/// Trains `leg`'s uninterrupted run to `n`, its interrupted run to `k`, and a
/// run that loads the interrupted run's checkpoint through the production
/// resume path and trains to `n`.
fn run_leg(leg: Leg) -> LegRuns {
    let source = leg.source.map(|source| train_source(&leg, source));
    let iterations = leg.resumed_iterations();

    let (full_system, mut full_setup) = leg_setup(&leg, leg.n, source.as_ref());
    let (full_result, full_events) = train_collecting(&mut full_setup, leg.threads_before);
    assert_eq!(full_result.iterations, u64::from(leg.n));
    let uninterrupted = RunOutcome::capture(
        &full_system,
        &full_setup,
        &full_result,
        full_events,
        &iterations,
    );

    let (system, mut first_setup) = leg_setup(&leg, leg.k, source.as_ref());
    let first_result = train_in_pool(&mut first_setup, leg.threads_before, None);
    assert_eq!(first_result.iterations, u64::from(leg.k));
    let at_k = SelectionFingerprint::capture(&first_setup, &first_result);
    let ownership_at_k = OwnershipPower::of(&first_setup);
    let out = tempfile::tempdir().expect("tempdir");
    let policy_dir = out.path().join(&first_setup.policy_path);
    write_checkpoint(
        &policy_dir,
        &first_setup,
        &system,
        &first_result,
        &checkpoint_params(&first_setup),
    )
    .expect("write_checkpoint must succeed");

    let (system, mut resumed_setup) = leg.setup(leg.configure, leg.n);
    load(
        FullFcfLoadKind::Resume,
        &policy_dir,
        &system,
        &mut resumed_setup,
    );
    let interrupted = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
    let restored = RestoredStore::capture(&resumed_setup, &interrupted, leg.forward_passes);
    let (resumed_result, resumed_events) = train_collecting(&mut resumed_setup, leg.threads_after);
    assert_eq!(resumed_result.iterations, u64::from(leg.n));

    LegRuns {
        leg,
        resumed: RunOutcome::capture(
            &system,
            &resumed_setup,
            &resumed_result,
            resumed_events,
            &iterations,
        ),
        uninterrupted,
        restored,
        at_k,
        ownership_at_k,
        checkpoint_prefix: interrupted
            .stage_cuts
            .iter()
            .map(|pool| pool.warm_start_count)
            .collect(),
        expected_prefix: source.map_or_else(
            || vec![0; first_setup.fcf.pools.len()],
            |source| source.rows,
        ),
    }
}

/// The `PolicyPoolSummary` events after iteration `after`, with the wall-clock
/// field reduced to whether it is set.
fn pool_summaries_after(events: &[TrainingEvent], after: u32) -> Vec<(u64, String)> {
    events
        .iter()
        .filter_map(|event| match event {
            TrainingEvent::PolicyPoolSummary { iteration, pools }
                if *iteration > u64::from(after) =>
            {
                let masked: Vec<_> = pools
                    .iter()
                    .cloned()
                    .map(|mut pool| {
                        pool.selection_time_ms = pool.selection_time_ms.map(|_| 0.0);
                        pool
                    })
                    .collect();
                Some((*iteration, format!("{masked:?}")))
            }
            _ => None,
        })
        .collect()
}

/// Asserts that the resumed run holds what the uninterrupted run holds, and
/// returns the largest relative lower-bound difference it saw.
///
/// Bit for bit: the cuts, flags, provenance, binding counters, slot roles,
/// stored trial states and ownership record of every pool, the final upper
/// bound, the simplex counts of the resumed iterations, the pool row counts of
/// the resumed iterations, the restored bases and both final checkpoints
/// (their lower-bound fields and `created_at` apart). Within
/// `LOWER_BOUND_REL_TOL`: the lower-bound series, because the lower-bound LP is
/// rebuilt warm in the uninterrupted run and cold after a resume.
fn assert_same_run(runs: &LegRuns) -> f64 {
    let (resumed, uninterrupted) = (&runs.resumed, &runs.uninterrupted);
    for run in [resumed, uninterrupted] {
        assert_eq!(run.resolved_cost_scale_factor, DEFAULT_COST_SCALE_FACTOR);
    }
    assert!(
        runs.restored.written.values().any(Option::is_some),
        "the interrupted run's checkpoint holds bases"
    );
    for (key, written) in &runs.restored.written {
        assert_eq!(
            &runs.restored.seeded[key], written,
            "(node, scenario) {key:?}: the restored basis is the one the interrupted run wrote"
        );
    }
    assert_eq!(
        runs.checkpoint_prefix, runs.expected_prefix,
        "the prefix length of each pool of the interrupted run's checkpoint"
    );

    resumed
        .fingerprint
        .assert_pools_match(&uninterrupted.fingerprint, "the resumed run");
    assert_eq!(
        resumed.fingerprint.bounds.final_ub_bits, uninterrupted.fingerprint.bounds.final_ub_bits,
        "final upper bound"
    );
    assert_eq!(
        resumed.simplex_iterations, uninterrupted.simplex_iterations,
        "forward and backward simplex iterations of the resumed iterations"
    );
    assert_eq!(
        pool_summaries_after(&resumed.events, runs.leg.k),
        pool_summaries_after(&uninterrupted.events, runs.leg.k),
        "pool row counts of the resumed iterations"
    );

    let (r_lb, u_lb) = (
        &resumed.fingerprint.bounds.lower_bound_bits,
        &uninterrupted.fingerprint.bounds.lower_bound_bits,
    );
    assert_eq!(r_lb.len(), u_lb.len(), "lower-bound history length");
    let mut largest = 0.0_f64;
    for (i, (r, u)) in r_lb.iter().zip(u_lb).enumerate() {
        let (r, u) = (f64::from_bits(*r), f64::from_bits(*u));
        largest = largest.max(assert_lower_bounds_agree(
            &format!("lower bound of iteration {}", i + 1),
            r,
            u,
        ));
    }

    largest.max(assert_same_checkpoint(
        &resumed.final_checkpoint,
        &uninterrupted.final_checkpoint,
        &runs.expected_prefix,
    ))
}

/// Asserts `resumed` and `uninterrupted` agree within `LOWER_BOUND_REL_TOL` and
/// returns their relative difference.
fn assert_lower_bounds_agree(what: &str, resumed: f64, uninterrupted: f64) -> f64 {
    let scale = 1.0_f64.max(resumed.abs()).max(uninterrupted.abs());
    let relative = (resumed - uninterrupted).abs() / scale;
    assert!(
        relative <= LOWER_BOUND_REL_TOL,
        "{what}: resumed {resumed}, uninterrupted {uninterrupted}"
    );
    relative
}

/// Asserts the two final checkpoints agree and returns the largest relative
/// difference of their lower-bound fields.
fn assert_same_checkpoint(
    resumed: &PolicyCheckpoint,
    uninterrupted: &PolicyCheckpoint,
    expected_prefix: &[u32],
) -> f64 {
    let (r_producer, u_producer) = (&resumed.metadata.producer, &uninterrupted.metadata.producer);
    assert_eq!(
        r_producer.lower_bound_history.len(),
        u_producer.lower_bound_history.len(),
        "checkpoint lower-bound history length"
    );
    let mut largest = 0.0_f64;
    for (i, (r, u)) in r_producer
        .lower_bound_history
        .iter()
        .zip(&u_producer.lower_bound_history)
        .enumerate()
    {
        largest = largest.max(assert_lower_bounds_agree(
            &format!("checkpoint lower bound of iteration {}", i + 1),
            *r,
            *u,
        ));
    }
    largest = largest.max(assert_lower_bounds_agree(
        "checkpoint final lower bound",
        r_producer.final_lower_bound,
        u_producer.final_lower_bound,
    ));

    let aligned = |checkpoint: &PolicyCheckpoint| {
        let mut aligned = checkpoint.clone();
        aligned.metadata.created_at.clear();
        aligned.metadata.producer.final_lower_bound = 0.0;
        aligned.metadata.producer.lower_bound_history.clear();
        aligned
    };
    let (r, u) = (aligned(resumed), aligned(uninterrupted));
    assert_eq!(
        format!("{:?}", r.metadata),
        format!("{:?}", u.metadata),
        "checkpoint manifest"
    );
    assert_eq!(r.stage_cuts.len(), u.stage_cuts.len());
    for (pool, (r, u)) in r.stage_cuts.iter().zip(&u.stage_cuts).enumerate() {
        assert_eq!(
            format!("{r:?}"),
            format!("{u:?}"),
            "checkpoint cuts of pool {pool}"
        );
        assert_eq!(
            (r.warm_start_count, u.warm_start_count),
            (expected_prefix[pool], expected_prefix[pool]),
            "pool {pool}: the prefix length of each run's checkpoint"
        );
    }
    assert_eq!(
        format!("{:?}", r.stage_bases),
        format!("{:?}", u.stage_bases),
        "checkpoint bases"
    );
    assert_eq!(
        format!("{:?}", r.stage_states),
        format!("{:?}", u.stage_states),
        "checkpoint trial states"
    );
    largest
}

fn finish(label: &str, runs: &LegRuns) {
    let largest = assert_same_run(runs);
    eprintln!("{label}: largest relative lower-bound difference {largest:e}");
}

/// Every iteration after `k` reports the row counts of its pools, so the
/// comparison of the summaries compares records.
fn assert_row_management_ran(runs: &LegRuns) {
    let leg = &runs.leg;
    assert_eq!(
        pool_summaries_after(&runs.uninterrupted.events, leg.k).len(),
        usize::try_from(leg.n - leg.k).unwrap(),
        "power: pool summaries of the iterations after the checkpoint"
    );
}

/// Power of a `level1` leg: a selection pass ran at the last iteration of both
/// runs, so the ownership record is current, and it changed the flag of a slot
/// the interrupted run had generated.
fn assert_selection_changed_a_flag_after_the_checkpoint(runs: &LegRuns) {
    assert_row_management_ran(runs);
    for (which, run) in [
        ("uninterrupted", &runs.uninterrupted),
        ("resumed", &runs.resumed),
    ] {
        assert!(
            run.selection_ran_at_the_last_iteration,
            "power: no selection pass ran at the last iteration of the {which} run"
        );
    }
    let flipped: usize = runs
        .at_k
        .cuts
        .iter()
        .zip(&runs.at_k.ownership)
        .zip(&runs.uninterrupted.fingerprint.cuts)
        .map(|((at_k, roles), at_n)| {
            (0..at_k.active.len())
                .filter(|&slot| {
                    roles.roles[slot].origin == SlotOrigin::Generated
                        && at_k.active[slot] != at_n.active[slot]
                })
                .count()
        })
        .sum();
    assert!(
        flipped > 0,
        "power: no slot generated by the checkpoint changed its flag afterwards"
    );
}

/// Power of the window legs: the window dropped the state of a scored slot
/// before the checkpoint, so the resume loads a scored cut with no state row.
fn assert_a_window_dropped_a_state_before_the_checkpoint(runs: &LegRuns) {
    assert!(
        runs.ownership_at_k.scored_slots_without_a_state > 0,
        "power: every scored slot holds its state at the checkpoint ({:?})",
        runs.ownership_at_k
    );
}

/// Power of the budget leg: the budget evicted cuts after the checkpoint, and a
/// slot generated by iteration `k` is an owner that is inactive at `n`. A
/// selection pass ran at `n`, so every owner was active before the budget of
/// `n`, and only that budget can have cleared the flag.
fn assert_the_budget_evicted_a_slot_generated_before_the_checkpoint(runs: &LegRuns) {
    let leg = &runs.leg;
    let power = SelectionPower::from_events(&row_management_events(
        &runs.uninterrupted.events,
        leg.resumed_iterations(),
    ));
    assert!(
        power.budget_evictions > 0,
        "power: the budget evicted no cut after the checkpoint ({power:?})"
    );
    let fingerprint = &runs.uninterrupted.fingerprint;
    let evicted_owners: usize = fingerprint
        .cuts
        .iter()
        .zip(&fingerprint.provenance)
        .zip(&fingerprint.ownership)
        .map(|((cuts, provenance), record)| {
            (0..cuts.active.len())
                .filter(|&slot| {
                    provenance.iteration_generated[slot] <= u64::from(leg.k)
                        && !cuts.active[slot]
                        && record.owned_states[slot] > 0
                })
                .count()
        })
        .sum();
    assert!(
        evicted_owners > 0,
        "power: no slot generated by the checkpoint is an owner that the budget evicted"
    );
}

/// Power of the `dynamic` leg: a slot generated by iteration `k` bound again
/// afterwards.
fn assert_a_slot_bound_again_after_the_checkpoint(runs: &LegRuns) {
    let bound_again: usize = runs
        .at_k
        .binding
        .iter()
        .zip(&runs.uninterrupted.fingerprint.binding)
        .map(|(at_k, at_n)| {
            (0..at_k.last_active_iter.len())
                .filter(|&slot| at_n.last_active_iter[slot] > at_k.last_active_iter[slot])
                .count()
        })
        .sum();
    assert!(
        bound_again > 0,
        "power: no slot generated by the checkpoint bound after it"
    );
}

const CHECK_FREQUENCY: u32 = 2;
const WINDOW: u32 = 2;
const BUDGET: u32 = 6;

fn level1_within(lookback_iterations: Option<u32>) -> SelectionMethod {
    SelectionMethod::Level1 {
        check_frequency: positive(CHECK_FREQUENCY),
        lookback_iterations: lookback_iterations.map(positive),
    }
}

fn level1(config: &mut Config) {
    config.training.cut_selection.selection = Some(level1_within(None));
}

fn level1_windowed(config: &mut Config) {
    config.training.cut_selection.selection = Some(level1_within(Some(WINDOW)));
}

fn level1_with_a_budget(config: &mut Config) {
    level1(config);
    config.training.cut_selection.max_active_per_stage = Some(BUDGET);
}

fn dynamic(config: &mut Config) {
    config.training.cut_selection.selection = Some(SelectionMethod::Dynamic {
        start_iteration: positive(2),
        seed_window: 1,
        candidate_recency: None,
        max_added_per_round: positive(10),
        violation_tolerance: None,
    });
}

const LEVEL1: Leg = Leg {
    forward_passes: 2,
    configure: level1,
    k: 3,
    n: 8,
    threads_before: 1,
    threads_after: 1,
    source: None,
};

#[test]
fn level1_resume_equals_the_uninterrupted_run() {
    let runs = run_leg(LEVEL1);
    assert_selection_changed_a_flag_after_the_checkpoint(&runs);
    finish("level1", &runs);
}

/// After the selection pass of iteration 4 has expired states; the pass of
/// iteration 2 runs at `WINDOW` completed iterations, where nothing has left
/// the window.
const CHECKPOINT_AFTER_AN_EXPIRY: u32 = 5;

#[test]
fn level1_with_a_window_resume_equals_the_uninterrupted_run() {
    let runs = run_leg(Leg {
        configure: level1_windowed,
        k: CHECKPOINT_AFTER_AN_EXPIRY,
        ..LEVEL1
    });
    assert_selection_changed_a_flag_after_the_checkpoint(&runs);
    assert_a_window_dropped_a_state_before_the_checkpoint(&runs);
    finish("window", &runs);
}

#[test]
fn level1_with_a_budget_resume_equals_the_uninterrupted_run() {
    let runs = run_leg(Leg {
        configure: level1_with_a_budget,
        ..LEVEL1
    });
    assert_selection_changed_a_flag_after_the_checkpoint(&runs);
    assert_the_budget_evicted_a_slot_generated_before_the_checkpoint(&runs);
    finish("budget", &runs);
}

#[test]
fn dynamic_resume_equals_the_uninterrupted_run() {
    let runs = run_leg(Leg {
        configure: dynamic,
        ..LEVEL1
    });
    assert_row_management_ran(&runs);
    assert_a_slot_bound_again_after_the_checkpoint(&runs);
    finish("dynamic", &runs);
}

#[test]
fn a_resume_on_more_threads_equals_the_uninterrupted_run() {
    let runs = run_leg(Leg {
        threads_after: 2,
        ..LEVEL1
    });
    assert_selection_changed_a_flag_after_the_checkpoint(&runs);
    finish("threads", &runs);
}

const WARM_START_SOURCE: SourceRun = SourceRun {
    iterations: 8,
    configure: level1_windowed,
};

#[test]
fn a_resumed_warm_started_run_equals_its_uninterrupted_run() {
    let runs = run_leg(Leg {
        configure: level1_windowed,
        k: CHECKPOINT_AFTER_AN_EXPIRY,
        source: Some(WARM_START_SOURCE),
        ..LEVEL1
    });
    assert!(
        runs.expected_prefix.iter().any(|&rows| rows > 0),
        "power: the source holds no cut to load"
    );
    assert_selection_changed_a_flag_after_the_checkpoint(&runs);
    assert_a_window_dropped_a_state_before_the_checkpoint(&runs);
    assert!(
        runs.ownership_at_k.loaded_scored_slots_without_a_state > 0,
        "power: every loaded scored slot holds its state at the checkpoint"
    );
    finish("warm", &runs);
}

#[test]
fn resumed_run_equals_the_uninterrupted_run_with_one_forward_pass() {
    finish("selection off, 1 pass", &run_leg(SELECTION_OFF));
}

#[test]
fn resumed_run_equals_the_uninterrupted_run_with_several_forward_passes() {
    finish(
        "selection off, 3 passes",
        &run_leg(Leg {
            forward_passes: 3,
            ..SELECTION_OFF
        }),
    );
}
