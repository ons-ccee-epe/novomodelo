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
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use novomodelo_core::{System, TrainingEvent};
use novomodelo_io::config::{
    CheckpointingConfig, SelectionMethod, StoppingRuleConfig, TrainingSelection,
};
use novomodelo_io::output::policy::{PolicyCheckpoint, read_policy_checkpoint};
use novomodelo_sddp::policy::full_fcf_load::{FullFcfLoadKind, check_full_fcf_load};
use novomodelo_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::{
    CapturedBasis, DEFAULT_COST_SCALE_FACTOR, SddpError, StudySetup, TrainingResult,
};
use novomodelo_solver::{ActiveSolver, BasisStatus};

use common::selection_fingerprint::SelectionFingerprint;
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
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(
            &mut solver,
            &StubComm,
            1,
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

/// Bound on two solves of one LP that start from different solver states.
const LOWER_BOUND_REL_TOL: f64 = 1e-6;
const RESUMED_ITERATIONS: std::ops::RangeInclusive<u64> = 4..=6;

struct RunOutcome {
    fingerprint: SelectionFingerprint,
    resolved_cost_scale_factor: f64,
    simplex_iterations: BTreeMap<(u64, &'static str, Option<i32>), u64>,
    final_checkpoint: PolicyCheckpoint,
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
        &CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
        },
    )
    .expect("write_checkpoint must succeed");
    read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint")
}

impl RunOutcome {
    fn capture(system: &System, setup: &StudySetup, result: &TrainingResult) -> Self {
        let mut simplex_iterations = BTreeMap::new();
        for entry in result
            .solver_stats_log
            .iter()
            .filter(|e| RESUMED_ITERATIONS.contains(&e.iteration))
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
        }
    }
}

fn selection_off_setup(iteration_limit: u32, forward_passes: u32) -> (System, StudySetup) {
    fresh_system_and_setup_with(&toy_case(), |config| {
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: iteration_limit,
        }]);
    })
}

/// One stored basis as the checkpoint writes it: column and row status codes
/// and the slot of each cut row.
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

/// Returns `(resumed, uninterrupted, restored)`: the resumed run trains 3
/// iterations, checkpoints, loads the checkpoint through the production resume
/// path and trains to 6.
fn resumed_and_uninterrupted(forward_passes: u32) -> (RunOutcome, RunOutcome, RestoredStore) {
    let (full_system, mut full_setup) = selection_off_setup(6, forward_passes);
    let full_result = train(&mut full_setup, None);
    assert_eq!(full_result.iterations, 6);
    let uninterrupted = RunOutcome::capture(&full_system, &full_setup, &full_result);

    let (system, mut first_setup) = selection_off_setup(3, forward_passes);
    let first_result = train(&mut first_setup, None);
    assert_eq!(first_result.iterations, 3);
    let out = tempfile::tempdir().expect("tempdir");
    let policy_dir = out.path().join(&first_setup.policy_path);
    write_checkpoint(
        &policy_dir,
        &first_setup,
        &system,
        &first_result,
        &CheckpointParams {
            max_iterations: first_setup.loop_params.max_iterations,
            forward_passes: first_setup.loop_params.forward_passes,
            seed: first_setup.loop_params.seed,
        },
    )
    .expect("write_checkpoint must succeed");

    let (system, mut resumed_setup) = selection_off_setup(6, forward_passes);
    check_full_fcf_load(
        FullFcfLoadKind::Resume,
        &policy_dir,
        &system,
        &resumed_setup,
        &mut |_msg: &str| {},
    )
    .expect("the checkpoint must load")
    .apply_to_training(&mut resumed_setup);
    let interrupted = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
    let restored = RestoredStore::capture(&resumed_setup, &interrupted, forward_passes);
    let resumed_result = train(&mut resumed_setup, None);
    assert_eq!(resumed_result.iterations, 6);
    (
        RunOutcome::capture(&system, &resumed_setup, &resumed_result),
        uninterrupted,
        restored,
    )
}

fn assert_same_run(resumed: &RunOutcome, uninterrupted: &RunOutcome, restored: &RestoredStore) {
    for run in [resumed, uninterrupted] {
        assert_eq!(run.resolved_cost_scale_factor, DEFAULT_COST_SCALE_FACTOR);
    }
    assert!(
        restored.written.values().any(Option::is_some),
        "the interrupted run's checkpoint holds bases"
    );
    for (key, written) in &restored.written {
        assert_eq!(
            &restored.seeded[key], written,
            "(node, scenario) {key:?}: the restored basis is the one the interrupted run wrote"
        );
    }

    assert_eq!(
        resumed.fingerprint.cuts.len(),
        uninterrupted.fingerprint.cuts.len()
    );
    for (pool, (r, u)) in resumed
        .fingerprint
        .cuts
        .iter()
        .zip(&uninterrupted.fingerprint.cuts)
        .enumerate()
    {
        assert_eq!(r.active, u.active, "pool {pool}: activity flags");
        assert_eq!(
            r.intercept_bits, u.intercept_bits,
            "pool {pool}: intercepts"
        );
        assert_eq!(
            r.coefficient_bits, u.coefficient_bits,
            "pool {pool}: coefficients"
        );
    }
    assert_eq!(
        resumed.fingerprint.provenance, uninterrupted.fingerprint.provenance,
        "generating iteration and forward pass of every slot"
    );
    assert_eq!(
        resumed.fingerprint.binding, uninterrupted.fingerprint.binding,
        "binding counters of every slot"
    );
    assert_eq!(
        resumed.fingerprint.bounds.final_ub_bits, uninterrupted.fingerprint.bounds.final_ub_bits,
        "final upper bound"
    );
    assert_eq!(
        resumed.simplex_iterations, uninterrupted.simplex_iterations,
        "forward and backward simplex iterations of the resumed iterations"
    );

    let (r_lb, u_lb) = (
        &resumed.fingerprint.bounds.lower_bound_bits,
        &uninterrupted.fingerprint.bounds.lower_bound_bits,
    );
    assert_eq!(r_lb.len(), u_lb.len(), "lower-bound history length");
    for (i, (r, u)) in r_lb.iter().zip(u_lb).enumerate() {
        let (r, u) = (f64::from_bits(*r), f64::from_bits(*u));
        assert_lower_bounds_agree(&format!("lower bound of iteration {}", i + 1), r, u);
    }

    assert_same_checkpoint(&resumed.final_checkpoint, &uninterrupted.final_checkpoint);
}

fn assert_lower_bounds_agree(what: &str, resumed: f64, uninterrupted: f64) {
    assert!(
        (resumed - uninterrupted).abs()
            <= LOWER_BOUND_REL_TOL * 1.0_f64.max(resumed.abs()).max(uninterrupted.abs()),
        "{what}: resumed {resumed}, uninterrupted {uninterrupted}"
    );
}

fn assert_same_checkpoint(resumed: &PolicyCheckpoint, uninterrupted: &PolicyCheckpoint) {
    let (r_producer, u_producer) = (&resumed.metadata.producer, &uninterrupted.metadata.producer);
    assert_eq!(
        r_producer.lower_bound_history.len(),
        u_producer.lower_bound_history.len(),
        "checkpoint lower-bound history length"
    );
    for (i, (r, u)) in r_producer
        .lower_bound_history
        .iter()
        .zip(&u_producer.lower_bound_history)
        .enumerate()
    {
        assert_lower_bounds_agree(
            &format!("checkpoint lower bound of iteration {}", i + 1),
            *r,
            *u,
        );
    }
    assert_lower_bounds_agree(
        "checkpoint final lower bound",
        r_producer.final_lower_bound,
        u_producer.final_lower_bound,
    );

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
    assert!(
        r.stage_cuts
            .iter()
            .chain(&u.stage_cuts)
            .all(|pool| pool.warm_start_count == 0),
        "neither run starts from a loaded prefix"
    );
}

#[test]
fn resumed_run_equals_the_uninterrupted_run_with_one_forward_pass() {
    let (resumed, uninterrupted, restored) = resumed_and_uninterrupted(1);
    assert_same_run(&resumed, &uninterrupted, &restored);
}

#[test]
fn resumed_run_equals_the_uninterrupted_run_with_several_forward_passes() {
    let (resumed, uninterrupted, restored) = resumed_and_uninterrupted(3);
    assert_same_run(&resumed, &uninterrupted, &restored);
}
