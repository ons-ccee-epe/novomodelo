//! The full-FCF load entry: directory lookup, check and apply against a
//! checkpoint written by the production writer.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use novomodelo_core::System;
use novomodelo_core::scenario::ScenarioSource;
use novomodelo_io::config::{
    BoundaryPolicy, Config, SelectionMethod, SimulationRowSet, StoppingRuleConfig,
    TrainingSelection,
};
use novomodelo_io::{OwnedPolicyCutRecord, read_policy_checkpoint};
use novomodelo_sddp::cut::{CutPool, CutSet, SlotOrigin};
use novomodelo_sddp::hydro_models::prepare_hydro_models;
use novomodelo_sddp::policy::full_fcf_load::{
    FullFcfLoadError, FullFcfLoadKind, check_full_fcf_load, locate_policy_dir,
};
use novomodelo_sddp::policy::orchestration::{
    CheckpointParams, partial_policy_dir, write_checkpoint,
};
use novomodelo_sddp::setup::prepare_stochastic;
use novomodelo_sddp::{
    BoundaryReconciliation, ErrorClass, SddpError, StudySetup, inject_boundary_cuts,
    reconcile_boundary_policy,
};
use novomodelo_solver::ActiveSolver;
use tempfile::TempDir;

mod common;
use common::{StubComm, truncated_toy_case};

const ITERATIONS: u32 = 3;

fn d01_case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch")
}

fn build_setup() -> (StudySetup, System) {
    build_setup_with(1, ITERATIONS)
}

fn build_setup_with(forward_passes: u32, iteration_limit: u32) -> (StudySetup, System) {
    let case_dir = d01_case_dir();
    let mut config = novomodelo_io::parse_config(&case_dir.join("config.json")).expect("config");
    config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: iteration_limit,
    }]);
    let system = novomodelo_io::load_case(&case_dir).expect("load_case");
    let prep = prepare_stochastic(
        system,
        &case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic");
    let hydro_models =
        prepare_hydro_models(&prep.system, &case_dir, false).expect("prepare_hydro_models");
    let setup = StudySetup::new(
        &prep.system,
        &config,
        prep.stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup::new");
    (setup, prep.system)
}

struct Trained {
    output: TempDir,
    completed_iterations: u64,
    final_lower_bound: f64,
}

fn train_and_write(setup: &mut StudySetup, system: &System) -> TempDir {
    let mut solver = ActiveSolver::new().expect("solver");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train");
    assert!(outcome.error.is_none());

    let output = TempDir::new().expect("tempdir");
    let policy_dir = output.path().join(&setup.policy_path);
    std::fs::create_dir_all(&policy_dir).expect("policy dir");
    let params = CheckpointParams {
        max_iterations: setup.loop_params.max_iterations,
        forward_passes: setup.loop_params.forward_passes,
        seed: setup.loop_params.seed,
    };
    write_checkpoint(&policy_dir, setup, system, &outcome.result, &params)
        .expect("write_checkpoint");
    output
}

fn train_and_write_checkpoint() -> Trained {
    let (mut setup, system) = build_setup();
    let output = train_and_write(&mut setup, &system);
    let producer = read_policy_checkpoint(&output.path().join(&setup.policy_path))
        .expect("read checkpoint")
        .metadata
        .producer;
    Trained {
        output,
        completed_iterations: u64::from(producer.completed_iterations),
        final_lower_bound: producer.final_lower_bound,
    }
}

#[test]
fn missing_policy_directory_names_the_directory_and_what_the_load_needs() {
    let (setup, _system) = build_setup();
    let output = TempDir::new().expect("tempdir");
    let dir = output.path().join(&setup.policy_path);
    for (kind, sentence) in [
        (
            FullFcfLoadKind::WarmStart,
            "Cannot warm-start without a prior policy.",
        ),
        (
            FullFcfLoadKind::Resume,
            "Cannot resume without a prior checkpoint.",
        ),
        (
            FullFcfLoadKind::SimulationOnly,
            "Cannot run simulation-only mode without a trained policy.",
        ),
    ] {
        let err = locate_policy_dir(kind, output.path(), &setup).expect_err("no directory");
        assert!(matches!(
            err,
            FullFcfLoadError::MissingPolicyDirectory { .. }
        ));
        assert_eq!(
            err.to_string(),
            format!("Policy directory not found: {}. {sentence}", dir.display())
        );
    }
}

#[test]
fn resume_apply_installs_the_checkpoint_cuts_and_the_start_iteration() {
    let trained = train_and_write_checkpoint();
    for kind in [FullFcfLoadKind::Resume, FullFcfLoadKind::WarmStart] {
        let (mut setup, system) = build_setup();
        assert_eq!(setup.fcf.total_active_cuts(), 0);
        let policy_dir = locate_policy_dir(kind, trained.output.path(), &setup).expect("locate");
        let checked =
            check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {}).expect("check");
        assert_eq!(checked.completed_iterations(), trained.completed_iterations);
        checked.apply_to_training(&mut setup);
        let loaded_cuts = setup.fcf.total_active_cuts();
        assert!(loaded_cuts > 0);
        let mut solver = ActiveSolver::new().expect("solver");
        let outcome = setup
            .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(outcome.error.is_none());
        // Resume starts past the iteration limit, so it adds no cuts.
        assert_eq!(
            setup.fcf.total_active_cuts() == loaded_cuts,
            kind == FullFcfLoadKind::Resume
        );
    }
}

#[test]
fn loaded_activity_clock_follows_the_load_kind() {
    let trained = train_and_write_checkpoint();
    for kind in [FullFcfLoadKind::Resume, FullFcfLoadKind::WarmStart] {
        let mut latest_loaded_iteration = 0;
        let (mut setup, system) = build_setup();
        let policy_dir = locate_policy_dir(kind, trained.output.path(), &setup).expect("locate");
        check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {})
            .expect("check")
            .apply_to_training(&mut setup);
        for (pool_id, pool) in setup.fcf.pools.iter().enumerate() {
            for slot in
                (0..pool.populated()).filter(|&slot| pool.role(slot).origin != SlotOrigin::Vacant)
            {
                assert_eq!(
                    pool.is_prefix_slot(slot),
                    kind == FullFcfLoadKind::WarmStart,
                    "{kind:?} pool {pool_id} slot {slot}: prefix role"
                );
                let meta = pool.metadata(slot);
                let expected_clock = match kind {
                    FullFcfLoadKind::Resume => meta.iteration_generated,
                    _ => 0,
                };
                assert_eq!(
                    (meta.active_count, meta.last_active_iter),
                    (0, expected_clock),
                    "{kind:?} pool {pool_id} slot {slot}: activity clock"
                );
                latest_loaded_iteration = latest_loaded_iteration.max(meta.iteration_generated);
            }
        }
        assert!(
            latest_loaded_iteration >= 1,
            "{kind:?}: a loaded cut must carry a generating iteration, or the two kinds are not told apart"
        );
    }
}

#[test]
fn warm_started_run_exports_loaded_cuts_with_their_iterations() {
    let (mut source_setup, source_system) = build_setup_with(2, ITERATIONS);
    let source_out = train_and_write(&mut source_setup, &source_system);
    let source = read_policy_checkpoint(&source_out.path().join(&source_setup.policy_path))
        .expect("read source checkpoint");

    let (mut setup, system) = build_setup_with(2, 2);
    let kind = FullFcfLoadKind::WarmStart;
    let policy_dir = locate_policy_dir(kind, source_out.path(), &setup).expect("locate");
    check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {})
        .expect("check")
        .apply_to_training(&mut setup);
    let warm_out = train_and_write(&mut setup, &system);
    let warm = read_policy_checkpoint(&warm_out.path().join(&setup.policy_path))
        .expect("read warm-started checkpoint");

    let coefficient_bits = |cut: &OwnedPolicyCutRecord| -> Vec<u64> {
        cut.coefficients.iter().map(|c| c.to_bits()).collect()
    };
    assert_eq!(source.stage_cuts.len(), setup.fcf.pools.len());
    assert_eq!(warm.stage_cuts.len(), setup.fcf.pools.len());
    for (pool_id, ((pool, source_pool), warm_pool)) in setup
        .fcf
        .pools
        .iter()
        .zip(&source.stage_cuts)
        .zip(&warm.stage_cuts)
        .enumerate()
    {
        let stride = u64::from(pool.visit_stride);
        if !warm_pool.cuts.is_empty() {
            assert_eq!(stride, 2, "pool {pool_id}: visit stride");
        }
        assert!(warm_pool.cuts.len() >= source_pool.cuts.len());
        for (slot, (loaded, exported)) in source_pool.cuts.iter().zip(&warm_pool.cuts).enumerate() {
            assert_eq!(
                (exported.iteration, exported.forward_pass_index),
                (loaded.iteration, loaded.forward_pass_index),
                "pool {pool_id} slot {slot}: generating iteration and forward pass"
            );
            assert_eq!(
                coefficient_bits(exported),
                coefficient_bits(loaded),
                "pool {pool_id} slot {slot}: coefficients"
            );
        }
        for (slot, cut) in warm_pool.cuts.iter().enumerate() {
            assert_ne!(cut.iteration, u32::MAX, "pool {pool_id} slot {slot}");
            assert!(
                (1..=ITERATIONS).contains(&cut.iteration),
                "pool {pool_id} slot {slot}: iteration {}",
                cut.iteration
            );
        }
    }

    assert!(
        source
            .stage_cuts
            .iter()
            .flat_map(|pool| &pool.cuts)
            .any(|cut| cut.forward_pass_index == 1),
        "the source run must have trained a second forward pass"
    );
    assert!(
        source
            .stage_cuts
            .iter()
            .zip(&warm.stage_cuts)
            .any(|(source_pool, warm_pool)| warm_pool.cuts.len() > source_pool.cuts.len()),
        "the warm-started run must have generated cuts"
    );
}

#[test]
fn a_resume_restores_the_prefix_length_of_a_warm_started_run() {
    let (mut source_setup, source_system) = build_setup_with(2, ITERATIONS);
    let source_out = train_and_write(&mut source_setup, &source_system);

    let (mut warm_setup, warm_system) = build_setup_with(2, 2);
    let warm_kind = FullFcfLoadKind::WarmStart;
    let source_dir = locate_policy_dir(warm_kind, source_out.path(), &warm_setup).expect("locate");
    check_full_fcf_load(
        warm_kind,
        &source_dir,
        &warm_system,
        &warm_setup,
        &mut |_| {},
    )
    .expect("check")
    .apply_to_training(&mut warm_setup);
    let warm_out = train_and_write(&mut warm_setup, &warm_system);

    let (mut resumed_setup, resumed_system) = build_setup_with(2, 2);
    let kind = FullFcfLoadKind::Resume;
    let policy_dir = locate_policy_dir(kind, warm_out.path(), &resumed_setup).expect("locate");
    check_full_fcf_load(
        kind,
        &policy_dir,
        &resumed_system,
        &resumed_setup,
        &mut |_| {},
    )
    .expect("check")
    .apply_to_training(&mut resumed_setup);

    assert!(
        warm_setup
            .fcf
            .pools
            .iter()
            .any(|pool| pool.prefix_len() > 0 && pool.generated() > pool.prefix_len()),
        "the warm-started run must hold a prefix and cuts of its own"
    );
    for (pool_id, (resumed, warm)) in resumed_setup
        .fcf
        .pools
        .iter()
        .zip(&warm_setup.fcf.pools)
        .enumerate()
    {
        assert_eq!(resumed.prefix_len(), warm.prefix_len(), "pool {pool_id}");
        assert_eq!(resumed.slot_base, warm.slot_base, "pool {pool_id}");
        assert_eq!(resumed.populated(), warm.populated(), "pool {pool_id}");
        let prefix_slots: Vec<usize> = (0..resumed.populated())
            .filter(|&slot| resumed.is_prefix_slot(slot))
            .collect();
        assert_eq!(
            prefix_slots,
            (0..warm.prefix_len()).collect::<Vec<_>>(),
            "pool {pool_id}: the slots that are loaded"
        );
        for slot in 0..resumed.populated() {
            assert_eq!(
                resumed.role(slot),
                warm.role(slot),
                "pool {pool_id} slot {slot}: role"
            );
        }
    }
}

/// 1dtoy with `forward_passes` and `iteration_limit`, then `edit`.
fn toy_setup(
    forward_passes: u32,
    iteration_limit: u32,
    edit: impl FnOnce(&mut Config),
) -> (System, StudySetup) {
    common::fresh_system_and_setup_with(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy"),
        |config| {
            config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
            config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
                limit: iteration_limit,
            }]);
            edit(config);
        },
    )
}

/// 1dtoy with two forward passes and an active-cut budget of 6, the budget the
/// only reader of binding activity.
fn budget_setup(iteration_limit: u32) -> (System, StudySetup) {
    toy_setup(2, iteration_limit, with_budget)
}

fn with_budget(config: &mut Config) {
    config.training.cut_selection.max_active_per_stage = Some(6);
}

fn load(kind: FullFcfLoadKind, output: &TempDir, setup: &mut StudySetup, system: &System) {
    let policy_dir = locate_policy_dir(kind, output.path(), setup).expect("locate");
    check_full_fcf_load(kind, &policy_dir, system, setup, &mut |_| {})
        .expect("check")
        .apply_to_training(setup);
}

/// Asserts that every written slot of `loaded` holds the binding counters of
/// the same slot of `written`, and returns whether some slot had bound.
fn assert_counters_restored(loaded: &StudySetup, written: &StudySetup) -> bool {
    let mut bound = false;
    assert_eq!(loaded.fcf.pools.len(), written.fcf.pools.len());
    for (pool_id, (loaded, written)) in loaded.fcf.pools.iter().zip(&written.fcf.pools).enumerate()
    {
        assert_eq!(loaded.populated(), written.populated(), "pool {pool_id}");
        for slot in
            (0..written.populated()).filter(|&slot| written.role(slot).origin != SlotOrigin::Vacant)
        {
            let (l, w) = (loaded.metadata(slot), written.metadata(slot));
            assert_eq!(
                (l.active_count, l.last_active_iter),
                (w.active_count, w.last_active_iter),
                "pool {pool_id} slot {slot}: binding counters"
            );
            bound |= w.active_count > 0;
        }
    }
    bound
}

#[test]
fn a_resumed_pool_restores_the_binding_counters_of_its_checkpoint() {
    let (system, mut written) = budget_setup(6);
    let output = train_and_write(&mut written, &system);
    assert!(
        read_policy_checkpoint(&output.path().join(&written.policy_path))
            .expect("read checkpoint")
            .metadata
            .producer
            .activity_counters
    );

    let (system, mut resumed) = budget_setup(6);
    load(FullFcfLoadKind::Resume, &output, &mut resumed, &system);
    assert!(
        assert_counters_restored(&resumed, &written),
        "some cut of the written run must have bound"
    );
}

/// A warm start reads no counter from a checkpoint that carries them, and a
/// resume of the warm-started run restores the counters its prefix cuts
/// gathered on that run's clock.
#[test]
fn a_warm_start_restarts_the_binding_counters_and_its_resume_restores_them() {
    let (system, mut source) = budget_setup(6);
    let source_out = train_and_write(&mut source, &system);

    let (system, mut warm) = budget_setup(2);
    load(FullFcfLoadKind::WarmStart, &source_out, &mut warm, &system);
    let mut prefix_slots = 0;
    for (pool_id, pool) in warm.fcf.pools.iter().enumerate() {
        for slot in (0..pool.populated()).filter(|&slot| pool.is_prefix_slot(slot)) {
            let meta = pool.metadata(slot);
            assert_eq!(
                (meta.active_count, meta.last_active_iter),
                (0, 0),
                "pool {pool_id} slot {slot}: a warm-started cut's counters"
            );
            prefix_slots += 1;
        }
    }
    assert!(prefix_slots > 0, "the warm start must load cuts");
    assert!(
        source
            .fcf
            .pools
            .iter()
            .any(|pool| (0..pool.populated()).any(|slot| pool.metadata(slot).active_count > 0)),
        "some cut of the source run must have bound, or the warm start reads nothing it could"
    );

    let warm_out = train_and_write(&mut warm, &system);
    let (system, mut resumed) = budget_setup(2);
    load(FullFcfLoadKind::Resume, &warm_out, &mut resumed, &system);
    assert_counters_restored(&resumed, &warm);
    assert!(
        warm.fcf.pools.iter().any(|pool| (0..pool.populated())
            .any(|slot| pool.is_prefix_slot(slot) && pool.metadata(slot).active_count > 0)),
        "some prefix cut of the warm-started run must have bound"
    );
}

#[test]
fn simulation_only_check_returns_the_recorded_bounds_without_frozen_templates() {
    let trained = train_and_write_checkpoint();
    let (setup, system) = build_setup();
    let kind = FullFcfLoadKind::SimulationOnly;
    let policy_dir = locate_policy_dir(kind, trained.output.path(), &setup).expect("locate");
    let checked =
        check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {}).expect("check");
    let (fcf, result) = checked.into_simulation_policy();
    assert_eq!(result.iterations, trained.completed_iterations);
    assert_eq!(result.final_lb, trained.final_lower_bound);
    assert!(result.frozen_templates.is_none());
    assert_eq!(fcf.state_dimension, setup.fcf.state_dimension);
}

fn relative_listing(root: &Path) -> Vec<PathBuf> {
    let mut listing = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path.clone());
            }
            listing.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
    listing.sort();
    listing
}

#[test]
fn locate_policy_dir_accepts_a_policy_caught_between_renames() {
    let trained = train_and_write_checkpoint();
    let (setup, system) = build_setup();
    let output_dir = trained.output.path();
    let policy_dir = output_dir.join(&setup.policy_path);
    let mut staging = policy_dir.clone().into_os_string();
    staging.push(".staging");
    std::fs::rename(&policy_dir, &staging).unwrap();
    let before = relative_listing(output_dir);

    let kind = FullFcfLoadKind::SimulationOnly;
    let located = locate_policy_dir(kind, output_dir, &setup).expect("locate");
    assert_eq!(located, policy_dir);
    let checked = check_full_fcf_load(kind, &located, &system, &setup, &mut |_| {}).expect("check");
    let (_fcf, result) = checked.into_simulation_policy();
    assert_eq!(result.iterations, trained.completed_iterations);
    assert_eq!(relative_listing(output_dir), before);
}

fn with_level1(window: Option<u32>) -> impl FnOnce(&mut Config) {
    move |config: &mut Config| {
        config.training.cut_selection.selection = Some(SelectionMethod::Level1 {
            check_frequency: NonZeroU32::MIN,
            lookback_iterations: window.and_then(NonZeroU32::new),
        });
    }
}

fn with_dynamic(config: &mut Config) {
    config.training.cut_selection.selection = Some(SelectionMethod::Dynamic {
        start_iteration: NonZeroU32::MIN,
        candidate_recency: None,
        seed_window: 2,
        max_added_per_round: NonZeroU32::MIN,
        violation_tolerance: None,
    });
}

/// The policy directory of a two-iteration 1dtoy run with `forward_passes`,
/// configured by `edit`, and the output directory holding it.
fn toy_checkpoint(forward_passes: u32, edit: impl FnOnce(&mut Config)) -> (TempDir, PathBuf) {
    let (system, mut setup) = toy_setup(forward_passes, 2, edit);
    let output = train_and_write(&mut setup, &system);
    let policy_dir = output.path().join(&setup.policy_path);
    (output, policy_dir)
}

/// The refusal a resume of `policy_dir` by a 1dtoy run with `forward_passes`,
/// configured by `edit`, gets, after a warm start of the same checkpoint by the
/// same run loads.
fn resume_refusal(
    policy_dir: &Path,
    forward_passes: u32,
    edit: impl FnOnce(&mut Config),
) -> String {
    let (system, setup) = toy_setup(forward_passes, 4, edit);
    check_full_fcf_load(
        FullFcfLoadKind::WarmStart,
        policy_dir,
        &system,
        &setup,
        &mut |_| {},
    )
    .expect("a warm start reads none of the fields a resume needs");
    let err = check_full_fcf_load(
        FullFcfLoadKind::Resume,
        policy_dir,
        &system,
        &setup,
        &mut |_| {},
    )
    .expect_err("the resume must be refused");
    assert!(
        matches!(err, FullFcfLoadError::Refused(SddpError::Validation(_))),
        "{err}"
    );
    assert_eq!(err.class(), ErrorClass::IncompatiblePolicy);
    err.to_string()
}

fn resumes(policy_dir: &Path, forward_passes: u32, edit: impl FnOnce(&mut Config)) {
    let (system, setup) = toy_setup(forward_passes, 4, edit);
    check_full_fcf_load(
        FullFcfLoadKind::Resume,
        policy_dir,
        &system,
        &setup,
        &mut |_| {},
    )
    .expect("the checkpoint serves the resume");
}

#[test]
fn a_whole_checkpoint_resumes_under_the_settings_that_wrote_it() {
    let edit = |config: &mut Config| {
        with_level1(Some(2))(config);
        with_budget(config);
    };
    let (_output, policy_dir) = toy_checkpoint(2, edit);
    let producer = read_policy_checkpoint(&policy_dir)
        .expect("read checkpoint")
        .metadata
        .producer;
    assert_eq!(producer.forward_passes, 2);
    assert_eq!(
        producer.state_history,
        novomodelo_io::StateHistory::Window(NonZeroU32::new(2).unwrap())
    );
    assert!(producer.activity_counters && producer.basis_store_complete);
    resumes(&policy_dir, 2, edit);
}

#[test]
fn a_resume_refuses_another_forward_pass_count() {
    let (_output, policy_dir) = toy_checkpoint(2, |_| {});
    let message = resume_refusal(&policy_dir, 1, |_| {});
    assert!(
        message.contains(
            "resume needs the forward-pass count the checkpoint was written with: \
             training.selection.forward_passes is 1 here and 2 in the checkpoint"
        ),
        "{message}"
    );
}

#[test]
fn a_level1_resume_refuses_a_checkpoint_without_the_trial_state_history() {
    let (_output, policy_dir) = toy_checkpoint(2, |_| {});
    let message = resume_refusal(&policy_dir, 2, with_level1(None));
    assert!(
        message.contains(
            "resume needs the trial-state history of training.cut_selection.selection.method = \
             level1; the checkpoint holds none"
        ),
        "{message}"
    );
    assert!(message.contains("policy.mode = warm_start"), "{message}");
}

/// A stored window serves a run that keeps the same or fewer iterations' states,
/// never one that keeps more or every one.
#[test]
fn a_level1_resume_refuses_a_shorter_stored_window() {
    let (_output, policy_dir) = toy_checkpoint(2, with_level1(Some(2)));

    let every_state = resume_refusal(&policy_dir, 2, with_level1(None));
    assert!(
        every_state.contains(
            "without lookback_iterations; the checkpoint holds those of the last 2 iterations only"
        ),
        "{every_state}"
    );
    let longer = resume_refusal(&policy_dir, 2, with_level1(Some(3)));
    assert!(
        longer.contains(
            "resume needs the trial states of the last 3 iterations \
             (training.cut_selection.selection.lookback_iterations = 3); the checkpoint holds \
             those of the last 2 iterations only"
        ),
        "{longer}"
    );

    resumes(&policy_dir, 2, with_level1(Some(2)));
    resumes(&policy_dir, 2, with_level1(Some(1)));
}

#[test]
fn a_resume_with_a_reader_refuses_a_checkpoint_without_the_binding_counters() {
    let (_output, policy_dir) = toy_checkpoint(2, |_| {});
    for (edit, reader) in [
        (
            with_budget as fn(&mut Config),
            "training.cut_selection.max_active_per_stage",
        ),
        (
            with_dynamic,
            "training.cut_selection.selection.method = dynamic",
        ),
    ] {
        let message = resume_refusal(&policy_dir, 2, edit);
        assert!(
            message.contains(&format!(
                "resume needs the binding counters read by {reader}; the checkpoint holds none"
            )),
            "{message}"
        );
    }
}

/// The sibling `.partial` policy directory a failed run leaves, and the output
/// directory holding it: a two-iteration 1dtoy run with `forward_passes`,
/// configured by `edit`, whose checkpoint is not whole.
fn partial_toy_checkpoint(
    forward_passes: u32,
    edit: impl FnOnce(&mut Config),
) -> (TempDir, PathBuf) {
    let (system, mut setup) = toy_setup(forward_passes, 2, edit);
    let mut solver = ActiveSolver::new().expect("solver");
    let mut outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train");
    assert!(outcome.error.is_none());
    outcome.result.checkpoint_bases.complete = false;
    let output = TempDir::new().expect("tempdir");
    let policy_dir = output.path().join(&setup.policy_path);
    let params = CheckpointParams {
        max_iterations: setup.loop_params.max_iterations,
        forward_passes: setup.loop_params.forward_passes,
        seed: setup.loop_params.seed,
    };
    let written = write_checkpoint(&policy_dir, &setup, &system, &outcome.result, &params)
        .expect("write_checkpoint");
    assert_eq!(Some(written.clone()), partial_policy_dir(&policy_dir));
    assert!(
        !read_policy_checkpoint(&written)
            .expect("read checkpoint")
            .metadata
            .producer
            .basis_store_complete
    );
    (output, written)
}

#[test]
fn a_resume_refuses_a_checkpoint_whose_bases_are_not_whole() {
    let (_output, written) = partial_toy_checkpoint(2, |_| {});

    let message = resume_refusal(&written, 2, |_| {});
    assert_eq!(
        message,
        "configuration validation error: resume needs a whole checkpoint; the checkpoint's \
         per-scenario bases are not whole (it was written when a run failed); resume from the \
         last whole checkpoint or continue with policy.mode = warm_start"
    );
}

/// When several rejects hold at once the first of this order is reported: the
/// forward-pass count, the trial-state history, its window, the binding
/// counters, the bases. A history that is missing and a window that is too short
/// exclude each other.
#[test]
fn a_resume_reports_the_first_refusal_in_its_fixed_order() {
    let level1_with_budget = |window| {
        move |config: &mut Config| {
            with_level1(window)(config);
            with_budget(config);
        }
    };
    let (_no_history, no_history) = partial_toy_checkpoint(2, |_| {});
    let (_windowed, windowed) = partial_toy_checkpoint(2, with_level1(Some(2)));

    for (checkpoint, passes, window, fragment) in [
        (&no_history, 1, None, "resume needs the forward-pass count"),
        (
            &no_history,
            2,
            None,
            "resume needs the trial-state history of",
        ),
        (&windowed, 2, None, "without lookback_iterations"),
        (&windowed, 2, Some(2), "resume needs the binding counters"),
    ] {
        let message = resume_refusal(checkpoint, passes, level1_with_budget(window));
        assert!(message.contains(fragment), "{fragment}: {message}");
    }
    let message = resume_refusal(&windowed, 2, with_level1(Some(2)));
    assert!(
        message.contains("resume needs a whole checkpoint"),
        "{message}"
    );
}

/// d01 has two stages, so no pool lies between the root and the leaf: a `level1`
/// run stores no trial state there, and its resume needs none.
#[test]
fn a_level1_resume_of_a_study_without_an_interior_pool_needs_no_history() {
    let trained = train_and_write_checkpoint();
    let (system, setup) = common::fresh_system_and_setup_with(&d01_case_dir(), |config| {
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 1 });
        with_level1(None)(config);
    });
    let policy_dir = trained.output.path().join(&setup.policy_path);
    assert_eq!(
        read_policy_checkpoint(&policy_dir)
            .expect("read checkpoint")
            .metadata
            .producer
            .state_history,
        novomodelo_io::StateHistory::None
    );
    check_full_fcf_load(
        FullFcfLoadKind::Resume,
        &policy_dir,
        &system,
        &setup,
        &mut |_| {},
    )
    .expect("a study without an interior pool needs no trial-state history");
}

const DEACTIVATING_ITERATIONS: u32 = 6;

/// A two-pass 1dtoy `level1` run under a budget, and its checkpoint: some pool of
/// the run holds a cut outside its active set. Returns the first such pool with
/// its cut count and its active count.
fn deactivating_checkpoint() -> (TempDir, PathBuf, (usize, usize, usize)) {
    deactivating_checkpoint_under(|config| {
        with_level1(None)(config);
        with_budget(config);
    })
}

/// [`deactivating_checkpoint`] for a two-pass run configured by `edit`.
fn deactivating_checkpoint_under(
    edit: impl FnOnce(&mut Config),
) -> (TempDir, PathBuf, (usize, usize, usize)) {
    let (system, mut setup) = toy_setup(2, DEACTIVATING_ITERATIONS, edit);
    let output = train_and_write(&mut setup, &system);
    let policy_dir = output.path().join(&setup.policy_path);
    let first = setup
        .fcf
        .pools
        .iter()
        .enumerate()
        .map(|(pool_id, pool)| (pool_id, pool.count_in(CutSet::All), pool.active_count()))
        .find(|(_, held, active)| held != active)
        .expect("power: the run must leave a cut outside its active set");
    (output, policy_dir, first)
}

fn load_under(
    kind: FullFcfLoadKind,
    policy_dir: &Path,
    edit: impl FnOnce(&mut Config),
) -> Result<(), FullFcfLoadError> {
    let (system, setup) = toy_setup(2, DEACTIVATING_ITERATIONS + 2, edit);
    check_full_fcf_load(kind, policy_dir, &system, &setup, &mut |_| {}).map(drop)
}

fn dynamic_simulating(cut_set: SimulationRowSet) -> impl FnOnce(&mut Config) {
    move |config| {
        with_dynamic(config);
        config.simulation.cut_set = cut_set;
    }
}

/// A `dynamic` run that simulates every cut refuses `kind`'s load of the
/// deactivating checkpoint, and loads it under `simulation.cut_set = active`.
fn assert_dynamic_all_is_refused(kind: FullFcfLoadKind) {
    let (_output, policy_dir, first) = deactivating_checkpoint();
    assert_dynamic_all_is_refused_over(kind, &policy_dir, first);
}

fn assert_dynamic_all_is_refused_over(
    kind: FullFcfLoadKind,
    policy_dir: &Path,
    (pool_id, held, active): (usize, usize, usize),
) {
    load_under(
        kind,
        policy_dir,
        dynamic_simulating(SimulationRowSet::Active),
    )
    .expect("the active cuts are what a dynamic lazy solve reads");
    let err = load_under(kind, policy_dir, dynamic_simulating(SimulationRowSet::All))
        .expect_err("the load must be refused");

    assert!(
        matches!(err, FullFcfLoadError::Refused(SddpError::Validation(_))),
        "{kind:?}: {err}"
    );
    assert_eq!(err.class(), ErrorClass::IncompatiblePolicy, "{kind:?}");
    let message = err.to_string();
    for fragment in [
        "training.cut_selection.selection.method = dynamic cannot simulate \
         simulation.cut_set = all over this policy: "
            .to_string(),
        format!("pool {pool_id} holds {held} cuts of which {active} are active"),
        "set simulation.cut_set = active or use the level1 method".to_string(),
    ] {
        assert!(message.contains(&fragment), "{kind:?}: {message}");
    }
}

#[test]
fn a_dynamic_simulation_only_run_refuses_to_simulate_all_over_deactivated_cuts() {
    assert_dynamic_all_is_refused(FullFcfLoadKind::SimulationOnly);
}

#[test]
fn a_dynamic_warm_start_refuses_to_simulate_all_over_deactivated_cuts() {
    assert_dynamic_all_is_refused(FullFcfLoadKind::WarmStart);
}

#[test]
fn a_dynamic_resume_refuses_to_simulate_all_over_deactivated_cuts() {
    assert_dynamic_all_is_refused(FullFcfLoadKind::Resume);
}

/// A row the cap alone evicted is `Generated`, which `all` holds whatever its flag:
/// a simulation-only load and a resume are refused, while a warm start loads
/// every row as `Prefix`, and an unscored inactive one is in neither set.
#[test]
fn a_dynamic_load_refuses_to_simulate_all_over_cap_evicted_cuts() {
    let (_output, policy_dir, first) = deactivating_checkpoint_under(with_budget);
    for kind in [FullFcfLoadKind::SimulationOnly, FullFcfLoadKind::Resume] {
        assert_dynamic_all_is_refused_over(kind, &policy_dir, first);
    }
    load_under(
        FullFcfLoadKind::WarmStart,
        &policy_dir,
        dynamic_simulating(SimulationRowSet::All),
    )
    .expect("a warm start loads an unscored inactive row as a prefix row");
}

/// The refusal needs all three of: the `dynamic` method, a simulation of `all`,
/// and a pool holding a cut outside its active set.
#[test]
fn a_load_is_refused_for_a_dynamic_all_simulation_over_deactivated_cuts_only() {
    let (_output, policy_dir, _) = deactivating_checkpoint();
    for kind in [FullFcfLoadKind::WarmStart, FullFcfLoadKind::Resume] {
        load_under(kind, &policy_dir, |config| {
            dynamic_simulating(SimulationRowSet::All)(config);
            config.simulation.enabled = false;
        })
        .unwrap_or_else(|err| panic!("{kind:?}: a run that does not simulate loads: {err}"));
    }
    for kind in [
        FullFcfLoadKind::WarmStart,
        FullFcfLoadKind::Resume,
        FullFcfLoadKind::SimulationOnly,
    ] {
        load_under(kind, &policy_dir, |config| {
            with_level1(None)(config);
            with_budget(config);
        })
        .unwrap_or_else(|err| panic!("{kind:?}: a level1 simulation of all loads: {err}"));
    }

    let (_dynamic_output, dynamic_dir) = toy_checkpoint(2, with_dynamic);
    for kind in [
        FullFcfLoadKind::WarmStart,
        FullFcfLoadKind::Resume,
        FullFcfLoadKind::SimulationOnly,
    ] {
        load_under(
            kind,
            &dynamic_dir,
            dynamic_simulating(SimulationRowSet::All),
        )
        .unwrap_or_else(|err| panic!("{kind:?}: a dynamic policy loads: {err}"));
    }
}

/// What a `dynamic` run simulating `cut_set` gets from using pool `pool_id` of
/// `policy_dir`, a 1dtoy policy, as its `policy.boundary`.
fn reconcile_boundary_of(
    policy_dir: &Path,
    pool_id: usize,
    cut_set: SimulationRowSet,
) -> Result<BoundaryReconciliation, SddpError> {
    let case = truncated_toy_case(pool_id + 1);
    let boundary = BoundaryPolicy {
        path: policy_dir.to_str().expect("utf-8 path").to_string(),
        strict: false,
    };
    let (system, setup) = common::fresh_system_and_setup_with(case.path(), |config| {
        dynamic_simulating(cut_set)(config);
        config.policy.boundary = Some(boundary.clone());
    });
    reconcile_boundary_policy(&setup, &system, &boundary, case.path())
}

/// The boundary rows of a source pool keep the `scored` flag of the source, so
/// `all` includes a scored row whatever its active flag, and a `dynamic` lazy
/// solve cannot read it.
#[test]
fn a_dynamic_simulation_of_all_refuses_boundary_cuts_outside_the_active_set() {
    let (_output, policy_dir, (pool_id, held, active)) = deactivating_checkpoint();

    let loaded = reconcile_boundary_of(&policy_dir, pool_id, SimulationRowSet::Active)
        .expect("the active rows are what a dynamic lazy solve reads");
    assert!(
        loaded
            .cuts
            .iter()
            .any(|record| record.scored && !record.is_active),
        "power: the boundary must hold a scored inactive row"
    );

    let err = reconcile_boundary_of(&policy_dir, pool_id, SimulationRowSet::All)
        .expect_err("the boundary must be refused");
    assert!(matches!(err, SddpError::Validation(_)), "{err}");
    let message = err.to_string();
    for fragment in [
        "training.cut_selection.selection.method = dynamic cannot simulate \
         simulation.cut_set = all over this policy: "
            .to_string(),
        format!("the policy.boundary pool holds {held} cuts of which {active} are active"),
        "set simulation.cut_set = active or use the level1 method".to_string(),
    ] {
        assert!(message.contains(&fragment), "{message}");
    }
}

/// 1dtoy over its first `pool_id + 1` stages as a `level1` run under a budget
/// with two forward passes, whose `policy.boundary` is pool `pool_id` of
/// `boundary_dir`.
fn boundary_run_setup(
    boundary_dir: &Path,
    pool_id: usize,
    iteration_limit: u32,
) -> (TempDir, BoundaryPolicy, System, StudySetup) {
    let case = truncated_toy_case(pool_id + 1);
    let boundary = BoundaryPolicy {
        path: boundary_dir.to_str().expect("utf-8 path").to_string(),
        strict: false,
    };
    let (system, setup) = common::fresh_system_and_setup_with(case.path(), |config| {
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: iteration_limit,
        }]);
        with_level1(None)(config);
        with_budget(config);
        config.policy.boundary = Some(boundary.clone());
    });
    (case, boundary, system, setup)
}

/// The policy directory of a `level1` run under a budget over the first
/// `pool_id + 1` stages of 1dtoy whose `policy.boundary` is pool `pool_id` of
/// `boundary_dir`: its terminal pool holds the boundary rows with the flags of
/// their source, and it carries the binding counters a `dynamic` resume reads.
fn boundary_run_checkpoint(boundary_dir: &Path, pool_id: usize) -> (TempDir, PathBuf) {
    let (case, boundary, system, mut setup) = boundary_run_setup(boundary_dir, pool_id, 2);
    let reconciled = reconcile_boundary_policy(&setup, &system, &boundary, case.path())
        .expect("the boundary reconciles");
    inject_boundary_cuts(&mut setup, &reconciled.cuts).expect("inject");
    let output = train_and_write(&mut setup, &system);
    let terminal = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .expect("terminal pool");
    let pool = &setup.fcf.pools[terminal];
    assert_ne!(
        pool.count_in(CutSet::All),
        pool.active_count(),
        "power: the terminal pool must hold a boundary row outside its active set"
    );
    let policy_dir = output.path().join(&setup.policy_path);
    (output, policy_dir)
}

/// `policy.boundary` replaces the loaded terminal pool, and the boundary is
/// checked on its own: the pool a checkpoint holds there is not checked.
#[test]
fn a_boundary_replaces_the_loaded_terminal_pool_the_dynamic_check_reads() {
    let (_source, source_dir, (pool_id, _, _)) = deactivating_checkpoint();
    let (_checkpoint, checkpoint_dir) = boundary_run_checkpoint(&source_dir, pool_id);
    let (_plain, plain_dir) = toy_checkpoint(2, |_| {});

    let case = truncated_toy_case(pool_id + 1);
    let boundary = BoundaryPolicy {
        path: plain_dir.to_str().expect("utf-8 path").to_string(),
        strict: false,
    };
    let (system, setup) = common::fresh_system_and_setup_with(case.path(), |config| {
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        dynamic_simulating(SimulationRowSet::All)(config);
        config.policy.boundary = Some(boundary.clone());
    });
    reconcile_boundary_policy(&setup, &system, &boundary, case.path())
        .expect("every row of the plain policy is active");
    for kind in [FullFcfLoadKind::WarmStart, FullFcfLoadKind::Resume] {
        check_full_fcf_load(kind, &checkpoint_dir, &system, &setup, &mut |_| {}).unwrap_or_else(
            |err| panic!("{kind:?}: the boundary replaces the terminal pool: {err}"),
        );
    }
    let err = check_full_fcf_load(
        FullFcfLoadKind::SimulationOnly,
        &checkpoint_dir,
        &system,
        &setup,
        &mut |_| {},
    )
    .expect_err("a simulation-only run injects no boundary and simulates the loaded pool");
    assert!(matches!(err, FullFcfLoadError::Refused(_)), "{err}");
}

/// A resume of a run that declared `policy.boundary` keeps the binding counters
/// its checkpoint holds for the terminal rows and refuses a boundary whose rows
/// differ from them; a warm start of the same checkpoint takes any boundary and
/// restarts the counters.
#[test]
fn a_resume_refuses_a_boundary_that_no_longer_matches_its_checkpoint() {
    let (_source, source_dir, (pool_id, _, _)) = deactivating_checkpoint();
    let (_checkpoint, checkpoint_dir) = boundary_run_checkpoint(&source_dir, pool_id);
    let (_plain, plain_dir) = toy_checkpoint(2, |_| {});

    let load_then_inject = |kind: FullFcfLoadKind, boundary_dir: &Path| {
        let (case, boundary, system, mut setup) = boundary_run_setup(boundary_dir, pool_id, 3);
        check_full_fcf_load(kind, &checkpoint_dir, &system, &setup, &mut |_| {})
            .expect("the checkpoint loads")
            .apply_to_training(&mut setup);
        let terminal = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .expect("terminal pool");
        let loaded = setup.fcf.pools[terminal].clone();
        let reconciled = reconcile_boundary_policy(&setup, &system, &boundary, case.path())
            .expect("the boundary reconciles");
        let injected = inject_boundary_cuts(&mut setup, &reconciled.cuts);
        (injected, loaded, setup.fcf.pools[terminal].clone())
    };
    let counters = |pool: &CutPool| -> Vec<(u64, u64)> {
        (0..pool.populated())
            .map(|slot| {
                let meta = pool.metadata(slot);
                (meta.active_count, meta.last_active_iter)
            })
            .collect()
    };

    let (injected, loaded, resumed) = load_then_inject(FullFcfLoadKind::Resume, &source_dir);
    injected.expect("the boundary the interrupted run used is admitted");
    assert!(
        counters(&loaded).iter().any(|&(count, _)| count > 0),
        "power: the checkpoint's terminal rows must have bound"
    );
    assert_eq!(counters(&resumed), counters(&loaded));

    let (injected, _, _) = load_then_inject(FullFcfLoadKind::Resume, &plain_dir);
    let err = injected.expect_err("another boundary is refused");
    assert!(matches!(err, SddpError::Validation(_)), "{err}");
    let message = err.to_string();
    for fragment in [
        "resume needs the policy.boundary and modeling.cost_scale_factor the interrupted run used",
        &format!("(pool {pool_id}) differs from the checkpoint's at slot count"),
        "policy.mode = warm_start",
    ] {
        assert!(message.contains(fragment), "{message}");
    }

    let (injected, _, restarted) = load_then_inject(FullFcfLoadKind::WarmStart, &plain_dir);
    injected.expect("a warm start takes any boundary");
    assert!(counters(&restarted).iter().all(|&c| c == (0, 0)));
}

/// A row a capped source evicted was never scored, so `all` leaves it out as it
/// leaves out its flag: the boundary pool's two counts are equal and the pairing
/// loads.
#[test]
fn a_dynamic_simulation_of_all_loads_boundary_cuts_a_cap_evicted() {
    let (system, mut source) = toy_setup(1, 8, with_budget);
    let output = train_and_write(&mut source, &system);
    let pool_id = source
        .fcf
        .pools
        .iter()
        .position(|pool| pool.active_count() < pool.populated())
        .expect("power: the cap must evict a cut");

    let loaded = reconcile_boundary_of(
        &output.path().join(&source.policy_path),
        pool_id,
        SimulationRowSet::All,
    )
    .expect("a boundary row outside the active set that is unscored is in neither set");
    assert!(
        loaded
            .cuts
            .iter()
            .any(|record| !record.scored && !record.is_active),
        "power: the boundary must hold an unscored inactive row"
    );
}
