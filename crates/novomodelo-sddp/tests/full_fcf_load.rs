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
use novomodelo_io::config::{Config, SelectionMethod, StoppingRuleConfig, TrainingSelection};
use novomodelo_io::{OwnedPolicyCutRecord, read_policy_checkpoint};
use novomodelo_sddp::cut::SlotOrigin;
use novomodelo_sddp::hydro_models::prepare_hydro_models;
use novomodelo_sddp::policy::full_fcf_load::{
    FullFcfLoadError, FullFcfLoadKind, check_full_fcf_load, locate_policy_dir,
};
use novomodelo_sddp::policy::orchestration::{
    CheckpointParams, partial_policy_dir, write_checkpoint,
};
use novomodelo_sddp::setup::prepare_stochastic;
use novomodelo_sddp::{ErrorClass, SddpError, StudySetup};
use novomodelo_solver::ActiveSolver;
use tempfile::TempDir;

mod common;
use common::StubComm;

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

#[test]
fn a_resume_refuses_a_checkpoint_whose_bases_are_not_whole() {
    let (system, mut setup) = toy_setup(2, 2, |_| {});
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

    let message = resume_refusal(&written, 2, |_| {});
    assert_eq!(
        message,
        "configuration validation error: resume needs a whole checkpoint; the checkpoint's \
         per-scenario bases are not whole (it was written when a run failed); resume from the \
         last whole checkpoint or continue with policy.mode = warm_start"
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
