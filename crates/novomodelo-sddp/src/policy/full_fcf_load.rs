//! The single full-FCF policy load shared by the CLI and the Python bindings.
//!
//! A warm-start, resume or simulation-only run loads a stored checkpoint in
//! two steps: [`check_full_fcf_load`] reads, validates and decodes it without
//! touching the study, and the returned [`CheckedFullFcfLoad`] is then applied.
//! Every MPI rank runs the load itself; no step is collective.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use novomodelo_core::System;
use novomodelo_io::output::policy::{
    ResolvedCheckpoint, read_policy_checkpoint, resolve_policy_checkpoint,
};
use novomodelo_io::{
    EntitySlot, OutputError, OwnedPolicyCutRecord, ProducerBlock, StageCutsReadResult,
    StageStatesReadResult, StateHistory,
};

use crate::cut::CutSelectionStrategy;
use crate::cut::fcf::FutureCostFunction;
use crate::error::{ErrorClass, SddpError};
use crate::policy::policy_load::{
    FullFcf, PolicyStageManifest, StoredBasisLoad, UnusedStoredBases,
    build_basis_cache_from_checkpoint, convert_checkpoint_cuts_for_load, validate_policy_load,
};
use crate::setup::StudySetup;
use crate::training::session::interior_cut_nodes;
use crate::training::training::{LoadedPolicy, MissingScenarioBasis, TrainingResult};
use crate::workspace::workspace::CapturedBasis;

/// Which run consumes the loaded policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullFcfLoadKind {
    /// Train on top of a prior policy's cuts.
    WarmStart,
    /// Continue a prior training run from its checkpoint.
    Resume,
    /// Simulate from a trained policy without training.
    SimulationOnly,
}

impl FullFcfLoadKind {
    fn unmet_requirement(self) -> &'static str {
        match self {
            Self::WarmStart => "Cannot warm-start without a prior policy.",
            Self::Resume => "Cannot resume without a prior checkpoint.",
            Self::SimulationOnly => "Cannot run simulation-only mode without a trained policy.",
        }
    }
}

/// The step of the load that failed.
#[derive(Debug, thiserror::Error)]
pub enum FullFcfLoadError {
    /// The policy directory does not exist.
    #[error("Policy directory not found: {}. {}", path.display(), kind.unmet_requirement())]
    MissingPolicyDirectory {
        /// The load that needed the directory.
        kind: FullFcfLoadKind,
        /// The directory that was looked up.
        path: PathBuf,
    },
    /// The checkpoint could not be read.
    #[error("failed to read policy checkpoint: {source}")]
    Read {
        /// The underlying read failure.
        source: OutputError,
    },
    /// The checkpoint is incompatible with the study: cost scale, software,
    /// manifests, or state a resume reads that it does not hold.
    #[error(transparent)]
    Refused(SddpError),
    /// The future-cost function could not be built from the cuts.
    #[error("{source}")]
    FcfConstruction {
        /// The load that was building the FCF.
        kind: FullFcfLoadKind,
        /// The underlying construction failure.
        source: SddpError,
    },
}

impl FullFcfLoadError {
    /// The [`ErrorClass`] of this failure.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::MissingPolicyDirectory { .. } => ErrorClass::InvalidInput,
            Self::Read { source } => match source {
                OutputError::IoError { source, .. } if source.kind() != ErrorKind::NotFound => {
                    ErrorClass::Io
                }
                OutputError::IoError { .. }
                | OutputError::SerializationError { .. }
                | OutputError::SchemaError { .. }
                | OutputError::ManifestError { .. } => ErrorClass::IncompatiblePolicy,
                OutputError::ForeignEntry { .. } => ErrorClass::InvalidInput,
            },
            // The inner `SddpError::Validation` classifies as `InvalidInput`, but a
            // refused checkpoint is an incompatible policy.
            Self::Refused(_) | Self::FcfConstruction { .. } => ErrorClass::IncompatiblePolicy,
        }
    }
}

/// Locate the policy directory of `setup` under `output_dir`.
///
/// The directory counts as present when [`resolve_policy_checkpoint`] finds it
/// or a committed sibling copy, beside a link's target when it is a symbolic
/// link.
///
/// # Errors
///
/// - [`FullFcfLoadError::MissingPolicyDirectory`] when the directory is absent
///   and no sibling copy is committed.
/// - [`FullFcfLoadError::Read`] when a probe fails for a reason other than
///   absence.
pub fn locate_policy_dir(
    kind: FullFcfLoadKind,
    output_dir: &Path,
    setup: &StudySetup,
) -> Result<PathBuf, FullFcfLoadError> {
    let path = output_dir.join(&setup.policy_path);
    match resolve_policy_checkpoint(&path) {
        Ok(ResolvedCheckpoint::NoDirectory) => {
            Err(FullFcfLoadError::MissingPolicyDirectory { kind, path })
        }
        Ok(ResolvedCheckpoint::NoManifest | ResolvedCheckpoint::Found(_)) => Ok(path),
        Err(source) => Err(FullFcfLoadError::Read { source }),
    }
}

/// Where the cuts a pool's checkpoint recorded lie once loaded: a resume keeps each
/// record at its stored slot, and the loads that pack the records put each at its
/// rank in ascending slot order.
#[derive(Debug, Clone)]
pub struct LoadedSlotMap {
    source_slots: Vec<u32>,
    packed: bool,
}

impl LoadedSlotMap {
    fn new(kind: FullFcfLoadKind, records: &[OwnedPolicyCutRecord]) -> Self {
        let packed = match kind {
            FullFcfLoadKind::WarmStart | FullFcfLoadKind::SimulationOnly => true,
            FullFcfLoadKind::Resume => false,
        };
        Self {
            source_slots: records.iter().map(|record| record.slot_index).collect(),
            packed,
        }
    }

    /// The map of every pool of a checkpoint, in pool id order.
    #[must_use]
    pub fn for_pools(kind: FullFcfLoadKind, stage_cuts: &[StageCutsReadResult]) -> Vec<Self> {
        stage_cuts
            .iter()
            .map(|pool| Self::new(kind, &pool.cuts))
            .collect()
    }

    /// The slot the record stored at `source_slot` occupies in the loaded pool, or
    /// `None` when no record held that slot.
    #[must_use]
    pub fn dest(&self, source_slot: u32) -> Option<u32> {
        let rank = self.source_slots.binary_search(&source_slot).ok()?;
        if self.packed {
            // Rationale: slots are strictly ascending `u32`s, so a rank never exceeds its slot.
            #[allow(clippy::cast_possible_truncation)]
            Some(rank as u32)
        } else {
            Some(source_slot)
        }
    }
}

/// A checkpoint that passed every check, ready to be applied.
#[derive(Debug)]
pub struct CheckedFullFcfLoad {
    kind: FullFcfLoadKind,
    fcf: FutureCostFunction,
    slot_maps: Vec<LoadedSlotMap>,
    basis_cache: Vec<Option<CapturedBasis>>,
    scenario_bases: Vec<Vec<(u32, CapturedBasis)>>,
    trial_states: Vec<StageStatesReadResult>,
    unused_stored_bases: Option<UnusedStoredBases>,
    producer: ProducerBlock,
}

/// `states` with each pool's stored slots replaced by the slots `slot_maps`
/// puts their cuts at, after checking each pool's row dimension against the
/// study's.
fn place_trial_states(
    kind: FullFcfLoadKind,
    states: Vec<StageStatesReadResult>,
    slot_maps: &[LoadedSlotMap],
    pool_state_dimensions: &[usize],
) -> Result<Vec<StageStatesReadResult>, FullFcfLoadError> {
    let refusal = |message: String| FullFcfLoadError::FcfConstruction {
        kind,
        source: SddpError::Validation(message),
    };
    states
        .into_iter()
        .map(|mut pool_states| {
            let pool = pool_states.stage_id as usize;
            let (Some(slot_map), Some(&dimension)) =
                (slot_maps.get(pool), pool_state_dimensions.get(pool))
            else {
                return Err(refusal(format!(
                    "pool {pool}: stored trial states for a pool the study does not have"
                )));
            };
            if pool_states.state_dimension as usize != dimension {
                return Err(refusal(format!(
                    "pool {pool}: stored trial states have {} values, the study's pool {dimension}",
                    pool_states.state_dimension
                )));
            }
            for slot in &mut pool_states.slots {
                *slot = slot_map.dest(*slot).ok_or_else(|| {
                    refusal(format!(
                        "pool {pool}: a trial state is stored at slot {slot}, which holds no cut"
                    ))
                })?;
            }
            Ok(pool_states)
        })
        .collect()
}

/// The first state a resume of `setup` reads that the checkpoint `producer`
/// describes does not hold, in a fixed order: the forward-pass count, the
/// trial-state history, its window, the binding counters, the basis store.
///
/// # Errors
///
/// [`SddpError::Validation`] naming the setting, what the checkpoint holds and
/// the way out.
fn check_resume_serves(producer: &ProducerBlock, setup: &StudySetup) -> Result<(), SddpError> {
    let refusal = |message: String| Err(SddpError::Validation(message));

    let (run_passes, stored_passes) = (setup.loop_params.forward_passes, producer.forward_passes);
    if stored_passes != run_passes {
        return refusal(format!(
            "resume needs the forward-pass count the checkpoint was written with: \
             training.selection.forward_passes is {run_passes} here and {stored_passes} in the \
             checkpoint; set it to {stored_passes} or continue with policy.mode = warm_start"
        ));
    }

    // A study without an interior pool stores no trial state, so its level1
    // checkpoint holds none either.
    let level1_window = match setup.inputs.cut_management.cut_selection {
        Some(CutSelectionStrategy::Level1 {
            lookback_iterations,
            ..
        }) if !interior_cut_nodes(&setup.inputs.node_graph).is_empty() => Some(lookback_iterations),
        Some(CutSelectionStrategy::Level1 { .. } | CutSelectionStrategy::Dynamic { .. }) | None => {
            None
        }
    };
    if let Some(run_window) = level1_window {
        let stored_window = match producer.state_history {
            StateHistory::None => {
                return refusal(
                    "resume needs the trial-state history of \
                     training.cut_selection.selection.method = level1; the checkpoint holds \
                     none (it was written by a run without level1 or exports.states); continue \
                     with policy.mode = warm_start or retrain"
                        .to_string(),
                );
            }
            StateHistory::Window(window) => Some(u64::from(window.get())),
            StateHistory::Whole => None,
        };
        match (run_window, stored_window) {
            (None, Some(stored)) => {
                return refusal(format!(
                    "resume needs every trial state of the run under \
                     training.cut_selection.selection.method = level1 without \
                     lookback_iterations; the checkpoint holds those of the last {stored} \
                     iterations only; set lookback_iterations to at most {stored} or continue \
                     with policy.mode = warm_start"
                ));
            }
            (Some(window), Some(stored)) if window.get() > stored => {
                return refusal(format!(
                    "resume needs the trial states of the last {window} iterations \
                     (training.cut_selection.selection.lookback_iterations = {window}); the \
                     checkpoint holds those of the last {stored} iterations only; set \
                     lookback_iterations to at most {stored} or continue with \
                     policy.mode = warm_start"
                ));
            }
            (_, None) | (Some(_), Some(_)) => {}
        }
    }

    let cut_management = &setup.inputs.cut_management;
    if cut_management.tracks_binding_activity() && !producer.activity_counters {
        let reader = if cut_management.budget.is_some() {
            "training.cut_selection.max_active_per_stage"
        } else {
            "training.cut_selection.selection.method = dynamic"
        };
        return refusal(format!(
            "resume needs the binding counters read by {reader}; the checkpoint holds none (it \
             was written by a run without max_active_per_stage or the dynamic method); continue \
             with policy.mode = warm_start or retrain"
        ));
    }

    if !producer.basis_store_complete {
        return refusal(
            "resume needs a whole checkpoint; the checkpoint's per-scenario bases are not whole \
             (it was written when a run failed); resume from the last whole checkpoint or \
             continue with policy.mode = warm_start"
                .to_string(),
        );
    }
    Ok(())
}

/// Read, validate and decode the checkpoint in `policy_dir` against `setup`.
///
/// `on_warning` receives each compatibility warning right after validation,
/// then, once the basis cache is built, one warning when stored bases do not fit
/// the study. `setup` is not mutated.
///
/// # Errors
///
/// The [`FullFcfLoadError`] variant naming the failing step.
pub fn check_full_fcf_load(
    kind: FullFcfLoadKind,
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
    on_warning: &mut dyn FnMut(&str),
) -> Result<CheckedFullFcfLoad, FullFcfLoadError> {
    let mut checkpoint =
        read_policy_checkpoint(policy_dir).map_err(|source| FullFcfLoadError::Read { source })?;
    convert_checkpoint_cuts_for_load(
        &mut checkpoint.stage_cuts,
        setup.inputs.stage_data.stage_templates.cost_scale_factor,
    )
    .map_err(FullFcfLoadError::Refused)?;

    // Rationale: the cast cannot truncate — `n_stages` is the validated study
    // horizon (a `u16`-scale stage count), far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
    // Rationale: the cast cannot truncate — `state_dimension` counts FCF state
    // variables (one per reservoir/lag), bounded by the validated study
    // dimensions and far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;

    // The terminal pool is always full-config, so its manifest witnesses every
    // state family's slot identity — a terminal-only comparison covers all stages.
    let current_manifest = setup.build_terminal_entity_manifest(system);
    let checkpoint_terminal_manifest: &[EntitySlot] = checkpoint
        .stage_cuts
        .last()
        .map_or(&[], |s| s.entity_manifest.as_slice());
    let source_state_dim = checkpoint
        .stage_cuts
        .last()
        .map_or(0, |s| s.state_dimension);
    let source_graph = &checkpoint.metadata.graph_manifest;
    let current_graph = setup.build_graph_manifest();

    let source = PolicyStageManifest {
        state_dimension: source_state_dim,
        num_stages: checkpoint.metadata.num_stages,
        n_pools: source_graph.n_pools,
        slots: checkpoint_terminal_manifest,
        graph: source_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: state_dim,
        num_stages: n_stages,
        n_pools: current_graph.n_pools,
        slots: &current_manifest,
        graph: &current_graph,
    };
    let proof =
        validate_policy_load::<FullFcf>(checkpoint.metadata.written_by(), &source, &current)
            .map_err(FullFcfLoadError::Refused)?;
    if kind == FullFcfLoadKind::Resume {
        check_resume_serves(&checkpoint.metadata.producer, setup)
            .map_err(FullFcfLoadError::Refused)?;
    }
    for msg in &proof.warnings {
        on_warning(msg);
    }

    // Reuse per-pool dimensions from the current study's FCF, not the checkpoint's.
    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let visit_bounds: Vec<u64> = setup
        .fcf
        .pools
        .iter()
        .map(|p| u64::from(p.visit_stride))
        .collect();
    // Reserve one extra slot for cuts added in the final iteration.
    let max_iterations = setup.loop_params.max_iterations.saturating_add(1);
    let mut fcf = match kind {
        FullFcfLoadKind::WarmStart => FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup.loop_params.forward_passes,
            max_iterations,
        ),
        FullFcfLoadKind::Resume => FutureCostFunction::new_resumed(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup.loop_params.forward_passes,
            max_iterations,
        ),
        FullFcfLoadKind::SimulationOnly => FutureCostFunction::from_deserialized(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
        ),
    }
    .map_err(|source| FullFcfLoadError::FcfConstruction { kind, source })?;
    if kind == FullFcfLoadKind::WarmStart {
        for pool in &mut fcf.pools {
            pool.restart_activity_clock();
        }
    }

    let slot_maps = LoadedSlotMap::for_pools(kind, &checkpoint.stage_cuts);
    let trial_states = match kind {
        FullFcfLoadKind::WarmStart | FullFcfLoadKind::Resume => place_trial_states(
            kind,
            checkpoint.stage_states,
            &slot_maps,
            &pool_state_dimensions,
        )?,
        FullFcfLoadKind::SimulationOnly => Vec::new(),
    };

    let StoredBasisLoad {
        cache: basis_cache,
        store: scenario_bases,
        unused: unused_stored_bases,
    } = build_basis_cache_from_checkpoint(&checkpoint.stage_bases, &slot_maps, setup);
    if let Some(unused) = &unused_stored_bases {
        on_warning(&unused.to_string());
    }

    Ok(CheckedFullFcfLoad {
        kind,
        fcf,
        slot_maps,
        basis_cache,
        scenario_bases,
        trial_states,
        unused_stored_bases,
        producer: checkpoint.metadata.producer,
    })
}

impl CheckedFullFcfLoad {
    /// Training iterations the checkpoint records as completed.
    #[must_use]
    pub fn completed_iterations(&self) -> u64 {
        u64::from(self.producer.completed_iterations)
    }

    /// The slot map of each pool, in pool id order.
    #[must_use]
    pub fn slot_maps(&self) -> &[LoadedSlotMap] {
        &self.slot_maps
    }

    /// The stored bases the load left out, when any did not fit the study.
    #[must_use]
    pub fn unused_stored_bases(&self) -> Option<&UnusedStoredBases> {
        self.unused_stored_bases.as_ref()
    }

    /// Install the loaded FCF, the stored bases of every scenario, the stored
    /// trial states and, for resume, the completed iterations and their
    /// recorded lower bounds into `setup`.
    pub fn apply_to_training(self, setup: &mut StudySetup) {
        debug_assert!(self.kind != FullFcfLoadKind::SimulationOnly);
        let completed = self.completed_iterations();
        setup.replace_fcf(self.fcf);
        let missing_scenario_basis = match self.kind {
            FullFcfLoadKind::Resume => MissingScenarioBasis::Cold,
            FullFcfLoadKind::WarmStart | FullFcfLoadKind::SimulationOnly => {
                MissingScenarioBasis::NodeLowestScenario
            }
        };
        setup.set_loaded_policy(LoadedPolicy {
            scenario_bases: self.scenario_bases,
            trial_states: self.trial_states,
            missing_scenario_basis,
        });
        if self.kind == FullFcfLoadKind::Resume {
            setup.set_resume_point(completed, self.producer.lower_bound_history);
        }
    }

    /// The loaded FCF and the synthetic training result that stands in for a
    /// training run in simulation-only mode.
    #[must_use]
    pub fn into_simulation_policy(self) -> (FutureCostFunction, TrainingResult) {
        debug_assert_eq!(self.kind, FullFcfLoadKind::SimulationOnly);
        let result = TrainingResult::new(
            self.producer.final_lower_bound,
            self.producer.best_upper_bound.unwrap_or(f64::INFINITY),
            0.0,
            0.0,
            self.producer.completed_iterations.into(),
            "loaded from checkpoint".to_string(),
            0,
            self.basis_cache,
            Vec::new(),
            // Checkpoints store no frozen templates; `simulate()` re-freezes from the
            // FCF row pool when this is None.
            None,
        );
        (self.fcf, result)
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use novomodelo_comm::LocalBackend;
    use novomodelo_io::test_support::owned_piece_defaults;
    use novomodelo_solver::{ActiveSolver, BasisStatus};

    use super::*;
    use crate::policy::orchestration::{CheckpointParams, write_checkpoint};
    use crate::setup::NodePos;
    use crate::test_support::{k_fan_setup, k_fan_system};

    const KINDS: [FullFcfLoadKind; 3] = [
        FullFcfLoadKind::WarmStart,
        FullFcfLoadKind::Resume,
        FullFcfLoadKind::SimulationOnly,
    ];

    fn read_io_error(kind: ErrorKind) -> FullFcfLoadError {
        FullFcfLoadError::Read {
            source: OutputError::IoError {
                path: PathBuf::from("policy/manifest.bin"),
                source: io::Error::from(kind),
            },
        }
    }

    fn record_at(slot_index: u32) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            slot_index,
            ..owned_piece_defaults()
        }
    }

    #[test]
    fn a_slot_map_places_each_record_by_kind_and_leaves_a_gap_empty() {
        let records: Vec<_> = [0, 1, 3, 4].map(record_at).into();
        let dests = |kind| {
            let map = LoadedSlotMap::new(kind, &records);
            (0..6).map(|slot| map.dest(slot)).collect::<Vec<_>>()
        };

        assert_eq!(
            dests(FullFcfLoadKind::Resume),
            [Some(0), Some(1), None, Some(3), Some(4), None]
        );
        for kind in [FullFcfLoadKind::WarmStart, FullFcfLoadKind::SimulationOnly] {
            assert_eq!(
                dests(kind),
                [Some(0), Some(1), None, Some(2), Some(3), None],
                "{kind:?}"
            );
        }
    }

    fn states_at(pool: u32, dimension: u32, slots: &[u32]) -> StageStatesReadResult {
        StageStatesReadResult {
            stage_id: pool,
            node_id: -1,
            state_dimension: dimension,
            count: slots.len() as u32,
            data: slots
                .iter()
                .flat_map(|&slot| (0..dimension).map(move |i| f64::from(slot) + f64::from(i) / 8.0))
                .collect(),
            entity_manifest: Vec::new(),
            slots: slots.to_vec(),
        }
    }

    #[test]
    fn a_stored_state_follows_its_cut_to_the_slot_the_load_puts_it_at() {
        let records: Vec<_> = [0, 1, 3, 4].map(record_at).into();
        let place = |kind| {
            let slot_maps = [LoadedSlotMap::new(kind, &records)];
            place_trial_states(kind, vec![states_at(0, 2, &[1, 3, 4])], &slot_maps, &[2])
                .expect("every state sits at a cut")
        };

        let resumed = place(FullFcfLoadKind::Resume);
        assert_eq!(resumed[0].slots, [1, 3, 4]);
        let warm_started = place(FullFcfLoadKind::WarmStart);
        assert_eq!(warm_started[0].slots, [1, 2, 3]);
        assert_eq!(
            resumed[0].data, warm_started[0].data,
            "the rows do not move"
        );
        assert_eq!(resumed[0].data[2..4], [3.0, 3.125]);
    }

    #[test]
    fn a_stored_state_the_study_cannot_hold_is_refused_naming_its_pool() {
        let records: Vec<_> = [0, 1].map(record_at).into();
        let kind = FullFcfLoadKind::Resume;
        let slot_maps = [LoadedSlotMap::new(kind, &records)];
        let refusal = |states: StageStatesReadResult, dimensions: &[usize]| {
            place_trial_states(kind, vec![states], &slot_maps, dimensions)
                .expect_err("the state cannot be held")
        };

        let wrong_dimension = refusal(states_at(0, 3, &[1]), &[2]);
        assert!(
            matches!(wrong_dimension, FullFcfLoadError::FcfConstruction { .. }),
            "{wrong_dimension}"
        );
        assert!(
            wrong_dimension.to_string().contains("pool 0"),
            "{wrong_dimension}"
        );
        let no_cut = refusal(states_at(0, 2, &[2]), &[2]);
        assert!(no_cut.to_string().contains("slot 2"), "{no_cut}");
        let no_pool = refusal(states_at(1, 2, &[0]), &[2, 2]);
        assert!(no_pool.to_string().contains("pool 1"), "{no_pool}");
    }

    /// Train a fan of [`k_fan_setup`] for `iterations`, from `policy` when given.
    fn train_fan(iterations: u32, policy: Option<&Path>) -> (StudySetup, TrainingResult) {
        let system = k_fan_system(FAN_WIDTH, false);
        let mut setup = k_fan_setup(FAN_WIDTH, FAN_PASSES, iterations).setup;
        if let Some(policy_dir) = policy {
            check_full_fcf_load(
                FullFcfLoadKind::Resume,
                policy_dir,
                &system,
                &setup,
                &mut |_: &str| {},
            )
            .expect("the fan checkpoint must load")
            .apply_to_training(&mut setup);
            assert_restores_every_stored_basis(&setup, policy_dir);
        }
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &LocalBackend, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        (setup, outcome.result)
    }

    /// Every entry of the checkpoint in `policy_dir` seeds its own scenario at
    /// its node, with the statuses and the cut-row slots it was written with.
    fn assert_restores_every_stored_basis(setup: &StudySetup, policy_dir: &Path) {
        let checkpoint = read_policy_checkpoint(policy_dir).expect("read_policy_checkpoint");
        let codes = |statuses: &[BasisStatus]| -> Vec<u8> {
            statuses.iter().map(|s| s.to_discriminant_code()).collect()
        };
        for record in &checkpoint.stage_bases {
            for entry in &record.entries {
                let node = NodePos(record.stage_id as usize);
                let seeded = setup
                    .loaded_policy()
                    .seed_basis(node, entry.scenario)
                    .expect("a stored basis seeds its scenario");
                assert_eq!(
                    (
                        codes(&seeded.basis.col_status),
                        codes(&seeded.basis.row_status),
                        &seeded.cut_row_slots,
                    ),
                    (
                        entry.column_status.clone(),
                        entry.row_status.clone(),
                        &entry.piece_row_slots,
                    ),
                    "node {}, scenario {}",
                    node.0,
                    entry.scenario
                );
            }
        }
    }

    fn write_fan_checkpoint(dir: &Path, setup: &StudySetup, result: &TrainingResult) {
        let params = CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
        };
        write_checkpoint(dir, setup, &k_fan_system(FAN_WIDTH, false), result, &params)
            .expect("write_checkpoint must succeed");
    }

    const FAN_WIDTH: usize = 8;
    const FAN_PASSES: u32 = 6;

    #[test]
    fn a_resumed_fan_run_restores_every_stored_basis_and_equals_the_uninterrupted_run() {
        let (uninterrupted, _) = train_fan(6, None);
        let (interrupted, interrupted_result) = train_fan(3, None);
        let dir = tempfile::tempdir().expect("tempdir");
        write_fan_checkpoint(dir.path(), &interrupted, &interrupted_result);

        let checkpoint = read_policy_checkpoint(dir.path()).expect("read_policy_checkpoint");
        let passes = FAN_PASSES as usize;
        assert!(
            checkpoint
                .stage_bases
                .iter()
                .any(|record| !record.entries.is_empty() && record.entries.len() < passes),
            "power: some node holds the bases of some scenarios only"
        );

        let (resumed, _) = train_fan(6, Some(dir.path()));
        for (pool, (r, u)) in resumed
            .fcf
            .pools
            .iter()
            .zip(&uninterrupted.fcf.pools)
            .enumerate()
        {
            assert_eq!(r.populated(), u.populated(), "pool {pool}: populated slots");
            for slot in 0..u.populated() {
                assert_eq!(r.role(slot), u.role(slot), "pool {pool}, slot {slot}: role");
                assert_eq!(
                    r.is_active(slot),
                    u.is_active(slot),
                    "pool {pool}, slot {slot}: activity"
                );
                assert_eq!(
                    r.intercept(slot).to_bits(),
                    u.intercept(slot).to_bits(),
                    "pool {pool}, slot {slot}: intercept"
                );
                let bits = |c: &[f64]| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(r.coefficient_row(slot)),
                    bits(u.coefficient_row(slot)),
                    "pool {pool}, slot {slot}: coefficients"
                );
            }
        }
    }

    /// On a sampled fan a scenario does not visit every node: a resume leaves
    /// that scenario's row cold there, as the run that wrote the checkpoint did,
    /// and a warm start seeds it from the node's lowest-scenario basis.
    #[test]
    fn only_a_resume_leaves_a_scenario_without_a_stored_basis_cold() {
        let (interrupted, interrupted_result) = train_fan(3, None);
        let dir = tempfile::tempdir().expect("tempdir");
        write_fan_checkpoint(dir.path(), &interrupted, &interrupted_result);
        let checkpoint = read_policy_checkpoint(dir.path()).expect("read_policy_checkpoint");
        let (node, scenario) = checkpoint
            .stage_bases
            .iter()
            .filter(|record| !record.entries.is_empty())
            .find_map(|record| {
                (0..FAN_PASSES)
                    .find(|s| record.entries.iter().all(|entry| entry.scenario != *s))
                    .map(|s| (NodePos(record.stage_id as usize), s))
            })
            .expect("power: some node holds the bases of some scenarios only");

        let system = k_fan_system(FAN_WIDTH, false);
        for kind in [FullFcfLoadKind::Resume, FullFcfLoadKind::WarmStart] {
            let mut setup = k_fan_setup(FAN_WIDTH, FAN_PASSES, 6).setup;
            check_full_fcf_load(kind, dir.path(), &system, &setup, &mut |_: &str| {})
                .expect("the fan checkpoint must load")
                .apply_to_training(&mut setup);
            let loaded = setup.loaded_policy();
            let lowest = loaded.scenario_bases[node.0]
                .first()
                .map(|(_, basis)| basis);
            assert!(lowest.is_some());
            let expected = match kind {
                FullFcfLoadKind::Resume => None,
                FullFcfLoadKind::WarmStart | FullFcfLoadKind::SimulationOnly => lowest,
            };
            let seeded = loaded.seed_basis(node, scenario);
            assert!(
                match (seeded, expected) {
                    (Some(seeded), Some(expected)) => std::ptr::eq(seeded, expected),
                    (seeded, expected) => seeded.is_none() && expected.is_none(),
                },
                "{kind:?}: node {}, scenario {scenario}",
                node.0
            );
        }
    }

    #[test]
    fn missing_policy_directory_classifies_as_invalid_input() {
        for kind in KINDS {
            let err = FullFcfLoadError::MissingPolicyDirectory {
                kind,
                path: PathBuf::from("out/policy"),
            };
            assert_eq!(err.class(), ErrorClass::InvalidInput, "{kind:?}");
        }
    }

    #[test]
    fn policy_file_not_found_classifies_as_incompatible_policy() {
        assert_eq!(
            read_io_error(ErrorKind::NotFound).class(),
            ErrorClass::IncompatiblePolicy
        );
    }

    #[test]
    fn unparseable_policy_file_classifies_as_incompatible_policy() {
        let sources = [
            OutputError::SerializationError {
                entity: "checkpoint_manifest".to_string(),
                message: "missing file identifier".to_string(),
            },
            OutputError::SchemaError {
                file: "cuts/0.bin".to_string(),
                column: "coefficients".to_string(),
                message: "unexpected type".to_string(),
            },
            OutputError::ManifestError {
                manifest_type: "checkpoint".to_string(),
                message: "inconsistent dates".to_string(),
            },
        ];
        for source in sources {
            let err = FullFcfLoadError::Read { source };
            assert_eq!(err.class(), ErrorClass::IncompatiblePolicy, "{err}");
        }
    }

    #[test]
    fn os_refused_policy_read_classifies_as_io() {
        assert_eq!(
            read_io_error(ErrorKind::PermissionDenied).class(),
            ErrorClass::Io
        );
    }

    #[test]
    fn refusals_and_fcf_construction_classify_as_incompatible_policy() {
        let mut errors = vec![
            FullFcfLoadError::Refused(SddpError::Validation("state_dimension mismatch".into())),
            FullFcfLoadError::Refused(SddpError::PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            }),
        ];
        errors.extend(KINDS.map(|kind| FullFcfLoadError::FcfConstruction {
            kind,
            source: SddpError::Validation("stage_results is empty".into()),
        }));
        for err in errors {
            assert_eq!(err.class(), ErrorClass::IncompatiblePolicy, "{err}");
        }
    }
}
