//! Case-load, communicator setup, broadcast, and pre-training phases for `novomodelo run`.
//!
//! Rank 0 loads from disk; `System` and config are broadcast to all ranks, which
//! then build `StudySetup` from the shared data. Hydro model preprocessing is the
//! exception: it is not broadcast. Rank 0 reuses the artifacts
//! `load_case_with_artifacts` already parsed (`prepare_hydro_models_from_artifacts`);
//! non-root ranks independently re-read the case directory from disk
//! (`prepare_hydro_models`), relying on a shared filesystem instead.

use std::path::Path;

use console::Term;

use novomodelo_comm::{Communicator, TopologyProvider, create_communicator, min_max_across_ranks};
use novomodelo_core::ScalarParameter;
use novomodelo_core::System;
use novomodelo_io::BroadcastScalarParameter;
use novomodelo_io::Config;
use novomodelo_io::PolicyMode;
use novomodelo_io::SetupTimings;
use novomodelo_io::load_case_with_artifacts;
use novomodelo_io::parse_config;
use novomodelo_io::remove_conditional_training_outputs;
use novomodelo_io::remove_simulation_outputs;
use novomodelo_io::remove_success_marker;
use novomodelo_io::write_hydro_model_summary;
use novomodelo_io::write_provenance_report;
use novomodelo_io::write_scaling_report;
use novomodelo_sddp::EstimationPath;
use novomodelo_sddp::HydroFitTimings;
use novomodelo_sddp::build_provenance_report;
use novomodelo_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use novomodelo_sddp::policy::orchestration::export_stochastic_artifacts;
use novomodelo_sddp::reconcile_global_ok;
use novomodelo_sddp::{
    CorrelationOrigin, EstimationReport, PrepareHydroModelsResult, PrepareStochasticResult,
    StudySetup, build_hydro_model_summary, build_stochastic_context_for_study,
    prepare_hydro_models, prepare_stochastic, resolve_boundary_state_requirements,
    setup::StudyParams,
};
use novomodelo_solver::active_solver_name;
use novomodelo_solver::active_solver_version;
use novomodelo_stochastic::HistoricalScenarioLibrary;
use novomodelo_stochastic::context::OpeningTree;
use novomodelo_stochastic::provenance::ComponentProvenance;

use crate::error::CliError;

use crate::commands::broadcast::{
    BroadcastConfig, BroadcastOpeningTree, broadcast_value, stopping_rules_from_broadcast,
};
use crate::commands::resolve_output_dir;

use super::{RunArgs, RunContext};
use crate::banner::print_banner;
use crate::progress::RenderMode;
use crate::progress::resolve_term_width;
use crate::summary::print_execution_topology;
use crate::summary::print_hydro_model_summary;
use crate::summary::print_provenance_summary;
use crate::summary::print_setup_summary;

pub(super) fn resolve_thread_count(cli_threads: Option<u32>) -> usize {
    match cli_threads {
        Some(n) => n as usize,
        None => 1,
    }
}

/// Values loaded on rank 0 by [`load_case_and_config`]. The trailing
/// [`novomodelo_io::SetupTimings`] leaves `broadcast_seconds` zero;
/// [`broadcast_and_build_setup`] fills it after the broadcast region runs.
type LoadedCase = (
    PrepareStochasticResult,
    PrepareHydroModelsResult,
    BroadcastConfig,
    Config,
    Vec<ScalarParameter>,
    SetupTimings,
);

/// Load case and config on rank 0, capturing errors for MPI collective participation.
fn load_case_and_config(
    args: &RunArgs,
    output_dir: &Path,
    quiet: bool,
    out: &Term,
) -> Result<LoadedCase, CliError> {
    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "case directory does not exist",
            ),
            context: args.case_dir.display().to_string(),
        });
    }
    if !quiet {
        let _ = out.write_line(&format!("Loading case: {}", args.case_dir.display()));
    }
    let mut timings = SetupTimings::default();

    let load_start = std::time::Instant::now();
    let novomodelo_io::LoadedCase { system, artifacts } = load_case_with_artifacts(&args.case_dir)?;
    let config_path = args.case_dir.join("config.json");
    let config = parse_config(&config_path)?;
    config
        .policy
        .check_dir(&config_path, output_dir, config.policy_dir_intent())?;
    timings.load_seconds = load_start.elapsed().as_secs_f64();

    // Resolve the boundary-derived state requirements once (rank 0, the sole
    // reader of the case dir) and carry them on the broadcast config, so both the
    // state layout and the boundary-load reject see the identical requirements on
    // every rank.
    let boundary_requirements = resolve_boundary_state_requirements(&args.case_dir, &config)?;
    if !quiet && let Some(d) = boundary_requirements.inflow_lag_depth() {
        let _ = out.write_line(&format!("Boundary policy: inflow-lag depth {d}"));
    }

    let mut bcast = BroadcastConfig::from_config(&config)?;
    bcast.boundary = boundary_requirements;
    let seed = bcast.seed;

    let stochastic_start = std::time::Instant::now();
    let prepared = prepare_stochastic(
        system,
        &args.case_dir,
        &config,
        seed,
        &bcast.training_source,
        bcast.boundary.inflow_lag_depth(),
    )
    .map_err(CliError::from)?;
    timings.stochastic_fit_seconds = stochastic_start.elapsed().as_secs_f64();

    let mut hydro_timings = HydroFitTimings::default();
    let hydro_models = prepare_hydro_models_from_artifacts(
        &prepared.system,
        &artifacts,
        config.exports.fpha_deviation_points,
        Some(&mut hydro_timings),
    )
    .map_err(CliError::from)?;
    timings.production_fit_seconds = hydro_timings.production_fit_seconds;
    timings.evaporation_fit_seconds = hydro_timings.evaporation_fit_seconds;

    Ok((
        prepared,
        hydro_models,
        bcast,
        config,
        artifacts.scalar_parameters,
        timings,
    ))
}

/// Output of [`broadcast_and_build_setup`]. `root_*` fields are `Some` only on
/// rank 0 (used for output writing); the flag fields are broadcast from rank 0.
pub(super) struct LoadBroadcastResult {
    pub(super) system: System,
    pub(super) setup: StudySetup,
    pub(super) root_config: Option<Config>,
    pub(super) root_estimation_report: Option<EstimationReport>,
    pub(super) root_estimation_path: Option<EstimationPath>,
    pub(super) root_correlation_origin: Option<CorrelationOrigin>,
    pub(super) training_enabled: bool,
    pub(super) policy_mode: PolicyMode,
    /// `None` on non-root ranks, which reconstruct setup independently and never
    /// write metadata.
    pub(super) setup_timings: Option<SetupTimings>,
}

/// Set up the communicator, terminal, rayon pool, and resolve the output directory.
pub(super) fn setup_communicator(
    args: &RunArgs,
) -> Result<RunContext<impl Communicator>, CliError> {
    let comm = create_communicator(args.comm_backend.into())?;
    if comm.size() > 1 && comm.rank() != 0 {
        crate::mark_non_root_rank();
    }
    let is_root = comm.rank() == 0;
    let quiet = args.quiet || !is_root;

    // mpiexec pipes rank 0's stdout and stderr without a PTY, so force colors
    // on; console would otherwise disable them on the non-TTY pipes.
    let mpi_active = comm.size() > 1;
    if mpi_active && is_root && !args.quiet {
        console::set_colors_enabled(true);
        console::set_colors_enabled_stderr(true);
    }

    let stdout = Term::stdout();
    let stderr = Term::stderr();

    // Gather topology while the concrete backend type is still in scope.
    let topology = comm.topology().clone();

    let configured_threads = resolve_thread_count(args.threads);
    let actual_threads = match rayon::ThreadPoolBuilder::new()
        .num_threads(configured_threads)
        .build_global()
    {
        Ok(()) => configured_threads,
        Err(err) => {
            let actual = rayon::current_num_threads();
            tracing::warn!(
                configured = configured_threads,
                actual,
                %err,
                "rayon global thread pool init failed; using existing pool",
            );
            actual
        }
    };
    if actual_threads == 0 {
        return Err(CliError::Internal {
            message: "rayon reported zero active threads — unexpected state".to_string(),
        });
    }

    let solver_version = active_solver_version();

    if !quiet {
        print_banner(&stdout);
        print_execution_topology(
            &stdout,
            &topology,
            actual_threads,
            active_solver_name(),
            Some(&solver_version),
        );
    }

    let output_dir = resolve_output_dir(&args.case_dir, args.output.as_deref());
    let term_width = resolve_term_width();
    let render_mode = RenderMode::auto();

    Ok(RunContext {
        comm,
        is_root,
        quiet,
        n_threads: actual_threads,
        output_dir,
        case_dir: args.case_dir.clone(),
        term_width,
        stdout,
        stderr,
        render_mode,
        topology,
        solver_version,
    })
}

/// Load the case on rank 0, broadcast system/config/tree, and build `StudySetup` on all ranks.
// Rationale: one MPI coordination seam — splitting it would scatter the ordered
// broadcast-receive-build sequence across callsites.
#[allow(clippy::too_many_lines)]
pub(super) fn broadcast_and_build_setup(
    ctx: &RunContext<impl Communicator>,
    args: &RunArgs,
) -> Result<LoadBroadcastResult, CliError> {
    // Kept out of the broadcast tuple: timings are never broadcast — only rank 0
    // writes metadata.
    let mut root_setup_timings: Option<SetupTimings> = None;
    let (
        raw_system,
        raw_bcast_config,
        root_config,
        root_stochastic,
        root_estimation_report,
        root_estimation_path,
        root_correlation_origin,
        raw_bcast_tree,
        root_hydro_models,
        raw_scalar_parameters,
        load_err,
    ) = if ctx.is_root {
        match load_case_and_config(args, &ctx.output_dir, ctx.quiet, &ctx.stdout) {
            Ok((prepared, hydro_models, bcast, config, scalar_parameters, timings)) => {
                root_setup_timings = Some(timings);
                let bcast_tree = user_supplied_opening_tree(&prepared.stochastic);
                let PrepareStochasticResult {
                    system,
                    stochastic,
                    estimation_report,
                    estimation_path,
                    correlation_origin,
                } = prepared;
                let bcast_params: Vec<BroadcastScalarParameter> = scalar_parameters
                    .iter()
                    .map(BroadcastScalarParameter::from)
                    .collect();
                (
                    Some(system),
                    Some(bcast),
                    Some(config),
                    Some(stochastic),
                    estimation_report,
                    Some(estimation_path),
                    Some(correlation_origin),
                    Some(bcast_tree),
                    Some(hydro_models),
                    Some(bcast_params),
                    None,
                )
            }
            Err(e) => (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(e),
            ),
        }
    } else {
        (
            None, None, None, None, None, None, None, None, None, None, None,
        )
    };
    let broadcast_start = std::time::Instant::now();
    let system_result = broadcast_value(raw_system, &ctx.comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &ctx.comm);
    let scalar_parameters_result = broadcast_value(raw_scalar_parameters, &ctx.comm);

    let tree_result = broadcast_value(raw_bcast_tree, &ctx.comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    let seed = bcast_config.seed;

    // Non-root reconstruction must reproduce rank 0's factor entries (load, NCS)
    // and forward seed exactly, for MPI reproducibility.
    let stochastic = if ctx.is_root {
        drop(tree_result);
        root_stochastic.ok_or_else(|| CliError::Internal {
            message: "stochastic context missing on rank 0 after successful load".to_string(),
        })?
    } else {
        let user_tree: Option<OpeningTree> =
            tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        reconstruct_stochastic_context_non_root(
            &system,
            &bcast_config,
            user_tree,
            seed,
            &args.case_dir,
        )?
    };

    let hydro_models = if ctx.is_root {
        root_hydro_models.ok_or_else(|| CliError::Internal {
            message: "hydro models missing on rank 0 after successful load".to_string(),
        })?
    } else {
        // Deviation-points opt-in is `false` regardless of the run-level export
        // flag: non-root ranks never reach the write site (only rank 0 writes).
        prepare_hydro_models(&system, &args.case_dir, false).map_err(|e| CliError::Internal {
            message: format!("hydro model preprocessing error on non-root rank: {e}"),
        })?
    };

    let training_enabled = bcast_config.training_enabled;
    let policy_mode = bcast_config.policy_mode;
    let scalar_parameters: Vec<ScalarParameter> = scalar_parameters_result?
        .into_iter()
        .map(ScalarParameter::from)
        .collect();
    let setup = build_study_setup(
        &system,
        &mut bcast_config,
        stochastic,
        hydro_models,
        scalar_parameters,
    )?;

    if ctx.comm.size() > 1 {
        let fingerprint = setup.inputs.stochastic.opening_tree().fingerprint();
        let (min, max) =
            min_max_across_ranks(&ctx.comm, fingerprint).map_err(|e| CliError::Internal {
                message: format!("opening tree fingerprint exchange failed: {e}"),
            })?;
        if min != max {
            return Err(CliError::Internal {
                message: format!(
                    "opening tree differs across MPI ranks (fingerprints {min:#018x} to \
                     {max:#018x}); every rank must build the same tree from the broadcast inputs"
                ),
            });
        }
    }

    if let Some(timings) = root_setup_timings.as_mut() {
        timings.broadcast_seconds = broadcast_start.elapsed().as_secs_f64();
    }

    Ok(LoadBroadcastResult {
        system,
        setup,
        root_config,
        root_estimation_report,
        root_estimation_path,
        root_correlation_origin,
        training_enabled,
        policy_mode,
        setup_timings: root_setup_timings,
    })
}

fn user_supplied_opening_tree(
    stochastic: &novomodelo_stochastic::StochasticContext,
) -> Option<BroadcastOpeningTree> {
    (stochastic.provenance().opening_tree == ComponentProvenance::UserSupplied).then(|| {
        let t = stochastic.opening_tree();
        BroadcastOpeningTree {
            data: t.data().to_vec(),
            openings_per_stage: t.openings_per_stage_slice().to_vec(),
            dim: t.dim(),
        }
    })
}

/// Reconstruct the non-root `StochasticContext` from broadcast parameters.
fn reconstruct_stochastic_context_non_root(
    system: &System,
    bcast_config: &BroadcastConfig,
    user_tree: Option<OpeningTree>,
    seed: u64,
    case_dir: &Path,
) -> Result<novomodelo_stochastic::StochasticContext, CliError> {
    build_stochastic_context_for_study(
        system,
        case_dir,
        seed,
        &bcast_config.training_source,
        bcast_config.boundary.inflow_lag_depth(),
        user_tree,
        bcast_config.training_moment_matching.into(),
    )
    .map_err(CliError::from)
}

/// Construct `StudySetup` on all ranks from broadcast parameters.
// Rationale: by-value because `StudySetup` construction moves these in; a
// reference would force an internal clone.
#[allow(clippy::needless_pass_by_value)]
fn build_study_setup(
    system: &System,
    bcast_config: &mut BroadcastConfig,
    stochastic: novomodelo_stochastic::StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    scalar_parameters: Vec<ScalarParameter>,
) -> Result<StudySetup, CliError> {
    let stopping_rule_set = stopping_rules_from_broadcast(bcast_config);
    let cut_selection = bcast_config.cut_selection.take();
    let training_solver_backward = bcast_config.training_solver_backward.take();
    let training_solver_forward = bcast_config.training_solver_forward.take();
    let simulation_solver = bcast_config.simulation_solver.take();
    let config = StudyParams {
        seed: bcast_config.seed,
        forward_passes: bcast_config.forward_passes,
        training_enumerated: bcast_config.training_enumerated,
        stopping_rule_set,
        n_scenarios: bcast_config.n_scenarios,
        simulation_enumerated: bcast_config.simulation_enumerated,
        simulation_cut_set: bcast_config.simulation_cut_set,
        io_channel_capacity: usize::try_from(bcast_config.io_channel_capacity).unwrap_or(64),
        policy_path: bcast_config.policy_path.clone(),
        inflow_method: bcast_config.inflow_method,
        cut_selection,
        cut_activity_tolerance: bcast_config.cut_activity_tolerance,
        budget: bcast_config.budget,
        export_states: bcast_config.export_states,
        checkpoint_schedule: bcast_config.checkpoint_schedule,
        scalar_parameters,
        training_solver_backward,
        training_solver_forward,
        simulation_solver,
        backward_scheduler: bcast_config.backward_scheduler.into(),
        cost_scale_factor: bcast_config.cost_scale_factor,
        boundary: bcast_config.boundary.clone(),
    };
    StudySetup::from_broadcast_params(
        system,
        stochastic,
        config,
        hydro_models,
        &bcast_config.training_source,
        &bcast_config.simulation_source,
    )
    .map_err(CliError::from)
}

pub(super) fn run_pre_training(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<EstimationPath>,
    root_correlation_origin: Option<CorrelationOrigin>,
    setup_timings: Option<&SetupTimings>,
) -> Result<(), CliError> {
    if ctx.is_root
        && !ctx.quiet
        && let Some(timings) = setup_timings
    {
        print_setup_summary(&ctx.stdout, timings);
    }

    let export_result: Result<(), CliError> = if ctx.is_root {
        run_root_exports(
            ctx,
            system,
            setup,
            root_config,
            root_estimation_report,
            root_estimation_path,
            root_correlation_origin,
        )
    } else {
        Ok(())
    };

    // Reconcile the rank-0-only export writes BEFORE the barrier: a rank-0 I/O
    // failure (disk full, permissions) would otherwise strand every peer at the
    // barrier while rank 0 returned early.
    let mut reconcile_scratch = [0_i32];
    let global_ok = reconcile_global_ok(export_result.is_ok(), &ctx.comm, &mut reconcile_scratch)
        .map_err(|e| CliError::Internal {
        message: format!("pre-training export reconcile error: {e}"),
    })?;
    export_result?;
    if !global_ok {
        return Err(CliError::Internal {
            message: "rank 0 pre-training export failed; failing on every rank in lockstep"
                .to_string(),
        });
    }

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

    Ok(())
}

/// Rank-0 pre-training exports: removes the stale `_SUCCESS` of each planned
/// phase, an earlier run's simulation outputs when simulation is planned and an
/// earlier run's conditional training outputs when training is planned, then
/// writes the hydro model summary, provenance report, stochastic artifacts
/// (non-fatal), and scaling report. Called only on rank 0; the returned
/// `Result` is reconciled across ranks before the post-export barrier.
fn run_root_exports(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<EstimationPath>,
    root_correlation_origin: Option<CorrelationOrigin>,
) -> Result<(), CliError> {
    if root_config.is_some_and(|c| c.training.enabled) {
        remove_success_marker(&ctx.output_dir.join("training")).map_err(CliError::from)?;
    }
    if setup.simulation_config.n_scenarios > 0 {
        remove_success_marker(&ctx.output_dir.join("simulation")).map_err(CliError::from)?;
        remove_simulation_outputs(&ctx.output_dir).map_err(CliError::from)?;
    }
    if root_config.is_some_and(|c| c.training.enabled) {
        remove_conditional_training_outputs(&ctx.output_dir).map_err(CliError::from)?;
    }

    // Built regardless of `quiet`: it feeds the `training/hydro_models.json`
    // output file, not just the optional print.
    let hydro_summary = build_hydro_model_summary(&setup.hydro_models, system);
    if !ctx.quiet {
        print_hydro_model_summary(&ctx.stdout, &hydro_summary);
    }
    let hydro_models_path = ctx.output_dir.join("training/hydro_models.json");
    write_hydro_model_summary(&hydro_models_path, &hydro_summary).map_err(|e| {
        CliError::Internal {
            message: format!("failed to write hydro model summary: {e}"),
        }
    })?;

    if let Some(path) = root_estimation_path {
        let Some(correlation_origin) = root_correlation_origin else {
            return Err(CliError::Internal {
                message: "rank 0 resolved an estimation path but no correlation origin".to_string(),
            });
        };
        let mut provenance = build_provenance_report(
            path,
            root_estimation_report,
            correlation_origin,
            setup.inputs.stochastic.provenance(),
            system.hydros(),
            &setup.hydro_models.provenance,
        );
        // Stale-library detection compares this digest on later runs.
        provenance.inflow.historical_library_seed_digest = setup
            .inputs
            .scenario_libraries
            .training
            .historical
            .as_ref()
            .map(HistoricalScenarioLibrary::seed_digest);
        if !ctx.quiet {
            print_provenance_summary(&ctx.stdout, &provenance);
        }
        let provenance_path = ctx.output_dir.join("training/model_provenance.json");
        write_provenance_report(&provenance_path, &provenance).map_err(|e| CliError::Internal {
            message: format!("failed to write provenance report: {e}"),
        })?;
    }

    if root_config.is_some_and(|c| c.exports.stochastic) {
        if !ctx.quiet {
            let _ = ctx.stdout.write_line("Exporting stochastic artifacts...");
        }
        let stderr = &ctx.stderr;
        let quiet = ctx.quiet;
        let mut on_warning = |msg: &str| {
            if !quiet {
                let _ = stderr.write_line(&format!("warning: stochastic export failed ({msg})"));
            }
        };
        export_stochastic_artifacts(
            &ctx.output_dir,
            &setup.inputs.stochastic,
            system,
            root_estimation_report,
            &mut on_warning,
        );
    }

    let scaling_path = ctx.output_dir.join("training/scaling_report.json");
    write_scaling_report(&scaling_path, &setup.inputs.stage_data.scaling_report).map_err(|e| {
        CliError::Internal {
            message: format!("failed to write scaling report: {e}"),
        }
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use console::Term;
    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use tempfile::TempDir;

    use novomodelo_core::ScalarParameter;
    use novomodelo_core::System;
    use novomodelo_io::BroadcastScalarParameter;
    use novomodelo_sddp::PrepareStochasticResult;
    use novomodelo_sddp::cut::CutSet;
    use novomodelo_sddp::prepare_hydro_models;
    use novomodelo_sddp::setup::study_stage_noise_group_ids;
    use novomodelo_sddp::test_support::decks::{Deck, SLOW_DECKS, committed_decks};
    use novomodelo_sddp::test_support::template_fact_groups;
    use novomodelo_stochastic::StochasticContext;
    use novomodelo_stochastic::context::OpeningTree;

    use super::{
        build_study_setup, load_case_and_config, reconstruct_stochastic_context_non_root,
        user_supplied_opening_tree,
    };
    use crate::commands::broadcast::BroadcastOpeningTree;
    use crate::commands::resolve_output_dir;
    use crate::commands::run::{CommBackendArg, RunArgs};
    use crate::error::CliError;

    fn d29_case_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d29-weekly-par-noise-sharing")
    }

    /// D29's 4 weekly study stages all declare `season_id=0`, so
    /// `study_stage_noise_group_ids` groups them into a single shared noise
    /// group — the sharing case a monthly (all-unique-groups) fixture cannot
    /// exercise.
    #[test]
    fn non_root_opening_tree_matches_rank_0_under_shared_noise_groups() {
        let (system, rank0, non_root) = load_rank0_and_rebuild_non_root(&d29_case_dir());
        let ids = study_stage_noise_group_ids(&system);
        assert!(
            ids.windows(2).any(|w| w[0] == w[1]),
            "D29 fixture must exercise shared noise groups; got {ids:?}",
        );
        assert_same_tree(
            rank0.opening_tree(),
            non_root.opening_tree(),
            "shared noise groups",
        );
    }

    #[test]
    fn non_root_opening_tree_matches_rank_0_under_shared_noise_groups_with_moment_matching() {
        let case = TempDir::new().expect("tempdir");
        copy_dir_recursive(&d29_case_dir(), case.path());
        set_moment_matching(case.path(), "per_coordinate");
        let (system, rank0, non_root) = load_rank0_and_rebuild_non_root(case.path());
        let ids = study_stage_noise_group_ids(&system);
        assert!(
            ids.windows(2).any(|w| w[0] == w[1]),
            "D29 fixture must exercise shared noise groups; got {ids:?}",
        );
        assert_same_tree(
            rank0.opening_tree(),
            non_root.opening_tree(),
            "shared noise groups, per_coordinate",
        );
        let (_system, drawn, _non_root) = load_rank0_and_rebuild_non_root(&d29_case_dir());
        assert_ne!(
            tree_bits(rank0.opening_tree()),
            tree_bits(drawn.opening_tree()),
            "per_coordinate must change the D29 tree, or the rank comparison above has no power"
        );
    }

    #[test]
    fn non_root_opening_tree_matches_rank_0_when_external_scenarios_clamp_openings() {
        let mut leg_bits = Vec::new();
        for moment_matching in ["none", "per_coordinate"] {
            let case = d56_with_num_openings(3, moment_matching);
            let (_system, rank0, non_root) = load_rank0_and_rebuild_non_root(case.path());
            assert_eq!(
                rank0.opening_tree().openings_per_stage_slice(),
                &[1, 1],
                "fixture must clamp every stage to the external library's scenario count"
            );
            assert_same_tree(
                rank0.opening_tree(),
                non_root.opening_tree(),
                &format!("external clamp, {moment_matching}"),
            );
            leg_bits.push(tree_bits(rank0.opening_tree()));
        }
        assert_eq!(
            leg_bits[0], leg_bits[1],
            "per_coordinate must leave a one-opening stage as drawn"
        );
    }

    #[test]
    fn non_root_opening_tree_matches_rank_0_when_external_scenarios_clamp_a_matched_tree() {
        let mut leg_bits = Vec::new();
        for moment_matching in ["none", "per_coordinate"] {
            let case = TempDir::new().expect("tempdir");
            copy_dir_recursive(&d56_case_dir(), case.path());
            crate::varying_external_deck::write_varying_external_deck(case.path(), moment_matching);
            let (_system, rank0, non_root) = load_rank0_and_rebuild_non_root(case.path());
            assert_eq!(
                rank0.opening_tree().openings_per_stage_slice(),
                &[3, 3],
                "fixture must clamp the 5 openings of every stage to the 3 external scenarios"
            );
            assert_same_tree(
                rank0.opening_tree(),
                non_root.opening_tree(),
                &format!("external clamp, {moment_matching}"),
            );
            leg_bits.push(tree_bits(rank0.opening_tree()));
        }
        assert_ne!(
            leg_bits[0], leg_bits[1],
            "per_coordinate must change a clamped stage with several openings, or the rank comparison above has no power"
        );
    }

    fn load_rank0_and_rebuild_non_root(
        case_dir: &Path,
    ) -> (System, StochasticContext, StochasticContext) {
        let args = RunArgs {
            case_dir: case_dir.to_path_buf(),
            output: None,
            quiet: true,
            threads: None,
            comm_backend: CommBackendArg::Local,
        };
        let (prepared, _hydro_models, bcast, _config, _scalars, _timings) = load_case_and_config(
            &args,
            &resolve_output_dir(&args.case_dir, None),
            true,
            &Term::stderr(),
        )
        .expect("case must load and prepare the stochastic context on rank 0");
        let bcast_tree = user_supplied_opening_tree(&prepared.stochastic);
        let PrepareStochasticResult {
            system, stochastic, ..
        } = prepared;
        let non_root_system = broadcast_round_trip(&system).expect("System must survive the wire");
        let non_root_bcast =
            broadcast_round_trip(&bcast).expect("BroadcastConfig must survive the wire");
        let non_root_tree: Option<BroadcastOpeningTree> =
            broadcast_round_trip(&bcast_tree).expect("opening tree must survive the wire");
        let user_tree =
            non_root_tree.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        let non_root = reconstruct_stochastic_context_non_root(
            &non_root_system,
            &non_root_bcast,
            user_tree,
            non_root_bcast.seed,
            case_dir,
        )
        .expect("non-root reconstruction must succeed");
        (system, stochastic, non_root)
    }

    fn assert_same_tree(rank0: &OpeningTree, non_root: &OpeningTree, what: &str) {
        assert_eq!(
            non_root.openings_per_stage_slice(),
            rank0.openings_per_stage_slice(),
            "non-root openings per stage must match rank 0's ({what})"
        );
        assert_eq!(
            non_root.dim(),
            rank0.dim(),
            "non-root dim must match rank 0's ({what})"
        );
        assert_eq!(
            tree_bits(non_root),
            tree_bits(rank0),
            "non-root tree data must match rank 0's bit for bit ({what})"
        );
    }

    fn tree_bits(tree: &OpeningTree) -> Vec<u64> {
        tree.data().iter().map(|v| v.to_bits()).collect()
    }

    fn set_moment_matching(case: &Path, moment_matching: &str) {
        crate::varying_external_deck::edit_json(&case.join("config.json"), |config| {
            config["training"]["scenario_source"]["openings"] =
                serde_json::json!({"source": "generated", "moment_matching": moment_matching});
        });
    }

    fn d56_case_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d56-external-authoritative")
    }

    fn d56_with_num_openings(num_openings: u32, moment_matching: &str) -> TempDir {
        let case = TempDir::new().expect("tempdir");
        copy_dir_recursive(&d56_case_dir(), case.path());
        set_moment_matching(case.path(), moment_matching);
        crate::varying_external_deck::edit_json(&case.path().join("stages.json"), |stages| {
            for stage in stages["stages"]
                .as_array_mut()
                .expect("stages.json must hold a stages array")
            {
                stage["num_openings"] = serde_json::json!(num_openings);
            }
        });
        case
    }

    fn copy_dir_recursive(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).expect("destination directory must be creatable");
        for entry in fs::read_dir(src).expect("source directory must be readable") {
            let entry = entry.expect("directory entry must be readable");
            let to = dst.join(entry.file_name());
            if entry
                .file_type()
                .expect("file type must be readable")
                .is_dir()
            {
                copy_dir_recursive(&entry.path(), &to);
            } else {
                fs::copy(entry.path(), &to).expect("file must be copyable");
            }
        }
    }

    /// Rank 0's own `broadcast_value` returns its input undecoded, so this is
    /// the actual wire a non-root rank exercises, not `broadcast_value` over a
    /// single-rank communicator.
    fn broadcast_round_trip<T: Serialize + DeserializeOwned>(value: &T) -> Result<T, CliError> {
        let bytes = postcard::to_allocvec(value).map_err(|e| CliError::Internal {
            message: format!("serialization error: {e}"),
        })?;
        postcard::from_bytes(&bytes).map_err(|e| CliError::Internal {
            message: format!("deserialization error: {e}"),
        })
    }

    /// A non-root rank rebuilds the stochastic context and hydro models from
    /// the case directory instead of rank 0's broadcast/artifact path, and
    /// this must produce byte-identical stage-LP template facts on every
    /// committed deck.
    #[test]
    fn non_root_rebuild_matches_rank_0_template_facts_on_every_committed_deck() {
        let build_both = |deck: &Deck| -> Result<[BTreeMap<&'static str, Vec<u8>>; 2], CliError> {
            let args = RunArgs {
                case_dir: deck.dir.clone(),
                output: None,
                quiet: true,
                threads: None,
                comm_backend: CommBackendArg::Local,
            };
            let (prepared, hydro_models, mut bcast, _config, scalars, _timings) =
                load_case_and_config(
                    &args,
                    &resolve_output_dir(&args.case_dir, None),
                    true,
                    &Term::stderr(),
                )?;

            let bcast_tree = user_supplied_opening_tree(&prepared.stochastic);
            let bcast_params: Vec<BroadcastScalarParameter> =
                scalars.iter().map(BroadcastScalarParameter::from).collect();

            let PrepareStochasticResult {
                system, stochastic, ..
            } = prepared;

            // Encode/decode every payload before rank 0's build_study_setup
            // takes fields from bcast and consumes stochastic.
            let non_root_system = broadcast_round_trip(&system)?;
            let mut non_root_bcast = broadcast_round_trip(&bcast)?;
            let non_root_tree: Option<BroadcastOpeningTree> = broadcast_round_trip(&bcast_tree)?;
            let non_root_params: Vec<BroadcastScalarParameter> =
                broadcast_round_trip(&bcast_params)?;

            let user_tree: Option<OpeningTree> = non_root_tree
                .map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
            let non_root_stochastic = reconstruct_stochastic_context_non_root(
                &non_root_system,
                &non_root_bcast,
                user_tree,
                non_root_bcast.seed,
                &deck.dir,
            )?;
            assert_same_tree(
                stochastic.opening_tree(),
                non_root_stochastic.opening_tree(),
                &deck.key,
            );
            let non_root_hydro_models = prepare_hydro_models(&non_root_system, &deck.dir, false)
                .map_err(|e| CliError::Internal {
                    message: format!("hydro model preprocessing error on non-root rank: {e}"),
                })?;
            let non_root_scalars: Vec<ScalarParameter> = non_root_params
                .into_iter()
                .map(ScalarParameter::from)
                .collect();
            let non_root_setup = build_study_setup(
                &non_root_system,
                &mut non_root_bcast,
                non_root_stochastic,
                non_root_hydro_models,
                non_root_scalars,
            )?;

            let rank0_scalars: Vec<ScalarParameter> = bcast_params
                .into_iter()
                .map(ScalarParameter::from)
                .collect();
            let rank0_setup =
                build_study_setup(&system, &mut bcast, stochastic, hydro_models, rank0_scalars)?;

            Ok([
                template_fact_groups(&rank0_setup),
                template_fact_groups(&non_root_setup),
            ])
        };

        let slow_tests_enabled = cfg!(feature = "slow-tests");
        let mut failures: Vec<String> = Vec::new();
        let mut compared = 0_usize;
        let mut skipped = 0_usize;

        for deck in committed_decks() {
            if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
                skipped += 1;
                continue;
            }
            let deck_key = deck.key.as_str();
            match build_both(&deck) {
                Err(e) => failures.push(format!("{deck_key}: {e}")),
                Ok([rank_0, non_root]) => {
                    compared += 1;
                    if rank_0.is_empty() {
                        failures.push(format!("{deck_key}: rank 0 has no template fact groups"));
                    }
                    for (group, bytes0) in &rank_0 {
                        match non_root.get(group) {
                            None => failures
                                .push(format!("{deck_key}: fact group `{group}` only on rank 0")),
                            Some(bytes1) => {
                                if bytes0 != bytes1 {
                                    let first_diff = bytes0
                                        .iter()
                                        .zip(bytes1.iter())
                                        .position(|(a, b)| a != b)
                                        .unwrap_or_else(|| bytes0.len().min(bytes1.len()));
                                    failures.push(format!(
                                        "{deck_key}: fact group `{group}` differs (rank 0 {} bytes, non-root {} bytes, first difference at byte {first_diff})",
                                        bytes0.len(),
                                        bytes1.len()
                                    ));
                                }
                            }
                        }
                    }
                    for group in non_root.keys() {
                        if !rank_0.contains_key(group) {
                            failures.push(format!(
                                "{deck_key}: fact group `{group}` only on the non-root rank"
                            ));
                        }
                    }
                }
            }
        }

        assert!(compared > 0, "no committed deck was compared");
        assert_eq!(
            compared,
            committed_decks().len() - skipped,
            "every committed deck (minus SLOW_DECKS skips) must be compared"
        );
        assert!(
            failures.is_empty(),
            "non-root setup diverges from rank 0 on {} item(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    #[test]
    fn non_root_rebuild_carries_the_simulation_cut_set() {
        let case_dir = d29_case_dir();
        let args = RunArgs {
            case_dir: case_dir.clone(),
            output: None,
            quiet: true,
            threads: None,
            comm_backend: CommBackendArg::Local,
        };
        let (prepared, _hydro_models, mut bcast, _config, scalars, _timings) =
            load_case_and_config(
                &args,
                &resolve_output_dir(&args.case_dir, None),
                true,
                &Term::stderr(),
            )
            .expect("D29 must load on rank 0");
        let system = broadcast_round_trip(&prepared.system).expect("system round trip");

        for cut_set in [CutSet::All, CutSet::Active] {
            bcast.simulation_cut_set = cut_set;
            let mut non_root_bcast = broadcast_round_trip(&bcast).expect("config round trip");
            let stochastic = reconstruct_stochastic_context_non_root(
                &system,
                &non_root_bcast,
                None,
                non_root_bcast.seed,
                &case_dir,
            )
            .expect("non-root reconstruction must succeed for D29");
            let hydro_models = prepare_hydro_models(&system, &case_dir, false)
                .expect("hydro model preprocessing must succeed for D29");
            let setup = build_study_setup(
                &system,
                &mut non_root_bcast,
                stochastic,
                hydro_models,
                scalars.clone(),
            )
            .expect("non-root setup must build for D29");

            assert_eq!(setup.simulation_cut_set, cut_set);
        }
    }
}
