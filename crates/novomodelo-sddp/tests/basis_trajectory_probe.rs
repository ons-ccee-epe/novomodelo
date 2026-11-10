//! Probe substrate and lag-fold probes over the reduced ago/26 case.
//!
//! Kept entry points:
//!
//! - `row_classifier::the_row_classifier_places_every_row_of_every_study` —
//!   fast, non-ignored, template-only sweep of the shared row classifier
//!   over every committed deck and structural study, in both inflow forms
//!   and the expanded rebuild of each compact form.
//! - `lag_fold_chain_rule_identity` (`#[ignore]`) — verifies the pinned
//!   inflow-lag columns' coupling channels and the chain-rule
//!   reconstruction of Benders cut coefficients from row duals + pool
//!   coefficients (~10 min, trains one checkpoint).
//! - `lag_fold_static_ab` (`#[ignore]`) — original-vs-cut-row-folded twin
//!   performance A/B over full backward chains (~1 h, trains three
//!   checkpoints).
//! - `by_node_backward_aggregation_share_on_the_reduced_case` (`#[ignore]`) —
//!   share of the compact-LP `by_node` backward pass spent in the serial
//!   aggregation after each stage's parallel solves, on 14 threads, projected
//!   to the production worker count (ten iterations).
//! - `compact_and_expanded_agree_at_identical_pinned_states_on_the_reduced_case`
//!   (`#[ignore]`) — the cut-row RHS and lifted-solution checks of the compact
//!   and expanded LPs at stored incoming states of three stages (three
//!   iterations, two forward passes).
//! - `cut_dual_counts` — the non-zero cut-dual count per solve, the touched-slot
//!   count per trial state and the stage union of non-zero slots, against their
//!   structural bounds: on the three-opening lag study, on every committed lagged
//!   deck (`slow-tests`) and, printed per stage, on the reduced ago/26 case
//!   (`#[ignore]`, the deck's own configuration).
//!
//! ```text
//! cargo nextest run --release -p novomodelo-sddp --features test-support --test basis_trajectory_probe \
//!     --run-ignored ignored-only -E 'test(lag_fold_chain_rule_identity)'
//! ```
//!
//! `case_dir` resolves the reduced ago/26 case, the bridge's conversion of the
//! raw `newave-ago-26-reduzido` deck, from `$HOME` — it lives outside this
//! repository (in the bridge checkout), so no repo-relative path can name it;
//! the probes panic with a clear message if the case is absent rather than
//! silently skipping.
//!
//! `StageGeometry::fpha` exposes the FPHA row range these probes read
//! (per-hydro plane counts are non-uniform, so no exposed count multiplies
//! out to it); `training_ctx()`/`solve_stage_for_probe` are the test-support
//! entry points supplying the rest of the reach.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::single_match_else
)]

use novomodelo_io::Config;
use novomodelo_io::config::{
    BackwardScheduler, SelectionMethod, StoppingRuleConfig, TrainingSelection,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use novomodelo_core::{TrainingEvent, WorkerTimingPhase};
use novomodelo_solver::{ActiveSolver, SolverInterface, StageTemplate};

use novomodelo_sddp::cut::pool::CutPool;
use novomodelo_sddp::forward::{ForwardPassBatch, run_forward_pass};
use novomodelo_sddp::indexer::StateDim;
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::setup::NodeId;
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::test_support::expander::expand_stage_templates;
use novomodelo_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
use novomodelo_sddp::workspace::{BasisStore, CapturedBasis, SolverWorkspace};
use novomodelo_sddp::{
    PrepareHydroModelsResult, PrepareStochasticResult, StudySetup, TrajectoryRecord,
};

mod common;
use common::expanded_solve::{PinnedStateRun, stored_incoming_states};
use common::row_families::{self, RowClass};
use common::{Emission, StubComm};

// ---------------------------------------------------------------------------
// Row-family classification
// ---------------------------------------------------------------------------

mod row_classifier {
    use std::collections::BTreeMap;

    use novomodelo_sddp::test_support::expander::expand_stage_templates;

    use super::common::row_families::{
        RowClass, census, classify_expanded_rows, classify_stage_rows,
    };
    use super::common::{expected_study_count, for_each_study_in_both_forms};

    const CUT_ROWS: usize = 2;

    fn assert_trailing_cuts(context: &str, classes: &[RowClass], base: usize) {
        assert_eq!(classes.len(), base + CUT_ROWS, "{context}: row count");
        assert_eq!(
            classes[base..],
            [RowClass::Cut; CUT_ROWS],
            "{context}: the rows past the structural rows"
        );
    }

    #[test]
    fn the_row_classifier_places_every_row_of_every_study() {
        let mut union: BTreeMap<&'static str, usize> = BTreeMap::new();
        let visited = for_each_study_in_both_forms(|key, system, logical, compact| {
            let expanded = expand_stage_templates(system, compact);
            let logical_templates = &logical.inputs.stage_data.stage_templates.templates;
            let compact_templates = &compact.inputs.stage_data.stage_templates.templates;
            for (stage, scaled) in expanded.scaled.iter().enumerate() {
                let context = format!("{key} stage {stage}");

                let logical_base = logical_templates[stage].num_rows;
                let logical_classes = classify_stage_rows(logical, stage, logical_base + CUT_ROWS);
                assert_trailing_cuts(
                    &format!("{context} logical"),
                    &logical_classes,
                    logical_base,
                );
                for (name, rows) in census(&logical_classes) {
                    *union.entry(name).or_insert(0) += rows;
                }

                let compact_base = compact_templates[stage].num_rows;
                let compact_classes = classify_stage_rows(compact, stage, compact_base + CUT_ROWS);
                assert_trailing_cuts(
                    &format!("{context} compact"),
                    &compact_classes,
                    compact_base,
                );

                let loaded = scaled.num_rows + CUT_ROWS;
                let expanded_classes =
                    classify_expanded_rows(compact, &expanded.layout, stage, loaded);
                let todays_classes = classify_stage_rows(logical, stage, loaded);
                let first_mismatch = (0..loaded)
                    .find(|&row| expanded_classes[row] != todays_classes[row])
                    .map(|row| (row, expanded_classes[row], todays_classes[row]));
                assert_eq!(
                    first_mismatch, None,
                    "{context}: first row (row, expanded, logical) whose class differs"
                );
            }
        });
        assert_eq!(
            visited,
            expected_study_count(),
            "the sweep visited a different number of studies than there are"
        );

        for family in [
            "water_balance",
            "load_balance",
            "fpha",
            "evaporation",
            "filling_target",
            "generic_constraint",
        ] {
            assert!(
                union.contains_key(family),
                "no study has a {family} row; reached {union:?}"
            );
        }
        assert!(
            [
                "anticipated_fishing",
                "anticipated_state_out_def",
                "anticipated_slot_definition",
            ]
            .iter()
            .any(|family| union.contains_key(family)),
            "no study has an anticipated row; reached {union:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Probe configuration
// ---------------------------------------------------------------------------

const SEED: u64 = 42;

fn case_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to resolve the reduced ago/26 case");
    let dir = PathBuf::from(home).join(
        "git/onstec-modelos/novomodelo-bridge/example/casos-exemplo-ct/newave/novomodelo-ago-26-reduzido",
    );
    assert!(
        dir.join("config.json").exists(),
        "reduced ago/26 case not found at {}; convert the raw deck newave-ago-26-reduzido with the \
         current bridge into novomodelo-ago-26-reduzido",
        dir.display()
    );
    dir
}

// ---------------------------------------------------------------------------
// Case loading (fresh per checkpoint)
// ---------------------------------------------------------------------------

/// The stochastic context and hydro models every probe setup is built from.
fn prepare_case(
    case_dir: &Path,
    config: &Config,
) -> (PrepareStochasticResult, PrepareHydroModelsResult) {
    let system =
        novomodelo_io::load_case(case_dir).expect("load_case must succeed for the reduced case");
    let source = config
        .training_scenario_source(&case_dir.join("config.json"))
        .expect("training_scenario_source must resolve");
    let prepared =
        novomodelo_sddp::setup::prepare_stochastic(system, case_dir, config, SEED, &source, None)
            .expect("prepare_stochastic must succeed for the reduced case");
    let hydro_models =
        novomodelo_sddp::hydro_models::prepare_hydro_models(&prepared.system, case_dir, false)
            .expect("prepare_hydro_models must succeed for the reduced case");
    (prepared, hydro_models)
}

/// Load a fresh `StudySetup` for `case_dir`, overriding `forward_passes` and
/// the iteration-limit stopping rule so each checkpoint is an independent,
/// from-scratch training run. With `cut_selection_off`, also overrides
/// `training.cut_selection` off: resident cuts then equal populated cuts,
/// reaching production-comparable density by deeper checkpoints.
fn fresh_setup(
    case_dir: &Path,
    forward_passes: u32,
    iteration_limit: u32,
    cut_selection_off: bool,
) -> StudySetup {
    let config_path = case_dir.join("config.json");
    let mut config = novomodelo_io::parse_config(&config_path).expect("config.json must parse");
    config.training.selection = Some(TrainingSelection::Sampled { forward_passes });
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: iteration_limit,
    }]);
    if cut_selection_off {
        config.training.cut_selection = novomodelo_io::config::RowSelectionConfig::default();
    }

    let (prepared, hydro_models) = prepare_case(case_dir, &config);
    StudySetup::new(
        &prepared.system,
        &config,
        prepared.stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup::new must build for the reduced case")
}

struct CheckpointRun {
    setup: StudySetup,
    frozen_templates: Vec<StageTemplate>,
}

/// Train `setup` to `iteration_limit`, returning the trained setup (its
/// `fcf` now holds every cut through that iteration) and the final frozen
/// templates.
fn train_checkpoint(
    case_dir: &Path,
    forward_passes: u32,
    iteration_limit: u32,
    threads: usize,
    cut_selection_off: bool,
) -> CheckpointRun {
    let mut setup = fresh_setup(case_dir, forward_passes, iteration_limit, cut_selection_off);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, threads, ActiveSolver::new, None, None)
        .expect("train() must not return Err on the reduced case");
    assert!(
        outcome.error.is_none(),
        "training error at checkpoint {iteration_limit}: {:?}",
        outcome.error
    );

    let frozen_templates = outcome
        .result
        .frozen_templates
        .expect("frozen_templates is always Some on a completed run");

    CheckpointRun {
        setup,
        frozen_templates,
    }
}

// ---------------------------------------------------------------------------
// Extra deterministic forward pass: trial states + forward bases
// ---------------------------------------------------------------------------

struct ForwardProbe {
    basis_store: BasisStore,
    records: Vec<TrajectoryRecord>,
}

/// Run one more deterministic forward pass against the checkpoint's trained
/// FCF, producing the trial states and forward-captured bases iteration
/// `checkpoint_iteration + 1`'s real backward pass would consume. Single-rank:
/// `local_forward_passes == total_forward_passes`, `fwd_offset == 0`; `threads`
/// only fans work across this rank's workspaces (`run_forward_pass`'s own
/// parallelism), it does not change which noise draws or which seed produce
/// the trajectory.
fn run_extra_forward_pass(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    forward_passes: u32,
    checkpoint_iteration: u64,
    threads: usize,
) -> ForwardProbe {
    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, threads, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let num_stages = setup.num_stages();
    let local_forward_passes = forward_passes as usize;

    let mut basis_store = BasisStore::new(local_forward_passes, num_stages);
    let mut records: Vec<TrajectoryRecord> = (0..local_forward_passes * num_stages)
        .map(|_| TrajectoryRecord::default())
        .collect();

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let batch = ForwardPassBatch {
        local_forward_passes,
        total_forward_passes: local_forward_passes,
        iteration: checkpoint_iteration + 1,
        fwd_offset: 0,
        event_sender: None,
    };

    let _forward_result = run_forward_pass(
        &mut pool.workspaces,
        &mut basis_store,
        &ctx,
        frozen_templates,
        &setup.fcf,
        &training_ctx,
        &batch,
        &mut records,
    )
    .expect("the extra deterministic forward pass must not error on the reduced case");

    ForwardProbe {
        basis_store,
        records,
    }
}

// ---------------------------------------------------------------------------
// Lag-fold chain-rule identity probe
// ---------------------------------------------------------------------------

/// Per-stage summary of the lag-fold identity checks (see
/// [`lag_fold_chain_rule_identity`]).
struct LagFoldStats {
    stage: usize,
    solves: usize,
    resident_cuts: usize,
    /// Max relative error of `rc == objective − Aᵀy` over every incoming
    /// state column's own CSC support, across all solves.
    max_rel_err_full_support: f64,
    /// Max relative error between frozen-template cut-row entries and
    /// `−coeff × col_scale` recomputed from the live pool.
    max_rel_err_pool_vs_template: f64,
    /// Max relative error of the chain-rule reconstruction (base-row dual
    /// terms + pool-coefficient GEMV) against the `rc / col_scale`
    /// cut-coefficient fingerprint, across all solves.
    max_rel_err_chain_rule: f64,
    /// Lag dims whose incoming column doubles as an outgoing render target
    /// (the shift alias); the remainder is the aged-out oldest lag per hydro.
    aliased_lag_dims: usize,
    unaliased_lag_dims: usize,
    /// The single non-state column cut rows touch (the theta column).
    theta_col: Option<usize>,
    /// Base-row families supporting the incoming storage columns (context
    /// for the fold design; storage stays in the LP).
    storage_support_families: BTreeMap<&'static str, usize>,
    /// Base-row entry counts per family over the incoming lag columns — the
    /// fold's per-node RHS patch sites and the chain rule's dual terms.
    lag_support_families: BTreeMap<&'static str, usize>,
    /// Base-row entry counts per family over the z-inflow columns — shows
    /// whether the water balance consumes `z_inflow` or bypasses it.
    z_col_support_families: BTreeMap<&'static str, usize>,
}

/// Replay every trial's opening chain at `stage` (same flow as
/// [`replay_stage`], no recording) and verify the identities a pinned-lag
/// fold relies on:
///
/// - **support**: an incoming inflow-lag column's base-template entries lie
///   only in its z-inflow and water-balance rows (measured: the PAR lag
///   combination is written into BOTH, at most one entry of each per lag
///   column) — any further family is a new coupling channel the fold design
///   has not accounted for (hard failure naming it);
/// - **full-support rc**: `reduced_cost == objective − Aᵀ·row_duals` over the
///   frozen template's CSC, per incoming state column (conventions check);
/// - **pool-vs-template**: every cut-row entry equals `−coeff × col_scale`
///   recomputed from `CutPool::active_cuts` at
///   `StateSpace::lp_column_for_state` (slot order = row order), and cut
///   rows touch no column outside the state render targets plus theta;
/// - **chain rule**: `rc/col_scale` on a lag column is reproduced from pool
///   coefficients and row duals alone — the z-row dual term plus the
///   shift-aliased entry of the cut-dual GEMV `g = Σ_c y_c · coeffs_c`.
fn lag_fold_check_stage(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    stage: usize,
    forward_probe: &ForwardProbe,
) -> LagFoldStats {
    assert!(stage >= 1, "the backward pass never solves stage 0");

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let state_space = setup.stage_state();
    let template = &frozen_templates[stage];
    let pool_ref: &CutPool = &setup.fcf.pools[stage];
    let opening_tree = training_ctx.stochastic.tree_view();
    let n_openings = opening_tree.n_openings(stage);
    let solve_order = opening_tree.solve_order_data(stage);

    let num_stages = setup.num_stages();
    let n_trials = forward_probe.records.len() / num_stages;
    let base_row_count = ctx.templates[stage].num_rows;
    let family_ids = row_families::classify_stage_rows(setup, stage, template.num_rows);

    let n_state = state_space.n_state;
    let n = state_space.hydro_count;
    let p = state_space.max_par_order;
    assert_eq!(
        n_state,
        n * (1 + p),
        "stage {stage}: state space is not storage+lags only (buckets/anticipated present) — \
         extend this probe's region arithmetic before trusting it"
    );

    let in_cols: Vec<usize> = (0..n_state)
        .map(|j| {
            state_space
                .state_to_lp_incoming_column(StateDim::new(j))
                .get()
        })
        .collect();
    let out_cols: Vec<usize> = (0..n_state)
        .map(|j| state_space.lp_column_for_state(StateDim::new(j)).get())
        .collect();
    let mut out_col_to_dim: BTreeMap<usize, usize> = BTreeMap::new();
    for (jj, &c) in out_cols.iter().enumerate() {
        assert!(
            out_col_to_dim.insert(c, jj).is_none(),
            "stage {stage}: two state dims render onto LP column {c}"
        );
    }

    let col_scale_at = |c: usize| -> f64 { template.col_scale.get(c).copied().unwrap_or(1.0) };
    let csc_col = |c: usize| {
        let s = template.col_starts[c] as usize;
        let e = template.col_starts[c + 1] as usize;
        (s..e).map(|k| (template.row_indices[k] as usize, template.values[k]))
    };
    let rel_err = |a: f64, b: f64| -> f64 { (a - b).abs() / b.abs().max(1.0) };

    // -- static structural scan (template-only, solve-independent) --

    let mut unexpected_lag_support: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut lag_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut storage_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut z_col_support_families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut aliased_lag_dims = 0usize;
    let mut unaliased_lag_dims = 0usize;
    for j in 0..n_state {
        let cin = in_cols[j];
        let mut z_rows = 0usize;
        let mut wb_rows = 0usize;
        for (r, _v) in csc_col(cin) {
            if r >= base_row_count {
                continue;
            }
            let fam = row_families::name(family_ids[r]);
            if j < n {
                *storage_support_families.entry(fam).or_insert(0) += 1;
            } else {
                *lag_support_families.entry(fam).or_insert(0) += 1;
                match family_ids[r] {
                    RowClass::Base(RowFamily::ZInflow) => z_rows += 1,
                    RowClass::Base(RowFamily::WaterBalance) => wb_rows += 1,
                    _ => {
                        *unexpected_lag_support.entry(fam).or_insert(0) += 1;
                    }
                }
            }
        }
        if j >= n {
            let lag = (j - n) / n;
            let h = (j - n) % n;
            assert!(
                z_rows <= 1 && wb_rows <= 1,
                "stage {stage}: incoming lag (h={h}, lag={lag}) has {z_rows} z-inflow and \
                 {wb_rows} water-balance entries — expected at most one of each (its own \
                 hydro's rows); a cross-hydro PAR coupling changes the fold's patch shape"
            );
            match out_col_to_dim.get(&cin) {
                Some(&alias) => {
                    aliased_lag_dims += 1;
                    assert_eq!(
                        alias,
                        n + (lag + 1) * n + h,
                        "stage {stage}: incoming lag (h={h}, lag={lag}) aliases state dim \
                         {alias}, not the expected (h={h}, lag={})",
                        lag + 1
                    );
                }
                None => {
                    unaliased_lag_dims += 1;
                    assert_eq!(
                        lag,
                        p - 1,
                        "stage {stage}: incoming lag (h={h}, lag={lag}) has no outgoing \
                         alias but is not the oldest lag"
                    );
                }
            }
        }
    }
    for h in 0..n {
        let zc = out_cols[n + h];
        for (r, _v) in csc_col(zc) {
            if r < base_row_count {
                *z_col_support_families
                    .entry(row_families::name(family_ids[r]))
                    .or_insert(0) += 1;
            }
        }
    }
    assert!(
        unexpected_lag_support.is_empty(),
        "stage {stage}: lag columns are supported by base rows outside \
         z_inflow/water_balance — a coupling channel the fold design has not accounted \
         for: {unexpected_lag_support:?}"
    );

    // -- pool-vs-template consistency (template-only) --

    let active: Vec<(usize, f64, &[f64])> = pool_ref.active_cuts().collect();
    let resident_cuts = active.len();
    assert_eq!(
        base_row_count + resident_cuts,
        template.num_rows,
        "stage {stage}: base_row_count + active cut count must equal frozen template rows"
    );

    let mut cut_entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for c in 0..template.num_cols {
        for (r, v) in csc_col(c) {
            if r >= base_row_count {
                cut_entries.insert((r, c), v);
            }
        }
    }
    let mut max_rel_err_pool_vs_template = 0.0f64;
    for (idx, (_slot, _intercept, coeffs)) in active.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            n_state,
            "stage {stage}: cut-state projection is not identity — map projected dims \
             before trusting the alias arithmetic"
        );
        let row = base_row_count + idx;
        for jj in 0..n_state {
            let col = out_cols[jj];
            let expected = -coeffs[jj] * col_scale_at(col);
            let actual = cut_entries.get(&(row, col)).copied().unwrap_or(0.0);
            max_rel_err_pool_vs_template =
                max_rel_err_pool_vs_template.max(rel_err(actual, expected));
        }
    }
    let extra_cols: BTreeSet<usize> = cut_entries
        .keys()
        .filter(|(_r, c)| !out_col_to_dim.contains_key(c))
        .map(|&(_r, c)| c)
        .collect();
    assert!(
        extra_cols.len() <= 1,
        "stage {stage}: cut rows touch more than one non-state column: {extra_cols:?}"
    );
    let theta_col = extra_cols.iter().next().copied();

    // -- per-solve numerical identities --

    let mut max_rel_err_full_support = 0.0f64;
    let mut max_rel_err_chain_rule = 0.0f64;
    let mut solves = 0usize;
    let mut gemv = vec![0.0f64; n_state];

    for m in 0..n_trials {
        let x_hat: &[f64] = &forward_probe.records[m * num_stages + (stage - 1)].state;
        let forward_basis: &CapturedBasis = forward_probe
            .basis_store
            .get(m, NodePos(stage))
            .unwrap_or_else(|| panic!("no forward-captured basis at (trial {m}, stage {stage})"));

        ws.solver.reset_solver_state();
        ws.solver.load_model(template);

        for omega_position in 0..n_openings {
            let omega = solve_order[omega_position] as usize;
            let raw_noise = opening_tree.opening(stage, omega);
            patch_backward_opening_for_probe(
                ws,
                &ctx,
                &training_ctx,
                StageIdx(stage),
                x_hat,
                raw_noise,
            );

            let stored_basis = (omega_position == 0).then_some(forward_basis);
            let view = solve_stage_for_probe(
                ws,
                &ctx,
                pool_ref,
                stored_basis,
                StageIdx(stage),
                m,
                NodeId(stage as i32),
            )
            .expect("stage solve must not error on the reduced case");
            let rc = view.reduced_costs;
            let y = view.dual;
            solves += 1;

            for j in 0..n_state {
                let cin = in_cols[j];
                let mut manual = template.objective[cin];
                for (r, v) in csc_col(cin) {
                    manual -= v * y[r];
                }
                max_rel_err_full_support = max_rel_err_full_support.max(rel_err(manual, rc[cin]));
            }

            // The fold's own kernel: g = Σ_c y_c · coeffs_c over resident cuts,
            // then per lag dim the z-row dual term plus the aliased g entry.
            gemv.fill(0.0);
            for (idx, (_slot, _intercept, coeffs)) in active.iter().enumerate() {
                let yc = y[base_row_count + idx];
                if yc != 0.0 {
                    for (g, &b) in gemv.iter_mut().zip(coeffs.iter()) {
                        *g += yc * b;
                    }
                }
            }
            for j in n..n_state {
                let cin = in_cols[j];
                let cs = col_scale_at(cin);
                let mut reconstructed = 0.0f64;
                for (r, v) in csc_col(cin) {
                    if r < base_row_count {
                        reconstructed -= (v / cs) * y[r];
                    }
                }
                if let Some(&alias) = out_col_to_dim.get(&cin) {
                    reconstructed += gemv[alias];
                }
                max_rel_err_chain_rule =
                    max_rel_err_chain_rule.max(rel_err(reconstructed, rc[cin] / cs));
            }
        }
    }

    LagFoldStats {
        stage,
        solves,
        resident_cuts,
        max_rel_err_full_support,
        max_rel_err_pool_vs_template,
        max_rel_err_chain_rule,
        aliased_lag_dims,
        unaliased_lag_dims,
        theta_col,
        storage_support_families,
        lag_support_families,
        z_col_support_families,
    }
}

/// Lag-fold feasibility probe: trains one checkpoint of the reduced ago/26
/// case (cut selection off, so resident-cut mass is production-shaped), then
/// verifies on every replayed backward solve that the pinned inflow-lag
/// columns couple to the LP through exactly three channels (their z-inflow
/// row, their water-balance row, and the resident cut rows) and that their
/// reduced costs — the Benders cut coefficients — are reproduced bit-tightly
/// from row duals and pool coefficients alone. See [`lag_fold_check_stage`]
/// for the four identities.
#[test]
#[ignore = "slow: trains a real checkpoint against the reduced ago/26 case; \
            run with --run-ignored --release"]
fn lag_fold_chain_rule_identity() {
    let case = case_dir();
    let forward_passes: u32 = 8;
    let checkpoint: u32 = 10;
    let threads: usize = 8;

    let ckpt = train_checkpoint(&case, forward_passes, checkpoint, threads, true);
    let fp = run_extra_forward_pass(
        &ckpt.setup,
        &ckpt.frozen_templates,
        forward_passes,
        u64::from(checkpoint),
        threads,
    );

    for &stage in &[5usize, 32, 60] {
        let stats = lag_fold_check_stage(&ckpt.setup, &ckpt.frozen_templates, stage, &fp);
        println!(
            "stage {}: solves={} resident_cuts={} | rc-vs-Aᵀy max rel {:.3e} | \
             pool-vs-template max rel {:.3e} | chain-rule max rel {:.3e} | \
             aliased/unaliased lag dims {}/{} | theta_col={:?} | \
             lag support {:?} | z-col support {:?} | storage support {:?}",
            stats.stage,
            stats.solves,
            stats.resident_cuts,
            stats.max_rel_err_full_support,
            stats.max_rel_err_pool_vs_template,
            stats.max_rel_err_chain_rule,
            stats.aliased_lag_dims,
            stats.unaliased_lag_dims,
            stats.theta_col,
            stats.lag_support_families,
            stats.z_col_support_families,
            stats.storage_support_families,
        );
        assert!(
            stats.max_rel_err_full_support <= 1e-6,
            "stage {stage}: rc != objective − Aᵀy beyond tolerance \
             (max rel {:.3e}) — the solver reports transformed duals/rcs",
            stats.max_rel_err_full_support
        );
        assert!(
            stats.max_rel_err_pool_vs_template <= 1e-12,
            "stage {stage}: frozen-template cut entries drift from pool coefficients \
             (max rel {:.3e})",
            stats.max_rel_err_pool_vs_template
        );
        assert!(
            stats.max_rel_err_chain_rule <= 1e-6,
            "stage {stage}: chain-rule reconstruction diverges from the recorded \
             cut-coefficient fingerprint (max rel {:.3e})",
            stats.max_rel_err_chain_rule
        );
    }
}

// ---------------------------------------------------------------------------
// Lag-fold static A/B: original vs cut-row-folded twin
// ---------------------------------------------------------------------------

/// Build this trial's folded twin: remove every pinned-lag-column entry from
/// the cut rows and absorb it into the cut-row bounds at the trial's pinned
/// (scaled) lag values. Base matrix, columns, pins, and row/column counts
/// are untouched, so the production patch path and the forward-basis warm
/// start drive the twin verbatim; only cut-row density changes (the fold's
/// dominant claimed effect — the measured speedup is therefore a lower
/// bound on a full fold, which would also drop the lag columns and their
/// two base-row entries each).
fn fold_cut_rows_for_trial(
    template: &StageTemplate,
    base_row_count: usize,
    lag_cols: &BTreeMap<usize, f64>,
) -> StageTemplate {
    let mut folded = template.clone();

    let mut row_delta = vec![0.0f64; template.num_rows];
    let mut new_col_starts: Vec<i32> = Vec::with_capacity(template.num_cols + 1);
    let mut new_row_indices: Vec<i32> = Vec::with_capacity(template.num_nz);
    let mut new_values: Vec<f64> = Vec::with_capacity(template.num_nz);
    new_col_starts.push(0);
    for c in 0..template.num_cols {
        let s = template.col_starts[c] as usize;
        let e = template.col_starts[c + 1] as usize;
        let pinned = lag_cols.get(&c);
        for k in s..e {
            let r = template.row_indices[k] as usize;
            match pinned {
                Some(&x_scaled) if r >= base_row_count => {
                    row_delta[r] += template.values[k] * x_scaled;
                }
                _ => {
                    new_row_indices.push(template.row_indices[k]);
                    new_values.push(template.values[k]);
                }
            }
        }
        new_col_starts.push(new_row_indices.len() as i32);
    }
    for r in base_row_count..template.num_rows {
        if folded.row_lower[r].is_finite() {
            folded.row_lower[r] -= row_delta[r];
        }
        if folded.row_upper[r].is_finite() {
            folded.row_upper[r] -= row_delta[r];
        }
    }
    folded.num_nz = new_values.len();
    folded.col_starts = new_col_starts;
    folded.row_indices = new_row_indices;
    folded.values = new_values;
    folded
}

/// One A/B row: chain totals for one (checkpoint, stage), both arms.
struct LagFoldAbStats {
    checkpoint: u32,
    stage: usize,
    resident_cuts: usize,
    nnz_orig: usize,
    nnz_folded: usize,
    solves_per_arm: usize,
    wall_orig_ms: f64,
    wall_folded_ms: f64,
    hop_wall_orig_ms: f64,
    hop_wall_folded_ms: f64,
    pivots_orig: u64,
    pivots_folded: u64,
    max_obj_rel_diff: f64,
}

/// Replay every trial's opening chain twice — original frozen template vs
/// its per-trial folded twin (`fold_cut_rows_for_trial`) — through the
/// identical patch/warm-start/solve flow, recording per-solve wall and
/// pivots for both arms and asserting per-opening objective equality (the
/// fold is an exact substitution of the pinned lag values; a bound bug or a
/// wrong pin reconstruction surfaces here, not as a silent timing artifact).
fn run_lag_fold_ab_stage(
    setup: &StudySetup,
    frozen_templates: &[StageTemplate],
    checkpoint: u32,
    stage: usize,
    forward_probe: &ForwardProbe,
) -> LagFoldAbStats {
    assert!(stage >= 1, "the backward pass never solves stage 0");

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let state_space = setup.stage_state();
    let template = &frozen_templates[stage];
    let pool_ref: &CutPool = &setup.fcf.pools[stage];
    let opening_tree = training_ctx.stochastic.tree_view();
    let n_openings = opening_tree.n_openings(stage);
    let solve_order = opening_tree.solve_order_data(stage);

    let num_stages = setup.num_stages();
    let n_trials = forward_probe.records.len() / num_stages;
    let base_row_count = ctx.templates[stage].num_rows;
    let resident_cuts = template.num_rows - base_row_count;

    let n_state = state_space.n_state;
    let n = state_space.hydro_count;

    let mut stats = LagFoldAbStats {
        checkpoint,
        stage,
        resident_cuts,
        nnz_orig: template.num_nz,
        nnz_folded: 0,
        solves_per_arm: 0,
        wall_orig_ms: 0.0,
        wall_folded_ms: 0.0,
        hop_wall_orig_ms: 0.0,
        hop_wall_folded_ms: 0.0,
        pivots_orig: 0,
        pivots_folded: 0,
        max_obj_rel_diff: 0.0,
    };

    let mut objectives_orig = vec![0.0f64; n_openings];

    for m in 0..n_trials {
        let x_hat: &[f64] = &forward_probe.records[m * num_stages + (stage - 1)].state;
        let forward_basis: &CapturedBasis = forward_probe
            .basis_store
            .get(m, NodePos(stage))
            .unwrap_or_else(|| panic!("no forward-captured basis at (trial {m}, stage {stage})"));

        // Pinned scaled lag values: the pin writes `v_orig / col_scale` (the
        // Benders subgradient contract's inverse direction).
        let lag_cols: BTreeMap<usize, f64> = (n..n_state)
            .map(|j| {
                let c = state_space
                    .state_to_lp_incoming_column(StateDim::new(j))
                    .get();
                let cs = template.col_scale.get(c).copied().unwrap_or(1.0);
                (c, x_hat[j] / cs)
            })
            .collect();
        let folded = fold_cut_rows_for_trial(template, base_row_count, &lag_cols);
        stats.nnz_folded = folded.num_nz;

        for (arm_folded, arm_template) in [(false, template), (true, &folded)] {
            ws.solver.reset_solver_state();
            ws.solver.load_model(arm_template);

            for omega_position in 0..n_openings {
                let omega = solve_order[omega_position] as usize;
                let raw_noise = opening_tree.opening(stage, omega);
                patch_backward_opening_for_probe(
                    ws,
                    &ctx,
                    &training_ctx,
                    StageIdx(stage),
                    x_hat,
                    raw_noise,
                );

                let stored_basis = (omega_position == 0).then_some(forward_basis);
                let view = solve_stage_for_probe(
                    ws,
                    &ctx,
                    pool_ref,
                    stored_basis,
                    StageIdx(stage),
                    m,
                    NodeId(stage as i32),
                )
                .expect("stage solve must not error on the reduced case");
                let wall_ms = view.solve_time_seconds * 1_000.0;
                let pivots = view.iterations;
                let objective = view.objective;

                if arm_folded {
                    let rel = (objective - objectives_orig[omega_position]).abs()
                        / objectives_orig[omega_position].abs().max(1.0);
                    stats.max_obj_rel_diff = stats.max_obj_rel_diff.max(rel);
                    assert!(
                        rel <= 1e-7,
                        "checkpoint {checkpoint} stage {stage} trial {m} chain_pos \
                         {omega_position}: folded objective {objective} diverges from \
                         original {} (rel {rel:.3e}) — the fold is not an exact \
                         substitution (bound arithmetic or pin reconstruction bug)",
                        objectives_orig[omega_position]
                    );
                    stats.wall_folded_ms += wall_ms;
                    stats.pivots_folded += pivots;
                    if omega_position > 0 {
                        stats.hop_wall_folded_ms += wall_ms;
                    }
                } else {
                    objectives_orig[omega_position] = objective;
                    stats.wall_orig_ms += wall_ms;
                    stats.pivots_orig += pivots;
                    if omega_position > 0 {
                        stats.hop_wall_orig_ms += wall_ms;
                    }
                    stats.solves_per_arm += 1;
                }
            }
        }
    }

    stats
}

/// Lag-fold static A/B (the fold's go/no-go performance readout): trains
/// checkpoints {3, 10, 30} of the reduced ago/26 case with cut selection off
/// (resident cuts ~24/80/240 — the density curve up to production-shaped
/// mass), then replays every trial's backward chain at stages {5, 32, 60}
/// on the original frozen template and on its per-trial cut-row-folded twin,
/// asserting per-opening objective equality and reporting wall/pivot ratios
/// per (checkpoint, stage).
#[test]
#[ignore = "slow (~1h): trains three checkpoints against the reduced ago/26 case; \
            run with --run-ignored --release"]
fn lag_fold_static_ab() {
    let case = case_dir();
    let forward_passes: u32 = 8;
    let threads: usize = 8;

    println!(
        "ckpt stage resident nnz_orig nnz_folded solves/arm | wall_orig_ms wall_folded_ms \
         speedup | hop_speedup | pivots_orig pivots_folded pivot_ratio | wall/pivot ratio | \
         max_obj_rel"
    );
    for &checkpoint in &[3u32, 10, 30] {
        let ckpt = train_checkpoint(&case, forward_passes, checkpoint, threads, true);
        let fp = run_extra_forward_pass(
            &ckpt.setup,
            &ckpt.frozen_templates,
            forward_passes,
            u64::from(checkpoint),
            threads,
        );
        for &stage in &[5usize, 32, 60] {
            let s =
                run_lag_fold_ab_stage(&ckpt.setup, &ckpt.frozen_templates, checkpoint, stage, &fp);
            let speedup = s.wall_orig_ms / s.wall_folded_ms.max(1e-9);
            let hop_speedup = s.hop_wall_orig_ms / s.hop_wall_folded_ms.max(1e-9);
            let pivot_ratio = s.pivots_orig as f64 / (s.pivots_folded as f64).max(1.0);
            let wpp_ratio = (s.wall_orig_ms / (s.pivots_orig as f64).max(1.0))
                / (s.wall_folded_ms / (s.pivots_folded as f64).max(1.0)).max(1e-12);
            println!(
                "{} {} {} {} {} {} | {:.1} {:.1} {:.3}x | {:.3}x | {} {} {:.3} | {:.3}x | {:.2e}",
                s.checkpoint,
                s.stage,
                s.resident_cuts,
                s.nnz_orig,
                s.nnz_folded,
                s.solves_per_arm,
                s.wall_orig_ms,
                s.wall_folded_ms,
                speedup,
                hop_speedup,
                s.pivots_orig,
                s.pivots_folded,
                pivot_ratio,
                wpp_ratio,
                s.max_obj_rel_diff,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// By-node backward aggregation share
// ---------------------------------------------------------------------------

/// Worker count of one production rank; the projection below moves with it.
const PRODUCTION_WORKERS: usize = 32;
const SHARE_PROBE_THREADS: usize = 14;
const SHARE_PROBE_ITERATIONS: u32 = 10;

/// Trains the reduced ago/26 case for ten iterations on the compact LP
/// under `by_node` with 14 threads and reports the share of the backward pass
/// spent in the serial aggregation after each stage's parallel solves:
/// `A / E` over the iterations, with `A` the reported aggregation time and `E`
/// the backward elapsed time.
///
/// The projection to `PRODUCTION_WORKERS` workers is `A / (A + K / W)`, with
/// `K` the backward wall time summed over every worker. That is the parallel
/// time when every stage has at least `W` work units, as the production deck
/// has, so the projected share is an upper estimate. The reduced deck keeps
/// the production deck's plants and PAR orders, so the aggregation work per
/// trial point matches and the share does not depend on the forward-pass
/// count. The probe measures and asserts no budget.
#[test]
#[ignore = "slow: trains ten iterations of the reduced ago/26 case on 14 threads; \
            run with --run-ignored --release"]
fn by_node_backward_aggregation_share_on_the_reduced_case() {
    let case = case_dir();
    let hardware = std::thread::available_parallelism()
        .expect("available_parallelism must resolve")
        .get();
    assert!(
        hardware >= SHARE_PROBE_THREADS,
        "the probe needs {SHARE_PROBE_THREADS} hardware threads, this host has {hardware}"
    );

    let mut config =
        novomodelo_io::parse_config(&case.join("config.json")).expect("config.json must parse");
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: SHARE_PROBE_ITERATIONS,
    }]);
    if matches!(
        config.training.cut_selection.selection,
        Some(SelectionMethod::Dynamic { .. })
    ) {
        config.training.cut_selection.selection = None;
        println!("row selection `dynamic` switched off: `by_node` never runs under it");
    }

    let (prepared, hydro_models) = prepare_case(&case, &config);
    let mut setup = common::build_setup_for_case_as(
        &case,
        &config,
        &prepared.system,
        prepared.stochastic,
        hydro_models,
        Emission::Compact,
    );
    setup.set_scheduler(BackwardScheduler::ByNode { block_size: None });

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
    let outcome = setup
        .train(
            &mut solver,
            &StubComm,
            SHARE_PROBE_THREADS,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train() must not return Err on the reduced case");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let mut elapsed_ms = 0.0;
    let mut aggregation_ms = 0.0;
    let mut worker_work_ms = 0.0;
    for event in event_rx.try_iter() {
        match event {
            TrainingEvent::BackwardPassComplete {
                elapsed_ms: wall,
                aggregation_time_ms,
                ..
            } => {
                elapsed_ms += wall as f64;
                aggregation_ms += aggregation_time_ms as f64;
            }
            TrainingEvent::WorkerTiming {
                phase: WorkerTimingPhase::Backward,
                timings,
                ..
            } => worker_work_ms += timings.backward_wall_ms,
            _ => {}
        }
    }

    let share = aggregation_ms / elapsed_ms;
    let projected = aggregation_ms / (aggregation_ms + worker_work_ms / PRODUCTION_WORKERS as f64);

    println!("backward aggregation share: {share:.4}");
    println!(
        "backward seconds: aggregation={:.3} worker work={:.3} elapsed={:.3}",
        aggregation_ms / 1000.0,
        worker_work_ms / 1000.0,
        elapsed_ms / 1000.0
    );
    println!(
        "projected backward aggregation share at {PRODUCTION_WORKERS} workers: {projected:.4}"
    );
    assert!(
        aggregation_ms <= elapsed_ms,
        "aggregation {aggregation_ms} ms exceeds the backward elapsed {elapsed_ms} ms"
    );
    assert!(worker_work_ms > 0.0, "no backward worker time was reported");
    for (name, value) in [("share", share), ("projected share", projected)] {
        assert!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "backward aggregation {name} {value} lies outside [0, 1]"
        );
    }
}

const PINNED_STATE_LABEL: &str = "reduced ago/26";
const PINNED_STATES_PER_STAGE: usize = 2;

/// Trains the reduced ago/26 case on the compact LP (two sampled forward passes, three iterations,
/// cut selection off, trial states exported) and checks the compact and expanded LPs at the first
/// two stored incoming states of stage 1, the middle stage and the second-to-last stage, every
/// opening each: each cut row's RHS against the expanded z rows' evaluation within its rounding
/// bound, and the lifted compact solution on every expanded row.
///
/// The deck must be converted by the current bridge; one converted by an older bridge is not the
/// case the probe is specified on:
///
/// ```text
/// $HOME/git/onstec-modelos/novomodelo-bridge/.venv/bin/novomodelo-bridge convert newave \
///     "$HOME/git/onstec-modelos/novomodelo-bridge/example/casos-exemplo-ct/newave/newave-ago-26-reduzido" \
///     "$HOME/git/onstec-modelos/novomodelo-bridge/example/casos-exemplo-ct/newave/novomodelo-ago-26-reduzido" \
///     --force
/// ```
#[test]
#[ignore = "trains the reduced ago/26 case; run with --run-ignored --release"]
fn compact_and_expanded_agree_at_identical_pinned_states_on_the_reduced_case() {
    let case = case_dir();
    let mut config =
        novomodelo_io::parse_config(&case.join("config.json")).expect("config.json must parse");
    config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]);
    config.training.cut_selection = novomodelo_io::config::RowSelectionConfig::default();

    let (prepared, hydro_models) = prepare_case(&case, &config);
    let mut compact = common::build_setup_for_case_as(
        &case,
        &config,
        &prepared.system,
        prepared.stochastic,
        hydro_models,
        Emission::Compact,
    );
    common::train_storing_trial_states(PINNED_STATE_LABEL, &mut compact);
    let expanded = expand_stage_templates(&prepared.system, &compact);

    let n_pools = compact.fcf.pools.len();
    let mut checked_stages = vec![1, n_pools / 2, n_pools - 2];
    checked_stages.dedup();
    let mut run = PinnedStateRun::new(PINNED_STATE_LABEL, &compact, &compact, &expanded);
    for stage in checked_stages {
        let mut states = stored_incoming_states(&compact, stage).unwrap_or_else(|| {
            panic!("{PINNED_STATE_LABEL} stage {stage}: the producer pool drops a state dimension")
        });
        states.truncate(PINNED_STATES_PER_STAGE);
        run.check_stage(stage, &states);
    }
    println!("{}", run.summary());
    run.assert_powers();
}

mod cut_dual_counts {
    use std::collections::BTreeSet;

    use novomodelo_core::System;
    use novomodelo_sddp::setup::StageIdx;
    use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
    use novomodelo_sddp::test_support::{
        FoldedProbeCut, FoldedProbeCuts, FoldedProbeOpening, cut_rows_for_slots_into,
        empty_row_batch, folded_opening_cut_for_probe,
    };
    use novomodelo_sddp::{RiskMeasure, StudySetup};
    use novomodelo_solver::{ActiveSolver, SolverInterface};

    use super::common::expanded_solve::{
        CutRowSet, stages_whose_producer_drops_a_state_dimension, stored_incoming_states,
    };
    use super::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
    use super::common::stored_cut_oracles::apply_oracle_config;
    use super::common::{
        Emission, StubComm, build_setup_for_case_as, build_setup_in_code_with_models_as,
        fresh_system_and_setup_with_emission, is_chain, train_storing_trial_states,
    };
    use super::{StoppingRuleConfig, case_dir, prepare_case};

    const LAG_STUDY_LABEL: &str = "fpha lag study";
    const LAG_STUDY_ITERATIONS: u32 = 5;
    const REDUCED_LABEL: &str = "reduced ago/26";

    struct StageCounts {
        stage: usize,
        bound: usize,
        resident_rows: usize,
        nnz: Vec<usize>,
        touched: Vec<usize>,
        union: usize,
    }

    fn record_cut_dual_counts(
        label: &str,
        system: &System,
        setup: &StudySetup,
        states_per_stage: usize,
    ) -> Vec<StageCounts> {
        let risk_measures: Vec<RiskMeasure> = system
            .stages()
            .iter()
            .filter(|stage| stage.id >= 0)
            .map(|stage| RiskMeasure::from(stage.risk_config))
            .collect();
        let ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let tree = training_ctx.stochastic.tree_view();
        let n_pools = setup.fcf.pools.len();
        let mut workspaces = setup
            .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
            .expect("the workspace pool must build");
        let ws = &mut workspaces.workspaces[0];
        let mut cut = FoldedProbeCut::new(&training_ctx);
        let quantiles = |values: &[usize]| -> [usize; 4] {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            let at = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
            [sorted[0], at(50), at(99), at(100)]
        };

        let mut dropped = Vec::new();
        let mut recorded = Vec::new();
        for t in 0..n_pools {
            let projection = &training_ctx.cut_state_layouts[t];
            let rows = CutRowSet::active(&setup.fcf, t, t.saturating_sub(1));
            let skipped = if t + 1 == n_pools {
                Some("terminal stage")
            } else if !projection.has_lag_slots() {
                Some("its pool has no lag slots")
            } else if rows.slots.is_empty() {
                Some("its pool holds no active cut")
            } else {
                None
            };
            if let Some(reason) = skipped {
                println!("{label}: stage {t} skipped: {reason}");
                continue;
            }

            let states = if t == 0 {
                vec![training_ctx.initial_state.to_vec()]
            } else {
                let Some(stored) = stored_incoming_states(setup, t) else {
                    println!(
                        "{label}: stage {t} dropped: its producer pool drops a state dimension"
                    );
                    dropped.push(t);
                    continue;
                };
                if stored.is_empty() {
                    println!("{label}: stage {t} skipped: its producer pool stored no state");
                    continue;
                }
                let picked = states_per_stage.min(stored.len());
                let wanted: Vec<usize> = (0..picked).map(|i| i * stored.len() / picked).collect();
                stored
                    .into_iter()
                    .enumerate()
                    .filter_map(|(k, x)| wanted.contains(&k).then_some(x))
                    .collect()
            };

            if let RiskMeasure::CVaR { alpha, lambda } = risk_measures[t] {
                assert!(
                    lambda < 1.0 || alpha >= 1.0,
                    "{label}: stage {t}: pure CVaR (lambda {lambda}, alpha {alpha}) can give an \
                     opening zero weight, so the openings with positive weight are not every opening"
                );
            }

            let template = &setup.inputs.stage_data.stage_templates.templates[t];
            let mut batch = empty_row_batch();
            cut_rows_for_slots_into(
                &mut batch,
                &setup.fcf.pools[t],
                &rows.slots,
                setup.stage_state(),
                projection,
                &template.col_scale,
            );
            let mut columns: BTreeSet<usize> = batch
                .col_indices
                .iter()
                .map(|&column| usize::try_from(column).expect("column index is non-negative"))
                .collect();
            columns.remove(&setup.stage_state().theta);
            let bound = 1 + columns.len();
            let resident_rows = rows.slots.len();
            ws.solver.load_model(template);
            ws.solver.add_rows(&batch);

            let openings = tree.n_openings(t);
            let touched_bound = (openings * bound).min(resident_rows);
            let (mut nnz, mut touched, mut union) = (Vec::new(), Vec::new(), BTreeSet::new());
            for (i, x) in states.iter().enumerate() {
                let mut aggregate = BTreeSet::new();
                for omega in 0..openings {
                    let context = format!("{label}: stage {t} state {i} opening {omega}");
                    folded_opening_cut_for_probe(
                        ws,
                        &ctx,
                        &training_ctx,
                        &FoldedProbeOpening {
                            stage: StageIdx(t),
                            pinned_state: x,
                            raw_noise: tree.opening(t, omega),
                            cuts: FoldedProbeCuts {
                                pool: &setup.fcf.pools[t],
                                pool_id: t,
                                slots: &rows.slots,
                            },
                            extracted_pool_id: rows.extracted_pool,
                        },
                        &mut cut,
                    )
                    .unwrap_or_else(|error| {
                        panic!("{context}: the stored-state solve failed: {error:?}")
                    });
                    let count = cut.cut_dual_slots.len();
                    assert!(
                        count <= bound,
                        "{context}: {count} non-zero cut duals exceed B_t {bound} (resident rows \
                         {resident_rows}, largest |y| {})",
                        cut.cut_dual_slots
                            .iter()
                            .map(|&(_, y)| y.abs())
                            .fold(0.0, f64::max)
                    );
                    nnz.push(count);
                    aggregate.extend(cut.cut_dual_slots.iter().map(|&(slot, _)| slot));
                }
                assert!(
                    aggregate.len() <= touched_bound,
                    "{label}: stage {t} state {i}: {} touched slots exceed min({openings} * B_t \
                     {bound}, resident rows {resident_rows})",
                    aggregate.len()
                );
                touched.push(aggregate.len());
                union.extend(aggregate);
            }

            println!(
                "{label}: t={t} R_t={resident_rows} B_t={bound} openings={openings} \
                 nnz[min/median/p99/max]={:?} touched[min/median/p99/max]={:?} stage_union={}",
                quantiles(&nnz),
                quantiles(&touched),
                union.len()
            );
            recorded.push(StageCounts {
                stage: t,
                bound,
                resident_rows,
                nnz,
                touched,
                union: union.len(),
            });
        }

        let listed = stages_whose_producer_drops_a_state_dimension(setup);
        for &stage in &dropped {
            assert!(
                listed.contains(&stage),
                "{label}: stage {stage} was dropped but is not among the stages whose producer \
                 drops a state dimension {listed:?}: pool state_dimension {}, projection n_slots {}",
                setup.fcf.pools[stage - 1].state_dimension,
                training_ctx.cut_state_layouts[stage - 1].n_slots()
            );
        }
        if !recorded.is_empty() {
            let pooled: Vec<usize> = recorded
                .iter()
                .flat_map(|counts| counts.touched.iter().copied())
                .collect();
            let (worst_stage, worst_ratio) = recorded
                .iter()
                .map(|counts| {
                    let peak = counts.nnz.iter().copied().max().unwrap_or(0);
                    (counts.stage, peak as f64 / counts.bound as f64)
                })
                .fold(
                    (0, 0.0),
                    |worst, next| if next.1 > worst.1 { next } else { worst },
                );
            println!(
                "{label}: {} stages recorded, dropped {dropped:?}; max nnz/B_t {worst_ratio:.4} \
                 (stage {worst_stage}); touched p99 {}; max stage union {}",
                recorded.len(),
                quantiles(&pooled)[2],
                recorded
                    .iter()
                    .map(|counts| counts.union)
                    .max()
                    .unwrap_or(0)
            );
        }
        recorded
    }

    #[test]
    fn cut_dual_counts_stay_within_the_structural_bound() {
        let options = FphaLagOptions {
            openings: 3,
            ..FphaLagOptions::default()
        };
        let (system, mut config, hydro_models) = fpha_lag_study_with(options);
        apply_oracle_config(&mut config);
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: LAG_STUDY_ITERATIONS,
        }]);
        let mut setup =
            build_setup_in_code_with_models_as(system, &config, hydro_models, Emission::Compact);
        train_storing_trial_states(LAG_STUDY_LABEL, &mut setup);
        let (system, _, _) = fpha_lag_study_with(options);

        let counts = record_cut_dual_counts(LAG_STUDY_LABEL, &system, &setup, 4);
        assert!(
            counts.iter().any(|stage| stage.stage >= 1),
            "{LAG_STUDY_LABEL}: no stage past the first was recorded"
        );
        assert!(
            counts.iter().flat_map(|stage| &stage.nnz).any(|&n| n > 0),
            "{LAG_STUDY_LABEL}: no recorded opening has a non-zero cut dual"
        );
        assert!(
            counts.iter().any(|stage| stage.resident_rows > stage.bound),
            "{LAG_STUDY_LABEL}: no recorded stage has more resident rows than its bound, so a \
             dense dual vector would pass; (stage, R_t, B_t) = {:?}",
            counts
                .iter()
                .map(|stage| (stage.stage, stage.resident_rows, stage.bound))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains every committed deck with lagged inflow"
    )]
    fn cut_dual_counts_stay_within_the_structural_bound_on_every_lagged_deck() {
        let (mut visited, mut not_chain, mut recorded) = (Vec::new(), Vec::new(), Vec::new());
        let mut some_non_zero = false;
        for deck in committed_decks() {
            if SLOW_DECKS.contains(&deck.key.as_str()) {
                continue;
            }
            let (system, mut setup) = fresh_system_and_setup_with_emission(
                &deck.dir,
                apply_oracle_config,
                Emission::Compact,
            );
            if setup.stage_state().max_par_order == 0 {
                continue;
            }
            if !is_chain(&setup) {
                println!("{}: skipped, not a chain", deck.key);
                not_chain.push(deck.key);
                continue;
            }
            visited.push(deck.key.clone());
            train_storing_trial_states(&deck.key, &mut setup);
            let counts = record_cut_dual_counts(&deck.key, &system, &setup, 8);
            some_non_zero |= counts.iter().flat_map(|stage| &stage.nnz).any(|&n| n > 0);
            if !counts.is_empty() {
                recorded.push(deck.key);
            }
        }
        println!(
            "visited {} lagged decks: {visited:?}; not chains: {not_chain:?}; recorded: {recorded:?}",
            visited.len()
        );
        assert!(
            !visited.is_empty(),
            "no committed lagged chain deck was visited"
        );
        assert!(
            some_non_zero,
            "no recorded opening of {recorded:?} has a non-zero cut dual"
        );
    }

    #[test]
    #[ignore = "trains the reduced ago/26 case; run with --run-ignored --release"]
    fn cut_dual_counts_on_the_reduced_case() {
        let case = case_dir();
        let config =
            novomodelo_io::parse_config(&case.join("config.json")).expect("config.json must parse");
        let (prepared, hydro_models) = prepare_case(&case, &config);
        let mut setup = build_setup_for_case_as(
            &case,
            &config,
            &prepared.system,
            prepared.stochastic,
            hydro_models,
            Emission::Compact,
        );
        train_storing_trial_states(REDUCED_LABEL, &mut setup);

        record_cut_dual_counts(REDUCED_LABEL, &prepared.system, &setup, 8);
    }
}
