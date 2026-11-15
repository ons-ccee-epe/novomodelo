//! Cut-validity, mask-soundness, and tightness oracles for stored Benders cuts, run over an in-code or committed study.

#![allow(clippy::expect_used, clippy::panic, clippy::cast_precision_loss)]

use std::sync::mpsc;

use novomodelo_core::TrainingEvent;
use novomodelo_io::Config;
use novomodelo_io::config::{RowSelectionConfig, SelectionMethod, StoppingRuleConfig};
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::forward::{ForwardPassBatch, run_forward_pass};
use novomodelo_sddp::indexer::{CutSlot, CutStateProjection, HydroSys};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::bound_recording::{BoundRecordingSolver, BoundWrite};
use novomodelo_sddp::test_support::{
    ProbeCut, ProbeCuts, ProbeOpening, inflow_slack_bounds_for_probe, opening_cut_for_probe,
    prep_for_probe_with_partial_sum, stage_state_box_bounds,
};
use novomodelo_sddp::workspace::{BasisStore, SolverWorkspace};
use novomodelo_sddp::{FutureCostFunction, SddpError, StudySetup, TrajectoryRecord};
use novomodelo_solver::{ActiveSolver, SolverError, SolverInterface, StageTemplate};
use novomodelo_stochastic::OpeningTreeView;

use super::expanded_solve::{CutRowSet, load_compact_cut_rows, scale_at};
use super::oracle_tolerance::value_tolerance;
use super::{StubComm, lift_to_global_state};

const ORACLE_ITERATIONS: u32 = 3;
const ORACLE_RANDOM_POINTS: usize = 5;
const ORACLE_SEED: u64 = 0x5EED_C0BE_5EED_C0BE;
const MASK_ABS_TOL: f64 = 1e-9;
const MASK_REL_TOL: f64 = 1e-9;
const ORACLE_MIN_FEASIBLE_POINTS: usize = 3;

/// Sets the training loop to run exactly [`ORACLE_ITERATIONS`] iterations with
/// cut selection disabled; every other config field is left as the study's own.
pub fn apply_oracle_config(config: &mut Config) {
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: ORACLE_ITERATIONS,
    }]);
    config.training.cut_selection = RowSelectionConfig::default();
}

/// [`apply_oracle_config`]'s settings with `selection` as the cut-selection method.
pub fn apply_oracle_config_with_selection(config: &mut Config, selection: SelectionMethod) {
    apply_oracle_config(config);
    config.training.cut_selection.selection = Some(selection);
}

/// Counters and violation lists from one [`run_cut_oracles`] run.
pub struct CutOracleReport {
    /// Producer-stage pools examined (those with at least one populated cut).
    pub pools_checked: usize,
    /// Total cuts walked across every checked pool, active or not.
    pub cuts_checked: usize,
    /// Of [`Self::cuts_checked`], the cuts whose activity flag is false.
    pub inactive_cuts_checked: usize,
    /// Checked pools whose successor pool holds an inactive cut, so validity ran
    /// against a different template than under-tightness.
    pub pools_with_inactive_successor_cuts: usize,
    /// Total feasible sample points examined across every checked pool.
    pub points_checked: usize,
    /// Final-iteration trial points whose pool envelope was compared with the
    /// successor value.
    pub under_tightness_points_checked: usize,
    /// Box-point probe solves that came back infeasible, skipped rather than
    /// counted as a violation.
    pub infeasible_sample_points: usize,
    /// Total off-mask cut-state slots examined across every checked pool.
    pub off_mask_slots_checked: usize,
    /// The minimum, over every checked pool, of its own feasible-point count.
    pub min_feasible_points: usize,
    /// (point, slack-bound signature) pairs whose successor value was re-solved
    /// with the slack bounds frozen at a cut's own trial state.
    pub frozen_resolves: usize,
    /// Of [`Self::frozen_resolves`], the re-solves that came back infeasible
    /// (the cut is then trivially valid at that point).
    pub frozen_infeasible: usize,
    /// Cut-validity violations.
    pub validity: Vec<String>,
    /// Under-tightness violations at a final-iteration trial point.
    pub tightness: Vec<String>,
    /// Mask-soundness violations.
    pub mask: Vec<String>,
}

impl Default for CutOracleReport {
    fn default() -> Self {
        Self {
            pools_checked: 0,
            cuts_checked: 0,
            inactive_cuts_checked: 0,
            pools_with_inactive_successor_cuts: 0,
            points_checked: 0,
            under_tightness_points_checked: 0,
            infeasible_sample_points: 0,
            off_mask_slots_checked: 0,
            min_feasible_points: usize::MAX,
            frozen_resolves: 0,
            frozen_infeasible: 0,
            validity: Vec::new(),
            tightness: Vec::new(),
            mask: Vec::new(),
        }
    }
}

impl CutOracleReport {
    /// Prints the report's counters, then panics naming every violation, or on
    /// vacuity (no pool checked, no cut checked, no trial point compared for
    /// under-tightness, or a checked pool with fewer than
    /// [`ORACLE_MIN_FEASIBLE_POINTS`] feasible sample points).
    pub fn assert_sound(&self, label: &str) {
        eprintln!(
            "{label}: pools_checked={} cuts_checked={} inactive_cuts_checked={} \
             pools_with_inactive_successor_cuts={} points_checked={} \
             under_tightness_points_checked={} infeasible_sample_points={} \
             off_mask_slots_checked={} min_feasible_points={} \
             frozen_resolves={} frozen_infeasible={}",
            self.pools_checked,
            self.cuts_checked,
            self.inactive_cuts_checked,
            self.pools_with_inactive_successor_cuts,
            self.points_checked,
            self.under_tightness_points_checked,
            self.infeasible_sample_points,
            self.off_mask_slots_checked,
            self.min_feasible_points,
            self.frozen_resolves,
            self.frozen_infeasible,
        );
        assert!(
            self.pools_checked > 0,
            "{label}: vacuity guard: no pool was checked"
        );
        assert!(
            self.cuts_checked > 0,
            "{label}: vacuity guard: no cut was checked"
        );
        assert!(
            self.under_tightness_points_checked > 0,
            "{label}: vacuity guard: no trial point was compared for under-tightness"
        );
        assert!(
            self.min_feasible_points >= ORACLE_MIN_FEASIBLE_POINTS,
            "{label}: vacuity guard: a checked pool had only {} feasible sample points (need >= {ORACLE_MIN_FEASIBLE_POINTS})",
            self.min_feasible_points
        );

        let mut violations: Vec<String> = Vec::new();
        violations.extend(self.validity.iter().map(|v| format!("[validity] {v}")));
        violations.extend(self.tightness.iter().map(|v| format!("[tightness] {v}")));
        violations.extend(self.mask.iter().map(|v| format!("[mask] {v}")));
        assert!(
            violations.is_empty(),
            "{label}: cut oracle violations:\n{}",
            violations.join("\n")
        );
    }
}

/// The open lag period's partial sum per hydro that every forward visit of every iteration of a
/// run read before its solve, from a replay over the cut-free setup: the sum is a function of the
/// noise and the PAR lags alone, so the replay of iteration `i` reproduces the run's.
pub struct PartialSums {
    n_stages: usize,
    forward_passes: usize,
    n_hydros: usize,
    visits: Vec<Vec<f64>>,
}

impl PartialSums {
    /// The sum the visit of `trial` at `stage` read in `iteration` (1-based); zeros where no pool
    /// of the study closes a lag period.
    ///
    /// # Panics
    ///
    /// Panics if `iteration` was not replayed.
    pub fn at(&self, iteration: u64, trial: usize, stage: usize) -> Vec<f64> {
        let index = usize::try_from(iteration)
            .ok()
            .and_then(|iteration| iteration.checked_sub(1))
            .map(|iteration| (iteration * self.forward_passes + trial) * self.n_stages + stage);
        let visit = index
            .and_then(|index| self.visits.get(index))
            .unwrap_or_else(|| panic!("iteration {iteration} was not replayed"));
        if visit.is_empty() {
            vec![0.0; self.n_hydros]
        } else {
            visit.clone()
        }
    }

    /// The sum the cut of `pool` at `slot` was generated under: the one its generating visit read
    /// at `stage`, the stage whose LP the cut is a function of.
    pub fn of_cut(&self, pool: &CutPool, slot: usize, stage: usize) -> Vec<f64> {
        let origin = pool.metadata(slot);
        self.at(
            origin.iteration_generated,
            origin.forward_pass_index as usize,
            stage,
        )
    }
}

/// Replays every iteration of `setup`'s run, one forward pass each over its cut-free templates,
/// and keeps the partial sum each visit's pin read.
///
/// # Panics
///
/// Panics if `setup` holds an active cut, or if a forward pass fails.
pub fn replay_partial_sums(setup: &StudySetup) -> PartialSums {
    assert_eq!(
        setup.fcf.total_active_cuts(),
        0,
        "the replay runs against cut-free templates"
    );
    let stage_ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let frozen: Vec<StageTemplate> = training_ctx
        .node_graph
        .pool_stage
        .iter()
        .map(|stage| stage_ctx.templates[stage.0].clone())
        .collect();
    let n_stages = setup.num_stages();
    let forward_passes = setup.fcf.forward_passes as usize;
    let mut pool = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("replay_partial_sums: workspace pool");
    let mut basis_store = BasisStore::new(forward_passes, training_ctx.node_graph.nodes.len());
    let mut visits = Vec::new();
    for iteration in 1..=setup.loop_params.max_iterations {
        let mut records = vec![TrajectoryRecord::default(); forward_passes * n_stages];
        let batch = ForwardPassBatch {
            local_forward_passes: forward_passes,
            total_forward_passes: forward_passes,
            iteration,
            fwd_offset: 0,
            event_sender: None,
        };
        let _ = run_forward_pass(
            &mut pool.workspaces,
            &mut basis_store,
            &stage_ctx,
            &frozen,
            &setup.fcf,
            &training_ctx,
            &batch,
            &mut records,
        )
        .expect("replay_partial_sums: forward pass");
        visits.extend(records.iter().map(|record| record.partial_sum().to_vec()));
    }
    PartialSums {
        n_stages,
        forward_passes,
        n_hydros: setup.stage_state().hydro_count,
        visits,
    }
}

fn all_zero(values: &[f64]) -> bool {
    values.iter().all(|&value| value == 0.0)
}

/// Per-run borrows [`run_cut_oracles`] threads into [`check_pool`]/[`evaluate_point`].
struct OracleCtx<'a> {
    setup: &'a StudySetup,
    tree_view: OpeningTreeView<'a>,
    partial_sums: &'a PartialSums,
    /// Per pool, how many cuts the final iteration's selection pass reactivated:
    /// such a cut is an active row that the final backward solve did not have, so
    /// under-tightness against the active rows is no invariant there.
    reactivated_by_final_pass: &'a [u32],
}

/// What [`evaluate_point`] gathers from one sample point's probe solves,
/// aggregated across every opening of the successor stage.
struct PointEval {
    q_hat: f64,
    per_opening_coeffs: Vec<Vec<f64>>,
}

/// Mirrors `production::fpha_fitting::rng::SplitMix64`'s algorithm and constants
/// byte-for-byte, hand-rolled here rather than reused: that generator is
/// `pub(crate)` inside a private submodule with no re-export reaching this
/// integration-test crate, and `box_random`'s sampled points must not move with
/// a `rand` upgrade the way a `rand::rngs::StdRng`-seeded draw could.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

fn box_corner(lo: &[f64], hi: &[f64], fallback: &[f64], pick: fn(f64, f64) -> f64) -> Vec<f64> {
    (0..lo.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                pick(lo[j], hi[j])
            } else {
                fallback[j]
            }
        })
        .collect()
}

fn box_random(lo: &[f64], hi: &[f64], fallback: &[f64], rng: &mut SplitMix64) -> Vec<f64> {
    (0..lo.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                lo[j] + rng.next_unit() * (hi[j] - lo[j])
            } else {
                fallback[j]
            }
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

fn check_mask(
    label: &str,
    t: usize,
    idx: usize,
    off_mask: &[usize],
    coeffs: &[f64],
    violations: &mut Vec<String>,
) {
    let scale = max_abs(coeffs);
    for &j in off_mask {
        if coeffs[j].abs() > MASK_ABS_TOL + MASK_REL_TOL * scale {
            violations.push(format!(
                "{label}: pool {t} point {idx}: off-mask slot {j} has |coefficient|={} (scale {scale})",
                coeffs[j]
            ));
        }
    }
}

/// Solves one opening of the successor stage, whose LP is the compact template plus `rows`,
/// cold: the LP is reloaded first, so the solve depends neither on the order of the points nor
/// on an earlier call's slack override.
///
/// # Errors
///
/// Returns the folded probe's [`SddpError`].
pub fn solve_successor_opening(
    setup: &StudySetup,
    ws: &mut SolverWorkspace<ActiveSolver>,
    rows: &CutRowSet,
    x: &[f64],
    raw_noise: &[f64],
    inflow_slack_bounds: Option<(&[f64], &[f64])>,
    out: &mut ProbeCut,
) -> Result<(), SddpError> {
    let stage_ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let s = rows.resident_pool;
    let pool = &setup.fcf.pools[s];
    ws.solver.reset_solver_state();
    load_compact_cut_rows(
        &mut ws.solver,
        setup,
        stage_ctx.template(StageIdx(s)),
        pool,
        &training_ctx.cut_state_layouts[s],
        rows,
    );
    opening_cut_for_probe(
        ws,
        &stage_ctx,
        &training_ctx,
        &ProbeOpening {
            stage: StageIdx(s),
            pinned_state: x,
            raw_noise,
            cuts: ProbeCuts {
                pool,
                pool_id: s,
                slots: &rows.slots,
            },
            extracted_pool_id: rows.extracted_pool,
            inflow_slack_bounds,
        },
        out,
    )
}

/// Solves one opening of the successor stage as [`solve_successor_opening`] does, with the cut rows
/// pinned at `partial_sum` instead of a fresh workspace's zero, and returns its objective. Only
/// the objective is read, so no cut is extracted.
///
/// # Errors
///
/// Returns [`SddpError::Infeasible`] when the LP is infeasible, or [`SddpError::Solver`] for any
/// other solver failure.
pub fn successor_objective_at(
    setup: &StudySetup,
    ws: &mut SolverWorkspace<ActiveSolver>,
    rows: &CutRowSet,
    x: &[f64],
    raw_noise: &[f64],
    inflow_slack_bounds: Option<(&[f64], &[f64])>,
    partial_sum: &[f64],
) -> Result<f64, SddpError> {
    let stage_ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let s = rows.resident_pool;
    let pool = &setup.fcf.pools[s];
    let template = stage_ctx.template(StageIdx(s));
    ws.solver.reset_solver_state();
    load_compact_cut_rows(
        &mut ws.solver,
        setup,
        template,
        pool,
        &training_ctx.cut_state_layouts[s],
        rows,
    );
    prep_for_probe_with_partial_sum(
        ws,
        &stage_ctx,
        &training_ctx,
        StageIdx(s),
        x,
        partial_sum,
        raw_noise,
        &ProbeCuts {
            pool,
            pool_id: s,
            slots: &rows.slots,
        },
    );
    if let Some((lower, upper)) = inflow_slack_bounds
        && !lower.is_empty()
    {
        let geometry = &stage_ctx.geometry_per_stage[s];
        let cols: Vec<usize> = (0..lower.len())
            .map(|h| geometry.inflow_slack_col(HydroSys::new(h)))
            .collect();
        let scaled = |bounds: &[f64]| -> Vec<f64> {
            cols.iter()
                .zip(bounds)
                .map(|(&col, &bound)| bound / scale_at(&template.col_scale, col))
                .collect()
        };
        ws.solver
            .set_col_bounds(&cols, &scaled(lower), &scaled(upper));
    }
    let view = ws.solver.solve(None).map_err(|error| match error {
        SolverError::Infeasible => SddpError::Infeasible {
            stage: s,
            iteration: 0,
            scenario: 0,
        },
        other => SddpError::Solver(other),
    })?;
    Ok(view.objective)
}

/// The RHS the production prep writes for the row of cut `slot` of pool `stage` when `stage`'s LP
/// is pinned at `incoming_state`, with the lag accumulator at `partial_sum` and the noise
/// `raw_noise`: the row of one forward visit, as that visit's solve saw it.
///
/// # Panics
///
/// Panics if the probe solve fails or the prep does not push the cut row's RHS.
pub fn pushed_cut_rhs_at_visit(
    setup: &StudySetup,
    stage: usize,
    slot: usize,
    incoming_state: &[f64],
    partial_sum: &[f64],
    raw_noise: &[f64],
) -> f64 {
    let stage_ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let pool = &setup.fcf.pools[stage];
    let rows = CutRowSet {
        resident_pool: stage,
        extracted_pool: stage,
        slots: vec![slot],
        frozen: 0,
    };
    let mut workspaces = setup
        .create_workspace_pool(&StubComm, 1, || {
            ActiveSolver::new().map(BoundRecordingSolver::new)
        })
        .expect("pushed_cut_rhs_at_visit: workspace pool");
    let ws = &mut workspaces.workspaces[0];
    load_compact_cut_rows(
        &mut ws.solver,
        setup,
        stage_ctx.template(StageIdx(stage)),
        pool,
        &training_ctx.cut_state_layouts[stage],
        &rows,
    );
    prep_for_probe_with_partial_sum(
        ws,
        &stage_ctx,
        &training_ctx,
        StageIdx(stage),
        incoming_state,
        partial_sum,
        raw_noise,
        &ProbeCuts {
            pool,
            pool_id: stage,
            slots: &rows.slots,
        },
    );
    ws.solver
        .solve(None)
        .expect("pushed_cut_rhs_at_visit: the visit's LP must solve");
    let solve = ws
        .solver
        .inner()
        .recorded()
        .last()
        .expect("pushed_cut_rhs_at_visit: the solve must be recorded");
    let row = solve.base_rows;
    solve
        .writes
        .iter()
        .rev()
        .find_map(|write| match write {
            BoundWrite::RowLowerRange { first_row, lower } => row
                .checked_sub(*first_row)
                .and_then(|offset| lower.get(offset))
                .copied(),
            _ => None,
        })
        .expect("pushed_cut_rhs_at_visit: the prep must push the cut row's RHS")
}

/// Solve every opening of stage `s = t + 1` at sample point `x` (stage `t`'s
/// outgoing state), aggregating the equal-weighted successor value `Q̂(x̃)` and
/// collecting each opening's raw (unaggregated) cut coefficients for the mask
/// child-side check.
fn evaluate_point(
    label: &str,
    occ: &OracleCtx<'_>,
    ws: &mut SolverWorkspace<ActiveSolver>,
    rows: &CutRowSet,
    t: usize,
    s: usize,
    x: &[f64],
) -> Result<PointEval, SddpError> {
    let n_openings = occ.tree_view.n_openings(s);
    assert!(
        n_openings >= 1,
        "{label}: evaluate_point: n_openings must be >= 1"
    );

    let mut q_hat = 0.0_f64;
    let weight = 1.0 / n_openings as f64;
    let mut per_opening_coeffs = Vec::with_capacity(n_openings);
    let mut cut = ProbeCut::new(&occ.setup.training_ctx());

    for o in 0..n_openings {
        solve_successor_opening(
            occ.setup,
            ws,
            rows,
            x,
            occ.tree_view.opening(s, o),
            None,
            &mut cut,
        )?;
        assert_eq!(
            cut.coefficients.len(),
            occ.setup.fcf.pools[t].state_dimension,
            "{label}: pool {t} coefficient length does not match cut_state_layouts[{t}]'s pairing"
        );
        q_hat += weight * cut.objective;
        per_opening_coeffs.push(cut.coefficients.clone());
    }

    Ok(PointEval {
        q_hat,
        per_opening_coeffs,
    })
}

/// Per-opening physical inflow-slack bounds `(lower, upper)` of the successor
/// stage at one state: empty bounds when the method does not truncate.
type SlackSignature = Vec<(Vec<f64>, Vec<f64>)>;

fn slack_signature(
    occ: &OracleCtx<'_>,
    ws: &mut SolverWorkspace<ActiveSolver>,
    s: usize,
    x: &[f64],
) -> SlackSignature {
    let training_ctx = occ.setup.training_ctx();
    (0..occ.tree_view.n_openings(s))
        .map(|o| {
            inflow_slack_bounds_for_probe(
                ws,
                &training_ctx,
                StageIdx(s),
                x,
                occ.tree_view.opening(s, o),
            )
        })
        .collect()
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn signature_eq(a: &SlackSignature, b: &SlackSignature) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((la, ua), (lb, ub))| bits_eq(la, lb) && bits_eq(ua, ub))
}

/// The successor value at `x` with the cut rows pinned at `partial_sum` and, when `signature` is
/// given, every opening's slack bounds frozen at it, or `+∞` when an opening is infeasible there.
// Rationale (too_many_arguments): `rows` selects the successor cut rows per
// call, so it cannot ride on the per-run `OracleCtx` with the other borrows.
#[allow(clippy::too_many_arguments)]
fn q_hat_at_partial_sum(
    label: &str,
    occ: &OracleCtx<'_>,
    ws: &mut SolverWorkspace<ActiveSolver>,
    rows: &CutRowSet,
    t: usize,
    s: usize,
    x: &[f64],
    signature: Option<&SlackSignature>,
    partial_sum: &[f64],
) -> f64 {
    let n_openings = occ.tree_view.n_openings(s);
    let weight = 1.0 / n_openings as f64;
    let mut q_hat = 0.0_f64;
    for o in 0..n_openings {
        let frozen = signature.map(|signature| (&signature[o].0[..], &signature[o].1[..]));
        match successor_objective_at(
            occ.setup,
            ws,
            rows,
            x,
            occ.tree_view.opening(s, o),
            frozen,
            partial_sum,
        ) {
            Ok(objective) => q_hat += weight * objective,
            Err(SddpError::Infeasible { .. }) => return f64::INFINITY,
            Err(other) => {
                panic!("{label}: pool {t} frozen-correction probe failed unexpectedly: {other}")
            }
        }
    }
    q_hat
}

/// What a checked cut's successor value is frozen at: the inflow-slack bounds of its own trial
/// state and the partial sum its generating visit read at the successor stage.
struct CutFreeze {
    signature: SlackSignature,
    partial_sum: Vec<f64>,
}

impl CutFreeze {
    fn same_as(&self, other: &Self) -> bool {
        signature_eq(&self.signature, &other.signature)
            && bits_eq(&self.partial_sum, &other.partial_sum)
    }
}

/// The freezes a pool's checked cuts are valued at: `by_slot` pairs each checked slot with its
/// index into `distinct`.
struct CutSignatures {
    distinct: Vec<CutFreeze>,
    by_slot: Vec<(usize, usize)>,
}

impl CutSignatures {
    fn push(&mut self, slot: usize, freeze: CutFreeze) {
        let k = if let Some(k) = self.distinct.iter().position(|d| d.same_as(&freeze)) {
            k
        } else {
            self.distinct.push(freeze);
            self.distinct.len() - 1
        };
        self.by_slot.push((slot, k));
    }
}

fn cut_signatures_at_trial_states(
    occ: &OracleCtx<'_>,
    ws: &mut SolverWorkspace<ActiveSolver>,
    t: usize,
    s: usize,
    cut_trial_states: &[(usize, Vec<f64>)],
) -> CutSignatures {
    let pool = &occ.setup.fcf.pools[t];
    let mut signatures = CutSignatures {
        distinct: Vec::new(),
        by_slot: Vec::new(),
    };
    for (slot, x_hat) in cut_trial_states {
        signatures.push(
            *slot,
            CutFreeze {
                signature: slack_signature(occ, ws, s, x_hat),
                partial_sum: occ.partial_sums.of_cut(pool, *slot, s),
            },
        );
    }
    signatures
}

/// Every populated slot of pool `t` under the one, empty signature of a successor
/// stage `s` with no truncation slack bounds, where the frozen successor value
/// does not depend on a cut's trial state.
///
/// # Panics
///
/// Panics if stage `s` has truncation slack bounds at `probe_state`, or if the
/// pool has a never-written slot.
fn state_free_cut_signatures(
    label: &str,
    occ: &OracleCtx<'_>,
    ws: &mut SolverWorkspace<ActiveSolver>,
    t: usize,
    s: usize,
    probe_state: &[f64],
) -> CutSignatures {
    let pool = &occ.setup.fcf.pools[t];
    let signature = slack_signature(occ, ws, s, probe_state);
    assert!(
        signature
            .iter()
            .all(|(lower, upper)| lower.is_empty() && upper.is_empty()),
        "{label}: pool {t}: stage {s} truncates inflow and the run applies cut selection: the \
         trial states of cuts older than the selection window are not available from the run, \
         so the inflow-slack bounds those cuts are frozen at cannot be computed"
    );
    assert_eq!(
        pool.populated(),
        pool.generated(),
        "{label}: pool {t} has a never-written slot"
    );
    let mut signatures = CutSignatures {
        distinct: Vec::new(),
        by_slot: Vec::new(),
    };
    for slot in 0..pool.populated() {
        signatures.push(
            slot,
            CutFreeze {
                signature: signature.clone(),
                partial_sum: occ.partial_sums.of_cut(pool, slot, s),
            },
        );
    }
    signatures
}

/// Every populated cut of pool `s`, active or not, as the cut rows of stage `s`'s LP, whose
/// cuts are written in pool `t`'s projection.
///
/// Validity is compared against these, never against the active rows alone: the pool is
/// append-only, so every successor row a cut's backward solve had is still populated, but
/// selection may have deactivated one since, and the cut may then lie above the successor value
/// built on the active rows only.
///
/// # Panics
///
/// Panics if pool `s` has a never-written slot.
fn populated_rows(label: &str, fcf: &FutureCostFunction, s: usize, t: usize) -> CutRowSet {
    assert_eq!(
        fcf.pools[s].populated(),
        fcf.pools[s].generated(),
        "{label}: pool {s} has a never-written slot"
    );
    CutRowSet::populated(fcf, s, t)
}

// Rationale (too_many_arguments): bundles the per-run oracle context, this
// pool's own trial/box points and per-cut slack signatures, the solver workspace,
// and the accumulating report; a second wrapper struct would just relocate the
// same values.
// Rationale (too_many_lines): one sequential per-point oracle (evaluate, mask, validity,
// tightness) whose steps share the point's evaluation and the report; helpers would each take the
// full context.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn check_pool(
    label: &str,
    t: usize,
    s: usize,
    occ: &OracleCtx<'_>,
    trial_points: &[Vec<f64>],
    trial_partial_sums: &[Vec<f64>],
    box_points: &[Vec<f64>],
    cut_signatures: &CutSignatures,
    ws: &mut SolverWorkspace<ActiveSolver>,
    report: &mut CutOracleReport,
) {
    let pool_t = &occ.setup.fcf.pools[t];
    let pool_s = &occ.setup.fcf.pools[s];
    report.pools_checked += 1;
    report.cuts_checked += cut_signatures.by_slot.len();
    report.inactive_cuts_checked += cut_signatures
        .by_slot
        .iter()
        .filter(|(slot, _)| !pool_t.is_active(*slot))
        .count();
    let active_rows = CutRowSet::active(&occ.setup.fcf, s, t);
    let validity_rows = (pool_s.active_count() < pool_s.populated())
        .then(|| populated_rows(label, &occ.setup.fcf, s, t));
    report.pools_with_inactive_successor_cuts += usize::from(validity_rows.is_some());
    let reactivated = occ.reactivated_by_final_pass[s];
    if reactivated > 0 {
        eprintln!(
            "{label}: pool {t}: under-tightness not compared: the final selection pass \
             reactivated {reactivated} cut(s) of pool {s}"
        );
    }

    let training_ctx = occ.setup.training_ctx();
    let cut_state_t = &training_ctx.cut_state_layouts[t];
    let nonzero = &occ.setup.stage_state().nonzero_state_indices;
    let off_mask: Vec<usize> = (0..cut_state_t.n_slots())
        .filter(|&j| {
            nonzero
                .binary_search(&cut_state_t.global_state_index(CutSlot::new(j)))
                .is_err()
        })
        .collect();
    report.off_mask_slots_checked += off_mask.len();

    let n_trial = trial_points.len();
    let mut feasible_count = 0usize;

    for (idx, x) in trial_points.iter().chain(box_points).enumerate() {
        let is_trial = idx < n_trial;
        match evaluate_point(label, occ, ws, &active_rows, t, s, x) {
            Ok(eval) => {
                feasible_count += 1;
                report.points_checked += 1;

                for coeffs in &eval.per_opening_coeffs {
                    check_mask(label, t, idx, &off_mask, coeffs, &mut report.mask);
                }

                let proj: Vec<f64> = (0..cut_state_t.n_slots())
                    .map(|j| x[cut_state_t.global_state_index(CutSlot::new(j)).get()])
                    .collect();

                let validity_q_hat = match &validity_rows {
                    None => eval.q_hat,
                    Some(rows) => {
                        evaluate_point(label, occ, ws, rows, t, s, x)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{label}: pool {t} point {idx} populated-cut probe failed \
                                     unexpectedly: {e}"
                                )
                            })
                            .q_hat
                    }
                };
                let point_signature = slack_signature(occ, ws, s, x);
                let frozen_values: Vec<f64> = cut_signatures
                    .distinct
                    .iter()
                    .map(|freeze| {
                        if signature_eq(&freeze.signature, &point_signature)
                            && all_zero(&freeze.partial_sum)
                        {
                            validity_q_hat
                        } else {
                            report.frozen_resolves += 1;
                            let value = q_hat_at_partial_sum(
                                label,
                                occ,
                                ws,
                                validity_rows.as_ref().unwrap_or(&active_rows),
                                t,
                                s,
                                x,
                                Some(&freeze.signature),
                                &freeze.partial_sum,
                            );
                            if value == f64::INFINITY {
                                report.frozen_infeasible += 1;
                            }
                            value
                        }
                    })
                    .collect();

                for &(slot, k) in &cut_signatures.by_slot {
                    let q_frozen = frozen_values[k];
                    if q_frozen == f64::INFINITY {
                        continue;
                    }
                    let lhs = pool_t.intercept(slot) + dot(pool_t.coefficient_row(slot), &proj);
                    let bound = value_tolerance(q_frozen, lhs);
                    if lhs > q_frozen + bound {
                        report.validity.push(format!(
                            "{label}: pool {t} point {idx} slot {slot} (active={}): cut value \
                             {lhs} exceeds the frozen-correction Q_hat={q_frozen} + tol={bound} \
                             (unfrozen Q_hat={validity_q_hat})",
                            pool_t.is_active(slot)
                        ));
                    }
                }

                // Under `level1` the active cuts reach this point's value because a
                // final-iteration trial point is a stored state: its owner is active
                // after a selection run and is worth at least the cut computed there.
                if is_trial && reactivated == 0 {
                    report.under_tightness_points_checked += 1;
                    let trial_q_hat = if all_zero(&trial_partial_sums[idx]) {
                        eval.q_hat
                    } else {
                        let value = q_hat_at_partial_sum(
                            label,
                            occ,
                            ws,
                            &active_rows,
                            t,
                            s,
                            x,
                            None,
                            &trial_partial_sums[idx],
                        );
                        assert!(
                            value.is_finite(),
                            "{label}: pool {t} trial point {idx} is infeasible at its own partial sum"
                        );
                        value
                    };
                    let evaluated = pool_t.evaluate_at_state(&proj);
                    let bound = value_tolerance(evaluated, trial_q_hat);
                    if evaluated < trial_q_hat - bound {
                        report.tightness.push(format!(
                            "{label}: pool {t} trial point {idx}: evaluate_at_state={evaluated} \
                             Q_hat={trial_q_hat} tol={bound}"
                        ));
                    }
                }
            }
            Err(SddpError::Infeasible { .. }) => {
                if is_trial {
                    report
                        .validity
                        .push(format!("{label}: pool {t} trial point {idx} is infeasible"));
                } else {
                    report.infeasible_sample_points += 1;
                }
            }
            Err(other) => {
                panic!("{label}: pool {t} point {idx} probe failed unexpectedly: {other}");
            }
        }
    }

    for &(slot, _) in &cut_signatures.by_slot {
        let beta = pool_t.coefficient_row(slot);
        let scale = max_abs(beta);
        for &j in &off_mask {
            if beta[j].abs() > MASK_ABS_TOL + MASK_REL_TOL * scale {
                report.mask.push(format!(
                    "{label}: pool {t} stored cut off-mask slot {j}: |beta|={} exceeds tol \
                     (scale {scale})",
                    beta[j]
                ));
            }
        }
    }

    report.min_feasible_points = report.min_feasible_points.min(feasible_count);
}

/// The values of the global dimensions pool `t` drops, as a state over the global
/// state (zeros for a pool that projects every dimension).
///
/// A pool stores the projected dimensions of its states only, and the run keeps
/// the whole state of one trial: `forward_state`, the state the first iteration's
/// forward pass ended stage `t` in. That state stands for every trial of the pool
/// when the dropped dimensions do not differ between trials, which holds with one
/// forward pass and one opening in every stage up to `t`.
///
/// # Panics
///
/// Panics if a pool that drops dimensions has several trials or openings.
fn dropped_dimension_fill(
    label: &str,
    t: usize,
    layout: &CutStateProjection,
    n_state: usize,
    single_trial: bool,
    forward_state: &[f64],
) -> Vec<f64> {
    if layout.n_slots() == n_state {
        return vec![0.0; n_state];
    }
    assert!(
        single_trial,
        "{label}: pool {t} drops state dimensions, which the run keeps for one trial only"
    );
    forward_state.to_vec()
}

/// Pairs every active cut slot `j` of pool `t` with the state it was generated at:
/// the state the pool stores at that slot, its dropped dimensions from `fill`.
///
/// # Panics
///
/// Panics if an active cut has no stored state.
fn recover_cut_trial_states(
    label: &str,
    t: usize,
    pool: &CutPool,
    layout: &CutStateProjection,
    fill: &[f64],
) -> Vec<(usize, Vec<f64>)> {
    pool.active_cuts()
        .map(|(j, _, _)| {
            let row = pool
                .trial_state(j)
                .unwrap_or_else(|| panic!("{label}: pool {t}: slot {j} has no stored state"));
            (j, lift_to_global_state(layout, row, fill))
        })
        .collect()
}

/// The final iteration's `forward_passes` trial states of pool `t` and the slots of their cuts,
/// its dropped dimensions from `fill`. When the pool drops some, `fill` is the state the first
/// iteration's forward pass ended the stage in, and the pool's first stored state
/// must equal it in the dimensions the pool keeps, bit for bit.
///
/// # Panics
///
/// Panics if the pool stores fewer states than `forward_passes`, or its first
/// stored state is not the forward one.
fn final_trial_points(
    label: &str,
    t: usize,
    pool: &CutPool,
    layout: &CutStateProjection,
    forward_passes: usize,
    fill: &[f64],
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let stored: Vec<(usize, &[f64])> = (0..pool.populated())
        .filter_map(|slot| pool.trial_state(slot).map(|row| (slot, row)))
        .collect();
    assert!(
        stored.len() >= forward_passes,
        "{label}: pool {t} stored {} states, fewer than forward_passes ({forward_passes})",
        stored.len()
    );
    let last = &stored[stored.len() - forward_passes..];
    let points: Vec<Vec<f64>> = last
        .iter()
        .map(|&(_, row)| lift_to_global_state(layout, row, fill))
        .collect();
    if layout.n_slots() < fill.len() {
        assert!(
            bits_eq(&lift_to_global_state(layout, stored[0].1, fill), fill),
            "{label}: pool {t}: the first stored state is not the state the first iteration's forward pass ended the stage in"
        );
    }
    (points, last.iter().map(|&(slot, _)| slot).collect())
}

/// Trains the compact `setup` with `HiGHS` on a single rank/thread, then gates every stored
/// cut, active or not, on the validity, under-tightness, and mask-soundness
/// oracles over each producer stage `t` whose pool has at least one populated cut.
/// The successor value is the folded solve of stage `t + 1`.
/// Validity compares each cut with the successor value whose inflow-slack bounds
/// are frozen at that cut's own trial state, read from the pool's per-slot store.
///
/// # Panics
///
/// Panics if `setup`'s node graph is not a chain, if training errors, if a cut is
/// inactive after a run with no selection pass, if a selection pass ran on a
/// study whose inflow method truncates, or if a probe solve fails for a reason
/// other than infeasibility.
pub fn run_cut_oracles(label: &str, mut setup: StudySetup) -> CutOracleReport {
    let n_stages = setup.inputs.stage_data.stage_templates.templates.len();
    assert_eq!(
        setup.inputs.node_graph.nodes.len(),
        n_stages,
        "{label}: run_cut_oracles requires a chain graph (one node per stage)"
    );
    for pos in 0..n_stages {
        assert_eq!(
            setup.inputs.node_graph.nodes[NodePos(pos)].stage.0,
            pos,
            "{label}: node {pos} is not chain-ordered onto stage {pos}"
        );
    }

    setup.set_export_states(true);
    let first_forward_states = super::forward_replay::first_iteration_forward_states(&setup);
    let partial_sums = replay_partial_sums(&setup);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("run_cut_oracles: ActiveSolver::new must succeed");
    let (event_sender, events) = mpsc::channel();
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            1,
            ActiveSolver::new,
            Some(event_sender),
            None,
        )
        .expect("run_cut_oracles: training must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: training must not error: {:?}",
        outcome.error
    );
    let events: Vec<TrainingEvent> = events.try_iter().collect();
    let last_selection_pass = events.iter().rev().find_map(|event| match event {
        TrainingEvent::PolicyPoolSummary { iteration, pools }
            if pools.iter().any(|record| record.rows_deactivated.is_some()) =>
        {
            Some((*iteration, pools))
        }
        _ => None,
    });
    let mut reactivated_by_final_pass = vec![0_u32; n_stages];
    match &last_selection_pass {
        None => assert_eq!(
            setup.fcf.total_active_cuts(),
            setup.fcf.total_generated_cuts(),
            "{label}: a cut is inactive after a run with no selection pass"
        ),
        Some((iteration, pools)) if *iteration == outcome.result.iterations => {
            assert_eq!(
                pools.len(),
                n_stages,
                "{label}: a chain has one record per pool, in pool order"
            );
            for (pool, record) in pools.iter().enumerate() {
                reactivated_by_final_pass[pool] = record.rows_reactivated.unwrap_or(0);
            }
        }
        Some(_) => {}
    }

    let forward_passes = setup.fcf.forward_passes as usize;

    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("run_cut_oracles: create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];

    let occ = OracleCtx {
        setup: &setup,
        tree_view: setup.training_ctx().stochastic.tree_view(),
        partial_sums: &partial_sums,
        reactivated_by_final_pass: &reactivated_by_final_pass,
    };

    let mut report = CutOracleReport::default();

    for (t, forward_state) in first_forward_states
        .iter()
        .enumerate()
        .take(n_stages.saturating_sub(1))
    {
        if setup.fcf.pools[t].populated() == 0 {
            continue;
        }
        let s = t + 1;
        let (lo, hi) = stage_state_box_bounds(&setup, t);
        let n_state = lo.len();

        let pool_t = &setup.fcf.pools[t];
        let layout_t = &setup.training_ctx().cut_state_layouts[t];
        let fill = dropped_dimension_fill(
            label,
            t,
            layout_t,
            n_state,
            forward_passes == 1 && (0..=t).all(|stage| occ.tree_view.n_openings(stage) == 1),
            forward_state,
        );
        let (trial_points, trial_slots) =
            final_trial_points(label, t, pool_t, layout_t, forward_passes, &fill);
        let trial_partial_sums: Vec<Vec<f64>> = trial_slots
            .iter()
            .map(|&slot| partial_sums.of_cut(pool_t, slot, s))
            .collect();

        let mut box_points = vec![
            box_corner(&lo, &hi, &trial_points[0], f64::midpoint),
            box_corner(&lo, &hi, &trial_points[0], |l, _h| l),
            box_corner(&lo, &hi, &trial_points[0], |_l, h| h),
        ];
        let mut rng = SplitMix64::new(ORACLE_SEED ^ (t as u64));
        for _ in 0..ORACLE_RANDOM_POINTS {
            box_points.push(box_random(&lo, &hi, &trial_points[0], &mut rng));
        }

        let cut_signatures = if last_selection_pass.is_some() {
            state_free_cut_signatures(label, &occ, ws, t, s, &trial_points[0])
        } else {
            let cut_trial_states = recover_cut_trial_states(label, t, pool_t, layout_t, &fill);
            cut_signatures_at_trial_states(&occ, ws, t, s, &cut_trial_states)
        };

        check_pool(
            label,
            t,
            s,
            &occ,
            &trial_points,
            &trial_partial_sums,
            &box_points,
            &cut_signatures,
            ws,
            &mut report,
        );
    }

    report
}
