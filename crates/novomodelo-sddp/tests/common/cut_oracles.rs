//! Cut-validity, mask-soundness, and tightness oracles for stored Benders cuts, run over an in-code or committed study.
//!
//! Validity compares each cut with the successor value whose inflow-slack bounds
//! are frozen at that cut's own trial state; when the study's inflow method does
//! not truncate, that is the plain successor value.

#![allow(clippy::expect_used, clippy::panic, clippy::cast_precision_loss)]

use std::sync::mpsc;

use novomodelo_core::TrainingEvent;
use novomodelo_io::Config;
use novomodelo_io::config::{RowSelectionConfig, SelectionMethod, StoppingRuleConfig};
use novomodelo_sddp::context::TrainingContext;
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::indexer::{CutSlot, CutStateProjection};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::{
    inflow_slack_bounds_for_probe, stage_state_box_bounds,
    write_backward_opening_outcome_at_canonical_state_for_probe,
    write_backward_opening_outcome_with_inflow_slack_bounds_for_probe,
};
use novomodelo_sddp::workspace::StageContext;
use novomodelo_sddp::{SddpError, StudySetup, build_cut_row_batch_into};
use novomodelo_solver::{
    ActiveSolver, FreezeScratch, RowBatch, StageTemplate, freeze_rows_into_template,
};
use novomodelo_stochastic::OpeningTreeView;

use super::StubComm;

const ORACLE_ITERATIONS: u32 = 3;
const ORACLE_RANDOM_POINTS: usize = 5;
const ORACLE_SEED: u64 = 0x5EED_C0BE_5EED_C0BE;
const ORACLE_ABS_TOL: f64 = 1e-6;
const ORACLE_REL_TOL: f64 = 1e-7;
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
    /// (Point, cut-signature) pairs re-solved with the inflow-slack bounds frozen
    /// at the cut's own trial state, because they differ from the point's own.
    pub frozen_resolves: usize,
    /// Of [`Self::frozen_resolves`], the ones infeasible on some opening (the cut
    /// is trivially valid there).
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

/// Per-run borrows [`run_cut_oracles`] threads into [`check_pool`]/[`evaluate_point`].
struct OracleCtx<'a> {
    setup: &'a StudySetup,
    tree_view: OpeningTreeView<'a>,
    /// One per pool (`pool_id == stage` on a chain), each its base template
    /// plus that pool's own active cuts as rows — unlike the setup's own
    /// `StageContext::templates`, which carries no cuts and would leave a
    /// successor's theta unconstrained if loaded directly.
    frozen_templates: &'a [StageTemplate],
    /// [`populated_cut_templates`]'s result.
    populated_templates: Option<&'a [StageTemplate]>,
    /// Per pool, how many cuts the final iteration's selection pass reactivated:
    /// such a cut is a row of `frozen_templates` that the final backward solve did
    /// not have, so under-tightness against that template is no invariant there.
    reactivated_by_final_pass: &'a [u32],
}

/// The three observations [`evaluate_point`] gathers from one sample point's
/// probe solves, aggregated across every opening of the successor stage.
struct PointEval {
    canonical_x: Vec<f64>,
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

fn q_tol(a: f64, b: f64) -> f64 {
    ORACLE_ABS_TOL + ORACLE_REL_TOL * a.abs().max(b.abs())
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

/// One cut's inflow-slack bounds `(lower, upper)` per opening of the successor
/// stage, at the cut's trial state; empty bounds when the method does not truncate.
type SlackSignature = Vec<(Vec<f64>, Vec<f64>)>;

fn signatures_equal(a: &SlackSignature, b: &SlackSignature) -> bool {
    let same_bits = |u: &[f64], v: &[f64]| {
        u.len() == v.len() && u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits())
    };
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| same_bits(&x.0, &y.0) && same_bits(&x.1, &y.1))
}

fn slack_signature_at(
    occ: &OracleCtx<'_>,
    ws: &mut novomodelo_sddp::workspace::SolverWorkspace<ActiveSolver>,
    training_ctx: &TrainingContext<'_>,
    s: usize,
    state: &[f64],
) -> SlackSignature {
    (0..occ.tree_view.n_openings(s))
        .map(|o| {
            inflow_slack_bounds_for_probe(
                ws,
                training_ctx,
                StageIdx(s),
                state,
                occ.tree_view.opening(s, o),
            )
        })
        .collect()
}

/// Probe every opening of stage `s = t + 1` at sample point `x` (stage `t`'s
/// outgoing state), aggregating the equal-weighted successor value `Q̂(x̃)` and
/// collecting each opening's raw (unaggregated) cut coefficients for the mask
/// child-side check. With `frozen`, each opening's inflow-slack bounds are
/// overwritten by the signature's instead of recomputed at `x`.
// Rationale (too_many_arguments): `templates` selects the successor cut rows per
// call, so it cannot ride on the per-run `OracleCtx` with the other borrows.
#[allow(clippy::too_many_arguments)]
fn evaluate_point(
    label: &str,
    occ: &OracleCtx<'_>,
    templates: &[StageTemplate],
    ws: &mut novomodelo_sddp::workspace::SolverWorkspace<ActiveSolver>,
    t: usize,
    s: usize,
    x: &[f64],
    frozen: Option<&SlackSignature>,
) -> Result<PointEval, SddpError> {
    let stage_ctx = StageContext {
        templates,
        ..occ.setup.stage_ctx()
    };
    let training_ctx = occ.setup.training_ctx();
    let successor_pool = &occ.setup.fcf.pools[s];
    let cut_state = &training_ctx.cut_state_layouts[t];
    let node_id = occ.setup.inputs.node_graph.node_ids[NodePos(s)];
    let n_openings = occ.tree_view.n_openings(s);

    let mut canonical_x: Option<Vec<f64>> = None;
    let mut q_hat = 0.0_f64;
    let weight = 1.0 / n_openings as f64;
    let mut per_opening_coeffs = Vec::with_capacity(n_openings);

    for o in 0..n_openings {
        let raw_noise = occ.tree_view.opening(s, o);
        let probe = match frozen {
            None => write_backward_opening_outcome_at_canonical_state_for_probe(
                ws,
                &stage_ctx,
                &training_ctx,
                successor_pool,
                cut_state,
                StageIdx(s),
                node_id,
                x,
                raw_noise,
            )?,
            Some(signature) => {
                write_backward_opening_outcome_with_inflow_slack_bounds_for_probe(
                    ws,
                    &stage_ctx,
                    &training_ctx,
                    successor_pool,
                    cut_state,
                    StageIdx(s),
                    node_id,
                    x,
                    raw_noise,
                    &signature[o].0,
                    &signature[o].1,
                )?
                .0
            }
        };
        assert_eq!(
            probe.outcome.coefficients.len(),
            occ.setup.fcf.pools[t].state_dimension,
            "{label}: pool {t} coefficient length does not match cut_state_layouts[{t}]'s pairing"
        );
        if canonical_x.is_none() {
            canonical_x = Some(probe.canonical_x_hat.clone());
        }
        q_hat += weight * probe.outcome.objective_value;
        per_opening_coeffs.push(probe.outcome.coefficients);
    }

    Ok(PointEval {
        canonical_x: canonical_x.expect("evaluate_point: n_openings must be >= 1"),
        q_hat,
        per_opening_coeffs,
    })
}

/// The slack signatures a pool's checked cuts are frozen at: `by_slot` pairs each
/// checked slot with its index into `distinct`.
struct CutSignatures {
    distinct: Vec<SlackSignature>,
    by_slot: Vec<(usize, usize)>,
}

fn cut_signatures_at_trial_states(
    occ: &OracleCtx<'_>,
    ws: &mut novomodelo_sddp::workspace::SolverWorkspace<ActiveSolver>,
    s: usize,
    cut_trial_states: &[(usize, Vec<f64>)],
) -> CutSignatures {
    let training_ctx = occ.setup.training_ctx();
    let mut distinct: Vec<SlackSignature> = Vec::new();
    let by_slot = cut_trial_states
        .iter()
        .map(|(slot, x_hat)| {
            let signature = slack_signature_at(occ, ws, &training_ctx, s, x_hat);
            let index = distinct
                .iter()
                .position(|known| signatures_equal(known, &signature))
                .unwrap_or_else(|| {
                    distinct.push(signature);
                    distinct.len() - 1
                });
            (*slot, index)
        })
        .collect();
    CutSignatures { distinct, by_slot }
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
    ws: &mut novomodelo_sddp::workspace::SolverWorkspace<ActiveSolver>,
    t: usize,
    s: usize,
    probe_state: &[f64],
) -> CutSignatures {
    let pool = &occ.setup.fcf.pools[t];
    let signature = slack_signature_at(occ, ws, &occ.setup.training_ctx(), s, probe_state);
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
    CutSignatures {
        distinct: vec![signature],
        by_slot: (0..pool.populated()).map(|slot| (slot, 0)).collect(),
    }
}

/// A copy of `frozen` in which each pool holding an inactive cut carries one row
/// per populated cut, active or not; `None` when no pool holds an inactive cut.
///
/// Validity is compared against these, never against `frozen` alone: the pool is
/// append-only, so every successor row a cut's backward solve had is still
/// populated, but selection may have deactivated one since, and the cut may then
/// lie above the successor value built on the active rows only.
///
/// # Panics
///
/// Panics if a pool holding an inactive cut has a never-written slot.
fn populated_cut_templates(
    label: &str,
    setup: &StudySetup,
    frozen: &[StageTemplate],
) -> Option<Vec<StageTemplate>> {
    let holds_inactive_cut = |pool: &CutPool| pool.active_count() < pool.populated();
    if !setup.fcf.pools.iter().any(holds_inactive_cut) {
        return None;
    }
    let training_ctx = setup.training_ctx();
    let base_templates = &setup.inputs.stage_data.stage_templates.templates;
    let mut fcf = setup.fcf.clone();
    let mut templates = frozen.to_vec();
    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    let mut scratch = FreezeScratch::new();
    for p in 0..fcf.pools.len() {
        if !holds_inactive_cut(&fcf.pools[p]) {
            continue;
        }
        let populated = fcf.pools[p].populated();
        assert_eq!(
            populated,
            fcf.pools[p].generated(),
            "{label}: pool {p} has a never-written slot"
        );
        for slot in 0..u32::try_from(populated).expect("slot count fits u32") {
            fcf.set_active(p, slot, true);
        }
        build_cut_row_batch_into(
            &mut batch,
            &fcf,
            p,
            training_ctx.state,
            &training_ctx.cut_state_layouts[p],
            &base_templates[p].col_scale,
        );
        freeze_rows_into_template(&base_templates[p], &batch, &mut templates[p], &mut scratch);
    }
    Some(templates)
}

// Rationale (too_many_arguments): bundles the per-run oracle context, this
// pool's own trial/box points and per-cut slack signatures, the solver workspace,
// and the accumulating report; a second wrapper struct would just relocate the
// same values.
#[allow(clippy::too_many_arguments)]
fn check_pool(
    label: &str,
    t: usize,
    s: usize,
    occ: &OracleCtx<'_>,
    trial_points: &[Vec<f64>],
    box_points: &[Vec<f64>],
    cut_signatures: &CutSignatures,
    ws: &mut novomodelo_sddp::workspace::SolverWorkspace<ActiveSolver>,
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
    let validity_templates = occ
        .populated_templates
        .filter(|_| pool_s.active_count() < pool_s.populated());
    report.pools_with_inactive_successor_cuts += usize::from(validity_templates.is_some());
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
        match evaluate_point(label, occ, occ.frozen_templates, ws, t, s, x, None) {
            Ok(eval) => {
                feasible_count += 1;
                report.points_checked += 1;

                for coeffs in &eval.per_opening_coeffs {
                    check_mask(label, t, idx, &off_mask, coeffs, &mut report.mask);
                }

                let mut proj = vec![0.0; cut_state_t.n_slots()];
                for (j, slot) in proj.iter_mut().enumerate() {
                    *slot = eval.canonical_x[cut_state_t.global_state_index(CutSlot::new(j)).get()];
                }

                let point_signature = slack_signature_at(occ, ws, &training_ctx, s, x);
                let validity_q_hat = match validity_templates {
                    None => eval.q_hat,
                    Some(templates) => {
                        evaluate_point(label, occ, templates, ws, t, s, x, None)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{label}: pool {t} point {idx} populated-cut probe failed \
                                     unexpectedly: {e}"
                                )
                            })
                            .q_hat
                    }
                };
                let frozen_q_hat: Vec<f64> = cut_signatures
                    .distinct
                    .iter()
                    .map(|signature| {
                        if signatures_equal(signature, &point_signature) {
                            return validity_q_hat;
                        }
                        report.frozen_resolves += 1;
                        match evaluate_point(
                            label,
                            occ,
                            validity_templates.unwrap_or(occ.frozen_templates),
                            ws,
                            t,
                            s,
                            x,
                            Some(signature),
                        ) {
                            Ok(frozen) => frozen.q_hat,
                            Err(SddpError::Infeasible { .. }) => {
                                report.frozen_infeasible += 1;
                                f64::INFINITY
                            }
                            Err(other) => panic!(
                                "{label}: pool {t} point {idx} frozen probe failed unexpectedly: {other}"
                            ),
                        }
                    })
                    .collect();

                for &(slot, sig) in &cut_signatures.by_slot {
                    let q_hat = frozen_q_hat[sig];
                    let lhs = pool_t.intercept(slot) + dot(pool_t.coefficient_row(slot), &proj);
                    let bound = q_tol(q_hat, lhs);
                    if lhs > q_hat + bound {
                        report.validity.push(format!(
                            "{label}: pool {t} point {idx}: slot {slot} (active={}) cut value \
                             {lhs} exceeds frozen Q_hat={q_hat} + tol={bound}",
                            pool_t.is_active(slot)
                        ));
                    }
                }

                // Under-tightness only: a cut frozen at another trial state may sit above
                // this point's own successor value (the paper's §2.5 gap); the frozen
                // comparator above is what bounds the cuts from above. Under `level1`
                // the active cuts reach this point's value because a final-iteration
                // trial point is a stored state: its owner is active after a selection
                // run and is worth at least the cut computed there.
                if is_trial && reactivated == 0 {
                    report.under_tightness_points_checked += 1;
                    let evaluated = pool_t.evaluate_at_state(&proj);
                    let bound = q_tol(evaluated, eval.q_hat);
                    if evaluated < eval.q_hat - bound {
                        report.tightness.push(format!(
                            "{label}: pool {t} trial point {idx}: evaluate_at_state={evaluated} \
                             below Q_hat={} tol={bound}",
                            eval.q_hat
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

/// A state stored in a pool, at its projected dimension, as a vector over the
/// global state: the dimensions the pool drops come from `fill`.
fn global_state(layout: &CutStateProjection, row: &[f64], fill: &[f64]) -> Vec<f64> {
    let mut state = fill.to_vec();
    for (j, &value) in row.iter().enumerate() {
        state[layout.global_state_index(CutSlot::new(j)).get()] = value;
    }
    state
}

/// The values of the global dimensions pool `t` drops, as a state over the global
/// state (zeros for a pool that projects every dimension).
///
/// A pool stores the projected dimensions of its states only, and the run keeps
/// the whole state of one trial: the one `captured`, the basis the final iteration
/// captured at the successor. That state stands for every trial of the pool when
/// the dropped dimensions do not differ between trials, which holds with one
/// forward pass and one opening in every stage up to `t`.
///
/// # Panics
///
/// Panics if a pool that drops dimensions has several trials or openings, or no
/// captured state.
fn dropped_dimension_fill(
    label: &str,
    t: usize,
    layout: &CutStateProjection,
    n_state: usize,
    single_trial: bool,
    captured: Option<&[f64]>,
) -> Vec<f64> {
    if layout.n_slots() == n_state {
        return vec![0.0; n_state];
    }
    assert!(
        single_trial,
        "{label}: pool {t} drops state dimensions, which the run keeps for one trial only"
    );
    captured
        .unwrap_or_else(|| panic!("{label}: pool {t}: the successor captured no basis"))
        .to_vec()
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
            (j, global_state(layout, row, fill))
        })
        .collect()
}

/// The final iteration's `forward_passes` trial states of pool `t`, its dropped
/// dimensions from `fill`. When the pool drops some, `fill` is the captured state
/// and the stored dimensions of the one trial must equal it bit for bit.
///
/// # Panics
///
/// Panics if the pool stores fewer states than `forward_passes`, or stores a state
/// other than the captured one.
fn final_trial_points(
    label: &str,
    t: usize,
    pool: &CutPool,
    layout: &CutStateProjection,
    forward_passes: usize,
    fill: &[f64],
) -> Vec<Vec<f64>> {
    let stored: Vec<&[f64]> = (0..pool.populated())
        .filter_map(|slot| pool.trial_state(slot))
        .collect();
    assert!(
        stored.len() >= forward_passes,
        "{label}: pool {t} stored {} states, fewer than forward_passes ({forward_passes})",
        stored.len()
    );
    let points: Vec<Vec<f64>> = stored[stored.len() - forward_passes..]
        .iter()
        .map(|row| global_state(layout, row, fill))
        .collect();
    if layout.n_slots() < fill.len() {
        assert!(
            points[0]
                .iter()
                .zip(fill)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label}: pool {t}: the stored state is not the state the final iteration captured"
        );
    }
    points
}

/// Trains `setup` with `HiGHS` on a single rank/thread, then gates every stored
/// cut, active or not, on the validity, under-tightness, and mask-soundness
/// oracles over each producer stage `t` whose pool has at least one populated cut.
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

    let frozen_templates = outcome
        .result
        .frozen_templates
        .expect("run_cut_oracles: training must populate frozen_templates");
    let basis_cache = outcome.result.basis_cache;
    let forward_passes = setup.fcf.forward_passes as usize;

    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("run_cut_oracles: create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];

    let populated_templates = populated_cut_templates(label, &setup, &frozen_templates);
    let occ = OracleCtx {
        setup: &setup,
        tree_view: setup.training_ctx().stochastic.tree_view(),
        frozen_templates: &frozen_templates,
        populated_templates: populated_templates.as_deref(),
        reactivated_by_final_pass: &reactivated_by_final_pass,
    };

    let mut report = CutOracleReport::default();

    for t in 0..n_stages.saturating_sub(1) {
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
            basis_cache[s]
                .as_ref()
                .map(|basis| basis.state_at_capture.as_slice()),
        );
        let trial_points = final_trial_points(label, t, pool_t, layout_t, forward_passes, &fill);

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
            cut_signatures_at_trial_states(&occ, ws, s, &cut_trial_states)
        };

        check_pool(
            label,
            t,
            s,
            &occ,
            &trial_points,
            &box_points,
            &cut_signatures,
            ws,
            &mut report,
        );
    }

    report
}
