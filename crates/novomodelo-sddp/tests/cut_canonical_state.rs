//! Pins that a Benders cut is built at the SAME canonical incoming state the LP
//! was pinned at: the real `assemble_outgoing_state` read-back seam
//! canonicalizes a producer state onto its admissible box, the backward opening
//! pins the LP and builds the intercept at that one canonical value, and the
//! cut is evaluated at the state the LP was ACTUALLY pinned at — recovered from
//! the solved primal (an independent data path), through
//! `CutStateProjection::dot_trial_state`'s own gather (`global_state_index`,
//! never a positional zip).
//!
//! The seam, pin, and intercept are all production functions driven through
//! `test_support::write_backward_opening_outcome_for_probe` (they are
//! `pub(crate)`, unreachable from this integration binary). The test supplies
//! only the RAW pre-clamp input and reads back three observations; it never
//! recomputes `write_opening_outcome`'s formula, and it evaluates the cut at a
//! value production produced, not one it threaded in.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_sddp::indexer::{CutSlot, CutStateProjection};
use cobre_sddp::setup::{NodeId, StageIdx};
use cobre_sddp::test_support::{
    CanonicalCutProbe, TrunkFanFixture, stage_state_box_bounds, trunk_fan_setup_enumerated,
    write_backward_opening_outcome_for_probe,
};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

/// The last trunk node's own stage — an interior node with a nonempty box
/// (`TRUNK_FAN_STORAGE` is scarce, so storage genuinely binds), reached with
/// `t_trunk = 2, k = 2` (`trunk_fan_setup_enumerated`'s smallest valid shape).
const STAGE: StageIdx = StageIdx(1);
const PRODUCER_STAGE: usize = 0;
/// `trunk_fan_policy_graph` gives every trunk node the same id as its stage.
const NODE_ID: NodeId = NodeId(1);

/// Storage state dimension of the single-hydro trunk+fan fixture.
const STORAGE_DIM: usize = 0;

/// Gather-based dot product over the enabled cut-state slots, reproducing
/// `CutStateProjection::dot_trial_state`'s own per-slot gather
/// (`global_state_index`) rather than a positional `zip` — the sddp.md
/// cut-intercept contract this file pins from outside the crate.
fn dot_via_projection(cut_state: &CutStateProjection, coefficients: &[f64], x_hat: &[f64]) -> f64 {
    (0..cut_state.n_slots())
        .map(|j| {
            let dim = cut_state.global_state_index(CutSlot::new(j)).get();
            coefficients[j] * x_hat[dim]
        })
        .sum()
}

#[test]
fn cut_evaluated_at_canonical_state_equals_lp_objective() {
    let TrunkFanFixture { setup, .. } = trunk_fan_setup_enumerated(2, 2, 1);
    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let cut_pool = &setup.fcf.pools[STAGE.0];
    let cut_state = &training_ctx.cut_state_layouts[STAGE.0];
    let opening_tree = training_ctx.stochastic.tree_view();
    let raw_noise = opening_tree.opening(STAGE.0, 0);

    let (lower, upper) = stage_state_box_bounds(&setup, PRODUCER_STAGE);
    // An in-box producer state: the seam's clamp is the identity, so the
    // canonical value equals the raw seed and the equality holds without drift.
    let raw_producer_state = vec![f64::midpoint(lower[STORAGE_DIM], upper[STORAGE_DIM])];

    let CanonicalCutProbe {
        outcome,
        canonical_x_hat,
        pinned_x_hat,
    } = write_backward_opening_outcome_for_probe(
        ws,
        &ctx,
        &training_ctx,
        cut_pool,
        cut_state,
        STAGE,
        NODE_ID,
        &raw_producer_state,
        raw_noise,
    )
    .expect("stage-1 backward solve must not error on the trunk+fan fixture");

    assert_eq!(
        canonical_x_hat, raw_producer_state,
        "an in-box seed must pass through the seam's clamp unchanged"
    );
    assert!(
        (pinned_x_hat[STORAGE_DIM] - canonical_x_hat[STORAGE_DIM]).abs() <= 1e-6,
        "the LP was pinned at {} (from the solved primal) but the seam produced {}",
        pinned_x_hat[STORAGE_DIM],
        canonical_x_hat[STORAGE_DIM]
    );
    assert!(
        outcome.coefficients[STORAGE_DIM].abs() > 1e-6,
        "fixture sanity: storage's subgradient must be non-degenerate, got {}",
        outcome.coefficients[STORAGE_DIM]
    );

    // cut(pinned) == objective: the intercept was built at the same state the LP
    // was solved at. `pinned_x_hat` is the solved-primal readback, NOT the value
    // threaded into write_opening_outcome, so the equality is not an identity.
    let cut_at_pin =
        outcome.intercept + dot_via_projection(cut_state, &outcome.coefficients, &pinned_x_hat);
    assert!(
        (cut_at_pin - outcome.objective_value).abs() <= 1e-6,
        "cut(pinned_x_hat) = {cut_at_pin} must equal the LP objective {}",
        outcome.objective_value
    );
}

#[test]
fn cut_from_drifted_seed_is_not_weakened_by_the_clamp() {
    let TrunkFanFixture { setup, .. } = trunk_fan_setup_enumerated(2, 2, 1);
    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut pool.workspaces[0];

    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let cut_pool = &setup.fcf.pools[STAGE.0];
    let cut_state = &training_ctx.cut_state_layouts[STAGE.0];
    let opening_tree = training_ctx.stochastic.tree_view();
    let raw_noise = opening_tree.opening(STAGE.0, 0);

    let (_, upper) = stage_state_box_bounds(&setup, PRODUCER_STAGE);
    let drift = 1.0_f64;
    // A producer state a hair outside the admissible box: the real seam MUST
    // clamp it before either the pin or the intercept sees it.
    let raw_producer_state = vec![upper[STORAGE_DIM] + drift];

    let CanonicalCutProbe {
        outcome,
        canonical_x_hat,
        pinned_x_hat,
    } = write_backward_opening_outcome_for_probe(
        ws,
        &ctx,
        &training_ctx,
        cut_pool,
        cut_state,
        STAGE,
        NODE_ID,
        &raw_producer_state,
        raw_noise,
    )
    .expect("stage-1 backward solve must not error on the trunk+fan fixture");

    // The seam clamped the out-of-box seed to the box edge — the
    // canonicalization this test exists to pin. If assemble_outgoing_state
    // skipped its clamp, canonical == raw and this fails (and the pin's own
    // producer-box guard would reject the raw value).
    assert!(
        (canonical_x_hat[STORAGE_DIM] - upper[STORAGE_DIM]).abs() <= 1e-9,
        "the seam must clamp the drifted seed to the box upper {}, got {}",
        upper[STORAGE_DIM],
        canonical_x_hat[STORAGE_DIM]
    );
    assert!(
        (canonical_x_hat[STORAGE_DIM] - raw_producer_state[STORAGE_DIM]).abs() > 1e-6,
        "the drifted seed must not survive the clamp unchanged"
    );
    // The LP was pinned at the clamped value (from the solved primal), not the
    // raw seed — the canonicalization reached the actual solve.
    assert!(
        (pinned_x_hat[STORAGE_DIM] - upper[STORAGE_DIM]).abs() <= 1e-6,
        "the LP was pinned at {} but the box upper is {}",
        pinned_x_hat[STORAGE_DIM],
        upper[STORAGE_DIM]
    );
    assert!(
        outcome.coefficients[STORAGE_DIM].abs() > 1e-6,
        "fixture sanity: storage's subgradient must be non-degenerate, got {}",
        outcome.coefficients[STORAGE_DIM]
    );

    // cut(pinned) == objective: the intercept was built at the clamped state the
    // LP solved, so the drifted seed did not bias it.
    let cut_at_pin =
        outcome.intercept + dot_via_projection(cut_state, &outcome.coefficients, &pinned_x_hat);
    assert!(
        (cut_at_pin - outcome.objective_value).abs() <= 1e-6,
        "cut(pinned_x_hat) = {cut_at_pin} must equal the LP objective {} even from a drifted seed",
        outcome.objective_value
    );

    // The cut is a genuine supporting hyperplane, not flat: evaluated at the RAW
    // pre-clamp seed it differs from the objective by ~|β|·drift. This is what
    // makes cut(pinned) == objective non-vacuous — the clamp genuinely moved the
    // evaluation point off the raw seed. A pin-time-only projection that left the
    // intercept on the raw seed would instead make cut(raw) == objective here.
    let cut_at_raw = outcome.intercept
        + dot_via_projection(cut_state, &outcome.coefficients, &raw_producer_state);
    let predicted_shift = outcome.coefficients[STORAGE_DIM] * drift;
    assert!(
        predicted_shift.abs() > 1e-3,
        "fixture sanity: |β|·drift = {predicted_shift} too small to discriminate"
    );
    assert!(
        (cut_at_raw - outcome.objective_value).abs() > 1e-3,
        "cut(raw_seed) = {cut_at_raw} must differ from the objective {} by ~|β|·drift = \
         {predicted_shift}; equality would mean the clamp never moved the point",
        outcome.objective_value
    );
}
