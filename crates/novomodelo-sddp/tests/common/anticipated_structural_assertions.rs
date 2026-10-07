//! Structural assertions shared by the anticipated-thermals integration tests.
//!
//! These helpers verify "training converged into a healthy state" without
//! claiming the converged value is correct. They survive legitimate code
//! changes (cut counts, basis-cache layout tweaks, FCF additions) and catch
//! regressions of the form "didn't crash, didn't fail to converge".
//!
//! Value-correctness coverage is the `anticipated_closed_form_lb_k1_single_thermal`
//! test in `crates/cobre-sddp/tests/anticipated_core.rs`, the hand-derivable canary. The two are complementary: the canary defends
//! the LP/cut math at a single trivial fixture; these structural assertions
//! defend the larger fixtures that have no closed form.

#![allow(clippy::expect_used, clippy::float_cmp, dead_code)]

use cobre_core::TrainingEvent;
use cobre_sddp::TrainingResult;

/// Assert the training run terminated cleanly with the expected iteration count,
/// a finite numeric lower bound, no NaNs in the iteration history, and a
/// non-decreasing lower-bound trajectory.
///
/// The non-decrease guarantee is the structural cousin of "SDDP cuts only
/// tighten the FCF approximation". Within a single training run with frozen
/// scenario data, the lower bound at iteration `k+1` must be at least the
/// lower bound at iteration `k`; otherwise either a cut was lost or the
/// stage-0 LP was re-solved with stale FCF.
///
/// # Arguments
///
/// - `events` — full event log; an empty slice skips the per-iteration history
///   check (only the `result`-level invariants run).
/// - `expected_iterations` — exact iteration count; use the configured
///   `IterationLimit`. Pins the counter without claiming the LP value is correct.
///
/// # Panics
///
/// On any failed invariant. Each assertion carries an explanatory message
/// naming the violated property.
pub fn assert_training_converged_structurally(
    result: &TrainingResult,
    events: &[TrainingEvent],
    expected_iterations: u64,
) {
    assert_eq!(
        result.iterations, expected_iterations,
        "structural: iteration count mismatch (expected {expected_iterations}, got {})",
        result.iterations,
    );

    let final_lb = result.final_lb;
    assert!(
        final_lb.is_finite(),
        "structural: final_lb must be finite; got {final_lb}",
    );

    let final_gap = result.final_gap;
    assert!(
        final_gap.is_finite(),
        "structural: final_gap must be finite; got {final_gap}",
    );

    let mut history: Vec<(u64, f64)> = events
        .iter()
        .filter_map(|event| match event {
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound,
                ..
            } => Some((*iteration, *lower_bound)),
            _ => None,
        })
        .collect();
    history.sort_by_key(|(iter, _)| *iter);

    if history.is_empty() {
        return;
    }

    for &(iter, lb) in &history {
        assert!(
            lb.is_finite(),
            "structural: lower bound at iteration {iter} must be finite; got {lb}",
        );
    }

    // Tolerance is scaled by the bound magnitude to cover double-precision
    // LP-solver noise.
    for window in history.windows(2) {
        let (prev_iter, prev_lb) = window[0];
        let (next_iter, next_lb) = window[1];
        let tolerance = 1e-9_f64 * prev_lb.abs().max(1.0);
        assert!(
            next_lb + tolerance >= prev_lb,
            "structural: lower bound decreased from iteration {prev_iter} \
             (lb={prev_lb}) to iteration {next_iter} (lb={next_lb}). SDDP \
             cuts must only tighten the FCF; a strict decrease indicates a \
             lost cut, an FCF reset, or a stale stage-0 solve.",
        );
    }
}

/// Assert the basis cache is fully populated: one entry per study stage,
/// every entry `Some(..)`, every captured state vector finite.
///
/// This catches the regression "some stage was never solved during the final
/// iteration", which is the symptom of a backward-pass abort or a
/// horizon-mode mismatch.
///
/// # Panics
///
/// If any stage has no captured basis, or if any state-at-capture entry is
/// non-finite.
pub fn assert_basis_cache_fully_populated(result: &TrainingResult, expected_stages: usize) {
    assert_eq!(
        result.basis_cache.len(),
        expected_stages,
        "structural: basis_cache must have one entry per study stage (expected {expected_stages}, got {})",
        result.basis_cache.len(),
    );
    for (stage_idx, entry) in result.basis_cache.iter().enumerate() {
        let captured = entry
            .as_ref()
            .unwrap_or_else(|| panic!("structural: basis_cache[{stage_idx}] must be Some"));
        for (state_idx, &value) in captured.state_at_capture.iter().enumerate() {
            assert!(
                value.is_finite(),
                "structural: basis_cache[{stage_idx}].state_at_capture[{state_idx}] must be finite; got {value}",
            );
        }
    }
}

/// Assert anticipated-state slots in the captured basis are finite and
/// non-negative at every delivery stage.
///
/// At a delivery stage `t` (where some past commitment matures), the
/// fishing constraint forces `gen_anticipated[i] = x_state[slot=0]`, and
/// the anticipated state must be ≥ 0 because it represents a committed MW.
///
/// # Arguments
///
/// - `anticipated_slots_out_start` — `indexer.commit_out.start`, the
///   state-vector offset of the anticipated ring's outgoing slots (the domain
///   `state_at_capture` is addressed in — never the relocated incoming
///   `commit_in` range).
/// - `n_anticipated` — `indexer.n_anticipated`.
/// - `delivery_stages` — pass the indices of stages where `K_i ≤ stage_idx` for
///   some plant; slot 0 (slot-major layout) must be populated and non-negative.
///
/// # Panics
///
/// If any delivery-stage slot 0 value is non-finite or strictly negative.
pub fn assert_anticipated_delivery_slots_populated(
    result: &TrainingResult,
    anticipated_slots_out_start: usize,
    n_anticipated: usize,
    delivery_stages: &[usize],
) {
    for &stage_idx in delivery_stages {
        let captured = result.basis_cache[stage_idx]
            .as_ref()
            .unwrap_or_else(|| panic!("structural: basis_cache[{stage_idx}] must be Some"));
        // Slot 0 holds the matured commitment about to be dispatched at
        // `stage_idx`. Layout is slot-major, plant-minor.
        for plant in 0..n_anticipated {
            let idx = anticipated_slots_out_start + plant;
            let value = captured.state_at_capture[idx];
            assert!(
                value.is_finite(),
                "structural: anticipated state at stage {stage_idx}, plant {plant}, slot 0 must be finite; got {value}",
            );
            // A committed-MW value below zero is a structural violation: the
            // ring-buffer either leaked a free LP slack or the decision
            // column bounds were broken upstream.
            assert!(
                value >= -1e-9,
                "structural: anticipated state at stage {stage_idx}, plant {plant}, slot 0 must be ≥ 0; got {value}",
            );
        }
    }
}
