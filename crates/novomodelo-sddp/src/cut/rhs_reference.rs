//! Scalar statement of the cut-row bracket B = fl(fl(α + S) + P) and of the noise term W, for tests
//! and benches to compare [`cut_brackets_into`](crate::cut::rhs::cut_brackets_into) and
//! [`cut_noise_terms_into`](crate::cut::rhs::cut_noise_terms_into) against bit for bit, as
//! `selection_reference` is for cut selection. One row at a time, one accumulator per dot from +0.0,
//! slots ascending.

use crate::cut::CutPool;
use crate::indexer::CutLagPoint;

/// B = fl(fl(α + S) + P) of one cut row. Reads `lag_state[..shifted.len()]` only.
#[must_use]
pub fn cut_bracket(
    intercept: f64,
    coefficients: &[f64],
    lag_point: &CutLagPoint,
    lag_state: &[f64],
    conditional_mean: &[f64],
) -> f64 {
    let (lag0, shifted) = (lag_point.lag0(), lag_point.shifted());
    let shifted_dot = ascending_dot(&coefficients[shifted.clone()], &lag_state[..shifted.len()]);
    let mean_dot = ascending_dot(&coefficients[lag0.clone()], &conditional_mean[..lag0.len()]);
    (intercept + shifted_dot) + mean_dot
}

/// W of one cut row; the RHS is `fl(bracket + W)`.
#[must_use]
pub fn cut_noise_term(coefficients: &[f64], lag_point: &CutLagPoint, noise_term: &[f64]) -> f64 {
    let (lag0, shifted) = (lag_point.lag0(), lag_point.shifted());
    debug_assert!(
        coefficients.len() >= shifted.end.max(lag0.end),
        "cut_noise_term: coefficients shorter than the lag slots"
    );

    ascending_dot(&coefficients[lag0.clone()], &noise_term[..lag0.len()])
}

/// Writes `brackets[i]` for the cut at `slots[i]` as the plain loop of [`cut_bracket`].
pub fn cut_brackets_reference_into(
    pool: &CutPool,
    slots: &[usize],
    lag_point: &CutLagPoint,
    lag_state: &[f64],
    conditional_mean: &[f64],
    brackets: &mut [f64],
) {
    debug_assert!(
        brackets.len() == slots.len(),
        "cut_brackets_reference_into: brackets and slots differ in length"
    );

    for (i, &slot) in slots.iter().enumerate() {
        brackets[i] = cut_bracket(
            pool.intercept(slot),
            pool.coefficient_row(slot),
            lag_point,
            lag_state,
            conditional_mean,
        );
    }
}

fn ascending_dot(coefficients: &[f64], values: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(values)
        .fold(0.0, |acc, (&b, &v)| acc + b * v)
}
