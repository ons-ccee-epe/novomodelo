//! Scalar statement of the cut-row bracket B = fl(fl(α + S) + P), for tests and benches to compare
//! [`cut_brackets_into`](crate::cut::rhs::cut_brackets_into) against bit for bit, as
//! `selection_reference` is for cut selection. One row at a time, one accumulator per dot from +0.0,
//! slots ascending.

use crate::cut::CutPool;
use crate::cut::rhs::ascending_dot;
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
    debug_assert!(
        conditional_mean.len() >= lag0.len(),
        "cut_bracket: conditional_mean shorter than the lag-0 range"
    );
    debug_assert!(
        lag_state.len() >= shifted.len(),
        "cut_bracket: lag_state shorter than the shifted range"
    );
    debug_assert!(
        coefficients.len() >= shifted.end.max(lag0.end),
        "cut_bracket: coefficients shorter than the lag slots"
    );

    let shifted_dot = ascending_dot(&coefficients[shifted.clone()], &lag_state[..shifted.len()]);
    let mean_dot = ascending_dot(&coefficients[lag0.clone()], &conditional_mean[..lag0.len()]);
    (intercept + shifted_dot) + mean_dot
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
