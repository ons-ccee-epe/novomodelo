//! Canonical cut-row RHS: B = fl(fl(α + S) + P) and RHS = fl(B + W), the one association every
//! cut-row RHS site uses. S, P and W are ascending-slot dots from +0.0: the shifted slots against
//! â, the lag-0 slots against p, and the lag-0 slots against σ·η. Not one dot over ẑ = fl(p + σ·η),
//! not the blocked GEMM kernel for S, not a dot started from an iterator sum's −0.0.

use crate::cut::CutPool;
use crate::indexer::CutLagPoint;

fn ascending_dot(coefficients: &[f64], values: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(values)
        .fold(0.0, |acc, (&b, &v)| acc + b * v)
}

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

/// W of one cut row; the RHS is `fl(bracket + W)`.
#[must_use]
pub fn cut_noise_term(coefficients: &[f64], lag_point: &CutLagPoint, noise_term: &[f64]) -> f64 {
    let (lag0, shifted) = (lag_point.lag0(), lag_point.shifted());
    debug_assert!(
        noise_term.len() >= lag0.len(),
        "cut_noise_term: noise_term shorter than the lag-0 range"
    );
    debug_assert!(
        coefficients.len() >= shifted.end.max(lag0.end),
        "cut_noise_term: coefficients shorter than the lag slots"
    );

    ascending_dot(&coefficients[lag0.clone()], &noise_term[..lag0.len()])
}

/// Writes `brackets[i]` for the cut at `slots[i]`, reading each row in place from `pool`.
pub fn cut_brackets_into(
    pool: &CutPool,
    slots: &[usize],
    lag_point: &CutLagPoint,
    lag_state: &[f64],
    conditional_mean: &[f64],
    brackets: &mut [f64],
) {
    debug_assert!(
        brackets.len() == slots.len(),
        "cut_brackets_into: brackets and slots differ in length"
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

/// Writes `rhs[i] = brackets[i] + W` for the cut at `slots[i]`, reading each row in place from
/// `pool`.
pub fn cut_rhs_into(
    pool: &CutPool,
    slots: &[usize],
    lag_point: &CutLagPoint,
    noise_term: &[f64],
    brackets: &[f64],
    rhs: &mut [f64],
) {
    debug_assert!(
        brackets.len() == slots.len(),
        "cut_rhs_into: brackets and slots differ in length"
    );
    debug_assert!(
        rhs.len() == slots.len(),
        "cut_rhs_into: rhs and slots differ in length"
    );

    for (i, &slot) in slots.iter().enumerate() {
        rhs[i] = brackets[i] + cut_noise_term(pool.coefficient_row(slot), lag_point, noise_term);
    }
}

#[cfg(test)]
mod tests {
    use novomodelo_core::temporal::StageStateConfig;

    use super::{cut_bracket, cut_brackets_into, cut_noise_term, cut_rhs_into};
    use crate::cut::CutPool;
    use crate::indexer::{CutStateProjection, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::setup::NodeId;

    const EPS: f64 = f64::EPSILON / 2.0;

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    const LAGS_ONLY: StageStateConfig = StageStateConfig {
        storage: false,
        inflow_lags: true,
    };

    fn projection(
        n_hydros: usize,
        lag_depth: usize,
        config: StageStateConfig,
    ) -> CutStateProjection {
        let global = StateSpace::new(
            n_hydros,
            lag_depth,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &vec![lag_depth; n_hydros],
        );
        CutStateProjection::new(&global, config)
    }

    #[test]
    fn bracket_adds_the_lag_shift_before_the_conditional_mean() {
        let lag_point = projection(1, 2, LAGS_ONLY);
        let lag_point = lag_point.lag_point();
        assert_eq!(lag_point.lag0(), 0..1);
        assert_eq!(lag_point.shifted(), 1..2);
        assert_ne!((1.0 + (EPS + EPS)).to_bits(), 1.0_f64.to_bits());

        let bracket = cut_bracket(1.0, &[1.0, 1.0], lag_point, &[EPS, 0.0], &[EPS]);

        assert_eq!(bracket.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn rhs_adds_the_noise_term_last_not_one_dot_over_z() {
        let lag_point = projection(1, 2, LAGS_ONLY);
        let lag_point = lag_point.lag_point();
        let coefficients = [1.0, 0.0];
        assert_ne!((1.0 + 1.0 * (EPS + EPS)).to_bits(), 1.0_f64.to_bits());

        let bracket = cut_bracket(1.0, &coefficients, lag_point, &[0.0, 0.0], &[EPS]);
        let noise = cut_noise_term(&coefficients, lag_point, &[EPS]);

        assert_eq!(bracket.to_bits(), 1.0_f64.to_bits());
        assert_eq!((bracket + noise).to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn shifted_dot_runs_in_ascending_slot_order() {
        let lag_point = projection(1, 4, LAGS_ONLY);
        let lag_point = lag_point.lag_point();
        assert_eq!(lag_point.shifted(), 1..4);
        assert_ne!(((EPS + EPS) + 1.0).to_bits(), 1.0_f64.to_bits());

        let bracket = cut_bracket(
            0.0,
            &[1.0, 1.0, 1.0, 1.0],
            lag_point,
            &[1.0, EPS, EPS, 0.0],
            &[0.0],
        );

        assert_eq!(bracket.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn bracket_accumulators_start_at_positive_zero() {
        let lag_point = projection(1, 2, LAGS_ONLY);
        let lag_point = lag_point.lag_point();
        let coefficients = [-1.0, -1.0];
        let values = [0.0, 0.0];
        let started_at_negative_zero = coefficients
            .iter()
            .zip(&values)
            .fold(-0.0_f64, |acc, (&b, &v)| acc + b * v);
        assert_eq!(started_at_negative_zero.to_bits(), (-0.0_f64).to_bits());

        let bracket = cut_bracket(-0.0, &coefficients, lag_point, &values, &[0.0]);

        assert_eq!(bracket.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn row_loops_equal_the_scalar_function_on_a_slot_subset() {
        let projection = projection(2, 2, ALL_ENABLED);
        let lag_point = projection.lag_point();
        let rows: [(f64, [f64; 6]); 3] = [
            (10.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (20.0, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            (30.0, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
        ];
        let mut pool = CutPool::new(3, projection.n_slots(), 1, 0);
        for (k, (intercept, row)) in (0_u64..).zip(&rows) {
            pool.add_cut(NodeId(0), k, 0, *intercept, row);
        }
        let (lag_state, mean, noise) = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0], [3.0, 4.0]);
        let slots = [2, 0];
        let (mut brackets, mut rhs) = ([f64::NAN; 2], [f64::NAN; 2]);

        cut_brackets_into(&pool, &slots, lag_point, &lag_state, &mean, &mut brackets);
        cut_rhs_into(&pool, &slots, lag_point, &noise, &brackets, &mut rhs);

        for (i, &slot) in slots.iter().enumerate() {
            let row = pool.coefficient_row(slot);
            let bracket = cut_bracket(pool.intercept(slot), row, lag_point, &lag_state, &mean);
            assert_eq!(brackets[i].to_bits(), bracket.to_bits());
            let expected_rhs = bracket + cut_noise_term(row, lag_point, &noise);
            assert_eq!(rhs[i].to_bits(), expected_rhs.to_bits());
        }
        assert_eq!(brackets.map(f64::to_bits), [56.0, 38.0].map(f64::to_bits));
        assert_eq!(rhs.map(f64::to_bits), [80.0, 63.0].map(f64::to_bits));
    }

    #[test]
    fn lag_ranges_come_from_the_projection_owner() {
        let all_enabled = projection(2, 2, ALL_ENABLED);
        let lags_only = projection(2, 2, LAGS_ONLY);
        let lag_coefficients = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let mut all_enabled_row = [1e300; 8];
        all_enabled_row[2..].copy_from_slice(&lag_coefficients);
        let (lag_state, mean, noise) = ([1.0, 2.0, 3.0, 4.0], [0.25, 0.5], [0.75, 1.0]);

        let from_all_enabled = (
            cut_bracket(
                1.0,
                &all_enabled_row[..6],
                all_enabled.lag_point(),
                &lag_state,
                &mean,
            ),
            cut_noise_term(&all_enabled_row[..6], all_enabled.lag_point(), &noise),
        );
        let from_lags_only = (
            cut_bracket(
                1.0,
                &lag_coefficients[..4],
                lags_only.lag_point(),
                &lag_state,
                &mean,
            ),
            cut_noise_term(&lag_coefficients[..4], lags_only.lag_point(), &noise),
        );

        assert_eq!(from_all_enabled.0.to_bits(), from_lags_only.0.to_bits());
        assert_eq!(from_all_enabled.1.to_bits(), from_lags_only.1.to_bits());

        let storage_only = projection(2, 2, STORAGE_ONLY);
        let bracket = cut_bracket(3.5, &[1.0, 2.0], storage_only.lag_point(), &[], &[]);
        let noise = cut_noise_term(&[1.0, 2.0], storage_only.lag_point(), &[]);
        assert_eq!(bracket.to_bits(), 3.5_f64.to_bits());
        assert_eq!((bracket + noise).to_bits(), 3.5_f64.to_bits());
    }
}
