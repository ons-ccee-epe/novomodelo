//! Canonical cut-row RHS: B = fl(fl(α + S) + P) and RHS = fl(B + W), the one association both RHS
//! writers (`cut_rhs_into` and `cut_rhs_from_noise_terms_into`) use. S, P and W are ascending-slot
//! dots from +0.0: the shifted slots against â, the lag-0 slots against p, and the lag-0 slots
//! against σ·η. Not one dot over ẑ = fl(p + σ·η), not the blocked GEMM kernel for S, not a dot
//! started from an iterator sum's −0.0, no fused multiply-add.
//!
//! `cut_brackets_into` and `cut_noise_terms_into` take `LANES` rows at a time, lanes across rows
//! only, each row ascending from +0.0; `rhs_reference` is the scalar statement they equal bit for
//! bit.

use std::array;

use crate::cut::CutPool;
use crate::indexer::CutLagPoint;

const LANES: usize = 8;
const CHUNK: usize = 256;

type BracketLanes = fn(&CutPool, &[usize], &CutLagPoint, &[f64], &[f64], &mut [f64]);
type NoiseLanes = fn(&CutPool, &[usize], &CutLagPoint, &[f64], &mut [f64]);

const BRACKET_LANES: [BracketLanes; LANES] = [
    bracket_lanes::<1>,
    bracket_lanes::<2>,
    bracket_lanes::<3>,
    bracket_lanes::<4>,
    bracket_lanes::<5>,
    bracket_lanes::<6>,
    bracket_lanes::<7>,
    bracket_lanes::<LANES>,
];

const NOISE_LANES: [NoiseLanes; LANES] = [
    noise_lanes::<1>,
    noise_lanes::<2>,
    noise_lanes::<3>,
    noise_lanes::<4>,
    noise_lanes::<5>,
    noise_lanes::<6>,
    noise_lanes::<7>,
    noise_lanes::<LANES>,
];

pub(crate) fn ascending_dot(coefficients: &[f64], values: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(values)
        .fold(0.0, |acc, (&b, &v)| acc + b * v)
}

/// Writes `noise_terms[i]`, the W of the cut at `slots[i]`, reading each row in place from `pool`;
/// the RHS is `fl(bracket + W)`.
pub fn cut_noise_terms_into(
    pool: &CutPool,
    slots: &[usize],
    lag_point: &CutLagPoint,
    noise_term: &[f64],
    noise_terms: &mut [f64],
) {
    debug_assert!(
        noise_terms.len() == slots.len(),
        "cut_noise_terms_into: noise_terms and slots differ in length"
    );

    for (group, out) in slots.chunks(LANES).zip(noise_terms.chunks_mut(LANES)) {
        NOISE_LANES[group.len() - 1](pool, group, lag_point, noise_term, out);
    }
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

    for (group, out) in slots.chunks(LANES).zip(brackets.chunks_mut(LANES)) {
        BRACKET_LANES[group.len() - 1](pool, group, lag_point, lag_state, conditional_mean, out);
    }
}

fn bracket_lanes<const R: usize>(
    pool: &CutPool,
    group: &[usize],
    lag_point: &CutLagPoint,
    lag_state: &[f64],
    conditional_mean: &[f64],
    brackets: &mut [f64],
) {
    let (lag0, shifted) = (lag_point.lag0(), lag_point.shifted());
    let rows: [&[f64]; R] = array::from_fn(|r| pool.coefficient_row(group[r]));
    let shifted_dots = ascending_dots(
        rows.map(|row| &row[shifted.clone()]),
        &lag_state[..shifted.len()],
    );
    let mean_dots = ascending_dots(
        rows.map(|row| &row[lag0.clone()]),
        &conditional_mean[..lag0.len()],
    );
    for (r, &slot) in group.iter().enumerate() {
        brackets[r] = (pool.intercept(slot) + shifted_dots[r]) + mean_dots[r];
    }
}

fn noise_lanes<const R: usize>(
    pool: &CutPool,
    group: &[usize],
    lag_point: &CutLagPoint,
    noise_term: &[f64],
    noise_terms: &mut [f64],
) {
    let lag0 = lag_point.lag0();
    let rows: [&[f64]; R] = array::from_fn(|r| &pool.coefficient_row(group[r])[lag0.clone()]);
    noise_terms.copy_from_slice(&ascending_dots(rows, &noise_term[..lag0.len()]));
}

// A chunk only blocks the loads: each lane's accumulator runs across chunks, never one partial
// per chunk.
fn ascending_dots<const R: usize>(rows: [&[f64]; R], values: &[f64]) -> [f64; R] {
    let mut acc = [0.0_f64; R];
    for (chunk_at, chunk) in values.chunks(CHUNK).enumerate() {
        let base = chunk_at * CHUNK;
        let tiles: [&[f64]; R] = array::from_fn(|r| &rows[r][base..base + chunk.len()]);
        for (j, &v) in chunk.iter().enumerate() {
            for (a, tile) in acc.iter_mut().zip(&tiles) {
                *a += tile[j] * v;
            }
        }
    }
    acc
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

    cut_noise_terms_into(pool, slots, lag_point, noise_term, rhs);
    for (out, &bracket) in rhs.iter_mut().zip(brackets) {
        *out += bracket;
    }
}

/// Writes `rhs[i] = brackets[i] + noise_terms[i]`, where `noise_terms[i]` is the W of row `i`
/// that [`cut_noise_terms_into`] wrote.
///
/// # Panics
///
/// If the three slices differ in length.
pub fn cut_rhs_from_noise_terms_into(brackets: &[f64], noise_terms: &[f64], rhs: &mut [f64]) {
    assert!(
        noise_terms.len() == brackets.len() && rhs.len() == brackets.len(),
        "cut_rhs_from_noise_terms_into: brackets, noise_terms and rhs differ in length"
    );

    for ((out, &bracket), &noise_term) in rhs.iter_mut().zip(brackets).zip(noise_terms) {
        *out = bracket + noise_term;
    }
}

#[cfg(test)]
mod tests {
    use novomodelo_core::temporal::StageStateConfig;
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{
        LANES, cut_brackets_into, cut_noise_terms_into, cut_rhs_from_noise_terms_into, cut_rhs_into,
    };
    use crate::cut::CutPool;
    use crate::cut::rhs_reference::{cut_bracket, cut_brackets_reference_into, cut_noise_term};
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
        let projection = projection(1, 2, LAGS_ONLY);
        let lag_point = projection.lag_point();
        let coefficients = [-1.0, -1.0];
        let values = [0.0, 0.0];
        let started_at_negative_zero = coefficients
            .iter()
            .zip(&values)
            .fold(-0.0_f64, |acc, (&b, &v)| acc + b * v);
        assert_eq!(started_at_negative_zero.to_bits(), (-0.0_f64).to_bits());

        let mut pool = CutPool::new(LANES, projection.n_slots(), 1, 0);
        for k in 0..LANES as u64 {
            pool.add_cut(NodeId(0), k, 0, -0.0, &coefficients);
        }
        for rows in [1, LANES] {
            let slots: Vec<usize> = (0..rows).collect();
            let mut brackets = vec![f64::NAN; rows];
            cut_brackets_into(&pool, &slots, lag_point, &values, &[0.0], &mut brackets);
            assert!(
                brackets
                    .iter()
                    .all(|bracket| bracket.to_bits() == 0.0_f64.to_bits()),
                "{rows} rows"
            );
        }
    }

    #[test]
    fn noise_term_accumulators_start_at_positive_zero() {
        let projection = projection(1, 2, LAGS_ONLY);
        let lag_point = projection.lag_point();
        let coefficients = [-1.0, -1.0];
        let mut pool = CutPool::new(LANES, projection.n_slots(), 1, 0);
        for k in 0..LANES as u64 {
            pool.add_cut(NodeId(0), k, 0, -0.0, &coefficients);
        }
        for rows in [1, LANES] {
            let slots: Vec<usize> = (0..rows).collect();
            let mut noise_terms = vec![f64::NAN; rows];
            cut_noise_terms_into(&pool, &slots, lag_point, &[0.0], &mut noise_terms);
            assert!(
                noise_terms.iter().all(|w| w.to_bits() == 0.0_f64.to_bits()),
                "{rows} rows"
            );
        }
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
    fn cut_rhs_from_noise_terms_equals_cut_rhs_into_bit_for_bit() {
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
        let (lag_state, mean, noise) = ([0.1, 0.2, 0.3, 0.4], [0.7, 0.9], [0.1, -0.3]);
        let slots = [2, 0, 1];
        let mut brackets = [f64::NAN; 3];
        cut_brackets_into(&pool, &slots, lag_point, &lag_state, &mean, &mut brackets);
        let noise_terms =
            slots.map(|slot| cut_noise_term(pool.coefficient_row(slot), lag_point, &noise));
        assert!(noise_terms.iter().all(|&w| w != 0.0));

        let mut expected = [f64::NAN; 3];
        let mut actual = [f64::NAN; 3];
        cut_rhs_into(&pool, &slots, lag_point, &noise, &brackets, &mut expected);
        cut_rhs_from_noise_terms_into(&brackets, &noise_terms, &mut actual);

        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }

    #[test]
    #[should_panic(expected = "brackets, noise_terms and rhs differ in length")]
    fn noise_terms_of_another_length_panic() {
        let brackets = [1.0, 2.0, 3.0];
        let mut rhs = [f64::NAN; 3];

        cut_rhs_from_noise_terms_into(&brackets, &[0.5, 0.5], &mut rhs);
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

    const MAX_ROWS: usize = 64;
    const MIN_ROWS: usize = 2 * LANES + 2;

    struct SplitMix(u64);

    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }

        fn value(&mut self) -> f64 {
            let r = self.next_u64();
            let sign = (r & 1) << 63;
            if (r >> 1).is_multiple_of(8) {
                return f64::from_bits(sign);
            }
            let exponent = 983 + (r >> 4) % 81;
            f64::from_bits(sign | exponent << 52 | self.next_u64() >> 12)
        }

        fn values(&mut self, len: usize) -> Vec<f64> {
            (0..len).map(|_| self.value()).collect()
        }

        fn subset(&mut self, n_slots: usize, n_rows: usize) -> Vec<usize> {
            let mut slots: Vec<usize> = (0..n_slots).collect();
            for i in 0..n_rows {
                let j = i + self.below(n_slots - i);
                slots.swap(i, j);
            }
            slots.truncate(n_rows);
            slots.sort_unstable();
            slots
        }
    }

    fn random_pool(rng: &mut SplitMix, n_slots: usize, state_dimension: usize) -> CutPool {
        let mut pool = CutPool::new(n_slots, state_dimension, 1, 0);
        for k in 0..n_slots as u64 {
            let (intercept, row) = (rng.value(), rng.values(state_dimension));
            pool.add_cut(NodeId(0), k, 0, intercept, &row);
        }
        pool
    }

    fn subset_draw() -> impl Strategy<Value = (usize, usize, u64)> {
        (MIN_ROWS..=MAX_ROWS)
            .prop_flat_map(|n_slots| (Just(n_slots), MIN_ROWS..=n_slots, any::<u64>()))
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::with_cases(32)
        })]

        #[test]
        fn batched_brackets_equal_the_scalar_reference_bit_for_bit(
            (n_slots, n_rows, seed) in subset_draw(),
        ) {
            let projection = projection(168, 12, LAGS_ONLY);
            let lag_point = projection.lag_point();
            prop_assert_eq!(projection.n_slots(), 2_016);
            prop_assert_eq!((lag_point.shifted().len(), lag_point.lag0().len()), (1_848, 168));

            let mut rng = SplitMix(seed);
            let pool = random_pool(&mut rng, n_slots, projection.n_slots());
            let (lag_state, mean) = (rng.values(1_848), rng.values(168));
            let slots = rng.subset(n_slots, n_rows);
            let mut reference = vec![f64::NAN; n_rows];
            cut_brackets_reference_into(&pool, &slots, lag_point, &lag_state, &mean, &mut reference);

            let mut batched = vec![f64::NAN; n_rows];
            for len in (1..=2 * LANES + 1).filter(|&len| len < n_rows).chain([n_rows]) {
                batched[..len].fill(f64::NAN);
                cut_brackets_into(&pool, &slots[..len], lag_point, &lag_state, &mean, &mut batched[..len]);
                let mismatch = batched[..len]
                    .iter()
                    .zip(&reference)
                    .position(|(got, want)| got.to_bits() != want.to_bits());
                prop_assert_eq!(mismatch, None, "{} rows", len);
            }
        }

        #[test]
        fn batched_noise_terms_equal_the_scalar_reference_bit_for_bit(
            (n_slots, n_rows, seed) in subset_draw(),
        ) {
            let projection = projection(168, 12, LAGS_ONLY);
            let lag_point = projection.lag_point();
            prop_assert_eq!(lag_point.lag0().len(), 168);

            let mut rng = SplitMix(seed);
            let pool = random_pool(&mut rng, n_slots, projection.n_slots());
            let noise = rng.values(168);
            let slots = rng.subset(n_slots, n_rows);
            let reference: Vec<f64> = slots
                .iter()
                .map(|&slot| cut_noise_term(pool.coefficient_row(slot), lag_point, &noise))
                .collect();

            let mut batched = vec![f64::NAN; n_rows];
            for len in (1..=2 * LANES + 1).filter(|&len| len < n_rows).chain([n_rows]) {
                batched[..len].fill(f64::NAN);
                cut_noise_terms_into(&pool, &slots[..len], lag_point, &noise, &mut batched[..len]);
                let mismatch = batched[..len]
                    .iter()
                    .zip(&reference)
                    .position(|(got, want)| got.to_bits() != want.to_bits());
                prop_assert_eq!(mismatch, None, "{} rows", len);
            }
        }
    }
}
