//! Latin Hypercube Sampling (LHS) for batch ([`generate_lhs`]) and point-wise
//! ([`sample_lhs_point`], no inter-worker coordination) noise generation.
//!
//! Output layout is opening-major: `output[opening * dim + entity]`. Determinism:
//! the same `(base_seed, stage_id)` always produces identical output.

use rand::Rng;
use rand::RngExt;
use rand_distr::Uniform;

use super::NoisePointSpec;
use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_forward_seed, derive_opening_seed, derive_stage_seed},
};

pub(crate) fn fisher_yates(perm: &mut [usize], rng: &mut impl Rng) {
    let n = perm.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        perm.swap(i, j);
    }
}

fn reset_identity_perm(perm: &mut [usize]) {
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
}

#[allow(clippy::expect_used)]
fn unit_uniform() -> Uniform<f64> {
    Uniform::new(0.0_f64, 1.0_f64).expect("0.0 < 1.0 is always a valid range")
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values using LHS.
///
/// Each dimension is independently stratified into `N = n_openings` strata, one
/// sample each, with a per-dimension Fisher-Yates shuffle assigning strata to
/// openings.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim`.
pub fn generate_lhs(
    base_seed: u64,
    stage_id: u32,
    n_openings: usize,
    dim: usize,
    output: &mut [f64],
) {
    assert!(
        output.len() >= n_openings * dim,
        "output slice too short: need {}, got {}",
        n_openings * dim,
        output.len(),
    );

    if n_openings == 0 || dim == 0 {
        return;
    }

    let seed = derive_stage_seed(base_seed, stage_id);
    let mut rng = rng_from_seed(seed);
    let uniform = unit_uniform();

    let mut samples = vec![0.0_f64; n_openings];
    let mut perm: Vec<usize> = (0..n_openings).collect();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n_openings as f64;

    for d in 0..dim {
        // norm_quantile requires strictly positive input; clamp guards the k=0
        // case where u[0] can be 0.0.
        #[allow(clippy::cast_precision_loss)]
        for (k, sample) in samples.iter_mut().enumerate() {
            let u = rng.sample(uniform);
            let s = (k as f64 + u) / n_f;
            *sample = s.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        }

        reset_identity_perm(&mut perm);
        fisher_yates(&mut perm, &mut rng);

        for k in 0..n_openings {
            output[perm[k] * dim + d] = norm_quantile(samples[k]);
        }
    }
}

/// Derives the per-dimension stratum permutations per call; this is the
/// reference `sample_lhs_point` is tested against.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or `perm_scratch.len() < spec.total_scenarios as usize`.
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn sample_lhs_point_reference(
    spec: &NoisePointSpec,
    output: &mut [f64],
    perm_scratch: &mut [usize],
) {
    let n = spec.total_scenarios as usize;
    assert!(
        perm_scratch.len() >= n,
        "perm_scratch too short: need {n}, got {}",
        perm_scratch.len(),
    );
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    let perm_seed = derive_opening_seed(spec.sampling_seed, spec.iteration, spec.stream_id);
    let mut perm_rng = rng_from_seed(perm_seed);

    let draw_seed = derive_forward_seed(
        spec.sampling_seed,
        spec.iteration,
        spec.scenario,
        spec.stream_id,
    );
    let mut draw_rng = rng_from_seed(draw_seed);

    let uniform = unit_uniform();

    let perm = &mut perm_scratch[..n];
    let scenario_idx = spec.scenario as usize;

    for slot in output.iter_mut().take(spec.dim) {
        // perm_rng advances identically on all workers, so the permutation matches.
        reset_identity_perm(perm);
        fisher_yates(perm, &mut perm_rng);

        let stratum = perm[scenario_idx];
        let u_raw = draw_rng.sample(uniform);
        let u_stratified = (stratum as f64 + u_raw) / n as f64;

        *slot = norm_quantile(u_stratified.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON));
    }
}

/// Per-dimension stratum permutations built once per
/// (`sampling_seed`, `iteration`, `stream_id`, `dim`, `total_scenarios`) tuple and
/// reused across all scenarios for that stream. `strata` is row-major: `strata[d *
/// n + k]` is the stratum assigned to scenario `k` in dimension `d`.
#[derive(Debug, Clone)]
pub struct LhsPrecomputed {
    dim: usize,
    n: usize,
    strata: Vec<u32>,
}

impl LhsPrecomputed {
    /// Builds the permutation tables described on [`LhsPrecomputed`].
    #[must_use]
    pub fn new(
        sampling_seed: u64,
        iteration: u32,
        stream_id: u32,
        dim: usize,
        total_scenarios: u32,
    ) -> Self {
        let perm_seed = derive_opening_seed(sampling_seed, iteration, stream_id);
        let mut perm_rng = rng_from_seed(perm_seed);

        let n = total_scenarios as usize;
        let mut strata = vec![0u32; dim * n];
        let mut work: Vec<usize> = (0..n).collect();

        for d in 0..dim {
            reset_identity_perm(&mut work);
            fisher_yates(&mut work, &mut perm_rng);

            let row = &mut strata[d * n..(d + 1) * n];
            // Values are strictly less than n, which itself derives from a u32,
            // so the cast cannot truncate.
            #[allow(clippy::cast_possible_truncation)]
            for (slot, &p) in row.iter_mut().zip(work.iter()) {
                *slot = p as u32;
            }
        }

        Self { dim, n, strata }
    }
}

/// Generate one scenario's noise vector from precomputed stratum permutations.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`. Panics (debug-only) if `spec.scenario`
/// is out of range `0..ctx.n`.
#[allow(clippy::cast_precision_loss)]
pub fn sample_lhs_point(spec: &NoisePointSpec, ctx: &LhsPrecomputed, output: &mut [f64]) {
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    debug_assert!(
        spec.dim <= ctx.dim,
        "spec.dim {} exceeds precomputed dim {}",
        spec.dim,
        ctx.dim,
    );

    let draw_seed = derive_forward_seed(
        spec.sampling_seed,
        spec.iteration,
        spec.scenario,
        spec.stream_id,
    );
    let mut draw_rng = rng_from_seed(draw_seed);

    let uniform = unit_uniform();

    let scenario_idx = spec.scenario as usize;
    debug_assert!(
        scenario_idx < ctx.n,
        "scenario {} out of range 0..{}",
        spec.scenario,
        ctx.n,
    );

    for (d, slot) in output.iter_mut().take(spec.dim).enumerate() {
        let stratum = ctx.strata[d * ctx.n + scenario_idx];
        let u_raw = draw_rng.sample(uniform);
        let u_stratified = (f64::from(stratum) + u_raw) / ctx.n as f64;

        *slot = norm_quantile(u_stratified.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON));
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    use super::{
        LhsPrecomputed, NoisePointSpec, fisher_yates, generate_lhs, sample_lhs_point,
        sample_lhs_point_reference,
    };

    #[test]
    fn fisher_yates_is_permutation() {
        let n = 20_usize;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = Pcg64::seed_from_u64(42);
        fisher_yates(&mut perm, &mut rng);

        let mut sorted = perm.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            sorted, expected,
            "shuffled slice is not a permutation of 0..{n}"
        );
    }

    #[test]
    fn fisher_yates_different_states_differ() {
        let n = 10_usize;
        let mut perm1: Vec<usize> = (0..n).collect();
        let mut perm2: Vec<usize> = (0..n).collect();
        let mut rng1 = Pcg64::seed_from_u64(1);
        let mut rng2 = Pcg64::seed_from_u64(999_999_999);
        fisher_yates(&mut perm1, &mut rng1);
        fisher_yates(&mut perm2, &mut rng2);
        // With n=10 the probability of a collision is 1/10! ≈ 2.76e-7.
        assert_ne!(
            perm1, perm2,
            "two different RNG states produced the same permutation"
        );
    }

    #[test]
    fn fisher_yates_edge_cases_do_not_panic() {
        let mut rng = Pcg64::seed_from_u64(0);
        let mut empty: Vec<usize> = vec![];
        fisher_yates(&mut empty, &mut rng);
        assert!(empty.is_empty());

        let mut single = vec![7_usize];
        fisher_yates(&mut single, &mut rng);
        assert_eq!(single, vec![7]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn lhs_determinism() {
        let n_openings = 50;
        let dim = 3;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut out1);
        generate_lhs(42, 0, n_openings, dim, &mut out2);
        assert_eq!(
            out1, out2,
            "generate_lhs is not deterministic for the same seed"
        );
    }

    #[test]
    fn lhs_different_seeds_differ() {
        let n_openings = 20;
        let dim = 2;
        let mut out_a = vec![0.0_f64; n_openings * dim];
        let mut out_b = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut out_a);
        generate_lhs(99, 0, n_openings, dim, &mut out_b);
        assert_ne!(out_a, out_b, "different seeds produced identical output");
    }

    #[test]
    fn lhs_correct_length() {
        let n_openings = 10;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(1, 2, n_openings, dim, &mut output);
        assert_eq!(output.len(), n_openings * dim);
    }

    #[test]
    fn lhs_all_finite() {
        let n_openings = 30;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    #[test]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn lhs_marginal_stratification() {
        let n_openings = 50_usize;
        let dim = 3_usize;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut output);

        let n_f = n_openings as f64;
        let approx_cdf = |z: f64| -> f64 { 0.5 * (1.0 + libm_erf(z / std::f64::consts::SQRT_2)) };

        for d in 0..dim {
            let mut strata: Vec<usize> = (0..n_openings)
                .map(|k| {
                    let z = output[k * dim + d];
                    let p = approx_cdf(z);
                    // Clamp to [0, N-1] to handle floating-point boundary cases.
                    let stratum = (p * n_f).floor() as usize;
                    stratum.min(n_openings - 1)
                })
                .collect();
            strata.sort_unstable();
            let expected: Vec<usize> = (0..n_openings).collect();
            assert_eq!(
                strata, expected,
                "dimension {d}: CDF-floor indices are not a permutation of 0..{n_openings}"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lhs_mean_and_std_within_tolerance() {
        let n_openings = 1000_usize;
        let dim = 1_usize;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut output);

        let n = n_openings as f64;
        let mean = output.iter().sum::<f64>() / n;
        let variance = output.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        assert!(
            mean.abs() < 0.1,
            "mean {mean:.4} too far from 0 (tolerance 0.1)"
        );
        assert!(
            (std - 1.0).abs() < 0.1,
            "std {std:.4} too far from 1 (tolerance 0.1)"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn lhs_zero_openings_does_not_panic() {
        let mut output = vec![99.0_f64; 0];
        generate_lhs(1, 0, 0, 3, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn lhs_zero_dim_does_not_panic() {
        let mut output: Vec<f64> = vec![];
        generate_lhs(1, 0, 5, 0, &mut output);
    }

    /// Approximate `erf(x)` using the Horner-form rational approximation
    /// (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
    fn libm_erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::cast_possible_truncation)]
    fn lhs_point_determinism() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        let mut perm = vec![0_usize; n];
        let spec = NoisePointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 0,
            stream_id: 0,
            total_scenarios: n as u32,
            dim,
        };

        sample_lhs_point_reference(&spec, &mut out1, &mut perm);
        sample_lhs_point_reference(&spec, &mut out2, &mut perm);

        assert_eq!(
            out1, out2,
            "sample_lhs_point is not deterministic for the same inputs"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lhs_point_different_seeds_differ() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        let mut perm = vec![0_usize; n];

        sample_lhs_point_reference(
            &NoisePointSpec {
                sampling_seed: 42,
                iteration: 0,
                scenario: 0,
                stream_id: 0,
                total_scenarios: n as u32,
                dim,
            },
            &mut out1,
            &mut perm,
        );
        sample_lhs_point_reference(
            &NoisePointSpec {
                sampling_seed: 43,
                iteration: 0,
                scenario: 0,
                stream_id: 0,
                total_scenarios: n as u32,
                dim,
            },
            &mut out2,
            &mut perm,
        );

        assert_ne!(
            out1, out2,
            "different sampling_seeds produced identical output"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lhs_point_all_finite() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut perm = vec![0_usize; n];

        for scenario in 0..n {
            let mut output = vec![0.0_f64; dim];
            sample_lhs_point_reference(
                &NoisePointSpec {
                    sampling_seed: 42,
                    iteration: 0,
                    scenario: scenario as u32,
                    stream_id: 0,
                    total_scenarios: n as u32,
                    dim,
                },
                &mut output,
                &mut perm,
            );
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite value at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }

    /// Point-wise sampling forms a valid LHS design with no inter-worker
    /// communication.
    #[test]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn lhs_point_stratum_coverage() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut perm = vec![0_usize; n];
        let n_f = n as f64;
        let approx_cdf = |z: f64| -> f64 { 0.5 * (1.0 + libm_erf(z / std::f64::consts::SQRT_2)) };

        let mut strata_by_dim: Vec<Vec<usize>> = (0..dim).map(|_| Vec::with_capacity(n)).collect();
        for scenario in 0..n {
            let mut output = vec![0.0_f64; dim];
            sample_lhs_point_reference(
                &NoisePointSpec {
                    sampling_seed: 42,
                    iteration: 0,
                    scenario: scenario as u32,
                    stream_id: 0,
                    total_scenarios: n as u32,
                    dim,
                },
                &mut output,
                &mut perm,
            );
            for (d, &v) in output.iter().enumerate() {
                let p = approx_cdf(v);
                let stratum = ((p * n_f).floor() as usize).min(n - 1);
                strata_by_dim[d].push(stratum);
            }
        }

        for (d, strata) in strata_by_dim.iter_mut().enumerate() {
            strata.sort_unstable();
            let expected: Vec<usize> = (0..n).collect();
            assert_eq!(
                *strata, expected,
                "dimension {d}: strata across all scenarios are not a permutation of 0..{n}"
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn lhs_point_matches_reference() {
        for (dim, total_scenarios) in [(1_usize, 4_u32), (3, 10), (2, 16)] {
            let ctx = LhsPrecomputed::new(11, 2, 1, dim, total_scenarios);
            let mut precomputed_out = vec![0.0_f64; dim];
            let mut direct_out = vec![0.0_f64; dim];
            let mut perm_scratch = vec![0_usize; total_scenarios as usize];

            for scenario in 0..total_scenarios {
                let spec = NoisePointSpec {
                    sampling_seed: 11,
                    iteration: 2,
                    scenario,
                    stream_id: 1,
                    total_scenarios,
                    dim,
                };

                sample_lhs_point(&spec, &ctx, &mut precomputed_out);
                sample_lhs_point_reference(&spec, &mut direct_out, &mut perm_scratch);

                assert_eq!(
                    precomputed_out, direct_out,
                    "mismatch at dim={dim}, total_scenarios={total_scenarios}, scenario={scenario}"
                );
            }
        }
    }

    #[test]
    fn lhs_precomputed_strata_rows_are_permutations() {
        let dim = 3_usize;
        let total_scenarios = 10_u32;
        let n = total_scenarios as usize;
        let ctx = LhsPrecomputed::new(11, 2, 1, dim, total_scenarios);

        for d in 0..dim {
            let mut row: Vec<u32> = ctx.strata[d * n..(d + 1) * n].to_vec();
            row.sort_unstable();
            let expected: Vec<u32> = (0..total_scenarios).collect();
            assert_eq!(
                row, expected,
                "dimension {d}: strata row is not a permutation of 0..{n}"
            );
        }
    }
}
