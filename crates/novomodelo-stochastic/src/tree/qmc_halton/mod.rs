//! Halton quasi-Monte Carlo sequence building blocks and generators.
//!
//! The Halton sequence assigns dimension `d` (1-indexed) the `d`-th prime base
//! (2, 3, 5, …); point `n`'s coordinate in dimension `d` is
//! `radical_inverse(n, p_d)`, with dimension 1 the van der Corput base-2 sequence.
//!
//! [`scrambled_halton_point`] generates one scenario independently of all others.
//!
//! Plain Halton suffers correlation artifacts in high dimensions (the "Halton
//! curse"). Owen-style random digit scrambling breaks these by permuting each
//! digit position per dimension via tables derived deterministically from the
//! stage seed, so output stays reproducible.

use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_opening_seed, derive_stage_seed},
};
use crate::tree::NoisePointSpec;

use super::lhs::fisher_yates;

/// Return the first `count` primes in ascending order via the sieve of
/// Eratosthenes; empty for `count == 0`.
#[must_use]
pub(crate) fn sieve_primes(count: usize) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }

    // Prime number theorem n-th-prime bound p_n < n*(ln n + ln ln n) + 2 for
    // n >= 6; hard floor 30 below that avoids non-positive bounds.
    let upper_bound: usize = if count < 6 {
        30
    } else {
        // Upper-bound estimate; f64's 52-bit mantissa covers any practical
        // sieve size (count near 2^52 is unreachable).
        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;
        let ln_n = n.ln();
        let ln_ln_n = ln_n.ln();
        // Positive and finite for n >= 6; truncation yields a small positive int.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bound = (n * (ln_n + ln_ln_n) + 4.0) as usize + 4;
        bound
    };

    let mut composite = vec![false; upper_bound + 1];
    composite[0] = true;
    composite[1] = true;

    let mut p = 2usize;
    while p * p <= upper_bound {
        if !composite[p] {
            let mut multiple = p * p;
            while multiple <= upper_bound {
                composite[multiple] = true;
                multiple += p;
            }
        }
        p += 1;
    }

    // Primes fit in u32 for any practical dimension count (the 1,000,000th prime
    // is 15,485,863).
    #[allow(clippy::cast_possible_truncation)]
    composite
        .iter()
        .enumerate()
        .filter(|&(_, &is_composite)| !is_composite)
        .map(|(i, _)| i as u32)
        .take(count)
        .collect()
}

/// Compute the base-`b` radical inverse of integer `n`: reflecting the base-`b`
/// digits of `n` about the point, so `n = d_k … d_0` maps to `0.d_0 … d_k`.
/// Returns `0.0` for `n == 0`, else a value in `[0.0, 1.0)`.
///
/// # Preconditions
///
/// `base >= 2` (a `debug_assert!` enforces it; in release `base < 2` returns
/// `0.0` without panicking).
#[cfg(test)]
#[must_use]
fn radical_inverse(n: u32, base: u32) -> f64 {
    debug_assert!(base >= 2, "radical_inverse requires base >= 2, got {base}");

    let mut result = 0.0_f64;
    let mut inv_base = 1.0 / f64::from(base);
    let mut n = n;

    while n > 0 {
        let digit = n % base;
        result += f64::from(digit) * inv_base;
        n /= base;
        inv_base /= f64::from(base);
    }

    result
}

/// Build Owen-style random digit scramble tables for all dimensions.
///
/// Returns `tables[d][j][digit]`: the scrambled value of `digit` in
/// `0..primes[d]` at digit position `j` (0-indexed, least-significant first) for
/// dimension `d`; each dimension has `max(1, ceil(log_{p_d}(max_n)))` positions.
/// A single `seed`-seeded RNG advances sequentially across all dimensions and
/// positions, so the tables are deterministic and independent between dimensions.
fn build_scramble_tables(seed: u64, primes: &[u32], max_n: usize) -> Vec<Vec<Vec<u32>>> {
    let mut rng = rng_from_seed(seed);
    let mut tables = Vec::with_capacity(primes.len());

    for &base in primes {
        let max_digits = if max_n <= 1 {
            1
        } else {
            // Divide (max_n - 1), not max_n: counts floor(log_base(max_n-1)) + 1
            // digits, the largest index actually generated.
            let mut digits = 0usize;
            let mut val = max_n - 1;
            while val > 0 {
                val /= base as usize;
                digits += 1;
            }
            digits.max(1)
        };

        let base_usize = base as usize;
        let mut dim_table: Vec<Vec<u32>> = Vec::with_capacity(max_digits);
        let mut work: Vec<usize> = (0..base_usize).collect();

        for _ in 0..max_digits {
            for (i, slot) in work.iter_mut().enumerate() {
                *slot = i;
            }
            fisher_yates(&mut work, &mut rng);
            // Values are in 0..base, so the u32 cast cannot truncate.
            #[allow(clippy::cast_possible_truncation)]
            let perm: Vec<u32> = work.iter().map(|&v| v as u32).collect();
            dim_table.push(perm);
        }

        tables.push(dim_table);
    }

    tables
}

/// Compute the scrambled base-`base` radical inverse of integer `n`.
///
/// Applies the Owen-style random digit permutation `perm_table[j][digit]`
/// at each digit position `j` before accumulating the radical inverse.
/// Returns a value in `[0.0, 1.0)`.
fn scrambled_radical_inverse(n: u32, base: u32, perm_table: &[Vec<u32>]) -> f64 {
    debug_assert!(base >= 2, "base must be >= 2, got {base}");

    let mut result = 0.0_f64;
    let mut inv_base = 1.0 / f64::from(base);
    let mut n = n;
    let mut j = 0usize;

    while n > 0 {
        let digit = n % base;
        // Identity fallback when n has more digits than the table has positions.
        let scrambled = if j < perm_table.len() {
            perm_table[j][digit as usize]
        } else {
            digit
        };
        result += f64::from(scrambled) * inv_base;
        n /= base;
        inv_base /= f64::from(base);
        j += 1;
    }

    result
}

fn scrambled_normal_sample(n: u32, base: u32, perm_table: &[Vec<u32>]) -> f64 {
    let u = scrambled_radical_inverse(n, base, perm_table);
    let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    norm_quantile(u)
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values using
/// scrambled Halton QMC. Output layout: opening-major
/// `output[opening * dim + entity]`. The scramble tables derive from the
/// stage-derived seed, so distinct stages produce distinct samples.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim`.
pub fn generate_qmc_halton(
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
    let primes = sieve_primes(dim);
    let tables = build_scramble_tables(seed, &primes, n_openings);

    for n in 0..n_openings {
        // Opening counts never exceed u32::MAX, so the cast cannot truncate.
        #[allow(clippy::cast_possible_truncation)]
        let n_u32 = n as u32;
        for d in 0..dim {
            output[n * dim + d] = scrambled_normal_sample(n_u32, primes[d], &tables[d]);
        }
    }
}

/// Prime table and scramble tables built once per
/// (`sampling_seed`, `iteration`, `stream_id`, `dim`, `total_scenarios`) tuple and
/// reused across all scenarios at that stage.
#[derive(Debug, Clone)]
pub struct HaltonPrecomputed {
    primes: Vec<u32>,
    tables: Vec<Vec<Vec<u32>>>,
}

impl HaltonPrecomputed {
    /// Builds the primes and scramble tables described on [`HaltonPrecomputed`].
    #[must_use]
    pub fn new(
        sampling_seed: u64,
        iteration: u32,
        stream_id: u32,
        dim: usize,
        total_scenarios: u32,
    ) -> Self {
        let seed = derive_opening_seed(sampling_seed, iteration, stream_id);
        let primes = sieve_primes(dim);
        let tables = build_scramble_tables(seed, &primes, total_scenarios as usize);
        Self { primes, tables }
    }
}

/// Generate one scenario's noise vector from precomputed primes and scramble
/// tables.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`.
pub fn scrambled_halton_point(spec: &NoisePointSpec, ctx: &HaltonPrecomputed, output: &mut [f64]) {
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    debug_assert!(
        spec.scenario < spec.total_scenarios,
        "scenario {} out of range 0..{}",
        spec.scenario,
        spec.total_scenarios,
    );

    if spec.dim == 0 {
        return;
    }

    for (d, out) in output.iter_mut().enumerate().take(spec.dim) {
        *out = scrambled_normal_sample(spec.scenario, ctx.primes[d], &ctx.tables[d]);
    }
}

/// Derives the primes and scramble tables per call; this is the reference
/// `scrambled_halton_point` is tested against.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`.
#[cfg(test)]
pub(crate) fn scrambled_halton_point_reference(spec: &NoisePointSpec, output: &mut [f64]) {
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    debug_assert!(
        spec.scenario < spec.total_scenarios,
        "scenario {} out of range 0..{}",
        spec.scenario,
        spec.total_scenarios,
    );

    if spec.dim == 0 {
        return;
    }

    let seed = derive_opening_seed(spec.sampling_seed, spec.iteration, spec.stream_id);
    let primes = sieve_primes(spec.dim);
    let tables = build_scramble_tables(seed, &primes, spec.total_scenarios as usize);

    for d in 0..spec.dim {
        let u = scrambled_radical_inverse(spec.scenario, primes[d], &tables[d]);
        let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        output[d] = norm_quantile(u);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::{
        HaltonPrecomputed, NoisePointSpec, generate_qmc_halton, radical_inverse,
        scrambled_halton_point, scrambled_halton_point_reference, sieve_primes,
    };

    #[test]
    fn test_sieve_first_10_primes() {
        assert_eq!(sieve_primes(10), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_zero_returns_empty() {
        assert_eq!(sieve_primes(0), Vec::<u32>::new());
    }

    #[test]
    fn test_sieve_one_returns_two() {
        assert_eq!(sieve_primes(1), vec![2]);
    }

    #[test]
    fn test_sieve_100_primes_count() {
        let primes = sieve_primes(100);
        assert_eq!(primes.len(), 100);
        assert_eq!(*primes.last().unwrap(), 541, "100th prime must be 541");
    }

    #[test]
    fn test_radical_inverse_base2_known_values() {
        // Van der Corput sequence in base 2: n -> radical_inverse(n, 2).
        let expected = [
            (0_u32, 0.0_f64),
            (1, 0.5),
            (2, 0.25),
            (3, 0.75),
            (4, 0.125),
            (5, 0.625),
            (6, 0.375),
            (7, 0.875),
        ];
        for (n, exp) in expected {
            let got = radical_inverse(n, 2);
            assert!(
                (got - exp).abs() < 1e-15,
                "radical_inverse({n}, 2): got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_radical_inverse_base3_known_values() {
        // n=3 is ternary "10", reflecting to 0.01 ternary = 1/9.
        let expected = [
            (0_u32, 0.0_f64),
            (1, 1.0 / 3.0),
            (2, 2.0 / 3.0),
            (3, 1.0 / 9.0),
        ];
        for (n, exp) in expected {
            let got = radical_inverse(n, 3);
            assert!(
                (got - exp).abs() < 1e-15,
                "radical_inverse({n}, 3): got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_radical_inverse_base5_n1() {
        // radical_inverse(1, 5) = 1/5 = 0.2
        let got = radical_inverse(1, 5);
        assert!(
            (got - 0.2_f64).abs() < 1e-15,
            "radical_inverse(1, 5): got {got}, expected 0.2"
        );
    }

    #[test]
    fn test_radical_inverse_range() {
        for base in [2_u32, 3, 5, 7] {
            for n in 1_u32..100 {
                let v = radical_inverse(n, base);
                assert!(
                    v > 0.0 && v < 1.0,
                    "radical_inverse({n}, {base}) = {v} is not in (0.0, 1.0)"
                );
            }
        }
    }

    // --- Batch generator tests ---

    #[test]
    fn test_halton_batch_determinism() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(42, 0, n_openings, dim, &mut out1);
        generate_qmc_halton(42, 0, n_openings, dim, &mut out2);
        assert_eq!(out1, out2, "generate_qmc_halton is not deterministic");
    }

    #[test]
    fn test_halton_batch_different_seeds_differ() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(42, 0, n_openings, dim, &mut out1);
        generate_qmc_halton(99, 0, n_openings, dim, &mut out2);
        assert_ne!(out1, out2, "different seeds produced identical output");
    }

    #[test]
    fn test_halton_batch_all_finite() {
        let n_openings = 64;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    /// The BSM approximation clamps at ±8.22.
    #[test]
    fn test_halton_batch_values_in_range() {
        let n_openings = 64;
        let dim = 2;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}");
            assert!(
                v > -8.22 && v < 8.22,
                "value {v} out of range (-8.22, 8.22) at index {i}"
            );
        }
    }

    #[test]
    fn test_halton_batch_correct_length() {
        let n_openings = 32;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(1, 2, n_openings, dim, &mut output);
        assert_eq!(output.len(), n_openings * dim);
    }

    #[test]
    fn test_halton_batch_zero_openings() {
        let mut output: Vec<f64> = vec![];
        generate_qmc_halton(42, 0, 0, 3, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_halton_batch_zero_dim() {
        let mut output: Vec<f64> = vec![];
        generate_qmc_halton(42, 0, 5, 0, &mut output);
        assert!(output.is_empty());
    }

    // --- Point-wise generator tests ---

    #[test]
    fn test_halton_point_determinism() {
        let dim = 2;
        let spec = NoisePointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 0,
            stream_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_halton_point_reference(&spec, &mut out1);
        scrambled_halton_point_reference(&spec, &mut out2);
        assert_eq!(out1, out2, "scrambled_halton_point is not deterministic");
    }

    #[test]
    fn test_halton_point_different_seeds_differ() {
        let dim = 3;
        let base_spec = NoisePointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 5,
            stream_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_halton_point_reference(&base_spec, &mut out1);
        scrambled_halton_point_reference(
            &NoisePointSpec {
                sampling_seed: 43,
                ..base_spec
            },
            &mut out2,
        );
        assert_ne!(
            out1, out2,
            "different sampling_seeds produced identical output"
        );
    }

    #[test]
    fn test_halton_point_all_finite() {
        let n = 64_usize;
        let dim = 4;
        for scenario in 0..n {
            let spec = NoisePointSpec {
                sampling_seed: 42,
                iteration: 0,
                #[allow(clippy::cast_possible_truncation)]
                scenario: scenario as u32,
                stream_id: 1,
                #[allow(clippy::cast_possible_truncation)]
                total_scenarios: n as u32,
                dim,
            };
            let mut output = vec![0.0_f64; dim];
            scrambled_halton_point_reference(&spec, &mut output);
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }

    #[test]
    fn halton_point_matches_reference() {
        for (dim, total_scenarios) in [(1_usize, 4_u32), (2, 16), (5, 8)] {
            let ctx = HaltonPrecomputed::new(42, 1, 3, dim, total_scenarios);
            let mut precomputed_out = vec![0.0_f64; dim];
            let mut direct_out = vec![0.0_f64; dim];

            for scenario in 0..total_scenarios {
                let spec = NoisePointSpec {
                    sampling_seed: 42,
                    iteration: 1,
                    scenario,
                    stream_id: 3,
                    total_scenarios,
                    dim,
                };

                scrambled_halton_point(&spec, &ctx, &mut precomputed_out);
                scrambled_halton_point_reference(&spec, &mut direct_out);

                assert_eq!(
                    precomputed_out, direct_out,
                    "mismatch at dim={dim}, total_scenarios={total_scenarios}, scenario={scenario}"
                );
            }
        }
    }
}
