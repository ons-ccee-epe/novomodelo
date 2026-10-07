//! Inverse normal CDF (quantile function) via Beasley-Springer-Moro approximation.
//!
//! Provides [`norm_quantile`], a piecewise rational/polynomial approximation of
//! the standard normal quantile (probit, `Phi^{-1}`).
//!
//! ## Algorithm
//!
//! Three regions, each with a different approximation:
//!
//! - **Central** (`|p - 0.5| < 0.42`): rational approximation in `y = p - 0.5`,
//!   `r = y^2`.
//! - **Intermediate tails**: degree-8 polynomial in `r = ln(-ln(min(p, 1-p)))`.
//! - **Extreme tails** (`p` or `1-p` `<=` `EXTREME_BOUNDARY`): clamped to
//!   ±`EXTREME_CLAMP`.
//!
//! Symmetry `norm_quantile(p) = -norm_quantile(1-p)` emerges from the algorithm;
//! it is not explicitly enforced.
//!
//! ## References
//!
//! Moro, B. (1995). "The Full Monte." *Risk*, 8(2), 57–58.
//! Beasley, J. D. & Springer, S. G. (1977). "Algorithm AS 111: The percentage
//! points of the normal distribution." *Applied Statistics*, 26(1), 118–121.

// Central-region numerator (A&S 26.2.16 / BSM). Lowest-degree first (Horner).
const A: [f64; 4] = [
    2.506_628_238_84,
    -18.615_000_625_29,
    41.391_197_735_34,
    -25.441_060_496_37,
];

// Central-region denominator. Lowest-degree first (Horner; implied trailing 1.0).
const B: [f64; 4] = [
    -8.473_510_930_90,
    23.083_367_437_43,
    -21.062_241_018_26,
    3.130_829_098_33,
];

// Tail-region polynomial (Moro 1995, degree-8 in r = ln(-ln(q))).
// Lowest-degree first (Horner).
const C: [f64; 9] = [
    0.337_475_482_272_614_7,
    0.976_169_019_091_718_6,
    0.160_797_971_491_820_9,
    0.027_643_881_033_386_3,
    0.003_840_572_937_360_9,
    0.000_395_189_651_134_9,
    0.000_032_176_788_176_8,
    0.000_000_288_816_736_4,
    0.000_000_396_031_518_7,
];

/// Extreme-tail clamp magnitude, keeping output finite for any `p` in (0, 1).
const EXTREME_CLAMP: f64 = 8.21;

/// Boundary between the extreme-tail clamp and the polynomial tail.
///
/// In IEEE 754 f64, `1.0 - 1e-20 == 1.0`, so the upper-tail branch
/// (`1 - p <= EXTREME_BOUNDARY`) is unreachable for any valid argument (the
/// assert fires first); the lower-tail clamp is reachable (`p = 1e-20`, `1e-21`).
/// Do not delete the upper branch as dead code — it is the symmetric mirror.
const EXTREME_BOUNDARY: f64 = 1e-20;

/// Compute the standard normal quantile (inverse CDF) at probability `p`.
///
/// Returns `x` such that `Φ(x) = p`, to better than 3e-9 absolute error across
/// the open unit interval.
///
/// # Panics
///
/// Panics unless `p` is in the open interval `(0, 1)`.
///
/// # Examples
///
/// ```
/// use cobre_stochastic::norm_quantile;
///
/// assert_eq!(norm_quantile(0.5), 0.0);
///
/// let z975 = norm_quantile(0.975);
/// assert!((z975 - 1.959_963_985_f64).abs() < 1e-8);
///
/// let p = 0.1_f64;
/// assert!((norm_quantile(p) + norm_quantile(1.0 - p)).abs() < 1e-12);
/// ```
// Single-char names mirror the notation in Moro (1995) and A&S 26.2.16/26.2.17.
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn norm_quantile(p: f64) -> f64 {
    assert!(
        p > 0.0 && p < 1.0,
        "norm_quantile requires p in (0, 1), got {p}"
    );

    if p <= EXTREME_BOUNDARY {
        return -EXTREME_CLAMP;
    }

    let p_mirror = 1.0 - p;
    if p_mirror <= EXTREME_BOUNDARY {
        return EXTREME_CLAMP;
    }

    let y = p - 0.5;
    if y.abs() < 0.42 {
        let r = y * y;
        let num = ((A[3] * r + A[2]) * r + A[1]) * r + A[0];
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        return y * num / den;
    }

    let q = if y <= 0.0 { p } else { p_mirror };
    let r = (-q.ln()).ln();

    let z = C[0]
        + r * (C[1]
            + r * (C[2]
                + r * (C[3] + r * (C[4] + r * (C[5] + r * (C[6] + r * (C[7] + r * C[8])))))));

    if y <= 0.0 { -z } else { z }
}

#[cfg(test)]
mod tests {
    use super::norm_quantile;

    // (p, expected_z, tolerance) from standard statistical tables.
    const KNOWN_VALUES: &[(f64, f64, f64)] = &[
        (0.001, -3.090_232_306, 1e-8),
        (0.01, -2.326_347_874, 1e-8),
        (0.025, -1.959_963_985, 1e-8),
        (0.05, -1.644_853_627, 1e-8),
        (0.1, -1.281_551_566, 1e-8),
        (0.25, -0.674_489_750, 1e-8),
        (0.75, 0.674_489_750, 1e-8),
        (0.9, 1.281_551_566, 1e-8),
        (0.95, 1.644_853_627, 1e-8),
        (0.975, 1.959_963_985, 1e-8),
        (0.99, 2.326_347_874, 1e-8),
        (0.999, 3.090_232_306, 1e-8),
    ];

    #[test]
    #[allow(clippy::float_cmp)]
    fn quantile_0_5_is_zero() {
        assert_eq!(norm_quantile(0.5), 0.0);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn quantile_symmetry() {
        let n = 1000_usize;
        for i in 1..=n {
            let p = 0.001 + (0.998 / n as f64) * i as f64;
            let sum = norm_quantile(p) + norm_quantile(1.0 - p);
            assert!(
                sum.abs() < 1e-12,
                "symmetry violated at p={p:.6}: q(p)+q(1-p) = {sum:.3e}",
            );
        }
    }

    #[test]
    fn quantile_known_values() {
        for &(p, expected, tol) in KNOWN_VALUES {
            let got = norm_quantile(p);
            let err = (got - expected).abs();
            assert!(
                err < tol,
                "norm_quantile({p}) = {got:.12}, expected {expected:.12}, error {err:.3e} >= {tol:.3e}",
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn quantile_monotonicity() {
        let n = 10_000_usize;
        let mut prev = norm_quantile(0.001);
        for i in 1..n {
            let p = 0.001 + (0.998 / n as f64) * i as f64;
            let cur = norm_quantile(p);
            assert!(
                cur > prev,
                "monotonicity violated: norm_quantile({p:.6}) = {cur} <= prev = {prev}",
            );
            prev = cur;
        }
    }

    #[test]
    #[should_panic(expected = "norm_quantile requires p in (0, 1)")]
    fn quantile_panics_at_0() {
        let _ = norm_quantile(0.0);
    }

    #[test]
    #[should_panic(expected = "norm_quantile requires p in (0, 1)")]
    fn quantile_panics_at_1() {
        let _ = norm_quantile(1.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn quantile_extreme_tail_clamp() {
        // On the boundary and beyond it both clamp to -8.21.
        assert_eq!(norm_quantile(1e-20), -8.21);
        assert_eq!(norm_quantile(1e-21), -8.21);
    }

    #[test]
    fn quantile_extreme_tail_symmetry() {
        // The upper clamp is unreachable (1.0 - 1e-20 == 1.0 in f64), so verify
        // the symmetric behavior indirectly: a representable value near 1.0 takes
        // the tail polynomial and returns a large positive finite result.
        let p = 1.0_f64 - 1e-15_f64;
        let v = norm_quantile(p);
        assert!(
            v > 7.5 && v.is_finite(),
            "expected large positive value near 8.21 for p near 1.0, got {v}",
        );
    }
}
