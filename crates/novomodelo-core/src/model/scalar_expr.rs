//! Scalar coefficient programs and the exact-zero rule.

/// Whether `v` is an exact zero: `+0.0` and `-0.0` both are, `NaN` never is.
#[must_use]
pub fn is_exact_zero(v: f64) -> bool {
    v == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_exact_zero_matches_both_signed_zeros_only() {
        assert!(is_exact_zero(0.0));
        assert!(is_exact_zero(-0.0));
        assert!(!is_exact_zero(f64::NAN));
        assert!(!is_exact_zero(f64::MIN_POSITIVE));
        assert!(!is_exact_zero(5e-324));
        assert!(!is_exact_zero(1.0));
    }
}
