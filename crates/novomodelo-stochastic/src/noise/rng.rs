//! PRNG wrapper: initialises a `Pcg64` from a derived seed for sampling
//! standard-normal (`N(0,1)`) variates.

use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Initialize a `Pcg64` RNG from a derived 64-bit seed.
///
/// Equal seeds yield identical sequences.
///
/// # Resume invariant
///
/// Because per-draw seeds derive from the *absolute* iteration number (via
/// [`derive_forward_seed`](crate::noise::seed::derive_forward_seed), not a
/// from-zero counter), a resumed run at iteration K+1 reproduces the same noise
/// as a continuous run — so RNG state need not be serialized for resume.
///
/// # Examples
///
/// ```
/// use rand::RngExt;
/// use cobre_stochastic::noise::rng::rng_from_seed;
///
/// let mut rng1 = rng_from_seed(12345);
/// let mut rng2 = rng_from_seed(12345);
///
/// assert_eq!(rng1.random::<f64>(), rng2.random::<f64>());
/// ```
#[must_use]
pub fn rng_from_seed(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::rng_from_seed;

    #[test]
    #[allow(clippy::float_cmp)]
    fn rng_from_seed_is_deterministic() {
        let mut rng1 = rng_from_seed(12345);
        let mut rng2 = rng_from_seed(12345);
        // Bitwise equality is the contract: identical seeds reproduce the same
        // bit pattern, not merely an approximately-equal float result.
        for _ in 0..10 {
            assert_eq!(rng1.random::<f64>(), rng2.random::<f64>());
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rng_from_seed_differs_for_different_seeds() {
        let mut rng1 = rng_from_seed(0);
        let mut rng2 = rng_from_seed(1);
        assert_ne!(rng1.random::<f64>(), rng2.random::<f64>());
    }

    #[test]
    fn rng_from_seed_zero_is_valid() {
        let mut rng = rng_from_seed(0);
        let v: f64 = rng.random();
        assert!(v.is_finite());
    }

    #[test]
    fn rng_from_seed_max_u64_is_valid() {
        let mut rng = rng_from_seed(u64::MAX);
        let v: f64 = rng.random();
        assert!(v.is_finite());
    }
}
