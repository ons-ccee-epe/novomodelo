//! Deterministic seeded PRNG and identity hash for the sampled plane reduction.
//!
//! # Determinism
//!
//! The sampled-distance reduction draws random `(V, Q)` points; seeding from a
//! STABLE (plant / entry / slot) identity via [`fnv1a64`] + [`SplitMix64`] — and
//! NOT a `rand`/`thread_rng` sampler — is what keeps the fitted policy bit-
//! identical across runs and MPI rank counts (the declaration-order hard rule).
//! Neither symbol reads any clock, thread id, or rank, and both are hand-rolled so
//! the dependency surface carries no entropy source at all.

/// A tiny, fully deterministic, seedable PRNG: the `SplitMix64` seed-expander.
///
/// # Contract — verbatim algorithm/constants
///
/// The mixing constants (`0x9E37_79B9_7F4A_7C15`, `0xBF58_476D_1CE4_E5B9`,
/// `0x94D0_49BB_1331_11EB`) and the shift amounts are the reference `SplitMix64`
/// finalizer and MUST stay byte-for-byte identical to the seeded-PRNG template
/// used elsewhere in the crate. A cross-check test pins that the `next_u64`
/// stream matches that template for a shared seed: drifting a single constant
/// would silently change every sampled point and so every merge decision, while
/// still compiling and passing most tests.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the generator from a `u64`.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next pseudo-random `u64` in the stream.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f64` in `[0, 1)` from the top 53 bits of the next `u64`.
    ///
    /// # Contract — top 53 bits, scaled by `2⁻⁵³`
    ///
    /// `(next_u64() >> 11) as f64 * 2.0_f64.powi(-53)` takes the high 53 bits
    /// (the exact mantissa width of an `f64`) and scales them into `[0, 1)`. The
    /// `>> 11` discards the low 11 bits that cannot survive the round to `f64`;
    /// keeping them and casting the full `u64` would round to exactly `1.0` for
    /// the largest inputs, breaking the half-open `[0, 1)` range the sampler
    /// relies on to stay inside the fitting box.
    pub(crate) fn next_unit_f64(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa * 2.0_f64.powi(-53)
    }
}

/// FNV-1a 64-bit hash of a byte slice.
///
/// # Contract — fixed offset basis and prime
///
/// The offset basis `0xcbf2_9ce4_8422_2325` and prime `0x0000_0100_0000_01b3`
/// are the standard FNV-1a 64-bit constants. They define the function: changing
/// either, or swapping the XOR-then-multiply order to multiply-then-XOR (FNV-1
/// rather than FNV-1a), would change every derived seed and so every sampled
/// point sequence. The hash is used only to compress a stable identity tuple
/// into a `u64` seed — it carries no clock, thread, or rank input.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::{SplitMix64, fnv1a64};

    /// The reference `SplitMix64` finalizer, mirrored from the crate's seeded-PRNG
    /// template, used only to cross-check that the production copy did not drift
    /// from the template constants.
    fn template_next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `SplitMix64::next_u64` reproduces the template sequence for a shared seed:
    /// the production copy must match the canonical seeded-PRNG template
    /// byte-for-byte over a long-enough prefix to catch any constant drift.
    #[test]
    fn next_u64_matches_template_sequence() {
        const SEED: u64 = 0x1234_5678_9abc_def0;
        let mut rng = SplitMix64::new(SEED);
        let mut template_state = SEED;
        for i in 0..64 {
            let got = rng.next_u64();
            let want = template_next_u64(&mut template_state);
            assert_eq!(
                got, want,
                "production SplitMix64 diverged from the template at draw {i}"
            );
        }
    }

    /// `next_unit_f64` is always in `[0, 1)` across a long stream.
    #[test]
    fn next_unit_f64_is_in_half_open_unit_interval() {
        let mut rng = SplitMix64::new(0xdead_beef_cafe_f00d);
        for _ in 0..10_000 {
            let u = rng.next_unit_f64();
            assert!(u >= 0.0, "next_unit_f64 must be >= 0.0, got {u}");
            assert!(u < 1.0, "next_unit_f64 must be < 1.0, got {u}");
        }
    }

    /// `SplitMix64` is reproducible: two generators from the same seed emit the
    /// identical `u64` stream.
    #[test]
    fn same_seed_yields_same_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    /// `fnv1a64` is a fixed-point function of its bytes: equal input ⇒ equal hash,
    /// and a one-byte change perturbs the hash.
    #[test]
    fn fnv1a64_is_deterministic_and_sensitive() {
        let a = fnv1a64(&[1, 2, 3, 4]);
        let b = fnv1a64(&[1, 2, 3, 4]);
        assert_eq!(a, b, "fnv1a64 must be deterministic");

        let c = fnv1a64(&[1, 2, 3, 5]);
        assert_ne!(a, c, "a one-byte change must perturb the hash");
    }

    /// `fnv1a64` matches its standard first-step value: the empty input hashes to
    /// the offset basis, and a single zero byte advances it by one prime
    /// multiply, pinning the constants against the FNV-1a reference.
    #[test]
    fn fnv1a64_matches_reference_offset_basis() {
        assert_eq!(
            fnv1a64(&[]),
            0xcbf2_9ce4_8422_2325_u64,
            "empty input must hash to the FNV-1a offset basis"
        );
        let expected_zero = 0xcbf2_9ce4_8422_2325_u64.wrapping_mul(0x0000_0100_0000_01b3);
        assert_eq!(
            fnv1a64(&[0]),
            expected_zero,
            "a single zero byte must apply exactly one prime multiply"
        );
    }
}
