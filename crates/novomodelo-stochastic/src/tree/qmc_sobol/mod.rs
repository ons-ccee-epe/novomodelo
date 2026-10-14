//! Sobol quasi-Monte Carlo sequence generation with Joe-Kuo direction tables.
//!
//! Embeds the Joe-Kuo direction-number dataset for up to `MAX_SOBOL_DIM`
//! dimensions. Dimension 1 uses the van der Corput sequence;
//! dimensions 2+ read `SOBOL_DIRECTIONS` (array index 0 is dimension 2).
//!
//! `SobolDirEntry::initial_dirs` holds raw, unshifted `m_i` values as listed in
//! the Joe-Kuo file; `build_direction_matrix` applies the left-shift — do not
//! pre-shift the stored values.
//!
//! [`generate_qmc_sobol`] walks the sequence by Gray-code recurrence (O(1) per
//! point); [`scrambled_sobol_point`] reaches one scenario by direct binary
//! decomposition. Both scramble via `scramble_to_normal`.

mod sobol_directions;

pub(crate) use sobol_directions::{SOBOL_DIRECTIONS, SOBOL_MAX_DIM};

use rand::RngExt;

use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_opening_seed, derive_stage_seed},
};
use crate::tree::NoisePointSpec;

/// Maximum supported Sobol dimension: dimension 1 (van der Corput) plus the
/// Joe-Kuo entries in `SOBOL_DIRECTIONS`.
pub(crate) const MAX_SOBOL_DIM: usize = SOBOL_MAX_DIM;

/// `1.0 / 2^32`, scaling a uniform u32 to a float in `[0, 1)`.
const INV_2_32: f64 = 1.0 / 4_294_967_296.0;

/// Applies Matousek linear scrambling to a raw Sobol coordinate, then maps the
/// result to N(0,1) via `norm_quantile`.
#[inline]
fn scramble_to_normal(x: u32, a: u32, b: u32) -> f64 {
    let xp = a.wrapping_mul(x).wrapping_add(b);
    let u = (f64::from(xp) * INV_2_32).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    norm_quantile(u)
}

/// Build the full 32-bit direction vectors for each of the `dim` dimensions.
///
/// Returns `result[d][j]`: the `j`-th direction number for dimension `d`, its
/// significant bit at `31 - j`. Dimension 1 is van der Corput; dimensions 2+ use
/// the Joe-Kuo polynomial recurrence.
///
/// # Panics
///
/// Panics if `dim > MAX_SOBOL_DIM`.
fn build_direction_matrix(dim: usize) -> Vec<[u32; 32]> {
    assert!(
        dim <= MAX_SOBOL_DIM,
        "dim {dim} exceeds MAX_SOBOL_DIM {MAX_SOBOL_DIM}"
    );

    let mut result: Vec<[u32; 32]> = Vec::with_capacity(dim);

    for d in 0..dim {
        let mut v = [0u32; 32];

        if d == 0 {
            for (j, slot) in v.iter_mut().enumerate() {
                *slot = 1u32 << (31 - j);
            }
        } else {
            let entry = &SOBOL_DIRECTIONS[d - 1];
            let s = entry.degree as usize;
            let a = entry.poly;

            // Raw m_j shifted into place: shift 31 - j (0-indexed j), i.e. the
            // m_j << (32 - (j+1)) convention.
            for (j, slot) in v[..s].iter_mut().enumerate() {
                *slot = entry.initial_dirs[j] << (31 - j);
            }

            // Joe-Kuo recurrence re-indexed: the source states it with 1-indexed j.
            for j in s..32 {
                let mut x = v[j - s] ^ (v[j - s] >> s);
                for k in 1..s {
                    if (a >> (s - 1 - k)) & 1 == 1 {
                        x ^= v[j - k];
                    }
                }
                v[j] = x;
            }
        }

        result.push(v);
    }

    result
}

/// Derive Matousek linear scrambling parameters `(a_d, b_d)` for each dimension,
/// with `a_d` forced odd to ensure bijection on `Z_{2^32}`.
fn derive_scramble_params(seed: u64, dim: usize) -> Vec<(u32, u32)> {
    let mut rng = rng_from_seed(seed);
    (0..dim)
        .map(|_| {
            let a: u32 = rng.random::<u32>() | 1;
            let b: u32 = rng.random();
            (a, b)
        })
        .collect()
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values using
/// Scrambled Sobol QMC with Gray-code recurrence for O(1) updates per point.
///
/// Output layout: opening-major `output[opening * dim + entity]`.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim` or `dim > MAX_SOBOL_DIM`.
pub fn generate_qmc_sobol(
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
    let directions = build_direction_matrix(dim);
    let scramble = derive_scramble_params(seed, dim);

    // Unscrambled coordinate per dimension; point 0 is the zero state.
    let mut sobol_state = vec![0u32; dim];

    for d in 0..dim {
        let (a, b) = scramble[d];
        output[d] = scramble_to_normal(sobol_state[d], a, b);
    }

    // Gray-code recurrence: XOR the direction at index `i.trailing_zeros()` into
    // each dimension's state.
    for i in 1..n_openings {
        #[allow(clippy::cast_possible_truncation)]
        let c = i.trailing_zeros() as usize;
        for d in 0..dim {
            sobol_state[d] ^= directions[d][c];
            let (a, b) = scramble[d];
            output[i * dim + d] = scramble_to_normal(sobol_state[d], a, b);
        }
    }
}

/// Direction matrix and scramble parameters built once per
/// (`sampling_seed`, `iteration`, `stream_id`, `dim`) tuple and reused across all
/// scenarios at that stage.
#[derive(Debug, Clone)]
pub struct SobolPrecomputed {
    directions: Vec<[u32; 32]>,
    scramble: Vec<(u32, u32)>,
}

impl SobolPrecomputed {
    /// Builds the direction matrix and scramble parameters described on
    /// [`SobolPrecomputed`].
    ///
    /// # Panics
    ///
    /// Panics if `dim > MAX_SOBOL_DIM`.
    #[must_use]
    pub fn new(sampling_seed: u64, iteration: u32, stream_id: u32, dim: usize) -> Self {
        let seed = derive_opening_seed(sampling_seed, iteration, stream_id);
        let directions = build_direction_matrix(dim);
        let scramble = derive_scramble_params(seed, dim);
        Self {
            directions,
            scramble,
        }
    }
}

/// Generate one scenario's noise vector from a precomputed direction matrix
/// and scramble parameters.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or `spec.dim > MAX_SOBOL_DIM`.
pub fn scrambled_sobol_point(spec: &NoisePointSpec, ctx: &SobolPrecomputed, output: &mut [f64]) {
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    for (d, out) in output.iter_mut().enumerate().take(spec.dim) {
        let mut xd = 0u32;
        let mut scenario = spec.scenario;
        let mut j = 0usize;
        while scenario != 0 {
            if scenario & 1 == 1 {
                xd ^= ctx.directions[d][j];
            }
            scenario >>= 1;
            j += 1;
        }

        let (a, b) = ctx.scramble[d];
        *out = scramble_to_normal(xd, a, b);
    }
}

/// Derives the direction matrix and scramble parameters per call; this is the
/// reference `scrambled_sobol_point` is tested against.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or `spec.dim > MAX_SOBOL_DIM`.
#[cfg(test)]
pub(crate) fn scrambled_sobol_point_reference(spec: &NoisePointSpec, output: &mut [f64]) {
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
    let directions = build_direction_matrix(spec.dim);
    let scramble = derive_scramble_params(seed, spec.dim);

    for d in 0..spec.dim {
        let mut xd = 0u32;
        let mut scenario = spec.scenario;
        let mut j = 0usize;
        while scenario != 0 {
            if scenario & 1 == 1 {
                xd ^= directions[d][j];
            }
            scenario >>= 1;
            j += 1;
        }

        let (a, b) = scramble[d];
        let xp = a.wrapping_mul(xd).wrapping_add(b);
        let u = (f64::from(xp) * INV_2_32).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        output[d] = norm_quantile(u);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::{
        INV_2_32, MAX_SOBOL_DIM, NoisePointSpec, SobolPrecomputed, build_direction_matrix,
        generate_qmc_sobol, scrambled_sobol_point, scrambled_sobol_point_reference,
    };

    /// Generate unscrambled Gray-code Sobol points in `[0,1)` for regression testing.
    fn generate_unscrambled_sobol(n_openings: usize, dim: usize) -> Vec<Vec<f64>> {
        let directions = build_direction_matrix(dim);
        let mut sobol_state = vec![0u32; dim];
        let mut result = Vec::with_capacity(n_openings);

        result.push((0..dim).map(|_| 0.0_f64).collect());

        for i in 1..n_openings {
            let c = i.trailing_zeros() as usize;
            for d in 0..dim {
                sobol_state[d] ^= directions[d][c];
            }
            let point: Vec<f64> = (0..dim)
                .map(|d| f64::from(sobol_state[d]) * INV_2_32)
                .collect();
            result.push(point);
        }

        result
    }

    #[test]
    fn test_sobol_batch_determinism() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out1);
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out2);
        assert_eq!(out1, out2, "generate_qmc_sobol is not deterministic");
    }

    #[test]
    fn test_sobol_batch_different_seeds_differ() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out1);
        generate_qmc_sobol(99, 0, n_openings, dim, &mut out2);
        assert_ne!(out1, out2, "different seeds produced identical output");
    }

    #[test]
    fn test_sobol_batch_all_finite() {
        let n_openings = 64;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    #[test]
    fn test_sobol_batch_correct_length() {
        let n_openings = 32;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(1, 2, n_openings, dim, &mut output);
        assert_eq!(output.len(), n_openings * dim);
    }

    #[test]
    fn test_sobol_point_determinism() {
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
        scrambled_sobol_point_reference(&spec, &mut out1);
        scrambled_sobol_point_reference(&spec, &mut out2);
        assert_eq!(out1, out2, "scrambled_sobol_point is not deterministic");
    }

    #[test]
    fn test_sobol_point_different_seeds_differ() {
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
        scrambled_sobol_point_reference(&base_spec, &mut out1);
        scrambled_sobol_point_reference(
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
    fn test_sobol_point_all_finite() {
        let n = 64_usize;
        let dim = 4;
        for scenario in 0..n {
            let spec = NoisePointSpec {
                sampling_seed: 42,
                iteration: 0,
                scenario: scenario as u32,
                stream_id: 1,
                total_scenarios: n as u32,
                dim,
            };
            let mut output = vec![0.0_f64; dim];
            scrambled_sobol_point_reference(&spec, &mut output);
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }

    #[test]
    fn sobol_point_matches_reference() {
        for (dim, total_scenarios) in [(1_usize, 4_u32), (2, 16), (5, 8)] {
            let ctx = SobolPrecomputed::new(42, 1, 3, dim);
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

                scrambled_sobol_point(&spec, &ctx, &mut precomputed_out);
                scrambled_sobol_point_reference(&spec, &mut direct_out);

                assert_eq!(
                    precomputed_out, direct_out,
                    "mismatch at dim={dim}, total_scenarios={total_scenarios}, scenario={scenario}"
                );
            }
        }
    }

    #[test]
    fn test_build_direction_matrix_dim1() {
        let dirs = build_direction_matrix(1);
        assert_eq!(dirs.len(), 1, "expected 1 direction vector");
        for (j, &v) in dirs[0].iter().enumerate() {
            let expected = 1u32 << (31 - j);
            assert_eq!(
                v, expected,
                "dim1 direction[{j}]: got {v:#010x}, expected {expected:#010x}"
            );
        }
    }

    /// The BSM approximation clamps at ±8.22.
    #[test]
    fn test_sobol_batch_values_in_range() {
        let n_openings = 64;
        let dim = 2;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}");
            assert!(
                v > -8.22 && v < 8.22,
                "value {v} out of range [-8.22, 8.22] at index {i}"
            );
        }
    }

    #[test]
    fn test_max_sobol_dim_constant() {
        assert_eq!(MAX_SOBOL_DIM, 21_201);
    }

    /// Dimension 1 yields the van der Corput base-2 values in Gray-code traversal
    /// order, not in van der Corput order.
    #[test]
    fn test_unscrambled_dim1_first_8_points() {
        let pts = generate_unscrambled_sobol(8, 1);
        // Gray-code Sobol sequence for dim 1 (van der Corput values, Gray-code order):
        // i=0: state=0                   → 0/2^32   = 0.0
        // i=1: c=0, state^=v[0]=2^31     → 2^31     = 0.5
        // i=2: c=1, state^=v[1]=2^30     → 3*2^30   = 0.75
        // i=3: c=0, state^=v[0]=2^31     → 2^30     = 0.25
        // i=4: c=2, state^=v[2]=2^29     → 3*2^29   = 0.375
        // i=5: c=0, state^=v[0]=2^31     → 7*2^29   = 0.875
        // i=6: c=1, state^=v[1]=2^30     → 5*2^29   = 0.625
        // i=7: c=0, state^=v[0]=2^31     → 1*2^29   = 0.125
        let expected = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
        for (i, (got, &exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got[0] - exp).abs() < 1e-12,
                "dim1 point[{i}]: got {}, expected {}",
                got[0],
                exp
            );
        }
    }

    /// Dimension 2 reads `SOBOL_DIRECTIONS[0]` (degree 1, poly 0).
    #[test]
    fn test_unscrambled_dim2_first_8_points() {
        let pts = generate_unscrambled_sobol(8, 2);
        // Gray-code Sobol sequence for dim 2 (Joe-Kuo degree=1, poly=0, m=[1,...]):
        // v[0]=0x80000000, v[1]=0xC0000000, v[2]=0xA0000000, ...
        // i=0: state=0                          → 0.0
        // i=1: c=0, state^=0x80000000           → 0.5
        // i=2: c=1, state^=0xC0000000           → 0x40000000/2^32 = 0.25
        // i=3: c=0, state^=0x80000000           → 0xC0000000/2^32 = 0.75
        // i=4: c=2, state^=0xA0000000           → 0x60000000/2^32 = 0.375
        // i=5: c=0, state^=0x80000000           → 0xE0000000/2^32 = 0.875
        // i=6: c=1, state^=0xC0000000           → 0x20000000/2^32 = 0.125
        // i=7: c=0, state^=0x80000000           → 0xA0000000/2^32 = 0.625
        let expected = [0.0, 0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625];
        for (i, (got, &exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got[1] - exp).abs() < 1e-12,
                "dim2 point[{i}]: got {}, expected {}",
                got[1],
                exp
            );
        }
    }
}
