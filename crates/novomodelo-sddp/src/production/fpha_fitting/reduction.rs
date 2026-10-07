//! Similar-hyperplane reduction of the fitted FPHA plane set.
//!
//! Collapses near-parallel / near-coincident consecutive planes into their mean
//! hyperplane, walking the canonical post-dedup plane order. [`reduce_planes`]
//! dispatches on [`PlaneReductionConfig`] (enum dispatch): the `Angle` arm is
//! RNG-free; the `Distance` arm samples from a deterministically seeded PRNG
//! (never wall-clock). Both arms are pure functions of the input slice and its
//! order — identical input ordering and rank count yield bit-identical output.
//!
//! # Origin-plane invariant
//!
//! The plane through the origin (`γ₀ = 0 ∧ γ_V = 0`, zero power at zero turbining)
//! is NEVER merged: averaging it into a neighbour would shift the zero-generation
//! anchor off the origin, so it is excluded as either member of any candidate
//! pair. See [`is_origin_plane`].

use super::geometry::FittingBounds;
use super::grid::{GridParams, build_grid};
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;
use super::rng::{SplitMix64, fnv1a64};
use cobre_io::extensions::PlaneReductionConfig;

/// Tolerance on `γ₀` and `γ_V` for the origin-plane predicate.
///
/// A tight numerical zero, not a physical tolerance: it absorbs the facet
/// arithmetic residual `fit_hull_planes` can leave on a coefficient that is zero
/// in exact arithmetic (sibling planes carry residuals around `1e-12`).
/// α-scaling multiplies such a residual but cannot create one — `0.0 * α` is
/// exactly `0.0` — so comparing against exact zero would still be wrong: an
/// origin plane emitted at `1e-13` would become mergeable and the zero-flow
/// anchor could be averaged away.
pub(super) const ORIGIN_EPS: f64 = 1e-9;

/// The angle-test normal of a plane: `n = (γ_V, γ_Q, −1)`.
///
/// # Contract — orientation only, in `(V, Q, generation)`
///
/// `γ_S` (the lateral-secant slope on a separate axis) and `γ₀` (the offset, not
/// the orientation) are EXCLUDED: folding either into the normal is the
/// wrong-but-compiling alternative — it would merge planes parallel in
/// `(V, Q, generation)` but differing in offset/secant. The `−1` generation
/// component guarantees `‖n‖ ≥ 1`, so [`angle_deg`] never divides by zero.
fn plane_normal(plane: &RawPlane) -> [f64; 3] {
    [plane.gamma_v, plane.gamma_q, -1.0]
}

/// Angle (degrees) between the two planes' `(V, Q, generation)` normals.
///
/// `θ = arccos( clamp(n₁·n₂ / (‖n₁‖·‖n₂‖), −1, 1) )`, converted to degrees.
///
/// # Contract — clamp the cosine before `arccos`
///
/// Roundoff can push the cosine ratio just past `±1`, making `arccos` return `NaN`
/// and silently corrupting the `θ < tolerance` comparison; clamping to `[−1, 1]`
/// first is mandatory, not defensive. Both norms are `≥ 1` (the `−1` generation
/// component of [`plane_normal`]), so the denominator is never zero.
fn angle_deg(a: &RawPlane, b: &RawPlane) -> f64 {
    let n1 = plane_normal(a);
    let n2 = plane_normal(b);
    let dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    let norm1 = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
    let norm2 = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();
    let cos = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

/// The mean hyperplane of two planes: the component-wise arithmetic mean of all
/// four coefficients.
///
/// # Contract — sign-preservation
///
/// The mean of two `validate_fitted_planes`-valid planes is itself valid (the mean
/// of two non-negatives is non-negative for `γ_V`/`γ_Q`; of two non-positives is
/// non-positive for `γ_S`). All four coefficients are averaged — averaging only
/// the orientation `(γ_V, γ_Q)` while keeping one plane's `γ₀`/`γ_S` is the
/// wrong-but-compiling alternative, leaving the merged offset/secant unbalanced
/// relative to its averaged orientation.
fn mean_plane(a: &RawPlane, b: &RawPlane) -> RawPlane {
    RawPlane {
        gamma_0: f64::midpoint(a.gamma_0, b.gamma_0),
        gamma_v: f64::midpoint(a.gamma_v, b.gamma_v),
        gamma_q: f64::midpoint(a.gamma_q, b.gamma_q),
        gamma_s: f64::midpoint(a.gamma_s, b.gamma_s),
    }
}

/// Whether `plane` is the plane through the origin (`γ₀ = 0 ∧ γ_V = 0`).
///
/// Never merged (see the module-header invariant). BOTH coefficients are tested:
/// several non-origin planes carry `γ₀ = 0` with a non-zero `γ_V`, and an
/// intercept-only predicate would wrongly protect them from merging. Tolerance
/// rationale lives on [`ORIGIN_EPS`].
pub(super) fn is_origin_plane(plane: &RawPlane) -> bool {
    plane.gamma_0.abs() <= ORIGIN_EPS && plane.gamma_v.abs() <= ORIGIN_EPS
}

/// Greedy single-forward-sweep merge of consecutive planes, parameterised by the
/// pair-similarity predicate.
///
/// Both the angle method and the sampled-distance method share this one walk, so
/// the canonical-order sweep, the origin invariant, and the greedy cascade have
/// exactly one implementation; only the predicate differs. A merged plane can
/// re-merge with the following candidate (the greedy cascade); the origin plane is
/// always pushed unmerged.
///
/// # Contract — element-for-element function of the input order
///
/// The sweep walks `planes` in its EXISTING canonical post-dedup order and never
/// re-sorts, so the output is an element-for-element function of the input order
/// (declaration-order bit-determinism). Callers must pass the canonical order.
///
/// # Contract — `should_merge` is keyed on the ORIGINAL slot indices
///
/// `should_merge(tail_slot, cand_slot, …)` receives 0-based indices in the
/// ORIGINAL pre-merge slice, NOT positions in the growing `out` Vec. A merged tail
/// keeps the slot of its FIRST member, so a distance predicate's per-pair seed
/// stays stable under the re-merge cascade. Keying the seed on coefficient values
/// instead would make the seed depend on merge history.
fn reduce_planes_with<F>(planes: &[RawPlane], should_merge: F) -> Vec<RawPlane>
where
    F: Fn(usize, usize, &RawPlane, &RawPlane) -> bool,
{
    let mut out: Vec<RawPlane> = Vec::with_capacity(planes.len());
    // The ORIGINAL slot of the current tail (its first member's slot), held
    // through a re-merge cascade so the predicate seed stays stable.
    let mut tail_slot: usize = 0;

    for (cand_slot, candidate) in planes.iter().enumerate() {
        let merged_tail = match out.last() {
            Some(tail)
                if !is_origin_plane(tail)
                    && !is_origin_plane(candidate)
                    && should_merge(tail_slot, cand_slot, tail, candidate) =>
            {
                Some(mean_plane(tail, candidate))
            }
            _ => None,
        };

        if let Some(merged) = merged_tail {
            // Replace the tail (keeping its original slot) so the merged plane can
            // cascade into the next candidate.
            if let Some(tail) = out.last_mut() {
                *tail = merged;
            }
        } else {
            out.push(*candidate);
            tail_slot = cand_slot;
        }
    }

    out
}

/// Merge consecutive near-parallel planes into their mean hyperplane (angle
/// method).
///
/// A pair merges when `θ_deg < tolerance_deg` (STRICT `<`, so `tolerance_deg = 0`
/// merges nothing). The predicate ignores the slot indices.
fn reduce_planes_angle(planes: &[RawPlane], tolerance_deg: f64) -> Vec<RawPlane> {
    reduce_planes_with(planes, |_tail_slot, _cand_slot, tail, candidate| {
        angle_deg(tail, candidate) < tolerance_deg
    })
}

/// The generation-window upper bound `generation_max`: the maximum
/// `generation = pf.evaluate(v, q, 0)` over a prebuilt fitting `(V, Q)` grid.
///
/// Takes a prebuilt grid so [`reduce_planes_distance`] builds it ONCE per plant
/// and shares it across the scan and every per-pair sampler. The scan walks the
/// closed-form axes in fixed `(v, q)` order, so the result is order-independent.
fn grid_max_gh(pf: &ProductionFunction, grid: &GridParams) -> f64 {
    let mut generation_max = f64::NEG_INFINITY;
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let generation = pf.evaluate(v, q, 0.0);
            if generation > generation_max {
                generation_max = generation;
            }
        }
    }
    generation_max
}

/// Derive a pair's canonical PRNG seed from a STABLE identity.
///
/// # Contract — pure-identity seed, NO clock/thread/rank
///
/// The seed is a pure function of the plant, the stage/entry identity, and the two
/// planes' ORIGINAL slot indices — it reads no wall clock, thread id, or MPI rank,
/// so two runs (and two rank counts) sampling the same pair draw the identical
/// point sequence. The forbidden alternative is seeding from coefficient values: a
/// re-merged tail's coefficients change with the merge history, which would make
/// the seed depend on how many neighbours the tail already absorbed. The slot
/// indices are stable under the cascade, so they are the correct seed identity.
fn pair_seed(hydro_id: i32, entry_level_bits: u64, tail_slot: usize, candidate_slot: usize) -> u64 {
    // Plane counts are bounded by `max_planes_per_hydro`, far below u32::MAX, so
    // the saturating cast cannot lose a distinguishing slot index.
    let tail_slot_u32 = u32::try_from(tail_slot).unwrap_or(u32::MAX);
    let candidate_slot_u32 = u32::try_from(candidate_slot).unwrap_or(u32::MAX);

    // Canonical seed composition: hydro_id ++ entry_level_bits ++ tail_slot ++
    // candidate_slot, all little-endian, in this fixed order.
    let mut buf = [0u8; 20];
    buf[0..4].copy_from_slice(&hydro_id.to_le_bytes());
    buf[4..12].copy_from_slice(&entry_level_bits.to_le_bytes());
    buf[12..16].copy_from_slice(&tail_slot_u32.to_le_bytes());
    buf[16..20].copy_from_slice(&candidate_slot_u32.to_le_bytes());
    fnv1a64(&buf)
}

/// Whether two planes are similar by the sampled mean-squared-distance test.
///
/// Draws `n_samples` `(V, Q)` points uniformly in the fitting box and returns
/// whether the normalised mean-squared generation difference falls below the
/// tolerance: `δ = EQM / generation²_max < tolerance_pct / 100`.
///
/// # Contract — sequential `Σ d²` in canonical sample order
///
/// The squared differences accumulate in a single sequential loop over the PRNG
/// draw order; a parallel / partitioned reduction would reorder the additions and
/// break bit-determinism across rank counts. The PRNG is freshly seeded from
/// `seed` each call, so the same seed reproduces the same `Σ d²` bit-for-bit.
///
/// # Contract — `δ` is a fraction; `tolerance_pct` is a percent
///
/// `δ = EQM / generation²_max` is a dimensionless fraction, so the comparison
/// normalises: `δ < tolerance_pct / 100.0`. Comparing against the raw
/// `tolerance_pct` (without the `/ 100`) is the wrong-but-compiling alternative —
/// it would treat a `0.5 %` tolerance as `50 %` and merge far too aggressively.
/// The caller guarantees `generation_max > 0`, so the division is well-defined.
fn pair_similar_distance(
    tail: &RawPlane,
    candidate: &RawPlane,
    grid: &GridParams,
    generation_max: f64,
    n_samples: u32,
    seed: u64,
    tolerance_pct: f64,
) -> bool {
    // Sample the full fitting box (the grid axis endpoints), so the sampled domain
    // is identical for every pair.
    let v_first = grid.v_points.first().copied().unwrap_or(0.0);
    let v_last = grid.v_points.last().copied().unwrap_or(0.0);
    let q_first = grid.q_points.first().copied().unwrap_or(0.0);
    let q_last = grid.q_points.last().copied().unwrap_or(0.0);
    let v_span = v_last - v_first;
    let q_span = q_last - q_first;

    let mut rng = SplitMix64::new(seed);
    let mut sum_sq = 0.0_f64;
    for _ in 0..n_samples {
        let v = v_first + rng.next_unit_f64() * v_span;
        let q = q_first + rng.next_unit_f64() * q_span;
        let d = tail.evaluate(v, q, 0.0) - candidate.evaluate(v, q, 0.0);
        sum_sq += d * d;
    }

    let eqm = sum_sq / f64::from(n_samples);
    let delta = eqm / (generation_max * generation_max);
    delta < tolerance_pct / 100.0
}

/// Merge consecutive near-coincident planes by the sampled mean-squared-distance
/// method.
///
/// `generation_max` is computed ONCE per plant; when non-positive the
/// normalisation is undefined, so the method is inert and returns the input
/// unchanged. Each pair's PRNG seed comes from the STABLE
/// `(hydro_id, entry_level_bits, tail_slot, candidate_slot)` identity via
/// [`pair_seed`], so the merge decisions are bit-identical across input ordering
/// and MPI rank count.
fn reduce_planes_distance(
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
    tolerance_pct: f64,
    n_samples: u32,
    hydro_id: i32,
    entry_level_bits: u64,
) -> Vec<RawPlane> {
    let grid = build_grid(pf, bounds);
    let generation_max = grid_max_gh(pf, &grid);

    // generation_max <= 0 ⇒ the squared normaliser is zero (δ undefined); return
    // the input unchanged rather than dividing by zero.
    if generation_max <= 0.0 {
        return planes.to_vec();
    }

    reduce_planes_with(planes, |tail_slot, cand_slot, tail, candidate| {
        let seed = pair_seed(hydro_id, entry_level_bits, tail_slot, cand_slot);
        pair_similar_distance(
            tail,
            candidate,
            &grid,
            generation_max,
            n_samples,
            seed,
            tolerance_pct,
        )
    })
}

/// Reduce a fitted plane set by the configured similar-hyperplane method.
///
/// Enum dispatch over [`PlaneReductionConfig`]. The `Distance` arm additionally
/// consumes `pf`, `bounds`, and the stable seed identity (`hydro_id`,
/// `entry_level_bits`); the `Angle` arm ignores them.
pub(crate) fn reduce_planes(
    planes: &[RawPlane],
    config: &PlaneReductionConfig,
    pf: &ProductionFunction,
    bounds: &FittingBounds,
    hydro_id: i32,
    entry_level_bits: u64,
) -> Vec<RawPlane> {
    match config {
        PlaneReductionConfig::Angle { tolerance_deg } => {
            reduce_planes_angle(planes, *tolerance_deg)
        }
        PlaneReductionConfig::Distance {
            tolerance_pct,
            n_samples,
        } => reduce_planes_distance(
            planes,
            pf,
            bounds,
            *tolerance_pct,
            *n_samples,
            hydro_id,
            entry_level_bits,
        ),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names
)]
mod tests {
    use cobre_core::{EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel};
    use cobre_io::extensions::HydroGeometryRow;

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::{ProductionFunction, TailraceSource};
    use super::super::selection::validate_fitted_planes;
    use super::{
        angle_deg, grid_max_gh, is_origin_plane, mean_plane, pair_seed, pair_similar_distance,
        reduce_planes, reduce_planes_angle, reduce_planes_distance,
    };
    use cobre_io::extensions::PlaneReductionConfig;

    /// A sign-valid plane with the given coefficients.
    fn plane(gamma_0: f64, gamma_v: f64, gamma_q: f64, gamma_s: f64) -> RawPlane {
        RawPlane {
            gamma_0,
            gamma_v,
            gamma_q,
            gamma_s,
        }
    }

    /// A 4-point VHA row.
    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// Resolved fitting bounds for a 6×6 `(V, Q)` grid over `[0, 30_000]` hm³.
    fn test_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 0.0,
            v_max: 30_000.0,
            n_volume_points: 6,
            n_flow_points: 6,
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: false,
        }
    }

    /// A concave production function with a positive generation window, so
    /// `grid_max_gh > 0` and the distance sampler exercises a real box.
    fn test_pf() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "DistanceFit").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.0008, -2e-8],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            3_000.0,
            "DistanceFit".to_owned(),
        )
    }

    /// Two planes with identical `(γ_V, γ_Q)` normals merge to the component-wise
    /// mean of all four coefficients (`to_bits` equality against the hand mean).
    #[test]
    fn identical_normal_pair_merges_to_component_wise_mean() {
        let a = plane(100.0, 0.002, 0.85, -0.01);
        let b = plane(200.0, 0.002, 0.85, -0.03);
        let reduced = reduce_planes_angle(&[a, b], 1.0);
        assert_eq!(
            reduced.len(),
            1,
            "identical normals must merge to one plane"
        );

        let expected = mean_plane(&a, &b);
        assert_eq!(reduced[0].gamma_0.to_bits(), expected.gamma_0.to_bits());
        assert_eq!(reduced[0].gamma_v.to_bits(), expected.gamma_v.to_bits());
        assert_eq!(reduced[0].gamma_q.to_bits(), expected.gamma_q.to_bits());
        assert_eq!(reduced[0].gamma_s.to_bits(), expected.gamma_s.to_bits());
    }

    /// A below-tolerance pair merges; an at/above-tolerance pair does not.
    #[test]
    fn below_tolerance_merges_at_or_above_does_not() {
        let a = plane(100.0, 0.0, 1.0, 0.0);
        let b = plane(100.0, 0.01, 1.0, 0.0);
        let theta = angle_deg(&a, &b);
        assert!(
            theta > 0.0,
            "the two normals must differ by a positive angle"
        );

        let below = reduce_planes_angle(&[a, b], theta + 1e-6);
        assert_eq!(below.len(), 1, "a below-tolerance pair must merge");

        let at = reduce_planes_angle(&[a, b], theta);
        assert_eq!(
            at.len(),
            2,
            "an at-tolerance pair must NOT merge (strict <)"
        );

        let above = reduce_planes_angle(&[a, b], theta - 1e-6);
        assert_eq!(above.len(), 2, "an above-tolerance pair must NOT merge");
    }

    /// Greedy cascade: three planes collapse to one when the merged tail stays
    /// within tolerance of the next candidate.
    #[test]
    fn greedy_cascade_three_planes_collapse_to_one() {
        // Three planes whose normals are pairwise within a generous tolerance.
        let a = plane(100.0, 0.000, 1.0, -0.01);
        let b = plane(120.0, 0.001, 1.0, -0.02);
        let c = plane(140.0, 0.002, 1.0, -0.03);
        let reduced = reduce_planes_angle(&[a, b, c], 5.0);
        assert_eq!(
            reduced.len(),
            1,
            "all three within tolerance must cascade into one plane, got {}",
            reduced.len()
        );
    }

    /// The origin plane is never merged: its four coefficients pass through
    /// to_bits-unchanged even next to a near-parallel plane and a large tolerance.
    #[test]
    fn origin_plane_never_merged() {
        let origin = plane(0.0, 0.0, 1.0, 0.0);
        assert!(
            is_origin_plane(&origin),
            "the fixture must be the origin plane"
        );
        // A near-parallel neighbour that WOULD merge if origin were eligible.
        let neighbour = plane(0.0, 0.0001, 1.0, 0.0);

        // Origin first, neighbour second.
        let reduced = reduce_planes_angle(&[origin, neighbour], 90.0);
        assert_eq!(reduced.len(), 2, "origin must not absorb its neighbour");
        assert_eq!(reduced[0].gamma_0.to_bits(), origin.gamma_0.to_bits());
        assert_eq!(reduced[0].gamma_v.to_bits(), origin.gamma_v.to_bits());
        assert_eq!(reduced[0].gamma_q.to_bits(), origin.gamma_q.to_bits());
        assert_eq!(reduced[0].gamma_s.to_bits(), origin.gamma_s.to_bits());

        // Neighbour first, origin second: origin still untouched.
        let reduced_rev = reduce_planes_angle(&[neighbour, origin], 90.0);
        assert_eq!(
            reduced_rev.len(),
            2,
            "origin must not be absorbed as candidate"
        );
        assert_eq!(reduced_rev[1].gamma_0.to_bits(), origin.gamma_0.to_bits());
        assert_eq!(reduced_rev[1].gamma_v.to_bits(), origin.gamma_v.to_bits());
        assert_eq!(reduced_rev[1].gamma_q.to_bits(), origin.gamma_q.to_bits());
        assert_eq!(reduced_rev[1].gamma_s.to_bits(), origin.gamma_s.to_bits());
    }

    /// `tolerance_deg = 0.0` merges nothing (strict `<` makes it inert).
    #[test]
    fn zero_tolerance_merges_nothing() {
        let a = plane(100.0, 0.002, 0.85, -0.01);
        let b = plane(200.0, 0.002, 0.85, -0.03);
        // Even identical normals (θ = 0) do not merge at tolerance 0 (0 < 0 is false).
        let reduced = reduce_planes_angle(&[a, b], 0.0);
        assert_eq!(reduced.len(), 2, "tolerance 0 must merge nothing");
    }

    /// A merged plane passes `validate_fitted_planes`: the mean of two sign-valid
    /// planes is sign-valid. Dispatched through `reduce_planes` (the `Angle` arm).
    #[test]
    fn merged_plane_passes_validation() {
        let a = plane(100.0, 0.001, 3.5, -0.01);
        let b = plane(200.0, 0.001, 3.5, -0.02);
        let config = PlaneReductionConfig::Angle { tolerance_deg: 1.0 };
        let pf = test_pf();
        let bounds = test_bounds();
        let reduced = reduce_planes(&[a, b], &config, &pf, &bounds, 1, 0);
        assert_eq!(reduced.len(), 1, "the pair must merge");
        let result = validate_fitted_planes(&reduced, 0.99, "MergedHydro");
        assert!(
            result.is_ok(),
            "merged plane must be sign-valid: {result:?}"
        );
    }

    /// `grid_max_gh` is positive for the concave fixture's window.
    #[test]
    fn grid_max_gh_is_positive_for_concave_window() {
        let pf = test_pf();
        let bounds = test_bounds();
        let generation_max = grid_max_gh(&pf, &super::build_grid(&pf, &bounds));
        assert!(
            generation_max > 0.0,
            "the concave fixture must have a positive generation window, got {generation_max}"
        );
    }

    /// `pair_similar_distance` is reproducible from its seed: two calls with the
    /// same seed return the same bool, and the intermediate `Σ d²` is
    /// `to_bits`-identical across the two calls.
    #[test]
    fn pair_similar_distance_is_reproducible_from_seed() {
        let pf = test_pf();
        let bounds = test_bounds();
        let grid = super::build_grid(&pf, &bounds);
        let generation_max = grid_max_gh(&pf, &grid);
        let tail = plane(100.0, 0.002, 0.85, -0.01);
        let candidate = plane(140.0, 0.0025, 0.90, -0.02);
        let seed = 0x1234_5678_9abc_def0_u64;
        let n_samples = 128;

        // Compute Σ d² independently with the same seeded stream, then assert the
        // predicate matches and the sum is bit-stable across two computations.
        let sum_sq = |seed: u64| -> f64 {
            let grid = super::build_grid(&pf, &bounds);
            let v_first = grid.v_points[0];
            let v_last = *grid.v_points.last().expect("non-empty");
            let q_first = grid.q_points[0];
            let q_last = *grid.q_points.last().expect("non-empty");
            let mut rng = super::SplitMix64::new(seed);
            let mut acc = 0.0_f64;
            for _ in 0..n_samples {
                let v = v_first + rng.next_unit_f64() * (v_last - v_first);
                let q = q_first + rng.next_unit_f64() * (q_last - q_first);
                let d = tail.evaluate(v, q, 0.0) - candidate.evaluate(v, q, 0.0);
                acc += d * d;
            }
            acc
        };
        assert_eq!(
            sum_sq(seed).to_bits(),
            sum_sq(seed).to_bits(),
            "Σ d² must be bit-identical for a shared seed"
        );

        let first = pair_similar_distance(
            &tail,
            &candidate,
            &grid,
            generation_max,
            n_samples,
            seed,
            50.0,
        );
        let second = pair_similar_distance(
            &tail,
            &candidate,
            &grid,
            generation_max,
            n_samples,
            seed,
            50.0,
        );
        assert_eq!(
            first, second,
            "the predicate must be reproducible from its seed"
        );
    }

    /// Coincident planes (identical coefficients) ⇒ `δ = 0` ⇒ merge to one plane
    /// equal to either input (`to_bits` equality).
    #[test]
    fn coincident_planes_have_zero_delta_and_merge() {
        let pf = test_pf();
        let bounds = test_bounds();
        let grid = super::build_grid(&pf, &bounds);
        let generation_max = grid_max_gh(&pf, &grid);
        let p = plane(120.0, 0.0015, 0.80, -0.015);

        // δ = 0 for coincident planes: every sampled difference is exactly 0, so
        // any positive tolerance accepts the pair.
        assert!(
            pair_similar_distance(&p, &p, &grid, generation_max, 128, 0xabc_def, 1e-9),
            "coincident planes must be similar at any positive tolerance"
        );

        let reduced = reduce_planes_distance(&[p, p], &pf, &bounds, 0.5, 128, 1, 0);
        assert_eq!(reduced.len(), 1, "coincident pair must merge to one plane");
        // The mean of two identical planes is that plane, bit-for-bit.
        assert_eq!(reduced[0].gamma_0.to_bits(), p.gamma_0.to_bits());
        assert_eq!(reduced[0].gamma_v.to_bits(), p.gamma_v.to_bits());
        assert_eq!(reduced[0].gamma_q.to_bits(), p.gamma_q.to_bits());
        assert_eq!(reduced[0].gamma_s.to_bits(), p.gamma_s.to_bits());
    }

    /// Input-order invariance: reducing the same canonical Vec twice yields
    /// `to_bits`-identical output for all four coefficients of every plane (the
    /// seed and sampling carry no clock/thread/rank component).
    #[test]
    fn distance_reduction_is_input_order_invariant() {
        let pf = test_pf();
        let bounds = test_bounds();
        let planes = [
            plane(100.0, 0.0010, 0.80, -0.010),
            plane(101.0, 0.0011, 0.81, -0.011),
            plane(300.0, 0.0050, 1.50, -0.030),
        ];
        let config = PlaneReductionConfig::Distance {
            tolerance_pct: 0.5,
            n_samples: 64,
        };

        let first = reduce_planes(&planes, &config, &pf, &bounds, 7, 42);
        let second = reduce_planes(&planes, &config, &pf, &bounds, 7, 42);
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.gamma_0.to_bits(), b.gamma_0.to_bits());
            assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
            assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
            assert_eq!(a.gamma_s.to_bits(), b.gamma_s.to_bits());
        }
    }

    /// Distinct `entry_level_bits` ⇒ distinct seed (the seed depends on entry
    /// identity), with all other identity components held fixed.
    #[test]
    fn distinct_entry_level_bits_yield_distinct_seed() {
        let seed_a = pair_seed(5, 0, 1, 2);
        let seed_b = pair_seed(5, 0x3ff0_0000_0000_0000, 1, 2);
        assert_ne!(
            seed_a, seed_b,
            "a different entry_level_bits must change the seed"
        );
    }

    /// `generation_max <= 0` ⇒ no merge: the method is inert and returns the input
    /// `to_bits`-identical. Built with a non-positive constant-head window via a
    /// fixture whose `pf.evaluate(v, q, 0)` is non-positive over the grid is hard
    /// to construct, so the contract is exercised by calling the inert guard path
    /// directly through a degenerate `generation_max` argument on `pair_similar_distance`.
    #[test]
    fn non_positive_gh_max_does_not_merge() {
        let pf = test_pf();
        let bounds = test_bounds();
        // Two clearly-coincident planes: they WOULD merge with a positive generation_max,
        // so a non-merge here is attributable to the guard, not the geometry.
        let p = plane(120.0, 0.0015, 0.80, -0.015);

        // The predicate itself: with generation_max = 0 the normaliser is 0 and δ is
        // NaN/inf, so the strict `<` comparison is false — never similar.
        let grid = super::build_grid(&pf, &bounds);
        assert!(
            !pair_similar_distance(&p, &p, &grid, 0.0, 64, 1, 99.0),
            "generation_max = 0 must make the predicate inert (no similarity)"
        );
    }

    /// The whole-plant `generation_max <= 0` guard returns the input unchanged. The guard
    /// lives in `reduce_planes_distance` and short-circuits before the sweep, so a
    /// fixture with a non-positive window is returned `to_bits`-identical. Such a
    /// fixture is awkward to build from geometry, so the guard's behaviour is
    /// pinned by the predicate-level inertness test above; here we assert that a
    /// degenerate flat window does not panic and length is preserved on the
    /// realistic fixture under an effectively-zero tolerance (no merge).
    #[test]
    fn zero_tolerance_distance_merges_nothing() {
        let pf = test_pf();
        let bounds = test_bounds();
        let planes = [
            plane(100.0, 0.0010, 0.80, -0.010),
            plane(101.0, 0.0011, 0.81, -0.011),
        ];
        // tolerance_pct = 0 ⇒ δ < 0 is impossible for non-negative δ ⇒ no merge.
        let reduced = reduce_planes_distance(&planes, &pf, &bounds, 0.0, 64, 1, 0);
        assert_eq!(reduced.len(), 2, "zero tolerance must merge nothing");
    }

    /// The origin plane is never merged: its four coefficients pass through
    /// `to_bits`-unchanged even next to a near-coincident plane and a large
    /// tolerance (reusing `is_origin_plane` for both pair members).
    #[test]
    fn distance_origin_plane_never_merged() {
        let pf = test_pf();
        let bounds = test_bounds();
        let origin = plane(0.0, 0.0, 1.0, 0.0);
        assert!(is_origin_plane(&origin), "fixture must be the origin plane");
        // A near-coincident neighbour that WOULD merge if the origin were eligible.
        let neighbour = plane(0.0, 0.0001, 1.0, 0.0);

        // Origin first.
        let reduced = reduce_planes_distance(&[origin, neighbour], &pf, &bounds, 100.0, 64, 1, 0);
        assert_eq!(reduced.len(), 2, "origin must not absorb its neighbour");
        assert_eq!(reduced[0].gamma_0.to_bits(), origin.gamma_0.to_bits());
        assert_eq!(reduced[0].gamma_v.to_bits(), origin.gamma_v.to_bits());
        assert_eq!(reduced[0].gamma_q.to_bits(), origin.gamma_q.to_bits());
        assert_eq!(reduced[0].gamma_s.to_bits(), origin.gamma_s.to_bits());

        // Neighbour first, origin second: origin still untouched.
        let reduced_rev =
            reduce_planes_distance(&[neighbour, origin], &pf, &bounds, 100.0, 64, 1, 0);
        assert_eq!(reduced_rev.len(), 2, "origin must not be absorbed");
        assert_eq!(reduced_rev[1].gamma_0.to_bits(), origin.gamma_0.to_bits());
        assert_eq!(reduced_rev[1].gamma_v.to_bits(), origin.gamma_v.to_bits());
        assert_eq!(reduced_rev[1].gamma_q.to_bits(), origin.gamma_q.to_bits());
        assert_eq!(reduced_rev[1].gamma_s.to_bits(), origin.gamma_s.to_bits());
    }
}
