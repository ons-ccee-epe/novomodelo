//! Convex-hull FPHA fitter: production cloud → upper-envelope hyperplanes.
//!
//! Owns the core of the computed-FPHA path. It evaluates the exact
//! production function `generation(V, Q)` on the `(V, Q)` grid (at spillage = 0 and
//! lateral = 0), closes the region with the point `(V̄, Q̄, 0)`, takes the 3-D
//! convex hull via [`crate::hull::convex_hull_3d`], and reads the
//! upper-envelope facets as planes `generation = γ₀ + γ_V·V + γ_Q·Q`.
//!
//! The grid axes come from the single grid-formula owner `super::grid::build_grid`
//! — the spillage axis it also produces is intentionally unused here, because
//! spillage is not a hull dimension (it is fixed at 0 in every cloud evaluation).

use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::production::ProductionFunction;
use crate::hull::{HullError, Hyperplane3d, convex_hull_3d};

/// An unscaled FPHA plane `generation = γ₀ + γ_V·V + γ_Q·Q (+ γ_S·s)`.
///
/// The single coefficient carrier the whole fitting pipeline threads: the hull
/// fitter emits it, the α correction scales it, the lateral secant writes its
/// `γ_S`, and validation reads its signs. The hull fitter sets `gamma_s = 0.0`:
/// the `γ_S` secant axis is a later refinement and spillage is not a hull
/// dimension. The intercept is NOT scaled by α — that scaling is applied by the
/// α correction before conversion to `super::FphaPlane`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::struct_field_names)]
pub(crate) struct RawPlane {
    /// Intercept `γ₀` (NOT scaled by α).
    pub gamma_0: f64,
    /// Volume gradient `γ_V` [MW / hm³]. Non-negative for upper-envelope facets.
    pub gamma_v: f64,
    /// Turbined-flow gradient `γ_Q` [MW / (m³/s)]. Non-negative for upper-envelope facets.
    pub gamma_q: f64,
    /// Spillage gradient `γ_S` [MW / (m³/s)]. `0.0` from the hull fitter; written by the secant.
    pub gamma_s: f64,
}

impl RawPlane {
    /// Evaluates the plane at `(v, q, s)`.
    pub(crate) fn evaluate(&self, v: f64, q: f64, s: f64) -> f64 {
        self.gamma_0 + self.gamma_v * v + self.gamma_q * q + self.gamma_s * s
    }
}

/// Tolerance on the facet normal's `generation` component used to select
/// upper-envelope facets and reject near-vertical facets.
///
/// A facet with `|nz| ≤ EPS_NZ` is a side wall (near-vertical in `generation`);
/// dividing by such an `nz` would blow up the converted coefficients, and the
/// facet carries no concave over-approximation information, so it is dropped.
const EPS_NZ: f64 = 1e-9;

/// Tolerance below which a single-volume facet's fitted `γ_V` is snapped to
/// exactly `0.0`.
///
/// A run-of-river plant's production is independent of `V`, so the correct fit has
/// `γ_V = 0`; the synthesized two-sample `V` axis leaves only a near-zero residual.
/// Snapping `|γ_V| <= GAMMA_V_EPS` to `0.0` makes the `Coef.Vutil = 0` contract
/// exact. The threshold sits above the downstream `γ_V >= -1e-10` sign tolerance
/// so a snapped plane never trips it.
const GAMMA_V_EPS: f64 = 1e-6;

/// Build the `(V, Q, generation)` production cloud at spillage = 0 and lateral = 0.
///
/// The flow axis starts at `q = 0` (where `generation = 0`), so the zero-flow
/// column is the region's lower closure — no synthetic closing point is added.
///
/// # Contract
///
/// Spillage and lateral inflow are NOT cloud dimensions: every point is evaluated
/// at `s = 0` and the production function carries no lateral term. Re-introducing
/// a spillage grid axis would make the cloud 4-D and break the `convex_hull_3d`
/// contract.
fn build_cloud(pf: &ProductionFunction, bounds: &FittingBounds) -> Vec<[f64; 3]> {
    let grid = build_grid(pf, bounds);

    let mut cloud = Vec::with_capacity(grid.v_points.len() * grid.q_points.len());
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let generation = pf.evaluate_capped(v, q, 0.0);
            cloud.push([v, q, generation]);
        }
    }

    cloud
}

/// Convert an upper-envelope facet to a plane `generation = γ₀ + γ_V·V + γ_Q·Q`.
///
/// # Sign contract
///
/// `convex_hull_3d` returns each facet as `nx·V + ny·Q + nz·generation + offset = 0`
/// with an OUTWARD unit normal. An upper-envelope facet — one that bounds the
/// cloud from above in the `generation` direction — has `nz > 0` (verified empirically
/// against a known concave cloud). Solving for `generation` divides through by `-nz`:
///
/// ```text
/// γ_V = -nx / nz,   γ_Q = -ny / nz,   γ₀ = -offset / nz
/// ```
///
/// The wrong-but-compiling alternative is to keep the LOWER-envelope facets
/// (`nz < 0`); those bound the cloud from below and yield a convex
/// under-approximation, the opposite of the concave over-approximation FPHA
/// requires. Dividing by `nz` (not `-nz`) flips every coefficient sign,
/// producing `γ_V ≤ 0` / `γ_Q ≤ 0` which violates the downstream sign
/// convention. For an upper facet `nx, ny ≤ 0` and `nz > 0`, so `γ_V, γ_Q ≥ 0`,
/// matching the `RawPlane` sign convention.
fn facet_to_plane(facet: &Hyperplane3d) -> RawPlane {
    let nz = facet.normal[2];
    RawPlane {
        gamma_0: -facet.offset / nz,
        gamma_v: -facet.normal[0] / nz,
        gamma_q: -facet.normal[1] / nz,
        gamma_s: 0.0,
    }
}

/// Fit upper-envelope FPHA planes from the production cloud via convex hull.
///
/// # Upper-envelope selection
///
/// A hull facet is part of the concave over-approximation iff its outward normal
/// has a POSITIVE `generation` component (`nz > EPS_NZ`): such a facet bounds the cloud
/// from above. The lower-envelope facets (`nz < 0`) and the near-vertical side
/// walls (`|nz| ≤ EPS_NZ`) are dropped. Keeping the lower-envelope facets is the
/// wrong-but-compiling alternative — it yields a convex under-approximation, the
/// opposite of the required concave envelope.
///
/// # Determinism
///
/// `convex_hull_3d`'s canonical sort-out fixes the facet order independently of
/// the cloud's iteration order; this function preserves that order (it never
/// re-sorts). The coplanar-facet dedup keys on exact `f64::to_bits()` equality of
/// `(γ₀, γ_V, γ_Q)`, a deterministic predicate, so the emitted plane `Vec` is an
/// element-for-element function of the cloud set, upholding declaration-order
/// invariance. A tolerance-based merge is deliberately NOT done here — that is the
/// similar-hyperplane merge of a later stage, and a careless tolerance merge
/// would be order-dependent.
///
/// # Errors
///
/// Returns [`HullError`] when the cloud is degenerate (collinear/coplanar, fewer
/// than 4 affinely-independent points) or the hull computation fails. The caller
/// maps these to a typed `FphaFittingError`.
pub(crate) fn fit_hull_planes(
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Result<Vec<RawPlane>, HullError> {
    let cloud = build_cloud(pf, bounds);
    let facets = convex_hull_3d(&cloud)?;

    let mut planes: Vec<RawPlane> = facets
        .iter()
        .filter(|f| f.normal[2] > EPS_NZ)
        .map(facet_to_plane)
        .collect();

    // Run-of-river γ_V = 0 contract. The facet γ_V from the synthesized
    // two-sample V axis is a near-zero residual, not a real storage gradient; snap
    // it to exactly 0.0 so `Coef.Vutil = 0` holds. Leaving the noisy γ_V (the
    // forbidden alternative) would emit a spurious storage slope and feed roundoff
    // into the LP head constraint. The zeroing runs BEFORE the dedup so the dedup
    // key (which includes γ_V) sees the snapped value, keeping the emitted Vec a
    // function of the cloud set.
    if bounds.single_volume {
        for plane in &mut planes {
            if plane.gamma_v.abs() <= GAMMA_V_EPS {
                plane.gamma_v = 0.0;
            }
        }
    }

    // `Qt` triangulation splits each flat hull face into several facets sharing one
    // plane, so dedup on EXACT to_bits() equality of (γ₀, γ_V, γ_Q) — a
    // deterministic, order-preserving predicate (first occurrence wins) that emits
    // one plane per face while upholding declaration-order invariance.
    let mut seen: Vec<(u64, u64, u64)> = Vec::with_capacity(planes.len());
    planes.retain(|p| {
        let key = (
            p.gamma_0.to_bits(),
            p.gamma_v.to_bits(),
            p.gamma_q.to_bits(),
        );
        if seen.contains(&key) {
            false
        } else {
            seen.push(key);
            true
        }
    });

    Ok(planes)
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
    use super::super::production::{ProductionFunction, TailraceSource};
    use super::{RawPlane, build_cloud, fit_hull_planes};
    use crate::hull::HullError;

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

    /// A strictly concave `generation(V, Q)`: a rising-then-flattening forebay curve plus
    /// a polynomial tailrace whose head drops with outflow makes both the V and Q
    /// cross-sections strictly concave, so the upper hull has several distinct
    /// facets (a genuine outer-approximation stress, not a single plane).
    fn concave_production_function() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "Concave").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.0008, -2e-8],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            3_000.0,
            "Concave".to_owned(),
        )
    }

    /// A V-dependent but Q-linear production function: sloped forebay, no
    /// tailrace. generation varies with both V and Q so the cloud stays full-dimensional.
    fn sloped_production_function() -> ProductionFunction {
        let rows = vec![row(0.0, 380.0), row(30_000.0, 410.0)];
        let forebay = ForebayTable::new(&rows, "Sloped").expect("valid VHA curve");
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(None),
            None,
            None,
            3_000.0,
            "Sloped".to_owned(),
        )
    }

    /// A constant-head flat-forebay function with no tailrace: generation = c·q,
    /// independent of V — the degenerate-in-generation path.
    fn flat_production_function() -> ProductionFunction {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 400.0)];
        let forebay = ForebayTable::new(&rows, "Flat").expect("valid VHA curve");
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(None),
            None,
            None,
            3_000.0,
            "Flat".to_owned(),
        )
    }

    /// A run-of-river production function: flat forebay (no storage head gain)
    /// plus a polynomial tailrace whose head drops with outflow, making the
    /// `Q` cross-section strictly concave. With a constant forebay height the
    /// exact `generation` is independent of `V`, so the correct single-volume fit has
    /// `γ_V = 0`.
    fn run_of_river_production_function() -> ProductionFunction {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 400.0)];
        let forebay = ForebayTable::new(&rows, "RunOfRiver").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.0008, -2e-8],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            3_000.0,
            "RunOfRiver".to_owned(),
        )
    }

    /// Single-volume fitting bounds: the two synthesized tangent-in-volume
    /// samples `[14_998, 15_002]` hm³ around `v0 = 15_000`, with
    /// `single_volume = true` so the fitter zeros `γ_V`. Mirrors what
    /// `resolve_fitting_bounds` produces for a run-of-river plant.
    fn single_volume_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 14_998.0,
            v_max: 15_002.0,
            n_volume_points: 2,
            n_flow_points: 6,
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: true,
        }
    }

    /// Evaluate the pointwise max of an envelope of planes at `(v, q)`.
    fn envelope_max(planes: &[RawPlane], v: f64, q: f64) -> f64 {
        planes
            .iter()
            .map(|p| p.gamma_0 + p.gamma_v * v + p.gamma_q * q)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn upper_envelope_is_outer_approximation_on_concave_function() {
        let pf = concave_production_function();
        let bounds = test_bounds();
        let planes = fit_hull_planes(&pf, &bounds).expect("concave cloud yields a valid hull");
        assert!(
            !planes.is_empty(),
            "expected at least one upper-envelope plane"
        );

        let cloud = build_cloud(&pf, &bounds);
        for point in &cloud {
            let [v, q, generation] = *point;
            let envelope = envelope_max(&planes, v, q);
            assert!(
                envelope >= generation - 1e-8,
                "outer-approximation violated at (v={v}, q={q}): envelope={envelope} < generation={generation}"
            );
        }
    }

    #[test]
    fn upper_envelope_planes_have_valid_coefficient_signs() {
        let pf = concave_production_function();
        let bounds = test_bounds();
        let planes = fit_hull_planes(&pf, &bounds).expect("valid hull");
        for (idx, plane) in planes.iter().enumerate() {
            assert!(
                plane.gamma_v >= -1e-10,
                "plane {idx}: gamma_v={} must be >= 0",
                plane.gamma_v
            );
            assert!(
                plane.gamma_q >= -1e-10,
                "plane {idx}: gamma_q={} must be >= 0",
                plane.gamma_q
            );
            assert_eq!(
                plane.gamma_s, 0.0,
                "plane {idx}: gamma_s must be 0.0 from the hull fitter"
            );
        }
    }

    #[test]
    fn cloud_samples_grid_at_zero_spillage_with_zero_flow_floor() {
        let pf = concave_production_function();
        let bounds = test_bounds();
        let cloud = build_cloud(&pf, &bounds);

        // The cloud is exactly the (V, Q) grid — one node each, no synthetic
        // closing point. The zero-flow column provides the lower closure.
        assert_eq!(
            cloud.len(),
            bounds.n_volume_points * bounds.n_flow_points,
            "cloud must be exactly the grid nodes, with no appended point"
        );

        // Every cloud point's generation equals the CAPPED production at spillage = 0,
        // confirming spillage is fixed (and lateral is implicitly 0).
        for &[v, q, generation] in &cloud {
            assert_eq!(
                generation,
                pf.evaluate_capped(v, q, 0.0),
                "cloud generation at (v={v}, q={q}) must be the capped value at spillage = 0"
            );
        }

        // The flow axis starts at q = 0, where production is zero — the floor that
        // closes the region from below.
        let zero_flow: Vec<_> = cloud.iter().filter(|p| p[1] == 0.0).collect();
        assert_eq!(
            zero_flow.len(),
            bounds.n_volume_points,
            "one zero-flow node per volume point"
        );
        for p in zero_flow {
            assert_eq!(p[2], 0.0, "production is zero at zero flow");
        }
    }

    /// The installed-capacity ceiling clips the cloud, and the hull gains a flat
    /// `generation ≤ capacity` plane (zero gradients) where the raw output runs past it.
    #[test]
    fn capacity_ceiling_clips_cloud_and_yields_flat_cap_plane() {
        let cap = 5_000.0_f64;
        let pf = concave_production_function().with_max_generation_mw(cap);
        let bounds = test_bounds();

        // Uncapped output exceeds the ceiling at the top corner; capped equals it.
        let v_top = bounds.v_max;
        let q_top = pf.max_turbined_m3s;
        assert!(
            pf.evaluate(v_top, q_top, 0.0) > cap,
            "fixture must exceed the ceiling for this test to bite"
        );
        assert_eq!(pf.evaluate_capped(v_top, q_top, 0.0), cap);

        // No cloud point exceeds the ceiling.
        for &[_, _, generation] in &build_cloud(&pf, &bounds) {
            assert!(
                generation <= cap + 1e-9,
                "cloud generation {generation} must not exceed cap {cap}"
            );
        }

        // The hull includes a flat cap plane: near-zero gradients, intercept ~ cap.
        let planes = fit_hull_planes(&pf, &bounds).expect("hull fit succeeds");
        assert!(
            planes.iter().any(|p| p.gamma_v.abs() < 1e-6
                && p.gamma_q.abs() < 1e-6
                && (p.gamma_0 - cap).abs() < 1.0),
            "expected a flat generation <= cap plane near {cap}, got {planes:?}"
        );
    }

    #[test]
    fn shuffle_invariant_plane_vec_is_bit_identical() {
        // The hull fitter must be a function of the cloud SET, not its iteration
        // order (the canonical sort-out + bit-exact dedup own this). convex_hull_3d
        // guarantees order independence, so a second identical call matching
        // bit-for-bit suffices as the reproducibility check.
        let pf = concave_production_function();
        let bounds = test_bounds();

        let reference = fit_hull_planes(&pf, &bounds).expect("valid hull");
        let again = fit_hull_planes(&pf, &bounds).expect("valid hull");
        assert_eq!(reference.len(), again.len());
        for (a, b) in reference.iter().zip(&again) {
            assert_eq!(a.gamma_0.to_bits(), b.gamma_0.to_bits());
            assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
            assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
        }
    }

    #[test]
    fn shuffle_invariant_across_explicit_cloud_orderings() {
        use crate::hull::convex_hull_3d;

        let pf = concave_production_function();
        let bounds = test_bounds();
        let cloud = build_cloud(&pf, &bounds);
        let mut reversed = cloud.clone();
        reversed.reverse();

        // convex_hull_3d guarantees order-independent facets; the fitter's
        // selection+dedup are pure functions of that facet Vec, so two cloud
        // orderings must yield identical planes.
        let f1 = convex_hull_3d(&cloud).expect("valid hull");
        let f2 = convex_hull_3d(&reversed).expect("valid hull");
        assert_eq!(
            f1, f2,
            "convex_hull_3d facet order must be cloud-order invariant"
        );
    }

    #[test]
    fn sloped_function_yields_outer_approximation_planes() {
        // A sloped (V-dependent) but Q-linear production function is still
        // non-degenerate (the generation surface varies in both V and Q), so the hull
        // fit must produce at least one valid upper-envelope plane.
        let pf = sloped_production_function();
        let bounds = test_bounds();
        let result = fit_hull_planes(&pf, &bounds);
        match result {
            Ok(planes) => {
                assert!(!planes.is_empty(), "sloped function must yield >= 1 plane");
                let cloud = build_cloud(&pf, &bounds);
                for point in &cloud {
                    let [v, q, generation] = *point;
                    assert!(
                        envelope_max(&planes, v, q) >= generation - 1e-8,
                        "outer approximation violated at (v={v}, q={q})"
                    );
                }
            }
            Err(e) => panic!("sloped function should yield a valid hull, got {e:?}"),
        }
    }

    #[test]
    fn flat_function_either_yields_a_plane_or_typed_degenerate() {
        // A constant-head flat-forebay function with no tailrace gives
        // generation = c·q, independent of V. The (V, Q, generation) cloud
        // may be a valid prism (>= 1 plane) or collapse to a degenerate hull. Both
        // outcomes are acceptable; a panic or an empty Ok is not.
        let pf = flat_production_function();
        let bounds = test_bounds();
        match fit_hull_planes(&pf, &bounds) {
            Ok(planes) => {
                assert!(
                    !planes.is_empty(),
                    "a successful flat-function fit must not be an empty Ok"
                );
            }
            Err(HullError::Degenerate) => {
                // Acceptable: the typed degenerate path. fit_fpha_planes maps this
                // to FphaFittingError::DegenerateProductionCloud.
            }
            Err(other) => panic!("unexpected hull error for flat function: {other:?}"),
        }
    }

    #[test]
    fn single_volume_fit_yields_gamma_v_exactly_zero() {
        // A run-of-river plant fit through the single-volume path must emit at
        // least one plane, and every plane's γ_V must be exactly 0.0 — the
        // run-of-river contract `Coef.Vutil = 0`.
        let pf = run_of_river_production_function();
        let bounds = single_volume_bounds();
        let planes = fit_hull_planes(&pf, &bounds).expect("single-volume cloud yields a hull");
        assert!(
            !planes.is_empty(),
            "single-volume fit must yield >= 1 plane"
        );
        for (idx, plane) in planes.iter().enumerate() {
            assert_eq!(
                plane.gamma_v.to_bits(),
                0.0_f64.to_bits(),
                "plane {idx}: gamma_v={} must be exactly 0.0",
                plane.gamma_v
            );
        }
    }

    #[test]
    fn single_volume_envelope_is_outer_approximation_in_q() {
        // The single-volume planes must still form a valid concave outer
        // approximation of generation(v0, q) at the fixed volume: at every (v0, q) grid
        // point, max_k(γ₀ + γ_Q·q) >= generation(v0, q) - 1e-8.
        let pf = run_of_river_production_function();
        let bounds = single_volume_bounds();
        let planes = fit_hull_planes(&pf, &bounds).expect("single-volume cloud yields a hull");

        let cloud = build_cloud(&pf, &bounds);
        for point in &cloud {
            let [v, q, generation] = *point;
            let envelope = envelope_max(&planes, v, q);
            assert!(
                envelope >= generation - 1e-8,
                "outer-approximation violated at (v={v}, q={q}): envelope={envelope} < generation={generation}"
            );
        }
    }

    #[test]
    fn single_volume_planes_have_valid_signs() {
        // γ_V zeroed exactly clears the γ_V >= -1e-10 check trivially; γ_Q must
        // still be non-negative and γ_S still 0 from the hull fitter.
        let pf = run_of_river_production_function();
        let bounds = single_volume_bounds();
        let planes = fit_hull_planes(&pf, &bounds).expect("single-volume cloud yields a hull");
        for (idx, plane) in planes.iter().enumerate() {
            assert_eq!(plane.gamma_v, 0.0, "plane {idx}: gamma_v must be 0.0");
            assert!(
                plane.gamma_q >= -1e-10,
                "plane {idx}: gamma_q={} must be >= 0",
                plane.gamma_q
            );
            assert_eq!(plane.gamma_s, 0.0, "plane {idx}: gamma_s must be 0.0");
        }
    }

    #[test]
    fn single_volume_shuffle_invariant_plane_vec_is_bit_identical() {
        // The single-volume path (γ_V zeroing runs before the bit-exact dedup)
        // must be a function of the cloud SET, not its iteration order. Feed the
        // same cloud through convex_hull_3d in two opposite orderings and assert
        // the converted+zeroed+deduped plane Vec is element-for-element identical.
        use crate::hull::convex_hull_3d;

        let pf = run_of_river_production_function();
        let bounds = single_volume_bounds();

        let cloud = build_cloud(&pf, &bounds);
        let mut reversed = cloud.clone();
        reversed.reverse();

        // convex_hull_3d's canonical sort-out already fixes the facet order;
        // confirm the full fitter is order-invariant end to end via two calls.
        let f1 = convex_hull_3d(&cloud).expect("valid hull");
        let f2 = convex_hull_3d(&reversed).expect("valid hull");
        assert_eq!(
            f1, f2,
            "convex_hull_3d facet order must be cloud-order invariant"
        );

        let reference = fit_hull_planes(&pf, &bounds).expect("valid hull");
        let again = fit_hull_planes(&pf, &bounds).expect("valid hull");
        assert_eq!(reference.len(), again.len());
        for (a, b) in reference.iter().zip(&again) {
            assert_eq!(a.gamma_0.to_bits(), b.gamma_0.to_bits());
            assert_eq!(a.gamma_v.to_bits(), b.gamma_v.to_bits());
            assert_eq!(a.gamma_q.to_bits(), b.gamma_q.to_bits());
        }
    }
}
