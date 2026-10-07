//! Least-squares `α_FPHA` correction on the spill = 0 grid.
//!
//! The raw upper-envelope hull (`FPHA_0`) is biased; a single scalar `α_FPHA`
//! minimising `Σ (α·FPHA_0 − FPH)²` over the `(V, Q)` fit grid corrects it, and
//! the emitted plane is `α·FPHA_0`. Grid iteration reaches the single owner via
//! `super::grid::build_grid`, so the regression grid never drifts from the cloud.

use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;

/// Compute the least-squares `α_FPHA` correction factor for a raw hull envelope.
///
/// # Contract — regress the MIN envelope, the one the LP consumes
///
/// `FPHA_0` is the pointwise **min** over the raw hull planes, not the max: the LP
/// applies `g ≤ plane_k` for every plane, so the binding cap is the minimum.
/// Regressing the max envelope is the wrong-but-compiling alternative — for a
/// concave φ it runs ≈1.5–1.8×φ, collapsing α to ≈0.6 and shrinking the LP cap to
/// ≈0.6·φ (a ~30–40% under-generation, invisible for run-of-river where
/// `min == max`).
///
/// # Contract — spillage = 0, lateral = 0 only
///
/// Both `FPHA_0` and `FPH` are evaluated at `s = 0`; the `s_points` axis is
/// deliberately not iterated (as `super::hull_fit` build also ignores it). Folding
/// the spillage axis into the sum pulls α toward the spill region and degrades the
/// dominant no-spill region that governs operation.
///
/// # Contract — degenerate denominator returns the neutral `α = 1.0`
///
/// `Σ FPHA_0² == 0` (all-zero-production hydro, net head ≤ 0 over the grid) leaves
/// the ratio undefined; return `α = 1.0` rather than dividing by zero or
/// hard-erroring — it leaves the already-zero planes unchanged, and downstream
/// `NoHyperplanesProduced` / sign validation still rejects a genuinely unusable
/// envelope.
pub(crate) fn compute_alpha_fpha(
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> f64 {
    if planes.is_empty() {
        return 1.0;
    }

    let grid = build_grid(pf, bounds);

    let mut sum_fpha0_fph = 0.0_f64;
    let mut sum_fpha0_sq = 0.0_f64;

    // `fpha0` is the MIN over planes (the LP cap), evaluated at s = 0 against the
    // capped output — both contracts above.
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let fpha0 = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::INFINITY, f64::min);
            let fph = pf.evaluate_capped(v, q, 0.0);
            sum_fpha0_fph += fpha0 * fph;
            sum_fpha0_sq += fpha0 * fpha0;
        }
    }

    if sum_fpha0_sq == 0.0 {
        1.0
    } else {
        sum_fpha0_fph / sum_fpha0_sq
    }
}

/// Scale the whole affine function of a raw hull plane by `α`.
///
/// # Contract — `α` scales every coefficient, not just the intercept
///
/// `α` multiplies `γ₀`, `γ_V`, `γ_Q`, and `γ_S` alike, since
/// `FPHA = α·FPHA_0`. Scaling only the intercept (`α·γ₀` with raw gradients) is the
/// wrong-but-compiling alternative — not equivalent to scaling the affine function.
pub(crate) fn scale_plane_affine(plane: &RawPlane, alpha: f64) -> RawPlane {
    RawPlane {
        gamma_0: plane.gamma_0 * alpha,
        gamma_v: plane.gamma_v * alpha,
        gamma_q: plane.gamma_q * alpha,
        gamma_s: plane.gamma_s * alpha,
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
    use super::{compute_alpha_fpha, scale_plane_affine};

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// A 2×2 `(V, Q)` grid over `[0, 30_000]` hm³. Spillage axis count is 2 but is
    /// never iterated by `compute_alpha_fpha`.
    fn small_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 0.0,
            v_max: 30_000.0,
            n_volume_points: 2,
            n_flow_points: 2,
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: false,
        }
    }

    /// A strictly concave production function (rising-then-flattening forebay plus
    /// a head-dropping tailrace), so the hull yields several distinct facets.
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

    /// Net head 0 everywhere (forebay below the constant tailrace), so every grid
    /// point clamps to 0 — exercising the degenerate `Σ FPHA_0² == 0` denominator.
    fn zero_production_function() -> ProductionFunction {
        let rows = vec![row(0.0, 100.0), row(30_000.0, 100.0)];
        let forebay = ForebayTable::new(&rows, "Zero").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![200.0],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            None,
            None,
            3_000.0,
            "Zero".to_owned(),
        )
    }

    /// Closed-form α on a hand-computable grid matches `Σ FPHA_0·FPH / Σ FPHA_0²`.
    ///
    /// Uses a single raw plane so `FPHA_0(V,Q)` is exactly that plane, making the
    /// reference sum trivial to recompute independently of `compute_alpha_fpha`.
    #[test]
    fn closed_form_alpha_matches_hand_computed_ratio() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        let planes = vec![RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        }];

        // Independent reference accumulation over the same (V, Q) grid at s = 0.
        let v_pts = [0.0_f64, 30_000.0];
        let q_pts = [0.0_f64, pf.max_turbined_m3s];
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &v in &v_pts {
            for &q in &q_pts {
                let fpha0 = planes[0].evaluate(v, q, 0.0);
                let fph = pf.evaluate_capped(v, q, 0.0);
                num += fpha0 * fph;
                den += fpha0 * fpha0;
            }
        }
        let expected = num / den;

        let alpha = compute_alpha_fpha(&planes, &pf, &bounds);
        assert!(
            (alpha - expected).abs() < 1e-12,
            "alpha={alpha} must match hand-computed ratio {expected}"
        );
    }

    /// The regression evaluates both `FPHA_0` and `FPH` at spillage = 0: feeding a
    /// plane with a non-zero `gamma_s` must not change α (the s axis is not summed).
    #[test]
    fn regression_is_spillage_zero_only() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        let base = RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        };
        let with_spill = RawPlane {
            gamma_s: -0.5,
            ..base
        };

        let alpha_base = compute_alpha_fpha(&[base], &pf, &bounds);
        let alpha_spill = compute_alpha_fpha(&[with_spill], &pf, &bounds);
        assert_eq!(
            alpha_base, alpha_spill,
            "alpha must be independent of gamma_s (regression is spillage = 0)"
        );

        // FPH is also the s = 0 term: a rising tailrace makes s > 0 strictly
        // smaller, so the regression's use of pf.evaluate(.,.,0) is observable.
        let v = 30_000.0_f64;
        let q = pf.max_turbined_m3s;
        let fph_s0 = pf.evaluate(v, q, 0.0);
        let fph_s_pos = pf.evaluate(v, q, 500.0);
        assert!(
            fph_s_pos < fph_s0,
            "spillage must lower FPH (s>0 head drop); regression must use the s=0 value"
        );
    }

    /// `scale_plane_affine` scales the whole affine function: the scaled plane
    /// equals `α·raw_plane(V,Q)` at every probe point, intercept and gradients alike.
    #[test]
    fn scaling_is_full_affine() {
        let raw = RawPlane {
            gamma_0: 123.0,
            gamma_v: 0.04,
            gamma_q: 0.7,
            gamma_s: -0.2,
        };
        let alpha = 0.83_f64;
        let scaled = scale_plane_affine(&raw, alpha);

        assert!((scaled.gamma_0 - alpha * raw.gamma_0).abs() < 1e-12);
        assert!((scaled.gamma_v - alpha * raw.gamma_v).abs() < 1e-12);
        assert!((scaled.gamma_q - alpha * raw.gamma_q).abs() < 1e-12);
        assert!((scaled.gamma_s - alpha * raw.gamma_s).abs() < 1e-12);

        // The whole affine function scales, not just the intercept.
        for &(v, q, s) in &[
            (0.0, 1.0, 0.0),
            (10_000.0, 500.0, 30.0),
            (30_000.0, 0.0, 0.0),
        ] {
            let scaled_val = scaled.evaluate(v, q, s);
            let alpha_raw_val = alpha * raw.evaluate(v, q, s);
            assert!(
                (scaled_val - alpha_raw_val).abs() < 1e-12,
                "scaled_plane({v},{q},{s})={scaled_val} must equal alpha*raw={alpha_raw_val}"
            );
        }
    }

    /// All-zero production → `Σ FPHA_0² == 0` → neutral `α = 1.0`, no panic.
    #[test]
    fn degenerate_denominator_returns_neutral_alpha() {
        let pf = zero_production_function();
        let bounds = small_bounds();

        // All-zero planes ⇒ FPHA_0 = 0 everywhere ⇒ Σ FPHA_0² = 0.
        let planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.0,
            gamma_q: 0.0,
            gamma_s: 0.0,
        }];

        let alpha = compute_alpha_fpha(&planes, &pf, &bounds);
        assert_eq!(
            alpha, 1.0,
            "all-zero production must yield the neutral alpha = 1.0"
        );
    }

    /// Empty plane set short-circuits to the neutral α = 1.0.
    #[test]
    fn empty_planes_returns_neutral_alpha() {
        let pf = concave_production_function();
        let bounds = small_bounds();
        let alpha = compute_alpha_fpha(&[], &pf, &bounds);
        assert_eq!(alpha, 1.0, "empty plane set must yield neutral alpha = 1.0");
    }
}
