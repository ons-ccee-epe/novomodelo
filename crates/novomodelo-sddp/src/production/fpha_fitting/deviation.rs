//! FPHA fit-quality deviation diagnostic.
//!
//! [`compute_fit_deviation`] measures how far the emitted concave
//! over-approximation sits from the exact production function on the spill = 0
//! grid — the fit-quality signal an operator sees when a badly non-concave surface
//! leaves a large residual that `α` alone cannot balance. Grid iteration reaches
//! the single owner via `super::grid::build_grid`.

use super::geometry::FittingBounds;
use super::grid::build_grid;
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;

/// Relative mean-absolute-deviation above which a fit is flagged to the operator.
///
/// A heuristic alarm, not a hard error. Raising it hides real misfits; lowering it
/// floods well-fit reservoirs with noise.
const WARN_RELATIVE_THRESHOLD: f64 = 0.05;

/// Aggregate FPHA-vs-exact deviation over the spill = 0 fit grid.
///
/// All magnitudes are in MW; [`Self::relative`] is the dimensionless figure the
/// warning threshold tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FphaFitDeviation {
    /// Mean of `|fitted − exact|` over the grid \[MW\].
    pub mean_abs_mw: f64,
    /// Maximum of `|fitted − exact|` over the grid \[MW\].
    pub max_abs_mw: f64,
    /// Mean signed residual `fitted − exact` \[MW\]: positive = optimistic cap,
    /// negative = pessimistic.
    pub mean_signed_mw: f64,
    /// [`Self::mean_abs_mw`] relative to the grid's peak exact generation
    /// (dimensionless, `≥ 0`). `0.0` when peak ≤ 0, so a zero-production hydro
    /// never trips [`Self::exceeds_warn_threshold`].
    pub relative: f64,
}

/// One `(V, Q)` grid point's FPHA-vs-exact residual at spillage = 0 — the
/// per-point detail the [`FphaFitDeviation`] aggregate is reduced from, computed
/// identically: same min-envelope `fpha_fitted`, same exact `fph_exact`, and a
/// `relative_to_peak` against the same grid peak (`0.0` when peak ≤ 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FphaDeviationPoint {
    /// Volume grid coordinate (hm³).
    pub v: f64,
    /// Turbined-flow grid coordinate (m³/s).
    pub q: f64,
    /// Exact production at `(v, q, 0.0)` (MW).
    pub fph_exact: f64,
    /// Fitted min-envelope value at `(v, q, 0.0)` (MW).
    pub fpha_fitted: f64,
    /// Signed residual `fpha_fitted − fph_exact` (MW).
    pub deviation: f64,
    /// `|deviation|` relative to the grid peak (dimensionless, `≥ 0`).
    pub relative_to_peak: f64,
}

impl FphaFitDeviation {
    /// The neutral diagnostic of a fit with nothing to measure (empty plane set).
    ///
    /// All-zero, so [`Self::exceeds_warn_threshold`] is `false`: an
    /// un-measurable fit never warns.
    const NEUTRAL: Self = Self {
        mean_abs_mw: 0.0,
        max_abs_mw: 0.0,
        mean_signed_mw: 0.0,
        relative: 0.0,
    };

    /// Whether this fit's relative deviation warrants an operator warning.
    pub(crate) fn exceeds_warn_threshold(&self) -> bool {
        self.relative > WARN_RELATIVE_THRESHOLD
    }
}

/// Measure the deviation of the emitted FPHA over-approximation from the exact
/// production function over the spill = 0 fit grid.
///
/// # Contract — compare the MIN envelope, the one the LP consumes
///
/// The fitted value is the pointwise **min** over `planes`, not the max (the LP
/// binds on the minimum). Measuring the max envelope is the wrong-but-compiling
/// alternative — it reports the deviation of a surface the LP never sees,
/// spuriously alarming a faithful fit. Mirrors `super::alpha::compute_alpha_fpha`.
///
/// # Contract — spillage = 0, lateral = 0 only
///
/// Both envelope and exact `FPH` are evaluated at `s = 0`; the `s_points` axis is
/// not iterated, as the cloud build and α regression also ignore it. At `s = 0`
/// the secant term vanishes, so the diagnostic measures the fit the operating
/// region actually uses.
pub(crate) fn compute_fit_deviation(
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> FphaFitDeviation {
    if planes.is_empty() {
        return FphaFitDeviation::NEUTRAL;
    }

    let grid = build_grid(pf, bounds);

    let mut sum_abs = 0.0_f64;
    let mut sum_signed = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut peak_fph = 0.0_f64;
    let mut count = 0_usize;

    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let fitted = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::INFINITY, f64::min);
            let exact = pf.evaluate_capped(v, q, 0.0);
            let signed = fitted - exact;
            let abs = signed.abs();
            sum_abs += abs;
            sum_signed += signed;
            max_abs = max_abs.max(abs);
            peak_fph = peak_fph.max(exact);
            count += 1;
        }
    }

    // `count.max(1)` guards the divisor against a hypothetical empty grid (NaN
    // into a warning), though build_grid always emits ≥ 1 point per axis.
    #[allow(clippy::cast_precision_loss)]
    let n = count.max(1) as f64;
    let mean_abs_mw = sum_abs / n;
    let relative = if peak_fph > 0.0 {
        mean_abs_mw / peak_fph
    } else {
        0.0
    };

    FphaFitDeviation {
        mean_abs_mw,
        max_abs_mw: max_abs,
        mean_signed_mw: sum_signed / n,
        relative,
    }
}

/// Collect the per-point FPHA-vs-exact residuals — the points
/// [`compute_fit_deviation`] aggregates — one [`FphaDeviationPoint`] per
/// `build_grid` `(V, Q)` node, computed identically to the aggregate.
pub(crate) fn collect_fit_deviation_points(
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Vec<FphaDeviationPoint> {
    if planes.is_empty() {
        return Vec::new();
    }

    let grid = build_grid(pf, bounds);

    // `relative_to_peak` needs the whole-grid peak, so the division is deferred to
    // the second walk — keeping the peak identical to the aggregate's.
    let mut points: Vec<FphaDeviationPoint> =
        Vec::with_capacity(grid.v_points.len() * grid.q_points.len());
    let mut peak_fph = 0.0_f64;
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let fpha_fitted = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::INFINITY, f64::min);
            let fph_exact = pf.evaluate_capped(v, q, 0.0);
            let deviation = fpha_fitted - fph_exact;
            peak_fph = peak_fph.max(fph_exact);
            points.push(FphaDeviationPoint {
                v,
                q,
                fph_exact,
                fpha_fitted,
                deviation,
                // Filled in the second walk once the grid peak is known.
                relative_to_peak: 0.0,
            });
        }
    }

    // peak ≤ 0 ⇒ relative pinned to 0 (same guard as `FphaFitDeviation::relative`).
    if peak_fph > 0.0 {
        for point in &mut points {
            point.relative_to_peak = point.deviation.abs() / peak_fph;
        }
    }

    points
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
    use super::{
        FphaFitDeviation, WARN_RELATIVE_THRESHOLD, collect_fit_deviation_points,
        compute_fit_deviation,
    };

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// A 2×2 `(V, Q)` grid over `[0, 30_000]` hm³ — small enough to recompute the
    /// deviation by hand. The spillage axis count is never iterated.
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
    /// a head-dropping tailrace).
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

    /// Net head 0 everywhere (forebay below a constant tailrace), exercising the
    /// `peak ≤ 0` relative-deviation guard.
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

    /// An empty plane set has nothing to measure → the all-zero neutral diagnostic,
    /// which never warns.
    #[test]
    fn empty_planes_is_neutral() {
        let pf = concave_production_function();
        let bounds = small_bounds();
        let dev = compute_fit_deviation(&[], &pf, &bounds);
        assert_eq!(dev, FphaFitDeviation::NEUTRAL);
        assert!(!dev.exceeds_warn_threshold());
    }

    /// The aggregate matches an independent hand accumulation over the same
    /// `(V, Q)` grid at spillage = 0, using the min envelope of a single plane.
    #[test]
    fn matches_hand_computed_deviation() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        let planes = vec![RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        }];

        let v_pts = [0.0_f64, 30_000.0];
        let q_pts = [0.0_f64, pf.max_turbined_m3s];
        let mut sum_abs = 0.0_f64;
        let mut sum_signed = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut peak = 0.0_f64;
        for &v in &v_pts {
            for &q in &q_pts {
                let fitted = planes[0].evaluate(v, q, 0.0);
                let exact = pf.evaluate_capped(v, q, 0.0);
                let signed = fitted - exact;
                sum_abs += signed.abs();
                sum_signed += signed;
                max_abs = max_abs.max(signed.abs());
                peak = peak.max(exact);
            }
        }
        let n = 4.0_f64;
        let expected_mean_abs = sum_abs / n;

        let dev = compute_fit_deviation(&planes, &pf, &bounds);
        assert!((dev.mean_abs_mw - expected_mean_abs).abs() < 1e-9);
        assert!((dev.max_abs_mw - max_abs).abs() < 1e-9);
        assert!((dev.mean_signed_mw - sum_signed / n).abs() < 1e-9);
        assert!((dev.relative - expected_mean_abs / peak).abs() < 1e-9);
    }

    /// The collector emits one point per `(V, Q)` grid node (grid-sized), and each
    /// point's `fph_exact`/`fpha_fitted`/`deviation`/`relative_to_peak` matches an
    /// independent hand recomputation on the same 2×2 grid at spillage = 0, using
    /// the min envelope of a single plane — mirroring `matches_hand_computed_deviation`.
    #[test]
    fn collect_points_match_hand_computed_on_2x2_grid() {
        let pf = concave_production_function();
        let bounds = small_bounds();

        let planes = vec![RawPlane {
            gamma_0: 50.0,
            gamma_v: 0.01,
            gamma_q: 0.3,
            gamma_s: 0.0,
        }];

        let v_pts = [0.0_f64, 30_000.0];
        let q_pts = [0.0_f64, pf.max_turbined_m3s];

        // First hand pass: residuals in canonical V-outer/Q-inner order + grid peak.
        let mut expected: Vec<(f64, f64, f64, f64, f64)> = Vec::new(); // (v, q, exact, fitted, dev)
        let mut peak = 0.0_f64;
        for &v in &v_pts {
            for &q in &q_pts {
                let fitted = planes[0].evaluate(v, q, 0.0);
                let exact = pf.evaluate_capped(v, q, 0.0);
                peak = peak.max(exact);
                expected.push((v, q, exact, fitted, fitted - exact));
            }
        }

        let points = collect_fit_deviation_points(&planes, &pf, &bounds);

        // Grid-sized: 2 × 2 = 4 points.
        assert_eq!(
            points.len(),
            4,
            "collector must emit one point per grid node"
        );

        for (point, (v, q, exact, fitted, dev)) in points.iter().zip(expected.iter()) {
            assert!((point.v - v).abs() < 1e-9, "v mismatch");
            assert!((point.q - q).abs() < 1e-9, "q mismatch");
            assert!((point.fph_exact - exact).abs() < 1e-9, "fph_exact mismatch");
            assert!(
                (point.fpha_fitted - fitted).abs() < 1e-9,
                "fpha_fitted mismatch"
            );
            assert!((point.deviation - dev).abs() < 1e-9, "deviation mismatch");
            let expected_relative = if peak > 0.0 { dev.abs() / peak } else { 0.0 };
            assert!(
                (point.relative_to_peak - expected_relative).abs() < 1e-9,
                "relative_to_peak mismatch"
            );
        }
    }

    /// An empty plane set has nothing to measure → no collected points, matching
    /// the aggregate's empty-plane short-circuit.
    #[test]
    fn collect_points_empty_planes_yields_no_points() {
        let pf = concave_production_function();
        let bounds = small_bounds();
        assert!(collect_fit_deviation_points(&[], &pf, &bounds).is_empty());
    }

    /// A zero-production hydro yields `peak ≤ 0`, so `relative` is pinned to 0 even
    /// though the absolute gap is non-zero — and it never warns.
    #[test]
    fn zero_production_has_zero_relative_and_does_not_warn() {
        let pf = zero_production_function();
        let bounds = small_bounds();
        // A constant cap of 100 MW sits entirely above the all-zero surface.
        let planes = vec![RawPlane {
            gamma_0: 100.0,
            gamma_v: 0.0,
            gamma_q: 0.0,
            gamma_s: 0.0,
        }];
        let dev = compute_fit_deviation(&planes, &pf, &bounds);
        assert!(dev.mean_abs_mw > 0.0, "absolute gap is real");
        assert_eq!(
            dev.relative, 0.0,
            "relative must be pinned to 0 when peak ≤ 0"
        );
        assert!(!dev.exceeds_warn_threshold());
    }

    /// A cap far above the exact surface produces a large relative deviation that
    /// trips the warning, with a positive (optimistic) signed bias.
    #[test]
    fn large_overestimate_warns() {
        let pf = concave_production_function();
        let bounds = small_bounds();
        let planes = vec![RawPlane {
            gamma_0: 1.0e6,
            gamma_v: 0.0,
            gamma_q: 0.0,
            gamma_s: 0.0,
        }];
        let dev = compute_fit_deviation(&planes, &pf, &bounds);
        assert!(dev.relative > WARN_RELATIVE_THRESHOLD);
        assert!(dev.exceeds_warn_threshold());
        assert!(
            dev.mean_signed_mw > 0.0,
            "an over-cap is optimistic (positive)"
        );
    }

    /// The warn predicate is strict (`>`, not `>=`) and keys ONLY on `relative`:
    /// a huge absolute deviation with zero relative does not warn.
    #[test]
    fn warn_threshold_is_strict_and_relative_only() {
        let at = FphaFitDeviation {
            mean_abs_mw: 0.0,
            max_abs_mw: 0.0,
            mean_signed_mw: 0.0,
            relative: WARN_RELATIVE_THRESHOLD,
        };
        assert!(
            !at.exceeds_warn_threshold(),
            "exactly at threshold does not warn"
        );

        let just_over = FphaFitDeviation {
            relative: WARN_RELATIVE_THRESHOLD + 1e-6,
            ..at
        };
        assert!(just_over.exceeds_warn_threshold());

        let huge_absolute_zero_relative = FphaFitDeviation {
            mean_abs_mw: 1.0e9,
            max_abs_mw: 1.0e9,
            mean_signed_mw: 1.0e9,
            relative: 0.0,
        };
        assert!(
            !huge_absolute_zero_relative.exceeds_warn_threshold(),
            "the threshold is relative-only"
        );
    }
}
