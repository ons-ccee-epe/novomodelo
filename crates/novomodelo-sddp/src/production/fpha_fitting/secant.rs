//! Per-plane lateral-flow secant `γ_S` on the `lateral_flow` axis.
//!
//! Each spill = 0 grid point `(v, q)` gets a zero-anchored least-squares slope of
//! `generation` against `lateral_flow ∈ [0, S_max]`; a plane's `γ_S` is the mean
//! of that slope over the grid points where the plane is the binding bound. The
//! sample is a closed-form evenly-spaced grid, so the fit is deterministic and
//! order-independent.
//!
//! # Meaning of `γ_S`
//!
//! `γ_S` is the plane's slope per m³/s of tailrace lateral flow `L`: the own spill
//! times the own outflow factor plus the other plants' contributions. It is fitted
//! on `L ∈ [0, S_max]` with `FphaColumnLayout::resolve_spillage_fit_max_m3s`'s
//! `S_max`; past `S_max` the plane extrapolates linearly.

use super::geometry::FittingBounds;
use super::hull_fit::RawPlane;
use super::production::ProductionFunction;

/// Number of evenly-spaced `lateral_flow` samples, `0` and `S_max` included, that
/// discretise the least-squares criterion over `[0, S_max]`.
///
/// Fixed (not config-driven) so the secant is deterministic and order-independent.
const N_LATERAL_FLOW_SAMPLES: usize = 101;

/// Magnitude below which a fitted `γ_S` slope is snapped to exactly 0.
///
/// The fit of a flat tailrace leaves floating-point residuals of EITHER sign;
/// a surviving near-zero structural coefficient in the spillage column wrecks LP
/// column scaling, so both signs MUST snap to `0.0`. The bound sits well below the
/// smallest physical spillage sensitivity (~`1e-6` MW/(m³/s)), so no real
/// coefficient is snapped, and at/above the `validate_fitted_planes` sign
/// tolerance so every snapped value is sign-valid.
const GAMMA_S_SNAP_EPS: f64 = 1e-10;

/// Zero-anchored least-squares slope of `generation` vs `lateral_flow` at
/// `(v, q)`: `Σ sᵢ·(g(sᵢ) − g(0)) / Σ sᵢ²` over the fixed sample of `[0, S_max]`.
///
/// # Contract — anchored at the spill = 0 generation, not a free intercept
///
/// The plane already carries the spill = 0 generation in its other terms, so the
/// line is forced through `(0, g(0))`. A free-intercept OLS slope is the
/// wrong-but-compiling alternative: it fits the curvature of `g(s)` with an
/// intercept the plane never uses, and over a wide `[0, S_max]` drifts from the
/// secant the plane needs.
///
/// # Contract — degenerate `S_max ≤ 0` returns the neutral `0.0`
///
/// `S_max ≤ 0` collapses the sample to the anchor (`Σ sᵢ² = 0`); return `0.0`
/// rather than dividing by zero.
fn anchored_slope(pf: &ProductionFunction, v: f64, q: f64, s_max: f64) -> f64 {
    if s_max <= 0.0 {
        return 0.0;
    }

    #[allow(clippy::cast_possible_truncation)]
    let denom = f64::from((N_LATERAL_FLOW_SAMPLES - 1) as u32);
    let generation_at_anchor = pf.evaluate(v, q, 0.0);

    let mut cross = 0.0_f64;
    let mut square = 0.0_f64;
    for i in 1..N_LATERAL_FLOW_SAMPLES {
        #[allow(clippy::cast_possible_truncation)]
        let lateral_flow = f64::from(i as u32) * s_max / denom;
        cross += lateral_flow * (pf.evaluate(v, q, lateral_flow) - generation_at_anchor);
        square += lateral_flow * lateral_flow;
    }

    cross / square
}

/// Fit `γ_S` in place for every plane on `lateral_flow ∈ [0, S_max]`.
///
/// `γ_S` is the only coefficient this step sets; the α-scaled gradients and
/// intercept are untouched. A plane's `γ_S` is the mean [`anchored_slope`] over
/// the spill = 0 grid points where it is active; a plane active nowhere keeps
/// `γ_S = 0`.
///
/// # Contract — average where the plane binds (argmin), not where it is loosest
///
/// A point counts for a plane when the plane is the pointwise **minimum** over
/// planes there (the LP applies `g ≤ plane_k`), matching the min-envelope
/// contract in `compute_alpha_fpha`. Averaging where a plane is the pointwise
/// **maximum** (the loosest bound) is the wrong-but-compiling alternative: it
/// reads the slope where the plane does not govern, over-steepening `γ_S` on
/// high-head planes.
///
/// The slopes are evaluated on the uncapped [`ProductionFunction::evaluate`],
/// once per grid point, and averaged in fixed `(v, q)` grid order.
///
/// # Contract — the plane is at or above 0 at `(bounds.v_min, 0, S_max)`
///
/// `γ_S ← max(γ_S, min(0, −(γ₀ + γ_V·v_min) / S_max))`, skipped for `S_max ≤ 0`, holds
/// the plane at or above 0 at the corner. Applied after the snap it leaves a −1e-16
/// residual in the spillage column; a `v` other than `bounds.v_min` breaks the bound.
/// Pinned by `gamma_s_limit_puts_the_plane_at_zero_at_the_window_corner`.
pub(crate) fn fit_gamma_s_for_planes(
    planes: &mut [RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
    s_max: f64,
) {
    let grid = super::grid::build_grid(pf, bounds);
    let points: Vec<(f64, f64)> = grid
        .v_points
        .iter()
        .flat_map(|&v| grid.q_points.iter().map(move |&q| (v, q)))
        .collect();
    let slopes: Vec<f64> = points
        .iter()
        .map(|&(v, q)| anchored_slope(pf, v, q, s_max))
        .collect();
    let envelope: Vec<f64> = points
        .iter()
        .map(|&(v, q)| {
            planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    for plane in planes.iter_mut() {
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for ((&(v, q), &slope), &min_val) in points.iter().zip(&slopes).zip(&envelope) {
            if plane.evaluate(v, q, 0.0) - min_val <= 1e-8 {
                sum += slope;
                count += 1;
            }
        }
        let mean = if count == 0 {
            0.0
        } else {
            sum / f64::from(count)
        };
        let limited = if s_max > 0.0 {
            mean.max((-(plane.gamma_0 + plane.gamma_v * bounds.v_min) / s_max).min(0.0))
        } else {
            mean
        };
        // Snap BOTH signs to 0 (see GAMMA_S_SNAP_EPS): `limited.min(0.0)` would let a
        // tiny NEGATIVE residual survive into the spillage column.
        plane.gamma_s = if limited > -GAMMA_S_SNAP_EPS {
            0.0
        } else {
            limited
        };
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]
mod tests {
    use novomodelo_core::{EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel};
    use novomodelo_io::extensions::HydroGeometryRow;
    use novomodelo_io::validation::semantic::envelope_tolerance;

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::{K, ProductionFunction, TailraceSource};
    use super::{GAMMA_S_SNAP_EPS, N_LATERAL_FLOW_SAMPLES, anchored_slope, fit_gamma_s_for_planes};

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    fn small_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 0.0,
            v_max: 30_000.0,
            n_volume_points: 3,
            n_flow_points: 3,
            single_volume: false,
        }
    }

    fn window_bounds() -> FittingBounds {
        FittingBounds {
            v_min: 10_000.0,
            ..small_bounds()
        }
    }

    /// A production function whose tailrace rises with total outflow, so lateral
    /// flow lowers net head. The linear tailrace gives a hand-computable secant
    /// slope (`dh_tail/dq_out = 0.001`).
    fn spill_sensitive_pf() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "SpillSensitive").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 0.0 }),
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            3_000.0,
            "SpillSensitive".to_owned(),
        )
    }

    /// A production function whose tailrace rises with outflow at a decreasing
    /// rate, so `generation` is curved in `lateral_flow`: the zero-anchored slope
    /// and a free-intercept OLS slope differ.
    fn curved_tailrace_pf() -> ProductionFunction {
        let rows = vec![
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ];
        let forebay = ForebayTable::new(&rows, "Curved").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.002, -1e-7],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            Some(&HydraulicLossesModel::Constant { value_m: 0.0 }),
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            3_000.0,
            "Curved".to_owned(),
        )
    }

    fn grid_points(pf: &ProductionFunction, bounds: &FittingBounds) -> Vec<(f64, f64)> {
        let grid = super::super::grid::build_grid(pf, bounds);
        grid.v_points
            .iter()
            .flat_map(|&v| grid.q_points.iter().map(move |&q| (v, q)))
            .collect()
    }

    fn lateral_flow_samples(s_max: f64) -> Vec<f64> {
        let denom = (N_LATERAL_FLOW_SAMPLES - 1) as f64;
        (0..N_LATERAL_FLOW_SAMPLES)
            .map(|i| i as f64 * s_max / denom)
            .collect()
    }

    /// A flat-tailrace production function: net head is independent of outflow, so
    /// `generation` is flat in `lateral_flow` and the secant slope is exactly 0.
    fn flat_tailrace_pf() -> ProductionFunction {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 420.0)];
        let forebay = ForebayTable::new(&rows, "Flat").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![10.0],
        };
        ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            None,
            Some(&EfficiencyModel::Constant { value: 1.0 }),
            3_000.0,
            "Flat".to_owned(),
        )
    }

    /// A plane active at every grid point gets the mean, over those points, of the
    /// hand-computed zero-anchored slope `Σ sᵢ·(g(sᵢ) − g(0)) / Σ sᵢ²`, and on a
    /// curved tailrace that differs from the free-intercept OLS slope.
    #[test]
    fn gamma_s_is_mean_zero_anchored_slope_over_active_points() {
        let pf = curved_tailrace_pf();
        let bounds = small_bounds();
        let s_max = 6_000.0;
        let xs = lateral_flow_samples(s_max);

        let mut anchored = Vec::new();
        let mut ols = Vec::new();
        for (v, q) in grid_points(&pf, &bounds) {
            let ys: Vec<f64> = xs.iter().map(|&x| pf.evaluate(v, q, x)).collect();
            let cross: f64 = xs.iter().zip(&ys).map(|(&x, &y)| x * (y - ys[0])).sum();
            let square: f64 = xs.iter().map(|&x| x * x).sum();
            anchored.push(cross / square);

            let n = xs.len() as f64;
            let mean_x = xs.iter().sum::<f64>() / n;
            let mean_y = ys.iter().sum::<f64>() / n;
            let cov: f64 = xs
                .iter()
                .zip(&ys)
                .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
                .sum();
            let var_x: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
            ols.push(cov / var_x);
        }
        let expected = anchored.iter().sum::<f64>() / anchored.len() as f64;
        let free_intercept = ols.iter().sum::<f64>() / ols.len() as f64;

        // A steep, flow-independent bound: the single plane is the argmin at every
        // grid point.
        let mut planes = vec![RawPlane {
            gamma_0: 1_000.0,
            gamma_v: 0.0,
            gamma_q: 1_000.0,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, s_max);

        assert!(
            (planes[0].gamma_s - expected).abs() < 1e-12,
            "gamma_s={} must equal the mean zero-anchored slope {expected}",
            planes[0].gamma_s
        );
        assert!(planes[0].gamma_s < 0.0);
        assert!(
            (expected - free_intercept).abs() > 1e-4 * expected.abs(),
            "fixture must separate the anchored slope {expected} from the free-intercept \
             slope {free_intercept}"
        );
    }

    /// The fit range changes `γ_S`: a wider `[0, S_max]` on a tailrace that
    /// flattens with outflow yields a gentler secant.
    #[test]
    fn wider_spillage_range_yields_gentler_gamma_s() {
        let pf = curved_tailrace_pf();
        let bounds = small_bounds();
        let plane = RawPlane {
            gamma_0: 1_000.0,
            gamma_v: 0.0,
            gamma_q: 1_000.0,
            gamma_s: 0.0,
        };
        let mut narrow = vec![plane];
        let mut wide = vec![plane];
        fit_gamma_s_for_planes(&mut narrow, &pf, &bounds, 6_000.0);
        fit_gamma_s_for_planes(&mut wide, &pf, &bounds, 9_000.0);
        assert!(
            wide[0].gamma_s.abs() < narrow[0].gamma_s.abs(),
            "wide |gamma_s|={} must be below narrow |gamma_s|={}",
            wide[0].gamma_s.abs(),
            narrow[0].gamma_s.abs()
        );
    }

    /// With no tailrace-flow declaration the lateral flow is the own spill: the tailrace
    /// is read at `q + lateral_flow`, so a rising tailrace yields a negative slope.
    #[test]
    fn default_axis_is_own_spill_qhat_zero() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 1_000.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];

        let s_max = 2.0 * pf.max_turbined_m3s;
        let (v, q) = (15_000.0, 1_500.0);
        assert!(
            pf.evaluate(v, q, s_max) < pf.evaluate(v, q, 0.0),
            "own-spill axis: generation at lateral_flow=S_max must be below the reference lateral flow=0 anchor"
        );

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, s_max);
        assert!(planes[0].gamma_s < 0.0);
    }

    /// The linear tailrace gives `∂g/∂L = −K·η·q·a₁` at any `own_outflow_factor`: the
    /// factor scales `q` inside the tailrace argument, so it never enters the slope in
    /// `L`.
    #[test]
    fn secant_slope_is_per_unit_lateral_flow_for_any_own_factor() {
        let (efficiency, tailrace_slope) = (0.9, 0.001);
        let (v, q, s_max) = (15_000.0, 1_500.0, 6_000.0);
        let expected = -K * efficiency * q * tailrace_slope;

        for own_outflow_factor in [1.0, 1.5] {
            let pf = spill_sensitive_pf().with_own_outflow_factor(own_outflow_factor);
            assert!(
                pf.net_head(v, q, s_max) > 0.0,
                "f={own_outflow_factor}: the grid point must keep the net head positive"
            );
            let slope = anchored_slope(&pf, v, q, s_max);
            assert!(
                ((slope - expected) / expected).abs() < 1e-9,
                "f={own_outflow_factor}: slope={slope} must equal the closed form {expected}"
            );
        }
    }

    /// derived: the plane `{γ₀ 0, γᵥ 0.001, γq 0.12}` reads `γ₀ + γᵥ·v_min = 10` MW at
    /// the window corner, so the limit binds at `γₛ = −10 / 6 000` and the corner
    /// `10 + γₛ·6 000` is 0. The twin with `γ₀ = 1 000` has the same argmin set and
    /// its own limit `−1 010 / 6 000` is below its mean, so it keeps the hand mean of
    /// the zero-anchored slopes; that mean lies below the first plane's bound, so the
    /// limit was the binding term for the first plane.
    #[test]
    fn gamma_s_limit_puts_the_plane_at_zero_at_the_window_corner() {
        let pf = spill_sensitive_pf();
        let bounds = window_bounds();
        let s_max = 6_000.0;
        let (gamma_0, gamma_v, v_min) = (0.0, 0.001, 10_000.0);

        let mut planes = vec![RawPlane {
            gamma_0,
            gamma_v,
            gamma_q: 0.12,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, s_max);

        let bound = -(gamma_0 + gamma_v * v_min) / s_max;
        let gamma_s = planes[0].gamma_s;
        assert!(
            (gamma_s - bound).abs() <= 4.0 * f64::EPSILON * bound.abs(),
            "gamma_s={gamma_s} must sit at the corner limit {bound}"
        );
        let corner = gamma_0 + gamma_v * v_min + gamma_s * s_max;
        assert!(
            corner.abs() <= envelope_tolerance(10.0),
            "corner={corner} must be 0"
        );

        let mut twin = vec![RawPlane {
            gamma_0: 1_000.0,
            ..planes[0]
        }];
        fit_gamma_s_for_planes(&mut twin, &pf, &bounds, s_max);

        let xs = lateral_flow_samples(s_max);
        let slopes: Vec<f64> = grid_points(&pf, &bounds)
            .into_iter()
            .map(|(v, q)| {
                let ys: Vec<f64> = xs.iter().map(|&x| pf.evaluate(v, q, x)).collect();
                let cross: f64 = xs.iter().zip(&ys).map(|(&x, &y)| x * (y - ys[0])).sum();
                let square: f64 = xs.iter().map(|&x| x * x).sum();
                cross / square
            })
            .collect();
        let mean = slopes.iter().sum::<f64>() / slopes.len() as f64;
        assert!(
            (twin[0].gamma_s - mean).abs() < 1e-12,
            "twin gamma_s={} must keep the mean {mean}",
            twin[0].gamma_s
        );
        assert!(
            twin[0].gamma_s < bound,
            "twin gamma_s={} must lie below {bound}, or the limit never bound",
            twin[0].gamma_s
        );
    }

    /// derived: the plane `{γ₀ 1e-12, γᵥ 0, γq 0.12}` has limit `−1e-12 / 6 000 ≈
    /// −1.7e-16`, above `−GAMMA_S_SNAP_EPS`, so the snap maps it to +0.0; a limit
    /// applied after the snap leaves that residual in the spillage column. The origin
    /// plane `{0, 0, 0.12}` has limit 0 and ends at +0.0.
    #[test]
    fn gamma_s_limit_runs_before_the_snap() {
        let pf = spill_sensitive_pf();
        let bounds = window_bounds();
        let s_max = 6_000.0;
        let residual_bound = -1e-12 / s_max;
        assert!(
            residual_bound < 0.0 && residual_bound > -GAMMA_S_SNAP_EPS,
            "fixture must put the limit inside the snap band, got {residual_bound}"
        );

        for gamma_0 in [1e-12, 0.0] {
            let mut planes = vec![RawPlane {
                gamma_0,
                gamma_v: 0.0,
                gamma_q: 0.12,
                gamma_s: 0.0,
            }];
            fit_gamma_s_for_planes(&mut planes, &pf, &bounds, s_max);
            assert_eq!(
                planes[0].gamma_s.to_bits(),
                0.0_f64.to_bits(),
                "gamma_0={gamma_0}: gamma_s={} must be +0.0",
                planes[0].gamma_s
            );
        }
    }

    /// Flat tailrace → flat `generation` in `lateral_flow` → secant slope exactly 0 (the neutral
    /// path for `Σ(x−x̄)(y−ȳ) = 0`).
    #[test]
    fn flat_tailrace_yields_zero_slope() {
        let pf = flat_tailrace_pf();
        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.0005,
            gamma_q: 0.2,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 2.0 * pf.max_turbined_m3s);
        assert_eq!(
            planes[0].gamma_s, 0.0,
            "flat tailrace must yield gamma_s = 0"
        );
    }

    /// Degenerate `S_max = 0` → neutral `γ_S = 0`, no divide-by-zero.
    #[test]
    fn degenerate_s_max_zero_is_neutral() {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 420.0)];
        let forebay = ForebayTable::new(&rows, "ZeroTurb").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001],
        };
        let pf = ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            None,
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            0.0,
            "ZeroTurb".to_owned(),
        );

        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
        assert_eq!(
            planes[0].gamma_s.to_bits(),
            0.0_f64.to_bits(),
            "S_max = 0 must yield neutral gamma_s"
        );
    }

    /// Every fitted plane satisfies the `γ_s ≤ 1e-10` sign rule on a multi-plane,
    /// spill-sensitive fixture (the validation contract `validate_fitted_planes`
    /// enforces downstream).
    #[test]
    fn all_planes_satisfy_sign_rule() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        let mut planes = vec![
            RawPlane {
                gamma_0: 0.0,
                gamma_v: 0.001,
                gamma_q: 0.12,
                gamma_s: 0.0,
            },
            RawPlane {
                gamma_0: 50.0,
                gamma_v: 0.0008,
                gamma_q: 0.08,
                gamma_s: 0.0,
            },
        ];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 2.0 * pf.max_turbined_m3s);
        for plane in &planes {
            assert!(
                plane.gamma_s <= 1e-10,
                "every plane must satisfy gamma_s <= 1e-10, got {}",
                plane.gamma_s
            );
        }
    }

    /// By-plane anchor regression: the high-`γ_q` (low-flow) plane must receive a
    /// GENTLER `γ_S` than the low-`γ_q` (high-flow) plane, because `|γ_S| ≈
    /// ρ·q_rep·|dh_tail/dq_out|` scales with the anchor's flow and the argmin anchor
    /// puts the high-`γ_q` plane at a small `q_rep`. The argmax anchor inverts this
    /// (over-steepens `γ_S`), so this test guards the anchor choice directly — the
    /// single-plane tests cannot, since one plane is argmin and argmax everywhere.
    #[test]
    fn high_gamma_q_plane_receives_gentler_gamma_s_than_low_gamma_q() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        // Grid q_points = [0, 1500, 3000]. The two planes cross at q = 1500 with a
        // shared γ_v, so the crossover is purely in q (independent of v): the high-γ_q
        // plane is the binding (min) bound for q ≤ 1500, the low-γ_q plane for q ≥ 1500.
        let mut planes = vec![
            RawPlane {
                gamma_0: 1_000.0,
                gamma_v: 0.001,
                gamma_q: 0.12, // high-γ_q (high head, governs low flow) — index 0
                gamma_s: 0.0,
            },
            RawPlane {
                gamma_0: 1_060.0,
                gamma_v: 0.001,
                gamma_q: 0.08, // low-γ_q (governs high flow) — index 1
                gamma_s: 0.0,
            },
        ];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 2.0 * pf.max_turbined_m3s);

        let high_gamma_q = planes[0].gamma_s;
        let low_gamma_q = planes[1].gamma_s;
        assert!(
            high_gamma_q < 0.0 && low_gamma_q < 0.0,
            "both rising-tailrace secants must be negative; got high={high_gamma_q}, low={low_gamma_q}"
        );
        assert!(
            high_gamma_q.abs() < low_gamma_q.abs(),
            "high-γ_q plane must get the GENTLER γ_S (argmin anchor); got |high|={} >= |low|={}",
            high_gamma_q.abs(),
            low_gamma_q.abs()
        );
    }
}
