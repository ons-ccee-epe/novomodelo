//! Per-plane lateral-flow secant `γ_S` on the `lateral_flow` axis.
//!
//! Each spill = 0 grid point `(v, q)` gets a zero-anchored least-squares slope of
//! `generation` against `lateral_flow ∈ [0, S_max]`; a plane's `γ_S` is the mean
//! of that slope over the grid points where the plane is the binding bound. The
//! sample is a closed-form evenly-spaced grid, so the fit is deterministic and
//! order-independent.
//!
//! # The smart default (`lateral_flow = S`, `reference lateral flow = 0`)
//!
//! The lateral axis collapses to own spill: feeding the lateral flow as the
//! spillage argument makes the tailrace see `outflow = q + lateral_flow` (it
//! already evaluates at `q_out = q + s`), exactly the spill-only default. Richer
//! `lateral_flow` composition and its LP wiring are a later refinement.

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

/// Resolve the lateral-flow sample upper bound `S_max` for the secant fit: the
/// input `spillage_fit_max_m3s` when given, else `2 × long-term mean inflow` when
/// positive, else `2 × max_turbined_m3s`.
///
/// # Contract — `long-term mean inflow > 0` strictly, else fall back
///
/// The test is strict: a history-less hydro yields `long_term_mean_inflow_m3s =
/// 0.0` and MUST take the fallback. `>= 0.0` would collapse `S_max = 0` and zero
/// out every `γ_S` (the degenerate guard in [`anchored_slope`]), silently dropping
/// the lateral correction.
pub(crate) fn resolve_s_max(
    spillage_fit_max_m3s: Option<f64>,
    long_term_mean_inflow_m3s: f64,
    max_turbined_m3s: f64,
) -> f64 {
    if let Some(s_max) = spillage_fit_max_m3s {
        s_max
    } else if long_term_mean_inflow_m3s > 0.0 {
        2.0 * long_term_mean_inflow_m3s
    } else {
        2.0 * max_turbined_m3s
    }
}

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

/// Fit `γ_S` in place for every plane, using the smart default `lateral_flow = S`.
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
        // Snap BOTH signs to 0 (see GAMMA_S_SNAP_EPS): `mean.min(0.0)` would let a
        // tiny NEGATIVE residual survive into the spillage column.
        plane.gamma_s = if mean > -GAMMA_S_SNAP_EPS { 0.0 } else { mean };
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

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::{ProductionFunction, TailraceSource};
    use super::{N_LATERAL_FLOW_SAMPLES, fit_gamma_s_for_planes, resolve_s_max};

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
            n_spillage_points: 2,
            max_planes_per_hydro: 10,
            single_volume: false,
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

    /// `resolve_s_max`: an input `spillage_fit_max_m3s` wins; otherwise
    /// long-term mean inflow > 0 ⇒ 2·long-term mean inflow, else 2·max_turbined.
    #[test]
    fn s_max_resolution_picks_input_then_mlt_then_falls_back() {
        assert_eq!(resolve_s_max(Some(42_000.0), 1_500.0, 3_000.0), 42_000.0);
        assert_eq!(resolve_s_max(Some(42_000.0), 0.0, 3_000.0), 42_000.0);
        assert_eq!(resolve_s_max(None, 1_500.0, 3_000.0), 3_000.0);
        assert_eq!(resolve_s_max(None, 0.0, 3_000.0), 6_000.0);
        // Negative/garbage long-term mean inflow also falls back (treated as not-supplied).
        assert_eq!(resolve_s_max(None, -1.0, 2_000.0), 4_000.0);
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

        // Zero intercept and a steep, flow-independent bound: the single plane is
        // the argmin at every grid point.
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
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
            gamma_0: 0.0,
            gamma_v: 0.0,
            gamma_q: 1_000.0,
            gamma_s: 0.0,
        };
        let mut narrow = vec![plane];
        let mut wide = vec![plane];
        fit_gamma_s_for_planes(&mut narrow, &pf, &bounds, resolve_s_max(None, 0.0, 3_000.0));
        fit_gamma_s_for_planes(
            &mut wide,
            &pf,
            &bounds,
            resolve_s_max(Some(9_000.0), 0.0, 3_000.0),
        );
        assert!(
            wide[0].gamma_s.abs() < narrow[0].gamma_s.abs(),
            "wide |gamma_s|={} must be below narrow |gamma_s|={}",
            wide[0].gamma_s.abs(),
            narrow[0].gamma_s.abs()
        );
    }

    /// Default axis is own-spill only: the secant uses `lateral_flow = S` (the spillage
    /// argument of `evaluate`) with `reference lateral flow = 0` (the lateral_flow = 0 anchor). Feeding the
    /// lateral flow as the spillage argument is exactly what the tailrace sees as
    /// `q + lateral_flow`, so a rising tailrace yields a negative slope.
    #[test]
    fn default_axis_is_own_spill_qhat_zero() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];

        let s_max = resolve_s_max(None, 0.0, pf.max_turbined_m3s);
        let (v, q) = (15_000.0, 1_500.0);
        assert!(
            pf.evaluate(v, q, s_max) < pf.evaluate(v, q, 0.0),
            "own-spill axis: generation at lateral_flow=S_max must be below the reference lateral flow=0 anchor"
        );

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, s_max);
        assert!(planes[0].gamma_s < 0.0);
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
        fit_gamma_s_for_planes(
            &mut planes,
            &pf,
            &bounds,
            resolve_s_max(None, 0.0, pf.max_turbined_m3s),
        );
        assert_eq!(
            planes[0].gamma_s, 0.0,
            "flat tailrace must yield gamma_s = 0"
        );
    }

    /// Degenerate `S_max == 0` (long-term mean inflow = 0 AND max_turbined = 0) → neutral `γ_S = 0`,
    /// no divide-by-zero.
    #[test]
    fn degenerate_s_max_zero_is_neutral() {
        let rows = vec![row(0.0, 400.0), row(30_000.0, 420.0)];
        let forebay = ForebayTable::new(&rows, "ZeroTurb").expect("valid VHA curve");
        let tailrace = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001],
        };
        // max_turbined = 0 → with long-term mean inflow = 0, S_max = 0.
        let pf = ProductionFunction::new(
            forebay,
            TailraceSource::Entity(Some(tailrace)),
            None,
            Some(&EfficiencyModel::Constant { value: 0.9 }),
            0.0,
            "ZeroTurb".to_owned(),
        );
        assert_eq!(resolve_s_max(None, 0.0, pf.max_turbined_m3s), 0.0);

        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(
            &mut planes,
            &pf,
            &bounds,
            resolve_s_max(None, 0.0, pf.max_turbined_m3s),
        );
        assert_eq!(
            planes[0].gamma_s, 0.0,
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
        fit_gamma_s_for_planes(
            &mut planes,
            &pf,
            &bounds,
            resolve_s_max(None, 0.0, pf.max_turbined_m3s),
        );
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
                gamma_0: 0.0,
                gamma_v: 0.001,
                gamma_q: 0.12, // high-γ_q (high head, governs low flow) — index 0
                gamma_s: 0.0,
            },
            RawPlane {
                gamma_0: 60.0,
                gamma_v: 0.001,
                gamma_q: 0.08, // low-γ_q (governs high flow) — index 1
                gamma_s: 0.0,
            },
        ];
        fit_gamma_s_for_planes(
            &mut planes,
            &pf,
            &bounds,
            resolve_s_max(None, 0.0, pf.max_turbined_m3s),
        );

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
