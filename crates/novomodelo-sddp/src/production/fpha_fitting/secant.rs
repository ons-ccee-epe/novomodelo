//! Per-plane lateral-flow secant `γ_S` on the `lateral_flow` axis.
//!
//! Fits one secant `γ_S` per plane by 1-D ordinary-least-squares of `generation`
//! against `lateral_flow ∈ [0, S_max]` at the plane's representative operating
//! point — NOT a hull/grid axis. The sample is a closed-form evenly-spaced grid,
//! so the fit is deterministic and order-independent.
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

/// Number of evenly-spaced `lateral_flow` samples for the per-plane secant fit.
///
/// Fixed (not config-driven) so the secant is deterministic and order-independent.
const N_LATERAL_FLOW_SAMPLES: usize = 9;

/// Magnitude below which a fitted `γ_S` slope is snapped to exactly 0.
///
/// The OLS fit of a flat tailrace leaves floating-point residuals of EITHER sign;
/// a surviving near-zero structural coefficient in the spillage column wrecks LP
/// column scaling, so both signs MUST snap to `0.0`. The bound sits well below the
/// smallest physical spillage sensitivity (~`1e-6` MW/(m³/s)), so no real
/// coefficient is snapped, and at/above the `validate_fitted_planes` sign
/// tolerance so every snapped value is sign-valid.
const GAMMA_S_SNAP_EPS: f64 = 1e-10;

/// Resolve the lateral-flow sample upper bound `S_max` for the secant fit:
/// `2 × long-term mean inflow` when positive, else `2 × max_turbined_m3s`.
///
/// # Contract — `long-term mean inflow > 0` strictly, else fall back
///
/// The test is strict: a history-less hydro yields `long_term_mean_inflow_m3s =
/// 0.0` and MUST take the fallback. `>= 0.0` would collapse `S_max = 0` and zero
/// out every `γ_S` (the degenerate guard in [`fit_gamma_s`]), silently dropping
/// the lateral correction.
pub(crate) fn resolve_s_max(long_term_mean_inflow_m3s: f64, max_turbined_m3s: f64) -> f64 {
    if long_term_mean_inflow_m3s > 0.0 {
        2.0 * long_term_mean_inflow_m3s
    } else {
        2.0 * max_turbined_m3s
    }
}

/// Representative operating point `(V, q)` at which a plane's `γ_S` secant is fit.
///
/// # Contract — anchor where the plane binds (argmin), not where it is loosest
///
/// Each plane anchors at the spill = 0 grid point where it is the *active* binding
/// bound — the pointwise **minimum** over planes (the LP applies `g ≤ plane_k`),
/// matching the min-envelope contract in `compute_alpha_fpha` — and attains the
/// largest `generation` among the points it governs. Anchoring at the pointwise
/// **maximum** (the loosest bound) is the wrong-but-compiling alternative: it
/// anchors each plane where it does not govern, over-steepening `γ_S` on high-head
/// planes and zeroing it on planes that never attain the max.
///
/// Determinism: the closed-form `build_grid` grid scanned in fixed `(v, q)`
/// order with a strict `>` test resolves ties to the first point.
///
/// Returns `None` for a plane never active on the grid; the caller leaves
/// `γ_S = 0`.
fn representative_operating_point(
    plane: &RawPlane,
    planes: &[RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
) -> Option<(f64, f64)> {
    let grid = super::grid::build_grid(pf, bounds);

    let mut best: Option<(f64, f64, f64)> = None; // (v, q, generation) at the binding point
    for &v in &grid.v_points {
        for &q in &grid.q_points {
            let this_val = plane.evaluate(v, q, 0.0);
            let min_val = planes
                .iter()
                .map(|p| p.evaluate(v, q, 0.0))
                .fold(f64::INFINITY, f64::min);

            // Active iff this plane is the binding bound: the argmin (the LP min),
            // not the argmax.
            if this_val - min_val <= 1e-8 {
                let generation = pf.evaluate(v, q, 0.0);
                if best.is_none_or(|(_, _, best_gh)| generation > best_gh) {
                    best = Some((v, q, generation));
                }
            }
        }
    }

    best.map(|(v, q, _)| (v, q))
}

/// Fit the lateral-flow secant `γ_S` for a single plane at `(v_rep, q_rep)` — the
/// OLS slope of `generation` vs `lateral_flow` over a fixed sample of `[0, S_max]`.
///
/// # Contract — `γ_S ≤ 0`
///
/// More lateral flow raises the tailrace and lowers `generation`, so the slope is
/// non-positive; floating-point noise is clamped to `0.0` to satisfy the
/// `validate_fitted_planes` rule `gamma_s ≤ 1e-10`. Returning the raw slope is the
/// wrong-but-compiling alternative — a `+1e-15` round-off would surface as an
/// invalid coefficient downstream.
///
/// # Contract — degenerate `S_max ≤ 0` returns the neutral `γ_S = 0`
///
/// `S_max ≤ 0` collapses the sample to one point (`Σ(x_i − x̄)² = 0`, slope
/// undefined); return `0.0` rather than dividing by zero. The same neutral path
/// covers a flat `generation`.
fn fit_gamma_s(pf: &ProductionFunction, v_rep: f64, q_rep: f64, s_max: f64) -> f64 {
    if s_max <= 0.0 {
        return 0.0;
    }

    #[allow(clippy::cast_possible_truncation)]
    let denom = f64::from((N_LATERAL_FLOW_SAMPLES - 1) as u32);

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for i in 0..N_LATERAL_FLOW_SAMPLES {
        #[allow(clippy::cast_possible_truncation)]
        let lateral_flow = f64::from(i as u32) * s_max / denom;
        let generation = pf.evaluate(v_rep, q_rep, lateral_flow);
        sum_x += lateral_flow;
        sum_y += generation;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = N_LATERAL_FLOW_SAMPLES as f64;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    for i in 0..N_LATERAL_FLOW_SAMPLES {
        #[allow(clippy::cast_possible_truncation)]
        let lateral_flow = f64::from(i as u32) * s_max / denom;
        let generation = pf.evaluate(v_rep, q_rep, lateral_flow);
        let dx = lateral_flow - mean_x;
        cov += dx * (generation - mean_y);
        var_x += dx * dx;
    }

    if var_x == 0.0 {
        return 0.0;
    }

    let slope = cov / var_x;
    // Snap BOTH signs to 0 (see GAMMA_S_SNAP_EPS): `slope.min(0.0)` would let a
    // tiny NEGATIVE residual survive into the spillage column.
    if slope > -GAMMA_S_SNAP_EPS {
        0.0
    } else {
        slope
    }
}

/// Fit `γ_S` in place for every plane, using the smart default `lateral_flow = S`.
///
/// `γ_S` is the only coefficient this step sets; the α-scaled gradients and
/// intercept are untouched. The secant is fit for every plant (spillage lowers net
/// head universally); only the degenerate `S_max ≤ 0` guard in [`fit_gamma_s`]
/// yields `γ_S = 0`. `long_term_mean_inflow_m3s = 0.0` selects the
/// `2 × max_turbined` fallback (see [`resolve_s_max`]).
pub(crate) fn fit_gamma_s_for_planes(
    planes: &mut [RawPlane],
    pf: &ProductionFunction,
    bounds: &FittingBounds,
    long_term_mean_inflow_m3s: f64,
) {
    let s_max = resolve_s_max(long_term_mean_inflow_m3s, pf.max_turbined_m3s);

    // Snapshot before mutation so the active-point selection is independent of fit
    // order.
    let snapshot: Vec<RawPlane> = planes.to_vec();
    for plane in planes.iter_mut() {
        plane.gamma_s = match representative_operating_point(plane, &snapshot, pf, bounds) {
            Some((v_rep, q_rep)) => fit_gamma_s(pf, v_rep, q_rep, s_max),
            None => 0.0,
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
    use cobre_core::{EfficiencyModel, EntityId, HydraulicLossesModel, TailraceModel};
    use cobre_io::extensions::HydroGeometryRow;

    use super::super::geometry::{FittingBounds, ForebayTable};
    use super::super::hull_fit::RawPlane;
    use super::super::production::{ProductionFunction, TailraceSource};
    use super::{
        N_LATERAL_FLOW_SAMPLES, fit_gamma_s_for_planes, representative_operating_point,
        resolve_s_max,
    };

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

    /// `resolve_s_max`: long-term mean inflow > 0 ⇒ 2·long-term mean inflow; long-term mean inflow = 0 ⇒ 2·max_turbined (both branches).
    #[test]
    fn s_max_resolution_picks_mlt_then_falls_back() {
        // long-term mean inflow > 0 branch.
        assert_eq!(resolve_s_max(1_500.0, 3_000.0), 3_000.0);
        // long-term mean inflow = 0 fallback branch.
        assert_eq!(resolve_s_max(0.0, 3_000.0), 6_000.0);
        // Negative/garbage long-term mean inflow also falls back (treated as not-supplied).
        assert_eq!(resolve_s_max(-1.0, 2_000.0), 4_000.0);
    }

    /// The fitted `γ_S` equals the hand-computed OLS secant slope of `generation` vs
    /// `lateral_flow` over `[0, S_max]` to within 1e-10, and is strictly negative.
    #[test]
    fn gamma_s_matches_hand_computed_ols_slope() {
        let pf = spill_sensitive_pf();
        let bounds = small_bounds();

        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];

        let (v_rep, q_rep) = representative_operating_point(&planes[0], &planes, &pf, &bounds)
            .expect("active plane");

        let s_max = resolve_s_max(0.0, pf.max_turbined_m3s);
        assert_eq!(s_max, 6_000.0);

        // Independent OLS slope over the same fixed lateral_flow sample.
        let denom = (N_LATERAL_FLOW_SAMPLES - 1) as f64;
        let xs: Vec<f64> = (0..N_LATERAL_FLOW_SAMPLES)
            .map(|i| i as f64 * s_max / denom)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| pf.evaluate(v_rep, q_rep, x)).collect();
        let n = N_LATERAL_FLOW_SAMPLES as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let cov: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum();
        let var_x: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
        let expected = cov / var_x;

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);

        assert!(
            (planes[0].gamma_s - expected).abs() < 1e-10,
            "gamma_s={} must equal hand-computed OLS slope {expected}",
            planes[0].gamma_s
        );
        assert!(
            planes[0].gamma_s < 0.0,
            "rising tailrace must yield a strictly negative gamma_s, got {}",
            planes[0].gamma_s
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

        // Own-spill axis: generation at lateral_flow = 0 exceeds generation at S_max.
        let (v_rep, q_rep) = representative_operating_point(&planes[0], &planes, &pf, &bounds)
            .expect("active plane");
        let s_max = resolve_s_max(0.0, pf.max_turbined_m3s);
        let gh_anchor = pf.evaluate(v_rep, q_rep, 0.0);
        let gh_smax = pf.evaluate(v_rep, q_rep, s_max);
        assert!(
            gh_smax < gh_anchor,
            "own-spill axis: generation at lateral_flow=S_max must be below the reference lateral flow=0 anchor"
        );

        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
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
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
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
        assert_eq!(resolve_s_max(0.0, pf.max_turbined_m3s), 0.0);

        let bounds = small_bounds();
        let mut planes = vec![RawPlane {
            gamma_0: 0.0,
            gamma_v: 0.001,
            gamma_q: 0.1,
            gamma_s: 0.0,
        }];
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
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
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);
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
        fit_gamma_s_for_planes(&mut planes, &pf, &bounds, 0.0);

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
