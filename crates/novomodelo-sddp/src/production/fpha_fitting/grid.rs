//! Single authoritative owner of the uniform `(V, Q)` FPHA fitting grid.
//!
//! The grid formula lives here only; every caller (`hull_fit` cloud builder,
//! `alpha` regression, `secant` scan) reaches it via `build_grid`. A second copy
//! of the axis formula would let the cloud grid and the regression grid drift
//! apart, silently invalidating the outer-approximation guarantee.

use super::geometry::FittingBounds;
use super::production::ProductionFunction;

/// Precomputed axis values for the fitting `(V, Q)` grid, built once by
/// [`build_grid`].
#[allow(clippy::struct_field_names)]
pub(crate) struct GridParams {
    /// Volume axis (hm³).
    pub(crate) v_points: Vec<f64>,
    /// Turbined-flow axis (m³/s).
    pub(crate) q_points: Vec<f64>,
}

/// Build the uniform `(V, Q)` grid for FPHA fitting (both axes inclusive at both
/// endpoints).
///
/// The flow axis starts at `q = 0` (zero production): that column anchors the
/// cloud at the zero-flow origin and forms its lower closure, so `hull_fit` needs
/// no synthetic closing point. Spillage is not a grid axis — the cloud and the α
/// regression fix `s = 0`, and the lateral secant sweeps its own `S_max` sample
/// independent of `bounds.n_spillage_points`.
pub(crate) fn build_grid(pf: &ProductionFunction, bounds: &FittingBounds) -> GridParams {
    let n_v = bounds.n_volume_points;
    let n_q = bounds.n_flow_points;

    let v_range = bounds.v_max - bounds.v_min;
    #[allow(clippy::cast_possible_truncation)]
    let v_denom = f64::from((n_v - 1) as u32);

    let q_min = 0.0_f64;
    let q_range = pf.max_turbined_m3s - q_min;
    #[allow(clippy::cast_possible_truncation)]
    let q_denom = f64::from((n_q - 1) as u32);

    #[allow(clippy::cast_possible_truncation)]
    let v_points: Vec<f64> = (0..n_v)
        .map(|i| bounds.v_min + f64::from(i as u32) * v_range / v_denom)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let q_points: Vec<f64> = (0..n_q)
        .map(|j| q_min + f64::from(j as u32) * q_range / q_denom)
        .collect();

    GridParams { v_points, q_points }
}
