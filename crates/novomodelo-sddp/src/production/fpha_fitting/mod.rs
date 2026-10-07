//! FPHA hyperplane fitting from reservoir geometry.
//!
//! Turns a hydro plant's Volume-Height-Area (VHA) curve into a set of
//! hyperplanes approximating the production function `phi(v, q, s)`. The pipeline
//! evaluates the production function on a uniform `(V, Q)` grid (at spillage = 0),
//! takes the 3-D convex hull of the resulting `(V, Q, generation)` cloud, and reads its
//! upper-envelope facets as a piecewise-linear concave outer approximation.
//!
//! # Submodule layout
//!
//! `grid` is the single authoritative owner of the uniform grid formula
//! (`GridParams` + `build_grid`), shared by the hull cloud, the α regression, the
//! deviation diagnostic, and the secant's representative-point scan. `deviation`
//! is the fit-quality measure that replaces the retired `kappa` shrink. The
//! remaining submodules (`error`, `geometry`, `production`, `hull_fit`, `alpha`,
//! `secant`, `tailrace`, `reduction`, `rng`, `selection`) own one pipeline stage
//! each, documented at their own module headers.
//!
//! The orchestration entry point `fit_fpha_planes` and its result `FphaFitResult`
//! live here in `mod`, co-located with the re-export surface.

use cobre_core::Hydro;
use cobre_io::extensions::{FphaColumnLayout, HydroGeometryRow};

use crate::hydro_models::FphaPlane;

mod alpha;
mod deviation;
mod error;
mod geometry;
mod grid;
mod hull_fit;
mod production;
mod reduction;
mod rng;
mod secant;
mod selection;
mod tailrace;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::similar_names
)]
mod tests;

use cobre_io::extensions::PlaneReductionConfig;

use alpha::{compute_alpha_fpha, scale_plane_affine};
use deviation::{collect_fit_deviation_points, compute_fit_deviation};
use geometry::resolve_fitting_bounds;
use hull_fit::{RawPlane, fit_hull_planes};
use production::ProductionFunction;
use reduction::reduce_planes;
use secant::fit_gamma_s_for_planes;
use selection::validate_fitted_planes;

pub(crate) use deviation::{FphaDeviationPoint, FphaFitDeviation};
pub(crate) use error::FphaFittingError;
pub(crate) use geometry::{ForebayTable, evaluate_losses, evaluate_tailrace};
pub(crate) use production::TailraceSource;
pub(crate) use tailrace::{TailraceFamilies, build_tailrace_families_map};

// ── Top-level fitting pipeline ────────────────────────────────────────────────

/// Combined result of the FPHA fitting pipeline.
///
/// # Contract — `planes` carry the α-scaled whole affine function
///
/// Each plane is `α·FPHA_0`: `intercept`, `gamma_v`, `gamma_q`, and `gamma_s` are
/// all `α · (raw hull coefficient)`. The export row builder writes them verbatim
/// with the IO `kappa` column pinned to `1.0`, so the precomputed read-back
/// (`intercept = gamma_0 · kappa`) round-trips `α·FPHA_0` exactly. Treating the
/// gradients as still-raw and re-applying an intercept-only shrink would
/// double-correct the export.
#[derive(Debug)]
pub(crate) struct FphaFitResult {
    /// Fitted hyperplanes, each the α-scaled whole affine function `α·FPHA_0`.
    pub planes: Vec<FphaPlane>,
    /// Least-squares `α_FPHA` correction every plane coefficient was scaled by.
    ///
    /// May be ≷ 1 (an MSE balance, not a bounded-below-1 factor). Divide a plane
    /// coefficient by `alpha` to recover the raw hull coefficient.
    // Surfaced for a future consumer that recovers the raw hull coefficients;
    // currently read only by the fitter unit tests, so the allow re-fires when
    // a non-test reader is removed.
    #[allow(dead_code)]
    pub alpha: f64,
    /// Fit-quality deviation of the FINAL (post-reduction) plane set — the set the
    /// LP applies — from the exact production function on the spill = 0 grid.
    ///
    /// The operator-facing signal that replaces the retired low-`kappa` warning;
    /// the resolver warns when [`FphaFitDeviation::exceeds_warn_threshold`].
    pub deviation: FphaFitDeviation,
    /// Per-sampled-point residuals the [`Self::deviation`] aggregate was reduced
    /// from, on the same spill = 0 grid.
    ///
    /// Populated only when `collect_deviation_points` is passed to
    /// [`fit_fpha_planes`]; otherwise empty with zero overhead, leaving the fit
    /// bit-identical. Feeds the opt-in `fpha_deviation_points.parquet` export.
    pub deviation_points: Vec<FphaDeviationPoint>,
}

/// Fit FPHA hyperplanes for a single hydro plant from its VHA curve geometry.
///
/// The top-level entry point for the computed FPHA path. The returned
/// `Vec<FphaPlane>` is structurally identical to what the precomputed path
/// produces from `fpha_hyperplanes.parquet`; the LP builder treats both identically.
///
/// # Errors
///
/// Any pipeline step can fail; all errors propagate via `?` as variants of
/// [`FphaFittingError`], each naming the hydro plant.
///
/// # Parameters
///
/// - `forebay_rows` — VHA curve rows sorted ascending by `volume_hm3` (as returned
///   by `cobre_io::extensions::parse_hydro_geometry`).
/// - `long_term_mean_inflow_m3s` — long-term mean natural inflow \[m³/s\] driving
///   the lateral-secant `S_max = 2·long-term mean inflow`; `0.0` (no inflow
///   history) selects the `2 × max_turbined` fallback (see `secant::resolve_s_max`).
/// - `tailrace_source` — resolved per (hydro, entry): [`TailraceSource::Families`]
///   when the plant has a `tailrace_curves` table, else [`TailraceSource::Entity`]
///   mirroring the entity-level model (the inert fallback).
/// - `reduction` — `None` is a literal skip: the validated `scaled` set is
///   converted unchanged, so a run without a reduction config is bit-identical to
///   the pre-reduction pipeline.
/// - `hydro_id`, `entry_level_bits` — the `Distance` reduction arm's stable seed
///   identity; the `Angle` arm and the `None` skip ignore them.
/// - `collect_deviation_points` — when `false` the residuals field stays empty
///   with zero overhead, so the fit is bit-identical regardless of the flag (the
///   collection is read-only over the emitted planes and never feeds back).
pub(crate) fn fit_fpha_planes(
    forebay_rows: &[HydroGeometryRow],
    hydro: &Hydro,
    config: &FphaColumnLayout,
    long_term_mean_inflow_m3s: f64,
    tailrace_source: TailraceSource,
    reduction: Option<&PlaneReductionConfig>,
    hydro_id: i32,
    entry_level_bits: u64,
    collect_deviation_points: bool,
) -> Result<FphaFitResult, FphaFittingError> {
    let forebay = ForebayTable::new(forebay_rows, &hydro.name)?;

    let pf = ProductionFunction::new(
        forebay.clone(),
        tailrace_source,
        hydro.hydraulic_losses.as_ref(),
        hydro.efficiency.as_ref(),
        hydro.max_turbined_m3s,
        hydro.name.clone(),
    )
    .with_max_generation_mw(hydro.max_generation_mw);

    let bounds = resolve_fitting_bounds(config, hydro, &forebay)?;

    // Any hull failure maps to a typed FPHA error instead of panicking, so one
    // pathological hydro does not abort the whole fitting loop. The dominant case
    // is a degenerate (collinear/coplanar) cloud.
    let hull_planes = fit_hull_planes(&pf, &bounds).map_err(|_err| {
        FphaFittingError::DegenerateProductionCloud {
            hydro_name: hydro.name.clone(),
        }
    })?;

    let alpha = compute_alpha_fpha(&hull_planes, &pf, &bounds);

    // Scale by α BEFORE fitting γ_S: α scaling does not touch the (already-0) γ_S,
    // and the secant reads each plane's representative point off the α-scaled
    // envelope, so scaling-then-fitting is the correct composition.
    let mut scaled: Vec<RawPlane> = hull_planes
        .iter()
        .map(|raw| scale_plane_affine(raw, alpha))
        .collect();

    fit_gamma_s_for_planes(&mut scaled, &pf, &bounds, long_term_mean_inflow_m3s);

    validate_fitted_planes(&scaled, alpha, &hydro.name)?;

    // The post-merge re-validate cannot fail (the mean of two sign-valid planes
    // is sign-valid) but makes the contract explicit.
    let reduced = match reduction {
        Some(cfg) => {
            let merged = reduce_planes(&scaled, cfg, &pf, &bounds, hydro_id, entry_level_bits);
            validate_fitted_planes(&merged, alpha, &hydro.name)?;
            merged
        }
        None => scaled,
    };

    // Measured on `reduced`, not `scaled`: the pre-reduction set would understate
    // the gap a merge introduces.
    let deviation = compute_fit_deviation(&reduced, &pf, &bounds);

    let deviation_points = if collect_deviation_points {
        collect_fit_deviation_points(&reduced, &pf, &bounds)
    } else {
        Vec::new()
    };

    let planes = reduced
        .iter()
        .map(|scaled_plane| FphaPlane {
            intercept: scaled_plane.gamma_0,
            gamma_v: scaled_plane.gamma_v,
            gamma_q: scaled_plane.gamma_q,
            gamma_s: scaled_plane.gamma_s,
        })
        .collect();

    Ok(FphaFitResult {
        planes,
        alpha,
        deviation,
        deviation_points,
    })
}
