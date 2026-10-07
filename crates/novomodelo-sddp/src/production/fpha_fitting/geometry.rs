//! Reservoir-geometry layer of the FPHA fitting pipeline.
//!
//! Owns the resolved fitting bounds ([`FittingBounds`] + [`resolve_fitting_bounds`]),
//! the forebay interpolation table ([`ForebayTable`]), and the tailrace / hydraulic-loss
//! evaluators ([`evaluate_tailrace`], [`evaluate_losses`]). The `production` submodule
//! reads these to assemble the complete production function.

use cobre_core::{HydraulicLossesModel, Hydro, TailraceModel};
use cobre_io::extensions::{FphaColumnLayout, HydroGeometryRow};

use super::error::FphaFittingError;

// ── FittingBounds ─────────────────────────────────────────────────────────────

/// Resolved volume range and discretization counts for the FPHA fitting grid.
///
/// Produced by [`resolve_fitting_bounds`] from an [`FphaColumnLayout`] and the hydro
/// plant entity. Consumed by the FPHA fitting grid construction step.
#[derive(Debug)]
pub(crate) struct FittingBounds {
    /// Resolved lower bound of the fitting volume range (hm³).
    pub v_min: f64,
    /// Resolved upper bound of the fitting volume range (hm³).
    pub v_max: f64,
    /// Number of volume grid points (>= 2).
    pub n_volume_points: usize,
    /// Number of turbined-flow grid points (>= 2).
    pub n_flow_points: usize,
    /// Number of spillage grid points (>= 2). Validated but not a fitting axis:
    /// the cloud and α regression fix `s = 0` and the lateral secant sweeps its
    /// own `S_max` sample, so no step builds a spillage grid axis.
    // Read by no fitting step; kept so the `spillage_discretization_points`
    // config field round-trips. The allow re-fires if a spillage-axis consumer
    // is reintroduced.
    #[allow(dead_code)]
    pub n_spillage_points: usize,
    /// Parsed plane-count ceiling (>= 1). No longer drives fitting: the hull fitter
    /// emits exactly the upper-envelope facets and there is no greedy selector to cap.
    // Read by no fitting step; kept so the config field round-trips and a
    // future trimming pass can consume it. The allow re-fires when that
    // reader lands.
    #[allow(dead_code)]
    pub max_planes_per_hydro: usize,
    /// Run-of-river marker: the requested volume range collapsed to a single
    /// point, so `[v_min, v_max]` are the two synthesized tangent-in-volume
    /// samples (see [`resolve_fitting_bounds`]) rather than a genuine storage
    /// window. The hull fitter reads this to zero `γ_V` exactly on the emitted
    /// planes — generation does not vary with storage for a run-of-river plant.
    pub single_volume: bool,
}

/// Absolute single-point detection tolerance for the volume range (hm³).
///
/// A range narrower than this is rerouted through the single-volume (run-of-river)
/// path, NOT rejected as an empty window. The reroute fires only for
/// `0 <= v_max - v_min <= V_EPS`; a genuinely inverted range (`v_max < v_min`) is
/// still an error.
const V_EPS: f64 = 1e-9;

/// Absolute fallback half-width for the synthesized volume samples (hm³).
///
/// When the entity's useful storage is itself zero, the `0.005·ΔV_useful` tangent
/// half-width collapses and the two synthesized samples would coincide, leaving
/// the cloud coplanar in `V`. This floor opens the `V` axis just enough for a
/// non-degenerate 3-D hull; `γ_V` is zeroed afterward, so its magnitude never
/// reaches the emitted planes.
const V_SYNTH_ABS: f64 = 1.0;

/// Resolve the fitting volume range and discretization counts from configuration.
///
/// # Volume range resolution
///
/// The volume range is resolved in three mutually exclusive modes — no fitting
/// window (full forebay range), absolute bounds, or percentile bounds — each
/// clamped to the forebay table range. Mixed min/max modes are accepted as long
/// as neither bound sets both absolute and percentile.
///
/// # Run-of-river reroute
///
/// A range collapsing to a single point (`0 <= v_max - v_min <= V_EPS`) or a
/// config requesting `NPTV = 1` is a run-of-river plant: rather than rejecting it,
/// the function synthesizes two tangent-in-volume samples and flags
/// `single_volume = true` so the hull fitter zeros `γ_V` exactly. The multi-volume
/// `n_volume_points >= 2` guard does NOT apply to this path.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | Both absolute and percentile set for the same bound (min or max) | [`FphaFittingError::ConflictingFittingWindow`] |
/// | Resolved `v_max < v_min` (inverted range, not single-point) | [`FphaFittingError::EmptyFittingWindow`] |
/// | A turbine/spillage count < 2, `max_planes_per_hydro` < 1, or (multi-volume only) `n_volume_points` < 2 | [`FphaFittingError::InsufficientDiscretization`] |
pub(crate) fn resolve_fitting_bounds(
    config: &FphaColumnLayout,
    hydro: &Hydro,
    forebay: &ForebayTable,
) -> Result<FittingBounds, FphaFittingError> {
    let hydro_name = &hydro.name;

    let (v_min, v_max) = match &config.fitting_window {
        None => (forebay.v_min(), forebay.v_max()),
        Some(fw) => {
            if fw.volume_min_hm3.is_some() && fw.volume_min_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_min_hm3 and volume_min_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }
            if fw.volume_max_hm3.is_some() && fw.volume_max_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_max_hm3 and volume_max_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }

            let entity_v_min = hydro.min_storage_hm3;
            let entity_v_max = hydro.max_storage_hm3;
            let entity_range = entity_v_max - entity_v_min;

            let v_min_raw = if let Some(abs) = fw.volume_min_hm3 {
                abs
            } else if let Some(pct) = fw.volume_min_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_min()
            };

            let v_max_raw = if let Some(abs) = fw.volume_max_hm3 {
                abs
            } else if let Some(pct) = fw.volume_max_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_max()
            };

            let v_min = v_min_raw.clamp(forebay.v_min(), forebay.v_max());
            let v_max = v_max_raw.clamp(forebay.v_min(), forebay.v_max());

            (v_min, v_max)
        }
    };

    // Inverted range is an error; the single-point case (`0 <= v_max - v_min <=
    // V_EPS`) is NOT inverted — it is rerouted below.
    if v_max < v_min {
        return Err(FphaFittingError::EmptyFittingWindow {
            hydro_name: hydro_name.clone(),
            v_min,
            v_max,
        });
    }

    #[allow(clippy::cast_sign_loss)]
    let requested_volume_points = config.volume_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_flow_points = config.turbine_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_spillage_points = config.spillage_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let max_planes = config.max_planes_per_hydro.unwrap_or(10) as usize;

    let single_volume = (v_max - v_min) <= V_EPS || requested_volume_points == 1;
    if single_volume {
        let v0 = v_min;
        // For a true run-of-river plant the useful storage band is itself 0, so
        // the proportional half-width collapses and `V_SYNTH_ABS` takes over.
        let useful = (hydro.max_storage_hm3 - hydro.min_storage_hm3).max(0.0);
        let half_width = (0.005 * useful).max(V_SYNTH_ABS);
        // Clamp to the forebay range ONLY when it spans a range. When the table is
        // a single point, clamping to `[v0, v0]` would collapse `v_lo == v_hi` into
        // a coplanar cloud the hull rejects; `height()` clamps queries internally,
        // so spreading beyond the degenerate range is safe and yields `γ_V = 0`.
        let (v_lo, v_hi) = if (forebay.v_max() - forebay.v_min()) <= V_EPS {
            (v0 - half_width, v0 + half_width)
        } else {
            (
                (v0 - half_width).clamp(forebay.v_min(), forebay.v_max()),
                (v0 + half_width).clamp(forebay.v_min(), forebay.v_max()),
            )
        };

        if n_flow_points < 2 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "turbine".to_owned(),
                value: n_flow_points,
            });
        }
        if n_spillage_points < 2 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "spillage".to_owned(),
                value: n_spillage_points,
            });
        }
        if max_planes < 1 {
            return Err(FphaFittingError::InsufficientDiscretization {
                hydro_name: hydro_name.clone(),
                dimension: "max_planes_per_hydro".to_owned(),
                value: max_planes,
            });
        }

        return Ok(FittingBounds {
            v_min: v_lo,
            v_max: v_hi,
            n_volume_points: 2,
            n_flow_points,
            n_spillage_points,
            max_planes_per_hydro: max_planes,
            single_volume: true,
        });
    }

    let n_volume_points = requested_volume_points;
    if n_volume_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "volume".to_owned(),
            value: n_volume_points,
        });
    }
    if n_flow_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "turbine".to_owned(),
            value: n_flow_points,
        });
    }
    if n_spillage_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "spillage".to_owned(),
            value: n_spillage_points,
        });
    }
    if max_planes < 1 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "max_planes_per_hydro".to_owned(),
            value: max_planes,
        });
    }

    Ok(FittingBounds {
        v_min,
        v_max,
        n_volume_points,
        n_flow_points,
        n_spillage_points,
        max_planes_per_hydro: max_planes,
        single_volume: false,
    })
}

// ── ForebayTable ──────────────────────────────────────────────────────────────

/// Linear interpolation table for forebay height `h_fore(v)`.
///
/// Stores the VHA curve for a single hydro plant as two parallel sorted vectors
/// of volume breakpoints (`volumes`, hm³) and corresponding surface elevations
/// (`heights`, m). All queries are clamped to `[v_min, v_max]`, so the table
/// never extrapolates and every method is infallible after construction.
///
/// # Construction
///
/// Build from a slice of [`HydroGeometryRow`] values (all rows for one hydro,
/// already sorted by ascending `volume_hm3` by the parser):
///
/// ```no_run
/// use cobre_io::extensions::HydroGeometryRow;
/// use cobre_core::EntityId;
///
/// // (ForebayTable is pub(crate); this example is for illustration only.)
/// let rows = vec![
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 0.0,    height_m: 386.5, area_km2: 2.5 },
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 2000.0, height_m: 390.0, area_km2: 3.1 },
/// ];
/// ```
#[derive(Debug, Clone)]
pub(crate) struct ForebayTable {
    /// Volume breakpoints (hm³), strictly increasing.
    volumes: Vec<f64>,
    /// Surface elevation breakpoints (m), monotonically non-decreasing.
    heights: Vec<f64>,
}

impl ForebayTable {
    /// Build a [`ForebayTable`] from a slice of VHA curve rows for one hydro plant.
    ///
    /// # Parameters
    ///
    /// - `rows` — all [`HydroGeometryRow`] entries for the hydro plant, sorted
    ///   by ascending `volume_hm3` (as returned by `cobre_io::extensions::parse_hydro_geometry`).
    /// - `hydro_name` — human-readable plant name used in error messages.
    ///
    /// A single row is accepted: a run-of-river plant has one operating volume
    /// (`v_min == v_max`), so its VHA curve is one row defining a CONSTANT forebay
    /// (`height()` returns that row's elevation for any query). Rejecting it here
    /// would make the single-volume FPHA fit (`resolve_fitting_bounds` →
    /// `single_volume`) unreachable end-to-end. Only an empty table is an error.
    ///
    /// # Errors
    ///
    /// | Condition | Error variant |
    /// |-----------|---------------|
    /// | Empty (zero rows) | [`FphaFittingError::InsufficientPoints`] |
    /// | `volume_hm3` not strictly increasing | [`FphaFittingError::NonMonotonicVolume`] |
    /// | `height_m` decreasing | [`FphaFittingError::NonMonotonicHeight`] |
    pub(crate) fn new(
        rows: &[HydroGeometryRow],
        hydro_name: &str,
    ) -> Result<Self, FphaFittingError> {
        if rows.is_empty() {
            return Err(FphaFittingError::InsufficientPoints {
                hydro_name: hydro_name.to_owned(),
                count: rows.len(),
            });
        }

        let mut volumes = Vec::with_capacity(rows.len());
        let mut heights = Vec::with_capacity(rows.len());

        volumes.push(rows[0].volume_hm3);
        heights.push(rows[0].height_m);

        for i in 1..rows.len() {
            let v_prev = rows[i - 1].volume_hm3;
            let v_curr = rows[i].volume_hm3;
            let h_prev = rows[i - 1].height_m;
            let h_curr = rows[i].height_m;

            if v_curr <= v_prev {
                return Err(FphaFittingError::NonMonotonicVolume {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    v_prev,
                    v_curr,
                });
            }

            if h_curr < h_prev {
                return Err(FphaFittingError::NonMonotonicHeight {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    h_prev,
                    h_curr,
                });
            }

            volumes.push(v_curr);
            heights.push(h_curr);
        }

        Ok(Self { volumes, heights })
    }

    /// Minimum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_min(&self) -> f64 {
        // INVARIANT: `volumes` is non-empty (enforced by `new`).
        self.volumes[0]
    }

    /// Maximum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_max(&self) -> f64 {
        // INVARIANT: `volumes` is non-empty (enforced by `new`).
        self.volumes[self.volumes.len() - 1]
    }

    /// Interpolated forebay surface elevation at `volume_hm3` (m).
    ///
    /// The query volume is clamped to `[v_min, v_max]` before interpolation,
    /// so this method is infallible and never extrapolates. Values at exact
    /// breakpoints are returned without rounding error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::extensions::HydroGeometryRow;
    /// use cobre_core::EntityId;
    ///
    /// // (ForebayTable is pub(crate); this example is for illustration only.)
    /// // let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
    /// // assert!((table.height(1000.0) - 388.25).abs() < 1e-10);
    /// ```
    pub(crate) fn height(&self, volume_hm3: f64) -> f64 {
        // Single-row table: returning `heights[0]` directly is REQUIRED — `locate`
        // computes `n - 2`, which underflows for `n == 1` and indexes out of bounds.
        if self.volumes.len() == 1 {
            return self.heights[0];
        }
        let v = volume_hm3.clamp(self.v_min(), self.v_max());
        let (i, t) = self.locate(v);
        self.heights[i] + t * (self.heights[i + 1] - self.heights[i])
    }

    /// Volume-averaged forebay elevation over `[v_lo, v_hi]` (m): the exact integral
    /// of the piecewise-linear `height` curve. Returns `height(v_lo)` for a
    /// zero-width (or single-row) range.
    pub(crate) fn mean_height(&self, v_lo: f64, v_hi: f64) -> f64 {
        if self.volumes.len() == 1 {
            return self.heights[0];
        }

        let v_lo = v_lo.clamp(self.v_min(), self.v_max());
        let v_hi = v_hi.clamp(self.v_min(), self.v_max());
        if v_hi <= v_lo {
            return self.height(v_lo);
        }

        let (seg_lo, frac_lo) = self.locate(v_lo);
        let (seg_hi, frac_hi) = self.locate(v_hi);
        let height_lo =
            self.heights[seg_lo] + frac_lo * (self.heights[seg_lo + 1] - self.heights[seg_lo]);
        let height_hi =
            self.heights[seg_hi] + frac_hi * (self.heights[seg_hi + 1] - self.heights[seg_hi]);

        if seg_lo == seg_hi {
            return 0.5 * (height_lo + height_hi);
        }

        let mut integral =
            0.5 * (height_lo + self.heights[seg_lo + 1]) * (self.volumes[seg_lo + 1] - v_lo);
        for seg in (seg_lo + 1)..seg_hi {
            integral += 0.5
                * (self.heights[seg] + self.heights[seg + 1])
                * (self.volumes[seg + 1] - self.volumes[seg]);
        }
        integral += 0.5 * (self.heights[seg_hi] + height_hi) * (v_hi - self.volumes[seg_hi]);

        integral / (v_hi - v_lo)
    }

    /// Find the segment index `i` and the fractional position `t` within it.
    ///
    /// Returns `(i, t)` such that:
    /// - `self.volumes[i] <= v <= self.volumes[i + 1]`
    /// - `t = (v - self.volumes[i]) / (self.volumes[i + 1] - self.volumes[i])`
    ///
    /// The caller must ensure `v` is already clamped to `[v_min, v_max]`.
    ///
    /// Uses `partition_point` (binary search) for O(log n) lookup.
    ///
    /// At `v_max` the last segment is returned (index `n - 2`) to avoid an
    /// out-of-bounds access on `i + 1`.
    fn locate(&self, v: f64) -> (usize, f64) {
        let n = self.volumes.len();
        let idx = self.volumes.partition_point(|&vk| vk <= v);
        let i = idx.saturating_sub(1).min(n - 2);
        let dv = self.volumes[i + 1] - self.volumes[i];
        let t = (v - self.volumes[i]) / dv;
        (i, t)
    }
}

// ── Tailrace and hydraulic loss evaluation ────────────────────────────────────

/// Tailrace elevation `h_tail(q_out)` for a total outflow of `outflow_m3s` (m).
///
/// - `Polynomial`: evaluates `c[0] + c[1]·q + c[2]·q² + …` via Horner's method.
/// - `Piecewise`: linearly interpolates between adjacent [`cobre_core::TailracePoint`]
///   breakpoints; the outflow is clamped to the table's range before lookup.
///
/// The function is infallible — the model invariants (≥ 1 coefficient; ≥ 2 points
/// sorted ascending) are enforced by the `cobre-io` parsing layer.
pub(crate) fn evaluate_tailrace(model: &TailraceModel, outflow_m3s: f64) -> f64 {
    match model {
        TailraceModel::Polynomial { coefficients } => coefficients
            .iter()
            .rev()
            .fold(0.0_f64, |acc, c| acc * outflow_m3s + c),
        TailraceModel::Piecewise { points } => {
            let n = points.len();
            if n == 0 {
                return 0.0;
            }
            if n == 1 {
                return points[0].height_m;
            }
            let q = outflow_m3s.clamp(points[0].outflow_m3s, points[n - 1].outflow_m3s);
            let (i, t) = locate_tailrace(points, q);
            points[i].height_m + t * (points[i + 1].height_m - points[i].height_m)
        }
    }
}

/// Head loss `h_loss` (m) for the given `gross_head` (m) and `turbined_m3s` (m³/s).
///
/// `_turbined_m3s` is reserved for future flow-dependent loss variants and ignored
/// by both current variants.
pub(crate) fn evaluate_losses(
    model: &HydraulicLossesModel,
    gross_head: f64,
    _turbined_m3s: f64,
) -> f64 {
    match model {
        HydraulicLossesModel::Factor { value } => value * gross_head,
        HydraulicLossesModel::Constant { value_m } => *value_m,
    }
}

/// Find segment index `i` and fractional position `t` in a piecewise tailrace table.
///
/// Returns `(i, t)` such that:
/// - `points[i].outflow_m3s <= q <= points[i+1].outflow_m3s`
/// - `t = (q - outflow[i]) / (outflow[i+1] - outflow[i])`
///
/// The caller must ensure `q` is already clamped to `[q_min, q_max]`. Uses
/// `partition_point` for O(log n) binary search; saturates at `n - 2` to keep
/// `i + 1` in bounds at `q == q_max`.
fn locate_tailrace(points: &[cobre_core::TailracePoint], q: f64) -> (usize, f64) {
    let n = points.len();
    let idx = points.partition_point(|p| p.outflow_m3s <= q);
    let i = idx.saturating_sub(1).min(n - 2);
    let dq = points[i + 1].outflow_m3s - points[i].outflow_m3s;
    let t = (q - points[i].outflow_m3s) / dq;
    (i, t)
}

#[cfg(test)]
mod tests {
    use cobre_core::EntityId;
    use cobre_io::extensions::HydroGeometryRow;
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::ForebayTable;

    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// Non-uniform 4-point table: segment widths 100/200/700 hm3.
    fn non_uniform_table() -> ForebayTable {
        ForebayTable::new(
            &[
                row(0.0, 10.0),
                row(100.0, 20.0),
                row(300.0, 50.0),
                row(1000.0, 80.0),
            ],
            "NonUniform",
        )
        .expect("strictly increasing volumes and non-decreasing heights")
    }

    fn assert_relative_close(actual: f64, expected: f64, msg: &str) {
        const REL_TOL: f64 = 1e-9;
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= REL_TOL * scale,
            "{msg}: got {actual}, expected {expected}"
        );
    }

    #[test]
    fn mean_height_single_row_returns_constant_height_for_any_range() {
        let table = ForebayTable::new(&[row(1500.0, 386.5)], "RunOfRiver").unwrap();
        assert_eq!(table.mean_height(0.0, 3000.0), 386.5);
        assert_eq!(table.mean_height(1500.0, 1500.0), 386.5);
        assert_eq!(table.mean_height(2000.0, 1000.0), 386.5);
    }

    #[test]
    fn mean_height_zero_width_range_returns_height_at_point() {
        let table = non_uniform_table();
        let v = 150.0;
        assert_eq!(table.mean_height(v, v), table.height(v));
    }

    #[test]
    fn mean_height_both_endpoints_within_one_segment_averages_the_two_heights() {
        let table = non_uniform_table();
        // Both v_lo and v_hi land strictly inside the [100, 300] segment, so
        // `locate` resolves the same segment index for each.
        let v_lo = 150.0;
        let v_hi = 250.0;
        let expected = 0.5 * (table.height(v_lo) + table.height(v_hi));
        assert_eq!(table.mean_height(v_lo, v_hi), expected);
    }

    #[test]
    fn mean_height_inverted_range_returns_height_at_clamped_lo() {
        let table = non_uniform_table();
        // v_lo clamps to v_max (1000.0), which then exceeds the clamped v_hi.
        assert_eq!(table.mean_height(1500.0, 50.0), table.height(1000.0));
    }

    #[test]
    fn mean_height_clamps_both_sides_matches_full_table_integral() {
        let table = non_uniform_table();
        let expected = (0.5 * (10.0 + 20.0) * 100.0
            + 0.5 * (20.0 + 50.0) * 200.0
            + 0.5 * (50.0 + 80.0) * 700.0)
            / 1000.0;
        assert_relative_close(
            table.mean_height(-500.0, 5000.0),
            expected,
            "clamped both sides",
        );
    }

    #[test]
    fn mean_height_breakpoint_aligned_endpoints_matches_single_segment_average() {
        let table = non_uniform_table();
        assert_relative_close(table.mean_height(100.0, 300.0), 35.0, "breakpoint-aligned");
    }

    #[test]
    fn mean_height_mid_segment_endpoints_spans_partial_and_full_segments() {
        let table = non_uniform_table();
        let height_lo = table.height(50.0);
        let height_hi = table.height(500.0);
        let expected = (0.5 * (height_lo + 20.0) * 50.0
            + 0.5 * (20.0 + 50.0) * 200.0
            + 0.5 * (50.0 + height_hi) * 200.0)
            / 450.0;
        assert_relative_close(
            table.mean_height(50.0, 500.0),
            expected,
            "mid-segment endpoints",
        );
    }

    #[test]
    fn mean_height_non_uniform_grid_spans_multiple_interior_segments() {
        let table = ForebayTable::new(
            &[
                row(0.0, 5.0),
                row(50.0, 15.0),
                row(200.0, 45.0),
                row(250.0, 50.0),
                row(800.0, 90.0),
                row(1000.0, 100.0),
            ],
            "MultiSegment",
        )
        .expect("strictly increasing volumes and non-decreasing heights");

        let height_lo = table.height(20.0);
        let height_hi = table.height(900.0);
        let expected = (0.5 * (height_lo + 15.0) * (50.0 - 20.0)
            + 0.5 * (15.0 + 45.0) * (200.0 - 50.0)
            + 0.5 * (45.0 + 50.0) * (250.0 - 200.0)
            + 0.5 * (50.0 + 90.0) * (800.0 - 250.0)
            + 0.5 * (90.0 + height_hi) * (900.0 - 800.0))
            / (900.0 - 20.0);

        assert_relative_close(
            table.mean_height(20.0, 900.0),
            expected,
            "non-uniform grid spanning multiple interior segments",
        );
    }

    /// A `ForebayTable` built from strictly-increasing volumes and non-decreasing
    /// heights (2 to 6 rows), generated from positive deltas so `new` never rejects.
    fn forebay_table_strategy() -> impl Strategy<Value = ForebayTable> {
        (2usize..=6).prop_flat_map(|n| {
            (
                prop::collection::vec(0.01f64..100.0, n),
                prop::collection::vec(0.0f64..50.0, n),
            )
                .prop_map(move |(volume_deltas, height_deltas)| {
                    let mut volume = 0.0;
                    let mut height = 100.0;
                    let rows: Vec<HydroGeometryRow> = (0..n)
                        .map(|idx| {
                            volume += volume_deltas[idx];
                            height += height_deltas[idx];
                            row(volume, height)
                        })
                        .collect();
                    ForebayTable::new(&rows, "PropTest")
                        .expect("strictly increasing volumes and non-decreasing heights")
                })
        })
    }

    fn table_and_three_volumes_strategy() -> impl Strategy<Value = (ForebayTable, f64, f64, f64)> {
        forebay_table_strategy().prop_flat_map(|table| {
            let lo = table.v_min();
            let hi = table.v_max();
            (Just(table), lo..=hi, lo..=hi, lo..=hi)
        })
    }

    /// Fixed cases/seed so a failing shrink is reproducible run-to-run.
    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    proptest! {
        #![proptest_config(fixed_config())]

        /// `mean_height` over `[a, c]` equals the width-weighted combination of
        /// `[a, b]` and `[b, c]` for any `a <= b <= c`: the split reorders the
        /// trapezoid sum and interpolates an extra height at `b`, so the two
        /// sides agree only up to floating-point tolerance, never bit-exact `==`.
        #[test]
        fn mean_height_partition_additivity(
            (table, x, y, z) in table_and_three_volumes_strategy()
        ) {
            let mut volumes = [x, y, z];
            volumes.sort_by(|p, q| p.partial_cmp(q).expect("proptest never generates NaN"));
            let [a, b, c] = volumes;

            let lhs = table.mean_height(a, c) * (c - a);
            let rhs = table.mean_height(a, b) * (b - a) + table.mean_height(b, c) * (c - b);
            let scale = lhs.abs().max(rhs.abs()).max(1.0);

            prop_assert!((lhs - rhs).abs() <= 1e-9 * scale);
        }
    }
}
