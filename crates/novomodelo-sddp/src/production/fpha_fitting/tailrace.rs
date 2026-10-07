//! Within-family piecewise-quartic tailrace evaluation.
//!
//! The exact tailrace `tailrace_level(outflow)` for one family is a piecewise degree-4
//! polynomial: the `outflow` domain is partitioned into contiguous segments, each
//! valid on `[outflow_min, outflow_max]` and evaluated as
//! `coefficient_0 + coefficient_1·Q + coefficient_2·Q² + coefficient_3·Q³ + coefficient_4·Q⁴`. This module owns the
//! segment-collection construction ([`TailraceSegments::from_rows`]), the
//! structural validation (contiguity + C0 continuity) that the IO layer
//! deliberately defers, and the infallible within-family evaluator
//! ([`TailraceSegments::evaluate`]).
//!
//! [`TailraceSegments`] evaluates ONE family in isolation. The backwater layer
//! ([`TailraceFamilies`]) groups a plant's rows into families keyed by their
//! downstream reference level (`downstream_reference_level_m`), orders them by that level, and
//! interpolates between the two bracketing families at a resolved downstream
//! level — collapsing to a single family (no interpolation) when the plant has
//! one family or the level is unresolved. [`build_tailrace_families_map`] groups
//! the whole table by `hydro_id` into one [`TailraceFamilies`] per plant.

use std::collections::HashMap;

use cobre_core::EntityId;
use cobre_io::extensions::TailraceCurveRow;

use super::error::FphaFittingError;

/// Absolute tolerance for the inter-segment contiguity check (m³/s).
///
/// Consecutive segments meet when `|outflow_min −
/// outflow_max| <= CONTIG_EPS`, never when `outflow_min == outflow_max`. The
/// source bounds are calibrated floats differing in their last ULPs, so an
/// exact-equality test would reject essentially every real family. Owning check:
/// [`TailraceSegments::from_rows`].
const CONTIG_EPS: f64 = 1e-6;

/// Absolute floor (m) for the inter-segment C0-continuity check, for a
/// near-zero tailrace level where a purely relative term would collapse to
/// near-zero.
const C0_EPS_ABS: f64 = 1e-3;

/// Relative tolerance for the inter-segment C0-continuity check.
///
/// Adjacent quartics are continuous when their boundary
/// elevations agree within `max(C0_EPS_ABS, C0_EPS_REL * max(|h_left|,
/// |h_right|))`, never bit-for-bit (`==`) — the two quartics are fit
/// independently and meet only to calibration precision, which scales with the
/// tailrace level rather than staying fixed in absolute terms (a fixed-`1e-3`-m
/// bound faithfully accepts a near-zero curve's residual but under-scales a
/// hundreds-of-metres curve's). `C0_EPS_REL` (100 ppm) is the owner-chosen
/// bound: loose enough to admit that calibration residual at any level, tight
/// enough that a genuine metre-scale discontinuity still exceeds it by roughly
/// an order of magnitude. Owning check: [`TailraceSegments::from_rows`].
const C0_EPS_REL: f64 = 1e-4;

/// One degree-4 piece of a family's tailrace curve.
///
/// Valid on `[outflow_min, outflow_max]` (m³/s). `coeffs[i]` is the coefficient of `Q^i`,
/// so `coeffs = [coefficient_0, coefficient_1, coefficient_2, coefficient_3, coefficient_4]`.
#[derive(Debug, Clone, PartialEq)]
struct QuarticSegment {
    /// Segment lower validity bound (m³/s).
    outflow_min: f64,
    /// Segment upper validity bound (m³/s), `>= outflow_min`.
    outflow_max: f64,
    /// Polynomial coefficients, `coeffs[i]` the coefficient of `Q^i`.
    coeffs: [f64; 5],
}

impl QuarticSegment {
    /// Evaluate the quartic at `q` via Horner's method.
    #[inline]
    fn eval(&self, q: f64) -> f64 {
        let [a0, a1, a2, a3, a4] = self.coeffs;
        (((a4 * q + a3) * q + a2) * q + a1) * q + a0
    }
}

/// One family's ordered, validated piecewise-quartic tailrace curve.
///
/// Built once from a family's `&[TailraceCurveRow]` slice via
/// [`TailraceSegments::from_rows`], which validates contiguity and C0
/// continuity. After construction [`TailraceSegments::evaluate`] is infallible,
/// pure, and allocation-free: identical inputs yield bit-identical outputs
/// regardless of call order.
#[derive(Debug, Clone, PartialEq)]
struct TailraceSegments {
    /// Contiguous segments ordered by ascending `outflow_min` (>= 1 element).
    segments: Vec<QuarticSegment>,
}

impl TailraceSegments {
    /// Build a validated [`TailraceSegments`] from one family's rows.
    ///
    /// `rows` is the slice for a single `(hydro_id, family_id)` group, already
    /// sorted by ascending `segment_id` (as returned by the tailrace-curve
    /// parser). The first segment's `outflow_min` is taken as given — not forced
    /// to 0. C1 continuity is preferred but NOT checked — a derivative break does
    /// not reject the family.
    ///
    /// # Errors
    ///
    /// | Condition | Error variant |
    /// |-----------|---------------|
    /// | `rows` empty | [`FphaFittingError::InsufficientPoints`] |
    /// | A gap or overlap between consecutive segments | [`FphaFittingError::TailraceGap`] |
    /// | A C0 break at an interior boundary | [`FphaFittingError::TailraceDiscontinuity`] |
    fn from_rows(rows: &[TailraceCurveRow], hydro_name: &str) -> Result<Self, FphaFittingError> {
        if rows.is_empty() {
            return Err(FphaFittingError::InsufficientPoints {
                hydro_name: hydro_name.to_owned(),
                count: 0,
            });
        }

        let segments: Vec<QuarticSegment> = rows
            .iter()
            .map(|r| QuarticSegment {
                outflow_min: r.outflow_min_m3s,
                outflow_max: r.outflow_max_m3s,
                coeffs: [
                    r.coefficient_0,
                    r.coefficient_1,
                    r.coefficient_2,
                    r.coefficient_3,
                    r.coefficient_4,
                ],
            })
            .collect();

        for k in 1..segments.len() {
            let prev = &segments[k - 1];
            let curr = &segments[k];

            // Both a gap and an overlap fail the same `<= CONTIG_EPS` test.
            if (curr.outflow_min - prev.outflow_max).abs() > CONTIG_EPS {
                return Err(FphaFittingError::TailraceGap {
                    hydro_name: hydro_name.to_owned(),
                    outflow_max_prev: prev.outflow_max,
                    outflow_min_curr: curr.outflow_min,
                });
            }

            let boundary = prev.outflow_max;
            let h_left = prev.eval(boundary);
            let h_right = curr.eval(boundary);
            let c0_tolerance = (C0_EPS_REL * h_left.abs().max(h_right.abs())).max(C0_EPS_ABS);
            if (h_left - h_right).abs() > c0_tolerance {
                return Err(FphaFittingError::TailraceDiscontinuity {
                    hydro_name: hydro_name.to_owned(),
                    boundary,
                    h_left,
                    h_right,
                });
            }
        }

        Ok(Self { segments })
    }

    /// Evaluate the family's tailrace elevation `tailrace_level` (m) at `outflow_m3s` (m³/s).
    ///
    /// `outflow_m3s` is clamped to `[segments[0].outflow_min, segments[last].outflow_max]` before
    /// locating (below → first `outflow_min`, above → last `outflow_max`), then the owning
    /// segment is found and its quartic evaluated at the clamped value. The
    /// method is infallible, pure, and allocation-free: validation already ran
    /// in [`TailraceSegments::from_rows`], and the segments live in `self`.
    fn evaluate(&self, outflow_m3s: f64) -> f64 {
        // INVARIANT: `segments` is non-empty (enforced by `from_rows`).
        let n = self.segments.len();
        let q_lo = self.segments[0].outflow_min;
        let q_hi = self.segments[n - 1].outflow_max;
        let q = outflow_m3s.clamp(q_lo, q_hi);

        // Saturate at `n - 1` so a `q` at the upper edge resolves to the last
        // segment instead of running past the end.
        let idx = self.segments.partition_point(|s| s.outflow_max <= q);
        let i = idx.min(n - 1);
        self.segments[i].eval(q)
    }
}

// ── Backwater (downstream-level-coupled) family collection ──────────────────────

/// One downstream-level-keyed family of a plant's tailrace curve.
#[derive(Debug, Clone)]
struct TailraceFamily {
    /// Downstream reference level keying the family (m); `None` for a
    /// single-family plant.
    downstream_reference_level_m: Option<f64>,
    /// The family's validated piecewise-quartic curve.
    segments: TailraceSegments,
}

/// All tailrace families for ONE plant, ordered for downstream-level bracketing.
///
/// Built from a plant's `&[TailraceCurveRow]` slice via
/// [`TailraceFamilies::from_rows`]. After construction the families are ordered
/// ascending by `downstream_reference_level_m`, and [`TailraceFamilies::evaluate`] returns the
/// effective tailrace elevation at a turbined-flow `outflow_m3s` and a resolved
/// downstream level, interpolating between the two bracketing families.
#[derive(Debug, Clone)]
pub(crate) struct TailraceFamilies {
    /// Families ordered ascending by `downstream_reference_level_m` (>= 1 element). A
    /// single-family plant may carry one `None`-keyed family; a multi-family
    /// plant has every `downstream_reference_level_m` populated (enforced by
    /// [`TailraceFamilies::from_rows`]).
    families: Vec<TailraceFamily>,
}

impl TailraceFamilies {
    /// Build a [`TailraceFamilies`] from one plant's tailrace rows.
    ///
    /// `rows` is the slice for a single `hydro_id`, already globally sorted by
    /// `(hydro_id, family_id, segment_id)` (as the tailrace-curve parser returns
    /// it). Rows are grouped into families by consecutive equal `family_id`, then
    /// ordered ascending by `downstream_reference_level_m` (`total_cmp`) with
    /// `family_id` as the secondary tie-break.
    ///
    /// # Family-key contract
    ///
    /// A plant with **more than one** family must carry a downstream reference
    /// level (`downstream_reference_level_m`) on **every** family — a multi-family table with any
    /// keyless family is rejected with
    /// [`FphaFittingError::TailraceFamilyKeyMissing`], **never** silently
    /// resolved by picking one family. The obvious-but-wrong alternative —
    /// treating a missing key as "ignore the level and use this family" — would
    /// make the choice of family depend on which row happened to lack a key, a
    /// non-deterministic, physically meaningless selection. A `None` key is only
    /// admissible when the plant has exactly one family, where the downstream
    /// level is ignored.
    ///
    /// # Errors
    ///
    /// | Condition | Error variant |
    /// |-----------|---------------|
    /// | A family group fails contiguity / C0 validation | propagated from [`TailraceSegments::from_rows`] |
    /// | `rows` empty | [`FphaFittingError::InsufficientPoints`] |
    /// | Multiple families with any `None` `downstream_reference_level_m` | [`FphaFittingError::TailraceFamilyKeyMissing`] |
    pub(crate) fn from_rows(
        rows: &[TailraceCurveRow],
        hydro_name: &str,
    ) -> Result<Self, FphaFittingError> {
        if rows.is_empty() {
            return Err(FphaFittingError::InsufficientPoints {
                hydro_name: hydro_name.to_owned(),
                count: 0,
            });
        }

        // Pre-sorted by `(hydro_id, family_id, segment_id)`, so equal `family_id`
        // rows are contiguous and already in `segment_id` order.
        let mut families: Vec<(i32, TailraceFamily)> = Vec::new();
        let mut group_start = 0_usize;
        for k in 1..=rows.len() {
            let at_boundary = k == rows.len() || rows[k].family_id != rows[group_start].family_id;
            if at_boundary {
                let group = &rows[group_start..k];
                let family_id = group[0].family_id;
                let downstream_reference_level_m = group[0].downstream_reference_level_m;
                let segments = TailraceSegments::from_rows(group, hydro_name)?;
                families.push((
                    family_id,
                    TailraceFamily {
                        downstream_reference_level_m,
                        segments,
                    },
                ));
                group_start = k;
            }
        }

        // A keyless family is admissible only for a single-family plant.
        if families.len() > 1
            && families
                .iter()
                .any(|(_, f)| f.downstream_reference_level_m.is_none())
        {
            return Err(FphaFittingError::TailraceFamilyKeyMissing {
                hydro_name: hydro_name.to_owned(),
                family_count: families.len(),
            });
        }

        // Level must be the PRIMARY key: `evaluate` clamps `L` to
        // `[family_level(0), family_level(last)]` and brackets by level, so a
        // `family_id`-primary sort would leave families out of level order, making
        // the clamp bounds `min > max` (a panic) and the bracket wrong. `total_cmp`
        // keeps the order total on equal levels; `family_id` breaks a shared-level
        // tie deterministically (declaration-order invariance).
        families.sort_by(|(fa, a), (fb, b)| {
            let la = a.downstream_reference_level_m.unwrap_or(f64::NEG_INFINITY);
            let lb = b.downstream_reference_level_m.unwrap_or(f64::NEG_INFINITY);
            la.total_cmp(&lb).then_with(|| fa.cmp(fb))
        });

        Ok(Self {
            families: families.into_iter().map(|(_, f)| f).collect(),
        })
    }

    /// Effective tailrace elevation `tailrace_level` (m) at `outflow_m3s` (m³/s) for a resolved
    /// downstream level.
    ///
    /// - **single family** ⇒ evaluate it directly; `downstream_level_m` is ignored;
    /// - **multiple families + `Some(L)`** ⇒ bracket `L` and linearly interpolate;
    /// - **multiple families + `None`** ⇒ the lowest-level family (a deterministic
    ///   fallback for an unresolved downstream level).
    ///
    /// # Clamp-not-extrapolate contract
    ///
    /// `L` is clamped to the calibrated level range before bracketing, so the
    /// result is NEVER extrapolated past it. Extending the linear blend beyond the
    /// bracket (the obvious-but-wrong alternative) would produce a non-physical
    /// tailrace elevation from a quartic-derived height outside its fitted band.
    /// Mirrors the clamp in
    /// [`ForebayTable::height`](super::geometry::ForebayTable::height).
    pub(crate) fn evaluate(&self, outflow_m3s: f64, downstream_level_m: Option<f64>) -> f64 {
        // INVARIANT: `families` is non-empty (enforced by `from_rows`).
        let n = self.families.len();
        if n == 1 {
            return self.families[0].segments.evaluate(outflow_m3s);
        }

        // Unresolved level falls back to the lowest-keyed family (index 0 after the
        // ascending sort).
        let Some(level) = downstream_level_m else {
            return self.families[0].segments.evaluate(outflow_m3s);
        };

        // Every multi-family family carries a level (enforced by `from_rows`);
        // `NEG_INFINITY` is an unreachable sentinel making the read total without
        // an `unwrap`.
        let family_level = |i: usize| {
            self.families[i]
                .downstream_reference_level_m
                .unwrap_or(f64::NEG_INFINITY)
        };
        let l_lo = family_level(0);
        let l_hi = family_level(n - 1);
        let l = level.clamp(l_lo, l_hi);

        // Saturate the upper bracket at `n - 1` so a level at the top edge resolves
        // the last pair instead of running past the end.
        let upper = self
            .families
            .partition_point(|f| level_le(f.downstream_reference_level_m, l));
        let hi = upper.min(n - 1).max(1);
        let lo = hi - 1;

        let h_lo = self.families[lo].segments.evaluate(outflow_m3s);
        let h_hi = self.families[hi].segments.evaluate(outflow_m3s);
        let level_lo = family_level(lo);
        let level_hi = family_level(hi);

        // A zero-width bracket (two same-level families) collapses to the lower
        // height, avoiding a divide-by-zero.
        let span = level_hi - level_lo;
        if span <= 0.0 {
            h_lo
        } else {
            let t = (l - level_lo) / span;
            h_lo + t * (h_hi - h_lo)
        }
    }
}

/// `downstream_reference_level_m <= l`, treating a `None` key as `-∞` (sorts first).
#[inline]
fn level_le(downstream_reference_level_m: Option<f64>, l: f64) -> bool {
    downstream_reference_level_m.unwrap_or(f64::NEG_INFINITY) <= l
}

/// Group a whole tailrace table into one [`TailraceFamilies`] per plant.
///
/// `rows` is the full table sorted by `(hydro_id, family_id, segment_id)`. Rows
/// are partitioned by `hydro_id` (a deterministic [`HashMap`] keyed by
/// [`EntityId`]); each plant's slice is built into a [`TailraceFamilies`]. A
/// plant absent from the returned map has no tailrace table and is handled by
/// the caller (the production-function sampler) falling back to the entity-level
/// tailrace model.
///
/// Mirrors `build_geometry_map` in the production-model layer: deterministic
/// grouping by [`EntityId`], independent of input row ordering.
///
/// # Errors
///
/// Propagates the first per-plant construction error from
/// [`TailraceFamilies::from_rows`] (a family-validation failure or a keyless
/// multi-family table).
pub(crate) fn build_tailrace_families_map(
    rows: &[TailraceCurveRow],
) -> Result<HashMap<EntityId, TailraceFamilies>, FphaFittingError> {
    // Pre-sorted by `(hydro_id, ...)`, so equal `hydro_id` rows are contiguous and
    // a single pass isolates each plant's slice without cloning rows.
    let mut map: HashMap<EntityId, TailraceFamilies> = HashMap::new();
    let mut group_start = 0_usize;
    for k in 1..=rows.len() {
        let at_boundary = k == rows.len() || rows[k].hydro_id != rows[group_start].hydro_id;
        if at_boundary {
            let group = &rows[group_start..k];
            let hydro_id = group[0].hydro_id;
            // The hydro name is unavailable at the table-grouping layer; the
            // EntityId stands in for error context until the sampler threads
            // the registry name through.
            let families = TailraceFamilies::from_rows(group, &format!("id={}", hydro_id.0))?;
            map.insert(hydro_id, families);
            group_start = k;
        }
    }
    Ok(map)
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
    use cobre_core::EntityId;
    use cobre_io::extensions::TailraceCurveRow;

    use super::super::error::FphaFittingError;
    use super::{TailraceFamilies, TailraceSegments, build_tailrace_families_map};

    /// Build a single segment's row with explicit bounds and coefficients.
    fn row(
        segment_id: i32,
        outflow_min: f64,
        outflow_max: f64,
        coeffs: [f64; 5],
    ) -> TailraceCurveRow {
        TailraceCurveRow {
            hydro_id: EntityId::from(1),
            family_id: 1,
            downstream_reference_level_m: None,
            segment_id,
            outflow_min_m3s: outflow_min,
            outflow_max_m3s: outflow_max,
            coefficient_0: coeffs[0],
            coefficient_1: coeffs[1],
            coefficient_2: coeffs[2],
            coefficient_3: coeffs[3],
            coefficient_4: coeffs[4],
        }
    }

    #[test]
    fn single_segment_linear_eval_matches_hand_value() {
        let rows = vec![row(1, 0.0, 1000.0, [5.0, 0.001, 0.0, 0.0, 0.0])];
        let seg = TailraceSegments::from_rows(&rows, "Plant").unwrap();
        // 5.0 + 0.001 * 400 = 5.4
        assert!((seg.evaluate(400.0) - 5.4).abs() < 1e-9);
    }

    #[test]
    fn two_contiguous_c0_matching_segments_ok_and_boundary_agrees() {
        let b = 408.649;
        // Left segment: h = 1.0 + 0.01·Q. Value at b is 1.0 + 0.01·b.
        // Right segment: a constant equal to the left value at b, so C0 holds.
        let left_at_b = 1.0 + 0.01 * b;
        let rows = vec![
            row(1, 0.0, b, [1.0, 0.01, 0.0, 0.0, 0.0]),
            row(2, b, 1000.0, [left_at_b, 0.0, 0.0, 0.0, 0.0]),
        ];
        let seg = TailraceSegments::from_rows(&rows, "Plant").unwrap();

        // Evaluate just below and at the boundary (left side), and just above
        // (right side); all three agree to tolerance.
        let at_boundary = seg.evaluate(b);
        let just_below = seg.evaluate(b - 1e-7);
        let just_above = seg.evaluate(b + 1e-7);
        assert!((at_boundary - left_at_b).abs() < 1e-9);
        assert!((just_below - left_at_b).abs() < 1e-6);
        assert!((just_above - left_at_b).abs() < 1e-9);
    }

    #[test]
    fn gap_between_segments_is_tailrace_gap() {
        // segments[0].outflow_max = 408.6 but segments[1].outflow_min = 410.0 → gap.
        let rows = vec![
            row(1, 0.0, 408.6, [1.0, 0.0, 0.0, 0.0, 0.0]),
            row(2, 410.0, 1000.0, [1.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let err = TailraceSegments::from_rows(&rows, "Plant").unwrap_err();
        match err {
            FphaFittingError::TailraceGap {
                outflow_max_prev,
                outflow_min_curr,
                ..
            } => {
                assert_eq!(outflow_max_prev, 408.6);
                assert_eq!(outflow_min_curr, 410.0);
            }
            other => panic!("expected TailraceGap, got {other:?}"),
        }
    }

    #[test]
    fn overlap_between_segments_is_tailrace_gap() {
        // segments[1].outflow_min (400.0) < segments[0].outflow_max (408.6) → overlap.
        let rows = vec![
            row(1, 0.0, 408.6, [1.0, 0.0, 0.0, 0.0, 0.0]),
            row(2, 400.0, 1000.0, [1.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let err = TailraceSegments::from_rows(&rows, "Plant").unwrap_err();
        match err {
            FphaFittingError::TailraceGap {
                outflow_max_prev,
                outflow_min_curr,
                ..
            } => {
                assert_eq!(outflow_max_prev, 408.6);
                assert_eq!(outflow_min_curr, 400.0);
            }
            other => panic!("expected TailraceGap (overlap), got {other:?}"),
        }
    }

    #[test]
    fn c0_break_is_tailrace_discontinuity() {
        let b = 408.6;
        // Left value at b = 1.0; right segment constant 1.5 → 0.5 m jump.
        let rows = vec![
            row(1, 0.0, b, [1.0, 0.0, 0.0, 0.0, 0.0]),
            row(2, b, 1000.0, [1.5, 0.0, 0.0, 0.0, 0.0]),
        ];
        let err = TailraceSegments::from_rows(&rows, "Plant").unwrap_err();
        match err {
            FphaFittingError::TailraceDiscontinuity {
                boundary,
                h_left,
                h_right,
                ..
            } => {
                assert_eq!(boundary, b);
                assert!((h_left - 1.0).abs() < 1e-12);
                assert!((h_right - 1.5).abs() < 1e-12);
            }
            other => panic!("expected TailraceDiscontinuity, got {other:?}"),
        }
    }

    /// A ~3.6e-6-relative knot gap at a ~775 m tailrace level (the reported
    /// false rejection: two independently-fit quartics meeting only to
    /// calibration precision) is accepted.
    #[test]
    fn c0_relative_gap_within_tolerance_at_high_magnitude_is_accepted() {
        let b = 416.0;
        let rows = vec![
            row(1, 0.0, b, [774.924_141_892_876_4, 0.0, 0.0, 0.0, 0.0]),
            row(2, b, 1000.0, [774.921_368_920_938, 0.0, 0.0, 0.0, 0.0]),
        ];
        let seg = TailraceSegments::from_rows(&rows, "Plant");
        assert!(
            seg.is_ok(),
            "a knot gap within the relative tolerance must be accepted, got: {seg:?}"
        );
    }

    /// A genuine metre-scale discontinuity at the same ~775 m level (well
    /// beyond the relative tolerance) is still rejected — the tolerance must
    /// have power, not just admit the reported calibration residual.
    #[test]
    fn c0_metre_scale_gap_beyond_tolerance_is_still_rejected() {
        let b = 416.0;
        let rows = vec![
            row(1, 0.0, b, [775.0, 0.0, 0.0, 0.0, 0.0]),
            row(2, b, 1000.0, [773.5, 0.0, 0.0, 0.0, 0.0]),
        ];
        let err = TailraceSegments::from_rows(&rows, "Plant").unwrap_err();
        match err {
            FphaFittingError::TailraceDiscontinuity {
                h_left, h_right, ..
            } => {
                assert!((h_left - 775.0).abs() < 1e-12);
                assert!((h_right - 773.5).abs() < 1e-12);
            }
            other => panic!("expected TailraceDiscontinuity, got {other:?}"),
        }
    }

    #[test]
    fn clamp_below_and_above_equals_edge_eval() {
        let rows = vec![row(1, 0.0, 1000.0, [5.0, 0.001, 0.0, 0.0, 0.0])];
        let seg = TailraceSegments::from_rows(&rows, "Plant").unwrap();
        assert_eq!(seg.evaluate(-50.0), seg.evaluate(0.0));
        assert_eq!(seg.evaluate(5000.0), seg.evaluate(1000.0));
    }

    #[test]
    fn genuine_quartic_with_negative_coeffs_matches_hand_horner() {
        // Coefficients carry negative coefficient_2 and coefficient_3 (reference data shape).
        let coeffs = [320.0, 1.0e-3, -3.1e-7, -2.0e-11, 5.0e-15];
        let rows = vec![row(1, 0.0, 1500.0, coeffs)];
        let seg = TailraceSegments::from_rows(&rows, "Plant").unwrap();

        let q = 900.0;
        let [a0, a1, a2, a3, a4] = coeffs;
        // Independent hand Horner.
        let hand = (((a4 * q + a3) * q + a2) * q + a1) * q + a0;
        assert!((seg.evaluate(q) - hand).abs() < 1e-9);
    }

    #[test]
    fn evaluate_is_deterministic_to_bits() {
        let rows = vec![row(1, 0.0, 1500.0, [320.0, 1.0e-3, -3.1e-7, 0.0, 0.0])];
        let seg_a = TailraceSegments::from_rows(&rows, "Plant").unwrap();
        // A clone built from an identical-content row slice.
        let seg_b = TailraceSegments::from_rows(&rows.clone(), "Plant").unwrap();

        let q = 723.456;
        assert_eq!(seg_a.evaluate(q).to_bits(), seg_a.evaluate(q).to_bits());
        assert_eq!(seg_a.evaluate(q).to_bits(), seg_b.evaluate(q).to_bits());
    }

    // ── Family-collection tests ─────────────────────────────────────────────

    /// Build a single-segment family row with explicit family key and constant
    /// height. The constant `h` is encoded as the degree-0 coefficient so the
    /// family evaluates to `h` for any `outflow_m3s`.
    fn family_row(
        hydro_id: i32,
        family_id: i32,
        downstream_reference_level_m: Option<f64>,
        h: f64,
    ) -> TailraceCurveRow {
        TailraceCurveRow {
            hydro_id: EntityId::from(hydro_id),
            family_id,
            downstream_reference_level_m,
            segment_id: 1,
            outflow_min_m3s: 0.0,
            outflow_max_m3s: 1000.0,
            coefficient_0: h,
            coefficient_1: 0.0,
            coefficient_2: 0.0,
            coefficient_3: 0.0,
            coefficient_4: 0.0,
        }
    }

    #[test]
    fn single_family_ignores_downstream_level() {
        // One keyless family with constant height 7.0; the level argument must
        // not change the result.
        let rows = vec![family_row(1, 1, None, 7.0)];
        let fams = TailraceFamilies::from_rows(&rows, "Plant").unwrap();

        assert!((fams.evaluate(400.0, Some(900.0)) - 7.0).abs() < 1e-9);
        assert!((fams.evaluate(400.0, None) - 7.0).abs() < 1e-9);
        // A different level still yields the same single-family height.
        assert_eq!(
            fams.evaluate(400.0, Some(100.0)).to_bits(),
            fams.evaluate(400.0, Some(900.0)).to_bits()
        );
    }

    #[test]
    fn two_family_mid_level_interpolates() {
        // Families at 880 m (height 10) and 890 m (height 20). At L = 885 the
        // linear blend is the midpoint, 15.
        let rows = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, Some(890.0), 20.0),
        ];
        let fams = TailraceFamilies::from_rows(&rows, "Plant").unwrap();

        assert!((fams.evaluate(400.0, Some(885.0)) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn level_below_and_above_range_clamp_to_nearest_family() {
        let rows = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, Some(890.0), 20.0),
        ];
        let fams = TailraceFamilies::from_rows(&rows, "Plant").unwrap();

        // 870 < 880 ⇒ clamp to the lowest family (height 10).
        assert!((fams.evaluate(400.0, Some(870.0)) - 10.0).abs() < 1e-9);
        // 900 > 890 ⇒ clamp to the highest family (height 20).
        assert!((fams.evaluate(400.0, Some(900.0)) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn none_level_multi_family_uses_lowest_family_level() {
        let rows = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, Some(890.0), 20.0),
        ];
        let fams = TailraceFamilies::from_rows(&rows, "Plant").unwrap();

        // Unresolved level ⇒ the lowest-family_level family (880 m, height 10).
        assert!((fams.evaluate(400.0, None) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn keyless_multi_family_is_family_key_missing() {
        // Two families but the second carries no downstream_reference_level_m ⇒ ambiguous.
        let rows = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, None, 20.0),
        ];
        let err = TailraceFamilies::from_rows(&rows, "Plant").unwrap_err();
        match err {
            FphaFittingError::TailraceFamilyKeyMissing { family_count, .. } => {
                assert_eq!(family_count, 2);
            }
            other => panic!("expected TailraceFamilyKeyMissing, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_is_deterministic_across_input_family_orderings() {
        // Same two families supplied in two different family_id input orders
        // within the slice. The total_cmp family sort makes the evaluated
        // result bit-identical regardless of input order.
        let forward = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, Some(890.0), 20.0),
        ];
        let reversed = vec![
            family_row(1, 2, Some(890.0), 20.0),
            family_row(1, 1, Some(880.0), 10.0),
        ];
        let fams_fwd = TailraceFamilies::from_rows(&forward, "Plant").unwrap();
        let fams_rev = TailraceFamilies::from_rows(&reversed, "Plant").unwrap();

        let l = Some(883.25);
        assert_eq!(
            fams_fwd.evaluate(412.0, l).to_bits(),
            fams_rev.evaluate(412.0, l).to_bits()
        );
    }

    #[test]
    fn family_id_order_inverted_from_level_order_still_brackets_by_level() {
        // family_id=1 carries the HIGHER level (890) and family_id=2 the LOWER
        // (880): the per-plant family_id order is the reverse of the level order.
        // Ordering by level (not family_id) is what makes `evaluate`'s clamp
        // bounds `[min, max]` well-formed and the bracketing correct; a family_id
        // primary sort would leave `[890, 880]`, panicking the clamp and
        // bracketing backwards.
        let rows = vec![
            family_row(1, 1, Some(890.0), 20.0),
            family_row(1, 2, Some(880.0), 10.0),
        ];
        let fams = TailraceFamilies::from_rows(&rows, "Plant").unwrap();

        // Mid-level interpolation, clamp-below, and clamp-above all resolve by
        // level regardless of the family_id ordering.
        assert!((fams.evaluate(400.0, Some(885.0)) - 15.0).abs() < 1e-9);
        assert!((fams.evaluate(400.0, Some(870.0)) - 10.0).abs() < 1e-9);
        assert!((fams.evaluate(400.0, Some(900.0)) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn build_map_groups_by_hydro_id() {
        // Two plants, one with two families and one with a single family.
        let rows = vec![
            family_row(1, 1, Some(880.0), 10.0),
            family_row(1, 2, Some(890.0), 20.0),
            family_row(2, 1, None, 5.0),
        ];
        let map = build_tailrace_families_map(&rows).unwrap();

        assert_eq!(map.len(), 2);
        let p1 = map.get(&EntityId::from(1)).unwrap();
        assert!((p1.evaluate(400.0, Some(885.0)) - 15.0).abs() < 1e-9);
        let p2 = map.get(&EntityId::from(2)).unwrap();
        assert!((p2.evaluate(400.0, Some(123.0)) - 5.0).abs() < 1e-9);
    }
}
