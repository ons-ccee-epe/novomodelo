//! FPHA fitting error type.

use std::fmt;
/// Errors that arise during FPHA fitting geometry validation or evaluation.
///
/// Every variant carries the rejected hydro's name in its `hydro_name` field.
#[derive(Debug)]
pub(crate) enum FphaFittingError {
    /// No VHA curve points were provided.
    ///
    /// A single point IS accepted (a constant run-of-river forebay yielding
    /// `γ_V = 0`); only the zero-row case reaches this variant.
    InsufficientPoints {
        hydro_name: String,
        /// Number of points provided.
        count: usize,
    },

    /// `volume_hm3` is not strictly increasing between consecutive rows.
    ///
    /// Strict monotonicity maps each volume to a unique interpolation interval;
    /// duplicate volumes produce a zero-length segment and undefined derivatives.
    NonMonotonicVolume {
        hydro_name: String,
        /// Row whose volume is not strictly greater than the previous row's.
        index: usize,
        /// Volume at the previous row (hm³).
        v_prev: f64,
        /// Volume at the current row (hm³); must satisfy `v_curr > v_prev`.
        v_curr: f64,
    },

    /// `height_m` decreases between consecutive rows.
    ///
    /// Heights must be non-decreasing: greater volume always gives a higher or
    /// equal water surface.
    NonMonotonicHeight {
        hydro_name: String,
        /// Row whose height is strictly less than the previous row's.
        index: usize,
        /// Height at the previous row (m).
        h_prev: f64,
        /// Height at the current row (m); must satisfy `h_curr >= h_prev`.
        h_curr: f64,
    },

    /// Both absolute and percentile bounds set for the same dimension (the min
    /// and max pairs are each mutually exclusive).
    ConflictingFittingWindow {
        hydro_name: String,
        /// Human-readable description of the conflict.
        detail: String,
    },

    /// The resolved volume range is inverted (`v_max < v_min`).
    ///
    /// A range that collapses to a single point (`v_min == v_max`) is NOT an
    /// error: it reroutes through the single-volume run-of-river path
    /// ([`resolve_fitting_bounds`](super::geometry::resolve_fitting_bounds)).
    EmptyFittingWindow {
        hydro_name: String,
        /// Resolved lower bound (hm³).
        v_min: f64,
        /// Resolved upper bound (hm³).
        v_max: f64,
    },

    /// A discretization count was too small to define a valid grid interval.
    ///
    /// `n_flow_points`/`n_spillage_points` must be `>= 2` and `max_planes_per_hydro`
    /// `>= 1`; `n_volume_points` must be `>= 2` on the multi-volume path only — the
    /// single-volume path synthesizes its own two samples and is exempt.
    InsufficientDiscretization {
        hydro_name: String,
        /// `"volume"`, `"turbine"`, `"spillage"`, or `"max_planes_per_hydro"`.
        dimension: String,
        /// The provided value (< 2 for grid dimensions, < 1 for max planes).
        value: usize,
    },

    /// The least-squares `α_FPHA` correction factor is not strictly positive.
    ///
    /// A non-positive `α` would flip every coefficient sign or collapse the
    /// envelope to zero, both physically invalid.
    NonPositiveAlpha {
        hydro_name: String,
        /// The computed `α_FPHA` value.
        alpha: f64,
    },

    /// The fitting pipeline produced zero valid hyperplanes (e.g. net head ≤ 0
    /// everywhere, so the hull yields no upper-envelope facet).
    NoHyperplanesProduced { hydro_name: String },

    /// The 3-D production cloud was too degenerate for a convex-hull fit.
    ///
    /// The hull needs at least four affinely-independent points; a cloud that
    /// collapses onto a line or plane (e.g. constant net head with no V/Q
    /// dependence) yields no facet. Mapped from the hull's degenerate status
    /// rather than panicking, so one pathological hydro does not abort the loop.
    DegenerateProductionCloud { hydro_name: String },

    /// A fitted hyperplane has a coefficient with the wrong sign.
    ///
    /// Physical planes satisfy `gamma_v ≥ 0`, `gamma_q ≥ 0`, `gamma_s ≤ 0`
    /// (spillage raises tailrace, reducing net head).
    InvalidCoefficient {
        hydro_name: String,
        /// Index of the offending hyperplane in the selected set.
        plane_index: usize,
        /// Which coefficient failed and its value.
        detail: String,
    },

    /// Consecutive tailrace segments do not tile the outflow domain.
    ///
    /// Each segment's lower bound must meet the previous segment's upper bound; a
    /// gap leaves outflow values unowned, an overlap makes ownership ambiguous.
    TailraceGap {
        hydro_name: String,
        /// Upper bound of the lower-indexed segment (m³/s).
        outflow_max_prev: f64,
        /// Lower bound of the higher-indexed segment (m³/s); must equal
        /// `outflow_max_prev` within tolerance.
        outflow_min_curr: f64,
    },

    /// Consecutive tailrace segments disagree at their shared boundary.
    ///
    /// The piecewise quartic must be C0-continuous: both segments must evaluate
    /// to the same elevation at the boundary they share.
    TailraceDiscontinuity {
        hydro_name: String,
        /// Shared boundary outflow (m³/s).
        boundary: f64,
        /// Elevation from the lower-indexed segment at `boundary` (m).
        h_left: f64,
        /// Elevation from the higher-indexed segment at `boundary` (m).
        h_right: f64,
    },

    /// A multi-family tailrace table has a family with no downstream reference
    /// level.
    ///
    /// Every family must carry a level so the families can be ordered and
    /// bracketed; a `None` level is admissible only for a single-family plant. The
    /// owning check is
    /// [`TailraceFamilies::from_rows`](super::tailrace::TailraceFamilies::from_rows).
    TailraceFamilyKeyMissing {
        hydro_name: String,
        /// Number of families found (> 1 when this fires).
        family_count: usize,
    },
}

impl std::fmt::Display for FphaFittingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientPoints { hydro_name, count } => write!(
                f,
                "hydro '{hydro_name}': VHA curve has {count} point(s); \
                 at least 1 is required (a single point defines a constant forebay)"
            ),
            Self::NonMonotonicVolume {
                hydro_name,
                index,
                v_prev,
                v_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': volume is not strictly increasing at index {index}: \
                 v[{index}]={v_curr} is not greater than v[{}]={v_prev}",
                index - 1
            ),
            Self::NonMonotonicHeight {
                hydro_name,
                index,
                h_prev,
                h_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': height decreases at index {index}: \
                 h[{index}]={h_curr} < h[{}]={h_prev}",
                index - 1
            ),
            Self::ConflictingFittingWindow { hydro_name, detail } => write!(
                f,
                "hydro '{hydro_name}': conflicting fitting window configuration: {detail}"
            ),
            Self::EmptyFittingWindow {
                hydro_name,
                v_min,
                v_max,
            } => write!(
                f,
                "hydro '{hydro_name}': fitting window is empty after resolution: \
                 v_min={v_min} >= v_max={v_max}"
            ),
            Self::InsufficientDiscretization {
                hydro_name,
                dimension,
                value,
            } => write!(
                f,
                "hydro '{hydro_name}': discretization count for '{dimension}' is {value}, \
                 which is below the minimum required"
            ),
            Self::NonPositiveAlpha { hydro_name, alpha } => write!(
                f,
                "hydro '{hydro_name}': least-squares alpha_FPHA {alpha} is not strictly positive; \
                 alpha_FPHA must be > 0"
            ),
            Self::DegenerateProductionCloud { hydro_name } => write!(
                f,
                "hydro '{hydro_name}': production cloud is degenerate (collinear or coplanar); \
                 a 3-D convex hull needs at least 4 affinely-independent points"
            ),
            Self::NoHyperplanesProduced { hydro_name } => write!(
                f,
                "hydro '{hydro_name}': fitting pipeline produced zero valid hyperplanes; \
                 check that net head is positive over the fitting grid"
            ),
            Self::InvalidCoefficient {
                hydro_name,
                plane_index,
                detail,
            } => write!(
                f,
                "hydro '{hydro_name}': hyperplane {plane_index} has an invalid coefficient: \
                 {detail}"
            ),
            Self::TailraceGap {
                hydro_name,
                outflow_max_prev,
                outflow_min_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace segments leave a gap or overlap at the \
                 boundary: previous outflow_max={outflow_max_prev} does not meet next outflow_min={outflow_min_curr}"
            ),
            Self::TailraceDiscontinuity {
                hydro_name,
                boundary,
                h_left,
                h_right,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace segments are discontinuous at q={boundary}: \
                 left tailrace_level={h_left} != right tailrace_level={h_right}"
            ),
            Self::TailraceFamilyKeyMissing {
                hydro_name,
                family_count,
            } => write!(
                f,
                "hydro '{hydro_name}': tailrace table has {family_count} families but at least \
                 one carries no downstream reference level (downstream_reference_level_m); a keyless family is \
                 only valid for a single-family plant"
            ),
        }
    }
}

impl std::error::Error for FphaFittingError {}
