//! Raw FFI binding to the qhull C shim (`csrc/qhull_wrapper.{c,h}`).
//!
//! These are the only `unsafe extern "C"` declarations for the convex-hull
//! path. They map 1:1 to the `cobre_qhull_*` functions in
//! `csrc/qhull_wrapper.h`. Use the safe wrapper in the parent module
//! ([`super::convex_hull_3d`]) rather than calling these directly.

use std::os::raw::{c_double, c_int};

// ============================================================
// Status codes (mirroring csrc/qhull_wrapper.h)
// ============================================================

/// Success; the out-params are populated.
pub const COBRE_QHULL_OK: c_int = 0;
/// qhull init/computation failure (neither degenerate input nor allocation).
pub const COBRE_QHULL_ERR_COMPUTE: c_int = 1;
/// Degenerate or insufficient input: fewer than 4 affinely-independent points.
pub const COBRE_QHULL_ERR_DEGENERATE: c_int = 2;
/// Memory allocation failure (qhull-internal or the shim's output buffer).
pub const COBRE_QHULL_ERR_ALLOC: c_int = 3;

// ============================================================
// Convex hull
// ============================================================

unsafe extern "C" {
    /// Compute the 3-D convex hull of `n_points` points and return its facet
    /// hyperplanes. Wraps `cobre_qhull_convex_hull_3d`.
    ///
    /// `points` is a borrowed array of `3 * n_points` doubles, row-major
    /// `[x0,y0,z0, ...]`. On `COBRE_QHULL_OK`, `*out_planes` is a
    /// shim-malloc'd array of `4 * (*out_n_facets)` doubles laid out
    /// `[nx,ny,nz,d, ...]` that the caller MUST release with
    /// [`cobre_qhull_free`]. On any non-zero return, `*out_planes` is NULL
    /// and `*out_n_facets` is 0.
    pub fn cobre_qhull_convex_hull_3d(
        points: *const c_double,
        n_points: c_int,
        out_planes: *mut *mut c_double,
        out_n_facets: *mut c_int,
    ) -> c_int;

    /// Free a plane array returned by [`cobre_qhull_convex_hull_3d`].
    /// Wraps `cobre_qhull_free`. Passing NULL is a no-op.
    pub fn cobre_qhull_free(planes: *mut c_double);
}
