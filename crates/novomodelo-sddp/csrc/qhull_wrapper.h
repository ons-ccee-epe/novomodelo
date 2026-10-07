#ifndef COBRE_QHULL_WRAPPER_H
#define COBRE_QHULL_WRAPPER_H

/* Thin C shim around the reentrant qhull library (libqhull_r) for use by
 * cobre-sddp FFI bindings.
 *
 * The shim presents a minimal, stable, flat C surface: it takes a flat array of
 * 3-D point coordinates, computes the convex hull, and writes back the facet
 * hyperplanes (unit normal + offset). The Rust side therefore never touches
 * qhull's internal data structures and exposes exactly one `unsafe extern "C"`.
 *
 * All functions use the `cobre_qhull_` prefix and fixed-width / standard C types
 * (double, int, size-free pointers) for FFI safety.
 *
 * This header is intentionally self-contained: it does NOT include any qhull
 * header and exposes NO qhull types, so Rust-facing code never needs to pull in
 * qhull's internal headers. Only <stdint.h>/<stddef.h> are referenced, and the
 * current surface needs neither — they are listed for forward compatibility if a
 * fixed-width count is later added.
 *
 * ---------------------------------------------------------------------------
 * Determinism contract
 * ---------------------------------------------------------------------------
 * qhull is invoked with the fixed option string "qhull Qt Pp":
 *   - "Qt"  triangulates the hull, so every output facet is simplicial and
 *           carries one unambiguous unit normal. This removes the per-facet
 *           normal ambiguity that merged (non-simplicial) facets introduce and
 *           keeps the output a deterministic function of the input ordering.
 *   - "Pp"  suppresses a separate class of qhull precision-warning text outside
 *           the hard errexit path. This is output-only and does not change the
 *           hull geometry.
 *   - Joggle ("QJ") is deliberately ABSENT: joggle perturbs the input with a
 *     pseudo-random offset, which destroys reproducibility. No randomized
 *     perturbation of any kind is enabled.
 * Canonical input/output SORTING is NOT performed here — it lives on the Rust
 * side. This shim is a pure transform: points in, facet hyperplanes out.
 *
 * ---------------------------------------------------------------------------
 * Error-trapping contract
 * ---------------------------------------------------------------------------
 * qhull reports errors via a setjmp/longjmp mechanism rooted at qh->errexit.
 * qh_new_qhull installs its own setjmp for the duration of hull construction and
 * RETURNS a non-zero qh_ERR* code on failure rather than long-jumping out; the
 * shim checks that return value and maps it to one of the status codes below.
 * The shim also installs an outer setjmp target around the surrounding qhull
 * calls (teardown), so an error there is trapped too. Recoverable qhull errors
 * (degenerate input, precision failure, normal out-of-memory) therefore do not
 * abort the host process — they surface as a status code. This is NOT an
 * absolute guarantee: qhull has a few unrecoverable paths (memory-arena
 * corruption, an error raised while already handling an error, a NULL error
 * file) that call qh_exit() and terminate the process, as in upstream qhull.
 * qhull resources are released on every return path — success and error alike.
 *
 * A narrow/degenerate cloud's hard errexit diagnostic (e.g. QH6154, "Initial
 * simplex is flat") is written to a captured in-memory stream
 * (`open_memstream`, falling back to `tmpfile()` and finally to `stderr` —
 * never NULL; Windows lacks `open_memstream` and starts the chain at
 * `tmpfile()`) instead of the process's stderr, so a near-coplanar cloud does
 * not dump multi-line text from every parallel worker. The captured text is
 * discarded unread on every path: the soft-recovery below already succeeds
 * without it, and the hard-failure status already maps to a hydro-naming
 * error on the Rust side.
 *
 * ---------------------------------------------------------------------------
 * Hyperplane convention
 * ---------------------------------------------------------------------------
 * Each output facet is four doubles laid out [nx, ny, nz, d], encoding the
 * plane  nx*x + ny*y + nz*z + d = 0  with the normal (nx, ny, nz) unit-length.
 * This is qhull's native convention: facet->normal is the unit normal and
 * facet->offset is d, so the values are copied through unchanged.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Status codes
 * ========================================================================= */

/** Success: `*out_planes` and `*out_n_facets` are populated. */
#define COBRE_QHULL_OK 0

/** qhull initialization or computation failed (internal / precision / other
 *  qhull error that is not specifically degenerate input or allocation). */
#define COBRE_QHULL_ERR_COMPUTE 1

/** Degenerate or insufficient input: fewer than 4 affinely-independent points,
 *  so no full-dimensional 3-D hull exists (qhull singular/topology error). */
#define COBRE_QHULL_ERR_DEGENERATE 2

/** Memory allocation failed (qhull-internal allocation or the shim's output
 *  buffer allocation). */
#define COBRE_QHULL_ERR_ALLOC 3

/* =========================================================================
 * Convex hull
 * ========================================================================= */

/** Compute the 3-D convex hull of `n_points` points and return its facet
 *  hyperplanes.
 *
 *  Inputs:
 *    - `points`   : pointer to `3 * n_points` doubles, row-major
 *                   `[x0,y0,z0, x1,y1,z1, ...]`. Borrowed; not retained.
 *    - `n_points` : number of input points.
 *
 *  Outputs (on COBRE_QHULL_OK only):
 *    - `*out_planes`   : a shim-malloc'd array of `4 * (*out_n_facets)` doubles,
 *                        laid out `[nx,ny,nz,d, ...]` (one quadruple per hull
 *                        facet, normal unit-length, plane nx*x+ny*y+nz*z+d=0).
 *                        The caller MUST release it with `cobre_qhull_free`.
 *    - `*out_n_facets` : the number of hull facets written.
 *  On any non-zero return, `*out_planes` is set to NULL and `*out_n_facets` to
 *  0; the caller must not free anything.
 *
 *  Returns one of the COBRE_QHULL_* status codes above.
 */
int cobre_qhull_convex_hull_3d(
    const double* points,
    int           n_points,
    double**      out_planes,
    int*          out_n_facets
);

/** Free a plane array returned by `cobre_qhull_convex_hull_3d`.
 *
 *  The free MUST happen through this function so the buffer is released with the
 *  same allocator that produced it (the shim's `malloc`/`free`), independent of
 *  the Rust-side allocator. Passing NULL is a no-op. */
void cobre_qhull_free(double* planes);

#ifdef __cplusplus
}
#endif

#endif /* COBRE_QHULL_WRAPPER_H */
