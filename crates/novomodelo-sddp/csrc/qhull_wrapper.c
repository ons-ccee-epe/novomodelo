/* Thin C shim around the reentrant qhull library (libqhull_r).
 *
 * Computes the 3-D convex hull of a flat point array and returns the facet
 * hyperplanes (unit normal + offset) through a stable, qhull-free C surface.
 * See qhull_wrapper.h for the full contract (determinism options, error
 * trapping, hyperplane convention, status codes).
 *
 * Every qhull call is confined to this translation unit; the public surface
 * exposes no qhull types.
 */

#include "qhull_wrapper.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

/* qhull_ra.h pulls in libqhull_r.h (data types, FORALLfacets, QHULL_LIB_CHECK,
 * qh_zero / qh_new_qhull / qh_freeqhull / qh_memfreeshort) plus the reentrant
 * internals. The header is included by relative name; the vendored
 * `libqhull_r/` directory is on the compiler include path (set in build.rs). */
#include "qhull_ra.h"

/* Dimension of the hull this shim computes. The flat point stride and the
 * per-facet output width (4 = 3 normal components + 1 offset) are derived from
 * it; only DIM == 3 is supported by the fixed [nx,ny,nz,d] output layout. */
#define COBRE_QHULL_DIM 3

/* Per-facet output width: nx, ny, nz, d. */
#define COBRE_QHULL_PLANE_STRIDE 4

/* Fixed qhull option string. "qhull" is the conventional command prefix; "Qt"
 * triangulates the hull into simplicial facets so each facet has one
 * unambiguous unit normal; "Pp" suppresses qhull's precision warnings (a
 * narrow/degenerate cloud would otherwise print multi-line diagnostics to
 * stderr from every parallel per-hydro worker — genuine errors are surfaced
 * through the returned status instead). "Pp" is output-only and does not change
 * the hull geometry (the determinism gate verifies the facet output stays
 * bit-identical). Joggle ("QJ") is deliberately omitted — no randomized
 * perturbation. See the header's determinism contract. */
static const char COBRE_QHULL_FLAGS[] = "qhull Qt Pp";

/* Map a qhull errexit code (qh_ERR*, libqhull_r.h) to a shim status code.
 * Called only on the error path (exitcode != 0). */
static int cobre_qhull_map_error(int exitcode) {
    switch (exitcode) {
        case qh_ERRmem:
            /* Insufficient memory inside qhull. */
            return COBRE_QHULL_ERR_ALLOC;
        case qh_ERRsingular:
        case qh_ERRprec:
        case qh_ERRtopology:
        case qh_ERRwide:
            /* Singular / precision / nearly-degenerate input: too few
             * affinely-independent points to form a full 3-D hull, or a
             * geometry too thin for qhull to resolve. */
            return COBRE_QHULL_ERR_DEGENERATE;
        case qh_ERRinput:
        case qh_ERRqhull:
        case qh_ERRother:
        default:
            /* Bad input dimensions, internal qhull error, or anything else. */
            return COBRE_QHULL_ERR_COMPUTE;
    }
}

int cobre_qhull_convex_hull_3d(
    const double* points,
    int           n_points,
    double**      out_planes,
    int*          out_n_facets
) {
    /* Establish the failure post-condition up front: on every non-OK return the
     * out-params are NULL/0 so the caller never frees a dangling buffer. */
    if (out_planes != NULL) {
        *out_planes = NULL;
    }
    if (out_n_facets != NULL) {
        *out_n_facets = 0;
    }

    if (points == NULL || out_planes == NULL || out_n_facets == NULL) {
        return COBRE_QHULL_ERR_COMPUTE;
    }

    /* A 3-D hull needs at least DIM+1 = 4 points; reject smaller clouds before
     * calling qhull so the caller gets the degenerate status without paying for
     * a guaranteed-failing qhull run. (qhull would also reject these, mapped to
     * the same code, but this keeps the contract explicit.) */
    if (n_points < COBRE_QHULL_DIM + 1) {
        return COBRE_QHULL_ERR_DEGENERATE;
    }

    /* Verify the linked libqhull_r matches the headers this shim compiled
     * against (struct sizes / reentrant ABI). On mismatch qh_lib_check aborts;
     * it is a build-configuration guard, not a runtime input error. */
    QHULL_LIB_CHECK

    /* Capture qhull's error stream instead of leaking it to the process's
     * stderr: a near-coplanar cloud's QH6154 precision diagnostic (and any
     * other qh_fprintf(qh, qh->ferr, ...) text) is written here instead, from
     * every parallel per-hydro worker. `open_memstream` grows as needed, so a
     * multi-line diagnostic never overflows a fixed buffer. On `ERRmem`
     * exhaustion (the only realistic `open_memstream` failure) fall back to
     * `tmpfile()`, and only if THAT also fails fall back to `stderr` — never
     * NULL, which `qh_fprintf` treats as a fatal internal abort (see the
     * header's error-trapping contract). The captured text is discarded
     * unread once qhull returns, on both outcomes: the soft-recovery path's
     * fit already succeeded (see below), and the hard-failure path already
     * returns a typed status the Rust caller maps to a hydro-naming error,
     * without needing the raw qhull text. */
    char*  diag_buf = NULL;
    size_t diag_size = 0;
    FILE*  diag_stream = NULL;
#if defined(_WIN32)
    /* open_memstream is POSIX-2008 and absent on MSVC — calling it is an
     * unresolved external symbol at link time, not a runtime NULL the check
     * below could catch — so Windows starts the fallback chain at tmpfile();
     * diag_buf/diag_size stay NULL/0 and free(diag_buf) below is a no-op. */
    diag_stream = tmpfile();
#else
    diag_stream = open_memstream(&diag_buf, &diag_size);
    if (diag_stream == NULL) {
        diag_stream = tmpfile();
    }
#endif
    FILE* err_stream = (diag_stream != NULL) ? diag_stream : stderr;

    /* Reentrant qhull instance lives on the stack — no global state, so
     * concurrent hulls on different threads cannot interfere. */
    qhT qh_qh;
    qhT* qh = &qh_qh;
    qh_zero(qh, err_stream);

    double* planes = NULL;
    int     status = COBRE_QHULL_OK;

    /* Install qhull's longjmp error target. A qhull error (degenerate input,
     * precision failure, OOM, internal error) jumps back here with a non-zero
     * exitcode instead of aborting the process. */
    int exitcode = setjmp(qh->errexit);
    if (!exitcode) {
        /* errexit is now valid: clear the "no errexit available" guard so
         * qh_errexit uses our setjmp target. */
        qh->NOerrexit = False;

        /* ismalloc = False: `points` is borrowed; qhull must not free or take
         * ownership of it. The cast drops const because qh_new_qhull takes a
         * non-const coordT*; qhull does not mutate the array when joggle is off
         * (no "QJ"), so this is sound. NULL outfile suppresses result printing;
         * `err_stream` (never NULL, see above) is the error sink qhull's
         * QH6154-class diagnostics land on instead of the process's stderr;
         * the "Pp" flag additionally suppresses the separate precision-warning
         * text qhull emits outside the hard errexit path. */
        int qhull_exit = qh_new_qhull(
            qh,
            COBRE_QHULL_DIM,
            n_points,
            (coordT*)points,
            False,
            (char*)COBRE_QHULL_FLAGS,
            NULL,
            err_stream
        );

        /* qh_new_qhull installs its OWN setjmp over qh->errexit for the duration
         * of hull construction and RETURNS a non-zero qh_ERR* code on failure
         * instead of long-jumping back to the setjmp above; `qhull_exit` captures
         * it. We do NOT reject outright on a non-zero code: a flat / coplanar
         * cloud — e.g. a constant-head production surface, whose generation is
         * linear in turbined flow — makes qhull raise a precision error ("initial
         * simplex is flat") yet leaves the correct narrow-hull facet in
         * facet_list, and that facet IS the plane the data lies in (the fit we
         * want; deterministic case D05 depends on it). Instead we read whatever
         * usable facets qhull produced and surface the error code only when none
         * exist. Skipping any facet with facet->normal == NULL makes a
         * partially-built list safe to read: a freshly allocated facet on the
         * precision-error path has a NULL normal, and dereferencing it is the
         * wrong-but-compiling hazard this guards. */
        facetT* facet = NULL;
        int     n_facets = 0;
        FORALLfacets {
            if (facet->upperdelaunay || facet->normal == NULL) {
                continue;
            }
            n_facets++;
        }

        if (n_facets == 0) {
            /* No usable hyperplane facet. If qhull reported an error, surface its
             * mapped code; otherwise the cloud is degenerate (too thin to hull). */
            status = (qhull_exit != 0)
                ? cobre_qhull_map_error(qhull_exit)
                : COBRE_QHULL_ERR_DEGENERATE;
        } else {
            /* Allocate the output buffer with the shim's allocator so the Rust
             * side releases it via cobre_qhull_free (same allocator). */
            size_t count = (size_t)n_facets * (size_t)COBRE_QHULL_PLANE_STRIDE;
            planes = (double*)malloc(count * sizeof(double));
            if (planes == NULL) {
                status = COBRE_QHULL_ERR_ALLOC;
            } else {
                /* Second pass: write [nx, ny, nz, d] per facet. qhull's
                 * facet->normal is already unit-length and facet->offset is d
                 * for the hyperplane normal·x + offset = 0, so the values are
                 * copied through unchanged. The skip predicate matches the
                 * counting pass so idx stays in lockstep with n_facets. */
                int idx = 0;
                FORALLfacets {
                    if (facet->upperdelaunay || facet->normal == NULL) {
                        continue;
                    }
                    double* dst = planes + (size_t)idx * COBRE_QHULL_PLANE_STRIDE;
                    dst[0] = (double)facet->normal[0];
                    dst[1] = (double)facet->normal[1];
                    dst[2] = (double)facet->normal[2];
                    dst[3] = (double)facet->offset;
                    idx++;
                }

                *out_planes = planes;
                *out_n_facets = n_facets;
            }
        }
    } else {
        /* An error from a qhull call OTHER than qh_new_qhull (e.g. resource
         * teardown) long-jumped here; map its exit code to a status. */
        status = cobre_qhull_map_error(exitcode);
    }

    /* Block further longjmp-based error handling while we tear down: any error
     * during freeing must not jump back into the (now-consumed) setjmp frame. */
    qh->NOerrexit = True;

    /* Release qhull resources on EVERY path — success and error alike — so no
     * qhull memory leaks. qh_freeqhull frees the qhull data structures;
     * qh_memfreeshort frees qhull's short-memory pools. !qh_ALL retains the
     * memory-arena bookkeeping that qh_memfreeshort then reclaims. */
    qh_freeqhull(qh, !qh_ALL);
    int curlong = 0;
    int totlong = 0;
    qh_memfreeshort(qh, &curlong, &totlong);

    /* If we hit an error after allocating the output buffer, drop it so we do
     * not leak and so the out-params stay NULL/0 (set above). With the current
     * control flow `planes` is only non-NULL on the success path, but free the
     * buffer defensively if a future edit allocates before a later failure. */
    if (status != COBRE_QHULL_OK && planes != NULL) {
        free(planes);
        *out_planes = NULL;
        *out_n_facets = 0;
    }

    /* Close and discard the captured diagnostic stream on every path. Closing
     * an `open_memstream` finalizes `diag_buf`/`diag_size`, which we then free
     * without reading; closing a `tmpfile()` fallback deletes it. `err_stream`
     * is `stderr` only when both captures failed, in which case `diag_stream`
     * is NULL and there is nothing to close here. */
    if (diag_stream != NULL) {
        fclose(diag_stream);
    }
    free(diag_buf);

    return status;
}

void cobre_qhull_free(double* planes) {
    /* free(NULL) is a no-op, so no NULL guard is needed; the same allocator
     * (malloc) that produced the buffer releases it here. */
    free(planes);
}
