//! Safe Rust wrapper over the qhull 3-D convex-hull C shim.
//!
//! The single entry point [`convex_hull_3d`] is the only safe caller of the
//! `unsafe extern "C"` shim binding in [`ffi`]. It owns the two contracts the
//! C shim deliberately does not: RAII release of the shim-allocated plane
//! buffer, and the canonical sort-in / sort-out that makes the output a
//! deterministic function of the *set* of input points. The shim
//! is a pure transform — points in, facet hyperplanes out — so determinism is
//! the wrapper's responsibility, not the shim's.

mod ffi;

use std::os::raw::{c_double, c_int};

use thiserror::Error;

/// A facet hyperplane of a 3-D convex hull.
///
/// Encodes the plane `nx·x + ny·y + nz·z + d = 0` with `normal = (nx, ny, nz)`
/// unit-length and `offset = d` — qhull's native `facet->normal` /
/// `facet->offset` convention, copied through unchanged by the shim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hyperplane3d {
    /// Unit-length outward normal `(nx, ny, nz)`.
    pub normal: [f64; 3],
    /// Plane offset `d` in `nx·x + ny·y + nz·z + d = 0`.
    pub offset: f64,
}

/// Failure modes of [`convex_hull_3d`].
///
/// Each non-zero shim status maps to a distinct variant; the input-validation
/// variant is raised by the wrapper *before* the FFI call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum HullError {
    /// An input coordinate was non-finite (NaN or ±∞). Guarded in Rust before
    /// the FFI call so the canonical total order never observes a NaN (it would
    /// otherwise be undefined).
    #[error("convex hull input contains a non-finite coordinate")]
    NonFiniteInput,

    /// Degenerate or insufficient input: fewer than 4 affinely-independent
    /// points, so no full-dimensional 3-D hull exists. Maps shim status
    /// `COBRE_QHULL_ERR_DEGENERATE`.
    #[error("convex hull input is degenerate or has fewer than 4 affinely-independent points")]
    Degenerate,

    /// qhull initialization or computation failed (internal/precision error
    /// that is neither degenerate input nor allocation). Maps shim status
    /// `COBRE_QHULL_ERR_COMPUTE`.
    #[error("qhull convex-hull computation failed")]
    Compute,

    /// Memory allocation failed (qhull-internal or the shim's output buffer).
    /// Maps shim status `COBRE_QHULL_ERR_ALLOC`.
    #[error("convex hull allocation failed")]
    Alloc,

    /// The shim returned a status code outside the documented set.
    #[error("qhull shim returned unknown status code {0}")]
    UnknownStatus(c_int),
}

/// RAII owner of the shim-malloc'd plane buffer.
///
/// Drop frees the buffer through [`ffi::cobre_qhull_free`] so the buffer is
/// released with the same allocator that produced it (the shim's
/// `malloc`/`free`) on *every* path — including an early `?` return or a panic
/// while copying the facets out. The buffer is never freed elsewhere.
struct PlaneBuffer {
    ptr: *mut c_double,
}

impl Drop for PlaneBuffer {
    fn drop(&mut self) {
        // SAFETY:
        //   * `self.ptr` is either NULL (a no-op for cobre_qhull_free, per the
        //     shim contract) or a pointer the shim returned from
        //     cobre_qhull_convex_hull_3d on a COBRE_QHULL_OK status.
        //   * This is the sole owner of that pointer and the sole free site;
        //     the buffer is freed exactly once.
        unsafe { ffi::cobre_qhull_free(self.ptr) };
    }
}

/// Compute the 3-D convex hull of `points`, returning its facet hyperplanes in
/// canonical order.
///
/// # Determinism (contract)
///
/// Canonical-sort-in / canonical-sort-out is the load-bearing invariant: the
/// input points are sorted by a NaN-free total order (`f64::total_cmp` on the
/// `(x, y, z)` tuple) *before* the FFI call, and the returned facets are sorted
/// by the same total order on the `(nx, ny, nz, d)` tuple *after*. Two clouds
/// that are permutations of each other therefore produce an element-for-element
/// bit-identical `Vec<Hyperplane3d>` — output ordering is independent of input
/// ordering. The wrong-but-compiling alternative — passing the caller's order
/// straight through, or sorting only one side — leaks input permutation into
/// the output and breaks the declaration-order-invariance hard rule.
///
/// # Errors
///
/// - [`HullError::NonFiniteInput`] if any coordinate is NaN or ±∞ (guarded
///   before the FFI call so the total order never observes a NaN).
/// - [`HullError::Degenerate`] for fewer than 4 affinely-independent points.
/// - [`HullError::Compute`] / [`HullError::Alloc`] for a qhull internal or
///   allocation failure.
pub(crate) fn convex_hull_3d(points: &[[f64; 3]]) -> Result<Vec<Hyperplane3d>, HullError> {
    // Guard before sorting so the total order below never observes a NaN.
    if points.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
        return Err(HullError::NonFiniteInput);
    }

    let mut sorted: Vec<[f64; 3]> = points.to_vec();
    sorted.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
            .then_with(|| a[2].total_cmp(&b[2]))
    });

    let n_points = sorted.len();
    let mut flat: Vec<f64> = Vec::with_capacity(n_points * 3);
    for p in &sorted {
        flat.extend_from_slice(p);
    }

    // An overflowing count is a caller bug; map it to Compute, never truncate.
    let Ok(n_points_c) = c_int::try_from(n_points) else {
        return Err(HullError::Compute);
    };

    let mut out_planes: *mut c_double = std::ptr::null_mut();
    let mut out_n_facets: c_int = 0;

    // SAFETY:
    //   * `flat.as_ptr()` is valid for reads of `3 * n_points` `f64`s: `flat`
    //     was built with exactly `n_points * 3` elements, matching the
    //     `n_points_c` count passed alongside it — the shim's `points` /
    //     `n_points` precondition.
    //   * `out_planes` and `out_n_facets` are addresses of live, initialized
    //     local variables (`null_mut()` / `0`); the shim writes through them
    //     exactly once and only on a `COBRE_QHULL_OK` return. They do not
    //     alias `flat` (distinct locals, distinct allocations).
    //   * The returned `out_planes` pointer is dereferenced ONLY when the
    //     status is `COBRE_QHULL_OK`; on any non-zero status the shim sets it
    //     to NULL and we never read it (the early `match` returns first).
    //   * On `COBRE_QHULL_OK` the buffer holds exactly `4 * out_n_facets`
    //     `f64`s (the shim's documented layout `[nx,ny,nz,d, ...]`), which is
    //     the slice length we reconstruct below.
    //   * The returned pointer is wrapped in `PlaneBuffer` immediately, whose
    //     `Drop` frees it via the paired `cobre_qhull_free` on every path.
    let status = unsafe {
        ffi::cobre_qhull_convex_hull_3d(
            flat.as_ptr(),
            n_points_c,
            std::ptr::addr_of_mut!(out_planes),
            std::ptr::addr_of_mut!(out_n_facets),
        )
    };

    if status != ffi::COBRE_QHULL_OK {
        // On any non-zero status the shim guarantees `out_planes == NULL`, so
        // there is nothing to free; map the code to a typed error.
        return Err(match status {
            ffi::COBRE_QHULL_ERR_DEGENERATE => HullError::Degenerate,
            ffi::COBRE_QHULL_ERR_COMPUTE => HullError::Compute,
            ffi::COBRE_QHULL_ERR_ALLOC => HullError::Alloc,
            other => HullError::UnknownStatus(other),
        });
    }

    // Take ownership of the shim buffer NOW, before any further fallible step,
    // so its `Drop` frees it even if a later `?`/panic unwinds.
    let buffer = PlaneBuffer { ptr: out_planes };

    let Ok(n_facets) = usize::try_from(out_n_facets) else {
        return Err(HullError::Compute);
    };

    let mut facets: Vec<Hyperplane3d> = Vec::with_capacity(n_facets);
    if n_facets > 0 {
        // SAFETY:
        //   * `buffer.ptr` is non-NULL (status was COBRE_QHULL_OK with
        //     `n_facets > 0`, so the shim malloc'd a populated buffer).
        //   * The shim's documented layout is `4 * n_facets` `f64`s laid out
        //     `[nx,ny,nz,d, ...]`; `4 * n_facets` is exactly the slice length.
        //   * `c_double` is `f64`; the buffer is suitably aligned (it came from
        //     the shim's `malloc`, which is aligned for `double`).
        //   * The slice borrow lives only within this block and `buffer`
        //     outlives it, so the pointer stays valid; the buffer is not
        //     mutated or freed until `buffer` drops after the copy completes.
        let planes: &[c_double] = unsafe { std::slice::from_raw_parts(buffer.ptr, n_facets * 4) };
        for q in planes.chunks_exact(4) {
            facets.push(Hyperplane3d {
                normal: [q[0], q[1], q[2]],
                offset: q[3],
            });
        }
    }

    // qhull's normals are finite, so total_cmp observes only finite values here too.
    facets.sort_by(|a, b| {
        a.normal[0]
            .total_cmp(&b.normal[0])
            .then_with(|| a.normal[1].total_cmp(&b.normal[1]))
            .then_with(|| a.normal[2].total_cmp(&b.normal[2]))
            .then_with(|| a.offset.total_cmp(&b.offset))
    });

    Ok(facets)
}

#[cfg(test)]
mod tests {
    use super::{HullError, Hyperplane3d, convex_hull_3d};

    /// The unit tetrahedron `[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]`.
    const TETRA: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    /// Deterministic seedable PRNG for the determinism gate's permutations.
    ///
    /// `thread_rng` is deliberately avoided: the gate must produce the identical
    /// permutation sequence on every machine and run, or a failure could not be
    /// reproduced from its reported seed.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A `usize` in `0..bound` (`bound > 0`) via Lemire's multiply-shift
        /// reduction; fully determined by the PRNG state.
        fn below(&mut self, bound: usize) -> usize {
            usize::try_from((u128::from(self.next_u64()) * (bound as u128)) >> 64)
                .expect("multiply-shift reduction is < bound, which fits usize")
        }
    }

    /// Fisher–Yates shuffle of `points` in place, driven entirely by `rng`.
    fn shuffle(points: &mut [[f64; 3]], rng: &mut SplitMix64) {
        for i in (1..points.len()).rev() {
            let j = rng.below(i + 1);
            points.swap(i, j);
        }
    }

    /// A production-like `(V, Q, z)` point cloud: generation-head samples on a
    /// `V × Q` storage/turbined-flow grid, plus the closing point `(V_max,
    /// Q_max, 0)`.
    ///
    /// `z` is a mildly concave surface (`a·√V + b·√Q − c·V·Q`) so the cloud is
    /// not a trivial box — its upper hull has several distinct facets, which is
    /// what makes it a representative determinism stress: a cloud whose hull is
    /// one or two facets would not exercise qhull's facet ordering. The closing
    /// point sits below that surface so it becomes a genuine hull vertex,
    /// mirroring how FPHA closes the tailrace cloud at zero generation.
    fn representative_cloud() -> Vec<[f64; 3]> {
        const V_MAX: f64 = 100.0;
        const Q_MAX: f64 = 50.0;
        const GRID: u32 = 6;

        // `f64::from(u32)`, not `as f64`: the latter trips cast_precision_loss.
        let span = f64::from(GRID - 1);
        let mut cloud = Vec::with_capacity((GRID * GRID + 1) as usize);
        for iv in 0..GRID {
            for iq in 0..GRID {
                let v = V_MAX * f64::from(iv) / span;
                let q = Q_MAX * f64::from(iq) / span;
                let z = 0.8 * v.sqrt() + 1.2 * q.sqrt() - 0.001 * v * q;
                cloud.push([v, q, z]);
            }
        }
        cloud.push([V_MAX, Q_MAX, 0.0]);
        cloud
    }

    /// Bit-identical equality of two facet lists: every normal component and
    /// every offset compared via `f64::to_bits()`, never a float tolerance.
    ///
    /// A tolerance is the wrong tool here on purpose — the determinism contract
    /// is that two permutations of one cloud yield *byte-identical* output, so a
    /// tolerance would mask exactly the ordering-dependent drift this gate must
    /// catch.
    fn facets_bit_identical(a: &[Hyperplane3d], b: &[Hyperplane3d]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(p, q)| {
                p.normal[0].to_bits() == q.normal[0].to_bits()
                    && p.normal[1].to_bits() == q.normal[1].to_bits()
                    && p.normal[2].to_bits() == q.normal[2].to_bits()
                    && p.offset.to_bits() == q.offset.to_bits()
            })
    }

    /// Does `plane` describe the same geometric plane as `nx·x+ny·y+nz·z+d=0`,
    /// up to sign (outward-normal orientation) and unit-normalization?
    ///
    /// qhull fixes the outward-normal sign and unit-normalizes; the wrapper's
    /// canonical sort fixes the facet *order*. But the test must not assume
    /// either sign or normalization constant, so it compares the *direction*
    /// of the normal and the plane's distance-from-origin, scale-free.
    fn matches_plane(plane: &Hyperplane3d, nx: f64, ny: f64, nz: f64, d: f64) -> bool {
        let want = [nx, ny, nz];
        let want_norm = (nx * nx + ny * ny + nz * nz).sqrt();
        let got = plane.normal;
        let got_norm = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2]).sqrt();
        if got_norm == 0.0 || want_norm == 0.0 {
            return false;
        }
        for sign in [1.0_f64, -1.0_f64] {
            let normal_matches =
                (0..3).all(|i| (got[i] / got_norm - sign * want[i] / want_norm).abs() < 1e-9);
            let offset_matches = (plane.offset / got_norm - sign * d / want_norm).abs() < 1e-9;
            if normal_matches && offset_matches {
                return true;
            }
        }
        false
    }

    #[test]
    fn tetrahedron_returns_four_faces_matching_hand_computed_planes() {
        let hull = convex_hull_3d(&TETRA).expect("tetrahedron is a valid 3-D hull");
        assert_eq!(hull.len(), 4, "tetrahedron has exactly 4 triangular faces");

        // The four faces of the unit tetrahedron, plane nx·x+ny·y+nz·z+d=0:
        //   x = 0          -> normal (1,0,0),  d = 0
        //   y = 0          -> normal (0,1,0),  d = 0
        //   z = 0          -> normal (0,0,1),  d = 0
        //   x + y + z = 1  -> normal (1,1,1),  d = -1
        let expected = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, -1.0),
        ];

        for (nx, ny, nz, d) in expected {
            assert!(
                hull.iter().any(|p| matches_plane(p, nx, ny, nz, d)),
                "expected a facet matching plane ({nx},{ny},{nz},{d}); got {hull:?}"
            );
        }
    }

    #[test]
    fn shuffled_input_yields_bit_identical_facets() {
        let baseline = convex_hull_3d(&TETRA).expect("baseline hull");

        let shuffled = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let permuted = convex_hull_3d(&shuffled).expect("shuffled hull");

        assert_eq!(
            baseline, permuted,
            "canonical sort-in/out must make the facet Vec element-for-element bit-identical"
        );
    }

    #[test]
    fn cube_returns_six_faces_matching_hand_computed_planes() {
        // The 8 corners of the unit cube [0,1]^3.
        let cube = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let hull = convex_hull_3d(&cube).expect("unit cube is a valid 3-D hull");

        // The six faces of the unit cube, plane nx·x+ny·y+nz·z+d=0:
        //   x = 0 -> (1,0,0), d =  0      x = 1 -> (1,0,0), d = -1
        //   y = 0 -> (0,1,0), d =  0      y = 1 -> (0,1,0), d = -1
        //   z = 0 -> (0,0,1), d =  0      z = 1 -> (0,0,1), d = -1
        let expected = [
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, -1.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, -1.0),
        ];

        // qhull triangulates each square face into two coplanar triangular
        // facets, so the cube returns 12 facets spanning 6 distinct planes —
        // assert set-membership of every hand-computed plane plus that the hull
        // covers exactly those 6 planes, rather than asserting a raw facet count
        // (which would encode qhull's triangulation policy, not the geometry).
        for (nx, ny, nz, d) in expected {
            assert!(
                hull.iter().any(|p| matches_plane(p, nx, ny, nz, d)),
                "expected a facet matching plane ({nx},{ny},{nz},{d}); got {hull:?}"
            );
        }
        assert!(
            hull.iter().all(|p| expected
                .iter()
                .any(|&(nx, ny, nz, d)| matches_plane(p, nx, ny, nz, d))),
            "every returned facet must lie on one of the 6 cube faces; got {hull:?}"
        );
    }

    #[test]
    fn representative_cloud_is_bit_identical_across_seeded_shuffles() {
        // Bit-identical output regardless of input ordering — the
        // declaration-order invariance hard rule, here as a seeded-permutation
        // Rust test (the hull is pure, with no MPI surface).
        //
        // A non-deterministic result is a BLOCKING ESCALATION (fallback:
        // qhull-sys + own safe wrapper), never a silent change of qhull options
        // or the sort to force this green.
        let base = representative_cloud();
        let reference = convex_hull_3d(&base).expect("representative cloud is a valid 3-D hull");
        assert!(
            reference.len() >= 4,
            "representative cloud must yield a multi-facet hull to stress facet \
             ordering; got {} facets",
            reference.len()
        );

        // 32 seeds, at or above the 16+ the determinism gate requires.
        for seed in 0u64..32 {
            let mut rng = SplitMix64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
            let mut permuted = base.clone();
            shuffle(&mut permuted, &mut rng);

            let result =
                convex_hull_3d(&permuted).expect("permuted representative cloud stays valid");

            assert!(
                facets_bit_identical(&reference, &result),
                "BLOCKING: qhull output is NOT bit-identical across input orderings \
                 (permutation seed = {seed}). The canonical sort-in/sort-out must \
                 make the facet Vec element-for-element to_bits()-equal to the \
                 reference. reference = {reference:?}, permuted = {result:?}"
            );
        }
    }

    #[test]
    fn coplanar_points_return_degenerate_error() {
        // Three coplanar points: no full-dimensional 3-D hull exists.
        let coplanar = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = convex_hull_3d(&coplanar);
        assert_eq!(
            result,
            Err(HullError::Degenerate),
            "a degenerate cloud must return the typed Degenerate error, not panic"
        );
    }

    #[test]
    fn coplanar_quad_is_handled_without_garbage_normals() {
        // Four points on the z = 0 plane: a 2-D set embedded in 3-D with enough
        // points (>= 4) to pass the shim's pre-check and reach qh_new_qhull, so
        // this exercises the qh_new_qhull return-code path that the 3-point
        // coplanar_points_return_degenerate_error case short-circuits before.
        // qhull may resolve a narrow hull (Ok) or report the thin geometry as a
        // precision/degenerate error; either is acceptable, but the shim must
        // never walk a partially-built facet list and emit a non-finite normal
        // or crash. The shim checks qh_new_qhull's return code before iterating
        // facets; this guards that contract.
        let coplanar = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        match convex_hull_3d(&coplanar) {
            Ok(facets) => {
                for f in &facets {
                    assert!(
                        f.normal.iter().all(|c| c.is_finite()) && f.offset.is_finite(),
                        "coplanar hull must not emit non-finite plane coefficients; got {f:?}"
                    );
                }
            }
            Err(HullError::Degenerate | HullError::Compute) => {}
            Err(other) => panic!("unexpected error variant for coplanar input: {other:?}"),
        }
    }
}
