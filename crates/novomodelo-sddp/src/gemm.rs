//! Cache-blocked GEMM primitive for batched matrix-vector evaluation.
//!
//! The single entry point [`gemm_block`] is the only call site in
//! `cobre-sddp` for [`matrixmultiply::dgemm`]. The wrapper isolates the
//! one `unsafe` block this crate's `cut_selection.rs` needs so the call
//! site itself stays in safe code.

/// Compute `v_block` (`k_rows × m_len`, row-major) = `coef` · `state_blockᵀ`.
///
/// `coef` is `k_rows × d`, row-major. `state_block` is `m_len × d`,
/// row-major. `v_block` is the caller-provided output buffer of length
/// `k_rows * m_len`, exclusively borrowed.
///
/// `matrixmultiply::dgemm`'s signature is `(m, k, n)` where `m` is rows of
/// A and C, `k` is the inner dimension, and `n` is cols of B and C. This
/// wrapper passes `(k_rows, d, m_len)` and supplies row-major strides for
/// A (`rsa=d, csa=1`) and C (`rsc=m_len, csc=1`). For B — the
/// `state_block` — the wrapper passes `rsb=1, csb=d` so dgemm reads
/// `state_block` as `d × m_len` (i.e., as the transpose of the caller's
/// `m_len × d` row-major layout). The resulting GEMM `A · Bᵀ` writes
/// `v_block[k * m_len + col]` = inner product of row `k` of `coef` with
/// row `col` of `state_block`.
///
/// Returns immediately (a no-op) if any dimension is zero — `matrixmultiply` is
/// undefined for zero-sized arguments. This early-return precedes the
/// `debug_assert` checks, so callers may pass empty slices for zero-sized inputs.
///
/// # Determinism
///
/// `matrixmultiply` is single-threaded (the workspace pins
/// `default-features = false`) and uses a fixed cache-blocked algorithm.
/// Same input → bit-identical f64 output on any IEEE-754 target compiled
/// with `target-feature=+avx2,+fma`.
///
/// # Panics
///
/// Panics in debug builds if a slice mismatches `(k_rows, d, m_len)` and no
/// dimension is zero. Release builds invoke UB on undersized slices for non-zero
/// dimensions — the caller's contract is to size them correctly.
#[inline]
pub(crate) fn gemm_block(
    coef: &[f64],
    state_block: &[f64],
    k_rows: usize,
    d: usize,
    m_len: usize,
    v_block: &mut [f64],
) {
    if k_rows == 0 || d == 0 || m_len == 0 {
        return;
    }

    debug_assert_eq!(
        coef.len(),
        k_rows * d,
        "gemm_block: coef slice must be exactly k_rows*d"
    );
    debug_assert_eq!(
        state_block.len(),
        m_len * d,
        "gemm_block: state_block must be exactly m_len*d"
    );
    debug_assert_eq!(
        v_block.len(),
        k_rows * m_len,
        "gemm_block: v_block must be exactly k_rows*m_len"
    );

    // SAFETY:
    //   * The zero-dim early-return above guarantees k_rows, d, m_len
    //     are all > 0. `coef.len() >= k_rows*d`,
    //     `state_block.len() >= m_len*d`, and `v_block.len() >= k_rows*m_len`
    //     are required by the caller's contract. Debug builds assert this
    //     via the three debug_assert_eq calls above; release builds rely on
    //     the caller — the function's doc comment warns that mismatched
    //     dimensions invoke UB in release.
    //   * `coef` and `state_block` are immutable shared borrows;
    //     `v_block` is an exclusive mutable borrow. The three slices
    //     come from distinct owners, so the raw pointers passed to
    //     dgemm cannot alias.
    //   * Row-major strides (rsa=d, csa=1) match the asserted layout of
    //     `coef`. Output strides (rsc=m_len, csc=1) match the asserted
    //     layout of `v_block`. Strides for `state_block` (rsb=1, csb=d)
    //     request the transpose access pattern; the `m_len*d` length is
    //     exactly enough storage either way.
    //   * dgemm is single-threaded in this build (workspace dep is
    //     `default-features = false`), so no thread-safety bug exists.
    //
    // `as isize`: matrixmultiply::dgemm's stride parameters are isize.
    // Workspace caps usable strides at the underlying slice lengths, all
    // of which fit comfortably in isize on every supported target.
    #[allow(clippy::cast_possible_wrap)]
    unsafe {
        matrixmultiply::dgemm(
            k_rows,
            d,
            m_len,
            1.0,
            coef.as_ptr(),
            d as isize,
            1,
            state_block.as_ptr(),
            1,
            d as isize,
            0.0,
            v_block.as_mut_ptr(),
            m_len as isize,
            1,
        );
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::gemm_block;

    #[test]
    fn gemm_block_matches_naive_reference_small() {
        const K: usize = 5;
        const D: usize = 3;
        const M_LEN: usize = 4;

        let coef: Vec<f64> = (0..K * D).map(|i| (i as f64) * 0.1).collect();
        let state: Vec<f64> = (0..M_LEN * D).map(|i| (i as f64) * 0.01 - 0.5).collect();
        let mut v = [0.0_f64; K * M_LEN];

        gemm_block(&coef, &state, K, D, M_LEN, &mut v);

        let mut expected = [0.0_f64; K * M_LEN];
        for k in 0..K {
            for m in 0..M_LEN {
                let mut acc = 0.0_f64;
                for d in 0..D {
                    acc += coef[k * D + d] * state[m * D + d];
                }
                expected[k * M_LEN + m] = acc;
            }
        }

        for i in 0..(K * M_LEN) {
            assert!(
                (v[i] - expected[i]).abs() < 1e-12,
                "gemm_block[{i}] = {} but expected {}",
                v[i],
                expected[i],
            );
        }
    }

    #[test]
    fn gemm_block_zero_dimensions_no_op() {
        // Three calls, one zero dimension each. None must panic
        // or trigger UB; v_block remains untouched.
        let mut v: Vec<f64> = Vec::new();
        gemm_block(&[], &[], 0, 5, 3, &mut v);

        let mut v2 = vec![1.0_f64; 15];
        gemm_block(&[], &[], 5, 0, 3, &mut v2);
        assert!(v2.iter().all(|&x| x == 1.0));

        let mut v3 = vec![2.0_f64; 0];
        gemm_block(&[], &[], 5, 3, 0, &mut v3);
    }
}
