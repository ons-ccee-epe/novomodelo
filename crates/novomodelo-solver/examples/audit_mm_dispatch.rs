//! Audit: verify `matrixmultiply::dgemm` produces byte-identical f64 output on
//! hosts of different microarchitecture when compiled with
//! `target-feature=+avx2,+fma,+sse4.2`.
//!
//! # Why this binary lives in `cobre-solver`
//!
//! `matrixmultiply::dgemm` is `pub unsafe fn`. The workspace default
//! `unsafe_code = "forbid"` (set in the root `Cargo.toml` under
//! `[workspace.lints.rust]`) blocks any `unsafe` block in crates that do not
//! override the lint. `cobre-solver` overrides `unsafe_code = "allow"` because
//! its `HiGHS` FFI bindings require it, so the audit example lives here.
//!
//! # Usage
//!
//! ```text
//! cargo build --release -p cobre-solver --example audit_mm_dispatch
//! ./target/release/examples/audit_mm_dispatch > /tmp/mm.bin
//! sha256sum /tmp/mm.bin
//! ```
//!
//! Repeat on a second host with a different microarchitecture (one without
//! AVX-512, one with) and compare the two SHA-256 hashes. If the hashes match,
//! `matrixmultiply::dgemm` is byte-identical across hosts under
//! `target-feature=+avx2,+fma,+sse4.2`. If they differ, the runtime CPU
//! dispatch is selecting different micro-kernels on the two hosts.
//!
//! # Output
//!
//! - **stdout**: raw little-endian bytes of the `K × N` output matrix.
//! - **stderr**: a hint to pipe stdout through `sha256sum` for cross-host
//!   comparison.
//!
//! The binary exits with status `1` if `dgemm` produces all-zero output.

// This example is a one-shot diagnostic, not library code. `expect`, `panic`,
// `print_stderr`, and `as_*` casts are intentional and not bugs to be linted.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_possible_wrap,
    clippy::float_cmp
)]

use std::io::Write;

/// Hand-rolled splitmix64 — reproducible PRNG with no external crate dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a `rows × cols` row-major matrix of f64 values in `[-0.5, 0.5)`,
/// seeded deterministically from `seed`.
fn fill_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut buf = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1_u64 << 52) - 1);
        let exp_bias = 1023_u64;
        let f = f64::from_bits((exp_bias << 52) | bits) - 1.5;
        buf.push(f);
    }
    buf
}

fn main() {
    const K: usize = 100;
    const D: usize = 50;
    const N: usize = 16;

    let a = fill_matrix(K, D, 0x1234_5678_9ABC_DEF0);
    let b = fill_matrix(D, N, 0x0FED_CBA9_8765_4321);
    let mut c = vec![0.0_f64; K * N];

    // SAFETY: `a` is `K * D = 5000` f64 values laid out row-major (rsa=D,
    // csa=1), `b` is `D * N = 800` f64 values row-major (rsb=N, csb=1), and
    // `c` is `K * N = 1600` f64 values row-major (rsc=N, csc=1). All three
    // buffers are heap-allocated `Vec<f64>` and properly aligned. `a` and `b`
    // are read-only borrows via `as_ptr`; `c` is the unique mutable borrow via
    // `as_mut_ptr`. There is no aliasing between the three buffers because
    // they are independent allocations. `dgemm` reads from `a` and `b` and
    // writes to `c`, which matches the borrow direction. The strides match the
    // documented row-major contract for `matrixmultiply::dgemm`.
    unsafe {
        matrixmultiply::dgemm(
            K,
            D,
            N,
            1.0,
            a.as_ptr(),
            D as isize,
            1,
            b.as_ptr(),
            N as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            N as isize,
            1,
        );
    }

    if c.iter().all(|&v| v == 0.0) {
        eprintln!("ERROR: dgemm produced all-zero output");
        std::process::exit(1);
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for v in &c {
        out.write_all(&v.to_le_bytes()).expect("write to stdout");
    }

    eprintln!(
        "Wrote {} bytes ({} f64 values) to stdout; pipe through `sha256sum` to compare across hosts.",
        c.len() * 8,
        c.len()
    );
}
