//! Build script for cobre-sddp.
//!
//! This script compiles the vendored `libqhull_r` reentrant convex-hull library
//! (`vendor/qhull/src/libqhull_r/`) into a static archive (`libqhull_r.a`) via the
//! `cc` crate — no `CMake`, no bindgen, no libclang. The produced archive is linked
//! into `cobre-sddp`; the Rust FFI bindings that consume its symbols land in a
//! later step. Until then the archive carries no undefined references back into
//! Rust, so linking it is a no-op-safe addition.
//!
//! The build runs identically for the `HiGHS` and `CLP` LP-solver backends: it reads
//! no `CARGO_FEATURE_*` variables and never invokes the LP solver's toolchain.

// Build scripts routinely use expect/panic for unrecoverable configuration
// errors. Allow these lints here since there is no caller to propagate errors to.
#![allow(clippy::expect_used, clippy::panic, clippy::manual_assert)]

use std::env;
use std::path::PathBuf;

/// The 17 `libqhull_r` translation units, enumerated explicitly rather than
/// runtime-globbed so the build inputs stay auditable and a stray `.c` dropped
/// into the vendored tree cannot silently change what gets compiled. This is the
/// full set the vendoring shipped, including `rboxlib_r.c` (the point-array
/// generator) — harmless to link even though the convex-hull path does not call
/// it.
const QHULL_SOURCES: [&str; 17] = [
    "geom2_r.c",
    "geom_r.c",
    "global_r.c",
    "io_r.c",
    "libqhull_r.c",
    "mem_r.c",
    "merge_r.c",
    "poly2_r.c",
    "poly_r.c",
    "qset_r.c",
    "random_r.c",
    "stat_r.c",
    "user_r.c",
    "usermem_r.c",
    "userprintf_r.c",
    "userprintf_rbox_r.c",
    "rboxlib_r.c",
];

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    let qhull_dir = manifest_dir.join("vendor/qhull/src/libqhull_r");

    // Probe a representative translation unit so an uninitialized submodule
    // yields a message naming the expected path instead of a raw `cc`/linker
    // error pointing at a missing file deep in the build output.
    if !qhull_dir.join("libqhull_r.c").exists() {
        panic!(
            "qhull source not found under crates/cobre-sddp/vendor/qhull/src/libqhull_r/. \
             Run: git submodule update --init --recursive"
        );
    }

    // The directory directive covers `.c`/`.h` edits and additions/removals
    // within the vendored tree, not just edits to already-tracked files.
    println!("cargo:rerun-if-changed=vendor/qhull/src/libqhull_r/");

    let mut build = cc::Build::new();
    build.include(&qhull_dir);

    // qhull is entirely third-party vendored source — we authored none of it, so
    // its C-compiler warnings (e.g. `-Wunused-but-set-variable`) are silenced
    // rather than forwarded as `cargo:warning=` lines that would clutter every
    // build and conflict with the zero-warnings policy. This mirrors the
    // LP-solver build, which keeps `.warnings(true)` only for cobre's own thin C
    // wrappers and silences vendored code. The shim added at the seam below gets
    // its own `cc::Build` with `.warnings(true)` — our code stays under scrutiny
    // while qhull stays quiet.
    build.warnings(false);

    // Always compile qhull optimized regardless of the Rust profile. qhull is
    // numeric library code; an unoptimized build is materially slower and would
    // mislead even during development — same release-always rationale the LP
    // solver build follows.
    build.opt_level(2);

    for source in QHULL_SOURCES {
        let path = qhull_dir.join(source);
        if !path.exists() {
            panic!(
                "qhull source {source} not found under \
                 crates/cobre-sddp/vendor/qhull/src/libqhull_r/. \
                 The vendored submodule is incomplete; run: \
                 git submodule update --init --recursive"
            );
        }
        build.file(path);
    }

    build.compile("qhull_r");

    // The C shim `csrc/qhull_wrapper.c` is a thin reentrant wrapper exposing a
    // stable convex-hull entry point to Rust; unlike the vendored build above,
    // it compiles with `.warnings(true)` since it's cobre's own code. It
    // includes `qhull_ra.h` from the vendored tree, so the same include dir is
    // required; it links against the `qhull_r` archive compiled above (both
    // archives are linked into the crate, resolving the shim's qhull symbols).
    let shim_dir = manifest_dir.join("csrc");
    let shim_source = shim_dir.join("qhull_wrapper.c");
    if !shim_source.exists() {
        panic!(
            "qhull shim source not found at crates/cobre-sddp/csrc/qhull_wrapper.c. \
             The thin reentrant wrapper is required."
        );
    }
    println!("cargo:rerun-if-changed=csrc/qhull_wrapper.c");
    println!("cargo:rerun-if-changed=csrc/qhull_wrapper.h");

    let mut shim = cc::Build::new();
    shim.include(&qhull_dir);
    shim.include(&shim_dir);
    shim.warnings(true);
    shim.opt_level(2);
    shim.file(&shim_source);
    shim.compile("cobre_qhull_shim");
}
