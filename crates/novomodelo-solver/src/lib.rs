//! # cobre-solver
//!
//! LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines a backend-agnostic interface for mathematical programming
//! solvers, with a default [HiGHS](https://highs.dev) backend:
//!
//! - **Solver trait**: unified API for LP and MIP problem construction, solving,
//!   and dual/basis extraction.
//! - **`HiGHS` backend** (`highs` feature, on by default): an open-source
//!   solver suited to iterative LP solving in power system optimization;
//!   exposed as `HighsSolver` and `HighsProfile`.
//! - **`CLP` backend** (`clp` feature, off by default): optional CLP/`CoinUtils`
//!   backend exposed as `ClpSolver` and `ClpProfile`; conformance-validated
//!   as a drop-in implementing the same [`SolverInterface`]. Requires the Clp
//!   and `CoinUtils` submodules to be initialized (`git submodule update --init
//!   --recursive`) before the first build with `--features clp`.
//! - **Basis management**: warm-starting support for iterative algorithms
//!   that solve sequences of related LPs.
//!
//! Exactly one LP backend is enabled at compile time — both-on and neither-on
//! are each a `compile_error!` — so the `Active*` aliases and `active_solver_*`
//! functions resolve to the CLP backend under `--features clp`, or the `HiGHS`
//! backend under the default `highs` feature.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic,
        clippy::too_many_lines
    )
)]

#[cfg(all(feature = "highs", feature = "clp"))]
compile_error!(
    "enable exactly one LP backend: `highs` OR `clp`. \
     To use CLP, build with `--no-default-features --features clp`."
);

#[cfg(not(any(feature = "highs", feature = "clp")))]
compile_error!("no LP backend selected: enable exactly one of `highs` or `clp`.");

pub(crate) mod ffi;

#[cfg(feature = "clp")]
pub(crate) use ffi::clp as clp_ffi;

pub mod basis_status;
pub use basis_status::BasisStatus;

pub mod trait_def;
pub use trait_def::SolverInterface;

pub mod types;
pub use types::{
    Basis, LpSolution, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate,
};

pub mod profile;
pub use profile::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL};

pub mod freeze;
pub use freeze::{FreezeScratch, freeze_rows_into_template};

pub mod backends;
pub use backends::profiled::ProfiledSolver;

#[cfg(feature = "highs")]
pub use backends::highs::{HighsProfile, HighsSolver, PresolveKind, highs_version};

#[cfg(feature = "clp")]
pub use backends::clp::{ClpAlgorithm, ClpProfile, ClpSolver, clp_version};

// Module-path shims exposing `cobre_solver::highs` / `cobre_solver::clp` for
// downstream code that imports a backend by module rather than via the re-exports
// above.
#[cfg(feature = "clp")]
pub use backends::clp;
#[cfg(feature = "highs")]
pub use backends::highs;

/// The compile-time-selected active LP solver backend, referenced by downstream
/// crates instead of a concrete type to bind one binary to one solver with zero
/// runtime dispatch.
#[cfg(feature = "clp")]
pub type ActiveSolver = ClpSolver;

/// The compile-time-selected active LP solver backend, referenced by downstream
/// crates instead of a concrete type to bind one binary to one solver with zero
/// runtime dispatch.
#[cfg(all(feature = "highs", not(feature = "clp")))]
pub type ActiveSolver = HighsSolver;

/// The solver profile type of the active backend; equals
/// `<ActiveSolver as SolverInterface>::Profile`.
#[cfg(feature = "clp")]
pub type ActiveProfile = ClpProfile;

/// The solver profile type of the active backend; equals
/// `<ActiveSolver as SolverInterface>::Profile`.
#[cfg(all(feature = "highs", not(feature = "clp")))]
pub type ActiveProfile = HighsProfile;

/// Returns the version string of the active backend (a backend-agnostic entry
/// point for metadata wiring).
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_version() -> String {
    clp_version()
}

/// Returns the version string of the active backend (a backend-agnostic entry
/// point for metadata wiring).
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_version() -> String {
    highs_version()
}

/// Returns the active backend's display name — the same string its
/// [`SolverInterface::name`] returns (`"CLP"` or `"HiGHS"`).
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_name() -> &'static str {
    "CLP"
}

/// Returns the active backend's display name — the same string its
/// [`SolverInterface::name`] returns (`"CLP"` or `"HiGHS"`).
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_name() -> &'static str {
    "HiGHS"
}

/// Returns the canonical lowercase backend id recorded in persisted output
/// metadata (`OutputContext.solver`) — the stable id, distinct from the
/// mixed-case display name of [`active_solver_name`].
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_metadata_id() -> &'static str {
    "clp"
}

/// Returns the canonical lowercase backend id recorded in persisted output
/// metadata (`OutputContext.solver`) — the stable id, distinct from the
/// mixed-case display name of [`active_solver_name`].
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_metadata_id() -> &'static str {
    "highs"
}

#[cfg(feature = "test-support")]
pub mod test_support {
    //! Test-only utilities for configuring solver options and probing the raw
    //! C API from integration tests.
    //!
    //! Do **not** enable this feature in production builds. Every item here is
    //! a thin, verbatim pass-through into the sealed `crate::ffi` module,
    //! bypassing all safe-wrapper validation, so integration tests can poke
    //! solver options or exercise the raw C API below the safe wrapper.

    #[cfg(feature = "highs")]
    mod highs_bridge {
        use std::os::raw::{c_char, c_double, c_int, c_void};

        /// # Safety
        /// Same contract as `crate::ffi::cobre_highs_get_double_option`.
        pub unsafe fn cobre_highs_get_double_option(
            highs: *const c_void,
            option: *const c_char,
            value: *mut c_double,
        ) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::cobre_highs_get_double_option(highs, option, value) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::cobre_highs_get_int_option`.
        pub unsafe fn cobre_highs_get_int_option(
            highs: *const c_void,
            option: *const c_char,
            value: *mut c_int,
        ) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::cobre_highs_get_int_option(highs, option, value) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::cobre_highs_set_double_option`.
        pub unsafe fn cobre_highs_set_double_option(
            highs: *mut c_void,
            option: *const c_char,
            value: c_double,
        ) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::cobre_highs_set_double_option(highs, option, value) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::cobre_highs_set_int_option`.
        pub unsafe fn cobre_highs_set_int_option(
            highs: *mut c_void,
            option: *const c_char,
            value: c_int,
        ) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::cobre_highs_set_int_option(highs, option, value) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::cobre_highs_set_string_option`.
        pub unsafe fn cobre_highs_set_string_option(
            highs: *mut c_void,
            option: *const c_char,
            value: *const c_char,
        ) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::cobre_highs_set_string_option(highs, option, value) }
        }
    }
    #[cfg(feature = "highs")]
    pub use highs_bridge::*;

    #[cfg(feature = "clp")]
    mod clp_bridge {
        use std::os::raw::{c_double, c_int, c_void};

        /// Thin pass-through of `crate::ffi::clp::CLP_STATUS_OPTIMAL`.
        pub const CLP_STATUS_OPTIMAL: i32 = crate::ffi::clp::CLP_STATUS_OPTIMAL;

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_create`.
        #[must_use]
        pub unsafe fn cobre_clp_create() -> *mut c_void {
            // SAFETY: verbatim forward; the wrapped fn takes no arguments.
            unsafe { crate::ffi::clp::cobre_clp_create() }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_destroy`.
        pub unsafe fn cobre_clp_destroy(model: *mut c_void) {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_destroy(model) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_set_log_level`.
        pub unsafe fn cobre_clp_set_log_level(model: *mut c_void, value: c_int) {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_set_log_level(model, value) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_load_problem`.
        // Rationale: mirrors `cobre_clp_load_problem`'s C signature verbatim; the
        // probe test needs the 1:1 forwarding contract, not a re-shaped API.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn cobre_clp_load_problem(
            model: *mut c_void,
            num_cols: c_int,
            num_rows: c_int,
            col_starts: *const c_int,
            row_indices: *const c_int,
            values: *const c_double,
            col_lower: *const c_double,
            col_upper: *const c_double,
            objective: *const c_double,
            row_lower: *const c_double,
            row_upper: *const c_double,
        ) {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe {
                crate::ffi::clp::cobre_clp_load_problem(
                    model,
                    num_cols,
                    num_rows,
                    col_starts,
                    row_indices,
                    values,
                    col_lower,
                    col_upper,
                    objective,
                    row_lower,
                    row_upper,
                );
            }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_dual`.
        pub unsafe fn cobre_clp_dual(model: *mut c_void, if_values_pass: c_int) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_dual(model, if_values_pass) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_objective_value`.
        #[must_use]
        pub unsafe fn cobre_clp_objective_value(model: *const c_void) -> c_double {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_objective_value(model) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_status`.
        #[must_use]
        pub unsafe fn cobre_clp_status(model: *const c_void) -> c_int {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_status(model) }
        }

        /// # Safety
        /// Same contract as `crate::ffi::clp::cobre_clp_get_row_price`.
        #[must_use]
        pub unsafe fn cobre_clp_get_row_price(model: *const c_void) -> *const c_double {
            // SAFETY: verbatim forward; caller upholds the wrapped fn's contract.
            unsafe { crate::ffi::clp::cobre_clp_get_row_price(model) }
        }
    }
    #[cfg(feature = "clp")]
    pub use clp_bridge::*;
}

#[cfg(all(test, any(feature = "highs", feature = "clp")))]
mod active_alias_tests {
    use crate::{
        ActiveProfile, ActiveSolver, SolverInterface, active_solver_metadata_id,
        active_solver_name, active_solver_version,
    };

    /// Compile-time assertion that `ActiveProfile` is exactly
    /// `<ActiveSolver as SolverInterface>::Profile`; never called — the body
    /// only type-checks when the two match, hence `dead_code`.
    #[allow(dead_code)]
    const fn _assert_profile_identity(
        p: ActiveProfile,
    ) -> <ActiveSolver as SolverInterface>::Profile {
        p
    }

    #[test]
    fn active_aliases_resolve_per_feature() {
        let name = active_solver_name();
        let version = active_solver_version();
        assert!(!version.is_empty(), "active version must be non-empty");
        assert!(
            version.contains('.'),
            "active version `{version}` should contain a `.`"
        );

        #[cfg(feature = "clp")]
        assert_eq!(name, "CLP");
        #[cfg(all(feature = "highs", not(feature = "clp")))]
        assert_eq!(name, "HiGHS");
    }

    #[test]
    fn active_solver_name_matches_instance_name() {
        let solver = ActiveSolver::new().expect("active solver must construct");
        assert_eq!(solver.name(), active_solver_name());
    }

    /// HiGHS-only build (`--features highs`, no `clp`): the active backend
    /// resolves to `HiGHS`. The both-on configuration cannot compile (a
    /// `compile_error!` in `lib.rs` rejects it), so no runnable test is needed
    /// for that contract.
    #[cfg(all(feature = "highs", not(feature = "clp")))]
    #[test]
    fn active_backend_is_highs_when_only_highs_enabled() {
        assert_eq!(active_solver_name(), "HiGHS");
        assert_eq!(active_solver_metadata_id(), "highs");
    }

    #[cfg(feature = "clp")]
    #[test]
    fn active_backend_is_clp_when_clp_enabled() {
        assert_eq!(active_solver_name(), "CLP");
        assert_eq!(active_solver_metadata_id(), "clp");
    }
}
