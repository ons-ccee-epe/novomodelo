//! The [`ClpSolver`] handle wrapper, its lifecycle primitives, and the
//! [`clp_version`] free function.

use std::os::raw::c_void;
use std::time::Instant;

use super::config::ClpProfile;
use crate::Basis;
use crate::{
    DEFAULT_PROFILE_HEURISTIC_SENTINEL, clp_ffi,
    types::{SolverError, SolverStatistics},
};

/// CLP LP solver backend.
///
/// Owns an opaque CLP model handle plus pre-allocated, reusable buffers resized
/// by `load_model` and reused across solves to avoid per-solve allocation. The
/// retained CSC arrays and bound vectors are the canonical, declaration-ordered
/// mirror of the loaded LP: `add_rows`/`set_*_bounds` patch them and reconcile
/// the change into CLP natively (`cobre_clp_add_rows` / `cobre_clp_chg_*`)
/// without rebuilding the model, preserving CLP's factorization/basis. Bound
/// vectors are forwarded to CLP verbatim.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "clp")]
/// # {
/// use cobre_solver::{ClpSolver, SolverInterface};
///
/// let solver = ClpSolver::new().expect("CLP initialisation failed");
/// assert_eq!(solver.name(), "CLP");
/// # }
/// ```
pub struct ClpSolver {
    /// Opaque pointer to the CLP model.
    pub(super) handle: *mut c_void,
    /// Primal column values extracted after each solve.
    pub(super) col_value: Vec<f64>,
    /// Column dual values (reduced costs).
    pub(super) col_dual: Vec<f64>,
    /// Row dual multipliers (shadow prices).
    pub(super) row_dual: Vec<f64>,
    /// Retained CSC column-start offsets (length `num_cols + 1`).
    pub(super) col_starts: Vec<i32>,
    /// Retained CSC row indices for each non-zero (length `num_nz`).
    pub(super) row_indices: Vec<i32>,
    /// Retained CSC non-zero values (length `num_nz`).
    pub(super) values: Vec<f64>,
    /// Retained column lower bounds (length `num_cols`).
    pub(super) col_lower: Vec<f64>,
    /// Retained column upper bounds (length `num_cols`).
    pub(super) col_upper: Vec<f64>,
    /// Retained row lower bounds (length `num_rows`).
    pub(super) row_lower: Vec<f64>,
    /// Retained row upper bounds (length `num_rows`).
    pub(super) row_upper: Vec<f64>,
    /// Retained non-zero count, kept in sync with `values.len()`.
    pub(super) num_nz: usize,
    /// Current number of LP columns (decision variables).
    pub(super) num_cols: usize,
    /// Current number of LP rows (constraints).
    pub(super) num_rows: usize,
    /// Whether a model is currently loaded; guards the `solve`/`get_basis` contract.
    pub(super) has_model: bool,
    /// Accumulated solver statistics. Counters grow monotonically from zero.
    pub(super) stats: SolverStatistics,
    /// Cached solver profile applied by the last profile-setter call.
    pub(super) current_profile: ClpProfile,
}

// SAFETY: `ClpSolver` holds a raw pointer to a CLP C++ object. The CLP handle
// is not thread-safe for concurrent access, but exclusive ownership is
// maintained at all times -- exactly one `ClpSolver` instance owns each handle
// and no shared references to the handle exist. Transferring the `ClpSolver`
// to another thread (via `Send`) is safe because there is no concurrent
// access; the new thread has exclusive ownership. `Sync` is intentionally NOT
// implemented.
unsafe impl Send for ClpSolver {}

impl ClpSolver {
    /// Creates a new CLP solver instance.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if `cobre_clp_create()`
    /// returns a null pointer.
    pub fn new() -> Result<Self, SolverError> {
        // SAFETY: `cobre_clp_create` is a C function with no preconditions.
        // It allocates and returns a new CLP model pointer, or null on
        // allocation failure. The returned pointer is opaque and must be
        // passed back to CLP API functions.
        let handle = unsafe { clp_ffi::cobre_clp_create() };

        if handle.is_null() {
            return Err(SolverError::InternalError {
                message: "CLP instance creation failed: Clp_newModel() returned null".to_string(),
                error_code: None,
            });
        }

        // CLP ships with log level 1 (per-solve progress to stdout), which would
        // pollute CLI/Python output on every solve; force level 0.
        //
        // SAFETY: `handle` is the non-null model just returned by
        // `cobre_clp_create`; `cobre_clp_set_log_level` only forwards the level
        // to `Clp_setLogLevel` on that model and has no other preconditions.
        unsafe { clp_ffi::cobre_clp_set_log_level(handle, 0) };

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_dual: Vec::new(),
            col_starts: Vec::new(),
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            num_nz: 0,
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics::default(),
            current_profile: ClpProfile::default(),
        })
    }

    /// Copies the three CLP-owned solution pointers into the owned buffers.
    ///
    /// CLP-owned pointers are valid only until the next solve. Each copy is
    /// guarded with `if len > 0` because passing a null or dangling pointer to
    /// `std::slice::from_raw_parts` is undefined behavior even with length 0 (a
    /// zero-column or zero-row LP may yield such a pointer from CLP).
    pub(super) fn copy_solution(&mut self) {
        if self.num_cols > 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved to optimality. `cobre_clp_get_col_solution`
            // returns a non-null pointer into CLP-owned memory of exactly
            // `num_cols` `f64`s (guarded `num_cols > 0`), valid until the next
            // solve. `self.col_value` was resized to `num_cols` in `load_model`.
            let primal = unsafe {
                let ptr = clp_ffi::cobre_clp_get_col_solution(self.handle);
                std::slice::from_raw_parts(ptr, self.num_cols)
            };
            self.col_value.copy_from_slice(primal);

            // SAFETY: as above; `cobre_clp_get_reduced_cost` returns a non-null
            // pointer into CLP-owned memory of exactly `num_cols` `f64`s, valid
            // until the next solve. `self.col_dual` was resized to `num_cols`.
            let reduced = unsafe {
                let ptr = clp_ffi::cobre_clp_get_reduced_cost(self.handle);
                std::slice::from_raw_parts(ptr, self.num_cols)
            };
            self.col_dual.copy_from_slice(reduced);
        }

        if self.num_rows > 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved to optimality. `cobre_clp_get_row_price` returns
            // a non-null pointer into CLP-owned memory of exactly `num_rows`
            // `f64`s (guarded `num_rows > 0`), valid until the next solve.
            let row_price = unsafe {
                let ptr = clp_ffi::cobre_clp_get_row_price(self.handle);
                std::slice::from_raw_parts(ptr, self.num_rows)
            };
            for (dst, &raw) in self.row_dual.iter_mut().zip(row_price) {
                *dst = normalize_row_dual(raw);
            }
        }
    }

    /// Reinstalls an offered warm-start basis into the CLP model element-by-element.
    ///
    /// CLP exposes basis status **per element**, not as a bulk array. An oversized
    /// row basis is tolerated (reinstalled up to `min(len, num_rows)`); an
    /// undersized one cannot be padded soundly and is rejected.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::BasisRowCountMismatch { lp_rows, basis_rows })`
    /// when `b.row_status.len() < self.num_rows`. On that path
    /// `basis_consistency_failures` is incremented and the basis is not offered
    /// to the solver.
    ///
    /// # Panics
    ///
    /// Panics if `b.col_status.len() != self.num_cols`; the LP column count is
    /// fixed at `load_model`, so a column mismatch is a genuine shape bug.
    pub(super) fn install_basis(&mut self, b: &Basis) -> Result<(), SolverError> {
        // CLP's per-element setters silently accept an inconsistent offered basis
        // and `Clp_dual` repairs it, so — unlike `HighsSolver::solve` — there is no
        // consistency check and no `SolverError::BasisInconsistent` surface here.
        assert!(
            b.col_status.len() == self.num_cols,
            "basis column count {} does not match LP column count {}",
            b.col_status.len(),
            self.num_cols
        );
        // Reject before `basis_offered` is incremented — a rejected basis was never offered.
        if b.row_status.len() < self.num_rows {
            self.stats.basis_consistency_failures += 1;
            return Err(SolverError::BasisRowCountMismatch {
                lp_rows: self.num_rows,
                basis_rows: b.row_status.len(),
            });
        }

        self.stats.basis_offered += 1;

        let row_copy_len = b.row_status.len().min(self.num_rows);

        let basis_set_start = Instant::now();
        // Rationale: indices bounded by `num_cols`/`num_rows`, asserted to fit in
        // i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `c` is in `0..num_cols`, a valid column sequence index, and
            // fits in i32. The setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_column_status(
                    self.handle,
                    c as i32,
                    b.col_status[c].to_clp_code(),
                );
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..row_copy_len {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..min(b.row_status.len(), num_rows)`, a valid row
            // sequence index, and fits in i32. The setter writes a single status
            // byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_row_status(
                    self.handle,
                    r as i32,
                    b.row_status[r].to_clp_code(),
                );
            }
        }
        self.stats.total_basis_set_time_seconds += basis_set_start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Resets the CLP model to a clean all-slack (cold) starting basis.
    ///
    /// After a failed `Clp_dual` the model retains CLP's failed internal basis;
    /// re-solving on it can inherit the bad state, so each escalation rung first
    /// drives every structural column nonbasic ([`clp_ffi::CLP_BASIS_AT_LOWER`])
    /// and every row slack basic ([`clp_ffi::CLP_BASIS_BASIC`]). Fully
    /// deterministic — the same status codes are written every time — so it
    /// cannot perturb bit-for-bit reproducibility.
    pub(super) fn reset_cold_basis(&mut self) {
        // Rationale: indices bounded by `num_cols`/`num_rows`, asserted to fit in
        // i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded (asserted via `has_model` in `solve`); `c` is in
            // `0..num_cols`, a valid column sequence index, and fits in i32. The
            // setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_column_status(
                    self.handle,
                    c as i32,
                    clp_ffi::CLP_BASIS_AT_LOWER,
                );
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..self.num_rows {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..num_rows`, a valid row sequence index, and
            // fits in i32. The setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_row_status(self.handle, r as i32, clp_ffi::CLP_BASIS_BASIC);
            }
        }
    }
    /// Resolves the per-attempt simplex iteration cap for `apply_profile`.
    ///
    /// At the [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`) applies the size-scaled
    /// heuristic `max(100_000, num_cols * 50)`; otherwise the profile value
    /// verbatim. Both branches clamp to `i32::MAX` for the FFI cast.
    pub(super) fn resolve_simplex_cap(&self) -> i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        if self.current_profile.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL {
            // Scale with LP size to avoid runaway cycling.
            let heuristic = self.num_cols.saturating_mul(50).max(100_000);
            (heuristic.min(i32::MAX as usize)) as i32
        } else {
            (self
                .current_profile
                .simplex_iteration_limit
                .min(i32::MAX as u32)) as i32
        }
    }

    /// Selects the dual-steepest-edge pricing rule on the underlying CLP model.
    ///
    /// `mode` 1 selects full DSE; 3 is the `ClpDualRowSteepest` default. Driven by
    /// [`Self::apply_profile`] only for a non-default mode (the `== 3` sentinel
    /// skips the call to keep the default profile byte-identical). The setter is
    /// idempotent and issues no solve.
    pub(super) fn set_dual_row_steepest(&mut self, mode: i32) {
        // SAFETY: `self.handle` is a valid, non-null CLP pointer from
        // `cobre_clp_create()`. The shim constructs a stack `ClpDualRowSteepest`
        // and installs it; it retains no pointer after the call returns and
        // cannot fail on a valid handle.
        unsafe {
            clp_ffi::cobre_clp_set_dual_row_steepest(self.handle, mode);
        }
    }
}

/// Normalizes a raw CLP row price into cobre's canonical dual-sign convention.
///
/// Identity is correct: `cobre_clp_get_row_price` already matches the canonical
/// convention. See `tests/_clp_sign_convention_probe.rs`.
const fn normalize_row_dual(raw: f64) -> f64 {
    raw
}

/// Converts a `usize` to `i32`, debug-asserting on overflow.
///
/// Each value is bounded by the merged nnz / row count, which `add_rows` asserts
/// fits in `i32`; this guards the per-entry writes in the CSR→CSC merge.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn i32_from_usize(v: usize) -> i32 {
    debug_assert!(
        i32::try_from(v).is_ok(),
        "value {v} overflows i32: LP exceeds CLP API limit"
    );
    v as i32
}

impl Drop for ClpSolver {
    fn drop(&mut self) {
        // SAFETY: valid CLP pointer from construction, called once per instance.
        unsafe { clp_ffi::cobre_clp_destroy(self.handle) };
    }
}

/// Returns the CLP version as a `"major.minor.patch"` string.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "clp")]
/// # {
/// let v = cobre_solver::clp_version();
/// assert!(v.contains('.'), "version string should be 'major.minor.patch'");
/// # }
/// ```
#[must_use]
pub fn clp_version() -> String {
    // SAFETY: These are pure query functions with no arguments. The CLP C API
    // documents them as safe to call without any prior initialisation; they
    // read only compile-time constants embedded in the library.
    let major = unsafe { clp_ffi::cobre_clp_version_major() };
    let minor = unsafe { clp_ffi::cobre_clp_version_minor() };
    let patch = unsafe { clp_ffi::cobre_clp_version_release() };
    format!("{major}.{minor}.{patch}")
}
