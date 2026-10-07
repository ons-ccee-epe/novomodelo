//! The [`HighsSolver`] handle wrapper, its lifecycle/solve primitives, and the
//! `highs_version` free function. The warm-start `solve_inner` orchestration is
//! determinism-sensitive.

use crate::ffi::{cobre_highs_version_major, cobre_highs_version_minor, cobre_highs_version_patch};
#[cfg(feature = "test-support")]
use std::ffi::CStr;
use std::os::raw::c_void;
use std::time::Instant;

use super::config::{HighsProfile, default_options};
use crate::{
    DEFAULT_PROFILE_HEURISTIC_SENTINEL, DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL, ffi,
    types::{SolutionView, SolverError, SolverStatistics},
};

/// `HiGHS` LP solver instance implementing [`SolverInterface`](crate::SolverInterface).
///
/// Construct with [`HighsSolver::new`]; the handle is destroyed on `Drop`. The
/// `Vec` buffers are reused across solves and grown but never shrunk, so the
/// solve hot path never reallocates.
///
/// # Example
///
/// ```rust
/// use cobre_solver::{HighsSolver, SolverInterface};
///
/// let solver = HighsSolver::new().expect("HiGHS initialisation failed");
/// assert_eq!(solver.name(), "HiGHS");
/// ```
pub struct HighsSolver {
    /// Opaque pointer to the `HiGHS` C++ instance, from `cobre_highs_create()`.
    pub(super) handle: *mut c_void,
    /// Primal column values extracted after each solve.
    pub(super) col_value: Vec<f64>,
    /// Column dual values (reduced costs from `HiGHS` perspective).
    pub(super) col_dual: Vec<f64>,
    /// Row primal values (constraint activity).
    pub(super) row_value: Vec<f64>,
    /// Row dual multipliers (shadow prices).
    pub(super) row_dual: Vec<f64>,
    /// `usize` → `i32` index conversion for the `HiGHS` C API.
    pub(super) scratch_i32: Vec<i32>,
    /// Column basis status codes.
    pub(super) basis_col_i32: Vec<i32>,
    /// Row basis status codes.
    pub(super) basis_row_i32: Vec<i32>,
    /// Dual-ray extraction scratch for `interpret_terminal_status`.
    pub(super) terminal_status_dual_scratch: Vec<f64>,
    /// Primal-ray extraction scratch for `interpret_terminal_status`.
    pub(super) terminal_status_primal_scratch: Vec<f64>,
    /// Current LP column count.
    pub(super) num_cols: usize,
    /// Current LP row count.
    pub(super) num_rows: usize,
    /// Guards the `solve`/`get_basis` "model must be loaded" contract.
    pub(super) has_model: bool,
    /// Accumulated statistics; counters grow monotonically and are not reset by `reset()`.
    pub(super) stats: SolverStatistics,
    /// Cached solver profile. Initialised to `HighsProfile::default()` (which
    /// preserves the historical hardcoded behaviour bit-for-bit) and read by
    /// `set_iteration_limits` on every solve attempt.
    pub(super) current_profile: HighsProfile,
}

// SAFETY: `HighsSolver` holds a raw pointer to a `HiGHS` C++ object. The `HiGHS`
// handle is not thread-safe for concurrent access, but exclusive ownership is
// maintained at all times -- exactly one `HighsSolver` instance owns each
// handle and no shared references to the handle exist. Transferring the
// `HighsSolver` to another thread (via `Send`) is safe because there is no
// concurrent access; the new thread has exclusive ownership. `Sync` is
// intentionally NOT implemented per `HiGHS` Implementation SS6.3.
unsafe impl Send for HighsSolver {}

impl HighsSolver {
    /// Creates a new `HiGHS` solver instance with the `default_options()`
    /// performance-tuned defaults (`HiGHS` Implementation SS4.1).
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if:
    /// - `cobre_highs_create()` returns a null pointer.
    /// - Any configuration call returns `HIGHS_STATUS_ERROR`.
    ///
    /// In both failure cases the `HiGHS` handle is destroyed before returning to
    /// prevent a resource leak.
    pub fn new() -> Result<Self, SolverError> {
        // SAFETY: `cobre_highs_create` is a C function with no preconditions.
        // It allocates and returns a new `HiGHS` instance, or null on allocation
        // failure. The returned pointer is opaque and must be passed back to
        // `HiGHS` API functions.
        let handle = unsafe { ffi::cobre_highs_create() };

        if handle.is_null() {
            return Err(SolverError::InternalError {
                message: "HiGHS instance creation failed: Highs_create() returned null".to_string(),
                error_code: None,
            });
        }

        if let Err(e) = Self::apply_default_config(handle) {
            // SAFETY: `handle` is a valid, non-null pointer obtained from
            // `cobre_highs_create()` in this same function. It has not been
            // passed to `cobre_highs_destroy()` yet. After this call, `handle`
            // must not be used again -- this function returns immediately with Err.
            unsafe { ffi::cobre_highs_destroy(handle) };
            return Err(e);
        }

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_value: Vec::new(),
            row_dual: Vec::new(),
            scratch_i32: Vec::new(),
            basis_col_i32: Vec::new(),
            basis_row_i32: Vec::new(),
            terminal_status_dual_scratch: Vec::new(),
            terminal_status_primal_scratch: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics {
                retry_level_histogram: vec![0u64; 12],
                ..SolverStatistics::default()
            },
            current_profile: HighsProfile::default(),
        })
    }

    /// Applies the `default_options()` table to a fresh handle.
    ///
    /// Returns `Err(SolverError::InternalError)` naming the failing option if any
    /// configuration call returns `HIGHS_STATUS_ERROR`.
    fn apply_default_config(handle: *mut c_void) -> Result<(), SolverError> {
        for opt in &default_options() {
            // SAFETY: `handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(handle) };
            if status == ffi::HIGHS_STATUS_ERROR {
                return Err(SolverError::InternalError {
                    message: format!(
                        "HiGHS configuration failed: {}",
                        opt.name.to_str().unwrap_or("?")
                    ),
                    error_code: Some(status),
                });
            }
        }
        Ok(())
    }

    /// Extracts the optimal solution from `HiGHS` into pre-allocated buffers and returns
    /// a [`SolutionView`] borrowing directly from those buffers.
    ///
    /// The returned view borrows solver-internal buffers and is valid until the next
    /// `&mut self` call. `col_dual` is the reduced cost vector. Row duals follow the
    /// canonical sign convention (per Solver Abstraction SS8).
    pub(super) fn extract_solution_view(&mut self, solve_time_seconds: f64) -> SolutionView<'_> {
        // SAFETY: buffers resized in `load_model`/`add_rows`; HiGHS writes within bounds.
        let status = unsafe {
            ffi::cobre_highs_get_solution(
                self.handle,
                self.col_value.as_mut_ptr(),
                self.col_dual.as_mut_ptr(),
                self.row_value.as_mut_ptr(),
                self.row_dual.as_mut_ptr(),
            )
        };
        // HiGHS guarantees non-ERROR status after an `OPTIMAL` model status.
        debug_assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_solution failed after optimal solve; HiGHS invariant violation"
        );

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let objective = unsafe { ffi::cobre_highs_get_objective_value(self.handle) };

        // SAFETY: iteration count is non-negative so cast is safe.
        #[allow(clippy::cast_sign_loss)]
        let iterations =
            unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;

        SolutionView {
            objective,
            primal: &self.col_value[..self.num_cols],
            dual: &self.row_dual[..self.num_rows],
            reduced_costs: &self.col_dual[..self.num_cols],
            iterations,
            solve_time_seconds,
        }
    }

    /// Re-applies every `current_profile` field `HiGHS` exposes as an option;
    /// restoring any of them anywhere else after `restore_default_settings`
    /// runs is a determinism bug. The profile's iteration limits are the one
    /// sanctioned exception — `restore_iteration_limits` re-installs them
    /// immediately after this call.
    pub(super) fn reapply_profile(&mut self) {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer obtained from
        // `cobre_highs_create()`. Option names are static C string literals with no
        // retained pointer after the call returns; `simplex_update_limit` is
        // clamped to `i32::MAX` before the u32 -> i32 cast so the cast cannot wrap.
        unsafe {
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"primal_feasibility_tolerance".as_ptr(),
                self.current_profile.primal_feasibility_tolerance,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_feasibility_tolerance".as_ptr(),
                self.current_profile.dual_feasibility_tolerance,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_dual_edge_weight_strategy".as_ptr(),
                self.current_profile.simplex_dual_edge_weight_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_scale_strategy".as_ptr(),
                self.current_profile.simplex_scale_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_price_strategy".as_ptr(),
                self.current_profile.simplex_price_strategy,
            );
            ffi::cobre_highs_set_string_option(
                self.handle,
                c"presolve".as_ptr(),
                self.current_profile.presolve.as_option().as_ptr(),
            );
            ffi::cobre_highs_set_bool_option(
                self.handle,
                c"use_warm_start".as_ptr(),
                i32::from(self.current_profile.use_warm_start),
            );
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let simplex_update_limit = self
                .current_profile
                .simplex_update_limit
                .min(i32::MAX as u32) as i32;
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_update_limit".as_ptr(),
                simplex_update_limit,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_simplex_cost_perturbation_multiplier".as_ptr(),
                self.current_profile.cost_perturbation,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"rebuild_refactor_solution_error_tolerance".as_ptr(),
                self.current_profile.refactor_error_tolerance,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"factor_pivot_threshold".as_ptr(),
                self.current_profile.factor_pivot_threshold,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_steepest_edge_weight_log_error_threshold".as_ptr(),
                self.current_profile.steepest_edge_devex_fallback_threshold,
            );
        }
    }

    /// Restores default options after retry escalation.
    ///
    /// Status codes are checked via `debug_assert!` to catch programming
    /// errors during development (e.g., invalid option name). In release
    /// builds, failures are silently ignored since we are already on the
    /// recovery path.
    pub(super) fn restore_default_settings(&mut self) {
        for opt in &default_options() {
            // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(self.handle) };
            debug_assert_eq!(
                status,
                ffi::HIGHS_STATUS_OK,
                "restore_default_settings: option {:?} failed with status {status}",
                opt.name,
            );
        }
    }

    /// Runs the solver once and returns the raw `HiGHS` model status.
    pub(super) fn run_once(&mut self) -> i32 {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let run_status = unsafe { ffi::cobre_highs_run(self.handle) };
        if run_status == ffi::HIGHS_STATUS_ERROR {
            return ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR;
        }
        // SAFETY: same.
        unsafe { ffi::cobre_highs_get_model_status(self.handle) }
    }

    /// Sets per-solve iteration limits before a `run_once()` call.
    ///
    /// Simplex cap: if `current_profile.simplex_iteration_limit` equals
    /// [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`), the historical heuristic
    /// `max(100_000, 50 × num_cols)` is used. Any non-zero profile value is
    /// applied verbatim (clamped to `i32::MAX` for the FFI call).
    ///
    /// IPM cap: if `current_profile.ipm_iteration_limit` equals
    /// [`DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL`] (`0`), `i32::MAX` is sent to
    /// `HiGHS` (no cap). Any positive value is applied verbatim (clamped to
    /// `i32::MAX` for the FFI call). The `Default` value is `10_000`, so
    /// existing callers see no behavioural change.
    ///
    /// **Note on `time_limit`**: `HiGHS` tracks elapsed time cumulatively from
    /// instance creation, not per-`run()` call — neither `clear_solver()` nor
    /// option changes reset the internal timer. This makes `time_limit`
    /// unusable across a long sequence of solves on one instance (thousands of
    /// solves is typical). Wall-clock measurement via `Instant` is used instead
    /// for time-based budget management.
    pub(super) fn set_iteration_limits(&mut self) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let simplex_iter_limit: i32 =
            if self.current_profile.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL {
                let heuristic = self.num_cols.saturating_mul(50).max(100_000);
                // Clamp to i32::MAX so the FFI cast cannot wrap.
                (heuristic.min(i32::MAX as usize)) as i32
            } else {
                (self
                    .current_profile
                    .simplex_iteration_limit
                    .min(i32::MAX as u32)) as i32
            };

        // Map the 0 sentinel to i32::MAX, else HiGHS reads 0 as "no iterations
        // allowed" rather than "unbounded".
        #[allow(clippy::cast_possible_wrap)]
        let ipm_iter_limit: i32 =
            if self.current_profile.ipm_iteration_limit == DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL {
                i32::MAX // "unbounded" per trait contract
            } else {
                (self
                    .current_profile
                    .ipm_iteration_limit
                    .min(i32::MAX as u32)) as i32
            };

        // SAFETY: handle is valid non-null HiGHS pointer; option names are
        // static C strings with no retained pointers.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                simplex_iter_limit,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"ipm_iteration_limit".as_ptr(),
                ipm_iter_limit,
            );
        }
    }

    /// Restores iteration limits to their unconstrained defaults.
    ///
    /// Called after `retry_escalation` completes (regardless of outcome).
    pub(super) fn restore_iteration_limits(&mut self) {
        // SAFETY: handle is valid non-null HiGHS pointer.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                i32::MAX,
            );
            ffi::cobre_highs_set_int_option(self.handle, c"ipm_iteration_limit".as_ptr(), i32::MAX);
        }
    }

    /// Interprets a non-optimal status as a terminal `SolverError`.
    ///
    /// Returns `None` for `SOLVE_ERROR` or `UNKNOWN` (retry continues),
    /// or `Some(error)` for terminal statuses.
    pub(super) fn interpret_terminal_status(
        &mut self,
        status: i32,
        solve_time_seconds: f64,
    ) -> Option<SolverError> {
        #[allow(clippy::match_same_arms)]
        match status {
            ffi::HIGHS_MODEL_STATUS_OPTIMAL => None,
            ffi::HIGHS_MODEL_STATUS_INFEASIBLE => Some(SolverError::Infeasible),
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                // A dual ray classifies as Infeasible, a primal ray as Unbounded.
                let mut has_dual_ray: i32 = 0;
                self.terminal_status_dual_scratch.resize(self.num_rows, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_dual_scratch` has been resized to at least
                // `self.num_rows` elements; HiGHS writes exactly `num_rows` values.
                let dual_status = unsafe {
                    ffi::cobre_highs_get_dual_ray(
                        self.handle,
                        &raw mut has_dual_ray,
                        self.terminal_status_dual_scratch.as_mut_ptr(),
                    )
                };
                if dual_status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
                    return Some(SolverError::Infeasible);
                }
                let mut has_primal_ray: i32 = 0;
                self.terminal_status_primal_scratch
                    .resize(self.num_cols, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_primal_scratch` has been resized to at least
                // `self.num_cols` elements; HiGHS writes exactly `num_cols` values.
                let primal_status = unsafe {
                    ffi::cobre_highs_get_primal_ray(
                        self.handle,
                        &raw mut has_primal_ray,
                        self.terminal_status_primal_scratch.as_mut_ptr(),
                    )
                };
                if primal_status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
                    return Some(SolverError::Unbounded);
                }
                Some(SolverError::Infeasible)
            }
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED => Some(SolverError::Unbounded),
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT => Some(SolverError::TimeLimitExceeded {
                elapsed_seconds: solve_time_seconds,
            }),
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT => {
                // SAFETY: handle is valid non-null pointer; iteration count is non-negative.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                Some(SolverError::IterationLimit { iterations })
            }
            // None = retryable, not terminal — do not fold into the `other` arm.
            ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR | ffi::HIGHS_MODEL_STATUS_UNKNOWN => None,
            other => Some(SolverError::InternalError {
                message: format!("HiGHS returned unexpected model status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Converts `usize` indices to `i32` in the internal scratch buffer.
    ///
    /// Grows but never shrinks the buffer. Each element is debug-asserted to fit in i32.
    pub(super) fn convert_to_i32_scratch(&mut self, source: &[usize]) -> &[i32] {
        if source.len() > self.scratch_i32.len() {
            self.scratch_i32.resize(source.len(), 0);
        }
        for (i, &v) in source.iter().enumerate() {
            debug_assert!(
                i32::try_from(v).is_ok(),
                "usize index {v} overflows i32::MAX at position {i}"
            );
            // SAFETY: debug_assert verifies v fits in i32; cast to HiGHS C API i32.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                self.scratch_i32[i] = v as i32;
            }
        }
        &self.scratch_i32[..source.len()]
    }

    /// Internal helper: run the simplex and update stats.
    ///
    /// Core simplex execution, called after (for warm-start) the basis has been
    /// installed. `HiGHS` retains its internal simplex basis across consecutive
    /// `solve_inner` calls on the same LP shape — the primary warm-start
    /// mechanism when a handle solves a long sequence of same-shape LPs. No
    /// `Highs_clearSolver` call is issued — clearing the solver discards the
    /// retained basis and forfeits the warm start.
    pub(super) fn solve_inner(&mut self) -> Result<SolutionView<'_>, SolverError> {
        // Iteration limits only, no time_limit (see `set_iteration_limits`):
        // wall-clock time is measured after `run_once` to detect stuck solves.
        self.set_iteration_limits();

        let t0 = Instant::now();
        let model_status = self.run_once();
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
            // SAFETY: handle is valid non-null HiGHS pointer.
            #[allow(clippy::cast_sign_loss)]
            let iterations =
                unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
            self.stats.success_count += 1;
            self.stats.first_try_successes += 1;
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;
            self.restore_iteration_limits();
            return Ok(self.extract_solution_view(solve_time));
        }

        // UNBOUNDED / ITERATION_LIMIT / TIME_LIMIT and a >15s wall-clock are all
        // retried, not treated as terminal: a warm-started dual simplex can
        // report any of them spuriously on numerically hard LPs, and HiGHS tracks
        // time cumulatively so TIME_LIMIT can fire even with time_limit=Infinity.
        let is_unbounded = model_status == ffi::HIGHS_MODEL_STATUS_UNBOUNDED;
        // INFEASIBLE is retried for the same reason: escalation level 0 clears the
        // warm basis and re-solves cold, rescuing a warm-start-only false
        // infeasible while a genuine one is confirmed and stays terminal inside
        // `retry_escalation`. Mirrors the cold-solve escalation on the CLP path.
        let is_infeasible = model_status == ffi::HIGHS_MODEL_STATUS_INFEASIBLE;
        let initial_retryable = is_unbounded
            || is_infeasible
            || model_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
            || model_status == ffi::HIGHS_MODEL_STATUS_TIME_LIMIT
            || solve_time > 15.0;
        if !initial_retryable
            && let Some(terminal_err) = self.interpret_terminal_status(model_status, solve_time)
        {
            self.restore_iteration_limits();
            self.stats.failure_count += 1;
            return Err(terminal_err);
        }

        match self.retry_escalation(is_unbounded) {
            Ok(outcome) => {
                self.stats.retry_count += outcome.attempts;
                self.stats.success_count += 1;
                self.stats.total_iterations += outcome.iterations;
                self.stats.total_solve_time_seconds += outcome.solve_time;
                self.stats.retry_level_histogram[outcome.level as usize] += 1;
                Ok(self.extract_solution_view(outcome.solve_time))
            }
            Err((attempts, err)) => {
                self.stats.retry_count += attempts;
                self.stats.failure_count += 1;
                Err(err)
            }
        }
    }
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: valid HiGHS pointer from construction, called once per instance.
        unsafe { ffi::cobre_highs_destroy(self.handle) };
    }
}

/// Returns the `HiGHS` version as a `"major.minor.patch"` string.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "highs")]
/// # {
/// let v = cobre_solver::highs_version();
/// assert!(v.contains('.'), "version string should be 'major.minor.patch'");
/// # }
/// ```
#[must_use]
pub fn highs_version() -> String {
    // SAFETY: These are pure query functions with no arguments. The HiGHS C API
    // documents them as safe to call without any prior initialisation; they read
    // only compile-time constants embedded in the library.
    let major = unsafe { cobre_highs_version_major() };
    let minor = unsafe { cobre_highs_version_minor() };
    let patch = unsafe { cobre_highs_version_patch() };
    format!("{major}.{minor}.{patch}")
}

/// Test-support accessors for integration tests that need to set raw `HiGHS` options.
///
/// Gated behind the `test-support` feature. The raw handle is intentionally not
/// part of the public API — callers use these methods to configure time/iteration
/// limits before a solve without going through the safe wrapper.
#[cfg(feature = "test-support")]
impl HighsSolver {
    /// Returns the raw `HiGHS` handle for use with test-support FFI helpers.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of `self`. The caller must
    /// not store the pointer beyond that lifetime, must not call
    /// `cobre_highs_destroy` on it, and must not alias it across threads.
    #[must_use]
    pub fn raw_handle(&self) -> *mut c_void {
        self.handle
    }

    /// Invoke `apply_retry_level_options` for a given level.
    ///
    /// Levels 0-4 only; for 5-11 use `apply_extended_retry_options_for_test`.
    pub fn apply_retry_level_options_for_test(&mut self, level: u32) {
        self.apply_retry_level_options(level);
    }

    /// Invoke `apply_extended_retry_options` for a given level (5-11).
    pub fn apply_extended_retry_options_for_test(&mut self, level: u32) {
        self.apply_extended_retry_options(level);
    }

    /// Invoke `restore_default_settings` then `reapply_profile`,
    /// mirroring the `retry_escalation` finalization path so tests can verify
    /// profile tolerances survive a defaults-restore.
    pub fn restore_defaults_then_reapply_profile_for_test(&mut self) {
        self.restore_default_settings();
        self.reapply_profile();
    }

    /// Read a double-valued `HiGHS` option by name.
    ///
    /// Returns `None` if the option name is unknown to `HiGHS`; `Some(value)`
    /// on success.
    #[must_use]
    pub fn get_double_option(&self, option: &CStr) -> Option<f64> {
        let mut out = 0.0_f64;
        // SAFETY: handle is valid non-null HiGHS pointer; option is a valid
        // null-terminated C string borrowed for the duration of the call;
        // `out` is stack-allocated and written by HiGHS on success.
        let status = unsafe {
            ffi::cobre_highs_get_double_option(self.handle, option.as_ptr(), &raw mut out)
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            None
        } else {
            Some(out)
        }
    }

    /// Read an integer-valued `HiGHS` option by name.
    ///
    /// Returns `None` if the option name is unknown to `HiGHS`; `Some(value)`
    /// on success.
    #[must_use]
    pub fn get_int_option(&self, option: &CStr) -> Option<i32> {
        let mut out = 0_i32;
        // SAFETY: handle is valid non-null HiGHS pointer; option is a valid
        // null-terminated C string borrowed for the duration of the call;
        // `out` is stack-allocated and written by HiGHS on success.
        let status =
            unsafe { ffi::cobre_highs_get_int_option(self.handle, option.as_ptr(), &raw mut out) };
        if status == ffi::HIGHS_STATUS_ERROR {
            None
        } else {
            Some(out)
        }
    }
}
