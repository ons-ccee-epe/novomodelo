//! Constants shared by the LP-solver profiles: iteration-limit sentinels and the
//! row-violation allowance.

/// Sentinel for the simplex iteration limit: when
/// `HighsProfile::simplex_iteration_limit` equals this, implementations MUST fall
/// back to their per-call heuristic rather than applying a flat cap; any non-zero
/// value is applied verbatim. `0` is safe as the sentinel because zero iterations
/// is never a legitimate cap.
pub const DEFAULT_PROFILE_HEURISTIC_SENTINEL: u32 = 0;

/// Sentinel for the IPM iteration limit: when `HighsProfile::ipm_iteration_limit`
/// equals this, implementations MUST apply no cap (pass `i32::MAX`); any positive
/// value is applied verbatim. `0` is safe as the sentinel because zero iterations
/// is never a legitimate cap.
pub const DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL: u32 = 0;

/// Row excess above which a returned solution counts as missing a row bound.
///
/// The row excess of a solution is the largest, over rows, of the bound violation
/// minus the solver's primal feasibility tolerance, divided by `max(1, Σ|a_ij x_j|)`.
/// A solution counts when that excess is strictly greater than this allowance.
pub const ROW_VIOLATION_RELATIVE_ALLOWANCE: f64 = 1e-11;

pub(crate) fn counts_as_row_violation(rho: f64) -> bool {
    rho > ROW_VIOLATION_RELATIVE_ALLOWANCE
}

/// Fieldless profile for in-crate test mocks, so test-only `SolverInterface`
/// impls need not name the feature-gated `HighsProfile`. Test-only — never export
/// it from the public API surface.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct MockProfile;
