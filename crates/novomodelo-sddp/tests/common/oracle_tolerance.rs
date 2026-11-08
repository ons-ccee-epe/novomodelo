//! Solver feasibility tolerances converted into the unscaled units of the row or column they
//! bound, and the check that every solve an oracle ran succeeded on its first attempt.

use novomodelo_solver::{ActiveProfile, SolverStatistics};

/// Primal and dual feasibility tolerances of a scaled LP.
#[derive(Clone, Copy, Debug)]
pub struct UnscaledTolerance {
    primal: f64,
    dual: f64,
}

fn assert_positive(name: &str, value: f64) {
    assert!(
        value.is_finite() && value > 0.0,
        "{name} must be finite and > 0.0, got {value}"
    );
}

impl UnscaledTolerance {
    /// Tolerances of explicit scaled-LP values.
    #[must_use]
    pub fn new(primal: f64, dual: f64) -> Self {
        assert_positive("primal tolerance", primal);
        assert_positive("dual tolerance", dual);
        Self { primal, dual }
    }

    /// The feasibility tolerances of the active backend's default profile.
    #[must_use]
    pub fn from_active_profile() -> Self {
        let profile = ActiveProfile::default();
        Self::new(
            profile.primal_feasibility_tolerance,
            profile.dual_feasibility_tolerance,
        )
    }

    /// The scaled primal feasibility tolerance.
    #[must_use]
    pub fn primal(self) -> f64 {
        self.primal
    }

    /// The scaled dual feasibility tolerance.
    #[must_use]
    pub fn dual(self) -> f64 {
        self.dual
    }

    /// Bounds an unscaled row residual of the row with scale `row_scale`.
    #[must_use]
    pub fn row_residual(self, row_scale: f64) -> f64 {
        assert_positive("row scale", row_scale);
        self.primal / row_scale
    }

    /// Bounds an unscaled column violation of its bound, for the column with scale `col_scale`.
    #[must_use]
    pub fn col_bound(self, col_scale: f64) -> f64 {
        assert_positive("column scale", col_scale);
        self.primal * col_scale
    }

    /// Bounds an unscaled reduced cost of the column with scale `col_scale`.
    #[must_use]
    pub fn reduced_cost(self, col_scale: f64) -> f64 {
        assert_positive("column scale", col_scale);
        self.dual / col_scale
    }

    /// Bounds an unscaled row dual of the row with scale `row_scale`.
    #[must_use]
    pub fn row_dual(self, row_scale: f64) -> f64 {
        assert_positive("row scale", row_scale);
        self.dual * row_scale
    }
}

/// Fails unless at least one solve ran between the snapshots and every one was a first-attempt optimum.
pub fn assert_first_try(label: &str, before: &SolverStatistics, after: &SolverStatistics) {
    let solves = after.solve_count.saturating_sub(before.solve_count);
    assert!(
        after.solve_count > before.solve_count,
        "{label}: no solve ran between the snapshots"
    );
    assert!(
        after.retry_count == before.retry_count,
        "{label}: a solve was retried: {} retry attempts",
        after.retry_count.saturating_sub(before.retry_count)
    );
    let first_try = after
        .first_try_successes
        .saturating_sub(before.first_try_successes);
    assert!(
        first_try == solves,
        "{label}: {} of {solves} solves did not succeed on the first attempt",
        solves.saturating_sub(first_try)
    );
}
