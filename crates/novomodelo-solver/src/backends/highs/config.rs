//! `HiGHS` tuning profile and the default configuration table.

use std::ffi::CStr;
use std::os::raw::c_void;

use crate::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, ffi};

/// HiGHS-specific solver profile carrying the full per-phase tuning surface.
///
/// Field defaults match the `default_options()` table bit-for-bit, so callers
/// that never set a non-default profile observe no behavioral change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighsProfile {
    /// Primal feasibility tolerance.
    pub primal_feasibility_tolerance: f64,
    /// Dual feasibility tolerance.
    pub dual_feasibility_tolerance: f64,
    /// Per-attempt simplex iteration cap. `DEFAULT_PROFILE_HEURISTIC_SENTINEL`
    /// (0) selects the historical heuristic `num_cols * 50 max 100_000`.
    pub simplex_iteration_limit: u32,
    /// Per-attempt IPM iteration cap. `DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL`
    /// (0) means unbounded (`i32::MAX`).
    pub ipm_iteration_limit: u32,
    /// `HiGHS` `simplex_dual_edge_weight_strategy` (`-1`=Choose, `0`=Dantzig,
    /// `1`=Devex, `2`=`SteepestEdge`).
    pub simplex_dual_edge_weight_strategy: i32,
    /// `HiGHS` `simplex_scale_strategy` (`0`=Off, `1`=Choose, `2`=Curtis–Reid,
    /// `4`=Equilibration); see `default_options()` for why the default is `0`.
    pub simplex_scale_strategy: i32,
    /// `HiGHS` `simplex_price_strategy` (`0`=Col, `1`=Row, `2`=`RowHyperSparse`,
    /// `3`=`RowSparse`).
    pub simplex_price_strategy: i32,
    /// `HiGHS` `presolve` (`off`/`choose`/`on`).
    pub presolve: PresolveKind,
    /// `HiGHS` `simplex_update_limit`: limit on the number of simplex UPDATE
    /// operations before a refactorization.
    pub simplex_update_limit: u32,
    /// `HiGHS` `dual_simplex_cost_perturbation_multiplier`. `0.0` disables cost
    /// perturbation to avoid wasted cleanup pivots when the basis is already
    /// near-optimal from a warm start (Bland's rule guards against cycling;
    /// retry level 0 restores perturbation as a cold-solve fallback).
    pub cost_perturbation: f64,
    /// `HiGHS` `rebuild_refactor_solution_error_tolerance`, loosened from the
    /// `HiGHS` default (`1e-8`) to skip conservative refactorizations that
    /// aren't earning their keep in this numerically benign regime.
    pub refactor_error_tolerance: f64,
    /// `HiGHS` `factor_pivot_threshold`: matrix factorization pivot threshold.
    pub factor_pivot_threshold: f64,
    /// `HiGHS` `use_warm_start`. Diagnostic-only: setting `false` forces every
    /// solve cold and is never the intended production configuration.
    pub use_warm_start: bool,
    /// `HiGHS` `dual_steepest_edge_weight_log_error_threshold`: the dual
    /// steepest-edge weight log-error threshold above which `HiGHS` falls back
    /// from `SteepestEdge` to Devex pricing mid-solve.
    pub steepest_edge_devex_fallback_threshold: f64,
}

impl Default for HighsProfile {
    fn default() -> Self {
        Self {
            primal_feasibility_tolerance: 1e-9,
            dual_feasibility_tolerance: 1e-9,
            simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
            ipm_iteration_limit: 10_000,
            simplex_dual_edge_weight_strategy: 1,
            simplex_scale_strategy: 0,
            simplex_price_strategy: 1,
            presolve: PresolveKind::On,
            simplex_update_limit: 5000,
            cost_perturbation: 0.0,
            refactor_error_tolerance: 1e-6,
            factor_pivot_threshold: 0.1,
            use_warm_start: true,
            steepest_edge_devex_fallback_threshold: 10.0,
        }
    }
}

/// `HiGHS` `presolve` mode (`kPresolveString`): `"off"`, `"choose"`, or `"on"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresolveKind {
    /// `HiGHS` presolve disabled.
    Off,
    /// `HiGHS` decides whether to presolve.
    Choose,
    /// `HiGHS` presolve enabled (cobre baseline).
    On,
}

impl PresolveKind {
    /// The `HiGHS` option string for this mode.
    #[must_use]
    pub const fn as_option(self) -> &'static CStr {
        match self {
            PresolveKind::Off => c"off",
            PresolveKind::Choose => c"choose",
            PresolveKind::On => c"on",
        }
    }
}

/// A typed `HiGHS` option value for the configuration table.
pub(super) enum OptionValue {
    /// String option (`cobre_highs_set_string_option`).
    Str(&'static CStr),
    /// Integer option (`cobre_highs_set_int_option`).
    Int(i32),
    /// Boolean option (`cobre_highs_set_bool_option`).
    Bool(i32),
    /// Double option (`cobre_highs_set_double_option`).
    Double(f64),
}

/// A named `HiGHS` option with its default value.
pub(super) struct DefaultOption {
    pub(super) name: &'static CStr,
    pub(super) value: OptionValue,
}

impl DefaultOption {
    /// Applies this option to a `HiGHS` handle. Returns the `HiGHS` status code.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid, non-null pointer from `cobre_highs_create()`.
    pub(super) unsafe fn apply(&self, handle: *mut c_void) -> i32 {
        unsafe {
            match &self.value {
                OptionValue::Str(val) => {
                    ffi::cobre_highs_set_string_option(handle, self.name.as_ptr(), val.as_ptr())
                }
                OptionValue::Int(val) => {
                    ffi::cobre_highs_set_int_option(handle, self.name.as_ptr(), *val)
                }
                OptionValue::Bool(val) => {
                    ffi::cobre_highs_set_bool_option(handle, self.name.as_ptr(), *val)
                }
                OptionValue::Double(val) => {
                    ffi::cobre_highs_set_double_option(handle, self.name.as_ptr(), *val)
                }
            }
        }
    }
}

/// Performance-tuned default options (`HiGHS` Implementation SS4.1).
///
/// Tuned for master LPs dominated by many slack rows that are warm-started
/// across consecutive solves.
///
/// `simplex_scale_strategy` is set to 0 (Off): cobre's offline prescaler
/// (`lp_builder/scaling.rs`, applied in `setup/template_postprocess`) conditions
/// every stage template via the per-column / per-row geometric-mean factors
/// stored in `StageTemplate.col_scale` / `row_scale`, so `HiGHS`'s internal
/// scaler is disabled to avoid double-scaling. Retry escalation levels 5+ keep
/// this strategy.
///
/// Several entries diverge from `HiGHS` defaults to suit warm-started solves
/// on master LPs with tens of thousands of mostly-slack rows: Devex pricing
/// avoids per-pivot edge-weight maintenance scaling with row count; skipping
/// the initial condition check eliminates O(m)–O(m²) work whose guarantees
/// are already provided by the caller's slot-tracked basis reconstruction;
/// row-wise PRICE wins on hyper-sparse basis-inverse rows. The
/// cost-perturbation and refactor-tolerance pins carry their rationale on
/// [`HighsProfile::cost_perturbation`] and
/// [`HighsProfile::refactor_error_tolerance`] — this table restores only
/// their base (restore-path) values.
pub(super) fn default_options() -> [DefaultOption; 17] {
    [
        DefaultOption {
            name: c"solver",
            value: OptionValue::Str(c"simplex"),
        },
        DefaultOption {
            name: c"simplex_strategy",
            value: OptionValue::Int(1), // Dual simplex
        },
        DefaultOption {
            name: c"simplex_scale_strategy",
            value: OptionValue::Int(0), // Off
        },
        DefaultOption {
            name: c"presolve",
            value: OptionValue::Str(c"on"),
        },
        DefaultOption {
            name: c"parallel",
            value: OptionValue::Str(c"off"),
        },
        DefaultOption {
            name: c"output_flag",
            value: OptionValue::Bool(0),
        },
        DefaultOption {
            name: c"primal_feasibility_tolerance",
            value: OptionValue::Double(1e-9),
        },
        DefaultOption {
            name: c"dual_feasibility_tolerance",
            value: OptionValue::Double(1e-9),
        },
        DefaultOption {
            name: c"simplex_dual_edge_weight_strategy",
            value: OptionValue::Int(1), // Devex
        },
        DefaultOption {
            name: c"dual_simplex_cost_perturbation_multiplier",
            value: OptionValue::Double(0.0), // Off (warm-start regime)
        },
        DefaultOption {
            name: c"simplex_initial_condition_check",
            value: OptionValue::Bool(0), // Off (caller manages basis quality)
        },
        DefaultOption {
            name: c"simplex_price_strategy",
            value: OptionValue::Int(1), // Row (hyper-sparse master LPs)
        },
        DefaultOption {
            name: c"rebuild_refactor_solution_error_tolerance",
            value: OptionValue::Double(1e-6), // Loosened from HiGHS default 1e-8
        },
        DefaultOption {
            name: c"simplex_update_limit",
            value: OptionValue::Int(5000),
        },
        DefaultOption {
            name: c"factor_pivot_threshold",
            value: OptionValue::Double(0.1),
        },
        DefaultOption {
            name: c"use_warm_start",
            value: OptionValue::Bool(1),
        },
        DefaultOption {
            name: c"dual_steepest_edge_weight_log_error_threshold",
            value: OptionValue::Double(10.0),
        },
    ]
}
