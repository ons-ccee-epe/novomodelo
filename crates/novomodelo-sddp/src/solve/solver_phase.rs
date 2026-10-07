//! SDDP algorithm phase enum, the backend-agnostic [`PhaseProfiles`] trait, and
//! the per-phase named solver-profile constants.
//!
//! All three phases solve the same cut-laden LPs, so they share one tuned
//! deep-cut-pool profile. For `HiGHS`, [`FORWARD_PROFILE`], [`BACKWARD_PROFILE`],
//! and [`SIMULATION_PROFILE`] are identical: each overrides only
//! `simplex_price_strategy` (`RowHyperSparse`, value `2`) relative to
//! [`HighsProfile::default()`] to exploit the sparse cut-subgradient rows. The
//! CLP backend pins full dual steepest-edge pricing (`dual_pricing_mode = 1`)
//! and `factorization_frequency = 200`. All values come from an empirical solver
//! sweep on production-scale cases.
//!
//! ## Why the per-phase profiles live here, not in `cobre-solver`
//!
//! The mapping "which phase wants which solver behaviour" is algorithm knowledge
//! that must live in the algorithm crate; `cobre-solver` stays strictly
//! backend-agnostic and cannot know about SDDP phases. The [`PhaseProfiles`]
//! trait abstracts only *which phase is running*, never the tuning content. The
//! trait is local and the implemented profile types are foreign, which the
//! orphan rule permits.

use cobre_io::config::{BackwardScheduler, PhaseSolverProfileConfig};
#[cfg(feature = "highs")]
use cobre_io::config::{DualEdgeWeight, PresolveMode, PriceStrategy, ScaleStrategy};
use cobre_solver::{ActiveProfile, DEFAULT_PROFILE_HEURISTIC_SENTINEL};
#[cfg(feature = "clp")]
use cobre_solver::{ClpAlgorithm, ClpProfile};
#[cfg(feature = "highs")]
use cobre_solver::{HighsProfile, PresolveKind};

use crate::SddpError;

/// The three algorithmic phases of the SDDP algorithm.
///
/// The solver is configured per phase via
/// `ProfiledSolver::set_profile(phase.profile())` at phase entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Forward pass: sampling trajectories by solving LPs from stage 1 to T.
    Forward,
    /// Backward pass: computing Benders cuts by solving LPs from stage T to 1.
    Backward,
    /// Policy simulation: evaluating the trained policy on out-of-sample
    /// scenarios.
    Simulation,
}

/// Resolved per-run backward/forward tuning, threaded from [`crate::setup::StudySetup`]
/// into `TrainingSession::new` once per training run: the solver profiles plus
/// the backward-pass scheduler (`training.parallelism.backward_scheduler`),
/// which has no forward-pass analogue but travels the same seam.
///
/// `Default` reproduces [`Phase::profile`]'s byte-neutral constants and the
/// byte-neutral trial-point scheduler, so a caller with no config override can
/// pass `SolverProfiles::default()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverProfiles {
    /// Resolved forward-pass profile.
    pub forward: ActiveProfile,
    /// Resolved backward-pass profile.
    pub backward: ActiveProfile,
    /// Backward-pass scheduler (`training.parallelism.backward_scheduler`),
    /// carrying the opening-block size when the `by_node` method is
    /// selected.
    pub backward_scheduler: BackwardScheduler,
    /// Opening-block-scheduler claim order
    /// (`BackwardPassState::set_hardest_first_claim_order`): `true` claims
    /// hardest-`(stage, block)`-first by the previous iteration's mean pivots
    /// (longest-processing-time-first); `false` forces the canonical ascending
    /// block order. No `training.*` config field resolves this yet — a
    /// reserved test-support seam for the byte-neutrality gate; production
    /// always resolves `true`.
    pub hardest_first_claim_order: bool,
}

impl Default for SolverProfiles {
    fn default() -> Self {
        Self {
            forward: Phase::Forward.profile(),
            backward: Phase::Backward.profile(),
            backward_scheduler: BackwardScheduler::default(),
            hardest_first_claim_order: true,
        }
    }
}

/// Per-phase identity selection of a backend solver profile (see the module
/// docs for why the tuned values live in `cobre-sddp`, not `cobre-solver`).
///
/// The members are **associated constants** because every backend profile field
/// is const-constructible.
pub trait PhaseProfiles: Sized {
    /// Profile applied when entering the forward pass.
    const FORWARD: Self;
    /// Profile applied when entering the backward pass.
    const BACKWARD: Self;
    /// Profile applied when entering policy simulation.
    const SIMULATION: Self;
}

/// Per-phase `HighsProfile` selection, bit-for-bit equal to the
/// [`FORWARD_PROFILE`]/[`BACKWARD_PROFILE`]/[`SIMULATION_PROFILE`] constants.
///
/// The full field literals are written out because associated constants cannot
/// use the `..HighsProfile::default()` struct-update spread in const context.
#[cfg(feature = "highs")]
impl PhaseProfiles for HighsProfile {
    const FORWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
        presolve: PresolveKind::On,
        simplex_update_limit: 5000,
        cost_perturbation: 0.0,
        refactor_error_tolerance: 1e-6,
        factor_pivot_threshold: 0.1,
        use_warm_start: true,
        steepest_edge_devex_fallback_threshold: 10.0,
    };
    const BACKWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
        presolve: PresolveKind::On,
        simplex_update_limit: 5000,
        cost_perturbation: 0.0,
        refactor_error_tolerance: 1e-6,
        factor_pivot_threshold: 0.1,
        use_warm_start: true,
        steepest_edge_devex_fallback_threshold: 10.0,
    };
    const SIMULATION: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
        presolve: PresolveKind::On,
        simplex_update_limit: 5000,
        cost_perturbation: 0.0,
        refactor_error_tolerance: 1e-6,
        factor_pivot_threshold: 0.1,
        use_warm_start: true,
        steepest_edge_devex_fallback_threshold: 10.0,
    };
}

/// Per-phase `ClpProfile` selection.
///
/// `FORWARD` and `BACKWARD` share the tuned deep-cut-pool profile
/// (`dual_pricing_mode = 1`, `factorization_frequency = 200`). These are
/// CLP-native values for CLP's own option surface — **not** a translation of the
/// `HiGHS` per-phase profiles. `SIMULATION` keeps those values but selects the
/// **primal** simplex (see the const's comment for why dual fails there).
///
/// The full field literals are written out because associated constants cannot
/// use the `..ClpProfile::default()` struct-update spread in const context.
#[cfg(feature = "clp")]
impl PhaseProfiles for ClpProfile {
    const FORWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
    const BACKWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
    // Simulation runs the PRIMAL simplex, not dual: CLP's dual falsely declares
    // these warm-started, fully-frozen cut-laden simulation LPs infeasible, while
    // the primal simplex solves them directly and deterministically (so
    // bit-for-bit reproducibility holds). The dual-row pivot rule
    // (`dual_pricing_mode`) is unused by primal.
    const SIMULATION: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Primal,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
}

/// Solver profile applied during the SDDP forward pass — the tuned
/// deep-cut-pool profile (see the module docs); equal to [`BACKWARD_PROFILE`]
/// and [`SIMULATION_PROFILE`].
#[cfg(feature = "highs")]
pub const FORWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
    presolve: PresolveKind::On,
    simplex_update_limit: 5000,
    cost_perturbation: 0.0,
    refactor_error_tolerance: 1e-6,
    factor_pivot_threshold: 0.1,
    use_warm_start: true,
    steepest_edge_devex_fallback_threshold: 10.0,
};

/// Solver profile applied during the SDDP backward pass — the tuned
/// deep-cut-pool profile (see the module docs); equal to [`FORWARD_PROFILE`]
/// and [`SIMULATION_PROFILE`].
#[cfg(feature = "highs")]
pub const BACKWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
    presolve: PresolveKind::On,
    simplex_update_limit: 5000,
    cost_perturbation: 0.0,
    refactor_error_tolerance: 1e-6,
    factor_pivot_threshold: 0.1,
    use_warm_start: true,
    steepest_edge_devex_fallback_threshold: 10.0,
};

/// Solver profile applied during policy simulation — the tuned deep-cut-pool
/// profile (see the module docs); equal to [`FORWARD_PROFILE`] and
/// [`BACKWARD_PROFILE`].
#[cfg(feature = "highs")]
pub const SIMULATION_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
    presolve: PresolveKind::On,
    simplex_update_limit: 5000,
    cost_perturbation: 0.0,
    refactor_error_tolerance: 1e-6,
    factor_pivot_threshold: 0.1,
    use_warm_start: true,
    steepest_edge_devex_fallback_threshold: 10.0,
};

impl Phase {
    /// Returns the [`ActiveProfile`] to apply when entering this
    /// phase, delegating to the active backend's [`PhaseProfiles`] impl. Pass it
    /// to `ProfiledSolver::set_profile` at phase entry.
    #[must_use]
    pub fn profile(self) -> ActiveProfile {
        match self {
            Phase::Forward => <ActiveProfile as PhaseProfiles>::FORWARD,
            Phase::Backward => <ActiveProfile as PhaseProfiles>::BACKWARD,
            Phase::Simulation => <ActiveProfile as PhaseProfiles>::SIMULATION,
        }
    }

    /// Resolves this phase's [`ActiveProfile`], layered as base constant
    /// ([`Self::profile`]) + per-field overrides (a later field wins).
    /// `config` absent returns [`Self::profile`] unchanged — the byte-neutral
    /// default.
    #[must_use]
    pub fn resolve_profile(self, config: Option<&PhaseSolverProfileConfig>) -> ActiveProfile {
        resolve_active_profile(self.profile(), config)
    }
}

fn phase_config_key(phase: Phase) -> &'static str {
    match phase {
        Phase::Forward => "forward",
        Phase::Backward => "backward",
        Phase::Simulation => "simulation",
    }
}

/// Validates a resolved per-phase solver config against the compiled
/// backend's support matrix (per-phase overrides are HiGHS-only; CLP
/// hard-rejects every override).
///
/// `config` absent is always `Ok(())` — the byte-neutral path
/// ([`Phase::resolve_profile`]) is untouched by this validator.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the offending field, its value,
/// and the compiled backend when `config` sets something the compiled
/// backend does not support.
pub(crate) fn validate_phase_solver_config(
    config: Option<&PhaseSolverProfileConfig>,
    phase: Phase,
) -> Result<(), SddpError> {
    match config {
        Some(config) => validate_backend_support(config, phase),
        None => Ok(()),
    }
}

/// `HiGHS` `factor_pivot_threshold`'s accepted range (`kMinPivotThreshold`/
/// `kMaxPivotThreshold`,
/// `crates/cobre-solver/vendor/HiGHS/highs/util/HFactorConst.h`).
#[cfg(feature = "highs")]
const MIN_FACTOR_PIVOT_THRESHOLD: f64 = 8e-4;
#[cfg(feature = "highs")]
const MAX_FACTOR_PIVOT_THRESHOLD: f64 = 0.5;

#[cfg(feature = "highs")]
const MIN_DUAL_FEASIBILITY_TOLERANCE: f64 = 1e-10;

#[cfg(feature = "highs")]
const MAX_SIMPLEX_UPDATE_LIMIT: u32 = i32::MAX as u32;

/// Under `highs`, every [`DualEdgeWeight`]/[`ScaleStrategy`]/[`PriceStrategy`]
/// variant is supported (the enums are already closed to the supported set),
/// and `presolve`/`use_warm_start` are unconditionally valid (closed enum /
/// bool); every other field is checked against `HighsProfile`'s accepted
/// range, each failure naming the field, its value, and the phase.
#[cfg(feature = "highs")]
fn validate_backend_support(
    config: &PhaseSolverProfileConfig,
    phase: Phase,
) -> Result<(), SddpError> {
    let phase_key = phase_config_key(phase);
    if let Some(tolerance) = config.primal_feasibility_tolerance
        && !tolerance.is_finite()
    {
        return Err(SddpError::Validation(format!(
            "unsupported primal_feasibility_tolerance {tolerance} for backend \"highs\" in phase \"{phase_key}\": value must be finite"
        )));
    }
    if let Some(tolerance) = config.dual_feasibility_tolerance
        && (!tolerance.is_finite() || tolerance < MIN_DUAL_FEASIBILITY_TOLERANCE)
    {
        return Err(SddpError::Validation(format!(
            "unsupported dual_feasibility_tolerance {tolerance} for backend \"highs\" in phase \"{phase_key}\": value must be finite and >= {MIN_DUAL_FEASIBILITY_TOLERANCE}"
        )));
    }
    if let Some(limit) = config.simplex_update_limit
        && limit > MAX_SIMPLEX_UPDATE_LIMIT
    {
        return Err(SddpError::Validation(format!(
            "unsupported simplex_update_limit {limit} for backend \"highs\" in phase \"{phase_key}\": value must be <= {MAX_SIMPLEX_UPDATE_LIMIT}"
        )));
    }
    if let Some(value) = config.cost_perturbation
        && (!value.is_finite() || value < 0.0)
    {
        return Err(SddpError::Validation(format!(
            "unsupported cost_perturbation {value} for backend \"highs\" in phase \"{phase_key}\": value must be finite and >= 0"
        )));
    }
    if let Some(value) = config.refactor_error_tolerance
        && (!value.is_finite() || value < 0.0)
    {
        return Err(SddpError::Validation(format!(
            "unsupported refactor_error_tolerance {value} for backend \"highs\" in phase \"{phase_key}\": value must be finite and >= 0"
        )));
    }
    if let Some(value) = config.factor_pivot_threshold
        && !(MIN_FACTOR_PIVOT_THRESHOLD..=MAX_FACTOR_PIVOT_THRESHOLD).contains(&value)
    {
        return Err(SddpError::Validation(format!(
            "unsupported factor_pivot_threshold {value} for backend \"highs\" in phase \"{phase_key}\": value must be in [{MIN_FACTOR_PIVOT_THRESHOLD}, {MAX_FACTOR_PIVOT_THRESHOLD}]"
        )));
    }
    if let Some(value) = config.steepest_edge_devex_fallback_threshold
        && (!value.is_finite() || value < 1.0)
    {
        return Err(SddpError::Validation(format!(
            "unsupported steepest_edge_devex_fallback_threshold {value} for backend \"highs\" in phase \"{phase_key}\": value must be finite and >= 1.0"
        )));
    }
    Ok(())
}

/// Name of the first field `config` sets, in declaration order, or `None` if
/// none are. Destructures the config struct exhaustively (no `..`), so a
/// field added to `PhaseSolverProfileConfig` fails this function to compile
/// until it is given an arm here — the single source of truth the CLP branch
/// of `validate_backend_support` dispatches through instead of a
/// field-by-field enumeration that a new field could silently bypass.
#[cfg(feature = "clp")]
fn first_set_override(config: &PhaseSolverProfileConfig) -> Option<&'static str> {
    let PhaseSolverProfileConfig {
        dual_edge_weight,
        scale,
        price,
        primal_feasibility_tolerance,
        dual_feasibility_tolerance,
        presolve,
        simplex_update_limit,
        cost_perturbation,
        refactor_error_tolerance,
        factor_pivot_threshold,
        use_warm_start,
        steepest_edge_devex_fallback_threshold,
    } = config;
    if dual_edge_weight.is_some() {
        Some("dual_edge_weight")
    } else if scale.is_some() {
        Some("scale")
    } else if price.is_some() {
        Some("price")
    } else if primal_feasibility_tolerance.is_some() {
        Some("primal_feasibility_tolerance")
    } else if dual_feasibility_tolerance.is_some() {
        Some("dual_feasibility_tolerance")
    } else if presolve.is_some() {
        Some("presolve")
    } else if simplex_update_limit.is_some() {
        Some("simplex_update_limit")
    } else if cost_perturbation.is_some() {
        Some("cost_perturbation")
    } else if refactor_error_tolerance.is_some() {
        Some("refactor_error_tolerance")
    } else if factor_pivot_threshold.is_some() {
        Some("factor_pivot_threshold")
    } else if use_warm_start.is_some() {
        Some("use_warm_start")
    } else if steepest_edge_devex_fallback_threshold.is_some() {
        Some("steepest_edge_devex_fallback_threshold")
    } else {
        None
    }
}

// CLP support is deliberately minimal: reject every override instead
// of silently applying a HiGHS-flavored value to CLP's own option surface.
#[cfg(feature = "clp")]
fn validate_backend_support(
    config: &PhaseSolverProfileConfig,
    phase: Phase,
) -> Result<(), SddpError> {
    let phase_key = phase_config_key(phase);
    let Some(field) = first_set_override(config) else {
        return Ok(());
    };
    if field == "primal_feasibility_tolerance" {
        let tolerance = config.primal_feasibility_tolerance.unwrap_or_default();
        return Err(clp_unsupported(
            format!("field \"primal_feasibility_tolerance\" ({tolerance})"),
            phase_key,
        ));
    }
    Err(clp_unsupported(format!("field \"{field}\""), phase_key))
}

#[cfg(feature = "clp")]
fn clp_unsupported(item: impl std::fmt::Display, phase_key: &str) -> SddpError {
    SddpError::Validation(format!(
        "solver profile {item} for phase \"{phase_key}\" is unsupported on backend \"clp\": rejected until CLP is re-measured"
    ))
}

/// Maps [`DualEdgeWeight`] to `HiGHS`'s `simplex_dual_edge_weight_strategy`.
///
/// Selecting `SteepestEdge` is a request, not a guarantee: under a
/// warm-started solve `HiGHS` silently falls back to Devex pricing once the
/// DSE weight log error exceeds
/// [`HighsProfile::steepest_edge_devex_fallback_threshold`], so a resolve may run
/// partly on Devex regardless of this setting.
#[cfg(feature = "highs")]
fn highs_dual_edge_weight(value: DualEdgeWeight) -> i32 {
    match value {
        DualEdgeWeight::Dantzig => 0,
        DualEdgeWeight::Devex => 1,
        DualEdgeWeight::SteepestEdge => 2,
    }
}

#[cfg(feature = "highs")]
fn highs_scale_strategy(value: ScaleStrategy) -> i32 {
    match value {
        ScaleStrategy::Off => 0,
        ScaleStrategy::SolverScaling => 2, // Curtis–Reid
    }
}

#[cfg(feature = "highs")]
fn highs_price_strategy(value: PriceStrategy) -> i32 {
    match value {
        PriceStrategy::Row => 1,
        PriceStrategy::RowHyperSparse => 2,
    }
}

#[cfg(feature = "highs")]
fn highs_presolve(value: PresolveMode) -> PresolveKind {
    match value {
        PresolveMode::On => PresolveKind::On,
        PresolveMode::Off => PresolveKind::Off,
        PresolveMode::Choose => PresolveKind::Choose,
    }
}

/// Layers `config` onto `base`: each `Some` per-field override replaces the
/// corresponding option, a later field always winning over `base`.
#[cfg(feature = "highs")]
fn resolve_active_profile(
    base: HighsProfile,
    config: Option<&PhaseSolverProfileConfig>,
) -> HighsProfile {
    let Some(config) = config else {
        return base;
    };
    let mut profile = base;
    if let Some(weight) = config.dual_edge_weight {
        profile.simplex_dual_edge_weight_strategy = highs_dual_edge_weight(weight);
    }
    if let Some(scale) = config.scale {
        profile.simplex_scale_strategy = highs_scale_strategy(scale);
    }
    if let Some(price) = config.price {
        profile.simplex_price_strategy = highs_price_strategy(price);
    }
    if let Some(tolerance) = config.primal_feasibility_tolerance {
        profile.primal_feasibility_tolerance = tolerance;
    }
    if let Some(tolerance) = config.dual_feasibility_tolerance {
        profile.dual_feasibility_tolerance = tolerance;
    }
    if let Some(mode) = config.presolve {
        profile.presolve = highs_presolve(mode);
    }
    if let Some(limit) = config.simplex_update_limit {
        profile.simplex_update_limit = limit;
    }
    if let Some(value) = config.cost_perturbation {
        profile.cost_perturbation = value;
    }
    if let Some(value) = config.refactor_error_tolerance {
        profile.refactor_error_tolerance = value;
    }
    if let Some(value) = config.factor_pivot_threshold {
        profile.factor_pivot_threshold = value;
    }
    if let Some(value) = config.use_warm_start {
        profile.use_warm_start = value;
    }
    if let Some(value) = config.steepest_edge_devex_fallback_threshold {
        profile.steepest_edge_devex_fallback_threshold = value;
    }
    profile
}

// CLP support is deliberately minimal: every override is deferred to
// the compiled-backend validation step, which hard-rejects a CLP config
// instead of silently applying a HiGHS-flavored override to CLP's own option
// surface.
#[cfg(feature = "clp")]
fn resolve_active_profile(
    base: ClpProfile,
    _config: Option<&PhaseSolverProfileConfig>,
) -> ClpProfile {
    base
}

// Compile-time drift guard: a field added to `HighsProfile` makes the compiler
// reject these const literals until they are updated, keeping the named
// constants and their documented field values in sync.
#[cfg(feature = "highs")]
const _: () = {
    assert!(FORWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(FORWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(FORWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(FORWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(FORWARD_PROFILE.simplex_price_strategy == 2);
    assert!(matches!(FORWARD_PROFILE.presolve, PresolveKind::On));
    assert!(FORWARD_PROFILE.simplex_update_limit == 5000);
    assert!(FORWARD_PROFILE.cost_perturbation == 0.0);
    assert!(FORWARD_PROFILE.refactor_error_tolerance == 1e-6);
    assert!(FORWARD_PROFILE.factor_pivot_threshold == 0.1);
    assert!(FORWARD_PROFILE.use_warm_start);
    assert!(FORWARD_PROFILE.steepest_edge_devex_fallback_threshold == 10.0);

    assert!(BACKWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(BACKWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(BACKWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(BACKWARD_PROFILE.simplex_price_strategy == 2);
    assert!(matches!(BACKWARD_PROFILE.presolve, PresolveKind::On));
    assert!(BACKWARD_PROFILE.simplex_update_limit == 5000);
    assert!(BACKWARD_PROFILE.cost_perturbation == 0.0);
    assert!(BACKWARD_PROFILE.refactor_error_tolerance == 1e-6);
    assert!(BACKWARD_PROFILE.factor_pivot_threshold == 0.1);
    assert!(BACKWARD_PROFILE.use_warm_start);
    assert!(BACKWARD_PROFILE.steepest_edge_devex_fallback_threshold == 10.0);

    assert!(SIMULATION_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(SIMULATION_PROFILE.ipm_iteration_limit == 10_000);
    assert!(SIMULATION_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(SIMULATION_PROFILE.simplex_scale_strategy == 0);
    assert!(SIMULATION_PROFILE.simplex_price_strategy == 2);
    assert!(matches!(SIMULATION_PROFILE.presolve, PresolveKind::On));
    assert!(SIMULATION_PROFILE.simplex_update_limit == 5000);
    assert!(SIMULATION_PROFILE.cost_perturbation == 0.0);
    assert!(SIMULATION_PROFILE.refactor_error_tolerance == 1e-6);
    assert!(SIMULATION_PROFILE.factor_pivot_threshold == 0.1);
    assert!(SIMULATION_PROFILE.use_warm_start);
    assert!(SIMULATION_PROFILE.steepest_edge_devex_fallback_threshold == 10.0);

    assert!(<HighsProfile as PhaseProfiles>::FORWARD.simplex_price_strategy == 2);
    assert!(<HighsProfile as PhaseProfiles>::BACKWARD.simplex_price_strategy == 2);
    assert!(<HighsProfile as PhaseProfiles>::SIMULATION.simplex_price_strategy == 2);
};

#[cfg(test)]
mod validate_phase_solver_config_tests {
    use super::{Phase, validate_phase_solver_config};

    /// A fully-absent config is always accepted, regardless of the compiled
    /// backend — the byte-neutral resolution path stays untouched.
    #[test]
    fn absent_config_is_accepted() {
        assert!(validate_phase_solver_config(None, Phase::Forward).is_ok());
        assert!(validate_phase_solver_config(None, Phase::Backward).is_ok());
        assert!(validate_phase_solver_config(None, Phase::Simulation).is_ok());
    }
}

#[cfg(all(test, feature = "highs"))]
mod highs_tests {
    use cobre_io::config::{DualEdgeWeight, PhaseSolverProfileConfig, PresolveMode, PriceStrategy};
    use cobre_solver::{HighsProfile, PresolveKind};

    use super::{
        BACKWARD_PROFILE, FORWARD_PROFILE, Phase, PhaseProfiles, SIMULATION_PROFILE,
        validate_phase_solver_config,
    };

    /// A `PhaseSolverProfileConfig` with every field absent — the base every
    /// per-field test starts from via `..blank_config()`.
    fn blank_config() -> PhaseSolverProfileConfig {
        PhaseSolverProfileConfig {
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
            dual_feasibility_tolerance: None,
            presolve: None,
            simplex_update_limit: None,
            cost_perturbation: None,
            refactor_error_tolerance: None,
            factor_pivot_threshold: None,
            use_warm_start: None,
            steepest_edge_devex_fallback_threshold: None,
        }
    }

    /// `Phase::profile()` returns the matching named constant for each variant.
    #[test]
    fn phase_profile_returns_matching_constant() {
        assert_eq!(Phase::Forward.profile(), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.profile(), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.profile(), SIMULATION_PROFILE);
    }

    /// Forward and simulation named constants equal the tuned deep-cut-pool
    /// profile (`BACKWARD_PROFILE`) and differ from [`HighsProfile::default()`]
    /// only in `simplex_price_strategy` (`2`, `RowHyperSparse`).
    #[test]
    fn forward_simulation_equal_tuned_profile() {
        let default = HighsProfile::default();
        assert_eq!(FORWARD_PROFILE, BACKWARD_PROFILE);
        assert_eq!(SIMULATION_PROFILE, BACKWARD_PROFILE);
        assert_ne!(FORWARD_PROFILE, default);
        assert_ne!(SIMULATION_PROFILE, default);
        assert_eq!(FORWARD_PROFILE.simplex_price_strategy, 2);
        assert_eq!(SIMULATION_PROFILE.simplex_price_strategy, 2);
        assert_eq!(default.simplex_price_strategy, 1);
        // Pinning the price strategy back to the default recovers the default.
        let mut forward_relaxed = FORWARD_PROFILE;
        forward_relaxed.simplex_price_strategy = default.simplex_price_strategy;
        assert_eq!(forward_relaxed, default);
    }

    #[test]
    fn backward_profile_overrides_only_price_strategy() {
        let default = HighsProfile::default();
        assert_ne!(BACKWARD_PROFILE, default);
        assert_eq!(BACKWARD_PROFILE.simplex_price_strategy, 2);
        // Fields that are NOT overridden must match the default.
        assert_eq!(
            BACKWARD_PROFILE.simplex_dual_edge_weight_strategy,
            default.simplex_dual_edge_weight_strategy
        );
        assert_eq!(
            BACKWARD_PROFILE.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            BACKWARD_PROFILE.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            BACKWARD_PROFILE.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(
            BACKWARD_PROFILE.ipm_iteration_limit,
            default.ipm_iteration_limit
        );
        assert_eq!(
            BACKWARD_PROFILE.simplex_scale_strategy,
            default.simplex_scale_strategy
        );
    }

    /// The `PhaseProfiles` impl's `FORWARD`/`SIMULATION` equal the tuned
    /// deep-cut-pool profile (`BACKWARD`) and differ from the default only in
    /// `simplex_price_strategy` (`2`, `RowHyperSparse`).
    #[test]
    fn phase_profiles_forward_simulation_equal_tuned_profile() {
        let default = HighsProfile::default();
        let forward = <HighsProfile as PhaseProfiles>::FORWARD;
        let simulation = <HighsProfile as PhaseProfiles>::SIMULATION;
        let backward = <HighsProfile as PhaseProfiles>::BACKWARD;
        assert_eq!(forward, backward);
        assert_eq!(simulation, backward);
        // They differ from the default only in the tuned price strategy.
        assert_ne!(forward, default);
        assert_ne!(simulation, default);
        assert_eq!(forward.simplex_price_strategy, 2);
        assert_eq!(simulation.simplex_price_strategy, 2);
        assert_eq!(default.simplex_price_strategy, 1);
    }

    /// The `PhaseProfiles` impl's `BACKWARD` overrides only
    /// `simplex_price_strategy` to `2`, matching the default elsewhere.
    #[test]
    fn phase_profiles_backward_overrides_only_price_strategy() {
        let default = HighsProfile::default();
        let backward = <HighsProfile as PhaseProfiles>::BACKWARD;
        assert_ne!(backward, default);
        assert_eq!(backward.simplex_price_strategy, 2);
        assert_eq!(
            backward.simplex_dual_edge_weight_strategy,
            default.simplex_dual_edge_weight_strategy
        );
        assert_eq!(
            backward.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            backward.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            backward.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(backward.ipm_iteration_limit, default.ipm_iteration_limit);
        assert_eq!(
            backward.simplex_scale_strategy,
            default.simplex_scale_strategy
        );
    }

    /// Numeric inertness: every `PhaseProfiles` value is bit-for-bit equal to
    /// the corresponding current named constant.
    #[test]
    fn phase_profiles_bit_for_bit_match_named_constants() {
        assert_eq!(<HighsProfile as PhaseProfiles>::FORWARD, FORWARD_PROFILE);
        assert_eq!(<HighsProfile as PhaseProfiles>::BACKWARD, BACKWARD_PROFILE);
        assert_eq!(
            <HighsProfile as PhaseProfiles>::SIMULATION,
            SIMULATION_PROFILE
        );
    }

    // ── `resolve_profile` ────────────────────────────────────────────────────

    /// Absent config resolves to the current per-phase constant, unchanged —
    /// the byte-neutrality gate.
    #[test]
    fn resolve_profile_absent_config_equals_named_constant() {
        assert_eq!(Phase::Forward.resolve_profile(None), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.resolve_profile(None), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.resolve_profile(None), SIMULATION_PROFILE);
    }

    /// A per-field override applies directly on the base constant; fields the
    /// override does not touch keep the base constant's values.
    #[test]
    fn resolve_profile_field_overrides_apply_on_base_constant() {
        let config = PhaseSolverProfileConfig {
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            ..blank_config()
        };
        let resolved = Phase::Forward.resolve_profile(Some(&config));
        assert_eq!(resolved.simplex_dual_edge_weight_strategy, 0);
        assert_eq!(
            resolved.simplex_scale_strategy,
            FORWARD_PROFILE.simplex_scale_strategy
        );
        assert_eq!(
            resolved.simplex_price_strategy,
            FORWARD_PROFILE.simplex_price_strategy
        );
        assert_eq!(
            resolved.primal_feasibility_tolerance,
            FORWARD_PROFILE.primal_feasibility_tolerance
        );
    }

    /// Each of the 8 new fields overrides the base constant when set to a
    /// valid non-default value, and `presolve` maps through `PresolveKind`;
    /// fields not set here keep the base constant's values.
    #[test]
    fn resolve_profile_new_fields_apply_overrides() {
        let config = PhaseSolverProfileConfig {
            dual_feasibility_tolerance: Some(1e-8),
            presolve: Some(PresolveMode::Off),
            simplex_update_limit: Some(2000),
            cost_perturbation: Some(0.5),
            refactor_error_tolerance: Some(1e-5),
            factor_pivot_threshold: Some(0.2),
            use_warm_start: Some(false),
            steepest_edge_devex_fallback_threshold: Some(20.0),
            ..blank_config()
        };
        let resolved = Phase::Backward.resolve_profile(Some(&config));
        assert_eq!(resolved.dual_feasibility_tolerance, 1e-8);
        assert!(matches!(resolved.presolve, PresolveKind::Off));
        assert_eq!(resolved.simplex_update_limit, 2000);
        assert_eq!(resolved.cost_perturbation, 0.5);
        assert_eq!(resolved.refactor_error_tolerance, 1e-5);
        assert_eq!(resolved.factor_pivot_threshold, 0.2);
        assert!(!resolved.use_warm_start);
        assert_eq!(resolved.steepest_edge_devex_fallback_threshold, 20.0);
        assert_eq!(
            resolved.primal_feasibility_tolerance,
            BACKWARD_PROFILE.primal_feasibility_tolerance
        );
        assert_eq!(
            resolved.simplex_dual_edge_weight_strategy,
            BACKWARD_PROFILE.simplex_dual_edge_weight_strategy
        );
    }

    // ── `validate_phase_solver_config` (HiGHS) ──────────────────────────────

    /// A non-finite `primal_feasibility_tolerance` is rejected, naming the
    /// field, its value, and the backend.
    #[test]
    fn validate_phase_solver_config_rejects_non_finite_tolerance() {
        let config = PhaseSolverProfileConfig {
            primal_feasibility_tolerance: Some(f64::NAN),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("non-finite tolerance must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("primal_feasibility_tolerance"),
            "message must name the field: {msg}"
        );
        assert!(
            msg.contains("highs"),
            "message must name the backend: {msg}"
        );
    }

    /// Several per-field overrides the closed enums allow are accepted
    /// together, without error.
    #[test]
    fn validate_phase_solver_config_accepts_multiple_overrides() {
        let config = PhaseSolverProfileConfig {
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            price: Some(PriceStrategy::RowHyperSparse),
            primal_feasibility_tolerance: Some(1e-8),
            ..blank_config()
        };
        assert!(validate_phase_solver_config(Some(&config), Phase::Backward).is_ok());
    }

    /// A `dual_feasibility_tolerance` below `1e-10` is rejected, naming the
    /// field.
    #[test]
    fn validate_phase_solver_config_rejects_dual_feasibility_tolerance_below_min() {
        let config = PhaseSolverProfileConfig {
            dual_feasibility_tolerance: Some(1e-11),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("dual_feasibility_tolerance below 1e-10 must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("dual_feasibility_tolerance"),
            "message must name the field: {msg}"
        );
    }

    /// A `simplex_update_limit` above `i32::MAX` is rejected, naming the field.
    #[test]
    fn validate_phase_solver_config_rejects_simplex_update_limit_above_i32_max() {
        let config = PhaseSolverProfileConfig {
            simplex_update_limit: Some(u32::MAX),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("simplex_update_limit above i32::MAX must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("simplex_update_limit"),
            "message must name the field: {msg}"
        );
    }

    /// A negative `cost_perturbation` (`-1.0`) is rejected, naming the field.
    #[test]
    fn validate_phase_solver_config_rejects_negative_cost_perturbation() {
        let config = PhaseSolverProfileConfig {
            cost_perturbation: Some(-1.0),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("negative cost_perturbation must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("cost_perturbation"),
            "message must name the field: {msg}"
        );
    }

    /// A negative `refactor_error_tolerance` is rejected, naming the field.
    #[test]
    fn validate_phase_solver_config_rejects_negative_refactor_error_tolerance() {
        let config = PhaseSolverProfileConfig {
            refactor_error_tolerance: Some(-1e-6),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("negative refactor_error_tolerance must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("refactor_error_tolerance"),
            "message must name the field: {msg}"
        );
    }

    /// A `factor_pivot_threshold` (`0.9`) above `kMaxPivotThreshold` (`0.5`,
    /// `HFactorConst.h`) is rejected, naming the field.
    #[test]
    fn validate_phase_solver_config_rejects_factor_pivot_threshold_above_max() {
        let config = PhaseSolverProfileConfig {
            factor_pivot_threshold: Some(0.9),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("factor_pivot_threshold above kMaxPivotThreshold must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("factor_pivot_threshold"),
            "message must name the field: {msg}"
        );
    }

    /// `factor_pivot_threshold` at exactly `kMinPivotThreshold`/
    /// `kMaxPivotThreshold` (`8e-4`/`0.5`, `HFactorConst.h`) is accepted —
    /// the range is inclusive.
    #[test]
    fn validate_phase_solver_config_accepts_factor_pivot_threshold_at_bounds() {
        for value in [8e-4, 0.5] {
            let config = PhaseSolverProfileConfig {
                factor_pivot_threshold: Some(value),
                ..blank_config()
            };
            assert!(
                validate_phase_solver_config(Some(&config), Phase::Forward).is_ok(),
                "factor_pivot_threshold={value} is within [kMinPivotThreshold, kMaxPivotThreshold] and must be accepted"
            );
        }
    }

    /// A `steepest_edge_devex_fallback_threshold` below `1.0` (`0.5`) is
    /// rejected, naming the field.
    #[test]
    fn validate_phase_solver_config_rejects_steepest_edge_devex_fallback_threshold_below_one() {
        let config = PhaseSolverProfileConfig {
            steepest_edge_devex_fallback_threshold: Some(0.5),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("steepest_edge_devex_fallback_threshold below 1.0 must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("steepest_edge_devex_fallback_threshold"),
            "message must name the field: {msg}"
        );
    }

    /// Every new field set to a valid non-default value is accepted.
    #[test]
    fn validate_phase_solver_config_accepts_new_fields_at_valid_values() {
        let config = PhaseSolverProfileConfig {
            dual_feasibility_tolerance: Some(1e-8),
            presolve: Some(PresolveMode::Off),
            simplex_update_limit: Some(2000),
            cost_perturbation: Some(0.5),
            refactor_error_tolerance: Some(1e-5),
            factor_pivot_threshold: Some(0.2),
            use_warm_start: Some(false),
            steepest_edge_devex_fallback_threshold: Some(20.0),
            ..blank_config()
        };
        assert!(validate_phase_solver_config(Some(&config), Phase::Backward).is_ok());
    }
}

#[cfg(all(test, feature = "clp"))]
mod clp_tests {
    use cobre_io::config::{
        DualEdgeWeight, PhaseSolverProfileConfig, PresolveMode, PriceStrategy, ScaleStrategy,
    };
    use cobre_solver::{ClpAlgorithm, ClpProfile};

    use super::{Phase, PhaseProfiles, first_set_override, validate_phase_solver_config};

    /// A `PhaseSolverProfileConfig` with every field absent — the base every
    /// per-field test starts from via `..blank_config()`.
    fn blank_config() -> PhaseSolverProfileConfig {
        PhaseSolverProfileConfig {
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
            dual_feasibility_tolerance: None,
            presolve: None,
            simplex_update_limit: None,
            cost_perturbation: None,
            refactor_error_tolerance: None,
            factor_pivot_threshold: None,
            use_warm_start: None,
            steepest_edge_devex_fallback_threshold: None,
        }
    }

    /// Runs `validate_phase_solver_config` on `config` and asserts it is
    /// rejected, naming both the backend and `field_name`.
    fn assert_clp_rejects_field(config: &PhaseSolverProfileConfig, field_name: &str) {
        let err = validate_phase_solver_config(Some(config), Phase::Backward)
            .expect_err("CLP must hard-reject a set field it does not support");
        let msg = err.to_string();
        assert!(msg.contains("clp"), "message must name the backend: {msg}");
        assert!(
            msg.contains(field_name),
            "message must name the field {field_name:?}: {msg}"
        );
    }

    /// The CLP `FORWARD` and `BACKWARD` profiles are the identical tuned
    /// deep-cut-pool profile (dual simplex, `dual_pricing_mode = 1`,
    /// `factorization_frequency = 200`). `SIMULATION` keeps those tuned values
    /// but selects the **primal** simplex, so it differs from `BACKWARD` in
    /// `algorithm` alone. All three differ from [`ClpProfile::default()`].
    #[test]
    fn clp_phase_profiles_tuned_with_primal_simulation() {
        let default = ClpProfile::default();
        let forward = <ClpProfile as PhaseProfiles>::FORWARD;
        let simulation = <ClpProfile as PhaseProfiles>::SIMULATION;
        let backward = <ClpProfile as PhaseProfiles>::BACKWARD;
        // FORWARD and BACKWARD are the identical tuned (dual) profile.
        assert_eq!(forward, backward);
        // SIMULATION runs the primal simplex to dodge CLP's dual
        // false-infeasibilities on warm-started, cut-laden simulation LPs, so it
        // differs from the tuned profile in `algorithm` ONLY.
        assert_ne!(simulation, backward);
        assert_eq!(forward.algorithm, ClpAlgorithm::Dual);
        assert_eq!(backward.algorithm, ClpAlgorithm::Dual);
        assert_eq!(simulation.algorithm, ClpAlgorithm::Primal);
        assert_eq!(
            ClpProfile {
                algorithm: ClpAlgorithm::Dual,
                ..simulation
            },
            backward,
            "SIMULATION must equal the tuned profile except for the primal algorithm"
        );
        // They differ from the default only in the tuned fields.
        assert_ne!(forward, default);
        assert_ne!(simulation, default);
        assert_eq!(forward.dual_pricing_mode, 1);
        assert_eq!(forward.factorization_frequency, 200);
        assert_eq!(simulation.dual_pricing_mode, 1);
        assert_eq!(simulation.factorization_frequency, 200);
        assert_eq!(default.dual_pricing_mode, 3);
        assert_eq!(default.factorization_frequency, 0);
    }

    /// The CLP `BACKWARD` profile overrides only `dual_pricing_mode` (to `1`,
    /// full DSE) and `factorization_frequency` (to `200`), matching the default
    /// on every other field.
    #[test]
    fn clp_backward_profile_overrides_only_pricing_and_factorization() {
        let default = ClpProfile::default();
        let backward = <ClpProfile as PhaseProfiles>::BACKWARD;
        assert_ne!(backward, default);
        assert_eq!(backward.dual_pricing_mode, 1);
        assert_eq!(backward.factorization_frequency, 200);
        // Fields that are NOT overridden must match the default.
        assert_eq!(backward.perturbation, default.perturbation);
        assert_eq!(backward.scaling, default.scaling);
        assert_eq!(
            backward.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            backward.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            backward.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(backward.algorithm, default.algorithm);
    }

    /// `Phase::profile()` returns the matching CLP per-phase profile for each
    /// variant (under `--features clp`, `ActiveProfile` resolves to `ClpProfile`).
    #[test]
    fn phase_profile_returns_matching_clp_profile() {
        assert_eq!(
            Phase::Forward.profile(),
            <ClpProfile as PhaseProfiles>::FORWARD
        );
        assert_eq!(
            Phase::Backward.profile(),
            <ClpProfile as PhaseProfiles>::BACKWARD
        );
        assert_eq!(
            Phase::Simulation.profile(),
            <ClpProfile as PhaseProfiles>::SIMULATION
        );
    }

    /// CLP support is deliberately minimal: `resolve_profile` returns the base
    /// constant regardless of `config` — absent config is byte-neutral, and a
    /// present one is silently ignored pending the compiled-backend reject.
    #[test]
    fn resolve_profile_ignores_config_under_clp() {
        assert_eq!(
            Phase::Backward.resolve_profile(None),
            Phase::Backward.profile()
        );
        let config = PhaseSolverProfileConfig {
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            ..blank_config()
        };
        assert_eq!(
            Phase::Backward.resolve_profile(Some(&config)),
            Phase::Backward.profile()
        );
    }

    // ── `validate_phase_solver_config` (CLP hard-reject) ────────────────────

    /// A per-field override is hard-rejected on CLP — every HiGHS-only
    /// override is rejected.
    #[test]
    fn validate_phase_solver_config_rejects_clp_override() {
        let config = PhaseSolverProfileConfig {
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            ..blank_config()
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("CLP must hard-reject an unsupported override");
        assert!(err.to_string().contains("clp"));
    }

    /// Each of the 8 new fields is hard-rejected on CLP when it is the only
    /// field set, naming both the backend and the field.
    #[test]
    fn validate_phase_solver_config_rejects_clp_new_fields() {
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                dual_feasibility_tolerance: Some(1e-8),
                ..blank_config()
            },
            "dual_feasibility_tolerance",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                presolve: Some(PresolveMode::Off),
                ..blank_config()
            },
            "presolve",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                simplex_update_limit: Some(2000),
                ..blank_config()
            },
            "simplex_update_limit",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                cost_perturbation: Some(0.5),
                ..blank_config()
            },
            "cost_perturbation",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                refactor_error_tolerance: Some(1e-5),
                ..blank_config()
            },
            "refactor_error_tolerance",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                factor_pivot_threshold: Some(0.2),
                ..blank_config()
            },
            "factor_pivot_threshold",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                use_warm_start: Some(false),
                ..blank_config()
            },
            "use_warm_start",
        );
        assert_clp_rejects_field(
            &PhaseSolverProfileConfig {
                steepest_edge_devex_fallback_threshold: Some(20.0),
                ..blank_config()
            },
            "steepest_edge_devex_fallback_threshold",
        );
    }

    /// `first_set_override` names each of the 12 fields when it is the only
    /// one set, and returns `None` for a fully-absent config — the single
    /// source of truth `validate_backend_support` dispatches through.
    #[test]
    fn first_set_override_detects_every_field() {
        assert_eq!(first_set_override(&blank_config()), None);
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                dual_edge_weight: Some(DualEdgeWeight::Dantzig),
                ..blank_config()
            }),
            Some("dual_edge_weight")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                scale: Some(ScaleStrategy::Off),
                ..blank_config()
            }),
            Some("scale")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                price: Some(PriceStrategy::Row),
                ..blank_config()
            }),
            Some("price")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                primal_feasibility_tolerance: Some(1e-8),
                ..blank_config()
            }),
            Some("primal_feasibility_tolerance")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                dual_feasibility_tolerance: Some(1e-8),
                ..blank_config()
            }),
            Some("dual_feasibility_tolerance")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                presolve: Some(PresolveMode::Off),
                ..blank_config()
            }),
            Some("presolve")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                simplex_update_limit: Some(2000),
                ..blank_config()
            }),
            Some("simplex_update_limit")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                cost_perturbation: Some(0.5),
                ..blank_config()
            }),
            Some("cost_perturbation")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                refactor_error_tolerance: Some(1e-5),
                ..blank_config()
            }),
            Some("refactor_error_tolerance")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                factor_pivot_threshold: Some(0.2),
                ..blank_config()
            }),
            Some("factor_pivot_threshold")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                use_warm_start: Some(false),
                ..blank_config()
            }),
            Some("use_warm_start")
        );
        assert_eq!(
            first_set_override(&PhaseSolverProfileConfig {
                steepest_edge_devex_fallback_threshold: Some(20.0),
                ..blank_config()
            }),
            Some("steepest_edge_devex_fallback_threshold")
        );
    }
}
