//! Periodic Autoregressive (PAR) model construction and validation.
//!
//! Fits PAR(p) models to periodic time series; the fitted parameters drive
//! correlated synthetic noise generation during scenario construction.
//!
//! See [`precompute`], [`validation`], [`evaluate`], [`fitting`], and
//! [`contribution`].

pub mod aggregate;
pub mod closure;
pub mod contribution;
pub mod evaluate;
pub mod fitting;
pub mod lag_kernel;
pub mod lag_transition;
pub mod precompute;
pub mod validation;

pub use aggregate::aggregate_observations_to_season;
pub use closure::{
    AnnualParams, ClosureRejection, check_stationarity, check_stationarity_annual,
    derive_residual_std_ratios, derive_residual_std_ratios_annual, implied_periodic_acf,
};
pub use contribution::{
    check_negative_contributions, compute_contributions, find_max_valid_order, has_negative_phi1,
};
#[allow(deprecated)]
pub use evaluate::{
    evaluate_par, evaluate_par_batch, evaluate_par_inflow, evaluate_par_inflows, solve_par_noise,
    solve_par_noise_batch, solve_par_noises,
};
pub use fitting::{
    AicSelectionResult, AnnualSeasonalStats, ArCoefficientEstimate, ArEstimationConfig,
    ContributionReduction, ContributionValidationResult, EstimationReport, HydroEstimationEntry,
    PacfSelectionResult, PeriodicYwAnnualResult, PeriodicYwResult, ReductionReason, SeasonalStats,
    StdRatioDivergence, build_estimation_report, build_extended_periodic_yw_matrix,
    build_periodic_yw_matrix, conditional_facp_partitioned, cross_correlation_a_z_neg1,
    cross_correlation_z_a, estimate_annual_seasonal_stats, estimate_ar_coefficients_with_selection,
    estimate_periodic_ar_annual_coefficients, estimate_periodic_ar_coefficients,
    find_season_for_date, periodic_autocorrelation, periodic_pacf, select_order_aic,
    select_order_pacf, select_order_pacf_annual, solve_linear_system,
};
pub use lag_kernel::{
    DownstreamLagAccum, EntityMajor, LagIndex, LagMajor, PrimaryLagAccum, advance_lag_chain,
};
pub use lag_transition::{
    derive_downstream_par_order, precompute_noise_groups, precompute_stage_lag_transitions,
    resolve_stage_lag_transition,
};
pub use precompute::PrecomputedPar;
pub use validation::validate_par_parameters;
