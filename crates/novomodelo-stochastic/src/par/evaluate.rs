//! PAR(p) evaluation and inverse functions.
//!
//! Forward: output from lag state and a noise realisation. Inverse: the noise
//! that produces a desired target output.
//!
//! ## PAR model equation
//!
//! Given precomputed parameters, the output for series `h` at stage `t` is:
//!
//! ```text
//! a_h = deterministic_base + sum_{l=0}^{psi.len()-1} psi[l] * lag[l] + sigma * eta
//! ```
//!
//! where:
//! - `deterministic_base` encodes `mu_m - sum psi_{m,l} * mu_{m-l}` (built by
//!   [`PrecomputedPar`])
//! - `psi[l]` are the materialised PAR coefficients in original units (lag 1 is
//!   at index 0); `psi.len()` is the number of lag terms the model uses at this
//!   stage — classical PAR(p) when no annual component is active, widened to 12
//!   when PAR(p)-A annual is in effect (annual contribution is spread across
//!   all 12 positions; see [`PrecomputedPar`])
//! - `lag[l]` is the observed series value at lag `l+1`
//! - `sigma` is the residual standard deviation
//! - `eta` is the standardised noise realisation
//!
//! The returned value may be negative; truncation to a physical minimum is the
//! caller's responsibility.
//!
//! [`PrecomputedPar`]: super::precompute::PrecomputedPar

use super::precompute::PrecomputedPar;

/// Evaluate the PAR(p) model equation (see the [module docs](self)) for a single
/// series element at a single stage. Every entry of `psi` contributes; the caller
/// must supply at least `psi.len()` lag values (`lags[0]` is lag-1). The returned
/// value may be negative (truncation is the caller's responsibility).
///
/// # Examples
///
/// AR(0) — mean plus noise:
///
/// ```
/// use cobre_stochastic::evaluate_par;
///
/// let a_h = evaluate_par(100.0, &[], &[], 30.0, 1.5);
/// assert!((a_h - 145.0).abs() < 1e-10);
/// ```
///
/// AR(1) — one lag:
///
/// ```
/// use cobre_stochastic::evaluate_par;
///
/// // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 127.51
/// let a_h = evaluate_par(70.0, &[0.48], &[90.0], 28.62, 0.5);
/// assert!((a_h - 127.51).abs() < 1e-10);
/// ```
#[must_use]
pub fn evaluate_par(
    deterministic_base: f64,
    psi: &[f64],
    lags: &[f64],
    sigma: f64,
    eta: f64,
) -> f64 {
    debug_assert!(
        lags.len() >= psi.len(),
        "lags.len() ({}) must be >= psi.len() ({})",
        lags.len(),
        psi.len()
    );
    let mut a_h = deterministic_base;
    for (l, &p) in psi.iter().enumerate() {
        a_h += p * lags[l];
    }
    a_h + sigma * eta
}

/// Evaluate the PAR(p) model equation for a single series element at a single stage.
///
/// # Deprecation
///
/// Renamed to [`evaluate_par`] for infrastructure-crate genericity.
/// This alias will be removed in a future minor version.
///
/// # Examples
///
/// AR(0) — mean plus noise:
///
/// ```
/// #[allow(deprecated)]
/// use cobre_stochastic::evaluate_par_inflow;
///
/// let a_h = evaluate_par_inflow(100.0, &[], &[], 30.0, 1.5);
/// assert!((a_h - 145.0).abs() < 1e-10);
/// ```
///
/// AR(1) — one lag:
///
/// ```
/// #[allow(deprecated)]
/// use cobre_stochastic::evaluate_par_inflow;
///
/// // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 127.51
/// let a_h = evaluate_par_inflow(70.0, &[0.48], &[90.0], 28.62, 0.5);
/// assert!((a_h - 127.51).abs() < 1e-10);
/// ```
#[must_use]
#[deprecated(since = "0.1.2", note = "use evaluate_par instead")]
pub fn evaluate_par_inflow(
    deterministic_base: f64,
    psi: &[f64],
    lags: &[f64],
    sigma: f64,
    eta: f64,
) -> f64 {
    evaluate_par(deterministic_base, psi, lags, sigma, eta)
}

/// Evaluate the PAR(p) model equation for all series elements at `stage`, writing
/// into `output` without allocating (values may be negative).
///
/// `lag_matrix` is flat, length `max_order * n_series`, indexed as
/// `[lag * n_series + element]` (lag 0 = lag-1 value, contiguous across elements).
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedPar;
/// use cobre_stochastic::evaluate_par_batch;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
///     annual: None,
/// };
/// let par_lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let noise = vec![0.5];
/// let mut output = vec![0.0];
/// evaluate_par_batch(&par_lp, 0, &lag_matrix, &noise, &mut output);
/// assert!((output[0] - 115.0).abs() < 1e-10); // 100.0 + 30.0 * 0.5
/// ```
pub fn evaluate_par_batch(
    par_lp: &PrecomputedPar,
    stage: usize,
    lag_matrix: &[f64],
    noise: &[f64],
    output: &mut [f64],
) {
    let n_series = par_lp.n_hydros();
    debug_assert!(
        noise.len() == n_series,
        "noise.len() ({}) must equal n_series ({})",
        noise.len(),
        n_series
    );
    debug_assert!(
        output.len() == n_series,
        "output.len() ({}) must equal n_series ({})",
        output.len(),
        n_series
    );

    for h in 0..n_series {
        let base = par_lp.deterministic_base(stage, h);
        let sigma = par_lp.sigma(stage, h);
        let psi = par_lp.psi_slice(stage, h);

        let mut a_h = base;
        for (l, &p) in psi.iter().enumerate() {
            a_h += p * lag_matrix[l * n_series + h];
        }
        a_h += sigma * noise[h];
        output[h] = a_h;
    }
}

/// Evaluate the PAR(p) model equation for all series elements at a given stage.
///
/// # Deprecation
///
/// Renamed to [`evaluate_par_batch`] for infrastructure-crate genericity.
/// This alias will be removed in a future minor version.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedPar;
/// #[allow(deprecated)]
/// use cobre_stochastic::evaluate_par_inflows;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
///     annual: None,
/// };
/// let par_lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let noise = vec![0.5];
/// let mut output = vec![0.0];
/// #[allow(deprecated)]
/// evaluate_par_inflows(&par_lp, 0, &lag_matrix, &noise, &mut output);
/// assert!((output[0] - 115.0).abs() < 1e-10); // 100.0 + 30.0 * 0.5
/// ```
#[deprecated(since = "0.1.2", note = "use evaluate_par_batch instead")]
pub fn evaluate_par_inflows(
    par_lp: &PrecomputedPar,
    stage: usize,
    lag_matrix: &[f64],
    noise: &[f64],
    output: &mut [f64],
) {
    evaluate_par_batch(par_lp, stage, lag_matrix, noise, output);
}

/// Solve the PAR(p) equation for the noise `η` producing a given `target` output
/// (set `a_h = target` in the PAR equation and solve):
///
/// ```text
/// η = (target - deterministic_base - Σ psi[l] * lags[l]) / sigma
/// ```
///
/// When `sigma == 0.0` noise cannot move the output: returns `0.0` if `target`
/// matches the deterministic output (within `1e-10`), else `f64::NEG_INFINITY`
/// to signal that no finite noise can reach `target`.
///
/// The caller must supply at least `psi.len()` lag values.
///
/// # Examples
///
/// Solve for noise that produces zero output (truncation):
///
/// ```
/// use cobre_stochastic::solve_par_noise;
///
/// // η = (0.0 - 70.0 - 0.48 * 90.0) / 28.62
/// let eta = solve_par_noise(70.0, &[0.48], &[90.0], 28.62, 0.0);
/// let expected = -(70.0 + 0.48 * 90.0) / 28.62;
/// assert!((eta - expected).abs() < 1e-10);
/// ```
///
/// Zero sigma with a matching target returns `0.0`:
///
/// ```
/// use cobre_stochastic::solve_par_noise;
///
/// // deterministic_value = 100.0 + 0.5 * 50.0 = 125.0; target matches → 0.0
/// let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 125.0);
/// assert_eq!(eta, 0.0_f64);
/// ```
///
/// Zero sigma with a non-matching target returns `f64::NEG_INFINITY`:
///
/// ```
/// use cobre_stochastic::solve_par_noise;
///
/// // deterministic_value = 125.0; target = 0.0 → impossible → NEG_INFINITY
/// let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 0.0);
/// assert_eq!(eta, f64::NEG_INFINITY);
/// ```
#[must_use]
pub fn solve_par_noise(
    deterministic_base: f64,
    psi: &[f64],
    lags: &[f64],
    sigma: f64,
    target: f64,
) -> f64 {
    debug_assert!(
        lags.len() >= psi.len(),
        "lags.len() ({}) must be >= psi.len() ({})",
        lags.len(),
        psi.len()
    );
    let mut deterministic_value = deterministic_base;
    for (l, &p) in psi.iter().enumerate() {
        deterministic_value += p * lags[l];
    }
    if sigma == 0.0 {
        return if (target - deterministic_value).abs() < 1e-10 {
            0.0
        } else {
            f64::NEG_INFINITY
        };
    }
    (target - deterministic_value) / sigma
}

/// Solve the PAR(p) equation for each series element's noise at `stage`, writing
/// into `output` without allocating. `lag_matrix` uses the [`evaluate_par_batch`]
/// layout `[lag * n_series + element]`.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedPar;
/// use cobre_stochastic::{solve_par_noise, solve_par_noise_batch};
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
///     annual: None,
/// };
/// let par_lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let targets = vec![0.0]; // solve for zero output (truncation)
/// let mut output = vec![0.0];
/// solve_par_noise_batch(&par_lp, 0, &lag_matrix, &targets, &mut output);
/// // η = (0.0 - 100.0) / 30.0
/// assert!((output[0] - (-100.0 / 30.0)).abs() < 1e-10);
/// ```
pub fn solve_par_noise_batch(
    par_lp: &PrecomputedPar,
    stage: usize,
    lag_matrix: &[f64],
    targets: &[f64],
    output: &mut [f64],
) {
    let n_series = par_lp.n_hydros();
    debug_assert!(
        targets.len() == n_series,
        "targets.len() ({}) must equal n_series ({})",
        targets.len(),
        n_series
    );
    debug_assert!(
        output.len() == n_series,
        "output.len() ({}) must equal n_series ({})",
        output.len(),
        n_series
    );

    for h in 0..n_series {
        let base = par_lp.deterministic_base(stage, h);
        let sigma = par_lp.sigma(stage, h);
        let psi = par_lp.psi_slice(stage, h);

        let mut deterministic_value = base;
        for (l, &p) in psi.iter().enumerate() {
            deterministic_value += p * lag_matrix[l * n_series + h];
        }
        if sigma == 0.0 {
            output[h] = if (targets[h] - deterministic_value).abs() < 1e-10 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
            continue;
        }
        output[h] = (targets[h] - deterministic_value) / sigma;
    }
}

/// Solve the PAR(p) equation for all series elements at a given stage.
///
/// # Deprecation
///
/// Renamed to [`solve_par_noise_batch`] for naming consistency with
/// [`evaluate_par_batch`]. This alias will be removed in a future minor
/// version.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedPar;
/// #[allow(deprecated)]
/// use cobre_stochastic::{solve_par_noise, solve_par_noises};
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
///     annual: None,
/// };
/// let par_lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let targets = vec![0.0]; // solve for zero output (truncation)
/// let mut output = vec![0.0];
/// #[allow(deprecated)]
/// solve_par_noises(&par_lp, 0, &lag_matrix, &targets, &mut output);
/// // η = (0.0 - 100.0) / 30.0
/// assert!((output[0] - (-100.0 / 30.0)).abs() < 1e-10);
/// ```
#[deprecated(since = "0.1.2", note = "use solve_par_noise_batch instead")]
pub fn solve_par_noises(
    par_lp: &PrecomputedPar,
    stage: usize,
    lag_matrix: &[f64],
    targets: &[f64],
    output: &mut [f64],
) {
    solve_par_noise_batch(par_lp, stage, lag_matrix, targets, output);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use cobre_core::{
        EntityId,
        scenario::InflowModel,
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{StageSpec, single_block},
    };

    use super::{
        evaluate_par, evaluate_par_batch, evaluate_par_inflow, evaluate_par_inflows,
        solve_par_noise, solve_par_noise_batch, solve_par_noises,
    };
    use crate::par::precompute::PrecomputedPar;
    use crate::test_support::InflowModelSpec;

    fn make_stage(index: usize, id: i32, season_id: Option<usize>) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id,
            blocks: single_block("SINGLE", 744.0),
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        })
    }

    fn make_model(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        coeffs: Vec<f64>,
        residual_ratio: f64,
    ) -> InflowModel {
        crate::test_support::make_inflow_model(InflowModelSpec {
            hydro_id,
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: coeffs,
            residual_std_ratio: residual_ratio,
            ..Default::default()
        })
    }

    #[test]
    fn ar0_produces_mean_plus_noise() {
        // a_h = 100.0 + 0 (no lags) + 30.0 * 1.5 = 145.0
        let a_h = evaluate_par_inflow(100.0, &[], &[], 30.0, 1.5);
        assert!(
            (a_h - 145.0).abs() < 1e-10,
            "AR(0): expected 145.0, got {a_h}"
        );
    }

    #[test]
    fn ar1_acceptance_criterion() {
        // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 70.0 + 43.2 + 14.31 = 127.51
        let a_h = evaluate_par_inflow(70.0, &[0.48], &[90.0], 28.62, 0.5);
        assert!(
            (a_h - 127.51).abs() < 1e-10,
            "AR(1): expected 127.51, got {a_h}"
        );
    }

    #[test]
    fn ar2_known_values() {
        // a_h = 50.0 + 0.4 * 80.0 + 0.2 * 60.0 + 20.0 * (-0.5)
        //     = 50.0 + 32.0 + 12.0 - 10.0
        //     = 84.0
        let a_h = evaluate_par_inflow(50.0, &[0.4, 0.2], &[80.0, 60.0], 20.0, -0.5);
        assert!(
            (a_h - 84.0).abs() < 1e-10,
            "AR(2): expected 84.0, got {a_h}"
        );
    }

    #[test]
    fn zero_sigma_returns_deterministic_value() {
        // a_h = 100.0 + 0.3 * 90.0 + 0.0 * anything = 100.0 + 27.0 = 127.0
        let a_h = evaluate_par_inflow(100.0, &[0.3], &[90.0], 0.0, 999.0);
        assert!(
            (a_h - 127.0).abs() < 1e-10,
            "zero sigma: expected 127.0, got {a_h}"
        );
    }

    #[test]
    fn negative_noise_can_produce_negative_inflow() {
        // a_h = 10.0 + 0.0 (no lags) + 100.0 * (-1.0) = -90.0
        // Truncation is not this function's responsibility.
        let a_h = evaluate_par_inflow(10.0, &[], &[], 100.0, -1.0);
        assert!(
            a_h < 0.0,
            "expected negative inflow for large negative noise, got {a_h}"
        );
        assert!((a_h - (-90.0)).abs() < 1e-10, "expected -90.0, got {a_h}");
    }

    #[test]
    fn full_psi_slice_participates_when_widened() {
        // All entries of psi contribute: with PAR(p)-A active, psi.len() exceeds
        // the AR order and the spread annual coefficient lives at the extra
        // positions. This test simulates that shape directly.
        // a_h = 50.0 + 0.4*80.0 + 0.9*60.0 + 0.9*40.0 + 20.0*0.0
        //     = 50.0 + 32.0 + 54.0 + 36.0 = 172.0
        let a_h = evaluate_par_inflow(50.0, &[0.4, 0.9, 0.9], &[80.0, 60.0, 40.0], 20.0, 0.0);
        assert!(
            (a_h - 172.0).abs() < 1e-10,
            "full-psi iteration: expected 172.0, got {a_h}"
        );
    }

    fn make_two_hydro_three_stage_par() -> PrecomputedPar {
        let hydro_ids = [EntityId(3), EntityId(5)];

        let stages: Vec<Stage> = (0..3)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some(0)))
            .collect();

        // Pre-study models needed for lag coefficient conversion.
        let pre_models = vec![
            make_model(3, -2, 80.0, 20.0, vec![], 1.0),
            make_model(3, -1, 90.0, 25.0, vec![], 1.0),
            make_model(5, -1, 60.0, 15.0, vec![], 1.0),
        ];

        let study_models = vec![
            make_model(3, 0, 100.0, 30.0, vec![0.4, 0.2], 0.9),
            make_model(3, 1, 110.0, 28.0, vec![0.35], 0.94),
            make_model(3, 2, 95.0, 25.0, vec![], 1.0),
            make_model(5, 0, 70.0, 18.0, vec![0.5], 0.87),
            make_model(5, 1, 75.0, 20.0, vec![0.45], 0.89),
            make_model(5, 2, 68.0, 17.0, vec![0.3], 0.95),
        ];

        let mut all_models = pre_models;
        all_models.extend(study_models);

        PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap()
    }

    #[test]
    fn batch_matches_single_hydro_at_stage_1() {
        let par_lp = make_two_hydro_three_stage_par();

        // Stage 1: hydro 3 (h_idx=0) has AR(1), hydro 5 (h_idx=1) has AR(1).
        // lag_matrix layout: [lag * n_hydros + hydro]
        // For max_order=2, n_hydros=2: [lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let lag_matrix = vec![
            85.0, // lag0, hydro 3
            55.0, // lag0, hydro 5
            0.0,  // lag1, hydro 3 (unused for AR(1))
            0.0,  // lag1, hydro 5 (unused for AR(1))
        ];
        let noise = vec![0.3_f64, -0.4_f64];
        let mut output = vec![0.0_f64; 2];

        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &noise, &mut output);

        let n_hydros = par_lp.n_hydros();
        let psi_h0 = par_lp.psi_slice(1, 0);
        let lags_h0: Vec<f64> = (0..psi_h0.len())
            .map(|l| lag_matrix[l * n_hydros])
            .collect();
        let expected_h0 = evaluate_par_inflow(
            par_lp.deterministic_base(1, 0),
            psi_h0,
            &lags_h0,
            par_lp.sigma(1, 0),
            noise[0],
        );
        let psi_h1 = par_lp.psi_slice(1, 1);
        let lags_h1: Vec<f64> = (0..psi_h1.len())
            .map(|l| lag_matrix[l * n_hydros + 1])
            .collect();
        let expected_h1 = evaluate_par_inflow(
            par_lp.deterministic_base(1, 1),
            psi_h1,
            &lags_h1,
            par_lp.sigma(1, 1),
            noise[1],
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 mismatch: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 mismatch: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn batch_matches_single_hydro_at_stage_0_ar2() {
        let par_lp = make_two_hydro_three_stage_par();

        // Stage 0: hydro 3 (h_idx=0) has AR(2), hydro 5 (h_idx=1) has AR(1).
        // lag_matrix layout: [lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let lag0_h0 = 90.0_f64;
        let lag0_h1 = 58.0_f64;
        let lag1_h0 = 78.0_f64;
        let lag_matrix = vec![lag0_h0, lag0_h1, lag1_h0, 0.0];
        let noise = vec![1.0_f64, -0.2_f64];
        let mut output = vec![0.0_f64; 2];

        evaluate_par_inflows(&par_lp, 0, &lag_matrix, &noise, &mut output);

        // max_order=2, so per-hydro lag buffers must be length 2.
        // Hydro 1 (AR(1)) gets a zero in the unused lag-1 slot — psi[1]==0 there.
        let expected_h0 = evaluate_par_inflow(
            par_lp.deterministic_base(0, 0),
            par_lp.psi_slice(0, 0),
            &[lag0_h0, lag1_h0],
            par_lp.sigma(0, 0),
            noise[0],
        );
        let expected_h1 = evaluate_par_inflow(
            par_lp.deterministic_base(0, 1),
            par_lp.psi_slice(0, 1),
            &[lag0_h1, 0.0],
            par_lp.sigma(0, 1),
            noise[1],
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 stage0 AR2 mismatch: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 stage0 AR1 mismatch: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn declaration_order_invariance_batch() {
        // Build par_lp with hydros in canonical order [3, 5].
        let hydro_ids_canonical = [EntityId(3), EntityId(5)];
        let stages = vec![make_stage(0, 0, Some(0))];

        let models = vec![
            make_model(3, 0, 100.0, 30.0, vec![], 1.0),
            make_model(5, 0, 200.0, 40.0, vec![], 1.0),
        ];
        // Also build with reversed model declaration order.
        let models_reversed = vec![
            make_model(5, 0, 200.0, 40.0, vec![], 1.0),
            make_model(3, 0, 100.0, 30.0, vec![], 1.0),
        ];

        let par_canonical =
            PrecomputedPar::build(&models, &stages, &hydro_ids_canonical, None).unwrap();
        let par_reversed = PrecomputedPar::build(
            &models_reversed,
            &[stages[0].clone()],
            &hydro_ids_canonical,
            None,
        )
        .unwrap();

        // AR(0) for both, so lag_matrix is empty (max_order=0 → no lag entries).
        let lag_matrix: Vec<f64> = vec![];
        let noise = vec![0.5_f64, -0.3_f64];
        let mut output_canonical = vec![0.0_f64; 2];
        let mut output_reversed = vec![0.0_f64; 2];

        evaluate_par_inflows(
            &par_canonical,
            0,
            &lag_matrix,
            &noise,
            &mut output_canonical,
        );
        evaluate_par_inflows(&par_reversed, 0, &lag_matrix, &noise, &mut output_reversed);

        // Both must produce bit-for-bit identical results.
        assert_eq!(
            output_canonical[0].to_bits(),
            output_reversed[0].to_bits(),
            "h_idx=0 output differs between canonical and reversed model order"
        );
        assert_eq!(
            output_canonical[1].to_bits(),
            output_reversed[1].to_bits(),
            "h_idx=1 output differs between canonical and reversed model order"
        );

        // Cross-check expected values: h_idx=0 is EntityId(3) (mean=100, sigma=30).
        // a_h = 100.0 + 30.0 * 0.5 = 115.0
        assert!(
            (output_canonical[0] - 115.0).abs() < 1e-10,
            "h_idx=0 (EntityId 3): expected 115.0, got {}",
            output_canonical[0]
        );
        // h_idx=1 is EntityId(5) (mean=200, sigma=40).
        // a_h = 200.0 + 40.0 * (-0.3) = 188.0
        assert!(
            (output_canonical[1] - 188.0).abs() < 1e-10,
            "h_idx=1 (EntityId 5): expected 188.0, got {}",
            output_canonical[1]
        );
    }

    #[test]
    fn solve_noise_ar1_for_zero_target() {
        // η = (0.0 - 70.0 - 0.48 * 90.0) / 28.62 = -113.2 / 28.62
        let expected = -(70.0_f64 + 0.48 * 90.0) / 28.62;
        let eta = solve_par_noise(70.0, &[0.48], &[90.0], 28.62, 0.0);
        assert!(
            (eta - expected).abs() < 1e-10,
            "AR(1) zero target: expected {expected}, got {eta}"
        );
    }

    #[test]
    fn solve_noise_roundtrip_zero_target() {
        // Given η from solve_par_noise(..., 0.0), evaluate_par_inflow(..., η) must be 0.0.
        let det_base = 70.0_f64;
        let psi = [0.48_f64];
        let lags = [90.0_f64];
        let sigma = 28.62_f64;

        let eta = solve_par_noise(det_base, &psi, &lags, sigma, 0.0);
        let inflow = evaluate_par_inflow(det_base, &psi, &lags, sigma, eta);
        assert!(
            inflow.abs() < 1e-10,
            "roundtrip: inflow with solved η must be 0.0, got {inflow}"
        );
    }

    #[test]
    fn solve_noise_roundtrip_ar2() {
        let det_base = 50.0_f64;
        let psi = [0.4_f64, 0.2];
        let lags = [80.0_f64, 60.0];
        let sigma = 20.0_f64;

        let eta = solve_par_noise(det_base, &psi, &lags, sigma, 0.0);
        let inflow = evaluate_par_inflow(det_base, &psi, &lags, sigma, eta);
        assert!(
            inflow.abs() < 1e-10,
            "AR(2) roundtrip: expected 0.0, got {inflow}"
        );
    }

    #[test]
    fn solve_noise_roundtrip_nonzero_target() {
        // Solve for a non-zero target and verify roundtrip.
        let det_base = 70.0_f64;
        let psi = [0.48_f64];
        let lags = [90.0_f64];
        let sigma = 28.62_f64;
        let target = 42.0_f64;

        let eta = solve_par_noise(det_base, &psi, &lags, sigma, target);
        let inflow = evaluate_par_inflow(det_base, &psi, &lags, sigma, eta);
        assert!(
            (inflow - target).abs() < 1e-10,
            "roundtrip with target={target}: expected {target}, got {inflow}"
        );
    }

    #[test]
    fn solve_noise_ar0_case() {
        // η = (0.0 - deterministic_base) / sigma
        let det_base = 120.0_f64;
        let sigma = 40.0_f64;
        let expected = -det_base / sigma;
        let eta = solve_par_noise(det_base, &[], &[], sigma, 0.0);
        assert!(
            (eta - expected).abs() < 1e-10,
            "AR(0): expected {expected}, got {eta}"
        );
    }

    #[test]
    fn solve_noise_zero_sigma_returns_neg_infinity() {
        let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 0.0);
        assert!(
            eta.is_infinite() && eta.is_sign_negative(),
            "zero sigma: expected NEG_INFINITY, got {eta}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_solve_par_noise_sigma_zero_matching_target() {
        // deterministic_value = 100.0 + 0.5 * 50.0 = 125.0; target matches → 0.0
        let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 125.0);
        assert_eq!(
            eta, 0.0_f64,
            "sigma=0, matching target: expected 0.0, got {eta}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_solve_par_noise_sigma_zero_non_matching_target() {
        // deterministic_value = 125.0; target = 200.0 → residual = 75.0 → NEG_INFINITY
        let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 200.0);
        assert_eq!(
            eta,
            f64::NEG_INFINITY,
            "sigma=0, non-matching target: expected NEG_INFINITY, got {eta}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_solve_par_noise_sigma_zero_near_matching_target() {
        // deterministic_value = 125.0; target within 1e-11 → still matches
        let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 125.0 + 1e-11);
        assert_eq!(
            eta, 0.0_f64,
            "sigma=0, target within 1e-11: expected 0.0, got {eta}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_solve_par_noise_batch_sigma_zero_matching_target() {
        // Build a single-hydro, single-stage PAR with sigma=0 (std_m3s=0 → sigma=0).
        // AR(0) with mean=125.0, std=0.0 → deterministic_value=125.0.
        use cobre_core::{EntityId, scenario::InflowModel};

        let stage = make_stage(0, 0, Some(0));
        let model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 125.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let par_lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();

        let lag_matrix: Vec<f64> = vec![];
        // Matching target: deterministic_value = 125.0
        let targets_match = vec![125.0_f64];
        let mut output_match = vec![0.0_f64];
        solve_par_noise_batch(&par_lp, 0, &lag_matrix, &targets_match, &mut output_match);
        assert_eq!(
            output_match[0], 0.0_f64,
            "batch sigma=0, matching target: expected 0.0, got {}",
            output_match[0]
        );

        // Non-matching target: deterministic_value = 125.0, target = 200.0
        let targets_no_match = vec![200.0_f64];
        let mut output_no_match = vec![0.0_f64];
        solve_par_noise_batch(
            &par_lp,
            0,
            &lag_matrix,
            &targets_no_match,
            &mut output_no_match,
        );
        assert_eq!(
            output_no_match[0],
            f64::NEG_INFINITY,
            "batch sigma=0, non-matching target: expected NEG_INFINITY, got {}",
            output_no_match[0]
        );
    }

    #[test]
    fn solve_noise_positive_deterministic_gives_negative_eta_for_zero() {
        // Positive deterministic inflow → need large negative noise to reach zero.
        let det_base = 100.0_f64;
        let sigma = 30.0_f64;
        let eta = solve_par_noise(det_base, &[], &[], sigma, 0.0);
        assert!(
            eta < 0.0,
            "positive deterministic inflow: η for zero target must be negative, got {eta}"
        );
    }

    #[test]
    fn solve_noise_negative_deterministic_gives_positive_eta_for_zero() {
        // Negative deterministic inflow → need positive noise to reach zero.
        // η = (0.0 - (-50.0)) / 20.0 = 2.5
        let eta = solve_par_noise(-50.0, &[], &[], 20.0, 0.0);
        assert!(
            eta > 0.0,
            "negative deterministic inflow: η for zero target must be positive, got {eta}"
        );
        assert!((eta - 2.5).abs() < 1e-10, "expected 2.5, got {eta}");
    }

    // -----------------------------------------------------------------------
    // solve_par_noises batch unit tests (deprecated alias)
    // -----------------------------------------------------------------------

    #[test]
    fn batch_solve_matches_single_hydro_at_stage_0() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        // Stage 0: hydro 3 (h_idx=0) has AR(2), hydro 5 (h_idx=1) has AR(1).
        let lag0_h0 = 90.0_f64;
        let lag0_h1 = 58.0_f64;
        let lag1_h0 = 78.0_f64;
        let lag_matrix = vec![lag0_h0, lag0_h1, lag1_h0, 0.0];
        let targets = vec![0.0_f64; n_hydros];

        let mut output = vec![0.0_f64; n_hydros];
        solve_par_noises(&par_lp, 0, &lag_matrix, &targets, &mut output);

        // Per-hydro single-call references. Lag buffer length = psi.len().
        let psi_h0 = par_lp.psi_slice(0, 0);
        let lags_for_h0: Vec<f64> = (0..psi_h0.len())
            .map(|l| lag_matrix[l * n_hydros])
            .collect();
        let expected_h0 = solve_par_noise(
            par_lp.deterministic_base(0, 0),
            psi_h0,
            &lags_for_h0,
            par_lp.sigma(0, 0),
            0.0,
        );
        let psi_h1 = par_lp.psi_slice(0, 1);
        let lags_for_h1: Vec<f64> = (0..psi_h1.len())
            .map(|l| lag_matrix[l * n_hydros + 1])
            .collect();
        let expected_h1 = solve_par_noise(
            par_lp.deterministic_base(0, 1),
            psi_h1,
            &lags_for_h1,
            par_lp.sigma(0, 1),
            0.0,
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 stage0: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 stage0: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn batch_solve_roundtrip_makes_all_inflows_hit_target() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        let lag_matrix = vec![
            85.0, // lag0, hydro 3
            55.0, // lag0, hydro 5
            0.0,  // lag1, hydro 3 (unused for AR(1))
            0.0,  // lag1, hydro 5 (unused for AR(1))
        ];
        let targets = vec![0.0_f64; n_hydros];

        let mut eta = vec![0.0_f64; n_hydros];
        solve_par_noises(&par_lp, 1, &lag_matrix, &targets, &mut eta);

        let mut inflows = vec![0.0_f64; n_hydros];
        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &eta, &mut inflows);

        for (h, &inflow) in inflows.iter().enumerate() {
            assert!(
                (inflow - targets[h]).abs() < 1e-10,
                "batch roundtrip: h={h} inflow must be {}, got {inflow}",
                targets[h]
            );
        }
    }

    #[test]
    fn batch_solve_roundtrip_nonzero_targets() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        let lag_matrix = vec![85.0, 55.0, 0.0, 0.0];
        let targets = vec![42.0_f64, 17.5_f64];

        let mut eta = vec![0.0_f64; n_hydros];
        solve_par_noises(&par_lp, 1, &lag_matrix, &targets, &mut eta);

        let mut inflows = vec![0.0_f64; n_hydros];
        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &eta, &mut inflows);

        for (h, &inflow) in inflows.iter().enumerate() {
            assert!(
                (inflow - targets[h]).abs() < 1e-10,
                "batch roundtrip nonzero: h={h} expected {}, got {inflow}",
                targets[h]
            );
        }
    }

    // -----------------------------------------------------------------------
    // New generic API tests
    // -----------------------------------------------------------------------

    /// `evaluate_par` and `evaluate_par_inflow` must produce bitwise identical
    /// results for the same inputs.
    #[test]
    fn test_evaluate_par_matches_inflow() {
        type EvalCase<'a> = (f64, &'a [f64], &'a [f64], f64, f64);
        let cases: &[EvalCase<'_>] = &[
            (100.0, &[], &[], 30.0, 1.5),
            (70.0, &[0.48], &[90.0], 28.62, 0.5),
            (50.0, &[0.4, 0.2], &[80.0, 60.0], 20.0, -0.5),
            (100.0, &[0.3], &[90.0], 0.0, 999.0),
        ];
        for &(base, psi, lags, sigma, eta) in cases {
            let via_new = evaluate_par(base, psi, lags, sigma, eta);
            let via_old = evaluate_par_inflow(base, psi, lags, sigma, eta);
            assert_eq!(
                via_new.to_bits(),
                via_old.to_bits(),
                "evaluate_par / evaluate_par_inflow mismatch for base={base} eta={eta}"
            );
        }
    }

    /// `evaluate_par_batch` and `evaluate_par_inflows` must fill output slices
    /// with bitwise identical values.
    #[test]
    fn test_evaluate_par_batch_matches_inflows() {
        let par_lp = make_two_hydro_three_stage_par();

        let lag_matrix = vec![85.0, 55.0, 0.0, 0.0];
        let noise = vec![0.3_f64, -0.4_f64];
        let mut output_new = vec![0.0_f64; 2];
        let mut output_old = vec![0.0_f64; 2];

        evaluate_par_batch(&par_lp, 1, &lag_matrix, &noise, &mut output_new);
        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &noise, &mut output_old);

        assert_eq!(
            output_new[0].to_bits(),
            output_old[0].to_bits(),
            "evaluate_par_batch / evaluate_par_inflows mismatch at h=0"
        );
        assert_eq!(
            output_new[1].to_bits(),
            output_old[1].to_bits(),
            "evaluate_par_batch / evaluate_par_inflows mismatch at h=1"
        );
    }

    /// `solve_par_noise_batch` and `solve_par_noises` must fill output slices
    /// with bitwise identical values.
    #[test]
    fn test_solve_par_noise_batch_matches_noises() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        let lag_matrix = vec![85.0, 55.0, 0.0, 0.0];
        let targets = vec![0.0_f64; n_hydros];
        let mut output_new = vec![0.0_f64; n_hydros];
        let mut output_old = vec![0.0_f64; n_hydros];

        solve_par_noise_batch(&par_lp, 1, &lag_matrix, &targets, &mut output_new);
        solve_par_noises(&par_lp, 1, &lag_matrix, &targets, &mut output_old);

        for h in 0..n_hydros {
            assert_eq!(
                output_new[h].to_bits(),
                output_old[h].to_bits(),
                "solve_par_noise_batch / solve_par_noises mismatch at h={h}"
            );
        }
    }
}
