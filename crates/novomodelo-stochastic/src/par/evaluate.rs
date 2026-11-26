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
/// use novomodelo_stochastic::evaluate_par;
///
/// let a_h = evaluate_par(100.0, &[], &[], 30.0, 1.5);
/// assert!((a_h - 145.0).abs() < 1e-10);
/// ```
///
/// AR(1) — one lag:
///
/// ```
/// use novomodelo_stochastic::evaluate_par;
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

#[inline]
fn conditional_mean_at(
    par_lp: &PrecomputedPar,
    stage: usize,
    h: usize,
    lag_matrix: &[f64],
    n_series: usize,
) -> f64 {
    let mut p_h = par_lp.deterministic_base(stage, h);
    for (l, &psi) in par_lp.psi_slice(stage, h).iter().enumerate() {
        p_h += psi * lag_matrix[l * n_series + h];
    }
    p_h
}

/// Evaluate the PAR(p) model equation for all series elements at `stage`, writing
/// into `output` without allocating (values may be negative).
///
/// `lag_matrix` is flat, length `max_order * n_series`, indexed as
/// `[lag * n_series + element]` (lag 0 = lag-1 value, contiguous across elements).
///
/// Bit-identical to [`evaluate_par_conditional_mean_batch`] and
/// [`evaluate_par_noise_term_batch`] followed by [`add_noise_term_batch`].
///
/// # Examples
///
/// ```
/// use novomodelo_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use novomodelo_stochastic::par::precompute::PrecomputedPar;
/// use novomodelo_stochastic::evaluate_par_batch;
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
        output[h] = conditional_mean_at(par_lp, stage, h, lag_matrix, n_series)
            + par_lp.sigma(stage, h) * noise[h];
    }
}

/// Evaluate the PAR(p) conditional mean for all series elements at `stage`,
/// writing into `output` without allocating:
///
/// ```text
/// p_h = b_h + sum_{l=0}^{max_order-1} psi[h][l] * lag_matrix[l * n_series + h]
/// ```
///
/// The sum starts from the deterministic base `b_h` and adds the lag terms in
/// ascending `l`. Each product and each addition is rounded separately (no fused
/// multiply-add). `lag_matrix` is laid out as in [`evaluate_par_batch`].
pub fn evaluate_par_conditional_mean_batch(
    par_lp: &PrecomputedPar,
    stage: usize,
    lag_matrix: &[f64],
    output: &mut [f64],
) {
    let n_series = par_lp.n_hydros();
    debug_assert_eq!(output.len(), n_series, "output.len() must equal n_series");

    for (h, out) in output[..n_series].iter_mut().enumerate() {
        *out = conditional_mean_at(par_lp, stage, h, lag_matrix, n_series);
    }
}

/// Evaluate the PAR(p) noise term `w_h = sigma_h * eta_h` for all series elements
/// at `stage`, writing into `output` without allocating. The product is rounded
/// once and is the same value [`evaluate_par_batch`] adds.
pub fn evaluate_par_noise_term_batch(
    par_lp: &PrecomputedPar,
    stage: usize,
    noise: &[f64],
    output: &mut [f64],
) {
    let n_series = par_lp.n_hydros();
    debug_assert_eq!(noise.len(), n_series, "noise.len() must equal n_series");
    debug_assert_eq!(output.len(), n_series, "output.len() must equal n_series");

    for h in 0..n_series {
        output[h] = par_lp.sigma(stage, h) * noise[h];
    }
}

/// Evaluate the noise term of a lognormal residual bounded below at the series'
/// non-negativity limit, writing into `output` without allocating:
///
/// ```text
/// w_h = p_h * expm1(s_h * eta_h - s_h^2 / 2)
/// s_h^2 = ln_1p((sigma_h / p_h)^2)
/// ```
///
/// `p_h + w_h` keeps the conditional mean `p_h` and the standard deviation
/// `sigma_h` and is never negative. Where `p_h` is not positive or
/// `(sigma_h / p_h)^2` is not finite, writes `sigma_h * eta_h` with the bits of
/// [`evaluate_par_noise_term_batch`]. Each operation is rounded separately (no
/// fused multiply-add), in the order shown. For `|eta_h|` near 40 with
/// `(sigma_h / p_h)^2` near `f64::MAX`, `exp_m1` overflows to infinity;
/// generated noise does not reach this, only a hand-written opening-tree file
/// can.
///
/// Every slice has length `par_lp.n_hydros()`.
pub fn evaluate_lognormal_noise_term_batch(
    par_lp: &PrecomputedPar,
    stage: usize,
    conditional_mean: &[f64],
    noise: &[f64],
    output: &mut [f64],
) {
    let n_series = par_lp.n_hydros();
    debug_assert_eq!(
        conditional_mean.len(),
        n_series,
        "conditional_mean.len() must equal n_series"
    );
    debug_assert_eq!(noise.len(), n_series, "noise.len() must equal n_series");
    debug_assert_eq!(output.len(), n_series, "output.len() must equal n_series");

    for h in 0..n_series {
        let sigma = par_lp.sigma(stage, h);
        let p = conditional_mean[h];
        let eta = noise[h];
        if p > 0.0 {
            let c = sigma / p;
            let c2 = c * c;
            if c2.is_finite() {
                let s2 = c2.ln_1p();
                let s = s2.sqrt();
                let x = s * eta - 0.5 * s2;
                output[h] = p * x.exp_m1();
                continue;
            }
        }
        output[h] = sigma * eta;
    }
}

/// Add a noise term to a conditional mean element-wise, writing
/// `output[h] = conditional_mean[h] + noise_term[h]` without allocating. Each
/// element is one rounded addition (no fused multiply-add).
///
/// With `conditional_mean` from [`evaluate_par_conditional_mean_batch`] and
/// `noise_term` from [`evaluate_par_noise_term_batch`], `output` equals
/// [`evaluate_par_batch`] bit for bit.
///
/// # Panics
///
/// Panics if the three slices differ in length.
pub fn add_noise_term_batch(conditional_mean: &[f64], noise_term: &[f64], output: &mut [f64]) {
    assert!(
        conditional_mean.len() == noise_term.len() && noise_term.len() == output.len(),
        "conditional_mean, noise_term and output must have equal lengths"
    );

    for ((out, &mean), &term) in output.iter_mut().zip(conditional_mean).zip(noise_term) {
        *out = mean + term;
    }
}

/// Solve the PAR(p) equation for the noise `η` producing a given `target` output
/// (set `a_h = target` in the PAR equation and solve):
///
/// ```text
/// η = (target - deterministic_base - Σ psi[l] * lags[l]) / sigma
/// ```
///
/// When `sigma == 0.0` noise cannot move the output, so the noise is `0.0`
/// whatever `target` is.
///
/// The caller must supply at least `psi.len()` lag values.
///
/// # Examples
///
/// Solve for the noise that produces a zero output:
///
/// ```
/// use novomodelo_stochastic::solve_par_noise;
///
/// // η = (0.0 - 70.0 - 0.48 * 90.0) / 28.62
/// let eta = solve_par_noise(70.0, &[0.48], &[90.0], 28.62, 0.0);
/// let expected = -(70.0 + 0.48 * 90.0) / 28.62;
/// assert!((eta - expected).abs() < 1e-10);
/// ```
///
/// Zero sigma gives zero noise, even for a target away from the deterministic
/// output `100.0 + 0.5 * 50.0 = 125.0`:
///
/// ```
/// use novomodelo_stochastic::solve_par_noise;
///
/// let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, 0.0);
/// assert_eq!(eta, 0.0_f64);
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
    if sigma == 0.0 {
        return 0.0;
    }
    let mut deterministic_value = deterministic_base;
    for (l, &p) in psi.iter().enumerate() {
        deterministic_value += p * lags[l];
    }
    (target - deterministic_value) / sigma
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use novomodelo_core::{
        EntityId,
        scenario::InflowModel,
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{InflowModelSpec, StageSpec, single_block},
    };

    use proptest::{prop_assert_eq, proptest};
    use rand::RngExt;

    use super::{
        add_noise_term_batch, evaluate_lognormal_noise_term_batch, evaluate_par,
        evaluate_par_batch, evaluate_par_conditional_mean_batch, evaluate_par_noise_term_batch,
        solve_par_noise,
    };
    use crate::noise::rng::rng_from_seed;
    use crate::par::precompute::PrecomputedPar;

    fn make_stage(index: usize, id: i32, season_id: Option<usize>) -> Stage {
        novomodelo_core::test_support::make_stage(StageSpec {
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
        novomodelo_core::test_support::make_inflow_model(InflowModelSpec {
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
        let a_h = evaluate_par(100.0, &[], &[], 30.0, 1.5);
        assert!(
            (a_h - 145.0).abs() < 1e-10,
            "AR(0): expected 145.0, got {a_h}"
        );
    }

    #[test]
    fn ar1_known_values() {
        // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 70.0 + 43.2 + 14.31 = 127.51
        let a_h = evaluate_par(70.0, &[0.48], &[90.0], 28.62, 0.5);
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
        let a_h = evaluate_par(50.0, &[0.4, 0.2], &[80.0, 60.0], 20.0, -0.5);
        assert!(
            (a_h - 84.0).abs() < 1e-10,
            "AR(2): expected 84.0, got {a_h}"
        );
    }

    #[test]
    fn zero_sigma_returns_deterministic_value() {
        // a_h = 100.0 + 0.3 * 90.0 + 0.0 * anything = 100.0 + 27.0 = 127.0
        let a_h = evaluate_par(100.0, &[0.3], &[90.0], 0.0, 999.0);
        assert!(
            (a_h - 127.0).abs() < 1e-10,
            "zero sigma: expected 127.0, got {a_h}"
        );
    }

    #[test]
    fn negative_noise_can_produce_negative_inflow() {
        // a_h = 10.0 + 0.0 (no lags) + 100.0 * (-1.0) = -90.0
        // Truncation is not this function's responsibility.
        let a_h = evaluate_par(10.0, &[], &[], 100.0, -1.0);
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
        let a_h = evaluate_par(50.0, &[0.4, 0.9, 0.9], &[80.0, 60.0, 40.0], 20.0, 0.0);
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

        evaluate_par_batch(&par_lp, 1, &lag_matrix, &noise, &mut output);

        let n_hydros = par_lp.n_hydros();
        let psi_h0 = par_lp.psi_slice(1, 0);
        let lags_h0: Vec<f64> = (0..psi_h0.len())
            .map(|l| lag_matrix[l * n_hydros])
            .collect();
        let expected_h0 = evaluate_par(
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
        let expected_h1 = evaluate_par(
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

        evaluate_par_batch(&par_lp, 0, &lag_matrix, &noise, &mut output);

        // max_order=2, so per-hydro lag buffers must be length 2.
        // Hydro 1 (AR(1)) gets a zero in the unused lag-1 slot — psi[1]==0 there.
        let expected_h0 = evaluate_par(
            par_lp.deterministic_base(0, 0),
            par_lp.psi_slice(0, 0),
            &[lag0_h0, lag1_h0],
            par_lp.sigma(0, 0),
            noise[0],
        );
        let expected_h1 = evaluate_par(
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

        evaluate_par_batch(
            &par_canonical,
            0,
            &lag_matrix,
            &noise,
            &mut output_canonical,
        );
        evaluate_par_batch(&par_reversed, 0, &lag_matrix, &noise, &mut output_reversed);

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
        // Given η from solve_par_noise(..., 0.0), evaluate_par(..., η) must be 0.0.
        let det_base = 70.0_f64;
        let psi = [0.48_f64];
        let lags = [90.0_f64];
        let sigma = 28.62_f64;

        let eta = solve_par_noise(det_base, &psi, &lags, sigma, 0.0);
        let inflow = evaluate_par(det_base, &psi, &lags, sigma, eta);
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
        let inflow = evaluate_par(det_base, &psi, &lags, sigma, eta);
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
        let inflow = evaluate_par(det_base, &psi, &lags, sigma, eta);
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
    fn zero_sigma_gives_zero_noise_whatever_the_target() {
        for target in [0.0, 125.0, 125.0 + 1e-11, 200.0, -50.0] {
            let eta = solve_par_noise(100.0, &[0.5], &[50.0], 0.0, target);
            assert_eq!(
                eta.to_bits(),
                0.0_f64.to_bits(),
                "sigma=0, target {target}: expected 0.0, got {eta}"
            );
        }
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

    // Lag-major `[lag * 2 + hydro]`; non-integral lags and noise.
    const SPLIT_LAGS: [f64; 4] = [85.3, 57.7, 78.9, 41.1];
    const SPLIT_NOISE: [f64; 2] = [0.37, -0.83];

    fn make_split_par(hydro_ids: [EntityId; 2]) -> PrecomputedPar {
        let stages = vec![make_stage(0, 0, Some(0))];
        let models = vec![
            make_model(3, -2, 80.0, 20.0, vec![], 1.0),
            make_model(3, -1, 90.0, 25.0, vec![], 1.0),
            make_model(5, -2, 55.0, 14.0, vec![], 1.0),
            make_model(5, -1, 60.0, 15.0, vec![], 1.0),
            make_model(3, 0, 100.0, 30.0, vec![0.4, 0.2], 0.93),
            make_model(5, 0, 70.0, 18.0, vec![0.5, 0.15], 0.87),
        ];
        let par_lp = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        assert!(par_lp.max_order() >= 2);
        for h in 0..par_lp.n_hydros() {
            assert!(
                par_lp.sigma(0, h).fract() != 0.0,
                "sigma must be non-integral"
            );
            assert!(
                par_lp.psi_slice(0, h).iter().all(|&psi| psi != 0.0),
                "psi must be nonzero at every lag"
            );
        }
        par_lp
    }

    /// Returns `[p, w, z]` with every output pre-filled with NaN to prove it is overwritten.
    fn run_split(par_lp: &PrecomputedPar, lags: &[f64], noise: &[f64]) -> [[f64; 2]; 3] {
        let mut p = [f64::NAN; 2];
        let mut w = [f64::NAN; 2];
        let mut z = [f64::NAN; 2];
        evaluate_par_conditional_mean_batch(par_lp, 0, lags, &mut p);
        evaluate_par_noise_term_batch(par_lp, 0, noise, &mut w);
        add_noise_term_batch(&p, &w, &mut z);
        [p, w, z]
    }

    fn swap_hydro_pairs(values: &[f64]) -> Vec<f64> {
        values.chunks_exact(2).flat_map(|c| [c[1], c[0]]).collect()
    }

    #[test]
    fn split_matches_single_call_bitwise() {
        let par_lp = make_split_par([EntityId(3), EntityId(5)]);
        let mut single = [f64::NAN; 2];
        evaluate_par_batch(&par_lp, 0, &SPLIT_LAGS, &SPLIT_NOISE, &mut single);

        let [_, _, z] = run_split(&par_lp, &SPLIT_LAGS, &SPLIT_NOISE);

        for h in 0..2 {
            assert_eq!(
                z[h].to_bits(),
                single[h].to_bits(),
                "split / single-call mismatch at h={h}"
            );
        }
    }

    #[test]
    fn noise_term_feeds_the_same_product_bits() {
        let par_lp = make_split_par([EntityId(3), EntityId(5)]);

        let [p, w, z] = run_split(&par_lp, &SPLIT_LAGS, &SPLIT_NOISE);

        for h in 0..2 {
            assert_eq!(
                w[h].to_bits(),
                (par_lp.sigma(0, h) * SPLIT_NOISE[h]).to_bits(),
                "noise term is not sigma * eta at h={h}"
            );
            assert_eq!(
                z[h].to_bits(),
                (p[h] + w[h]).to_bits(),
                "realised value is not p + w at h={h}"
            );
        }
    }

    #[test]
    fn split_declaration_order_invariance() {
        let canonical = make_split_par([EntityId(3), EntityId(5)]);
        let permuted = make_split_par([EntityId(5), EntityId(3)]);

        let out_canonical = run_split(&canonical, &SPLIT_LAGS, &SPLIT_NOISE);
        let out_permuted = run_split(
            &permuted,
            &swap_hydro_pairs(&SPLIT_LAGS),
            &swap_hydro_pairs(&SPLIT_NOISE),
        );

        for (name, (c, p)) in ["p", "w", "z"]
            .into_iter()
            .zip(out_canonical.iter().zip(&out_permuted))
        {
            assert_eq!(
                c[0].to_bits(),
                p[1].to_bits(),
                "{name}: canonical h=0 differs from permuted h=1"
            );
            assert_eq!(
                c[1].to_bits(),
                p[0].to_bits(),
                "{name}: canonical h=1 differs from permuted h=0"
            );
        }
    }

    fn make_orders_0_to_2_par() -> PrecomputedPar {
        let stages = vec![make_stage(0, 0, Some(0))];
        let models = vec![
            make_model(3, -2, 80.0, 20.0, vec![], 1.0),
            make_model(3, -1, 90.0, 25.0, vec![], 1.0),
            make_model(5, -2, 55.0, 14.0, vec![], 1.0),
            make_model(5, -1, 60.0, 15.0, vec![], 1.0),
            make_model(7, -2, 40.0, 9.0, vec![], 1.0),
            make_model(7, -1, 45.0, 11.0, vec![], 1.0),
            make_model(3, 0, 100.0, 30.0, vec![0.4, 0.2], 0.93),
            make_model(5, 0, 70.0, 18.0, vec![0.5], 0.87),
            make_model(7, 0, 50.0, 12.0, vec![], 1.0),
        ];
        PrecomputedPar::build(
            &models,
            &stages,
            &[EntityId(3), EntityId(5), EntityId(7)],
            None,
        )
        .unwrap()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn adding_the_noise_term_equals_the_fused_add_bit_for_bit() {
        let par_lp = make_orders_0_to_2_par();
        let n = par_lp.n_hydros();
        assert_eq!(n, 3);
        assert_eq!(par_lp.max_order(), 2);

        let mut rng = rng_from_seed(0x5eed_0073);
        for draw in 0..512 {
            let lags: Vec<f64> = (0..par_lp.max_order() * n)
                .map(|_| 20.0 + 180.0 * rng.random::<f64>())
                .collect();
            let noise: Vec<f64> = match draw {
                0 => vec![0.0, -0.0, 0.0],
                1 => vec![-0.0, 0.0, -0.0],
                _ => (0..n).map(|_| 8.0 * rng.random::<f64>() - 4.0).collect(),
            };

            let mut mean = vec![f64::NAN; n];
            let mut term = vec![f64::NAN; n];
            let mut added = vec![f64::NAN; n];
            let mut fused = vec![f64::NAN; n];
            evaluate_par_conditional_mean_batch(&par_lp, 0, &lags, &mut mean);
            evaluate_par_noise_term_batch(&par_lp, 0, &noise, &mut term);
            add_noise_term_batch(&mean, &term, &mut added);
            evaluate_par_batch(&par_lp, 0, &lags, &noise, &mut fused);

            assert_eq!(bits(&added), bits(&fused), "draw {draw}");
        }
    }

    fn make_ar0_par(stds: &[f64]) -> PrecomputedPar {
        let ids: Vec<EntityId> = (1..).map(EntityId).take(stds.len()).collect();
        let models: Vec<_> = ids
            .iter()
            .zip(stds)
            .map(|(id, &std)| make_model(id.0, 0, 100.0, std, vec![], 1.0))
            .collect();
        PrecomputedPar::build(&models, &[make_stage(0, 0, Some(0))], &ids, None).unwrap()
    }

    fn lognormal_term<const N: usize>(
        par_lp: &PrecomputedPar,
        p: [f64; N],
        eta: [f64; N],
    ) -> [f64; N] {
        let mut w = [f64::NAN; N];
        evaluate_lognormal_noise_term_batch(par_lp, 0, &p, &eta, &mut w);
        w
    }

    fn normal_pdf(eta: f64) -> f64 {
        (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    fn trapezoid(lo: f64, hi: f64, step: f64, f: impl Fn(f64) -> f64) -> f64 {
        let mut sum = 0.0;
        let mut last = 0.0;
        let mut i = 0u32;
        while lo + f64::from(i) * step <= hi {
            last = f(lo + f64::from(i) * step);
            sum += if i == 0 { 0.5 * last } else { last };
            i += 1;
        }
        (sum - 0.5 * last) * step
    }

    #[test]
    fn lognormal_term_keeps_the_conditional_mean_and_std() {
        let par_lp = make_ar0_par(&[1.0]);
        assert_eq!(par_lp.sigma(0, 0).to_bits(), 1.0f64.to_bits());
        let sigma = par_lp.sigma(0, 0);

        for cv in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0] {
            let p = sigma / cv;
            let hi = 2.0 * (cv * cv).ln_1p().sqrt() + 12.0;
            let z_at = |eta: f64| p + lognormal_term(&par_lp, [p], [eta])[0];
            let moments = |step: f64| {
                let mean = trapezoid(-12.0, hi, step, |eta| z_at(eta) * normal_pdf(eta));
                let var = trapezoid(-12.0, hi, step, |eta| {
                    (z_at(eta) - p) * (z_at(eta) - p) * normal_pdf(eta)
                });
                (mean, var.sqrt())
            };
            let (mean, sd) = moments(1e-3);
            let (mean_coarse, sd_coarse) = moments(2e-3);

            assert!((mean - p).abs() / p <= 1e-8, "cv {cv}: mean {mean}, p {p}");
            assert!(
                (sd - sigma).abs() / sigma <= 1e-8,
                "cv {cv}: sd {sd}, sigma {sigma}"
            );
            assert!(
                (mean - mean_coarse).abs() / mean <= 1e-10,
                "cv {cv}: mean {mean} vs coarse {mean_coarse}"
            );
            assert!(
                (sd - sd_coarse).abs() / sd <= 1e-10,
                "cv {cv}: sd {sd} vs coarse {sd_coarse}"
            );
        }
    }

    #[test]
    fn lognormal_term_matches_the_second_order_gaussian_limit() {
        let cases: [(f64, &[f64]); 2] = [(1e-6, &[-2.0, -1.5, 1.5, 2.0]), (1e-10, &[2.0])];
        for (c, etas) in cases {
            let par_lp = make_ar0_par(&[c]);
            let sigma = par_lp.sigma(0, 0);
            for &eta in etas {
                let w = lognormal_term(&par_lp, [1.0], [eta])[0];
                let w2 = sigma * eta + sigma * c * (eta * eta - 1.0) / 2.0;
                assert!(
                    (w - w2).abs() <= 1e-12 * w2.abs(),
                    "c {c}, eta {eta}: w {w:e}, second order {w2:e}"
                );
            }
        }
    }

    #[test]
    fn lognormal_inflow_is_finite_and_non_negative_at_sampler_extremes() {
        let par_lp = make_ar0_par(&[1.0, 1e-290, 1e150, 1e-320]);
        let p = [1.0, 1e-300, 1.0, 5e-324];

        for eta in [-38.5, -8.6, 8.6, 38.5] {
            let w = lognormal_term(&par_lp, p, [eta; 4]);
            let mut z = [f64::NAN; 4];
            add_noise_term_batch(&p, &w, &mut z);

            for h in 0..4 {
                let case = format!("eta {eta}, p {:e}, sigma {:e}", p[h], par_lp.sigma(0, h));
                assert!(z[h].is_finite(), "{case}: z {}", z[h]);
                assert!(z[h] >= 0.0, "{case}: z {}", z[h]);
                assert_ne!(z[h].to_bits(), (-0.0f64).to_bits(), "{case}: z is -0.0");
                assert_ne!(
                    w[h].to_bits(),
                    (par_lp.sigma(0, h) * eta).to_bits(),
                    "{case}: the fallback ran"
                );
            }
        }
    }

    #[test]
    fn lognormal_term_rounds_each_operation_in_the_documented_order() {
        let par_lp = make_ar0_par(&[0.7, 3.0, 40.0]);
        let p = [2.3, 0.9, 1.7];
        for eta in [-2.4, -0.3, 0.6, 1.9] {
            let w = lognormal_term(&par_lp, p, [eta; 3]);
            for h in 0..3 {
                let c = par_lp.sigma(0, h) / p[h];
                let s2 = (c * c).ln_1p();
                let x = s2.sqrt() * eta - s2 / 2.0;
                let expected = p[h] * x.exp_m1();
                assert_eq!(w[h].to_bits(), expected.to_bits(), "h {h}, eta {eta}");
            }
        }
    }

    #[test]
    fn lognormal_fallback_has_the_bits_of_the_normal_term() {
        let par_lp = make_split_par([EntityId(3), EntityId(5)]);
        let mut normal = [f64::NAN; 2];
        evaluate_par_noise_term_batch(&par_lp, 0, &SPLIT_NOISE, &mut normal);

        for p in [[-3.0, 0.0], [-0.0, 1e-300]] {
            let mut w = [f64::NAN; 2];
            evaluate_lognormal_noise_term_batch(&par_lp, 0, &p, &SPLIT_NOISE, &mut w);
            assert_eq!(bits(&w), bits(&normal), "p {p:?}");
        }
    }

    #[test]
    fn lognormal_zero_sigma_leaves_the_conditional_mean() {
        let par_lp = make_ar0_par(&[0.0]);
        for p in [5.0, -1.0] {
            for eta in [1.5, -1.5, 0.0, -0.0] {
                let w = lognormal_term(&par_lp, [p], [eta]);
                let mut z = [f64::NAN];
                add_noise_term_batch(&[p], &w, &mut z);

                assert_eq!(w[0].to_bits(), (0.0 * eta).to_bits(), "p {p}, eta {eta}");
                assert_eq!(z[0].to_bits(), p.to_bits(), "p {p}, eta {eta}");
            }
        }
    }

    proptest! {
        #[test]
        fn lognormal_term_is_invariant_to_hydro_order(
            p0 in -50.0..500.0f64,
            p1 in -50.0..500.0f64,
            eta0 in -9.0..9.0f64,
            eta1 in -9.0..9.0f64,
        ) {
            let canonical = make_split_par([EntityId(3), EntityId(5)]);
            let permuted = make_split_par([EntityId(5), EntityId(3)]);

            let w_c = lognormal_term(&canonical, [p0, p1], [eta0, eta1]);
            let w_p = lognormal_term(&permuted, [p1, p0], [eta1, eta0]);

            prop_assert_eq!(w_c[0].to_bits(), w_p[1].to_bits());
            prop_assert_eq!(w_c[1].to_bits(), w_p[0].to_bits());
        }
    }
}
