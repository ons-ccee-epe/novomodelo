//! Integration test: effective lag coefficients ψ_eff of PAR(2)-A vs classical PAR.
//!
//! Builds two small synthetic fixtures (24 stages × 12 seasons) and reads
//! [`PrecomputedPar`] directly. No LP build, no solve, no filesystem I/O.
//! Sub-second runtime; not gated behind `slow-tests`.
//!
//! ## What is being verified
//!
//! The PAR(p)-A architectural insight is that the annual component is absorbed
//! into a single effective coefficient per lag:
//!
//! ```text
//! ψ_eff[lag] = φ̂_{lag+1} + ψ̂/12   for lag ∈ [0, ar_order)
//! ψ_eff[lag] =             ψ̂/12   for lag ∈ [ar_order, 12)
//! ```
//!
//! where `ψ̂ = ψ · σ_m / σ^A`. These tests assert those values on
//! [`PrecomputedPar::psi_slice`], the single owner of ψ_eff.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use chrono::NaiveDate;
use novomodelo_core::{
    EntityId,
    scenario::{AnnualComponent, InflowModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use novomodelo_stochastic::PrecomputedPar;

mod common;
use common::builders::{StageSpec, make_stage};

/// Number of study stages in both fixtures (2 full cycles of 12 months).
const N_STUDY: usize = 24;
/// Number of seasons (months per year).
const N_SEASONS: usize = 12;
/// PAR classical order used in study models.
const AR_ORDER: usize = 2;

/// Monthly inflow σ used in all seasonal models.
const SIGMA_M: f64 = 200.0;
/// Annual inflow σ used in `AnnualComponent`.
const SIGMA_A: f64 = 250.0;
/// Annual coefficient ψ in `AnnualComponent`.
const PSI: f64 = 0.1;
/// AR-1 coefficient φ₁ in study models.
const PHI_1: f64 = 0.5;
/// AR-2 coefficient φ₂ in study models.
const PHI_2: f64 = 0.2;

/// Build the 2-hydro, 24-stage, 12-season [`PrecomputedPar`]. `annual: Some(_)`
/// reproduces the PAR(2)-A fixture; `annual: None` the classical PAR(2) fixture.
///
/// Uniform σ_m = 200, σ^A = 250 across all stages, so σ_m / σ_{m-1} = 1.0 and
/// φ̂_j = φ_j. Pre-study models (stage ids -1 and -2) let the builder resolve
/// lag-stage statistics for stage 0.
fn build_par_a_fixture_core(annual: Option<&AnnualComponent>) -> PrecomputedPar {
    let hydro_ids = [EntityId(1), EntityId(2)];

    let study_stages: Vec<Stage> = (0..N_STUDY)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    season_id: Some(i % N_SEASONS),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: true,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let mut all_inflow_models: Vec<InflowModel> = Vec::new();

    for &h_id in &hydro_ids {
        for pre_id in [-2_i32, -1_i32] {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: pre_id,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: annual.cloned(),
            });
        }

        for i in 0..N_STUDY {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: i as i32,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![PHI_1, PHI_2],
                residual_std_ratio: 0.7,
                annual: annual.cloned(),
            });
        }
    }

    PrecomputedPar::build(&all_inflow_models, &study_stages, &hydro_ids, None)
        .expect("PrecomputedPar::build must succeed for a valid PAR(2) fixture")
}

/// PAR(2)-A fixture: [`build_par_a_fixture_core`] with the annual component present.
fn build_par_a_fixture() -> PrecomputedPar {
    let annual = AnnualComponent {
        coefficient: PSI,
        mean_m3s: 1000.0,
        std_m3s: SIGMA_A,
    };
    build_par_a_fixture_core(Some(&annual))
}

/// Classical PAR(2) fixture: [`build_par_a_fixture_core`] with no annual
/// component; `max_order` stays at 2.
fn build_classical_fixture() -> PrecomputedPar {
    build_par_a_fixture_core(None)
}

/// Lag 11 of hydro 0 at stage 0 carries `ψ̂/12` alone, beyond the AR order.
#[test]
fn lag_11_psi_equals_psi_hat_over_twelve() {
    let par = build_par_a_fixture();

    assert_eq!(
        par.max_order(),
        12,
        "PAR(2)-A must widen max_order to 12; got {}",
        par.max_order()
    );
    assert_eq!(
        par.effective_lag_count(0),
        12,
        "PAR(2)-A must report 12 effective lags for hydro 0; got {}",
        par.effective_lag_count(0)
    );
    assert!(par.has_annual(0), "hydro 0 must report an annual component");

    let psi = par.psi_slice(0, 0);
    assert_eq!(
        psi.len(),
        12,
        "psi_slice(0, 0) must hold 12 coefficients; got {}",
        psi.len()
    );

    let psi_hat = PSI * SIGMA_M / SIGMA_A; // 0.1 * 200 / 250 = 0.08
    let expected = psi_hat / 12.0;
    let value = psi[11];
    assert!(
        (value - expected).abs() < 1e-12,
        "lag-11 psi: got {value:.15}, expected {expected:.15} (diff = {:.3e})",
        (value - expected).abs()
    );
}

/// Lags 0 and 1 of hydro 0 at stage 0 combine the AR term and `ψ̂/12`. With
/// uniform σ_m across stages, φ̂_j = φ_j · (σ_m / σ_m) = φ_j.
#[test]
fn lag_0_psi_combines_ar_and_annual() {
    let par = build_par_a_fixture();
    let psi = par.psi_slice(0, 0);

    let psi_hat = PSI * SIGMA_M / SIGMA_A; // 0.08
    for (lag, phi) in [PHI_1, PHI_2].into_iter().enumerate() {
        let expected = phi * (SIGMA_M / SIGMA_M) + psi_hat / 12.0;
        let value = psi[lag];
        assert!(
            (value - expected).abs() < 1e-12,
            "lag-{lag} psi: got {value:.15}, expected {expected:.15} (diff = {:.3e})",
            (value - expected).abs()
        );
    }
}

/// A classical PAR(2) fixture (annual: None) keeps `max_order == 2`, so no
/// lag-11 slot exists.
#[test]
fn classical_par_has_no_lag_11_slot() {
    let par = build_classical_fixture();

    assert_eq!(
        par.max_order(),
        AR_ORDER,
        "classical PAR(2) must keep max_order == {AR_ORDER}; got {}",
        par.max_order()
    );
    assert_eq!(
        par.psi_slice(0, 0).len(),
        AR_ORDER,
        "classical PAR(2) psi_slice(0, 0) must hold {AR_ORDER} coefficients; got {}",
        par.psi_slice(0, 0).len()
    );
    assert_eq!(
        par.effective_lag_count(0),
        AR_ORDER,
        "classical PAR(2) must report {AR_ORDER} effective lags for hydro 0; got {}",
        par.effective_lag_count(0)
    );
    assert!(
        !par.has_annual(0),
        "classical PAR(2) hydro 0 must report no annual component"
    );
}
