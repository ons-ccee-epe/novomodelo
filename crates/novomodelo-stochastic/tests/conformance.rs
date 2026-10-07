//! End-to-end pipeline conformance tests for `cobre-stochastic`, from `System`
//! input through `sample_forward` output, over a shared AR(1) fixture.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cobre_core::{InflowModel, SamplingScheme, SystemBuilder};
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context, sample_forward};

mod common;
use common::{InflowModelSpec, deficit_bus, identity_correlation_model, saa_stage, sized_hydro};

fn make_inflow_model(
    hydro_id: i32,
    stage_id: i32,
    mean_m3s: f64,
    std_m3s: f64,
    ar_coefficients: Vec<f64>,
    residual_std_ratio: f64,
) -> InflowModel {
    common::make_inflow_model(InflowModelSpec {
        hydro_id,
        stage_id,
        mean_m3s,
        std_m3s,
        ar_coefficients,
        residual_std_ratio,
        ..Default::default()
    })
}

fn fixture_with_openings(n_openings: usize) -> cobre_core::System {
    let hydros = vec![sized_hydro(1), sized_hydro(2)];

    // Stage id=-1 is pre-study (excluded from the opening tree); it supplies the
    // lag-1 statistics the PAR coefficient conversion needs.
    let stages = vec![
        saa_stage(0, -1, n_openings),
        saa_stage(1, 0, n_openings),
        saa_stage(2, 1, n_openings),
        saa_stage(3, 2, n_openings),
    ];

    let inflow_models = vec![
        make_inflow_model(1, -1, 100.0, 30.0, vec![], 1.0),
        make_inflow_model(1, 0, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 1, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(1, 2, 100.0, 30.0, vec![0.3], 0.954),
        make_inflow_model(2, -1, 200.0, 40.0, vec![], 1.0),
        make_inflow_model(2, 0, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 1, 200.0, 40.0, vec![0.4], 0.917),
        make_inflow_model(2, 2, 200.0, 40.0, vec![0.4], 0.917),
    ];

    SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation_model(&[1, 2]))
        .build()
        .expect("fixture_with_openings: system build must succeed")
}

fn shared_fixture() -> cobre_core::System {
    fixture_with_openings(5)
}

fn build_shared_ctx(system: &cobre_core::System) -> cobre_stochastic::StochasticContext {
    build_stochastic_context(
        system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context must succeed for the shared fixture")
}

#[test]
fn pipeline_builds_with_correct_dimensions() {
    let system = shared_fixture();
    let ctx = build_shared_ctx(&system);

    assert_eq!(ctx.dim(), 2, "expected dim=2 (two hydros)");
    assert_eq!(ctx.n_stages(), 3, "expected n_stages=3 (study stages only)");
    assert_eq!(ctx.base_seed(), 42, "expected base_seed=42");
}

/// PAR coefficient cache contains the hand-computed reference values.
///
/// For stage 0 (study stage id=0), the lag-1 stage has id=-1 with the same
/// mean and std (stationary series with a same-std pre-study stage).
///
/// Hand-computed expected values follow the formula
/// `psi = psi_star * s_m / s_lag` and
/// `base = mu - psi * mu_lag`.
///
/// Hydro 1 (hydro index 0, sorted by `EntityId`):
///   psi = 0.3 * 30.0 / 30.0 = 0.3, base = 100.0 - 0.3*100.0 = 70.0,
///   sigma = 30.0 * 0.954 = 28.62
///
/// Hydro 2 (hydro index 1, sorted by `EntityId`):
///   psi = 0.4 * 40.0 / 40.0 = 0.4, base = 200.0 - 0.4*200.0 = 120.0,
///   sigma = 40.0 * 0.917 = 36.68
#[test]
fn par_lp_coefficients_match_hand_computed() {
    let system = shared_fixture();
    let ctx = build_shared_ctx(&system);

    let par = ctx.par();
    let tol = 1e-10;

    let expected_base_h1 = 70.0_f64;
    assert!(
        (par.deterministic_base(0, 0) - expected_base_h1).abs() < tol,
        "hydro 1 stage 0 deterministic_base: expected {expected_base_h1}, got {}",
        par.deterministic_base(0, 0)
    );

    let expected_sigma_h1 = 30.0 * 0.954;
    assert!(
        (par.sigma(0, 0) - expected_sigma_h1).abs() < tol,
        "hydro 1 stage 0 sigma: expected {expected_sigma_h1}, got {}",
        par.sigma(0, 0)
    );

    let expected_psi_h1 = 0.3_f64;
    let psi_h1 = par.psi_slice(0, 0);
    assert!(!psi_h1.is_empty(), "psi_slice for AR(1) must not be empty");
    assert!(
        (psi_h1[0] - expected_psi_h1).abs() < tol,
        "hydro 1 stage 0 psi[0]: expected {expected_psi_h1}, got {}",
        psi_h1[0]
    );

    let expected_base_h2 = 120.0_f64;
    assert!(
        (par.deterministic_base(0, 1) - expected_base_h2).abs() < tol,
        "hydro 2 stage 0 deterministic_base: expected {expected_base_h2}, got {}",
        par.deterministic_base(0, 1)
    );

    let expected_sigma_h2 = 40.0 * 0.917;
    assert!(
        (par.sigma(0, 1) - expected_sigma_h2).abs() < tol,
        "hydro 2 stage 0 sigma: expected {expected_sigma_h2}, got {}",
        par.sigma(0, 1)
    );

    let expected_psi_h2 = 0.4_f64;
    let psi_h2 = par.psi_slice(0, 1);
    assert!(!psi_h2.is_empty(), "psi_slice for AR(1) must not be empty");
    assert!(
        (psi_h2[0] - expected_psi_h2).abs() < tol,
        "hydro 2 stage 0 psi[0]: expected {expected_psi_h2}, got {}",
        psi_h2[0]
    );
}

#[test]
fn opening_tree_structure_correct() {
    let system = shared_fixture();
    let ctx = build_shared_ctx(&system);

    let tree = ctx.opening_tree();

    assert_eq!(tree.n_stages(), 3, "expected 3 study stages");
    assert_eq!(tree.n_openings(0), 5, "stage 0 must have 5 openings");
    assert_eq!(tree.n_openings(1), 5, "stage 1 must have 5 openings");
    assert_eq!(tree.n_openings(2), 5, "stage 2 must have 5 openings");
    assert_eq!(tree.dim(), 2, "dim must equal number of hydros");

    for stage in 0..tree.n_stages() {
        for opening in 0..tree.n_openings(stage) {
            for &v in tree.opening(stage, opening) {
                assert!(
                    v.is_finite(),
                    "non-finite value at stage={stage} opening={opening}"
                );
            }
        }
    }
}

#[test]
fn sample_forward_returns_valid_output() {
    let system = shared_fixture();
    let ctx = build_shared_ctx(&system);

    let view = ctx.tree_view();
    let base_seed = ctx.base_seed();

    for iteration in 0_u32..3 {
        for scenario in 0_u32..5 {
            for (stage_idx, stage_domain_id) in [(0usize, 0u32), (1, 1), (2, 2)] {
                let (idx, slice) = sample_forward(
                    &view,
                    base_seed,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                    0,
                    view.n_openings(stage_idx),
                );

                assert!(
                    idx < 5,
                    "index {idx} out of bounds (n_openings=5) for \
                     iteration={iteration} scenario={scenario} stage_idx={stage_idx}"
                );
                assert_eq!(
                    slice.len(),
                    2,
                    "slice length must equal dim=2 for \
                     iteration={iteration} scenario={scenario} stage_idx={stage_idx}"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn opening_tree_marginal_statistics() {
    let n_openings = 500_usize;
    let system = fixture_with_openings(n_openings);

    let ctx = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context must succeed for marginal statistics test");

    let tree = ctx.opening_tree();

    for stage in 0..tree.n_stages() {
        for dim_idx in 0..tree.dim() {
            let values: Vec<f64> = (0..tree.n_openings(stage))
                .map(|o| tree.opening(stage, o)[dim_idx])
                .collect();

            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let std = variance.sqrt();

            assert!(
                mean.abs() < 0.15,
                "stage={stage} dim={dim_idx}: mean {mean:.4} too far from 0 \
                 (expected N(0,1), |mean| < 0.15)"
            );
            assert!(
                (std - 1.0).abs() < 0.15,
                "stage={stage} dim={dim_idx}: std {std:.4} too far from 1 \
                 (expected N(0,1), |std - 1| < 0.15)"
            );
        }
    }
}
