//! Integration tests for the [`ForwardSampler`] abstraction.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use cobre_core::{NoiseMethod, SamplingScheme};
use cobre_stochastic::{SampleRequest, StochasticError, build_forward_sampler, sample_forward};

mod common;
use common::{
    build_test_ctx, build_test_system, correlated_correlation_model, identity_correlation_model,
    make_sampler_config, stages_from_system, tables_for,
};

#[test]
fn insample_dispatch_returns_tree_slice_of_correct_dim() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);
    let sampler =
        build_forward_sampler(make_sampler_config(SamplingScheme::InSample, &ctx, &stages))
            .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, 5, &[]);

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: ctx.tree_view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let slice = result.as_slice();
    assert_eq!(
        slice.len(),
        dim,
        "noise slice length {} != dim {}",
        slice.len(),
        dim
    );
}

#[test]
fn insample_copy_equivalence_matches_direct_call() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);
    let sampler =
        build_forward_sampler(make_sampler_config(SamplingScheme::InSample, &ctx, &stages))
            .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, 5, &[]);

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: ctx.tree_view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let tree_view = ctx.tree_view();
    let (_direct_idx, direct_slice) = sample_forward(
        &tree_view,
        ctx.base_seed(),
        0,
        0,
        0,
        0,
        0,
        tree_view.n_openings(0),
    );

    assert_eq!(
        result.as_slice(),
        direct_slice,
        "ForwardSampler::InSample and direct sample_forward must return bitwise-identical slices"
    );
}

#[test]
fn out_of_sample_dispatch_returns_fresh_noise_of_correct_dim() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, 5, &[]);

    let result = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let slice = result.as_slice();
    assert_eq!(
        slice.len(),
        dim,
        "fresh noise slice length {} != dim {}",
        slice.len(),
        dim
    );
}

#[test]
fn out_of_sample_is_deterministic() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_a = vec![0.0f64; dim];
    let mut buf_b = vec![0.0f64; dim];
    let mut corr_a = vec![0.0f64; 2 * dim];
    let mut corr_b = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, 5, &[]);

    let a = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_a,
            corr_scratch: &mut corr_a,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let b = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_b,
            corr_scratch: &mut corr_b,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    assert_eq!(
        a.as_slice(),
        b.as_slice(),
        "identical OutOfSample calls must produce bitwise-identical noise"
    );
}

#[test]
fn out_of_sample_scenario_changes_noise() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_0 = vec![0.0f64; dim];
    let mut buf_1 = vec![0.0f64; dim];
    let mut corr_0 = vec![0.0f64; 2 * dim];
    let mut corr_1 = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, 5, &[]);

    let result_0 = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_0,
            corr_scratch: &mut corr_0,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let result_1 = sampler
        .sample(SampleRequest {
            iteration: 0,
            scenario: 1,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_1,
            corr_scratch: &mut corr_1,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let any_differ = result_0
        .as_slice()
        .iter()
        .zip(result_1.as_slice())
        .any(|(a, b)| a != b);

    assert!(
        any_differ,
        "noise for scenario=0 and scenario=1 must differ in at least one element"
    );
}

#[test]
fn out_of_sample_noise_is_finite() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 100;

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, total_scenarios, &[]);

    for scenario in 0..total_scenarios {
        let result = sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        for (i, &v) in result.as_slice().iter().enumerate() {
            assert!(
                v.is_finite(),
                "scenario={scenario} element[{i}] is not finite: {v}"
            );
        }
    }
}

#[test]
fn out_of_sample_correlation_matches_target() {
    let rho = 0.8_f64;
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        correlated_correlation_model(&[1, 2], rho),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    assert_eq!(dim, 2, "expected dim=2 for 2 hydros");

    let n_scenarios: u32 = 2000;
    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_scenarios as usize);
    let tables = tables_for(&sampler, 0, n_scenarios, &[]);

    for scenario in 0..n_scenarios {
        let result = sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios: n_scenarios,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        let s = result.as_slice();
        pairs.push((s[0], s[1]));
    }

    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

    let cov_xy = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / (n - 1.0);
    let var_x = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / (n - 1.0);
    let var_y = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / (n - 1.0);

    let sample_corr = cov_xy / (var_x.sqrt() * var_y.sqrt());

    assert!(
        (sample_corr - rho).abs() < 0.15,
        "sample correlation {sample_corr:.4} not within 0.15 of target {rho}"
    );
}

#[test]
fn out_of_sample_per_stage_method_mixing() {
    let system = build_test_system(
        &[NoiseMethod::Lhs, NoiseMethod::Saa, NoiseMethod::QmcHalton],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 10;

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 0, total_scenarios, &[]);

    for stage_idx in 0..3_usize {
        let stage_id = stage_idx as u32;
        for scenario in 0..total_scenarios {
            let result = sampler
                .sample(SampleRequest {
                    iteration: 0,
                    scenario,
                    stage: stage_id,
                    stage_idx,
                    noise_buf: &mut noise_buf,
                    corr_scratch: &mut corr_scratch,
                    total_scenarios,
                    noise_group_id: stage_id,
                    node_opening_offset: 0,
                    node_opening_len: 0,
                    pinned_scenario: None,
                    tables: &tables,
                })
                .unwrap();

            let slice = result.as_slice();
            assert_eq!(
                slice.len(),
                dim,
                "stage_idx={stage_idx} scenario={scenario}: expected dim={dim}, got {}",
                slice.len()
            );
            for (i, &v) in slice.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "stage_idx={stage_idx} scenario={scenario} element[{i}] is not finite: {v}"
                );
            }
        }
    }
}

#[test]
fn factory_rejects_out_of_sample_without_seed() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);

    let result = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("out_of_sample"),
                "expected scheme to contain 'out_of_sample', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn factory_rejects_historical() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);

    let result = build_forward_sampler(make_sampler_config(
        SamplingScheme::Historical,
        &ctx,
        &stages,
    ));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("historical"),
                "expected scheme to contain 'historical', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn factory_rejects_external() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, None);
    let stages = stages_from_system(&system);

    let result =
        build_forward_sampler(make_sampler_config(SamplingScheme::External, &ctx, &stages));

    match result {
        Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
            assert!(
                scheme.contains("external"),
                "expected scheme to contain 'external', got: {scheme}"
            );
        }
        other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
    }
}

#[test]
fn out_of_sample_resume_invariance() {
    let system = build_test_system(
        &[NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();

    let mut buf_first = vec![0.0f64; dim];
    let mut buf_resume = vec![0.0f64; dim];
    let mut corr_first = vec![0.0f64; 2 * dim];
    let mut corr_resume = vec![0.0f64; 2 * dim];
    let tables = tables_for(&sampler, 5, 5, &[]);

    let first = sampler
        .sample(SampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_first,
            corr_scratch: &mut corr_first,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    let resumed = sampler
        .sample(SampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut buf_resume,
            corr_scratch: &mut corr_resume,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        })
        .unwrap();

    assert_eq!(
        first.as_slice(),
        resumed.as_slice(),
        "OutOfSample noise must be identical for a resumed call with the same (iteration, scenario, stage)"
    );
}
