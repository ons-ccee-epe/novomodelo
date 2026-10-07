//! Reproducibility and invariance integration tests for `cobre-stochastic`:
//! deterministic reproducibility, declaration-order invariance (the pipeline
//! sorts entities by `EntityId` internally), seed sensitivity, and infrastructure
//! genericity (the crate source carries zero algorithm-specific references).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_core::{Hydro, SystemBuilder, scenario::SamplingScheme};
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context, sample_forward};

mod common;
use common::{
    default_inflow_model, deficit_bus, identity_correlation_model, saa_stage, sized_hydro,
};

fn build_fixture(hydros: Vec<Hydro>, base_seed: u64) -> cobre_stochastic::StochasticContext {
    let stages = vec![saa_stage(0, 0, 5), saa_stage(1, 1, 5), saa_stage(2, 2, 5)];
    let inflow_models = vec![
        default_inflow_model(1, 0),
        default_inflow_model(1, 1),
        default_inflow_model(1, 2),
        default_inflow_model(2, 0),
        default_inflow_model(2, 1),
        default_inflow_model(2, 2),
    ];

    let system = SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation_model(&[1, 2]))
        .build()
        .expect("build_fixture: system build must succeed");

    build_stochastic_context(
        &system,
        base_seed,
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
    .expect("build_fixture: build_stochastic_context must succeed")
}

#[test]
fn deterministic_reproducibility() {
    let hydros = vec![sized_hydro(1), sized_hydro(2)];

    let ctx_a = build_fixture(hydros.clone(), 42);
    let ctx_b = build_fixture(hydros, 42);

    let tree_a = ctx_a.opening_tree();
    let tree_b = ctx_b.opening_tree();

    assert_eq!(
        tree_a.n_stages(),
        tree_b.n_stages(),
        "n_stages must match between two identical builds"
    );

    for stage in 0..tree_a.n_stages() {
        assert_eq!(
            tree_a.n_openings(stage),
            tree_b.n_openings(stage),
            "n_openings at stage={stage} must match"
        );
        for opening in 0..tree_a.n_openings(stage) {
            assert_eq!(
                tree_a.opening(stage, opening),
                tree_b.opening(stage, opening),
                "opening values at stage={stage} opening={opening} must be bit-identical"
            );
        }
    }

    let view_a = ctx_a.tree_view();
    let view_b = ctx_b.tree_view();
    let seed_a = ctx_a.base_seed();
    let seed_b = ctx_b.base_seed();

    for iteration in 0_u32..3 {
        for scenario in 0_u32..5 {
            for (stage_idx, stage_domain_id) in [(0usize, 0u32), (1, 1), (2, 2)] {
                let (idx_a, slice_a) = sample_forward(
                    &view_a,
                    seed_a,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                    0,
                    view_a.n_openings(stage_idx),
                );
                let (idx_b, slice_b) = sample_forward(
                    &view_b,
                    seed_b,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                    0,
                    view_b.n_openings(stage_idx),
                );

                assert_eq!(
                    idx_a, idx_b,
                    "sample_forward index must be identical for iteration={iteration} \
                     scenario={scenario} stage_idx={stage_idx}"
                );
                assert_eq!(
                    slice_a, slice_b,
                    "sample_forward slice must be bit-identical for iteration={iteration} \
                     scenario={scenario} stage_idx={stage_idx}"
                );
            }
        }
    }
}

/// Reversing the hydro entity list in the input produces an identical opening
/// tree, because `SystemBuilder` sorts hydros by `EntityId` internally.
#[test]
fn declaration_order_invariance() {
    let hydros_forward = vec![sized_hydro(1), sized_hydro(2)];
    let hydros_reversed = vec![sized_hydro(2), sized_hydro(1)];

    let ctx_forward = build_fixture(hydros_forward, 42);
    let ctx_reversed = build_fixture(hydros_reversed, 42);

    let tree_fwd = ctx_forward.opening_tree();
    let tree_rev = ctx_reversed.opening_tree();

    assert_eq!(
        tree_fwd.n_stages(),
        tree_rev.n_stages(),
        "n_stages must match regardless of hydro insertion order"
    );

    for stage in 0..tree_fwd.n_stages() {
        assert_eq!(
            tree_fwd.n_openings(stage),
            tree_rev.n_openings(stage),
            "n_openings at stage={stage} must match regardless of hydro insertion order"
        );
        for opening in 0..tree_fwd.n_openings(stage) {
            assert_eq!(
                tree_fwd.opening(stage, opening),
                tree_rev.opening(stage, opening),
                "opening values at stage={stage} opening={opening} must be identical \
                 regardless of hydro insertion order"
            );
        }
    }
}

#[test]
fn seed_sensitivity() {
    let hydros = vec![sized_hydro(1), sized_hydro(2)];

    let ctx_42 = build_fixture(hydros.clone(), 42);
    let ctx_99 = build_fixture(hydros, 99);

    let tree_42 = ctx_42.opening_tree();
    let tree_99 = ctx_99.opening_tree();

    let any_differ = (0..tree_42.n_stages()).any(|stage| {
        (0..tree_42.n_openings(stage)).any(|opening| {
            tree_42
                .opening(stage, opening)
                .iter()
                .zip(tree_99.opening(stage, opening).iter())
                .any(|(a, b)| a != b)
        })
    });

    assert!(
        any_differ,
        "expected at least one differing noise value between seed=42 and seed=99 trees"
    );
}

#[test]
fn infrastructure_genericity_no_sddp_references() {
    use std::path::Path;
    use std::process::Command;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");
    let src_path = Path::new(manifest_dir).join("src");

    let output = Command::new("grep")
        .args(["-riE", "sddp"])
        .arg(&src_path)
        .current_dir(&workspace_root)
        .output()
        .expect("infrastructure_genericity: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "grep found algorithm-specific references in cobre-stochastic/src/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
