//! Shared fixtures for the `novomodelo-stochastic` integration-test suite:
//! correlation-model builders, entity-order dimensions, and preset
//! bus/hydro/stage constructors built on top of
//! `novomodelo_core::test_support` and `novomodelo_stochastic::test_support`.

#![allow(dead_code, unused_imports)]
// Each `tests/*.rs` binary compiles this module separately and uses only the
// subset of items it needs, so an item or re-export unused by one binary is
// not dead code.

use std::collections::BTreeMap;

pub use novomodelo_core::test_support::{
    InflowModelSpec, StageSpec, make_inflow_model, make_stage, norm_cdf, single_block,
};
use novomodelo_core::{
    EntityId, SamplingScheme, Stage,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
};
pub use novomodelo_stochastic::test_support::{
    build_test_system, deficit_bus, method_stage, saa_stage, sized_hydro,
};
use novomodelo_stochastic::{
    ClassDimensions, ClassSchemes, ForwardNoiseTables, ForwardSampler, ForwardSamplerConfig,
    OpeningTreeInputs, StochasticContext, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation,
};

/// The shared [`CorrelationModel`]: one `"default"` profile with a single group `"g1"` of
/// inflow entities whose off-diagonal correlation is `rho`.
#[must_use]
pub fn correlated_correlation_model(entity_ids: &[i32], rho: f64) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn inflow_entity_order_and_dims(entity_ids: &[i32]) -> (Vec<EntityId>, ClassDimensions) {
    (
        entity_ids.iter().map(|&id| EntityId(id)).collect(),
        ClassDimensions {
            n_hydros: entity_ids.len(),
            n_load_buses: 0,
            n_ncs: 0,
        },
    )
}

pub fn identity_correlation_model(entity_ids: &[i32]) -> CorrelationModel {
    correlated_correlation_model(entity_ids, 0.0)
}

pub fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(&identity_correlation_model(entity_ids), &entity_order, dims)
        .unwrap()
}

pub fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
    let (entity_order, dims) = inflow_entity_order_and_dims(entity_ids);
    DecomposedCorrelation::build(
        &correlated_correlation_model(entity_ids, rho),
        &entity_order,
        dims,
    )
    .unwrap()
}

/// Build a [`ForwardSamplerConfig`] over `stages` and `ctx`, with `scheme`
/// set on every noise class.
#[must_use]
pub fn make_sampler_config<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> ForwardSamplerConfig<'a> {
    ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(scheme),
            load: Some(scheme),
            ncs: Some(scheme),
        },
        ctx,
        forward_seed: ctx.forward_seed(),
        stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
    }
}

/// The in-sample [`StochasticContext`] for `system`, base seed `42`.
#[must_use]
pub fn build_test_ctx(
    system: &novomodelo_core::System,
    forward_seed: Option<u64>,
) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

/// The study stages of `system` (excludes the pre-study `id < 0` stage).
#[must_use]
pub fn stages_from_system(system: &novomodelo_core::System) -> Vec<Stage> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect()
}

/// Rebuild `sampler`'s noise tables for one `(iteration, total, groups)`
/// triple — the `SampleRequest.tables` every `sample()` call in
/// `forward_sampler.rs` and `forward_sampler_golden.rs` needs.
#[must_use]
pub fn tables_for(
    sampler: &ForwardSampler<'_>,
    iteration: u32,
    total: u32,
    groups: &[u32],
) -> ForwardNoiseTables {
    let mut tables = ForwardNoiseTables::default();
    sampler
        .rebuild_noise_tables(iteration, total, groups, &mut tables)
        .expect("test fixtures never exceed the Sobol dimension cap");
    tables
}
