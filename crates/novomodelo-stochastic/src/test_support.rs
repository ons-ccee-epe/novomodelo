//! Test-only fixture builders for this crate's `OpeningTree` shape, the
//! forward-sampler system preset, and a WARN-message recorder, shared across
//! this crate's unit and integration tests and reachable from downstream crates
//! via the `test-support` feature.
//! Compiles only under `cfg(test)` or that feature; must never be enabled in a
//! production build.

use std::sync::{Arc, Mutex, Once};

use novomodelo_core::{
    Bus, CorrelationModel, DeficitSegment, Hydro, NoiseMethod, ScenarioSourceConfig, Stage, System,
    SystemBuilder,
    test_support::{
        BusSpec, HydroSpec, InflowModelSpec, MirrorUnitGroup, StageSpec, date, make_bus,
        make_hydro, make_inflow_model, make_stage, single_block,
    },
};
use tracing::subscriber::Interest;
use tracing::{Event, Level, Metadata, Subscriber, span};

use crate::OpeningTree;

/// The shared [`Bus`]: one uncapped deficit segment at `1000.0`/`MWh`, zero excess cost.
#[must_use]
pub fn deficit_bus(id: i32) -> Bus {
    make_bus(BusSpec {
        id,
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        ..Default::default()
    })
}

/// The shared [`Hydro`]: `100.0` hm³ storage, `100.0` m³/s turbined capacity, `100.0` `MW`.
#[must_use]
pub fn sized_hydro(id: i32) -> Hydro {
    make_hydro(HydroSpec {
        id,
        name: format!("H{id}"),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
        ..Default::default()
    })
}

/// The shared unmirrored [`Hydro`]: [`sized_hydro`]'s capacities, in service from 2020-01-01, with no mirror unit group.
#[must_use]
pub fn unmirrored_hydro(id: i32) -> Hydro {
    make_hydro(HydroSpec {
        id,
        name: format!("H{id}"),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
        operational_start_date: date(2020, 1, 1),
        mirror_unit_group: MirrorUnitGroup::None,
        ..Default::default()
    })
}

/// The shared SAA [`Stage`]: one `"SINGLE"` 744-hour block, season `0`.
#[must_use]
pub fn saa_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
    method_stage(index, id, branching_factor, NoiseMethod::Saa)
}

/// The forward-sampler stage preset: one `"SINGLE"` 744-hour block, season 0,
/// with the per-stage noise method supplied by the caller.
#[must_use]
pub fn method_stage(
    index: usize,
    id: i32,
    branching_factor: usize,
    noise_method: NoiseMethod,
) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: single_block("SINGLE", 744.0),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method,
        },
        ..Default::default()
    })
}

/// The two-hydro, three-stage forward-sampler fixture system for `methods`
/// (one per stage) and `correlation`.
///
/// # Panics
///
/// If `methods.len() != 3`, or if `SystemBuilder` rejects the preset system
/// built with `correlation`.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: the preset's own inputs are valid, so a build failure means the
// caller's `correlation` is wrong, which is a test-authoring bug.
pub fn build_test_system(methods: &[NoiseMethod], correlation: CorrelationModel) -> System {
    assert_eq!(methods.len(), 3, "must supply exactly 3 per-stage methods");
    let hydros = vec![sized_hydro(1), sized_hydro(2)];
    let stages = vec![
        method_stage(0, 0, 5, methods[0]),
        method_stage(1, 1, 5, methods[1]),
        method_stage(2, 2, 5, methods[2]),
    ];
    let inflow_models: Vec<_> = (1..=2)
        .flat_map(|hydro_id| {
            (0..3).map(move |stage_id| {
                make_inflow_model(InflowModelSpec {
                    hydro_id,
                    stage_id,
                    ..Default::default()
                })
            })
        })
        .collect();
    SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .expect("the preset system passes SystemBuilder validation")
}

/// A uniformly-valued [`OpeningTree`] with `n_stages` stages, each holding
/// `openings` openings of dimension `dim`: values `0.0, 1.0, 2.0, ..` filled
/// row-major over (stage, opening, dim).
///
/// # Panics
///
/// Never for the stage/opening/dim triples the fixtures below pass.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: every call site below passes a triple whose product fits in
// `u32`, so `u32::try_from` cannot return `Err`.
pub fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
    let total = n_stages * openings * dim;
    let data: Vec<f64> = (0_u32..u32::try_from(total).expect("total fits in u32"))
        .map(f64::from)
        .collect();
    OpeningTree::from_parts(data, vec![openings; n_stages], dim)
}

/// Installs, once per test binary, a global subscriber that records nothing and
/// answers [`Interest::sometimes`] for every callsite, so a scoped subscriber
/// sees every event it enables.
///
/// Without it, tracing caches a callsite's interest from the thread that first
/// reaches it while only one subscriber is registered: a concurrent test with no
/// subscriber caches the callsite as "never" and a scoped recorder on another
/// thread then misses its events.
pub fn keep_callsite_interest_dynamic() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // Err only when the binary already set a global subscriber; that one stays.
        let _ = tracing::subscriber::set_global_default(DynamicInterest);
    });
}

struct DynamicInterest;

impl Subscriber for DynamicInterest {
    fn register_callsite(&self, _metadata: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        false
    }

    fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    fn event(&self, _event: &Event<'_>) {}

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

/// Records the `message` of every WARN event.
pub struct WarnRecorder {
    messages: Arc<Mutex<Vec<String>>>,
}

impl WarnRecorder {
    /// A recorder and the shared list it appends to.
    #[must_use]
    pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        keep_callsite_interest_dynamic();
        let messages = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                messages: Arc::clone(&messages),
            },
            messages,
        )
    }
}

impl Subscriber for WarnRecorder {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= Level::WARN
    }

    fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    #[allow(clippy::expect_used)]
    // Rationale: only a panic in another test thread poisons the list, and that test has failed.
    fn event(&self, event: &Event<'_>) {
        if *event.metadata().level() == Level::WARN {
            struct MessageVisitor(String);
            impl tracing::field::Visit for MessageVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }

                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "message" {
                        self.0 = value.to_string();
                    }
                }
            }
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.messages
                .lock()
                .expect("warn message list is never poisoned")
                .push(visitor.0);
        }
    }

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

#[cfg(test)]
mod tests {
    use novomodelo_core::{CorrelationModel, EntityId, NoiseMethod};

    use super::build_test_system;

    #[test]
    fn build_test_system_preset_shape() {
        let system = build_test_system(&[NoiseMethod::Saa; 3], CorrelationModel::default());

        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
        assert_eq!(hydro_ids, [EntityId(1), EntityId(2)]);

        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
        assert_eq!(study_stages.len(), 3);
        assert!(
            study_stages
                .iter()
                .all(|s| s.scenario_config.branching_factor == 5)
        );

        assert_eq!(system.inflow_models().len(), 6);
    }
}
