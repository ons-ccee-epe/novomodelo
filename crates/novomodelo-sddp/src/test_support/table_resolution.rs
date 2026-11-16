//! Resolver-backed `ResolvedBounds` / `ResolvedPenalties` for sddp tests: the
//! tables come from the entities through the production resolvers instead of
//! being hand-built per site, so a new resolved field needs no per-test edit.

use novomodelo_core::{ResolvedBounds, ResolvedPenalties, Stage, System, SystemBuilder};
use novomodelo_io::StageIdResolver;
use novomodelo_io::resolution::{
    BoundsEntitySlices, BoundsOverrides, PenaltiesEntitySlices, PenaltiesOverrides, resolve_bounds,
    resolve_penalties,
};

/// Bounds and penalties for `system` with no override rows, rows aligned to the
/// `System`'s canonical entity positions.
///
/// `k_max` is explicit because the pipeline's lead-stage rule is private; pass
/// `0` unless the replaced hand-built table set it. The hydro unit-group overlay
/// stays as [`resolve_bounds`] returns it.
#[must_use]
pub fn resolved_tables(system: &System, k_max: usize) -> (ResolvedBounds, ResolvedPenalties) {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let stage_index = stage_resolver.index_map();
    let n_stages = study_stages.len();
    let blocks_per_stage: Vec<usize> = study_stages.iter().map(|s| s.blocks.len()).collect();

    let bounds = resolve_bounds(
        &BoundsEntitySlices {
            hydros: system.hydros(),
            thermals: system.thermals(),
            lines: system.lines(),
            pumping_stations: system.pumping_stations(),
            contracts: system.contracts(),
        },
        n_stages,
        k_max,
        stage_index,
        &BoundsOverrides {
            hydro: &[],
            thermal: &[],
            line: &[],
            pumping: &[],
            contract: &[],
        },
        &blocks_per_stage,
    );
    let penalties = resolve_penalties(
        &PenaltiesEntitySlices {
            hydros: system.hydros(),
            buses: system.buses(),
            lines: system.lines(),
            ncs_sources: system.non_controllable_sources(),
        },
        n_stages,
        stage_index,
        &PenaltiesOverrides {
            hydro: &[],
            bus: &[],
            line: &[],
            ncs: &[],
        },
    );
    (bounds, penalties)
}

/// Builds `make()` twice: the first `System` fixes the canonical entity order the
/// tables are resolved against, `adjust` patches them by canonical index, and the
/// second `System` carries them.
///
/// # Panics
///
/// If either build fails validation, or if the two builds hold different entity
/// or stage id sequences (a non-deterministic `make`).
#[expect(
    clippy::expect_used,
    reason = "a fixture that fails validation is a fixture bug, not a runtime error to propagate"
)]
#[must_use]
pub fn build_with_resolved_tables(
    make: impl Fn() -> SystemBuilder,
    k_max: usize,
    adjust: impl FnOnce(&System, &mut ResolvedBounds, &mut ResolvedPenalties),
) -> System {
    let pass1 = make()
        .build()
        .expect("build_with_resolved_tables: pass 1 failed validation");
    let (mut bounds, mut penalties) = resolved_tables(&pass1, k_max);
    adjust(&pass1, &mut bounds, &mut penalties);
    let system = make()
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("build_with_resolved_tables: pass 2 failed validation");
    for ((kind, first), (_, second)) in id_sequences(&pass1).into_iter().zip(id_sequences(&system))
    {
        assert_eq!(
            first, second,
            "build_with_resolved_tables: {kind} ids differ between the two builds"
        );
    }
    system
}

fn id_sequences(system: &System) -> [(&'static str, Vec<i32>); 8] {
    [
        ("bus", system.buses().iter().map(|e| e.id.0).collect()),
        ("line", system.lines().iter().map(|e| e.id.0).collect()),
        ("hydro", system.hydros().iter().map(|e| e.id.0).collect()),
        (
            "thermal",
            system.thermals().iter().map(|e| e.id.0).collect(),
        ),
        (
            "pumping station",
            system.pumping_stations().iter().map(|e| e.id.0).collect(),
        ),
        (
            "contract",
            system.contracts().iter().map(|e| e.id.0).collect(),
        ),
        (
            "non-controllable source",
            system
                .non_controllable_sources()
                .iter()
                .map(|e| e.id.0)
                .collect(),
        ),
        ("stage", system.stages().iter().map(|s| s.id).collect()),
    ]
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use novomodelo_core::{EntityId, Hydro, Stage, SystemBuilder};
    use novomodelo_io::resolution::{
        BoundsEntitySlices, BoundsOverrides, PenaltiesEntitySlices, PenaltiesOverrides,
        resolve_bounds, resolve_penalties,
    };

    use super::{build_with_resolved_tables, resolved_tables};
    use crate::test_support::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    fn hydro(id: i32, max_storage_hm3: f64) -> Hydro {
        make_hydro(
            EntityId(id),
            HydroSpec {
                max_storage_hm3,
                ..HydroSpec::default()
            },
        )
    }

    fn one_bus_two_stages() -> SystemBuilder {
        SystemBuilder::new()
            .buses(vec![make_bus(EntityId(0), BusSpec::default())])
            .stages(vec![
                make_stage(0, StageSpec::default()),
                make_stage(1, StageSpec::default()),
            ])
    }

    #[test]
    fn rows_follow_canonical_order_not_input_order() {
        let system = build_with_resolved_tables(
            || one_bus_two_stages().hydros(vec![hydro(2, 300.0), hydro(1, 100.0)]),
            0,
            |_, _, _| {},
        );

        assert_eq!(system.hydros()[0].id, EntityId(1));
        for stage in 0..2 {
            assert_eq!(
                system.bounds().hydro_bounds(0, stage).max_storage_hm3,
                100.0
            );
            assert_eq!(
                system.bounds().hydro_bounds(1, stage).max_storage_hm3,
                300.0
            );
        }
    }

    #[test]
    fn resolved_tables_equals_pipeline_style_resolution() {
        let system = build_with_resolved_tables(
            || {
                SystemBuilder::new()
                    .buses(vec![make_bus(EntityId(0), BusSpec::default())])
                    .hydros(vec![hydro(2, 300.0), hydro(1, 100.0)])
                    .thermals(vec![make_thermal(
                        EntityId(1),
                        ThermalSpec {
                            cost_per_mwh: 45.0,
                            ..ThermalSpec::default()
                        },
                    )])
                    .stages(vec![
                        Stage {
                            id: -1,
                            ..make_stage(0, StageSpec::default())
                        },
                        make_stage(0, StageSpec::default()),
                        make_stage(1, StageSpec::default()),
                    ])
            },
            0,
            |_, _, _| {},
        );

        let stage_index = HashMap::from([(0, 0), (1, 1)]);
        let expected_bounds = resolve_bounds(
            &BoundsEntitySlices {
                hydros: system.hydros(),
                thermals: system.thermals(),
                lines: system.lines(),
                pumping_stations: system.pumping_stations(),
                contracts: system.contracts(),
            },
            2,
            0,
            &stage_index,
            &BoundsOverrides {
                hydro: &[],
                thermal: &[],
                line: &[],
                pumping: &[],
                contract: &[],
            },
            &[1, 1],
        );
        let expected_penalties = resolve_penalties(
            &PenaltiesEntitySlices {
                hydros: system.hydros(),
                buses: system.buses(),
                lines: system.lines(),
                ncs_sources: system.non_controllable_sources(),
            },
            2,
            &stage_index,
            &PenaltiesOverrides {
                hydro: &[],
                bus: &[],
                line: &[],
                ncs: &[],
            },
        );

        assert_eq!(
            resolved_tables(&system, 0),
            (expected_bounds, expected_penalties)
        );
    }

    #[test]
    fn adjust_patch_reaches_the_built_system() {
        let system = build_with_resolved_tables(
            || {
                one_bus_two_stages().thermals(vec![make_thermal(
                    EntityId(1),
                    ThermalSpec {
                        cost_per_mwh: 45.0,
                        ..ThermalSpec::default()
                    },
                )])
            },
            0,
            |_, bounds, _| bounds.thermal_block_base_mut(0, 1).cost_per_mwh = 123.0,
        );

        assert_eq!(system.bounds().thermal_block_base(0, 1).cost_per_mwh, 123.0);
        assert_eq!(system.bounds().thermal_block_base(0, 0).cost_per_mwh, 45.0);
    }

    #[test]
    #[should_panic(expected = "hydro")]
    fn nondeterministic_factory_panics() {
        let calls = Cell::new(0);
        let _ = build_with_resolved_tables(
            || {
                let hydro_id = if calls.get() == 0 { 1 } else { 7 };
                calls.set(calls.get() + 1);
                one_bus_two_stages().hydros(vec![hydro(hydro_id, 100.0)])
            },
            0,
            |_, _, _| {},
        );
    }
}
