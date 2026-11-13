use std::collections::{BTreeMap, HashMap};

use chrono::{NaiveDate, TimeDelta};
use novomodelo_core::test_support::{
    BusSpec, HydroSpec, StageSpec, date, make_bus, make_hydro, make_stage,
};
use novomodelo_core::{
    Block, BlockMode, DeficitSegment, Hydro, PostStudyStage, PostStudyStages, Stage, System,
    SystemBuilder, WaterBucketStages,
};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

use super::{TransitBucketTopology, build_transit_bucket_topology};
use crate::lead_time::SpreadResolution;
use crate::time_value::{DeliveryCalendar, WaterDeliveryBasis};

#[derive(Debug, Clone)]
struct StageShape {
    hours: f64,
    n_blocks: u8,
    block_mode: BlockMode,
}

#[derive(Debug, Clone)]
struct Case {
    study: Vec<StageShape>,
    travel_times: Vec<f64>,
    water: Vec<f64>,
    boundary_present: bool,
}

#[derive(Debug, PartialEq)]
struct SpreadBits {
    stage_reach: usize,
    stage_weights: Vec<u64>,
    block_deposits: Vec<Vec<u64>>,
    within_stage_routing: Vec<Vec<u64>>,
    arrival_density: Vec<Vec<u64>>,
}

impl SpreadBits {
    fn of(spread: &SpreadResolution) -> Self {
        Self {
            stage_reach: spread.stage_reach,
            stage_weights: bits(&spread.stage_weights),
            block_deposits: spread.block_deposits.iter().map(|row| bits(row)).collect(),
            within_stage_routing: spread
                .within_stage_routing
                .iter()
                .map(|row| bits(row))
                .collect(),
            arrival_density: spread.arrival_density.iter().map(|row| bits(row)).collect(),
        }
    }
}

fn duration_hours() -> impl Strategy<Value = f64> {
    prop::sample::select(vec![168.0, 720.0, 744.0])
}

fn stage_shape() -> impl Strategy<Value = StageShape> {
    (
        duration_hours(),
        1..=3_u8,
        prop_oneof![Just(BlockMode::Parallel), Just(BlockMode::Chronological)],
    )
        .prop_map(|(hours, n_blocks, block_mode)| StageShape {
            hours,
            n_blocks,
            block_mode,
        })
}

fn extend_until_covering(seed: &[f64], travel_time_hours: f64) -> Vec<f64> {
    let mut water = seed.to_vec();
    while water.iter().sum::<f64>() < travel_time_hours {
        water.push(seed[water.len() % seed.len()]);
    }
    water
}

fn covering_case() -> impl Strategy<Value = Case> {
    (
        prop::collection::vec(stage_shape(), 1..=4),
        prop::collection::vec(1.0..=2000.0_f64, 1..=2),
        prop::collection::vec(duration_hours(), 1..=4),
        any::<bool>(),
    )
        .prop_map(|(study, travel_times, water_seed, boundary_present)| {
            let deepest = travel_times.iter().copied().fold(0.0, f64::max);
            Case {
                water: extend_until_covering(&water_seed, deepest),
                study,
                travel_times,
                boundary_present,
            }
        })
}

fn after(start: NaiveDate, hours: f64) -> NaiveDate {
    start + TimeDelta::days((hours / 24.0) as i64)
}

fn study_stages(shapes: &[StageShape]) -> Vec<Stage> {
    let mut start = date(2024, 1, 1);
    shapes
        .iter()
        .enumerate()
        .map(|(id, shape)| {
            let end_date = after(start, shape.hours);
            let block_hours = shape.hours / f64::from(shape.n_blocks);
            let blocks = (0..usize::from(shape.n_blocks))
                .map(|index| Block {
                    index,
                    name: format!("B{index}"),
                    duration_hours: block_hours,
                })
                .collect();
            let stage = Stage {
                block_mode: shape.block_mode,
                ..make_stage(StageSpec {
                    id: i32::try_from(id).expect("at most four study stages"),
                    start_date: start,
                    end_date,
                    blocks,
                    ..StageSpec::default()
                })
            };
            start = end_date;
            stage
        })
        .collect()
}

fn water_stages(study_end: NaiveDate, hours: &[f64]) -> Vec<PostStudyStage> {
    let mut start_date = study_end;
    hours
        .iter()
        .map(|&duration_hours| {
            let stage = PostStudyStage {
                start_date,
                duration_hours,
            };
            start_date = after(start_date, duration_hours);
            stage
        })
        .collect()
}

fn system(case: &Case, declared: bool) -> System {
    let stages = study_stages(&case.study);
    let study_end = stages.last().expect("at least one study stage").end_date;
    let water = water_stages(study_end, &case.water);
    let (thermal_stages, water_bucket_stages) = if declared {
        (Vec::new(), WaterBucketStages::Declared(water))
    } else {
        (water, WaterBucketStages::FromThermalStages)
    };

    let hydro = |id: i32, downstream_id: Option<i32>| {
        make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            bus_id: 1,
            downstream_id,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            max_storage_hm3: 100.0,
            ..HydroSpec::default()
        })
    };
    let mut hydros = vec![hydro(1, None)];
    hydros.extend(
        case.travel_times
            .iter()
            .zip(2..)
            .map(|(&travel_time_hours, id)| Hydro {
                travel_time_hours: Some(travel_time_hours),
                ..hydro(id, Some(1))
            }),
    );

    SystemBuilder::new()
        .buses(vec![make_bus(BusSpec {
            id: 1,
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            ..BusSpec::default()
        })])
        .hydros(hydros)
        .stages(stages)
        .post_study_stages(Some(PostStudyStages {
            thermal_stages,
            thermal_bounds: Vec::new(),
            water_bucket_stages,
        }))
        .build()
        .expect("valid system")
}

fn topology(system: &System, boundary_present: bool) -> TransitBucketTopology {
    let basis = WaterDeliveryBasis::of(system);
    let calendar = DeliveryCalendar::water(system, basis);
    build_transit_bucket_topology(system, &calendar, basis, boundary_present)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn arc_rows<T, U>(
    arcs: &HashMap<usize, Vec<T>>,
    project: impl Fn(&T) -> U,
) -> BTreeMap<usize, Vec<U>> {
    arcs.iter()
        .map(|(&arc, rows)| (arc, rows.iter().map(&project).collect()))
        .collect()
}

fn assert_neutral(case: &Case) -> TransitBucketTopology {
    let declared = system(case, true);
    let padded = system(case, false);
    assert!(matches!(
        WaterDeliveryBasis::of(&declared),
        WaterDeliveryBasis::Declared(_)
    ));
    assert_eq!(WaterDeliveryBasis::of(&padded), WaterDeliveryBasis::Legacy);

    let a = topology(&declared, case.boundary_present);
    let b = topology(&padded, case.boundary_present);

    assert_eq!(a.column_order, b.column_order, "column_order");
    assert_eq!(a.per_stage_mask, b.per_stage_mask, "per_stage_mask");
    assert_eq!(
        arc_rows(&a.arc_stage_weights, |weights| bits(weights)),
        arc_rows(&b.arc_stage_weights, |weights| bits(weights)),
        "stage weights"
    );
    assert_eq!(
        arc_rows(&a.arc_spread_chrono, |spread| spread
            .as_ref()
            .map(SpreadBits::of)),
        arc_rows(&b.arc_spread_chrono, |spread| spread
            .as_ref()
            .map(SpreadBits::of)),
        "chronological spread, stage_reach included"
    );
    assert_eq!(
        arc_rows(&a.arc_arrival_density, |density| density
            .as_deref()
            .map(bits)),
        arc_rows(&b.arc_arrival_density, |density| density
            .as_deref()
            .map(bits)),
        "arrival density"
    );
    a
}

proptest! {
    #![proptest_config(ProptestConfig {
        rng_seed: RngSeed::Fixed(42),
        ..ProptestConfig::default()
    })]

    #[test]
    fn declared_and_padded_bases_agree_on_covering_water_calendars(case in covering_case()) {
        assert_neutral(&case);
    }
}

#[test]
fn declared_and_padded_bases_agree_at_equality_coverage() {
    for boundary_present in [false, true] {
        let case = Case {
            study: vec![
                StageShape {
                    hours: 168.0,
                    n_blocks: 2,
                    block_mode: BlockMode::Chronological,
                };
                3
            ],
            travel_times: vec![336.0],
            water: vec![168.0, 168.0],
            boundary_present,
        };

        let topology = assert_neutral(&case);

        assert_eq!(
            topology.n_buckets(),
            2,
            "a 336 h travel time over 168 h stages reaches lag 2, so the equality is not vacuous"
        );
    }
}
