use std::collections::BTreeMap;

use novomodelo_core::{
    AnticipatedConfig, Block, BlockMode, Bus, DeficitSegment, EntityId, NoiseMethod,
    PostStudyStage, PostStudyStages, PostStudyThermalBound, ScenarioSourceConfig, Stage,
    StageRiskConfig, StageStateConfig, System, SystemBuilder, Thermal, WaterBucketStages,
};
use novomodelo_stochastic::par::precompute::PrecomputedPar;

use super::{ResolvedStateLayout, build_extended_delivery_anchors, resolve_state_and_topology};
use crate::bucket_topology::TransitBucketTopology;
use crate::lp::builder::{FactGroups, encode_time_value_facts};
use crate::lp::indexer::AnticipatedPlants;
use crate::test_support::{geometry_hydro, ymd};
use crate::time_value::{DeliveryCalendar, TimeValue, WaterDeliveryBasis};

const STUDY_MONTHS: [(u32, f64); 6] = [
    (1, 744.0),
    (2, 696.0),
    (3, 744.0),
    (4, 720.0),
    (5, 744.0),
    (6, 720.0),
];
const POST_STUDY_MONTHS: [(u32, f64); 3] = [(7, 744.0), (8, 744.0), (9, 720.0)];
const THERMAL_ID: EntityId = EntityId(7);
const ARC_TRAVEL_HOURS: f64 = 2300.0;
const LEAD_TIME_HOURS: f64 = 1000.0;

fn system_with_post_study(n_post: usize) -> System {
    let stages: Vec<Stage> = STUDY_MONTHS
        .iter()
        .enumerate()
        .map(|(index, &(month, duration_hours))| Stage {
            index,
            id: i32::try_from(index).expect("small stage index"),
            start_date: ymd(2024, month, 1),
            end_date: ymd(2024, month + 1, 1),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();
    let post_study = PostStudyStages {
        thermal_stages: POST_STUDY_MONTHS[..n_post]
            .iter()
            .map(|&(month, duration_hours)| PostStudyStage {
                start_date: ymd(2024, month, 1),
                duration_hours,
            })
            .collect(),
        thermal_bounds: (0..n_post)
            .map(|post_study_stage_index| PostStudyThermalBound {
                thermal_id: THERMAL_ID,
                post_study_stage_index,
                cost_per_mwh: 50.0,
                min_mw: 0.0,
                max_mw: 500.0,
            })
            .collect(),
        water_bucket_stages: WaterBucketStages::FromThermalStages,
    };
    let mut upstream = geometry_hydro(1);
    upstream.downstream_id = Some(EntityId(0));
    upstream.travel_time_hours = Some(ARC_TRAVEL_HOURS);
    SystemBuilder::new()
        .buses(vec![Bus {
            id: EntityId(0),
            name: "B".to_string(),
            operational_start_date: ymd(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        }])
        .hydros(vec![geometry_hydro(0), upstream])
        .thermals(vec![Thermal {
            id: THERMAL_ID,
            name: "T".to_string(),
            operational_start_date: ymd(2024, 1, 1),
            bus_id: EntityId(0),
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(LEAD_TIME_HOURS)),
            entry_stage_id: None,
            exit_stage_id: None,
        }])
        .stages(stages)
        .post_study_stages(Some(post_study))
        .build()
        .expect("fixture system builds")
}

fn system_a() -> System {
    system_with_post_study(1)
}

fn system_b() -> System {
    system_with_post_study(POST_STUDY_MONTHS.len())
}

fn resolve(
    system: &System,
    thermal: &DeliveryCalendar,
    water: &DeliveryCalendar,
) -> (TransitBucketTopology, ResolvedStateLayout) {
    resolve_state_and_topology(
        system,
        thermal,
        water,
        WaterDeliveryBasis::of(system),
        &PrecomputedPar::default(),
        None,
        false,
    )
    .expect("fixture resolves")
}

#[derive(Debug, PartialEq)]
struct ThermalFacts {
    time_value: FactGroups,
    n_delivery: usize,
    max_fanout: usize,
    anchors: Vec<i32>,
}

fn thermal_facts(
    system: &System,
    thermal: &DeliveryCalendar,
    water: &DeliveryCalendar,
) -> ThermalFacts {
    let (_, layout) = resolve(system, thermal, water);
    let plants = AnticipatedPlants::build(system.thermals());
    let mut time_value = FactGroups::new();
    encode_time_value_facts(
        &TimeValue::from_system(system, &plants, thermal.clone()),
        &mut time_value,
    );
    ThermalFacts {
        time_value,
        n_delivery: layout.state.n_delivery(),
        max_fanout: layout.state.anticipated_resolution.max_fanout(),
        anchors: build_extended_delivery_anchors(system, thermal),
    }
}

fn water_facts(topology: &TransitBucketTopology) -> String {
    format!(
        "{:?}{:?}{:?}{:?}{:?}",
        topology.column_order,
        topology.per_stage_mask,
        topology
            .arc_stage_weights
            .iter()
            .collect::<BTreeMap<_, _>>(),
        topology
            .arc_spread_chrono
            .iter()
            .collect::<BTreeMap<_, _>>(),
        topology
            .arc_arrival_density
            .iter()
            .collect::<BTreeMap<_, _>>(),
    )
}

#[test]
fn thermal_consumers_ignore_the_water_calendar() {
    let (a, b) = (system_a(), system_b());
    let thermal = DeliveryCalendar::thermal(&a);

    assert_eq!(
        thermal_facts(
            &a,
            &thermal,
            &DeliveryCalendar::water(&a, WaterDeliveryBasis::of(&a))
        ),
        thermal_facts(
            &a,
            &thermal,
            &DeliveryCalendar::water(&b, WaterDeliveryBasis::of(&b))
        ),
    );
}

#[test]
fn water_consumers_ignore_the_thermal_calendar() {
    let (a, b) = (system_a(), system_b());
    let (thermal_a, thermal_b) = (DeliveryCalendar::thermal(&a), DeliveryCalendar::thermal(&b));
    let (water_a, water_b) = (
        DeliveryCalendar::water(&a, WaterDeliveryBasis::of(&a)),
        DeliveryCalendar::water(&b, WaterDeliveryBasis::of(&b)),
    );

    assert_eq!(
        water_facts(&resolve(&a, &thermal_a, &water_b).0),
        water_facts(&resolve(&a, &thermal_b, &water_b).0),
    );
    assert_ne!(
        water_facts(&resolve(&a, &thermal_a, &water_a).0),
        water_facts(&resolve(&a, &thermal_a, &water_b).0),
    );
}

#[test]
fn thermal_consumers_follow_the_thermal_calendar() {
    let (a, b) = (system_a(), system_b());
    let water = DeliveryCalendar::water(&a, WaterDeliveryBasis::of(&a));

    let from_a = thermal_facts(&a, &DeliveryCalendar::thermal(&a), &water);
    let from_b = thermal_facts(&a, &DeliveryCalendar::thermal(&b), &water);

    assert!(from_b.n_delivery > from_a.n_delivery);
    assert_ne!(from_b.time_value, from_a.time_value);
    assert!(from_b.anchors.len() > from_a.anchors.len());
}
