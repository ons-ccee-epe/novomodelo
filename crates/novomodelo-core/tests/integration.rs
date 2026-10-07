//! Integration tests exercising the full `SystemBuilder::build()` pipeline
//! through the public API only.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::test_support::{
    BusSpec, ContractSpec, HydroSpec, LineSpec, NcsSpec, PumpingSpec, ThermalSpec, UnitGroupSpec,
    make_bus, make_contract, make_hydro, make_line, make_ncs, make_pumping_station, make_thermal,
    make_unit_group,
};
use cobre_core::{
    Bus, DeficitSegment, DiversionChannel, EnergyContract, EntityId, FillingConfig, Hydro,
    HydroUnitGroup, Line, NonControllableSource, PumpingStation, SystemBuilder, Thermal,
    ValidationError,
};

fn deficit_bus(id: i32) -> Bus {
    make_bus(BusSpec {
        id,
        name: format!("bus-{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: Some(100.0),
            cost_per_mwh: 500.0,
        }],
        ..Default::default()
    })
}

fn sized_hydro(id: i32, bus_id: i32, downstream_id: Option<i32>) -> Hydro {
    make_hydro(HydroSpec {
        id,
        name: format!("hydro-{id}"),
        bus_id,
        downstream_id,
        max_storage_hm3: 100.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 450.0,
        ..Default::default()
    })
}

fn sized_thermal(id: i32, bus_id: i32) -> Thermal {
    make_thermal(ThermalSpec {
        id,
        name: format!("thermal-{id}"),
        bus_id,
        cost_per_mwh: 80.0,
        max_generation_mw: 300.0,
        ..Default::default()
    })
}

fn sized_ncs(id: i32, bus_id: i32) -> NonControllableSource {
    make_ncs(NcsSpec {
        id,
        name: format!("ncs-{id}"),
        bus_id,
        max_generation_mw: 80.0,
        curtailment_cost: 5.0,
        ..Default::default()
    })
}

fn sized_contract(id: i32, bus_id: i32) -> EnergyContract {
    make_contract(ContractSpec {
        id,
        name: format!("contract-{id}"),
        bus_id,
        price_per_mwh: 50.0,
        max_mw: 150.0,
        ..Default::default()
    })
}

fn sized_pumping(id: i32, source_hydro_id: i32, destination_hydro_id: i32) -> PumpingStation {
    make_pumping_station(PumpingSpec {
        id,
        name: format!("ps-{id}"),
        bus_id: 2,
        source_hydro_id,
        destination_hydro_id,
        max_flow_m3s: 20.0,
        ..Default::default()
    })
}

fn sized_line(id: i32, source_bus_id: i32, target_bus_id: i32) -> Line {
    make_line(LineSpec {
        id,
        name: format!("line-{id}"),
        source_bus_id,
        target_bus_id,
        direct_capacity_mw: 200.0,
        reverse_capacity_mw: 200.0,
        ..Default::default()
    })
}

fn sized_unit_group(id: i32, bus_id: i32) -> HydroUnitGroup {
    make_unit_group(UnitGroupSpec {
        id,
        name: format!("group-{id}"),
        bus_id,
        max_generation_mw: 450.0,
        max_turbined_m3s: 500.0,
        ..Default::default()
    })
}

#[test]
fn test_declaration_order_invariance() {
    let buses_fwd = vec![deficit_bus(1), deficit_bus(2)];
    let lines_fwd = vec![sized_line(10, 1, 2)];
    let hydros_fwd = vec![sized_hydro(20, 1, Some(21)), sized_hydro(21, 1, None)];
    let thermals_fwd = vec![sized_thermal(30, 2)];
    let pumping_fwd = vec![sized_pumping(40, 20, 21)];
    let contracts_fwd = vec![sized_contract(50, 1)];
    let ncs_fwd = vec![sized_ncs(60, 2)];

    let system_fwd = SystemBuilder::new()
        .buses(buses_fwd)
        .lines(lines_fwd)
        .hydros(hydros_fwd)
        .thermals(thermals_fwd)
        .pumping_stations(pumping_fwd)
        .contracts(contracts_fwd)
        .non_controllable_sources(ncs_fwd)
        .build()
        .expect("forward-order system must be valid");

    let buses_rev = vec![deficit_bus(2), deficit_bus(1)];
    let lines_rev = vec![sized_line(10, 1, 2)];
    let hydros_rev = vec![sized_hydro(21, 1, None), sized_hydro(20, 1, Some(21))];
    let thermals_rev = vec![sized_thermal(30, 2)];
    let pumping_rev = vec![sized_pumping(40, 20, 21)];
    let contracts_rev = vec![sized_contract(50, 1)];
    let ncs_rev = vec![sized_ncs(60, 2)];

    let system_rev = SystemBuilder::new()
        .buses(buses_rev)
        .lines(lines_rev)
        .hydros(hydros_rev)
        .thermals(thermals_rev)
        .pumping_stations(pumping_rev)
        .contracts(contracts_rev)
        .non_controllable_sources(ncs_rev)
        .build()
        .expect("reverse-order system must be valid");

    assert_eq!(
        system_fwd, system_rev,
        "System must be identical regardless of input entity ordering"
    );
}

#[test]
fn test_realistic_multi_entity_system() {
    let mut hydro_10 = sized_hydro(10, 1, Some(12));
    let mut hydro_11 = sized_hydro(11, 2, Some(12));
    let hydro_12 = sized_hydro(12, 3, None);

    hydro_10.name = "upstream-A".to_string();
    hydro_11.name = "upstream-B".to_string();

    let system = SystemBuilder::new()
        .buses(vec![deficit_bus(1), deficit_bus(2), deficit_bus(3)])
        .lines(vec![sized_line(100, 1, 2), sized_line(101, 2, 3)])
        .hydros(vec![hydro_10, hydro_11, hydro_12])
        .thermals(vec![sized_thermal(20, 1), sized_thermal(21, 3)])
        .pumping_stations(vec![sized_pumping(30, 10, 12)])
        .contracts(vec![sized_contract(40, 1)])
        .non_controllable_sources(vec![sized_ncs(50, 3)])
        .build()
        .expect("realistic multi-entity system must be valid");

    assert_eq!(system.n_buses(), 3);
    assert_eq!(system.n_lines(), 2);
    assert_eq!(system.n_hydros(), 3);
    assert_eq!(system.n_thermals(), 2);
    assert_eq!(system.n_pumping_stations(), 1);
    assert_eq!(system.n_contracts(), 1);
    assert_eq!(system.n_non_controllable_sources(), 1);

    assert!(system.bus(EntityId(1)).is_some());
    assert!(system.bus(EntityId(2)).is_some());
    assert!(system.bus(EntityId(3)).is_some());
    assert!(system.bus(EntityId(999)).is_none());

    assert!(system.line(EntityId(100)).is_some());
    assert!(system.line(EntityId(101)).is_some());

    let h10 = system.hydro(EntityId(10)).expect("hydro 10 must exist");
    assert_eq!(h10.name, "upstream-A");
    assert!(system.hydro(EntityId(11)).is_some());
    assert!(system.hydro(EntityId(12)).is_some());
    assert!(system.hydro(EntityId(999)).is_none());

    assert!(system.thermal(EntityId(20)).is_some());
    assert!(system.thermal(EntityId(21)).is_some());

    assert!(system.pumping_station(EntityId(30)).is_some());
    assert!(system.contract(EntityId(40)).is_some());
    assert!(system.non_controllable_source(EntityId(50)).is_some());

    let buses = system.buses();
    assert_eq!(buses[0].id, EntityId(1));
    assert_eq!(buses[1].id, EntityId(2));
    assert_eq!(buses[2].id, EntityId(3));

    // Equal dates sort by id ascending.
    let hydros = system.hydros();
    assert_eq!(hydros[0].id, EntityId(10));
    assert_eq!(hydros[1].id, EntityId(11));
    assert_eq!(hydros[2].id, EntityId(12));

    let cascade = system.cascade();
    assert_eq!(cascade.len(), 3);

    assert_eq!(cascade.downstream(EntityId(10)), Some(EntityId(12)));
    assert_eq!(cascade.downstream(EntityId(11)), Some(EntityId(12)));
    assert_eq!(cascade.downstream(EntityId(12)), None);
    let upstream_12 = cascade.upstream(EntityId(12));
    assert_eq!(upstream_12.len(), 2);
    assert_eq!(upstream_12[0], EntityId(10));
    assert_eq!(upstream_12[1], EntityId(11));
    assert!(cascade.is_headwater(EntityId(10)));
    assert!(cascade.is_headwater(EntityId(11)));
    assert!(!cascade.is_headwater(EntityId(12)));
    assert!(!cascade.is_terminal(EntityId(10)));
    assert!(!cascade.is_terminal(EntityId(11)));
    assert!(cascade.is_terminal(EntityId(12)));

    let topo = cascade.topological_order();
    let pos_10 = topo
        .iter()
        .position(|&id| id == EntityId(10))
        .expect("10 in topo");
    let pos_11 = topo
        .iter()
        .position(|&id| id == EntityId(11))
        .expect("11 in topo");
    let pos_12 = topo
        .iter()
        .position(|&id| id == EntityId(12))
        .expect("12 in topo");
    assert!(
        pos_10 < pos_12,
        "hydro 10 must precede hydro 12 in topo order"
    );
    assert!(
        pos_11 < pos_12,
        "hydro 11 must precede hydro 12 in topo order"
    );
}

#[test]
fn test_hydro_on_unknown_bus_is_accepted_groups_own_the_bus() {
    let mut hydro = sized_hydro(1, 999, None);
    hydro.unit_groups = vec![sized_unit_group(0, 1)];

    let result = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_ok(),
        "a plant's own bus argument naming a nonexistent bus must not fail \
         validation — only its groups' own buses are validated; got: {:?}",
        result.err()
    );
}

#[test]
fn test_cascade_cycle_rejected() {
    let hydro_1 = sized_hydro(1, 1, Some(2));
    let hydro_2 = sized_hydro(2, 1, Some(1));

    let result = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![hydro_1, hydro_2])
        .build();

    assert!(result.is_err(), "cyclic cascade must fail validation");

    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { .. })),
        "expected CascadeCycle error; got: {errors:?}"
    );
}

#[test]
fn test_large_order_invariance() {
    let make_system = |bus_order: Vec<i32>, hydro_order: Vec<(i32, Option<i32>)>| {
        let buses = bus_order.into_iter().map(deficit_bus).collect();
        let hydros = hydro_order
            .into_iter()
            .map(|(id, ds)| sized_hydro(id, 1, ds))
            .collect();

        SystemBuilder::new()
            .buses(buses)
            .lines(vec![sized_line(10, 1, 2), sized_line(11, 2, 3)])
            .hydros(hydros)
            .thermals(vec![sized_thermal(20, 2), sized_thermal(21, 3)])
            .pumping_stations(vec![sized_pumping(30, 1, 3)])
            .contracts(vec![sized_contract(40, 1)])
            .non_controllable_sources(vec![sized_ncs(50, 3)])
            .build()
            .expect("system must be valid")
    };

    let system_asc = make_system(vec![1, 2, 3], vec![(1, Some(3)), (2, Some(3)), (3, None)]);
    let system_desc = make_system(vec![3, 2, 1], vec![(3, None), (2, Some(3)), (1, Some(3))]);

    assert_eq!(
        system_asc, system_desc,
        "System must be identical regardless of input ordering (large test)"
    );
}

#[test]
fn test_invalid_filling_config_rejected() {
    let mut hydro = sized_hydro(1, 1, None);
    hydro.entry_stage_id = Some(0);
    hydro.filling = Some(FillingConfig {
        start_stage_id: 0,
        filling_min_rate_m3s: -5.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_err(),
        "invalid filling config must fail validation"
    );

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidFillingConfig {
                hydro_id: EntityId(1),
                ..
            }
        )),
        "expected InvalidFillingConfig for hydro 1; got: {errors:?}"
    );
}

#[test]
fn test_diversion_invalid_reference_rejected() {
    let mut hydro = sized_hydro(1, 1, None);
    hydro.diversion = Some(DiversionChannel {
        downstream_id: EntityId(999),
        max_flow_m3s: 10.0,
    });

    let result = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![hydro])
        .build();

    assert!(
        result.is_err(),
        "hydro with bad diversion.downstream_id must fail validation"
    );

    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                field_name: "diversion.downstream_id",
                referenced_id: EntityId(999),
                ..
            }
        )),
        "expected InvalidReference for Hydro.diversion.downstream_id -> Hydro 999; got: {errors:?}"
    );
}

#[test]
fn test_canonical_order_stable_under_name_changes() {
    let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let mut hydro_a = sized_hydro(1, 1, None);
    hydro_a.name = "alpha".to_string();
    hydro_a.operational_start_date = date;
    let mut hydro_b = sized_hydro(2, 1, None);
    hydro_b.name = "bravo".to_string();
    hydro_b.operational_start_date = date;

    let system_a = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![hydro_a.clone(), hydro_b.clone()])
        .build()
        .expect("system must be valid");

    let mut a_renamed = hydro_a;
    a_renamed.name = "zulu".to_string();
    let mut b_renamed = hydro_b;
    b_renamed.name = "alfa".to_string();

    let system_b = SystemBuilder::new()
        .buses(vec![deficit_bus(1)])
        .hydros(vec![b_renamed, a_renamed])
        .build()
        .expect("system must be valid");

    let ids_a: Vec<i32> = system_a.hydros().iter().map(|h| h.id.0).collect();
    let ids_b: Vec<i32> = system_b.hydros().iter().map(|h| h.id.0).collect();
    assert_eq!(
        ids_a, ids_b,
        "renaming with (date, id) held constant must not change processing order"
    );
    assert_eq!(
        ids_a,
        vec![1, 2],
        "same-date entities order by id ascending"
    );
}

#[test]
fn test_canonical_order_sorts_by_distinct_date() {
    let early = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let late = NaiveDate::from_ymd_opt(2024, 6, 1).unwrap();

    let mut bus_late = deficit_bus(1);
    bus_late.operational_start_date = late;
    let mut bus_early = deficit_bus(2);
    bus_early.operational_start_date = early;

    let system = SystemBuilder::new()
        .buses(vec![bus_late, bus_early])
        .build()
        .expect("system must be valid");

    assert!(
        system.buses()[0].operational_start_date < system.buses()[1].operational_start_date,
        "distinct dates supplied in reverse must come out date-ascending"
    );
}

#[test]
fn test_canonical_order_id_tiebreak_on_equal_date() {
    let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    // id 1 has the name that sorts LAST ("B"); id 2 the name that sorts first ("A").
    // The id tiebreak must win, so id 1 comes first regardless of name.
    let mut bus_b = deficit_bus(1);
    bus_b.name = "B".to_string();
    bus_b.operational_start_date = date;
    let mut bus_a = deficit_bus(2);
    bus_a.name = "A".to_string();
    bus_a.operational_start_date = date;

    let system = SystemBuilder::new()
        .buses(vec![bus_a, bus_b])
        .build()
        .expect("system must be valid");

    assert_eq!(system.buses()[0].id, EntityId(1));
    assert_eq!(system.buses()[1].id, EntityId(2));
}
