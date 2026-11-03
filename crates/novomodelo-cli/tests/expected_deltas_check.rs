//! The frozen expected-delta list parses, names every planned deck and follows
//! its own water calendars; the diff and check helpers behave on in-memory input.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::Duration;

use chrono::{Datelike, NaiveDate, TimeDelta};
use novomodelo_io::{
    ENTITY_SLOT_DATE_SENTINEL, EntitySlot, OwnedPolicyCutRecord, StageCutsReadResult, StateFamily,
};
use serde_json::{Value, json};

mod common;
use common::expected_deltas::{
    CoefficientDelta, ExpectedDeck, ExpectedList, SlotDelta, check_expected_deltas,
    diff_coefficients, diff_manifests,
};

const BOUNDARY_KEY: &str = "crates/novomodelo-sddp/tests/gate_fixtures/dual_calendar_boundary";

const PLANNED_DECKS: [&str; 11] = [
    "examples/deterministic/d44-travel-time-substage",
    "examples/deterministic/d45-travel-time-mixed-calendar",
    "examples/deterministic/d46-travel-time-chronological",
    "examples/deterministic/d47-travel-time-confluence",
    "examples/deterministic/d48-travel-time-ic-seed",
    "examples/deterministic/d49-travel-time-chronological-arrival",
    "examples/deterministic/d50-travel-time-plain-tributary-confluence",
    "examples/deterministic/d55-post-study-anticipated-lanes",
    "crates/novomodelo-sddp/tests/fixtures/travel_time_arc",
    "crates/novomodelo-sddp/tests/fixtures/exited_plant_transit",
    BOUNDARY_KEY,
];

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bit_identity")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn load_list() -> ExpectedList {
    serde_json::from_str(&fixture("phase2_expected_deltas.json")).unwrap()
}

fn encode(date: NaiveDate) -> i32 {
    date.year() * 10_000 + date.month().cast_signed() * 100 + date.day().cast_signed()
}

#[test]
fn expected_delta_list_parses_and_names_every_planned_deck() {
    let list = load_list();
    assert!(!list.frozen_before.is_empty());
    assert!(!list.informational.is_empty());

    let listed: BTreeSet<&str> = list.decks.keys().map(String::as_str).collect();
    assert_eq!(listed, BTreeSet::from(PLANNED_DECKS));

    let baseline: Value = serde_json::from_str(&fixture("highs.json")).unwrap();
    for deck in PLANNED_DECKS {
        assert!(
            baseline["decks"].get(deck).is_some(),
            "{deck} is not a deck key of highs.json"
        );
    }

    for (deck, expected) in &list.decks {
        for slot in &expected.slots {
            let file = Path::new(&slot.file);
            assert_eq!(file.parent(), Some(Path::new("policy/cuts")));
            assert_eq!(file.extension().unwrap(), "bin");
            assert_ne!(slot.legacy, slot.migrated, "{deck}: {slot:?} does not move");
        }
        for coefficient in &expected.coefficients {
            assert_eq!(coefficient.rows, "boundary");
            assert_eq!(
                deck, BOUNDARY_KEY,
                "only the boundary fixture moves coefficients"
            );
        }
    }
}

#[test]
fn expected_delta_dates_follow_the_water_calendar() {
    for (deck, expected) in load_list().decks {
        let calendar: Vec<(i32, i32)> = expected
            .water_bucket_stages
            .iter()
            .map(|stage| {
                let start = NaiveDate::parse_from_str(&stage.start_date, "%Y-%m-%d").unwrap();
                let end = start
                    + TimeDelta::from_std(Duration::from_secs_f64(stage.duration_hours * 3600.0))
                        .unwrap();
                (encode(start), encode(end))
            })
            .collect();
        for slot in &expected.slots {
            if let (Some(start), Some(end)) = slot.migrated {
                assert!(
                    calendar.contains(&(start, end)),
                    "{deck}: {} slot ({}, {}) migrates to [{start}, {end}), no water stage matches",
                    slot.file,
                    slot.entity_id,
                    slot.subindex
                );
            }
        }
    }
}

fn bucket(entity_id: i32, lag: u32, interval: (i32, i32)) -> EntitySlot {
    EntitySlot::transit_bucket(entity_id, lag, true).with_interval(interval.0, interval.1)
}

fn sentinel_bucket(entity_id: i32, lag: u32) -> EntitySlot {
    bucket(
        entity_id,
        lag,
        (ENTITY_SLOT_DATE_SENTINEL, ENTITY_SLOT_DATE_SENTINEL),
    )
}

#[test]
fn diff_manifests_of_equal_manifests_has_no_delta() {
    let manifest = vec![
        EntitySlot::storage(1, true),
        bucket(1, 1, (20_240_301, 20_240_401)),
        sentinel_bucket(1, 2),
    ];
    assert_eq!(
        diff_manifests("policy/cuts/001.bin", &manifest, &manifest).unwrap(),
        vec![]
    );
}

#[test]
fn diff_manifests_reports_a_changed_interval_with_the_sentinel_as_none() {
    let legacy = vec![EntitySlot::storage(1, true), sentinel_bucket(1, 1)];
    let migrated = vec![
        EntitySlot::storage(1, true),
        bucket(1, 1, (20_240_301, 20_240_401)),
    ];
    assert_eq!(
        diff_manifests("policy/cuts/001.bin", &legacy, &migrated).unwrap(),
        vec![SlotDelta {
            file: "policy/cuts/001.bin".to_string(),
            entity_type: StateFamily::HydroTransitBucket.code(),
            entity_id: 1,
            subindex: 1,
            legacy: (None, None),
            migrated: (Some(20_240_301), Some(20_240_401)),
        }]
    );
}

#[test]
fn diff_manifests_refuses_a_length_or_identity_mismatch() {
    let one = vec![sentinel_bucket(1, 1)];
    let two = vec![sentinel_bucket(1, 1), sentinel_bucket(1, 2)];
    assert!(
        diff_manifests("f", &one, &two)
            .unwrap_err()
            .contains("length")
    );

    let renamed = vec![sentinel_bucket(2, 1)];
    assert!(
        diff_manifests("f", &one, &renamed)
            .unwrap_err()
            .contains("beyond its interval")
    );
    let deactivated = vec![EntitySlot::transit_bucket(1, 1, false)];
    assert!(diff_manifests("f", &one, &deactivated).is_err());
}

fn cut(slot_index: u32, coefficients: &[f64]) -> OwnedPolicyCutRecord {
    OwnedPolicyCutRecord {
        slot_index,
        iteration: 1,
        forward_pass_index: 0,
        intercept: 0.0,
        coefficients: coefficients.to_vec(),
        is_active: true,
        scored: false,
        active_count: None,
        last_active_iteration: None,
    }
}

fn pool(warm_start_count: u32, cuts: Vec<OwnedPolicyCutRecord>) -> StageCutsReadResult {
    StageCutsReadResult {
        stage_id: 5,
        state_dimension: 2,
        capacity: 8,
        warm_start_count,
        populated_count: u32::try_from(cuts.len()).unwrap(),
        cuts,
        entity_manifest: vec![
            EntitySlot::storage(1, true),
            bucket(1, 3, (20_240_901, 20_241_001)),
        ],
        cost_scale_factor: Some(1.0),
        node_id: -1,
        graph_stage_id: -1,
        priced_state_date: ENTITY_SLOT_DATE_SENTINEL,
    }
}

#[test]
fn diff_coefficients_compares_bits_and_marks_the_boundary_rows() {
    let legacy = pool(1, vec![cut(0, &[1.0, 2.5]), cut(1, &[1.0, 2.5])]);
    let migrated = pool(1, vec![cut(0, &[1.0, 0.0]), cut(1, &[1.0, 2.5])]);
    assert_eq!(
        diff_coefficients("policy/cuts/005.bin", &legacy, &migrated),
        vec![CoefficientDelta {
            file: "policy/cuts/005.bin".to_string(),
            row: 0,
            boundary: true,
            entity_type: StateFamily::HydroTransitBucket.code(),
            entity_id: 1,
            subindex: 3,
            legacy: 2.5,
            migrated: 0.0,
        }]
    );
    assert!(diff_coefficients("f", &legacy, &legacy).is_empty());
}

#[test]
fn diff_coefficients_reports_a_negative_zero_change() {
    let legacy = pool(0, vec![cut(0, &[0.0, 0.0])]);
    let migrated = pool(0, vec![cut(0, &[0.0, -0.0])]);
    let deltas = diff_coefficients("f", &legacy, &migrated);
    assert_eq!(deltas.len(), 1);
    assert!(!deltas[0].boundary);
    assert_eq!(deltas[0].migrated.to_bits(), (-0.0f64).to_bits());
}

fn doctored_deck(slots: &[Value], coefficients: &[Value]) -> ExpectedDeck {
    serde_json::from_value(json!({
        "water_bucket_stages": [],
        "slots": slots,
        "coefficients": coefficients,
    }))
    .unwrap()
}

fn listed_slot() -> Value {
    json!({
        "file": "policy/cuts/005.bin", "entity_id": 1, "subindex": 3,
        "legacy": [20_240_901, 20_241_001], "migrated": [null, null],
    })
}

fn listed_coefficient() -> Value {
    json!({
        "file": "policy/cuts/005.bin", "slot": {"entity_id": 1, "subindex": 3},
        "rows": "boundary", "migrated": 0.0,
    })
}

fn actual_slot() -> SlotDelta {
    SlotDelta {
        file: "policy/cuts/005.bin".to_string(),
        entity_type: StateFamily::HydroTransitBucket.code(),
        entity_id: 1,
        subindex: 3,
        legacy: (Some(20_240_901), Some(20_241_001)),
        migrated: (None, None),
    }
}

fn actual_coefficient() -> CoefficientDelta {
    CoefficientDelta {
        file: "policy/cuts/005.bin".to_string(),
        row: 0,
        boundary: true,
        entity_type: StateFamily::HydroTransitBucket.code(),
        entity_id: 1,
        subindex: 3,
        legacy: 2.5,
        migrated: 0.0,
    }
}

#[test]
fn check_expected_deltas_accepts_exactly_the_listed_deltas() {
    let expected = doctored_deck(&[listed_slot()], &[listed_coefficient()]);
    check_expected_deltas("deck", &[actual_slot()], &[actual_coefficient()], &expected).unwrap();

    let empty = doctored_deck(&[], &[]);
    check_expected_deltas("deck", &[], &[], &empty).unwrap();
}

#[test]
fn check_expected_deltas_names_an_unlisted_actual_delta() {
    let expected = doctored_deck(&[], &[]);
    let failures =
        check_expected_deltas("deck", &[actual_slot()], &[actual_coefficient()], &expected)
            .unwrap_err();
    assert_eq!(failures.len(), 2);
    assert!(
        failures
            .iter()
            .all(|f| f.starts_with("unexpected delta: deck policy/cuts/005.bin"))
    );
}

#[test]
fn check_expected_deltas_names_a_listed_delta_nobody_observed() {
    let expected = doctored_deck(&[listed_slot()], &[listed_coefficient()]);
    let failures = check_expected_deltas("deck", &[], &[], &expected).unwrap_err();
    assert_eq!(failures.len(), 2);
    assert!(
        failures
            .iter()
            .all(|f| f.starts_with("missing expected delta: deck policy/cuts/005.bin"))
    );
}

#[test]
fn check_expected_deltas_rejects_a_delta_that_differs_from_the_listed_one() {
    let expected = doctored_deck(&[listed_slot()], &[listed_coefficient()]);

    let mut moved = actual_slot();
    moved.migrated = (Some(20_241_001), Some(20_241_101));
    let mut anticipated = actual_slot();
    anticipated.entity_type = StateFamily::AnticipatedThermalState.code();
    let mut negative_zero = actual_coefficient();
    negative_zero.migrated = -0.0;
    let mut generated = actual_coefficient();
    generated.boundary = false;

    let failures = check_expected_deltas(
        "deck",
        &[moved, anticipated],
        &[negative_zero, generated],
        &expected,
    )
    .unwrap_err();
    let count = |prefix: &str| failures.iter().filter(|f| f.starts_with(prefix)).count();
    assert_eq!(count("unexpected delta:"), 4);
    assert_eq!(count("missing expected delta:"), 2);
}
