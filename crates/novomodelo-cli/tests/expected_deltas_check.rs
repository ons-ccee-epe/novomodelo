//! The expected-delta list parses, names every planned deck and follows its own
//! water calendars; the diff and check helpers behave on in-memory input, and each
//! list kind (slots, coefficients, bases, files) fails in both directions.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Duration;

use chrono::{Datelike, NaiveDate, TimeDelta};
use novomodelo_io::{
    ENTITY_SLOT_DATE_SENTINEL, EntitySlot, OwnedBasisEntry, OwnedPolicyBasisRecord,
    OwnedPolicyCutRecord, StageCutsReadResult, StateFamily,
};
use serde_json::{Value, json};

mod common;
use common::expected_deltas::{
    BasisDelta, CoefficientDelta, ExpectedDeck, ExpectedList, ExpectedRows, SlotDelta, SlotValue,
    check_expected_deltas, check_files, diff_bases, diff_coefficients, diff_manifests,
};

const BOUNDARY_KEY: &str = "crates/novomodelo-sddp/tests/gate_fixtures/dual_calendar_boundary";
const TRAVEL_TIME_ARC_KEY: &str = "crates/novomodelo-sddp/tests/fixtures/travel_time_arc";

const PLANNED_DECKS: [&str; 11] = [
    "examples/deterministic/d44-travel-time-substage",
    "examples/deterministic/d45-travel-time-mixed-calendar",
    "examples/deterministic/d46-travel-time-chronological",
    "examples/deterministic/d47-travel-time-confluence",
    "examples/deterministic/d48-travel-time-ic-seed",
    "examples/deterministic/d49-travel-time-chronological-arrival",
    "examples/deterministic/d50-travel-time-plain-tributary-confluence",
    "examples/deterministic/d55-post-study-anticipated-lanes",
    TRAVEL_TIME_ARC_KEY,
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
            let rows = if deck == BOUNDARY_KEY {
                ExpectedRows::Boundary
            } else {
                ExpectedRows::All
            };
            assert_eq!(coefficient.rows, rows, "{deck}: {coefficient:?}");
            assert!(
                deck == BOUNDARY_KEY || deck == TRAVEL_TIME_ARC_KEY,
                "{deck}: only the boundary fixture and travel_time_arc move coefficients"
            );
        }
        for basis in &expected.bases {
            let file = Path::new(&basis.file);
            assert_eq!(file.parent(), Some(Path::new("policy/basis")));
            assert_ne!(basis.legacy, basis.migrated, "{deck}: {basis:?}");
        }
        assert!(
            expected.bases.is_empty() || deck == BOUNDARY_KEY,
            "{deck}: only the boundary fixture moves a basis"
        );
        let unique: BTreeSet<_> = expected.files.iter().chain(&expected.lp_facts).collect();
        assert_eq!(
            unique.len(),
            expected.files.len() + expected.lp_facts.len(),
            "{deck}: files and lp_facts list each name once"
        );
    }
}

#[test]
fn travel_time_arc_lists_the_depth_three_slot_added_by_the_unpadded_calendar() {
    let list = load_list();
    let expected = &list.decks[TRAVEL_TIME_ARC_KEY];

    let added: BTreeMap<&str, SlotValue> = expected
        .slots
        .iter()
        .filter(|slot| slot.legacy == SlotValue::Absent)
        .inspect(|slot| assert_eq!((slot.entity_id, slot.subindex), (1, 3)))
        .map(|slot| (slot.file.as_str(), slot.migrated))
        .collect();
    assert_eq!(
        added,
        BTreeMap::from([
            (
                "policy/cuts/000.bin",
                SlotValue::Interval((Some(20_240_401), Some(20_240_501)))
            ),
            (
                "policy/cuts/001.bin",
                SlotValue::Interval((Some(20_240_501), Some(20_240_601)))
            ),
            ("policy/cuts/002.bin", SlotValue::Interval((None, None))),
        ])
    );
    assert_eq!(expected.slots.len(), 6);

    let rows: BTreeSet<&str> = expected
        .coefficients
        .iter()
        .inspect(|c| assert_eq!(c.migrated.to_bits(), 0.0f64.to_bits()))
        .map(|c| c.file.as_str())
        .collect();
    assert_eq!(
        rows,
        BTreeSet::from(["policy/cuts/000.bin", "policy/cuts/001.bin"]),
        "policy/cuts/002.bin holds no cut row, so it has no coefficient to list"
    );

    assert!(
        expected
            .files
            .contains(&"training/scaling_report.json".to_string())
    );
    assert!(expected.lp_facts.contains(&"parity_hash".to_string()));
}

#[test]
fn boundary_fixture_lists_the_frozen_bucket_column_flip() {
    let list = load_list();
    let bases = &list.decks[BOUNDARY_KEY].bases;
    assert_eq!(bases.len(), 1);
    assert_eq!(bases[0].file, "policy/basis/005.bin");
    assert_eq!(
        (
            bases[0].column,
            bases[0].legacy.as_str(),
            bases[0].migrated.as_str()
        ),
        (4, "Upper", "Lower")
    );
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
            if let SlotValue::Interval((Some(start), Some(end))) = slot.migrated {
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
            legacy: SlotValue::Interval((None, None)),
            migrated: SlotValue::Interval((Some(20_240_301), Some(20_240_401))),
        }]
    );
}

fn bucket_delta(
    entity_id: i32,
    subindex: u32,
    legacy: SlotValue,
    migrated: SlotValue,
) -> SlotDelta {
    SlotDelta {
        file: "f".to_string(),
        entity_type: StateFamily::HydroTransitBucket.code(),
        entity_id,
        subindex,
        legacy,
        migrated,
    }
}

#[test]
fn diff_manifests_aligns_slots_by_key_not_by_position() {
    let legacy = vec![
        EntitySlot::storage(1, true),
        bucket(1, 1, (20_240_301, 20_240_401)),
        sentinel_bucket(1, 2),
    ];
    let migrated = vec![
        sentinel_bucket(2, 1),
        EntitySlot::storage(1, true),
        bucket(1, 2, (20_240_401, 20_240_501)),
        bucket(1, 1, (20_240_301, 20_240_401)),
    ];
    assert_eq!(
        diff_manifests("f", &legacy, &migrated).unwrap(),
        vec![
            bucket_delta(
                1,
                2,
                SlotValue::Interval((None, None)),
                SlotValue::Interval((Some(20_240_401), Some(20_240_501)))
            ),
            bucket_delta(2, 1, SlotValue::Absent, SlotValue::Interval((None, None))),
        ]
    );
}

#[test]
fn diff_manifests_reports_a_slot_only_one_side_holds_as_absent() {
    let one = vec![sentinel_bucket(1, 1)];
    let two = vec![
        sentinel_bucket(1, 1),
        bucket(1, 2, (20_240_401, 20_240_501)),
    ];
    let added = bucket_delta(
        1,
        2,
        SlotValue::Absent,
        SlotValue::Interval((Some(20_240_401), Some(20_240_501))),
    );
    assert_eq!(diff_manifests("f", &one, &two).unwrap(), vec![added]);

    let dropped = bucket_delta(
        1,
        2,
        SlotValue::Interval((Some(20_240_401), Some(20_240_501))),
        SlotValue::Absent,
    );
    assert_eq!(diff_manifests("f", &two, &one).unwrap(), vec![dropped]);
}

#[test]
fn diff_manifests_refuses_an_identity_mismatch_or_a_duplicate_key() {
    let one = vec![sentinel_bucket(1, 1)];
    let deactivated = vec![EntitySlot::transit_bucket(1, 1, false)];
    assert!(
        diff_manifests("f", &one, &deactivated)
            .unwrap_err()
            .contains("beyond its interval")
    );

    let twice = vec![sentinel_bucket(1, 1), sentinel_bucket(1, 1)];
    assert!(
        diff_manifests("f", &one, &twice)
            .unwrap_err()
            .contains("appears twice in the migrated manifest")
    );
    assert!(
        diff_manifests("f", &twice, &one)
            .unwrap_err()
            .contains("appears twice in the legacy manifest")
    );
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

fn pool_over(
    entity_manifest: Vec<EntitySlot>,
    warm_start_count: u32,
    cuts: Vec<OwnedPolicyCutRecord>,
) -> StageCutsReadResult {
    StageCutsReadResult {
        stage_id: 5,
        state_dimension: u32::try_from(entity_manifest.len()).unwrap(),
        capacity: 8,
        warm_start_count,
        populated_count: u32::try_from(cuts.len()).unwrap(),
        cuts,
        entity_manifest,
        cost_scale_factor: Some(1.0),
        node_id: -1,
        graph_stage_id: -1,
        priced_state_date: ENTITY_SLOT_DATE_SENTINEL,
    }
}

fn pool(warm_start_count: u32, cuts: Vec<OwnedPolicyCutRecord>) -> StageCutsReadResult {
    pool_over(
        vec![
            EntitySlot::storage(1, true),
            bucket(1, 3, (20_240_901, 20_241_001)),
        ],
        warm_start_count,
        cuts,
    )
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
            legacy: Some(2.5),
            migrated: 0.0,
        }]
    );
    assert!(diff_coefficients("f", &legacy, &legacy).is_empty());
}

#[test]
fn diff_coefficients_aligns_by_slot_key_and_reports_an_added_slot_on_every_row() {
    let storage = EntitySlot::storage(1, true);
    let lag_one = bucket(1, 1, (20_240_301, 20_240_401));
    let lag_two = bucket(1, 2, (20_240_401, 20_240_501));
    let legacy = pool_over(
        vec![storage.clone(), lag_one.clone()],
        0,
        vec![cut(0, &[1.0, 2.5]), cut(1, &[3.0, 4.5])],
    );
    let migrated = pool_over(
        vec![storage, lag_two, lag_one],
        0,
        vec![cut(0, &[1.0, 0.0, 2.5]), cut(1, &[3.0, 0.0, 4.5])],
    );
    let deltas = diff_coefficients("f", &legacy, &migrated);
    assert_eq!(deltas.len(), 2);
    for (row, delta) in deltas.iter().enumerate() {
        assert_eq!(delta.row, u32::try_from(row).unwrap());
        assert_eq!((delta.entity_id, delta.subindex), (1, 2));
        assert_eq!(delta.legacy, None);
        assert_eq!(delta.migrated.to_bits(), 0.0f64.to_bits());
    }
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

fn deck_with_bases(bases: &[Value]) -> ExpectedDeck {
    serde_json::from_value(json!({
        "water_bucket_stages": [],
        "slots": [],
        "coefficients": [],
        "bases": bases,
    }))
    .unwrap()
}

fn listed_slot() -> Value {
    json!({
        "file": "policy/cuts/005.bin", "entity_id": 1, "subindex": 3,
        "legacy": [20_240_901, 20_241_001], "migrated": [null, null],
    })
}

fn listed_added_slot() -> Value {
    json!({
        "file": "policy/cuts/000.bin", "entity_id": 1, "subindex": 3,
        "legacy": "absent", "migrated": [20_240_401, 20_240_501],
    })
}

fn listed_coefficient() -> Value {
    json!({
        "file": "policy/cuts/005.bin", "slot": {"entity_id": 1, "subindex": 3},
        "rows": "boundary", "migrated": 0.0,
    })
}

fn listed_added_coefficient() -> Value {
    json!({
        "file": "policy/cuts/000.bin", "slot": {"entity_id": 1, "subindex": 3},
        "rows": "all", "migrated": 0.0,
    })
}

fn listed_basis() -> Value {
    json!({
        "file": "policy/basis/005.bin", "column": 4,
        "legacy": "Upper", "migrated": "Lower",
    })
}

fn actual_slot() -> SlotDelta {
    SlotDelta {
        file: "policy/cuts/005.bin".to_string(),
        entity_type: StateFamily::HydroTransitBucket.code(),
        entity_id: 1,
        subindex: 3,
        legacy: SlotValue::Interval((Some(20_240_901), Some(20_241_001))),
        migrated: SlotValue::Interval((None, None)),
    }
}

fn actual_added_slot() -> SlotDelta {
    SlotDelta {
        file: "policy/cuts/000.bin".to_string(),
        legacy: SlotValue::Absent,
        migrated: SlotValue::Interval((Some(20_240_401), Some(20_240_501))),
        ..actual_slot()
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
        legacy: Some(2.5),
        migrated: 0.0,
    }
}

fn actual_added_coefficient(row: u32) -> CoefficientDelta {
    CoefficientDelta {
        file: "policy/cuts/000.bin".to_string(),
        row,
        boundary: false,
        legacy: None,
        ..actual_coefficient()
    }
}

fn actual_basis() -> BasisDelta {
    BasisDelta {
        file: "policy/basis/005.bin".to_string(),
        scenario: 0,
        column: 4,
        legacy: 2,
        migrated: 0,
    }
}

#[test]
fn check_expected_deltas_accepts_exactly_the_listed_deltas() {
    let expected = doctored_deck(&[listed_slot()], &[listed_coefficient()]);
    check_expected_deltas(
        "deck",
        &[actual_slot()],
        &[actual_coefficient()],
        &[],
        &expected,
    )
    .unwrap();

    let empty = doctored_deck(&[], &[]);
    check_expected_deltas("deck", &[], &[], &[], &empty).unwrap();
}

#[test]
fn check_expected_deltas_names_an_unlisted_actual_delta() {
    let expected = doctored_deck(&[], &[]);
    let failures = check_expected_deltas(
        "deck",
        &[actual_slot()],
        &[actual_coefficient()],
        &[],
        &expected,
    )
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
    let failures = check_expected_deltas("deck", &[], &[], &[], &expected).unwrap_err();
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
    moved.migrated = SlotValue::Interval((Some(20_241_001), Some(20_241_101)));
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
        &[],
        &expected,
    )
    .unwrap_err();
    let count = |prefix: &str| failures.iter().filter(|f| f.starts_with(prefix)).count();
    assert_eq!(count("unexpected delta:"), 4);
    assert_eq!(count("missing expected delta:"), 2);
}

#[test]
fn absent_slots_fail_both_ways_and_are_not_interchangeable_with_the_sentinel_interval() {
    let expected = doctored_deck(&[listed_added_slot()], &[]);
    check_expected_deltas("deck", &[actual_added_slot()], &[], &[], &expected).unwrap();

    let removed = doctored_deck(&[], &[]);
    let failures =
        check_expected_deltas("deck", &[actual_added_slot()], &[], &[], &removed).unwrap_err();
    assert_eq!(failures.len(), 1);
    assert!(failures[0].starts_with("unexpected delta: deck policy/cuts/000.bin"));
    assert!(failures[0].contains("absent -> [20240401, 20240501)"));

    let spurious = check_expected_deltas("deck", &[], &[], &[], &expected).unwrap_err();
    assert_eq!(spurious.len(), 1);
    assert!(spurious[0].starts_with("missing expected delta: deck policy/cuts/000.bin"));

    let mut sentinel = actual_added_slot();
    sentinel.legacy = SlotValue::Interval((None, None));
    let failures = check_expected_deltas("deck", &[sentinel], &[], &[], &expected).unwrap_err();
    assert_eq!(failures.len(), 2);

    let mut dropped = actual_added_slot();
    dropped.legacy = dropped.migrated;
    dropped.migrated = SlotValue::Absent;
    let failures = check_expected_deltas("deck", &[dropped], &[], &[], &expected).unwrap_err();
    assert_eq!(failures.len(), 2);
}

#[test]
fn all_rows_coefficients_list_every_row_and_fail_both_ways() {
    let expected = doctored_deck(&[], &[listed_added_coefficient()]);
    let rows = [actual_added_coefficient(0), actual_added_coefficient(1)];
    check_expected_deltas("deck", &[], &rows, &[], &expected).unwrap();

    let mut boundary_row = actual_added_coefficient(2);
    boundary_row.boundary = true;
    check_expected_deltas("deck", &[], &[boundary_row], &[], &expected).unwrap();

    let removed = doctored_deck(&[], &[]);
    let failures = check_expected_deltas("deck", &[], &rows, &[], &removed).unwrap_err();
    assert_eq!(failures.len(), 2);
    assert!(failures.iter().all(|f| f.starts_with("unexpected delta:")));

    let spurious = check_expected_deltas("deck", &[], &[], &[], &expected).unwrap_err();
    assert_eq!(spurious.len(), 1);
    assert!(spurious[0].starts_with("missing expected delta:"));

    let mut stray = actual_added_coefficient(1);
    stray.migrated = 0.5;
    let failures =
        check_expected_deltas("deck", &[], &[rows[0].clone(), stray], &[], &expected).unwrap_err();
    assert_eq!(failures.len(), 1);
    assert!(failures[0].starts_with("unexpected delta:"));
}

#[test]
fn a_boundary_rows_entry_does_not_cover_generated_rows() {
    let expected = doctored_deck(&[], &[listed_coefficient()]);
    let mut generated = actual_coefficient();
    generated.boundary = false;
    let failures = check_expected_deltas("deck", &[], &[generated], &[], &expected).unwrap_err();
    assert_eq!(failures.len(), 2);
}

#[test]
fn bases_fail_both_ways_and_on_a_different_column_or_status() {
    let expected = deck_with_bases(&[listed_basis()]);
    check_expected_deltas("deck", &[], &[], &[actual_basis()], &expected).unwrap();

    let removed = deck_with_bases(&[]);
    let failures =
        check_expected_deltas("deck", &[], &[], &[actual_basis()], &removed).unwrap_err();
    assert_eq!(failures.len(), 1);
    assert!(failures[0].starts_with(
        "unexpected delta: deck policy/basis/005.bin scenario 0 column 4: Upper -> Lower"
    ));

    let spurious = check_expected_deltas("deck", &[], &[], &[], &expected).unwrap_err();
    assert_eq!(spurious.len(), 1);
    assert!(
        spurious[0].starts_with(
            "missing expected delta: deck policy/basis/005.bin column 4: Upper -> Lower"
        )
    );

    let mut other_column = actual_basis();
    other_column.column = 3;
    let mut other_status = actual_basis();
    other_status.migrated = 1;
    for actual in [other_column, other_status] {
        let failures = check_expected_deltas("deck", &[], &[], &[actual], &expected).unwrap_err();
        assert_eq!(failures.len(), 2);
    }
}

fn basis_entry(scenario: u32, columns: &[u8], rows: &[u8], slots: &[u32]) -> OwnedBasisEntry {
    OwnedBasisEntry {
        scenario,
        column_status: columns.to_vec(),
        row_status: rows.to_vec(),
        piece_row_slots: slots.to_vec(),
    }
}

fn basis(stage_id: u32, iteration: u32, entries: Vec<OwnedBasisEntry>) -> OwnedPolicyBasisRecord {
    OwnedPolicyBasisRecord {
        stage_id,
        iteration,
        entries,
    }
}

#[test]
fn diff_bases_reports_only_the_changed_column_statuses() {
    let file = "policy/basis/005.bin";
    let legacy = basis(
        5,
        5,
        vec![basis_entry(0, &[1, 1, 1, 1, 2], &[1, 0], &[4, 7])],
    );
    assert_eq!(diff_bases(file, &legacy, &legacy).unwrap(), vec![]);

    let migrated = basis(
        5,
        5,
        vec![basis_entry(0, &[1, 1, 1, 1, 0], &[1, 0], &[4, 7])],
    );
    assert_eq!(
        diff_bases(file, &legacy, &migrated).unwrap(),
        vec![BasisDelta {
            file: file.to_string(),
            scenario: 0,
            column: 4,
            legacy: 2,
            migrated: 0,
        }]
    );
}

#[test]
fn diff_bases_refuses_a_difference_outside_the_column_statuses() {
    let file = "policy/basis/005.bin";
    let entry = basis_entry(0, &[1, 2], &[1, 0], &[4, 7]);
    let legacy = basis(5, 5, vec![entry.clone()]);
    let differs =
        |migrated: &OwnedPolicyBasisRecord| diff_bases(file, &legacy, migrated).unwrap_err();

    assert!(differs(&basis(6, 5, vec![entry.clone()])).contains("(stage, iteration, entries)"));
    assert!(differs(&basis(5, 6, vec![entry.clone()])).contains("(stage, iteration, entries)"));
    assert!(differs(&basis(5, 5, vec![entry.clone(), entry.clone()])).contains("entries"));
    for other in [
        basis_entry(1, &[1, 2], &[1, 0], &[4, 7]),
        basis_entry(0, &[1, 2, 2], &[1, 0], &[4, 7]),
        basis_entry(0, &[1, 2], &[1, 1], &[4, 7]),
        basis_entry(0, &[1, 2], &[1, 0], &[4]),
        basis_entry(0, &[1, 2], &[1, 0], &[4, 8]),
    ] {
        assert!(differs(&basis(5, 5, vec![other])).contains("beyond its column statuses"));
    }
}

fn paths(files: &[&str]) -> BTreeSet<String> {
    files.iter().map(|file| (*file).to_string()).collect()
}

fn listed_files(files: &[&str]) -> Vec<String> {
    files.iter().map(|file| (*file).to_string()).collect()
}

#[test]
fn files_fail_both_ways() {
    let differing = paths(&["training/scaling_report.json", "policy/basis/000.bin"]);
    let listed = listed_files(&["training/scaling_report.json", "policy/basis/000.bin"]);
    assert!(check_files("deck", &differing, &BTreeSet::new(), &listed).is_empty());

    let removed = listed_files(&["policy/basis/000.bin"]);
    assert_eq!(
        check_files("deck", &differing, &BTreeSet::new(), &removed),
        ["unexpected delta: deck training/scaling_report.json differs from the baseline"]
    );

    let spurious = listed_files(&[
        "training/scaling_report.json",
        "policy/basis/000.bin",
        "training/convergence.parquet",
    ]);
    assert_eq!(
        check_files("deck", &differing, &BTreeSet::new(), &spurious),
        ["missing expected delta: deck training/convergence.parquet equals the baseline"]
    );
}

#[test]
fn a_decoded_file_is_exempt_from_the_hash_check_but_a_listed_one_must_still_differ() {
    let differing = paths(&["policy/cuts/001.bin"]);
    let exempt = BTreeSet::from(["policy/cuts/001.bin"]);
    assert!(check_files("deck", &differing, &exempt, &[]).is_empty());

    let listed = listed_files(&["policy/cuts/000.bin"]);
    assert_eq!(
        check_files(
            "deck",
            &BTreeSet::new(),
            &BTreeSet::from(["policy/cuts/000.bin"]),
            &listed
        ),
        ["missing expected delta: deck policy/cuts/000.bin equals the baseline"]
    );
}

#[test]
fn expected_deck_names_the_files_it_decodes() {
    let expected: ExpectedDeck = serde_json::from_value(json!({
        "water_bucket_stages": [],
        "slots": [listed_slot(), listed_added_slot()],
        "coefficients": [listed_added_coefficient()],
        "bases": [listed_basis()],
    }))
    .unwrap();
    assert_eq!(
        expected.cut_files(),
        BTreeSet::from(["policy/cuts/000.bin", "policy/cuts/005.bin"])
    );
    assert_eq!(
        expected.basis_files(),
        BTreeSet::from(["policy/basis/005.bin"])
    );
}

#[test]
fn a_slot_value_other_than_absent_or_an_interval_is_refused() {
    let refused = serde_json::from_value::<ExpectedDeck>(json!({
        "water_bucket_stages": [],
        "slots": [{
            "file": "policy/cuts/000.bin", "entity_id": 1, "subindex": 3,
            "legacy": "missing", "migrated": [null, null],
        }],
        "coefficients": [],
    }));
    assert!(refused.is_err());
}
