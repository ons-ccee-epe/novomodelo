//! Diffs of two `policy/cuts/NNN.bin` reads, checked against the committed list
//! in `fixtures/bit_identity/phase2_expected_deltas.json`.

use std::collections::BTreeMap;

use novomodelo_io::{ENTITY_SLOT_DATE_SENTINEL, EntitySlot, StageCutsReadResult, StateFamily};
use serde::Deserialize;

pub type Interval = (Option<i32>, Option<i32>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotDelta {
    pub file: String,
    pub entity_type: u8,
    pub entity_id: i32,
    pub subindex: u32,
    pub legacy: Interval,
    pub migrated: Interval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoefficientDelta {
    pub file: String,
    pub row: u32,
    pub boundary: bool,
    pub entity_type: u8,
    pub entity_id: i32,
    pub subindex: u32,
    pub legacy: f64,
    pub migrated: f64,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedList {
    pub frozen_before: String,
    pub decks: BTreeMap<String, ExpectedDeck>,
    pub informational: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedDeck {
    pub water_bucket_stages: Vec<ExpectedWaterStage>,
    pub slots: Vec<ExpectedSlot>,
    pub coefficients: Vec<ExpectedCoefficient>,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedWaterStage {
    pub start_date: String,
    pub duration_hours: f64,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedSlot {
    pub file: String,
    pub entity_id: i32,
    pub subindex: u32,
    pub legacy: Interval,
    pub migrated: Interval,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedCoefficient {
    pub file: String,
    pub slot: ExpectedSlotKey,
    pub rows: String,
    pub migrated: f64,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedSlotKey {
    pub entity_id: i32,
    pub subindex: u32,
}

const BOUNDARY_ROWS: &str = "boundary";

fn interval(slot: &EntitySlot) -> Interval {
    let date = |date: i32| (date != ENTITY_SLOT_DATE_SENTINEL).then_some(date);
    (date(slot.interval_start), date(slot.interval_end))
}

fn show_interval(interval: Interval) -> String {
    let show = |date: Option<i32>| date.map_or_else(|| "sentinel".to_string(), |d| d.to_string());
    format!("[{}, {})", show(interval.0), show(interval.1))
}

fn slot_label(deck: &str, file: &str, entity_type: u8, entity_id: i32, subindex: u32) -> String {
    let family = StateFamily::from_code(entity_type).map_or_else(
        || format!("entity_type {entity_type}"),
        |family| format!("{family:?}"),
    );
    format!("{deck} {file} {family} slot (entity {entity_id}, subindex {subindex})")
}

pub fn diff_manifests(
    file: &str,
    legacy: &[EntitySlot],
    migrated: &[EntitySlot],
) -> Result<Vec<SlotDelta>, String> {
    if legacy.len() != migrated.len() {
        return Err(format!(
            "{file}: manifest length {} (legacy) != {} (migrated)",
            legacy.len(),
            migrated.len()
        ));
    }
    let identity = |slot: &EntitySlot| {
        (
            slot.entity_type,
            slot.entity_id,
            slot.subindex,
            slot.was_active,
            slot.reference_date,
        )
    };
    let mut deltas = Vec::new();
    for (position, (before, after)) in legacy.iter().zip(migrated).enumerate() {
        if identity(before) != identity(after) {
            return Err(format!(
                "{file}: slot {position} differs beyond its interval: {:?} (legacy) vs {:?} (migrated)",
                identity(before),
                identity(after)
            ));
        }
        if interval(before) != interval(after) {
            deltas.push(SlotDelta {
                file: file.to_string(),
                entity_type: after.entity_type,
                entity_id: after.entity_id,
                subindex: after.subindex,
                legacy: interval(before),
                migrated: interval(after),
            });
        }
    }
    Ok(deltas)
}

/// Cuts pair by position and coefficients by index; slot identity and the
/// boundary prefix (`warm_start_count`) come from `migrated`. The caller
/// guarantees both pools hold the same cuts over the same manifest.
pub fn diff_coefficients(
    file: &str,
    legacy: &StageCutsReadResult,
    migrated: &StageCutsReadResult,
) -> Vec<CoefficientDelta> {
    let mut deltas = Vec::new();
    for (before, after) in legacy.cuts.iter().zip(&migrated.cuts) {
        let boundary = after.slot_index < migrated.warm_start_count;
        for ((was, now), slot) in before
            .coefficients
            .iter()
            .zip(&after.coefficients)
            .zip(&migrated.entity_manifest)
        {
            if was.to_bits() != now.to_bits() {
                deltas.push(CoefficientDelta {
                    file: file.to_string(),
                    row: after.slot_index,
                    boundary,
                    entity_type: slot.entity_type,
                    entity_id: slot.entity_id,
                    subindex: slot.subindex,
                    legacy: *was,
                    migrated: *now,
                });
            }
        }
    }
    deltas
}

fn slot_is_listed(actual: &SlotDelta, expected: &ExpectedSlot) -> bool {
    actual.file == expected.file
        && actual.entity_type == StateFamily::HydroTransitBucket.code()
        && actual.entity_id == expected.entity_id
        && actual.subindex == expected.subindex
        && actual.legacy == expected.legacy
        && actual.migrated == expected.migrated
}

fn coefficient_is_listed(actual: &CoefficientDelta, expected: &ExpectedCoefficient) -> bool {
    actual.file == expected.file
        && expected.rows == BOUNDARY_ROWS
        && actual.boundary
        && actual.entity_type == StateFamily::HydroTransitBucket.code()
        && actual.entity_id == expected.slot.entity_id
        && actual.subindex == expected.slot.subindex
        && actual.migrated.to_bits() == expected.migrated.to_bits()
}

pub fn check_expected_deltas(
    deck: &str,
    actual_slots: &[SlotDelta],
    actual_coeffs: &[CoefficientDelta],
    expected: &ExpectedDeck,
) -> Result<(), Vec<String>> {
    let bucket = StateFamily::HydroTransitBucket.code();
    let mut failures = Vec::new();
    for actual in actual_slots {
        if !expected.slots.iter().any(|e| slot_is_listed(actual, e)) {
            failures.push(format!(
                "unexpected delta: {} {} -> {}",
                slot_label(
                    deck,
                    &actual.file,
                    actual.entity_type,
                    actual.entity_id,
                    actual.subindex
                ),
                show_interval(actual.legacy),
                show_interval(actual.migrated)
            ));
        }
    }
    for actual in actual_coeffs {
        if !expected
            .coefficients
            .iter()
            .any(|e| coefficient_is_listed(actual, e))
        {
            failures.push(format!(
                "unexpected delta: {} coefficient of {} row {}: {:?} -> {:?}",
                slot_label(
                    deck,
                    &actual.file,
                    actual.entity_type,
                    actual.entity_id,
                    actual.subindex
                ),
                if actual.boundary {
                    BOUNDARY_ROWS
                } else {
                    "generated"
                },
                actual.row,
                actual.legacy,
                actual.migrated
            ));
        }
    }
    for listed in &expected.slots {
        if !actual_slots.iter().any(|a| slot_is_listed(a, listed)) {
            failures.push(format!(
                "missing expected delta: {} {} -> {}",
                slot_label(
                    deck,
                    &listed.file,
                    bucket,
                    listed.entity_id,
                    listed.subindex
                ),
                show_interval(listed.legacy),
                show_interval(listed.migrated)
            ));
        }
    }
    for listed in &expected.coefficients {
        if !actual_coeffs
            .iter()
            .any(|a| coefficient_is_listed(a, listed))
        {
            failures.push(format!(
                "missing expected delta: {} coefficient of {} rows -> {:?}",
                slot_label(
                    deck,
                    &listed.file,
                    bucket,
                    listed.slot.entity_id,
                    listed.slot.subindex
                ),
                listed.rows,
                listed.migrated
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}
