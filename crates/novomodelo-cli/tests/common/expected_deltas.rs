//! Diffs of two `policy/cuts/NNN.bin` and `policy/basis/NNN.bin` reads, and of two
//! output-file and LP-fact sets, checked against the committed list in
//! `fixtures/bit_identity/phase2_expected_deltas.json`.

use std::collections::{BTreeMap, BTreeSet};

use novomodelo_io::{
    ENTITY_SLOT_DATE_SENTINEL, EntitySlot, OwnedBasisEntry, OwnedPolicyBasisRecord,
    StageCutsReadResult, StateFamily,
};
use novomodelo_solver::BasisStatus;
use serde::Deserialize;

pub type Interval = (Option<i32>, Option<i32>);

type SlotKey = (u8, i32, u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "RawSlotValue")]
pub enum SlotValue {
    Absent,
    Interval(Interval),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawSlotValue {
    Marker(String),
    Interval(Interval),
}

impl TryFrom<RawSlotValue> for SlotValue {
    type Error = String;

    fn try_from(raw: RawSlotValue) -> Result<Self, String> {
        match raw {
            RawSlotValue::Marker(marker) if marker == "absent" => Ok(Self::Absent),
            RawSlotValue::Marker(marker) => Err(format!(
                "slot value {marker:?} is neither `absent` nor an interval"
            )),
            RawSlotValue::Interval(interval) => Ok(Self::Interval(interval)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotDelta {
    pub file: String,
    pub entity_type: u8,
    pub entity_id: i32,
    pub subindex: u32,
    pub legacy: SlotValue,
    pub migrated: SlotValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoefficientDelta {
    pub file: String,
    pub row: u32,
    pub boundary: bool,
    pub entity_type: u8,
    pub entity_id: i32,
    pub subindex: u32,
    pub legacy: Option<f64>,
    pub migrated: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisDelta {
    pub file: String,
    pub scenario: u32,
    pub column: usize,
    pub legacy: u8,
    pub migrated: u8,
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
    #[serde(default)]
    pub bases: Vec<ExpectedBasis>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub lp_facts: Vec<String>,
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
    pub legacy: SlotValue,
    pub migrated: SlotValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpectedRows {
    Boundary,
    All,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedCoefficient {
    pub file: String,
    pub slot: ExpectedSlotKey,
    pub rows: ExpectedRows,
    pub migrated: f64,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedSlotKey {
    pub entity_id: i32,
    pub subindex: u32,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedBasis {
    pub file: String,
    pub column: usize,
    pub legacy: String,
    pub migrated: String,
}

impl ExpectedDeck {
    pub fn cut_files(&self) -> BTreeSet<&str> {
        self.slots
            .iter()
            .map(|slot| slot.file.as_str())
            .chain(self.coefficients.iter().map(|c| c.file.as_str()))
            .collect()
    }

    pub fn basis_files(&self) -> BTreeSet<&str> {
        self.bases.iter().map(|basis| basis.file.as_str()).collect()
    }
}

fn key(slot: &EntitySlot) -> SlotKey {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

fn interval(slot: &EntitySlot) -> Interval {
    let date = |date: i32| (date != ENTITY_SLOT_DATE_SENTINEL).then_some(date);
    (date(slot.interval_start), date(slot.interval_end))
}

fn show_interval(interval: Interval) -> String {
    let show = |date: Option<i32>| date.map_or_else(|| "sentinel".to_string(), |d| d.to_string());
    format!("[{}, {})", show(interval.0), show(interval.1))
}

fn show_value(value: SlotValue) -> String {
    match value {
        SlotValue::Absent => "absent".to_string(),
        SlotValue::Interval(interval) => show_interval(interval),
    }
}

fn slot_label(deck: &str, file: &str, entity_type: u8, entity_id: i32, subindex: u32) -> String {
    let family = StateFamily::from_code(entity_type).map_or_else(
        || format!("entity_type {entity_type}"),
        |family| format!("{family:?}"),
    );
    format!("{deck} {file} {family} slot (entity {entity_id}, subindex {subindex})")
}

fn shape(entry: &OwnedBasisEntry) -> (u32, usize, &[u8], &[u32]) {
    (
        entry.scenario,
        entry.column_status.len(),
        &entry.row_status,
        &entry.piece_row_slots,
    )
}

fn status_name(code: u8) -> String {
    format!("{:?}", BasisStatus::from_discriminant_code(code))
}

fn slots_by_key<'a>(
    file: &str,
    side: &str,
    manifest: &'a [EntitySlot],
) -> Result<BTreeMap<SlotKey, &'a EntitySlot>, String> {
    let mut by_key = BTreeMap::new();
    for slot in manifest {
        if by_key.insert(key(slot), slot).is_some() {
            let (entity_type, entity_id, subindex) = key(slot);
            return Err(format!(
                "{file}: slot (type {entity_type}, entity {entity_id}, subindex {subindex}) appears twice in the {side} manifest"
            ));
        }
    }
    Ok(by_key)
}

/// Slots pair by (entity type, entity, subindex); a slot only one manifest
/// holds is a delta against [`SlotValue::Absent`].
pub fn diff_manifests(
    file: &str,
    legacy: &[EntitySlot],
    migrated: &[EntitySlot],
) -> Result<Vec<SlotDelta>, String> {
    let before = slots_by_key(file, "legacy", legacy)?;
    let after = slots_by_key(file, "migrated", migrated)?;
    let identity = |slot: &EntitySlot| (slot.was_active, slot.reference_date);
    let value = |slot: Option<&&EntitySlot>| {
        slot.map_or(SlotValue::Absent, |slot| {
            SlotValue::Interval(interval(slot))
        })
    };
    let mut deltas = Vec::new();
    for slot_key in before.keys().chain(after.keys()).collect::<BTreeSet<_>>() {
        let (was, now) = (before.get(slot_key), after.get(slot_key));
        if let (Some(was), Some(now)) = (was, now)
            && identity(was) != identity(now)
        {
            return Err(format!(
                "{file}: slot (type {}, entity {}, subindex {}) differs beyond its interval: {:?} (legacy) vs {:?} (migrated)",
                slot_key.0,
                slot_key.1,
                slot_key.2,
                identity(was),
                identity(now)
            ));
        }
        if value(was) != value(now) {
            deltas.push(SlotDelta {
                file: file.to_string(),
                entity_type: slot_key.0,
                entity_id: slot_key.1,
                subindex: slot_key.2,
                legacy: value(was),
                migrated: value(now),
            });
        }
    }
    Ok(deltas)
}

/// Cuts pair by position and coefficients by slot key; the boundary prefix
/// (`warm_start_count`) comes from `migrated`. The caller guarantees both pools
/// hold the same number of cuts.
pub fn diff_coefficients(
    file: &str,
    legacy: &StageCutsReadResult,
    migrated: &StageCutsReadResult,
) -> Vec<CoefficientDelta> {
    let position: BTreeMap<SlotKey, usize> = legacy
        .entity_manifest
        .iter()
        .enumerate()
        .map(|(position, slot)| (key(slot), position))
        .collect();
    let mut deltas = Vec::new();
    for (before, after) in legacy.cuts.iter().zip(&migrated.cuts) {
        let boundary = after.slot_index < migrated.warm_start_count;
        for (now, slot) in after.coefficients.iter().zip(&migrated.entity_manifest) {
            let was = position
                .get(&key(slot))
                .and_then(|position| before.coefficients.get(*position))
                .copied();
            if was.is_none_or(|was| was.to_bits() != now.to_bits()) {
                deltas.push(CoefficientDelta {
                    file: file.to_string(),
                    row: after.slot_index,
                    boundary,
                    entity_type: slot.entity_type,
                    entity_id: slot.entity_id,
                    subindex: slot.subindex,
                    legacy: was,
                    migrated: *now,
                });
            }
        }
    }
    deltas
}

/// Everything but the column statuses must be equal: stage, iteration, entry
/// count, and per entry the scenario, column count, row statuses and the piece
/// row slots.
pub fn diff_bases(
    file: &str,
    legacy: &OwnedPolicyBasisRecord,
    migrated: &OwnedPolicyBasisRecord,
) -> Result<Vec<BasisDelta>, String> {
    let header =
        |basis: &OwnedPolicyBasisRecord| (basis.stage_id, basis.iteration, basis.entries.len());
    if header(legacy) != header(migrated) {
        return Err(format!(
            "{file}: (stage, iteration, entries) is {:?} (legacy) vs {:?} (migrated)",
            header(legacy),
            header(migrated)
        ));
    }
    let mut deltas = Vec::new();
    for (before, after) in legacy.entries.iter().zip(&migrated.entries) {
        if shape(before) != shape(after) {
            return Err(format!(
                "{file}: scenario {} differs beyond its column statuses (scenario, column count, row statuses, piece row slots)",
                before.scenario
            ));
        }
        for (column, (was, now)) in before
            .column_status
            .iter()
            .zip(&after.column_status)
            .enumerate()
        {
            if was != now {
                deltas.push(BasisDelta {
                    file: file.to_string(),
                    scenario: after.scenario,
                    column,
                    legacy: *was,
                    migrated: *now,
                });
            }
        }
    }
    Ok(deltas)
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
        && (expected.rows == ExpectedRows::All || actual.boundary)
        && actual.entity_type == StateFamily::HydroTransitBucket.code()
        && actual.entity_id == expected.slot.entity_id
        && actual.subindex == expected.slot.subindex
        && actual.migrated.to_bits() == expected.migrated.to_bits()
}

fn basis_is_listed(actual: &BasisDelta, expected: &ExpectedBasis) -> bool {
    actual.file == expected.file
        && actual.column == expected.column
        && status_name(actual.legacy) == expected.legacy
        && status_name(actual.migrated) == expected.migrated
}

fn reconcile<A, E>(
    actual: &[A],
    listed: &[E],
    is_listed: fn(&A, &E) -> bool,
    unexpected: impl Fn(&A) -> String,
    missing: impl Fn(&E) -> String,
) -> Vec<String> {
    let unexpected = actual
        .iter()
        .filter(|a| !listed.iter().any(|e| is_listed(a, e)))
        .map(|a| format!("unexpected delta: {}", unexpected(a)));
    let missing = listed
        .iter()
        .filter(|e| !actual.iter().any(|a| is_listed(a, e)))
        .map(|e| format!("missing expected delta: {}", missing(e)));
    unexpected.chain(missing).collect()
}

pub fn check_expected_deltas(
    deck: &str,
    actual_slots: &[SlotDelta],
    actual_coeffs: &[CoefficientDelta],
    actual_bases: &[BasisDelta],
    expected: &ExpectedDeck,
) -> Result<(), Vec<String>> {
    let bucket = StateFamily::HydroTransitBucket.code();
    let mut failures = reconcile(
        actual_slots,
        &expected.slots,
        slot_is_listed,
        |a| {
            format!(
                "{} {} -> {}",
                slot_label(deck, &a.file, a.entity_type, a.entity_id, a.subindex),
                show_value(a.legacy),
                show_value(a.migrated)
            )
        },
        |e| {
            format!(
                "{} {} -> {}",
                slot_label(deck, &e.file, bucket, e.entity_id, e.subindex),
                show_value(e.legacy),
                show_value(e.migrated)
            )
        },
    );
    failures.extend(reconcile(
        actual_coeffs,
        &expected.coefficients,
        coefficient_is_listed,
        |a| {
            format!(
                "{} coefficient of {} row {}: {:?} -> {:?}",
                slot_label(deck, &a.file, a.entity_type, a.entity_id, a.subindex),
                if a.boundary { "boundary" } else { "generated" },
                a.row,
                a.legacy,
                a.migrated
            )
        },
        |e| {
            format!(
                "{} coefficient of {:?} rows -> {:?}",
                slot_label(deck, &e.file, bucket, e.slot.entity_id, e.slot.subindex),
                e.rows,
                e.migrated
            )
        },
    ));
    failures.extend(reconcile(
        actual_bases,
        &expected.bases,
        basis_is_listed,
        |a| {
            format!(
                "{deck} {} scenario {} column {}: {} -> {}",
                a.file,
                a.scenario,
                a.column,
                status_name(a.legacy),
                status_name(a.migrated)
            )
        },
        |e| {
            format!(
                "{deck} {} column {}: {} -> {}",
                e.file, e.column, e.legacy, e.migrated
            )
        },
    ));
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// `differing` holds every output path whose hash differs from the baseline;
/// `exempt` holds the paths compared by decoding instead. Every other differing
/// path must be listed, and every listed path must differ.
pub fn check_files(
    deck: &str,
    differing: &BTreeSet<String>,
    exempt: &BTreeSet<&str>,
    listed: &[String],
) -> Vec<String> {
    let unexpected = differing
        .iter()
        .filter(|path| !exempt.contains(path.as_str()) && !listed.contains(path))
        .map(|path| format!("unexpected delta: {deck} {path} differs from the baseline"));
    let missing = listed
        .iter()
        .filter(|path| !differing.contains(*path))
        .map(|path| format!("missing expected delta: {deck} {path} equals the baseline"));
    unexpected.chain(missing).collect()
}
