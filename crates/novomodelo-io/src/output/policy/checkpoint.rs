//! Filesystem write and read entry points for value-function artifacts.
//!
//! `manifest.bin` is written last so its presence is the commit signal of a
//! complete artifact, and it carries the `format_version` marker the reader
//! checks first — an artifact without it is cleanly rejected before any payload
//! is parsed.

use std::collections::BTreeMap;
use std::fs::FileType;
use std::io;
use std::path::{Path, PathBuf};

use flatbuffers::FLATBUFFERS_MAX_BUFFER_SIZE;

use super::super::atomic::{
    ensure_parent_dir, sync_dir, write_bytes_atomic_synced, write_bytes_synced,
};
use super::super::error::OutputError;
use super::codec::{
    build_checkpoint_manifest, build_stage_basis, build_stage_cuts, build_stage_states,
    checkpoint_manifest_size_bound, deserialize_checkpoint_manifest, deserialize_stage_basis,
    deserialize_stage_cuts, deserialize_stage_states, read_sorted_bin_files,
    stage_basis_size_bound, stage_cuts_size_bound, stage_states_size_bound,
};
use super::records::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, ManifestNode,
    OwnedPolicyBasisRecord, PolicyBasisRecord, PolicyCheckpoint, StageCutsPayload,
    StageCutsReadResult, StageStatesPayload, StageStatesReadResult, StateFamily, decode_slot_date,
};

fn is_well_formed_slot_date(value: i32) -> bool {
    value == ENTITY_SLOT_DATE_SENTINEL || decode_slot_date(value).is_some()
}

fn slot_date_error(pool_id: u32, slot: &EntitySlot, detail: &str) -> OutputError {
    OutputError::serialization(
        "policy_checkpoint_dates",
        format!(
            "pool {pool_id} entity {} subindex {} {detail}",
            slot.entity_id, slot.subindex
        ),
    )
}

fn check_well_formed_slot_dates(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    for (field_name, value) in [
        ("reference_date", slot.reference_date),
        ("interval_start", slot.interval_start),
        ("interval_end", slot.interval_end),
    ] {
        if !is_well_formed_slot_date(value) {
            return Err(slot_date_error(
                pool_id,
                slot,
                &format!("carries malformed {field_name} {value}"),
            ));
        }
    }
    Ok(())
}

fn check_interval_pairing(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let start_live = slot.interval_start != ENTITY_SLOT_DATE_SENTINEL;
    let end_live = slot.interval_end != ENTITY_SLOT_DATE_SENTINEL;
    match (start_live, end_live) {
        (true, false) => Err(slot_date_error(
            pool_id,
            slot,
            "carries a live interval_start with no interval_end",
        )),
        (false, true) => Err(slot_date_error(
            pool_id,
            slot,
            "carries a live interval_end with no interval_start",
        )),
        _ => Ok(()),
    }
}

fn check_interval_ordering(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let start = slot.interval_start;
    let end = slot.interval_end;
    if start != ENTITY_SLOT_DATE_SENTINEL && end != ENTITY_SLOT_DATE_SENTINEL && start >= end {
        return Err(slot_date_error(
            pool_id,
            slot,
            &format!("carries interval_start {start} not before interval_end {end}"),
        ));
    }
    Ok(())
}

fn check_family_applicability(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let interval_live = slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
        || slot.interval_end != ENTITY_SLOT_DATE_SENTINEL;
    let reference_live = slot.reference_date != ENTITY_SLOT_DATE_SENTINEL;
    match slot.family() {
        Some(StateFamily::HydroStorage) => {
            if reference_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    &format!(
                        "is a storage slot, which carries no per-slot date, but carries a live reference_date {}",
                        slot.reference_date
                    ),
                ));
            }
            if interval_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    "is a storage slot, which carries no per-slot date, but carries a live interval",
                ));
            }
        }
        Some(StateFamily::HydroInflowLag) => {
            if interval_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    "is an inflow-lag slot, which carries no interval, but carries a live interval",
                ));
            }
        }
        other => {
            if reference_live {
                let noun = match other {
                    Some(StateFamily::HydroTransitBucket) => "a transit-bucket slot",
                    Some(StateFamily::AnticipatedThermalState) => {
                        "an anticipated-thermal-state slot"
                    }
                    _ => "a slot with no recognized family",
                };
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    &format!(
                        "is {noun}, which carries no reference_date (only inflow-lag slots do), but carries a live reference_date {}",
                        slot.reference_date
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Non-sentinel `interval_start`s must be monotone non-decreasing in `subindex`
/// for [`StateFamily::HydroTransitBucket`] slots. Only this family is checked:
/// the other calendar-shaped family's modular delivery-target-residue `subindex`
/// wraps across the horizon, so monotonicity there would reject valid dates.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the pool, the offending
/// subindex, and its `interval_start`.
fn check_transit_bucket_monotonicity(
    pool_id: u32,
    manifest: &[EntitySlot],
) -> Result<(), OutputError> {
    let mut by_entity: BTreeMap<i32, Vec<(u32, i32)>> = BTreeMap::new();
    for slot in manifest {
        if slot.family() == Some(StateFamily::HydroTransitBucket)
            && slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
        {
            by_entity
                .entry(slot.entity_id)
                .or_default()
                .push((slot.subindex, slot.interval_start));
        }
    }
    for starts in by_entity.values_mut() {
        starts.sort_by_key(|&(subindex, _)| subindex);
        for pair in starts.windows(2) {
            let (prev_subindex, prev_start) = pair[0];
            let (subindex, start) = pair[1];
            if start < prev_start {
                return Err(OutputError::serialization(
                    "policy_checkpoint_dates",
                    format!(
                        "pool {pool_id} subindex {subindex} carries interval_start {start}, \
                         earlier than subindex {prev_subindex}'s {prev_start}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validate that one pool's entity manifest is date-consistent, returning the
/// first violation found.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the pool and the
/// offending slot.
fn check_pool_dates(pool_id: u32, manifest: &[EntitySlot]) -> Result<(), OutputError> {
    // Canonical order so the first reported error is declaration-order invariant.
    let mut slots: Vec<&EntitySlot> = manifest.iter().collect();
    slots.sort_by_key(|slot| (slot.entity_type, slot.entity_id, slot.subindex));
    for slot in slots {
        check_well_formed_slot_dates(pool_id, slot)?;
        check_interval_pairing(pool_id, slot)?;
        check_interval_ordering(pool_id, slot)?;
        check_family_applicability(pool_id, slot)?;
    }
    check_transit_bucket_monotonicity(pool_id, manifest)
}

/// Zero-padded `.bin` name for stable on-disk sort. Identity comes from the
/// buffer content, never this filename.
fn bin_file_name(id: u32) -> String {
    format!("{id:03}.bin")
}

fn structure_error(dir: &str, id: u32, detail: &str) -> OutputError {
    OutputError::serialization(
        "policy_checkpoint_structure",
        format!("{dir}/{}: {detail}", bin_file_name(id)),
    )
}

fn manifest_structure_error(detail: &str) -> OutputError {
    OutputError::serialization(
        "policy_checkpoint_structure",
        format!("manifest.bin: {detail}"),
    )
}

/// The first adjacent pair of `values` that is not strictly ascending.
fn first_not_ascending(values: impl IntoIterator<Item = u32>) -> Option<(u32, u32)> {
    let mut values = values.into_iter();
    let mut previous = values.next()?;
    for next in values {
        if previous >= next {
            return Some((previous, next));
        }
        previous = next;
    }
    None
}

/// One pool's `cuts/` payload as the rules read it: record slots and scored
/// slots in record order, and each record's counters.
struct PoolView<'a> {
    pool_id: u32,
    capacity: u32,
    populated_count: u32,
    entity_manifest: &'a [EntitySlot],
    slots: Vec<u32>,
    scored_slots: Vec<u32>,
    counters: Vec<(u32, Option<u64>, Option<u64>)>,
}

/// A record's `(slot, scored, active_count, last_active_iteration)`.
type RecordFacts = (u32, bool, Option<u64>, Option<u64>);

impl<'a> PoolView<'a> {
    fn new(
        pool_id: u32,
        capacity: u32,
        populated_count: u32,
        entity_manifest: &'a [EntitySlot],
        records: impl Iterator<Item = RecordFacts>,
    ) -> Self {
        let mut slots = Vec::new();
        let mut scored_slots = Vec::new();
        let mut counters = Vec::new();
        for (slot, scored, active_count, last_active_iteration) in records {
            slots.push(slot);
            if scored {
                scored_slots.push(slot);
            }
            counters.push((slot, active_count, last_active_iteration));
        }
        Self {
            pool_id,
            capacity,
            populated_count,
            entity_manifest,
            slots,
            scored_slots,
            counters,
        }
    }
}

struct StatesView<'a> {
    pool_id: u32,
    count: u32,
    state_dimension: u32,
    data_len: usize,
    slots: &'a [u32],
}

struct EntryView<'a> {
    scenario: u32,
    row_count: usize,
    piece_row_slots: &'a [u32],
}

struct BasisView<'a> {
    node: u32,
    entries: Vec<EntryView<'a>>,
}

fn find_pool<'a, 'b>(pools: &'b [PoolView<'a>], pool_id: u32) -> Option<&'b PoolView<'a>> {
    pools.iter().find(|pool| pool.pool_id == pool_id)
}

/// The pool id and record slots of the pool at graph position `node`; the
/// slots are empty when no `cuts/` file holds that pool.
fn node_pool<'b>(
    nodes: &[ManifestNode],
    pools: &'b [PoolView<'_>],
    node: u32,
) -> Option<(u32, &'b [u32])> {
    let pool_id = nodes.get(node as usize)?.pool_id;
    let slots = find_pool(pools, pool_id).map_or(&[][..], |pool| pool.slots.as_slice());
    Some((pool_id, slots))
}

fn check_record_slots(
    pool_id: u32,
    capacity: u32,
    populated_count: u32,
    slots: &[u32],
) -> Result<(), OutputError> {
    let fail = |detail: String| structure_error("cuts", pool_id, &detail);
    if let Some((previous, next)) = first_not_ascending(slots.iter().copied()) {
        return Err(fail(format!(
            "records not strictly ascending by slot_index: slot {next} after slot {previous}"
        )));
    }
    if let Some(slot) = slots.iter().find(|&&slot| slot >= capacity) {
        return Err(fail(format!(
            "record slot_index {slot} not below capacity {capacity}"
        )));
    }
    if populated_count as usize != slots.len() {
        return Err(fail(format!(
            "populated_count {populated_count} differs from its {} records",
            slots.len()
        )));
    }
    Ok(())
}

/// `scored_slots` ascend once [`check_record_slots`] passed for the pool.
fn check_state_rows(
    pool_id: u32,
    count: u32,
    state_dimension: u32,
    data_len: usize,
    slots: &[u32],
    scored_slots: &[u32],
    manifest_len: usize,
) -> Result<(), OutputError> {
    let fail = |detail: String| structure_error("states", pool_id, &detail);
    if data_len as u64 != u64::from(count) * u64::from(state_dimension) {
        return Err(fail(format!(
            "data holds {data_len} values, not count {count} times state_dimension {state_dimension}"
        )));
    }
    if slots.len() != count as usize {
        return Err(fail(format!("{} slots for count {count}", slots.len())));
    }
    if let Some((previous, next)) = first_not_ascending(slots.iter().copied()) {
        return Err(fail(format!(
            "slots not strictly ascending: slot {next} after slot {previous}"
        )));
    }
    if let Some(slot) = slots
        .iter()
        .find(|slot| scored_slots.binary_search(slot).is_err())
    {
        return Err(fail(format!(
            "slot {slot} has no scored record in the pool's `cuts/` file"
        )));
    }
    if state_dimension as usize != manifest_len {
        return Err(fail(format!(
            "state_dimension {state_dimension} differs from the pool's entity-manifest length {manifest_len}"
        )));
    }
    Ok(())
}

fn check_scenario_order(
    node: u32,
    scenarios: impl IntoIterator<Item = u32>,
) -> Result<(), OutputError> {
    match first_not_ascending(scenarios) {
        Some((previous, next)) => Err(structure_error(
            "basis",
            node,
            &format!(
                "entries not strictly ascending by scenario: scenario {next} after scenario {previous}"
            ),
        )),
        None => Ok(()),
    }
}

/// `node_pool` is [`node_pool`] of the entry's node; its slots ascend once
/// [`check_record_slots`] passed for that pool.
fn check_basis_entry(
    node: u32,
    scenario: u32,
    row_count: usize,
    piece_row_slots: &[u32],
    node_pool: Option<(u32, &[u32])>,
) -> Result<(), OutputError> {
    let fail = |detail: String| structure_error("basis", node, &detail);
    if row_count < piece_row_slots.len() {
        return Err(fail(format!(
            "scenario {scenario} has {} piece-row slots but only {row_count} rows",
            piece_row_slots.len()
        )));
    }
    if piece_row_slots.is_empty() {
        return Ok(());
    }
    let Some((pool_id, pool_slots)) = node_pool else {
        return Err(fail(
            "names a node position the manifest's graph does not hold".to_string(),
        ));
    };
    if let Some(slot) = piece_row_slots
        .iter()
        .find(|slot| pool_slots.binary_search(slot).is_err())
    {
        return Err(fail(format!(
            "scenario {scenario} piece-row slot {slot} has no record in pool {pool_id}'s `cuts/` file"
        )));
    }
    Ok(())
}

fn check_stored_row_total(total_visited_states: u64, stored_rows: u64) -> Result<(), OutputError> {
    if total_visited_states == stored_rows {
        Ok(())
    } else {
        Err(manifest_structure_error(&format!(
            "total_visited_states {total_visited_states} differs from the {stored_rows} rows of the states files"
        )))
    }
}

fn check_counter_presence(
    activity_counters: bool,
    pool_id: u32,
    slot: u32,
    active_count: Option<u64>,
    last_active_iteration: Option<u64>,
) -> Result<(), OutputError> {
    if (active_count.is_some() && last_active_iteration.is_some()) == activity_counters {
        Ok(())
    } else {
        Err(manifest_structure_error(&format!(
            "activity_counters is {activity_counters} but pool {pool_id} slot {slot} carries \
             active_count {active_count:?} and last_active_iteration {last_active_iteration:?}"
        )))
    }
}

/// Everything the date and structure rules read of a checkpoint, borrowed from
/// either the files a read parsed or the payloads a write receives. The rules
/// run in [`CheckpointView::check`] alone, so a rule added there binds both.
struct CheckpointView<'a> {
    pools: Vec<PoolView<'a>>,
    states: Vec<StatesView<'a>>,
    bases: Vec<BasisView<'a>>,
    metadata: &'a CheckpointManifest,
}

impl<'a> CheckpointView<'a> {
    /// The parsed files, already in ascending id order.
    fn of_checkpoint(checkpoint: &'a PolicyCheckpoint) -> Self {
        Self {
            pools: checkpoint
                .stage_cuts
                .iter()
                .map(|pool| {
                    PoolView::new(
                        pool.stage_id,
                        pool.capacity,
                        pool.populated_count,
                        &pool.entity_manifest,
                        pool.cuts.iter().map(|r| {
                            (
                                r.slot_index,
                                r.scored,
                                r.active_count,
                                r.last_active_iteration,
                            )
                        }),
                    )
                })
                .collect(),
            states: checkpoint
                .stage_states
                .iter()
                .map(|s| StatesView {
                    pool_id: s.stage_id,
                    count: s.count,
                    state_dimension: s.state_dimension,
                    data_len: s.data.len(),
                    slots: &s.slots,
                })
                .collect(),
            bases: checkpoint
                .stage_bases
                .iter()
                .map(|b| BasisView {
                    node: b.stage_id,
                    entries: b
                        .entries
                        .iter()
                        .map(|e| EntryView {
                            scenario: e.scenario,
                            row_count: e.row_status.len(),
                            piece_row_slots: &e.piece_row_slots,
                        })
                        .collect(),
                })
                .collect(),
            metadata: &checkpoint.metadata,
        }
    }

    /// The payloads a write receives, in the read's file order so the first
    /// violation reported is the same.
    fn of_payloads(
        stage_cuts: &'a [StageCutsPayload<'_>],
        stage_bases: &'a [PolicyBasisRecord<'_>],
        stage_states: &'a [StageStatesPayload<'_>],
        metadata: &'a CheckpointManifest,
    ) -> Self {
        Self {
            pools: sorted_by_id(stage_cuts, |p| p.stage_id)
                .into_iter()
                .map(|pool| {
                    PoolView::new(
                        pool.stage_id,
                        pool.capacity,
                        pool.populated_count,
                        pool.entity_manifest,
                        pool.cuts.iter().map(|r| {
                            (
                                r.slot_index,
                                r.scored,
                                r.active_count,
                                r.last_active_iteration,
                            )
                        }),
                    )
                })
                .collect(),
            states: sorted_by_id(stage_states, |s| s.stage_id)
                .into_iter()
                .map(|s| StatesView {
                    pool_id: s.stage_id,
                    count: s.count,
                    state_dimension: s.state_dimension,
                    data_len: s.data.len(),
                    slots: s.slots,
                })
                .collect(),
            bases: sorted_by_id(stage_bases, |b| b.stage_id)
                .into_iter()
                .map(|b| BasisView {
                    node: b.stage_id,
                    entries: b
                        .entries
                        .iter()
                        .map(|e| EntryView {
                            scenario: e.scenario,
                            row_count: e.row_status.len(),
                            piece_row_slots: e.piece_row_slots,
                        })
                        .collect(),
                })
                .collect(),
            metadata,
        }
    }

    /// Check that the checkpoint's entity manifests are date-consistent and its
    /// payloads agree with each other and with the manifest, returning the first
    /// violation found.
    ///
    /// # Errors
    ///
    /// Returns [`OutputError::SerializationError`] naming the offending file and
    /// the rule it breaks.
    fn check(&self) -> Result<(), OutputError> {
        for pool in &self.pools {
            check_pool_dates(pool.pool_id, pool.entity_manifest)?;
        }
        for pool in &self.pools {
            check_record_slots(
                pool.pool_id,
                pool.capacity,
                pool.populated_count,
                &pool.slots,
            )?;
        }
        for states in &self.states {
            let pool = find_pool(&self.pools, states.pool_id);
            check_state_rows(
                states.pool_id,
                states.count,
                states.state_dimension,
                states.data_len,
                states.slots,
                pool.map_or(&[], |p| p.scored_slots.as_slice()),
                pool.map_or(0, |p| p.entity_manifest.len()),
            )?;
        }
        for basis in &self.bases {
            check_scenario_order(basis.node, basis.entries.iter().map(|e| e.scenario))?;
            let node_pool = node_pool(&self.metadata.graph_manifest.nodes, &self.pools, basis.node);
            for entry in &basis.entries {
                check_basis_entry(
                    basis.node,
                    entry.scenario,
                    entry.row_count,
                    entry.piece_row_slots,
                    node_pool,
                )?;
            }
        }
        let producer = &self.metadata.producer;
        check_stored_row_total(
            producer.total_visited_states,
            self.states.iter().map(|s| u64::from(s.count)).sum(),
        )?;
        for pool in &self.pools {
            for &(slot, active_count, last_active_iteration) in &pool.counters {
                check_counter_presence(
                    producer.activity_counters,
                    pool.pool_id,
                    slot,
                    active_count,
                    last_active_iteration,
                )?;
            }
        }
        Ok(())
    }
}

/// The first id, in ascending order, that `ids` holds twice.
fn first_repeated(ids: impl IntoIterator<Item = u32>) -> Option<u32> {
    let mut ids: Vec<u32> = ids.into_iter().collect();
    ids.sort_unstable();
    ids.windows(2)
        .find(|pair| pair[0] == pair[1])
        .map(|pair| pair[0])
}

/// Each id names one file, so two payloads of one kind with the same id would
/// overwrite one another.
fn check_unique_ids(
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    stage_states: &[StageStatesPayload<'_>],
) -> Result<(), OutputError> {
    let repeated = [
        (
            "cuts",
            first_repeated(stage_cuts.iter().map(|p| p.stage_id)),
        ),
        (
            "basis",
            first_repeated(stage_bases.iter().map(|b| b.stage_id)),
        ),
        (
            "states",
            first_repeated(stage_states.iter().map(|s| s.stage_id)),
        ),
    ];
    for (dir, id) in repeated {
        if let Some(id) = id {
            return Err(structure_error(
                dir,
                id,
                "more than one payload carries this id",
            ));
        }
    }
    Ok(())
}

fn sorted_by_id<T>(items: &[T], id: impl Fn(&T) -> u32) -> Vec<&T> {
    let mut sorted: Vec<&T> = items.iter().collect();
    sorted.sort_by_key(|item| id(item));
    sorted
}

fn check_size(file: &str, bound: usize, limit: usize) -> Result<(), OutputError> {
    if bound <= limit {
        Ok(())
    } else {
        Err(OutputError::serialization(
            "policy_checkpoint_size",
            format!(
                "{file}: encoded size bound {bound} bytes exceeds the FlatBuffers limit of {limit} bytes"
            ),
        ))
    }
}

/// Refuse a payload whose file's encoded size bound exceeds `limit` bytes.
///
/// # Errors
///
/// [`OutputError::SerializationError`] naming the first such file, its bound
/// and the limit.
pub(super) fn check_encoded_sizes(
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    stage_states: &[StageStatesPayload<'_>],
    metadata: &CheckpointManifest,
    limit: usize,
) -> Result<(), OutputError> {
    for payload in sorted_by_id(stage_cuts, |p| p.stage_id) {
        let file = format!("cuts/{}", bin_file_name(payload.stage_id));
        check_size(&file, stage_cuts_size_bound(payload), limit)?;
    }
    for record in sorted_by_id(stage_bases, |b| b.stage_id) {
        let file = format!("basis/{}", bin_file_name(record.stage_id));
        check_size(&file, stage_basis_size_bound(record), limit)?;
    }
    for payload in sorted_by_id(stage_states, |s| s.stage_id) {
        let file = format!("states/{}", bin_file_name(payload.stage_id));
        check_size(&file, stage_states_size_bound(payload), limit)?;
    }
    check_size(
        "manifest.bin",
        checkpoint_manifest_size_bound(metadata),
        limit,
    )
}

/// Check, before anything is written, that no two payloads of one kind share an
/// id, that a checkpoint of these payloads is one [`read_policy_checkpoint`]
/// accepts (its entity manifests date-consistent and its payloads
/// structurally sound), and that no file of it can exceed the 2 GiB
/// `FlatBuffers` buffer limit. [`write_policy_checkpoint`] runs it first.
///
/// # Errors
///
/// [`OutputError::SerializationError`] naming the first offending file
/// (`cuts/<pool>.bin`, `basis/<node>.bin`, `states/<pool>.bin` or
/// `manifest.bin`) and the rule it breaks, or its size bound and the limit.
pub fn check_checkpoint_payloads(
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    stage_states: &[StageStatesPayload<'_>],
    metadata: &CheckpointManifest,
) -> Result<(), OutputError> {
    check_unique_ids(stage_cuts, stage_bases, stage_states)?;
    CheckpointView::of_payloads(stage_cuts, stage_bases, stage_states, metadata).check()?;
    check_encoded_sizes(
        stage_cuts,
        stage_bases,
        stage_states,
        metadata,
        FLATBUFFERS_MAX_BUFFER_SIZE,
    )
}

/// Write a complete value-function artifact to `path`, replacing the copy
/// there.
///
/// ## Directory layout produced
///
/// ```text
/// path/
///   manifest.bin
///   cuts/
///     000.bin        (one per pool, keyed by pool id; a shared leaf pool once)
///     001.bin
///     ...
///   basis/
///     000.bin        (one per node with a basis, keyed by node position)
///     001.bin
///     ...
///   states/          (only when stage_states is non-empty)
///     000.bin        (one per pool that stores trial points, keyed by pool id)
///     ...
/// ```
///
/// `basis/` is created even when `stage_bases` is empty.
///
/// ## Commit protocol
///
/// The copy is staged in `<path>.staging` and swapped in with two renames.
/// [`resolve_policy_checkpoint`] reads `<path>`, else `<path>.staging`, else
/// `<path>.previous`, whichever first holds a `manifest.bin`, and after an
/// interruption at any step that is a complete copy. A symbolic link at `path`
/// is read once: the protocol runs at its target, beside which both siblings
/// sit, and the link is kept, pointing at the new copy. Only one process may
/// write a given `path` at a time.
///
/// Before staging, the write refuses a directory holding anything a
/// checkpoint writer does not leave there, and completes or discards what an
/// interrupted write left beside `path`, keeping the copy readers resolve. The write then commits in this order, syncing
/// directories on unix only:
///
/// 1. each payload file is written into `<path>.staging` and synced;
/// 2. each payload subdirectory is synced;
/// 3. `manifest.bin` is written to a temporary file, synced and renamed;
/// 4. `<path>.staging` is synced;
/// 5. `<path>` is renamed to `<path>.previous`, then `<path>.staging` to
///    `<path>`;
/// 6. the parent directory is synced;
/// 7. `<path>.previous` is removed.
///
/// # Errors
///
/// - [`OutputError::SerializationError`] — [`check_checkpoint_payloads`]
///   refused the payloads. Nothing on disk is changed.
/// - [`OutputError::ForeignEntry`] — `path`, its target or a sibling is not
///   a directory or holds an entry no checkpoint writer leaves there. Nothing
///   on disk is changed.
/// - [`OutputError::IoError`] — a probe, write, sync, rename or removal
///   failed, or `path` has no file name ([`io::ErrorKind::InvalidInput`]). A
///   failure before step 5 leaves the previous copy where readers find it; a
///   failure at step 7 leaves the new copy committed.
///
/// # Examples
///
/// ```no_run
/// use novomodelo_io::{
///     write_policy_checkpoint, FORMAT_VERSION, GraphManifest, PolicyBasisRecord,
///     CheckpointManifest, PolicyCutRecord, ProducerBlock, SeasonManifest,
///     SOFTWARE_NAME, SOFTWARE_VERSION, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
///     StateHistory,
/// };
/// use std::path::Path;
///
/// # fn main() -> Result<(), novomodelo_io::OutputError> {
/// let coefficients = [1.0_f64, 2.0, 3.0];
/// let piece = PolicyCutRecord {
///     slot_index: 0,
///     iteration: 1,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &coefficients,
///     is_active: true,
///     scored: false,
///     active_count: None,
///     last_active_iteration: None,
/// };
/// let stage_cuts = [StageCutsPayload {
///     stage_id: 0,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[piece],
///     populated_count: 1,
///     entity_manifest: &[],
///     cost_scale_factor: 1_000_000.0,
///     node_id: 0,
///     graph_stage_id: 0,
///     priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
/// }];
/// let metadata = CheckpointManifest {
///     format_version: FORMAT_VERSION,
///     software: Some(SOFTWARE_NAME.to_string()),
///     software_version: SOFTWARE_VERSION.to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     num_stages: 1,
///     graph_manifest: GraphManifest::default(),
///     producer: ProducerBlock {
///         completed_iterations: 1,
///         final_lower_bound: 42.0,
///         best_upper_bound: None,
///         max_iterations: 100,
///         forward_passes: 4,
///         warm_start_cuts: 0,
///         warm_start_counts: vec![0],
///         rng_seed: 0,
///         total_visited_states: 0,
///         training_block_mode: "parallel".to_string(),
///         training_block_mode_per_stage: vec![],
///         cost_scale_factor: None,
///         lower_bound_history: Vec::new(),
///         state_history: StateHistory::None,
///         activity_counters: false,
///         basis_store_complete: false,
///     },
///     season_manifest: SeasonManifest::default(),
/// };
/// write_policy_checkpoint(Path::new("/tmp/policy"), &stage_cuts, &[], &metadata, &[])?;
/// # Ok(())
/// # }
/// ```
pub fn write_policy_checkpoint(
    path: &Path,
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    metadata: &CheckpointManifest,
    stage_states: &[StageStatesPayload<'_>],
) -> Result<(), OutputError> {
    check_checkpoint_payloads(stage_cuts, stage_bases, stage_states, metadata)?;
    let target = check_checkpoint_replaceable(path)?;
    let (staging, previous) = swap_siblings(&target).ok_or_else(|| without_file_name(path))?;
    finish_interrupted_swap(&target)?;

    ensure_parent_dir(&target)?;
    stage_checkpoint(&staging, stage_cuts, stage_bases, metadata, stage_states)?;

    let replaces = target
        .try_exists()
        .map_err(|e| OutputError::io(&target, e))?;
    if replaces {
        rename_dir(&target, &previous)?;
    }
    rename_dir(&staging, &target)?;
    sync_dir(parent_or_current(&target))?;
    if replaces {
        std::fs::remove_dir_all(&previous).map_err(|e| OutputError::io(&previous, e))?;
    }
    Ok(())
}

/// Write a complete copy into the new directory `staging`, syncing the
/// payloads, then each payload subdirectory, then the manifest, then `staging`.
fn stage_checkpoint(
    staging: &Path,
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    metadata: &CheckpointManifest,
    stage_states: &[StageStatesPayload<'_>],
) -> Result<(), OutputError> {
    let cuts_dir = staging.join("cuts");
    let basis_dir = staging.join("basis");
    let states_dir = staging.join("states");
    let exports_states = !stage_states.is_empty();

    create_dir(staging)?;
    create_dir(&cuts_dir)?;
    create_dir(&basis_dir)?;
    if exports_states {
        create_dir(&states_dir)?;
    }

    for payload in stage_cuts {
        let file_path = cuts_dir.join(bin_file_name(payload.stage_id));
        write_bytes_synced(&file_path, build_stage_cuts(payload).finished_data())?;
    }
    for record in stage_bases {
        let file_path = basis_dir.join(bin_file_name(record.stage_id));
        write_bytes_synced(&file_path, build_stage_basis(record).finished_data())?;
    }
    for payload in stage_states {
        let file_path = states_dir.join(bin_file_name(payload.stage_id));
        write_bytes_synced(&file_path, build_stage_states(payload).finished_data())?;
    }

    sync_dir(&cuts_dir)?;
    sync_dir(&basis_dir)?;
    if exports_states {
        sync_dir(&states_dir)?;
    }

    let manifest_builder = build_checkpoint_manifest(metadata);
    write_bytes_atomic_synced(
        &staging.join("manifest.bin"),
        manifest_builder.finished_data(),
    )?;
    sync_dir(staging)
}

/// Complete or discard what an interrupted write left beside `target`,
/// keeping the copy [`resolve_policy_checkpoint`] finds. Afterwards neither
/// sibling exists, and `target` holds that copy, or is absent or without a
/// `manifest.bin` as before.
///
/// `target` is a directory [`check_checkpoint_replaceable`] returned, so every
/// directory removed holds only checkpoint entries.
///
/// # Errors
///
/// [`OutputError::IoError`] naming the path a probe, removal, rename or sync
/// failed on.
pub(super) fn finish_interrupted_swap(target: &Path) -> Result<(), OutputError> {
    let (staging, previous) = swap_siblings(target).ok_or_else(|| without_file_name(target))?;
    match resolve_policy_checkpoint(target)? {
        ResolvedCheckpoint::Found(copy) if copy == staging => {
            remove_dir_if_present(target)?;
            rename_dir(&staging, target)?;
            sync_dir(parent_or_current(target))?;
            remove_dir_if_present(&previous)
        }
        ResolvedCheckpoint::Found(copy) if copy == previous => {
            remove_dir_if_present(target)?;
            remove_dir_if_present(&staging)?;
            rename_dir(&previous, target)?;
            sync_dir(parent_or_current(target))
        }
        ResolvedCheckpoint::Found(_)
        | ResolvedCheckpoint::NoManifest
        | ResolvedCheckpoint::NoDirectory => {
            remove_dir_if_present(&staging)?;
            remove_dir_if_present(&previous)
        }
    }
}

/// Top-level files a checkpoint writer leaves. `metadata.json` is the commit
/// file of releases up to 0.14: it is replaced with the copy, never read.
const CHECKPOINT_FILES: [&str; 3] = ["manifest.bin", "manifest.bin.tmp", "metadata.json"];

const PAYLOAD_DIRS: [&str; 3] = ["cuts", "basis", "states"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    Absent,
    Directory,
    SymbolicLink,
    Other,
}

fn entry_kind(path: &Path) -> Result<EntryKind, OutputError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Ok(EntryKind::Directory),
        Ok(metadata) if metadata.file_type().is_symlink() => Ok(EntryKind::SymbolicLink),
        Ok(_) => Ok(EntryKind::Other),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(EntryKind::Absent),
        Err(e) => Err(OutputError::io(path, e)),
    }
}

/// Check that a checkpoint write may replace what is at `path`, and return
/// the directory the write runs on: `path`, or the recorded target of a
/// symbolic link there. The check only reads.
///
/// `path` must be absent, a directory, or a symbolic link to a directory. A
/// link pointing at nothing is accepted only while its target's `.staging` or
/// `.previous` sibling holds a `manifest.bin`, as a swap interrupted between
/// its two renames leaves it. Both siblings must be absent or directories.
/// Each existing directory among the target and its siblings may hold only
/// the regular files `manifest.bin`, `manifest.bin.tmp` and `metadata.json`,
/// and the directories `cuts`, `basis` and `states` holding only regular
/// `*.bin` and `*.bin.tmp` files. Links are never followed, except the one at
/// `path`.
///
/// # Errors
///
/// - [`OutputError::ForeignEntry`] naming the first refused entry in sorted
///   order: a non-directory at one of the paths, with its parent as `dir`, or
///   an entry of a directory, with that directory as `dir`.
/// - [`OutputError::IoError`] naming the path a probe or listing failed on,
///   or `path` with [`io::ErrorKind::InvalidInput`] when it has no file name.
pub(crate) fn check_checkpoint_replaceable(path: &Path) -> Result<PathBuf, OutputError> {
    let path_kind = entry_kind(path)?;
    let target = if path_kind == EntryKind::SymbolicLink {
        checkpoint_target(path)?
    } else {
        path.to_path_buf()
    };
    let (staging, previous) = swap_siblings(&target).ok_or_else(|| without_file_name(path))?;
    let staging_kind = entry_kind(&staging)?;
    let previous_kind = entry_kind(&previous)?;

    let target_kind = match path_kind {
        EntryKind::Absent | EntryKind::Directory => path_kind,
        EntryKind::SymbolicLink => match entry_kind(&target)? {
            EntryKind::Directory => EntryKind::Directory,
            EntryKind::Absent
                if holds_manifest(&staging, staging_kind)?
                    || holds_manifest(&previous, previous_kind)? =>
            {
                EntryKind::Absent
            }
            _ => return Err(non_directory(path)),
        },
        EntryKind::Other => return Err(non_directory(path)),
    };
    for (sibling, kind) in [(&staging, staging_kind), (&previous, previous_kind)] {
        if !matches!(kind, EntryKind::Absent | EntryKind::Directory) {
            return Err(non_directory(sibling));
        }
    }

    for (dir, kind) in [
        (&target, target_kind),
        (&staging, staging_kind),
        (&previous, previous_kind),
    ] {
        if kind == EntryKind::Directory
            && let Some(entry) = first_foreign_entry(dir)?
        {
            return Err(OutputError::ForeignEntry {
                dir: dir.clone(),
                entry,
            });
        }
    }
    Ok(target)
}

fn holds_manifest(dir: &Path, kind: EntryKind) -> Result<bool, OutputError> {
    if kind != EntryKind::Directory {
        return Ok(false);
    }
    let manifest_path = dir.join("manifest.bin");
    manifest_path
        .try_exists()
        .map_err(|e| OutputError::io(&manifest_path, e))
}

/// The first entry of `dir`, in sorted order, that no checkpoint writer
/// leaves there.
fn first_foreign_entry(dir: &Path) -> Result<Option<PathBuf>, OutputError> {
    for (entry, file_type) in sorted_entries(dir)? {
        if file_type.is_dir() && has_name_in(&entry, &PAYLOAD_DIRS) {
            let foreign_payload = sorted_entries(&entry)?
                .into_iter()
                .find(|(payload, payload_type)| !(payload_type.is_file() && is_payload(payload)));
            if let Some((payload, _)) = foreign_payload {
                return Ok(Some(payload));
            }
        } else if !(file_type.is_file() && has_name_in(&entry, &CHECKPOINT_FILES)) {
            return Ok(Some(entry));
        }
    }
    Ok(None)
}

fn sorted_entries(dir: &Path) -> Result<Vec<(PathBuf, FileType)>, OutputError> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| OutputError::io(dir, e))? {
        let path = entry.map_err(|e| OutputError::io(dir, e))?.path();
        let file_type = std::fs::symlink_metadata(&path)
            .map_err(|e| OutputError::io(&path, e))?
            .file_type();
        entries.push((path, file_type));
    }
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    Ok(entries)
}

fn has_name_in(path: &Path, names: &[&str]) -> bool {
    path.file_name()
        .is_some_and(|name| names.iter().any(|candidate| name == *candidate))
}

fn is_payload(path: &Path) -> bool {
    let has_extension = |path: &Path, extension: &str| path.extension() == Some(extension.as_ref());
    has_extension(path, "bin")
        || (has_extension(path, "tmp")
            && path
                .file_stem()
                .is_some_and(|stem| has_extension(Path::new(stem), "bin")))
}

fn non_directory(entry: &Path) -> OutputError {
    OutputError::ForeignEntry {
        dir: parent_or_current(entry).to_path_buf(),
        entry: entry.to_path_buf(),
    }
}

fn without_file_name(path: &Path) -> OutputError {
    OutputError::io(path, io::Error::from(io::ErrorKind::InvalidInput))
}

/// The directory holding `path`: its parent, or `.` for a bare name.
fn parent_or_current(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn create_dir(dir: &Path) -> Result<(), OutputError> {
    std::fs::create_dir(dir).map_err(|e| OutputError::io(dir, e))
}

fn rename_dir(from: &Path, to: &Path) -> Result<(), OutputError> {
    std::fs::rename(from, to).map_err(|e| OutputError::io(from, e))
}

fn remove_dir_if_present(dir: &Path) -> Result<(), OutputError> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(OutputError::io(dir, e)),
    }
}

const STAGING_SUFFIX: &str = ".staging";
const PREVIOUS_SUFFIX: &str = ".previous";

fn sibling(path: &Path, suffix: &str) -> Option<PathBuf> {
    let mut name = path.file_name()?.to_os_string();
    name.push(suffix);
    Some(path.with_file_name(name))
}

/// The `.staging` and `.previous` siblings of `target`; none for a path with
/// no file name.
fn swap_siblings(target: &Path) -> Option<(PathBuf, PathBuf)> {
    Some((
        sibling(target, STAGING_SUFFIX)?,
        sibling(target, PREVIOUS_SUFFIX)?,
    ))
}

/// The directory a checkpoint at `path` lives in: the recorded target of a
/// symbolic link at `path`, else `path` itself.
pub(crate) fn checkpoint_target(path: &Path) -> Result<PathBuf, OutputError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            // `read_link`, not `canonicalize`: a link left pointing at nothing by
            // an interrupted swap still names the target's siblings.
            let target = std::fs::read_link(path).map_err(|e| OutputError::io(path, e))?;
            Ok(match path.parent() {
                Some(parent) => parent.join(target),
                None => target,
            })
        }
        Ok(_) => Ok(path.to_path_buf()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(path.to_path_buf()),
        Err(e) => Err(OutputError::io(path, e)),
    }
}

/// Which copy of a checkpoint a read uses, as [`resolve_policy_checkpoint`]
/// finds it. The target is the checkpoint path itself, or the recorded target
/// of a symbolic link there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedCheckpoint {
    /// The directory a read uses: the target, its `.staging` sibling or its
    /// `.previous` sibling.
    Found(PathBuf),
    /// The target exists, and no candidate holds a `manifest.bin`.
    NoManifest,
    /// The target is absent, and no candidate holds a `manifest.bin`.
    NoDirectory,
}

/// Resolve which copy of the checkpoint at `path` a read uses.
///
/// The candidates, in order, are `path`, its `.staging` sibling and its
/// `.previous` sibling; the first that holds a `manifest.bin` is the copy. A
/// present `manifest.bin` is the only completeness signal, because a writer
/// puts it in place last. A symbolic link at `path` is read once, and its
/// recorded target and that target's siblings are the candidates.
///
/// The call changes nothing on disk, so any number of processes may resolve
/// the same `path` at once.
///
/// # Errors
///
/// [`OutputError::IoError`] naming the probed path when a probe fails for any
/// reason other than absence, or naming `path` when a link there cannot be
/// inspected or read.
pub fn resolve_policy_checkpoint(path: &Path) -> Result<ResolvedCheckpoint, OutputError> {
    let target = checkpoint_target(path)?;
    let candidates = [
        Some(target.clone()),
        sibling(&target, STAGING_SUFFIX),
        sibling(&target, PREVIOUS_SUFFIX),
    ];
    for dir in candidates.into_iter().flatten() {
        let manifest_path = dir.join("manifest.bin");
        if manifest_path
            .try_exists()
            .map_err(|e| OutputError::io(&manifest_path, e))?
        {
            return Ok(ResolvedCheckpoint::Found(dir));
        }
    }
    if target
        .try_exists()
        .map_err(|e| OutputError::io(&target, e))?
    {
        Ok(ResolvedCheckpoint::NoManifest)
    } else {
        Ok(ResolvedCheckpoint::NoDirectory)
    }
}

/// Read a complete value-function artifact from the copy of `path` that
/// [`resolve_policy_checkpoint`] finds.
///
/// That copy is `path` (for a symbolic link, its target), else its `.staging`
/// sibling, else its `.previous` sibling, whichever first holds a
/// `manifest.bin`; with none, the read fails on `<path>/manifest.bin`. The
/// read changes nothing on disk.
///
/// `manifest.bin` is read first and its `format_version` is checked
/// **before any `.bin` payload is parsed**: an absent `manifest.bin`, a missing
/// `CBVF` identifier, or a mismatched version is a named error, so an artifact
/// this build cannot read is cleanly rejected rather than read positionally.
///
/// Per-pool/-stage results are sorted by `stage_id` in the returned
/// [`PolicyCheckpoint`].
///
/// # Errors
///
/// - [`OutputError::IoError`] — a probe, directory or file read failed (a
///   missing `manifest.bin`, i.e. a pre-`manifest.bin` artifact, included).
/// - [`OutputError::SerializationError`] — a `FlatBuffers` parse failure, a
///   missing `CBVF` identifier or a `format_version` mismatch (both enforced by
///   [`deserialize_checkpoint_manifest`]), or a date-consistency or structural
///   violation, the rules [`check_checkpoint_payloads`] applies before a write.
///
/// # Examples
///
/// ```no_run
/// use novomodelo_io::read_policy_checkpoint;
/// use std::path::Path;
///
/// # fn main() -> Result<(), novomodelo_io::OutputError> {
/// let checkpoint = read_policy_checkpoint(Path::new("/tmp/policy"))?;
/// println!("metadata: {} stages", checkpoint.metadata.num_stages);
/// println!("stages loaded: {}", checkpoint.stage_cuts.len());
/// # Ok(())
/// # }
/// ```
pub fn read_policy_checkpoint(path: &Path) -> Result<PolicyCheckpoint, OutputError> {
    match resolve_policy_checkpoint(path)? {
        ResolvedCheckpoint::Found(dir) => read_checkpoint_dir(&dir),
        ResolvedCheckpoint::NoManifest | ResolvedCheckpoint::NoDirectory => {
            read_checkpoint_dir(path)
        }
    }
}

fn read_checkpoint_dir(path: &Path) -> Result<PolicyCheckpoint, OutputError> {
    let manifest_path = path.join("manifest.bin");
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| OutputError::io(&manifest_path, e))?;
    let metadata = deserialize_checkpoint_manifest(&manifest_bytes)?;

    let cuts_dir = path.join("cuts");
    let mut stage_cuts: Vec<StageCutsReadResult> =
        read_sorted_bin_files(&cuts_dir, "stage_cuts", deserialize_stage_cuts)?;
    stage_cuts.sort_by_key(|r| r.stage_id);

    // Absent reads as empty, like `states/`: zip tools drop the empty `basis/` the writer leaves.
    let basis_dir = path.join("basis");
    let stage_bases: Vec<OwnedPolicyBasisRecord> = if basis_dir.is_dir() {
        let mut sb = read_sorted_bin_files(&basis_dir, "stage_basis", deserialize_stage_basis)?;
        sb.sort_by_key(|r| r.stage_id);
        sb
    } else {
        Vec::new()
    };

    let states_dir = path.join("states");
    let stage_states: Vec<StageStatesReadResult> = if states_dir.is_dir() {
        let mut ss = read_sorted_bin_files(&states_dir, "stage_states", deserialize_stage_states)?;
        ss.sort_by_key(|r| r.stage_id);
        ss
    } else {
        Vec::new()
    };

    let checkpoint = PolicyCheckpoint {
        metadata,
        stage_cuts,
        stage_bases,
        stage_states,
    };
    CheckpointView::of_checkpoint(&checkpoint).check()?;
    Ok(checkpoint)
}
