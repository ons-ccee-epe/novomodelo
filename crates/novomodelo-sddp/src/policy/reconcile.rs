//! Boundary-policy coefficient reconciliation: per-target-slot rebind
//! operations that map a source cut's coefficients onto a target manifest of
//! possibly different shape than `source`.
//!
//! Storage and inflow-lag are the state's must-correspond core: a target slot
//! of either family with no source counterpart is [`RebindOp::Reject`] — the
//! entity (`entity_type`/`entity_id`) is never relaxed, only matched exactly.
//! Inflow-lag additionally validates a matched slot's `reference_date`
//! ([`resolve_inflow_lag`]), so identity alone does not guarantee a `Copy`.
//! Only a forward-dated family's calendar `subindex` is relaxed, by
//! [`resolve_by_interval_overlap`].
//!
//! Strict-superset guarantee: when every storage/lag/unclassified target slot
//! has a same-identity source counterpart and every forward-family slot is
//! still sentinel-dated — the shape an already target-aligned, pre-fan-out
//! boundary policy has — the rebind reproduces the source cut's own
//! coefficients bit-for-bit, the sentinel slots' `Zero` included: a masked
//! state dimension's source coefficient is itself always `0.0`.

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::AnticipatedCommitmentHistory;
use cobre_io::ENTITY_SLOT_DATE_SENTINEL;
use cobre_io::EntitySlot;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::StateFamily;
use cobre_io::decode_slot_date;
use serde::Serialize;

use crate::SddpError;

/// Tolerance (hours) for treating a target slot's covered-month overlap as
/// exactly `H_w` — the `Blend`-vs-`Renormalize` dividing line. Calendar-day
/// arithmetic converted to hours is exact in `f64` for any realistic
/// horizon, so this only absorbs a multi-term summation's rounding, never
/// masks a genuine gap.
const COVERAGE_TOLERANCE_HOURS: f64 = 1e-6;

/// One reconciliation operation, producing one target-manifest slot's
/// coefficient from a source cut's coefficients.
///
/// [`build_rebind`] assigns exactly one op per target slot; [`rebind_cut`]
/// applies the assignment to a source cut.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RebindOp {
    /// Take the source coefficient at this position verbatim.
    Copy(usize),
    /// No source counterpart contributes; the coefficient is `0.0`.
    Zero,
    /// An hours-weighted blend of source positions: [`rebind_cut`] applies
    /// `Σ cut.coefficients[p] · w` over the `(source_position, weight)`
    /// terms, whose weights [`resolve_by_interval_overlap`] owns.
    Blend(Vec<(usize, f64)>),
    /// A [`Self::Blend`] whose weights are additionally scaled by
    /// `H_w / Σ_covered overlap` so a partially covered target slot
    /// replicates the covered intervals' density across its uncovered span
    /// instead of deflating it with an implicit `0.0` term.
    Renormalize(Vec<(usize, f64)>),
    /// The target slot cannot be resolved from `source`. A sentinel:
    /// [`build_rebind`] converts this into an [`SddpError::Validation`]
    /// rather than returning it, so it never appears in a successfully built
    /// op vector.
    Reject {
        /// Human-readable rejection reason.
        reason: String,
    },
}

/// `(entity_type, entity_id, subindex)` — the identity a reconciliation join
/// keys on.
pub(crate) type SlotKey = (u8, i32, u32);

fn slot_key(slot: &EntitySlot) -> SlotKey {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// One [`SlotKey`] -> source position index, built once per `source`
/// manifest and shared by [`build_rebind`]'s identity-keyed resolution arms
/// and the topology-subset pre-check that runs ahead of it — the same
/// `source` manifest, so both consumers read one hashing of it.
pub(crate) fn build_identity_index(source: &[EntitySlot]) -> HashMap<SlotKey, usize> {
    let mut index = HashMap::with_capacity(source.len());
    for (pos, slot) in source.iter().enumerate() {
        index.insert(slot_key(slot), pos);
    }
    index
}

/// One forward-family source slot's decoded delivery/arrival interval and its
/// position in `source`.
#[derive(Debug)]
pub(crate) struct SourceInterval {
    source_pos: usize,
    start: NaiveDate,
    end: NaiveDate,
    hours: f64,
}

/// `(entity_type, entity_id)` — the forward-family fan-out join key, shared
/// by `AnticipatedThermalState` and `HydroTransitBucket`. The family byte
/// keeps the two families' entries disjoint in
/// [`build_source_interval_index`]'s map, so one index serves both without a
/// discriminator parameter.
pub(crate) type SourceKey = (u8, i32);

/// Positive `[start, end)` span in hours; `0.0` for a degenerate or
/// backwards interval (`end <= start`), never a negative value.
fn positive_hours(start: NaiveDate, end: NaiveDate) -> f64 {
    if end <= start {
        return 0.0;
    }
    let days = u32::try_from((end - start).num_days()).unwrap_or(0);
    f64::from(days) * 24.0
}

/// Hours of overlap between two `[start, end)` calendar intervals; `0.0` when
/// they do not intersect.
pub(crate) fn overlap_hours(a: (NaiveDate, NaiveDate), b: (NaiveDate, NaiveDate)) -> f64 {
    positive_hours(a.0.max(b.0), a.1.min(b.1))
}

/// Index every LIVE (non-sentinel) forward-family source slot by its owning
/// [`SourceKey`], decoding the `[start, end)` span from the slot's own
/// `interval_start`/`interval_end` rather than reconstructing a month anchor.
///
/// # Errors
///
/// Returns [`SddpError::Validation`], naming the slot's identity and the raw
/// values, if a live forward-family source slot's
/// `interval_start`/`interval_end` fails to decode, or decodes to a
/// non-positive span.
pub(crate) fn build_source_interval_index(
    source: &[EntitySlot],
) -> Result<HashMap<SourceKey, Vec<SourceInterval>>, SddpError> {
    let mut index: HashMap<SourceKey, Vec<SourceInterval>> = HashMap::new();
    for (pos, slot) in source.iter().enumerate() {
        let is_forward_family = slot.entity_type == StateFamily::AnticipatedThermalState.code()
            || slot.entity_type == StateFamily::HydroTransitBucket.code();
        if !is_forward_family || slot.interval_start == ENTITY_SLOT_DATE_SENTINEL {
            continue;
        }
        let Some(start) = decode_slot_date(slot.interval_start) else {
            return Err(SddpError::Validation(format!(
                "boundary policy source forward-family slot (entity_id={}, subindex={}) has an \
                 undecodable interval_start {}",
                slot.entity_id, slot.subindex, slot.interval_start
            )));
        };
        let Some(end) = decode_slot_date(slot.interval_end) else {
            return Err(SddpError::Validation(format!(
                "boundary policy source forward-family slot (entity_id={}, subindex={}) has an \
                 undecodable interval_end {}",
                slot.entity_id, slot.subindex, slot.interval_end
            )));
        };
        let hours = positive_hours(start, end);
        if hours <= 0.0 {
            return Err(SddpError::Validation(format!(
                "boundary policy source forward-family slot (entity_id={}, subindex={}) has a \
                 non-positive delivery interval: interval_start={}, interval_end={}",
                slot.entity_id, slot.subindex, slot.interval_start, slot.interval_end
            )));
        }
        index
            .entry((slot.entity_type, slot.entity_id))
            .or_default()
            .push(SourceInterval {
                source_pos: pos,
                start,
                end,
                hours,
            });
    }
    Ok(index)
}

/// The boundary intercept-fold vector `(source_pos, factor)` pricing a study's
/// fixed post-horizon commitments, reusing [`resolve_by_interval_overlap`]'s
/// per-source-interval `÷H_m` distribute against each fixed window's declared
/// MW instead of a state dimension. A window overlapping no source interval
/// contributes nothing (mirrors [`RebindOp::Zero`], never an error), and only
/// non-zero factors are emitted — an empty vector is the byte-neutral
/// no-contribution case the intercept fold relies on. `source_index` is
/// [`build_source_interval_index`]'s output, built once by the caller and
/// shared with [`build_rebind`]/[`build_reconciliation_report`] so a source
/// slot's span is never read two different ways.
///
/// Determinism (D5): accumulates into a `source_pos`-indexed `Vec`, iterating
/// `fixed_windows` and each plant's `source_index` records in given order;
/// the sole map use is the index lookup, never a `HashMap` iteration. The
/// lookup key's family byte is load-bearing: it is the only thing that keeps
/// a transit slot sharing an `entity_id` with an anticipated slot out of the
/// fold, so the key must never collapse to the bare `entity_id`
/// (`boundary_fold_ignores_transit_slots_sharing_an_entity_id`).
pub(crate) fn build_boundary_fold(
    source: &[EntitySlot],
    fixed_windows: &[AnticipatedCommitmentHistory],
    source_index: &HashMap<SourceKey, Vec<SourceInterval>>,
) -> Vec<(usize, f64)> {
    let mut factor = vec![0.0_f64; source.len()];
    for window in fixed_windows {
        if window.value_mw == 0.0 {
            continue;
        }
        let key = (
            StateFamily::AnticipatedThermalState.code(),
            window.thermal_id.0,
        );
        let Some(intervals) = source_index.get(&key) else {
            continue;
        };
        for interval in intervals {
            let overlap = overlap_hours(
                (window.start_date, window.end_date),
                (interval.start, interval.end),
            );
            if overlap > 0.0 {
                factor[interval.source_pos] += (overlap / interval.hours) * window.value_mw;
            }
        }
    }
    factor
        .into_iter()
        .enumerate()
        .filter(|&(_, f)| f != 0.0)
        .collect()
}

/// Build one [`RebindOp`] per `target` slot, dispatched by family
/// ([`resolve_target_slot`]). `source_index` is
/// [`build_source_interval_index`]'s output and `identity_index` is
/// [`build_identity_index`]'s output, both built once by the caller and
/// shared with [`build_reconciliation_report`] (`source_index`) and the
/// topology-subset pre-check (`identity_index`) so a source slot's span or
/// identity is never read two different ways.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a storage, inflow-lag, or
/// unclassified `target` slot has no `source` counterpart under identity
/// resolution, or a live forward-family `target` slot's
/// `interval_start`/`interval_end` fails to decode. A live forward-family
/// `target` slot whose interval ends at or before `boundary_date` is an
/// in-study delivery, resolved to [`RebindOp::Zero`] rather than rejected
/// (see [`resolve_by_interval_overlap`]).
pub(crate) fn build_rebind(
    source: &[EntitySlot],
    target: &[EntitySlot],
    boundary_date: NaiveDate,
    source_index: &HashMap<SourceKey, Vec<SourceInterval>>,
    identity_index: &HashMap<SlotKey, usize>,
) -> Result<Vec<RebindOp>, SddpError> {
    let mut ops = Vec::with_capacity(target.len());
    for (i, slot) in target.iter().enumerate() {
        match resolve_target_slot(i, slot, source, identity_index, source_index, boundary_date) {
            RebindOp::Reject { reason } => return Err(SddpError::Validation(reason)),
            op => ops.push(op),
        }
    }
    Ok(ops)
}

/// Dispatch one target slot to its family's resolution rule.
fn resolve_target_slot(
    i: usize,
    slot: &EntitySlot,
    source: &[EntitySlot],
    by_identity: &HashMap<SlotKey, usize>,
    source_index: &HashMap<SourceKey, Vec<SourceInterval>>,
    boundary_date: NaiveDate,
) -> RebindOp {
    match slot.family() {
        Some(StateFamily::HydroStorage) => resolve_storage(slot, by_identity),
        Some(StateFamily::HydroInflowLag) => resolve_inflow_lag(slot, source, by_identity),
        Some(StateFamily::HydroTransitBucket | StateFamily::AnticipatedThermalState) => {
            resolve_by_interval_overlap(slot, boundary_date, source_index)
        }
        None => resolve_by_identity(i, slot, by_identity),
    }
}

/// `HydroStorage` resolution, by `(entity_type, entity_id)` identity. Through
/// [`load_boundary_cuts`](crate::policy::policy_load::load_boundary_cuts),
/// the `check_topology_subset` gate normally reports this condition first,
/// over every missing hydro at once; this arm remains the enforcement of
/// [`build_rebind`]'s own postcondition for a caller that invokes it
/// directly.
fn resolve_storage(slot: &EntitySlot, by_identity: &HashMap<SlotKey, usize>) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Reject {
            reason: format!(
                "boundary policy does not price hydro {}; it was trained on a different set of \
                 plants",
                slot.entity_id
            ),
        },
    }
}

/// `HydroInflowLag` resolution, by `(entity_type, entity_id, subindex)`
/// identity, under the same `check_topology_subset` relationship as
/// [`resolve_storage`].
///
/// An identity hit is then validated on `reference_date`: the two raw `i32`
/// stamps, never decoded `NaiveDate`s — two undecodable stamps must not fold
/// into a spurious equality. Both dated and DIFFERENT rejects, naming the
/// hydro, the lag depth, and both dates — a "different past" diagnosis,
/// worded distinctly from the identity-miss reject above so the two failures,
/// which have different remedies, are never conflated. Either side at
/// `ENTITY_SLOT_DATE_SENTINEL` copies by identity regardless: the
/// [`reserve_boundary_inflow_lag_slots`](crate::policy_export::reserve_boundary_inflow_lag_slots)
/// bridge builds slots from a manifest and coefficients, never a calendar, so
/// it can only ever emit the sentinel.
///
/// Two wrong-but-compiling alternatives: rejecting whenever either side is
/// undated breaks that bridge, the only supported external-authoring path;
/// keying the join on `(hydro, reference_date)` instead of `(hydro,
/// lag-index)` would make every undated slot unjoinable, to guard against a
/// convention drift
/// [`build_stage_entity_manifest`](crate::policy_export::build_stage_entity_manifest)
/// already precludes as the sole owner of the 1-based `subindex` convention
/// on both sides.
fn resolve_inflow_lag(
    slot: &EntitySlot,
    source: &[EntitySlot],
    by_identity: &HashMap<SlotKey, usize>,
) -> RebindOp {
    let Some(&pos) = by_identity.get(&slot_key(slot)) else {
        return RebindOp::Reject {
            reason: format!(
                "boundary policy has no inflow-lag coefficient for hydro {} at lag depth {}: the \
                 boundary is lag-depth-incompatible with the current study",
                slot.entity_id, slot.subindex
            ),
        };
    };

    let target_date = slot.reference_date;
    let source_date = source[pos].reference_date;
    let target_live = target_date != ENTITY_SLOT_DATE_SENTINEL;
    let source_live = source_date != ENTITY_SLOT_DATE_SENTINEL;

    match (target_live, source_live) {
        (true, true) if target_date == source_date => RebindOp::Copy(pos),
        (true, true) => RebindOp::Reject {
            reason: format!(
                "boundary policy's inflow-lag coefficient for hydro {} at lag depth {} \
                 references a different past than the current study: boundary {}, current {}",
                slot.entity_id,
                slot.subindex,
                render_reference_date(source_date),
                render_reference_date(target_date),
            ),
        },
        (false, _) | (_, false) => RebindOp::Copy(pos),
    }
}

/// Renders an [`EntitySlot::reference_date`] stamp for a reject message,
/// never for the equality comparison itself (that stays on the raw `i32` in
/// [`resolve_inflow_lag`]).
fn render_reference_date(raw: i32) -> String {
    decode_slot_date(raw).map_or_else(|| raw.to_string(), |d| d.to_string())
}

/// Forward-dated resolution shared by `AnticipatedThermalState` and
/// `HydroTransitBucket` target slots — the two families whose delivery/
/// arrival intervals, not `subindex` identity, determine which source
/// position(s) a target coefficient draws from. `subindex` plays no part in
/// the join for either family: two transit buckets of one downstream plant at
/// different maturity lags, or two anticipated ring slots of one thermal, are
/// told apart entirely by their disjoint intervals.
///
/// `interval_start == SENTINEL` (pre-fan-out padding) resolves to `Zero`. A
/// live interval that fails to decode (never expected past
/// `read_policy_checkpoint`'s date validation, but not ruled out by the type
/// system) is [`RebindOp::Reject`] — loud rather than silently `Zero`. A live
/// interval ending at or before `boundary_date` is an IN-STUDY delivery,
/// already discharged inside the study, and resolves to `Zero`, never a
/// reject: the terminal boundary FCF prices only post-study obligations, and
/// rejecting here aborts a legitimate boundary load the moment any
/// anticipated thermal delivers in-horizon (the `K = 0` sub-stage-lead case).
///
/// Under a shared stage calendar, a live interval reaching past
/// `boundary_date` always STARTS at or after it too, so none straddles the
/// boundary. That is asserted by a `debug_assert!` rather than enforced by
/// clipping to `[max(start, boundary_date), end)` — clipping adds machinery
/// for a case the calendar forbids and masks the producer bug the assertion
/// surfaces.
///
/// Such an interval fans out against `source_index`'s same-family intervals:
/// no covered interval yields `Zero` — the expected outcome for a
/// `HydroTransitBucket` target against a NEWAVE-shaped source carrying no
/// transit arcs at all, an expected boundary shape rather than an
/// incompatibility; full coverage yields [`RebindOp::Blend`] (`weight =
/// overlap(w, m) / H_m`, the `÷H_m` distribute); partial coverage (a
/// boundary-edge slot straddling into unpriced time) yields
/// [`RebindOp::Renormalize`], scaling the covered intervals' density up to
/// the full slot instead of an implicit `0.0` deflation term.
fn resolve_by_interval_overlap(
    slot: &EntitySlot,
    boundary_date: NaiveDate,
    source_index: &HashMap<SourceKey, Vec<SourceInterval>>,
) -> RebindOp {
    if slot.interval_start == ENTITY_SLOT_DATE_SENTINEL {
        return RebindOp::Zero;
    }
    let Some(start_w) = decode_slot_date(slot.interval_start) else {
        return RebindOp::Reject {
            reason: format!(
                "boundary policy target forward-family slot (entity_id={}, subindex={}) has an \
                 undecodable interval_start {}",
                slot.entity_id, slot.subindex, slot.interval_start
            ),
        };
    };
    let Some(end_w) = decode_slot_date(slot.interval_end) else {
        return RebindOp::Reject {
            reason: format!(
                "boundary policy target forward-family slot (entity_id={}, subindex={}) has an \
                 undecodable interval_end {}",
                slot.entity_id, slot.subindex, slot.interval_end
            ),
        };
    };

    if end_w <= boundary_date {
        return RebindOp::Zero;
    }
    debug_assert!(
        start_w >= boundary_date,
        "a live target forward-family interval reaching past the boundary date must start at or \
         after it under a shared stage calendar: [{start_w}, {end_w}) vs boundary {boundary_date}"
    );

    let h_w = positive_hours(start_w, end_w);
    if h_w <= 0.0 {
        return RebindOp::Zero;
    }
    let Some(intervals) = source_index.get(&(slot.entity_type, slot.entity_id)) else {
        return RebindOp::Zero;
    };

    let mut terms = Vec::new();
    let mut covered = 0.0;
    for interval in intervals {
        let overlap = overlap_hours((start_w, end_w), (interval.start, interval.end));
        if overlap > 0.0 {
            terms.push((interval.source_pos, overlap, interval.hours));
            covered += overlap;
        }
    }
    if terms.is_empty() {
        return RebindOp::Zero;
    }

    if (h_w - covered).abs() <= COVERAGE_TOLERANCE_HOURS {
        RebindOp::Blend(
            terms
                .into_iter()
                .map(|(pos, overlap, hours)| (pos, overlap / hours))
                .collect(),
        )
    } else {
        let scale = h_w / covered;
        RebindOp::Renormalize(
            terms
                .into_iter()
                .map(|(pos, overlap, hours)| (pos, (overlap / hours) * scale))
                .collect(),
        )
    }
}

/// Generic identity resolution: the default for any family without its own
/// arm. Unmatched rejects naming the target's own position and identity.
fn resolve_by_identity(
    i: usize,
    slot: &EntitySlot,
    by_identity: &HashMap<SlotKey, usize>,
) -> RebindOp {
    match by_identity.get(&slot_key(slot)) {
        Some(&pos) => RebindOp::Copy(pos),
        None => RebindOp::Reject {
            reason: format!(
                "target slot {i} (entity_type={}, entity_id={}, subindex={}) has no source \
                 counterpart",
                slot.entity_type, slot.entity_id, slot.subindex
            ),
        },
    }
}

/// Produce `rebind`'s target-aligned coefficient vector from one source cut.
///
/// `Blend` and `Renormalize` both apply their precomputed `(source_position,
/// weight)` terms identically — `Σ cut.coefficients[p] · w` — the
/// distribute-vs-renormalize distinction lives entirely in how
/// [`build_rebind`] computed the weights, never in this application.
///
/// # Panics
///
/// Panics if `rebind` contains a [`RebindOp::Reject`] — a [`build_rebind`]
/// postcondition violation: `build_rebind` converts every `Reject` into an
/// error before returning.
pub(crate) fn rebind_cut(cut: &OwnedPolicyCutRecord, rebind: &[RebindOp]) -> Vec<f64> {
    rebind
        .iter()
        .map(|op| match op {
            RebindOp::Copy(pos) => cut.coefficients[*pos],
            RebindOp::Zero => 0.0,
            RebindOp::Blend(terms) | RebindOp::Renormalize(terms) => terms
                .iter()
                .map(|&(pos, w)| cut.coefficients[pos] * w)
                .sum(),
            RebindOp::Reject { reason } => unreachable!(
                "build_rebind must convert Reject into an error before rebind_cut sees it: \
                 {reason}"
            ),
        })
        .collect()
}

/// `source` positions no op in `rebind` ever references, excluding a
/// structural pad ([`is_structural_pad`]) — the source-side counterpart to
/// [`classify_op`]'s target-side pad exclusion: a pad prices nothing, so an
/// unreferenced pad is not a genuinely dropped coefficient.
fn dropped_source_positions(source: &[EntitySlot], rebind: &[RebindOp]) -> Vec<usize> {
    let len = source.len();
    let mut referenced = vec![false; len];
    for op in rebind {
        match op {
            RebindOp::Copy(pos) => {
                if *pos < len {
                    referenced[*pos] = true;
                }
            }
            RebindOp::Blend(terms) | RebindOp::Renormalize(terms) => {
                for &(pos, _) in terms {
                    if pos < len {
                        referenced[pos] = true;
                    }
                }
            }
            RebindOp::Zero | RebindOp::Reject { .. } => {}
        }
    }
    referenced
        .into_iter()
        .enumerate()
        .filter_map(|(pos, was_referenced)| {
            (!was_referenced && !is_structural_pad(&source[pos])).then_some(pos)
        })
        .collect()
}

/// Human-readable name for a slot's state family — the single owner of every
/// family's rendered name.
fn family_label(family: Option<StateFamily>) -> &'static str {
    match family {
        Some(StateFamily::HydroStorage) => "storage",
        Some(StateFamily::HydroInflowLag) => "inflow-lag",
        Some(StateFamily::HydroTransitBucket) => "transit-bucket",
        Some(StateFamily::AnticipatedThermalState) => "anticipated",
        None => "other-identity",
    }
}

/// Per-family tally of one [`BoundaryReconciliationReport`]'s per-operation
/// classification ([`classify_op`]).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct FamilyTally {
    /// Target slots resolved by [`RebindOp::Copy`].
    pub copy: usize,
    /// Target slots resolved by [`RebindOp::Blend`] or [`RebindOp::Renormalize`].
    pub fan_out: usize,
    /// The [`RebindOp::Renormalize`] subset of `fan_out`.
    pub straddling: usize,
    /// Target-only [`RebindOp::Zero`] slots (excludes a sentinel-anticipated pad).
    pub default_zero: usize,
    /// This family's source positions no op references.
    pub dropped_source: usize,
}

/// Anticipated-family fan-out coverage.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct AnticipatedCoverage {
    /// Live, dated anticipated source slots contributing a decoded delivery interval.
    pub source_interval_count: usize,
}

/// One [`EntitySlot`]'s identity and dating, as rendered into
/// [`BoundaryReconciliationReport::dropped_source_slots`] and
/// [`BoundaryReconciliationReport::straddling_slots`].
#[derive(Debug, Clone, Serialize)]
pub struct SlotDetail {
    /// This slot's state family, via [`family_label`].
    pub family: &'static str,
    /// The owning entity's id.
    pub entity_id: i32,
    /// The slot's secondary index (per-family meaning is the caller's).
    pub subindex: u32,
    /// The slot's own `[start, end)` delivery/arrival span, decoded, for the
    /// two forward-dated families; `None` otherwise, including an
    /// undecodable stamp.
    pub interval: Option<(NaiveDate, NaiveDate)>,
    /// The slot's own reference date, decoded, for `HydroInflowLag`; `None`
    /// otherwise, including an undecodable stamp.
    pub reference_date: Option<NaiveDate>,
}

/// Builds `slot`'s [`SlotDetail`], decoding only the date field(s) `slot`'s
/// own family populates. A stamp that fails to decode yields `None`, never an
/// error — the report is a diagnostic and must never fail a load that
/// otherwise succeeded.
fn slot_detail(slot: &EntitySlot) -> SlotDetail {
    let family = slot.family();
    let interval = matches!(
        family,
        Some(StateFamily::AnticipatedThermalState | StateFamily::HydroTransitBucket)
    )
    .then(|| decode_slot_date(slot.interval_start).zip(decode_slot_date(slot.interval_end)))
    .flatten();
    let reference_date = matches!(family, Some(StateFamily::HydroInflowLag))
        .then(|| decode_slot_date(slot.reference_date))
        .flatten();
    SlotDetail {
        family: family_label(family),
        entity_id: slot.entity_id,
        subindex: slot.subindex,
        interval,
        reference_date,
    }
}

/// The "which boundary policy we have + what got reconciled" diagnostic:
/// [`build_reconciliation_report`]'s pure per-family tally of one
/// `load_boundary_cuts` reconciliation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BoundaryReconciliationReport {
    /// `false` on the skipped (dimension-only) load path; `true` when rebind ran.
    pub reconciled: bool,
    /// `HydroStorage` tally.
    pub storage: FamilyTally,
    /// `HydroInflowLag` tally.
    pub inflow_lag: FamilyTally,
    /// `HydroTransitBucket` tally.
    pub transit_bucket: FamilyTally,
    /// `AnticipatedThermalState` tally.
    pub anticipated: FamilyTally,
    /// The anticipated family's fan-out coverage summary.
    pub anticipated_coverage: AnticipatedCoverage,
    /// Every family without its own reconcile arm (the identity-reject default).
    pub other_identity: FamilyTally,
    /// Every dropped source slot's own identity and dating, in ascending
    /// source position — [`dropped_source_positions`] made readable.
    pub dropped_source_slots: Vec<SlotDetail>,
    /// Every straddling ([`RebindOp::Renormalize`]) target slot's own
    /// identity and dating, in ascending target position.
    pub straddling_slots: Vec<SlotDetail>,
}

impl BoundaryReconciliationReport {
    fn tally_mut(&mut self, family: Option<StateFamily>) -> &mut FamilyTally {
        match family {
            Some(StateFamily::HydroStorage) => &mut self.storage,
            Some(StateFamily::HydroInflowLag) => &mut self.inflow_lag,
            Some(StateFamily::HydroTransitBucket) => &mut self.transit_bucket,
            Some(StateFamily::AnticipatedThermalState) => &mut self.anticipated,
            None => &mut self.other_identity,
        }
    }

    fn families(&self) -> [(&'static str, FamilyTally); 5] {
        [
            (family_label(Some(StateFamily::HydroStorage)), self.storage),
            (
                family_label(Some(StateFamily::HydroInflowLag)),
                self.inflow_lag,
            ),
            (
                family_label(Some(StateFamily::HydroTransitBucket)),
                self.transit_bucket,
            ),
            (
                family_label(Some(StateFamily::AnticipatedThermalState)),
                self.anticipated,
            ),
            (family_label(None), self.other_identity),
        ]
    }

    /// The four aggregate tallies (copy, fan-out, default-zero, dropped-source)
    /// summed across every family, shared with the CLI's own rendering.
    #[must_use]
    pub fn tally_totals(&self) -> (usize, usize, usize, usize) {
        let families = self.families();
        let copy: usize = families.iter().map(|(_, t)| t.copy).sum();
        let fan_out: usize = families.iter().map(|(_, t)| t.fan_out).sum();
        let default_zero: usize = families.iter().map(|(_, t)| t.default_zero).sum();
        let dropped: usize = families.iter().map(|(_, t)| t.dropped_source).sum();
        (copy, fan_out, default_zero, dropped)
    }

    /// One-line reconciliation summary: the dimension-only notice on the skip
    /// path, otherwise the totals across every family.
    #[must_use]
    pub fn summary_line(&self) -> String {
        if !self.reconciled {
            return "boundary reconciliation: dimension-only load (entity manifest absent); no \
                    per-family fan-out tally"
                .to_string();
        }
        let (total_copy, total_fan_out, total_default_zero, total_dropped) = self.tally_totals();
        format!(
            "boundary reconciliation: {total_copy} copied, {total_fan_out} fanned out, \
             {total_default_zero} defaulted to 0.0, {total_dropped} source slots dropped"
        )
    }

    /// The dropped-source counts behind a superset boundary (a source pricing
    /// state this study does not model), or `None` when every family's count
    /// is zero. Carries no framing and no per-slot examples — the caller
    /// supplies the framing.
    #[must_use]
    pub fn superset_summary(&self) -> Option<String> {
        let families = self.families();
        let total: usize = families.iter().map(|(_, tally)| tally.dropped_source).sum();
        if total == 0 {
            return None;
        }
        let per_family = families
            .iter()
            .filter(|(_, tally)| tally.dropped_source > 0)
            .map(|(name, tally)| format!("{name}: {}", tally.dropped_source))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "{total} source slot(s) price entities this study does not model ({per_family})"
        ))
    }

    /// Per-family reconciliation breakdown, the anticipated coverage line,
    /// then the capped `dropped:`/`straddling:` slot lines — the verbose
    /// detail behind [`Self::summary_line`]. Empty on the dimension-only skip
    /// path.
    #[must_use]
    pub fn detail_lines(&self) -> Vec<String> {
        if !self.reconciled {
            return Vec::new();
        }
        let families = self.families();
        let mut lines = Vec::with_capacity(families.len() + 1 + 2 * (DETAIL_LINES_SLOT_CAP + 1));
        for (name, tally) in families {
            lines.push(format!(
                "{name}: COPY={}, FAN-OUT={}, DEFAULT-0.0={}, DROP={}",
                tally.copy, tally.fan_out, tally.default_zero, tally.dropped_source
            ));
        }
        lines.push(format!(
            "anticipated: {} source delivery intervals fanned to {} target slots ({} \
             straddling, overlap-blended), {} months defaulted",
            self.anticipated_coverage.source_interval_count,
            self.anticipated.fan_out,
            self.anticipated.straddling,
            self.anticipated.default_zero
        ));
        push_capped_slot_lines(&mut lines, "dropped", &self.dropped_source_slots);
        push_capped_slot_lines(&mut lines, "straddling", &self.straddling_slots);
        lines
    }
}

/// [`BoundaryReconciliationReport::detail_lines`]'s per-list rendering cap for
/// `dropped_source_slots`/`straddling_slots` — a terminal-rendering concern
/// only; the report's own vectors stay uncapped.
const DETAIL_LINES_SLOT_CAP: usize = 5;

/// Renders `details`' first [`DETAIL_LINES_SLOT_CAP`] entries, appending a
/// `"... and N more"` line past the cap.
fn push_capped_slot_lines(lines: &mut Vec<String>, prefix: &str, details: &[SlotDetail]) {
    for detail in details.iter().take(DETAIL_LINES_SLOT_CAP) {
        let date_clause = match (detail.interval, detail.reference_date) {
            (Some((start, end)), _) => format!(" @ [{start}, {end})"),
            (None, Some(reference_date)) => format!(" @ {reference_date}"),
            (None, None) => String::new(),
        };
        lines.push(format!(
            "{prefix}: {} {}/{}{date_clause}",
            detail.family, detail.entity_id, detail.subindex
        ));
    }
    if details.len() > DETAIL_LINES_SLOT_CAP {
        lines.push(format!(
            "... and {} more",
            details.len() - DETAIL_LINES_SLOT_CAP
        ));
    }
}

/// A sentinel-interval forward-family slot — the ring buffer's structural
/// pad: it carries no delivery or arrival and prices nothing, so it is
/// excluded from both [`classify_op`]'s target-side `default_zero` tally and
/// [`dropped_source_positions`]'s source-side `dropped_source` tally.
fn is_structural_pad(slot: &EntitySlot) -> bool {
    let is_forward_family = matches!(
        slot.family(),
        Some(StateFamily::AnticipatedThermalState | StateFamily::HydroTransitBucket)
    );
    is_forward_family && slot.interval_start == ENTITY_SLOT_DATE_SENTINEL
}

/// Classify one target slot's `(op, slot)` into `tally`. A `Zero` on a
/// structural pad ([`is_structural_pad`]) is excluded from every tally.
fn classify_op(op: &RebindOp, slot: &EntitySlot, tally: &mut FamilyTally) {
    match op {
        RebindOp::Copy(_) => tally.copy += 1,
        RebindOp::Blend(_) => tally.fan_out += 1,
        RebindOp::Renormalize(_) => {
            tally.fan_out += 1;
            tally.straddling += 1;
        }
        RebindOp::Zero => {
            if !is_structural_pad(slot) {
                tally.default_zero += 1;
            }
        }
        RebindOp::Reject { reason } => unreachable!(
            "build_rebind must convert Reject into an error before build_reconciliation_report \
             sees it: {reason}"
        ),
    }
}

/// Build a [`BoundaryReconciliationReport`] over one `load_boundary_cuts`
/// reconciliation: a pure pass over the aligned `target`/`rebind` vectors,
/// against the same [`build_source_interval_index`] output [`build_rebind`]
/// resolved against. The two slot-detail vectors are filled in ascending
/// source and target position respectively — positional orders, never sorted.
///
/// # Panics
///
/// Panics if `rebind` contains a [`RebindOp::Reject`] — the same
/// `build_rebind` postcondition violation [`rebind_cut`] guards against.
pub(crate) fn build_reconciliation_report(
    source: &[EntitySlot],
    target: &[EntitySlot],
    rebind: &[RebindOp],
    source_index: &HashMap<SourceKey, Vec<SourceInterval>>,
) -> BoundaryReconciliationReport {
    let mut report = BoundaryReconciliationReport {
        reconciled: true,
        ..BoundaryReconciliationReport::default()
    };

    for (slot, op) in target.iter().zip(rebind) {
        let family = slot.family();
        if matches!(op, RebindOp::Renormalize(_)) {
            report.straddling_slots.push(slot_detail(slot));
        }
        classify_op(op, slot, report.tally_mut(family));
    }

    for pos in dropped_source_positions(source, rebind) {
        if let Some(slot) = source.get(pos) {
            report.tally_mut(slot.family()).dropped_source += 1;
            report.dropped_source_slots.push(slot_detail(slot));
        }
    }

    let mut source_interval_count = 0;
    // AnticipatedCoverage is anticipated-only; the shared index also carries
    // HydroTransitBucket entries.
    for (&(family_byte, _), intervals) in source_index {
        if family_byte != StateFamily::AnticipatedThermalState.code() {
            continue;
        }
        source_interval_count += intervals.len();
    }
    report.anticipated_coverage.source_interval_count = source_interval_count;

    report
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::{
        BoundaryReconciliationReport, FamilyTally, RebindOp, SlotDetail, build_boundary_fold,
        build_identity_index, build_rebind, build_reconciliation_report,
        build_source_interval_index, dropped_source_positions, rebind_cut, slot_detail,
    };
    use crate::SddpError;
    use crate::test_support::{
        anticipated_slot, anticipated_slot_at, inflow_lag_slot, storage_slot, transit_bucket_slot,
        transit_bucket_slot_over, ymd,
    };
    use cobre_core::{AnticipatedCommitmentHistory, EntityId};
    use cobre_io::{EntitySlot, OwnedPolicyCutRecord, encode_slot_date};

    /// The two-step index-then-rebind call `load_boundary_cuts` performs,
    /// collapsed into one.
    fn rebind_via_index(
        source: &[EntitySlot],
        target: &[EntitySlot],
        boundary_date: NaiveDate,
    ) -> Result<Vec<RebindOp>, SddpError> {
        let source_index = build_source_interval_index(source)?;
        let identity_index = build_identity_index(source);
        build_rebind(
            source,
            target,
            boundary_date,
            &source_index,
            &identity_index,
        )
    }

    /// [`rebind_via_index`]'s counterpart for
    /// [`build_reconciliation_report`].
    fn report_via_index(
        source: &[EntitySlot],
        target: &[EntitySlot],
        rebind: &[RebindOp],
    ) -> BoundaryReconciliationReport {
        let source_index = build_source_interval_index(source).unwrap();
        build_reconciliation_report(source, target, rebind, &source_index)
    }

    /// [`build_boundary_fold`]'s index-then-fold call, collapsed into one —
    /// mirrors [`rebind_via_index`].
    fn fold_via_index(
        source: &[EntitySlot],
        fixed_windows: &[AnticipatedCommitmentHistory],
    ) -> Vec<(usize, f64)> {
        let source_index = build_source_interval_index(source).unwrap();
        build_boundary_fold(source, fixed_windows, &source_index)
    }

    fn owned_cut(coefficients: Vec<f64>) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 0.0,
            coefficients,
            is_active: true,
        }
    }

    /// For tests exercising only the identity families or a sentinel-padded
    /// forward-family slot, none of which reads `boundary_date`.
    fn arbitrary_boundary_date() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid calendar date")
    }

    fn fixed_window(
        thermal_id: i32,
        start: NaiveDate,
        end: NaiveDate,
        value_mw: f64,
    ) -> AnticipatedCommitmentHistory {
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(thermal_id),
            start_date: start,
            end_date: end,
            value_mw,
        }
    }

    /// Given a `source` and `target` manifest of equal shape (all storage),
    /// when `build_rebind` runs, then it returns one `Copy` per slot at the
    /// matching source position.
    #[test]
    fn build_rebind_equal_shape_all_storage_yields_identity_copy() {
        let source = vec![storage_slot(1), storage_slot(2), storage_slot(3)];
        let target = source.clone();

        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Copy(0), RebindOp::Copy(1), RebindOp::Copy(2)]
        );
    }

    /// Given a target manifest mixing storage and inflow-lag slots that both
    /// match the source by identity, when `build_rebind` runs, then every slot
    /// resolves to `Copy` at its identity-matched source position — order
    /// independent, since matching is by identity, not position.
    #[test]
    fn build_rebind_storage_and_lag_identity_match_yields_copy() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            storage_slot(2),
            inflow_lag_slot(2, 1),
        ];
        let target = vec![storage_slot(2), inflow_lag_slot(1, 1)];

        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(2), RebindOp::Copy(1)]);
    }

    /// Given a target storage slot for a hydro absent from the source, when
    /// `build_rebind` runs, then it rejects, naming the unpriced hydro.
    #[test]
    fn build_rebind_storage_miss_rejects_naming_hydro() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(42)];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("42"), "must name the unpriced hydro: {msg}");
    }

    /// Given a target inflow-lag slot whose `(entity_id, subindex)` is absent
    /// from the source, when `build_rebind` runs, then it rejects as a
    /// lag-depth incompatibility, naming the offending hydro and lag depth.
    #[test]
    fn build_rebind_lag_miss_rejects_naming_lag_depth() {
        let source = vec![storage_slot(1), inflow_lag_slot(1, 1)];
        let target = vec![storage_slot(1), inflow_lag_slot(1, 2)];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains('1'), "must name hydro 1: {msg}");
        assert!(msg.contains('2'), "must name lag depth 2: {msg}");
    }

    /// Given a source and a target inflow-lag slot for the same hydro and lag
    /// depth, but with DIFFERENT non-sentinel `reference_date`s, when
    /// `build_rebind` runs, then it rejects naming the hydro, the lag depth,
    /// and both dates — never the identity-miss lag-depth-incompatibility
    /// wording.
    #[test]
    fn inflow_lag_differing_reference_dates_reject_naming_both_dates() {
        let source = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_301)];
        let target = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_401)];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("hydro 1"), "must name hydro 1: {msg}");
        assert!(msg.contains("lag depth 2"), "must name lag depth 2: {msg}");
        assert!(
            msg.contains("2030-03-01"),
            "must name the source date: {msg}"
        );
        assert!(
            msg.contains("2030-04-01"),
            "must name the target date: {msg}"
        );
        assert!(
            !msg.contains("lag-depth-incompatible"),
            "a date mismatch is not a depth incompatibility: {msg}"
        );
    }

    /// Given a source and a target inflow-lag slot agreeing on a non-sentinel
    /// `reference_date`, when `build_rebind` runs, then the coefficient still
    /// copies by identity — matching dates is not a new requirement, only a
    /// mismatch is newly rejected.
    #[test]
    fn inflow_lag_matching_reference_dates_copy() {
        let source = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_301)];
        let target = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_301)];

        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0)]);
    }

    /// Given an inflow-lag slot pair where exactly one side carries a real
    /// `reference_date` and the other the sentinel — both directions — when
    /// `build_rebind` runs, then it still copies by identity: the
    /// `reserve_boundary_inflow_lag_slots` bridge carve-out.
    #[test]
    fn inflow_lag_sentinel_on_either_side_copies_by_identity() {
        let dated_source = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_301)];
        let sentinel_target = vec![inflow_lag_slot(1, 2)];
        let rebind =
            rebind_via_index(&dated_source, &sentinel_target, arbitrary_boundary_date()).unwrap();
        assert_eq!(
            rebind,
            vec![RebindOp::Copy(0)],
            "dated source, sentinel target"
        );

        let sentinel_source = vec![inflow_lag_slot(1, 2)];
        let dated_target = vec![inflow_lag_slot(1, 2).with_reference_date(20_300_301)];
        let rebind =
            rebind_via_index(&sentinel_source, &dated_target, arbitrary_boundary_date()).unwrap();
        assert_eq!(
            rebind,
            vec![RebindOp::Copy(0)],
            "sentinel source, dated target"
        );
    }

    /// A target storage slot never matches a source slot sharing its
    /// `entity_id` under a different family (inflow-lag): the entity is
    /// identified by `(entity_type, entity_id[, subindex])` jointly, never
    /// `entity_id` alone — the entity is never relaxed across families.
    #[test]
    fn build_rebind_entity_never_crosses_family() {
        let source = vec![inflow_lag_slot(5, 1)];
        let target = vec![storage_slot(5)];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
    }

    /// Given a rebind of all `Copy` ops and a source cut, when `rebind_cut`
    /// runs, then the returned coefficient vector equals the source cut's
    /// coefficients verbatim.
    #[test]
    fn rebind_cut_all_copy_matches_source_coefficients_verbatim() {
        let source = vec![storage_slot(1), storage_slot(2)];
        let target = source.clone();
        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();
        let cut = owned_cut(vec![10.0, -5.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, cut.coefficients);
    }

    /// Given a rebind containing a `Zero` op at position `j`, when
    /// `rebind_cut` runs, then `coefficients[j] == 0.0`.
    #[test]
    fn rebind_cut_zero_op_yields_zero_coefficient() {
        let rebind = vec![RebindOp::Copy(0), RebindOp::Zero];
        let cut = owned_cut(vec![7.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![7.0, 0.0]);
    }

    /// Given a `Blend` op with two weighted source positions, when
    /// `rebind_cut` runs, then it returns their weighted sum — the mechanical
    /// application `build_rebind`'s weight computation feeds.
    #[test]
    fn rebind_cut_blend_op_sums_weighted_source_positions() {
        let rebind = vec![RebindOp::Blend(vec![(0, 0.25), (1, 0.75)])];
        let cut = owned_cut(vec![100.0, 200.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![175.0]);
    }

    /// Given a `Renormalize` op, when `rebind_cut` runs, then it applies the
    /// identical weighted-sum mechanics as `Blend` — the distinction is only
    /// in the weight value, never in how `rebind_cut` uses it.
    #[test]
    fn rebind_cut_renormalize_op_sums_weighted_source_positions() {
        let rebind = vec![RebindOp::Renormalize(vec![(0, 0.5)])];
        let cut = owned_cut(vec![40.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients, vec![20.0]);
    }

    #[test]
    fn build_rebind_rejects_target_slot_with_no_source_counterpart() {
        let source = vec![storage_slot(1)];
        let target = vec![storage_slot(1), storage_slot(2)];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        assert!(matches!(err, SddpError::Validation(_)));
    }

    #[test]
    fn dropped_source_positions_reports_unreferenced_source_slots() {
        let source = vec![storage_slot(1), storage_slot(2), storage_slot(3)];
        let rebind = vec![RebindOp::Copy(1)];

        let dropped = dropped_source_positions(&source, &rebind);

        assert_eq!(dropped, vec![0, 2]);
    }

    /// A storage slot no op references is a genuine superset drop, never a
    /// structural pad — `is_structural_pad` never widens to storage.
    #[test]
    fn dropped_source_positions_still_reports_an_unreferenced_storage_slot() {
        let source = vec![storage_slot(1), storage_slot(2)];
        let rebind = vec![RebindOp::Copy(0)];

        let dropped = dropped_source_positions(&source, &rebind);

        assert_eq!(dropped, vec![1]);
    }

    /// Given a source mixing a referenced storage slot, an unreferenced
    /// dated anticipated slot, and an unreferenced sentinel anticipated pad,
    /// when `dropped_source_positions` runs, then only the dated slot's
    /// position is returned — the pad is a structural pad and is excluded.
    #[test]
    fn dropped_source_positions_excludes_sentinel_forward_family_pads() {
        let source = vec![
            storage_slot(1),
            anticipated_slot_at(9, 1, 20_320_101),
            anticipated_slot(9, 2),
        ];
        let rebind = vec![RebindOp::Copy(0)];

        let dropped = dropped_source_positions(&source, &rebind);

        assert_eq!(dropped, vec![1]);
    }

    /// Given a target transit-bucket slot at the sentinel `interval_start`,
    /// when `build_rebind` runs, then it resolves to `Zero` even when the
    /// source carries the same `(entity_id, subindex)` transit slot at the
    /// same sentinel — transit no longer resolves by identity, so a
    /// same-identity match with no arrival interval never copies.
    #[test]
    fn build_rebind_sentinel_transit_bucket_target_is_always_zero() {
        let source = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let matching_target = vec![storage_slot(1), transit_bucket_slot(2, 1)];

        let rebind =
            rebind_via_index(&source, &matching_target, arbitrary_boundary_date()).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// Given a source and a target transit-bucket slot sharing the SAME
    /// arrival interval, when `build_rebind` runs, then it resolves to
    /// `Blend` with a single unit-weight term — the exact-match acceptance
    /// case, whose coefficient is bit-identical to the source's own — and
    /// the reconciliation report tallies it as `fan_out`, never `copy`: the
    /// join mechanism changed even though the coefficient did not.
    #[test]
    fn transit_bucket_identical_arrival_interval_blends_at_unit_weight() {
        let start = encode_slot_date(ymd(2026, 4, 1));
        let end = encode_slot_date(ymd(2026, 5, 1));
        let source = vec![transit_bucket_slot_over(2, 1, start, end)];
        let target = vec![transit_bucket_slot_over(2, 1, start, end)];
        let boundary_date = ymd(2026, 4, 1);

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        assert_eq!(rebind, vec![RebindOp::Blend(vec![(0, 1.0)])]);

        let cut = owned_cut(vec![77.0]);
        let coefficients = rebind_cut(&cut, &rebind);
        assert_eq!(
            coefficients[0].to_bits(),
            cut.coefficients[0].to_bits(),
            "an identical-interval unit-weight Blend term must reproduce the source \
             coefficient bit-for-bit"
        );

        let report = report_via_index(&source, &target, &rebind);
        assert_eq!(
            report.transit_bucket.fan_out, 1,
            "an interval-join match tallies as fan_out, not copy"
        );
        assert_eq!(report.transit_bucket.copy, 0);
    }

    /// Given a target transit-bucket slot whose arrival interval is fully
    /// inside the ONE wider source interval covering it, when `build_rebind`
    /// runs, then it resolves to `Blend` (full coverage, never
    /// `Renormalize`) with the `÷H_m`-conserving weight — a target narrower
    /// than its covering source interval draws a fractional share, not the
    /// source's own coefficient verbatim, exactly as
    /// `build_rebind_anticipated_partial_month_yields_blend_fractional_weight`
    /// establishes for the anticipated family this resolver is shared with.
    #[test]
    fn transit_bucket_target_inside_one_source_interval_blends_at_fractional_source_hours() {
        let source = vec![transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        )];
        let boundary_date = ymd(2026, 4, 1);
        let target = vec![transit_bucket_slot_over(
            2,
            100,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 4, 8)),
        )];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        match &rebind[0] {
            RebindOp::Blend(terms) => {
                assert_eq!(terms.len(), 1);
                assert_eq!(terms[0].0, 0);
                let expected_weight = (7.0 * 24.0) / (30.0 * 24.0);
                assert!(
                    (terms[0].1 - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {} != expected {expected_weight}",
                    terms[0].1
                );
            }
            other => panic!("expected Blend, got {other:?}"),
        }
    }

    /// Given a target transit-bucket slot whose arrival interval does not
    /// overlap the source's, when `build_rebind` runs, then it resolves to
    /// `Zero` — no covered interval, nothing to reconcile to.
    #[test]
    fn transit_bucket_non_overlapping_arrival_interval_zeroes() {
        let source = vec![transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        )];
        let boundary_date = ymd(2026, 4, 1);
        let target = vec![transit_bucket_slot_over(
            2,
            100,
            encode_slot_date(ymd(2026, 5, 1)),
            encode_slot_date(ymd(2026, 6, 1)),
        )];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        assert_eq!(rebind, vec![RebindOp::Zero]);
    }

    /// Given a target manifest with two transit buckets for one downstream
    /// hydro at lags 1 and 2 whose arrival intervals are disjoint, and a
    /// source carrying only the lag-2 interval AT A DIFFERENT subindex, when
    /// `build_rebind` runs, then the lag-2 target draws the source
    /// coefficient and the lag-1 target resolves to `Zero` — the join
    /// discriminates by interval alone, with `subindex` playing no part.
    #[test]
    fn transit_bucket_join_ignores_subindex() {
        let lag1 = (ymd(2026, 4, 1), ymd(2026, 4, 8));
        let lag2 = (ymd(2026, 4, 8), ymd(2026, 4, 15));
        let source = vec![transit_bucket_slot_over(
            2,
            7,
            encode_slot_date(lag2.0),
            encode_slot_date(lag2.1),
        )];
        let target = vec![
            transit_bucket_slot_over(2, 1, encode_slot_date(lag1.0), encode_slot_date(lag1.1)),
            transit_bucket_slot_over(2, 2, encode_slot_date(lag2.0), encode_slot_date(lag2.1)),
        ];

        let rebind = rebind_via_index(&source, &target, lag1.0).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Zero, RebindOp::Blend(vec![(0, 1.0)])],
            "the lag-1 target has no overlapping source interval and zeroes; the lag-2 target \
             draws the source's coefficient by interval overlap alone, its differing subindex \
             (7 on the source, 2 on the target) irrelevant"
        );
    }

    /// Given a target transit-bucket slot at the sentinel `interval_start`
    /// (pre-fan-out padding), when `build_rebind` runs, then it resolves to
    /// `Zero` regardless of `boundary_date`, and the reconciliation report
    /// excludes it from every tally — the structural-pad exclusion, mirroring
    /// `anticipated_sentinel_interval_is_a_structural_pad`.
    #[test]
    fn transit_bucket_sentinel_interval_is_a_structural_pad() {
        let source = vec![transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        )];
        let target = vec![transit_bucket_slot(2, 1)];
        let boundary_date = arbitrary_boundary_date();

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();
        assert_eq!(rebind, vec![RebindOp::Zero]);

        let report = report_via_index(&source, &target, &rebind);
        assert_eq!(
            report.transit_bucket.default_zero, 0,
            "a sentinel-interval Zero must be excluded from the default_zero tally"
        );
        assert_eq!(report.transit_bucket.copy, 0);
        assert_eq!(report.transit_bucket.fan_out, 0);
    }

    /// Given a target slot whose `entity_type` is a byte no `StateFamily`
    /// variant defines, when `build_rebind` runs, then it still rejects by identity
    /// via `resolve_by_identity`, naming the target's own position and
    /// identity — unaffected by the forward-family dispatch this module adds.
    #[test]
    fn unrecognized_entity_type_still_rejects_by_identity() {
        let mut unrecognized = storage_slot(42);
        unrecognized.entity_type = 99;
        let source: Vec<EntitySlot> = vec![];
        let target = vec![unrecognized];

        let err = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("entity_type=99"),
            "must name the unrecognized entity_type: {msg}"
        );
    }

    /// Given a target anticipated slot with `interval_start == SENTINEL`,
    /// when `build_rebind` runs, then it resolves to `Zero` — even when the
    /// source carries an identical sentinel-dated slot at the same identity,
    /// proving the sentinel case is dispatched purely on the interval.
    #[test]
    fn build_rebind_sentinel_anticipated_target_is_always_zero() {
        let source = vec![storage_slot(1), anticipated_slot(9, 0)];
        let target = vec![storage_slot(1), anticipated_slot(9, 0)];

        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// Given a source manifest with one monthly anticipated slot and a
    /// target ring slot whose interval lies fully inside that month, when
    /// `build_rebind` runs, then it resolves to `Blend` with a single term
    /// weighted `overlap/H_M`.
    #[test]
    fn build_rebind_dated_anticipated_target_fully_covered_yields_blend() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 3, 1);
        let target = vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(
            encode_slot_date(boundary_date),
            encode_slot_date(ymd(2026, 4, 1)),
        )];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Blend(vec![(0, 1.0)])],
            "a monthly target fully inside one source month yields a single unit-weight term"
        );

        let cut = owned_cut(vec![42.5]);
        let coefficients = rebind_cut(&cut, &rebind);
        assert_eq!(
            coefficients[0].to_bits(),
            cut.coefficients[0].to_bits(),
            "a single unit-weight Blend term must reproduce the source coefficient \
             bit-for-bit, copy-equivalent"
        );
    }

    /// Given a target ring slot spanning one week fully inside a priced
    /// source month, when `build_rebind` runs, then it resolves to `Blend`
    /// with a fractional `overlap/H_M` weight.
    #[test]
    fn build_rebind_anticipated_partial_month_yields_blend_fractional_weight() {
        let source = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_401)];
        let start_w = ymd(2026, 4, 8);
        let end_w = ymd(2026, 4, 15);
        let target = vec![
            storage_slot(1),
            anticipated_slot_at(9, 100, 20_260_401)
                .with_interval(encode_slot_date(start_w), encode_slot_date(end_w)),
        ];

        let rebind = rebind_via_index(&source, &target, start_w).unwrap();

        match &rebind[1] {
            RebindOp::Blend(terms) => {
                assert_eq!(terms.len(), 1);
                assert_eq!(terms[0].0, 1);
                let expected_weight = (7.0 * 24.0) / (30.0 * 24.0);
                assert!(
                    (terms[0].1 - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {} != expected {expected_weight}",
                    terms[0].1
                );
            }
            other => panic!("expected Blend, got {other:?}"),
        }
    }

    /// Given a source anticipated slot whose OWN delivery interval is a
    /// 7-day stage (not a calendar month) and a target slot covering that
    /// same span, when `build_rebind` runs, then the fanned coefficient
    /// equals the source's own value (weight `1.0`) — under the retired
    /// month-reconstruction the same fixture would have produced `7/30` of
    /// it.
    #[test]
    fn anticipated_source_interval_shorter_than_a_month_weights_by_its_own_hours() {
        let source_coeff = 42.0;
        let start = ymd(2026, 4, 1);
        let end = ymd(2026, 4, 8);
        let source = vec![
            anticipated_slot_at(9, 0, 20_260_401)
                .with_interval(encode_slot_date(start), encode_slot_date(end)),
        ];
        let target = vec![
            anticipated_slot_at(9, 100, 20_260_401)
                .with_interval(encode_slot_date(start), encode_slot_date(end)),
        ];

        let rebind = rebind_via_index(&source, &target, start).unwrap();

        assert_eq!(
            rebind,
            vec![RebindOp::Blend(vec![(0, 1.0)])],
            "a target fully covering a 7-day source stage weights 1.0, not 7/30"
        );

        let cut = owned_cut(vec![source_coeff]);
        assert_eq!(rebind_cut(&cut, &rebind), vec![source_coeff]);
    }

    /// Given a source anticipated slot whose `interval_start` is a raw,
    /// undecodable value, when `build_source_interval_index` runs, then it
    /// rejects, naming the slot's identity and the undecodable raw value.
    #[test]
    fn source_interval_index_rejects_an_undecodable_live_interval() {
        let source =
            vec![anticipated_slot_at(9, 0, 20_260_301).with_interval(20_261_301, 20_260_401)];

        let err = build_source_interval_index(&source).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("entity_id=9"), "must name the entity: {msg}");
        assert!(
            msg.contains("20261301"),
            "must name the undecodable raw value: {msg}"
        );
    }

    /// Given a target ring slot whose interval straddles a priced month and
    /// an unpriced one, when `build_rebind` runs, then it resolves to
    /// `Renormalize` with a single covered term scaled to the full slot —
    /// never an implicit `0.0` deflation term for the uncovered days.
    #[test]
    fn build_rebind_anticipated_straddle_into_unpriced_yields_renormalize_no_zero_term() {
        let source_coeff = 300.0;
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let start_w = ymd(2026, 2, 26);
        let end_w = ymd(2026, 3, 5);
        let target = vec![
            anticipated_slot_at(9, 100, 20_260_301)
                .with_interval(encode_slot_date(start_w), encode_slot_date(end_w)),
        ];

        let rebind = rebind_via_index(&source, &target, start_w).unwrap();

        let h_w = f64::from(u32::try_from((end_w - start_w).num_days()).unwrap()) * 24.0;
        let h_m = 31.0 * 24.0;
        match &rebind[0] {
            RebindOp::Renormalize(terms) => {
                assert_eq!(
                    terms.len(),
                    1,
                    "no 0.0 deflation term for the uncovered (unpriced February) days"
                );
                let (pos, weight) = terms[0];
                assert_eq!(pos, 0);
                let expected_weight = h_w / h_m;
                assert!(
                    (weight - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {weight} != expected {expected_weight}"
                );
            }
            other => panic!("expected Renormalize, got {other:?}"),
        }

        let cut = owned_cut(vec![source_coeff]);
        let coefficients = rebind_cut(&cut, &rebind);
        let expected_coeff = source_coeff * h_w / h_m;
        assert!(
            (coefficients[0] - expected_coeff).abs() < expected_coeff.abs() * 1e-9,
            "renormalized coefficient {} != expected {expected_coeff} (source · H_w/H_priced)",
            coefficients[0]
        );
    }

    /// Given a target ring slot whose interval falls in a month the source
    /// carries no anticipated slot for, when `build_rebind` runs, then it
    /// resolves to `Zero` — no covered month, nothing to reconcile to.
    #[test]
    fn build_rebind_anticipated_no_covered_month_yields_zero() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let start_w = ymd(2026, 5, 1);
        let end_w = ymd(2026, 5, 8);
        let target = vec![
            anticipated_slot_at(9, 100, 20_260_301)
                .with_interval(encode_slot_date(start_w), encode_slot_date(end_w)),
        ];

        let rebind = rebind_via_index(&source, &target, start_w).unwrap();

        assert_eq!(rebind, vec![RebindOp::Zero]);
    }

    /// Given a target anticipated slot whose fan-out weights come purely from
    /// its OWN `interval_start`/`interval_end` fields, when `build_rebind`
    /// runs, then it fans out exactly as the (retired) parallel interval
    /// vector would have.
    #[test]
    fn anticipated_target_reads_its_own_manifest_interval() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 3, 1);
        let target = vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(
            encode_slot_date(ymd(2026, 3, 1)),
            encode_slot_date(ymd(2026, 3, 15)),
        )];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        let h_w = 14.0 * 24.0;
        let h_m = 31.0 * 24.0;
        match &rebind[0] {
            RebindOp::Blend(terms) => {
                assert_eq!(terms.len(), 1);
                assert_eq!(terms[0].0, 0);
                let expected_weight = h_w / h_m;
                assert!(
                    (terms[0].1 - expected_weight).abs() < expected_weight * 1e-9,
                    "weight {} != expected {expected_weight}",
                    terms[0].1
                );
            }
            other => panic!("expected Blend, got {other:?}"),
        }
    }

    /// Given a live anticipated target slot whose interval ENDS at the
    /// boundary date — an in-study delivery, discharged inside the current
    /// horizon (e.g. a `K = 0` sub-stage-lead thermal maturing at the
    /// terminal stage) — when `build_rebind` runs, then it resolves to
    /// `Zero`: the terminal boundary prices no within-horizon delivery. It
    /// must never reject (which would abort a legitimate load) and never fan
    /// out against the source months (which would wrongly `Blend`), even
    /// though the source carries a month whose date would overlap it if it
    /// were wrongly fanned out.
    #[test]
    fn anticipated_target_ending_at_the_boundary_date_zeroes() {
        let source = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 4, 1);
        let target = vec![
            storage_slot(1),
            anticipated_slot_at(9, 0, 20_260_301).with_interval(
                encode_slot_date(ymd(2026, 3, 1)),
                encode_slot_date(boundary_date),
            ),
        ];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        assert_eq!(rebind, vec![RebindOp::Copy(0), RebindOp::Zero]);
    }

    /// The predicate's other side: a live anticipated target slot whose
    /// interval reaches PAST the boundary date fans out against the
    /// overlapping source month, never zeroes.
    #[test]
    fn anticipated_target_reaching_past_the_boundary_date_fans_out() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 2, 1);
        let target = vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(
            encode_slot_date(ymd(2026, 3, 1)),
            encode_slot_date(ymd(2026, 4, 1)),
        )];

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        assert!(
            matches!(rebind[0], RebindOp::Blend(_)),
            "an interval reaching past the boundary date must fan out, not zero: {:?}",
            rebind[0]
        );
    }

    /// Given a target anticipated slot at the sentinel `interval_start`
    /// (pre-fan-out padding), when `build_rebind` runs, then it resolves to
    /// `Zero` regardless of `boundary_date`, and the reconciliation report
    /// excludes it from every tally — the structural-pad exclusion.
    #[test]
    fn anticipated_sentinel_interval_is_a_structural_pad() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let target = vec![anticipated_slot(9, 100)];
        let boundary_date = arbitrary_boundary_date();

        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();
        assert_eq!(rebind, vec![RebindOp::Zero]);

        let report = report_via_index(&source, &target, &rebind);
        assert_eq!(
            report.anticipated.default_zero, 0,
            "a sentinel-interval Zero must be excluded from the default_zero tally"
        );
        assert_eq!(report.anticipated.copy, 0);
        assert_eq!(report.anticipated.fan_out, 0);
    }

    /// Given a target anticipated slot whose `interval_start` is a raw,
    /// undecodable value (never expected past `read_policy_checkpoint`'s own
    /// date validation, but not ruled out by the type system), when
    /// `build_rebind` runs, then it rejects, naming the slot's identity and
    /// the undecodable raw value.
    #[test]
    fn anticipated_undecodable_interval_rejects_naming_the_slot() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let target =
            vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(20_261_301, 20_260_401)];

        let err = rebind_via_index(&source, &target, ymd(2026, 3, 1)).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("entity_id=9"), "must name the entity: {msg}");
        assert!(
            msg.contains("subindex=100"),
            "must name the subindex: {msg}"
        );
        assert!(
            msg.contains("20261301"),
            "must name the undecodable raw value: {msg}"
        );
    }

    /// Given a source already shaped identically to the target (storage, lag,
    /// and a sentinel-dated anticipated slot), when `build_rebind` and
    /// `rebind_cut` run over the source cut, then every resulting coefficient
    /// is bit-identical to the source's own coefficient (`f64::to_bits`) — the
    /// superset property: reconciling a target-shaped source never regresses
    /// today's exact-match load. The sentinel-anticipated position stays
    /// `0.0` on both sides — a masked state dimension never holds a value —
    /// so the forced `Zero` output equals `source`'s own coefficient there.
    #[test]
    fn superset_target_shaped_source_reconciles_bit_identically() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            anticipated_slot(9, 0),
        ];
        let target = source.clone();
        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();
        let cut = owned_cut(vec![10.5, -3.25, 0.0]);

        let coefficients = rebind_cut(&cut, &rebind);

        assert_eq!(coefficients.len(), cut.coefficients.len());
        for (actual, expected) in coefficients.iter().zip(cut.coefficients.iter()) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "reconciling a target-shaped source must reproduce its coefficients \
                 bit-for-bit: {actual} != {expected}"
            );
        }
    }

    /// Given a source carrying a DATED transit bucket and a DATED anticipated
    /// slot that no target op references (the target has neither family),
    /// when `dropped_source_positions` runs, then both positions are reported
    /// dropped, in ascending order: a dated forward-family slot prices a real
    /// interval the study does not model, so it is a genuine superset drop and
    /// not a structural pad (`is_structural_pad`).
    #[test]
    fn dropped_source_positions_reports_unreferenced_transit_and_anticipated_source_slots() {
        let source = vec![
            storage_slot(1),
            transit_bucket_slot_over(2, 1, 20_260_401, 20_260_501),
            anticipated_slot_at(9, 0, 20_260_401),
        ];
        let target = vec![storage_slot(1)];
        let rebind = rebind_via_index(&source, &target, arbitrary_boundary_date()).unwrap();

        let dropped = dropped_source_positions(&source, &rebind);

        assert_eq!(dropped, vec![1, 2]);
    }

    /// A source position referenced only by a `Blend` or `Renormalize` term
    /// (never a `Copy`) is not reported dropped — `dropped_source_positions`
    /// counts every referencing op, not `Copy` alone; an unreferenced pad
    /// still is.
    #[test]
    fn dropped_source_positions_blend_and_renormalize_referenced_slots_are_not_dropped() {
        let source = vec![
            storage_slot(1),
            storage_slot(2),
            storage_slot(3),
            storage_slot(4),
        ];
        let rebind = vec![
            RebindOp::Blend(vec![(0, 0.5), (1, 0.5)]),
            RebindOp::Renormalize(vec![(2, 1.0)]),
        ];

        let dropped = dropped_source_positions(&source, &rebind);

        assert_eq!(
            dropped,
            vec![3],
            "positions 0/1 (Blend terms) and 2 (Renormalize term) are referenced; only 3 is \
             dropped"
        );
    }

    /// Given a source with one monthly anticipated slot and a target with two
    /// ring slots exactly tiling that month, when `build_reconciliation_report`
    /// runs over the resulting `Blend` ops, then the anticipated family's
    /// `fan_out` equals the target slot count, `straddling`/`default_zero` are
    /// `0`, and the coverage line renders the expected shape.
    #[test]
    fn build_reconciliation_report_full_coverage_fan_out_matches_target_slot_count() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_401)];
        let boundary_date = ymd(2026, 4, 1);
        let target = vec![
            anticipated_slot_at(9, 100, 20_260_401).with_interval(
                encode_slot_date(ymd(2026, 4, 1)),
                encode_slot_date(ymd(2026, 4, 16)),
            ),
            anticipated_slot_at(9, 101, 20_260_401).with_interval(
                encode_slot_date(ymd(2026, 4, 16)),
                encode_slot_date(ymd(2026, 5, 1)),
            ),
        ];
        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        let report = report_via_index(&source, &target, &rebind);

        assert_eq!(
            report.anticipated.fan_out, 2,
            "N target slots -> fan_out == N"
        );
        assert_eq!(
            report.anticipated.straddling, 0,
            "full coverage: no straddle"
        );
        assert_eq!(report.anticipated.default_zero, 0);
        assert_eq!(
            report.anticipated_coverage.source_interval_count, 1,
            "one source delivery interval (K = 1)"
        );

        let lines = report.detail_lines();
        assert!(
            lines.iter().any(|l| l
                == "anticipated: 1 source delivery intervals fanned to 2 target slots (0 \
                    straddling, overlap-blended), 0 months defaulted"),
            "coverage line must match the expected shape: {lines:?}"
        );
    }

    /// Given a target ring slot straddling a priced month and an unpriced one
    /// (a `Renormalize` op), when `build_reconciliation_report` runs, then the
    /// anticipated family's `straddling` includes that slot and it is also
    /// counted in `fan_out`.
    #[test]
    fn build_reconciliation_report_renormalize_counts_in_fan_out_and_straddling() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 2, 26);
        let target = vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(
            encode_slot_date(ymd(2026, 2, 26)),
            encode_slot_date(ymd(2026, 3, 5)),
        )];
        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();
        assert!(
            matches!(rebind[0], RebindOp::Renormalize(_)),
            "fixture must straddle into unpriced time"
        );

        let report = report_via_index(&source, &target, &rebind);

        assert_eq!(
            report.anticipated.fan_out, 1,
            "Renormalize counts toward fan_out"
        );
        assert_eq!(report.anticipated.straddling, 1, "and toward straddling");
    }

    /// Given a target dated anticipated slot with no covered source month
    /// (`Zero`) and a sentinel anticipated slot (also `Zero`), when
    /// `build_reconciliation_report` runs, then the dated slot counts as
    /// `default_zero` while the sentinel slot is excluded — not a default.
    #[test]
    fn build_reconciliation_report_dated_zero_defaults_sentinel_zero_excluded() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 5, 1);
        let target = vec![
            anticipated_slot_at(9, 100, 20_260_301).with_interval(
                encode_slot_date(ymd(2026, 5, 1)),
                encode_slot_date(ymd(2026, 5, 8)),
            ),
            anticipated_slot(9, 101),
        ];
        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();
        assert_eq!(rebind, vec![RebindOp::Zero, RebindOp::Zero]);

        let report = report_via_index(&source, &target, &rebind);

        assert_eq!(
            report.anticipated.default_zero, 1,
            "only the dated no-covered-month Zero counts as a default"
        );
        assert_eq!(report.anticipated.fan_out, 0);
        assert_eq!(report.anticipated.copy, 0);
    }

    /// Given a target-shaped source (storage, lag, sentinel anticipated) that
    /// loads bit-identically, when `build_reconciliation_report` runs, then
    /// every family reports only `copy` (the sentinel-anticipated slot is
    /// excluded per its own classification, not counted as `copy` or
    /// `default_zero`), and `fan_out == 0` everywhere.
    #[test]
    fn build_reconciliation_report_target_shaped_superset_reports_copy_only() {
        let source = vec![
            storage_slot(1),
            inflow_lag_slot(1, 1),
            anticipated_slot(9, 0),
        ];
        let target = source.clone();
        let boundary_date = arbitrary_boundary_date();
        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();

        let report = report_via_index(&source, &target, &rebind);

        assert_eq!(report.storage.copy, 1);
        assert_eq!(report.inflow_lag.copy, 1);
        assert_eq!(
            report.anticipated.copy, 0,
            "sentinel Zero is excluded, not copy"
        );
        assert_eq!(
            report.anticipated.default_zero, 0,
            "sentinel Zero is excluded, not a default"
        );
        assert_eq!(report.anticipated.fan_out, 0);
        assert_eq!(report.transit_bucket.fan_out, 0);
        assert_eq!(report.other_identity.fan_out, 0);
    }

    /// Given a dated anticipated slot, when `slot_detail` reads it, then the
    /// `SlotDetail` carries the family, identity, and decoded `[start, end)`
    /// interval.
    #[test]
    fn dropped_slot_detail_carries_the_anticipated_interval() {
        let slot = anticipated_slot_at(33, 1, 20_320_101);

        let detail = slot_detail(&slot);

        assert_eq!(detail.family, "anticipated");
        assert_eq!(detail.entity_id, 33);
        assert_eq!(detail.subindex, 1);
        assert_eq!(detail.interval, Some((ymd(2032, 1, 1), ymd(2032, 2, 1))));
        assert_eq!(detail.reference_date, None);
    }

    /// Given an inflow-lag slot carrying a `reference_date`, when
    /// `slot_detail` reads it, then the `SlotDetail` carries the decoded
    /// `reference_date` and no interval.
    #[test]
    fn dropped_slot_detail_carries_the_inflow_lag_reference_date() {
        let slot = inflow_lag_slot(4, 2).with_reference_date(20_300_301);

        let detail = slot_detail(&slot);

        assert_eq!(detail.family, "inflow-lag");
        assert_eq!(detail.entity_id, 4);
        assert_eq!(detail.subindex, 2);
        assert_eq!(detail.reference_date, Some(ymd(2030, 3, 1)));
        assert_eq!(detail.interval, None);
    }

    /// Given a storage slot, when `slot_detail` reads it, then the
    /// `SlotDetail` carries neither an interval nor a reference date —
    /// storage populates no date field.
    #[test]
    fn dropped_slot_detail_for_storage_carries_no_date() {
        let slot = storage_slot(7);

        let detail = slot_detail(&slot);

        assert_eq!(detail.family, "storage");
        assert_eq!(detail.entity_id, 7);
        assert_eq!(detail.subindex, 0);
        assert_eq!(detail.interval, None);
        assert_eq!(detail.reference_date, None);
    }

    /// Given a target ring slot straddling a priced month and an unpriced one
    /// (the same `Renormalize` fixture as
    /// `build_reconciliation_report_renormalize_counts_in_fan_out_and_straddling`),
    /// when `build_reconciliation_report` runs, then `straddling_slots`
    /// carries that TARGET slot's own `[start, end)` span, never the source
    /// month it straddles into.
    #[test]
    fn straddling_slot_detail_carries_the_target_span() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_301)];
        let boundary_date = ymd(2026, 2, 26);
        let target = vec![anticipated_slot_at(9, 100, 20_260_301).with_interval(
            encode_slot_date(ymd(2026, 2, 26)),
            encode_slot_date(ymd(2026, 3, 5)),
        )];
        let rebind = rebind_via_index(&source, &target, boundary_date).unwrap();
        assert!(
            matches!(rebind[0], RebindOp::Renormalize(_)),
            "fixture must straddle into unpriced time"
        );

        let report = report_via_index(&source, &target, &rebind);

        assert_eq!(report.anticipated.straddling, 1);
        assert_eq!(report.straddling_slots.len(), 1);
        let detail = &report.straddling_slots[0];
        assert_eq!(detail.family, "anticipated");
        assert_eq!(detail.entity_id, 9);
        assert_eq!(detail.subindex, 100);
        assert_eq!(
            detail.interval,
            Some((ymd(2026, 2, 26), ymd(2026, 3, 5))),
            "the target slot's OWN span, not the source month it straddles into"
        );
    }

    /// A source manifest's two dropped slots (a transit bucket and an
    /// anticipated slot the target models neither of) are declared in one
    /// order; a second manifest declares the SAME two slots at swapped
    /// positions. `dropped_source_slots` must reflect each manifest's own
    /// ascending source position — checked with TWO simultaneous differences
    /// so a content-derived order (e.g. sorted by entity id, which would
    /// coincidentally match both manifests here) cannot pass unnoticed — and
    /// the two reports' entry sets must be equal regardless of declaration
    /// order.
    #[test]
    fn dropped_slot_details_are_declaration_order_invariant() {
        let transit = transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        );
        let anticipated = anticipated_slot_at(9, 0, 20_320_101);
        let target = vec![storage_slot(1)];
        let boundary_date = arbitrary_boundary_date();

        let source_a = vec![storage_slot(1), transit.clone(), anticipated.clone()];
        let rebind_a = rebind_via_index(&source_a, &target, boundary_date).unwrap();
        let report_a = report_via_index(&source_a, &target, &rebind_a);

        let source_b = vec![storage_slot(1), anticipated.clone(), transit.clone()];
        let rebind_b = rebind_via_index(&source_b, &target, boundary_date).unwrap();
        let report_b = report_via_index(&source_b, &target, &rebind_b);

        assert_eq!(report_a.dropped_source_slots.len(), 2);
        assert_eq!(report_b.dropped_source_slots.len(), 2);

        assert_eq!(
            report_a.dropped_source_slots[0].family, "transit-bucket",
            "manifest A declares transit (source position 1) before anticipated (position 2)"
        );
        assert_eq!(report_a.dropped_source_slots[1].family, "anticipated");

        assert_eq!(
            report_b.dropped_source_slots[0].family, "anticipated",
            "manifest B declares anticipated (position 1) before transit (position 2) — swapped \
             from A"
        );
        assert_eq!(report_b.dropped_source_slots[1].family, "transit-bucket");

        let entry_set = |report: &BoundaryReconciliationReport| {
            let mut entries: Vec<(&'static str, i32, u32)> = report
                .dropped_source_slots
                .iter()
                .map(|d| (d.family, d.entity_id, d.subindex))
                .collect();
            entries.sort_unstable();
            entries
        };
        assert_eq!(
            entry_set(&report_a),
            entry_set(&report_b),
            "the two reports must carry the same dropped entries regardless of declaration order"
        );
    }

    /// Given a report with seven dropped slots, when `detail_lines` renders,
    /// then it emits five `dropped:` lines and one `... and 2 more` line —
    /// the rendering cap.
    #[test]
    fn detail_lines_caps_dropped_entries_at_five_with_a_remainder_line() {
        let dropped_source_slots: Vec<SlotDetail> = (0..7)
            .map(|i| SlotDetail {
                family: "anticipated",
                entity_id: i,
                subindex: 0,
                interval: None,
                reference_date: None,
            })
            .collect();
        let report = BoundaryReconciliationReport {
            reconciled: true,
            dropped_source_slots,
            ..BoundaryReconciliationReport::default()
        };

        let lines = report.detail_lines();

        let dropped_lines: Vec<&String> = lines
            .iter()
            .filter(|l| l.starts_with("dropped: "))
            .collect();
        assert_eq!(
            dropped_lines.len(),
            5,
            "cap at five dropped lines: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "... and 2 more"),
            "must state the remainder count: {lines:?}"
        );
    }

    /// The default report (the shape `load_boundary_cuts` stores on its
    /// empty-manifest / dimension-only skip path) is `reconciled == false`,
    /// renders a dimension-only summary line, and carries no per-family detail.
    #[test]
    fn boundary_reconciliation_report_default_is_unreconciled_with_dimension_only_render() {
        let report = BoundaryReconciliationReport::default();

        assert!(!report.reconciled);
        assert!(
            report.summary_line().contains("dimension-only"),
            "must state a dimension-only load: {}",
            report.summary_line()
        );
        assert!(
            report.detail_lines().is_empty(),
            "the skip path carries no per-family detail"
        );
    }

    /// `summary_line()`'s reconciled-path wording is a correctness contract for
    /// `cobre validate` (asserted byte-exact by `cli_validate.rs`).
    #[test]
    fn summary_line_reconciled_wording_is_byte_exact() {
        let report = BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                copy: 2184,
                ..FamilyTally::default()
            },
            inflow_lag: FamilyTally {
                fan_out: 1,
                ..FamilyTally::default()
            },
            transit_bucket: FamilyTally {
                dropped_source: 2,
                ..FamilyTally::default()
            },
            anticipated: FamilyTally {
                default_zero: 1,
                ..FamilyTally::default()
            },
            ..BoundaryReconciliationReport::default()
        };

        assert_eq!(
            report.summary_line(),
            "boundary reconciliation: 2184 copied, 1 fanned out, 1 defaulted to 0.0, 2 source \
             slots dropped"
        );
    }

    /// Given a report whose `storage.dropped_source` is 2 and whose
    /// `anticipated.dropped_source` is 1, when `superset_summary` runs, then
    /// it names both dropping families with their counts, with no prefix,
    /// suffix, or trailing punctuation.
    #[test]
    fn superset_summary_names_every_dropping_family_with_its_count() {
        let report = BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                dropped_source: 2,
                ..FamilyTally::default()
            },
            anticipated: FamilyTally {
                dropped_source: 1,
                ..FamilyTally::default()
            },
            ..BoundaryReconciliationReport::default()
        };

        assert_eq!(
            report.superset_summary(),
            Some(
                "3 source slot(s) price entities this study does not model (storage: 2, \
                 anticipated: 1)"
                    .to_string()
            )
        );
    }

    /// A reconciled report whose every family's `dropped_source` is `0`
    /// carries no superset — `superset_summary` returns `None`.
    #[test]
    fn superset_summary_is_none_when_nothing_dropped() {
        let report = BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                copy: 4,
                ..FamilyTally::default()
            },
            ..BoundaryReconciliationReport::default()
        };

        assert_eq!(report.superset_summary(), None);
    }

    /// Given a report dropping more in `anticipated` than in `storage`, when
    /// `superset_summary` runs, then `storage` still appears first — the
    /// structural-families-first ordering is [`Self::families`]'s
    /// declaration order, not a count sort.
    #[test]
    fn superset_summary_orders_storage_before_anticipated() {
        let report = BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                dropped_source: 1,
                ..FamilyTally::default()
            },
            anticipated: FamilyTally {
                dropped_source: 5,
                ..FamilyTally::default()
            },
            ..BoundaryReconciliationReport::default()
        };

        assert_eq!(
            report.superset_summary(),
            Some(
                "6 source slot(s) price entities this study does not model (storage: 1, \
                 anticipated: 5)"
                    .to_string()
            )
        );
    }

    /// The unreconciled default report ([`BoundaryReconciliationReport::default`],
    /// the shape `load_boundary_cuts` stores on its skip path) carries every
    /// tally at zero, so `superset_summary` returns `None` by construction.
    #[test]
    fn superset_summary_is_none_on_the_unreconciled_default_report() {
        assert_eq!(
            BoundaryReconciliationReport::default().superset_summary(),
            None
        );
    }

    /// A single fixed window one whole week fully inside a priced source month
    /// (April, `H_M = 30·24`) folds to the hand-computed `(overlap/H_M)·value`
    /// at the source month's own position.
    #[test]
    fn build_boundary_fold_single_window_yields_hand_computed_factor() {
        let source = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_401)];
        let k = 1;
        let value = 50.0;
        let windows = vec![fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), value)];

        let fold = fold_via_index(&source, &windows);

        assert_eq!(fold.len(), 1);
        assert_eq!(fold[0].0, k);
        let expected = (7.0 * 24.0 / (30.0 * 24.0)) * value;
        assert!(
            (fold[0].1 - expected).abs() < expected * 1e-9,
            "factor {} != expected {expected}",
            fold[0].1
        );
    }

    /// A transit-bucket slot sharing the SAME `entity_id` and delivery span
    /// as an anticipated slot must never be folded in: the index's lookup
    /// key carries the family byte, so a transit slot (a different family,
    /// same id) is invisible to the anticipated fold — the guarantee that
    /// keeps a future transit-widened index from double-counting.
    #[test]
    fn boundary_fold_ignores_transit_slots_sharing_an_entity_id() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_401)];
        let windows = vec![fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 50.0)];
        let baseline = fold_via_index(&source, &windows);

        let mut with_transit = source.clone();
        with_transit.push(transit_bucket_slot(9, 1).with_interval(
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        ));
        let with_transit_fold = fold_via_index(&with_transit, &windows);

        assert_eq!(
            baseline, with_transit_fold,
            "a transit slot sharing entity_id must never be folded into the anticipated intercept"
        );
    }

    /// A fixed window in a month the source carries no anticipated slot for
    /// contributes nothing — mirrors `RebindOp::Zero`, an empty fold.
    #[test]
    fn build_boundary_fold_no_overlapping_source_month_is_empty() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_401)];
        let windows = vec![fixed_window(9, ymd(2026, 5, 1), ymd(2026, 5, 8), 50.0)];

        let fold = fold_via_index(&source, &windows);

        assert_eq!(fold, vec![]);
    }

    /// A `value_mw == 0.0` window overlapping a source month contributes
    /// nothing, keeping the fold empty for the all-zero horizon-end stub.
    #[test]
    fn build_boundary_fold_zero_value_window_is_empty() {
        let source = vec![anticipated_slot_at(9, 0, 20_260_401)];
        let windows = vec![fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 0.0)];

        let fold = fold_via_index(&source, &windows);

        assert_eq!(fold, vec![]);
    }

    /// Two windows overlapping the one source month accumulate into a single
    /// emitted term at that source position — a per-position accumulation.
    #[test]
    fn build_boundary_fold_accumulates_windows_at_one_source_position() {
        let source = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_401)];
        let k = 1;
        let value_a = 30.0;
        let value_b = 60.0;
        let windows = vec![
            fixed_window(9, ymd(2026, 4, 1), ymd(2026, 4, 8), value_a),
            fixed_window(9, ymd(2026, 4, 20), ymd(2026, 4, 27), value_b),
        ];

        let fold = fold_via_index(&source, &windows);

        assert_eq!(fold.len(), 1, "one source position -> one emitted term");
        assert_eq!(fold[0].0, k);
        let w = 7.0 * 24.0 / (30.0 * 24.0);
        let expected = w * value_a + w * value_b;
        assert!(
            (fold[0].1 - expected).abs() < expected * 1e-9,
            "factor {} != expected sum {expected}",
            fold[0].1
        );
    }

    /// Reordering the fixed windows (here across two plants at distinct source
    /// positions) never changes the emitted fold — the sole map use is a
    /// lookup, the accumulation a `source_pos`-indexed `Vec`.
    #[test]
    fn build_boundary_fold_is_order_invariant() {
        let source = vec![
            anticipated_slot_at(9, 0, 20_260_401),
            anticipated_slot_at(7, 0, 20_260_401),
        ];
        let w9 = fixed_window(9, ymd(2026, 4, 8), ymd(2026, 4, 15), 50.0);
        let w7 = fixed_window(7, ymd(2026, 4, 1), ymd(2026, 4, 8), 20.0);

        let fold_ab = fold_via_index(&source, &[w9.clone(), w7.clone()]);
        let fold_ba = fold_via_index(&source, &[w7, w9]);

        assert_eq!(fold_ab, fold_ba);
        assert_eq!(fold_ab.len(), 2);
        assert!(
            fold_ab[0].0 < fold_ab[1].0,
            "the fold is emitted ascending by source position"
        );
    }
}
