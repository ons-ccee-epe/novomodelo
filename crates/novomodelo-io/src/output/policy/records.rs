//! Input and output record types for value-function artifact serialization.
//!
//! Input types (`PolicyCutRecord`, `PolicyBasisRecord`, `BasisEntry`,
//! `StageStatesPayload`, `StageCutsPayload`) borrow from caller-owned buffers; owned output types
//! (`Owned*`, `*ReadResult`, `PolicyCheckpoint`) own their vectors. All use
//! generic names to maintain infrastructure crate genericity; conversion from
//! algorithm-specific types is the calling crate's responsibility. Field names
//! correspond to the tables in `schemas/policy.fbs`.

use std::num::NonZeroU32;

use chrono::{Datelike, NaiveDate};

use crate::output::SoftwareIdentity;

/// Current on-disk value-function artifact format version.
///
/// [`CheckpointManifest::format_version`] must equal this;
/// [`crate::read_policy_checkpoint`] rejects any other value — and absence —
/// with a named error before parsing any payload, so a pre-marker artifact is
/// cleanly rejected, never read positionally. Version 4 stores every piece's
/// `intercept` and `coefficients` in the writing study's scaled units, the
/// scale its pool's [`StageCutsPayload::cost_scale_factor`] names.
pub const FORMAT_VERSION: u32 = 4;

/// Sentinel [`EntitySlot`] date-field value — [`EntitySlot::reference_date`],
/// [`EntitySlot::interval_start`], and [`EntitySlot::interval_end`] all
/// default to it — for a slot whose family does not populate that field; also
/// the value a reader yields when a field is absent from a buffer older than
/// that field's own id (see `schemas/policy.fbs` for each field's introducing
/// id).
pub const ENTITY_SLOT_DATE_SENTINEL: i32 = i32::MIN;

/// One per-slot entity-identity record for a state-vector dimension.
///
/// `entity_type` is the raw discriminant byte of the `EntityType` enum in
/// `schemas/policy.fbs`; [`EntitySlot::family`] reads it as the typed
/// [`StateFamily`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntitySlot {
    /// Raw [`StateFamily`] discriminant byte; see [`EntitySlot::family`].
    pub entity_type: u8,
    /// Owning entity's id; `int32` because a sentinel id can be `-1`.
    pub entity_id: i32,
    /// Secondary index within the owning entity (per-type meaning is the caller's).
    pub subindex: u32,
    /// Whether the owning entity was operationally active at this slot's stage.
    pub was_active: bool,
    /// `HydroInflowLag`'s reference past stage's `start_date`, `YYYYMMDD`
    /// encoded (`year * 10000 + month * 100 + day`);
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for every other family.
    pub reference_date: i32,
    /// Half-open delivery/arrival interval's inclusive start, `YYYYMMDD`
    /// encoded: `HydroTransitBucket`'s `arrival_start` or
    /// `AnticipatedThermalState`'s `delivery_start`;
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for storage and inflow-lag.
    pub interval_start: i32,
    /// Half-open delivery/arrival interval's exclusive end, paired with
    /// [`interval_start`](Self::interval_start);
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for storage and inflow-lag.
    pub interval_end: i32,
}

impl EntitySlot {
    /// Builds a slot for `family` with every date field at
    /// [`ENTITY_SLOT_DATE_SENTINEL`] — the shared body of the four
    /// per-family constructors below.
    fn at_sentinel_dates(
        family: StateFamily,
        entity_id: i32,
        subindex: u32,
        was_active: bool,
    ) -> Self {
        Self {
            entity_type: family.code(),
            entity_id,
            subindex,
            was_active,
            reference_date: ENTITY_SLOT_DATE_SENTINEL,
            interval_start: ENTITY_SLOT_DATE_SENTINEL,
            interval_end: ENTITY_SLOT_DATE_SENTINEL,
        }
    }

    /// Builds a [`StateFamily::HydroStorage`] slot; `subindex` is always `0`.
    #[must_use]
    pub fn storage(entity_id: i32, was_active: bool) -> Self {
        Self::at_sentinel_dates(StateFamily::HydroStorage, entity_id, 0, was_active)
    }

    /// Builds a [`StateFamily::HydroInflowLag`] slot; `subindex` is the
    /// 1-based AR lag order.
    #[must_use]
    pub fn inflow_lag(entity_id: i32, lag_order: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::HydroInflowLag,
            entity_id,
            lag_order,
            was_active,
        )
    }

    /// Builds a [`StateFamily::HydroTransitBucket`] slot; `entity_id` is the
    /// downstream hydro, `subindex` the maturity lag.
    #[must_use]
    pub fn transit_bucket(downstream_entity_id: i32, maturity_lag: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::HydroTransitBucket,
            downstream_entity_id,
            maturity_lag,
            was_active,
        )
    }

    /// Builds a [`StateFamily::AnticipatedThermalState`] slot; `subindex` is
    /// the ring-buffer slot.
    #[must_use]
    pub fn anticipated(entity_id: i32, ring_slot: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::AnticipatedThermalState,
            entity_id,
            ring_slot,
            was_active,
        )
    }

    /// Returns `self` with `reference_date` replaced; every other field is
    /// unchanged.
    #[must_use]
    pub fn with_reference_date(self, reference_date: i32) -> Self {
        Self {
            reference_date,
            ..self
        }
    }

    /// Returns `self` with `interval_start` and `interval_end` replaced; every
    /// other field is unchanged.
    #[must_use]
    pub fn with_interval(self, start: i32, end: i32) -> Self {
        Self {
            interval_start: start,
            interval_end: end,
            ..self
        }
    }
}

/// State-vector dimension class of an [`EntitySlot`] — the typed Rust view of
/// the `EntityType` enum in `schemas/policy.fbs`. The wire representation stays
/// the raw [`EntitySlot::entity_type`] byte; this enum is the checked reading of
/// it, so the discriminants MUST match the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StateFamily {
    /// Reservoir storage volume; see [`EntitySlot::storage`].
    HydroStorage = 0,
    /// Hydro inflow AR lag; see [`EntitySlot::inflow_lag`].
    HydroInflowLag = 1,
    /// Anticipated thermal commitment; see [`EntitySlot::anticipated`].
    AnticipatedThermalState = 2,
    /// Water in-transit bucket; see [`EntitySlot::transit_bucket`].
    HydroTransitBucket = 3,
}

impl StateFamily {
    /// The raw `EntityType` discriminant byte for this family.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The family for a raw `EntityType` byte, or `None` for a discriminant no
    /// `schemas/policy.fbs` `EntityType` variant defines.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::HydroStorage),
            1 => Some(Self::HydroInflowLag),
            2 => Some(Self::AnticipatedThermalState),
            3 => Some(Self::HydroTransitBucket),
            _ => None,
        }
    }
}

impl EntitySlot {
    /// The typed [`StateFamily`] of this slot, or `None` when [`Self::entity_type`]
    /// is a byte no `EntityType` variant defines.
    #[must_use]
    pub fn family(&self) -> Option<StateFamily> {
        StateFamily::from_code(self.entity_type)
    }
}

/// One affine-piece record for value-function artifact serialization.
///
/// `'a` borrows the coefficient slice without copying (vectors can be large).
/// `intercept` and `coefficients` are in the writer's scaled units, the scale
/// [`StageCutsPayload::cost_scale_factor`] names.
#[derive(Debug, Clone)]
pub struct PolicyCutRecord<'a> {
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients; length equals the pool's `entity_manifest` length.
    ///
    /// Positional only: index `i` is the i-th state-vector dimension, whose
    /// identity is carried by slot `i` of the co-located [`EntitySlot`] manifest
    /// (`entity_manifest`); no labels are stored inline.
    pub coefficients: &'a [f64],
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
    /// Whether the writer stored a trial point for this piece when it wrote the
    /// piece's slot; kept when that point is later dropped, and read as written,
    /// never inferred from the states table.
    pub scored: bool,
    /// The writer's count of the piece's activity; `None` when the checkpoint
    /// carries no activity counters ([`ProducerBlock::activity_counters`]).
    pub active_count: Option<u64>,
    /// The writer's last iteration of the piece's activity; `None` when the
    /// checkpoint carries no activity counters.
    pub last_active_iteration: Option<u64>,
}

/// One scenario's solver basis at a node. Element type of
/// [`PolicyBasisRecord::entries`].
#[derive(Debug, Clone)]
pub struct BasisEntry<'a> {
    /// Global scenario index holding this basis.
    pub scenario: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: &'a [u8],
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: &'a [u8],
    /// Pool slot of each trailing affine-piece row of `row_status`: entry `i`
    /// is the slot of the `i`-th piece row.
    pub piece_row_slots: &'a [u32],
}

/// One node's solver bases for value-function artifact serialization.
#[derive(Debug, Clone)]
pub struct PolicyBasisRecord<'a> {
    /// Node position (0-based) — the storage-unit key naming `basis/<node>.bin`.
    pub stage_id: u32,
    /// Training iteration that produced these bases.
    pub iteration: u32,
    /// One entry per scenario holding a basis at the node, ascending by
    /// `scenario`.
    pub entries: &'a [BasisEntry<'a>],
}

/// Sentinel [`StageStatesPayload::node_id`]/[`StageStatesReadResult::node_id`]
/// value for a pool with no single owning node, or a buffer without the field.
pub const STAGE_STATES_NODE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::node_id`]/[`StageCutsReadResult::node_id`] value
/// for a pool with no single owning node (a shared pool, never a boundary
/// source) or a pre-`id:8` buffer (forward-compatible default).
pub const STAGE_CUTS_NODE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::graph_stage_id`]/[`StageCutsReadResult::graph_stage_id`]
/// value for an unresolved owning-stage key or a pre-`id:8` buffer
/// (forward-compatible default).
pub const STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::priced_state_date`]/[`StageCutsReadResult::priced_state_date`]
/// value for a not-yet-recorded priced instant or a pre-`id:11` buffer
/// (forward-compatible default). `i32::MIN` rather than `-1`, which is a
/// decodable value in the `YYYYMMDD` space this field encodes.
pub const STAGE_CUTS_PRICED_STATE_DATE_SENTINEL: i32 = i32::MIN;

/// Encodes `date` as `YYYYMMDD` (`year * 10000 + month * 100 + day`) — the wire
/// encoding [`EntitySlot`]'s and [`StageCutsPayload`]'s date fields document.
/// Total; [`decode_slot_date`] is its exact inverse.
#[must_use]
pub fn encode_slot_date(date: NaiveDate) -> i32 {
    date.year() * 10_000
        + i32::try_from(date.month()).unwrap_or(1) * 100
        + i32::try_from(date.day()).unwrap_or(1)
}

/// Decodes `value` from the `YYYYMMDD` encoding [`encode_slot_date`] produces —
/// its exact inverse. `None` for [`ENTITY_SLOT_DATE_SENTINEL`]/
/// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] (`i32::MIN`) and for any other
/// value that is not a real calendar date; `-1` is deliberately rejected here
/// too, since it is a distinct non-date sentinel elsewhere in this module
/// (e.g. [`STAGE_CUTS_NODE_ID_SENTINEL`]) that must never be mistaken for one.
#[must_use]
pub fn decode_slot_date(value: i32) -> Option<NaiveDate> {
    let year = value / 10_000;
    let month = (value / 100) % 100;
    let day = value % 100;
    let (Ok(month), Ok(day)) = (u32::try_from(month), u32::try_from(day)) else {
        return None;
    };
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Payload for writing one pool's stored trial points to a value-function
/// artifact, `states/<pool>.bin`.
///
/// The `data` slice contains the flat state vectors (row-major, each of length
/// `state_dimension`). The total number of stored states is `count`.
#[derive(Debug, Clone)]
pub struct StageStatesPayload<'a> {
    /// Pool id (0-based) — the same key as [`StageCutsPayload::stage_id`].
    pub stage_id: u32,
    /// Declared id of the pool's sole owning node, or
    /// [`STAGE_STATES_NODE_ID_SENTINEL`] for a shared pool.
    pub node_id: i32,
    /// Length of each state vector: the pool's `entity_manifest` length.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer: `count * state_dimension` f64 elements.
    pub data: &'a [f64],
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
    /// Pool slot of each stored row, strictly ascending; length equals `count`.
    pub slots: &'a [u32],
}

/// Per-pool affine-piece data payload for [`crate::write_policy_checkpoint`],
/// grouping the arguments of [`crate::serialize_stage_cuts`].
#[derive(Debug)]
pub struct StageCutsPayload<'a> {
    /// Pool id (0-based) — the storage-unit key naming this payload's file
    /// `cuts/<pool>.bin`. Equals the stage index on a chain.
    pub stage_id: u32,
    /// The writing study's global state-vector length; a pool that projects
    /// fewer dimensions holds shorter coefficient vectors, of its
    /// `entity_manifest`'s length.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Protected-prefix length: the leading slots the writing run started with.
    pub warm_start_count: u32,
    /// Slice of affine-piece records to serialize; length equals `populated_count`.
    pub cuts: &'a [PolicyCutRecord<'a>],
    /// Number of affine-piece records.
    pub populated_count: u32,
    /// Per-slot entity identity; length equals each piece's coefficient count
    /// when populated. An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
    /// The scale of every piece's `intercept` and `coefficients`: the objective
    /// cost-scale factor of the writing study, whose scaled units they are in.
    pub cost_scale_factor: f64,
    /// Owning node's policy-graph id, or [`STAGE_CUTS_NODE_ID_SENTINEL`] for a
    /// shared pool.
    pub node_id: i32,
    /// Graph-stage id of the node(s) owning this pool — the boundary-resolution
    /// key; [`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`] when unresolved.
    pub graph_stage_id: i32,
    /// Owning stage's `end_date`, encoded `year * 10000 + month * 100 + day`
    /// — the instant this pool's pieces price. [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`]
    /// when not recorded.
    pub priced_state_date: i32,
}

/// One node of the value-function artifact's graph manifest: its declared id,
/// the stage it sits at, and the pool whose payload carries its affine pieces.
///
/// `pool_id` **is** the node → pool map: a node references one pool, and a
/// reader resolves node `id`'s pieces as `pool_id`'s payload (leaf nodes sharing
/// a pool all name the same `pool_id`).
#[derive(Debug, Clone)]
pub struct ManifestNode {
    /// Declared node id.
    pub id: i32,
    /// Stage id this node sits at.
    pub stage_id: i32,
    /// Pool whose payload holds this node's affine pieces.
    pub pool_id: u32,
}

/// One directed edge of the graph manifest, with its transition probability.
#[derive(Debug, Clone)]
pub struct ManifestEdge {
    /// Source node id.
    pub source_id: i32,
    /// Target node id.
    pub target_id: i32,
    /// Transition probability `P(source -> target)`.
    pub probability: f64,
}

/// The graph manifest: the node list (each node carrying its own node → pool
/// assignment), the edge list, and the pool-set size.
///
/// This is the identity source the positional format never had — a reader
/// resolves a node's payload through it (node `n`'s pieces are `pool(n)`'s
/// payload), rather than trusting a filename.
#[derive(Debug, Clone, Default)]
pub struct GraphManifest {
    /// Number of distinct pools (the pool-set size).
    pub n_pools: u32,
    /// Every node, in canonical order, each with its stage and pool.
    pub nodes: Vec<ManifestNode>,
    /// Every directed edge with its transition probability.
    pub edges: Vec<ManifestEdge>,
}

/// [`SeasonManifest::cycle_code`] discriminant: a monthly season cycle.
pub const SEASON_CYCLE_CODE_MONTHLY: u8 = 0;
/// [`SeasonManifest::cycle_code`] discriminant: a weekly season cycle.
pub const SEASON_CYCLE_CODE_WEEKLY: u8 = 1;
/// [`SeasonManifest::cycle_code`] discriminant: a custom (neither monthly nor
/// weekly) season cycle.
pub const SEASON_CYCLE_CODE_CUSTOM: u8 = 2;
/// [`SeasonManifest::cycle_code`] discriminant: no season map declared — the
/// value [`SeasonManifest::default`] carries and a reader yields for a
/// pre-`id:19` buffer.
pub const SEASON_CYCLE_CODE_ABSENT: u8 = 255;

/// One hydro's per-season autoregressive order vector. Element type of
/// [`SeasonManifest::hydro_orders`].
#[derive(Debug, Clone)]
pub struct HydroSeasonOrders {
    /// Owning hydro's id.
    pub hydro_id: i32,
    /// Consecutive-lag AR order per dense season ordinal — a count, never a
    /// lag set (PAR order selection returns a scalar and the coefficient
    /// vector is dense). Length equals [`SeasonManifest::n_seasons`].
    pub orders: Vec<u32>,
}

/// Study-global season cycle and per-hydro PAR-order descriptor carried on
/// [`CheckpointManifest::season_manifest`]. The date-driven boundary gate
/// compares two studies' descriptors to reject a source whose season
/// definitions or per-season AR orders differ from the loading study's.
#[derive(Debug, Clone)]
pub struct SeasonManifest {
    /// Season cycle discriminant; one of the `SEASON_CYCLE_CODE_*`
    /// constants. A raw `u8`, never a `novomodelo-core` season type — the
    /// crate-genericity rule forbids the algorithm-specific dependency here.
    pub cycle_code: u8,
    /// Number of distinct seasons in the cycle; the length of every
    /// [`HydroSeasonOrders::orders`] vector.
    pub n_seasons: u32,
    /// Per-hydro AR order vectors, in canonical ascending `hydro_id` order.
    pub hydro_orders: Vec<HydroSeasonOrders>,
}

impl CheckpointManifest {
    /// The software this checkpoint records as its writer.
    #[must_use]
    pub fn written_by(&self) -> SoftwareIdentity<'_> {
        SoftwareIdentity {
            name: self.software.as_deref(),
            version: &self.software_version,
        }
    }
}

impl Default for SeasonManifest {
    /// The absent descriptor: [`SEASON_CYCLE_CODE_ABSENT`], zero seasons, no
    /// hydros — what a pre-`id:19` buffer decodes to.
    fn default() -> Self {
        Self {
            cycle_code: SEASON_CYCLE_CODE_ABSENT,
            n_seasons: 0,
            hydro_orders: Vec::new(),
        }
    }
}

/// Producer-namespaced metadata: everything specific to how the artifact was
/// produced (the training algorithm's own recorded state). Segregated from the
/// neutral core carried on [`CheckpointManifest`], whose doc states the
/// segregation rationale.
#[derive(Debug, Clone)]
pub struct ProducerBlock {
    /// Number of training iterations completed at write time.
    pub completed_iterations: u32,
    /// Lower bound value after the final completed iteration.
    pub final_lower_bound: f64,
    /// Last iteration's upper bound, if available (the final value, not a
    /// min-tracked best).
    pub best_upper_bound: Option<f64>,
    /// Maximum number of iterations configured for the run.
    pub max_iterations: u32,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Number of pieces loaded from a previous artifact at run start.
    pub warm_start_cuts: u32,
    /// Per-pool warm-start piece counts, in pool-id order.
    ///
    /// When non-empty, supersedes [`warm_start_cuts`] for per-pool accuracy.
    ///
    /// [`warm_start_cuts`]: Self::warm_start_cuts
    pub warm_start_counts: Vec<u32>,
    /// RNG seed used by the scenario sampler.
    ///
    /// Per-draw seeds are derived from `(rng_seed, iteration, scenario, stage)`,
    /// so resume needs only the seed — no accumulated RNG state is persisted.
    pub rng_seed: u64,
    /// Total visited states across all nodes.
    pub total_visited_states: u64,
    /// Block mode the artifact was trained under: the shared lowercase mode when
    /// every study stage agrees, else `"mixed"`.
    pub training_block_mode: String,
    /// Per-study-stage training block modes, in study-stage order.
    ///
    /// Populated only for mixed-mode studies.
    pub training_block_mode_per_stage: Vec<String>,
    /// Objective cost-scale factor the writing study resolved
    /// (`modeling.cost_scale_factor`): the scale of the writer's scaled units,
    /// which every piece's `intercept` and `coefficients` are in. Each pool
    /// carries its own copy, [`StageCutsPayload::cost_scale_factor`].
    ///
    /// Absent when unmarked.
    pub cost_scale_factor: Option<f64>,
    /// Lower bound after each recorded iteration, oldest first; the last entry
    /// is iteration [`completed_iterations`]. A writer records every completed
    /// iteration it has; an empty vector means none was recorded, and is what a
    /// buffer without the field reads as.
    ///
    /// [`completed_iterations`]: Self::completed_iterations
    pub lower_bound_history: Vec<f64>,
    /// Which trial points the pools that are neither the root's nor the
    /// terminal's store.
    pub state_history: StateHistory,
    /// Whether every piece of every pool carries both activity counters.
    pub activity_counters: bool,
    /// Whether `basis/` holds a basis for every scenario that held one, rather
    /// than one representative per node.
    pub basis_store_complete: bool,
}

/// [`StateHistory`] wire code: no pool stores trial points.
pub const STATE_HISTORY_SCOPE_NONE: u8 = 0;
/// [`StateHistory`] wire code: each pool stores the trial points of a window
/// of iterations.
pub const STATE_HISTORY_SCOPE_WINDOW: u8 = 1;
/// [`StateHistory`] wire code: each pool stores every trial point.
pub const STATE_HISTORY_SCOPE_WHOLE: u8 = 2;

/// Which trial points a checkpoint's pools store, stated over the pools that
/// are neither the root's nor the terminal's. On the wire it is the raw code
/// `state_history_scope` (one of the `STATE_HISTORY_SCOPE_*` constants) plus
/// `state_history_window`, read only under the window code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateHistory {
    /// No pool stores trial points.
    None,
    /// Each pool stores the trial points of the last `n` iterations.
    Window(NonZeroU32),
    /// Each pool stores every trial point.
    Whole,
}

impl StateHistory {
    /// The raw `state_history_scope` code.
    #[must_use]
    pub const fn scope_code(self) -> u8 {
        match self {
            Self::None => STATE_HISTORY_SCOPE_NONE,
            Self::Window(_) => STATE_HISTORY_SCOPE_WINDOW,
            Self::Whole => STATE_HISTORY_SCOPE_WHOLE,
        }
    }

    /// The raw `state_history_window` value: the window, `0` for every other
    /// scope.
    #[must_use]
    pub const fn window(self) -> u32 {
        match self {
            Self::Window(window) => window.get(),
            Self::None | Self::Whole => 0,
        }
    }

    /// The scope for a raw code and window, or `None` for an unknown code or a
    /// window code with a zero window.
    #[must_use]
    pub const fn from_codes(scope_code: u8, window: u32) -> Option<Self> {
        match scope_code {
            STATE_HISTORY_SCOPE_NONE => Some(Self::None),
            STATE_HISTORY_SCOPE_WHOLE => Some(Self::Whole),
            STATE_HISTORY_SCOPE_WINDOW => match NonZeroU32::new(window) {
                Some(window) => Some(Self::Window(window)),
                None => None,
            },
            _ => None,
        }
    }
}

/// Study-global checkpoint metadata carried on the `FlatBuffers`
/// `CheckpointManifest` root at `manifest.bin`: the neutral core
/// (`format_version`, `software`, `software_version`, `created_at`,
/// `num_stages`, the [`GraphManifest`] descriptors) plus the namespaced
/// [`ProducerBlock`]. Read
/// first by [`crate::read_policy_checkpoint`], whose version gate rejects a
/// stale `format_version` before any payload is parsed.
///
/// The neutral core describes the artifact itself; the algorithm's own recorded
/// state lives under the namespaced [`producer`] block, so a reader that does
/// not know the producer reads the core from the core's own vocabulary.
///
/// [`producer`]: Self::producer
#[derive(Debug, Clone)]
pub struct CheckpointManifest {
    /// On-disk format version; must equal [`FORMAT_VERSION`] on read.
    pub format_version: u32,
    /// Name of the software that wrote this checkpoint; `None` for a buffer
    /// older than the field.
    pub software: Option<String>,
    /// Version of the software that wrote this checkpoint.
    pub software_version: String,
    /// ISO 8601 timestamp when the checkpoint was written.
    pub created_at: String,
    /// Number of stages the graph manifest spans.
    pub num_stages: u32,
    /// Graph manifest: node list, edge list, node → pool map, and pool-set size.
    pub graph_manifest: GraphManifest,
    /// Producer-namespaced metadata (the training algorithm's own state).
    pub producer: ProducerBlock,
    /// Study-global season cycle and per-hydro PAR-order descriptor; absent
    /// ([`SEASON_CYCLE_CODE_ABSENT`]) for a pre-`id:19` buffer.
    pub season_manifest: SeasonManifest,
}

// ── Owned output types for deserialization ───────────────────────────────────

/// Owned version of [`PolicyCutRecord`] returned by [`crate::deserialize_stage_cuts`].
///
/// Unlike [`PolicyCutRecord<'a>`], this type owns its `coefficients` vector so it
/// can be returned from a deserialization function that does not borrow from the
/// input buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyCutRecord {
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients; positional per the [`PolicyCutRecord::coefficients`] contract.
    pub coefficients: Vec<f64>,
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
    /// Per the [`PolicyCutRecord::scored`] contract.
    pub scored: bool,
    /// Per the [`PolicyCutRecord::active_count`] contract.
    pub active_count: Option<u64>,
    /// Per the [`PolicyCutRecord::last_active_iteration`] contract.
    pub last_active_iteration: Option<u64>,
}

/// Owned version of [`BasisEntry`]. Element type of
/// [`OwnedPolicyBasisRecord::entries`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedBasisEntry {
    /// Global scenario index holding this basis.
    pub scenario: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: Vec<u8>,
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: Vec<u8>,
    /// Per the [`BasisEntry::piece_row_slots`] contract.
    pub piece_row_slots: Vec<u32>,
}

/// Owned version of [`PolicyBasisRecord`] returned by [`crate::deserialize_stage_basis`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyBasisRecord {
    /// Node position (0-based).
    pub stage_id: u32,
    /// Training iteration that produced these bases.
    pub iteration: u32,
    /// One entry per scenario holding a basis at the node, ascending by
    /// `scenario`.
    pub entries: Vec<OwnedBasisEntry>,
}

/// Stage-level metadata and affine-piece records returned by [`crate::deserialize_stage_cuts`].
///
/// Contains the stage-level fields stored in the `StageCuts` root table plus the
/// vector of deserialized affine-piece records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageCutsReadResult {
    /// Pool id (0-based), as written by the pool-keyed payload.
    pub stage_id: u32,
    /// Per the [`StageCutsPayload::state_dimension`] contract.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Per the [`StageCutsPayload::warm_start_count`] contract.
    pub warm_start_count: u32,
    /// Number of affine-piece records.
    pub populated_count: u32,
    /// Deserialized affine-piece records.
    pub cuts: Vec<OwnedPolicyCutRecord>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
    /// The scale of every piece's `intercept` and `coefficients`, per the
    /// [`StageCutsPayload::cost_scale_factor`] contract; `None` when absent.
    pub cost_scale_factor: Option<f64>,
    /// Owning node's policy-graph id; [`STAGE_CUTS_NODE_ID_SENTINEL`] for a shared
    /// pool or a pre-`id:8` buffer.
    pub node_id: i32,
    /// Graph-stage id key; [`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`] when unresolved
    /// or absent from a pre-`id:8` buffer.
    pub graph_stage_id: i32,
    /// Owning stage's `end_date`, encoded `year * 10000 + month * 100 + day`;
    /// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] when not recorded or absent
    /// from a pre-`id:11` buffer.
    pub priced_state_date: i32,
}

/// Owned version of [`StageStatesPayload`] returned by [`crate::deserialize_stage_states`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageStatesReadResult {
    /// Pool id (0-based).
    pub stage_id: u32,
    /// Per the [`StageStatesPayload::node_id`] contract.
    pub node_id: i32,
    /// Length of each state vector.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer (owned).
    pub data: Vec<f64>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
    /// Per the [`StageStatesPayload::slots`] contract.
    pub slots: Vec<u32>,
}

/// Complete deserialized value-function artifact returned by [`crate::read_policy_checkpoint`].
#[derive(Debug, Clone)]
pub struct PolicyCheckpoint {
    /// Checkpoint manifest read from `manifest.bin`.
    pub metadata: CheckpointManifest,
    /// Per-pool affine-piece collections, sorted by pool id.
    pub stage_cuts: Vec<StageCutsReadResult>,
    /// Per-node solver bases, sorted by node position (`stage_id`).
    pub stage_bases: Vec<OwnedPolicyBasisRecord>,
    /// Per-pool stored trial points, sorted by pool id (`stage_id`).
    ///
    /// Empty when no pool stores trial points.
    pub stage_states: Vec<StageStatesReadResult>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use chrono::NaiveDate;

    use std::num::NonZeroU32;

    use super::{
        ENTITY_SLOT_DATE_SENTINEL, EntitySlot, STATE_HISTORY_SCOPE_NONE, STATE_HISTORY_SCOPE_WHOLE,
        STATE_HISTORY_SCOPE_WINDOW, StateFamily, StateHistory, decode_slot_date, encode_slot_date,
    };

    #[test]
    fn state_history_codes_round_trip_and_reject_a_zero_window_or_unknown_code() {
        let window = NonZeroU32::new(6).unwrap();
        for (scope, code, raw_window) in [
            (StateHistory::None, STATE_HISTORY_SCOPE_NONE, 0),
            (StateHistory::Window(window), STATE_HISTORY_SCOPE_WINDOW, 6),
            (StateHistory::Whole, STATE_HISTORY_SCOPE_WHOLE, 0),
        ] {
            assert_eq!((scope.scope_code(), scope.window()), (code, raw_window));
            assert_eq!(StateHistory::from_codes(code, raw_window), Some(scope));
        }
        assert_eq!(
            StateHistory::from_codes(STATE_HISTORY_SCOPE_WINDOW, 0),
            None
        );
        assert_eq!(StateHistory::from_codes(3, 0), None);
    }

    #[test]
    fn state_family_codes_match_policy_fbs_entity_type() {
        assert_eq!(StateFamily::HydroStorage.code(), 0);
        assert_eq!(StateFamily::HydroInflowLag.code(), 1);
        assert_eq!(StateFamily::AnticipatedThermalState.code(), 2);
        assert_eq!(StateFamily::HydroTransitBucket.code(), 3);
    }

    #[test]
    fn state_family_from_code_round_trips_and_rejects_unknown() {
        for family in [
            StateFamily::HydroStorage,
            StateFamily::HydroInflowLag,
            StateFamily::AnticipatedThermalState,
            StateFamily::HydroTransitBucket,
        ] {
            assert_eq!(StateFamily::from_code(family.code()), Some(family));
        }
        assert_eq!(StateFamily::from_code(4), None);
        assert_eq!(StateFamily::from_code(u8::MAX), None);
    }

    #[test]
    fn entity_slot_family_reads_the_raw_byte() {
        let slot = EntitySlot::anticipated(7, 0, true);
        assert_eq!(slot.family(), Some(StateFamily::AnticipatedThermalState));
    }

    #[test]
    fn entity_slot_constructors_set_family_code_and_sentinel() {
        let storage = EntitySlot::storage(7, true);
        assert_eq!(storage.entity_type, StateFamily::HydroStorage.code());
        assert_eq!(storage.entity_id, 7);
        assert_eq!(storage.subindex, 0);
        assert!(storage.was_active);

        let inflow_lag = EntitySlot::inflow_lag(2, 3, false);
        assert_eq!(inflow_lag.entity_type, StateFamily::HydroInflowLag.code());
        assert_eq!(inflow_lag.entity_id, 2);
        assert_eq!(inflow_lag.subindex, 3);
        assert!(!inflow_lag.was_active);

        let transit_bucket = EntitySlot::transit_bucket(4, 5, true);
        assert_eq!(
            transit_bucket.entity_type,
            StateFamily::HydroTransitBucket.code()
        );
        assert_eq!(transit_bucket.entity_id, 4);
        assert_eq!(transit_bucket.subindex, 5);
        assert!(transit_bucket.was_active);

        let anticipated = EntitySlot::anticipated(6, 1, false);
        assert_eq!(
            anticipated.entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(anticipated.entity_id, 6);
        assert_eq!(anticipated.subindex, 1);
        assert!(!anticipated.was_active);
    }

    #[test]
    fn entity_slot_constructors_leave_new_dates_at_sentinel() {
        for slot in [
            EntitySlot::storage(1, true),
            EntitySlot::inflow_lag(1, 1, true),
            EntitySlot::transit_bucket(1, 1, true),
            EntitySlot::anticipated(1, 0, true),
        ] {
            assert_eq!(slot.reference_date, ENTITY_SLOT_DATE_SENTINEL);
            assert_eq!(slot.interval_start, ENTITY_SLOT_DATE_SENTINEL);
            assert_eq!(slot.interval_end, ENTITY_SLOT_DATE_SENTINEL);
        }
    }

    #[test]
    fn entity_slot_with_interval_sets_both_endpoints() {
        let dated = EntitySlot::transit_bucket(9, 2, true).with_interval(20_311_201, 20_320_101);
        assert_eq!(dated.interval_start, 20_311_201);
        assert_eq!(dated.interval_end, 20_320_101);
        assert_eq!(dated.reference_date, ENTITY_SLOT_DATE_SENTINEL);
    }

    #[test]
    fn slot_date_codec_round_trips_and_rejects_non_dates() {
        for (year, month, day) in [(2031, 11, 10), (2024, 2, 29), (2030, 1, 1)] {
            let date = NaiveDate::from_ymd_opt(year, month, day).unwrap();
            assert_eq!(decode_slot_date(encode_slot_date(date)), Some(date));
        }
        for value in [i32::MIN, -1, 0, 20_240_230] {
            assert_eq!(decode_slot_date(value), None);
        }
    }
}
