//! Per-thread solver workspace, workspace pool, and per-scenario basis store.
//!
//! [`SolverWorkspace`] bundles all mutable per-thread resources needed for one LP solve sequence.
//! [`WorkspacePool`] allocates one workspace per worker thread.
//! [`BasisStore`] provides per-scenario, per-stage basis storage for warm-starting LP solves.

use std::ops::Range;

use novomodelo_core::WorkerPhaseTimings;
use novomodelo_io::config::BackwardScheduler;
use novomodelo_solver::{Basis, BasisStatus, ProfiledSolver, SolverInterface, SolverStatistics};

use crate::SddpError;
use crate::SddpError::Validation;
use crate::backward::StagedCut;
use crate::context::{StageContext, TrainingContext};
use crate::cut::chain_rule::{AggregateScratch, ChainRuleScratch, ChildOpenings, OpeningDuals};
use crate::dcs::DcsSolveScratch;
use crate::indexer::CutStateProjection;
use crate::lp::builder::{PatchBuffer, StageGeometry};
use crate::risk_measure::RiskMeasureScratch;
use crate::setup::{NodeId, NodePos};
use crate::solve::partition;
use crate::solver_stats::SolverStatsDelta;

// ---------------------------------------------------------------------------
// CapturedBasis
// ---------------------------------------------------------------------------

/// A solver basis augmented with slot-tracking metadata for cut-set-aware
/// warm-start reconstruction.
///
/// `cut_row_slots` maps each captured cut row to the cut pool slot it occupied
/// at capture, so reconstruction binds stored statuses to current LP cut rows by
/// slot identity rather than row count. Pre-sized via [`CapturedBasis::new`] so
/// the forward capture site reuses one `CapturedBasis` across iterations without
/// reallocation.
///
/// `node_id` tags the node this basis was captured at; the apply sites
/// (`run_stage_solve`, `lazy_solve_preloaded`) treat a stored basis whose
/// `node_id` differs from the node being solved as cold, never warm-started
/// from it — a resampled path visiting a different node than the one that
/// captured the basis. Carried on the broadcast wire (see
/// [`CapturedBasis::to_broadcast_payload`]).
#[derive(Clone, Debug)]
pub struct CapturedBasis {
    /// Row and column statuses from the solver.
    pub basis: Basis,
    /// Number of template (non-cut) LP rows at capture time; row statuses at
    /// `0..base_row_count` are template rows.
    pub base_row_count: usize,
    /// Cut pool slot for each cut row in `basis.row_status[base_row_count..]`;
    /// length equals `basis.row_status.len() - base_row_count` when populated.
    pub cut_row_slots: Vec<u32>,
    /// Node this basis was captured at (`NodeGraph::node_ids[node]`).
    pub node_id: NodeId,
}

/// Wire-format version for `CapturedBasis` broadcast payloads.
///
/// Stored as the second `i32` in every `Some`-path payload, immediately after
/// the presence sentinel (`1_i32`). Bump this constant and update
/// `try_from_broadcast_payload` whenever the field layout changes.
///
/// `cut_row_slots` is the load-bearing reconstruction key —
/// `build_slot_lookup` binds stored cut-row statuses to target-LP cut rows by
/// slot id. The payload is a single `i32` buffer.
pub const BASIS_BROADCAST_FORMAT_TAG: i32 = 3;

/// Converts via [`BasisStatus::to_discriminant_code`], the single owner of the numeric
/// mapping (injective for round-trip). Payload byte layout owned by the
/// `to_broadcast_payload`/`try_from_broadcast_payload` pair.
fn basis_status_to_wire_code(status: BasisStatus) -> i32 {
    i32::from(status.to_discriminant_code())
}

/// Out-of-range codes decode to [`BasisStatus::Nonbasic`] via
/// [`BasisStatus::from_discriminant_code`].
fn basis_status_from_wire_code(code: i32) -> BasisStatus {
    u8::try_from(code).map_or(BasisStatus::Nonbasic, BasisStatus::from_discriminant_code)
}

impl CapturedBasis {
    /// Pre-sizes `cut_row_slots` but leaves it empty (length 0).
    #[must_use]
    pub fn new(
        num_cols: usize,
        num_rows: usize,
        base_row_count: usize,
        cut_slot_capacity: usize,
        node_id: NodeId,
    ) -> Self {
        Self {
            basis: Basis::new(num_cols, num_rows),
            base_row_count,
            cut_row_slots: Vec::with_capacity(cut_slot_capacity),
            node_id,
        }
    }

    /// Grows the buffers to `rows` row statuses and `cut_rows` cut-row slots, never shrinking and
    /// changing no length.
    pub(crate) fn reserve_cut_rows(&mut self, rows: usize, cut_rows: usize) {
        reserve_to(&mut self.basis.row_status, rows);
        reserve_to(&mut self.cut_row_slots, cut_rows);
    }

    /// Retains capacity. Does not touch `basis` (next `get_basis` overwrites it).
    pub fn clear(&mut self) {
        self.cut_row_slots.clear();
    }

    /// Append this basis's wire-format payload to the output buffer.
    ///
    /// Single owner of the wire format; any change must update both this method and
    /// [`CapturedBasis::try_from_broadcast_payload`] together.
    ///
    /// Pushes the following into `i32_buf` in order:
    /// - `1_i32` sentinel (present)
    /// - [`BASIS_BROADCAST_FORMAT_TAG`] as `i32` (format tag)
    /// - `node_id` as `i32` (the capture node)
    /// - `col_status.len()` as `i32`
    /// - `row_status.len()` as `i32`
    /// - `base_row_count` as `i32`
    /// - `cut_row_slots.len()` as `i32`
    /// - `col_status[..]` mapped through `basis_status_to_wire_code`
    /// - `row_status[..]` mapped through `basis_status_to_wire_code`
    /// - `cut_row_slots[..]` cast to `i32`
    ///
    /// Callers write the `0_i32` "no basis" sentinel when
    /// `Option<CapturedBasis>` is `None`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn to_broadcast_payload(&self, i32_buf: &mut Vec<i32>) {
        i32_buf.push(1_i32);
        i32_buf.push(BASIS_BROADCAST_FORMAT_TAG);
        i32_buf.push(self.node_id.0);
        i32_buf.push(self.basis.col_status.len() as i32);
        i32_buf.push(self.basis.row_status.len() as i32);
        i32_buf.push(self.base_row_count as i32);
        i32_buf.push(self.cut_row_slots.len() as i32);
        for &status in &self.basis.col_status {
            i32_buf.push(basis_status_to_wire_code(status));
        }
        for &status in &self.basis.row_status {
            i32_buf.push(basis_status_to_wire_code(status));
        }
        // u32 -> i32: slot values are LP pool indices (always
        // non-negative) that fit comfortably in i32.
        for &slot in &self.cut_row_slots {
            i32_buf.push(slot as i32);
        }
    }

    /// Deserialise one stage's payload from the wire-format buffer,
    /// advancing the cursor past the consumed bytes.
    ///
    /// Returns `Ok(None)` when the sentinel read is `0` (no basis
    /// for this stage). Returns `Ok(Some(captured))` when the
    /// sentinel is `1`, the format tag matches [`BASIS_BROADCAST_FORMAT_TAG`],
    /// and the payload is complete.
    ///
    /// # Layout (`Some` path)
    ///
    /// Reads from `i32_buf` in order:
    /// - `1_i32` sentinel (present)
    /// - [`BASIS_BROADCAST_FORMAT_TAG`] as `i32` (format tag)
    /// - `node_id` as `i32` (the generating/capture node, read straight into
    ///   the reconstructed [`CapturedBasis::node_id`])
    /// - `col_status.len()` as `i32`
    /// - `row_status.len()` as `i32`
    /// - `base_row_count` as `i32`
    /// - `cut_row_slots.len()` as `i32`
    /// - `col_status[..]` decoded through `basis_status_from_wire_code`
    /// - `row_status[..]` decoded through `basis_status_from_wire_code`
    /// - `cut_row_slots[..]` (stored as `i32`, cast back to `u32`)
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Validation` if the `i32_buf` is truncated at any of
    /// the bounded reads (sentinel, format tag, `node_id` + four length fields,
    /// `col_status`, `row_status`, `cut_row_slots`), or if the format-tag
    /// field does not match [`BASIS_BROADCAST_FORMAT_TAG`]. The error message
    /// names the affected stage and the expected vs. available byte count.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn try_from_broadcast_payload(
        stage: usize,
        i32_buf: &[i32],
        i32_cursor: &mut usize,
    ) -> Result<Option<Self>, SddpError> {
        if *i32_cursor >= i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated at stage {stage} \
                 (pos={}, len={})",
                *i32_cursor,
                i32_buf.len()
            )));
        }
        let sentinel = i32_buf[*i32_cursor];
        *i32_cursor += 1;

        if sentinel == 0 {
            return Ok(None);
        }

        // Format tag is present only on the Some path, after the sentinel.
        if *i32_cursor >= i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading version at stage {stage}"
            )));
        }
        let version = i32_buf[*i32_cursor];
        *i32_cursor += 1;
        if version != BASIS_BROADCAST_FORMAT_TAG {
            return Err(Validation(format!(
                "try_from_broadcast_payload: unsupported wire version {version} at stage \
                 {stage} (expected {BASIS_BROADCAST_FORMAT_TAG})"
            )));
        }

        if *i32_cursor + 5 > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading node_id + lengths at stage \
                 {stage}"
            )));
        }
        let node_id = NodeId(i32_buf[*i32_cursor]);
        *i32_cursor += 1;
        let col_len = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let row_len = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let base_row_count = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;
        let cut_slot_count = i32_buf[*i32_cursor] as usize;
        *i32_cursor += 1;

        if *i32_cursor + col_len > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading col_status at stage \
                 {stage} (need {col_len}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        let col_status: Vec<BasisStatus> = i32_buf[*i32_cursor..*i32_cursor + col_len]
            .iter()
            .map(|&code| basis_status_from_wire_code(code))
            .collect();
        *i32_cursor += col_len;

        if *i32_cursor + row_len > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading row_status at stage \
                 {stage} (need {row_len}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        let row_status: Vec<BasisStatus> = i32_buf[*i32_cursor..*i32_cursor + row_len]
            .iter()
            .map(|&code| basis_status_from_wire_code(code))
            .collect();
        *i32_cursor += row_len;

        if *i32_cursor + cut_slot_count > i32_buf.len() {
            return Err(Validation(format!(
                "try_from_broadcast_payload: buffer truncated reading cut_row_slots at stage \
                 {stage} (need {cut_slot_count}, have {})",
                i32_buf.len() - *i32_cursor
            )));
        }
        // cast_sign_loss: values were originally u32 LP pool indices cast to
        // i32 on the pack side; casting back to u32 is lossless.
        let cut_row_slots: Vec<u32> = i32_buf[*i32_cursor..*i32_cursor + cut_slot_count]
            .iter()
            .map(|&v| v as u32)
            .collect();
        *i32_cursor += cut_slot_count;

        Ok(Some(Self {
            basis: Basis {
                col_status,
                row_status,
            },
            base_row_count,
            cut_row_slots,
            node_id,
        }))
    }
}

/// Execution-count sizing hints shared by [`SolverWorkspace`], [`WorkspacePool`],
/// and `ScratchBuffers` constructors.
///
/// Grouping these dimensions into one struct keeps constructor argument counts
/// within the clippy budget. Every study-shape count these constructors need
/// (hydro count, PAR order, load buses, block counts, bucket/anticipated
/// counts, `n_state`, the downstream PAR order) is read from the
/// `training_ctx`/`stage_ctx` owners each constructor also takes; this struct
/// carries only counts with no other owner.
///
/// Any dimension may be `0`: the corresponding buffer starts empty and grows
/// on-demand via the growth-only resize semantics. Pass `0` for
/// `max_openings`/`initial_pool_capacity` on simulation-only workspaces, and
/// for `max_local_fwd` on backward-only or simulation-only workspaces.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspaceSizing {
    /// Maximum openings across all successor stages; pre-sizes
    /// `BackwardAccumulators::successor_aggregate`.
    pub max_openings: usize,
    /// Initial cut pool capacity; pre-sizes
    /// `BackwardAccumulators::slot_increments`.
    pub initial_pool_capacity: usize,
    /// Maximum forward-pass scenarios assigned to this rank; pre-sizes
    /// `ScratchBuffers::trajectory_costs_buf`.
    pub max_local_fwd: usize,
    /// Whether `ScratchBuffers::raw_noise_buf`/`corr_scratch` pre-reserve to
    /// the study's noise dimension, or start empty and grow on demand.
    pub noise: NoisePreallocation,
}

/// [`WorkspaceSizing::noise`]'s two preallocation strategies for
/// `ScratchBuffers::raw_noise_buf`/`corr_scratch`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoisePreallocation {
    /// Start empty; grow on first use.
    #[default]
    OnDemand,
    /// Pre-reserve to `TrainingContext::stochastic`'s noise dimension.
    StochasticDim,
}

/// Pre-allocated accumulators for the backward pass trial-point loop.
///
/// All buffers grow monotonically (never shrink) and are reused across stages
/// and trial points without per-call allocation. Each rayon worker owns an
/// exclusive [`SolverWorkspace`] and therefore an exclusive instance, so the
/// per-worker fields need no synchronisation; the sequential merge phase after
/// the parallel region sums their contributions.
///
/// `slot_increments` and `metadata_sync_contribution` are grown, zeroed and
/// filled only while the pass records binding activity (the successor view's
/// `cut_activity_tolerance` is `Some`). `cut_dual_slots` is filled on every pool
/// whose cut-state projection has lag slots, whatever the binding switch; only
/// the binding count reads the switch.
#[derive(Default)]
pub(crate) struct BackwardAccumulators {
    /// Per-slot binding count over the node's successor pool regions, concatenated
    /// per child (`SuccessorChild::metadata_offset + slot`), so a fan's distinct
    /// sibling pools never collide on a shared slot index. Filled only while
    /// [`tracks_binding_activity`](crate::config::CutManagementConfig::tracks_binding_activity)
    /// holds; zeroed per trial point by the by-scenario path, once per stage by
    /// the by-node path.
    pub(crate) slot_increments: Vec<u64>,
    /// Aggregated cut coefficients (`n_state` entries), written by
    /// `aggregate_weighted_into` then copied into [`agg_arena`](Self::agg_arena)
    /// at the trial point's slot.
    pub(crate) agg_coefficients: Vec<f64>,
    /// Per-worker flat arena holding every staged cut's coefficient vector for
    /// the current stage.
    ///
    /// Sized lazily to at least `(end_m - start_m) * n_state` `f64`s; slot `i` of
    /// the trial point with local index `local_idx = m - start_m` lives at
    /// `agg_arena[local_idx * n_state + i]`. Each [`StagedCut`] stores a
    /// `Range<usize>` into this arena instead of an owned `Vec<f64>`; the FCF
    /// merge reads the slice after the parallel region returns. Overwritten per
    /// trial point before any read, so no zero-fill is required.
    pub(crate) agg_arena: Vec<f64>,
    /// Per-worker metadata sync contribution over the node's successor pool
    /// regions, concatenated per child (same layout as `slot_increments`). Zeroed
    /// once per stage (not per trial point); each pool's own region slice is merged
    /// into `metadata_sync_buf` and synced separately.
    pub(crate) metadata_sync_contribution: Vec<u64>,
    /// Per-worker per-opening solver-statistics accumulator, at least
    /// `n_openings` long and read within `[..n_openings]`. Re-initialised once
    /// per stage by [`Self::reset_per_opening_stats`]; gathered into the entries
    /// `BackwardPassState::stage_stats` yields.
    pub(crate) per_opening_stats: Vec<SolverStatsDelta>,
    /// Per-block `simplex_iterations` sum for the by-node
    /// scheduler's pivot accumulator, index `b` in `[0, n_blocks)` for
    /// the current stage. Re-initialised once per stage (never per trial
    /// point); merged into `ByNodeScratch::block_pivots` after the
    /// parallel region.
    pub(crate) block_pivot_sum: Vec<u64>,
    /// Per-block solved-opening count, parallel to `block_pivot_sum`.
    pub(crate) block_pivot_count: Vec<u64>,
    /// Per-opening scratch for the incoming-state columns' reduced costs, reserved to the widest
    /// pool's slot count by [`WorkspacePool::reserve_backward_scratch`].
    pub(crate) state_duals_buf: Vec<f64>,
    /// Per-opening scratch of (pool slot, raw cut-row dual) pairs, zero-free and
    /// by ascending slot; reserved to the largest `structural_dual_bound` by
    /// [`WorkspacePool::reserve_backward_scratch`].
    pub(crate) cut_dual_slots: Vec<(usize, f64)>,
    /// Reusable `[hydro | load-bus | NCS]` opening-noise buffer for an External
    /// successor node, assembled from the external libraries at the node's
    /// column. Empty and never touched on the generated/sampled path.
    pub(crate) external_noise_buf: Vec<f64>,
    /// Per-worker staging buffer for the cuts one stage's fan-out produces; the
    /// by-scenario finish drains the workers' buffers in worker order.
    pub(crate) staged_cuts_buf: Vec<StagedCut>,
    /// The error that ended this worker's share of the stage's fan-out, taken in
    /// worker order by `take_first_error` before the finish reads the workers.
    pub(crate) fan_out_error: Option<SddpError>,
    /// Scratch for `CVaR` weight computation; the internal `Vec`s grow to
    /// `n_openings` on the first `CVaR` call. Never accessed under
    /// `RiskMeasure::Expectation`.
    pub(crate) risk_scratch: RiskMeasureScratch,
    /// Reusable scratch for the dynamic-cut-selection lazy solve, only touched
    /// when that method is active.
    pub(crate) dcs_solve: DcsSolveScratch,
    /// Metadata-seeded initial resident-cut set for the dynamic-cut-selection
    /// lazy solve; refilled per (trial point, opening).
    pub(crate) dcs_initial_resident: Vec<u32>,
    /// Per-opening before-solve statistics snapshot, filled via `statistics_into`
    /// (reusing the histogram allocation) and diffed against `stats_after_buf`.
    pub(crate) stats_before_buf: SolverStatistics,
    /// Per-opening after-solve statistics snapshot; counterpart to
    /// `stats_before_buf`.
    pub(crate) stats_after_buf: SolverStatistics,
    /// Per-opening solver-statistics delta, written from the two snapshots and
    /// accumulated into `per_opening_stats`.
    pub(crate) opening_delta_buf: SolverStatsDelta,
    /// Length of the backend's `retry_level_histogram`, learned from the
    /// worker's solver when the workspace is built.
    pub(crate) retry_histogram_len: usize,
    /// (Gᵀπ)_h per hydro for the current opening, in unscaled cost units.
    pub(crate) coupled_duals_buf: Vec<f64>,
    /// Per-opening chain-rule accumulators, shaped by the layout's (N, L).
    pub(crate) chain_scratch: ChainRuleScratch,
    /// Openings this worker solved, read by the aggregate-first cut.
    pub(crate) opening_records: OpeningRecords,
    /// Reusable state of one aggregate-first call.
    pub(crate) successor_aggregate: SuccessorAggregateScratch,
}

impl BackwardAccumulators {
    /// Allocate accumulators pre-sized from the given dimensions; any
    /// (`n_hydros` and `lag_depth` included) may be `0`, in which case those
    /// buffers start empty and grow lazily.
    pub(crate) fn new(
        max_openings: usize,
        initial_pool_capacity: usize,
        n_state: usize,
        columned_len: usize,
        n_hydros: usize,
        lag_depth: usize,
    ) -> Self {
        let mut dcs_solve = DcsSolveScratch::default();
        dcs_solve.reserve(columned_len, initial_pool_capacity);
        Self {
            slot_increments: vec![0u64; initial_pool_capacity],
            agg_coefficients: vec![0.0_f64; n_state],
            agg_arena: Vec::new(),
            metadata_sync_contribution: vec![0u64; initial_pool_capacity],
            per_opening_stats: vec![SolverStatsDelta::default(); max_openings],
            block_pivot_sum: Vec::new(),
            block_pivot_count: Vec::new(),
            state_duals_buf: Vec::new(),
            cut_dual_slots: Vec::new(),
            external_noise_buf: Vec::new(),
            staged_cuts_buf: Vec::new(),
            fan_out_error: None,
            risk_scratch: RiskMeasureScratch::new(),
            dcs_solve,
            dcs_initial_resident: Vec::with_capacity(initial_pool_capacity),
            stats_before_buf: SolverStatistics::default(),
            stats_after_buf: SolverStatistics::default(),
            opening_delta_buf: SolverStatsDelta::default(),
            retry_histogram_len: 0,
            coupled_duals_buf: vec![0.0; n_hydros],
            chain_scratch: ChainRuleScratch::new(n_hydros, lag_depth),
            opening_records: OpeningRecords::default(),
            successor_aggregate: SuccessorAggregateScratch::new(n_hydros, lag_depth, max_openings),
        }
    }

    /// Reserves `histogram_len` entries in every statistics histogram this
    /// accumulator owns, so no first use allocates whichever opening a worker
    /// claims first. Changes no length.
    pub(crate) fn reserve_statistics_histograms(&mut self, histogram_len: usize) {
        self.retry_histogram_len = histogram_len;
        let deltas = self
            .per_opening_stats
            .iter_mut()
            .chain([&mut self.opening_delta_buf]);
        for delta in deltas {
            delta.retry_level_histogram.reserve_exact(histogram_len);
        }
        for stats in [&mut self.stats_before_buf, &mut self.stats_after_buf] {
            stats.retry_level_histogram.reserve_exact(histogram_len);
        }
    }

    /// Zeroes the first `n_openings` per-opening accumulators in place. The
    /// vector only grows, so a stage with fewer openings leaves the later
    /// slots' histograms allocated for the stage that follows.
    pub(crate) fn reset_per_opening_stats(&mut self, n_openings: usize) {
        if self.per_opening_stats.len() < n_openings {
            self.per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
        }
        for slot in &mut self.per_opening_stats[..n_openings] {
            slot.reset_in_place();
        }
    }
}

/// Grows `buf` to `target` capacity, growth-only: `additional` is computed from
/// `len()`, which `Vec::reserve` counts from.
pub(crate) fn reserve_to<T>(buf: &mut Vec<T>, target: usize) {
    if buf.capacity() < target {
        buf.reserve(target - buf.len());
    }
}

/// Reuses the allocation of `v` for a vector of a same-layout type, so a buffer of
/// borrowing views keeps its capacity across calls that borrow for different lifetimes.
pub(crate) fn recycle<T, U>(mut v: Vec<T>) -> Vec<U> {
    v.clear();
    v.into_iter().map(|_| unreachable!("cleared")).collect()
}

/// Packed openings of one worker. A record holds its `(trial position, ω)` key,
/// Q, `state_len` lag-free state duals, N coupled duals and a range of the shared
/// zero-free `(slot, y)` pair buffer.
///
/// Reserved once, serially, through [`Self::reserve`] and never inside a claim, child or
/// unit loop: reserved capacity is virtual memory, and a worker commits only the pages
/// its own pushes write. A record with more pairs than the hint grows the pair buffer;
/// nothing is truncated.
#[derive(Default)]
pub(crate) struct OpeningRecords {
    state_len: usize,
    n_hydros: usize,
    keys: Vec<(usize, usize)>,
    objectives: Vec<f64>,
    state_duals: Vec<f64>,
    coupled_duals: Vec<f64>,
    pair_ends: Vec<usize>,
    pairs: Vec<(usize, f64)>,
}

impl OpeningRecords {
    /// Empties the store and fixes the per-record shape of the pushes that follow.
    pub(crate) fn clear(&mut self, state_len: usize, n_hydros: usize) {
        self.state_len = state_len;
        self.n_hydros = n_hydros;
        self.keys.clear();
        self.objectives.clear();
        self.state_duals.clear();
        self.coupled_duals.clear();
        self.pair_ends.clear();
        self.pairs.clear();
    }

    /// Grows every buffer to hold `records` records of the given shape and `pair_hint`
    /// pairs, never shrinking.
    pub(crate) fn reserve(
        &mut self,
        records: usize,
        state_len: usize,
        n_hydros: usize,
        pair_hint: usize,
    ) {
        reserve_to(&mut self.keys, records);
        reserve_to(&mut self.objectives, records);
        reserve_to(&mut self.pair_ends, records);
        reserve_to(&mut self.state_duals, records * state_len);
        reserve_to(&mut self.coupled_duals, records * n_hydros);
        reserve_to(&mut self.pairs, pair_hint);
    }

    pub(crate) fn push(
        &mut self,
        key: (usize, usize),
        objective: f64,
        state_duals: &[f64],
        coupled_duals: &[f64],
        pairs: &[(usize, f64)],
    ) {
        debug_assert_eq!(state_duals.len(), self.state_len);
        debug_assert_eq!(coupled_duals.len(), self.n_hydros);
        self.keys.push(key);
        self.objectives.push(objective);
        self.state_duals.extend_from_slice(state_duals);
        self.coupled_duals.extend_from_slice(coupled_duals);
        self.pairs.extend_from_slice(pairs);
        self.pair_ends.push(self.pairs.len());
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn key(&self, i: usize) -> (usize, usize) {
        self.keys[i]
    }

    #[cfg(test)]
    pub(crate) fn buffer_identity(&self) -> [(usize, usize); 6] {
        [
            (self.keys.as_ptr().addr(), self.keys.capacity()),
            (self.objectives.as_ptr().addr(), self.objectives.capacity()),
            (
                self.state_duals.as_ptr().addr(),
                self.state_duals.capacity(),
            ),
            (
                self.coupled_duals.as_ptr().addr(),
                self.coupled_duals.capacity(),
            ),
            (self.pair_ends.as_ptr().addr(), self.pair_ends.capacity()),
            (self.pairs.as_ptr().addr(), self.pairs.capacity()),
        ]
    }

    pub(crate) fn view(&self, i: usize) -> OpeningDuals<'_> {
        let pairs_start = if i == 0 { 0 } else { self.pair_ends[i - 1] };
        OpeningDuals {
            objective: self.objectives[i],
            state_duals: &self.state_duals[i * self.state_len..(i + 1) * self.state_len],
            coupled_duals: &self.coupled_duals[i * self.n_hydros..(i + 1) * self.n_hydros],
            cut_duals: &self.pairs[pairs_start..self.pair_ends[i]],
        }
    }
}

/// Reusable state of one aggregate-first cut: the kernel scratch, the flattened
/// opening costs, and the view vectors recycled across lifetimes so the call
/// allocates nothing after its first use.
pub(crate) struct SuccessorAggregateScratch {
    pub(crate) aggregate: AggregateScratch,
    pub(crate) costs: Vec<f64>,
    pub(crate) views: Vec<OpeningDuals<'static>>,
    pub(crate) children: Vec<ChildOpenings<'static>>,
    /// Record index of each canonical ω within one by-scenario unit.
    pub(crate) location: Vec<usize>,
}

impl SuccessorAggregateScratch {
    /// Scratch for a layout of `n_hydros` and `lag_depth`, holding up to `max_openings`
    /// openings per call.
    pub(crate) fn new(n_hydros: usize, lag_depth: usize, max_openings: usize) -> Self {
        Self {
            aggregate: AggregateScratch::new(n_hydros, lag_depth),
            costs: Vec::with_capacity(max_openings),
            views: Vec::with_capacity(max_openings),
            children: Vec::with_capacity(max_openings),
            location: Vec::with_capacity(max_openings),
        }
    }

    /// Grows the kernel scratch to `pool_capacity` pool slots and `pair_hint` pairs,
    /// never shrinking; the kernel itself does not grow it.
    pub(crate) fn reserve(&mut self, pool_capacity: usize, pair_hint: usize) {
        self.aggregate.reserve(pool_capacity, pair_hint);
    }
}

impl Default for SuccessorAggregateScratch {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

/// Per-record dimensions every pool of the study needs: the largest `columned_len` (the
/// lag-free state duals of a record) and the largest `structural_dual_bound` (the pair
/// count of a record, a reservation and not a capacity).
pub(crate) fn record_dims(layouts: &[CutStateProjection]) -> (usize, usize) {
    layouts.iter().fold((0, 0), |(state_len, bound), layout| {
        (
            state_len.max(layout.columned_len()),
            bound.max(layout.structural_dual_bound()),
        )
    })
}

// ---------------------------------------------------------------------------
// ByNodeScratch
// ---------------------------------------------------------------------------

/// What one chunk of `by_node_finish`'s position loop reuses across its positions.
pub(crate) struct ByNodeFinishScratch {
    /// Aggregate-first scratch of `aggregate_successor_cut`.
    pub(crate) aggregate: SuccessorAggregateScratch,
    /// `CVaR` weight-computation scratch.
    pub(crate) risk_scratch: RiskMeasureScratch,
    /// Aggregated cut coefficients; starts at `n_state` length, resized (within reserved
    /// capacity) to each stage's `cut_n_state`.
    pub(crate) coeffs_buf: Vec<f64>,
}

impl ByNodeFinishScratch {
    fn sized(n_state: usize, n_hydros: usize, lag_depth: usize, bwd_max_openings: usize) -> Self {
        Self {
            aggregate: SuccessorAggregateScratch::new(n_hydros, lag_depth, bwd_max_openings),
            risk_scratch: RiskMeasureScratch::new(),
            coeffs_buf: vec![0.0_f64; n_state],
        }
    }
}

/// Rank-level pre-allocated scratch for the by-node scheduler
/// (`training.parallelism.backward_scheduler`), held on `BackwardPassState`
/// and reused across every backward stage for the lifetime of a training run.
///
/// Empty (`location` capacity `0`) unless `ByNode` could be the
/// scheduler a run actually dispatches to: sized by
/// `BackwardPassState::resize_by_node_scratch` (called from `set_scheduler`)
/// whenever the configured scheduler is literally `ByNode`. Untouched by the
/// enumerated traversal — its per-node solve
/// (`BackwardPassState::run_enumerated_backward`) never dispatches through the
/// by-node scheduler at all. Never resized on the hot path (sddp.md "By-node
/// scheduler is warm-start-only").
#[derive(Default)]
pub(crate) struct ByNodeScratch {
    /// `(worker, record)` of each `(m, ω)`, addressed `location[m * n_openings + omega]`
    /// for the CURRENT stage's `n_openings`.
    pub(crate) location: Vec<(usize, usize)>,
    /// Outcome count each worker recorded in the stage, in worker order.
    pub(crate) worker_counts: Vec<usize>,
    /// Each worker's [`OpeningRecords`] as `by_node_finish` borrows them, recycled across
    /// lifetimes, so its parallel loop captures no `SolverWorkspace`.
    pub(crate) worker_records: Vec<&'static OpeningRecords>,
    /// One scratch per chunk of `by_node_finish`'s position loop, `n_workers.max(1)` entries.
    pub(crate) finish: Vec<ByNodeFinishScratch>,
    /// The stage's aggregated cuts, one `cut_n_state` row and one intercept per
    /// routed position, held until `by_node_finish` commits them: the aggregation
    /// reads the successor pools, which the commit mutates.
    pub(crate) cut_arena: Vec<f64>,
    /// Intercept of each staged cut, by routed position.
    pub(crate) cut_intercepts: Vec<f64>,
    /// Per-`(generating node, block-index)` `(simplex_iterations sum,
    /// solved-opening count)` pivot accumulator, addressed via
    /// [`Self::block_pivot_row`] — never re-derived at a call site. Keyed by
    /// the GENERATING node's canonical position, never the successor stage
    /// (`merge_block_pivots`'s own contract): sibling fan nodes at one level
    /// own distinct rows, so each computes its hardest-first order from its
    /// own history. Aggregated over every trial point `m` claimed for that
    /// block, never keyed per-`m` (the wrong-but-plausible variant, since
    /// resampled trial points make per-`m` hardness noise where the
    /// opening-block component is iteration-stable). Reset to `(0, 0)` at the
    /// start of every `BackwardPassState::run` call, never per stage.
    pub(crate) block_pivots: Vec<(u64, u64)>,
    /// Double-buffer companion to [`Self::block_pivots`]: holds the
    /// PREVIOUS iteration's fully-merged row, swapped in at the top of
    /// `BackwardPassState::run` (never per stage) so the hardest-first claim-order
    /// computation reads a row the current sweep has not yet started
    /// overwriting. Reading `block_pivots` directly during the sweep is the
    /// wrong-but-compiling alternative it rules out — that buffer is
    /// reset-then-partially-filled mid-run, yielding zeros or a half-filled
    /// row. Same shape as `block_pivots`.
    pub(crate) block_pivots_prev: Vec<(u64, u64)>,
    /// Row width of `block_pivots`/`block_pivots_prev` (block slots per
    /// generating node) — the run's max opening count, an upper bound on any
    /// node's actual block count (block size >= 1 implies
    /// `n_blocks <= n_openings <= n_blocks_max`).
    pub(crate) n_blocks_max: usize,
    /// Hardest-first claim-order scratch: the successor stage's sorted (or identity)
    /// `0..n_blocks` block-index permutation, recomputed into this buffer
    /// once per stage from [`Self::block_pivots_prev`] — never a fresh `Vec`
    /// per stage. Length `n_blocks_max`, an upper bound on any stage's
    /// `n_blocks` (see [`Self::n_blocks_max`]).
    pub(crate) block_order: Vec<u32>,
}

impl ByNodeScratch {
    /// Allocate the record-location table and aggregation scratch for `ByNode`:
    /// `location` is reserved for `max_local_fwd * bwd_max_openings` entries.
    /// `block_pivots` and
    /// `block_pivots_prev` each hold `row_axis_len * bwd_max_openings`
    /// accumulator cells, `row_axis_len` an upper bound on the GENERATING
    /// node count (see [`Self::block_pivots`] — never a stage count, though
    /// it coincides with one on a chain); `block_order` holds
    /// `bwd_max_openings` claim-order scratch slots; `worker_counts` holds one
    /// count per each of `n_workers` workers.
    #[must_use]
    pub(crate) fn sized(
        max_local_fwd: usize,
        bwd_max_openings: usize,
        n_state: usize,
        n_hydros: usize,
        lag_depth: usize,
        row_axis_len: usize,
        n_workers: usize,
    ) -> Self {
        let outcomes = max_local_fwd * bwd_max_openings;
        Self {
            location: Vec::with_capacity(outcomes),
            worker_counts: Vec::with_capacity(n_workers),
            worker_records: Vec::with_capacity(n_workers),
            finish: (0..n_workers.max(1))
                .map(|_| ByNodeFinishScratch::sized(n_state, n_hydros, lag_depth, bwd_max_openings))
                .collect(),
            cut_arena: vec![0.0_f64; max_local_fwd * n_state],
            cut_intercepts: vec![0.0_f64; max_local_fwd],
            block_pivots: vec![(0u64, 0u64); row_axis_len * bwd_max_openings],
            block_pivots_prev: vec![(0u64, 0u64); row_axis_len * bwd_max_openings],
            n_blocks_max: bwd_max_openings,
            block_order: vec![0u32; bwd_max_openings],
        }
    }

    /// Grows every finish entry's aggregate scratch to `pool_capacity` pool slots and
    /// `pair_hint` pairs, never shrinking.
    pub(crate) fn reserve_aggregate(&mut self, pool_capacity: usize, pair_hint: usize) {
        for finish in &mut self.finish {
            finish.aggregate.reserve(pool_capacity, pair_hint);
        }
    }

    /// `block_pivots`/`block_pivots_prev`'s row range for the GENERATING
    /// node `node_pos` — single owner of the `node_pos * n_blocks_max`
    /// row-start arithmetic (`NodePos` carries no arithmetic of its own), so
    /// a caller never recomputes it independently of this method.
    #[must_use]
    pub(crate) fn block_pivot_row(&self, node_pos: NodePos, n_blocks: usize) -> Range<usize> {
        let start = node_pos.0 * self.n_blocks_max;
        start..start + n_blocks
    }
}

/// Pre-allocated scratch buffers for noise transformation and simulation,
/// passed by `&mut` to the transformation functions in `noise.rs`.
///
/// `pub` type with every field kept `pub(crate)`: [`crate::lower_bound::evaluate_lower_bound`]'s
/// public signature must name the type, but an external caller must still go
/// through [`ScratchBuffers::new`] rather than construct or mutate one field-by-field.
#[allow(clippy::struct_field_names)]
pub struct ScratchBuffers {
    pub(crate) incoming_lags_buf: Vec<f64>,
    pub(crate) realized_inflow_buf: Vec<f64>,
    /// p, the PAR conditional mean at the pinned lags, per hydro.
    pub(crate) par_mean_buf: Vec<f64>,
    /// σ·η per hydro for the current opening.
    pub(crate) par_noise_term_buf: Vec<f64>,
    /// Pool slot of each cut row of the loaded LP, in LP row order.
    pub(crate) cut_slots_buf: Vec<usize>,
    /// The bracket B of each cut row, in LP row order.
    pub(crate) cut_bracket_buf: Vec<f64>,
    /// Pin phases that found no slab for the frozen prefix, counted over the workspace's lifetime.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) slab_misses: u64,
    /// The cut-row lower bound B + W, in LP row order.
    pub(crate) cut_rhs_buf: Vec<f64>,
    pub(crate) ncs_col_upper_buf: Vec<f64>,
    pub(crate) ncs_col_lower_buf: Vec<f64>,
    pub(crate) ncs_col_indices_buf: Vec<usize>,
    // Active-subset gather targets, parallel to `ncs_col_indices_buf`:
    // `transform_ncs_noise` writes `ncs_col_{lower,upper}_buf` in full slot order
    // (incl. slots inactive at this stage); the patch gathers only active slots'
    // bounds here so set-bounds gets equal-length buffers. Passing the slot-order
    // buffers verbatim would over-feed the call at a strict-subset stage.
    pub(crate) ncs_col_lower_active_buf: Vec<f64>,
    pub(crate) ncs_col_upper_active_buf: Vec<f64>,
    /// Per-stage NCS column start `ncs_col_indices_buf` was last built for;
    /// `usize::MAX` initially so the first patch always rebuilds.
    ///
    /// The lazy rebuild must key on the **start**, not the length: two stages can
    /// share the same `active_count * n_blks` length yet address different columns
    /// (mid-horizon commissioning / differing block counts), so keying on length
    /// would write `set_col_bounds` onto the wrong LP columns. Comparing
    /// `ncs_col_indices_buf[0]` is also insufficient — a simultaneous change of
    /// start and first-active-local can collide on the first index while the rest
    /// of the buffer is stale.
    pub(crate) last_ncs_col_start: usize,
    // NCS upper bounds for simulation extraction, length `n_ncs * n_blks` in
    // active-local column order (NOT slot order): template `col_upper` overwritten
    // per stochastic active column's block with the realized availability from
    // `ncs_col_upper_buf`. `ncs_col_upper_buf` (slot order) cannot be consumed
    // directly because extraction indexes by active-local column.
    pub(crate) ncs_col_upper_extract_buf: Vec<f64>,
    /// Per-hydro inflow-slack bounds in physical m³/s, written by
    /// `realize_inflow` under the truncation methods; the patch divides
    /// them by `col_scale` into the `_scaled` buffers.
    pub(crate) inflow_slack_lower_buf: Vec<f64>,
    pub(crate) inflow_slack_upper_buf: Vec<f64>,
    pub(crate) inflow_slack_col_indices_buf: Vec<usize>,
    pub(crate) inflow_slack_col_lower_scaled_buf: Vec<f64>,
    pub(crate) inflow_slack_col_upper_scaled_buf: Vec<f64>,
    /// Inflow-slack column start `inflow_slack_col_indices_buf` was last built
    /// for; `usize::MAX` initially. Keyed on the start, never the length, which
    /// is always `hydro_count`.
    pub(crate) last_inflow_slack_col_start: usize,
    pub(crate) load_rhs_buf: Vec<f64>,
    pub(crate) row_lower_buf: Vec<f64>,
    pub(crate) unscaled_primal: Vec<f64>,
    pub(crate) unscaled_dual: Vec<f64>,
    pub(crate) lag_accumulator: Vec<f64>,
    pub(crate) lag_weight_accum: Vec<f64>,
    pub(crate) downstream_accumulator: Vec<f64>,
    pub(crate) downstream_weight_accum: f64,
    // Slot-major `[slot * hydro_count + hydro]`; slot 0 = oldest completed
    // quarter, slot n-1 = most recent.
    pub(crate) downstream_completed_lags: Vec<f64>,
    pub(crate) downstream_n_completed: usize,
    /// Maps each cut pool slot to its position in the stored
    /// `CapturedBasis::cut_row_slots`, so reconstruction locates any active cut's
    /// row status in O(1). Pre-filled with `None` to `initial_pool_capacity`.
    pub(crate) recon_slot_lookup: Vec<Option<u32>>,

    /// Per-worker forward-pass trajectory-cost accumulator, taken via
    /// `std::mem::take` at the worker boundary. Named `..._buf` (not
    /// `trajectory_costs`) to avoid collision with the identically-named field on
    /// `ForwardWorkerResult`.
    pub(crate) trajectory_costs_buf: Vec<f64>,

    /// Per-worker raw-noise scratch for the forward-pass sampler and simulation
    /// worker loop.
    pub(crate) raw_noise_buf: Vec<f64>,

    /// Per-worker gather and correlate scratch for a correlation group wider
    /// than the applier's stack bound, sized `2 * noise_dim`: a group's width
    /// never exceeds its class dimension, hence never `noise_dim`, and the
    /// applier needs one gather array and one correlate array of that width.
    pub(crate) corr_scratch: Vec<f64>,

    /// Per-trajectory sampled-walk node carrier: `current_node_buf[local_m]` is
    /// trajectory `local_m`'s [`crate::setup::node_graph::NodeGraph`] position at
    /// the stage the forward loop is about to visit, root-initialized then
    /// advanced by the transition draw. Resized (never reallocated once at
    /// capacity) at each `run_forward_worker` call.
    pub(crate) current_node_buf: Vec<NodePos>,
}

/// All per-thread mutable resources required for one LP solve sequence.
///
/// Each instance is exclusively owned by one worker thread — no shared state
/// between workspaces — and is distributed by mutable reference from a
/// [`WorkspacePool`]. `rank` and `worker_id` are assigned once at
/// [`WorkspacePool::new`] and never change.
pub struct SolverWorkspace<S: SolverInterface> {
    /// MPI rank that owns this workspace. Stable across the run.
    pub rank: i32,
    /// Rayon worker index within this rank's pool, range `0..n_workers_local`.
    /// Stable across the run.
    pub worker_id: i32,
    /// LP solver, wrapped in [`ProfiledSolver`] so per-phase configuration
    /// applies via [`ProfiledSolver::set_profile`] without touching call sites
    /// that use the [`SolverInterface`] trait.
    pub solver: ProfiledSolver<S>,
    /// Pre-allocated row-bound patch buffer.
    pub patch_buf: PatchBuffer,
    /// Scratch buffer for the current state vector.
    pub current_state: Vec<f64>,
    /// Pre-allocated scratch buffers for noise transformation and simulation.
    pub(crate) scratch: ScratchBuffers,
    /// Pre-allocated destination basis for [`reconstruct_basis`], filled in-place
    /// from the read-only [`CapturedBasis`] in [`BasisStore`] then passed to
    /// `solve(Some(&basis))`. Sized via [`WorkspacePool::resize_scratch_bases`]
    /// so reconstruction never reallocates on the hot path.
    ///
    /// [`reconstruct_basis`]: crate::basis_reconstruct::reconstruct_basis
    pub(crate) scratch_basis: Basis,
    /// Pre-allocated accumulators for the backward pass trial-point loop.
    /// Empty on simulation-only workspaces (`max_openings = 0`), which the
    /// backward pass never touches.
    pub(crate) backward_accum: BackwardAccumulators,

    /// Zero-allocation timing payload for [`novomodelo_core::TrainingEvent::WorkerTiming`],
    /// accumulated in the parallel region and moved into the event payload after
    /// it completes. Reset to default at each iteration boundary.
    ///
    /// Field-to-slot mapping for the writer record:
    /// `forward_wall_ms` → `WORKER_TIMING_SLOT_FWD_WALL`,
    /// `backward_wall_ms` → `WORKER_TIMING_SLOT_BWD_WALL`,
    /// `bwd_setup_ms` → `WORKER_TIMING_SLOT_BWD_SETUP`,
    /// `fwd_setup_ms` → `WORKER_TIMING_SLOT_FWD_SETUP`.
    pub worker_timing_buf: WorkerPhaseTimings,
}

impl<S: SolverInterface> SolverWorkspace<S> {
    /// Construct a workspace with explicit identity, solver, and patch buffer.
    ///
    /// `scratch_basis` starts empty; call `WorkspacePool::resize_scratch_bases`
    /// after construction to pre-allocate it for in-place basis reconstruction.
    #[must_use]
    pub fn new(
        rank: i32,
        worker_id: i32,
        solver: S,
        patch_buf: PatchBuffer,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> Self {
        let mut backward_accum = BackwardAccumulators::new(
            sizing.max_openings,
            sizing.initial_pool_capacity,
            training_ctx.state.n_state,
            record_dims(training_ctx.cut_state_layouts).0,
            training_ctx.state.hydro_count,
            training_ctx.state.max_par_order,
        );
        backward_accum
            .reserve_statistics_histograms(solver.statistics().retry_level_histogram.len());
        Self {
            rank,
            worker_id,
            solver: ProfiledSolver::new(solver),
            patch_buf,
            current_state: Vec::with_capacity(training_ctx.state.n_state),
            scratch: ScratchBuffers::new(training_ctx, stage_ctx, sizing),
            scratch_basis: Basis::new(0, 0),
            backward_accum,
            worker_timing_buf: WorkerPhaseTimings::default(),
        }
    }
}

impl ScratchBuffers {
    /// Allocate scratch buffers sized from the study owners `training_ctx`/
    /// `stage_ctx` carry, plus the execution-count hints in `sizing`.
    #[must_use]
    pub fn new(
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> Self {
        let hydro_count = training_ctx.state.hydro_count;
        let max_par_order = training_ctx.state.max_par_order;
        let downstream_par_order = training_ctx.study_dims.downstream_par_order;
        let load_row_capacity = stage_ctx.load_bus_indices.len()
            * StageGeometry::max_blocks(stage_ctx.geometry_per_stage);
        let WorkspaceSizing {
            initial_pool_capacity,
            max_local_fwd,
            noise,
            // `max_openings` sizes `BackwardAccumulators`, not `ScratchBuffers`.
            ..
        } = sizing;
        let noise_dim = match noise {
            NoisePreallocation::StochasticDim => training_ctx.stochastic.dim(),
            NoisePreallocation::OnDemand => 0,
        };
        Self {
            incoming_lags_buf: Vec::with_capacity(max_par_order * hydro_count),
            realized_inflow_buf: Vec::with_capacity(hydro_count),
            par_mean_buf: vec![0.0; hydro_count],
            par_noise_term_buf: vec![0.0; hydro_count],
            cut_slots_buf: Vec::with_capacity(initial_pool_capacity),
            cut_bracket_buf: Vec::with_capacity(initial_pool_capacity),
            #[cfg(any(test, feature = "test-support"))]
            slab_misses: 0,
            cut_rhs_buf: Vec::with_capacity(initial_pool_capacity),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            inflow_slack_lower_buf: Vec::with_capacity(hydro_count),
            inflow_slack_upper_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_indices_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_lower_scaled_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_upper_scaled_buf: Vec::with_capacity(hydro_count),
            last_inflow_slack_col_start: usize::MAX,
            load_rhs_buf: Vec::with_capacity(load_row_capacity),
            row_lower_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![0.0_f64; hydro_count],
            lag_weight_accum: vec![0.0_f64; hydro_count],
            downstream_accumulator: if downstream_par_order > 0 {
                vec![0.0_f64; hydro_count]
            } else {
                Vec::new()
            },
            downstream_weight_accum: 0.0,
            downstream_completed_lags: vec![0.0_f64; hydro_count * downstream_par_order],
            downstream_n_completed: 0,
            recon_slot_lookup: vec![None; initial_pool_capacity],
            trajectory_costs_buf: Vec::with_capacity(max_local_fwd),
            raw_noise_buf: Vec::with_capacity(noise_dim),
            corr_scratch: Vec::with_capacity(2 * noise_dim),
            current_node_buf: Vec::with_capacity(max_local_fwd),
        }
    }

    /// Grows the three cut-row buffers to `capacity`, never shrinking.
    pub(crate) fn reserve_cut_rows(&mut self, capacity: usize) {
        reserve_to(&mut self.cut_slots_buf, capacity);
        reserve_to(&mut self.cut_bracket_buf, capacity);
        reserve_to(&mut self.cut_rhs_buf, capacity);
    }
}

/// A pool of [`SolverWorkspace`] instances, one per worker thread.
///
/// Create once at algorithm startup via [`WorkspacePool::new`] and distribute
/// workspaces to threads by indexing into [`workspaces`](WorkspacePool::workspaces).
///
/// The pool size equals the number of worker threads. Each workspace is
/// independently allocated and does not share any mutable state with the others.
pub struct WorkspacePool<S: SolverInterface> {
    /// The individual workspaces, indexed by thread number.
    pub workspaces: Vec<SolverWorkspace<S>>,
}

impl<S: SolverInterface> WorkspacePool<S> {
    /// Build one workspace: size its [`PatchBuffer`] from `training_ctx`/`stage_ctx`
    /// and construct the [`SolverWorkspace`]. Shared tail of [`WorkspacePool::new`]
    /// and [`WorkspacePool::try_new`], which differ only in how they obtain
    /// `worker_id` and `solver`.
    fn build_workspace(
        rank: i32,
        worker_id: i32,
        solver: S,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
    ) -> SolverWorkspace<S> {
        let patch_buf = PatchBuffer::new(
            training_ctx.state,
            stage_ctx.load_bus_indices,
            stage_ctx.geometry_per_stage,
            stage_ctx.inflow_coupling,
        );
        SolverWorkspace::new(
            rank,
            worker_id,
            solver,
            patch_buf,
            training_ctx,
            stage_ctx,
            sizing,
        )
    }

    /// Construct a pool of `n_threads` independently allocated workspaces, each
    /// with a sequentially assigned `worker_id` in `0..n_threads` and a fresh
    /// solver from `solver_factory` (called once per thread).
    ///
    /// # Panics
    ///
    /// Panics if `n_threads > i32::MAX` — physically impossible since rayon pools
    /// are bounded by CPU count.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new(
        rank: i32,
        n_threads: usize,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> S,
    ) -> Self {
        let workspaces = (0..n_threads)
            .map(|idx| {
                let worker_id =
                    i32::try_from(idx).expect("worker_id fits in i32 (rayon pools are small)");
                let solver = solver_factory();
                Self::build_workspace(rank, worker_id, solver, training_ctx, stage_ctx, sizing)
            })
            .collect();
        Self { workspaces }
    }

    /// Like [`WorkspacePool::new`] but with a fallible `solver_factory`; the
    /// first factory error is returned immediately and no partial pool is built.
    ///
    /// # Errors
    ///
    /// Returns `Err(E)` if any call to `solver_factory` fails.
    ///
    /// # Panics
    ///
    /// Panics if `n_threads > i32::MAX` — physically impossible since rayon pools
    /// are bounded by CPU count.
    #[allow(clippy::expect_used)]
    pub fn try_new<E>(
        rank: i32,
        n_threads: usize,
        training_ctx: &TrainingContext<'_>,
        stage_ctx: &StageContext<'_>,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> Result<S, E>,
    ) -> Result<Self, E> {
        let mut workspaces = Vec::with_capacity(n_threads);
        for idx in 0..n_threads {
            let worker_id =
                i32::try_from(idx).expect("worker_id fits in i32 (rayon pools are small)");
            let solver = solver_factory()?;
            workspaces.push(Self::build_workspace(
                rank,
                worker_id,
                solver,
                training_ctx,
                stage_ctx,
                sizing,
            ));
        }
        Ok(Self { workspaces })
    }

    /// Pre-allocate each workspace's `scratch_basis` to the given LP dimensions.
    ///
    /// The allocation happens once during setup; basis reconstruction on the
    /// hot path then reuses the existing capacity without reallocating.
    pub(crate) fn resize_scratch_bases(&mut self, max_cols: usize, max_rows: usize) {
        for ws in &mut self.workspaces {
            ws.scratch_basis = Basis::new(max_cols, max_rows);
        }
    }

    /// Re-reserve every workspace's DCS solve scratch ([`DcsSolveScratch`]),
    /// its three cut-row buffers (`cut_slots_buf`, `cut_bracket_buf`,
    /// `cut_rhs_buf`) and its aggregate-first scratch to `pool_capacity`
    /// (`pair_hint` pairs for the latter), growth-only — a no-op once every
    /// workspace already covers it.
    ///
    /// `pool_capacity` must be the LARGEST capacity across every pool a
    /// workspace's sweep may touch, never a single pool's own value: the
    /// scratch is shared per WORKER, not per pool, so sizing it from one
    /// pool under-covers every other pool that worker also solves against.
    pub(crate) fn reserve_cut_scratch(
        &mut self,
        columned_len: usize,
        pool_capacity: usize,
        pair_hint: usize,
    ) {
        for ws in &mut self.workspaces {
            ws.backward_accum
                .dcs_solve
                .reserve(columned_len, pool_capacity);
            ws.backward_accum
                .successor_aggregate
                .reserve(pool_capacity, pair_hint);
            ws.scratch.reserve_cut_rows(pool_capacity);
        }
    }

    /// Reserves every workspace's [`OpeningRecords`] for `records` records of
    /// `state_len` state duals and `pair_hint_per_record` pairs each,
    /// growth-only. Serial, once at construction: the claim loops only push.
    pub(crate) fn reserve_opening_records(
        &mut self,
        records: usize,
        state_len: usize,
        pair_hint_per_record: usize,
    ) {
        for ws in &mut self.workspaces {
            let n_hydros = ws.backward_accum.coupled_duals_buf.len();
            ws.backward_accum.opening_records.reserve(
                records,
                state_len,
                n_hydros,
                records * pair_hint_per_record,
            );
        }
    }

    /// Reserves the scratch whose size follows `pool_capacity`: the cut scratch of
    /// [`Self::reserve_cut_scratch`] with the study's pair bound. Growth-only, so it
    /// serves the construction and every point where the pools have grown;
    /// [`Self::reserve_backward_scratch`] fixes everything else at construction.
    pub(crate) fn reserve_pool_scratch(
        &mut self,
        training_ctx: &TrainingContext<'_>,
        sizing: WorkspaceSizing,
        pool_capacity: usize,
    ) {
        let (columned_len, bound) = record_dims(training_ctx.cut_state_layouts);
        self.reserve_cut_scratch(columned_len, pool_capacity, sizing.max_openings * bound);
    }

    /// Reserves what a backward sweep of `sizing`'s shape pushes into, whatever the pools'
    /// capacity: the opening records of every trial point `scheduler` keeps a worker
    /// holding, and each worker's per-opening dual buffers. Once, at construction.
    pub(crate) fn reserve_backward_scratch(
        &mut self,
        training_ctx: &TrainingContext<'_>,
        sizing: WorkspaceSizing,
        scheduler: &BackwardScheduler,
    ) {
        let (state_len, bound) = record_dims(training_ctx.cut_state_layouts);
        let max_slots = training_ctx
            .cut_state_layouts
            .iter()
            .map(CutStateProjection::n_slots)
            .max()
            .unwrap_or(0);
        let trial_points_held = match scheduler {
            BackwardScheduler::ByScenario {} => sizing.max_local_fwd.min(1),
            BackwardScheduler::ByNode { .. } => sizing.max_local_fwd,
        };
        self.reserve_opening_records(trial_points_held * sizing.max_openings, state_len, bound);
        for ws in &mut self.workspaces {
            reserve_to(&mut ws.backward_accum.state_duals_buf, max_slots);
            reserve_to(&mut ws.backward_accum.cut_dual_slots, bound);
        }
    }
}

// ---------------------------------------------------------------------------
// BasisStore
// ---------------------------------------------------------------------------

/// Per-scenario, per-node basis storage for warm-starting LP solves.
///
/// The store is indexed as `store[scenario_index][node_position]`. The forward
/// pass writes each visit at its stage position and the backward pass
/// warm-starts a cut's successor at that successor's canonical node position;
/// absent `nodes[]` node position == stage, so the two coincide. During the
/// forward pass, each worker writes to its disjoint scenario range; during the
/// backward pass, all workers read from any scenario's basis.
///
/// Internally, data is stored flat as
/// `bases[scenario * num_nodes + node]` for cache-friendly sequential
/// access within a single scenario's node loop.
///
/// # Cut selection interaction
///
/// Row statuses are positional (`row_status[i]` is LP row `i`). When cut
/// selection changes the active cut set, the cut-row count changes and the
/// stored cut-row statuses become stale; the degraded warm-start is accepted —
/// `HiGHS` detects the dimension mismatch on `solve(Some(&basis))`, crash-starts,
/// and records a `basis_rejection` in [`SolverStatistics`]. Template (non-cut)
/// row statuses remain valid.
///
/// [`SolverStatistics`]: novomodelo_solver::SolverStatistics
pub struct BasisStore {
    /// Flat storage: `bases[scenario * num_nodes + node]`.
    bases: Vec<Option<CapturedBasis>>,
    /// Node-axis length (canonical node count; equals `num_stages` on the chain).
    num_nodes: usize,
}

impl BasisStore {
    /// Allocate a new store for `num_scenarios` scenarios and `num_nodes`
    /// node positions, with every slot initialised to `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use novomodelo_sddp::setup::NodePos;
    /// use novomodelo_sddp::workspace::BasisStore;
    ///
    /// let store = BasisStore::new(4, 10);
    /// assert_eq!(store.num_scenarios(), 4);
    /// assert_eq!(store.num_nodes(), 10);
    /// assert!(store.get(0, NodePos(0)).is_none());
    /// ```
    #[must_use]
    pub fn new(num_scenarios: usize, num_nodes: usize) -> Self {
        let len = num_scenarios * num_nodes;
        Self {
            bases: vec![None; len],
            num_nodes,
        }
    }

    /// Return the number of scenarios this store was allocated for.
    #[must_use]
    pub fn num_scenarios(&self) -> usize {
        self.bases.len().checked_div(self.num_nodes).unwrap_or(0)
    }

    /// Return the node-axis length.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get an immutable reference to the basis at `[scenario][node]`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    #[must_use]
    pub fn get(&self, scenario: usize, node: NodePos) -> Option<&CapturedBasis> {
        self.bases[scenario * self.num_nodes + node.0].as_ref()
    }

    /// Get a mutable reference to the basis slot at `[scenario][node]`.
    pub fn get_mut(&mut self, scenario: usize, node: NodePos) -> &mut Option<CapturedBasis> {
        &mut self.bases[scenario * self.num_nodes + node.0]
    }

    /// Grows every stored basis to its node's `(rows, cut_rows)` bound, never shrinking and
    /// changing no length. Call it wherever the resident pools grow.
    pub(crate) fn reserve_cut_rows(&mut self, bound: impl Fn(NodePos) -> (usize, usize)) {
        for (i, slot) in self.bases.iter_mut().enumerate() {
            if let Some(captured) = slot {
                let (rows, cut_rows) = bound(NodePos(i % self.num_nodes));
                captured.reserve_cut_rows(rows, cut_rows);
            }
        }
    }

    /// Clears `out` and fills it with `n_workers` disjoint mutable sub-views of the
    /// first `n_active` scenarios, worker `w` receiving `partition(n_active,
    /// n_workers, w)` — the same windows the caller's per-worker trajectory records
    /// use. A vector kept across calls keeps its allocation.
    ///
    /// The store is sized for the largest per-rank share, so a rank holding fewer
    /// scenarios must pass its own count: splitting the full store instead shifts
    /// every worker window off its records. Scenarios `n_active..num_scenarios()`
    /// belong to no slice. A worker beyond `n_active` receives an empty slice.
    ///
    /// # Panics
    ///
    /// Panics if `n_active > num_scenarios()`, and in debug builds if
    /// `n_workers == 0`.
    pub(crate) fn split_workers_into<'a>(
        &'a mut self,
        n_active: usize,
        n_workers: usize,
        out: &mut Vec<BasisStoreSliceMut<'a>>,
    ) {
        debug_assert!(n_workers > 0, "n_workers must be > 0");
        assert!(
            n_active <= self.num_scenarios(),
            "split_workers_into: {n_active} active scenarios exceed the store's {}",
            self.num_scenarios()
        );
        out.clear();
        let mut bases_rem = self.bases.as_mut_slice();

        for w in 0..n_workers {
            let (start, end) = partition(n_active, n_workers, w);
            let chunk = (end - start) * self.num_nodes;
            let (bases_left, bases_rest) = bases_rem.split_at_mut(chunk);
            bases_rem = bases_rest;
            out.push(BasisStoreSliceMut {
                bases: bases_left,
                scenario_offset: start,
                scenario_end: end,
                num_nodes: self.num_nodes,
            });
        }
    }
}

/// A mutable sub-view of a [`BasisStore`] covering a contiguous range of
/// scenarios.
///
/// Produced by [`BasisStore::split_workers_into`]. Each slice is exclusive
/// to one worker thread; multiple slices can coexist because they cover
/// disjoint memory regions.
pub struct BasisStoreSliceMut<'a> {
    /// Sub-slice of the flat basis array for this worker's scenario range.
    bases: &'a mut [Option<CapturedBasis>],
    /// Absolute scenario index of the first scenario in this slice.
    scenario_offset: usize,
    /// One past the absolute index of the last scenario in this slice.
    scenario_end: usize,
    /// Node-axis length (see [`BasisStore`]).
    num_nodes: usize,
}

impl BasisStoreSliceMut<'_> {
    /// Get an immutable reference to the basis at absolute scenario index
    /// `scenario` and node position `node`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    ///
    /// # Panics
    ///
    /// Panics if `scenario` lies outside [`Self::scenario_window`].
    #[must_use]
    pub fn get(&self, scenario: usize, node: NodePos) -> Option<&CapturedBasis> {
        self.bases[self.flat_index(scenario, node)].as_ref()
    }

    /// This slice's global scenario window `[start, end)` — the scenarios
    /// `get`/`get_mut` address in-bounds. Each worker solves exactly the
    /// scenarios in its own window, so its `get(m, node)` never indexes outside
    /// this slice.
    #[must_use]
    pub fn scenario_window(&self) -> (usize, usize) {
        (self.scenario_offset, self.scenario_end)
    }

    /// Get a mutable reference to the basis slot at absolute scenario index
    /// `scenario` and node position `node`.
    ///
    /// # Panics
    ///
    /// Panics if `scenario` lies outside [`Self::scenario_window`].
    pub fn get_mut(&mut self, scenario: usize, node: NodePos) -> &mut Option<CapturedBasis> {
        let idx = self.flat_index(scenario, node);
        &mut self.bases[idx]
    }

    fn flat_index(&self, scenario: usize, node: NodePos) -> usize {
        debug_assert!(
            (self.scenario_offset..self.scenario_end).contains(&scenario),
            "scenario {scenario} outside this slice's window [{}, {})",
            self.scenario_offset,
            self.scenario_end
        );
        (scenario - self.scenario_offset) * self.num_nodes + node.0
    }
}

#[cfg(test)]
impl ScratchBuffers {
    /// Hydro-indexed noise buffers sized by `hydro_count`, everything else empty;
    /// lazy-rebuild keys at their never-built sentinel.
    #[must_use]
    pub(crate) fn for_test(hydro_count: usize) -> Self {
        Self {
            incoming_lags_buf: Vec::new(),
            realized_inflow_buf: Vec::with_capacity(hydro_count),
            par_mean_buf: vec![0.0; hydro_count],
            par_noise_term_buf: vec![0.0; hydro_count],
            cut_slots_buf: Vec::new(),
            cut_bracket_buf: Vec::new(),
            slab_misses: 0,
            cut_rhs_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            inflow_slack_lower_buf: Vec::with_capacity(hydro_count),
            inflow_slack_upper_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_indices_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_lower_scaled_buf: Vec::with_capacity(hydro_count),
            inflow_slack_col_upper_scaled_buf: Vec::with_capacity(hydro_count),
            last_inflow_slack_col_start: usize::MAX,
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: Vec::new(),
            lag_weight_accum: Vec::new(),
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            corr_scratch: Vec::new(),
            current_node_buf: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "highs")]
    use super::PatchBuffer;
    use super::{
        BasisStore, BasisStoreSliceMut, ByNodeScratch, CapturedBasis, NodeId, NodePos,
        OpeningRecords, ScratchBuffers, SolverWorkspace, WorkspacePool, WorkspaceSizing,
        record_dims, recycle,
    };
    use crate::context::StageContext;
    use crate::lead_time::AnticipatedResolution;
    use crate::lp::indexer::StateSpace;
    use crate::solve::partition;
    use crate::test_support::{TrainingContextFixture, inflow_coupling_with_entries, state_layout};
    use novomodelo_io::config::BackwardScheduler;
    use novomodelo_solver::{
        Basis, BasisStatus, SolutionView, SolverError, SolverInterface, SolverStatistics,
        types::{RowBatch, StageTemplate},
    };

    /// Minimal no-op solver for workspace tests.
    struct MockSolver;

    impl SolverInterface for MockSolver {
        type Profile = novomodelo_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &novomodelo_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_row_lower_bounds_range(&mut self, _first_row: usize, _lower: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".into(),
                error_code: None,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    /// Compile-time assertion that `SolverWorkspace<MockSolver>` is `Send`.
    fn assert_send<T: Send>() {}

    #[test]
    fn test_workspace_send_bound() {
        assert_send::<SolverWorkspace<MockSolver>>();
    }

    /// A [`StageContext`] over empty slices, for pool-sizing tests that never
    /// exercise load buses or geometry.
    fn empty_stage_ctx() -> StageContext<'static> {
        StageContext {
            templates: &[],
            state_boxes: &[],
            geometry_per_stage: &[],
            inflow_coupling: &[],
            cost_scale_factor: 1.0,
            load_bus_indices: &[],
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
        }
    }

    #[test]
    fn test_workspace_pool_size() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 4);
    }

    #[test]
    fn test_workspace_buffer_dimensions() {
        // M=0, B=0, three rows shifted by the coupling → patch_buf length 3
        // n_state=9 → current_state capacity = 9
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let couplings = [inflow_coupling_with_entries(3, &[(0, 0), (1, 1), (2, 2)])];
        let stage_ctx = StageContext {
            inflow_coupling: &couplings,
            ..empty_stage_ctx()
        };
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &stage_ctx,
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 3, "patch_buf length");
            assert_eq!(ws.current_state.capacity(), 9, "current_state capacity");
            assert_eq!(ws.current_state.len(), 0, "current_state starts empty");
        }
    }

    #[test]
    fn scratch_buffers_size_the_folded_prep_vectors() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let scratch = ScratchBuffers::new(
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing {
                initial_pool_capacity: 7,
                ..WorkspaceSizing::default()
            },
        );
        for (name, buf) in [
            ("par_mean_buf", &scratch.par_mean_buf),
            ("par_noise_term_buf", &scratch.par_noise_term_buf),
        ] {
            assert_eq!(buf.len(), 3, "{name} must hold one value per hydro");
            assert!(
                buf.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()),
                "{name} must start at +0.0"
            );
        }
        for (name, len, capacity) in [
            (
                "cut_slots_buf",
                scratch.cut_slots_buf.len(),
                scratch.cut_slots_buf.capacity(),
            ),
            (
                "cut_bracket_buf",
                scratch.cut_bracket_buf.len(),
                scratch.cut_bracket_buf.capacity(),
            ),
            (
                "cut_rhs_buf",
                scratch.cut_rhs_buf.len(),
                scratch.cut_rhs_buf.capacity(),
            ),
        ] {
            assert_eq!(len, 0, "{name} must start empty");
            assert!(
                capacity >= 7,
                "{name} must reserve initial_pool_capacity (7), got {capacity}"
            );
        }
    }

    #[test]
    fn backward_accumulators_size_the_coupled_duals_to_the_hydro_count() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 2);
        for (i, ws) in pool.workspaces.iter().enumerate() {
            let coupled = &ws.backward_accum.coupled_duals_buf;
            assert_eq!(coupled.len(), 3, "workspace {i}: one value per hydro");
            assert!(
                coupled.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()),
                "workspace {i}: coupled_duals_buf must start at +0.0"
            );
        }
    }

    #[test]
    fn reserve_cut_scratch_grows_the_cut_row_buffers_only_upward() {
        let ctx = TrainingContextFixture::new(state_layout(2, 1));
        let columned_len = record_dims(ctx.training_ctx().cut_state_layouts).0;
        let mut pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        let cut_row_buffers = |s: &ScratchBuffers| {
            [
                (
                    "cut_slots_buf",
                    s.cut_slots_buf.len(),
                    s.cut_slots_buf.capacity(),
                ),
                (
                    "cut_bracket_buf",
                    s.cut_bracket_buf.len(),
                    s.cut_bracket_buf.capacity(),
                ),
                ("cut_rhs_buf", s.cut_rhs_buf.len(), s.cut_rhs_buf.capacity()),
            ]
        };

        pool.reserve_cut_scratch(columned_len, 5, 0);
        for (i, ws) in pool.workspaces.iter().enumerate() {
            for (name, len, capacity) in cut_row_buffers(&ws.scratch) {
                assert_eq!(len, 0, "workspace {i}: {name} must stay empty");
                assert!(
                    capacity >= 5,
                    "workspace {i}: {name} must reserve 5, got {capacity}"
                );
            }
        }

        pool.workspaces[0].scratch.cut_bracket_buf.resize(5, 1.0);
        pool.reserve_cut_scratch(columned_len, 3, 0);
        for (i, ws) in pool.workspaces.iter().enumerate() {
            for (name, _, capacity) in cut_row_buffers(&ws.scratch) {
                assert!(
                    capacity >= 5,
                    "workspace {i}: {name} must not shrink, got {capacity}"
                );
            }
        }
        let bracket = &pool.workspaces[0].scratch.cut_bracket_buf;
        assert_eq!(bracket.len(), 5, "a smaller reserve must keep the contents");
        assert!(
            bracket.iter().all(|v| v.to_bits() == 1.0_f64.to_bits()),
            "a smaller reserve must keep the contents"
        );

        let scratch = &mut pool.workspaces[1].scratch;
        scratch.cut_slots_buf = Vec::with_capacity(10);
        scratch.cut_bracket_buf = Vec::with_capacity(10);
        scratch.cut_rhs_buf = Vec::with_capacity(10);
        scratch.cut_slots_buf.resize(2, 0);
        scratch.cut_bracket_buf.resize(2, 0.0);
        scratch.cut_rhs_buf.resize(2, 0.0);
        for (name, _, capacity) in cut_row_buffers(scratch) {
            assert!(
                (10..12).contains(&capacity),
                "{name}: the case needs a capacity below the target, got {capacity}"
            );
        }
        pool.reserve_cut_scratch(columned_len, 12, 0);
        for (name, len, capacity) in cut_row_buffers(&pool.workspaces[1].scratch) {
            assert_eq!(len, 2, "{name} must keep its contents");
            assert!(
                capacity >= 12,
                "{name}: a target past the capacity must grow a partly filled buffer, got {capacity}"
            );
        }
    }

    #[test]
    fn workspace_shift_region_follows_the_most_touched_stage() {
        let couplings = [
            inflow_coupling_with_entries(2, &[(0, 1)]),
            inflow_coupling_with_entries(2, &[(0, 2), (1, 2), (0, 5), (1, 7)]),
            inflow_coupling_with_entries(2, &[(0, 3), (1, 4)]),
        ];
        let stage_ctx = StageContext {
            inflow_coupling: &couplings,
            ..empty_stage_ctx()
        };
        let state = StateSpace::new(
            2,
            1,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[1, 1],
        );

        let ctx = TrainingContextFixture::new(state);
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &stage_ctx,
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 2);
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 3);
        }
    }

    #[test]
    fn test_workspace_pool_zero_threads() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let pool = WorkspacePool::new(
            0,
            0,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 0);
    }

    #[test]
    fn test_workspace_pool_single_thread() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            1,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), 1);
        assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 0);
    }

    #[test]
    fn test_workspace_pool_each_solver_independent() {
        // Factory is called n_threads times; each workspace gets its own instance.
        // Verify by checking pool size matches factory call expectation.
        let n = 6;
        let ctx = TrainingContextFixture::new(state_layout(1, 0));
        let pool = WorkspacePool::new(
            0,
            n,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(pool.workspaces.len(), n);
    }

    #[test]
    fn test_scratch_buffers_zero_downstream_par_order_empty_buffers() {
        let ctx = TrainingContextFixture::new(state_layout(5, 2)).downstream_par_order(0);
        let scratch = ScratchBuffers::new(
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert!(
            scratch.downstream_accumulator.is_empty(),
            "downstream_accumulator must be empty when downstream_par_order=0"
        );
        assert!(
            scratch.downstream_completed_lags.is_empty(),
            "downstream_completed_lags must be empty when downstream_par_order=0"
        );
        assert_eq!(
            scratch.downstream_weight_accum, 0.0,
            "downstream_weight_accum must be 0.0"
        );
        assert_eq!(
            scratch.downstream_n_completed, 0,
            "downstream_n_completed must be 0"
        );
    }

    #[test]
    fn test_scratch_buffers_nonzero_downstream_par_order_allocates_correctly() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2)).downstream_par_order(2);
        let scratch = ScratchBuffers::new(
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert_eq!(
            scratch.downstream_accumulator.len(),
            3,
            "downstream_accumulator.len() must equal hydro_count"
        );
        assert_eq!(
            scratch.downstream_completed_lags.len(),
            6,
            "downstream_completed_lags.len() must equal hydro_count * downstream_par_order"
        );
        assert!(
            scratch.downstream_accumulator.iter().all(|&v| v == 0.0),
            "downstream_accumulator must be initialized to 0.0"
        );
        assert!(
            scratch.downstream_completed_lags.iter().all(|&v| v == 0.0),
            "downstream_completed_lags must be initialized to 0.0"
        );
        assert_eq!(scratch.downstream_weight_accum, 0.0);
        assert_eq!(scratch.downstream_n_completed, 0);
    }

    #[test]
    fn test_workspace_pool_propagates_downstream_par_order() {
        let ctx = TrainingContextFixture::new(state_layout(3, 2)).downstream_par_order(2);
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(
                ws.scratch.downstream_accumulator.len(),
                3,
                "downstream_accumulator.len() per workspace"
            );
            assert_eq!(
                ws.scratch.downstream_completed_lags.len(),
                6,
                "downstream_completed_lags.len() per workspace"
            );
            assert_eq!(ws.scratch.downstream_weight_accum, 0.0);
            assert_eq!(ws.scratch.downstream_n_completed, 0);
        }
    }

    // ---------------------------------------------------------------------------
    // ByNodeScratch tests
    // ---------------------------------------------------------------------------

    #[test]
    fn backward_by_node_scratch_default_is_empty() {
        let scratch = ByNodeScratch::default();
        assert_eq!(
            scratch.location.capacity(),
            0,
            "location must start capacity 0"
        );
        assert!(scratch.finish.is_empty());
        assert!(scratch.block_pivots.is_empty());
        assert!(scratch.block_pivots_prev.is_empty());
        assert_eq!(scratch.n_blocks_max, 0);
        assert!(scratch.block_order.is_empty());
    }

    #[test]
    fn backward_by_node_scratch_sized_matches_study_dims() {
        let max_local_fwd = 3_usize;
        let bwd_max_openings = 4_usize;
        let n_state = 5_usize;
        let num_stages = 6_usize;

        let scratch = ByNodeScratch::sized(
            max_local_fwd,
            bwd_max_openings,
            n_state,
            0,
            0,
            num_stages,
            2,
        );

        assert!(
            scratch.location.capacity() >= max_local_fwd * bwd_max_openings,
            "location must reserve one entry per (trial point, opening)"
        );
        assert_eq!(scratch.finish.len(), 2, "one finish scratch per worker");
        assert!(
            scratch
                .finish
                .iter()
                .all(|finish| finish.coeffs_buf.len() == n_state),
            "every finish entry's coeffs_buf must be pre-sized to n_state"
        );
        assert_eq!(scratch.cut_arena.len(), max_local_fwd * n_state);
        assert_eq!(scratch.cut_intercepts.len(), max_local_fwd);
        assert_eq!(
            scratch.block_pivots.len(),
            num_stages * bwd_max_openings,
            "block_pivots must hold num_stages * bwd_max_openings accumulator cells"
        );
        assert!(
            scratch
                .block_pivots
                .iter()
                .all(|&(sum, count)| sum == 0 && count == 0),
            "a freshly sized accumulator must start zeroed"
        );
        assert_eq!(
            scratch.block_pivots_prev.len(),
            num_stages * bwd_max_openings,
            "block_pivots_prev must mirror block_pivots' shape"
        );
        assert!(
            scratch
                .block_pivots_prev
                .iter()
                .all(|&(sum, count)| sum == 0 && count == 0),
            "a freshly sized double-buffer companion must start zeroed"
        );
        assert_eq!(scratch.n_blocks_max, bwd_max_openings);
        assert_eq!(
            scratch.block_order.len(),
            bwd_max_openings,
            "block_order must hold n_blocks_max claim-order scratch slots"
        );
    }

    #[test]
    fn backward_by_node_scratch_sized_zero_dims_yields_empty_locations() {
        let scratch = ByNodeScratch::sized(0, 0, 0, 0, 0, 0, 0);
        assert_eq!(scratch.location.capacity(), 0);
        assert_eq!(
            scratch.finish.len(),
            1,
            "a zero-worker scratch still holds one finish entry"
        );
        assert!(scratch.finish[0].coeffs_buf.is_empty());
        assert!(scratch.block_pivots.is_empty());
        assert!(scratch.block_pivots_prev.is_empty());
        assert!(scratch.block_order.is_empty());
    }

    #[test]
    fn by_node_scratch_reserves_the_aggregate_children_of_each_worker() {
        let scratch = ByNodeScratch::sized(3, 4, 5, 2, 3, 6, 2);
        assert!(scratch.location.capacity() >= 12);
        assert_eq!(scratch.finish.len(), 2);
        for finish in &scratch.finish {
            assert_eq!(finish.coeffs_buf.len(), 5);
            assert!(finish.aggregate.children.capacity() >= 4);
        }
        assert_eq!(scratch.block_pivots.len(), 6 * 4);
        assert!(scratch.worker_counts.capacity() >= 2);
    }

    #[test]
    fn opening_records_round_trip_without_reallocating() {
        let pairs = [(2_usize, 0.5_f64), (5, -1.25), (9, 3.0)];
        let fill = |records: &mut OpeningRecords| {
            records.clear(2, 1);
            records.reserve(2, 2, 1, 3);
            records.push((1, 0), 7.0, &[1.0, -0.0], &[4.0], &[]);
            records.push((0, 1), 9.5, &[2.0, 3.0], &[-6.0], &pairs);
        };
        let mut records = OpeningRecords::default();
        fill(&mut records);
        let first = records.buffer_identity();

        assert_eq!(records.len(), 2);
        assert_eq!(records.key(0), (1, 0));
        assert_eq!(records.key(1), (0, 1));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let view = records.view(0);
        assert_eq!(view.objective.to_bits(), 7.0_f64.to_bits());
        assert_eq!(bits(view.state_duals), bits(&[1.0, -0.0]));
        assert_eq!(bits(view.coupled_duals), bits(&[4.0]));
        assert!(view.cut_duals.is_empty());
        let view = records.view(1);
        assert_eq!(view.objective.to_bits(), 9.5_f64.to_bits());
        assert_eq!(bits(view.state_duals), bits(&[2.0, 3.0]));
        assert_eq!(bits(view.coupled_duals), bits(&[-6.0]));
        assert_eq!(view.cut_duals, pairs.as_slice());

        fill(&mut records);
        assert_eq!(
            records.buffer_identity(),
            first,
            "a refill must reuse its buffers"
        );

        let views: Vec<super::OpeningDuals<'static>> = Vec::with_capacity(4);
        let (ptr, capacity) = (views.as_ptr(), views.capacity());
        let reborrowed: Vec<super::OpeningDuals<'_>> = recycle(views);
        assert_eq!(
            (reborrowed.as_ptr(), reborrowed.capacity()),
            (ptr, capacity)
        );
        let views: Vec<super::OpeningDuals<'static>> = recycle(reborrowed);
        assert_eq!((views.as_ptr(), views.capacity()), (ptr, capacity));
    }

    #[test]
    fn a_record_with_more_pairs_than_reserved_keeps_every_pair() {
        let wide = [
            (1_usize, 0.5_f64),
            (3, -0.0),
            (4, 2.25),
            (6, -7.0),
            (8, 1e-3),
            (11, 4.0),
            (12, -0.75),
        ];
        let narrow = [(0_usize, -2.0_f64), (5, 3.5)];
        let mut records = OpeningRecords::default();
        records.clear(1, 1);
        records.reserve(2, 1, 1, 1);
        let (_, reserved_pairs) = records.buffer_identity()[5];
        assert!(reserved_pairs < wide.len());

        records.push((0, 0), 1.0, &[0.0], &[0.0], &wide);
        records.push((0, 1), 2.0, &[0.0], &[0.0], &narrow);

        let bits = |pairs: &[(usize, f64)]| {
            pairs
                .iter()
                .map(|&(slot, y)| (slot, y.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(records.view(0).cut_duals), bits(&wide));
        assert_eq!(bits(records.view(1).cut_duals), bits(&narrow));
    }

    #[test]
    fn opening_records_are_sized_by_scheduler() {
        const MAX_OPENINGS: usize = 5;
        const MAX_LOCAL_FWD: usize = 4;
        let ctx = TrainingContextFixture::new(state_layout(3, 2));
        let reserved_slots = |scheduler: BackwardScheduler, max_local_fwd: usize| {
            let sizing = WorkspaceSizing {
                max_openings: MAX_OPENINGS,
                max_local_fwd,
                ..WorkspaceSizing::default()
            };
            let mut pool = WorkspacePool::new(
                0,
                1,
                &ctx.training_ctx(),
                &empty_stage_ctx(),
                sizing,
                || MockSolver,
            );
            pool.reserve_backward_scratch(&ctx.training_ctx(), sizing, &scheduler);
            pool.workspaces[0]
                .backward_accum
                .opening_records
                .buffer_identity()[0]
                .1
        };
        let by_scenario = BackwardScheduler::ByScenario {};
        let by_node = BackwardScheduler::ByNode { block_size: None };

        let by_scenario_slots = reserved_slots(by_scenario, MAX_LOCAL_FWD);
        assert!(
            (MAX_OPENINGS..MAX_LOCAL_FWD * MAX_OPENINGS).contains(&by_scenario_slots),
            "by-scenario reserved {by_scenario_slots} record slots"
        );
        assert!(reserved_slots(by_node, MAX_LOCAL_FWD) >= MAX_LOCAL_FWD * MAX_OPENINGS);
        assert_eq!(reserved_slots(by_scenario, 0), 0);
        assert_eq!(reserved_slots(by_node, 0), 0);
    }

    // ---------------------------------------------------------------------------
    // BasisStore tests
    // ---------------------------------------------------------------------------

    #[test]
    fn basis_store_new_all_none() {
        let store = BasisStore::new(3, 5);
        assert_eq!(store.num_scenarios(), 3);
        assert_eq!(store.num_nodes(), 5);
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must start as None"
                );
            }
        }
    }

    #[test]
    fn basis_store_get_mut_set_and_retrieve() {
        let mut store = BasisStore::new(2, 3);
        *store.get_mut(1, NodePos(2)) = Some(CapturedBasis::new(4, 2, 0, 0, NodeId(0)));
        assert!(store.get(1, NodePos(2)).is_some());
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(1, NodePos(0)).is_none());
    }

    #[test]
    fn basis_store_zero_scenarios() {
        let store = BasisStore::new(0, 5);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_nodes(), 5);
    }

    #[test]
    fn basis_store_zero_stages() {
        let store = BasisStore::new(3, 0);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_nodes(), 0);
    }

    #[test]
    fn basis_store_split_workers_into_disjoint_writes() {
        // 4 scenarios, 3 stages, 2 workers.
        // Worker 0 covers scenarios 0..2; worker 1 covers scenarios 2..4.
        let mut store = BasisStore::new(4, 3);
        let mut slices = Vec::new();
        store.split_workers_into(4, 2, &mut slices);

        // Worker 0 writes to scenario 0 stage 1.
        *slices[0].get_mut(0, NodePos(1)) = Some(CapturedBasis::new(2, 1, 0, 0, NodeId(0)));
        // Worker 1 writes to scenario 3 stage 2.
        *slices[1].get_mut(3, NodePos(2)) = Some(CapturedBasis::new(2, 1, 0, 0, NodeId(0)));

        // Drop slices to release the borrow on store.
        drop(slices);

        assert!(
            store.get(0, NodePos(1)).is_some(),
            "scenario 0 stage 1 must be populated"
        );
        assert!(
            store.get(3, NodePos(2)).is_some(),
            "scenario 3 stage 2 must be populated"
        );
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(3, NodePos(0)).is_none());
    }

    #[test]
    fn basis_store_split_single_worker() {
        let mut store = BasisStore::new(3, 2);
        let mut slices = Vec::new();
        store.split_workers_into(3, 1, &mut slices);
        *slices[0].get_mut(2, NodePos(1)) = Some(CapturedBasis::new(1, 0, 0, 0, NodeId(0)));
        drop(slices);
        assert!(store.get(2, NodePos(1)).is_some());
    }

    #[test]
    fn basis_store_split_more_workers_than_scenarios() {
        // 2 scenarios, 4 workers → workers 2 and 3 receive empty slices.
        let mut store = BasisStore::new(2, 3);
        let mut slices = Vec::new();
        store.split_workers_into(2, 4, &mut slices);
        assert_eq!(slices.len(), 4);
        // Workers 0 and 1 cover 1 scenario each; workers 2 and 3 cover 0.
        assert_eq!(slices[0].bases.len(), 3); // 1 scenario × 3 stages
        assert_eq!(slices[1].bases.len(), 3);
        assert_eq!(slices[2].bases.len(), 0);
        assert_eq!(slices[3].bases.len(), 0);
    }

    #[test]
    fn basis_store_slice_offset_correct() {
        // 6 scenarios, 2 stages, 3 workers → 2 scenarios each.
        let mut store = BasisStore::new(6, 2);
        let mut slices = Vec::new();
        store.split_workers_into(6, 3, &mut slices);

        // Worker 1 covers absolute scenarios 2..4.
        *slices[1].get_mut(2, NodePos(0)) = Some(CapturedBasis::new(1, 0, 0, 0, NodeId(0)));
        *slices[1].get_mut(3, NodePos(1)) = Some(CapturedBasis::new(1, 0, 0, 0, NodeId(0)));
        drop(slices);

        assert!(store.get(2, NodePos(0)).is_some());
        assert!(store.get(3, NodePos(1)).is_some());
        assert!(store.get(0, NodePos(0)).is_none());
        assert!(store.get(4, NodePos(0)).is_none());
    }

    #[test]
    fn basis_store_split_tiles_only_the_active_scenarios() {
        // A rank with 5 of a 6-scenario store's scenarios: the windows must be
        // the record windows `partition(5, 4, w)`, not `partition(6, 4, w)`.
        let mut store = BasisStore::new(6, 2);
        let mut slices = Vec::new();
        store.split_workers_into(5, 4, &mut slices);
        let windows: Vec<(usize, usize)> = slices
            .iter()
            .map(BasisStoreSliceMut::scenario_window)
            .collect();
        assert_eq!(windows, vec![(0, 2), (2, 3), (3, 4), (4, 5)]);
        for (w, window) in windows.iter().enumerate() {
            assert_eq!(*window, partition(5, 4, w));
        }

        for slice in &mut slices {
            let (start, end) = slice.scenario_window();
            for m in start..end {
                *slice.get_mut(m, NodePos(1)) = Some(CapturedBasis::new(1, 0, 0, 0, NodeId(0)));
            }
        }
        drop(slices);

        assert!((0..5).all(|m| store.get(m, NodePos(1)).is_some()));
        assert!(store.get(5, NodePos(1)).is_none());
    }

    #[test]
    fn split_workers_into_reuses_the_vector_and_gives_the_partition_windows() {
        let mut store = BasisStore::new(6, 2);
        let mut kept: Vec<BasisStoreSliceMut<'static>> = Vec::with_capacity(4);
        let allocation = (kept.as_ptr().addr(), kept.capacity());
        for (n_active, n_workers) in [(6, 3), (5, 4), (0, 2), (6, 4)] {
            let expected: Vec<(usize, usize)> = (0..n_workers)
                .map(|w| partition(n_active, n_workers, w))
                .collect();
            let mut slices: Vec<BasisStoreSliceMut<'_>> = recycle(std::mem::take(&mut kept));
            store.split_workers_into(n_active, n_workers, &mut slices);
            let windows: Vec<(usize, usize)> = slices
                .iter()
                .map(BasisStoreSliceMut::scenario_window)
                .collect();
            assert_eq!(windows, expected, "{n_active} active, {n_workers} workers");
            kept = recycle(slices);
            assert_eq!(
                (kept.as_ptr().addr(), kept.capacity()),
                allocation,
                "{n_active} active, {n_workers} workers"
            );
        }
    }

    #[test]
    fn basis_store_split_with_no_active_scenarios_yields_empty_windows() {
        let mut store = BasisStore::new(3, 2);
        let mut slices = Vec::new();
        store.split_workers_into(0, 2, &mut slices);
        assert!(
            slices
                .iter()
                .all(|s| s.scenario_window() == (0, 0) && s.bases.is_empty())
        );
    }

    #[test]
    #[should_panic(expected = "4 active scenarios exceed the store's 3")]
    fn basis_store_split_rejects_more_active_scenarios_than_slots() {
        let mut store = BasisStore::new(3, 2);
        store.split_workers_into(4, 2, &mut Vec::new());
    }

    #[test]
    fn test_captured_basis_new_capacities() {
        let cb = CapturedBasis::new(4, 6, 3, 10, NodeId(9));
        assert_eq!(cb.basis.row_status.len(), 6, "row_status length");
        assert_eq!(cb.base_row_count, 3, "base_row_count");
        assert_eq!(cb.node_id, NodeId(9), "node_id");
        assert!(
            cb.cut_row_slots.capacity() >= 10,
            "cut_row_slots capacity must be >= 10 (got {})",
            cb.cut_row_slots.capacity()
        );
        assert_eq!(cb.cut_row_slots.len(), 0, "cut_row_slots starts empty");
    }

    #[test]
    fn test_basis_store_holds_captured_basis() {
        let mut store = BasisStore::new(3, 5);
        // All 15 slots start as None.
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must be None before any write"
                );
            }
        }
        // Write a CapturedBasis at [1][3]; read it back.
        *store.get_mut(1, NodePos(3)) = Some(CapturedBasis::new(4, 6, 3, 10, NodeId(0)));
        let retrieved = store.get(1, NodePos(3));
        assert!(retrieved.is_some(), "slot [1][3] must be Some after write");
        let cb = retrieved.expect("just checked is_some");
        assert_eq!(cb.base_row_count, 3);
        // All other slots remain None.
        for s in 0..3 {
            for t in 0..5 {
                if s == 1 && t == 3 {
                    continue;
                }
                assert!(
                    store.get(s, NodePos(t)).is_none(),
                    "slot [{s}][{t}] must remain None"
                );
            }
        }
    }

    #[test]
    fn test_recon_slot_lookup_presized() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            4,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing {
                initial_pool_capacity: 50,
                ..WorkspaceSizing::default()
            },
            || MockSolver,
        );
        for (i, ws) in pool.workspaces.iter().enumerate() {
            assert_eq!(
                ws.scratch.recon_slot_lookup.len(),
                50,
                "workspace {i}: recon_slot_lookup.len() must equal initial_pool_capacity (50)"
            );
            assert!(
                ws.scratch.recon_slot_lookup.iter().all(Option::is_none),
                "workspace {i}: all recon_slot_lookup entries must be None"
            );
        }
        // Verify zero initial_pool_capacity produces an empty vec.
        let pool_empty = WorkspacePool::new(
            0,
            1,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        assert_eq!(
            pool_empty.workspaces[0].scratch.recon_slot_lookup.len(),
            0,
            "initial_pool_capacity=0 must produce empty recon_slot_lookup"
        );
    }

    #[test]
    fn test_workspace_pool_assigns_sequential_worker_ids() {
        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            /* rank = */ 3,
            /* n_workers = */ 5,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        let ws_slice = &pool.workspaces;
        assert_eq!(ws_slice.len(), 5);
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for ws in ws_slice {
            assert_eq!(ws.rank, 3);
            assert!(ws.worker_id >= 0 && ws.worker_id < 5);
            assert!(seen.insert(ws.worker_id), "worker_id duplicated");
        }
    }

    // ---------------------------------------------------------------------------
    // CapturedBasis wire-format round-trip tests
    // ---------------------------------------------------------------------------

    /// Round-trip test: single stage with fully populated metadata.
    ///
    /// Constructs a `CapturedBasis` with known fields, packs via
    /// `to_broadcast_payload`, then unpacks via `try_from_broadcast_payload`.
    /// Asserts field-by-field equality with the original.
    #[test]
    fn test_captured_basis_round_trip_populated() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower, BasisStatus::Basic, BasisStatus::Upper],
                row_status: vec![BasisStatus::Zero, BasisStatus::Nonbasic],
            },
            base_row_count: 1,
            cut_row_slots: vec![10_u32, 20],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        let mut i32_cursor = 0_usize;
        let result = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail");
        let recovered = result.expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.basis.col_status, original.basis.col_status,
            "col_status"
        );
        assert_eq!(
            recovered.basis.row_status, original.basis.row_status,
            "row_status"
        );
        assert_eq!(
            recovered.base_row_count, original.base_row_count,
            "base_row_count"
        );
        assert_eq!(
            recovered.cut_row_slots, original.cut_row_slots,
            "cut_row_slots"
        );
        // Cursors must have advanced past the full payload.
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
    }

    /// Round-trip test: single stage with empty `cut_row_slots`.
    #[test]
    fn test_captured_basis_round_trip_empty_metadata() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Superbasic, BasisStatus::Fixed],
                row_status: vec![BasisStatus::Lower],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        let mut i32_cursor = 0_usize;
        let result = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail");
        let recovered = result.expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert!(
            recovered.cut_row_slots.is_empty(),
            "cut_row_slots must be empty"
        );
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
    }

    /// Multi-stage round-trip: one `Some` stage followed by one `None` stage.
    ///
    /// Packs the `Some` stage via `to_broadcast_payload`, writes a `0_i32`
    /// sentinel for the `None` stage, then loops `try_from_broadcast_payload`
    /// twice and asserts the recovered `Option`s match.
    #[test]
    fn test_captured_basis_round_trip_multi_stage() {
        let populated = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic, BasisStatus::Upper, BasisStatus::Zero],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![100_u32, 200, 300],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();

        // Stage 0: Some — pack via method.
        populated.to_broadcast_payload(&mut i32_buf);
        // Stage 1: None — caller writes the 0 sentinel directly.
        i32_buf.push(0_i32);

        let mut i32_cursor = 0_usize;

        // Unpack stage 0.
        let stage0 = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("stage 0 must not fail")
            .expect("stage 0 sentinel is 1; must return Some");

        assert_eq!(stage0.basis.col_status, populated.basis.col_status);
        assert_eq!(stage0.basis.row_status, populated.basis.row_status);
        assert_eq!(stage0.base_row_count, populated.base_row_count);
        assert_eq!(stage0.cut_row_slots, populated.cut_row_slots);

        // Unpack stage 1 — must return None, advancing cursor by 1.
        let stage1 = CapturedBasis::try_from_broadcast_payload(1, &i32_buf, &mut i32_cursor)
            .expect("stage 1 must not fail");
        assert!(stage1.is_none(), "stage 1 sentinel is 0; must return None");

        assert_eq!(
            i32_cursor,
            i32_buf.len(),
            "i32_cursor must be at end after both stages"
        );
    }

    /// Truncated i32 buffer: truncate the packed buffer by 1 element, assert
    /// `Err(SddpError::Validation(msg))` where `msg` contains "truncated" and
    /// the stage index.
    #[test]
    fn test_captured_basis_truncated_i32_buffer() {
        use crate::SddpError;

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower, BasisStatus::Basic],
                row_status: vec![BasisStatus::Upper],
            },
            base_row_count: 1,
            cut_row_slots: vec![5_u32],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf);

        // Truncate i32 buffer by 1.
        i32_buf.pop();

        let mut i32_cursor = 0_usize;
        let err = CapturedBasis::try_from_broadcast_payload(7, &i32_buf, &mut i32_cursor)
            .expect_err("truncated buffer must return Err");

        match err {
            SddpError::Validation(ref msg) => {
                assert!(
                    msg.contains("truncated"),
                    "error message must contain 'truncated', got: {msg}"
                );
                assert!(
                    msg.contains('7'),
                    "error message must contain stage index 7, got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------------------
    // Version-byte tests
    // ---------------------------------------------------------------------------

    /// Round-trip verification that `to_broadcast_payload` emits
    /// `BASIS_BROADCAST_FORMAT_TAG` at offset 1 of the `i32_buf` (immediately
    /// after the presence sentinel).
    #[test]
    fn to_broadcast_payload_emits_version_byte() {
        use super::BASIS_BROADCAST_FORMAT_TAG;

        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                    BasisStatus::Lower,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![10_u32, 20],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        // Offset 0 is the presence sentinel (1_i32).
        assert_eq!(i32_buf[0], 1_i32, "offset 0 must be the presence sentinel");
        // Offset 1 must be the wire version.
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_FORMAT_TAG,
            "offset 1 must be BASIS_BROADCAST_FORMAT_TAG"
        );

        // Full round-trip must return a bit-equal CapturedBasis.
        let mut i32_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail")
            .expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert_eq!(recovered.cut_row_slots, original.cut_row_slots);
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
    }

    /// Manually overwrite the version field (offset 1) to `1_i32` (the removed
    /// v1 format) and assert that `try_from_broadcast_payload` returns
    /// `Err(SddpError::Validation)` whose message contains
    /// `"unsupported wire version 1"`.
    #[test]
    fn try_from_broadcast_payload_rejects_wrong_version() {
        use crate::SddpError;

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![
                    BasisStatus::Lower,
                    BasisStatus::Basic,
                    BasisStatus::Upper,
                    BasisStatus::Zero,
                ],
                row_status: vec![
                    BasisStatus::Nonbasic,
                    BasisStatus::Superbasic,
                    BasisStatus::Fixed,
                    BasisStatus::Lower,
                ],
            },
            base_row_count: 2,
            cut_row_slots: vec![10_u32, 20],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf);

        // Corrupt the version field (offset 1) to the removed v1 to simulate a
        // stale-version peer.
        i32_buf[1] = 1_i32;

        let mut i32_cursor = 0_usize;
        let err = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect_err("mismatched version must return Err");

        match err {
            SddpError::Validation(ref msg) => {
                assert!(
                    msg.contains("unsupported wire version 1"),
                    "error must contain 'unsupported wire version 1', got: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// A `None` payload (sentinel `0_i32`) returns `Ok(None)` and advances the
    /// i32 cursor by exactly 1 — the version byte is never consumed.
    #[test]
    fn try_from_broadcast_payload_none_does_not_consume_version_byte() {
        // Build a buffer that starts with a 0 sentinel followed by sentinel=1
        // data for a second stage.  After unpacking stage 0 the cursor must
        // sit at offset 1 (i.e. the 0 sentinel was the only consumed element).
        let populated = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Superbasic],
                row_status: vec![BasisStatus::Fixed],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        // Stage 0: None sentinel.
        i32_buf.push(0_i32);
        // Stage 1: Some.
        populated.to_broadcast_payload(&mut i32_buf);

        let mut i32_cursor = 0_usize;

        // Unpack stage 0 — must return None.
        let stage0 = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("stage 0 must not fail");
        assert!(stage0.is_none(), "stage 0 sentinel is 0; must return None");
        // Only the sentinel was consumed (1 element).
        assert_eq!(
            i32_cursor, 1,
            "None path must advance cursor by exactly 1 (only the sentinel)"
        );

        // Unpack stage 1 — must still succeed (version byte is intact).
        let stage1 = CapturedBasis::try_from_broadcast_payload(1, &i32_buf, &mut i32_cursor)
            .expect("stage 1 must not fail")
            .expect("stage 1 sentinel is 1; must return Some");
        assert_eq!(stage1.basis.col_status, populated.basis.col_status);
        assert_eq!(stage1.basis.row_status, populated.basis.row_status);
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
    }

    /// `node_id` rides the wire (version 2): it is written at a fixed header
    /// slot and recovered by `try_from_broadcast_payload`, so the caller no
    /// longer fills it out-of-band. A distinct `node_id` changes exactly one
    /// wire word (the byte count is unchanged), and the value round-trips.
    #[test]
    fn captured_basis_node_id_round_trips_on_the_wire() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower],
                row_status: vec![BasisStatus::Basic],
            },
            base_row_count: 1,
            cut_row_slots: vec![],
            node_id: NodeId(77),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);
        let bytes_with_node_id = i32_buf.len();

        let mut other = original.clone();
        other.node_id = NodeId(-1);
        let mut i32_buf2: Vec<i32> = Vec::new();
        other.to_broadcast_payload(&mut i32_buf2);

        assert_ne!(
            i32_buf, i32_buf2,
            "a distinct node_id must change the payload (it is on the wire)"
        );
        assert_eq!(
            bytes_with_node_id,
            i32_buf2.len(),
            "node_id is one fixed-width word; it must not change the wire byte count"
        );

        let mut i32_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail")
            .expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.node_id, original.node_id,
            "node_id must round-trip through the wire (version 2)"
        );
    }

    // ---------------------------------------------------------------------------
    // Payload layout test
    // ---------------------------------------------------------------------------

    /// The header words sit at fixed offsets, followed by `col_status`, `row_status`
    /// and `cut_row_slots`, in that order.
    #[test]
    fn test_captured_basis_payload_layout_is_header_then_statuses_then_slots() {
        use super::{BASIS_BROADCAST_FORMAT_TAG, basis_status_to_wire_code};

        let cb = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Lower, BasisStatus::Basic],
                row_status: vec![BasisStatus::Upper, BasisStatus::Zero, BasisStatus::Fixed],
            },
            base_row_count: 1,
            cut_row_slots: vec![11_u32, 22],
            node_id: NodeId(5),
        };
        let mut i32_buf: Vec<i32> = Vec::new();
        cb.to_broadcast_payload(&mut i32_buf);

        let code = basis_status_to_wire_code;
        let expected = vec![
            1,
            BASIS_BROADCAST_FORMAT_TAG,
            5,
            2,
            3,
            1,
            2,
            code(BasisStatus::Lower),
            code(BasisStatus::Basic),
            code(BasisStatus::Upper),
            code(BasisStatus::Zero),
            code(BasisStatus::Fixed),
            11,
            22,
        ];
        assert_eq!(i32_buf, expected);
    }

    // ---------------------------------------------------------------------------
    // Layout-invariance tests
    // ---------------------------------------------------------------------------

    /// A basis whose columns include the anticipated-ring slots round-trips its
    /// column count and `cut_row_slots`.
    #[test]
    fn test_captured_basis_column_count_round_trips_with_anticipated_ring_columns() {
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic; 12],
                row_status: vec![BasisStatus::Nonbasic; 8],
            },
            base_row_count: 6,
            cut_row_slots: vec![10_u32, 20],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        let mut i32_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail")
            .expect("sentinel is 1; must return Some");

        assert_eq!(
            recovered.basis.col_status.len(),
            12,
            "col_status length round-trips bit-identically (including the anticipated-ring column slots)"
        );
        assert_eq!(
            recovered.cut_row_slots, original.cut_row_slots,
            "cut_row_slots round-trips bit-identically"
        );
    }

    /// Locks `BASIS_BROADCAST_FORMAT_TAG` at 3: widening the anticipated
    /// ring does not bump the wire version.
    #[test]
    fn test_basis_broadcast_wire_version_stays_stable_with_state_out_column() {
        use super::BASIS_BROADCAST_FORMAT_TAG;

        // Representative basis with enough col_status entries to include the
        // anticipated-ring columns.
        let original = CapturedBasis {
            basis: Basis {
                col_status: vec![BasisStatus::Basic; 16], // includes the anticipated-ring block
                row_status: vec![BasisStatus::Basic; 10],
            },
            base_row_count: 8,
            cut_row_slots: vec![],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        // i32_buf layout per workspace.rs to_broadcast_payload:
        //   [0]: sentinel (1)
        //   [1]: BASIS_BROADCAST_FORMAT_TAG
        //   [2]: node_id
        //   [3]: col_status.len()
        //   [4]: row_status.len()
        //   [5]: base_row_count
        //   [6]: cut_row_slots.len()
        //   [7..]: col_status elements, then row_status, then cut_row_slots
        assert_eq!(i32_buf[0], 1, "sentinel must be 1");
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_FORMAT_TAG,
            "wire version field must equal BASIS_BROADCAST_FORMAT_TAG (= 3)"
        );
        assert_eq!(
            BASIS_BROADCAST_FORMAT_TAG, 3,
            "broadcast wire-format version constant must remain stable across releases"
        );

        // Full round-trip confirms no data drift.
        let mut i32_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail")
            .expect("sentinel is 1; must return Some");

        assert_eq!(recovered.basis.col_status, original.basis.col_status);
        assert_eq!(recovered.basis.row_status, original.basis.row_status);
        assert_eq!(recovered.base_row_count, original.base_row_count);
        assert_eq!(recovered.cut_row_slots, original.cut_row_slots);
    }

    /// Full-`Basis` round-trip on a wide-anticipated-ring solve: the
    /// `col_status` count carries the anticipated ring's outgoing+incoming
    /// blocks in the state region (column count is wider by `2*n_anticipated*k_max`
    /// than the no-anticipated baseline), and `cut_row_slots` carries a slot that
    /// the reconstruction path will match by identity. The wire format serialises
    /// `col_status.len()`/`row_status.len()` as live counts, so the wider column
    /// block round-trips byte-for-byte with no format change.
    #[test]
    fn test_captured_basis_round_trip_relocated_anticipated_column_layout() {
        // Anticipated layout sketch: N=2, L=1, A=2, k_max=2.
        //   n_state             = N*(1+L) + A*k_max = 2*2 + 2*2 = 8.
        //   col_status models a relocated-layout LP whose column count grew by
        //   A=2 relative to the no-anticipated layout (every later column shifted).
        // Two full BasisStatus cycles (7 variants each) so a within-cycle
        // positional mix-up (e.g. cut row written onto a state column) shows as
        // a diff; a shift by exactly 7 is the one class this fixture cannot
        // discriminate.
        let col_status: Vec<BasisStatus> = [
            BasisStatus::Lower,
            BasisStatus::Basic,
            BasisStatus::Upper,
            BasisStatus::Zero,
            BasisStatus::Nonbasic,
            BasisStatus::Superbasic,
            BasisStatus::Fixed,
        ]
        .iter()
        .copied()
        .cycle()
        .take(14)
        .collect();
        // 6 template rows (uniform) then 2 distinct cut rows.
        let row_status: Vec<BasisStatus> = vec![
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Superbasic,
            BasisStatus::Basic,
            BasisStatus::Zero,
        ];
        let original = CapturedBasis {
            basis: Basis {
                col_status: col_status.clone(),
                row_status: row_status.clone(),
            },
            base_row_count: 6,
            cut_row_slots: vec![7_u32, 42],
            node_id: NodeId(0),
        };

        let mut i32_buf: Vec<i32> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf);

        let mut i32_cursor = 0_usize;
        let recovered = CapturedBasis::try_from_broadcast_payload(0, &i32_buf, &mut i32_cursor)
            .expect("round-trip must not fail")
            .expect("sentinel is 1; must return Some");

        // The wider column block (incl. the anticipated-ring columns) round-trips
        // byte-for-byte.
        assert_eq!(
            recovered.basis.col_status, col_status,
            "relocated-layout col_status must round-trip bit-identically"
        );
        assert_eq!(
            recovered.basis.row_status, row_status,
            "template+cut row_status must round-trip bit-identically"
        );
        assert_eq!(recovered.base_row_count, 6);
        assert_eq!(
            recovered.cut_row_slots,
            vec![7_u32, 42],
            "cut_row_slots (the slot-identity key) must round-trip bit-identically"
        );
        assert_eq!(i32_cursor, i32_buf.len(), "i32_cursor must be at end");
    }

    // ---------------------------------------------------------------------------
    // ProfiledSolver integration
    // ---------------------------------------------------------------------------

    /// A freshly constructed workspace must expose a solver whose
    /// `current_profile()` equals `HighsProfile::default()`.
    ///
    /// Confirms that `ProfiledSolver::new` wraps the inner solver without
    /// issuing any FFI calls and that `WorkspacePool::new` correctly initialises
    /// every workspace.
    ///
    /// Gated to `feature = "highs"` because it embeds the backend-specific
    /// default-profile value `HighsProfile::default()`; a CLP analog is a
    /// separate assertion.
    #[cfg(all(test, feature = "highs"))]
    #[test]
    fn workspace_solver_initialised_with_default_profile() {
        use novomodelo_solver::HighsProfile;

        let ctx = TrainingContextFixture::new(state_layout(0, 0));
        let pool = WorkspacePool::new(
            0,
            2,
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(
                ws.solver.current_profile(),
                &HighsProfile::default(),
                "solver.current_profile() must equal HighsProfile::default() after construction"
            );
        }

        // Also verify SolverWorkspace::new directly.
        let ws = SolverWorkspace::new(
            0,
            0,
            MockSolver,
            PatchBuffer::new(ctx.training_ctx().state, &[], &[], &[]),
            &ctx.training_ctx(),
            &empty_stage_ctx(),
            WorkspaceSizing::default(),
        );
        assert_eq!(
            ws.solver.current_profile(),
            &HighsProfile::default(),
            "SolverWorkspace::new must initialise solver with default profile"
        );
    }
}
