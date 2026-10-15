//! Per-stage cut pool for the Future Cost Function (FCF).
//!
//! Each stage owns one [`CutPool`]. Cuts occupy pre-allocated slots with a
//! deterministic slot-assignment formula so results are bit-for-bit identical
//! regardless of execution timing or ordering (the declaration-order hard rule).
//!
//! ## Slot assignment
//!
//! ```text
//! slot = slot_base + iteration * visit_stride + forward_pass_index
//! ```
//!
//! ## Activity tracking
//!
//! The pool is append-only: deactivated cuts are retained at their stable slot
//! (never removed), so the slot layout stays deterministic. Inactive cuts are
//! excluded from LP construction and [`evaluate_at_state`]. [`set_active`] is the
//! canonical toggle that keeps [`active_count`] consistent; [`deactivate`] wraps
//! `set_active(slot, false)`. [`populated`] is the slot bound — the LP-row metric
//! for append-only LP tracking.
//!
//! [`evaluate_at_state`]: CutPool::evaluate_at_state
//! [`set_active`]: CutPool::set_active
//! [`deactivate`]: CutPool::deactivate
//! [`active_count`]: CutPool::active_count
//! [`populated`]: CutPool::populated
//!
//! ## Example
//!
//! ```rust
//! use novomodelo_sddp::cut::pool::CutPool;
//! use novomodelo_sddp::setup::NodeId;
//!
//! // 100-slot pool, 9-dimensional state, a per-pool visit stride of 10,
//! // slot base 0.
//! let mut pool = CutPool::new(100, 9, 10, 0);
//! assert_eq!(pool.active_count(), 0);
//!
//! let coeffs = vec![1.0; 9];
//! pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);
//! assert_eq!(pool.active_count(), 1);
//! assert_eq!(pool.populated(), 1);
//! ```

use std::num::NonZeroU64;

use super::ownership::Ownership;
use super::trial_states::TrialStates;
use crate::SddpError;
use crate::SddpError::Validation;
use crate::cut_selection::CutActivityUpdates;
use crate::cut_selection::CutMetadata;
use crate::lp::indexer::CutStateProjection;
use crate::setup::NodeId;

use novomodelo_io::OwnedPolicyCutRecord;

/// Where the cut in a pool slot came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotOrigin {
    /// No cut was ever written to the slot.
    Vacant,
    /// A cut loaded before the run started.
    Prefix,
    /// A cut the run generated.
    Generated,
}

/// The static facts of a pool slot, fixed when the slot is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRole {
    /// Where the slot's cut came from.
    pub origin: SlotOrigin,
    /// Whether the pool stored trial states when the slot was written; a loaded slot keeps the
    /// flag its record carries.
    pub scored: bool,
}

/// Which cuts of a pool a solve loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CutSet {
    /// The cuts whose activity flag is set.
    Active,
    /// Every cut a run generated, flag aside; every loaded cut that carries a
    /// trial state, flag aside; every other loaded cut whose flag is set.
    All,
}

const VACANT_ROLE: SlotRole = SlotRole {
    origin: SlotOrigin::Vacant,
    scored: false,
};

fn unwritten_metadata() -> CutMetadata {
    CutMetadata {
        iteration_generated: 0,
        forward_pass_index: 0,
        node: NodeId(0),
        active_count: 0,
        last_active_iter: 0,
    }
}

/// Pre-allocated per-stage cut pool for the Future Cost Function (FCF).
///
/// Cut storage is allocated at construction time — no heap allocation on the
/// training-loop hot path; the trial-state rows are reserved by
/// [`enable_trial_states`](CutPool::enable_trial_states) or
/// [`enable_trial_states_within`](CutPool::enable_trial_states_within). Slots
/// are addressed by a deterministic formula of the iteration counter and
/// forward-pass index.
#[derive(Debug, Clone)]
pub struct CutPool {
    /// Flat coefficient storage. Slot `i` occupies
    /// `i * state_dimension .. (i + 1) * state_dimension`.
    coefficients: Vec<f64>,

    /// Per-slot intercept values.
    intercepts: Vec<f64>,

    /// Per-slot cut-selection bookkeeping.
    metadata: Vec<CutMetadata>,

    /// Per-slot role, parallel to `metadata` and written only with the slot.
    roles: Vec<SlotRole>,

    /// The trial state of each scored cut, by slot; `None` while the pool
    /// stores no states.
    trial_states: Option<TrialStates>,

    /// The per-state ownership record, allocated with the trial states.
    ownership: Option<Ownership>,

    /// Iterations whose trial states are kept; `None` keeps every state.
    lookback: Option<NonZeroU64>,

    /// Per-slot activity flags. `false` excludes the cut from LP construction and
    /// evaluation; the slot is retained so the layout stays deterministic.
    active: Vec<bool>,

    /// One past the highest written slot; bounds iteration so trailing
    /// unwritten slots are skipped.
    populated_count: usize,

    /// Total number of pre-allocated slots. Fixed after construction.
    pub capacity: usize,

    /// Length of each coefficient vector. Fixed after construction.
    pub state_dimension: usize,

    /// Per-pool cut-receipt stride: how many fresh slots this pool receives
    /// per iteration, a factor in the slot formula. Exact where the graph
    /// makes visits statically known (a chain node's is `forward_passes`),
    /// or an expected-value floor otherwise — see
    /// [`NodeGraph::pool_cut_stride`](crate::setup::node_graph::NodeGraph::pool_cut_stride).
    /// Fixed after construction.
    pub visit_stride: u32,

    /// Slot of the first training cut: the offset of the slot formula, not a
    /// count of loaded cuts ([`prefix_len`](CutPool::prefix_len)). Fixed after
    /// construction.
    pub slot_base: u32,

    /// Number of `Prefix` slots, written by the loading constructors.
    prefix_len: usize,

    /// Iteration that maps to the first training slot (`slot_base`). The
    /// slot formula subtracts it so 1-based iterations pack densely. Default 0 is
    /// the legacy layout that leaves the block `[slot_base, +visit_stride)`
    /// unused; production sets 1 via [`set_iteration_base`], on a resumed run
    /// too, so an iteration's slots do not depend on where the run started.
    /// Both are correct; dense is tighter.
    ///
    /// [`set_iteration_base`]: CutPool::set_iteration_base
    pub iteration_base: u64,

    /// Active-cut count, maintained incrementally so [`active_count`] is O(1).
    ///
    /// [`active_count`]: CutPool::active_count
    cached_active_count: usize,

    /// Cuts ever inserted (including warm-start cuts). Unlike
    /// [`populated`](CutPool::populated) — a slot high-water mark that includes
    /// reserved-but-unwritten leading slots — this counts only real insertions:
    /// the true policy-row count.
    pub generated_count: usize,

    /// Scratch buffer for [`enforce_budget`] candidate collection, reused across
    /// calls to avoid per-call allocation.
    ///
    /// [`enforce_budget`]: CutPool::enforce_budget
    pub(crate) candidates_buf: Vec<u32>,
}

impl CutPool {
    /// Create a `CutPool` with all slots pre-allocated and zero / inactive.
    /// `slot_base` reserves that many vacant leading slots; none is a loaded cut.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    ///
    /// let pool = CutPool::new(50, 4, 5, 0);
    /// assert_eq!(pool.capacity, 50);
    /// assert_eq!(pool.state_dimension, 4);
    /// assert_eq!(pool.active_count(), 0);
    /// assert_eq!(pool.populated(), 0);
    /// ```
    #[must_use]
    pub fn new(capacity: usize, state_dimension: usize, visit_stride: u32, slot_base: u32) -> Self {
        Self {
            coefficients: vec![0.0; capacity * state_dimension],
            intercepts: vec![0.0; capacity],
            metadata: vec![unwritten_metadata(); capacity],
            roles: vec![VACANT_ROLE; capacity],
            trial_states: None,
            ownership: None,
            lookback: None,
            active: vec![false; capacity],
            populated_count: 0,
            capacity,
            state_dimension,
            visit_stride,
            slot_base,
            prefix_len: 0,
            iteration_base: 0,
            cached_active_count: 0,
            generated_count: 0,
            candidates_buf: Vec::new(),
        }
    }

    /// Compute the deterministic slot index for a cut.
    ///
    /// ```text
    /// slot = slot_base
    ///      + (iteration - iteration_base) * visit_stride
    ///      + forward_pass_index
    /// ```
    #[inline]
    fn slot_index(&self, iteration: u64, forward_pass_index: u32) -> usize {
        debug_assert!(
            iteration >= self.iteration_base,
            "slot_index: iteration {iteration} < iteration_base {}",
            self.iteration_base
        );
        // Cast cannot truncate: SDDP runs only on 64-bit targets and capacity < usize::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let iter_usize = (iteration - self.iteration_base) as usize;
        self.slot_base as usize
            + iter_usize * self.visit_stride as usize
            + forward_pass_index as usize
    }

    /// The first slot whose trial state is inside a window of `lookback`
    /// iterations at a selection run of `iteration`: slot 0 while this run has
    /// completed fewer than `lookback` iterations, a loaded state included, and
    /// the block of iteration `iteration - lookback + 1` after.
    fn window_start_slot(&self, lookback: NonZeroU64, iteration: u64) -> usize {
        debug_assert!(
            iteration >= self.iteration_base,
            "window_start_slot: iteration {iteration} < iteration_base {}",
            self.iteration_base
        );
        let completed = iteration - self.iteration_base + 1;
        match completed.checked_sub(lookback.get()) {
            None => 0,
            Some(left) => self.slot_index(self.iteration_base + left, 0),
        }
    }

    /// Set the iteration that maps to the first training slot.
    ///
    /// Pass 1 for dense packing from slot `slot_base`, also on a resumed run:
    /// the resumed cuts sit where the recorded run put them, so the loop's cuts
    /// continue at the slots the uninterrupted run gives them. Default 0 leaves
    /// the block `[slot_base, +visit_stride)` unused.
    ///
    /// Call before the first [`add_cut`](CutPool::add_cut) of a run. Changing the
    /// base while the pool still holds *active* training cuts is caught by
    /// `add_cut`'s no-overwrite guard; re-setting when all cuts are inactive
    /// safely reuses slots.
    pub fn set_iteration_base(&mut self, iteration_base: u64) {
        self.iteration_base = iteration_base;
    }

    /// Insert a Benders cut at its deterministic `slot_index`, marked active.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the computed slot is >= `capacity` or if
    /// `coefficients.len() != state_dimension`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(20, 3, 5, 0);
    /// pool.add_cut(NodeId(0), 1, 2, 10.0, &[1.0, 2.0, 3.0]);
    /// // slot = 0 + 1*5 + 2 = 7
    /// assert!(pool.is_active(7));
    /// assert_eq!(pool.intercept(7), 10.0);
    /// ```
    ///
    /// `node_id` is the declared id of the generating node
    /// (`NodeGraph::node_ids`), recorded in the slot's [`CutMetadata::node`] and
    /// later carried onto the MPI cut wire. It is provenance only — it never
    /// affects the slot the cut lands in (the append-only, slot-identity
    /// contract is independent of `node_id`).
    ///
    /// [`CutMetadata::node`]: crate::cut_selection::CutMetadata::node
    pub fn add_cut(
        &mut self,
        node_id: NodeId,
        iteration: u64,
        forward_pass_index: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        let slot = self.slot_index(iteration, forward_pass_index);

        debug_assert!(
            slot < self.capacity,
            "cut slot {slot} is out of bounds (capacity = {})",
            self.capacity
        );
        debug_assert!(
            coefficients.len() == self.state_dimension,
            "coefficients length {} != state_dimension {}",
            coefficients.len(),
            self.state_dimension
        );

        self.intercepts[slot] = intercept;
        let start = slot * self.state_dimension;
        self.coefficients[start..start + self.state_dimension].copy_from_slice(coefficients);
        debug_assert!(
            !self.active[slot],
            "add_cut: slot {slot} is already active (double-insert)"
        );
        self.active[slot] = true;
        self.cached_active_count += 1;
        self.metadata[slot] = CutMetadata {
            iteration_generated: iteration,
            forward_pass_index,
            node: node_id,
            active_count: 0,
            last_active_iter: iteration,
        };
        self.roles[slot] = SlotRole {
            origin: SlotOrigin::Generated,
            scored: self.stores_trial_states(),
        };

        if slot >= self.populated_count {
            self.populated_count = slot + 1;
        }
        self.generated_count += 1;
    }

    /// Iterate over active cuts as `(slot_index, intercept, coefficient_slice)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 3.0, &[1.0, 2.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 7.0, &[3.0, 4.0]);
    ///
    /// let active: Vec<_> = pool.active_cuts().collect();
    /// assert_eq!(active.len(), 2);
    /// ```
    pub fn active_cuts(&self) -> impl Iterator<Item = (usize, f64, &[f64])> {
        let mut remaining = self.cached_active_count;
        self.active[..self.populated_count]
            .iter()
            .enumerate()
            .scan((), move |(), (i, &is_active)| {
                if remaining == 0 {
                    return None;
                }
                if is_active {
                    remaining -= 1;
                    let start = i * self.state_dimension;
                    Some(Some((
                        i,
                        self.intercepts[i],
                        &self.coefficients[start..start + self.state_dimension],
                    )))
                } else {
                    Some(None)
                }
            })
            .flatten()
    }

    /// Iterate over the cuts of `set` as `(slot_index, intercept, coefficient_slice)`,
    /// in ascending slot order. [`CutSet::Active`] yields what
    /// [`active_cuts`](CutPool::active_cuts) yields.
    pub fn cuts_in(&self, set: CutSet) -> impl Iterator<Item = (usize, f64, &[f64])> {
        (0..self.populated_count)
            .filter(move |&slot| self.is_in(slot, set))
            .map(move |slot| {
                let start = slot * self.state_dimension;
                (
                    slot,
                    self.intercepts[slot],
                    &self.coefficients[start..start + self.state_dimension],
                )
            })
    }

    /// Count the cuts of `set`.
    #[must_use]
    pub fn count_in(&self, set: CutSet) -> usize {
        match set {
            CutSet::Active => self.active_count(),
            CutSet::All => self.cuts_in(set).count(),
        }
    }

    fn is_in(&self, slot: usize, set: CutSet) -> bool {
        match set {
            CutSet::Active => self.active[slot],
            CutSet::All => {
                let role = self.roles[slot];
                match role.origin {
                    SlotOrigin::Vacant => false,
                    SlotOrigin::Generated => true,
                    SlotOrigin::Prefix => role.scored || self.active[slot],
                }
            }
        }
    }

    /// Iterate over active cuts whose `iteration_generated == current_iteration`,
    /// in insertion order (declaration-order invariance).
    ///
    /// A loaded cut ([`is_prefix_slot`](CutPool::is_prefix_slot)) is never a
    /// delta cut: a restarted iteration counter can equal its generating
    /// iteration.
    pub(crate) fn active_delta_cuts(
        &self,
        current_iteration: u64,
    ) -> impl Iterator<Item = (usize, f64, &[f64])> {
        self.active_cuts().filter(move |&(slot, _, _)| {
            !self.is_prefix_slot(slot)
                && self.metadata[slot].iteration_generated == current_iteration
        })
    }

    /// Count the active cuts (cached, O(1)).
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// assert_eq!(pool.active_count(), 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn active_count(&self) -> usize {
        debug_assert_eq!(
            self.cached_active_count,
            self.active[..self.populated_count]
                .iter()
                .filter(|&&a| a)
                .count(),
            "cached active_count disagrees with the active flags"
        );
        self.cached_active_count
    }

    /// Number of cuts loaded before the run, the number of `Prefix` slots.
    #[must_use]
    #[inline]
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Whether any cut was loaded before the run; on the terminal stage's pool,
    /// the boundary flag.
    #[must_use]
    #[inline]
    pub fn has_prefix_cuts(&self) -> bool {
        self.prefix_len > 0
    }

    /// Whether `slot` holds a cut loaded before the run started.
    #[must_use]
    #[inline]
    pub fn is_prefix_slot(&self, slot: usize) -> bool {
        self.roles[slot].origin == SlotOrigin::Prefix
    }

    /// Read a single slot's role.
    #[must_use]
    #[inline]
    pub fn role(&self, slot: usize) -> SlotRole {
        self.roles[slot]
    }

    /// Reserve a trial-state row for each of the pool's `capacity` slots, so
    /// that every cut written afterwards is scored. Idempotent.
    ///
    /// Memory: `capacity * state_dimension * 8` bytes reserved, of which
    /// `stored_trial_states() * state_dimension * 8` hold a state, and one `f64` and
    /// three `u32` per slot for the lookup and the ownership record.
    pub fn enable_trial_states(&mut self) {
        let (capacity, dimension) = (self.capacity, self.state_dimension);
        self.trial_states
            .get_or_insert_with(|| TrialStates::new(capacity, dimension));
        self.ownership
            .get_or_insert_with(|| Ownership::new(capacity));
    }

    /// Like [`enable_trial_states`](CutPool::enable_trial_states), for a window of
    /// `lookback_iterations`: [`expire_trial_states`](CutPool::expire_trial_states)
    /// drops the older states and their rows are reused. A pool that holds more
    /// states than rows extends the store.
    ///
    /// Memory: `min((lookback_iterations + check_frequency) * visit_stride,
    /// capacity) * state_dimension * 8` bytes reserved, independent of the run
    /// length, plus 20 bytes per slot for the lookup and the ownership record.
    pub fn enable_trial_states_within(
        &mut self,
        lookback_iterations: NonZeroU64,
        check_frequency: u64,
    ) {
        let (capacity, dimension) = (self.capacity, self.state_dimension);
        let iterations = lookback_iterations.get().saturating_add(check_frequency);
        let rows = usize::try_from(iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(self.visit_stride as usize)
            .min(capacity);
        self.trial_states
            .get_or_insert_with(|| TrialStates::bounded(capacity, rows, dimension));
        self.ownership
            .get_or_insert_with(|| Ownership::new(capacity));
        self.lookback = Some(lookback_iterations);
    }

    /// Whether the pool stores the trial state of each cut written.
    #[must_use]
    #[inline]
    pub fn stores_trial_states(&self) -> bool {
        self.trial_states.is_some()
    }

    /// The window of iterations whose trial states the pool keeps; `None`
    /// keeps every state, or the pool stores none.
    #[must_use]
    pub(crate) fn trial_state_window(&self) -> Option<NonZeroU64> {
        self.lookback
    }

    /// Store the state the cut at `(iteration, forward_pass_index)` was computed
    /// at: `x_hat`, a full state vector, gathered through `projection`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the pool stores no trial states, the slot is not a scored
    /// generated slot, or the slot already holds a state.
    pub(crate) fn store_trial_state(
        &mut self,
        iteration: u64,
        forward_pass_index: u32,
        x_hat: &[f64],
        projection: &CutStateProjection,
    ) {
        let slot = self.slot_index(iteration, forward_pass_index);
        debug_assert!(
            self.stores_trial_states(),
            "store_trial_state: the pool stores no trial states"
        );
        debug_assert_eq!(
            self.roles[slot],
            SlotRole {
                origin: SlotOrigin::Generated,
                scored: true,
            },
            "store_trial_state: slot {slot} is not a scored generated slot"
        );
        if let Some(trial_states) = self.trial_states.as_mut() {
            projection.gather_trial_state(x_hat, trial_states.claim(slot));
        }
    }

    /// Store `row`, a state already projected to `state_dimension` values, at a
    /// scored `slot` of either origin: the only production path that stores a
    /// state for a loaded slot.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the pool stores no trial states, the slot is not scored, the
    /// slot already holds a state, or `row.len() != state_dimension`.
    pub(crate) fn load_trial_state(&mut self, slot: usize, row: &[f64]) {
        debug_assert!(
            self.stores_trial_states(),
            "load_trial_state: the pool stores no trial states"
        );
        debug_assert!(
            self.roles[slot].scored,
            "load_trial_state: slot {slot} is not scored"
        );
        debug_assert_eq!(
            row.len(),
            self.state_dimension,
            "load_trial_state: row length differs from the state dimension"
        );
        if let Some(trial_states) = self.trial_states.as_mut() {
            trial_states.claim(slot).copy_from_slice(row);
        }
    }

    /// The trial state stored for `slot`; `None` for a slot with none.
    #[must_use]
    pub fn trial_state(&self, slot: usize) -> Option<&[f64]> {
        self.trial_states.as_ref()?.get(slot)
    }

    /// Number of slots that hold a trial state.
    #[must_use]
    pub fn stored_trial_states(&self) -> usize {
        self.trial_states
            .as_ref()
            .map_or(0, TrialStates::stored_count)
    }

    #[cfg(test)]
    pub(crate) fn reserved_trial_rows(&self) -> usize {
        self.trial_states
            .as_ref()
            .map_or(0, TrialStates::reserved_rows)
    }

    /// Drop the trial states that left the window at a selection run of
    /// `current_iteration`: each leaves the ownership record, its owner loses
    /// it, and its row returns to the store. The cuts, roles and activity flags
    /// are untouched, and a scored cut without a state still competes at the
    /// remaining ones. A no-op for a pool without a window.
    pub fn expire_trial_states(&mut self, current_iteration: u64) {
        let Some(lookback) = self.lookback else {
            return;
        };
        let until = self
            .window_start_slot(lookback, current_iteration)
            .min(self.populated_count);
        if let (Some(states), Some(ownership)) =
            (self.trial_states.as_mut(), self.ownership.as_mut())
        {
            ownership.expire(states, until);
        }
    }

    /// Bring the ownership record up to date with the slots written since the
    /// last call. A no-op for a pool that stores no trial states.
    pub fn update_ownership(&mut self) {
        if let Some(mut ownership) = self.ownership.take() {
            ownership.update(self);
            self.ownership = Some(ownership);
        }
    }

    /// Recompute the ownership record from every slot, role and stored state.
    ///
    /// Cost: about the pool's cut count squared times its dimension in
    /// multiply-adds, on every rank. A no-op for a pool that stores no trial
    /// states.
    pub fn rebuild_ownership(&mut self) {
        if let Some(ownership) = self.ownership.as_mut() {
            ownership.reset();
        }
        self.update_ownership();
    }

    /// The slot owning the state stored at `slot`; `None` when no state is
    /// stored there or the record is not up to date for it.
    #[must_use]
    pub fn state_owner(&self, slot: usize) -> Option<usize> {
        self.ownership.as_ref()?.owner_of(slot)
    }

    /// The owner's value at the state stored at `slot`; `None` as for
    /// [`state_owner`](CutPool::state_owner).
    #[must_use]
    pub fn state_best_value(&self, slot: usize) -> Option<f64> {
        self.ownership.as_ref()?.best_of(slot)
    }

    /// Number of stored states the cut at `slot` owns.
    #[must_use]
    pub fn owned_states(&self, slot: usize) -> usize {
        self.ownership
            .as_ref()
            .map_or(0, |ownership| ownership.owned_by(slot))
    }

    /// The activity changes the ownership record implies: an active scored slot
    /// that owns no stored state is a deactivation, an inactive scored slot that
    /// owns one is a reactivation, and an unscored or vacant slot is in neither
    /// list. Valid after [`update_ownership`](CutPool::update_ownership); the
    /// pool is not modified.
    #[must_use]
    pub fn ownership_activity_updates(&self) -> CutActivityUpdates {
        let mut updates = Vec::new();
        let mut reactivations = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        for slot in (0..self.populated_count).filter(|&slot| self.roles[slot].scored) {
            match (self.active[slot], self.owned_states(slot) > 0) {
                (true, false) => updates.push(slot as u32),
                (false, true) => reactivations.push(slot as u32),
                (true, true) | (false, false) => {}
            }
        }
        CutActivityUpdates {
            updates,
            reactivations,
        }
    }

    /// Return the slot bound — one past the highest written slot, independent of
    /// activity state. A slot below it can be vacant, so it bounds slots and does
    /// not count cuts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// assert_eq!(pool.populated(), 2);
    ///
    /// // Deactivation does not change the LP-row count.
    /// pool.deactivate(&[0]);
    /// assert_eq!(pool.populated(), 2);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn populated(&self) -> usize {
        self.populated_count
    }

    /// Read a single slot's activity flag.
    #[must_use]
    #[inline]
    pub fn is_active(&self, slot: usize) -> bool {
        self.active[slot]
    }

    /// Read a single slot's cut-selection metadata.
    #[must_use]
    #[inline]
    pub fn metadata(&self, slot: usize) -> &CutMetadata {
        &self.metadata[slot]
    }

    /// Read a single slot's intercept.
    #[must_use]
    #[inline]
    pub fn intercept(&self, slot: usize) -> f64 {
        self.intercepts[slot]
    }

    /// Read a single slot's coefficient row (`state_dimension` elements).
    #[must_use]
    #[inline]
    pub fn coefficient_row(&self, slot: usize) -> &[f64] {
        let start = slot * self.state_dimension;
        &self.coefficients[start..start + self.state_dimension]
    }

    /// Read every populated slot's coefficients as one flat, row-major slice
    /// (`populated() * state_dimension` elements) — the batched-GEMM
    /// counterpart to [`coefficient_row`](CutPool::coefficient_row).
    #[must_use]
    #[inline]
    pub fn coefficients_prefix(&self) -> &[f64] {
        &self.coefficients[..self.populated_count * self.state_dimension]
    }

    /// Read every populated slot's intercepts as one flat slice
    /// (`populated()` elements) — the batched counterpart to
    /// [`intercept`](CutPool::intercept).
    #[must_use]
    #[inline]
    pub fn intercepts_prefix(&self) -> &[f64] {
        &self.intercepts[..self.populated_count]
    }

    /// Record that a cut was binding `increment` more times, most recently at
    /// `iteration`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= populated_count`.
    #[inline]
    pub fn record_binding(&mut self, slot: usize, increment: u64, iteration: u64) {
        debug_assert!(
            slot < self.populated_count,
            "record_binding: slot {slot} out of populated range"
        );
        self.metadata[slot].active_count += increment;
        self.metadata[slot].last_active_iter = iteration;
    }

    /// Zero `active_count` and `last_active_iter` of the loaded prefix, so they
    /// count this run from 0. `iteration_generated` keeps the record's
    /// iteration, which the checkpoint writer exports.
    pub(crate) fn restart_activity_clock(&mut self) {
        for (meta, role) in self.metadata.iter_mut().zip(&self.roles) {
            if role.origin == SlotOrigin::Prefix {
                meta.active_count = 0;
                meta.last_active_iter = 0;
            }
        }
    }

    /// Test-only: overwrite a written slot's role, keeping `prefix_len` in step.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= populated_count` or the role is a scored vacant slot.
    #[cfg(test)]
    pub(crate) fn set_role_for_test(&mut self, slot: usize, role: SlotRole) {
        assert!(
            slot < self.populated_count,
            "set_role_for_test: slot {slot} out of populated range"
        );
        assert!(
            role.origin != SlotOrigin::Vacant || !role.scored,
            "set_role_for_test: a vacant slot is not scored"
        );
        self.prefix_len -= usize::from(self.roles[slot].origin == SlotOrigin::Prefix);
        self.prefix_len += usize::from(role.origin == SlotOrigin::Prefix);
        self.roles[slot] = role;
    }

    /// Test-only: store `state` at a scored slot, bypassing the projection.
    ///
    /// # Panics
    ///
    /// Panics if the pool stores no trial states, `slot` is not a scored written
    /// slot, or it already holds a state.
    #[cfg(any(test, feature = "test-support"))]
    #[allow(clippy::expect_used)]
    pub fn set_trial_state_for_test(&mut self, slot: usize, state: &[f64]) {
        assert!(
            slot < self.populated_count && self.roles[slot].scored,
            "set_trial_state_for_test: slot {slot} is not a scored written slot"
        );
        self.trial_states
            .as_mut()
            .expect("set_trial_state_for_test: the pool stores no trial states")
            .claim(slot)
            .copy_from_slice(state);
    }

    /// Test-only: overwrite one slot's metadata directly, bypassing
    /// `add_cut`'s iteration/slot coupling.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= populated_count`.
    #[cfg(test)]
    pub(crate) fn set_metadata_for_test(&mut self, slot: usize, metadata: CutMetadata) {
        assert!(
            slot < self.populated_count,
            "set_metadata_for_test: slot {slot} out of populated range"
        );
        self.metadata[slot] = metadata;
    }

    /// Test-only: overwrite one slot's `iteration_generated`, decoupling a
    /// cut's metadata age from the slot `add_cut` placed it at. Every other
    /// call site fixtures test scenarios (cut-selection windows, DCS
    /// candidate age) around this one field.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= populated_count`.
    #[cfg(test)]
    pub(crate) fn set_iteration_generated_for_test(
        &mut self,
        slot: usize,
        iteration_generated: u64,
    ) {
        assert!(
            slot < self.populated_count,
            "set_iteration_generated_for_test: slot {slot} out of populated range"
        );
        self.metadata[slot].iteration_generated = iteration_generated;
    }

    /// Cuts ever inserted, including warm-start cuts.
    #[must_use]
    #[inline]
    pub fn generated(&self) -> usize {
        self.generated_count
    }

    /// Deactivate the cuts at the given slot indices via [`set_active`].
    ///
    /// [`set_active`]: CutPool::set_active
    ///
    /// # Idempotency
    ///
    /// An already-inactive index is a no-op (no second decrement → no underflow),
    /// so a duplicate index is safe.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.deactivate(&[0]);
    /// assert_eq!(pool.active_count(), 1);
    /// assert!(!pool.is_active(0));
    /// assert!(pool.is_active(1));
    /// ```
    pub fn deactivate(&mut self, indices: &[u32]) {
        for &idx in indices {
            self.set_active(idx, false);
        }
    }

    /// Apply a batch of activity changes from a [`CutActivityUpdates`] result.
    ///
    /// Deactivates `updates.updates` and reactivates `updates.reactivations` via
    /// [`set_active`]. Folding both into one call is why the training loop never
    /// silently drops reactivation entries. Idempotency follows [`set_active`].
    ///
    /// [`set_active`]: CutPool::set_active
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::cut_selection::CutActivityUpdates;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
    ///
    /// // Deactivate slot 1, then reactivate it via apply_updates.
    /// pool.deactivate(&[1]);
    /// assert_eq!(pool.active_count(), 2);
    ///
    /// let updates = CutActivityUpdates {
    ///     updates: vec![0],          // deactivate slot 0
    ///     reactivations: vec![1],    // reactivate slot 1
    /// };
    /// pool.apply_updates(&updates);
    /// assert!(!pool.is_active(0));
    /// assert!(pool.is_active(1));
    /// assert!(pool.is_active(2));
    /// assert_eq!(pool.active_count(), 2);
    /// ```
    pub fn apply_updates(&mut self, updates: &CutActivityUpdates) {
        for &slot in &updates.updates {
            self.set_active(slot, false);
        }
        for &slot in &updates.reactivations {
            self.set_active(slot, true);
        }
    }

    /// Toggle a single slot's activity flag, keeping `cached_active_count`
    /// consistent. A no-op when the slot already holds the requested state.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `slot >= populated_count`, or if `active` is set on a vacant slot.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
    ///
    /// pool.set_active(1, false);
    /// assert_eq!(pool.active_count(), 2);
    /// assert!(!pool.is_active(1));
    /// assert_eq!(pool.populated(), 3); // populated count unchanged
    ///
    /// pool.set_active(1, true);
    /// assert_eq!(pool.active_count(), 3);
    /// assert!(pool.is_active(1));
    /// ```
    pub fn set_active(&mut self, slot: u32, active: bool) {
        let i = slot as usize;
        debug_assert!(
            i < self.populated_count,
            "set_active slot {i} out of populated range"
        );
        debug_assert!(
            !active || self.roles[i].origin != SlotOrigin::Vacant,
            "set_active: slot {i} is vacant"
        );
        if self.active[i] == active {
            return;
        }
        self.active[i] = active;
        if active {
            self.cached_active_count += 1;
        } else {
            self.cached_active_count -= 1;
        }
    }

    /// Evaluate the FCF: the max over active cuts of `intercept + coefficients ·
    /// state`, or [`f64::NEG_INFINITY`] when no cut is active.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `state.len() != state_dimension`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
    /// pool.add_cut(NodeId(0), 1, 0,  5.0, &[0.0, 2.0]);
    ///
    /// // max(10 + 1*3 + 0*4, 5 + 0*3 + 2*4) = max(13, 13) = 13
    /// assert_eq!(pool.evaluate_at_state(&[3.0, 4.0]), 13.0);
    ///
    /// // Empty pool returns NEG_INFINITY.
    /// let empty = CutPool::new(10, 2, 1, 0);
    /// assert_eq!(empty.evaluate_at_state(&[1.0, 1.0]), f64::NEG_INFINITY);
    /// ```
    #[must_use]
    pub fn evaluate_at_state(&self, state: &[f64]) -> f64 {
        debug_assert!(
            state.len() == self.state_dimension,
            "state length {} != state_dimension {}",
            state.len(),
            self.state_dimension
        );

        self.active_cuts()
            .map(|(_, intercept, coeffs)| {
                let dot: f64 = coeffs.iter().zip(state).map(|(a, b)| a * b).sum();
                intercept + dot
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Diagnostic: count exact-zero coefficients (`value == 0.0`) across active
    /// cuts into a [`SparsityReport`].
    ///
    /// Allocates per call (one `Vec<usize>`); offline use, not per-iteration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(10, 3, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 2.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 3.0]);
    ///
    /// let report = pool.sparsity_report();
    /// assert_eq!(report.total_coefficients, 6);   // 2 cuts * 3 dims
    /// assert_eq!(report.exact_zero_count, 3);      // (0,1), (1,0), (1,1)
    /// assert!((report.sparsity_fraction - 0.5).abs() < 1e-10);
    /// assert_eq!(report.per_dimension_zeros, vec![1, 2, 0]);
    /// ```
    #[must_use]
    pub fn sparsity_report(&self) -> SparsityReport {
        let mut exact_zero_count = 0usize;
        let mut per_dimension_zeros = vec![0usize; self.state_dimension];

        for (_slot, _intercept, coeffs) in self.active_cuts() {
            for (j, &c) in coeffs.iter().enumerate() {
                if c == 0.0 {
                    exact_zero_count += 1;
                    per_dimension_zeros[j] += 1;
                }
            }
        }

        let total = self.active_count() * self.state_dimension;
        #[allow(clippy::cast_precision_loss)]
        let fraction = if total > 0 {
            exact_zero_count as f64 / total as f64
        } else {
            0.0
        };

        SparsityReport {
            total_coefficients: total,
            exact_zero_count,
            sparsity_fraction: fraction,
            per_dimension_zeros,
        }
    }

    /// Construct a `CutPool` from deserialized cut records for FCF evaluation
    /// during simulation.
    ///
    /// Capacity is exactly `records.len()` with `visit_stride = 0` — this pool
    /// takes no new training cuts. Record `i` sits at slot `i`. A record stored
    /// below `prefix_len` (the recorded pool's protected-prefix length) is `Prefix`
    /// and every other record `Generated`, each with the `scored` flag it was
    /// written with, so [`CutSet::All`] holds what the recorded run's held.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_io::OwnedPolicyCutRecord;
    /// use novomodelo_sddp::cut::pool::CutPool;
    ///
    /// let records = vec![
    ///     OwnedPolicyCutRecord {
    ///         slot_index: 0,
    ///         iteration: 0,
    ///         forward_pass_index: 0,
    ///         intercept: 5.0,
    ///         coefficients: vec![1.0, 2.0],
    ///         is_active: true,
    ///         scored: false,
    ///         active_count: None,
    ///         last_active_iteration: None,
    ///     },
    /// ];
    ///
    /// let pool = CutPool::from_deserialized(2, 1, &records);
    /// assert_eq!(pool.capacity, 1);
    /// assert_eq!(pool.populated(), 1);
    /// assert_eq!(pool.prefix_len(), 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn from_deserialized(
        state_dimension: usize,
        prefix_len: u32,
        records: &[OwnedPolicyCutRecord],
    ) -> Self {
        let capacity = records.len();
        let mut coefficients = Vec::with_capacity(capacity * state_dimension);
        let mut intercepts = Vec::with_capacity(capacity);
        let mut active = Vec::with_capacity(capacity);
        let mut metadata = Vec::with_capacity(capacity);
        let mut roles = Vec::with_capacity(capacity);
        let mut cached_active_count = 0usize;
        let mut prefix_records = 0usize;

        for record in records {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "from_deserialized: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            coefficients.extend_from_slice(&record.coefficients);
            intercepts.push(record.intercept);
            active.push(record.is_active);
            if record.is_active {
                cached_active_count += 1;
            }
            metadata.push(CutMetadata {
                iteration_generated: u64::from(record.iteration),
                forward_pass_index: record.forward_pass_index,
                // Checkpoint records carry no generating-node id; the reload path
                // never re-emits these cuts on the MPI wire, so 0 is inert.
                node: NodeId(0),
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
            });
            let origin = if record.slot_index < prefix_len {
                prefix_records += 1;
                SlotOrigin::Prefix
            } else {
                SlotOrigin::Generated
            };
            roles.push(SlotRole {
                origin,
                scored: record.scored,
            });
        }

        #[allow(clippy::cast_possible_truncation)]
        let pool = Self {
            coefficients,
            intercepts,
            metadata,
            roles,
            trial_states: None,
            ownership: None,
            lookback: None,
            active,
            populated_count: capacity,
            capacity,
            state_dimension,
            visit_stride: 0,
            slot_base: capacity as u32,
            prefix_len: prefix_records,
            iteration_base: 0,
            cached_active_count,
            generated_count: capacity,
            candidates_buf: Vec::new(),
        };
        pool.debug_assert_prefix_len_matches_roles();
        pool
    }

    /// Construct a `CutPool` with warm-start cuts plus capacity for training.
    ///
    /// Loaded cuts occupy the first `records.len()` slots, each `Prefix` with the
    /// `scored` flag it was written with; the remaining `max_iterations *
    /// visit_stride` slots take new training cuts (offset past the warm-start
    /// region by the slot formula).
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_io::OwnedPolicyCutRecord;
    /// use novomodelo_sddp::cut::pool::CutPool;
    ///
    /// let records = vec![
    ///     OwnedPolicyCutRecord {
    ///         slot_index: 0, iteration: 0, forward_pass_index: 0,
    ///         intercept: 5.0, coefficients: vec![1.0, 2.0],
    ///         is_active: true, scored: false, active_count: None,
    ///         last_active_iteration: None,
    ///     },
    /// ];
    /// let pool = CutPool::new_with_warm_start(2, 4, 10, &records);
    /// assert_eq!(pool.prefix_len(), 1);
    /// assert_eq!(pool.capacity, 1 + 10 * 4); // 41
    /// assert_eq!(pool.populated(), 1);
    /// assert_eq!(pool.active_count(), 1);
    /// ```
    #[must_use]
    pub fn new_with_warm_start(
        state_dimension: usize,
        visit_stride: u32,
        max_iterations: u64,
        records: &[OwnedPolicyCutRecord],
    ) -> Self {
        let prefix_len = records.len();
        #[allow(clippy::cast_possible_truncation)]
        let capacity = prefix_len + (max_iterations as usize) * (visit_stride as usize);

        let mut coefficients = vec![0.0_f64; capacity * state_dimension];
        let mut intercepts = vec![0.0; capacity];
        let mut active = vec![false; capacity];
        let mut metadata = vec![unwritten_metadata(); capacity];
        let mut roles = vec![VACANT_ROLE; capacity];
        let mut cached_active_count = 0usize;

        for (i, record) in records.iter().enumerate() {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "new_with_warm_start: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            let start = i * state_dimension;
            coefficients[start..start + state_dimension].copy_from_slice(&record.coefficients);
            intercepts[i] = record.intercept;
            active[i] = record.is_active;
            if record.is_active {
                cached_active_count += 1;
            }
            metadata[i] = CutMetadata {
                iteration_generated: u64::from(record.iteration),
                forward_pass_index: record.forward_pass_index,
                // Prefix slots are never packed for cut sync, so their generating
                // node never reaches the wire; 0 is inert.
                node: NodeId(0),
                active_count: 0,
                last_active_iter: u64::from(record.iteration),
            };
            roles[i] = SlotRole {
                origin: SlotOrigin::Prefix,
                scored: record.scored,
            };
        }

        #[allow(clippy::cast_possible_truncation)]
        let pool = Self {
            coefficients,
            intercepts,
            metadata,
            roles,
            trial_states: None,
            ownership: None,
            lookback: None,
            active,
            populated_count: prefix_len,
            capacity,
            state_dimension,
            visit_stride,
            slot_base: prefix_len as u32,
            prefix_len,
            iteration_base: 0,
            cached_active_count,
            generated_count: prefix_len,
            candidates_buf: Vec::new(),
        };
        pool.debug_assert_prefix_len_matches_roles();
        pool
    }

    /// Construct the pool a checkpoint recorded: each record at its own
    /// `slot_index`, the first `prefix_len` slots `Prefix` and every other
    /// record `Generated`, a record's `scored` flag and binding counters as
    /// written. A record without counters starts at `active_count = 0` and
    /// `last_active_iter = iteration`, or at `last_active_iter = 0` below
    /// `prefix_len`, where [`restart_activity_clock`](CutPool::restart_activity_clock)
    /// left the recorded run's. A slot with no record stays vacant, so a later
    /// cut lands where the recorded run's did.
    ///
    /// `prefix_len` is the number of `Prefix` records the recorded pool held
    /// (the writer stores [`CutPool::prefix_len`] as the pool's
    /// `warm_start_count`) and becomes the slot base. Capacity is the larger of
    /// `stored_capacity` and `prefix_len + max_iterations * visit_stride`, so a
    /// pool grown by [`grow`](CutPool::grow) keeps its slots.
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] naming the slot when two records share it, a
    /// record's slot is not below the capacity, or a slot below `prefix_len`
    /// holds no record.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_io::OwnedPolicyCutRecord;
    /// use novomodelo_sddp::cut::pool::CutPool;
    ///
    /// let record = |slot_index| OwnedPolicyCutRecord {
    ///     slot_index,
    ///     iteration: 1,
    ///     forward_pass_index: 0,
    ///     intercept: 5.0,
    ///     coefficients: vec![1.0, 2.0],
    ///     is_active: true,
    ///     scored: false,
    ///     active_count: None,
    ///     last_active_iteration: None,
    /// };
    /// let pool = CutPool::new_resumed(2, 4, 10, 1, 0, &[record(0), record(5)])?;
    /// assert_eq!(pool.prefix_len(), 1);
    /// assert_eq!(pool.capacity, 1 + 10 * 4);
    /// assert_eq!(pool.populated(), 6);
    /// assert_eq!(pool.generated(), 2);
    /// # Ok::<(), novomodelo_sddp::SddpError>(())
    /// ```
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn new_resumed(
        state_dimension: usize,
        visit_stride: u32,
        max_iterations: u64,
        prefix_len: u32,
        stored_capacity: u32,
        records: &[OwnedPolicyCutRecord],
    ) -> Result<Self, SddpError> {
        let prefix = prefix_len as usize;
        #[allow(clippy::cast_possible_truncation)]
        let capacity = (stored_capacity as usize)
            .max(prefix + (max_iterations as usize) * (visit_stride as usize));

        let mut coefficients = vec![0.0_f64; capacity * state_dimension];
        let mut intercepts = vec![0.0; capacity];
        let mut active = vec![false; capacity];
        let mut metadata = vec![unwritten_metadata(); capacity];
        let mut roles = vec![VACANT_ROLE; capacity];
        let mut cached_active_count = 0usize;
        let mut populated_count = 0usize;

        for record in records {
            debug_assert!(
                record.coefficients.len() == state_dimension,
                "new_resumed: coefficients length {} != state_dimension {}",
                record.coefficients.len(),
                state_dimension
            );
            let slot = record.slot_index as usize;
            if slot >= capacity {
                return Err(Validation(format!(
                    "slot {slot} is not below the capacity {capacity}"
                )));
            }
            if roles[slot].origin != SlotOrigin::Vacant {
                return Err(Validation(format!("slot {slot} holds two records")));
            }
            let start = slot * state_dimension;
            coefficients[start..start + state_dimension].copy_from_slice(&record.coefficients);
            intercepts[slot] = record.intercept;
            active[slot] = record.is_active;
            if record.is_active {
                cached_active_count += 1;
            }
            let (active_count, last_active_iter) =
                match (record.active_count, record.last_active_iteration) {
                    (Some(count), Some(last)) => (count, last),
                    _ if slot < prefix => (0, 0),
                    _ => (0, u64::from(record.iteration)),
                };
            metadata[slot] = CutMetadata {
                iteration_generated: u64::from(record.iteration),
                forward_pass_index: record.forward_pass_index,
                // A reloaded cut was generated in an earlier iteration, so it is
                // never packed for cut sync and its generating node never reaches
                // the wire; 0 is inert.
                node: NodeId(0),
                active_count,
                last_active_iter,
            };
            roles[slot] = SlotRole {
                origin: if slot < prefix {
                    SlotOrigin::Prefix
                } else {
                    SlotOrigin::Generated
                },
                scored: record.scored,
            };
            populated_count = populated_count.max(slot + 1);
        }
        if let Some(slot) = roles[..prefix]
            .iter()
            .position(|role| role.origin == SlotOrigin::Vacant)
        {
            return Err(Validation(format!(
                "slot {slot} is inside the protected prefix of {prefix} but holds no record"
            )));
        }

        let pool = Self {
            coefficients,
            intercepts,
            metadata,
            roles,
            trial_states: None,
            ownership: None,
            lookback: None,
            active,
            populated_count,
            capacity,
            state_dimension,
            visit_stride,
            slot_base: prefix_len,
            prefix_len: prefix,
            iteration_base: 0,
            cached_active_count,
            generated_count: records.len(),
            candidates_buf: Vec::new(),
        };
        pool.debug_assert_prefix_len_matches_roles();
        Ok(pool)
    }

    fn debug_assert_prefix_len_matches_roles(&self) {
        debug_assert_eq!(
            self.prefix_len,
            self.roles
                .iter()
                .filter(|role| role.origin == SlotOrigin::Prefix)
                .count(),
            "prefix_len disagrees with the slot roles"
        );
    }

    /// Grow this pool's capacity in place. `Vec::resize` only appends new
    /// (zeroed/inactive) slots when growing — it never relocates an existing
    /// element — so every populated slot keeps its index, the invariant a
    /// stored basis's slot-identity match depends on (the "Cut pool is
    /// append-only; basis matches by slot identity" contract). The training
    /// session's between-iteration growth hook calls this when an iteration's
    /// visits exceed a pool's free slots: on an enumerated run with two or more
    /// paths, possibly on a sampled branching pool whose `visit_stride` is below
    /// the forward-pass count, and never on a chain.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `new_capacity < self.capacity` — growth-only, never shrinks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::pool::CutPool;
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let mut pool = CutPool::new(2, 2, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);
    /// pool.grow(10);
    /// assert_eq!(pool.capacity, 10);
    /// assert!(pool.is_active(0));
    /// assert_eq!(pool.intercept(0), 5.0);
    /// ```
    pub fn grow(&mut self, new_capacity: usize) {
        debug_assert!(
            new_capacity >= self.capacity,
            "grow: new_capacity {new_capacity} < current capacity {}",
            self.capacity
        );
        self.coefficients
            .resize(new_capacity * self.state_dimension, 0.0);
        self.intercepts.resize(new_capacity, 0.0);
        self.metadata.resize(new_capacity, unwritten_metadata());
        self.roles.resize(new_capacity, VACANT_ROLE);
        if let Some(trial_states) = self.trial_states.as_mut() {
            trial_states.grow(new_capacity);
        }
        if let Some(ownership) = self.ownership.as_mut() {
            ownership.grow(new_capacity);
        }
        self.active.resize(new_capacity, false);
        self.capacity = new_capacity;
    }
}

/// Diagnostic report of exact-zero coefficients across active cuts in a [`CutPool`].
///
/// Counts exact zeros (`value == 0.0`) only; near-zero values are not collapsed.
#[derive(Debug, Clone)]
pub struct SparsityReport {
    /// Total number of coefficients scanned (`active_count * state_dimension`).
    pub total_coefficients: usize,
    /// Number of exact-zero coefficients (`value == 0.0`).
    pub exact_zero_count: usize,
    /// Fraction of exact-zero coefficients (0.0 to 1.0).
    pub sparsity_fraction: f64,
    /// Per-dimension zero count (length = `state_dimension`). Entry `j` is the
    /// number of active cuts where `coefficient[j] == 0.0`.
    pub per_dimension_zeros: Vec<usize>,
}

/// Result of a [`CutPool::enforce_budget`] call.
#[derive(Debug, Clone)]
pub struct BudgetEnforcementResult {
    /// Number of cuts deactivated during this enforcement pass.
    pub evicted_count: u32,
    /// Active cut count before enforcement.
    pub active_before: u32,
    /// Active cut count after enforcement (`active_before - evicted_count`).
    pub active_after: u32,
}

impl CutPool {
    /// Evict the active cuts this run generated down to `budget`, smallest
    /// `(last_active_iter, active_count, slot)` first.
    ///
    /// A loaded cut ([`is_prefix_slot`](CutPool::is_prefix_slot)) neither counts
    /// toward `budget` nor is evicted. A cut generated in `current_iteration`
    /// counts and is never evicted, so the generated cuts can stay above
    /// `budget`. The result's active counts cover the whole pool, loaded cuts
    /// included.
    pub fn enforce_budget(
        &mut self,
        budget: u32,
        current_iteration: u64,
    ) -> BudgetEnforcementResult {
        #[allow(clippy::cast_possible_truncation)]
        let active_before = self.active_count() as u32;
        let budget_usize = budget as usize;

        if self.cached_active_count <= budget_usize {
            return BudgetEnforcementResult {
                evicted_count: 0,
                active_before,
                active_after: active_before,
            };
        }

        self.candidates_buf.clear();
        let mut generated = 0usize;
        #[allow(clippy::cast_possible_truncation)]
        for (slot, &is_active) in self.active[..self.populated_count].iter().enumerate() {
            if is_active && !self.is_prefix_slot(slot) {
                generated += 1;
                if self.metadata[slot].iteration_generated != current_iteration {
                    self.candidates_buf.push(slot as u32);
                }
            }
        }

        if generated <= budget_usize || self.candidates_buf.is_empty() {
            return BudgetEnforcementResult {
                evicted_count: 0,
                active_before,
                active_after: active_before,
            };
        }

        let evict_count = (generated - budget_usize).min(self.candidates_buf.len());

        let mut candidates = std::mem::take(&mut self.candidates_buf);
        candidates.sort_unstable_by_key(|&slot| {
            let meta = &self.metadata[slot as usize];
            (meta.last_active_iter, meta.active_count, slot)
        });
        self.deactivate(&candidates[..evict_count]);
        self.candidates_buf = candidates;

        #[allow(clippy::cast_possible_truncation)]
        let evicted_count = evict_count as u32;
        #[allow(clippy::cast_possible_truncation)]
        let active_after = self.active_count() as u32;

        BudgetEnforcementResult {
            evicted_count,
            active_before,
            active_after,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroU64;

    use super::{CutPool, CutSet, SlotOrigin, SlotRole};
    use crate::cut_selection::{CutActivityUpdates, CutMetadata};
    use crate::lp::indexer::CutStateProjection;
    use crate::setup::NodeId;
    use crate::test_support::state_layout;
    use novomodelo_core::temporal::StageStateConfig;
    use novomodelo_io::test_support::owned_piece_defaults;
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    #[test]
    fn new_creates_pool_with_correct_capacity_and_all_inactive() {
        let pool = CutPool::new(100, 9, 10, 0);
        assert_eq!(pool.capacity, 100);
        assert_eq!(pool.state_dimension, 9);
        assert_eq!(pool.visit_stride, 10);
        assert_eq!(pool.slot_base, 0);
        assert_eq!(pool.populated_count, 0);
        assert_eq!(pool.active_count(), 0);
        assert!(pool.active.iter().all(|&a| !a));
        assert_eq!(pool.coefficients.len(), 100 * 9);
        assert!(pool.coefficients.iter().all(|&v| v == 0.0));
        assert!(pool.intercepts.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn new_zero_capacity_is_valid() {
        let pool = CutPool::new(0, 4, 5, 0);
        assert_eq!(pool.capacity, 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn add_cut_at_slot_zero_stores_intercept_coefficients_and_active_flag() {
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![1.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);

        assert_eq!(pool.active_count(), 1);
        assert!(pool.active[0]);
        assert_eq!(pool.intercepts[0], 5.0);
        assert_eq!(&pool.coefficients[0..9], vec![1.0; 9].as_slice());
        assert_eq!(pool.metadata[0].iteration_generated, 0);
        assert_eq!(pool.metadata[0].forward_pass_index, 0);
        assert_eq!(pool.populated_count, 1);
    }

    #[test]
    fn add_cut_deterministic_slot_formula_no_warmstart() {
        // slot = 0 + iteration * visit_stride + forward_pass_index
        let mut pool = CutPool::new(200, 2, 10, 0);

        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]); // slot = 0
        pool.add_cut(NodeId(0), 0, 3, 2.0, &[3.0, 4.0]); // slot = 3
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[5.0, 6.0]); // slot = 10
        pool.add_cut(NodeId(0), 2, 5, 4.0, &[7.0, 8.0]); // slot = 25

        assert!(pool.active[0]);
        assert_eq!(pool.intercepts[0], 1.0);

        assert!(pool.active[3]);
        assert_eq!(pool.intercepts[3], 2.0);

        assert!(pool.active[10]);
        assert_eq!(pool.intercepts[10], 3.0);

        assert!(pool.active[25]);
        assert_eq!(pool.intercepts[25], 4.0);
    }

    #[test]
    fn add_cut_slot_base_offsets_slot() {
        // slot = 5 + 0*10 + 0 = 5
        let mut pool = CutPool::new(100, 9, 10, 5);
        let coeffs = vec![0.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 42.0, &coeffs);

        assert!(pool.active[5]);
        assert_eq!(pool.intercepts[5], 42.0);
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn add_cut_metadata_initialized_correctly() {
        let mut pool = CutPool::new(50, 3, 5, 0);
        pool.add_cut(NodeId(0), 3, 2, 7.0, &[1.0, 2.0, 3.0]);
        // slot = 0 + 3*5 + 2 = 17
        let meta = &pool.metadata[17];
        assert_eq!(meta.iteration_generated, 3);
        assert_eq!(meta.forward_pass_index, 2);
        assert_eq!(meta.active_count, 0);
        assert_eq!(meta.last_active_iter, 3);
    }

    #[test]
    fn populated_count_tracks_high_water_mark() {
        let mut pool = CutPool::new(50, 1, 5, 0);

        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0 → populated_count = 1
        assert_eq!(pool.populated_count, 1);

        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 5 → populated_count = 6
        assert_eq!(pool.populated_count, 6);

        pool.add_cut(NodeId(0), 0, 2, 3.0, &[3.0]); // slot 2 → no change (2 < 6)
        assert_eq!(pool.populated_count, 6);
    }

    #[test]
    fn set_iteration_base_packs_training_cuts_densely() {
        // base = 1 (start_iteration 0) maps iteration 1 to slot slot_base
        // (0 here), so 1-based iterations leave no reserved leading block.
        let mut pool = CutPool::new(30, 1, 3, 0);
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 1, 2.0, &[1.0]); // slot 1
        pool.add_cut(NodeId(0), 1, 2, 3.0, &[1.0]); // slot 2
        pool.add_cut(NodeId(0), 2, 0, 4.0, &[1.0]); // slot 3
        assert!(pool.active[0] && pool.active[1] && pool.active[2] && pool.active[3]);
        assert_eq!(pool.populated_count, 4, "dense packing leaves no gap");
        assert_eq!(pool.generated_count, 4);
        // metadata keeps the TRUE iteration, not the re-based slot iteration.
        assert_eq!(pool.metadata[0].iteration_generated, 1);
        assert_eq!(pool.metadata[3].iteration_generated, 2);
    }

    #[test]
    fn active_cuts_returns_only_active_cuts() {
        let mut pool = CutPool::new(20, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[3.0, 4.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[5.0, 6.0]); // slot 2

        pool.deactivate(&[1]);

        let active: Vec<_> = pool.active_cuts().collect();
        assert_eq!(active.len(), 2);

        let slots: Vec<usize> = active.iter().map(|(s, _, _)| *s).collect();
        assert!(slots.contains(&0));
        assert!(slots.contains(&2));
        assert!(!slots.contains(&1));
    }

    #[test]
    fn active_cuts_empty_pool_returns_empty_iterator() {
        let pool = CutPool::new(10, 3, 5, 0);
        let active: Vec<_> = pool.active_cuts().collect();
        assert!(active.is_empty());
    }

    #[test]
    fn active_count_is_correct_after_add_and_deactivate() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        assert_eq!(pool.active_count(), 3);
        pool.deactivate(&[1]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_sets_flags_correctly() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[1]);

        assert!(pool.active[0]);
        assert!(!pool.active[1]);
        assert!(pool.active[2]);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn deactivate_multiple_indices() {
        let mut pool = CutPool::new(20, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2

        pool.deactivate(&[0, 2]);

        assert!(!pool.active[0]);
        assert!(pool.active[1]);
        assert!(!pool.active[2]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn deactivate_empty_slice_is_noop() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.deactivate(&[]);
        assert_eq!(pool.active_count(), 1);
    }

    /// `deactivate` silently skips already-inactive indices.
    ///
    /// In debug builds the `debug_assert!` would fire on a duplicate
    /// input, so the test is gated to release-only. When a release
    /// build receives `&[0, 0, 0]`, the active count drops by exactly
    /// 1 (not 3) and `active[0]` becomes `false`.
    #[test]
    #[cfg(not(debug_assertions))]
    fn deactivate_duplicate_index_is_silently_skipped() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        assert_eq!(pool.active_count(), 2);
        pool.deactivate(&[0, 0, 0]);
        assert_eq!(
            pool.active_count(),
            1,
            "duplicate indices must not double-decrement"
        );
        assert!(!pool.active[0]);
        assert!(pool.active[1]);
    }

    #[test]
    fn evaluate_at_state_returns_max_cut_value() {
        // cuts: (intercept=10, coeffs=[1, 0]) and (intercept=5, coeffs=[0, 2])
        // state = [3, 4]
        // cut 0: 10 + 1*3 + 0*4 = 13
        // cut 1:  5 + 0*3 + 2*4 = 13
        // max = 13
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0, 2.0]);

        let result = pool.evaluate_at_state(&[3.0, 4.0]);
        assert_eq!(result, 13.0);
    }

    #[test]
    fn evaluate_at_state_selects_correct_max() {
        // cut 0: intercept=2, coeffs=[1] → at state [10]: 2 + 10 = 12
        // cut 1: intercept=5, coeffs=[2] → at state [10]: 5 + 20 = 25
        // max = 25
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 2.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[2.0]);

        let result = pool.evaluate_at_state(&[10.0]);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn evaluate_at_state_empty_pool_returns_neg_infinity() {
        let pool = CutPool::new(10, 3, 5, 0);
        assert_eq!(pool.evaluate_at_state(&[1.0, 2.0, 3.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn evaluate_at_state_all_deactivated_returns_neg_infinity() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 100.0, &[1.0]);
        pool.deactivate(&[0]);
        assert_eq!(pool.evaluate_at_state(&[5.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn evaluate_at_state_ignores_deactivated_cuts() {
        // slot 0: active, intercept=10, coeff=[1]  → at state [3]: 13
        // slot 1: INACTIVE, intercept=100, coeff=[1] → would be 103, but ignored
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 100.0, &[1.0]);
        pool.deactivate(&[1]);

        assert_eq!(pool.evaluate_at_state(&[3.0]), 13.0);
    }

    #[test]
    fn ac_add_cut_stores_at_slot_zero_and_active_count_is_one() {
        let mut pool = CutPool::new(100, 9, 10, 0);
        let coeffs = vec![0.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 5.0, &coeffs);

        assert!(pool.active[0]);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn ac_deactivate_reduces_active_count_correctly() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);

        pool.deactivate(&[1]);

        assert_eq!(pool.active_count(), 2);
        assert!(!pool.active[1]);
    }

    #[test]
    fn ac_evaluate_at_state_returns_correct_max() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0, 2.0]);

        assert_eq!(pool.evaluate_at_state(&[3.0, 4.0]), 13.0);
    }

    #[test]
    fn ac_slot_base_offsets_slot() {
        let mut pool = CutPool::new(100, 9, 10, 5);
        let coeffs = vec![0.0; 9];
        pool.add_cut(NodeId(0), 0, 0, 1.0, &coeffs);

        assert!(pool.active[5]);
        assert!(!pool.active[0]);
    }

    #[test]
    fn ac_empty_pool_evaluate_returns_neg_infinity() {
        let pool = CutPool::new(10, 2, 1, 0);
        assert_eq!(pool.evaluate_at_state(&[1.0, 2.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn cut_pool_derives_debug_and_clone() {
        let mut pool = CutPool::new(5, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 3.0, &[1.0, 2.0]);

        let cloned = pool.clone();
        assert_eq!(cloned.active_count(), 1);
        assert_eq!(cloned.intercepts[0], 3.0);

        let debug_str = format!("{pool:?}");
        assert!(!debug_str.is_empty());
    }

    // ── SparsityReport tests ──────────────────────────────────────────

    #[test]
    fn sparsity_report_empty_pool() {
        let pool = CutPool::new(10, 3, 1, 0);
        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 0);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn sparsity_report_all_nonzero() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0, 3.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[4.0, 5.0, 6.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn sparsity_report_all_zero() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 0.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 6);
        assert!((report.sparsity_fraction - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.per_dimension_zeros, vec![2, 2, 2]);
    }

    #[test]
    fn sparsity_report_mixed() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 2.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 0.0, 3.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 6);
        assert_eq!(report.exact_zero_count, 3);
        assert!((report.sparsity_fraction - 0.5).abs() < 1e-10);
        assert_eq!(report.per_dimension_zeros, vec![1, 2, 0]);
    }

    #[test]
    fn sparsity_report_excludes_inactive_cuts() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 0.0]); // all zero, then deactivate
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[1.0, 2.0]); // all non-zero
        pool.deactivate(&[0]);

        let report = pool.sparsity_report();
        // Only the second cut is active.
        assert_eq!(report.total_coefficients, 2);
        assert_eq!(report.exact_zero_count, 0);
        assert!((report.sparsity_fraction - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sparsity_report_per_dimension_zeros_correct() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        // Cut 0: dims 0,2 are zero
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 1.0, 0.0, 3.0]);
        // Cut 1: dims 0,3 are zero
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 2.0, 4.0, 0.0]);
        // Cut 2: no zeros
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[5.0, 6.0, 7.0, 8.0]);

        let report = pool.sparsity_report();
        assert_eq!(report.total_coefficients, 12);
        assert_eq!(report.exact_zero_count, 4);
        assert_eq!(report.per_dimension_zeros, vec![2, 0, 1, 1]);
    }

    #[test]
    fn warm_start_cuts_keep_their_generating_iteration() {
        use novomodelo_io::OwnedPolicyCutRecord;

        let records = vec![
            OwnedPolicyCutRecord {
                slot_index: 0,
                coefficients: vec![1.0, 2.0],
                intercept: 10.0,
                is_active: true,
                iteration: 5,
                ..owned_piece_defaults()
            },
            OwnedPolicyCutRecord {
                slot_index: 1,
                coefficients: vec![3.0, 4.0],
                intercept: 20.0,
                is_active: true,
                iteration: 7,
                forward_pass_index: 1,
                scored: false,
                active_count: None,
                last_active_iteration: None,
            },
        ];

        let pool = CutPool::new_with_warm_start(2, 4, 100, &records);
        assert_eq!(pool.prefix_len(), 2);
        assert_eq!(pool.populated_count, 2);
        assert_eq!(pool.metadata[0].iteration_generated, 5);
        assert_eq!(pool.metadata[1].iteration_generated, 7);
        assert_eq!(pool.metadata[0].last_active_iter, 5);
        assert_eq!(pool.metadata[1].last_active_iter, 7);
        assert!(pool.is_prefix_slot(0));
        assert!(pool.is_prefix_slot(1));
        assert!(!pool.is_prefix_slot(2));
    }

    fn loaded_cut(iteration: u32) -> novomodelo_io::OwnedPolicyCutRecord {
        novomodelo_io::OwnedPolicyCutRecord {
            slot_index: 0,
            iteration,
            intercept: 1.0,
            coefficients: vec![1.0],
            is_active: true,
            ..owned_piece_defaults()
        }
    }

    #[test]
    fn restart_activity_clock_touches_only_the_loaded_prefix() {
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &[loaded_cut(5), loaded_cut(7)]);
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0]);
        pool.record_binding(0, 3, 6);
        pool.record_binding(1, 1, 9);
        pool.record_binding(2, 2, 4);
        pool.set_active(1, false);

        pool.restart_activity_clock();

        let clock = |slot: usize| {
            let meta = pool.metadata(slot);
            (
                meta.iteration_generated,
                meta.active_count,
                meta.last_active_iter,
            )
        };
        assert_eq!(clock(0), (5, 0, 0));
        assert_eq!(clock(1), (7, 0, 0));
        assert_eq!(clock(2), (1, 2, 4));
        assert_eq!(
            (pool.is_active(0), pool.is_active(1), pool.is_active(2)),
            (true, false, true)
        );
    }

    #[test]
    fn a_loaded_cut_sharing_the_current_iteration_number_is_not_a_delta_cut() {
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &[loaded_cut(2)]);
        pool.set_iteration_base(2);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[1.0]);

        let delta: Vec<usize> = pool.active_delta_cuts(2).map(|(slot, _, _)| slot).collect();

        assert_eq!(delta, vec![1]);
    }

    fn unscored(origin: SlotOrigin) -> SlotRole {
        SlotRole {
            origin,
            scored: false,
        }
    }

    fn scored_generated() -> SlotRole {
        SlotRole {
            origin: SlotOrigin::Generated,
            scored: true,
        }
    }

    fn identity_projection(n_state: usize) -> CutStateProjection {
        CutStateProjection::new(
            &state_layout(n_state, 0),
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        )
    }

    fn roles_of(pool: &CutPool) -> Vec<SlotRole> {
        (0..pool.capacity).map(|slot| pool.role(slot)).collect()
    }

    #[test]
    fn a_slot_role_is_fixed_when_the_slot_is_written() {
        use SlotOrigin::{Generated, Prefix, Vacant};

        let fresh = CutPool::new(3, 1, 1, 0);
        assert_eq!(roles_of(&fresh), vec![unscored(Vacant); 3]);

        let records = [loaded_cut(5), loaded_cut(7)];
        let from_records = CutPool::from_deserialized(1, 2, &records);
        assert_eq!(roles_of(&from_records), vec![unscored(Prefix); 2]);

        let mut pool = CutPool::new_with_warm_start(1, 2, 4, &records);
        assert_eq!(
            roles_of(&pool)[..3],
            [unscored(Prefix), unscored(Prefix), unscored(Vacant)]
        );
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0]);
        pool.enable_trial_states();
        pool.add_cut(NodeId(0), 2, 0, 4.0, &[1.0]);
        pool.grow(12);
        let expected = [
            unscored(Prefix),
            unscored(Prefix),
            unscored(Generated),
            unscored(Vacant),
            scored_generated(),
        ];
        assert_eq!(roles_of(&pool)[..5], expected);
        assert_eq!(roles_of(&pool)[5..], vec![unscored(Vacant); 7]);

        let metadata = |slot: usize| pool.metadata(slot).clone();
        let kept: Vec<CutMetadata> = (0..pool.populated()).map(metadata).collect();
        let before = roles_of(&pool);

        pool.set_active(2, false);
        assert_eq!(roles_of(&pool), before, "set_active");
        pool.set_active(2, true);
        pool.deactivate(&[4]);
        assert_eq!(roles_of(&pool), before, "deactivate");
        pool.apply_updates(&CutActivityUpdates {
            updates: vec![2],
            reactivations: vec![4],
        });
        assert_eq!(roles_of(&pool), before, "apply_updates");
        pool.set_metadata_for_test(2, kept[0].clone());
        assert_eq!(roles_of(&pool), before, "set_metadata_for_test");
        pool.record_binding(2, 1, 3);
        assert_eq!(roles_of(&pool), before, "record_binding");
        pool.restart_activity_clock();
        assert_eq!(roles_of(&pool), before, "restart_activity_clock");
        pool.store_trial_state(2, 0, &[9.0], &identity_projection(1));
        assert_eq!(roles_of(&pool), before, "store_trial_state");
        let evicted = pool.enforce_budget(0, 9).evicted_count;
        assert!(evicted > 0, "the budget must evict for the check to bite");
        assert_eq!(roles_of(&pool), before, "enforce_budget");
    }

    fn record_at(slot: u32, iteration: u32) -> novomodelo_io::OwnedPolicyCutRecord {
        novomodelo_io::OwnedPolicyCutRecord {
            slot_index: slot,
            iteration,
            forward_pass_index: slot % 2,
            intercept: f64::from(slot) + 0.5,
            coefficients: vec![f64::from(slot) + 0.25],
            is_active: true,
            ..owned_piece_defaults()
        }
    }

    #[test]
    fn a_resumed_pool_keeps_each_record_at_its_slot_and_a_hole_vacant() {
        use SlotOrigin::{Generated, Prefix, Vacant};

        let mut records = vec![
            record_at(0, 1),
            record_at(1, 1),
            record_at(3, 2),
            record_at(6, 3),
        ];
        records[2].scored = true;
        records[3].is_active = false;
        let pool = CutPool::new_resumed(1, 3, 4, 2, 5, &records).unwrap();

        assert_eq!(
            roles_of(&pool)[..8],
            [
                unscored(Prefix),
                unscored(Prefix),
                unscored(Vacant),
                scored_generated(),
                unscored(Vacant),
                unscored(Vacant),
                unscored(Generated),
                unscored(Vacant),
            ]
        );
        assert_eq!(
            (
                pool.slot_base,
                pool.prefix_len(),
                pool.populated(),
                pool.generated(),
                pool.active_count()
            ),
            (2, 2, 7, 4, 3)
        );
        for slot in [2, 4, 5] {
            assert!(!pool.is_active(slot), "slot {slot}");
            assert_eq!(pool.intercept(slot), 0.0, "slot {slot}");
            assert_eq!(pool.coefficient_row(slot), [0.0], "slot {slot}");
        }
        for record in &records {
            let slot = record.slot_index as usize;
            let meta = pool.metadata(slot);
            assert_eq!(pool.intercept(slot), record.intercept, "slot {slot}");
            assert_eq!(
                pool.coefficient_row(slot),
                record.coefficients,
                "slot {slot}"
            );
            assert_eq!(pool.is_active(slot), record.is_active, "slot {slot}");
            assert_eq!(
                (
                    meta.iteration_generated,
                    meta.forward_pass_index,
                    meta.last_active_iter,
                    meta.active_count
                ),
                (
                    u64::from(record.iteration),
                    record.forward_pass_index,
                    if slot < pool.prefix_len() {
                        0
                    } else {
                        u64::from(record.iteration)
                    },
                    0
                ),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn a_resumed_record_without_counters_resumes_on_the_clock_its_run_left_it() {
        let counted = |slot, iteration| novomodelo_io::OwnedPolicyCutRecord {
            active_count: Some(4),
            last_active_iteration: Some(7),
            ..record_at(slot, iteration)
        };
        let clock = |pool: &CutPool, slot: usize| {
            let meta = pool.metadata(slot);
            (meta.active_count, meta.last_active_iter)
        };
        let pool = CutPool::new_resumed(
            1,
            2,
            4,
            2,
            6,
            &[
                record_at(0, 3),
                counted(1, 3),
                record_at(2, 3),
                counted(3, 3),
            ],
        )
        .unwrap();
        assert_eq!(clock(&pool, 0), (0, 0), "prefix record without counters");
        assert_eq!(clock(&pool, 1), (4, 7), "prefix record with counters");
        assert_eq!(clock(&pool, 2), (0, 3), "generated record without counters");
        assert_eq!(clock(&pool, 3), (4, 7), "generated record with counters");

        let mut restarted = CutPool::new_resumed(
            1,
            2,
            4,
            2,
            6,
            &[counted(0, 3), counted(1, 3), record_at(2, 3)],
        )
        .unwrap();
        restarted.restart_activity_clock();
        assert_eq!(
            clock(&restarted, 0),
            clock(&pool, 0),
            "a restarted prefix slot is the one a record without counters resumes at"
        );
    }

    #[test]
    fn a_resumed_pool_is_as_large_as_the_recorded_pool_or_the_fresh_formula() {
        let recorded = [record_at(0, 1)];
        let grown = CutPool::new_resumed(1, 3, 4, 1, 20, &recorded).unwrap();
        assert_eq!(grown.capacity, 20);
        let fresh = CutPool::new_resumed(1, 3, 4, 1, 5, &recorded).unwrap();
        assert_eq!(fresh.capacity, 1 + 4 * 3);
    }

    #[test]
    fn a_resumed_pool_takes_the_loop_cuts_where_the_recorded_run_put_them() {
        let records = [record_at(0, 1), record_at(1, 1), record_at(2, 2)];
        let mut pool = CutPool::new_resumed(1, 2, 4, 0, 6, &records).unwrap();
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 3, 0, 9.0, &[1.0]);
        pool.add_cut(NodeId(0), 3, 1, 9.0, &[1.0]);

        assert_eq!(pool.role(3), unscored(SlotOrigin::Vacant));
        assert_eq!((pool.populated(), pool.generated()), (6, 5));
        let delta: Vec<usize> = pool.active_delta_cuts(3).map(|(slot, _, _)| slot).collect();
        assert_eq!(delta, vec![4, 5]);
    }

    fn refusal(
        prefix_len: u32,
        stored_capacity: u32,
        records: &[novomodelo_io::OwnedPolicyCutRecord],
    ) -> String {
        CutPool::new_resumed(1, 1, 2, prefix_len, stored_capacity, records)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn a_resumed_pool_refuses_two_records_at_one_slot() {
        let msg = refusal(0, 4, &[record_at(2, 1), record_at(2, 2)]);
        assert!(msg.contains("slot 2 holds two records"), "{msg}");
        assert!(CutPool::new_resumed(1, 1, 2, 0, 4, &[record_at(2, 1), record_at(3, 2)]).is_ok());
    }

    #[test]
    fn a_resumed_pool_refuses_a_slot_at_or_above_its_capacity() {
        let msg = refusal(0, 4, &[record_at(4, 1)]);
        assert!(msg.contains("slot 4 is not below the capacity 4"), "{msg}");
        assert!(CutPool::new_resumed(1, 1, 2, 0, 4, &[record_at(3, 1)]).is_ok());
    }

    #[test]
    fn a_resumed_pool_refuses_an_empty_slot_inside_its_prefix() {
        let msg = refusal(3, 4, &[record_at(0, 1), record_at(2, 1)]);
        assert!(
            msg.contains("slot 1 is inside the protected prefix of 3"),
            "{msg}"
        );
        let filled = [record_at(0, 1), record_at(1, 1), record_at(2, 1)];
        assert!(CutPool::new_resumed(1, 1, 2, 3, 4, &filled).is_ok());
    }

    fn slots_in(pool: &CutPool, set: CutSet) -> Vec<usize> {
        pool.cuts_in(set).map(|(slot, _, _)| slot).collect()
    }

    #[test]
    fn cuts_in_all_follows_the_slot_roles() {
        use SlotOrigin::{Generated, Prefix};

        let scored = |origin| SlotRole {
            origin,
            scored: true,
        };
        let mut pool = CutPool::new(8, 1, 1, 0);
        for slot in 0..7_u32 {
            pool.add_cut(NodeId(0), u64::from(slot), 0, f64::from(slot), &[1.0]);
        }
        let roles = [
            (scored(Generated), true),
            (scored(Generated), false),
            (unscored(Generated), false),
            (unscored(Prefix), true),
            (unscored(Prefix), false),
            (scored(Prefix), false),
        ];
        for (slot, (role, active)) in (0..).zip(roles) {
            pool.set_role_for_test(slot as usize, role);
            pool.set_active(slot, active);
        }
        pool.set_active(6, false);
        pool.set_role_for_test(6, unscored(SlotOrigin::Vacant));

        assert_eq!(slots_in(&pool, CutSet::All), vec![0, 1, 2, 3, 5]);
        assert_eq!(pool.count_in(CutSet::All), 5);
        assert_eq!(slots_in(&pool, CutSet::Active), vec![0, 3]);
        assert_eq!(pool.count_in(CutSet::Active), 2);
    }

    #[test]
    fn cuts_in_active_equals_active_cuts() {
        let mut pool = CutPool::new(8, 2, 1, 0);
        for slot in 0..6_u32 {
            pool.add_cut(NodeId(0), u64::from(slot), 0, f64::from(slot), &[1.0, 2.0]);
        }
        pool.deactivate(&[1, 4]);

        let active: Vec<_> = pool.active_cuts().collect();
        let in_set: Vec<_> = pool.cuts_in(CutSet::Active).collect();

        assert_eq!(in_set, active);
        assert_eq!(pool.count_in(CutSet::Active), active.len());
        assert_eq!(active.len(), 4);
    }

    #[test]
    fn a_loaded_pool_keeps_the_cut_sets_of_its_run() {
        let record = |slot, is_active, scored| novomodelo_io::OwnedPolicyCutRecord {
            is_active,
            scored,
            ..record_at(slot, 1)
        };
        let records = [
            record(0, true, false),
            record(1, false, true),
            record(2, false, false),
            record(3, false, false),
        ];
        let pool = CutPool::from_deserialized(1, 3, &records);

        assert_eq!(pool.prefix_len(), 3);
        assert_eq!(
            slots_in(&pool, CutSet::All),
            vec![0, 1, 3],
            "the inactive scored prefix cut and the inactive generated cut stay in `All`, the \
             inactive unscored prefix cut does not"
        );
        assert_eq!(slots_in(&pool, CutSet::Active), vec![0]);
        assert_eq!(pool.count_in(CutSet::All), 3);
        assert_eq!(pool.count_in(CutSet::Active), pool.active_count());
    }

    #[test]
    fn the_loading_constructors_read_each_records_scored_flag() {
        use SlotOrigin::{Generated, Prefix};

        let record = |slot, scored| novomodelo_io::OwnedPolicyCutRecord {
            scored,
            ..record_at(slot, 1)
        };
        let records = [record(0, true), record(1, false), record(2, true)];
        let role = |origin, scored| SlotRole { origin, scored };

        let warm = CutPool::new_with_warm_start(1, 2, 4, &records);
        assert_eq!(
            roles_of(&warm)[..3],
            [role(Prefix, true), role(Prefix, false), role(Prefix, true)]
        );
        assert_eq!(warm.prefix_len(), 3);

        let simulated = CutPool::from_deserialized(1, 2, &records);
        assert_eq!(
            roles_of(&simulated),
            [
                role(Prefix, true),
                role(Prefix, false),
                role(Generated, true)
            ]
        );
        assert_eq!(
            (
                simulated.prefix_len(),
                simulated.slot_base,
                simulated.capacity
            ),
            (2, 3, 3)
        );
        let unprotected = CutPool::from_deserialized(1, 0, &records);
        assert_eq!(unprotected.prefix_len(), 0);
        assert!(!unprotected.has_prefix_cuts());
    }

    #[test]
    fn a_reserved_leading_slot_is_vacant_not_loaded() {
        let mut pool = CutPool::new(4, 1, 1, 2);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[1.0]);

        assert_eq!(pool.role(0), unscored(SlotOrigin::Vacant));
        assert_eq!(pool.role(1), unscored(SlotOrigin::Vacant));
        assert_eq!(pool.role(2), unscored(SlotOrigin::Generated));
        assert_eq!(pool.role(3), unscored(SlotOrigin::Generated));
        assert!((0..4).all(|slot| !pool.is_prefix_slot(slot)));
        assert_eq!(pool.generated(), 2);
    }

    #[test]
    fn a_pool_that_stores_trial_states_marks_its_cuts_scored() {
        let projection = identity_projection(1);
        let mut pool = CutPool::new_with_warm_start(1, 2, 4, &[loaded_cut(5)]);
        pool.set_iteration_base(1);
        assert!(!pool.stores_trial_states());
        assert_eq!(pool.trial_state(0), None);
        assert_eq!(pool.stored_trial_states(), 0);

        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0]);
        pool.enable_trial_states();
        pool.add_cut(NodeId(0), 2, 0, 4.0, &[1.0]);
        pool.add_cut(NodeId(0), 2, 1, 5.0, &[1.0]);
        assert!(pool.stores_trial_states());
        assert_eq!(pool.role(0), unscored(SlotOrigin::Prefix));
        assert_eq!(pool.role(1), unscored(SlotOrigin::Generated));
        assert_eq!(pool.role(3), scored_generated());
        assert_eq!(pool.role(4), scored_generated());

        pool.store_trial_state(2, 0, &[7.0], &projection);
        pool.enable_trial_states();
        assert_eq!(pool.stored_trial_states(), 1);
        assert_eq!(pool.trial_state(3), Some(&[7.0][..]));
        pool.store_trial_state(2, 1, &[8.0], &projection);
        assert_eq!(pool.stored_trial_states(), 2);
        assert_eq!(pool.trial_state(4), Some(&[8.0][..]));
        for slot in [0, 1, 2, 5] {
            assert_eq!(pool.trial_state(slot), None, "slot {slot}");
        }
    }

    /// derived: the flat cuts are worth 9, 1, 5 and 3 at every state. Slot 0 was
    /// written before the pool stored states, so it is not live: slot 2 owns
    /// both states, the scored slots 1 and 3 own none, and slot 0, the highest
    /// cut, is in neither list whatever its flag.
    #[test]
    fn ownership_activity_updates_leave_an_unscored_generated_cut_alone() {
        let mut pool = CutPool::new(4, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 9.0, &[0.0]);
        pool.enable_trial_states();
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 3, 0, 3.0, &[0.0]);
        pool.set_trial_state_for_test(1, &[0.0]);
        pool.set_trial_state_for_test(2, &[1.0]);
        pool.update_ownership();
        let expected = CutActivityUpdates {
            updates: vec![1, 3],
            reactivations: vec![],
        };

        assert_eq!(pool.ownership_activity_updates(), expected);
        pool.set_active(0, false);
        assert_eq!(pool.ownership_activity_updates(), expected);
    }

    /// derived: two loaded records put `slot_base` at 2; with `iteration_base` 4
    /// the run has completed `r = k - 3` iterations at a run of iteration `k`.
    /// For `r < 2` the start is slot 0, then it is `2 + (r - 2) * 3`: 0 at
    /// `k = 4`, then 2, 5, 8, 11 and 14.
    #[test]
    fn the_window_start_slot_agrees_with_the_slot_formula() {
        let lookback = NonZeroU64::new(2).unwrap();
        let mut pool = CutPool::new_with_warm_start(1, 3, 10, &[loaded_cut(5), loaded_cut(7)]);
        pool.set_iteration_base(4);

        for (k, start) in (4..=9).zip([0, 2, 5, 8, 11, 14]) {
            assert_eq!(pool.window_start_slot(lookback, k), start, "start at {k}");
            for generating in 4..=9 {
                for pass in 0..3 {
                    assert_eq!(
                        pool.slot_index(generating, pass) >= start,
                        generating > k - 2,
                        "iteration {generating} pass {pass} at {k}"
                    );
                }
            }
            for loaded in 0..2 {
                assert_eq!(loaded >= start, k == 4, "loaded slot {loaded} at {k}");
            }
        }
    }

    /// derived: with stride 2, lookback 4 and a run every 2 iterations, a pool
    /// holds the states of all its iterations until the run of iteration 4, then
    /// those of the last 4 iterations: at most `(4 + 2) * 2` states before a
    /// run, which is the reserve, and `4 * 2` after it.
    #[test]
    fn a_windowed_pool_never_holds_more_rows_than_its_reserve() {
        const STRIDE: u32 = 2;
        const ITERATIONS: u64 = 12;
        let mut pool = CutPool::new(24, 1, STRIDE, 0);
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(NonZeroU64::new(4).unwrap(), 2);
        let reserve = (4 + 2) * STRIDE as usize;
        assert_eq!(pool.reserved_trial_rows(), reserve);

        for iteration in 1..=ITERATIONS {
            for pass in 0..STRIDE {
                let slot = pool.slot_index(iteration, pass);
                pool.add_cut(NodeId(0), iteration, pass, 1.0, &[1.0]);
                pool.set_trial_state_for_test(slot, &[0.0]);
            }
            if iteration % 2 == 0 {
                assert!(pool.stored_trial_states() <= reserve, "before the run");
                pool.expire_trial_states(iteration);
                pool.update_ownership();
                assert_eq!(
                    pool.stored_trial_states(),
                    iteration.min(4) as usize * STRIDE as usize,
                    "after the run of iteration {iteration}"
                );
            }
            assert_eq!(pool.reserved_trial_rows(), reserve, "at {iteration}");
        }
        assert_eq!(pool.stored_trial_states(), 4 * STRIDE as usize);
    }

    /// derived: with stride 3 and a window of 1, the run of iteration `k` starts
    /// at slot `3 * (k - 1)`. Iteration 3 writes slot 6 only, so slots 7 and 8
    /// are never written, and iteration 4 writes none: at its run the start, 9,
    /// lies above `populated() = 7` and the state at slot 6 goes. Iteration 5
    /// writes slots 12 and 13 and iteration 6 slots 15 to 17, so the six runs
    /// leave 3, 3, 1, 0, 2 and 3 states. The cuts are flat, so the written cut of
    /// highest intercept owns every stored state: slot 1 (4.0) at the first run,
    /// slot 4 (5.0) at the second and third, slot 12 (6.0) at the fifth and
    /// slot 16 (7.0) at the sixth.
    #[test]
    fn a_window_start_above_the_populated_slots_releases_every_older_state() {
        const STRIDE: u32 = 3;
        let lookback = NonZeroU64::new(1).unwrap();
        let mut pool = CutPool::new(18, 1, STRIDE, 0);
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(lookback, 1);
        let written: [&[(u32, f64)]; 6] = [
            &[(0, 1.0), (1, 4.0), (2, 2.0)],
            &[(0, 3.0), (1, 5.0), (2, 0.0)],
            &[(0, 2.5)],
            &[],
            &[(0, 6.0), (1, 1.0)],
            &[(0, 0.5), (1, 7.0), (2, 1.5)],
        ];
        let expected = [
            (3, Some(1)),
            (3, Some(4)),
            (1, Some(4)),
            (0, None),
            (2, Some(12)),
            (3, Some(16)),
        ];

        for (index, (cuts, (stored, owner))) in written.iter().zip(expected).enumerate() {
            let iteration = index as u64 + 1;
            for &(pass, intercept) in *cuts {
                let slot = pool.slot_index(iteration, pass);
                pool.add_cut(NodeId(0), iteration, pass, intercept, &[0.0]);
                pool.set_trial_state_for_test(slot, &[0.0]);
            }
            if iteration == 4 {
                assert!(pool.window_start_slot(lookback, iteration) > pool.populated());
            }
            pool.expire_trial_states(iteration);
            pool.update_ownership();

            let mut rebuilt = pool.clone();
            rebuilt.rebuild_ownership();
            assert_eq!(pool.stored_trial_states(), stored, "run of {iteration}");
            for slot in 0..pool.capacity {
                let label = format!("run of {iteration}, slot {slot}");
                assert_eq!(
                    pool.state_owner(slot),
                    pool.trial_state(slot).and(owner),
                    "{label}"
                );
                assert_eq!(
                    pool.owned_states(slot),
                    if owner == Some(slot) { stored } else { 0 },
                    "{label}"
                );
                assert_eq!(rebuilt.state_owner(slot), pool.state_owner(slot), "{label}");
                assert_eq!(
                    rebuilt.state_best_value(slot).map(f64::to_bits),
                    pool.state_best_value(slot).map(f64::to_bits),
                    "{label}"
                );
                assert_eq!(
                    rebuilt.owned_states(slot),
                    pool.owned_states(slot),
                    "{label}"
                );
            }
        }
        assert!((0..15).all(|slot| pool.trial_state(slot).is_none()));
    }

    fn scored_cut(intercept: f64, coefficient: f64) -> novomodelo_io::OwnedPolicyCutRecord {
        novomodelo_io::OwnedPolicyCutRecord {
            intercept,
            coefficients: vec![coefficient],
            scored: true,
            ..loaded_cut(1)
        }
    }

    /// derived: the cuts' values at `x` are `x`, `3`, `2x - 5` and `0.5`. The
    /// states `0, 2, 4, 6` have the maxima `3, 3, 4, 7`, owned by slots `1, 1, 0,
    /// 2`, so slots 0 to 3 own `1, 2, 1, 0` states. A window of 1 with a check
    /// frequency of 1 reserves 2 rows for the 4 loaded states.
    #[test]
    fn a_loaded_state_is_stored_at_a_scored_prefix_slot_past_the_reserve() {
        let records = [
            scored_cut(0.0, 1.0),
            scored_cut(3.0, 0.0),
            scored_cut(-5.0, 2.0),
            scored_cut(0.5, 0.0),
        ];
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &records);
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(NonZeroU64::new(1).unwrap(), 1);
        assert_eq!(pool.reserved_trial_rows(), 2);

        let states = [0.0, 2.0, 4.0, 6.0];
        for (slot, state) in states.into_iter().enumerate() {
            pool.load_trial_state(slot, &[state]);
        }

        for (slot, state) in states.into_iter().enumerate() {
            assert_eq!(
                pool.role(slot),
                SlotRole {
                    origin: SlotOrigin::Prefix,
                    scored: true
                }
            );
            assert_eq!(pool.trial_state(slot), Some(&[state][..]));
        }
        assert_eq!(pool.stored_trial_states(), 4);
        assert_eq!(pool.reserved_trial_rows(), 4, "a state past the reserve");

        let mut rebuilt = pool.clone();
        rebuilt.rebuild_ownership();
        pool.update_ownership();
        for p in [&pool, &rebuilt] {
            let owners: Vec<_> = (0..4).map(|slot| p.state_owner(slot)).collect();
            assert_eq!(owners, [Some(1), Some(1), Some(0), Some(2)]);
            let best: Vec<_> = (0..4).map(|slot| p.state_best_value(slot)).collect();
            assert_eq!(best, [Some(3.0), Some(3.0), Some(4.0), Some(7.0)]);
            let owned: Vec<_> = (0..4).map(|slot| p.owned_states(slot)).collect();
            assert_eq!(owned, [1, 2, 1, 0]);
        }
    }

    /// derived: two loaded records put `slot_base` at 2 and `iteration_base` is
    /// 1, so a window of 2 starts at slot 0 for the run of iteration 1 and at
    /// slot 2 for the run of iteration 2: a loaded state leaves at the run's
    /// iteration 2, whatever iteration its record carries. At the state 1 the
    /// values are `0.75` and `2.75` for the loaded cuts and `0` for the cut
    /// generated at slot 2, so slot 1 owns all three states until the loaded two
    /// leave.
    #[test]
    fn a_loaded_state_leaves_a_window_at_the_runs_iteration_w() {
        let records = [scored_cut(0.5, 0.25), scored_cut(1.5, 1.25)];
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &records);
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(NonZeroU64::new(2).unwrap(), 1);
        pool.load_trial_state(0, &[1.0]);
        pool.load_trial_state(1, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 0.0, &[0.0]);
        pool.set_trial_state_for_test(2, &[1.0]);

        pool.expire_trial_states(1);
        pool.update_ownership();
        assert_eq!(pool.stored_trial_states(), 3);
        assert_eq!(pool.owned_states(1), 3);

        pool.expire_trial_states(2);
        pool.update_ownership();
        assert_eq!(pool.stored_trial_states(), 1);
        assert_eq!(
            [0, 1, 2].map(|slot| pool.trial_state(slot).is_some()),
            [false, false, true]
        );
        assert_eq!(pool.owned_states(1), 1);
        assert_eq!(pool.state_owner(0), None);
        assert_eq!(pool.state_owner(2), Some(1));
    }

    #[test]
    fn new_with_warm_start_zero_max_iterations_yields_fixed_capacity() {
        use novomodelo_io::OwnedPolicyCutRecord;

        let records = vec![
            OwnedPolicyCutRecord {
                slot_index: 0,
                coefficients: vec![1.0, 2.0],
                intercept: 5.0,
                is_active: true,
                iteration: 3,
                ..owned_piece_defaults()
            },
            OwnedPolicyCutRecord {
                slot_index: 1,
                coefficients: vec![3.0, 4.0],
                intercept: 6.0,
                is_active: true,
                iteration: 4,
                forward_pass_index: 1,
                scored: false,
                active_count: None,
                last_active_iteration: None,
            },
        ];

        let pool = CutPool::new_with_warm_start(2, 4, 0, &records);
        assert_eq!(
            pool.capacity,
            records.len(),
            "max_iterations = 0 must leave no growable training slack"
        );
        assert_eq!(pool.prefix_len(), records.len());
        assert_eq!(pool.populated(), records.len());
    }

    #[test]
    fn has_prefix_cuts_follows_the_loaded_records() {
        use novomodelo_io::OwnedPolicyCutRecord;

        let empty = CutPool::new(100, 2, 10, 0);
        assert!(!empty.has_prefix_cuts());

        let active_record = vec![OwnedPolicyCutRecord {
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: true,
            iteration: 0,
            ..owned_piece_defaults()
        }];
        let active = CutPool::new_with_warm_start(1, 4, 100, &active_record);
        assert!(active.has_prefix_cuts());

        let inactive_record = vec![OwnedPolicyCutRecord {
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: false,
            iteration: 0,
            ..owned_piece_defaults()
        }];
        let inactive = CutPool::new_with_warm_start(1, 4, 100, &inactive_record);
        assert!(inactive.has_prefix_cuts());
        assert_eq!(inactive.active_count(), 0);
    }

    #[test]
    fn no_prefix_cuts_when_no_record_is_loaded() {
        let loaded_nothing = CutPool::new_with_warm_start(2, 10, 100, &[]);
        assert!(!loaded_nothing.has_prefix_cuts());
        assert_eq!(loaded_nothing.prefix_len(), 0);

        let reserved_slots = CutPool::new(100, 2, 10, 5);
        assert!(!reserved_slots.has_prefix_cuts());
        assert_eq!(reserved_slots.prefix_len(), 0);
        assert_eq!(reserved_slots.slot_base, 5);
    }

    // ── enforce_budget tests ────────────────────────────────────────────────

    #[test]
    fn enforce_budget_noop_when_under_budget() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 2.0]);
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[3.0, 4.0]);
        assert_eq!(pool.active_count(), 2);
        let result = pool.enforce_budget(5, 1);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(result.active_after, 2);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn enforce_budget_evicts_oldest_last_active_iter() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // Add 5 cuts at iterations 0-4
        for iter in 0..5_u64 {
            pool.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
            // Set last_active_iter to make older cuts staler
            pool.metadata[pool.populated_count - 1].last_active_iter = iter;
        }
        assert_eq!(pool.active_count(), 5);
        // Budget = 3, current_iteration = 5 → evict 2 oldest
        let result = pool.enforce_budget(3, 5);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(result.active_after, 3);
        assert_eq!(pool.active_count(), 3);
        // The 2 oldest (last_active_iter 0 and 1) should be evicted
        // Slot for (iter=0, fp=0) = 0*10+0 = 0
        // Slot for (iter=1, fp=0) = 1*10+0 = 10
        assert!(!pool.active[0], "oldest cut should be evicted");
        assert!(!pool.active[10], "second oldest should be evicted");
    }

    #[test]
    fn enforce_budget_tiebreaks_by_active_count() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // Two cuts with same last_active_iter but different active_count
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 1;
        pool.metadata[0].active_count = 5;
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[0.0, 1.0]);
        pool.metadata[1].last_active_iter = 1;
        pool.metadata[1].active_count = 2;
        assert_eq!(pool.active_count(), 2);
        // Budget = 1 → evict the one with lower active_count (slot 1)
        let result = pool.enforce_budget(1, 1);
        assert_eq!(result.evicted_count, 1);
        assert!(pool.active[0], "higher active_count survives");
        assert!(!pool.active[1], "lower active_count evicted");
    }

    #[test]
    fn enforce_budget_protects_current_iteration() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // 3 cuts: 2 from iteration 0, 1 from iteration 1 (current)
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.metadata[0].last_active_iter = 0;
        pool.add_cut(NodeId(0), 0, 1, 2.0, &[0.0, 1.0]);
        pool.metadata[1].last_active_iter = 0;
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0, 1.0]);
        pool.metadata[10].last_active_iter = 1;
        assert_eq!(pool.active_count(), 3);
        // Budget = 1, current_iteration = 1 → can only evict iter-0 cuts
        let result = pool.enforce_budget(1, 1);
        assert_eq!(result.evicted_count, 2);
        // Current-iteration cut (slot 10) survives
        assert!(pool.active[10], "current iteration cut preserved");
    }

    #[test]
    fn enforce_budget_all_current_iteration_no_eviction() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        // All cuts from current iteration
        pool.add_cut(NodeId(0), 5, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 5, 1, 2.0, &[0.0, 1.0]);
        pool.add_cut(NodeId(0), 5, 2, 3.0, &[1.0, 1.0]);
        assert_eq!(pool.active_count(), 3);
        // Budget = 1, current_iteration = 5 → no candidates, no eviction
        let result = pool.enforce_budget(1, 5);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn enforce_budget_result_fields() {
        let mut pool = CutPool::new(100, 2, 10, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[1.0, 1.0]);
        assert_eq!(pool.active_count(), 3);
        let result = pool.enforce_budget(1, 3);
        assert_eq!(result.active_before, 3);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(result.active_after, 1);
    }

    #[test]
    fn enforce_budget_never_evicts_a_loaded_cut() {
        let loaded = [loaded_cut(0), loaded_cut(0), loaded_cut(3)];
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &loaded);
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[1.0]);

        let result = pool.enforce_budget(1, 3);

        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.active_before, 5);
        assert_eq!(result.active_after, 4);
        assert!(
            (0..3).all(|slot| pool.is_active(slot)),
            "a loaded cut stays"
        );
        assert!(!pool.is_active(3), "the oldest generated cut is evicted");
        assert!(pool.is_active(4));
    }

    #[test]
    fn enforce_budget_does_not_count_loaded_cuts() {
        let loaded = [loaded_cut(0), loaded_cut(0), loaded_cut(0)];
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &loaded);
        pool.set_iteration_base(1);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[1.0]);

        let result = pool.enforce_budget(1, 3);

        assert_eq!(result.evicted_count, 0);
        assert_eq!(result.active_before, 4);
        assert_eq!(result.active_after, 4);
    }

    /// `(iteration_generated, active, last_active_iter, active_count)`.
    type CutDraw = (u64, bool, u64, u64);

    /// Loaded cuts fill the first slots; the generated cuts follow, one per slot.
    fn pool_with(loaded: &[CutDraw], generated: &[CutDraw]) -> CutPool {
        let records: Vec<_> = loaded.iter().map(|_| loaded_cut(0)).collect();
        let mut pool = CutPool::new_with_warm_start(1, 1, generated.len() as u64, &records);
        pool.set_iteration_base(1);
        for i in 0..generated.len() {
            pool.add_cut(NodeId(0), 1 + i as u64, 0, 1.0, &[1.0]);
        }
        for (slot, &(iteration_generated, active, last_active_iter, active_count)) in
            loaded.iter().chain(generated).enumerate()
        {
            pool.set_metadata_for_test(
                slot,
                CutMetadata {
                    iteration_generated,
                    forward_pass_index: 0,
                    node: NodeId(0),
                    active_count,
                    last_active_iter,
                },
            );
            pool.set_active(slot as u32, active);
        }
        pool
    }

    fn active_slots(pool: &CutPool) -> BTreeSet<usize> {
        (0..pool.populated())
            .filter(|&slot| pool.is_active(slot))
            .collect()
    }

    /// derived: four generated cuts with the key `(2, 1)` are ordered by slot
    /// alone, so budget 2 evicts slots 0 and 1. The keys `(3, 0)`, `(2, 5)`,
    /// `(2, 1)`, `(2, 1)` sort to slots 2, 3, 1, 0, so budget 1 evicts the first
    /// three of them and keeps slot 0, the one with the largest
    /// `last_active_iter`.
    #[test]
    fn enforce_budget_breaks_ties_by_lowest_slot() {
        let mut equal_keys = pool_with(&[], &[(1, true, 2, 1); 4]);
        let result = equal_keys.enforce_budget(2, 3);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(active_slots(&equal_keys), BTreeSet::from([2, 3]));

        let mut mixed_keys = pool_with(
            &[],
            &[
                (1, true, 3, 0),
                (1, true, 2, 5),
                (1, true, 2, 1),
                (1, true, 2, 1),
            ],
        );
        let result = mixed_keys.enforce_budget(1, 3);
        assert_eq!(result.evicted_count, 3);
        assert_eq!(active_slots(&mixed_keys), BTreeSet::from([0]));
    }

    const CURRENT_ITERATION: u64 = 4;

    fn cut_draw() -> impl Strategy<Value = CutDraw> {
        (
            0..=CURRENT_ITERATION,
            prop::bool::weighted(0.8),
            0..=3u64,
            0..=3u64,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        })]

        #[test]
        fn evicted_set_is_the_lexicographic_minimum(
            loaded in prop::collection::vec(cut_draw(), 0..=4),
            generated in prop::collection::vec(cut_draw(), 1..=40),
            budget in any::<prop::sample::Index>(),
        ) {
            let budget = budget.index(generated.len() + 1) as u32;
            let mut pool = pool_with(&loaded, &generated);
            let before = active_slots(&pool);

            let mut candidates: Vec<(u64, u64, usize)> = Vec::new();
            let mut generated_active = 0usize;
            for (i, &(iteration_generated, active, last_active_iter, active_count)) in
                generated.iter().enumerate()
            {
                if active {
                    generated_active += 1;
                    if iteration_generated != CURRENT_ITERATION {
                        candidates.push((last_active_iter, active_count, loaded.len() + i));
                    }
                }
            }
            candidates.sort_unstable();
            let evict = generated_active
                .saturating_sub(budget as usize)
                .min(candidates.len());
            let expected: BTreeSet<usize> =
                candidates[..evict].iter().map(|&(_, _, slot)| slot).collect();

            let result = pool.enforce_budget(budget, CURRENT_ITERATION);
            let after = active_slots(&pool);

            prop_assert_eq!(before.difference(&after).copied().collect::<BTreeSet<_>>(), expected);
            prop_assert!(after.is_subset(&before));
            prop_assert_eq!(result.evicted_count as usize, evict);
            for (slot, &(iteration_generated, ..)) in
                loaded.iter().chain(&generated).enumerate()
            {
                if slot < loaded.len() || iteration_generated == CURRENT_ITERATION {
                    prop_assert_eq!(before.contains(&slot), after.contains(&slot));
                }
            }

            let second = pool.enforce_budget(budget, CURRENT_ITERATION);
            prop_assert_eq!(second.evicted_count, 0);
            prop_assert_eq!(active_slots(&pool), after);
        }
    }

    type IterationDraw = (Vec<u32>, usize);

    /// A stride and, per iteration, the pass indices inserted in a drawn order and
    /// the extra slots to grow by before inserting.
    fn insertion_plan() -> impl Strategy<Value = (u32, Vec<IterationDraw>)> {
        (1..=4u32).prop_flat_map(|stride| {
            let iteration = (
                prop::sample::subsequence((0..stride).collect::<Vec<_>>(), 0..=stride as usize)
                    .prop_shuffle(),
                0..=2 * stride as usize,
            );
            (Just(stride), prop::collection::vec(iteration, 1..=8))
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        })]

        #[test]
        fn generated_slots_follow_generation_order_across_growth(
            (stride, plan) in insertion_plan(),
        ) {
            let mut pool = CutPool::new(stride as usize, 1, stride, 0);
            let mut written = BTreeSet::new();
            let mut marker = 0.0;
            for (iteration, (passes, extra)) in plan.iter().enumerate() {
                let iteration = iteration as u64;
                let needed = (iteration as usize + 1) * stride as usize;
                if pool.capacity < needed || *extra > 0 {
                    let snapshot = |pool: &CutPool| -> Vec<(SlotRole, f64, Vec<f64>)> {
                        (0..pool.capacity)
                            .map(|slot| {
                                (pool.role(slot), pool.intercept(slot), pool.coefficient_row(slot).to_vec())
                            })
                            .collect()
                    };
                    let before = snapshot(&pool);
                    pool.grow(pool.capacity.max(needed) + extra);
                    let after = snapshot(&pool);
                    prop_assert_eq!(&after[..before.len()], &before[..]);
                    prop_assert!(after[before.len()..].iter().all(|(role, ..)| role.origin == SlotOrigin::Vacant));
                }
                for &pass in passes {
                    marker += 1.0;
                    pool.add_cut(NodeId(0), iteration, pass, marker, &[-marker]);
                    written.insert((iteration, pass));
                }
            }

            let mut keys = Vec::new();
            for slot in 0..pool.populated() {
                let meta = pool.metadata(slot);
                let key = (meta.iteration_generated, meta.forward_pass_index);
                match pool.role(slot).origin {
                    SlotOrigin::Generated => {
                        prop_assert!(written.contains(&key));
                        keys.push((slot, key));
                    }
                    origin => prop_assert_eq!(origin, SlotOrigin::Vacant),
                }
            }
            prop_assert_eq!(keys.len(), written.len());
            prop_assert!(keys.windows(2).all(|pair| pair[0].1 < pair[1].1));

            let mut above: Option<usize> = None;
            for iteration in 0..plan.len() as u64 {
                let slots: Vec<usize> = keys
                    .iter()
                    .filter(|(_, (generated_in, _))| *generated_in == iteration)
                    .map(|&(slot, _)| slot)
                    .collect();
                if let (Some(&lowest), Some(floor)) = (slots.iter().min(), above) {
                    prop_assert!(lowest > floor);
                }
                above = slots.iter().max().copied().or(above);
            }
        }
    }

    // ── active_cuts early-exit tests ─────────────────────────────────────────

    /// Verify that `active_cuts()` stops iterating once all active cuts have
    /// been yielded — it must not scan up to `populated_count` when
    /// `cached_active_count` is small.
    ///
    /// Pool has `populated_count = 100` and only slot 0 is active
    /// (`cached_active_count = 1`).  The early-exit iterator must stop after
    /// visiting slot 0, yielding exactly 1 item.  If the old O(populated)
    /// walk were still in place the count would still be 1, but the
    /// scan-based implementation verifies correctness by construction:
    /// `remaining` hits 0 after the first active slot and `scan` returns
    /// `None` for all subsequent elements, preventing any further polling.
    #[test]
    fn active_cuts_early_exit_stops_at_cached_count() {
        // visit_stride = 1 so slot = slot_base + iteration * 1 + fp_index
        let mut pool = CutPool::new(100, 2, 1, 0);

        // Add a cut at slot 0 (iteration 0, fp 0).
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);

        // Manually populate a further 99 slots as inactive to extend
        // populated_count to 100 without going through add_cut (which marks
        // them active).  We write directly to the active flag so that
        // populated_count is extended to 100 while cached_active_count stays 1.
        //
        // We do this by exploiting that slot_index(i, 0) = i * 1 + 0 = i.
        // We need slot 99 to be "populated" (high-water mark = 100) so that
        // the old O(populated) walk would have to visit all 100 slots.
        pool.populated_count = 100;
        // active[0] is already true; active[1..100] are all false (default).

        assert_eq!(pool.cached_active_count, 1);
        assert_eq!(pool.populated_count, 100);

        // The iterator must yield exactly 1 item.
        let result: Vec<_> = pool.active_cuts().collect();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0, "yielded slot must be 0");
        assert_eq!(result[0].1, 5.0, "intercept must match");
        assert_eq!(result[0].2, &[1.0, 2.0], "coefficients must match");
    }

    /// Verify that `candidates_buf` retains its allocation across successive
    /// `enforce_budget` calls (i.e., the scratch buffer is reused, not dropped
    /// and reallocated each time).
    #[test]
    fn enforce_budget_candidates_buf_is_reused() {
        let mut pool = CutPool::new(100, 2, 10, 0);

        // Add cuts spread across several iterations so there are always
        // eviction candidates regardless of current_iteration.
        for iter in 0..5_u64 {
            pool.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
        }
        assert_eq!(pool.active_count(), 5);

        // First enforce: evicts some cuts, candidates_buf gets populated.
        pool.enforce_budget(3, 5);
        let cap_after_first = pool.candidates_buf.capacity();
        assert!(
            cap_after_first >= 1,
            "candidates_buf must have acquired capacity after first enforce_budget"
        );

        // Re-add cuts so the second call also has candidates.
        // We need iteration offsets past the existing slots; restart with a
        // new pool to keep slot arithmetic simple.
        let mut pool2 = CutPool::new(100, 2, 10, 0);
        for iter in 0..5_u64 {
            pool2.add_cut(NodeId(0), iter, 0, 1.0, &[1.0, 0.0]);
        }
        pool2.enforce_budget(3, 5);
        let cap_after_first2 = pool2.candidates_buf.capacity();

        // Second call on the same pool2 — re-add some cuts first.
        // Since slots 0, 10, 20, 30, 40 are now inactive, add at iter 6..=8.
        pool2.add_cut(NodeId(0), 6, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(NodeId(0), 7, 0, 2.0, &[0.0, 1.0]);
        pool2.add_cut(NodeId(0), 8, 0, 2.0, &[0.0, 1.0]);
        pool2.enforce_budget(2, 9);
        let cap_after_second2 = pool2.candidates_buf.capacity();

        // The capacity must not have shrunk — Vec::clear() preserves the heap
        // allocation, so the second call reuses the buffer.
        assert!(
            cap_after_second2 >= cap_after_first2,
            "candidates_buf capacity must not shrink across calls (was {cap_after_first2}, now {cap_after_second2})"
        );
    }

    // ── set_active tests ─────────────────────────────────────────────────────

    #[test]
    fn set_active_false_decrements_active_count() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);

        pool.set_active(1, false);

        assert!(!pool.active[1]);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(pool.populated(), 3);
    }

    #[test]
    fn set_active_true_reactivates_deactivated_slot() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool.deactivate(&[1]);

        pool.set_active(1, true);

        assert!(pool.active[1]);
        assert_eq!(pool.active_count(), 3);
        assert_eq!(pool.populated(), 3);
    }

    #[test]
    fn set_active_idempotent_when_state_unchanged() {
        let mut pool = CutPool::new(10, 4, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);

        // slot 1 is already active — second call must be a no-op
        pool.set_active(1, true);
        pool.set_active(1, true);

        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn deactivate_delegates_to_set_active() {
        let mut pool_a = CutPool::new(10, 4, 1, 0);
        pool_a.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool_a.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool_a.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool_a.deactivate(&[1, 2]);

        let mut pool_b = CutPool::new(10, 4, 1, 0);
        pool_b.add_cut(NodeId(0), 0, 0, 1.0, &[1.0, 0.0, 0.0, 0.0]);
        pool_b.add_cut(NodeId(0), 1, 0, 2.0, &[0.0, 1.0, 0.0, 0.0]);
        pool_b.add_cut(NodeId(0), 2, 0, 3.0, &[0.0, 0.0, 1.0, 0.0]);
        pool_b.set_active(1, false);
        pool_b.set_active(2, false);

        assert_eq!(pool_a.active, pool_b.active);
        assert_eq!(pool_a.cached_active_count, pool_b.cached_active_count);
    }

    // ── apply_updates tests ──────────────────────────────────────────────────

    /// `apply_updates` deactivates every slot in `updates.updates` and
    /// reactivates every slot in `updates.reactivations`. Mixed batches
    /// must leave the pool in the expected state and the cached counter
    /// must match the bitmap.
    #[test]
    fn apply_updates_applies_mixed_deactivate_and_reactivate() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0 active
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]); // slot 1 active
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]); // slot 2 active
        pool.add_cut(NodeId(0), 3, 0, 4.0, &[4.0]); // slot 3 active

        // Pre-deactivate slot 2 so the reactivation has something to flip.
        pool.deactivate(&[2]);
        assert_eq!(pool.active_count(), 3);
        assert!(!pool.active[2]);

        let updates = CutActivityUpdates {
            updates: vec![0, 3],    // deactivate slots 0 and 3
            reactivations: vec![2], // reactivate slot 2
        };
        pool.apply_updates(&updates);

        assert!(!pool.active[0], "slot 0 must be deactivated");
        assert!(pool.active[1], "slot 1 must remain active");
        assert!(pool.active[2], "slot 2 must be reactivated");
        assert!(!pool.active[3], "slot 3 must be deactivated");
        assert_eq!(pool.active_count(), 2);
        // populated is unaffected by activity changes.
        assert_eq!(pool.populated(), 4);
    }

    /// `apply_updates` must be idempotent: applying the same updates twice
    /// must leave the pool in the same state as applying it once. This
    /// follows from `set_active`'s "already-in-target-state is a no-op"
    /// contract.
    #[test]
    fn apply_updates_is_idempotent() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);

        // Pre-deactivate slot 1 so the reactivation has something to flip.
        pool.deactivate(&[1]);
        assert_eq!(pool.active_count(), 2);

        let updates = CutActivityUpdates {
            updates: vec![0],
            reactivations: vec![1],
        };

        // First application flips slot 0 off and slot 1 on.
        pool.apply_updates(&updates);
        let active_snapshot = pool.active.clone();
        let count_snapshot = pool.active_count();

        // Second application: every requested state already holds → no-op.
        pool.apply_updates(&updates);
        assert_eq!(
            pool.active, active_snapshot,
            "active bitmap must not change"
        );
        assert_eq!(
            pool.active_count(),
            count_snapshot,
            "active count must not change"
        );
        assert_eq!(pool.active_count(), 2);
        assert!(!pool.active[0]);
        assert!(pool.active[1]);
        assert!(pool.active[2]);
    }

    #[test]
    fn apply_updates_empty_is_noop() {
        use crate::cut_selection::CutActivityUpdates;

        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
        let active_before = pool.active.clone();
        let count_before = pool.active_count();

        let updates = CutActivityUpdates {
            updates: vec![],
            reactivations: vec![],
        };
        pool.apply_updates(&updates);

        assert_eq!(pool.active, active_before);
        assert_eq!(pool.active_count(), count_before);
    }

    /// `apply_updates` must match the behavior of calling `set_active`
    /// directly for each entry, regardless of which list (`updates` or
    /// `reactivations`) drives the change.
    #[test]
    fn apply_updates_matches_manual_set_active_loop() {
        use crate::cut_selection::CutActivityUpdates;

        let build_pool = || {
            let mut pool = CutPool::new(10, 1, 1, 0);
            pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
            pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
            pool.add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);
            pool.add_cut(NodeId(0), 3, 0, 4.0, &[4.0]);
            pool.deactivate(&[2]); // pre-deactivate so reactivation flips a bit
            pool
        };

        let updates = CutActivityUpdates {
            updates: vec![0, 3],
            reactivations: vec![2],
        };

        let mut pool_a = build_pool();
        pool_a.apply_updates(&updates);

        let mut pool_b = build_pool();
        for &slot in &updates.updates {
            pool_b.set_active(slot, false);
        }
        for &slot in &updates.reactivations {
            pool_b.set_active(slot, true);
        }

        assert_eq!(pool_a.active, pool_b.active);
        assert_eq!(pool_a.cached_active_count, pool_b.cached_active_count);
    }

    // ── read accessor equivalence tests ──────────────────────────────────────

    /// Every new read accessor must return exactly the value its corresponding
    /// direct field read would, on a pool with a mix of active and inactive
    /// populated slots.
    #[test]
    fn accessors_match_direct_field_reads() {
        let mut pool = CutPool::new(10, 3, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0, 2.0, 3.0]);
        pool.add_cut(NodeId(0), 1, 0, 20.0, &[4.0, 5.0, 6.0]);
        pool.add_cut(NodeId(0), 2, 0, 30.0, &[7.0, 8.0, 9.0]);
        pool.deactivate(&[1]);

        assert_eq!(pool.populated(), pool.populated_count);
        assert_eq!(pool.generated(), pool.generated_count);

        for slot in 0..pool.populated_count {
            assert_eq!(pool.is_active(slot), pool.active[slot], "slot {slot}");
            assert_eq!(pool.intercept(slot), pool.intercepts[slot], "slot {slot}");

            let start = slot * pool.state_dimension;
            assert_eq!(
                pool.coefficient_row(slot),
                &pool.coefficients[start..start + pool.state_dimension],
                "slot {slot}"
            );

            let via_accessor = pool.metadata(slot);
            let direct = &pool.metadata[slot];
            assert_eq!(
                via_accessor.iteration_generated, direct.iteration_generated,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.forward_pass_index, direct.forward_pass_index,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.active_count, direct.active_count,
                "slot {slot}"
            );
            assert_eq!(
                via_accessor.last_active_iter, direct.last_active_iter,
                "slot {slot}"
            );
        }
    }

    // ── grow ─────────────────────────────────────────────────────────────

    #[test]
    fn grow_preserves_populated_slot_positions_and_values() {
        let mut pool = CutPool::new(2, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[1.0, 2.0]);
        pool.add_cut(NodeId(0), 1, 0, 6.0, &[3.0, 4.0]);

        pool.grow(10);

        assert_eq!(pool.capacity, 10);
        assert_eq!(pool.populated(), 2, "growth does not touch populated_count");
        assert!(pool.is_active(0));
        assert_eq!(pool.intercept(0), 5.0);
        assert_eq!(pool.coefficient_row(0), &[1.0, 2.0]);
        assert!(pool.is_active(1));
        assert_eq!(pool.intercept(1), 6.0);
        assert_eq!(pool.coefficient_row(1), &[3.0, 4.0]);
    }

    #[test]
    fn grow_new_slots_are_writable_via_add_cut() {
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]); // slot 0
        pool.grow(5);
        // slot = 0 + 1*1 + 0 = 1 is already populated; grow one more forward
        // pass worth of stride to reach a genuinely new slot at iteration 4.
        pool.add_cut(NodeId(0), 4, 0, 9.0, &[7.0]); // slot 4, beyond the original capacity
        assert!(pool.is_active(4));
        assert_eq!(pool.intercept(4), 9.0);
        assert_eq!(pool.coefficient_row(4), &[7.0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is vacant")]
    fn activating_a_vacant_slot_panics_in_debug() {
        let mut pool = CutPool::new(4, 1, 2, 0);
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[1.0]);
        pool.set_active(1, true);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not scored")]
    fn loading_a_state_at_an_unscored_slot_panics_in_debug() {
        let mut pool = CutPool::new_with_warm_start(1, 1, 4, &[loaded_cut(1)]);
        pool.enable_trial_states();
        pool.load_trial_state(0, &[1.0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "new_capacity")]
    fn grow_to_a_smaller_capacity_panics_in_debug() {
        let mut pool = CutPool::new(10, 2, 1, 0);
        pool.grow(5);
    }
}
