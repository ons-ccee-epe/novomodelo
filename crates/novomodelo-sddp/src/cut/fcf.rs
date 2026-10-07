//! Future Cost Function (FCF) — all-pools container for Benders cuts.
//!
//! [`FutureCostFunction`] wraps one [`CutPool`] per pool id as a thin
//! orchestration layer over the training loop's cut operations. A pool id is
//! resolved from the node graph's `node → pool` map
//! ([`crate::setup::node_graph::NodeGraph::n_pools`],
//! `NodeRuntime.pool_id`); on the degenerate one-node-per-stage graph
//! (`nodes[]` absent), `pool_id == stage`, so every read below reduces
//! byte-for-byte to the pre-node-native stage read.
//!
//! ## Pool-id indexing
//!
//! FCF methods use **0-based** pool ids, resolved via the node graph — never a
//! bare stage number on a declared (non-chain) graph.
//!
//! ## Memory allocation
//!
//! All pool storage is pre-allocated by [`FutureCostFunction::new`] — no heap
//! allocation on the training-loop hot path.
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::fcf::FutureCostFunction;
//! use cobre_sddp::setup::NodeId;
//!
//! let mut fcf = FutureCostFunction::new(3, 4, 10, 50, &[0; 3]);
//! assert_eq!(fcf.pools.len(), 3);
//!
//! let coeffs = vec![1.0, 0.0, 0.0, 0.0];
//! fcf.add_cut(NodeId(0), 1, 0, 0, 5.0, &coeffs);
//!
//! let cuts: Vec<_> = fcf.active_cuts(1).collect();
//! assert_eq!(cuts.len(), 1);
//! assert_eq!(fcf.total_active_cuts(), 1);
//! assert_eq!(fcf.cuts_in_lp(1), 1);
//! ```

use super::pool::CutPool;
use crate::FullFcf;
use crate::PolicyLoadProof;
use crate::SddpError;
use crate::SddpError::Validation;
use crate::setup::NodeId;

use cobre_io::StageCutsReadResult;

/// All-pools container for the Future Cost Function (FCF): one [`CutPool`] per
/// pool id. Per-cut logic is delegated to [`CutPool`].
///
/// Each pool carries its own [`CutPool::state_dimension`] — its cut-slot-space
/// dimension, the length each stored cut's `coefficients` slice spans
/// ([`CutStateProjection::n_slots`](crate::indexer::CutStateProjection::n_slots));
/// a pool whose cuts span fewer state dimensions (a successor with
/// `inflow_lags: false`) sizes a smaller pool. [`Self::state_dimension`] remains
/// the global state-vector length (`StateSpace::n_state`) the checkpoint and
/// boundary paths validate on — the space `validate_policy_load`'s
/// `state_dimension` check operates in — equal to every pool's dimension when
/// all stages enable all dimensions (the default).
#[derive(Debug, Clone)]
pub struct FutureCostFunction {
    /// One cut pool per pool id, indexed 0-based — resolved from the node
    /// graph's `node → pool` map (`pool_id == stage` on the chain degeneracy).
    pub pools: Vec<CutPool>,

    /// Global state-vector length (`StateSpace::n_state`); a pool's own
    /// cut-slot-space [`CutPool::state_dimension`] may be smaller.
    pub state_dimension: usize,

    /// Forward passes per training iteration. Immutable after construction.
    pub forward_passes: u32,
}

impl FutureCostFunction {
    /// Construct a `FutureCostFunction` with pre-allocated pools, one per pool id,
    /// each sized `warm_start_counts[i] + max_iterations * forward_passes` via
    /// [`pool_capacity`] — the chain-degenerate case, where every pool's visit
    /// bound is `forward_passes` uniformly.
    ///
    /// # Parameters
    ///
    /// - `warm_start_counts`: per-pool warm-start cut counts; length must equal
    ///   `num_pools`. Pass `&vec![0; num_pools]` for the no-warm-start case.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::fcf::FutureCostFunction;
    ///
    /// let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
    /// assert_eq!(fcf.pools.len(), 5);
    /// // capacity = 0 + 100 * 10 = 1000
    /// assert_eq!(fcf.pools[0].capacity, 1000);
    /// ```
    #[must_use]
    pub fn new(
        num_pools: usize,
        state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        warm_start_counts: &[u32],
    ) -> Self {
        Self::new_per_pool(
            &vec![state_dimension; num_pools],
            state_dimension,
            forward_passes,
            max_iterations,
            warm_start_counts,
            &vec![u64::from(forward_passes); num_pools],
        )
    }

    /// Construct a `FutureCostFunction` sizing each pool id's [`CutPool`] by its
    /// own cut state dimension and its own per-iteration visit bound, for the
    /// per-pool `state_config` projection.
    ///
    /// `pool_state_dimensions[p]` is the dimension of pool `p` (the count from
    /// `CutStateProjection(successor_stage.state_config)` for non-terminal
    /// pools; the global `n_state` for the terminal pool). `global_state_dimension`
    /// is `StateSpace::n_state` — the value stored in [`Self::state_dimension`]
    /// and used by the checkpoint/boundary paths, independent of any reduced pool.
    ///
    /// `visit_bounds[p]` is pool `p`'s per-iteration visit bound — exact where
    /// the graph makes visits statically known (a chain node's is
    /// `forward_passes`; see
    /// [`NodeGraph::pool_cut_stride`](crate::setup::node_graph::NodeGraph::pool_cut_stride)
    /// and
    /// [`NodeGraph::forward_solve_counts`](crate::setup::node_graph::NodeGraph::forward_solve_counts))
    /// or an expected-value floor otherwise. [`pool_capacity`] sizes the pool
    /// from it, and each pool's own [`CutPool::visit_stride`] is set from the
    /// SAME value — capacity and stride must agree, or the slot formula can
    /// address past whichever is smaller.
    ///
    /// # Parameters
    ///
    /// - `warm_start_counts`: per-pool warm-start cut counts; length must equal
    ///   `pool_state_dimensions.len()`. Pass `&vec![0; n]` for no warm-start.
    /// - `visit_bounds`: per-pool visit bound; length must equal
    ///   `pool_state_dimensions.len()`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `warm_start_counts.len() != pool_state_dimensions.len()` or
    /// `visit_bounds.len() != pool_state_dimensions.len()`.
    #[must_use]
    pub fn new_per_pool(
        pool_state_dimensions: &[usize],
        global_state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        warm_start_counts: &[u32],
        visit_bounds: &[u64],
    ) -> Self {
        debug_assert_eq!(
            warm_start_counts.len(),
            pool_state_dimensions.len(),
            "warm_start_counts.len() ({}) != pool_state_dimensions.len() ({})",
            warm_start_counts.len(),
            pool_state_dimensions.len()
        );
        debug_assert_eq!(
            visit_bounds.len(),
            pool_state_dimensions.len(),
            "visit_bounds.len() ({}) != pool_state_dimensions.len() ({})",
            visit_bounds.len(),
            pool_state_dimensions.len()
        );

        let pools = warm_start_counts
            .iter()
            .zip(pool_state_dimensions)
            .zip(visit_bounds)
            .map(|((&wsc, &pool_dim), &visit_bound)| {
                let capacity = pool_capacity(wsc, max_iterations, visit_bound);
                // The live (sampled) arm caps visit_bound at forward_passes per
                // pool, and a chain's visit_bound equals forward_passes exactly
                // (pool_cut_stride), so this never truncates on that path.
                #[allow(clippy::cast_possible_truncation)]
                let stride = visit_bound as u32;
                CutPool::new(capacity, pool_dim, stride, wsc)
            })
            .collect();

        Self {
            pools,
            state_dimension: global_state_dimension,
            forward_passes,
        }
    }

    /// Reconstruct a `FutureCostFunction` from deserialized policy checkpoint data
    /// via [`CutPool::from_deserialized`] (tightly sized, no training capacity),
    /// each pool sized by the study's OWN `pool_state_dimensions[p]` — the same
    /// per-pool array [`Self::new_per_pool`] takes — never a single dimension
    /// broadcast to every pool.
    ///
    /// `_proof` is compile-time evidence from
    /// [`validate_policy_load`](crate::validate_policy_load) that
    /// `stage_results` already passed the [`FullFcf`] compatibility check; it is
    /// never read at runtime.
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] if `stage_results` is empty or its own
    /// per-stage `state_dimension` field is inconsistent across stages (a
    /// malformed checkpoint) — independent of `pool_state_dimensions`, which
    /// this study's setup pipeline already sizes.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool_state_dimensions.len() != stage_results.len()`.
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn from_deserialized(
        _proof: &PolicyLoadProof<FullFcf>,
        stage_results: &[StageCutsReadResult],
        pool_state_dimensions: &[usize],
    ) -> Result<Self, SddpError> {
        let state_dimension =
            validate_consistent_state_dimension("from_deserialized", stage_results)?;
        debug_assert_eq!(
            pool_state_dimensions.len(),
            stage_results.len(),
            "pool_state_dimensions.len() ({}) != stage_results.len() ({})",
            pool_state_dimensions.len(),
            stage_results.len()
        );

        let pools = stage_results
            .iter()
            .zip(pool_state_dimensions)
            .map(|(sr, &pool_dim)| CutPool::from_deserialized(pool_dim, &sr.cuts))
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes: 0,
        })
    }

    /// Build an FCF with warm-start cuts plus training capacity via
    /// [`CutPool::new_with_warm_start`], each pool sized by the study's OWN
    /// `pool_state_dimensions[p]` and given its OWN `visit_bounds[p]` stride —
    /// the same two per-pool arrays [`Self::new_per_pool`] takes, so the resume
    /// path builds pools identically to the cold path. Neither array may be
    /// omitted in favor of a scalar broadcast to every pool: a resumed pool's
    /// stride must equal the value the cold-start run derived for it
    /// (`NodeGraph::pool_cut_stride`), never `forward_passes` uniformly.
    ///
    /// `_proof` is compile-time evidence from
    /// [`validate_policy_load`](crate::validate_policy_load) that
    /// `stage_results` already passed the [`FullFcf`] compatibility check; it is
    /// never read at runtime. Omitting it, or handing a
    /// [`BoundaryInjection`](crate::BoundaryInjection)-typed proof instead,
    /// fails to compile.
    ///
    /// ```compile_fail
    /// use cobre_sddp::FutureCostFunction;
    ///
    /// fn call_without_proof(stage_results: &[cobre_io::StageCutsReadResult]) {
    ///     let _ = FutureCostFunction::new_with_warm_start(stage_results, &[], &[], 4, 10);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use cobre_sddp::{BoundaryInjection, FutureCostFunction, PolicyLoadProof};
    ///
    /// fn call_with_wrong_kind(
    ///     proof: &PolicyLoadProof<BoundaryInjection>,
    ///     stage_results: &[cobre_io::StageCutsReadResult],
    /// ) {
    ///     let _ = FutureCostFunction::new_with_warm_start(proof, stage_results, &[], &[], 4, 10);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] if `stage_results` is empty or its own
    /// per-stage `state_dimension` field is inconsistent across stages.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool_state_dimensions.len() != stage_results.len()` or
    /// `visit_bounds.len() != stage_results.len()`.
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn new_with_warm_start(
        _proof: &PolicyLoadProof<FullFcf>,
        stage_results: &[StageCutsReadResult],
        pool_state_dimensions: &[usize],
        visit_bounds: &[u64],
        forward_passes: u32,
        max_iterations: u64,
    ) -> Result<Self, SddpError> {
        let state_dimension =
            validate_consistent_state_dimension("new_with_warm_start", stage_results)?;
        debug_assert_eq!(
            pool_state_dimensions.len(),
            stage_results.len(),
            "pool_state_dimensions.len() ({}) != stage_results.len() ({})",
            pool_state_dimensions.len(),
            stage_results.len()
        );
        debug_assert_eq!(
            visit_bounds.len(),
            stage_results.len(),
            "visit_bounds.len() ({}) != stage_results.len() ({})",
            visit_bounds.len(),
            stage_results.len()
        );

        let pools = stage_results
            .iter()
            .zip(pool_state_dimensions)
            .zip(visit_bounds)
            .map(|((sr, &pool_dim), &visit_bound)| {
                // The live (sampled) arm caps visit_bound at forward_passes per
                // pool, and a chain's visit_bound equals forward_passes exactly
                // (pool_cut_stride), so this never truncates on that path — the
                // same cap `new_per_pool` relies on.
                #[allow(clippy::cast_possible_truncation)]
                let stride = visit_bound as u32;
                CutPool::new_with_warm_start(pool_dim, stride, max_iterations, &sr.cuts)
            })
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes,
        })
    }

    /// Insert a Benders cut at the given (0-based) pool id, generated at node
    /// `node_id` (`NodeGraph::node_ids`). `coefficients` is the state gradient;
    /// length must be `state_dimension`.
    ///
    /// `node_id` is recorded as provenance (the generating node, distinct per
    /// node even when several nodes share `pool`) and never affects which slot
    /// the cut lands in — the append-only, slot-identity contract is unchanged.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()` or if `coefficients.len() !=
    /// state_dimension`.
    pub fn add_cut(
        &mut self,
        node_id: NodeId,
        pool: usize,
        iteration: u64,
        forward_pass_index: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].add_cut(
            node_id,
            iteration,
            forward_pass_index,
            intercept,
            coefficients,
        );
    }

    /// Iterate over active cuts at the given (0-based) pool id as
    /// `(slot_index, intercept, coefficient_slice)`, for LP construction.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()`.
    pub fn active_cuts(&self, pool: usize) -> impl Iterator<Item = (usize, f64, &[f64])> {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].active_cuts()
    }

    /// Evaluate the FCF at `values` for the given (0-based) pool id: the max over
    /// active cuts, or [`f64::NEG_INFINITY`] when the pool has none. `values`
    /// length must be `state_dimension`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()` or if `values.len() !=
    /// state_dimension`.
    #[must_use]
    pub fn evaluate_at_state(&self, pool: usize, values: &[f64]) -> f64 {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].evaluate_at_state(values)
    }

    /// Return the total active-cut count across all pools.
    #[must_use]
    pub fn total_active_cuts(&self) -> usize {
        self.pools.iter().map(CutPool::active_count).sum()
    }

    /// Return the total cuts ever generated across all pools (sums
    /// `generated_count`), excluding reserved-but-unwritten slots — the true
    /// policy-row count.
    #[must_use]
    pub fn total_generated_cuts(&self) -> usize {
        self.pools.iter().map(|p| p.generated_count).sum()
    }

    /// Return the total warm-start (loaded boundary) cuts across all pools
    /// (sums `warm_start_count`) — the subset of [`Self::total_generated_cuts`]
    /// that predates this run rather than being produced by it.
    #[must_use]
    pub fn total_warm_start_cuts(&self) -> usize {
        self.pools.iter().map(|p| p.warm_start_count as usize).sum()
    }

    /// Propagate [`CutPool::set_iteration_base`] to every pool. Call once before
    /// training with `start_iteration + 1` for dense slot packing.
    pub fn set_iteration_base(&mut self, iteration_base: u64) {
        for pool in &mut self.pools {
            pool.set_iteration_base(iteration_base);
        }
    }

    /// Deactivate the cuts at `indices` for the given (0-based) pool id.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()`.
    pub fn deactivate(&mut self, pool: usize, indices: &[u32]) {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].deactivate(indices);
    }

    /// Toggle a single slot's activity flag at the given (0-based) pool id via
    /// [`CutPool::set_active`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()` or if `slot >= pools[pool].populated()`.
    ///
    /// [`CutPool::set_active`]: super::pool::CutPool::set_active
    pub fn set_active(&mut self, pool: usize, slot: u32, active: bool) {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].set_active(slot, active);
    }

    /// Return the LP-row count (populated slots, independent of activity) for the
    /// given (0-based) pool id via [`CutPool::cuts_in_lp`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `pool >= pools.len()`.
    ///
    /// [`CutPool::cuts_in_lp`]: super::pool::CutPool::cuts_in_lp
    #[must_use]
    pub fn cuts_in_lp(&self, pool: usize) -> usize {
        debug_assert!(
            pool < self.pools.len(),
            "pool id {pool} is out of bounds (num_pools = {})",
            self.pools.len()
        );
        self.pools[pool].cuts_in_lp()
    }

    /// One [`SparsityReport`] per pool, in pool-id order.
    ///
    /// [`SparsityReport`]: super::pool::SparsityReport
    #[must_use]
    pub fn sparsity_reports(&self) -> Vec<super::pool::SparsityReport> {
        self.pools.iter().map(CutPool::sparsity_report).collect()
    }
}

/// One pool's capacity: `warm_start_count + max_iterations * visit_bound` —
/// exact where `visit_bound` is a chain's `forward_passes` or a declared
/// graph's static per-node visit count, an expected-value-plus-margin floor
/// where it is not (sampled general, inert until a general-graph traversal
/// supplies realized visit counts). `u64` arithmetic guards the product
/// against overflow before the `usize` cast.
#[allow(clippy::cast_possible_truncation)]
fn pool_capacity(warm_start_count: u32, max_iterations: u64, visit_bound: u64) -> usize {
    (u64::from(warm_start_count) + max_iterations * visit_bound) as usize
}

fn validate_consistent_state_dimension(
    context: &str,
    stage_results: &[StageCutsReadResult],
) -> Result<usize, SddpError> {
    if stage_results.is_empty() {
        return Err(Validation(format!("{context}: stage_results is empty")));
    }

    let state_dimension = stage_results[0].state_dimension as usize;
    for sr in &stage_results[1..] {
        if sr.state_dimension as usize != state_dimension {
            return Err(Validation(format!(
                "{context}: inconsistent state_dimension: stage {} has {}, \
                 expected {} (from stage {})",
                sr.stage_id, sr.state_dimension, state_dimension, stage_results[0].stage_id
            )));
        }
    }

    Ok(state_dimension)
}

#[cfg(test)]
mod tests {
    use super::{FutureCostFunction, NodeId};

    #[test]
    fn new_creates_correct_number_of_pools() {
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn new_each_pool_has_correct_capacity_no_warmstart() {
        // capacity = 0 + 100 * 10 = 1000
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1000);
            assert_eq!(pool.state_dimension, 9);
            assert_eq!(pool.visit_stride, 10);
            assert_eq!(pool.warm_start_count, 0);
        }
    }

    #[test]
    fn new_each_pool_has_correct_capacity_with_warmstart() {
        // capacity = 5 + 100 * 10 = 1005
        let fcf = FutureCostFunction::new(3, 4, 10, 100, &[5; 3]);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1005);
            assert_eq!(pool.warm_start_count, 5);
        }
    }

    #[test]
    fn new_all_pools_start_with_zero_active_cuts() {
        let fcf = FutureCostFunction::new(4, 3, 5, 20, &[0; 4]);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn new_zero_pools_is_valid() {
        let fcf = FutureCostFunction::new(0, 4, 5, 10, &[]);
        assert_eq!(fcf.pools.len(), 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn new_non_uniform_warm_start_counts_per_pool_capacity() {
        // warm_start_counts = [5, 3, 0], max_iterations = 10, forward_passes = 2
        // capacity[0] = 5 + 10*2 = 25
        // capacity[1] = 3 + 10*2 = 23
        // capacity[2] = 0 + 10*2 = 20
        let fcf = FutureCostFunction::new(3, 4, 2, 10, &[5, 3, 0]);
        assert_eq!(fcf.pools[0].capacity, 25);
        assert_eq!(fcf.pools[0].warm_start_count, 5);
        assert_eq!(fcf.pools[1].capacity, 23);
        assert_eq!(fcf.pools[1].warm_start_count, 3);
        assert_eq!(fcf.pools[2].capacity, 20);
        assert_eq!(fcf.pools[2].warm_start_count, 0);
    }

    #[test]
    fn new_uniform_zero_counts_matches_old_scalar_zero_behavior() {
        let fcf = FutureCostFunction::new(3, 4, 2, 10, &[0, 0, 0]);
        for pool in &fcf.pools {
            // capacity = 0 + 10*2 = 20
            assert_eq!(pool.capacity, 20);
            assert_eq!(pool.warm_start_count, 0);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "warm_start_counts.len()")]
    fn new_mismatched_length_panics_in_debug() {
        // Length 2 != num_pools 3: should trigger debug_assert_eq!
        let _ = FutureCostFunction::new(3, 4, 2, 10, &[0, 0]);
    }

    #[test]
    fn add_cut_and_active_cuts_round_trip_at_specific_pool() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, &[0; 5]);
        let coeffs = [3.0, 7.0];
        fcf.add_cut(NodeId(0), 2, 0, 0, 42.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
        let (_, intercept, c) = active[0];
        assert_eq!(intercept, 42.0);
        assert_eq!(c, &[3.0, 7.0]);
    }

    #[test]
    fn active_cuts_at_other_pool_returns_empty() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, &[0; 5]);
        fcf.add_cut(NodeId(0), 2, 0, 0, 42.0, &[1.0, 2.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn add_cut_multiple_pools_are_independent() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 10, &[0; 4]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(NodeId(0), 3, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.active_cuts(0).count(), 1);
        assert_eq!(fcf.active_cuts(1).count(), 1);
        assert_eq!(fcf.active_cuts(2).count(), 0);
        assert_eq!(fcf.active_cuts(3).count(), 1);
    }

    #[test]
    fn evaluate_at_state_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 2, 1, 10, &[0; 3]);
        // pool 1: cut with intercept=10, coeffs=[1,0]
        fcf.add_cut(NodeId(0), 1, 0, 0, 10.0, &[1.0, 0.0]);
        // pool 2: cut with intercept=5, coeffs=[0,2]
        fcf.add_cut(NodeId(0), 2, 0, 0, 5.0, &[0.0, 2.0]);

        // pool 1: 10 + 1*3 + 0*4 = 13
        assert_eq!(fcf.evaluate_at_state(1, &[3.0, 4.0]), 13.0);
        // pool 2: 5 + 0*3 + 2*4 = 13
        assert_eq!(fcf.evaluate_at_state(2, &[3.0, 4.0]), 13.0);
        // pool 0: no cuts → NEG_INFINITY
        assert_eq!(fcf.evaluate_at_state(0, &[3.0, 4.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn total_active_cuts_sums_across_pools() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 20, &[0; 4]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(NodeId(0), 1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(NodeId(0), 3, 0, 0, 4.0, &[4.0]);

        // pool 0: 1, pool 1: 2, pool 2: 0, pool 3: 1 → total = 4
        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn total_active_cuts_reflects_deactivation() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]); // slot 0
        fcf.add_cut(NodeId(0), 0, 1, 0, 2.0, &[2.0]); // slot 1
        fcf.add_cut(NodeId(0), 1, 0, 0, 3.0, &[3.0]); // slot 0

        assert_eq!(fcf.total_active_cuts(), 3);
        fcf.deactivate(0, &[0]);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn total_warm_start_cuts_sums_across_pools() {
        let fcf = FutureCostFunction::new(4, 1, 1, 20, &[0, 3, 0, 7]);
        assert_eq!(fcf.total_warm_start_cuts(), 10);
    }

    #[test]
    fn total_warm_start_cuts_is_unaffected_by_cuts_added_this_run() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 20, &[0, 5]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 2.0, &[2.0]);

        // warm_start_count is fixed at construction; only generated_count grows.
        assert_eq!(fcf.total_warm_start_cuts(), 5);
        assert_eq!(fcf.total_generated_cuts(), 7);
    }

    #[test]
    fn set_iteration_base_propagates_and_packs_densely() {
        let mut fcf = FutureCostFunction::new(3, 1, 2, 10, &[0; 3]);
        fcf.set_iteration_base(1);
        for pool in &fcf.pools {
            assert_eq!(pool.iteration_base, 1);
        }
        // With base 1, iteration 1 maps to slot 0: no reserved leading block
        // (base 0 would place these at slots 2,3 with populated_count 4).
        fcf.add_cut(NodeId(0), 0, 1, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 0, 1, 1, 2.0, &[1.0]);
        assert_eq!(fcf.pools[0].populated(), 2);
        assert_eq!(fcf.pools[0].generated_count, 2);
    }

    #[test]
    fn deactivate_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 1, 1, 10, &[0; 3]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 10.0, &[1.0]); // slot 0 of pool[1]
        fcf.add_cut(NodeId(0), 1, 1, 0, 20.0, &[2.0]); // slot 1 of pool[1]
        fcf.add_cut(NodeId(0), 2, 0, 0, 30.0, &[3.0]); // slot 0 of pool[2]

        fcf.deactivate(1, &[0]);

        // pool[1] should now have 1 active cut
        assert_eq!(fcf.active_cuts(1).count(), 1);
        // pool[2] should still have 1 active cut
        assert_eq!(fcf.active_cuts(2).count(), 1);
    }

    #[test]
    fn ac_new_5_pools_pools_len_is_5() {
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn ac_active_cuts_at_pool_with_cut_yields_it() {
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, &[0; 5]);
        let coeffs = [1.0, 2.0, 3.0];
        fcf.add_cut(NodeId(0), 2, 0, 0, 99.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn ac_active_cuts_at_different_pool_yields_none() {
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, &[0; 5]);
        fcf.add_cut(NodeId(0), 2, 0, 0, 99.0, &[1.0, 2.0, 3.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn ac_total_active_cuts_is_sum_across_pools() {
        let mut fcf = FutureCostFunction::new(5, 1, 1, 10, &[0; 5]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(NodeId(0), 1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(NodeId(0), 4, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn fcf_derives_debug_and_clone() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 5, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 7.0, &[1.0, 2.0]);

        let cloned = fcf.clone();
        assert_eq!(cloned.total_active_cuts(), 1);
        assert_eq!(cloned.evaluate_at_state(0, &[0.0, 0.0]), 7.0);

        let debug_str = format!("{fcf:?}");
        assert!(!debug_str.is_empty());
    }

    // ── from_deserialized tests ──────────────────────────────────────────

    fn make_record(
        intercept: f64,
        coefficients: Vec<f64>,
        is_active: bool,
    ) -> cobre_io::OwnedPolicyCutRecord {
        cobre_io::OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active,
        }
    }

    fn make_stage(
        stage_id: u32,
        state_dimension: u32,
        cuts: Vec<cobre_io::OwnedPolicyCutRecord>,
    ) -> cobre_io::StageCutsReadResult {
        let populated_count = u32::try_from(cuts.len()).expect("cuts count fits in u32");
        cobre_io::StageCutsReadResult {
            stage_id,
            state_dimension,
            capacity: populated_count,
            warm_start_count: 0,
            populated_count,
            cuts,
            entity_manifest: Vec::new(),
            cost_scale_factor: None,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: cobre_io::STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        }
    }

    #[test]
    fn from_deserialized_empty_input_returns_err() {
        let proof = crate::test_support::trivial_full_fcf_proof(0, 0);
        let result = FutureCostFunction::from_deserialized(&proof, &[], &[]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty"), "{msg}");
    }

    #[test]
    fn from_deserialized_inconsistent_dimensions_returns_err() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 3, vec![make_record(2.0, vec![1.0, 0.0, 0.0], true)]),
        ];
        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let result = FutureCostFunction::from_deserialized(&proof, &stages, &[2, 3]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("inconsistent"), "{msg}");
    }

    #[test]
    fn from_deserialized_preserves_active_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
                make_record(3.0, vec![1.0, 1.0], true),
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages, &[2]).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        assert_eq!(fcf.total_active_cuts(), 2);
        assert_eq!(fcf.pools[0].populated(), 3);
    }

    #[test]
    fn from_deserialized_evaluate_at_state_matches_original() {
        // Build original FCF with known cuts, then reconstruct via deserialized.
        let mut original = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        original.add_cut(NodeId(0), 0, 0, 0, 10.0, &[1.0, 0.0]);
        original.add_cut(NodeId(0), 0, 1, 0, 5.0, &[0.0, 2.0]);
        original.add_cut(NodeId(0), 1, 0, 0, 3.0, &[1.0, 1.0]);

        let state = [3.0, 4.0];
        let orig_val_s0 = original.evaluate_at_state(0, &state);
        let orig_val_s1 = original.evaluate_at_state(1, &state);

        // Construct deserialized data matching the original cuts.
        let stages = vec![
            make_stage(
                0,
                2,
                vec![
                    make_record(10.0, vec![1.0, 0.0], true),
                    make_record(5.0, vec![0.0, 2.0], true),
                ],
            ),
            make_stage(1, 2, vec![make_record(3.0, vec![1.0, 1.0], true)]),
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let reconstructed =
            FutureCostFunction::from_deserialized(&proof, &stages, &[2, 2]).unwrap();
        assert_eq!(reconstructed.evaluate_at_state(0, &state), orig_val_s0);
        assert_eq!(reconstructed.evaluate_at_state(1, &state), orig_val_s1);
    }

    #[test]
    fn from_deserialized_empty_stage_is_valid() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]), // empty stage
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages, &[2, 2]).unwrap();
        assert_eq!(fcf.pools.len(), 2);
        assert_eq!(fcf.pools[1].capacity, 0);
        assert_eq!(fcf.pools[1].active_count(), 0);
        assert_eq!(fcf.evaluate_at_state(1, &[1.0, 1.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn from_deserialized_single_cut_stage() {
        let stages = vec![make_stage(
            0,
            3,
            vec![make_record(7.0, vec![1.0, 2.0, 3.0], true)],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(3, 1);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages, &[3]).unwrap();
        assert_eq!(fcf.state_dimension, 3);
        assert_eq!(fcf.total_active_cuts(), 1);
        // 7 + 1*1 + 2*2 + 3*3 = 7 + 1 + 4 + 9 = 21
        assert_eq!(fcf.evaluate_at_state(0, &[1.0, 2.0, 3.0]), 21.0);
    }

    // ── new_with_warm_start tests ────────────────────────────────────────

    #[test]
    fn warm_start_capacity_includes_training_slots() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], true),
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf =
            FutureCostFunction::new_with_warm_start(&proof, &stages, &[2], &[4], 4, 10).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        // capacity = 2 warm-start + 10*4 training = 42
        assert_eq!(fcf.pools[0].capacity, 42);
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.pools[0].visit_stride, 4);
        assert_eq!(fcf.pools[0].populated(), 2);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn warm_start_training_cuts_at_correct_offset() {
        let stages = vec![make_stage(0, 1, vec![make_record(10.0, vec![1.0], true)])];

        let proof = crate::test_support::trivial_full_fcf_proof(1, 1);
        let mut fcf =
            FutureCostFunction::new_with_warm_start(&proof, &stages, &[1], &[2], 2, 5).unwrap();
        // warm_start_count = 1, forward_passes = 2
        // Training cut at iteration=0, fwd_idx=0: slot = 1 + 0*2 + 0 = 1
        fcf.add_cut(NodeId(0), 0, 0, 0, 20.0, &[2.0]);
        // Training cut at iteration=0, fwd_idx=1: slot = 1 + 0*2 + 1 = 2
        fcf.add_cut(NodeId(0), 0, 0, 1, 30.0, &[3.0]);

        assert_eq!(fcf.total_active_cuts(), 3);
        assert_eq!(fcf.pools[0].populated(), 3);
        // Warm-start cut at slot 0
        assert_eq!(fcf.pools[0].intercept(0), 10.0);
        // Training cuts at slots 1 and 2
        assert_eq!(fcf.pools[0].intercept(1), 20.0);
        assert_eq!(fcf.pools[0].intercept(2), 30.0);
    }

    #[test]
    fn warm_start_empty_stage_has_training_capacity() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]),
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let fcf = FutureCostFunction::new_with_warm_start(&proof, &stages, &[2, 2], &[3, 3], 3, 5)
            .unwrap();
        // Stage 0: warm_start=1, capacity=1+15=16
        assert_eq!(fcf.pools[0].capacity, 16);
        assert_eq!(fcf.pools[0].warm_start_count, 1);
        // Stage 1: warm_start=0, capacity=0+15=15
        assert_eq!(fcf.pools[1].capacity, 15);
        assert_eq!(fcf.pools[1].warm_start_count, 0);
    }

    #[test]
    fn warm_start_preserves_inactive_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf =
            FutureCostFunction::new_with_warm_start(&proof, &stages, &[2], &[1], 1, 5).unwrap();
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.total_active_cuts(), 1); // only the active one
    }

    // ── set_active tests ────────────────────────────────────────────────────

    #[test]
    fn fcf_set_active_delegates_to_pool() {
        let mut fcf = FutureCostFunction::new(3, 1, 1, 10, &[0; 3]);
        // Add 3 cuts to stage 1: slots 0, 1, 2
        fcf.add_cut(NodeId(0), 1, 0, 0, 10.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 1, 0, 20.0, &[2.0]);
        fcf.add_cut(NodeId(0), 1, 2, 0, 30.0, &[3.0]);
        let prior = fcf.total_active_cuts();

        fcf.set_active(1, 0, false);

        assert!(!fcf.pools[1].is_active(0));
        assert_eq!(fcf.total_active_cuts(), prior - 1);
    }

    // ── new_per_pool tests ──────────────────────────────────────────────────

    #[test]
    fn new_per_pool_sizes_each_pool_by_its_own_dimension() {
        // Three pools with distinct dimensions (e.g. a declared graph's
        // per-pool successor `state_config` projections), a global
        // `state_dimension` that need not match any single pool's,
        // non-uniform warm-start counts, and non-uniform visit bounds
        // (independent of both the dimension and the global forward_passes).
        let pool_state_dimensions = [2usize, 5usize, 1usize];
        let warm_start_counts = [1u32, 0u32, 3u32];
        let visit_bounds = [4u64, 2u64, 6u64];
        let fcf = FutureCostFunction::new_per_pool(
            &pool_state_dimensions,
            9,
            4,
            10,
            &warm_start_counts,
            &visit_bounds,
        );

        assert_eq!(fcf.pools.len(), 3);
        assert_eq!(
            fcf.state_dimension, 9,
            "global dimension is stored verbatim"
        );
        for (pool, ((&dim, &wsc), &vb)) in fcf.pools.iter().zip(
            pool_state_dimensions
                .iter()
                .zip(&warm_start_counts)
                .zip(&visit_bounds),
        ) {
            assert_eq!(
                pool.state_dimension, dim,
                "each pool keeps its own dimension"
            );
            assert_eq!(pool.warm_start_count, wsc);
            // capacity = warm_start_count + max_iterations * visit_bound
            assert_eq!(pool.capacity, wsc as usize + 10 * vb as usize);
            assert_eq!(pool.visit_stride, vb as u32);
        }
    }
}
