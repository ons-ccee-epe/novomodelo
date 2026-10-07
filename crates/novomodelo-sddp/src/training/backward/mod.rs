//! Backward pass execution for the SDDP training loop.
//!
//! Sweeps stages in reverse, evaluating the cost-to-go at each trial point
//! assigned to this rank, extracting LP duals into Benders cut coefficients, and
//! aggregating per-opening outcomes via
//! [`RiskMeasure::aggregate_cut`](crate::RiskMeasure::aggregate_cut) into one cut
//! per trial point per stage inserted into the [`crate::FutureCostFunction`].
//!
//! Each rank processes only its own forward-pass assignments to avoid generating
//! duplicate cuts; cut synchronization (`allgatherv`) distributes them to all
//! ranks afterward.
//!
//! ## Stage indexing convention
//!
//! The backward pass generates a cut **at stage `t`** by solving the LP
//! **at stage `t + 1`** (the successor) under each opening noise vector from
//! that successor stage. The opening tree provides noise at `t + 1`.
//!
//! ## Cut coefficient formula
//!
//! `coefficients = reduced_cost` (raw, no sign flip at extraction); the LP cut row
//! negates it in `cut::row::build_cut_row_batch_into`. The full subgradient
//! contract (`pi[i] = reduced_cost[col_i] / col_scale[col_i]`, divided not
//! multiplied) lives in `duals_extraction` and sddp.md "Benders cut sign &
//! subgradient extraction".
//!
//! ### Anticipated-ring cut gradient flow
//!
//! Anticipated-ring slots resolve by identity (`state_to_lp_column`, the
//! `transit_buckets_out` convention): the in-LP ring's definition rows — a
//! same-slot carry (`out − in = 0`) for a not-yet-due slot, the
//! delivery-decision deposit into the slot of a decision's own delivery target
//! (`delivery_stage mod k_max`) — resolve the ring transition, so cuts apply
//! directly against the outgoing `commit_out` column. The fishing
//! constraint is emitted at every stage unconditionally, so every slot
//! participates in the dual chain. See the `StateSpace::state_to_lp_column`
//! rustdoc.
//!
//! ## Cut activity tracking
//!
//! After each backward solve, the duals of the appended cut rows are inspected
//! to determine which existing cuts at the successor stage are binding. The
//! metadata of binding cuts is updated in-place so that cut selection
//! strategies have accurate activity counts at the end of the iteration.
//!
//! ## Thread-level parallelism
//!
//! The outer per-stage loop is sequential (stage `t` depends on cuts generated at
//! stage `t+1`); the inner trial-point loop is parallelised across
//! [`SolverWorkspace`](crate::workspace::SolverWorkspace) instances with static
//! scenario partitioning. Each worker generates cuts into a thread-local
//! `StagedCut` buffer, sorted by `trial_state_idx` after the parallel region for
//! deterministic FCF insertion regardless of thread completion order.

use cobre_solver::{RowBatch, StageTemplate};

use crate::{
    context::{StageContext, TrainingContext},
    cut::{FutureCostFunction, pool::CutPool},
    forward::build_delta_cut_row_batch_into,
    indexer::CutStateProjection,
    setup::node_graph::{NodeId, NodeOpenings, NodePos, StageIdx},
    solver_stats::SolverStatsDelta,
};

use std::ops::Range;

mod by_node;
mod by_scenario;
mod duals_extraction;
mod lp_setup;
mod outcome_aggregation;
mod replicated;

#[cfg(test)]
mod tests;

pub(crate) use by_node::{
    OpeningOutcome, by_node_block_count, by_node_finish, hardest_first_block_order,
    identity_block_order, merge_block_pivots, process_stage_backward_by_node, resolve_block_size,
};
pub(crate) use by_scenario::{
    StageOpeningSolver, by_scenario_finish, process_by_scenario_backward,
};
pub(crate) use duals_extraction::extract_state_duals_only;
pub(crate) use lp_setup::fill_external_opening_noise;
pub(crate) use replicated::{ReplicatedScratch, run_backward_node_replicated};

#[cfg(test)]
pub(crate) use lp_setup::{load_backward_lp, patch_opening_bounds, resolve_backward_basis};

/// `any(test, feature = "test-support")`, not the bare `#[cfg(test)]` above:
/// an external `tests/` binary drives this through `test_support` under
/// `--features test-support`, which never sets `cfg(test)` on this library.
#[cfg(any(test, feature = "test-support"))]
pub(crate) use outcome_aggregation::write_opening_outcome;

/// Per-`(rank, worker_id, opening)` solver delta collected during a single
/// backward stage, as returned inside [`BackwardResult::stage_stats`].
///
/// Layout: `(rank, worker_id, opening_index, delta)`.
pub type StageWorkerOpeningDelta = (i32, i32, usize, SolverStatsDelta);

/// Result produced by the backward pass on a single rank.
///
/// The per-worker timing data carried inside `stage_stats` is keyed
/// by the `WORKER_TIMING_SLOT_*` constants exported from
/// `cobre-core`. New per-worker timing slots should be added to
/// that constant set (and the `WORKER_TIMING_SLOT_COUNT` updated)
/// rather than as standalone fields on this struct, so the parquet
/// timing schema picks them up automatically.
#[derive(Debug, Clone)]
#[must_use]
pub struct BackwardResult {
    /// Cuts generated, summed across all ranks (rank-count invariant).
    pub cuts_generated: usize,

    /// This rank's wall time (milliseconds).
    pub elapsed_ms: u64,

    /// LP solves performed.
    pub lp_solves: u64,

    /// Per-stage, per-`(rank, worker_id, opening)` solver statistics deltas.
    ///
    /// Each outer entry is `(successor_stage_index, per_worker_opening_deltas)`;
    /// the inner `Vec` element is `(rank, worker_id, omega, delta)`. The sampled
    /// path gathers one entry per `(MPI rank, rayon worker, opening)` triple via
    /// `allgatherv`, filtered to `omega < n_openings(successor)` and
    /// `delta.lp_solves > 0 || omega == 0` (the omega=0 "stage visited"
    /// sentinel), in reverse stage order. The enumerated path has no
    /// per-worker/per-opening breakdown: one `(rank, worker_id, 0, delta)` entry
    /// per visited successor stage (`delta.lp_solves > 0`), in ascending stage
    /// order.
    pub stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)>,

    /// State exchange time, accumulated across all stages (milliseconds).
    pub state_exchange_time_ms: u64,

    /// `build_cut_row_batch_into` time, accumulated across all stages (milliseconds).
    pub cut_batch_build_time_ms: u64,

    /// Aggregate non-solve work inside the parallel region, accumulated across
    /// all stages (milliseconds). Computed per-stage as the sum over all workers of
    /// `load_model_time_ms + set_bounds_time_ms + basis_set_time_ms`.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead, accumulated across all
    /// stages (milliseconds). Computed per-stage as `max_worker_total_ms - avg_worker_total_ms`,
    /// where `worker_total_ms = solve + load_model + set_bounds + basis_set` for each worker.
    pub load_imbalance_ms: u64,

    /// Rayon scheduling overhead, accumulated across all stages (milliseconds).
    /// Computed per-stage as `parallel_wall_ms - max_worker_total_ms`.
    pub scheduling_overhead_ms: u64,

    /// Per-stage cut synchronization time, accumulated across all stages (milliseconds).
    pub cut_sync_time_ms: u64,
}

/// Per-thread staging buffer for one aggregated cut produced at a single trial
/// point during the parallel backward sweep.
///
/// Each worker thread populates one `StagedCut` per trial point instead of
/// writing directly into the `FutureCostFunction`. After the parallel region,
/// staged cuts are sorted by `trial_state_idx` and merged into the FCF in
/// deterministic order regardless of thread completion order.
pub(crate) struct StagedCut {
    /// Local trial-point index within `0..local_work`. Used for deterministic
    /// merge ordering after the parallel region.
    pub(crate) trial_state_idx: usize,

    /// Aggregated cut intercept (result of `RiskMeasure::aggregate_cut`).
    pub(crate) intercept: f64,

    /// Range into the producing worker's `agg_arena` holding this cut's
    /// aggregated coefficients (length = `n_state`). The arena is owned by the
    /// rayon worker that produced this cut; the FCF merge resolves the slice
    /// after the parallel region returns via
    /// `workspaces[w].backward_accum.agg_arena[coefficients_range]`.
    pub(crate) coefficients_range: Range<usize>,

    /// Global forward-pass index (`fwd_offset + m`), stored as `u32` for the
    /// FCF slot formula.
    pub(crate) forward_pass_index: u32,
}

/// Node-level backward-pass arguments at one cut-generating node, shared across
/// every successor child. The per-child successor data lives in
/// [`SuccessorOutcomes`]; a child is priced against ITS OWN LP, never child 0's.
pub(crate) struct SuccessorSpec<'a> {
    /// Stage index being cut (the stage whose cost-to-go we are computing).
    pub(crate) t: StageIdx,
    /// Successor stage index (`t + 1`), where the LPs are solved. Every child of
    /// the node sits at this one stage (`t -> t+1`, no stage-skipping — validation
    /// rule 39), so it is node-level, not per-child.
    pub(crate) successor: StageIdx,
    /// This rank's MPI rank index (used to address exchange buffer state).
    pub(crate) my_rank: usize,
    /// Product weights `P(n→m) · q_{m,ψ}` over the current node's successor
    /// outcome set `O(n) = {(m, ψ): m ∈ n⁺, ψ ∈ Ω_m}`, flattened in canonical
    /// order (ascending child node id, then within-child ω) — the same order the
    /// [`SuccessorOutcomes`] entries and their `outcome_range`s follow.
    pub(crate) probabilities: &'a [f64],
    /// Cut-state projection for the pool this stage's cut is inserted into
    /// (the pool id resolved from the node graph for the node at stage `t`,
    /// sized from its successor's `state_config`): the LP incoming-state
    /// columns dual extraction reads. `n_slots()` equals every successor pool's
    /// `state_dimension` (all children share one stage and `state_config`).
    pub(crate) cut_state: &'a CutStateProjection,
}

/// Owned per-child metadata for one successor outcome. Borrow-free so it lives in a
/// reused per-node buffer (no hot-path allocation); the pool-indexed LP data
/// (frozen template, delta cut batch, active slots, cut pool) is resolved on demand
/// by [`SuccessorOutcomes::child`].
pub(crate) struct SuccessorEntry {
    /// Child's canonical node position (`NodeGraph::nodes` index) — the warm-start
    /// `BasisStore` node-axis key. On the chain degeneracy equal to the successor stage.
    pub(crate) successor_node: NodePos,
    /// Child's declared node id (`NodeGraph::node_ids[successor_node]`).
    pub(crate) successor_node_id: NodeId,
    /// Child's pool id (`FutureCostFunction::pools` index).
    pub(crate) pool_id: usize,
    /// Total active cuts at the child for dual extraction — the child's OWN frozen
    /// pool rows plus the child's OWN delta (mixing pools corrupts warm-start slots).
    pub(crate) num_cuts_at_successor: usize,
    /// Populated count of the child's cut pool.
    pub(crate) populated_count: usize,
    /// This child's slice of the shared active-slots buffer.
    pub(crate) active_slots: Range<usize>,
    /// Base offset of this child's cut pool's region in the concatenated
    /// binding-metadata buffers (`slot_increments`/`metadata_sync_contribution`) —
    /// so each child's binding metadata lands in ITS OWN pool's slots, never child
    /// 0's. Two children never share a non-empty pool (a non-leaf owns its own pool;
    /// leaves share an empty terminal pool), so per-child offsets separate distinct
    /// pools while shared empty-pool children stay collision-free (`populated == 0`).
    pub(crate) metadata_offset: usize,
    /// Child's Ω view (source + offset + len).
    pub(crate) openings: NodeOpenings,
    /// This child's contiguous slice of the flattened outcome/weight vector.
    pub(crate) outcome_range: Range<usize>,
}

/// A fully-resolved successor child: [`SuccessorEntry`] metadata plus the
/// pool-indexed LP data resolved from the node's arrays. The leaf backward helpers
/// read per-child data from this, so pricing a child against another child's LP —
/// the child-0 collapse — is unrepresentable.
pub(crate) struct SuccessorChild<'a> {
    /// Child's canonical node position — the warm-start `BasisStore` node-axis key.
    pub(crate) successor_node: NodePos,
    /// Child's declared node id — tags the captured basis and the solve target.
    pub(crate) successor_node_id: NodeId,
    /// Child's pool id.
    pub(crate) pool_id: usize,
    /// Child's frozen LP template (`frozen[pool_id]`).
    pub(crate) frozen_template: &'a StageTemplate,
    /// Child's delta cut rows to append to its LP (`cut_batches[pool_id]`).
    pub(crate) cut_batch: &'a RowBatch,
    /// Total active cuts at the child (frozen pool rows + delta).
    pub(crate) num_cuts_at_successor: usize,
    /// Base row count of the successor stage template (node-level; excludes cuts).
    pub(crate) template_num_rows: usize,
    /// Minimum dual multiplier for a cut to count as binding (node-level).
    pub(crate) cut_activity_tolerance: f64,
    /// Ordered slot indices of the child's active cuts.
    pub(crate) successor_active_slots: &'a [usize],
    /// Populated count of the child's cut pool.
    pub(crate) populated_count: usize,
    /// Base offset of the child's pool region in the binding-metadata buffers.
    pub(crate) metadata_offset: usize,
    /// Child's cut pool (binding-activity tracking, basis-capture metadata).
    pub(crate) successor_pool: &'a CutPool,
    /// Child's Ω view (source + offset + len).
    pub(crate) openings: NodeOpenings,
    /// This child's contiguous slice of the flattened outcome/weight vector.
    pub(crate) outcome_range: Range<usize>,
}

/// The current node's reified successor outcome set: one [`SuccessorEntry`] per
/// child, canonical order (ascending child node id, then within-child ω — the order
/// `assemble_outcome_weights` produces, so the weight slices align with the outcome
/// arena by construction), over the node's pool-indexed LP arrays. A chain is the
/// one-element case — chain byte-parity is preserved by that degeneracy, never by a
/// graph-shape predicate.
pub(crate) struct SuccessorOutcomes<'a> {
    entries: &'a [SuccessorEntry],
    active_slots_buf: &'a [usize],
    frozen: &'a [StageTemplate],
    cut_batches: &'a [RowBatch],
    pools: &'a [CutPool],
    template_num_rows: usize,
    cut_activity_tolerance: f64,
}

impl<'a> SuccessorOutcomes<'a> {
    /// Build the view over the reused metadata buffer and the node's pool-indexed
    /// arrays.
    pub(crate) fn new(
        entries: &'a [SuccessorEntry],
        active_slots_buf: &'a [usize],
        frozen: &'a [StageTemplate],
        cut_batches: &'a [RowBatch],
        pools: &'a [CutPool],
        template_num_rows: usize,
        cut_activity_tolerance: f64,
    ) -> Self {
        Self {
            entries,
            active_slots_buf,
            frozen,
            cut_batches,
            pools,
            template_num_rows,
            cut_activity_tolerance,
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        self.entries.len()
    }

    /// Total flattened outcomes `Σ_(m∈successors)|Ω_m|`; must equal `probabilities.len()`.
    pub(crate) fn total_outcomes(&self) -> usize {
        self.entries.iter().map(|e| e.outcome_range.len()).sum()
    }

    /// Total length of the per-worker binding-metadata buffers: the sum of every
    /// child's cut-pool populated count, so each child's pool occupies its own
    /// non-overlapping slot region (`metadata_offset..metadata_offset + populated`).
    pub(crate) fn total_metadata_len(&self) -> usize {
        self.entries.iter().map(|e| e.populated_count).sum()
    }

    /// Resolve child `i`'s full LP bundle from the pool-indexed arrays.
    pub(crate) fn child(&self, i: usize) -> SuccessorChild<'a> {
        let e = &self.entries[i];
        SuccessorChild {
            successor_node: e.successor_node,
            successor_node_id: e.successor_node_id,
            pool_id: e.pool_id,
            frozen_template: &self.frozen[e.pool_id],
            cut_batch: &self.cut_batches[e.pool_id],
            num_cuts_at_successor: e.num_cuts_at_successor,
            template_num_rows: self.template_num_rows,
            cut_activity_tolerance: self.cut_activity_tolerance,
            successor_active_slots: &self.active_slots_buf[e.active_slots.clone()],
            populated_count: e.populated_count,
            metadata_offset: e.metadata_offset,
            successor_pool: &self.pools[e.pool_id],
            openings: e.openings,
            outcome_range: e.outcome_range.clone(),
        }
    }
}

/// Reify the node's successor outcome set into the caller-owned `meta_buf` /
/// `active_slots_buf`: one [`SuccessorEntry`] per successor child in canonical
/// order (ascending child node id, then within-child ω), each child's delta cut
/// batch built against its own pool.
///
/// Backs [`SuccessorOutcomes`]; the own-pool `num_cuts_at_successor` and per-child
/// `metadata_offset` non-overlap invariants are documented on [`SuccessorEntry`].
/// Child order and offset accumulation are load-bearing — the flattened outcome
/// weights align to this exact order.
pub(crate) fn reify_successor_outcomes(
    meta_buf: &mut Vec<SuccessorEntry>,
    active_slots_buf: &mut Vec<usize>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    fcf: &FutureCostFunction,
    cut_batches: &mut [RowBatch],
    frozen: &[StageTemplate],
    node_pos: NodePos,
    iteration: u64,
) {
    let node_graph = training_ctx.node_graph;
    let successor_stage = node_graph.nodes[node_pos].stage.next();
    let template_num_rows = ctx.template(successor_stage).num_rows;
    let cut_state = training_ctx.state;

    meta_buf.clear();
    active_slots_buf.clear();
    let mut outcome_offset = 0usize;
    let mut metadata_offset = 0usize;
    for succ_edge in &node_graph.successors[node_pos] {
        let child_node = succ_edge.child;
        let child_pool = node_graph.nodes[child_node].pool_id;
        let child_openings = node_graph.nodes[child_node].openings;
        let child_cut_layout = &training_ctx.cut_state_layouts[child_pool];
        build_delta_cut_row_batch_into(
            &mut cut_batches[child_pool],
            fcf,
            child_pool,
            cut_state,
            child_cut_layout,
            &ctx.template(successor_stage).col_scale,
            iteration,
        );
        let num_cuts_at_successor =
            (frozen[child_pool].num_rows - template_num_rows) + cut_batches[child_pool].num_rows;
        let slots_start = active_slots_buf.len();
        active_slots_buf.extend(fcf.active_cuts(child_pool).map(|(slot, _, _)| slot));
        let slots_end = active_slots_buf.len();
        let populated_count = fcf.pools[child_pool].populated();
        let outcome_len = child_openings.len;
        meta_buf.push(SuccessorEntry {
            successor_node: child_node,
            successor_node_id: node_graph.node_ids[child_node],
            pool_id: child_pool,
            num_cuts_at_successor,
            populated_count,
            active_slots: slots_start..slots_end,
            metadata_offset,
            openings: child_openings,
            outcome_range: outcome_offset..outcome_offset + outcome_len,
        });
        outcome_offset += outcome_len;
        metadata_offset += populated_count;
    }
}
