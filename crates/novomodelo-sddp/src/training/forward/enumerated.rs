//! Enumerated all-paths forward execution engine.
//!
//! Selected by `selection = enumerated`; the sampled M-trajectory driver
//! (`forward_pass_state::run_forward_worker`) owns `selection = sampled` and is
//! untouched. The engine visits every distinct node once per iteration
//! (`enumerated_node_visit_counts`), pins each visit at its parent visit's
//! outgoing state, and reconstructs each root→leaf path's cost from its visited
//! nodes' stage costs — the exact bound `Σ_ℓ P(ℓ)·C(ℓ)` the caller reduces with
//! the compensated `Σ w·c` primitive.
//!
//! Determinism mirrors the by-node backward scheduler's own
//! "By-node scheduler is warm-start-only" contract: a stage's
//! visits are claimed from a shared atomic counter in any order, written into a
//! per-node arena keyed by canonical node position, and aggregated in canonical
//! order — so the exact bound, the lower bound, and the generated cut set are
//! bit-identical across worker, thread, and rank counts.

use std::sync::mpsc::Sender;
use std::time::Instant;

use novomodelo_core::{TrainingEvent, WorkerTimingPhase};
use novomodelo_solver::{SolverInterface, StageTemplate};
use novomodelo_stochastic::{
    ClassSampleRequest, ForwardNoiseTables, ForwardSampler, SampleRequest,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    claim_scatter::{ClaimCursor, canonical_scatter},
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    dcs::DcsParams,
    error::SddpError,
    noise::AccumSnapshot,
    setup::node_graph::{EnumeratedPlan, NodeGraph, NodeId, NodePos, StageIdx, TypedVec},
    solve::StageOpeningSolver,
    solver_stats::SolverStatsDelta,
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, CapturedBasis, SolverWorkspace},
};

use super::{
    capture_forward_basis,
    node_solve::{
        ForwardBasisApply, ForwardNodeSolve, FusedTerminalSlice, solve_forward_node_core,
    },
};

/// One node's forward visit outcome, keyed by canonical node position in the
/// per-node arena and produced worker-locally in the claim loop.
#[derive(Clone, Default)]
struct NodeVisit {
    /// Canonical node position this visit solved.
    node: NodePos,
    /// Raw (pre-cumulative-discount) stage cost, the same quantity the sampled
    /// driver accumulates via `cum_d * stage_cost`.
    stage_cost: f64,
    node_id: NodeId,
    /// Outgoing LP state (length `n_state`) — a child visit's incoming state and
    /// the value scattered into every visiting path's record `state` field.
    out_state: Vec<f64>,
    accum: AccumSnapshot,
    /// Post-solve captured basis (frozen path only; `None` on the DCS path).
    basis: Option<CapturedBasis>,
    /// Populated only for an eligible External terminal leaf; see [`FusedTerminalSlice`].
    fused: FusedTerminalSlice,
}

impl NodeVisit {
    /// A slot whose outgoing state and fused subgradient never grow: both hold at
    /// most `n_state` entries.
    fn with_state_capacity(n_state: usize) -> Self {
        Self {
            out_state: Vec::with_capacity(n_state),
            fused: FusedTerminalSlice {
                subgradient: Vec::with_capacity(n_state),
                ..FusedTerminalSlice::default()
            },
            ..Self::default()
        }
    }
}

/// Reused per-iteration mutable scratch for the enumerated forward engine —
/// the per-node visit arena and per-worker capture buffers. Meaningful only
/// while [`crate::setup::node_graph::Traversal`] is `Enumerated`; grows to
/// size against the graph-invariant [`EnumeratedPlan`] on the first call and
/// is reused in place thereafter, so no allocation occurs after warm-up.
/// Always present on `ForwardPassState` (never `Option`): dispatch is driven
/// solely by the resolved `Traversal`, read fresh each call, so there is no
/// separate flag this scratch's population could drift out of sync with.
/// `pub` (not `pub(crate)`) solely so `BackwardPassInputs::enumerated_state`
/// (a public field) stays nameable from the crate's own integration tests;
/// every field stays module-private, read only through [`Self::out_state`].
#[derive(Default)]
pub struct EnumeratedForwardScratch {
    /// Per-node visit arena, indexed by canonical node position.
    arena: TypedVec<NodePos, NodeVisit>,
    /// Per-node `arena` population flag, reset each run.
    solved: TypedVec<NodePos, bool>,
    /// Nodes this rank must solve this run (on its assigned paths), reset each run.
    on_my_paths: TypedVec<NodePos, bool>,
    /// Representative local path index for each node — the smallest path in
    /// this rank's range visiting it, minus `fwd_offset`. One value, three
    /// consumption roles at its read sites: the noise-sampling seed
    /// (`local_m`/`global_scenario`), the basis-store row
    /// (`BasisStore::get`/`get_mut`), and the solve/stats bookkeeping index
    /// (`solve_forward_node`'s `scenario_index`).
    m_rep: TypedVec<NodePos, usize>,
    /// Reused per-stage claim unit list (canonical node order).
    stage_units: Vec<NodePos>,
    /// Per-worker capture buffers (worker-local; scattered into `arena` after
    /// each stage's parallel region).
    worker_captures: Vec<Vec<NodeVisit>>,
    /// Per-worker, per-stage solver-stats accumulator, reused across iterations.
    worker_stage_stats: Vec<Vec<SolverStatsDelta>>,
}

impl EnumeratedForwardScratch {
    /// Grow `arena`/`solved`/`on_my_paths`/`m_rep`/`stage_units` to `plan`'s
    /// node count (each `arena` slot's `out_state` to `n_state`); a no-op once
    /// sized, since neither dimension changes across a training run's
    /// iterations. `worker_captures`/`worker_stage_stats` grow separately,
    /// against the worker count, inside [`run_enumerated_forward`].
    fn ensure_sized(&mut self, plan: &EnumeratedPlan, n_state: usize) {
        let n_nodes = plan.parent.len();
        if self.arena.len() == n_nodes {
            return;
        }
        self.arena = (0..n_nodes)
            .map(|_| NodeVisit {
                node: NodePos(usize::MAX),
                stage_cost: 0.0,
                node_id: NodeId(0),
                out_state: vec![0.0; n_state],
                accum: AccumSnapshot::default(),
                basis: None,
                fused: FusedTerminalSlice {
                    captured: false,
                    objective: 0.0,
                    subgradient: Vec::with_capacity(n_state),
                },
            })
            .collect();
        self.solved = vec![false; n_nodes].into();
        self.on_my_paths = vec![false; n_nodes].into();
        self.m_rep = vec![0; n_nodes].into();
        self.stage_units = Vec::with_capacity(n_nodes);
    }

    /// Node `node`'s outgoing state from the most recent [`run_enumerated_forward`]
    /// call — persisted across the fwd→bwd boundary via
    /// `ForwardPassState::enumerated_state` so the enumerated backward can read a
    /// cut-generating node's own trial state directly, never through
    /// `records`/`exchange.state_at`.
    pub(crate) fn out_state(&self, node: NodePos) -> &[f64] {
        &self.arena[node].out_state
    }

    /// Node `node`'s captured terminal-leaf subgradient from the most recent
    /// [`run_enumerated_forward`] call — `Some((objective, subgradient))` for
    /// an eligible External terminal leaf, `None` otherwise. Mirrors
    /// [`Self::out_state`]'s node-indexed read.
    pub(crate) fn fused_terminal_slice(&self, node: NodePos) -> Option<(f64, &[f64])> {
        let fused = &self.arena[node].fused;
        fused
            .captured
            .then_some((fused.objective, fused.subgradient.as_slice()))
    }

    /// Test-only: directly seed node `node`'s `out_state`, growing the arena to
    /// `n_nodes` positions first if needed. Lets a `BackwardPassState`-level unit
    /// test hand-set the persisted arena without running a full forward solve.
    #[cfg(test)]
    pub(crate) fn set_out_state_for_test(&mut self, node: NodePos, n_nodes: usize, state: &[f64]) {
        if self.arena.len() < n_nodes {
            self.arena = (0..n_nodes).map(|_| NodeVisit::default()).collect();
        }
        self.arena[node].out_state.clear();
        self.arena[node].out_state.extend_from_slice(state);
    }
}

/// Read-only per-run captures the enumerated claim loop shares across workers.
pub(crate) struct EnumeratedParams<'a> {
    pub iteration: u64,
    pub fwd_offset: usize,
    pub local_forward_passes: usize,
    pub total_forward_passes: usize,
    pub ctx: &'a StageContext<'a>,
    pub frozen: &'a [StageTemplate],
    pub fcf: &'a FutureCostFunction,
    pub training_ctx: &'a TrainingContext<'a>,
    pub sampler: &'a ForwardSampler<'a>,
    pub noise_tables: &'a ForwardNoiseTables,
    pub dcs: Option<DcsParams>,
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

/// Aggregate return of the enumerated forward driver.
pub(crate) struct EnumeratedForwardResult {
    pub scenario_costs: Vec<f64>,
    pub lp_solves: u64,
    pub stage_stats: Vec<SolverStatsDelta>,
}

/// Solve one node's LP at the incoming state already installed on
/// `ws.current_state`, warm-started from `stored_basis` (matched by node id, or
/// applied 1:1 at the terminal stage — `ForwardBasisApply`); leaves
/// `ws.current_state` holding the outgoing state and returns the raw stage cost
/// plus the captured basis (frozen path only) and fused terminal-leaf
/// subgradient.
///
/// Keys the basis by node: `stored_basis` is read immutably (shared across the
/// claim loop) and the frozen-path capture is written into `basis_out` in place,
/// for the caller's sequential post-region scatter, so a dynamically-claimed node
/// never races the session basis store. The DCS path captures no basis, leaving
/// `basis_out` untouched. `fused_out` is populated independently of the basis
/// capture, gated on `NodeGraph::is_external_terminal_leaf` — see
/// [`FusedTerminalSlice`].
///
/// `scenario_index` (the caller's `EnumeratedForwardScratch::m_rep[node]`) is
/// this call's solve/stats-bookkeeping role only; the SAME underlying value's
/// basis-store and noise-sampling roles are the caller's concern
/// (`BasisStore::get`/`get_mut`, `global_scenario`).
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from the LP solve.
// RATIONALE: disjoint partial borrows of the solver workspace, the shared run
// params and the per-node outputs (`basis_out`, `fused_out`); grouping them would
// only re-bundle the same borrows.
#[allow(clippy::too_many_arguments)]
fn solve_forward_node<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
    node: NodePos,
    parent: Option<NodePos>,
    t: StageIdx,
    scenario_index: usize,
    raw_noise: &[f64],
    stored_basis: Option<&CapturedBasis>,
    basis_out: &mut Option<CapturedBasis>,
    fused_out: &mut FusedTerminalSlice,
) -> Result<f64, SddpError> {
    let training_ctx = params.training_ctx;
    let ctx = params.ctx;
    let node_graph = training_ctx.node_graph;
    let horizon = training_ctx.horizon;
    let pool_id = node_graph.nodes[node].pool_id;
    let pool = &params.fcf.pools[pool_id];
    // Fuse the leaf's forward slice only with its CUT-GENERATING PARENT's
    // cut-state projection, never the leaf's own pool — the parent-pool
    // fusion-projection contract (sddp.md). Disabled under DCS: the forward's
    // lazily cut-reduced LP need not match the full frozen template the backward
    // loads (mirrors the `params.dcs.is_none()` basis-capture gate below). A
    // parentless leaf captures nothing → direct backward solve.
    let fusion_cut_state = (params.dcs.is_none()
        && node_graph.is_external_terminal_leaf(node, horizon.num_stages()))
    .then_some(parent)
    .flatten()
    .map(|p| &training_ctx.cut_state_layouts[node_graph.nodes[p].pool_id]);

    // Reset + reload per solve so the landed vertex cannot depend on which nodes
    // a worker solved before it (determinism across worker counts); the DCS path
    // loads the cut-free core the lazy loop appends cuts to.
    ws.solver.reset_solver_state();
    if params.dcs.is_none() {
        ws.solver.load_model(&params.frozen[pool_id]);
    } else {
        ws.solver.load_model(ctx.template(t));
    }

    let opening_solver = StageOpeningSolver::from_dcs_params(params.dcs);
    let stage_cost = solve_forward_node_core(
        ws,
        ctx,
        training_ctx,
        &ForwardNodeSolve {
            t,
            node,
            scenario_index,
            iteration: params.iteration,
            raw_noise,
            pool,
            stored_basis,
            opening_solver: &opening_solver,
            basis_apply: if horizon.is_terminal(t.next().0) {
                ForwardBasisApply::TerminalOneToOne
            } else {
                ForwardBasisApply::SlotReconstruct
            },
            fusion_cut_state,
        },
        Some(fused_out),
    )?;

    if opening_solver.captures_basis() {
        capture_forward_basis(
            &mut ws.solver,
            basis_out,
            ctx.template(t),
            params.frozen[pool_id].num_rows,
            pool,
            &ws.current_state[..training_ctx.state.n_state],
            node_graph.node_ids[node],
        );
    }

    Ok(stage_cost)
}

/// Execute the enumerated all-paths forward for one iteration on this rank.
///
/// Solves every node on this rank's assigned root→leaf paths exactly once
/// (trunk nodes shared by all paths are replicated across ranks; a rank owns
/// the complete subtrees under its path range, so no cross-rank forward state
/// exchange is needed), scatters each node's outcome into the dense
/// `records[m * num_stages + t]` layout the backward pass reads, and returns the
/// per-path costs in canonical path order for the exact `Σ P(ℓ)·C(ℓ)` bound.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from any node solve.
// RATIONALE: the stage-synced outer loop, per-stage (node, prefix) claim distribution, and
// canonical-order arena scatter are one reproducibility-critical sequence; splitting it
// would fragment the claim-order-independence contract for no clarity gain.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_enumerated_forward<S>(
    plan: &EnumeratedPlan,
    scratch: &mut EnumeratedForwardScratch,
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    records: &mut [TrajectoryRecord],
    params: &EnumeratedParams<'_>,
) -> Result<EnumeratedForwardResult, SddpError>
where
    S: SolverInterface + Send,
{
    let node_graph = params.training_ctx.node_graph;
    let num_stages = params.training_ctx.horizon.num_stages();
    let n_state = params.training_ctx.state.n_state;
    let n_workers = workspaces.len().max(1);
    let path_range = params.fwd_offset..params.fwd_offset + params.local_forward_passes;

    scratch.ensure_sized(plan, n_state);

    // Reset the per-run arena membership. `on_my_paths`/`m_rep` derive from this
    // rank's path range; `solved` tracks arena population for the child lookup.
    for flag in &mut scratch.on_my_paths {
        *flag = false;
    }
    for flag in &mut scratch.solved {
        *flag = false;
    }
    // Mark every node on this rank's paths and record each node's representative
    // (smallest) local path index for its warm-start basis slot. Paths are
    // canonical, so the smallest visiting path in range is the first walk to
    // reach the node.
    let mut node_seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        node_seq.clear();
        plan.walk_path(m, num_stages, &mut node_seq);
        let local_m = m - params.fwd_offset;
        for &node in &node_seq {
            if !scratch.on_my_paths[node] {
                scratch.on_my_paths[node] = true;
                scratch.m_rep[node] = local_m;
            }
        }
    }

    if scratch.worker_captures.len() != n_workers {
        scratch.worker_captures = (0..n_workers).map(|_| Vec::new()).collect();
        scratch.worker_stage_stats = (0..n_workers)
            .map(|_| {
                (0..num_stages)
                    .map(|_| SolverStatsDelta::default())
                    .collect()
            })
            .collect();
    }
    for inner in &mut scratch.worker_stage_stats {
        for d in inner.iter_mut() {
            d.reset_in_place();
        }
    }

    let mut lp_solves = 0u64;
    for t in (0..num_stages).map(StageIdx) {
        // Units: the nodes at stage `t` on this rank's paths, canonical order.
        scratch.stage_units.clear();
        for node in node_graph.stage_frontier(t) {
            if scratch.on_my_paths[node] {
                scratch.stage_units.push(node);
            }
        }
        if scratch.stage_units.is_empty() {
            continue;
        }

        let cursor = ClaimCursor::new(scratch.stage_units.len());
        let capture_basis = params.dcs.is_none();
        // Immutable borrows for the parallel region: the previous stage's arena
        // (children read parents), the plan, and the session basis store (warm
        // start read only — captures are written into per-worker slots for a
        // sequential post-region scatter).
        let arena = &scratch.arena;
        let solved = &scratch.solved;
        let m_rep = &scratch.m_rep;
        let stage_units = &scratch.stage_units;
        let parent = &plan.parent;
        let store: &BasisStore = basis_store;

        let worker_counts: Vec<Result<usize, SddpError>> = workspaces
            .par_iter_mut()
            .zip(scratch.worker_captures.par_iter_mut())
            .zip(scratch.worker_stage_stats.par_iter_mut())
            .map(|((ws, captures), stage_stats)| {
                let worker_start = Instant::now();
                let result = enumerated_stage_worker(
                    ws,
                    captures,
                    stage_stats,
                    &cursor,
                    t,
                    params,
                    stage_units,
                    parent,
                    arena,
                    solved,
                    m_rep,
                    store,
                );
                ws.worker_timing_buf.forward_wall_ms +=
                    worker_start.elapsed().as_secs_f64() * 1_000.0;
                result
            })
            .collect();

        // Propagate the first worker error by value (`SddpError` is not `Clone`),
        // mirroring `by_node_finish`'s `res?` handling.
        let counts = worker_counts
            .into_iter()
            .collect::<Result<Vec<usize>, SddpError>>()?;
        lp_solves += counts.iter().map(|&c| c as u64).sum::<u64>();

        // Sequential scatter: copy each worker's captures into the per-node arena
        // and swap each captured basis into the session store, keyed by canonical
        // node — claim/worker order cannot reach the arena, so the outcome is
        // worker-count invariant. Copying (not moving) keeps every worker-slot
        // buffer alive for reuse next stage.
        for (w, i) in canonical_scatter(&counts) {
            let node = scratch.worker_captures[w][i].node;
            // The basis-store role of `m_rep`: the row `BasisStore::get_mut`
            // swaps this node's freshly captured basis into.
            let basis_scenario = scratch.m_rep[node];
            {
                let src = &scratch.worker_captures[w][i];
                let dst = &mut scratch.arena[node];
                dst.node = node;
                dst.stage_cost = src.stage_cost;
                dst.node_id = src.node_id;
                dst.out_state.clear();
                dst.out_state.extend_from_slice(&src.out_state);
                src.accum.copy_into(&mut dst.accum);
                dst.fused.captured = src.fused.captured;
                dst.fused.objective = src.fused.objective;
                dst.fused.subgradient.clear();
                dst.fused
                    .subgradient
                    .extend_from_slice(&src.fused.subgradient);
            }
            scratch.solved[node] = true;
            if capture_basis {
                std::mem::swap(
                    basis_store.get_mut(basis_scenario, node),
                    &mut scratch.worker_captures[w][i].basis,
                );
            }
        }
    }

    let mut stage_stats: Vec<SolverStatsDelta> = (0..num_stages)
        .map(|_| SolverStatsDelta::default())
        .collect();
    for inner in &scratch.worker_stage_stats {
        for (dst, src) in stage_stats.iter_mut().zip(inner.iter()) {
            SolverStatsDelta::accumulate_into(dst, src);
        }
    }

    // Scatter the arena into the dense record layout and reconstruct per-path
    // costs in canonical path order (ascending trajectory id within this rank).
    let mut scenario_costs = Vec::with_capacity(params.local_forward_passes);
    let mut seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        let local_m = m - params.fwd_offset;
        seq.clear();
        plan.walk_path(m, num_stages, &mut seq);
        debug_assert_eq!(seq.len(), num_stages, "path must span every stage");
        let mut cost = 0.0_f64;
        for (t, &node) in seq.iter().enumerate() {
            debug_assert!(scratch.solved[node], "node on an owned path was not solved");
            let visit = &scratch.arena[node];
            let cum_d = params
                .ctx
                .cumulative_discount_factors
                .get(t)
                .copied()
                .unwrap_or(1.0);
            cost += cum_d * visit.stage_cost;
            let rec = &mut records[local_m * num_stages + t];
            rec.stage_cost = visit.stage_cost;
            rec.node_id = visit.node_id;
            rec.state.clear();
            rec.state.extend_from_slice(&visit.out_state);
            debug_assert_eq!(
                rec.state.len(),
                n_state,
                "record state must be n_state long"
            );
        }
        scenario_costs.push(cost);
    }

    // Single-rank pins the dedup factor: solving every node once totals
    // `Σ forward_solve_counts`. Multi-rank replicates trunk nodes per rank, so
    // the equality holds only when this rank owns every path.
    #[cfg(debug_assertions)]
    if params.local_forward_passes == params.total_forward_passes {
        let expected = expected_single_rank_solves(node_graph)?;
        debug_assert_eq!(
            lp_solves, expected,
            "single-rank enumerated solve total must equal Σ forward_solve_counts"
        );
    }

    if let Some(sender) = params.event_sender {
        for ws in workspaces.iter() {
            let _ = sender.send(TrainingEvent::WorkerTiming {
                rank: ws.rank,
                worker_id: ws.worker_id,
                iteration: params.iteration,
                phase: WorkerTimingPhase::Forward,
                timings: ws.worker_timing_buf,
            });
        }
    }

    Ok(EnumeratedForwardResult {
        scenario_costs,
        lp_solves,
        stage_stats,
    })
}

/// One worker's claim loop over a stage's node units. Claims units from the
/// shared atomic counter in any order; each solved node's outcome is written into
/// a reused worker-local capture slot for the caller's sequential, canonical-order
/// scatter. Returns the count of slots this worker filled (`captures[..count]`).
/// The capture slots — and every buffer they hold (`out_state`, the accumulator
/// snapshot, the captured basis) — persist across stages and iterations, so no
/// allocation occurs after warm-up.
// RATIONALE: the per-worker claim loop threads disjoint partial borrows of one
// &mut SolverWorkspace plus the shared claim counter and per-node arena; extracting would
// pass the whole workspace for no gain or re-introduce the argument count.
#[allow(clippy::too_many_arguments)]
fn enumerated_stage_worker<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    captures: &mut Vec<NodeVisit>,
    stage_stats: &mut [SolverStatsDelta],
    cursor: &ClaimCursor,
    t: StageIdx,
    params: &EnumeratedParams<'_>,
    stage_units: &[NodePos],
    parent: &TypedVec<NodePos, Option<NodePos>>,
    arena: &TypedVec<NodePos, NodeVisit>,
    solved: &TypedVec<NodePos, bool>,
    m_rep: &TypedVec<NodePos, usize>,
    store: &BasisStore,
) -> Result<usize, SddpError> {
    let node_graph = params.training_ctx.node_graph;
    let state_space = params.training_ctx.state;
    let n_state = state_space.n_state;

    let noise_dim = params.training_ctx.stochastic.dim();
    let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
    raw_noise_buf.resize(noise_dim, 0.0_f64);
    let mut corr_scratch = std::mem::take(&mut ws.scratch.corr_scratch);
    corr_scratch.resize(2 * noise_dim, 0.0_f64);

    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios_u32 = params.total_forward_passes as u32;

    let mut count = 0usize;
    while let Some(u) = cursor.claim() {
        let node = stage_units[u];
        let local_m = m_rep[node];
        let global_scenario = params.fwd_offset + local_m;
        let parent_node = parent[node];

        // Install the incoming state: the parent visit's outgoing state (already
        // scattered in the previous stage's sequential pass), or the initial
        // state at a root.
        ws.current_state.clear();
        if let Some(p) = parent_node {
            debug_assert!(solved[p], "parent visit must be solved before its child");
            ws.current_state.extend_from_slice(&arena[p].out_state);
            arena[p].accum.restore_into(&mut ws.scratch);
        } else {
            ws.current_state
                .extend_from_slice(params.training_ctx.initial_state);
            seed_root_accumulators(ws, params);
        }

        #[allow(clippy::cast_possible_truncation)]
        let (i32_it, s32, t32) = (params.iteration as u32, global_scenario as u32, t.0 as u32);
        let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(node);
        let pinned_scenario = node_graph.node_pinned_scenario(node);

        if parent_node.is_none() {
            let class_req = ClassSampleRequest {
                iteration: i32_it,
                scenario: s32,
                stage: t32,
                stage_idx: t.0,
                total_scenarios: total_scenarios_u32,
                noise_group_id: 0,
                node_opening_offset,
                node_opening_len,
                pinned_scenario,
            };
            params.sampler.apply_initial_state(
                &class_req,
                &mut ws.current_state,
                state_space.state_dim_lag_range().start,
            );
        }

        let noise = params.sampler.sample(SampleRequest {
            iteration: i32_it,
            scenario: s32,
            stage: t32,
            stage_idx: t.0,
            noise_buf: &mut raw_noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: total_scenarios_u32,
            noise_group_id: params.ctx.noise_group_id_at(t),
            node_opening_offset,
            node_opening_len,
            pinned_scenario,
            tables: params.noise_tables,
        })?;

        // Reuse the capture slot at `count`, growing only until the widest stage's
        // frontier is covered; every buffer inside is then reused across stages
        // and iterations.
        if captures.len() <= count {
            captures.push(NodeVisit::with_state_capacity(n_state));
        }
        let stored = store.get(local_m, node);
        let stats_before = ws.solver.statistics();
        // `noise`/`stored` borrow the worker-local `raw_noise_buf` and the shared
        // read-only basis store, never `ws`, so both coexist with the `&mut ws`
        // solve — no per-solve copy needed (mirrors the sampled driver).
        let raw_noise = noise.as_slice();
        let visit = &mut captures[count];
        let stage_cost = solve_forward_node(
            ws,
            params,
            node,
            parent_node,
            t,
            local_m,
            raw_noise,
            stored,
            &mut visit.basis,
            &mut visit.fused,
        )?;
        let delta = SolverStatsDelta::from_snapshots(&stats_before, &ws.solver.statistics());
        SolverStatsDelta::accumulate_into(&mut stage_stats[t.0], &delta);

        visit.node = node;
        visit.stage_cost = stage_cost;
        visit.node_id = node_graph.node_ids[node];
        visit.out_state.clear();
        visit
            .out_state
            .extend_from_slice(&ws.current_state[..n_state]);
        visit.accum.capture_from(&ws.scratch);
        count += 1;
    }

    ws.scratch.raw_noise_buf = raw_noise_buf;
    ws.scratch.corr_scratch = corr_scratch;
    Ok(count)
}

/// Seed the trajectory-carried scratch accumulators at a root visit, matching
/// the sampled driver's stage-0 seed (`DerivedInflowSeeds` copy or zero-fill).
fn seed_root_accumulators<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
) {
    if params.training_ctx.lag_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum.fill(0.0);
    } else {
        ws.scratch.lag_accumulator[..params.training_ctx.lag_accum_seed.len()]
            .copy_from_slice(params.training_ctx.lag_accum_seed);
        ws.scratch.lag_weight_accum[..params.training_ctx.lag_weight_seed.len()]
            .copy_from_slice(params.training_ctx.lag_weight_seed);
    }
    ws.scratch.downstream_accumulator.fill(0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch.downstream_completed_lags.fill(0.0);
    ws.scratch.downstream_n_completed = 0;
}

/// The single-rank per-iteration solve total: `Σ forward_solve_counts`.
/// Exposed so the caller can `debug_assert` the enumerated engine's realized
/// single-rank solve count against it (the dedup scale invariant).
///
/// # Errors
///
/// Propagates the `u64` path-product overflow guard.
// Consumed only by the single-rank solve-count debug-assert above; legitimately
// unused in a release non-test build (debug-assertions off), so allow it there.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(crate) fn expected_single_rank_solves(node_graph: &NodeGraph) -> Result<u64, SddpError> {
    Ok(node_graph.forward_solve_counts()?.into_iter().sum())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

    use super::*;
    use novomodelo_comm::LocalBackend;
    use novomodelo_core::temporal::StageStateConfig;
    use novomodelo_solver::{ActiveSolver, SolutionView};

    use crate::{
        indexer::{CutStateProjection, InflowForm, StateSpace},
        lead_time::AnticipatedResolution,
        setup::{StudySetup, node_graph::Traversal},
        solver_phase::Phase,
        test_support,
        training::{
            backward::extract_state_duals_only,
            forward::{build_sampler_from_ctx, node_solve::capture_fused_terminal_slice},
        },
        workspace::WorkspacePool,
    };

    /// An eligible node's captured `(objective, duals)` equals a direct
    /// [`extract_state_duals_only`] call on the same `view`/`cut_state`/`col_scale` —
    /// [`capture_fused_terminal_slice`]'s "reuse verbatim" contract, proved by
    /// construction rather than by re-deriving the `rc`/`col_scale` math a second time.
    #[test]
    fn capture_fused_terminal_slice_matches_extract_state_duals_only() {
        let state = StateSpace::new(
            2,
            1,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &[1, 1],
        );
        let cut_state = CutStateProjection::new(
            &state,
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        );
        // A lag slot's incoming column (`inflow_lags` block) lands past
        // `state.n_state`, so this must exceed the largest incoming column
        // `cut_state` resolves, not just `[0, n_state)`.
        let n_cols = 64;
        let reduced_costs: Vec<f64> = (0..n_cols).map(|c| c as f64 + 1.0).collect();
        let col_scale: Vec<f64> = vec![2.0; n_cols];
        let view = SolutionView {
            objective: 42.0,
            primal: &[],
            dual: &[],
            reduced_costs: &reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        };

        let mut expected_duals = Vec::new();
        let expected_objective =
            extract_state_duals_only(&view, &cut_state, &col_scale, &mut expected_duals);

        let mut fused_out = FusedTerminalSlice::default();
        capture_fused_terminal_slice(&view, &col_scale, Some(&cut_state), &mut fused_out);

        assert!(fused_out.captured);
        assert_eq!(fused_out.objective, expected_objective);
        assert_eq!(fused_out.subgradient, expected_duals);
    }

    /// A `None` `cut_state` (the non-eligible-node signal) resets an
    /// already-populated `fused_out` to uncaptured, not merely leaves it as
    /// constructed — catching a "only ever sets true" regression.
    #[test]
    fn capture_fused_terminal_slice_none_when_ineligible() {
        let view = SolutionView {
            objective: 7.0,
            primal: &[],
            dual: &[],
            reduced_costs: &[],
            iterations: 0,
            solve_time_seconds: 0.0,
        };
        let mut fused_out = FusedTerminalSlice {
            captured: true,
            objective: 99.0,
            subgradient: vec![1.0, 2.0],
        };

        capture_fused_terminal_slice(&view, &[], None, &mut fused_out);

        assert!(!fused_out.captured);
    }

    /// Fresh, unpopulated scaffolding for one [`run_enumerated_forward`] call
    /// over `setup`'s whole graph: a single-worker pool, an empty
    /// [`BasisStore`], and zeroed [`TrajectoryRecord`]s.
    fn fresh_rig(
        setup: &StudySetup,
    ) -> (
        WorkspacePool<ActiveSolver>,
        BasisStore,
        Vec<TrajectoryRecord>,
    ) {
        let node_graph = &setup.inputs.node_graph;
        let total_forward_passes =
            usize::try_from(test_support::node_scenario_count(node_graph).expect("scenario count"))
                .expect("fits usize");
        let comm = LocalBackend;
        let pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool");
        let basis_store = BasisStore::new(total_forward_passes, node_graph.nodes.len());
        let records: Vec<TrajectoryRecord> = (0..total_forward_passes * setup.num_stages())
            .map(|_| TrajectoryRecord::default())
            .collect();
        (pool, basis_store, records)
    }

    /// Run one `enumerated` forward iteration over `setup`'s whole graph on a
    /// single worker, mirroring `ForwardPassState::run_enumerated`'s param
    /// assembly. `scratch`/`workspaces`/`basis_store`/`records` persist across
    /// repeated calls, so a second call at `iteration + 1` exercises
    /// cross-iteration basis warm-start.
    fn run_iteration(
        setup: &StudySetup,
        scratch: &mut EnumeratedForwardScratch,
        workspaces: &mut [SolverWorkspace<ActiveSolver>],
        basis_store: &mut BasisStore,
        records: &mut [TrajectoryRecord],
        iteration: u64,
        event_sender: Option<&Sender<TrainingEvent>>,
    ) -> EnumeratedForwardResult {
        let node_graph = &setup.inputs.node_graph;
        let stage_ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let sampler = build_sampler_from_ctx(&training_ctx).expect("forward sampler");
        let frozen: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| stage_ctx.templates[node_graph.pool_stage[p].0].clone())
            .collect();
        let traversal = Traversal::resolve(node_graph, true, 0);
        let Traversal::Enumerated(plan) = &traversal else {
            unreachable!("resolve(is_enumerated=true, ..) always yields Enumerated");
        };
        let total_forward_passes =
            usize::try_from(test_support::node_scenario_count(node_graph).expect("scenario count"))
                .expect("fits usize");
        let dcs = training_ctx
            .dcs
            .map(|p| p.resolve(Phase::Forward.profile().primal_feasibility_tolerance))
            .filter(|p| p.is_active(iteration));
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                u32::try_from(iteration).expect("fits u32"),
                u32::try_from(total_forward_passes).expect("fits u32"),
                stage_ctx.noise_group_ids,
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");

        let params = EnumeratedParams {
            iteration,
            fwd_offset: 0,
            local_forward_passes: total_forward_passes,
            total_forward_passes,
            ctx: &stage_ctx,
            frozen: &frozen,
            fcf: &setup.fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
            dcs,
            event_sender,
        };

        run_enumerated_forward(plan, scratch, workspaces, basis_store, records, &params)
            .expect("run_enumerated_forward must succeed on a well-formed fixture")
    }

    /// An External, single-opening terminal leaf is captured; the interior
    /// External root is not; the eligible leaves' basis is still captured into
    /// [`BasisStore`] and a second iteration (warm-started from it) succeeds.
    #[test]
    fn enumerated_forward_captures_fused_slice_only_for_eligible_external_terminal_leaves() {
        let setup = test_support::external_distinct_fan_setup(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();

        let eligible = eligible_external_terminal_leaves(&setup);
        let ineligible: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&p| !node_graph.is_external_terminal_leaf(p, num_stages))
            .collect();
        assert_eq!(
            eligible.len(),
            2,
            "both leaves of a 2-leaf external-distinct fan are eligible"
        );
        assert_eq!(
            ineligible.len(),
            1,
            "only the interior External root is ineligible"
        );

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &ineligible {
            assert!(
                scratch.fused_terminal_slice(node).is_none(),
                "non-eligible node {node:?} must not be captured"
            );
        }
        for &node in &eligible {
            let (objective, duals) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            assert!(objective.is_finite());
            let pool_id = node_graph.nodes[node].pool_id;
            assert_eq!(
                duals.len(),
                setup.inputs.cut_state_layouts[pool_id].n_slots(),
                "captured duals must span the leaf's own cut-state projection"
            );
            let local_m = scratch.m_rep[node];
            assert!(
                basis_store.get(local_m, node).is_some(),
                "eligible leaf {node:?}'s basis must still be captured into BasisStore"
            );
        }

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            2,
            None,
        );

        for &node in &eligible {
            assert!(
                scratch.fused_terminal_slice(node).is_some(),
                "eligible leaf {node:?} must stay captured across a warm-started iteration"
            );
            let local_m = scratch.m_rep[node];
            assert!(
                basis_store.get(local_m, node).is_some(),
                "eligible leaf {node:?}'s basis must still be present after iteration 2"
            );
        }
    }

    /// An eligible leaf's captured dual length must span its CUT-GENERATING
    /// PARENT's cut-state projection — never the leaf's own terminal (no-successor,
    /// always-full) pool. [`test_support::external_distinct_fan_setup_heterogeneous_cut_state`]
    /// declares an active inflow-lag slot so the two genuinely diverge
    /// (`build_cut_state_layouts`'s no-successor rule keeps every leaf's own pool
    /// full-dimension regardless of its declared `state_config`); on the ORIGINAL
    /// `external_distinct_fan_setup` fixture (no lag slot) the two pools coincide,
    /// which is exactly the coverage gap that let the wrong-projection capture ship.
    #[test]
    fn enumerated_forward_fused_slice_projects_with_parent_pool_not_leaf_pool() {
        let setup = test_support::external_distinct_fan_setup_heterogeneous_cut_state(2, 1);
        let node_graph = &setup.inputs.node_graph;

        let eligible = eligible_external_terminal_leaves(&setup);
        assert_eq!(
            eligible.len(),
            2,
            "both leaves of a 2-leaf external-distinct fan are eligible"
        );

        for &node in &eligible {
            let leaf_pool = node_graph.nodes[node].pool_id;
            let parent = node_graph
                .node_parent(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must have a parent"));
            let parent_pool = node_graph.nodes[parent].pool_id;
            assert_ne!(
                setup.inputs.cut_state_layouts[leaf_pool].n_slots(),
                setup.inputs.cut_state_layouts[parent_pool].n_slots(),
                "fixture power check: leaf pool {leaf_pool} and parent pool {parent_pool} must \
                 project DIFFERENT dimensions, or this test cannot distinguish the fix from the bug"
            );
        }

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &eligible {
            let (_, duals) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            let leaf_pool = node_graph.nodes[node].pool_id;
            let parent = node_graph.node_parent(node).expect("checked above");
            let parent_pool = node_graph.nodes[parent].pool_id;
            assert_eq!(
                duals.len(),
                setup.inputs.cut_state_layouts[parent_pool].n_slots(),
                "captured duals must span the CUT-GENERATING PARENT's cut-state projection \
                 (the backward's own `SuccessorSpec::cut_state`), not the leaf's own pool"
            );
            assert_ne!(
                duals.len(),
                setup.inputs.cut_state_layouts[leaf_pool].n_slots(),
                "power check: the leaf's own pool dimension must differ from the parent's, or \
                 this assertion cannot distinguish the fix from the wrong-projection bug"
            );
        }
    }

    fn eligible_external_terminal_leaves(setup: &StudySetup) -> Vec<NodePos> {
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();
        (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&p| node_graph.is_external_terminal_leaf(p, num_stages))
            .collect()
    }

    /// `(leaf pool projection, cut-generating parent pool projection)` of `leaf`.
    fn leaf_and_parent_projections(
        setup: &StudySetup,
        leaf: NodePos,
    ) -> (&CutStateProjection, &CutStateProjection) {
        let node_graph = &setup.inputs.node_graph;
        let parent = node_graph
            .node_parent(leaf)
            .unwrap_or_else(|| panic!("eligible leaf {leaf:?} must have a parent"));
        (
            &setup.inputs.cut_state_layouts[node_graph.nodes[leaf].pool_id],
            &setup.inputs.cut_state_layouts[node_graph.nodes[parent].pool_id],
        )
    }

    /// On the compact LP the captured slice spans the cut-generating PARENT's
    /// projection, which here differs in size from the leaf's own pool.
    #[test]
    fn enumerated_forward_compact_fused_slice_spans_the_parent_projection() {
        let (_, setup) =
            test_support::external_distinct_fan_setup_heterogeneous_cut_state_in_both_forms(2, 1);
        assert_eq!(setup.stage_state().inflow_form, InflowForm::Compact);
        let eligible = eligible_external_terminal_leaves(&setup);
        assert_eq!(eligible.len(), 2, "both leaves of the fan are eligible");
        for &node in &eligible {
            let (leaf, parent) = leaf_and_parent_projections(&setup, node);
            assert!(leaf.has_lag_slots(), "power check: the leaf projects lag");
            assert!(!parent.has_lag_slots(), "power check: the parent does not");
            assert_ne!(
                leaf.n_slots(),
                parent.n_slots(),
                "power check: a swapped projection must change the slice length"
            );
        }

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &eligible {
            let (leaf, parent) = leaf_and_parent_projections(&setup, node);
            let (objective, subgradient) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            assert!(objective.is_finite());
            assert_eq!(subgradient.len(), parent.n_slots());
            assert_ne!(subgradient.len(), leaf.n_slots());
            assert!(subgradient.iter().all(|c| c.is_finite()));
        }

        let n_state = setup.stage_state().n_state;
        let slots = scratch.worker_captures.iter().flatten();
        assert!(slots.clone().count() > 0, "power check: the run captured");
        for slot in slots {
            assert_eq!(slot.out_state.capacity(), n_state);
            assert_eq!(slot.fused.subgradient.capacity(), n_state);
        }
    }

    /// A parent that projects a lag slot is the case a storage-only read of the
    /// compact LP cannot serve: the incoming lag column does not exist there.
    #[test]
    fn enumerated_forward_compact_fused_slice_reads_a_lagged_parent() {
        let (_, setup) =
            test_support::external_distinct_fan_setup_lagged_parent_in_both_forms(2, 1);
        assert_eq!(setup.stage_state().inflow_form, InflowForm::Compact);
        let eligible = eligible_external_terminal_leaves(&setup);
        assert_eq!(eligible.len(), 2, "both leaves of the fan are eligible");
        for &node in &eligible {
            let (_, parent) = leaf_and_parent_projections(&setup, node);
            assert!(
                parent.has_lag_slots(),
                "power check: the parent projects a lag slot"
            );
        }

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &eligible {
            let (_, parent) = leaf_and_parent_projections(&setup, node);
            let (objective, subgradient) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            assert!(objective.is_finite());
            assert_eq!(subgradient.len(), parent.n_slots());
        }
    }

    /// The compact terminal leaf stores a basis in the compact layout and the
    /// next iteration offers it back 1:1, never through reconstruction.
    #[test]
    fn enumerated_forward_compact_terminal_leaf_warm_starts_one_to_one() {
        let (logical, setup) =
            test_support::external_distinct_fan_setup_lagged_parent_in_both_forms(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let eligible = eligible_external_terminal_leaves(&setup);
        assert_eq!(eligible.len(), 2, "both leaves of the fan are eligible");

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        assert!(
            pool.workspaces.iter().all(|ws| ws.solver.warm_starts()),
            "power check: the default profile warm-starts, or the offer counts below are vacuous"
        );
        let mut scratch = EnumeratedForwardScratch::default();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        let terminal = StageIdx(setup.num_stages() - 1);
        let compact_template = setup.stage_ctx().template(terminal).clone();
        let logical_cols = logical.stage_ctx().template(terminal).num_cols;
        assert!(
            compact_template.num_cols < logical_cols,
            "power check: the compact template has fewer columns than its logical twin"
        );
        for &node in &eligible {
            let stored = basis_store
                .get(scratch.m_rep[node], node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?}'s basis must be stored"));
            let leaf_pool = &setup.fcf.pools[node_graph.nodes[node].pool_id];
            assert_eq!(stored.basis.col_status.len(), compact_template.num_cols);
            assert_eq!(
                stored.basis.row_status.len(),
                compact_template.num_rows + leaf_pool.active_count()
            );
        }

        let totals = |workspaces: &[SolverWorkspace<ActiveSolver>]| {
            workspaces.iter().fold((0, 0, 0), |(o, r, f), ws| {
                let stats = ws.solver.statistics();
                (
                    o + stats.basis_offered,
                    r + stats.basis_reconstructions,
                    f + stats.basis_consistency_failures,
                )
            })
        };
        let (offered_before, reconstructed_before, failures_before) = totals(&pool.workspaces);
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            2,
            None,
        );
        let (offered_after, reconstructed_after, failures_after) = totals(&pool.workspaces);

        let one_to_one = (offered_after - offered_before)
            .saturating_sub(reconstructed_after - reconstructed_before);
        assert!(
            one_to_one >= eligible.len() as u64,
            "each terminal leaf must be offered its stored basis without reconstruction, got {one_to_one}"
        );
        assert_eq!(failures_after, failures_before);
    }

    /// A Generated terminal leaf is never captured, even though its basis is —
    /// fusion eligibility and basis capture are independent mechanisms.
    #[test]
    fn enumerated_forward_generated_terminal_leaf_stays_uncaptured() {
        let setup = test_support::terminal_generated_fan_setup(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();

        assert!(
            (0..node_graph.nodes.len())
                .map(NodePos)
                .all(|p| !node_graph.is_external_terminal_leaf(p, num_stages)),
            "a terminal-Generated fan has no eligible node"
        );

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for pos in (0..node_graph.nodes.len()).map(NodePos) {
            assert!(
                scratch.fused_terminal_slice(pos).is_none(),
                "node {pos:?} in a terminal-Generated fan must not be captured"
            );
            let local_m = scratch.m_rep[pos];
            assert!(
                basis_store.get(local_m, pos).is_some(),
                "node {pos:?}'s basis must still be captured regardless of fusion eligibility"
            );
        }
    }

    /// `run_enumerated_forward` emits one `WorkerTiming { phase: Forward }`
    /// event per workspace, each carrying a non-zero `forward_wall_ms` no
    /// larger than the call's own wall-clock elapsed — every worker's
    /// accumulated per-stage busy time is a subset of the call's total wall.
    #[test]
    fn enumerated_forward_emits_worker_timing_per_workspace() {
        let setup = test_support::terminal_generated_fan_setup(2, 1);
        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();

        let call_start = Instant::now();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            Some(&tx),
        );
        let call_elapsed_ms = call_start.elapsed().as_secs_f64() * 1_000.0;
        drop(tx);

        let n_workers = pool.workspaces.len();
        let forward_walls: Vec<f64> = rx
            .try_iter()
            .filter_map(|e| match e {
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Forward,
                    timings,
                    ..
                } => Some(timings.forward_wall_ms),
                _ => None,
            })
            .collect();

        assert_eq!(
            forward_walls.len(),
            n_workers,
            "one Forward WorkerTiming event must be emitted per workspace"
        );
        assert!(
            forward_walls.iter().all(|&ms| ms > 0.0),
            "every worker must report a non-zero forward_wall_ms on a genuine (non-fused) \
             real solve, got {forward_walls:?}"
        );
        assert!(
            forward_walls.iter().all(|&ms| ms <= call_elapsed_ms + 1.0),
            "a worker's own accumulated wall time cannot exceed run_enumerated_forward's own \
             call wall ({call_elapsed_ms} ms), got {forward_walls:?}"
        );
    }
}
