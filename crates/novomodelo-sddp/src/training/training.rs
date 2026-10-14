//! Training loop orchestrator for the SDDP algorithm.
//!
//! [`train`] wires the forward pass, forward sync, state exchange, backward
//! pass, cut sync, lower-bound evaluation, and convergence check into a single
//! iterative loop. The per-iteration ordering is enforced by
//! `TrainingSession::run_iteration`.
//!
//! All workspace buffers are allocated once before the loop and reused: no heap
//! allocation occurs on the hot path.

use novomodelo_comm::CommError::InvalidBufferSize;
use novomodelo_comm::{Communicator, per_rank_counts};
use novomodelo_io::StageStatesReadResult;
use novomodelo_solver::ActiveProfile;
use novomodelo_solver::SolverError;
use novomodelo_solver::{SolverInterface, StageTemplate};

use crate::training::session::rank_distribution::RankDistribution;
use crate::workspace::BasisStore;
use crate::{
    SddpError, SolverProfiles, TrainingConfig,
    context::{StageContext, TrainingContext},
    cut::fcf::FutureCostFunction,
    setup::NodePos,
    solver_stats::SolverStatsLogEntry,
    stopping_rule::StopDecision,
    training_session::{IterationOutcome, TrainingSession},
    workspace::CapturedBasis,
};

// ---------------------------------------------------------------------------
// TrainingResult
// ---------------------------------------------------------------------------

/// Result of a training run that always carries partial results.
///
/// On normal completion `error` is `None` and `result` holds the full
/// statistics. On mid-iteration failure `error` carries the cause and `result`
/// holds statistics from the fully completed iterations (the failing one
/// excluded).
#[derive(Debug)]
pub struct TrainingOutcome {
    /// Training result from the completed iterations; always populated.
    pub result: TrainingResult,

    /// If training was interrupted by an error, the cause. `None` when
    /// training completed normally (convergence, iteration limit, or
    /// time limit).
    pub error: Option<SddpError>,
}

/// Summary statistics produced when the training loop terminates.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final lower bound at termination.
    pub final_lb: f64,

    /// The lower bound of each completed iteration the run or its resumed
    /// checkpoint recorded, oldest first; empty when the result was not
    /// produced by a training loop.
    pub lower_bound_history: Vec<f64>,

    /// Final upper bound mean at termination.
    pub final_ub: f64,

    /// Final upper bound standard deviation at termination.
    pub final_ub_std: f64,

    /// Final convergence gap: `(UB - LB) / max(1.0, |UB|)`.
    pub final_gap: f64,

    /// Total number of iterations completed.
    pub iterations: u64,

    /// Human-readable termination reason (e.g., `"iteration_limit"`, `"graceful_shutdown"`).
    pub reason: String,

    /// The stop decision of the iteration where training stopped; the default
    /// (nothing triggered) when training ended otherwise or the result was not
    /// produced by a training loop.
    pub stop_decision: StopDecision,

    /// Total wall-clock time for the training run, in milliseconds.
    pub total_time_ms: u64,

    /// Per-stage captured basis from the last iteration, 0-based. `None` when
    /// the stage was never solved in the final iteration (e.g. the last
    /// finite-horizon stage has no successor cuts).
    pub basis_cache: Vec<Option<CapturedBasis>>,

    /// Per-iteration, per-phase solver statistics log. Stage index is `-1` for
    /// `"lower_bound"` entries (rank-aggregated, no stage axis).
    pub solver_stats_log: Vec<SolverStatsLogEntry>,

    /// Final-iteration frozen templates, one per POOL (each is a pool's base
    /// stage template plus that pool's active cuts). Always `Some`: freeze runs
    /// unconditionally before the first iteration and is never reverted.
    pub frozen_templates: Option<Vec<StageTemplate>>,

    /// The per-scenario bases the policy checkpoint writes, on rank 0; empty on
    /// every other rank and when the result was not produced by a training loop.
    pub checkpoint_bases: CheckpointBasisStore,
}

impl TrainingResult {
    /// Construct a `TrainingResult` with explicit field values.
    ///
    /// Prefer this over struct-literal syntax: adding a field then fails to
    /// compile until every call site is updated.
    #[must_use]
    // Rationale: the arguments are independently sourced phase outputs and every
    // call site constructs the full result, so a context struct would not reduce
    // the arity.
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn new(
        final_lb: f64,
        final_ub: f64,
        final_ub_std: f64,
        final_gap: f64,
        iterations: u64,
        reason: String,
        total_time_ms: u64,
        basis_cache: Vec<Option<CapturedBasis>>,
        solver_stats_log: Vec<SolverStatsLogEntry>,
        frozen_templates: Option<Vec<StageTemplate>>,
    ) -> Self {
        Self {
            final_lb,
            lower_bound_history: Vec::new(),
            final_ub,
            final_ub_std,
            final_gap,
            iterations,
            reason,
            stop_decision: StopDecision::default(),
            total_time_ms,
            basis_cache,
            solver_stats_log,
            frozen_templates,
            checkpoint_bases: CheckpointBasisStore::default(),
        }
    }
}

/// The solver bases a checkpoint holds: one per global scenario and node, each
/// with an empty `state_at_capture`.
#[derive(Debug, Clone, Default)]
pub struct CheckpointBasisStore {
    /// Per canonical node, `(global scenario, basis)` ascending by scenario.
    pub nodes: Vec<Vec<(u32, CapturedBasis)>>,
    /// Whether every rank's rows are held; a failure exit holds rank 0's only.
    pub complete: bool,
}

impl CheckpointBasisStore {
    /// This rank's rows keyed by global scenario, marked incomplete. Issues no
    /// collective, so a failure exit can call it.
    pub(crate) fn rank_local(basis_store: &BasisStore, ranks: &RankDistribution) -> Self {
        let nodes = (0..basis_store.num_nodes())
            .map(|node| {
                (0..ranks.my_actual_fwd)
                    .filter_map(|m| {
                        basis_store.get(m, NodePos(node)).map(|captured| {
                            (
                                global_scenario(ranks.my_fwd_offset, m),
                                CapturedBasis {
                                    basis: captured.basis.clone(),
                                    base_row_count: captured.base_row_count,
                                    cut_row_slots: captured.cut_row_slots.clone(),
                                    state_at_capture: Vec::new(),
                                    node_id: captured.node_id,
                                },
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        Self {
            nodes,
            complete: false,
        }
    }
}

/// What a policy load hands to [`train`].
#[derive(Debug, Default)]
pub struct LoadedPolicy {
    /// Per canonical node, the stored bases with their global scenario,
    /// ascending by scenario; empty for a fresh start.
    pub scenario_bases: Vec<Vec<(u32, CapturedBasis)>>,

    /// The stored trial states of the loaded pools, their slots already placed
    /// in the loaded pools.
    pub trial_states: Vec<StageStatesReadResult>,
}

impl LoadedPolicy {
    /// The basis that seeds global scenario `scenario`'s row of the basis store
    /// at `node`: that scenario's own stored basis, else the node's
    /// lowest-scenario one (a source that ran fewer forward passes), else none.
    #[must_use]
    pub fn seed_basis(&self, node: NodePos, scenario: u32) -> Option<&CapturedBasis> {
        let entries = self.scenario_bases.get(node.0)?;
        entries
            .binary_search_by_key(&scenario, |&(stored, _)| stored)
            .map_or_else(|_| entries.first(), |i| entries.get(i))
            .map(|(_, basis)| basis)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a buffer length to `i32` for use as an MPI count.
///
/// # Errors
///
/// Returns `SddpError::Communication(CommError::InvalidBufferSize { .. })`
/// when `len > i32::MAX` (the MPI count limit).
fn checked_mpi_count(len: usize, operation: &'static str) -> Result<i32, SddpError> {
    i32::try_from(len).map_err(|_| {
        SddpError::Communication(InvalidBufferSize {
            operation,
            expected: i32::MAX as usize,
            actual: len,
        })
    })
}

/// Broadcast rank 0's buffer length. A length past the `i32` count limit
/// travels as `-1`, so every rank fails after the same broadcast instead of
/// rank 0 leaving the peers blocked in it.
///
/// # Errors
///
/// Rank 0 gets `SddpError::Communication(CommError::InvalidBufferSize { .. })`
/// for such a length; every other rank gets `SddpError::Validation`.
fn broadcast_len<C: Communicator>(
    comm: &C,
    len: usize,
    operation: &'static str,
) -> Result<usize, SddpError> {
    let own = checked_mpi_count(len, operation);
    let mut buf = [own.as_ref().map_or(-1, |&count| count)];
    comm.broadcast(&mut buf, 0).map_err(SddpError::from)?;
    own?;
    usize::try_from(buf[0]).map_err(|_| {
        SddpError::Validation(format!(
            "{operation}: rank 0's payload does not fit an i32 count"
        ))
    })
}

/// This rank's local scenario-0 basis at every node, metadata included
/// (`cut_row_slots`, `state_at_capture`, `base_row_count`), so a reconstruction
/// from it keeps full slot identity. On rank 0 it is what
/// [`broadcast_basis_cache`] delivers to every rank.
fn rank_local_basis_cache(basis_store: &BasisStore) -> Vec<Option<CapturedBasis>> {
    (0..basis_store.num_nodes())
        .map(|t| basis_store.get(0, NodePos(t)).cloned())
        .collect()
}

/// Build a `basis_cache` from global scenario 0, broadcasting rank 0's bases to
/// all ranks so every rank starts simulation from an identical warm-start vertex.
///
/// Scenario 0 (not the last local scenario) is broadcast because `basis_store`
/// is local-indexed: each rank's last local scenario follows a distinct noise
/// realisation and would yield divergent bases, whereas rank 0's local
/// scenario 0 is always global scenario 0 (`my_fwd_offset == 0`).
///
/// The two-buffer wire layout is owned by
/// [`CapturedBasis::to_broadcast_payload`](crate::workspace::CapturedBasis::to_broadcast_payload)
/// /
/// [`try_from_broadcast_payload`](crate::workspace::CapturedBasis::try_from_broadcast_payload);
/// this function only sequences the four MPI broadcasts (length then payload,
/// for the i32 and f64 buffers). Single-rank runs skip the broadcast and return
/// [`rank_local_basis_cache`].
///
/// Each reconstructed basis's `node_id` now rides the wire payload
/// (`try_from_broadcast_payload` reads it), so no out-of-band fill from a
/// per-stage node array is needed. The single-rank clone path likewise keeps
/// the `node_id` capture set via `write_capture_metadata`. The cache is sized
/// and keyed by `basis_store.num_nodes()` (one slot per canonical node), so a
/// branching graph's leaf nodes are broadcast too, never truncated at
/// `num_stages`.
///
/// # Errors
///
/// Returns `SddpError::Communication(CommError::InvalidBufferSize { .. })` on
/// rank 0 when a buffer length exceeds `i32::MAX` (every other rank gets
/// `SddpError::Validation` after the same broadcast, see [`broadcast_len`]),
/// `SddpError::Communication` when a `comm.broadcast` fails, or
/// `SddpError::Validation` when a length prefix is inconsistent with the
/// received buffer (naming the offending node).
pub(crate) fn broadcast_basis_cache<C: Communicator>(
    basis_store: &BasisStore,
    comm: &C,
) -> Result<Vec<Option<CapturedBasis>>, SddpError> {
    let num_nodes = basis_store.num_nodes();
    if comm.size() == 1 {
        return Ok(rank_local_basis_cache(basis_store));
    }

    // Multi-rank path: pack rank 0's scenario-0 full CapturedBasis into two
    // flat buffers (one i32 for integer/status fields, one f64 for
    // state_at_capture) and broadcast both.
    //
    // Wire format is owned by CapturedBasis::to_broadcast_payload /
    // CapturedBasis::try_from_broadcast_payload. The None case writes a
    // 0_i32 sentinel directly; the Some case delegates to the method.
    let mut buf: Vec<i32> = Vec::new();
    let mut f64_buf: Vec<f64> = Vec::new();
    if comm.rank() == 0 {
        for t in 0..num_nodes {
            match basis_store.get(0, NodePos(t)) {
                None => buf.push(0_i32),
                Some(captured) => captured.to_broadcast_payload(&mut buf, &mut f64_buf),
            }
        }
    }

    let total_len = broadcast_len(comm, buf.len(), "broadcast_basis_cache_i32")?;
    buf.resize(total_len, 0_i32);
    comm.broadcast(&mut buf, 0).map_err(SddpError::from)?;

    let f64_total_len = broadcast_len(comm, f64_buf.len(), "broadcast_basis_cache_f64")?;
    f64_buf.resize(f64_total_len, 0.0_f64);
    comm.broadcast(&mut f64_buf, 0).map_err(SddpError::from)?;

    let mut cache: Vec<Option<CapturedBasis>> = Vec::with_capacity(num_nodes);
    let mut pos = 0_usize;
    let mut f64_pos = 0_usize;
    for node in 0..num_nodes {
        let captured = CapturedBasis::try_from_broadcast_payload(
            node,
            &buf,
            &mut pos,
            &f64_buf,
            &mut f64_pos,
        )?;
        cache.push(captured);
    }

    Ok(cache)
}

/// Gather every rank's stored bases to rank 0, keyed by global scenario
/// (`fwd_offset + m` for local row `m`). Every rank calls it under the same
/// condition; per canonical node it issues one `allgatherv` of the two payload
/// lengths and one each of the `i32` and `f64` payloads, whose layout
/// [`CapturedBasis::to_broadcast_payload`] owns. Rank 0 returns the complete
/// store, every other rank `None`.
///
/// # Errors
///
/// `SddpError::Communication` when a collective fails or a node's gathered
/// payload exceeds an `i32` count, and `SddpError::Validation` when a rank's
/// payload does not fit an `i32` count or rank 0 cannot decode a segment. Every
/// rank meets a count error at the same node; a decode error is returned only
/// after every node has been gathered, so no rank leaves the collectives early.
pub(crate) fn gather_basis_store<C: Communicator>(
    basis_store: &BasisStore,
    ranks: &RankDistribution,
    comm: &C,
) -> Result<Option<CheckpointBasisStore>, SddpError> {
    let num_ranks = ranks.num_ranks;
    let num_nodes = basis_store.num_nodes();
    let is_root = comm.rank() == 0;
    let rank_passes = per_rank_counts(ranks.num_total_forward_passes, num_ranks);

    let mut int_send: Vec<i32> = Vec::new();
    let mut float_send: Vec<f64> = Vec::new();
    let mut int_recv: Vec<i32> = Vec::new();
    let mut float_recv: Vec<f64> = Vec::new();
    let mut lengths = vec![0_i32; 2 * num_ranks];
    let length_counts = vec![2_usize; num_ranks];
    let length_displs: Vec<usize> = (0..num_ranks).map(|r| 2 * r).collect();
    let (mut int_counts, mut int_displs) = (vec![0_usize; num_ranks], vec![0_usize; num_ranks]);
    let (mut float_counts, mut float_displs) = (vec![0_usize; num_ranks], vec![0_usize; num_ranks]);

    let mut gathered = CheckpointBasisStore {
        nodes: Vec::with_capacity(if is_root { num_nodes } else { 0 }),
        complete: true,
    };
    let mut decode_error = None;

    for node in 0..num_nodes {
        int_send.clear();
        float_send.clear();
        for m in 0..ranks.my_actual_fwd {
            match basis_store.get(m, NodePos(node)) {
                None => int_send.push(0_i32),
                Some(captured) => captured.to_broadcast_payload(&mut int_send, &mut float_send),
            }
        }
        // A length past `i32::MAX` travels as -1, so every rank fails at this node.
        let own_lengths = [
            i32::try_from(int_send.len()).unwrap_or(-1),
            i32::try_from(float_send.len()).unwrap_or(-1),
        ];
        lengths.fill(0);
        comm.allgatherv(&own_lengths, &mut lengths, &length_counts, &length_displs)
            .map_err(SddpError::from)?;

        let int_total = segment_layout(
            lengths.iter().step_by(2),
            node,
            "gather_basis_store_i32",
            &mut int_counts,
            &mut int_displs,
        )?;
        let float_total = segment_layout(
            lengths.iter().skip(1).step_by(2),
            node,
            "gather_basis_store_f64",
            &mut float_counts,
            &mut float_displs,
        )?;
        int_recv.resize(int_total, 0_i32);
        float_recv.resize(float_total, 0.0_f64);
        comm.allgatherv(&int_send, &mut int_recv, &int_counts, &int_displs)
            .map_err(SddpError::from)?;
        comm.allgatherv(&float_send, &mut float_recv, &float_counts, &float_displs)
            .map_err(SddpError::from)?;

        if !is_root || decode_error.is_some() {
            continue;
        }
        let decode_node = || {
            let mut entries = Vec::new();
            let mut first_scenario = 0;
            for (rank, &passes) in rank_passes.iter().enumerate() {
                decode_rank_segment(
                    node,
                    rank,
                    &int_recv[int_displs[rank]..int_displs[rank] + int_counts[rank]],
                    &float_recv[float_displs[rank]..float_displs[rank] + float_counts[rank]],
                    first_scenario,
                    passes,
                    &mut entries,
                )?;
                first_scenario += passes;
            }
            Ok::<_, SddpError>(entries)
        };
        match decode_node() {
            Ok(entries) => gathered.nodes.push(entries),
            Err(e) => decode_error = Some(e),
        }
    }

    if let Some(e) = decode_error {
        return Err(e);
    }
    Ok(is_root.then_some(gathered))
}

/// Fill one payload buffer's per-rank counts and displacements from the
/// gathered lengths, returning the buffer's total length.
///
/// # Errors
///
/// `SddpError::Validation` naming the rank whose length is negative (it did
/// not fit an `i32`), `SddpError::Communication` when the total exceeds
/// `i32::MAX`.
fn segment_layout<'a>(
    lengths: impl Iterator<Item = &'a i32>,
    node: usize,
    operation: &'static str,
    counts: &mut [usize],
    displs: &mut [usize],
) -> Result<usize, SddpError> {
    let mut total = 0_usize;
    for (rank, ((count, displ), &length)) in counts
        .iter_mut()
        .zip(displs.iter_mut())
        .zip(lengths)
        .enumerate()
    {
        *count = usize::try_from(length).map_err(|_| {
            SddpError::Validation(format!(
                "{operation}: rank {rank}'s basis payload at node {node} does not fit an i32 count"
            ))
        })?;
        *displ = total;
        total += *count;
    }
    checked_mpi_count(total, operation)?;
    Ok(total)
}

/// Decode one rank's segment of a node into `entries`, numbering its rows from
/// `first_scenario` and keeping the rows that hold a basis.
///
/// # Errors
///
/// `SddpError::Validation` when the segment holds more rows than the rank's
/// `passes`, leaves `f64` values unread, or fails
/// [`CapturedBasis::try_from_broadcast_payload`].
fn decode_rank_segment(
    node: usize,
    rank: usize,
    i32_segment: &[i32],
    f64_segment: &[f64],
    first_scenario: usize,
    passes: usize,
    entries: &mut Vec<(u32, CapturedBasis)>,
) -> Result<(), SddpError> {
    let (mut i32_pos, mut f64_pos) = (0_usize, 0_usize);
    let mut row = 0_usize;
    while i32_pos < i32_segment.len() {
        if row == passes {
            return Err(SddpError::Validation(format!(
                "gather_basis_store: rank {rank} sent more than its {passes} rows at node {node}"
            )));
        }
        if let Some(mut captured) = CapturedBasis::try_from_broadcast_payload(
            node,
            i32_segment,
            &mut i32_pos,
            f64_segment,
            &mut f64_pos,
        )? {
            captured.state_at_capture = Vec::new();
            entries.push((global_scenario(first_scenario, row), captured));
        }
        row += 1;
    }
    if f64_pos != f64_segment.len() {
        return Err(SddpError::Validation(format!(
            "gather_basis_store: rank {rank} sent {} f64 values its rows at node {node} do not read",
            f64_segment.len() - f64_pos
        )));
    }
    Ok(())
}

// Rationale: `LoopConfig::forward_passes` is a `u32`, so every global scenario fits.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn global_scenario(first_scenario: usize, row: usize) -> u32 {
    (first_scenario + row) as u32
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

/// Execute the SDDP training loop.
///
/// Allocates all workspace buffers, runs the iteration loop until a stopping
/// rule triggers or `config.max_iterations` is reached, and returns a
/// [`TrainingOutcome`] summarising the final convergence statistics.
///
/// ## Event channel
///
/// When `config.event_sender` is `Some`, typed [`novomodelo_core::TrainingEvent`]
/// values are emitted at each lifecycle boundary. Send failures (receiver
/// dropped) are silently ignored so they cannot interrupt training.
///
/// ## Cut selection
///
/// When `cut_selection` is `Some(strategy)`, its `should_run(iteration)` gate
/// controls how often each stage's pool is scanned for inactive cuts. When
/// `None`, no [`novomodelo_core::TrainingEvent::PolicySelectionComplete`] events are
/// emitted.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when an LP has no feasible
/// solution. Returns `Err(SddpError::Solver(_))` for other solver failures.
/// Returns `Err(SddpError::Communication(_))` when a collective operation
/// fails.
///
/// # Examples
///
/// `fcf`, `stage_ctx`, `training_ctx` and `comm` come from a built
/// [`StudySetup`](crate::setup::StudySetup), so the example is not compiled.
///
/// ```rust,ignore
/// use novomodelo_sddp::config::{CutManagementConfig, EventConfig, LoopConfig};
/// use novomodelo_sddp::{LoadedPolicy, RiskMeasure, SolverProfiles, TrainingConfig, train};
/// use novomodelo_solver::ActiveSolver;
///
/// let mut solver = ActiveSolver::new()?;
/// let config = TrainingConfig {
///     loop_config: LoopConfig { forward_passes: 100, max_iterations: 100, ..LoopConfig::default() },
///     cut_management: CutManagementConfig {
///         risk_measures: vec![RiskMeasure::Expectation; num_stages],
///         ..CutManagementConfig::default()
///     },
///     events: EventConfig::default(),
/// };
///
/// let outcome = train(
///     &mut solver, config, &mut fcf, &stage_ctx, &training_ctx, &comm,
///     ActiveSolver::new, LoadedPolicy::default(), SolverProfiles::default(),
/// )?;
///
/// println!("stopped after {} iterations, gap={:.4}", outcome.result.iterations, outcome.result.final_gap);
/// ```
///
/// # Panics
///
/// In debug builds, panics if `templates.len() != horizon.num_stages()` or if
/// `config.cut_management.risk_measures.len() != horizon.num_stages()` or if
/// `training_ctx.stochastic.opening_tree().n_openings(0) == 0`.
///
/// Always panics if `comm.rank() > i32::MAX`. MPI world sizes are bounded well
/// below this on all real systems.
pub fn train<S, C: Communicator>(
    solver: &mut S,
    config: TrainingConfig,
    fcf: &mut FutureCostFunction,
    stage_ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    comm: &C,
    solver_factory: impl Fn() -> Result<S, SolverError>,
    mut loaded: LoadedPolicy,
    solver_profiles: SolverProfiles,
) -> Result<TrainingOutcome, SddpError>
where
    S: SolverInterface<Profile = ActiveProfile> + Send,
{
    let mut session = TrainingSession::new(
        solver,
        config,
        fcf,
        stage_ctx,
        training_ctx,
        comm,
        solver_factory,
        std::mem::take(&mut loaded.trial_states),
        solver_profiles,
    )?;
    // Must seed the basis store before `prime_frozen_templates` freezes the loaded
    // cuts into the templates.
    session.seed_basis_store(&loaded);
    session.prime_frozen_templates();
    for iteration in session.iteration_range() {
        match session.run_iteration(iteration) {
            Ok(IterationOutcome::Continue) => {}
            Ok(IterationOutcome::Converged | IterationOutcome::Shutdown) => break,
            Err(e) => return Ok(session.finalize_with_error(e)),
        }
    }
    session.finalize()
}

#[cfg(test)]
mod tests;
