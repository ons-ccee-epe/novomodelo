use crate::SddpError;

/// Test-only backward-pass shim that owns per-call scratch, so the tests in
/// this module can exercise the backward pass without threading a full
/// `TrainingSession` through every fixture.
fn run_backward_pass<S, C: Communicator>(
    inputs: &mut BackwardPassInputs<'_, S, C>,
) -> Result<BackwardResult, SddpError>
where
    S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
{
    run_backward_pass_configured(inputs, |_| {})
}

/// [`run_backward_pass`] with `configure` applied to the state before the run.
fn run_backward_pass_configured<S, C: Communicator>(
    inputs: &mut BackwardPassInputs<'_, S, C>,
    configure: impl FnOnce(&mut BackwardPassState),
) -> Result<BackwardResult, SddpError>
where
    S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
{
    run_backward_pass_inspected(inputs, configure, |_| {})
}

/// [`run_backward_pass`] returning a copy of the stage statistics the run recorded.
fn run_backward_pass_stage_stats<S, C: Communicator>(
    inputs: &mut BackwardPassInputs<'_, S, C>,
) -> Result<Vec<(usize, Vec<StageWorkerOpeningDelta>)>, SddpError>
where
    S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
{
    let mut stage_stats = Vec::new();
    let _ = run_backward_pass_inspected(
        inputs,
        |_| {},
        |state| {
            stage_stats = state
                .stage_stats()
                .map(|(stage, entries)| (stage, entries.to_vec()))
                .collect();
        },
    )?;
    Ok(stage_stats)
}

/// [`run_backward_pass_configured`] with `inspect` applied to the state after `configure` and
/// again after the run.
fn run_backward_pass_inspected<S, C: Communicator>(
    inputs: &mut BackwardPassInputs<'_, S, C>,
    configure: impl FnOnce(&mut BackwardPassState),
    mut inspect: impl FnMut(&BackwardPassState),
) -> Result<BackwardResult, SddpError>
where
    S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
{
    let n_workers_local = inputs.workspaces.len();
    let n_ranks = inputs.comm.size();
    let num_stages = inputs.training_ctx.horizon.num_stages();
    let bwd_max_openings = (0..num_stages)
        .map(|t| inputs.training_ctx.stochastic.opening_tree().n_openings(t))
        .max()
        .unwrap_or(0);
    let mut bwd_state = BackwardPassState::new(
        n_workers_local,
        n_ranks,
        bwd_max_openings,
        inputs.local_work,
        inputs.training_ctx.state,
        inputs.training_ctx.horizon,
    );
    configure(&mut bwd_state);
    inspect(&bwd_state);
    let result = bwd_state.run(inputs);
    inspect(&bwd_state);
    result
}

use std::any::TypeId;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use novomodelo_comm::{CommData, CommError, Communicator, ReduceOp};
use novomodelo_solver::{
    Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};

use novomodelo_core::scenario::SamplingScheme;
use novomodelo_core::{StageStateConfig, WorkerPhaseTimings};
use novomodelo_io::config::BackwardScheduler;
use novomodelo_stochastic::{StochasticContext, evaluate_par_noise_term_batch};

use super::{
    BackwardResult, StageNoiseTable, StageWorkerOpeningDelta, by_scenario::prepare_child_lp,
    take_first_error,
};
use crate::{
    backward_pass_state::{BackwardPassInputs, BackwardPassState},
    build_cut_row_batch_into,
    context::{StageContext, TrainingContext},
    cut::{
        CutPool, FutureCostFunction,
        chain_rule::{ChainRuleInputs, ChainRuleScratch, OpeningDuals, per_opening_cut_into},
        rhs_reference::cut_noise_term,
        row::build_cut_row_batch,
    },
    cut_sync::CutSyncBuffers,
    forward::{EnumeratedForwardScratch, StageKey, run_forward_stage},
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::{InflowCoupling, PatchBuffer, StageGeometry},
    lp::indexer::{
        ChainRuleSlots, CutStateProjection, HydroSys, StateDim, StateRegion, StateSpace,
        StudyDimensions,
    },
    risk_measure::{BackwardOutcome, RiskMeasure, RiskMeasureScratch},
    setup::NodeId,
    setup::node_graph::{NodeGraph, NodePos, StageIdx, Traversal},
    solve::StageOpeningSolver,
    solver_stats::SolverStatsDelta,
    state_exchange::ExchangeBuffers,
    test_support::{
        self, StageContextFixture,
        bound_recording::{BoundRecordingSolver, BoundWrite, RecordedSolve},
        empty_row_batch, equipment_free_geometry, permissive_state_boxes,
    },
    training::stage_solve_prep::{BracketSource, LoadedCutRows, StageSolvePrep, StateSource},
    trajectory::{FrozenBrackets, TrajectoryRecord},
    workspace::{BackwardAccumulators, BasisStore, CapturedBasis, ScratchBuffers, SolverWorkspace},
};

/// Owned backing arrays for a single-successor (chain-degenerate) reified
/// [`super::SuccessorOutcomes`], used by the backward-pass unit tests. The
/// pool-indexed `frozen`/`cut_batches` are padded up to `pool_id` so `child(0)`
/// resolves the successor's LP data.
struct SingleSuccessor {
    entries: Vec<super::SuccessorEntry>,
    active_slots: Vec<usize>,
    frozen: Vec<StageTemplate>,
    cut_batches: Vec<RowBatch>,
}

impl SingleSuccessor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        successor_node: NodePos,
        successor_node_id: NodeId,
        pool_id: usize,
        num_cuts_at_successor: usize,
        populated_count: usize,
        active_slots: Vec<usize>,
        openings: crate::setup::node_graph::NodeOpenings,
        n_openings: usize,
        frozen_template: StageTemplate,
        cut_batch: RowBatch,
        chain_rule_slots: ChainRuleSlots,
    ) -> Self {
        let mut frozen = Vec::with_capacity(pool_id + 1);
        let mut cut_batches = Vec::with_capacity(pool_id + 1);
        for _ in 0..pool_id {
            frozen.push(frozen_template.clone());
            cut_batches.push(empty_row_batch());
        }
        frozen.push(frozen_template);
        cut_batches.push(cut_batch);
        let active_len = active_slots.len();
        Self {
            entries: vec![super::SuccessorEntry {
                successor_node,
                successor_node_id,
                pool_id,
                num_cuts_at_successor,
                populated_count,
                active_slots: 0..active_len,
                metadata_offset: 0,
                openings,
                outcome_range: 0..n_openings,
                chain_rule_slots,
            }],
            active_slots,
            frozen,
            cut_batches,
        }
    }

    fn outcomes<'a>(
        &'a self,
        pools: &'a [crate::cut::pool::CutPool],
        template_num_rows: usize,
        cut_activity_tolerance: Option<f64>,
    ) -> super::SuccessorOutcomes<'a> {
        super::SuccessorOutcomes::new(
            &self.entries,
            &self.active_slots,
            &self.frozen,
            &self.cut_batches,
            pools,
            template_num_rows,
            cut_activity_tolerance,
        )
    }
}

/// One successor child for the multi-child reified-fan builder [`MultiSuccessor`].
struct ChildSpec {
    successor_node: NodePos,
    successor_node_id: NodeId,
    pool_id: usize,
    openings: crate::setup::node_graph::NodeOpenings,
    num_cuts_at_successor: usize,
    populated_count: usize,
    active_slots: Vec<usize>,
    chain_rule_slots: ChainRuleSlots,
}

/// Owned backing arrays for a MULTI-child reified [`super::SuccessorOutcomes`] — a
/// genuine fan the single-child [`SingleSuccessor`] cannot represent. Children carry
/// their own pool, External column, node id, and per-pool binding-metadata region
/// (`metadata_offset`), so a test can assert none of child 0's data is reused.
struct MultiSuccessor {
    entries: Vec<super::SuccessorEntry>,
    active_slots: Vec<usize>,
    frozen: Vec<StageTemplate>,
    cut_batches: Vec<RowBatch>,
}

impl MultiSuccessor {
    fn new(
        children: Vec<ChildSpec>,
        frozen: Vec<StageTemplate>,
        cut_batches: Vec<RowBatch>,
    ) -> Self {
        let mut entries = Vec::with_capacity(children.len());
        let mut active_slots = Vec::new();
        let mut outcome_offset = 0usize;
        let mut metadata_offset = 0usize;
        for c in children {
            let outcome_len = c.openings.len;
            let slots_start = active_slots.len();
            active_slots.extend(c.active_slots.iter().copied());
            entries.push(super::SuccessorEntry {
                successor_node: c.successor_node,
                successor_node_id: c.successor_node_id,
                pool_id: c.pool_id,
                num_cuts_at_successor: c.num_cuts_at_successor,
                populated_count: c.populated_count,
                active_slots: slots_start..active_slots.len(),
                metadata_offset,
                openings: c.openings,
                outcome_range: outcome_offset..outcome_offset + outcome_len,
                chain_rule_slots: c.chain_rule_slots,
            });
            outcome_offset += outcome_len;
            metadata_offset += c.populated_count;
        }
        Self {
            entries,
            active_slots,
            frozen,
            cut_batches,
        }
    }

    fn outcomes<'a>(
        &'a self,
        pools: &'a [crate::cut::pool::CutPool],
        template_num_rows: usize,
        cut_activity_tolerance: Option<f64>,
    ) -> super::SuccessorOutcomes<'a> {
        super::SuccessorOutcomes::new(
            &self.entries,
            &self.active_slots,
            &self.frozen,
            &self.cut_batches,
            pools,
            template_num_rows,
            cut_activity_tolerance,
        )
    }
}

fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
    (0..n_stages).map(|_| empty_row_batch()).collect()
}

fn chain_rule_slots_of(
    training_ctx: &TrainingContext<'_>,
    child_pool: usize,
    generating: NodePos,
) -> ChainRuleSlots {
    ChainRuleSlots::compose(
        training_ctx.state,
        &training_ctx.cut_state_layouts[child_pool],
        &training_ctx.cut_state_layouts[training_ctx.node_graph.nodes[generating].pool_id],
    )
}

/// Stub communicator for tests (single-rank).
struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        unreachable!("StubComm broadcast not used in backward pass tests")
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Mock solver for testing: returns fixed solution or infeasible error on demand.
///
/// Buffer fields (`buf_primal`, `buf_dual`, `buf_reduced_costs`) store the
/// solution data that [`SolutionView`] borrows from. They are filled in
/// `solve` before the borrow is established.
struct MockSolver {
    solution: LpSolution,
    infeasible_at: Option<usize>,
    call_count: usize,
    /// Tracks the current number of rows (template + appended cuts).
    current_num_rows: usize,
    /// Number of times `solve(Some(&basis))` was called (warm-start calls).
    warm_start_calls: usize,
    /// Dual padding value for rows beyond the base template (cuts).
    /// Defaults to 0.0 (cuts not binding). Set to a positive value
    /// to make all cuts appear binding in tests.
    cut_dual_padding: f64,
    buf_primal: Vec<f64>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
}

impl MockSolver {
    fn always_ok(solution: LpSolution) -> Self {
        let base_rows = solution.dual.len();
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: None,
            call_count: 0,
            current_num_rows: base_rows,
            warm_start_calls: 0,
            cut_dual_padding: 0.0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
        }
    }

    fn infeasible_on(solution: LpSolution, n: usize) -> Self {
        Self {
            infeasible_at: Some(n),
            ..Self::always_ok(solution)
        }
    }

    /// Like `always_ok` but added cut rows return positive duals,
    /// making all existing cuts appear binding in subsequent solves.
    fn always_ok_with_binding_cuts(solution: LpSolution) -> Self {
        let mut s = Self::always_ok(solution);
        s.cut_dual_padding = 1.0;
        s
    }
}

impl SolverInterface for MockSolver {
    type Profile = novomodelo_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &novomodelo_solver::ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, template: &StageTemplate) {
        self.current_num_rows = template.num_rows;
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.current_num_rows += cuts.num_rows;
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_row_lower_bounds_range(&mut self, _first_row: usize, _lower: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<novomodelo_solver::SolutionView<'_>, SolverError> {
        if basis.is_some() {
            self.warm_start_calls += 1;
        }
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_at == Some(call) {
            return Err(SolverError::Infeasible);
        }
        // Fill internal buffers, resizing dual to match current LP row count.
        self.buf_primal.clone_from(&self.solution.primal);
        self.buf_dual.clone_from(&self.solution.dual);
        self.buf_dual
            .resize(self.current_num_rows, self.cut_dual_padding);
        self.buf_reduced_costs
            .clone_from(&self.solution.reduced_costs);
        Ok(novomodelo_solver::SolutionView {
            objective: self.solution.objective,
            primal: &self.buf_primal,
            dual: &self.buf_dual,
            reduced_costs: &self.buf_reduced_costs,
            iterations: self.solution.iterations,
            solve_time_seconds: self.solution.solve_time_seconds,
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

/// Records, per solve, the concatenated `set_row_bounds` lower values patched since
/// the previous solve and whether a warm basis was offered — so a fan test can prove
/// each child saw its OWN patched noise and its OWN `(m, child node)` basis. Returns
/// binding cut duals (`cut_dual > tolerance`) so per-child binding metadata fires.
struct PerChildProbeSolver {
    current_num_rows: usize,
    current_num_cols: usize,
    binding_dual: f64,
    pending_row_lower: Vec<f64>,
    per_solve_row_lower: Vec<Vec<f64>>,
    per_solve_basis_offered: Vec<bool>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
}

impl PerChildProbeSolver {
    fn new(binding_dual: f64) -> Self {
        Self {
            current_num_rows: 0,
            current_num_cols: 0,
            binding_dual,
            pending_row_lower: Vec::new(),
            per_solve_row_lower: Vec::new(),
            per_solve_basis_offered: Vec::new(),
            buf_dual: Vec::new(),
            buf_reduced_costs: Vec::new(),
        }
    }
}

impl SolverInterface for PerChildProbeSolver {
    type Profile = novomodelo_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &novomodelo_solver::ActiveProfile) {}
    fn solver_name_version(&self) -> String {
        "PerChildProbe 0.0.0".to_string()
    }
    fn load_model(&mut self, template: &StageTemplate) {
        self.current_num_rows = template.num_rows;
        self.current_num_cols = template.num_cols;
    }
    fn add_rows(&mut self, cuts: &RowBatch) {
        self.current_num_rows += cuts.num_rows;
    }
    fn set_row_bounds(&mut self, _indices: &[usize], lower: &[f64], _upper: &[f64]) {
        self.pending_row_lower.extend_from_slice(lower);
    }
    fn set_row_lower_bounds_range(&mut self, _first_row: usize, _lower: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<novomodelo_solver::SolutionView<'_>, SolverError> {
        self.per_solve_basis_offered.push(basis.is_some());
        self.per_solve_row_lower
            .push(std::mem::take(&mut self.pending_row_lower));
        self.buf_dual.clear();
        self.buf_dual
            .resize(self.current_num_rows, self.binding_dual);
        self.buf_reduced_costs.clear();
        self.buf_reduced_costs.resize(self.current_num_cols, 0.0);
        Ok(novomodelo_solver::SolutionView {
            objective: 0.0,
            primal: &[],
            dual: &self.buf_dual,
            reduced_costs: &self.buf_reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
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
        "PerChildProbe"
    }
}

// Columns: storage_out (0), storage_in (1), theta (2); one row.
fn minimal_template_1_0() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn solution_1_0(objective: f64, dual_storage: f64) -> LpSolution {
    // For the N=1, L=0 state layout: storage_in.start = N = 1.
    // Cut subgradients are read from reduced_costs[storage_in_col], so
    // reduced_costs[1] carries the incoming-storage column's reduced cost.
    let mut reduced_costs = vec![0.0; 3];
    reduced_costs[1] = dual_storage;
    LpSolution {
        objective,
        primal: vec![0.0, 0.0, 0.0],
        dual: vec![dual_storage],
        reduced_costs,
        iterations: 0,
        solve_time_seconds: 0.0,
    }
}

/// Resolve a [`StagedCut`]'s coefficient slice from its producing worker's
/// arena. Mirrors the production merge's
/// `workspaces[w].backward_accum.agg_arena[cut.coefficients_range]` read so
/// the in-file backward tests observe the same bytes the FCF receives.
fn staged_cut_coefficients<'a>(cut: &super::StagedCut, arena: &'a [f64]) -> &'a [f64] {
    &arena[cut.coefficients_range.clone()]
}

/// Wrap a `MockSolver` into a single-element `Vec<SolverWorkspace<MockSolver>>`
/// for tests that exercise the workspace-based backward-pass API.
///
/// The workspace is sized for `n_hydro=1`, `max_par_order=0`, and `n_state`
/// state dimensions.
fn single_workspace<S: SolverInterface + Send>(
    solver: S,
    n_state: usize,
) -> Vec<SolverWorkspace<S>> {
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
        current_state: Vec::with_capacity(n_state),
        scratch: ScratchBuffers::for_test(1),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    }]
}

/// Single `MockSolver` workspace with a bucket-aware `PatchBuffer`
/// (`hydro_count = 0`, `max_par_order = 0`), for the `pin_and_bracket`
/// bucket-pinning regression.
fn transit_bucket_only_workspace(
    solver: MockSolver,
    n_buckets: usize,
) -> SolverWorkspace<MockSolver> {
    SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(
            &test_support::state_layout_with_transit_buckets(
                0,
                0,
                (0..n_buckets).map(|d| (HydroSys::new(0), d)).collect(),
                vec![],
            ),
            &[],
            &[],
            &[],
        ),
        current_state: Vec::new(),
        scratch: ScratchBuffers::for_test(0),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    }
}

/// Create an empty `BasisStore` for `num_scenarios` scenarios and
/// `num_stages` stages (all slots `None`).
fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
    BasisStore::new(num_scenarios, num_stages)
}

/// Create a `BasisStore` with one slot pre-populated at
/// `[scenario][stage]` with the given `Basis`.
fn basis_store_with_one(
    num_scenarios: usize,
    num_stages: usize,
    scenario: usize,
    stage: usize,
    basis: Basis,
) -> BasisStore {
    let mut store = BasisStore::new(num_scenarios, num_stages);
    // Set `base_row_count` to the full row_status length and leave
    // `cut_row_slots` empty so the CapturedBasis invariant
    // (`row_status.len() == base_row_count + cut_row_slots.len()`) holds
    // by construction. These tests exercise the warm-start propagation
    // path; the reconstruction copies the template rows verbatim and
    // emits an empty cut block.
    let base_row_count = basis.row_status.len();
    // `stage` doubles as the node id: every fixture in this module drives a
    // chain (`chain_node_graph`), where `node_ids[t] == t`, so this must equal
    // the successor's `node_id` for the warm-start propagation path this
    // helper exercises to actually reach the warm path.
    let node_id = NodeId(i32::try_from(stage).expect("test fixture stage count fits in i32"));
    *store.get_mut(scenario, NodePos(stage)) = Some(CapturedBasis {
        basis,
        base_row_count,
        cut_row_slots: Vec::new(),
        node_id,
    });
    store
}

/// Gather buffers holding `states` as trial points, plus the records the backward
/// pass repacks them from.
///
/// The pass repacks `exchange` from `records` once per stage, so the two must
/// agree: both are derived here from the same [`test_support::trial_state_records`]
/// (which replicates each scenario's state across every stage), and the buffers are
/// pre-populated by driving the real `exchange` those records feed. A caller that
/// instead crafted the buffers independently of `records` would have its trial
/// points silently overwritten by the pass's own repack.
fn exchange_and_records(
    n_state: usize,
    states: &[Vec<f64>],
    n_stages: usize,
) -> (ExchangeBuffers, Vec<TrajectoryRecord>) {
    use novomodelo_comm::LocalBackend;

    let records = test_support::trial_state_records(states, n_stages);
    let mut bufs = ExchangeBuffers::new(&test_support::state_layout(n_state, 0), states.len(), 1);
    bufs.exchange(&records, StageIdx(0), n_stages, &LocalBackend)
        .unwrap();
    (bufs, records)
}

fn exchange_with_states(n_state: usize, states: Vec<Vec<f64>>) -> ExchangeBuffers {
    use novomodelo_comm::LocalBackend;

    let local_count = states.len();
    let mut bufs = ExchangeBuffers::new(&test_support::state_layout(n_state, 0), local_count, 1);
    let records: Vec<TrajectoryRecord> = states
        .into_iter()
        .map(|state| TrajectoryRecord {
            state,
            ..TrajectoryRecord::default()
        })
        .collect();

    let comm = LocalBackend;
    bufs.exchange(&records, StageIdx(0), 1, &comm).unwrap();
    bufs
}

fn make_stochastic_context(
    n_stages: usize,
    branching_factor: usize,
) -> novomodelo_stochastic::StochasticContext {
    make_stochastic_context_with_ar(n_stages, branching_factor, &[])
}

#[allow(clippy::too_many_lines)]
fn make_stochastic_context_with_ar(
    n_stages: usize,
    branching_factor: usize,
    ar_coefficients: &[f64],
) -> novomodelo_stochastic::StochasticContext {
    use chrono::NaiveDate;
    use novomodelo_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use novomodelo_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use novomodelo_stochastic::context::{
        ClassSchemes, OpeningTreeInputs, build_stochastic_context,
    };
    use std::collections::BTreeMap;

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(EntityId(0));

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    #[allow(clippy::cast_possible_truncation)]
    let inflow = |stage_idx: usize| InflowModel {
        hydro_id: EntityId(1),
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        stage_id: stage_idx as i32,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: ar_coefficients.to_vec(),
        residual_std_ratio: 1.0,
        annual: None,
    };

    let inflow_models: Vec<InflowModel> = (0..n_stages).map(inflow).collect();

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap();

    build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[test]
fn backward_result_fields_accessible() {
    let r = BackwardResult {
        cuts_generated: 6,
        elapsed_ms: 42,
        lp_solves: 0,
        state_exchange_time_ms: 0,
        cut_batch_build_time_ms: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        cut_sync_time_ms: 0,
    };
    assert_eq!(r.cuts_generated, 6);
    assert_eq!(r.elapsed_ms, 42);
    assert_eq!(r.state_exchange_time_ms, 0);
    assert_eq!(r.cut_batch_build_time_ms, 0);
    assert_eq!(r.setup_time_ms, 0);
    assert_eq!(r.load_imbalance_ms, 0);
    assert_eq!(r.scheduling_overhead_ms, 0);
    assert_eq!(r.cut_sync_time_ms, 0);
}

#[test]
fn backward_result_clone_and_debug() {
    let r = BackwardResult {
        cuts_generated: 3,
        elapsed_ms: 100,
        lp_solves: 0,
        state_exchange_time_ms: 0,
        cut_batch_build_time_ms: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        cut_sync_time_ms: 0,
    };
    let c = r.clone();
    assert_eq!(c.cuts_generated, 3);
    let s = format!("{r:?}");
    assert!(s.contains("BackwardResult"));
}

#[test]
fn dual_extraction_formula_coefficients_are_negated_duals() {
    let d0 = 3.5_f64;
    let d1 = -1.2_f64;
    let dual = [d0, d1];

    let coefficients: Vec<f64> = dual.iter().map(|&d| -d).collect();

    assert!((coefficients[0] - (-d0)).abs() < f64::EPSILON);
    assert!((coefficients[1] - (-d1)).abs() < f64::EPSILON);
}

#[test]
fn intercept_formula_matches_spec() {
    // alpha = Q - pi^T * x_hat
    let objective = 50.0_f64;
    let coefficients = [2.0_f64, -1.0_f64];
    let x_hat = [10.0_f64, 5.0_f64];
    let pi_dot_x: f64 = coefficients
        .iter()
        .zip(x_hat.iter())
        .map(|(p, x)| p * x)
        .sum();
    let intercept = objective - pi_dot_x;
    assert!((intercept - 35.0).abs() < f64::EPSILON);
}

#[test]
fn single_stage_system_produces_no_cuts() {
    // A 1-stage system has no stages with a successor, so the backward
    // sweep (0..0) is empty — zero cuts are generated.
    let stochastic = make_stochastic_context(1, 2);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0()];

    let n_state = state.n_state;
    let n_stages = 1_usize;
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![10.0], vec![20.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    assert_eq!(result.cuts_generated, 0);
    assert_eq!(fcf.total_active_cuts(), 0);
}

#[test]
fn two_stage_system_two_trial_states_generates_two_cuts_at_stage_0() {
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state; // 1
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);

    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![10.0], vec![20.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // 2 trial points × 1 stage with a successor = 2 cuts at stage 0.
    assert_eq!(result.cuts_generated, 2);
    assert_eq!(fcf.active_cuts(0).count(), 2);
    // Stage 1 (the last stage) gets no cuts.
    assert_eq!(fcf.active_cuts(1).count(), 0);
}

#[test]
fn cut_inserted_with_correct_stage_iteration_and_forward_pass_index() {
    // Acceptance criterion: iteration=2, forward_passes=3, global
    // trial point m=1 → fcf.add_cut(NodeId(0),stage=0, iteration=2, fpi=1, ...).
    // slot = warm_start + 2*3 + 1 = 7.
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

    // 3 trial points (forward_passes=3 on a single rank).
    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![5.0], vec![10.0], vec![15.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(50.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 2,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // Trial point m=1: slot = 0 + 2*3 + 1 = 7
    // Verify that pool[0].metadata(7) has the correct iteration and fpi.
    let meta = fcf.pools[0].metadata(7);
    assert_eq!(meta.iteration_generated, 2);
    assert_eq!(meta.forward_pass_index, 1);
}

#[test]
fn no_cuts_generated_at_last_stage() {
    let n_stages = 5_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -3.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // 1 trial point × 4 stages with successors = 4 cuts total.
    assert_eq!(result.cuts_generated, 4);
    for t in 0..4 {
        assert_eq!(fcf.active_cuts(t).count(), 1, "stage {t} should have 1 cut");
    }
    // The last stage (4) must have no cuts.
    assert_eq!(fcf.active_cuts(4).count(), 0, "stage 4 must have no cuts");
}

#[test]
fn elapsed_ms_is_non_negative() {
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 2);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![5.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(10.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // elapsed_ms is u64, so it is always >= 0.
    let _ = result.elapsed_ms;
}

#[test]
fn infeasible_solver_returns_sddp_infeasible_error() {
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(0.0, 0.0);
    let solver = MockSolver::infeasible_on(solution, 0);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    });

    assert!(
        matches!(result, Err(SddpError::Infeasible { .. })),
        "expected SddpError::Infeasible, got: {result:?}",
    );
}

#[test]
fn expectation_aggregation_mean_of_per_opening_intercepts() {
    use BackwardOutcome as BO;

    let outcomes = vec![
        BO {
            intercept: 10.0,
            coefficients: vec![],
            objective_value: 10.0,
        },
        BO {
            intercept: 20.0,
            coefficients: vec![],
            objective_value: 20.0,
        },
        BO {
            intercept: 30.0,
            coefficients: vec![],
            objective_value: 30.0,
        },
    ];
    let probs = vec![1.0 / 3.0; 3];
    let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
    assert!(
        (intercept - 20.0).abs() < 1e-10,
        "expected 20.0, got {intercept}"
    );
}

// ── Integration tests ─────────────────────────────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn cut_coefficients_and_intercept_match_dual_extraction_formula() {
    // Cut coefficients equal the raw dual, not its negation (sddp.md
    // Benders cut sign & subgradient extraction).
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(80.0, -3.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, intercept, coefficients) = &cuts[0];

    assert!(
        (intercept - 110.0).abs() < 1e-10,
        "expected intercept=110.0, got {intercept}"
    );
    assert_eq!(coefficients.len(), 1);
    assert!(
        (coefficients[0] - (-3.0)).abs() < 1e-10,
        "expected coefficient=-3.0, got {}",
        coefficients[0]
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn cut_gradient_sign_physically_correct() {
    // Physical invariant: more initial storage → lower future cost, so the
    // incoming-storage column's reduced cost π (negative) is kept as the cut
    // coefficient, not negated (sddp.md Benders cut sign & subgradient extraction).
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![50.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -2.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1, "expected exactly one cut");
    let (_, _intercept, coefficients) = &cuts[0];

    assert!(
        coefficients[0] < 0.0,
        "cut coefficient must be negative (more storage → less future cost), \
         got {} — likely the Benders cut sign bug has been reintroduced",
        coefficients[0]
    );
    assert!(
        (coefficients[0] - (-2.0)).abs() < 1e-10,
        "expected coefficient=-2.0, got {}",
        coefficients[0]
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn cut_is_tight_at_trial_state() {
    // Regression test: a Benders cut must be tight (exact) at the trial
    // point x̂ where it was generated. That is:
    //   intercept + coefficient * x̂ = Q(x̂)
    // where Q(x̂) = objective value of the subproblem at x̂.
    //
    // The cut equation is: θ ≥ intercept + coefficient * x
    // At x = x̂: θ ≥ Q(x̂) + π'(x̂ - x̂) = Q(x̂)
    //
    // If the sign is wrong (coefficient = -π instead of π), then:
    //   intercept + (-π) * x̂ ≠ Q(x̂) in general
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let x_hat = 30.0_f64;
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![x_hat]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let q_xhat = 200.0_f64; // subproblem objective at x̂
    let dual_storage = -4.0_f64;
    let solution = solution_1_0(q_xhat, dual_storage);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, intercept, coefficients) = &cuts[0];

    // Evaluate the cut at x̂: cut_value = intercept + coeff * x̂
    let cut_at_xhat = intercept + coefficients[0] * x_hat;

    // Must equal Q(x̂) (tightness property)
    assert!(
        (cut_at_xhat - q_xhat).abs() < 1e-10,
        "cut must be tight at trial point: \
         cut_value={cut_at_xhat}, Q(x̂)={q_xhat}, \
         intercept={intercept}, coeff={}, x̂={x_hat}",
        coefficients[0]
    );
}

#[test]
fn single_rank_backward_pass_with_local_backend_produces_correct_fcf() {
    use novomodelo_comm::LocalBackend;

    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 2_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![10.0], vec![20.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // 3-stage system: cuts at stages 0 and 1; 2 trial points each.
    // Total cuts = 2 stages × 2 trial points = 4.
    assert_eq!(result.cuts_generated, 4);
    assert_eq!(fcf.active_cuts(0).count(), 2);
    assert_eq!(fcf.active_cuts(1).count(), 2);
    assert_eq!(fcf.active_cuts(2).count(), 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn forward_pass_index_matches_global_scenario_index() {
    // Acceptance criterion: when a cut is generated for global trial point
    // m=5, then `fcf.add_cut(NodeId(0),stage, iteration, 5, ...)` is called with
    // forward_pass_index = m = 5.
    //
    // Setup: iteration=2, forward_passes=6 (6 scenarios on 1 rank), 1 opening.
    // ExchangeBuffers: local_count=6, num_ranks=1, total_scenarios=6.
    // state_at(5/6, 5%6) = state_at(0, 5) — valid.
    //
    // Slot formula: slot = warm_start(0) + 2*6 + 5 = 17.
    // The key invariant: forward_pass_index = m = 5.
    let n_stages = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 6_u32; // 6 scenarios on a single rank
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

    // 6 trial points (m = 0..5). ExchangeBuffers: local_count=6, num_ranks=1.
    let (mut exchange, records) = exchange_and_records(
        n_state,
        &[
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ],
        n_stages,
    );

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(50.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 2,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // m=5: slot = warm_start(0) + 2*6 + 5 = 17
    // The critical check: forward_pass_index in metadata equals global m=5.
    let meta = fcf.pools[0].metadata(17);
    assert_eq!(meta.iteration_generated, 2, "iteration_generated must be 2");
    assert_eq!(
        meta.forward_pass_index, 5,
        "forward_pass_index must be 5 (= global m)"
    );
}

// ── Unit tests: warm-start basis caching (backward pass) ──────────────────

#[test]
fn warm_start_uses_prepopulated_forward_basis() {
    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;

    // Pre-populate the basis store at (scenario=0, stage=1).
    // This simulates a forward pass having already solved stage 1 and cached its basis.
    let mut pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
    test_support::fill_consistent_basis(&mut pre_basis);
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
    assert_eq!(
        warm_start_calls, 1,
        "first opening at successor stage must call solve(Some(&basis)) \
         when basis_store.get(0, NodePos(1)) is pre-populated (warm_start_calls == 1, got {warm_start_calls})"
    );
}

#[test]
fn multi_opening_subsequent_openings_use_internal_hotstart() {
    let n_stages = 2_usize;
    let n_openings = 3_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm;

    // Start with an empty store — opening 1 must cold-start.
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // Opening 0 cold-starts (no basis in store); openings 1 and 2 use
    // solve(None) (HiGHS internal hot-start) instead of solve(Some(&working_basis)),
    // so no explicit warm-start calls are issued for subsequent openings.
    let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
    assert_eq!(
        warm_start_calls, 0,
        "P3b: no warm-start calls expected when BasisStore is empty \
         (warm_start_calls == 0, got {warm_start_calls})"
    );
}

#[test]
fn backward_solver_error_propagates() {
    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(0.0, 0.0);
    // The first backward solve (call 0) returns infeasible.
    let solver = MockSolver::infeasible_on(solution, 0);
    let comm = StubComm;

    // Pre-populate the store — error should propagate regardless.
    let mut pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
    test_support::fill_consistent_basis(&mut pre_basis);
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    });

    assert!(
        matches!(result, Err(SddpError::Infeasible { .. })),
        "expected SddpError::Infeasible, got: {result:?}",
    );
    // The BasisStore is not mutated by the backward pass — the working_basis
    // is a local variable dropped on error. The store slot at (0, 1) remains
    // as it was; this just verifies the store is untouched by an error path.
    assert!(
        basis_store.get(0, NodePos(1)).is_some(),
        "BasisStore must not be mutated by the backward pass error path"
    );
}

// ── Parallel cut determinism ──────────────────────────────────────────────

/// When `run_backward_pass` runs with 1 workspace vs 4 workspaces given
/// the same input data, the FCF pools contain identical cuts (same intercept,
/// coefficient vectors, and slot assignments for each trial point).
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn test_backward_pass_parallel_cut_determinism() {
    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let n_trial_states = 8_usize;

    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    #[allow(clippy::cast_possible_truncation)]
    let forward_passes = n_trial_states as u32;

    // Build 8 distinct trial-point states.
    let states: Vec<Vec<f64>> = (0..n_trial_states).map(|i| vec![i as f64 + 1.0]).collect();
    let (mut exchange, records) = exchange_and_records(n_state, &states, n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let solution = solution_1_0(100.0, -5.0);
    let comm = StubComm;

    // --- Run with 1 workspace ---
    let mut fcf_1 =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let solver_1 = MockSolver::always_ok(solution.clone());
    let mut workspaces_1 = vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver_1),
        patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
        current_state: Vec::with_capacity(n_state),
        scratch: ScratchBuffers::for_test(1),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    }];
    let mut basis_store_1 = empty_basis_store(exchange.local_count(), n_stages);
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let geometry = equipment_free_geometry(&vec![0; templates.len()]);
    let fixture =
        StageContextFixture::new(&templates, &state_boxes, &geometry).without_inflow_entries(1);
    let ctx = fixture.ctx();
    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces_1,
        basis_store: &mut basis_store_1,
        ctx: &ctx,
        frozen: &mut templates.clone(),
        fcf: &mut fcf_1,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // --- Run with 4 workspaces ---
    let mut fcf_4 =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4_i32)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: idx,
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::for_test(1),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        })
        .collect();
    let mut basis_store_4 = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces_4,
        basis_store: &mut basis_store_4,
        ctx: &ctx,
        frozen: &mut templates.clone(),
        fcf: &mut fcf_4,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // --- Verify identical FCF contents for all non-last stages ---
    for t in 0..(n_stages - 1) {
        let cuts_1: Vec<_> = fcf_1.active_cuts(t).collect();
        let cuts_4: Vec<_> = fcf_4.active_cuts(t).collect();

        assert_eq!(
            cuts_1.len(),
            cuts_4.len(),
            "stage {t}: cut count differs (1 workspace: {}, 4 workspaces: {})",
            cuts_1.len(),
            cuts_4.len()
        );

        for (idx, ((slot_1, intercept_1, coeff_1), (slot_4, intercept_4, coeff_4))) in
            cuts_1.iter().zip(cuts_4.iter()).enumerate()
        {
            assert_eq!(
                slot_1, slot_4,
                "stage {t} cut {idx}: slot mismatch ({slot_1} vs {slot_4})"
            );
            assert!(
                (intercept_1 - intercept_4).abs() < 1e-12,
                "stage {t} cut {idx}: intercept mismatch ({intercept_1} vs {intercept_4})"
            );
            assert_eq!(
                coeff_1.len(),
                coeff_4.len(),
                "stage {t} cut {idx}: coefficient vector length mismatch"
            );
            for (j, (c1, c4)) in coeff_1.iter().zip(coeff_4.iter()).enumerate() {
                assert!(
                    (c1 - c4).abs() < 1e-12,
                    "stage {t} cut {idx} coeff[{j}]: {c1} vs {c4}"
                );
            }
        }
    }

    // Last stage must have no cuts in both.
    assert_eq!(fcf_1.active_cuts(n_stages - 1).count(), 0);
    assert_eq!(fcf_4.active_cuts(n_stages - 1).count(), 0);
}

// ── Load noise wiring tests (backward pass) ──────────────────────────────

/// Build a 2-stage `StochasticContext` with 1 hydro and 1 stochastic load bus.
///
/// The noise vector dimension is `n_hydros + n_load_buses = 2`.
/// Stage 0 uses `branching_factor` openings; stage 1 is the successor solved
/// in the backward pass opening loop.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context_with_load(
    n_stages: usize,
    branching_factor: usize,
    mean_mw: f64,
    std_mw: f64,
) -> novomodelo_stochastic::StochasticContext {
    use chrono::NaiveDate;
    use novomodelo_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::{CorrelationModel, InflowModel, LoadModel};
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use novomodelo_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use novomodelo_stochastic::context::{
        ClassSchemes, OpeningTreeInputs, build_stochastic_context,
    };

    let bus0 = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(10),
        name: "H10".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(EntityId(0));

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|idx| InflowModel {
            hydro_id: EntityId(10),
            stage_id: idx as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|idx| LoadModel {
            bus_id: EntityId(1),
            stage_id: idx as i32,
            mean_mw,
            std_mw,
        })
        .collect();

    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: std::collections::BTreeMap::new(),
        schedule: vec![],
    };

    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .correlation(correlation)
        .build()
        .unwrap();

    build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

/// Given a backward pass with 1 stochastic load bus and opening noise
/// that includes a load component eta, the load balance row RHS in the patch
/// buffer is set to `max(0, mean + std * eta) * block_factor` before the solve.
///
/// We verify this indirectly: after the backward pass runs, `ws.scratch.load_rhs_buf`
/// must be non-empty and must contain a positive value (with mean=300, std=30
/// any reasonable eta produces a positive realization).
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_load_patches_applied() {
    // 2-stage system: backward pass solves at successor=1 for each opening.
    // n_hydros=1, n_load_buses=1, 1 block per stage.
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    // mean_mw=300 guarantees a positive realization for any reasonable eta draw.
    let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 300.0, 30.0);
    let state = test_support::state_layout(1, 0);

    // geometry_per_stage[successor=1].load_balance starts at 10; load_bus_indices=[0]; 1 block/stage.
    let load_bus_indices = vec![0_usize];
    let geometry_per_stage = vec![test_support::geometry_with_load_balance(10, 1, 1); n_stages];

    // PatchBuffer: n_hydros=1, max_par_order=0, n_load_buses=1, max_blocks=1.
    let patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage, &[]);

    // Template: 2 rows; row content is irrelevant here (this test exercises
    // only the load-balance patch, addressed via geometry_per_stage).
    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // MockSolver returns a fixed solution (1 state var, 1 dual entry).
    let solution = solution_1_0(100.0, -2.0);

    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: ScratchBuffers::for_test(1),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];

    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
            .without_inflow_entries(1)
            .load_bus_indices(&load_bus_indices)
            .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // After the backward pass, load_rhs_buf must have been populated with a
    // positive value for the last opening solved (mean=300, std=30 → positive).
    assert_eq!(
        workspaces[0].scratch.load_rhs_buf.len(),
        1,
        "load_rhs_buf must have 1 entry (1 load bus × 1 block)"
    );
    assert!(
        workspaces[0].scratch.load_rhs_buf[0] > 0.0,
        "load realization must be positive with mean=300, std=30: got {}",
        workspaces[0].scratch.load_rhs_buf[0]
    );
}

/// Given a backward pass with 0 stochastic load buses, `patch_count` equals
/// the rows the inflow coupling shifts (no load patches) and `load_rhs_buf`
/// stays empty.
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_no_load_buses_unchanged() {
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);

    let inflow_coupling = vec![test_support::inflow_coupling_with_entries(1, &[(0, 0)]); n_stages];
    let patch_buf = PatchBuffer::new(
        &test_support::state_layout(1, 0),
        &[],
        &[],
        &inflow_coupling,
    );

    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];

    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -2.0);
    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: ScratchBuffers::for_test(1),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];
    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let _ = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&[1_usize; 2]),
        )
        .inflow_coupling(&inflow_coupling)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    assert_eq!(
        workspaces[0].patch_buf.forward_patch_count(),
        inflow_coupling[0].touched_rows().len(),
        "forward_patch_count must equal the rows the inflow coupling shifts when n_load_buses=0, got {}",
        workspaces[0].patch_buf.forward_patch_count()
    );
    // load_rhs_buf must remain empty.
    assert!(
        workspaces[0].scratch.load_rhs_buf.is_empty(),
        "load_rhs_buf must be empty when n_load_buses=0"
    );
}

/// Given a backward pass with stochastic load, when Benders cut
/// coefficients are extracted, the cut coefficient array has length `n_state`
/// unchanged — load adds no state variables.
///
/// Setup: N=1 hydro, L=0 PAR lags → `n_state=1`. After the backward pass with
/// 1 load bus, each generated cut must have exactly 1 coefficient.
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_cut_coefficients_unaffected() {
    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 200.0, 20.0);
    let state = test_support::state_layout(1, 0);

    let load_bus_indices = vec![0_usize];
    let geometry_per_stage = vec![test_support::geometry_with_load_balance(10, 1, 1); n_stages];

    let patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage, &[]);

    let template = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![50.0, 100.0],
        row_upper: vec![50.0, 100.0],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let templates = vec![template; n_stages];

    let n_state = state.n_state; // 1
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(80.0, -3.0);
    let ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
        patch_buf,
        current_state: Vec::with_capacity(n_state),
        scratch: ScratchBuffers::for_test(1),
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    };
    let mut workspaces = vec![ws];
    let comm = StubComm;
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
            .without_inflow_entries(1)
            .load_bus_indices(&load_bus_indices)
            .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // Exactly 1 cut generated (1 trial point × 1 stage with a successor).
    assert_eq!(result.cuts_generated, 1);

    // The cut must have exactly n_state=1 coefficient.
    let cuts: Vec<_> = fcf.active_cuts(0).collect();
    assert_eq!(cuts.len(), 1);
    let (_, _intercept, coefficients) = &cuts[0];
    assert_eq!(
        coefficients.len(),
        n_state,
        "cut coefficients length must be n_state={n_state}, got {} — \
         load buses must not add state variables",
        coefficients.len()
    );
}

/// Structural invariant: per-stage cut sync inside the backward loop.
///
/// Verifies that after `run_backward_pass`, the cut synchronization has been
/// performed per-stage (not as a separate post-sweep loop). The structural
/// evidence is:
///
/// 1. `BackwardResult.cut_sync_time_ms` is populated (timing was captured).
/// 2. The FCF has the expected number of cuts per stage — same as single-rank
///    without sync, because single-rank sync is a no-op that does not change
///    results but exercises the code path.
/// 3. Using `LocalBackend` (the production single-rank communicator) instead
///    of `StubComm` exercises the full `sync_level_records` → allgatherv → deserialize
///    path, confirming no panics or data corruption.
///
/// True multi-rank correctness testing requires actual MPI and is out of
/// scope for CI. This test validates the structural invariant (sync is
/// called per-stage inside the loop) and exercises the full code path.
#[test]
#[allow(clippy::too_many_lines)]
fn per_stage_cut_sync_invariant_after_bug1_fix() {
    use novomodelo_comm::LocalBackend;

    let n_stages = 4_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![10.0], vec![20.0], vec![30.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok(solution);
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 1,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // 4-stage system: cuts at stages 0, 1, 2; 3 trial points each.
    // Total cuts = 3 stages × 3 trial points = 9.
    assert_eq!(result.cuts_generated, 9);

    // Each non-terminal stage has 3 cuts (one per trial point).
    assert_eq!(fcf.active_cuts(0).count(), 3, "stage 0 must have 3 cuts");
    assert_eq!(fcf.active_cuts(1).count(), 3, "stage 1 must have 3 cuts");
    assert_eq!(fcf.active_cuts(2).count(), 3, "stage 2 must have 3 cuts");
    assert_eq!(
        fcf.active_cuts(3).count(),
        0,
        "terminal stage must have 0 cuts"
    );

    // Verify cut_sync_time_ms was captured (structural evidence that
    // sync_level_records was called inside the backward loop).
    // For single-rank LocalBackend, sync is a no-op, so time should be
    // very small but the field must be populated (not default/garbage).
    // We just verify it's a valid non-negative value.
    assert!(
        result.cut_sync_time_ms < 10_000,
        "cut_sync_time_ms should be reasonable, got {}",
        result.cut_sync_time_ms
    );
}

/// What [`backward_iteration_3_over_three_old_cuts`] leaves behind.
struct BindingFixtureRun {
    result: BackwardResult,
    fcf: FutureCostFunction,
    workspaces: Vec<SolverWorkspace<MockSolver>>,
}

/// One backward pass at iteration 3 over a 3-stage chain whose pool 1 already
/// holds 3 cuts generated at iteration 1, with 3 trial points and 2 openings.
/// `MockSolver::always_ok_with_binding_cuts` returns a positive dual on every
/// cut row, so every cut a solve carries appears binding.
#[allow(clippy::too_many_lines)]
fn backward_iteration_3_over_three_old_cuts<C: Communicator>(
    comm: &C,
    configure: impl FnOnce(&mut BackwardPassState),
) -> BindingFixtureRun {
    // 3-stage system: backward loop processes t=1 then t=0.
    // Pool[1] already holds 3 cuts generated at iteration 1, which the iteration-3
    // frozen LP of stage 1 carries as 3 extra rows.
    // At t=1: generates cuts into pool[1], successor pool[2] is empty.
    // At t=0: generates cuts into pool[0], successor pool[1] has the old and the
    //         new cuts. Mock duals make them appear binding.
    let n_stages = 3_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];

    let n_state = state.n_state;
    let forward_passes = 3_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
    for forward_pass_index in 0..forward_passes {
        fcf.add_cut(
            NodeId(1),
            1,
            1,
            forward_pass_index,
            0.0,
            &vec![0.0; n_state],
        );
    }
    let mut frozen = templates.clone();
    frozen[1].num_rows += forward_passes as usize;
    let (mut exchange, records) =
        exchange_and_records(n_state, &[vec![10.0], vec![20.0], vec![30.0]], n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let solution = solution_1_0(100.0, -5.0);
    let solver = MockSolver::always_ok_with_binding_cuts(solution);
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

    let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let result = run_backward_pass_configured(
        &mut BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContextFixture::new(
                &templates,
                &state_boxes,
                &equipment_free_geometry(&vec![0; templates.len()]),
            )
            .without_inflow_entries(1)
            .ctx(),
            frozen: &mut frozen,
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                node_graph: &crate::test_support::chain_node_graph(&stochastic),
                horizon: &horizon,
                state: &state,
                cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
                study_dims: &test_support::study_dims(),
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                lag_accum_seed: &[],
                lag_weight_seed: &[],
                dcs: None,
            },
            comm,
            records: &records,
            iteration: 3,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            event_sender: None,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        },
        configure,
    )
    .unwrap();

    BindingFixtureRun {
        result,
        fcf,
        workspaces,
    }
}

/// Cuts generated at iteration 1 and found binding at iteration 3 carry
/// `last_active_iter == 3`, a value only the metadata sync can have written
/// (`add_cut` stamps the generating iteration), and the sync accumulates
/// `active_count`.
#[test]
fn metadata_sync_advances_last_active_iter_past_the_generating_iteration() {
    use novomodelo_comm::LocalBackend;

    let BindingFixtureRun { result, fcf, .. } =
        backward_iteration_3_over_three_old_cuts(&LocalBackend, |_| {});

    // 3 stages × (n_stages-1=2 non-terminal) × 3 trial points = 6 cuts.
    assert_eq!(result.cuts_generated, 6);

    // Pool[1] holds the 3 cuts generated at iteration 1 and the 3 generated by
    // this pass at iteration 3.
    // Slot formula: warm_start(0) + iteration * fwd_passes(3) + fpi
    // → slots 3, 4, 5 and 9, 10, 11. Populated count = 12 (high-water mark).
    assert_eq!(fcf.pools[1].populated(), 12);

    // At t=0, the 3 cuts in pool[1] generated at iteration 1 (slots 3,4,5)
    // were evaluated for binding. The mock solver returns positive duals
    // (cut_dual_padding = 1.0) for all cut rows. Each trial point has
    // n_openings=2 openings, and the binding check runs per opening. So each
    // slot gets 3 trial points × 2 openings = 6 increments.
    for slot in 3..6 {
        assert!(
            fcf.pools[1].metadata(slot).active_count > 0,
            "slot {slot} active_count should be > 0 (cuts were binding)"
        );
        assert_eq!(
            fcf.pools[1].metadata(slot).active_count,
            6,
            "slot {slot} active_count should be 6 (3 trial points × 2 openings)"
        );
        assert_eq!(
            fcf.pools[1].metadata(slot).last_active_iter,
            3,
            "slot {slot} was generated at iteration 1 and found binding at iteration 3"
        );
    }

    // Pool[2] (terminal successor) received no cuts and no binding
    // was checked against it — metadata should be at defaults.
    assert_eq!(fcf.pools[2].populated(), 0);
}

/// Single-rank communicator counting the `u64` `Sum` reductions; the binding
/// metadata sync issues the only ones of a backward pass (the worker-count
/// check reduces `u64` with `Min`/`Max`, every other `Sum` reduces `f64`).
#[derive(Default)]
struct MetadataReductionCountingComm {
    metadata_reductions: AtomicUsize,
}

impl Communicator for MetadataReductionCountingComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        if matches!(op, ReduceOp::Sum) && TypeId::of::<T>() == TypeId::of::<u64>() {
            self.metadata_reductions.fetch_add(1, Ordering::Relaxed);
        }
        recv[..send.len()].copy_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        unreachable!("the backward pass issues no broadcast")
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// With binding tracking off the backward pass issues no metadata reduction,
/// writes no binding metadata, copies no cut-row dual and never touches the
/// workers' binding buffers, and it generates the same cuts as with tracking on;
/// under either scheduler.
#[test]
fn binding_is_recorded_only_when_tracking_is_on() {
    let cut_bits = |fcf: &FutureCostFunction| -> Vec<(Vec<u64>, Vec<u64>)> {
        fcf.pools
            .iter()
            .map(|pool| {
                (
                    pool.intercepts_prefix()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                    pool.coefficients_prefix()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                )
            })
            .collect()
    };
    let binding_buffers = |workspaces: &[SolverWorkspace<MockSolver>]| -> Vec<u64> {
        workspaces
            .iter()
            .flat_map(|ws| {
                ws.backward_accum
                    .slot_increments
                    .iter()
                    .chain(&ws.backward_accum.metadata_sync_contribution)
                    .copied()
            })
            .collect()
    };

    for scheduler in [
        BackwardScheduler::ByScenario {},
        BackwardScheduler::ByNode { block_size: None },
    ] {
        let comm_on = MetadataReductionCountingComm::default();
        let on = backward_iteration_3_over_three_old_cuts(&comm_on, |state| {
            state.set_scheduler(scheduler);
        });
        assert!(
            comm_on.metadata_reductions.load(Ordering::Relaxed) >= 1,
            "{scheduler:?}"
        );
        for slot in 3..6 {
            let metadata = on.fcf.pools[1].metadata(slot);
            assert!(metadata.active_count > 0, "{scheduler:?} slot {slot}");
            assert_eq!(metadata.last_active_iter, 3, "{scheduler:?} slot {slot}");
        }
        assert!(
            binding_buffers(&on.workspaces).iter().any(|&v| v > 0),
            "{scheduler:?}: the fixture must fill a binding buffer for the off leg to have power"
        );
        assert!(
            on.workspaces
                .iter()
                .any(|ws| !ws.backward_accum.cut_dual_slots.is_empty()),
            "{scheduler:?}: the fixture must copy cut-row duals for the off leg to have power"
        );

        let comm_off = MetadataReductionCountingComm::default();
        let off = backward_iteration_3_over_three_old_cuts(&comm_off, |state| {
            state.set_scheduler(scheduler);
            state.set_binding_tracking(false);
        });
        assert_eq!(
            comm_off.metadata_reductions.load(Ordering::Relaxed),
            0,
            "{scheduler:?}"
        );
        for (pool_id, pool) in off.fcf.pools.iter().enumerate() {
            for slot in 0..pool.populated() {
                let metadata = pool.metadata(slot);
                assert_eq!(
                    metadata.active_count, 0,
                    "{scheduler:?} pool {pool_id} slot {slot}"
                );
                assert_eq!(
                    metadata.last_active_iter, metadata.iteration_generated,
                    "{scheduler:?} pool {pool_id} slot {slot}"
                );
            }
        }
        // The fixture's buffers start empty, so an empty buffer is an untouched one.
        assert!(
            binding_buffers(&off.workspaces).is_empty(),
            "{scheduler:?}: a worker grew, scanned or folded a binding buffer with tracking off"
        );
        assert!(
            off.workspaces
                .iter()
                .all(|ws| ws.backward_accum.cut_dual_slots.is_empty()),
            "{scheduler:?}: cut-row duals were copied with tracking off"
        );

        assert_eq!(on.result.cuts_generated, off.result.cuts_generated);
        assert_eq!(cut_bits(&on.fcf), cut_bits(&off.fcf), "{scheduler:?}");
    }
}

/// One backward pass over a 3-stage chain with a single trial point and a single
/// opening: stage 1 stores one cut, stage 0 solves against it once, and the mock
/// returns `cut_row_dual` as that cut row's dual. Returns the stored cut's
/// `active_count` after the metadata sync.
///
/// The mock supplies the dual because a real optimum never makes it negative.
fn active_count_after_one_binding_solve(cut_row_dual: f64, tolerance: f64) -> u64 {
    use novomodelo_comm::LocalBackend;

    let n_stages = 3_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let n_state = state.n_state;
    let mut fcf = FutureCostFunction::new(n_stages, n_state, 1, 20, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut solver = MockSolver::always_ok(solution_1_0(100.0, -5.0));
    solver.cut_dual_padding = cut_row_dual;
    let comm = LocalBackend;
    let mut workspaces = single_workspace(solver, n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::new(n_state, 1, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: tolerance,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    assert_eq!(
        result.cuts_generated, 2,
        "1 trial point × 2 non-terminal stages"
    );
    assert_eq!(
        fcf.pools[1].populated(),
        1,
        "pool 1 must hold exactly the one cut stage 0 solves against, in slot 0"
    );
    fcf.pools[1].metadata(0).active_count
}

#[test]
fn a_negative_cut_row_dual_is_not_binding() {
    assert_eq!(active_count_after_one_binding_solve(-0.5, 0.0), 0);
}

#[test]
fn a_cut_row_dual_at_or_below_the_tolerance_is_not_binding() {
    let tolerance = 0.5;
    assert_eq!(
        active_count_after_one_binding_solve(0.25, tolerance),
        0,
        "a dual below the tolerance must not bind"
    );
    assert_eq!(
        active_count_after_one_binding_solve(0.5, tolerance),
        0,
        "a dual equal to the tolerance must not bind"
    );
    assert_eq!(
        active_count_after_one_binding_solve(0.75, tolerance),
        1,
        "a dual above the tolerance must bind once"
    );
}

/// Build N identical `SolverWorkspace<MockSolver>` instances and run a
/// 2-stage backward pass with 6 trial points. Returns the resulting FCF.
///
/// Used by `work_stealing_produces_identical_results_across_worker_counts`
/// to compare FCF state across different worker counts.
///
/// The `MockSolver` returns objective=100.0 and dual[0]=-5.0 for every solve,
/// which is deterministic (no dependence on call order or worker identity).
/// Each trial point i gets state [(i + 1) as f64 * 10.0] so that distinct
/// cuts are generated and the ordering invariant is meaningful.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn run_backward_pass_with_n_workers(n_workers: usize) -> FutureCostFunction {
    let n_stages = 2_usize;
    let local_work = 6_usize;
    let n_openings = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let n_state = state.n_state; // 1

    // Use forward_passes = local_work so the FCF pool is large enough for
    // all trial points in a single iteration (iteration 0, slots 0..5).
    #[allow(clippy::cast_possible_truncation)]
    let forward_passes = local_work as u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 64, &vec![0; n_stages]);

    // Build `local_work` trial points with distinct states so each cut
    // has a different intercept. State for trial point i = (i+1)*10.0.
    let states: Vec<Vec<f64>> = (0..local_work)
        .map(|i| vec![(i + 1) as f64 * 10.0])
        .collect();
    let (mut exchange, records) = exchange_and_records(n_state, &states, n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    // Each workspace gets the same deterministic solution.
    // MockSolver::always_ok returns objective=100.0, dual[0]=-5.0 for
    // every call regardless of call order or worker identity.
    let solution = solution_1_0(100.0, -5.0);
    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("worker_id fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::for_test(1),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let comm = StubComm;
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .unwrap();

    // Confirm all 6 trial points produced cuts at stage 0.
    assert_eq!(
        result.cuts_generated, local_work,
        "n_workers={n_workers}: expected {local_work} cuts, got {}",
        result.cuts_generated,
    );

    fcf
}

#[test]
fn work_stealing_produces_identical_results_across_worker_counts() {
    // Acceptance criterion: the FCF state after running the backward pass
    // with 1 workspace must be bit-identical to the state after running
    // with 3 workspaces, given the same inputs. This verifies that the
    // sort-by-trial_state_idx post-processing in the work-stealing
    // implementation produces a deterministic FCF regardless of which
    // worker claims which trial point.
    let fcf_1 = run_backward_pass_with_n_workers(1);
    let fcf_3 = run_backward_pass_with_n_workers(3);

    let num_stages = 2;

    // Verify that both runs produced cuts (belt-and-suspenders guard so
    // that an empty FCF cannot cause a false positive).
    assert!(
        fcf_1.active_cuts(0).count() > 0,
        "1-worker run produced no cuts at stage 0"
    );

    for stage in 0..num_stages {
        let cuts_1: Vec<_> = fcf_1.active_cuts(stage).collect();
        let cuts_3: Vec<_> = fcf_3.active_cuts(stage).collect();
        assert_eq!(
            cuts_1.len(),
            cuts_3.len(),
            "stage {stage}: cut count mismatch ({} vs {})",
            cuts_1.len(),
            cuts_3.len(),
        );
        for (i, ((s1, int1, c1), (s3, int3, c3))) in cuts_1.iter().zip(&cuts_3).enumerate() {
            assert_eq!(
                s1, s3,
                "stage {stage}, cut {i}: slot mismatch ({s1} vs {s3})"
            );
            assert_eq!(
                int1, int3,
                "stage {stage}, cut {i}: intercept mismatch ({int1} vs {int3})"
            );
            assert_eq!(
                c1, c3,
                "stage {stage}, cut {i}: coefficients mismatch ({c1:?} vs {c3:?})"
            );
        }
    }
}

// ── Parallel overhead decomposition unit tests ────────────────────────────

/// Build a `SolverStatistics` snapshot with the given cumulative times (in seconds).
fn make_stats(solve_s: f64, load_s: f64, set_bounds_s: f64, basis_set_s: f64) -> SolverStatistics {
    SolverStatistics {
        total_solve_time_seconds: solve_s,
        total_load_model_time_seconds: load_s,
        total_set_bounds_time_seconds: set_bounds_s,
        total_basis_set_time_seconds: basis_set_s,
        ..SolverStatistics::default()
    }
}

/// Decompose parallel overhead into (`setup_ms`, `imbalance_ms`, `scheduling_ms`)
/// from per-worker before/after snapshots.
fn decompose_overhead(
    pairs: &[(SolverStatistics, SolverStatistics)],
    parallel_wall_ms: u64,
) -> (u64, u64, u64) {
    use crate::solver_stats::SolverStatsDelta;

    #[allow(clippy::cast_precision_loss)]
    let n_workers = pairs.len() as f64;

    let worker_deltas: Vec<SolverStatsDelta> = pairs
        .iter()
        .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after))
        .collect();

    let stage_setup_ms: f64 = worker_deltas
        .iter()
        .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
        .sum();

    let worker_totals: Vec<f64> = worker_deltas
        .iter()
        .map(|d| {
            d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        })
        .collect();

    let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
    let avg_worker_ms = if worker_totals.is_empty() {
        0.0_f64
    } else {
        worker_totals.iter().sum::<f64>() / n_workers
    };

    let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
    #[allow(clippy::cast_precision_loss)]
    let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (
        stage_setup_ms as u64,
        stage_imbalance_ms as u64,
        stage_scheduling_ms as u64,
    )
}

/// 4 workers with different solve times: imbalance equals
/// `trunc(max - mean_f64)` of worker totals, scheduling equals
/// `parallel_wall - max`.
///
/// Worker solve times: 100 ms, 200 ms, 150 ms, 180 ms.
/// Setup per worker: 0 (this sub-test isolates solve imbalance).
/// Mean of totals (f64) = 630.0 / 4 = 157.5.
/// Imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42.
/// Scheduling = 250 - 200 = 50.
#[test]
fn decompose_four_workers_different_solve_times() {
    let zero = SolverStatistics::default();
    let pairs = vec![
        (zero.clone(), make_stats(0.1, 0.0, 0.0, 0.0)), // 100 ms solve
        (zero.clone(), make_stats(0.2, 0.0, 0.0, 0.0)), // 200 ms solve
        (zero.clone(), make_stats(0.15, 0.0, 0.0, 0.0)), // 150 ms solve
        (zero.clone(), make_stats(0.18, 0.0, 0.0, 0.0)), // 180 ms solve
    ];
    let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 250);

    assert_eq!(setup_ms, 0, "no setup work expected");
    assert_eq!(
        imbalance_ms, 42,
        "imbalance = trunc(max(200.0) - avg(157.5)) = trunc(42.5) = 42"
    );
    assert_eq!(scheduling_ms, 50, "scheduling overhead = wall - max_worker");
}

/// Acceptance criterion: `setup_time_ms` is the sum of all workers' non-solve
/// work.
///
/// Workers have setup costs (load+add+bounds+basis): 20, 25, 15, 22 ms.
/// Expected `setup_ms` = 20 + 25 + 15 + 22 = 82.
#[test]
fn decompose_setup_time_is_aggregate_non_solve_work() {
    let zero = SolverStatistics::default();
    // Each worker: 0 solve + known setup split across the three sub-timers.
    // Worker setup totals: 20, 25, 15, 22 ms (put entirely in load_model timer).
    let pairs = vec![
        (zero.clone(), make_stats(0.0, 0.020, 0.0, 0.0)), // 20 ms total setup
        (zero.clone(), make_stats(0.0, 0.025, 0.0, 0.0)), // 25 ms
        (zero.clone(), make_stats(0.0, 0.015, 0.0, 0.0)), // 15 ms
        (zero.clone(), make_stats(0.0, 0.022, 0.0, 0.0)), // 22 ms
    ];
    let (setup_ms, _imbalance_ms, _scheduling_ms) = decompose_overhead(&pairs, 300);
    assert_eq!(
        setup_ms, 82,
        "aggregate setup must sum all workers' non-solve work"
    );
}

/// Edge case: all workers have identical timing → imbalance must be 0.
#[test]
fn decompose_identical_workers_zero_imbalance() {
    let zero = SolverStatistics::default();
    let after = make_stats(0.1, 0.01, 0.002, 0.001);
    let pairs = vec![
        (zero.clone(), after.clone()),
        (zero.clone(), after.clone()),
        (zero.clone(), after.clone()),
    ];
    let (_, imbalance_ms, _) = decompose_overhead(&pairs, 200);
    assert_eq!(
        imbalance_ms, 0,
        "identical workers must have zero imbalance"
    );
}

/// Edge case: single worker → imbalance is 0, setup equals that worker's
/// setup, scheduling is the residual.
#[test]
fn decompose_single_worker() {
    let zero = SolverStatistics::default();
    // 100 ms solve + 20 ms setup = 120 ms worker total.
    let after = make_stats(0.1, 0.020, 0.0, 0.0);
    let pairs = vec![(zero.clone(), after)];
    let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 150);

    assert_eq!(setup_ms, 20, "single worker: setup = 20 ms");
    assert_eq!(imbalance_ms, 0, "single worker: imbalance must be 0");
    // scheduling = 150 - 120 = 30
    assert_eq!(
        scheduling_ms, 30,
        "single worker: scheduling = wall - worker_total"
    );
}

/// Edge case: `scheduling_overhead_ms` is clamped to 0 when `max_worker_total`
/// exceeds `parallel_wall_ms` (clock skew or measurement granularity).
#[test]
fn decompose_scheduling_clamped_when_worker_exceeds_wall() {
    let zero = SolverStatistics::default();
    // Worker total = 200 ms, but wall = 180 ms → scheduling would be negative.
    let after = make_stats(0.2, 0.0, 0.0, 0.0);
    let pairs = vec![(zero.clone(), after)];
    let (_, _, scheduling_ms) = decompose_overhead(&pairs, 180);
    assert_eq!(scheduling_ms, 0, "negative scheduling must be clamped to 0");
}

// ── allgatherv per-worker stats unit tests ────────────────────────────────

/// Single-rank (np=1) backward pass with 2 workers.
///
/// Constructs a 2-worker `StageWorkerStatsBuffer::new(2, 4)` and uses
/// `StubComm` (which echoes send→recv, simulating `LocalBackend` np=1).
/// After one backward iteration, the recorded stage statistics must
/// contain 2 entries per non-zero opening (`worker_id` 0 and `worker_id` 1),
/// both with `rank = 0`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn allgatherv_single_rank_two_workers_stage_stats_has_per_worker_entries() {
    let n_stages = 2_usize;
    let n_openings = 4_usize;
    let n_workers = 2_usize;
    let local_work = 4_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let n_state = state.n_state;

    let solution = solution_1_0(100.0, -5.0);
    let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
    let (mut exchange, records) = exchange_and_records(n_state, &states, n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::for_test(1),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let stage_stats = run_backward_pass_stage_stats(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &StubComm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .expect("single-rank 2-worker backward must not error");

    // The 2-stage system has 1 backward stage (t=0, successor=1).
    // stage_stats must contain exactly 1 entry (one successor).
    assert_eq!(
        stage_stats.len(),
        1,
        "expected 1 backward stage entry (successor=1)"
    );
    let (successor, entries) = &stage_stats[0];
    assert_eq!(*successor, 1_usize, "successor index must be 1");

    // Every entry must have rank=0 (np=1 StubComm).
    for (rank, _wid, _omega, _delta) in entries {
        assert_eq!(*rank, 0_i32, "all entries must have rank=0 for np=1");
    }
    // Both worker_id values (0 and 1) must appear at omega=0.
    let omega0_wids: Vec<i32> = entries
        .iter()
        .filter(|(_, _, omega, _)| *omega == 0)
        .map(|(_, wid, _, _)| *wid)
        .collect();
    assert!(
        omega0_wids.contains(&0),
        "worker_id=0 must appear at omega=0"
    );
    assert!(
        omega0_wids.contains(&1),
        "worker_id=1 must appear at omega=0"
    );
}

/// Multi-rank (np=2) backward pass with stub communicator.
///
/// Uses a `DualRankStubComm` whose `size()` returns 2 and whose
/// `allgatherv` concatenates a manually injected "remote rank" payload
/// (a copy of the send buffer). Asserts that the unpacked
/// `stage_stats` contains entries for both `rank=0` and `rank=1`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn allgatherv_dual_rank_stub_stage_stats_contains_both_ranks() {
    /// Stub communicator simulating np=2: `allgatherv` fills recv with
    /// `[send, send]` (rank-0 and a synthetic rank-1 copy).
    struct DualRankStubComm;

    impl Communicator for DualRankStubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            // Fill each rank's slot in recv using the provided counts/displs.
            // Both ranks contribute `send` (rank-1 is a synthetic copy of rank-0).
            for (&count, &displ) in counts.iter().zip(displs) {
                let src = &send[..count.min(send.len())];
                recv[displ..displ + src.len()].copy_from_slice(src);
            }
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    let n_stages = 2_usize;
    let n_openings = 2_usize;
    let n_workers = 1_usize;
    let local_work = 2_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0(); n_stages];
    let n_state = state.n_state;

    let solution = solution_1_0(100.0, -5.0);
    let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
    let (mut exchange, records) = exchange_and_records(n_state, &states, n_stages);

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits in i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::for_test(1),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
    let mut csb = CutSyncBuffers::new(n_state, local_work, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let stage_stats = run_backward_pass_stage_stats(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(templates.len()),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &DualRankStubComm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    })
    .expect("dual-rank stub backward must not error");

    // With np=2, stage_stats for successor=1 must contain entries from
    // both rank=0 and rank=1 (DualRankStubComm copies the rank-0 block
    // into the rank-1 slot, so both appear in the unpacked output).
    assert_eq!(stage_stats.len(), 1);
    let (_, entries) = &stage_stats[0];

    let ranks_seen: Vec<i32> = entries
        .iter()
        .map(|(rank, _, _, _)| *rank)
        .collect::<std::collections::HashSet<i32>>()
        .into_iter()
        .collect();
    assert!(
        ranks_seen.contains(&0),
        "rank=0 must appear in stage_stats; got {ranks_seen:?}"
    );
    assert!(
        ranks_seen.contains(&1),
        "rank=1 must appear in stage_stats; got {ranks_seen:?}"
    );
}

// ── read-site prefer-with-fallback unit tests ────────────────

/// Run `process_by_scenario_backward` for stage 0 → successor 1 with
/// explicitly-provided backward and forward basis stores.
///
/// `basis_store` is taken by `&mut` so a `BasisStoreSliceMut` can be
/// derived from it and passed to `process_by_scenario_backward`.
///
/// Returns the mutated workspace so the caller can inspect
/// `ws.solver.warm_start_calls`.
#[allow(clippy::too_many_lines)]
fn run_one_trial_state_with_stores(
    basis_store: &mut BasisStore,
) -> Result<Vec<SolverWorkspace<MockSolver>>, crate::SddpError> {
    let n_stages = 2_usize;
    let n_openings = 1_usize;
    let n_state = 1_usize;
    let stochastic = make_stochastic_context(n_stages, n_openings);
    let state = test_support::state_layout(n_state, 0);

    let solver = MockSolver::always_ok(solution_1_0(100.0, -5.0));
    let mut workspaces = single_workspace(solver, n_state);
    let ws = &mut workspaces[0];
    ws.backward_accum
        .per_opening_stats
        .push(SolverStatsDelta::default());
    ws.backward_accum.agg_coefficients.resize(n_state, 0.0);

    let exchange = exchange_with_states(n_state, vec![vec![5.0]]);

    let templates = vec![minimal_template_1_0(), minimal_template_1_0()];
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let geometry = equipment_free_geometry(&vec![0; templates.len()]);
    let fixture =
        StageContextFixture::new(&templates, &state_boxes, &geometry).without_inflow_entries(1);
    let ctx = fixture.ctx();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let study_dims = test_support::study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };

    let iteration: u64 = 1;
    let fwd_offset: usize = 0;
    let succ_probabilities = vec![1.0_f64; n_openings];
    let successor_active_slots: Vec<usize> = vec![];
    let frozen_template = minimal_template_1_0();

    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0u32; n_stages]);
    let empty_cut_batch = empty_row_batch();

    let cut_state_projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let template_num_rows = frozen_template.num_rows;
    let succ_spec = super::SuccessorSpec {
        successor: StageIdx(1),
        iteration,
        my_rank: 0,
        probabilities: &succ_probabilities,
        cut_state: &cut_state_projection,
        slab: super::SlabLookup::Unavailable,
        stage_noise: None,
    };
    let single = SingleSuccessor::new(
        NodePos(1),
        training_ctx.node_graph.node_ids[NodePos(1)],
        1,
        0,
        fcf.pools[1].populated(),
        successor_active_slots.clone(),
        training_ctx.node_graph.nodes[NodePos(1)].openings,
        n_openings,
        frozen_template,
        empty_cut_batch,
        chain_rule_slots_of(&training_ctx, 1, NodePos(0)),
    );
    let outcomes = single.outcomes(&fcf.pools, template_num_rows, Some(0.0));
    let child0 = outcomes.child(0);

    // Derive a single-worker BasisStoreSliceMut covering all scenarios.
    let mut basis_slices = Vec::new();
    basis_store.split_workers_into(basis_store.num_scenarios(), 1, &mut basis_slices);
    let mut basis_slice = basis_slices.remove(0);

    let ws = &mut workspaces[0];
    super::load_backward_lp(ws, &child0);
    ws.backward_accum.reset_per_opening_stats(n_openings);
    ws.backward_accum.slot_increments.resize(1, 0);
    ws.backward_accum.slot_increments[..1].fill(0);
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0_f64);
    }

    let params = super::BackwardNodeParams {
        trial_points: &[0],
        exchange: &exchange,
        fwd_offset,
        node_visit_offset: fwd_offset,
        risk_measure: &risk_measures[0],
        succ: &succ_spec,
        outcomes: &outcomes,
    };
    super::process_by_scenario_backward(
        ws,
        &ctx,
        &training_ctx,
        &params,
        &mut basis_slice,
        &StageOpeningSolver::Frozen,
        0,
        0,
        0,
    )?;
    Ok(workspaces)
}

/// Regression: the backward trial-point path (`StageSolvePrep::pin_and_bracket`) inherits
/// the `PatchBuffer` single-owner fix — every travel-time bucket incoming column
/// is pinned to `x_hat`, a value constant across the opening loop (the
/// decision-driven trial point), not re-derived per opening (contrast NCS
/// availability, which genuinely varies per opening).
#[test]
fn pin_folded_pins_transit_bucket_incoming_columns_per_stage_visit() {
    let state = test_support::state_layout_with_transit_buckets(
        0,
        0,
        vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
        vec![],
    );
    assert_eq!(state.n_state, 2);

    let stochastic = test_support::hydro_free_stochastic_context(1, 1);
    let template = test_support::transit_bucket_only_template(state.theta + 1);

    let templates = vec![template];
    let state_boxes = permissive_state_boxes(state.n_state, 1);
    let geometry = equipment_free_geometry(&[0]);
    let fixture =
        StageContextFixture::new(&templates, &state_boxes, &geometry).without_inflow_entries(0);
    let ctx = fixture.ctx();

    let horizon = HorizonMode::Finite { num_stages: 1 };
    let study_dims = test_support::study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, 1),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };

    let x_hat = vec![7.0_f64, 11.0];
    let mut ws = transit_bucket_only_workspace(MockSolver::always_ok(solution_1_0(0.0, 0.0)), 2);
    let pool = CutPool::new(1, state.n_state, 1, 0);
    let projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );

    StageSolvePrep::pin_and_bracket(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        &ctx,
        &training_ctx,
        StageIdx(0),
        StateSource(&x_hat),
        &LoadedCutRows {
            pool: &pool,
            projection: &projection,
            slots: &[],
            first_row: 0,
        },
        BracketSource::Computed,
    );

    let cp = ws.patch_buf.state_col_patch_count();
    assert_eq!(
        cp, 2,
        "state_col_patch_count must equal n_buckets when N=0, A=0"
    );
    for (i, &expected) in x_hat.iter().enumerate() {
        let col = state.transit_buckets_in.start + i;
        let pos = ws.patch_buf.col_indices[..cp]
            .iter()
            .position(|&c| c == col)
            .unwrap_or_else(|| panic!("bucket incoming column {col} must be pinned"));
        assert_eq!(ws.patch_buf.col_lower[pos], expected);
        assert_eq!(ws.patch_buf.col_upper[pos], expected);
    }
}

/// Per-child isolation on a genuine fan: root → two children at stage 1 with
/// DISTINCT External inflow columns AND DISTINCT non-leaf pools. Asserts none of
/// child 0's data is reused for child 1: (a) each child reads its OWN declared
/// External column (distinct patched inflow noise), (b) each warm-starts from its
/// OWN `(m, child node)` basis key (distinct basis-store slots captured), (c) each
/// child's binding activity lands in ITS OWN pool's metadata region.
#[test]
#[allow(clippy::too_many_lines)]
fn per_child_backward_isolates_column_basis_and_pool_metadata() {
    use crate::setup::node_graph::{NodeOpenings, OpeningSource};

    let n_stages = 2_usize;
    let n_pools = 3;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let n_state = state.n_state;

    // Two external inflow columns at stage 1 carrying materially different eta.
    let mut inflow_lib =
        novomodelo_stochastic::ExternalScenarioLibrary::new(n_stages, 2, 1, "inflow", vec![1, 2]);
    inflow_lib.eta_slice_mut(1, 0).copy_from_slice(&[2.0]);
    inflow_lib.eta_slice_mut(1, 1).copy_from_slice(&[-2.0]);

    let templates = vec![minimal_template_1_0(); n_stages];
    let state_boxes = permissive_state_boxes(n_state, n_stages);
    let geometry = equipment_free_geometry(&vec![0; templates.len()]);
    let inflow_coupling = vec![test_support::inflow_coupling_with_entries(1, &[(0, 0)]); n_stages];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
        .inflow_coupling(&inflow_coupling);
    let ctx = fixture.ctx();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let study_dims = test_support::study_dims();
    let node_graph = crate::test_support::chain_node_graph(&stochastic);
    let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_pools);
    let training_ctx = TrainingContext {
        node_graph: &node_graph,
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &cut_state_layouts,
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
        inflow_scheme: SamplingScheme::External,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: Some(&inflow_lib),
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };

    // Two distinct non-leaf pools (1 and 2), each carrying active cuts so binding
    // metadata is non-trivial; pool 0 is the generating (root) pool. The populated
    // counts and active slots are read back from the FCF rather than assumed.
    let mut fcf = FutureCostFunction::new(n_pools, n_state, 1, 10, &vec![0u32; n_pools]);
    fcf.add_cut(NodeId(1), 1, 1, 0, 0.0, &vec![0.0; n_state]);
    fcf.add_cut(NodeId(2), 2, 1, 0, 0.0, &vec![0.0; n_state]);
    let base_num_rows = minimal_template_1_0().num_rows;
    let pool_child = |pool: usize, node: usize, node_id: NodeId, offset: usize| {
        let active: Vec<usize> = fcf.active_cuts(pool).map(|(s, _, _)| s).collect();
        ChildSpec {
            successor_node: NodePos(node),
            successor_node_id: node_id,
            pool_id: pool,
            openings: NodeOpenings {
                source: OpeningSource::External,
                offset,
                len: 1,
                q: 1.0,
            },
            num_cuts_at_successor: active.len(),
            populated_count: fcf.pools[pool].populated(),
            active_slots: active,
            chain_rule_slots: chain_rule_slots_of(&training_ctx, pool, NodePos(0)),
        }
    };
    let child_1 = pool_child(1, 1, NodeId(101), 0);
    let child_2 = pool_child(2, 2, NodeId(102), 1);
    let (pop1, pop2) = (child_1.populated_count, child_2.populated_count);
    assert!(
        pop1 >= 1 && pop2 >= 1,
        "both fan pools must carry active cuts"
    );
    let num_cuts_1 = child_1.num_cuts_at_successor;
    let num_cuts_2 = child_2.num_cuts_at_successor;

    // Each pool's frozen template = base rows + that pool's own active cut rows.
    let frozen_for = |n_cuts: usize| {
        let mut t = minimal_template_1_0();
        t.num_rows = base_num_rows + n_cuts;
        t
    };
    let frozen = vec![
        minimal_template_1_0(),
        frozen_for(num_cuts_1),
        frozen_for(num_cuts_2),
    ];
    let cut_batches = empty_cut_batches(n_pools);
    let multi = MultiSuccessor::new(vec![child_1, child_2], frozen, cut_batches);
    let cut_state_projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let succ_spec = super::SuccessorSpec {
        successor: StageIdx(1),
        iteration: 1,
        my_rank: 0,
        probabilities: &[0.5_f64, 0.5],
        cut_state: &cut_state_projection,
        slab: super::SlabLookup::Unavailable,
        stage_noise: None,
    };
    let outcomes = multi.outcomes(&fcf.pools, base_num_rows, Some(0.0));
    let total_metadata = outcomes.total_metadata_len();
    assert_eq!(
        total_metadata,
        pop1 + pop2,
        "the metadata buffer concatenates each pool's own slot region"
    );

    let trial_states = vec![vec![10.0_f64]];
    let records = test_support::trial_state_records(&trial_states, n_stages);
    let mut exchange = ExchangeBuffers::new(&state, trial_states.len(), 1);
    exchange
        .exchange(&records, StageIdx(1), n_stages, &StubComm)
        .expect("exchange must succeed");

    // Basis node axis covers the child node positions (1, 2), not just n_stages.
    let mut basis_store = BasisStore::new(exchange.local_count(), 3);
    let mut basis_slices = Vec::new();
    basis_store.split_workers_into(basis_store.num_scenarios(), 1, &mut basis_slices);

    let mut workspaces = single_workspace(PerChildProbeSolver::new(1.0), n_state);
    let ws = &mut workspaces[0];
    ws.patch_buf = PatchBuffer::new(&state, &[], &[], &inflow_coupling);
    ws.backward_accum
        .per_opening_stats
        .resize_with(2, SolverStatsDelta::default);
    ws.backward_accum.slot_increments.clear();
    ws.backward_accum.slot_increments.resize(total_metadata, 0);
    ws.backward_accum
        .metadata_sync_contribution
        .resize(total_metadata, 0);
    ws.backward_accum.metadata_sync_contribution[..total_metadata].fill(0);
    ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    ws.backward_accum.agg_arena.resize(n_state, 0.0);

    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let params = super::BackwardNodeParams {
        trial_points: &[0],
        exchange: &exchange,
        fwd_offset: 0,
        node_visit_offset: 0,
        risk_measure: &risk_measures[0],
        succ: &succ_spec,
        outcomes: &outcomes,
    };
    super::process_by_scenario_backward(
        ws,
        &ctx,
        &training_ctx,
        &params,
        &mut basis_slices[0],
        &StageOpeningSolver::Frozen,
        0,
        0,
        0,
    )
    .expect("per-child backward must succeed");

    // (a) Each child read its OWN External column: the two solves patched DIFFERENT
    // inflow noise. If child 0's column were reused, both patches would be equal.
    let probe = ws.solver.inner();
    assert_eq!(
        probe.per_solve_row_lower.len(),
        2,
        "one solve per child (each External child has one opening)"
    );
    assert_ne!(
        probe.per_solve_row_lower[0], probe.per_solve_row_lower[1],
        "each child must patch its OWN External column's noise, not a reused child-0 column"
    );

    // (b) Each child warm-started from and captured its OWN (m, child node) basis.
    for node in [NodePos(1), NodePos(2)] {
        assert!(
            basis_slices[0].get(0, node).is_some(),
            "child at node {node} must capture its own (m, child node) basis"
        );
    }

    // (c) Each child's binding activity landed in ITS OWN pool's metadata region:
    // child 0 → region `[0, pop1)` (pool 1), child 1 → region `[pop1, pop1+pop2)`
    // (pool 2). Both regions record bindings, and neither is empty — a child-0
    // collapse would leave pool 2's region untouched.
    let contrib = &ws.backward_accum.metadata_sync_contribution;
    let pool1_region: u64 = contrib[0..pop1].iter().sum();
    let pool2_region: u64 = contrib[pop1..pop1 + pop2].iter().sum();
    assert_eq!(
        pool1_region, num_cuts_1 as u64,
        "pool 1's region must record child 0's binding cuts"
    );
    assert_eq!(
        pool2_region, num_cuts_2 as u64,
        "pool 2's region must record child 1's binding cuts, not collide onto pool 1"
    );
}

// ---------------------------------------------------------------------------
// resolve_backward_basis_* unit tests
// ---------------------------------------------------------------------------

#[test]
fn resolve_backward_basis_returns_some_when_slot_is_populated() {
    use crate::workspace::{BasisStore, CapturedBasis};

    let b = CapturedBasis::new(2, 2, 0, 0, NodeId(0));
    let mut store = BasisStore::new(1, 2);
    *store.get_mut(0, NodePos(1)) = Some(b);

    let mut slices = Vec::new();
    store.split_workers_into(store.num_scenarios(), 1, &mut slices);
    let slice = &slices[0];
    let basis_ref = super::resolve_backward_basis(slice, 0, NodePos(1));

    assert!(basis_ref.is_some(), "expected Some when slot has a basis");
}

#[test]
fn resolve_backward_basis_returns_none_when_slot_is_empty() {
    let mut store = BasisStore::new(1, 2);
    let mut slices = Vec::new();
    store.split_workers_into(store.num_scenarios(), 1, &mut slices);
    let slice = &slices[0];
    let basis_ref = super::resolve_backward_basis(slice, 0, NodePos(1));

    assert!(basis_ref.is_none(), "expected None for empty slot");
}

// ---------------------------------------------------------------------------
// Integration tests (backward write populates BasisStore)
// ---------------------------------------------------------------------------

#[test]
fn backward_write_populates_basis_store_at_omega_zero() {
    let mut basis_store = BasisStore::new(1, 2);
    let workspaces = run_one_trial_state_with_stores(&mut basis_store).unwrap();

    assert!(
        basis_store.get(0, NodePos(1)).is_some(),
        "BasisStore[0, 1] must be Some after backward write at omega=0"
    );
    let captured = basis_store.get(0, NodePos(1)).unwrap();
    assert_eq!(
        captured.base_row_count, 1,
        "the written basis must carry the successor template's row count"
    );
    assert_eq!(
        workspaces[0].solver.inner().call_count,
        1,
        "solver must be called exactly once for a 1-opening backward pass"
    );
}

#[test]
fn backward_write_preserves_slot_on_infeasibility_at_omega_zero() {
    use novomodelo_solver::Basis;

    use crate::workspace::{BasisStore, CapturedBasis};

    // Pre-populate slot [0, 1] with a sentinel basis. `base_row_count = 2`
    // differs from the successor template's row count, so the reuse-path
    // overwrite must replace it. The fields satisfy the `CapturedBasis`
    // invariant `row_status.len() == base_row_count + cut_row_slots.len()`.
    let pre_existing = CapturedBasis {
        basis: Basis::new(2, 2),
        base_row_count: 2,
        cut_row_slots: Vec::new(),
        // Matches the chain's node_ids[1] == 1 so the pre-existing slot is
        // still eligible as a warm-start input, not silently dropped cold by
        // the node-tag check this fixture is not exercising.
        node_id: NodeId(1),
    };
    let mut basis_store = BasisStore::new(1, 2);
    *basis_store.get_mut(0, NodePos(1)) = Some(pre_existing);

    assert_eq!(
        basis_store.get(0, NodePos(1)).unwrap().base_row_count,
        2,
        "sentinel must be in place before the infeasible solve"
    );

    // run_one_trial_state_with_stores uses MockSolver::always_ok, so we
    // exercise the reuse path (successful solve overwrites slot). For the
    // infeasibility path, the structural guarantee is: `?` in
    // process_by_scenario_backward propagates Err before the write site.
    // That path is integration-tested by `backward_pass_propagates_infeasible_error`.
    //
    // Here we test the complementary invariant: a *successful* solve at ω=0
    // with a pre-existing slot uses the reuse branch (get_basis into the
    // existing allocation) and leaves the slot Some (not None).
    let result = run_one_trial_state_with_stores(&mut basis_store);
    assert!(result.is_ok(), "expected Ok for successful solve");

    assert!(
        basis_store.get(0, NodePos(1)).is_some(),
        "BasisStore[0, 1] must not be None after successful reuse-path write at ω=0"
    );
    assert_eq!(
        basis_store.get(0, NodePos(1)).unwrap().base_row_count,
        1,
        "the reuse path must overwrite the sentinel with the successor template's row count"
    );
}

/// Handshake passes when all ranks agree on `n_workers_local`.
///
/// Uses `StubComm` (echoes send→recv, i.e. min==max==local) with a
/// 2-worker setup and a 1-stage system so no backward stages are swept.
/// The test only validates that the uniformity check does not reject a
/// consistent 2-worker configuration.
#[test]
#[allow(clippy::too_many_lines)]
fn handshake_passes_with_local_backend() {
    let n_stages = 1_usize;
    let n_workers = 2_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0()];
    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let solution = solution_1_0(100.0, -5.0);

    // Build 2 workspaces to exercise n_workers_local=2.
    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|idx| SolverWorkspace {
            rank: 0,
            worker_id: i32::try_from(idx).expect("idx fits i32"),
            solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
            patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[], &[]),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::for_test(1),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        })
        .collect();

    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::new(n_state, 1, 1);
    let comm = StubComm;
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    });

    assert!(
        result.is_ok(),
        "handshake must pass when all ranks have the same n_workers_local; got: {result:?}"
    );
}

/// Handshake rejects non-uniform `n_workers_local` across ranks.
///
/// `NonUniformStubComm` simulates a 2-rank cluster where min and max
/// worker counts differ. Its `allreduce(Min)` returns all `T::default()`
/// (zeros), while `allreduce(Max)` copies `send` to `recv` (the local
/// value). With `local_workers = 1`, `min_recv[0] = 0` and
/// `max_recv[0] = 1`, so `0 != 1` triggers the uniformity check.
/// `BackwardPassState::run` must return `SddpError::Validation` with the
/// expected substring before entering the stage loop.
#[test]
#[allow(clippy::too_many_lines)]
fn handshake_rejects_nonuniform_workers() {
    /// Stub communicator that forces `allreduce(Min)` to return zeros and
    /// `allreduce(Max)` to echo the send buffer, producing `min != max`
    /// for any non-zero local value.
    struct NonUniformStubComm;

    impl Communicator for NonUniformStubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            match op {
                // Min: return T::default() (0) to simulate a remote rank
                // with zero workers, creating a min != max discrepancy.
                ReduceOp::Min => {
                    for r in recv.iter_mut() {
                        *r = T::default();
                    }
                }
                // Max and all others: echo send so max == local value.
                _ => {
                    recv[..send.len()].copy_from_slice(send);
                }
            }
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    let n_stages = 1_usize;
    let stochastic = make_stochastic_context(n_stages, 1);
    let state = test_support::state_layout(1, 0);
    let templates = vec![minimal_template_1_0()];
    let n_state = state.n_state;
    let forward_passes = 1_u32;
    let mut fcf =
        FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
    let (mut exchange, records) = exchange_and_records(n_state, &[vec![10.0]], n_stages);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let comm = NonUniformStubComm;
    // n_workers_local = 1 on this rank; allreduce(Min) returns 0 and
    // allreduce(Max) returns 1 → 0 != 1 triggers the validation error.
    let mut workspaces =
        single_workspace(MockSolver::always_ok(solution_1_0(100.0, -5.0)), n_state);
    let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
    let mut csb = CutSyncBuffers::new(n_state, 1, 1);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let result = run_backward_pass(&mut BackwardPassInputs {
        workspaces: &mut workspaces,
        basis_store: &mut basis_store,
        ctx: &StageContextFixture::new(
            &templates,
            &state_boxes,
            &equipment_free_geometry(&vec![0; templates.len()]),
        )
        .without_inflow_entries(1)
        .ctx(),
        frozen: &mut templates.clone(),
        fcf: &mut fcf,
        cut_batches: &mut empty_cut_batches(n_stages),
        training_ctx: &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        comm: &comm,
        records: &records,
        iteration: 0,
        local_work: exchange.local_count(),
        fwd_offset: 0,
        risk_measures: &risk_measures,
        exchange: &mut exchange,
        cut_activity_tolerance: 0.0,
        cut_sync_bufs: &mut csb,
        event_sender: None,

        traversal: &Traversal::default(),
        enumerated_state: &EnumeratedForwardScratch::default(),
    });

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("non-uniform n_workers_local"),
                "error message must contain 'non-uniform n_workers_local'; got: {msg}"
            );
            assert!(
                msg.contains("min=0"),
                "error message must mention min=0 (stub Min returns T::default()); got: {msg}"
            );
            assert!(
                msg.contains("max=1"),
                "error message must mention max=1 (stub Max echoes local=1); got: {msg}"
            );
            assert!(
                msg.contains("local=1"),
                "error message must mention local=1 (single workspace); got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) from non-uniform handshake, got: {other:?}"
        ),
    }
}

/// Verify cut sign convention: an anticipated ring slot's stored coefficient
/// 7.5 is negated to -7.5 at the cut-target column. Drives the cut-row
/// builder against a finalized anticipated [`StateSpace`] (the role-(a)
/// owner of the resolver).
#[test]
fn cut_coefficient_sign_convention_slot_zero_k2() {
    let state = test_support::state_layout_full(0, 0, vec![2]);
    assert_eq!(state.commit_out.start, 0);
    assert_eq!(state.n_state, 2);

    let mut fcf = FutureCostFunction::new(3, state.n_state, 1, 10, &[0; 3]);
    let mut coefficients = vec![0.0_f64; state.n_state];
    coefficients[state.commit_out.start] = 7.5;
    fcf.add_cut(NodeId(0), 1, 0, 0, 0.0, &coefficients);

    let mut batch = empty_row_batch();
    build_cut_row_batch_into(
        &mut batch,
        &fcf,
        1,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
    );

    // Slot 0 (j = commit_out.start) resolves by identity — the
    // in-LP ring's definition row (not `state_to_lp_column`) resolves the
    // ring transition, so the cut renders directly onto the outgoing column.
    let lp_col = state
        .state_to_lp_column(StateDim::new(state.commit_out.start))
        .get();
    assert_eq!(lp_col, state.commit_out.start);

    let pos = batch
        .col_indices
        .iter()
        .position(|&c| c == lp_col as i32)
        .expect("lp_col must appear in batch.col_indices");
    assert_eq!(
        batch.values[pos], -7.5,
        "stored subgradient 7.5 must be negated to -7.5 in the cut row"
    );
}

// -----------------------------------------------------------------------
// DCS backward-integration tests
// -----------------------------------------------------------------------

use crate::cut_selection::{CutMetadata, CutSelectionStrategy};
use crate::dcs::{DcsConfig, DcsParams};
use crate::solver_phase::Phase;
use crate::workspace::{NoisePreallocation, WorkspacePool, WorkspaceSizing};
use novomodelo_solver::ActiveSolver;

/// Cut-free successor core for a 1-hydro, no-lag stage:
/// columns `[storage_out=0, storage_in=1, theta=2]`.
/// Minimize `theta`. The pin phase pins `storage_in` (col 1, the
/// incoming-state column) to `x_hat`. A single coupling row
/// `storage_out - storage_in = 0` ties the outgoing-state column (col 0,
/// which the cuts reference) to the pinned incoming state, so the cut floor
/// is evaluated at `x_hat` and the cut subgradient flows back to the pinned
/// column — the minimal structure that makes the backward dual a real
/// subgradient with respect to the incoming state.
fn dcs_core_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 2,
        // CSC by column: col0 → (row0, +1), col1 → (row0, -1); col 2 empty.
        col_starts: vec![0_i32, 1, 2, 2],
        row_indices: vec![0_i32, 0],
        values: vec![1.0, -1.0],
        col_lower: vec![0.0, 0.0, -1.0e6],
        col_upper: vec![f64::INFINITY, f64::INFINITY, 1.0e6],
        objective: vec![0.0, 0.0, 1.0],
        // Coupling equality: storage_out - storage_in = 0.
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build a two-stage `FutureCostFunction` whose successor (stage 1) pool
/// carries three cuts on the incoming-storage state (index 0):
///   slot 0: intercept 1, coeff [0]   → floor 1.0 at `x_hat` = 2
///   slot 1: intercept 0, coeff [2]   → floor 4.0  (the binding cut)
///   slot 2: intercept 3, coeff [0]   → floor 3.0
/// Metadata is set so that the metadata-seeded initial set omits the
/// binding slot 1 (its `last_active_iter` is stale) — the lazy loop must
/// add it.
fn dcs_two_stage_fcf() -> FutureCostFunction {
    let n_stages = 2;
    let mut fcf = FutureCostFunction::new(n_stages, 1, 8, 10, &vec![0; n_stages]);
    fcf.add_cut(NodeId(0), 1, 0, 0, 1.0, &[0.0]);
    fcf.add_cut(NodeId(0), 1, 0, 1, 0.0, &[2.0]);
    fcf.add_cut(NodeId(0), 1, 0, 2, 3.0, &[0.0]);
    // Seed metadata: slots 0 and 2 are "recently active"; the binding slot 1
    // is stale so the seed window excludes it (forcing the lazy add). All were
    // generated before the current iteration so none is current-iteration
    // protected.
    let meta = |generated: u64, last: u64| CutMetadata {
        iteration_generated: generated,
        forward_pass_index: 0,
        node: NodeId(0),
        active_count: 0,
        last_active_iter: last,
    };
    fcf.pools[1].set_metadata_for_test(0, meta(1, 5));
    fcf.pools[1].set_metadata_for_test(1, meta(1, 1)); // stale → outside seed_window=2 at iter 5
    fcf.pools[1].set_metadata_for_test(2, meta(1, 5));
    fcf
}

/// Single real-solver (`ActiveSolver`) workspace sized for `n_state = 1`.
fn dcs_active_workspace() -> Vec<SolverWorkspace<ActiveSolver>> {
    let sizing = WorkspaceSizing {
        max_openings: 1,
        initial_pool_capacity: 16,
        max_local_fwd: 1,
        noise: NoisePreallocation::StochasticDim,
    };
    let solver = ActiveSolver::new().expect("ActiveSolver::new()");
    let state = test_support::state_layout(1, 0);
    let stochastic = test_support::hydro_free_stochastic_context(1, 1);
    let node_graph = crate::test_support::chain_node_graph(&stochastic);
    let study_dims = test_support::study_dims();
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
    let initial_state: Vec<f64> = Vec::new();
    let training_ctx = TrainingContext {
        node_graph: &node_graph,
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &cut_state_layouts,
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };
    vec![SolverWorkspace::new(
        0,
        0,
        solver,
        PatchBuffer::new(&state, &[], &[], &[]),
        &training_ctx,
        &StageContextFixture::new(&[], &[], &[]).ctx(),
        sizing,
    )]
}

/// Run one backward trial point at the successor stage with the real solver
/// and return the produced [`StagedCut`] plus the post-call per-slot
/// `metadata_sync_contribution` snapshot. `dcs` toggles the path. The
/// incoming state is pinned to `x_hat = 2.0`.
fn run_dcs_backward_trial_state(
    dcs: Option<DcsParams>,
    iteration: u64,
) -> (super::StagedCut, Vec<f64>, Vec<u64>) {
    run_dcs_backward_trial_state_at(dcs, iteration, 2.0)
}

/// `run_dcs_backward_trial_state` with the incoming-state pin `x_hat`
/// parameterized, so a sweep can vary the pinned state (which cut binds).
///
/// Returns the produced [`StagedCut`], its coefficient slice resolved from
/// the worker arena (the bytes the FCF would receive), and the post-call
/// `metadata_sync_contribution` snapshot.
#[allow(clippy::too_many_lines)]
fn run_dcs_backward_trial_state_at(
    dcs: Option<DcsParams>,
    iteration: u64,
    x_hat: f64,
) -> (super::StagedCut, Vec<f64>, Vec<u64>) {
    let state = test_support::state_layout(1, 0);
    let n_state = state.n_state;
    let n_stages = 2;
    let core = dcs_core_template();
    let templates = vec![core.clone(), core.clone()];
    let stochastic = make_stochastic_context(2, 1);
    let horizon = HorizonMode::Finite { num_stages: 2 };
    let risk_measure = RiskMeasure::Expectation;

    let fcf = dcs_two_stage_fcf();
    // All-cuts batch for the frozen path (delta == all cuts here).
    let cut_batch = build_cut_row_batch(
        &fcf,
        1,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
    );
    let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated()).collect();
    let num_cuts = successor_active_slots.len();

    let exchange = exchange_with_states(n_state, vec![vec![x_hat]]);
    let mut workspaces = dcs_active_workspace();
    let mut basis_store = empty_basis_store(exchange.local_count(), 2);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let geometry = equipment_free_geometry(&vec![0; templates.len()]);
    let fixture =
        StageContextFixture::new(&templates, &state_boxes, &geometry).without_inflow_entries(1);
    let ctx = fixture.ctx();
    let study_dims = test_support::study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };

    let probabilities = vec![1.0_f64];
    let cut_state_projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let template_num_rows = core.num_rows;
    let succ = super::SuccessorSpec {
        successor: StageIdx(1),
        iteration,
        my_rank: 0,
        probabilities: &probabilities,
        cut_state: &cut_state_projection,
        slab: super::SlabLookup::Unavailable,
        stage_noise: None,
    };
    let single = SingleSuccessor::new(
        NodePos(1),
        training_ctx.node_graph.node_ids[NodePos(1)],
        1,
        num_cuts,
        fcf.pools[1].populated(),
        successor_active_slots.clone(),
        training_ctx.node_graph.nodes[NodePos(1)].openings,
        probabilities.len(),
        core.clone(),
        cut_batch.clone(),
        chain_rule_slots_of(&training_ctx, 1, NodePos(0)),
    );
    let outcomes = single.outcomes(&fcf.pools, template_num_rows, Some(0.0));
    let child0 = outcomes.child(0);

    let mut basis_slices = Vec::new();
    basis_store.split_workers_into(basis_store.num_scenarios(), 1, &mut basis_slices);
    let ws = &mut workspaces[0];
    // Choose the opening-solve strategy exactly as the driver does, then issue
    // the per-child prepare/load (mirrors process_by_scenario_backward's per-child
    // load after the per-child solver-state reset). The `Frozen` variant loads the
    // child's frozen all-cuts LP; the `Lazy` variant loads the cut-free core +
    // builds the metadata seed.
    let opening_solver =
        StageOpeningSolver::from_dcs_params(dcs.filter(|params| params.is_active(iteration)));
    prepare_child_lp(&opening_solver, ws, &ctx, &succ, &child0);
    // Initialise the per-opening accumulator buffers the trial-point helper
    // expects (mirrors process_stage_backward's per-stage setup).
    let n_openings = succ.probabilities.len();
    let pop = child0.populated_count;
    if ws.backward_accum.slot_increments.len() < pop {
        ws.backward_accum.slot_increments.resize(pop, 0);
    }
    ws.backward_accum.slot_increments[..pop].fill(0);
    if ws.backward_accum.agg_coefficients.len() < n_state {
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    }
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0);
    }
    if ws.backward_accum.metadata_sync_contribution.len() < pop {
        ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
    }
    ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
    ws.backward_accum.reset_per_opening_stats(n_openings);

    let params = super::BackwardNodeParams {
        trial_points: &[0],
        exchange: &exchange,
        fwd_offset: 0,
        node_visit_offset: 0,
        risk_measure: &risk_measure,
        succ: &succ,
        outcomes: &outcomes,
    };
    let cut = super::process_by_scenario_backward(
        ws,
        &ctx,
        &training_ctx,
        &params,
        &mut basis_slices[0],
        &opening_solver,
        0,
        0,
        0,
    )
    .expect("backward trial-point solve must succeed");

    // Resolve the coefficient slice from the worker arena while `ws` is
    // still in scope (the bytes the FCF merge would read).
    let coefficients = staged_cut_coefficients(&cut, &ws.backward_accum.agg_arena).to_vec();
    let meta_sync = ws.backward_accum.metadata_sync_contribution[..pop].to_vec();
    (cut, coefficients, meta_sync)
}

fn dcs_params(start_iteration: u64) -> DcsParams {
    dcs_params_with_candidate_recency(start_iteration, None)
}

/// Exactness (real solver, both backends via the active solver): the
/// DCS-built cut equals the all-cuts cut (intercept + every coefficient
/// within 1e-9). DCS seeds an initial set that omits the binding cut; the
/// lazy loop must add it and reach the same dual.
#[test]
fn backward_dcs_cut_equals_all_cuts_cut() {
    let iteration = 5;
    let (frozen_cut, frozen_coefficients, _) = run_dcs_backward_trial_state(None, iteration);
    let (dcs_cut, dcs_coefficients, _) =
        run_dcs_backward_trial_state(Some(dcs_params(2)), iteration);

    assert!(
        (frozen_cut.intercept - dcs_cut.intercept).abs() < 1e-9,
        "intercept: frozen {} vs DCS {}",
        frozen_cut.intercept,
        dcs_cut.intercept
    );
    assert_eq!(frozen_coefficients.len(), dcs_coefficients.len());
    for (i, (b, d)) in frozen_coefficients
        .iter()
        .zip(&dcs_coefficients)
        .enumerate()
    {
        assert!(
            (b - d).abs() < 1e-9,
            "coefficient[{i}]: frozen {b} vs DCS {d}"
        );
    }
    // The binding cut has gradient 2.0 on the incoming storage; both paths
    // must recover it.
    assert!(
        (frozen_coefficients[0] - 2.0).abs() < 1e-9,
        "frozen gradient must be the binding cut's 2.0, got {}",
        frozen_coefficients[0]
    );
}

/// `dcs = None` ⇒ the frozen all-cuts path is taken and the cut is identical
/// to the pre-DCS baseline (same fixture run with `None`).
#[test]
fn backward_dcs_off_is_identical_to_baseline() {
    let (cut_a, coefficients_a, _) = run_dcs_backward_trial_state(None, 5);
    let (cut_b, coefficients_b, _) = run_dcs_backward_trial_state(None, 5);
    assert_eq!(cut_a.intercept, cut_b.intercept);
    assert_eq!(coefficients_a, coefficients_b);
    // Baseline binding gradient.
    assert!((coefficients_a[0] - 2.0).abs() < 1e-9);
}

/// `dcs = Some` but `iteration < start_iteration` ⇒ the frozen path is used
/// (DCS not yet active), so the cut equals the frozen cut.
#[test]
fn backward_dcs_inactive_before_start_iteration() {
    // start_iteration = 4, iteration = 1 → inactive.
    let (frozen_cut, frozen_coefficients, frozen_meta) = run_dcs_backward_trial_state(None, 1);
    let (early_cut, early_coefficients, early_meta) =
        run_dcs_backward_trial_state(Some(dcs_params(4)), 1);
    assert_eq!(frozen_cut.intercept, early_cut.intercept);
    assert_eq!(frozen_coefficients, early_coefficients);
    // Frozen path updates binding-count metadata; the inactive-DCS run takes
    // the frozen path, so its metadata contribution matches the frozen run.
    assert_eq!(frozen_meta, early_meta);
}

/// The DCS path extracts the cut gradient from the final all-satisfied
/// LP AND maintains the binding-count metadata slot-correct under the
/// resident `CutRowMap`. For this fixture the only cut that binds at the
/// converged optimum (`x_hat` = 2, theta = 4) is the binding slot 1; slots 0
/// (floor 1) and 2 (floor 3) are resident from the seed but slack, so their
/// cut-row duals are zero. The DCS binding-count contribution must therefore
/// equal the frozen path's — slot 1 bumped, all others zero — proving the
/// slot-correct translation maps the resident binding row back to slot 1 and
/// to no other.
#[test]
fn backward_dcs_binding_counts_match_frozen() {
    let (_, _, frozen_meta) = run_dcs_backward_trial_state(None, 5);
    let (_, _, dcs_meta) = run_dcs_backward_trial_state(Some(dcs_params(2)), 5);

    // Frozen path bumps exactly the binding slot 1 (the floor-4 cut at x=2).
    assert_eq!(
        frozen_meta,
        vec![0, 1, 0],
        "frozen path must bump exactly binding slot 1, got {frozen_meta:?}"
    );
    // DCS path records the SAME binding-count contribution: the resident
    // binding row maps back to slot 1, and to no other slot.
    assert_eq!(
        dcs_meta, frozen_meta,
        "DCS binding-count metadata must match frozen (slot-correct via the \
         resident CutRowMap), got DCS {dcs_meta:?} vs frozen {frozen_meta:?}"
    );
}

/// `parse_cut_selection_config` for `method = "dynamic"` flows into
/// `DcsConfig::from_strategy`, so the backward context's `dcs` is `Some`
/// for the dynamic variant and `None` otherwise.
#[test]
fn from_strategy_gates_the_backward_dcs_field() {
    let dynamic = CutSelectionStrategy::Dynamic {
        candidate_recency: None,
        seed_window: 5,
        max_added_per_round: 10,
        violation_tolerance: None,
        start_iteration: 2,
    };
    assert!(DcsConfig::from_strategy(&dynamic).is_some());
    let level1 = CutSelectionStrategy::Level1 {
        check_frequency: 5,
        lookback_iterations: None,
    };
    assert!(DcsConfig::from_strategy(&level1).is_none());
}

// -----------------------------------------------------------------------
// DCS backward validation gates (exactness / finite candidate_recency /
// determinism / slow sweep). Default `candidate_recency = None` (∞) —
// exactness holds only when every pool cut is a candidate.
// -----------------------------------------------------------------------

/// `DcsParams` with an explicit `candidate_recency` window.
fn dcs_params_with_candidate_recency(
    start_iteration: u64,
    candidate_recency: Option<u32>,
) -> DcsParams {
    DcsParams {
        candidate_recency,
        seed_window: 2,
        max_added_per_round: 10,
        violation_tolerance: 1e-9,
        start_iteration,
        max_inner_iterations: 50,
    }
}

/// Exactness + "never spins": with `candidate_recency = None` the DCS backward cut equals
/// the all-cuts cut (intercept + every coefficient within 1e-9), and the
/// lazy loop terminates — even with `max_inner_iterations = 1`, which lifts
/// the per-round limit after one round — so it can never spin unbounded.
#[test]
fn backward_dcs_exactness_and_terminates() {
    let iteration = 5;
    let (frozen, frozen_coefficients, _) = run_dcs_backward_trial_state(None, iteration);

    // Default-cap DCS reaches the no-violation stop and matches all-cuts.
    let (dcs, dcs_coefficients, _) = run_dcs_backward_trial_state(Some(dcs_params(2)), iteration);
    assert!((frozen.intercept - dcs.intercept).abs() < 1e-9);
    for (b, d) in frozen_coefficients.iter().zip(&dcs_coefficients) {
        assert!((b - d).abs() < 1e-9, "coeff mismatch frozen {b} vs DCS {d}");
    }

    // A round cap of 1 lifts the per-round limit after one round; the call must
    // still return (no spin) and land on the exact all-cuts cut.
    let tight = DcsParams {
        max_inner_iterations: 1,
        ..dcs_params(2)
    };
    let (dcs_tc, dcs_tc_coefficients, _) = run_dcs_backward_trial_state(Some(tight), iteration);
    assert!((frozen.intercept - dcs_tc.intercept).abs() < 1e-9);
    for (b, d) in frozen_coefficients.iter().zip(&dcs_tc_coefficients) {
        assert!(
            (b - d).abs() < 1e-9,
            "round-cap-1 coeff mismatch frozen {b} vs DCS {d}"
        );
    }
}

/// A finite `candidate_recency` window demonstrably takes effect (guards
/// against `candidate_recency` being silently ignored): with
/// `candidate_recency = Some(1)` at iteration 5, the binding cut (slot 1,
/// generated at iteration 1 → age 4 ≥ 1) is windowed out of candidacy and is
/// also outside the `seed_window = 2` initial-set window, so it is never
/// added. The DCS optimum then differs from the all-cuts optimum — the
/// deliberately-non-exact windowed mode.
#[test]
fn backward_dcs_finite_candidate_recency_takes_effect() {
    let iteration = 5;
    let (frozen, frozen_coefficients, _) = run_dcs_backward_trial_state(None, iteration);
    // Sanity: the all-cuts (and candidate_recency=None DCS) gradient is the binding cut's 2.0.
    assert!((frozen_coefficients[0] - 2.0).abs() < 1e-9);

    let (windowed, windowed_coefficients, _) = run_dcs_backward_trial_state(
        Some(dcs_params_with_candidate_recency(2, Some(1))),
        iteration,
    );
    // The binding cut is windowed out, so the windowed optimum differs:
    // the surviving cuts (slots 0,2, both gradient 0) give a 0 gradient and
    // a different intercept than the all-cuts cut.
    assert!(
        (windowed_coefficients[0] - frozen_coefficients[0]).abs() > 1e-6
            || (windowed.intercept - frozen.intercept).abs() > 1e-6,
        "finite candidate_recency must change the cut vs all-cuts (windowed coeff {} intercept {}; \
         all-cuts coeff {} intercept {})",
        windowed_coefficients[0],
        windowed.intercept,
        frozen_coefficients[0],
        frozen.intercept,
    );
}

/// Determinism: running the integrated DCS backward trial point twice on
/// identical inputs yields bit-identical cuts AND bit-identical
/// binding-count metadata. A non-deterministic inner-loop insert order on
/// identical deterministic inputs would perturb the converged cut or the
/// metadata, so cut + metadata bit-identity is the determinism surface.
#[test]
fn backward_dcs_run_to_run_determinism() {
    let (cut_a, coefficients_a, meta_a) = run_dcs_backward_trial_state(Some(dcs_params(2)), 5);
    let (cut_b, coefficients_b, meta_b) = run_dcs_backward_trial_state(Some(dcs_params(2)), 5);
    assert_eq!(
        cut_a.intercept.to_bits(),
        cut_b.intercept.to_bits(),
        "intercept must be bit-identical run-to-run"
    );
    assert_eq!(coefficients_a.len(), coefficients_b.len());
    for (a, b) in coefficients_a.iter().zip(&coefficients_b) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "coefficient must be bit-identical run-to-run"
        );
    }
    assert_eq!(
        meta_a, meta_b,
        "binding-count metadata must be bit-identical run-to-run"
    );
}

/// Slow exactness sweep: across a handful of pinned incoming states (which
/// vary the binding cut), the `candidate_recency = None` DCS backward cut equals the
/// all-cuts cut within 1e-9 at every point. Gated behind `slow-tests`.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow DCS exactness sweep")]
fn backward_dcs_exactness_sweep() {
    // x_hat values span the regimes where different cuts bind:
    //   slot0 floor = 1, slot1 floor = 2*x_hat, slot2 floor = 3.
    // x_hat < 1.5 → slot2 binds (3); x_hat > 1.5 → slot1 binds (2*x_hat).
    let x_hats = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
    let iterations = [3_u64, 5, 7];
    for &iteration in &iterations {
        for &x in &x_hats {
            let (frozen, frozen_coefficients, _) =
                run_dcs_backward_trial_state_at(None, iteration, x);
            let (dcs, dcs_coefficients, _) =
                run_dcs_backward_trial_state_at(Some(dcs_params(2)), iteration, x);
            assert!(
                (frozen.intercept - dcs.intercept).abs() < 1e-9,
                "sweep iter {iteration} x_hat {x}: intercept frozen {} vs DCS {}",
                frozen.intercept,
                dcs.intercept
            );
            for (i, (b, d)) in frozen_coefficients
                .iter()
                .zip(&dcs_coefficients)
                .enumerate()
            {
                assert!(
                    (b - d).abs() < 1e-9,
                    "sweep iter {iteration} x_hat {x}: coeff[{i}] frozen {b} vs DCS {d}"
                );
            }
        }
    }
}

/// Frozen successor template for the regression fixture: the cut-free base
/// (`dcs_core_template`, the coupling row `col0 - col1 = 0`) PLUS the binding
/// cut (`-2*col0 + theta >= 0`) frozen as a second structural row. This
/// mimics freeze being active (`frozen_template.num_rows = 2 >
/// template_num_rows = 1`), so the all-cuts/frozen successor LP already
/// carries a cut row that the DCS path must NOT re-append.
///
/// CSC by column (3 cols, 2 rows):
///   col0 -> (row0, +1), (row1, -5);  col1 -> (row0, -1);  col2 -> (row1, +1)
///
/// The frozen cut intentionally DOMINATES the pool's true binding cut: it is
/// `-5*col0 + theta >= 0`, i.e. `theta >= 5*col0`, giving floor `10` at the
/// pinned `x_hat = 2` versus the pool's true optimum floor `4` (gradient 2).
/// This is a cut that is NOT in the DCS resident pool (the pool's cuts have
/// gradients 0 and 2, never 5). If the DCS path erroneously loaded this
/// frozen template as its core, the LP would carry the spurious floor-10
/// constraint and the produced cut would be `theta = 10, gradient = 5` —
/// observably different from the correct all-cuts cut. Loading the cut-free
/// base (the fix) ignores this frozen row, so the DCS cut matches all-cuts.
fn dcs_frozen_template_with_one_cut() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 4,
        col_starts: vec![0_i32, 2, 3, 4],
        row_indices: vec![0_i32, 1, 0, 1],
        values: vec![1.0, -5.0, -1.0, 1.0],
        col_lower: vec![0.0, 0.0, -1.0e6],
        col_upper: vec![f64::INFINITY, f64::INFINITY, 1.0e6],
        objective: vec![0.0, 0.0, 1.0],
        // row0: coupling equality (=0); row1: spurious frozen cut
        // -5*col0 + theta >= 0 (NOT a DCS pool cut).
        row_lower: vec![0.0, 0.0],
        row_upper: vec![0.0, f64::INFINITY],
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Regression for the frozen-template-as-core bug: when freeze is active
/// (`frozen_template.num_rows > template_num_rows`), the DCS path must load
/// the cut-free base `ctx.templates[s]` — NOT `succ.frozen_template`, which
/// already carries the active cut rows. Loading the frozen template would
/// leave its frozen cut rows resident in the LP even though the lazy loop's
/// fresh `CutRowMap` does not own them, so DCS would solve against cut rows
/// it never selected.
///
/// Here `succ.frozen_template` carries one frozen cut row
/// (`dcs_frozen_template_with_one_cut`, `num_rows = 2`) that is a spurious
/// floor-10 / gradient-5 constraint NOT present in the DCS pool, while
/// `ctx.templates[s]` is the cut-free base (`num_rows = 1`). With the fix
/// the DCS cut equals the all-cuts cut (gradient 2) within 1e-9. Against the
/// old `core = succ.frozen_template`, the spurious frozen cut dominates the
/// solve and DCS returns gradient 5 — observably wrong — failing this
/// assertion. (Verified: this test fails on the buggy code, passes on the
/// fix.)
#[test]
// Rationale: one end-to-end DCS-vs-frozen scenario whose setup (templates, fcf,
// SuccessorSpec, contexts) and sequential assertions form a single coherent
// fixture; splitting would fragment the cut-row-identity checks it verifies.
#[allow(clippy::too_many_lines)]
fn backward_dcs_frozen_cuts_present_no_duplicate_rows() {
    let iteration = 5;
    let state = test_support::state_layout(1, 0);
    let n_state = state.n_state;
    let n_stages = 2;

    // Cut-free base (loaded by the DCS path) and the frozen successor
    // template that carries the binding cut as a structural row.
    let base = dcs_core_template();
    let frozen = dcs_frozen_template_with_one_cut();
    // ctx.templates carries the cut-free base for the successor stage.
    let templates = vec![base.clone(), base.clone()];
    let stochastic = make_stochastic_context(2, 1);
    let horizon = HorizonMode::Finite { num_stages: 2 };
    let risk_measure = RiskMeasure::Expectation;

    let fcf = dcs_two_stage_fcf();
    // All-cuts batch (delta) for the frozen path; with frozen carrying the
    // binding cut, the delta is the remaining (non-frozen) cuts. For the DCS
    // exactness comparison we only need the all-cuts reference, computed
    // from the full pool against the cut-free base below.
    let cut_batch = build_cut_row_batch(
        &fcf,
        1,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
    );
    let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated()).collect();
    let num_cuts = successor_active_slots.len();

    let exchange = exchange_with_states(n_state, vec![vec![2.0]]);
    let mut workspaces = dcs_active_workspace();
    let mut basis_store = empty_basis_store(exchange.local_count(), 2);
    let state_boxes = permissive_state_boxes(n_state, n_stages);

    let geometry = equipment_free_geometry(&vec![0; templates.len()]);
    let fixture =
        StageContextFixture::new(&templates, &state_boxes, &geometry).without_inflow_entries(1);
    let ctx = fixture.ctx();
    let study_dims = test_support::study_dims();
    let training_ctx = TrainingContext {
        node_graph: &crate::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
        study_dims: &study_dims,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &[],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: Some(DcsConfig {
            candidate_recency: None,
            seed_window: 2,
            max_added_per_round: 10,
            violation_tolerance: None,
            start_iteration: 2,
        }),
    };

    let probabilities = vec![1.0_f64];
    // `frozen_template` carries a frozen cut row (num_rows = 2); the cut-free
    // base has template_num_rows = 1. This is the freeze-active shape that
    // exposed the bug.
    let cut_state_projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let template_num_rows = base.num_rows;
    let succ = super::SuccessorSpec {
        successor: StageIdx(1),
        iteration,
        my_rank: 0,
        probabilities: &probabilities,
        cut_state: &cut_state_projection,
        slab: super::SlabLookup::Unavailable,
        stage_noise: None,
    };
    let single = SingleSuccessor::new(
        NodePos(1),
        training_ctx.node_graph.node_ids[NodePos(1)],
        1,
        num_cuts,
        fcf.pools[1].populated(),
        successor_active_slots.clone(),
        training_ctx.node_graph.nodes[NodePos(1)].openings,
        probabilities.len(),
        frozen.clone(),
        cut_batch.clone(),
        chain_rule_slots_of(&training_ctx, 1, NodePos(0)),
    );
    let outcomes = single.outcomes(&fcf.pools, template_num_rows, Some(0.0));
    let child0 = outcomes.child(0);

    let mut basis_slices = Vec::new();
    basis_store.split_workers_into(basis_store.num_scenarios(), 1, &mut basis_slices);
    let ws = &mut workspaces[0];
    // Choose the opening-solve strategy as the driver does, then issue the
    // per-child prepare/load (mirrors process_by_scenario_backward's per-child
    // load). A dynamic `dcs` active from iteration 2 with `iteration = 5` selects `Lazy`,
    // whose prepare loads the cut-free core + builds the metadata seed.
    let opening_solver = StageOpeningSolver::from_dcs_params(
        training_ctx
            .dcs
            .map(|config| config.resolve(Phase::Backward.profile().primal_feasibility_tolerance))
            .filter(|params| params.is_active(iteration)),
    );
    prepare_child_lp(&opening_solver, ws, &ctx, &succ, &child0);
    let n_openings = succ.probabilities.len();
    let pop = child0.populated_count;
    if ws.backward_accum.slot_increments.len() < pop {
        ws.backward_accum.slot_increments.resize(pop, 0);
    }
    ws.backward_accum.slot_increments[..pop].fill(0);
    if ws.backward_accum.agg_coefficients.len() < n_state {
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    }
    // Size the coefficient arena for the single trial point (offset 0).
    if ws.backward_accum.agg_arena.len() < n_state {
        ws.backward_accum.agg_arena.resize(n_state, 0.0);
    }
    if ws.backward_accum.metadata_sync_contribution.len() < pop {
        ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
    }
    ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
    ws.backward_accum.reset_per_opening_stats(n_openings);

    let params = super::BackwardNodeParams {
        trial_points: &[0],
        exchange: &exchange,
        fwd_offset: 0,
        node_visit_offset: 0,
        risk_measure: &risk_measure,
        succ: &succ,
        outcomes: &outcomes,
    };
    let dcs_cut = super::process_by_scenario_backward(
        ws,
        &ctx,
        &training_ctx,
        &params,
        &mut basis_slices[0],
        &opening_solver,
        0,
        0,
        0,
    )
    .expect("DCS backward solve with frozen cuts present must succeed");
    // Resolve the DCS coefficient slice from the worker arena while `ws` is
    // still in scope.
    let dcs_coefficients = staged_cut_coefficients(&dcs_cut, &ws.backward_accum.agg_arena).to_vec();

    // The all-cuts reference cut (cut-free base + full pool, no DCS).
    let (allcuts, allcuts_coefficients, _) = run_dcs_backward_trial_state(None, iteration);

    // With the fix (core = cut-free ctx.templates[s]), the binding cut is
    // added exactly once and the DCS cut matches the all-cuts cut. With the
    // bug (core = frozen_template), the frozen cut is double-added and the
    // solve/extraction is malformed, so this fails.
    assert!(
        (dcs_cut.intercept - allcuts.intercept).abs() < 1e-9,
        "intercept: DCS {} vs all-cuts {}",
        dcs_cut.intercept,
        allcuts.intercept
    );
    assert_eq!(dcs_coefficients.len(), allcuts_coefficients.len());
    for (i, (d, a)) in dcs_coefficients
        .iter()
        .zip(&allcuts_coefficients)
        .enumerate()
    {
        assert!((d - a).abs() < 1e-9, "coeff[{i}]: DCS {d} vs all-cuts {a}");
    }
    // The binding gradient (2.0) must be recovered.
    assert!((dcs_coefficients[0] - 2.0).abs() < 1e-9);
}

const COMPACT_TEMPLATE_ROWS: usize = 1;
const COMPACT_OBJECTIVE: f64 = 50.0;
const COMPACT_STORAGE_REDUCED_COST: f64 = 6.0;
const COMPACT_BASE_ROW_DUAL: f64 = 0.5;
const COMPACT_CUT_DUALS: [f64; 2] = [2.0, 0.75];
const COMPACT_X_HAT: [f64; 3] = [42.0, 5.0, 7.0];
const COMPACT_PASS_ITERATION: u64 = 1;
const COMPACT_SUCCESSOR_CUTS: [(f64, [f64; 2]); 3] =
    [(10.0, [1.0, 0.5]), (20.0, [4.0, 3.0]), (-3.0, [2.0, 1.5])];

/// One hydro with PAR(2) inflow over two stages. Stage 0's pool spans the whole
/// state; the successor's pool spans its lag slots only and holds three cuts, the
/// middle one deactivated, so the frozen successor LP carries the rows of slots 0
/// and 2.
struct CompactBackwardCase {
    state: StateSpace,
    stochastic: StochasticContext,
    templates: Vec<StageTemplate>,
    frozen: StageTemplate,
    geometry: Vec<StageGeometry>,
    couplings: Vec<InflowCoupling>,
    study_dims: StudyDimensions,
    node_graph: NodeGraph,
    layouts: Vec<CutStateProjection>,
    pools: Vec<CutPool>,
    probabilities: Vec<f64>,
}

struct CompactRun {
    solves: Vec<RecordedSolve>,
    state_duals: Vec<f64>,
    coupled_duals: Vec<f64>,
    cut_dual_slots: Vec<(usize, f64)>,
    intercept: f64,
    coefficients: Vec<f64>,
}

struct CompactPass {
    solves: Vec<RecordedSolve>,
    intercepts: Vec<f64>,
    coefficients: Vec<f64>,
    slab_misses: u64,
    successor_active_counts: Vec<u64>,
}

struct CompactUnit<S: SolverInterface> {
    intercept: f64,
    coefficients: Vec<f64>,
    ws: SolverWorkspace<BoundRecordingSolver<S>>,
}

struct PassSpec<'a> {
    scheduler: BackwardScheduler,
    states: &'a [Vec<f64>],
    risk_measure: RiskMeasure,
    hardest_first: bool,
    bypass_solve_caches: bool,
}

/// Solves to the fixture solution scaled by a factor read from the opening's z-row
/// noise bound, so every opening has its own objective and duals whichever worker or
/// position solves it.
struct NoiseKeyedSolver {
    base: LpSolution,
    solved: LpSolution,
    num_rows: usize,
    key: f64,
}

impl NoiseKeyedSolver {
    fn new(base: LpSolution) -> Self {
        Self {
            solved: base.clone(),
            num_rows: base.dual.len(),
            base,
            key: 0.0,
        }
    }
}

impl SolverInterface for NoiseKeyedSolver {
    type Profile = novomodelo_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &novomodelo_solver::ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "NoiseKeyedSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.num_rows = template.num_rows;
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.num_rows += cuts.num_rows;
    }

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], _upper: &[f64]) {
        if let Some(i) = indices.iter().position(|&row| row == 0) {
            self.key = lower[i];
        }
    }

    fn set_row_lower_bounds_range(&mut self, _first_row: usize, _lower: &[f64]) {}

    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        _basis: Option<&Basis>,
    ) -> Result<novomodelo_solver::SolutionView<'_>, SolverError> {
        let k = 1.0 + 0.1 * self.key.sin();
        let solved = &mut self.solved;
        solved.objective = self.base.objective * k;
        solved.reduced_costs.clone_from(&self.base.reduced_costs);
        for rc in &mut solved.reduced_costs {
            *rc *= k * k;
        }
        solved.dual.clone_from(&self.base.dual);
        solved.dual.resize(self.num_rows, 0.0);
        for (row, dual) in solved.dual.iter_mut().enumerate() {
            *dual *= if row == 0 { k } else { 2.0 - k };
        }
        Ok(novomodelo_solver::SolutionView {
            objective: solved.objective,
            primal: &solved.primal,
            dual: &solved.dual,
            reduced_costs: &solved.reduced_costs,
            iterations: solved.iterations,
            solve_time_seconds: solved.solve_time_seconds,
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
        "NoiseKeyed"
    }
}

impl CompactBackwardCase {
    fn new(probabilities: &[f64]) -> Self {
        Self::with_successor_cuts(probabilities, COMPACT_SUCCESSOR_CUTS)
    }

    fn with_successor_cuts(probabilities: &[f64], cuts: [(f64, [f64; 2]); 3]) -> Self {
        let state = StateSpace::new(
            1,
            2,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[2],
        );
        let stochastic = make_stochastic_context_with_ar(2, probabilities.len(), &[0.4, 0.2]);
        let node_graph = test_support::chain_node_graph(&stochastic);
        let num_cols = state.theta + 2;
        let template = StageTemplate {
            num_rows: COMPACT_TEMPLATE_ROWS,
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            col_scale: vec![2.0; num_cols],
            row_scale: vec![4.0],
            ..test_support::transit_bucket_only_template(num_cols)
        };
        let frozen_rows = COMPACT_TEMPLATE_ROWS + 2;
        let frozen = StageTemplate {
            num_rows: frozen_rows,
            row_lower: vec![0.0; frozen_rows],
            row_upper: vec![0.0, f64::INFINITY, f64::INFINITY],
            row_scale: vec![4.0; frozen_rows],
            ..template.clone()
        };
        let mut geometry = equipment_free_geometry(&[1, 1]);
        for stage_geometry in &mut geometry {
            stage_geometry.inflow_slack = state.theta + 1..state.theta + 2;
        }
        let layouts = vec![
            CutStateProjection::new(
                &state,
                StageStateConfig {
                    storage: true,
                    inflow_lags: true,
                },
            ),
            CutStateProjection::new(
                &state,
                StageStateConfig {
                    storage: false,
                    inflow_lags: true,
                },
            ),
        ];
        assert!(layouts[0].n_slots() > layouts[1].n_slots());

        let mut successor_pool = CutPool::new(3, layouts[1].n_slots(), 1, 0);
        let successor_node_id = node_graph.node_ids[NodePos(1)];
        for (k, (intercept, row)) in (0_u64..).zip(cuts) {
            successor_pool.add_cut(successor_node_id, k, 0, intercept, &row);
        }
        successor_pool.set_active(1, false);
        let pools = vec![CutPool::new(6, layouts[0].n_slots(), 2, 0), successor_pool];

        Self {
            templates: vec![template.clone(), template],
            frozen,
            geometry,
            couplings: vec![
                test_support::inflow_coupling_with_entries(1, &[(0, 0)]),
                test_support::inflow_coupling_with_entries(1, &[(0, 0)]),
            ],
            study_dims: test_support::study_dims(),
            node_graph,
            layouts,
            pools,
            probabilities: probabilities.to_vec(),
            state,
            stochastic,
        }
    }

    fn solution(&self) -> LpSolution {
        let num_cols = self.templates[1].num_cols;
        let mut reduced_costs = vec![0.0; num_cols];
        reduced_costs[self.state.incoming_block_start(StateRegion::Storage)] =
            COMPACT_STORAGE_REDUCED_COST;
        LpSolution {
            objective: COMPACT_OBJECTIVE,
            primal: vec![0.0; num_cols],
            dual: vec![
                COMPACT_BASE_ROW_DUAL,
                COMPACT_CUT_DUALS[0],
                COMPACT_CUT_DUALS[1],
            ],
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn training_ctx(&self) -> TrainingContext<'_> {
        TrainingContext {
            node_graph: &self.node_graph,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            state: &self.state,
            cut_state_layouts: &self.layouts,
            study_dims: &self.study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &self.stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        }
    }

    fn sizing(&self, max_local_fwd: usize) -> WorkspaceSizing {
        WorkspaceSizing {
            max_openings: self.probabilities.len(),
            initial_pool_capacity: self.pools[1].populated(),
            max_local_fwd,
            noise: NoisePreallocation::StochasticDim,
        }
    }

    fn workspace_with<S: SolverInterface>(
        &self,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        max_local_fwd: usize,
        scheduler: BackwardScheduler,
        solver: S,
    ) -> SolverWorkspace<BoundRecordingSolver<S>> {
        let sizing = self.sizing(max_local_fwd);
        let mut pool = WorkspacePool {
            workspaces: vec![SolverWorkspace::new(
                0,
                0,
                BoundRecordingSolver::new(solver),
                PatchBuffer::new(&self.state, &[], &self.geometry, &self.couplings),
                training_ctx,
                ctx,
                sizing,
            )],
        };
        pool.reserve_pool_scratch(training_ctx, sizing, self.pools[1].populated());
        pool.reserve_backward_scratch(training_ctx, sizing, &scheduler);
        pool.workspaces.remove(0)
    }

    fn with_contexts<R>(&self, f: impl FnOnce(&StageContext<'_>, &TrainingContext<'_>) -> R) -> R {
        let state_boxes = permissive_state_boxes(self.state.n_state, 2);
        let fixture = StageContextFixture::new(&self.templates, &state_boxes, &self.geometry)
            .without_inflow_entries(1)
            .inflow_coupling(&self.couplings);
        let ctx = fixture.ctx();
        let training_ctx = self.training_ctx();
        f(&ctx, &training_ctx)
    }

    fn run_pass(&self, scheduler: BackwardScheduler) -> CompactPass {
        let spec = PassSpec {
            scheduler,
            states: &[COMPACT_X_HAT.to_vec()],
            risk_measure: RiskMeasure::Expectation,
            hardest_first: true,
            bypass_solve_caches: false,
        };
        self.run_pass_with(&spec, 1, || MockSolver::always_ok(self.solution()))
    }

    fn run_pass_with<S>(
        &self,
        spec: &PassSpec<'_>,
        workers: usize,
        make_solver: impl Fn() -> S,
    ) -> CompactPass
    where
        S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
    {
        self.with_contexts(|ctx, training_ctx| {
            let mut workspaces: Vec<_> = (0..workers)
                .map(|_| {
                    self.workspace_with(
                        ctx,
                        training_ctx,
                        spec.states.len(),
                        spec.scheduler,
                        make_solver(),
                    )
                })
                .collect();
            self.run_pass_on(ctx, training_ctx, spec, &mut workspaces, |_| {})
        })
    }

    fn run_pass_on<S>(
        &self,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        spec: &PassSpec<'_>,
        workspaces: &mut [SolverWorkspace<BoundRecordingSolver<S>>],
        prepare_records: impl FnOnce(&mut [TrajectoryRecord]),
    ) -> CompactPass
    where
        S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
    {
        self.run_pass_inspected(ctx, training_ctx, spec, workspaces, prepare_records, |_| {})
    }

    /// [`Self::run_pass_on`] with `inspect` applied to the backward state after its reserves
    /// and again after the pass.
    fn run_pass_inspected<S>(
        &self,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        spec: &PassSpec<'_>,
        workspaces: &mut [SolverWorkspace<BoundRecordingSolver<S>>],
        prepare_records: impl FnOnce(&mut [TrajectoryRecord]),
        inspect: impl FnMut(&BackwardPassState),
    ) -> CompactPass
    where
        S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
    {
        let n_trials = spec.states.len();
        let mut fcf = FutureCostFunction {
            pools: self.pools.clone(),
            state_dimension: self.state.n_state,
            forward_passes: u32::try_from(n_trials).expect("trial count fits u32"),
        };
        let (mut exchange, mut records) = exchange_and_records(self.state.n_state, spec.states, 2);
        prepare_records(&mut records);
        let mut frozen = vec![self.templates[0].clone(), self.frozen.clone()];
        let mut basis_store = empty_basis_store(n_trials, 2);
        let mut cut_sync_bufs = CutSyncBuffers::new(self.state.n_state, n_trials, 1);

        let result = run_backward_pass_inspected(
            &mut BackwardPassInputs {
                workspaces,
                basis_store: &mut basis_store,
                ctx,
                frozen: &mut frozen,
                fcf: &mut fcf,
                cut_batches: &mut empty_cut_batches(2),
                training_ctx,
                comm: &novomodelo_comm::LocalBackend,
                records: &records,
                iteration: COMPACT_PASS_ITERATION,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &[spec.risk_measure; 2],
                exchange: &mut exchange,
                cut_activity_tolerance: 1e-9,
                cut_sync_bufs: &mut cut_sync_bufs,
                event_sender: None,
                traversal: &Traversal::default(),
                enumerated_state: &EnumeratedForwardScratch::default(),
            },
            |state| {
                state.set_scheduler(spec.scheduler);
                state.set_hardest_first_claim_order(spec.hardest_first);
                state.set_solve_cache_bypass(spec.bypass_solve_caches);
                state.reserve_by_node_aggregate(&self.layouts, self.pools[1].populated());
                state.reserve_stage_noise(
                    self.stochastic.class_dimensions().total(),
                    self.pools[1].populated(),
                );
            },
            inspect,
        )
        .expect("the mock solve succeeds");
        assert_eq!(result.cuts_generated, n_trials);

        CompactPass {
            solves: workspaces[0].solver.inner().recorded().to_vec(),
            intercepts: fcf.pools[0].intercepts_prefix().to_vec(),
            coefficients: fcf.pools[0].coefficients_prefix().to_vec(),
            slab_misses: workspaces.iter().map(|ws| ws.scratch.slab_misses).sum(),
            successor_active_counts: (0..fcf.pools[1].populated())
                .map(|slot| fcf.pools[1].metadata(slot).active_count)
                .collect(),
        }
    }

    fn run(&self, cut_activity_tolerance: Option<f64>) -> CompactRun {
        let unit = self.run_unit(
            MockSolver::always_ok(self.solution()),
            RiskMeasure::Expectation,
            cut_activity_tolerance,
            false,
        );
        CompactRun {
            solves: unit.ws.solver.inner().recorded().to_vec(),
            state_duals: unit.ws.backward_accum.state_duals_buf.clone(),
            coupled_duals: unit.ws.backward_accum.coupled_duals_buf.clone(),
            cut_dual_slots: unit.ws.backward_accum.cut_dual_slots.clone(),
            intercept: unit.intercept,
            coefficients: unit.coefficients,
        }
    }

    fn run_unit<S>(
        &self,
        solver: S,
        risk_measure: RiskMeasure,
        cut_activity_tolerance: Option<f64>,
        with_stage_noise: bool,
    ) -> CompactUnit<S>
    where
        S: SolverInterface<Profile = novomodelo_solver::ActiveProfile> + Send,
    {
        let n_openings = self.probabilities.len();
        self.with_contexts(|ctx, training_ctx| {
            let mut ws = self.workspace_with(
                ctx,
                training_ctx,
                1,
                BackwardScheduler::ByScenario {},
                solver,
            );
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            ws.backward_accum.agg_arena.resize(self.state.n_state, 0.0);

            let successor = self.successor();
            let outcomes =
                successor.outcomes(&self.pools, COMPACT_TEMPLATE_ROWS, cut_activity_tolerance);
            let mut table = StageNoiseTable::default();
            if with_stage_noise {
                table
                    .fill(training_ctx, StageIdx(1), &outcomes)
                    .expect("generated openings need no library");
            }
            let spec = super::SuccessorSpec {
                successor: StageIdx(1),
                iteration: 1,
                my_rank: 0,
                probabilities: &self.probabilities,
                cut_state: &self.layouts[0],
                slab: super::SlabLookup::Unavailable,
                stage_noise: with_stage_noise.then_some(&table),
            };
            let exchange = exchange_with_states(self.state.n_state, vec![COMPACT_X_HAT.to_vec()]);
            let mut basis_store = empty_basis_store(1, 2);
            let mut basis_slices = Vec::new();
            basis_store.split_workers_into(1, 1, &mut basis_slices);

            let params = super::BackwardNodeParams {
                trial_points: &[0],
                exchange: &exchange,
                fwd_offset: 0,
                node_visit_offset: 0,
                risk_measure: &risk_measure,
                succ: &spec,
                outcomes: &outcomes,
            };
            let staged = super::process_by_scenario_backward(
                &mut ws,
                ctx,
                training_ctx,
                &params,
                &mut basis_slices[0],
                &StageOpeningSolver::Frozen,
                0,
                0,
                0,
            )
            .expect("the mock solve succeeds");

            let coefficients =
                staged_cut_coefficients(&staged, &ws.backward_accum.agg_arena).to_vec();
            CompactUnit {
                intercept: staged.intercept,
                coefficients,
                ws,
            }
        })
    }

    fn successor(&self) -> SingleSuccessor {
        let active_slots: Vec<usize> = self.pools[1]
            .active_cuts()
            .map(|(slot, _, _)| slot)
            .collect();
        SingleSuccessor::new(
            NodePos(1),
            self.node_graph.node_ids[NodePos(1)],
            1,
            active_slots.len(),
            self.pools[1].populated(),
            active_slots,
            self.node_graph.nodes[NodePos(1)].openings,
            self.probabilities.len(),
            self.frozen.clone(),
            empty_row_batch(),
            self.chain_rule_slots(),
        )
    }

    fn chain_rule_slots(&self) -> ChainRuleSlots {
        ChainRuleSlots::compose(&self.state, &self.layouts[1], &self.layouts[0])
    }

    fn fill_stage_noise(&self, table: &mut StageNoiseTable) {
        let successor = self.successor();
        self.with_contexts(|_, training_ctx| {
            let outcomes = successor.outcomes(&self.pools, COMPACT_TEMPLATE_ROWS, None);
            table
                .fill(training_ctx, StageIdx(1), &outcomes)
                .expect("generated openings need no library");
        });
    }

    fn chain_rule_cut(&self, run: &CompactRun, cut_duals: &[(usize, f64)]) -> (f64, Vec<f64>) {
        self.reference_cut(&OpeningDuals {
            objective: COMPACT_OBJECTIVE,
            state_duals: &run.state_duals,
            coupled_duals: &run.coupled_duals,
            cut_duals,
        })
    }

    fn reference_cut(&self, opening: &OpeningDuals<'_>) -> (f64, Vec<f64>) {
        let slots = self.chain_rule_slots();
        let mut scratch = ChainRuleScratch::new(self.state.hydro_count, self.state.max_par_order);
        let mut coefficients = vec![0.0; self.layouts[0].n_slots()];
        let intercept = per_opening_cut_into(
            &ChainRuleInputs {
                slots: &slots,
                resident_pool: &self.pools[1],
                extracted: &self.layouts[0],
                par: self.stochastic.par(),
                stage: 1,
            },
            opening,
            &COMPACT_X_HAT,
            &mut scratch,
            &mut coefficients,
        );
        (intercept, coefficients)
    }
}

#[test]
fn compact_by_scenario_pins_once_per_child_and_pushes_one_rhs_range_per_opening() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let storage_col = case.state.incoming_block_start(StateRegion::Storage);

    let run = case.run(Some(1e-9));

    assert_eq!(run.solves.len(), 2);
    let pinned_in: Vec<usize> = run
        .solves
        .iter()
        .enumerate()
        .flat_map(|(solve, recorded)| {
            recorded
                .writes
                .iter()
                .filter(|write| {
                    matches!(write, BoundWrite::ColBounds { indices, .. } if indices.contains(&storage_col))
                })
                .map(move |_| solve)
        })
        .collect();
    assert_eq!(
        pinned_in,
        [0],
        "the incoming storage column is pinned once, in the first solve"
    );
    for recorded in &run.solves {
        let ranges: Vec<(usize, usize)> = recorded
            .writes
            .iter()
            .filter_map(|write| match write {
                BoundWrite::RowLowerRange { first_row, lower } => Some((*first_row, lower.len())),
                _ => None,
            })
            .collect();
        assert_eq!(ranges, [(COMPACT_TEMPLATE_ROWS, 2)]);
        for write in &recorded.writes {
            if let BoundWrite::RowBounds { indices, .. } = write {
                assert!(indices.iter().all(|&row| row < COMPACT_TEMPLATE_ROWS));
            }
        }
    }
}

#[test]
fn compact_by_scenario_maps_cut_rows_to_their_pool_slots() {
    let case = CompactBackwardCase::new(&[1.0]);
    let [y0, y1] = COMPACT_CUT_DUALS;

    let run = case.run(Some(1e-9));

    assert_eq!(run.cut_dual_slots, [(0, y0), (2, y1)]);
    let (intercept, coefficients) = case.chain_rule_cut(&run, &[(0, y0), (2, y1)]);
    assert_eq!(run.intercept, intercept);
    assert_eq!(run.coefficients, coefficients);
    let (_, by_position) = case.chain_rule_cut(&run, &[(0, y0), (1, y1)]);
    assert_ne!(by_position, coefficients);
}

#[test]
fn compact_backward_reads_cut_duals_without_binding_tracking() {
    let case = CompactBackwardCase::new(&[1.0]);
    let [y0, y1] = COMPACT_CUT_DUALS;

    let run = case.run(None);

    let with_pairs = case.chain_rule_cut(&run, &[(0, y0), (2, y1)]);
    let without_pairs = case.chain_rule_cut(&run, &[]);
    assert_ne!(with_pairs, without_pairs);
    assert_eq!((run.intercept, run.coefficients.clone()), with_pairs);
    assert_eq!(run.cut_dual_slots, [(0, y0), (2, y1)]);
}

#[test]
fn compact_forward_stage_writes_the_brackets_of_its_frozen_rows() {
    const ITERATION: u64 = 3;
    const BARE_INTERCEPTS: [f64; 2] = [10.0, -3.0];
    let case = CompactBackwardCase::new(&[1.0]);
    let node = NodePos(1);
    let node_id = case.node_graph.node_ids[node];
    let raw_noise = vec![0.0; case.stochastic.dim()];

    case.with_contexts(|ctx, training_ctx| {
        let forward_record = |dcs: Option<DcsParams>| {
            let mut ws = case.workspace_with(
                ctx,
                training_ctx,
                1,
                compact_by_node(1),
                MockSolver::always_ok(case.solution()),
            );
            ws.solver.load_model(&case.frozen);
            ws.current_state.clear();
            ws.current_state.extend_from_slice(&COMPACT_X_HAT);
            let mut basis_store = empty_basis_store(1, 2);
            let mut slices = Vec::new();
            basis_store.split_workers_into(1, 1, &mut slices);
            let mut records = vec![TrajectoryRecord::default(); 2];
            records[1].frozen_brackets.reserve(2);
            let key = StageKey {
                t: StageIdx(1),
                m: 0,
                local_m: 0,
                iteration: ITERATION,
                raw_noise: &raw_noise,
                basis_row_capacity: case.frozen.num_rows,
                pool: &case.pools[1],
                dcs: dcs.filter(|p| p.is_active(ITERATION)),
                node,
            };
            run_forward_stage(
                &mut ws,
                &mut slices[0],
                ctx,
                training_ctx,
                &key,
                &mut records,
            )
            .expect("the mock solve succeeds");
            records.swap_remove(1)
        };

        let expected = to_bits(&compact_slab_reference(&case, &COMPACT_X_HAT));
        assert_eq!(expected.len(), 2);
        assert_ne!(expected, to_bits(&BARE_INTERCEPTS));

        let frozen = forward_record(None);
        let written = frozen
            .frozen_brackets
            .read(ITERATION, node_id, expected.len())
            .expect("a frozen compact stage writes its brackets");
        assert_eq!(to_bits(written), expected);
        assert_eq!(
            frozen
                .frozen_brackets
                .read(ITERATION + 1, node_id, expected.len()),
            None
        );

        let start = ITERATION - 1;
        assert!(dcs_params(start).is_active(ITERATION));
        let lazy = forward_record(Some(dcs_params(start)));
        assert_eq!(
            lazy.frozen_brackets
                .read(ITERATION, node_id, expected.len()),
            None
        );
    });
}

fn pins_per_solve(solves: &[RecordedSolve], column: usize) -> Vec<usize> {
    solves
        .iter()
        .map(|recorded| {
            recorded
                .writes
                .iter()
                .filter(|write| {
                    matches!(write, BoundWrite::ColBounds { indices, .. } if indices.contains(&column))
                })
                .count()
        })
        .collect()
}

fn range_pushes_per_solve(solves: &[RecordedSolve]) -> Vec<Vec<(usize, usize)>> {
    solves
        .iter()
        .map(|recorded| {
            recorded
                .writes
                .iter()
                .filter_map(|write| match write {
                    BoundWrite::RowLowerRange { first_row, lower } => {
                        Some((*first_row, lower.len()))
                    }
                    _ => None,
                })
                .collect()
        })
        .collect()
}

fn to_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn compact_by_node_pins_once_per_claimed_block() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let storage_col = case.state.incoming_block_start(StateRegion::Storage);
    let by_node = |block_size| BackwardScheduler::ByNode {
        block_size: NonZeroUsize::new(block_size),
    };

    let by_scenario = case.run_pass(BackwardScheduler::ByScenario {});
    let one_claim = case.run_pass(by_node(2));
    let two_claims = case.run_pass(by_node(1));

    assert_eq!(
        pins_per_solve(&one_claim.solves, storage_col),
        [1, 0],
        "one claim covers both openings: one load, one pin"
    );
    assert_eq!(
        pins_per_solve(&two_claims.solves, storage_col),
        [1, 1],
        "two claims load twice: each load is pinned"
    );
    for pass in [&one_claim, &two_claims] {
        assert_eq!(
            range_pushes_per_solve(&pass.solves),
            [[(COMPACT_TEMPLATE_ROWS, 2)], [(COMPACT_TEMPLATE_ROWS, 2)]]
        );
    }

    assert!(by_scenario.intercepts.iter().any(|v| v.abs() > 0.0));
    for pass in [&one_claim, &two_claims] {
        assert_eq!(to_bits(&pass.intercepts), to_bits(&by_scenario.intercepts));
        assert_eq!(
            to_bits(&pass.coefficients),
            to_bits(&by_scenario.coefficients)
        );
    }
}

const COMPACT_CVAR: RiskMeasure = RiskMeasure::CVaR {
    alpha: 0.5,
    lambda: 0.5,
};

fn compact_two_trial_states() -> [Vec<f64>; 2] {
    [COMPACT_X_HAT.to_vec(), vec![40.0, 6.0, 9.0]]
}

fn compact_by_node(block_size: usize) -> BackwardScheduler {
    BackwardScheduler::ByNode {
        block_size: NonZeroUsize::new(block_size),
    }
}

fn compact_slab_schedulers() -> [BackwardScheduler; 2] {
    [BackwardScheduler::ByScenario {}, compact_by_node(1)]
}

fn compact_slab_reference(case: &CompactBackwardCase, state: &[f64]) -> Vec<f64> {
    case.with_contexts(|ctx, training_ctx| {
        let mut reference = case.workspace_with(
            ctx,
            training_ctx,
            1,
            compact_by_node(1),
            MockSolver::always_ok(case.solution()),
        );
        reference.solver.load_model(&case.frozen);
        StageSolvePrep::pin_and_bracket(
            &mut reference.solver,
            &mut reference.patch_buf,
            &mut reference.scratch,
            ctx,
            training_ctx,
            StageIdx(1),
            StateSource(state),
            &LoadedCutRows {
                pool: &case.pools[1],
                projection: &case.layouts[1],
                slots: &[0, 2],
                first_row: COMPACT_TEMPLATE_ROWS,
            },
            BracketSource::Computed,
        );
        reference.scratch.cut_bracket_buf.clone()
    })
}

fn write_compact_slab(
    record: usize,
    iteration: u64,
    node_id: NodeId,
    brackets: &[f64],
) -> impl FnOnce(&mut [TrajectoryRecord]) + '_ {
    move |records| {
        let slab = &mut records[record].frozen_brackets;
        slab.reserve(brackets.len());
        slab.write(iteration, node_id, brackets);
    }
}

fn compact_slab_pass(
    case: &CompactBackwardCase,
    scheduler: BackwardScheduler,
    states: &[Vec<f64>],
    prepare_records: impl FnOnce(&mut [TrajectoryRecord]),
) -> CompactPass {
    let spec = PassSpec {
        scheduler,
        states,
        risk_measure: RiskMeasure::Expectation,
        hardest_first: true,
        bypass_solve_caches: false,
    };
    case.with_contexts(|ctx, training_ctx| {
        let mut workspaces = vec![case.workspace_with(
            ctx,
            training_ctx,
            states.len(),
            scheduler,
            MockSolver::always_ok(case.solution()),
        )];
        case.run_pass_on(ctx, training_ctx, &spec, &mut workspaces, prepare_records)
    })
}

fn compact_pin_count(case: &CompactBackwardCase, pass: &CompactPass) -> u64 {
    let storage_col = case.state.incoming_block_start(StateRegion::Storage);
    let pins: usize = pins_per_solve(&pass.solves, storage_col).iter().sum();
    u64::try_from(pins).expect("pin count fits u64")
}

fn row_lower_range_bits(solves: &[RecordedSolve]) -> Vec<Vec<u64>> {
    solves
        .iter()
        .map(|recorded| {
            recorded
                .writes
                .iter()
                .flat_map(|write| match write {
                    BoundWrite::RowLowerRange { lower, .. } => to_bits(lower),
                    _ => Vec::new(),
                })
                .collect()
        })
        .collect()
}

fn assert_same_cut_bits(pass: &CompactPass, unwritten: &CompactPass, what: &str) {
    assert_eq!(
        row_lower_range_bits(&pass.solves),
        row_lower_range_bits(&unwritten.solves),
        "{what}: pushed rhs"
    );
    assert_eq!(
        to_bits(&pass.intercepts),
        to_bits(&unwritten.intercepts),
        "{what}: intercepts"
    );
    assert_eq!(
        to_bits(&pass.coefficients),
        to_bits(&unwritten.coefficients),
        "{what}: coefficients"
    );
}

#[test]
fn compact_slab_unwritten_slab_misses_on_every_pin() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = [COMPACT_X_HAT.to_vec()];

    for scheduler in compact_slab_schedulers() {
        let pass = compact_slab_pass(&case, scheduler, &states, |_| {});

        let pins = compact_pin_count(&case, &pass);
        assert!(pins > 0, "{scheduler:?}");
        assert_eq!(pass.slab_misses, pins, "{scheduler:?}");
    }
}

#[test]
fn compact_slab_hit_reads_without_a_miss_and_matches_the_unwritten_run() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = [COMPACT_X_HAT.to_vec()];
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let brackets = compact_slab_reference(&case, &COMPACT_X_HAT);

    for scheduler in compact_slab_schedulers() {
        let unwritten = compact_slab_pass(&case, scheduler, &states, |_| {});
        let hit = compact_slab_pass(
            &case,
            scheduler,
            &states,
            write_compact_slab(1, COMPACT_PASS_ITERATION, node_id, &brackets),
        );

        assert_eq!(hit.slab_misses, 0, "{scheduler:?}");
        assert!(unwritten.intercepts.iter().any(|v| v.abs() > 0.0));
        assert_same_cut_bits(&hit, &unwritten, &format!("{scheduler:?}"));
    }
}

#[test]
fn compact_slab_foreign_tags_miss_on_every_pin_and_match_the_unwritten_run() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = [COMPACT_X_HAT.to_vec()];
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let mut brackets = compact_slab_reference(&case, &COMPACT_X_HAT);
    brackets[0] += 1.0;
    let foreign = [
        (COMPACT_PASS_ITERATION - 1, node_id),
        (COMPACT_PASS_ITERATION, NodeId(99)),
    ];

    for scheduler in compact_slab_schedulers() {
        let unwritten = compact_slab_pass(&case, scheduler, &states, |_| {});
        let pins = compact_pin_count(&case, &unwritten);
        for (iteration, tagged) in foreign {
            let pass = compact_slab_pass(
                &case,
                scheduler,
                &states,
                write_compact_slab(1, iteration, tagged, &brackets),
            );

            assert_eq!(pass.slab_misses, pins, "{scheduler:?} {iteration} {tagged}");
            assert_same_cut_bits(
                &pass,
                &unwritten,
                &format!("{scheduler:?} {iteration} {tagged}"),
            );
        }
    }
}

#[test]
fn compact_slab_is_read_at_each_trials_own_record() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = compact_two_trial_states();
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let brackets = compact_slab_reference(&case, &states[1]);

    for scheduler in compact_slab_schedulers() {
        let unwritten = compact_slab_pass(&case, scheduler, &states, |_| {});
        let second_trial_hit = compact_slab_pass(
            &case,
            scheduler,
            &states,
            write_compact_slab(3, COMPACT_PASS_ITERATION, node_id, &brackets),
        );

        let pins = compact_pin_count(&case, &unwritten);
        assert_eq!(second_trial_hit.slab_misses * 2, pins, "{scheduler:?}");
        assert_same_cut_bits(&second_trial_hit, &unwritten, &format!("{scheduler:?}"));
    }
}

#[test]
fn compact_slab_values_reach_the_pushed_cut_rhs() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = [COMPACT_X_HAT.to_vec()];
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let mut perturbed = compact_slab_reference(&case, &COMPACT_X_HAT);
    perturbed[0] += 1.0;

    for scheduler in compact_slab_schedulers() {
        let unwritten = compact_slab_pass(&case, scheduler, &states, |_| {});
        let pass = compact_slab_pass(
            &case,
            scheduler,
            &states,
            write_compact_slab(1, COMPACT_PASS_ITERATION, node_id, &perturbed),
        );

        assert_eq!(pass.slab_misses, 0, "{scheduler:?}");
        let pushed = row_lower_range_bits(&pass.solves);
        let recomputed = row_lower_range_bits(&unwritten.solves);
        assert_eq!(pushed.len(), recomputed.len());
        for (solve, (pushed, recomputed)) in pushed.iter().zip(&recomputed).enumerate() {
            assert_ne!(pushed[0], recomputed[0], "{scheduler:?} solve {solve}");
            assert_eq!(pushed[1], recomputed[1], "{scheduler:?} solve {solve}");
        }
    }
}

#[test]
fn compact_slab_lookup_maps_to_the_pin_source() {
    let case = CompactBackwardCase::new(&[1.0]);
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let values = [1.5, -2.5];
    let mut records = vec![TrajectoryRecord::default(); 4];
    write_compact_slab(3, 7, node_id, &values)(&mut records);
    let spec = |slab, iteration| super::SuccessorSpec {
        successor: StageIdx(1),
        iteration,
        my_rank: 0,
        probabilities: &case.probabilities,
        cut_state: &case.layouts[0],
        slab,
        stage_noise: None,
    };
    let frozen = super::SlabLookup::Frozen {
        records: &records,
        n_stages: 2,
    };
    let successor = case.successor();
    let outcomes = successor.outcomes(&case.pools, COMPACT_TEMPLATE_ROWS, None);
    let child = outcomes.child(0);
    assert_eq!(child.successor_node_id, node_id);
    assert_eq!(child.frozen_rows(), values.len());
    let other_node = super::SuccessorChild {
        successor_node_id: NodeId(99),
        ..outcomes.child(0)
    };
    let other_rows = super::SuccessorChild {
        num_cuts_at_successor: child.num_cuts_at_successor + 1,
        ..outcomes.child(0)
    };

    assert_eq!(
        spec(super::SlabLookup::Unavailable, 7).bracket_source(1, &child),
        BracketSource::Computed
    );
    assert_eq!(
        spec(frozen, 7).bracket_source(1, &child),
        BracketSource::Slab(Some(values.as_slice()))
    );
    assert_eq!(
        spec(frozen, 6).bracket_source(1, &child),
        BracketSource::Slab(None)
    );
    assert_eq!(
        spec(frozen, 7).bracket_source(1, &other_node),
        BracketSource::Slab(None)
    );
    assert_eq!(
        spec(frozen, 7).bracket_source(0, &child),
        BracketSource::Slab(None)
    );
    assert_eq!(
        spec(frozen, 7).bracket_source(1, &other_rows),
        BracketSource::Slab(None)
    );
}

#[test]
fn a_slab_read_with_another_row_count_is_a_miss() {
    const ITERATION: u64 = 4;
    let node_id = NodeId(2);
    let brackets = [1.5, -2.5, 0.25];
    let rows = brackets.len();
    let mut slab = FrozenBrackets::default();
    slab.reserve(rows);
    slab.write(ITERATION, node_id, &brackets);

    assert_eq!(
        slab.read(ITERATION, node_id, rows).map(to_bits),
        Some(to_bits(&brackets))
    );
    assert_eq!(slab.read(ITERATION, node_id, rows - 1), None);
    assert_eq!(slab.read(ITERATION, node_id, rows + 1), None);
}

fn row_and_col_bound_bits(solves: &[RecordedSolve]) -> Vec<Vec<Vec<u64>>> {
    let write_bits = |kind: u64, indices: &[usize], lower: &[f64], upper: &[f64]| {
        let indices = indices
            .iter()
            .map(|&index| u64::try_from(index).expect("index fits u64"))
            .collect();
        [vec![kind], indices, to_bits(lower), to_bits(upper)].concat()
    };
    solves
        .iter()
        .map(|recorded| {
            recorded
                .writes
                .iter()
                .filter_map(|write| match write {
                    BoundWrite::RowBounds {
                        indices,
                        lower,
                        upper,
                    } => Some(write_bits(0, indices, lower, upper)),
                    BoundWrite::ColBounds {
                        indices,
                        lower,
                        upper,
                    } => Some(write_bits(1, indices, lower, upper)),
                    _ => None,
                })
                .collect()
        })
        .collect()
}

#[test]
fn compact_solve_cache_bypass_forces_the_miss_and_gives_the_cached_bits() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let states = compact_two_trial_states();
    let node_id = case.node_graph.node_ids[NodePos(1)];
    let first = compact_slab_reference(&case, &states[0]);
    let second = compact_slab_reference(&case, &states[1]);
    let hit_slab = |records: &mut [TrajectoryRecord]| {
        write_compact_slab(1, COMPACT_PASS_ITERATION, node_id, &first)(records);
        write_compact_slab(3, COMPACT_PASS_ITERATION, node_id, &second)(records);
    };
    let pass = |scheduler, bypass_solve_caches| {
        let spec = PassSpec {
            scheduler,
            states: &states,
            risk_measure: RiskMeasure::Expectation,
            hardest_first: true,
            bypass_solve_caches,
        };
        case.with_contexts(|ctx, training_ctx| {
            let mut workspaces = vec![case.workspace_with(
                ctx,
                training_ctx,
                states.len(),
                scheduler,
                NoiseKeyedSolver::new(case.solution()),
            )];
            case.run_pass_on(ctx, training_ctx, &spec, &mut workspaces, hit_slab)
        })
    };

    for scheduler in compact_slab_schedulers() {
        let cached = pass(scheduler, false);
        let bypassed = pass(scheduler, true);

        let pins = compact_pin_count(&case, &bypassed);
        assert!(pins > 0, "{scheduler:?}");
        assert_eq!(cached.slab_misses, 0, "{scheduler:?}");
        assert_eq!(bypassed.slab_misses, pins, "{scheduler:?}");
        assert!(cached.intercepts.iter().any(|v| v.abs() > 0.0));
        assert_eq!(
            row_and_col_bound_bits(&bypassed.solves),
            row_and_col_bound_bits(&cached.solves),
            "{scheduler:?}: pushed row and column bounds"
        );
        assert_same_cut_bits(&bypassed, &cached, &format!("{scheduler:?}"));
    }
}

#[test]
fn compact_aggregate_first_cut_is_scheduler_and_worker_independent() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let states = compact_two_trial_states();
    let run = |scheduler, workers, hardest_first, risk_measure| {
        let spec = PassSpec {
            scheduler,
            states: &states,
            risk_measure,
            hardest_first,
            bypass_solve_caches: false,
        };
        case.run_pass_with(&spec, workers, || NoiseKeyedSolver::new(case.solution()))
    };
    let by_scenario = BackwardScheduler::ByScenario {};

    let reference = run(by_scenario, 1, true, COMPACT_CVAR);

    assert!(reference.intercepts.iter().any(|v| v.abs() > 0.0));
    let expectation = run(by_scenario, 1, true, RiskMeasure::Expectation);
    assert_ne!(
        to_bits(&expectation.coefficients),
        to_bits(&reference.coefficients),
        "the openings must differ in cost for the CVaR weights to differ from the probabilities"
    );
    for (scheduler, workers, hardest_first) in [
        (by_scenario, 2, true),
        (compact_by_node(1), 1, true),
        (compact_by_node(1), 2, true),
        (compact_by_node(1), 2, false),
    ] {
        let pass = run(scheduler, workers, hardest_first, COMPACT_CVAR);
        assert_eq!(
            to_bits(&pass.intercepts),
            to_bits(&reference.intercepts),
            "{scheduler:?} x {workers} workers, hardest-first {hardest_first}"
        );
        assert_eq!(
            to_bits(&pass.coefficients),
            to_bits(&reference.coefficients),
            "{scheduler:?} x {workers} workers, hardest-first {hardest_first}"
        );
    }
}

#[test]
fn compact_aggregate_first_agrees_with_weighted_per_opening_cuts() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);

    let unit = case.run_unit(
        NoiseKeyedSolver::new(case.solution()),
        COMPACT_CVAR,
        Some(1e-9),
        false,
    );

    let records = &unit.ws.backward_accum.opening_records;
    assert_eq!(records.len(), 2);
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&i| records.key(i).1);
    let openings: Vec<OpeningDuals<'_>> = order.into_iter().map(|i| records.view(i)).collect();
    let costs: Vec<f64> = openings.iter().map(|opening| opening.objective).collect();
    let mut risk_scratch = RiskMeasureScratch::new();
    let weights = COMPACT_CVAR
        .opening_weights_into(&costs, &case.probabilities, &mut risk_scratch)
        .to_vec();
    let cuts: Vec<(f64, Vec<f64>)> = openings
        .iter()
        .map(|opening| case.reference_cut(opening))
        .collect();
    let weighted = |weights: &[f64]| -> Vec<f64> {
        let mut sum = vec![0.0; 1 + case.layouts[0].n_slots()];
        for (weight, (intercept, coefficients)) in weights.iter().zip(&cuts) {
            sum[0] += weight * intercept;
            for (slot, coefficient) in sum[1..].iter_mut().zip(coefficients) {
                *slot += weight * coefficient;
            }
        }
        sum
    };

    let staged: Vec<f64> = std::iter::once(unit.intercept)
        .chain(unit.coefficients.iter().copied())
        .collect();
    for (a, b) in staged.iter().zip(weighted(&weights)) {
        assert!(
            (a - b).abs() <= 1e-12 * 1.0_f64.max(a.abs()).max(b.abs()),
            "aggregate-first {a} vs weighted per-opening {b}"
        );
    }
    let by_probability = weighted(&case.probabilities);
    assert!(
        staged
            .iter()
            .zip(&by_probability)
            .any(|(a, b)| (a - b).abs() > 1e-9),
        "the CVaR weights must move the cut off the probability-weighted one"
    );
}

#[test]
fn pure_cvar_counts_every_binding_opening_as_expectation_does() {
    const PURE_CVAR: RiskMeasure = RiskMeasure::CVaR {
        alpha: 0.5,
        lambda: 1.0,
    };
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let states = compact_two_trial_states();

    let unit = case.run_unit(
        NoiseKeyedSolver::new(case.solution()),
        PURE_CVAR,
        Some(1e-9),
        false,
    );
    let records = &unit.ws.backward_accum.opening_records;
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&i| records.key(i).1);
    let costs: Vec<f64> = order.iter().map(|&i| records.view(i).objective).collect();
    let mut risk_scratch = RiskMeasureScratch::new();
    let weights = PURE_CVAR.opening_weights_into(&costs, &case.probabilities, &mut risk_scratch);
    assert!(
        weights.contains(&0.0),
        "pure CVaR must give an opening an exact zero weight: {weights:?}"
    );

    let counts = |scheduler, risk_measure| {
        let spec = PassSpec {
            scheduler,
            states: &states,
            risk_measure,
            hardest_first: false,
            bypass_solve_caches: false,
        };
        case.run_pass_with(&spec, 2, || NoiseKeyedSolver::new(case.solution()))
            .successor_active_counts
    };
    let openings_times_trials = (case.probabilities.len() * states.len()) as u64;
    for scheduler in [BackwardScheduler::ByScenario {}, compact_by_node(2)] {
        let pure = counts(scheduler, PURE_CVAR);
        let expectation = counts(scheduler, RiskMeasure::Expectation);

        assert_eq!(pure, expectation, "{scheduler:?}");
        assert_eq!(
            pure,
            [openings_times_trials, 0, openings_times_trials],
            "{scheduler:?}"
        );
    }
}

fn assert_opening_records_never_reallocate(
    scheduler: BackwardScheduler,
    trial_points_held: impl FnOnce(usize) -> usize,
) {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let states = compact_two_trial_states();
    let held_records = trial_points_held(states.len()) * case.probabilities.len();
    let spec = PassSpec {
        scheduler,
        states: &states,
        risk_measure: RiskMeasure::Expectation,
        hardest_first: true,
        bypass_solve_caches: false,
    };

    case.with_contexts(|ctx, training_ctx| {
        let mut workspaces = vec![case.workspace_with(
            ctx,
            training_ctx,
            states.len(),
            spec.scheduler,
            NoiseKeyedSolver::new(case.solution()),
        )];
        let identity = |ws: &SolverWorkspace<_>| {
            let accum = &ws.backward_accum;
            (
                accum.opening_records.buffer_identity(),
                [
                    (
                        accum.state_duals_buf.as_ptr().addr(),
                        accum.state_duals_buf.capacity(),
                    ),
                    (
                        accum.cut_dual_slots.as_ptr().addr(),
                        accum.cut_dual_slots.capacity(),
                    ),
                ],
            )
        };
        let reserved = identity(&workspaces[0]);
        assert!(reserved.0.iter().all(|&(_, capacity)| capacity > 0));
        assert!(reserved.1.iter().all(|&(_, capacity)| capacity > 0));

        for _ in 0..2 {
            case.run_pass_on(ctx, training_ctx, &spec, &mut workspaces, |_| {});

            let accum = &workspaces[0].backward_accum;
            assert_eq!(accum.opening_records.len(), held_records);
            assert!(!accum.cut_dual_slots.is_empty());
            assert_eq!(identity(&workspaces[0]), reserved);
        }
    });
}

#[test]
fn compact_aggregate_first_records_never_reallocate_inside_the_claim_loop() {
    assert_opening_records_never_reallocate(compact_by_node(1), |trial_points| trial_points);
}

#[test]
fn compact_by_scenario_records_never_reallocate_inside_the_unit_loop() {
    assert_opening_records_never_reallocate(BackwardScheduler::ByScenario {}, |_| 1);
}

const COMPACT_ACTIVE_SLOTS: [usize; 2] = [0, 2];

fn stage_noise_bits(table: &StageNoiseTable, n_outcomes: usize, rows: usize) -> Vec<u64> {
    (0..n_outcomes)
        .flat_map(|omega| {
            to_bits(table.noise_term(omega))
                .into_iter()
                .chain(to_bits(table.cut_noise(omega, rows)))
        })
        .collect()
}

fn has_nonzero_cut_noise(case: &CompactBackwardCase) -> bool {
    let mut table = StageNoiseTable::default();
    case.fill_stage_noise(&mut table);
    (0..case.probabilities.len()).any(|omega| {
        table
            .cut_noise(omega, COMPACT_ACTIVE_SLOTS.len())
            .iter()
            .any(|&w| w != 0.0)
    })
}

fn bound_write_bits(solves: &[RecordedSolve]) -> Vec<Vec<u64>> {
    solves
        .iter()
        .map(|recorded| {
            recorded
                .writes
                .iter()
                .flat_map(|write| match write {
                    BoundWrite::RowBounds { lower, upper, .. }
                    | BoundWrite::ColBounds { lower, upper, .. } => {
                        [to_bits(lower), to_bits(upper)].concat()
                    }
                    BoundWrite::RowLowerRange { lower, .. } => to_bits(lower),
                    BoundWrite::AddRows { .. } => Vec::new(),
                })
                .collect()
        })
        .collect()
}

#[test]
fn compact_stage_noise_table_holds_each_opening_noise_term_and_cut_noise() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let mut table = StageNoiseTable::default();
    case.fill_stage_noise(&mut table);

    case.with_contexts(|_, training_ctx| {
        let stochastic = training_ctx.stochastic;
        let tree_view = stochastic.tree_view();
        let lag_point = case.layouts[1].lag_point();
        for omega in 0..case.probabilities.len() {
            let raw_noise = tree_view.opening(1, omega);
            let mut noise_term = vec![0.0; case.state.hydro_count];
            evaluate_par_noise_term_batch(
                stochastic.par(),
                1,
                &raw_noise[stochastic.class_dimensions().hydro_range()],
                &mut noise_term,
            );
            let cut_noise: Vec<f64> = COMPACT_ACTIVE_SLOTS
                .map(|slot| {
                    cut_noise_term(case.pools[1].coefficient_row(slot), lag_point, &noise_term)
                })
                .to_vec();

            assert_eq!(to_bits(table.noise_term(omega)), to_bits(&noise_term));
            assert_eq!(
                to_bits(table.cut_noise(omega, COMPACT_ACTIVE_SLOTS.len())),
                to_bits(&cut_noise)
            );
        }
    });
}

#[test]
fn compact_stage_noise_table_prices_each_child_with_its_own_slots() {
    let case = CompactBackwardCase::new(&[0.5, 0.5]);
    let openings = case.node_graph.nodes[NodePos(1)].openings;
    let child = |node_id: NodeId, active_slots: Vec<usize>| ChildSpec {
        successor_node: NodePos(1),
        successor_node_id: node_id,
        pool_id: 1,
        openings,
        num_cuts_at_successor: active_slots.len(),
        populated_count: case.pools[1].populated(),
        active_slots,
        chain_rule_slots: case.chain_rule_slots(),
    };
    let fan = MultiSuccessor::new(
        vec![child(NodeId(10), vec![0, 2]), child(NodeId(11), vec![2])],
        vec![case.frozen.clone(); 2],
        empty_cut_batches(2),
    );
    let mut table = StageNoiseTable::default();

    case.with_contexts(|_, training_ctx| {
        let outcomes = fan.outcomes(&case.pools, COMPACT_TEMPLATE_ROWS, None);
        table
            .fill(training_ctx, StageIdx(1), &outcomes)
            .expect("generated openings need no library");
    });

    let lag_point = case.layouts[1].lag_point();
    let cut_noise = |slot: usize, noise_term: &[f64]| {
        cut_noise_term(case.pools[1].coefficient_row(slot), lag_point, noise_term)
    };
    let n_openings = openings.len;
    for local_omega in 0..n_openings {
        let noise_term = table.noise_term(local_omega).to_vec();
        assert_eq!(
            to_bits(&noise_term),
            to_bits(table.noise_term(n_openings + local_omega)),
            "both children draw the same opening tree"
        );
        assert_eq!(
            to_bits(table.cut_noise(local_omega, 2)),
            to_bits(&[cut_noise(0, &noise_term), cut_noise(2, &noise_term)])
        );
        assert_eq!(
            to_bits(table.cut_noise(n_openings + local_omega, 1)),
            to_bits(&[cut_noise(2, &noise_term)])
        );
    }
}

#[test]
fn compact_stage_noise_table_is_thread_count_independent() {
    const OPENINGS: usize = 512;
    let case = CompactBackwardCase::new(&[1.0 / 512.0; OPENINGS]);
    let rows = COMPACT_ACTIVE_SLOTS.len();
    let filled_in_pool = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a rayon pool builds");
        let mut table = StageNoiseTable::default();
        pool.install(|| case.fill_stage_noise(&mut table));
        table
    };

    let single = filled_in_pool(1);

    assert!(
        (0..OPENINGS).any(|omega| single.cut_noise(omega, rows).iter().any(|&w| w != 0.0)),
        "the parallel W pass must write a nonzero cut-row term"
    );
    assert_eq!(
        stage_noise_bits(&filled_in_pool(4), OPENINGS, rows),
        stage_noise_bits(&single, OPENINGS, rows)
    );
}

#[test]
fn compact_stage_noise_table_never_reallocates_after_reserve() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let mut table = StageNoiseTable::default();
    table.reserve(
        case.probabilities.len(),
        case.state.hydro_count,
        case.pools[1].populated(),
        case.stochastic.class_dimensions().total(),
    );
    let reserved = table.buffer_identity();
    assert!(reserved.iter().all(|&(_, capacity)| capacity > 0));

    for _ in 0..2 {
        case.fill_stage_noise(&mut table);
        assert_eq!(table.buffer_identity(), reserved);
    }
}

#[test]
fn compact_backward_pass_fills_the_stage_noise_table_inside_its_reserve() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let states = compact_two_trial_states();
    let rows = COMPACT_ACTIVE_SLOTS.len();

    for scheduler in compact_slab_schedulers() {
        let spec = PassSpec {
            scheduler,
            states: &states,
            risk_measure: RiskMeasure::Expectation,
            hardest_first: true,
            bypass_solve_caches: false,
        };
        let mut identities = Vec::new();
        let mut wrote_cut_noise = false;
        case.with_contexts(|ctx, training_ctx| {
            let mut workspaces = vec![case.workspace_with(
                ctx,
                training_ctx,
                states.len(),
                scheduler,
                NoiseKeyedSolver::new(case.solution()),
            )];
            case.run_pass_inspected(
                ctx,
                training_ctx,
                &spec,
                &mut workspaces,
                |_| {},
                |state| {
                    let table = state.stage_noise();
                    if !identities.is_empty() {
                        wrote_cut_noise = (0..case.probabilities.len())
                            .any(|omega| table.cut_noise(omega, rows).iter().any(|&w| w != 0.0));
                    }
                    identities.push(table.buffer_identity());
                },
            );
        });

        let [reserved, after_pass] = identities[..] else {
            panic!("the state is inspected after its reserves and after the pass");
        };
        assert!(reserved.iter().all(|&(_, capacity)| capacity > 0));
        assert!(
            wrote_cut_noise,
            "{scheduler:?} must fill the table it reads"
        );
        assert_eq!(after_pass, reserved, "{scheduler:?} reallocated the table");
    }
}

#[test]
fn compact_stage_noise_power_check_fails_without_lag_coefficients() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let without_lag_coefficients = CompactBackwardCase::with_successor_cuts(
        &[0.2, 0.3, 0.5],
        COMPACT_SUCCESSOR_CUTS.map(|(intercept, _)| (intercept, [0.0; 2])),
    );

    assert!(has_nonzero_cut_noise(&case));
    assert!(!has_nonzero_cut_noise(&without_lag_coefficients));
}

#[test]
fn compact_stage_noise_unit_matches_the_on_the_fly_unit() {
    let case = CompactBackwardCase::new(&[0.2, 0.3, 0.5]);
    let run = |with_stage_noise| {
        case.run_unit(
            NoiseKeyedSolver::new(case.solution()),
            COMPACT_CVAR,
            Some(1e-9),
            with_stage_noise,
        )
    };

    let on_the_fly = run(false);
    let staged = run(true);

    let (expected, actual) = (
        on_the_fly.ws.solver.inner().recorded(),
        staged.ws.solver.inner().recorded(),
    );
    assert_eq!(expected.len(), case.probabilities.len());
    assert!(
        row_lower_range_bits(expected)
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f64.to_bits())
    );
    assert_eq!(bound_write_bits(actual), bound_write_bits(expected));
    for (staged_solve, on_the_fly_solve) in actual.iter().zip(expected) {
        assert_eq!(staged_solve.writes, on_the_fly_solve.writes);
    }
    assert_eq!(staged.intercept.to_bits(), on_the_fly.intercept.to_bits());
    assert_eq!(
        to_bits(&staged.coefficients),
        to_bits(&on_the_fly.coefficients)
    );
}

#[test]
fn take_first_error_returns_the_first_worker_error_and_clears_every_slot() {
    let mut slots = [
        None,
        Some(SddpError::Validation("first".to_string())),
        Some(SddpError::Validation("second".to_string())),
    ];

    let first = take_first_error(slots.iter_mut());

    assert!(matches!(first, Err(SddpError::Validation(message)) if message == "first"));
    assert!(slots.iter().all(Option::is_none));
    assert!(take_first_error(slots.iter_mut()).is_ok());
}
