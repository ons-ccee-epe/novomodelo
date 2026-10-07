//! Per-phase delta of solver counters for LP statistics collection.
//!
//! [`SolverStatsDelta`] is computed from before/after [`SolverStatistics`] snapshots
//! taken around each training phase (forward pass, backward pass, lower bound evaluation).
//! The deltas are stored per-iteration and per-phase for later Parquet output and
//! CLI display.

use cobre_io::SolverStatsRow;
use cobre_solver::SolverStatistics;

/// Delta of solver counters between two snapshots.
///
/// All fields represent the difference: after minus before snapshot.
#[derive(Debug, Clone, Default)]
pub struct SolverStatsDelta {
    /// Number of LP solves in this phase.
    pub lp_solves: u64,

    /// Solves that returned optimal (including retried solves that eventually succeeded).
    pub lp_successes: u64,

    /// Solves that returned optimal on first attempt (before any retry).
    pub first_try_successes: u64,

    /// Solves that exhausted all retry levels.
    pub lp_failures: u64,

    /// Total retry attempts across all retried solves.
    pub retry_attempts: u64,

    /// Number of warm-start `solve(Some(&basis))` calls.
    pub basis_offered: u64,

    /// Times the offered basis was rejected because `isBasisConsistent` returned false.
    pub basis_consistency_failures: u64,

    /// Total simplex iterations across all solves.
    pub simplex_iterations: u64,

    /// Cumulative wall-clock solve time in milliseconds.
    pub solve_time_ms: f64,

    /// Number of `load_model` calls in this phase.
    pub load_model_count: u64,

    /// Cumulative wall-clock time spent in `load_model` calls, in milliseconds.
    pub load_model_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_row_bounds`/`set_col_bounds` calls, in milliseconds.
    pub set_bounds_time_ms: f64,

    /// Cumulative wall-clock time spent in `set_basis` FFI calls, in milliseconds.
    pub basis_set_time_ms: f64,

    /// Per-level retry success histogram delta. Length depends on the solver
    /// backend (e.g. 12 for `HiGHS`).
    pub retry_level_histogram: Vec<u64>,
}

/// Resize histogram on first use to match the source histogram length.
fn ensure_histogram_capacity(result: &mut Vec<u64>, source: &[u64]) {
    if result.is_empty() && !source.is_empty() {
        result.resize(source.len(), 0);
    }
}

impl SolverStatsDelta {
    /// Compute the delta between two [`SolverStatistics`] snapshots.
    #[must_use]
    pub fn from_snapshots(before: &SolverStatistics, after: &SolverStatistics) -> Self {
        Self {
            lp_solves: after.solve_count - before.solve_count,
            lp_successes: after.success_count - before.success_count,
            first_try_successes: after.first_try_successes - before.first_try_successes,
            lp_failures: after.failure_count - before.failure_count,
            retry_attempts: after.retry_count - before.retry_count,
            basis_offered: after.basis_offered - before.basis_offered,
            basis_consistency_failures: after.basis_consistency_failures
                - before.basis_consistency_failures,
            simplex_iterations: after.total_iterations - before.total_iterations,
            solve_time_ms: (after.total_solve_time_seconds - before.total_solve_time_seconds)
                * 1000.0,
            load_model_count: after.load_model_count - before.load_model_count,
            load_model_time_ms: (after.total_load_model_time_seconds
                - before.total_load_model_time_seconds)
                * 1000.0,
            set_bounds_time_ms: (after.total_set_bounds_time_seconds
                - before.total_set_bounds_time_seconds)
                * 1000.0,
            basis_set_time_ms: (after.total_basis_set_time_seconds
                - before.total_basis_set_time_seconds)
                * 1000.0,
            retry_level_histogram: after
                .retry_level_histogram
                .iter()
                .zip(&before.retry_level_histogram)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Add `rhs` element-wise into `dst` in place — allocation-free for the hot backward-pass path.
    pub fn accumulate_into(dst: &mut Self, rhs: &Self) {
        dst.lp_solves += rhs.lp_solves;
        dst.lp_successes += rhs.lp_successes;
        dst.first_try_successes += rhs.first_try_successes;
        dst.lp_failures += rhs.lp_failures;
        dst.retry_attempts += rhs.retry_attempts;
        dst.basis_offered += rhs.basis_offered;
        dst.basis_consistency_failures += rhs.basis_consistency_failures;
        dst.simplex_iterations += rhs.simplex_iterations;
        dst.solve_time_ms += rhs.solve_time_ms;
        dst.load_model_count += rhs.load_model_count;
        dst.load_model_time_ms += rhs.load_model_time_ms;
        dst.set_bounds_time_ms += rhs.set_bounds_time_ms;
        dst.basis_set_time_ms += rhs.basis_set_time_ms;
        ensure_histogram_capacity(&mut dst.retry_level_histogram, &rhs.retry_level_histogram);
        for (d, s) in dst
            .retry_level_histogram
            .iter_mut()
            .zip(&rhs.retry_level_histogram)
        {
            *d += s;
        }
    }

    /// Copy `self` into `dst`, reusing `dst`'s `retry_level_histogram`
    /// allocation. Prefer over `dst = self.clone()` on hot paths where the
    /// histogram length is stable (no heap allocation then).
    pub fn clone_into_reuse(&self, dst: &mut Self) {
        dst.lp_solves = self.lp_solves;
        dst.lp_successes = self.lp_successes;
        dst.first_try_successes = self.first_try_successes;
        dst.lp_failures = self.lp_failures;
        dst.retry_attempts = self.retry_attempts;
        dst.basis_offered = self.basis_offered;
        dst.basis_consistency_failures = self.basis_consistency_failures;
        dst.simplex_iterations = self.simplex_iterations;
        dst.solve_time_ms = self.solve_time_ms;
        dst.load_model_count = self.load_model_count;
        dst.load_model_time_ms = self.load_model_time_ms;
        dst.set_bounds_time_ms = self.set_bounds_time_ms;
        dst.basis_set_time_ms = self.basis_set_time_ms;
        let n = self.retry_level_histogram.len();
        dst.retry_level_histogram.resize(n, 0);
        dst.retry_level_histogram
            .copy_from_slice(&self.retry_level_histogram);
    }

    /// Zero all scalar fields and clear the histogram in place, retaining its
    /// capacity (unlike `*self = SolverStatsDelta::default()`).
    pub fn reset_in_place(&mut self) {
        self.lp_solves = 0;
        self.lp_successes = 0;
        self.first_try_successes = 0;
        self.lp_failures = 0;
        self.retry_attempts = 0;
        self.basis_offered = 0;
        self.basis_consistency_failures = 0;
        self.simplex_iterations = 0;
        self.solve_time_ms = 0.0;
        self.load_model_count = 0;
        self.load_model_time_ms = 0.0;
        self.set_bounds_time_ms = 0.0;
        self.basis_set_time_ms = 0.0;
        self.retry_level_histogram.clear();
    }

    /// Sum an iterator of deltas element-wise into a single aggregate.
    ///
    /// Returns `Default` (all zeros) for an empty iterator.
    #[must_use]
    pub fn aggregate<'a>(deltas: impl Iterator<Item = &'a Self>) -> Self {
        let mut result = Self::default();
        for d in deltas {
            Self::accumulate_into(&mut result, d);
        }
        result
    }
}

/// Sum solver counters across a workspace pool into one aggregate snapshot.
#[must_use]
pub fn aggregate_solver_statistics(
    stats: impl Iterator<Item = SolverStatistics>,
) -> SolverStatistics {
    let mut result = SolverStatistics::default();
    for s in stats {
        result.solve_count += s.solve_count;
        result.success_count += s.success_count;
        result.failure_count += s.failure_count;
        result.total_iterations += s.total_iterations;
        result.retry_count += s.retry_count;
        result.total_solve_time_seconds += s.total_solve_time_seconds;
        result.basis_consistency_failures += s.basis_consistency_failures;
        result.first_try_successes += s.first_try_successes;
        result.basis_offered += s.basis_offered;
        result.load_model_count += s.load_model_count;
        result.total_load_model_time_seconds += s.total_load_model_time_seconds;
        result.total_set_bounds_time_seconds += s.total_set_bounds_time_seconds;
        result.total_basis_set_time_seconds += s.total_basis_set_time_seconds;
        result.basis_reconstructions += s.basis_reconstructions;
        ensure_histogram_capacity(&mut result.retry_level_histogram, &s.retry_level_histogram);
        for (dst, src) in result
            .retry_level_histogram
            .iter_mut()
            .zip(&s.retry_level_histogram)
        {
            *dst += src;
        }
    }
    result
}

/// A single row in the per-iteration, per-phase solver-stats log.
///
/// The `opening`/`worker_id` `-1 → None` sentinel decode is owned by
/// [`SolverStatsLogEntry::from_raw`] — the sole construction path on the hot
/// push sites — so no consumer re-decodes a sentinel. `stage_id` carries no
/// sentinel: it is a nullable domain id (a domain stage id may be negative), so
/// the "no stage" case is `None`, never `-1`.
#[derive(Debug, Clone)]
pub struct SolverStatsLogEntry {
    /// 1-based training iteration number.
    pub iteration: u64,
    /// `"forward"`, `"backward"`, or `"lower_bound"`. `&'static str` to avoid
    /// per-entry heap allocation on the hot push path.
    pub phase: &'static str,
    /// Declared study `stage_id` (domain id) for forward/backward rows; `None`
    /// for the lower-bound phase (no per-stage attribution).
    pub stage_id: Option<i32>,
    /// Opening index `Some(ω)` for backward rows; `None` otherwise (writer → NULL).
    pub opening: Option<i32>,
    /// MPI rank that produced this row; never a sentinel (writer wraps in `Some`).
    pub rank: i32,
    /// rayon worker for backward rows (from the `allgatherv` unpack); `None`
    /// otherwise (writer → NULL).
    pub worker_id: Option<i32>,
    /// Solver-counter delta for this entry.
    pub delta: SolverStatsDelta,
}

impl SolverStatsLogEntry {
    /// Build an entry from raw producer values. The `-1` sentinel on `opening`/`worker_id`
    /// decodes to `None`; `stage_id` passes through; `rank` is verbatim.
    #[must_use]
    pub fn from_raw(
        iteration: u64,
        phase: &'static str,
        stage_id: Option<i32>,
        opening: i32,
        rank: i32,
        worker_id: i32,
        delta: SolverStatsDelta,
    ) -> Self {
        Self {
            iteration,
            phase,
            stage_id,
            opening: (opening != -1).then_some(opening),
            rank,
            worker_id: (worker_id != -1).then_some(worker_id),
            delta,
        }
    }
}

/// Convert a [`SolverStatsDelta`] into a [`SolverStatsRow`] for Parquet output.
///
/// A training row fills `iteration` (leaving `scenario_id = None`); a simulation
/// row fills `scenario_id` (leaving `iteration = None`). `stage_id` is `None` for
/// the lower-bound and simulation phases that carry no per-stage attribution.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn delta_to_stats_row(
    iteration: Option<i32>,
    scenario_id: Option<i32>,
    phase: &'static str,
    stage_id: Option<i32>,
    opening_index: Option<i32>,
    rank: Option<i32>,
    worker_id: Option<i32>,
    delta: &SolverStatsDelta,
) -> SolverStatsRow {
    SolverStatsRow {
        iteration,
        scenario_id,
        phase,
        stage_id,
        opening_index,
        rank,
        worker_id,
        lp_solves: delta.lp_solves as u32,
        lp_successes: delta.lp_successes as u32,
        lp_retries: delta.lp_successes.saturating_sub(delta.first_try_successes) as u32,
        lp_failures: delta.lp_failures as u32,
        retry_attempts: delta.retry_attempts as u32,
        basis_offered: delta.basis_offered as u32,
        basis_consistency_failures: delta.basis_consistency_failures as u32,
        simplex_iterations: delta.simplex_iterations,
        solve_time_ms: delta.solve_time_ms,
        load_model_time_ms: delta.load_model_time_ms,
        set_bounds_time_ms: delta.set_bounds_time_ms,
        basis_set_time_ms: delta.basis_set_time_ms,
        retry_level_histogram: delta.retry_level_histogram.clone(),
    }
}

/// Fold a solver-stats log into the five phase-derived training-summary
/// counts: `(first_try, retried, failed, forward_solve_seconds,
/// backward_solve_seconds)`. `rank_filter = Some(rank)` keeps only entries
/// from that rank (CLI backward-entry filter); `None` folds every entry.
///
/// `total_lp_solves` is deliberately NOT derived here — it stays sourced from
/// the per-iteration convergence records (`IterationRecord.lp_solves`), which
/// can diverge from this log's own per-entry `lp_solves` on a multi-stage
/// case. Pinned by `aggregate_solver_stats_log_ignores_lp_solves`.
#[must_use]
pub fn aggregate_solver_stats_log(
    entries: &[SolverStatsLogEntry],
    rank_filter: Option<u32>,
) -> (u64, u64, u64, f64, f64) {
    let rank_filter = rank_filter.map(|r| i32::try_from(r).unwrap_or(i32::MAX));
    let mut first_try = 0u64;
    let mut retried = 0u64;
    let mut failed = 0u64;
    let mut forward_solve_ms = 0.0_f64;
    let mut backward_solve_ms = 0.0_f64;
    for entry in entries {
        if rank_filter.is_some_and(|rank| entry.rank != rank) {
            continue;
        }
        let delta = &entry.delta;
        // lower_bound solve counts are included, but its solve_time_ms is
        // deliberately dropped by the `_ => {}` arm: no output field receives it.
        first_try += delta.first_try_successes;
        retried += delta.lp_successes.saturating_sub(delta.first_try_successes);
        failed += delta.lp_failures;
        match entry.phase {
            "forward" => forward_solve_ms += delta.solve_time_ms,
            "backward" => backward_solve_ms += delta.solve_time_ms,
            _ => {}
        }
    }
    (
        first_try,
        retried,
        failed,
        forward_solve_ms / 1000.0,
        backward_solve_ms / 1000.0,
    )
}

/// Map a per-iteration solver-stats log into Parquet rows.
#[must_use]
pub fn solver_stats_log_to_rows(log: &[SolverStatsLogEntry]) -> Vec<SolverStatsRow> {
    log.iter()
        .map(|entry| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let iteration = entry.iteration as i32;
            delta_to_stats_row(
                Some(iteration),
                None,
                entry.phase,
                entry.stage_id,
                entry.opening,
                Some(entry.rank),
                entry.worker_id,
                &entry.delta,
            )
        })
        .collect()
}

/// Number of scalar fields in [`SolverStatsDelta`] (excludes histogram `Vec`).
/// Buffer size for MPI allreduce/allgatherv: 8 `u64` fields cast to `f64` + 5 native `f64` fields.
pub const SOLVER_STATS_DELTA_SCALAR_FIELDS: usize = 13;

/// Packed `f64` stride per scenario: `scenario_id` + 13 scalar fields.
pub const SCENARIO_STATS_STRIDE: usize = 1 + SOLVER_STATS_DELTA_SCALAR_FIELDS;

/// Packed `f64` stride per entry: `worker_id`, `slot_idx`, + 13 scalar fields (total 15).
pub const WORKER_STATS_ENTRY_STRIDE: usize = 2 + SOLVER_STATS_DELTA_SCALAR_FIELDS;

/// Preallocate buffer size for per-worker per-slot MPI payloads.
#[must_use]
#[inline]
pub fn worker_opening_stats_buffer_size(n_workers: usize, n_slots: usize) -> usize {
    n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE
}

/// Pack the 13 scalar fields of a [`SolverStatsDelta`] into a fixed-size `f64` array.
/// Indices 0–7: eight `u64` fields cast to `f64`. Indices 8–12: five native `f64` fields.
/// The `retry_level_histogram` is excluded from MPI packing and Parquet output.
///
/// # Precision contract
///
/// `u64` casts are exact for event counts up to `2^53` (9,007,199,254,740,992). Beyond
/// that threshold `f64` cannot represent consecutive integers exactly, and an
/// `allreduce(Sum)` across MPI ranks would silently lose precision.
///
/// **The caller is responsible for enforcing the `2^53` ceiling before calling this
/// function.** This function does not validate counter values. For the CLI training
/// path the guard is `check_stats_overflow`, which runs before the pack and returns
/// `Err(CliError::Internal)` if any counter exceeds the limit.
///
/// # Timing precision
///
/// `solve_time_ms` (index 9) and the other `f64` timing fields are summed natively
/// without a cast. `f64` addition is non-associative, so rank-order changes across
/// MPI runs can cause ULP-level drift in `total_solve_time_seconds`. This is
/// acceptable: timing metrics require semantic parity (same order of magnitude),
/// not bit-for-bit reproducibility. Solve-count correctness is the strict invariant;
/// timing totals across ranks are informational.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pack_delta_scalars(delta: &SolverStatsDelta) -> [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] {
    [
        delta.lp_solves as f64,                  // index 0
        delta.lp_successes as f64,               // index 1
        delta.first_try_successes as f64,        // index 2
        delta.lp_failures as f64,                // index 3
        delta.retry_attempts as f64,             // index 4
        delta.basis_offered as f64,              // index 5
        delta.basis_consistency_failures as f64, // index 6
        delta.simplex_iterations as f64,         // index 7
        delta.load_model_count as f64,           // index 8
        delta.solve_time_ms,                     // index 9
        delta.load_model_time_ms,                // index 10
        delta.set_bounds_time_ms,                // index 11
        delta.basis_set_time_ms,                 // index 12
    ]
}

/// Unpack a fixed-size `f64` array (from [`pack_delta_scalars`]) back into a [`SolverStatsDelta`].
/// The `retry_level_histogram` is excluded from MPI packing and reset to its default.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unpack_delta_scalars(buf: &[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS]) -> SolverStatsDelta {
    SolverStatsDelta {
        lp_solves: buf[0] as u64,
        lp_successes: buf[1] as u64,
        first_try_successes: buf[2] as u64,
        lp_failures: buf[3] as u64,
        retry_attempts: buf[4] as u64,
        basis_offered: buf[5] as u64,
        basis_consistency_failures: buf[6] as u64,
        simplex_iterations: buf[7] as u64,
        load_model_count: buf[8] as u64,
        solve_time_ms: buf[9],
        load_model_time_ms: buf[10],
        set_bounds_time_ms: buf[11],
        basis_set_time_ms: buf[12],
        retry_level_histogram: Vec::new(),
    }
}

/// Pack per-scenario `(scenario_id, delta)` pairs into a flat `f64` buffer for `allgatherv`.
/// Each scenario contributes [`SCENARIO_STATS_STRIDE`] values. Histogram is excluded.
#[must_use]
pub fn pack_scenario_stats(stats: &[(u32, SolverStatsDelta)]) -> Vec<f64> {
    let mut buf = Vec::with_capacity(stats.len() * SCENARIO_STATS_STRIDE);
    for (scenario_id, delta) in stats {
        buf.push(f64::from(*scenario_id));
        buf.extend_from_slice(&pack_delta_scalars(delta));
    }
    buf
}

/// Unpack a flat `f64` buffer (from [`pack_scenario_stats`]) back into a `Vec<(u32, SolverStatsDelta)>`.
/// Buffer length must be a multiple of [`SCENARIO_STATS_STRIDE`].
/// Panics in debug if length is not a multiple of `SCENARIO_STATS_STRIDE`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unpack_scenario_stats(buf: &[f64]) -> Vec<(u32, SolverStatsDelta)> {
    debug_assert_eq!(
        buf.len() % SCENARIO_STATS_STRIDE,
        0,
        "buffer length must be a multiple of SCENARIO_STATS_STRIDE"
    );
    buf.chunks_exact(SCENARIO_STATS_STRIDE)
        .map(|chunk| {
            let scenario_id = chunk[0] as u32;
            let arr = [
                chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7], chunk[8],
                chunk[9], chunk[10], chunk[11], chunk[12], chunk[13],
            ];
            (scenario_id, unpack_delta_scalars(&arr))
        })
        .collect()
}

/// Pack per-worker per-slot buffer into a flat `f64` buffer for `allgatherv`.
/// Fixed stride per entry: `[w as f64, k as f64, <SOLVER_STATS_DELTA_SCALAR_FIELDS scalar fields>]` in row-major order.
/// `out` length must be `n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE`.
/// `stats` length must be `n_workers * n_slots` (row-major: `stats[w * n_slots + k]`).
/// Panics in debug if sizes don't match.
#[allow(clippy::cast_precision_loss)]
pub fn pack_worker_opening_stats(
    out: &mut [f64],
    stats: &[SolverStatsDelta],
    n_workers: usize,
    n_slots: usize,
) {
    debug_assert_eq!(stats.len(), n_workers * n_slots);
    debug_assert_eq!(out.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
    for w in 0..n_workers {
        for k in 0..n_slots {
            let entry_base = (w * n_slots + k) * WORKER_STATS_ENTRY_STRIDE;
            out[entry_base] = w as f64;
            out[entry_base + 1] = k as f64;
            let scalars = pack_delta_scalars(&stats[w * n_slots + k]);
            out[entry_base + 2..entry_base + WORKER_STATS_ENTRY_STRIDE].copy_from_slice(&scalars);
        }
    }
}

/// Unpack a flat `f64` buffer (from [`pack_worker_opening_stats`]) into `out`.
/// `buf` must be `n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE` floats.
/// `out` must be a slice of length `n_workers * n_slots` (row-major order; contents overwritten).
/// The prefix `[worker_id, slot_idx]` per entry is informational (not asserted on unpack for ranks > 0).
///
/// # Panics
///
/// Panics (in debug builds) if buffer lengths don't match the expected sizes.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used
)]
pub fn unpack_worker_opening_stats(
    buf: &[f64],
    out: &mut [SolverStatsDelta],
    n_workers: usize,
    n_slots: usize,
) {
    debug_assert_eq!(buf.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
    debug_assert_eq!(out.len(), n_workers * n_slots);
    for w in 0..n_workers {
        for k in 0..n_slots {
            let entry_base = (w * n_slots + k) * WORKER_STATS_ENTRY_STRIDE;
            // Do not assert the prefix worker_id/slot_idx against w/k: each rank
            // writes LOCAL indices, so the combined flat w != local worker_id for
            // ranks > 0. Prefix is informational.
            let scalars: [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] = buf
                [entry_base + 2..entry_base + WORKER_STATS_ENTRY_STRIDE]
                .try_into()
                .expect("slice length equals SOLVER_STATS_DELTA_SCALAR_FIELDS");
            out[w * n_slots + k] = unpack_delta_scalars(&scalars);
        }
    }
}

/// Flat per-(worker, slot) gather buffer for a single training pass.
///
/// Shape `n_workers × n_slots`, indexed by `worker_id * n_slots + slot`. `n_slots`
/// is `max_openings` (backward), `n_stages` (forward), or `1` (lower-bound).
/// Allocated once at `train` and reused; only `new` allocates, so no other method
/// touches the heap (the no-hot-path-allocation rule).
///
/// Call [`Self::reset`] before the parallel region — per backward stage, or per
/// training iteration for the forward/lower-bound buffers.
#[derive(Debug)]
pub struct StageWorkerStatsBuffer {
    data: Vec<SolverStatsDelta>,
    n_workers: usize,
    n_slots: usize,
}

impl StageWorkerStatsBuffer {
    /// Allocate a new buffer of shape `n_workers × n_slots`, initialised to
    /// `SolverStatsDelta::default()` (all zeros).
    #[must_use]
    pub fn new(n_workers: usize, n_slots: usize) -> Self {
        Self {
            data: vec![SolverStatsDelta::default(); n_workers * n_slots],
            n_workers,
            n_slots,
        }
    }

    /// Flat index `worker_id * n_slots + slot`; `pub` so callers can do bulk
    /// slice arithmetic without re-deriving it.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `worker_id >= self.n_workers` or `slot >= self.n_slots`.
    #[inline]
    #[must_use]
    pub fn index(&self, worker_id: usize, slot: usize) -> usize {
        debug_assert!(
            worker_id < self.n_workers,
            "worker_id {worker_id} >= n_workers {}",
            self.n_workers
        );
        debug_assert!(
            slot < self.n_slots,
            "slot {slot} >= n_slots {}",
            self.n_slots
        );
        worker_id * self.n_slots + slot
    }

    /// Return a shared reference to the delta at `(worker_id, slot)`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn get(&self, worker_id: usize, slot: usize) -> &SolverStatsDelta {
        let idx = self.index(worker_id, slot);
        &self.data[idx]
    }

    /// Write `delta` into the slot at `(worker_id, slot)`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, worker_id: usize, slot: usize, delta: SolverStatsDelta) {
        let idx = self.index(worker_id, slot);
        self.data[idx] = delta;
    }

    /// Zero all slots without re-allocating (cadence on the type doc).
    pub fn reset(&mut self) {
        for slot in &mut self.data {
            slot.reset_in_place();
        }
    }

    /// Return the flat backing slice of length `n_workers * n_slots`.
    #[must_use]
    pub fn as_slice(&self) -> &[SolverStatsDelta] {
        &self.data
    }

    /// Number of workers in the buffer (first dimension).
    #[must_use]
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Number of slots per worker (second dimension); see the type doc for its
    /// per-pass meaning.
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn test_from_snapshots_all_deltas() {
        let before = SolverStatistics {
            solve_count: 10,
            success_count: 9,
            failure_count: 1,
            total_iterations: 500,
            retry_count: 3,
            total_solve_time_seconds: 2.0,
            basis_consistency_failures: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            total_load_model_time_seconds: 1.0,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.1,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };
        let after = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_consistency_failures: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            total_load_model_time_seconds: 3.0,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.3,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };

        let delta = SolverStatsDelta::from_snapshots(&before, &after);
        assert_eq!(delta.lp_solves, 10);
        assert_eq!(delta.lp_successes, 9);
        assert_eq!(delta.first_try_successes, 8);
        assert_eq!(delta.lp_failures, 1);
        assert_eq!(delta.retry_attempts, 2);
        assert_eq!(delta.basis_offered, 9);
        assert_eq!(delta.basis_consistency_failures, 2);
        assert_eq!(delta.simplex_iterations, 600);
        assert!((delta.solve_time_ms - 2500.0).abs() < 1e-6);
        assert_eq!(delta.load_model_count, 7);
        assert!((delta.load_model_time_ms - 2000.0).abs() < 1e-6);
        assert!((delta.set_bounds_time_ms - 500.0).abs() < 1e-6);
        assert!((delta.basis_set_time_ms - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_snapshots_zero_delta() {
        let snap = SolverStatistics {
            solve_count: 5,
            success_count: 5,
            failure_count: 0,
            total_iterations: 200,
            retry_count: 0,
            total_solve_time_seconds: 1.0,
            basis_consistency_failures: 0,
            first_try_successes: 5,
            basis_offered: 3,
            load_model_count: 3,
            total_load_model_time_seconds: 0.1,
            total_set_bounds_time_seconds: 0.02,
            total_basis_set_time_seconds: 0.01,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        };
        let delta = SolverStatsDelta::from_snapshots(&snap, &snap);
        assert_eq!(delta.lp_solves, 0);
        assert_eq!(delta.lp_successes, 0);
        assert_eq!(delta.first_try_successes, 0);
        assert_eq!(delta.lp_failures, 0);
        assert_eq!(delta.retry_attempts, 0);
        assert_eq!(delta.basis_offered, 0);
        assert_eq!(delta.basis_consistency_failures, 0);
        assert_eq!(delta.simplex_iterations, 0);
        assert!((delta.solve_time_ms).abs() < 1e-10);
        assert!((delta.load_model_time_ms).abs() < 1e-10);
        assert!((delta.set_bounds_time_ms).abs() < 1e-10);
        assert!((delta.basis_set_time_ms).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_empty_returns_default() {
        let agg = SolverStatsDelta::aggregate(std::iter::empty());
        assert_eq!(agg.lp_solves, 0);
        assert_eq!(agg.solve_time_ms, 0.0);
    }

    #[test]
    fn test_aggregate_sums_all_fields() {
        let d1 = SolverStatsDelta {
            lp_solves: 10,
            lp_successes: 9,
            first_try_successes: 8,
            lp_failures: 1,
            retry_attempts: 2,
            basis_offered: 7,
            basis_consistency_failures: 1,
            simplex_iterations: 500,
            solve_time_ms: 100.0,
            load_model_count: 5,
            load_model_time_ms: 10.0,
            set_bounds_time_ms: 2.0,
            basis_set_time_ms: 1.0,
            retry_level_histogram: vec![0; 12],
        };
        let d2 = SolverStatsDelta {
            lp_solves: 20,
            lp_successes: 19,
            first_try_successes: 17,
            lp_failures: 1,
            retry_attempts: 3,
            basis_offered: 15,
            basis_consistency_failures: 2,
            simplex_iterations: 800,
            solve_time_ms: 200.0,
            load_model_count: 10,
            load_model_time_ms: 20.0,
            set_bounds_time_ms: 4.0,
            basis_set_time_ms: 2.0,
            retry_level_histogram: vec![0; 12],
        };

        let agg = SolverStatsDelta::aggregate([d1, d2].iter());
        assert_eq!(agg.lp_solves, 30);
        assert_eq!(agg.lp_successes, 28);
        assert_eq!(agg.first_try_successes, 25);
        assert_eq!(agg.lp_failures, 2);
        assert_eq!(agg.retry_attempts, 5);
        assert_eq!(agg.basis_offered, 22);
        assert_eq!(agg.basis_consistency_failures, 3);
        assert_eq!(agg.simplex_iterations, 1300);
        assert!((agg.solve_time_ms - 300.0).abs() < 1e-6);
        assert_eq!(agg.load_model_count, 15);
        assert!((agg.load_model_time_ms - 30.0).abs() < 1e-6);
        assert!((agg.set_bounds_time_ms - 6.0).abs() < 1e-6);
        assert!((agg.basis_set_time_ms - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_solver_statistics_sums_all_fields() {
        let s1 = SolverStatistics {
            solve_count: 10,
            success_count: 9,
            failure_count: 1,
            total_iterations: 500,
            retry_count: 3,
            total_solve_time_seconds: 2.0,
            basis_consistency_failures: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 5,
            total_load_model_time_seconds: 1.0,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.05,
            basis_reconstructions: 4,
            retry_level_histogram: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let s2 = SolverStatistics {
            solve_count: 20,
            success_count: 18,
            failure_count: 2,
            total_iterations: 1100,
            retry_count: 5,
            total_solve_time_seconds: 4.5,
            basis_consistency_failures: 3,
            first_try_successes: 15,
            basis_offered: 17,
            load_model_count: 12,
            total_load_model_time_seconds: 3.0,
            total_set_bounds_time_seconds: 0.75,
            total_basis_set_time_seconds: 0.15,
            basis_reconstructions: 10,
            retry_level_histogram: vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        let agg = aggregate_solver_statistics([s1, s2].into_iter());
        assert_eq!(agg.solve_count, 30);
        assert_eq!(agg.success_count, 27);
        assert_eq!(agg.failure_count, 3);
        assert_eq!(agg.total_iterations, 1600);
        assert_eq!(agg.retry_count, 8);
        assert!((agg.total_solve_time_seconds - 6.5).abs() < 1e-10);
        assert_eq!(agg.basis_consistency_failures, 4);
        assert_eq!(agg.first_try_successes, 22);
        assert_eq!(agg.basis_offered, 25);
        assert_eq!(agg.load_model_count, 17);
        assert!((agg.total_load_model_time_seconds - 4.0).abs() < 1e-10);
        assert!((agg.total_set_bounds_time_seconds - 1.0).abs() < 1e-10);
        assert!((agg.total_basis_set_time_seconds - 0.2).abs() < 1e-10);
        assert_eq!(agg.retry_level_histogram[0], 1);
        assert_eq!(agg.retry_level_histogram[1], 2);
        assert_eq!(agg.retry_level_histogram[2], 0);
        assert_eq!(agg.basis_reconstructions, 14);
    }

    fn make_delta(lp_solves: u64) -> SolverStatsDelta {
        SolverStatsDelta {
            lp_solves,
            lp_successes: lp_solves,
            first_try_successes: lp_solves / 2,
            lp_failures: 0,
            retry_attempts: 1,
            basis_offered: lp_solves,
            basis_consistency_failures: 2,
            simplex_iterations: lp_solves * 10,
            solve_time_ms: lp_solves as f64 * 0.5,
            load_model_count: 3,
            load_model_time_ms: 1.5,
            set_bounds_time_ms: 0.25,
            basis_set_time_ms: 0.125,
            retry_level_histogram: vec![0; 12],
        }
    }

    #[test]
    fn test_pack_unpack_delta_scalars_round_trip() {
        let delta = make_delta(600);
        let packed = pack_delta_scalars(&delta);
        assert_eq!(packed.len(), SOLVER_STATS_DELTA_SCALAR_FIELDS);
        let unpacked = unpack_delta_scalars(&packed);

        assert_eq!(unpacked.lp_solves, delta.lp_solves);
        assert_eq!(unpacked.lp_successes, delta.lp_successes);
        assert_eq!(unpacked.first_try_successes, delta.first_try_successes);
        assert_eq!(unpacked.lp_failures, delta.lp_failures);
        assert_eq!(unpacked.retry_attempts, delta.retry_attempts);
        assert_eq!(unpacked.basis_offered, delta.basis_offered);
        assert_eq!(
            unpacked.basis_consistency_failures,
            delta.basis_consistency_failures
        );
        assert_eq!(unpacked.simplex_iterations, delta.simplex_iterations);
        assert_eq!(unpacked.load_model_count, delta.load_model_count);
        assert!((unpacked.solve_time_ms - delta.solve_time_ms).abs() < 1e-10);
        assert!((unpacked.load_model_time_ms - delta.load_model_time_ms).abs() < 1e-10);
        assert!((unpacked.set_bounds_time_ms - delta.set_bounds_time_ms).abs() < 1e-10);
        assert!((unpacked.basis_set_time_ms - delta.basis_set_time_ms).abs() < 1e-10);
        // histogram is excluded from pack/unpack
        assert!(unpacked.retry_level_histogram.is_empty());
    }

    #[test]
    fn test_pack_unpack_delta_scalars_identity_for_lp_solves_600() {
        // Acceptance criterion: identity property for allreduce with LocalBackend.
        let delta = make_delta(600);
        let packed = pack_delta_scalars(&delta);
        let unpacked = unpack_delta_scalars(&packed);
        assert_eq!(unpacked.lp_solves, 600);
    }

    #[test]
    fn test_pack_unpack_scenario_stats_round_trip_three_entries() {
        // Acceptance criterion: pack/unpack round-trip for Vec<(u32, SolverStatsDelta)>
        // with scenario IDs 7, 12, 25.
        let stats = vec![
            (7u32, make_delta(100)),
            (12u32, make_delta(200)),
            (25u32, make_delta(300)),
        ];
        let buf = pack_scenario_stats(&stats);
        assert_eq!(buf.len(), 3 * SCENARIO_STATS_STRIDE);

        let unpacked = unpack_scenario_stats(&buf);
        assert_eq!(unpacked.len(), 3);

        // Verify scenario IDs
        assert_eq!(unpacked[0].0, 7);
        assert_eq!(unpacked[1].0, 12);
        assert_eq!(unpacked[2].0, 25);

        // Verify field values for each scenario
        assert_eq!(unpacked[0].1.lp_solves, 100);
        assert_eq!(unpacked[1].1.lp_solves, 200);
        assert_eq!(unpacked[2].1.lp_solves, 300);

        assert!((unpacked[0].1.solve_time_ms - 50.0).abs() < 1e-10);
        assert!((unpacked[1].1.solve_time_ms - 100.0).abs() < 1e-10);
        assert!((unpacked[2].1.solve_time_ms - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_pack_scenario_stats_empty_round_trip() {
        let buf = pack_scenario_stats(&[]);
        assert!(buf.is_empty());
        let unpacked = unpack_scenario_stats(&buf);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_pack_delta_scalars_field_count() {
        // Regression: pack_delta_scalars must produce a 13-element array.
        let delta = SolverStatsDelta::default();
        let packed = pack_delta_scalars(&delta);
        assert_eq!(
            packed.len(),
            13,
            "pack_delta_scalars must return 13 elements"
        );
        assert_eq!(packed.len(), SOLVER_STATS_DELTA_SCALAR_FIELDS);
        assert_eq!(SOLVER_STATS_DELTA_SCALAR_FIELDS, 13);
    }

    #[test]
    fn test_accumulate_into_all_fields() {
        // AC-001: accumulate_into sums every scalar field and extends the histogram.
        let mut dst = make_delta(10);
        dst.retry_level_histogram = vec![1, 0, 2, 0];
        let rhs = make_delta(5);
        let mut rhs_full = rhs.clone();
        rhs_full.retry_level_histogram = vec![0, 3, 0, 1];

        SolverStatsDelta::accumulate_into(&mut dst, &rhs_full);

        assert_eq!(dst.lp_solves, 15);
        assert_eq!(dst.lp_successes, 15);
        assert_eq!(dst.first_try_successes, 7); // 5 + 2
        assert_eq!(dst.lp_failures, 0);
        assert_eq!(dst.retry_attempts, 2); // 1 + 1
        assert_eq!(dst.basis_offered, 15);
        assert_eq!(dst.basis_consistency_failures, 4); // 2 + 2
        assert_eq!(dst.simplex_iterations, 150); // 100 + 50
        assert!((dst.solve_time_ms - 7.5).abs() < 1e-10); // 5.0 + 2.5
        assert_eq!(dst.load_model_count, 6); // 3 + 3
        assert!((dst.load_model_time_ms - 3.0).abs() < 1e-10);
        assert!((dst.set_bounds_time_ms - 0.5).abs() < 1e-10);
        assert!((dst.basis_set_time_ms - 0.25).abs() < 1e-10);
        // Histogram: [1+0, 0+3, 2+0, 0+1]
        assert_eq!(dst.retry_level_histogram, vec![1, 3, 2, 1]);
    }

    #[test]
    fn test_solver_stats_log_per_opening_shape() {
        // Forward entries carry a real (domain) stage_id, opening == None,
        // worker_id == None (no per-worker dimension yet). LB entries use
        // stage_id == None, opening == None, worker_id == None. Backward entries
        // carry real (rank, worker_id) from allgatherv unpack.
        let fwd_entry =
            SolverStatsLogEntry::from_raw(1, "forward", Some(0), -1, 0, -1, make_delta(4));
        let bwd_entry_0 =
            SolverStatsLogEntry::from_raw(1, "backward", Some(2), 0, 0, 0, make_delta(2));
        let bwd_entry_1 =
            SolverStatsLogEntry::from_raw(1, "backward", Some(2), 1, 0, 1, make_delta(3));
        let lb_entry =
            SolverStatsLogEntry::from_raw(1, "lower_bound", None, -1, 0, -1, make_delta(1));

        let log: Vec<SolverStatsLogEntry> = vec![fwd_entry, bwd_entry_0, bwd_entry_1, lb_entry];

        // Verify the forward entry has a real stage_id, opening None, worker_id None.
        assert_eq!(log[0].phase, "forward");
        assert_eq!(
            log[0].stage_id,
            Some(0),
            "forward stage_id must be a real (domain) stage id"
        );
        assert_eq!(log[0].opening, None);
        assert_eq!(
            log[0].worker_id, None,
            "forward rows have no per-worker dimension"
        );

        // Verify backward entries carry correct opening indices and worker ids.
        assert_eq!(log[1].stage_id, Some(2));
        assert_eq!(log[1].opening, Some(0));
        assert_eq!(log[1].rank, 0);
        assert_eq!(log[1].worker_id, Some(0));
        assert_eq!(log[1].delta.lp_solves, 2);

        assert_eq!(log[2].stage_id, Some(2));
        assert_eq!(log[2].opening, Some(1));
        assert_eq!(log[2].rank, 0);
        assert_eq!(log[2].worker_id, Some(1));
        assert_eq!(log[2].delta.lp_solves, 3);

        // Verify that collapsing across openings/workers yields the per-stage total.
        let backward_entries: Vec<&SolverStatsDelta> = log
            .iter()
            .filter(|e| e.phase == "backward")
            .map(|e| &e.delta)
            .collect();
        let collapsed = SolverStatsDelta::aggregate(backward_entries.into_iter());
        assert_eq!(collapsed.lp_solves, 5); // 2 + 3

        // Verify that LB entry has stage_id None, opening None, worker_id None.
        assert_eq!(log[3].stage_id, None);
        assert_eq!(log[3].opening, None);
        assert_eq!(log[3].worker_id, None);
    }

    #[test]
    fn solver_stats_log_entry_from_raw_decodes_minus_one_to_none() {
        // The -1 sentinel on opening/worker_id decodes to None on the struct;
        // a non-negative index decodes to Some. stage_id is passed through and rank
        // is verbatim.
        let forward =
            SolverStatsLogEntry::from_raw(3, "forward", Some(1), -1, 0, -1, make_delta(5));
        assert_eq!(forward.opening, None);
        assert_eq!(forward.worker_id, None);

        let backward =
            SolverStatsLogEntry::from_raw(3, "backward", Some(2), 0, 1, 4, make_delta(2));
        assert_eq!(backward.opening, Some(0));
        assert_eq!(backward.worker_id, Some(4));
        assert_eq!(backward.rank, 1);
        assert_eq!(backward.stage_id, Some(2));
    }

    #[test]
    fn test_solver_stats_log_to_rows_decodes_minus_one_to_none() {
        // The -1 sentinel on opening/worker_id maps to None in the Parquet row;
        // a non-negative index maps to Some. rank is always wrapped in Some.
        let backward =
            SolverStatsLogEntry::from_raw(3, "backward", Some(2), 0, 1, 4, make_delta(2));
        let forward =
            SolverStatsLogEntry::from_raw(3, "forward", Some(1), -1, 1, -1, make_delta(5));
        let log: Vec<SolverStatsLogEntry> = vec![backward, forward];

        let rows = solver_stats_log_to_rows(&log);
        assert_eq!(rows.len(), 2);

        // Backward: non-negative opening/worker_id → Some; rank → Some. A training
        // row fills iteration and leaves scenario_id None.
        assert_eq!(rows[0].opening_index, Some(0));
        assert_eq!(rows[0].worker_id, Some(4));
        assert_eq!(rows[0].rank, Some(1));
        assert_eq!(rows[0].iteration, Some(3));
        assert_eq!(rows[0].scenario_id, None);
        assert_eq!(rows[0].stage_id, Some(2));
        assert_eq!(rows[0].phase, "backward");
        assert_eq!(rows[0].lp_solves, 2);

        // Forward: opening == -1 → None; worker_id == -1 → None; rank → Some.
        assert_eq!(rows[1].opening_index, None);
        assert_eq!(rows[1].worker_id, None);
        assert_eq!(rows[1].rank, Some(1));
        assert_eq!(rows[1].phase, "forward");
        assert_eq!(rows[1].lp_solves, 5);
    }

    #[test]
    fn aggregate_solver_stats_log_folds_five_phase_counts() {
        let forward_delta = SolverStatsDelta {
            lp_solves: 10,
            first_try_successes: 7,
            lp_successes: 9,
            lp_failures: 1,
            solve_time_ms: 2500.0,
            ..SolverStatsDelta::default()
        };
        let backward_delta = SolverStatsDelta {
            lp_solves: 4,
            first_try_successes: 2,
            lp_successes: 4,
            lp_failures: 0,
            solve_time_ms: 1500.0,
            ..SolverStatsDelta::default()
        };
        let log = vec![
            SolverStatsLogEntry::from_raw(0, "forward", Some(0), -1, 0, -1, forward_delta),
            SolverStatsLogEntry::from_raw(0, "backward", Some(0), 0, 0, 0, backward_delta),
        ];

        let (first_try, retried, failed, forward_seconds, backward_seconds) =
            aggregate_solver_stats_log(&log, None);

        assert_eq!(first_try, 9); // 7 + 2
        assert_eq!(retried, 4); // (9-7) + (4-2)
        assert_eq!(failed, 1);
        assert_eq!(forward_seconds, 2.5);
        assert_eq!(backward_seconds, 1.5);
    }

    #[test]
    fn aggregate_solver_stats_log_all_ranks_identity_on_one_rank_log() {
        // On a one-rank log, folding all ranks (None) must equal folding
        // filtered to that one rank (Some(rank)).
        let log = vec![
            SolverStatsLogEntry::from_raw(0, "forward", Some(0), -1, 0, -1, make_delta(6)),
            SolverStatsLogEntry::from_raw(0, "backward", Some(0), 0, 0, 0, make_delta(3)),
        ];

        assert_eq!(
            aggregate_solver_stats_log(&log, None),
            aggregate_solver_stats_log(&log, Some(0))
        );
    }

    #[test]
    fn aggregate_solver_stats_log_ignores_lp_solves() {
        // total_lp_solves is sourced from the convergence records, never from
        // this fold: varying delta.lp_solves alone must not change the result.
        let low = SolverStatsLogEntry::from_raw(
            0,
            "forward",
            Some(0),
            -1,
            0,
            -1,
            SolverStatsDelta {
                lp_solves: 1,
                first_try_successes: 5,
                lp_successes: 5,
                solve_time_ms: 100.0,
                ..SolverStatsDelta::default()
            },
        );
        let high = SolverStatsLogEntry::from_raw(
            0,
            "forward",
            Some(0),
            -1,
            0,
            -1,
            SolverStatsDelta {
                lp_solves: 999,
                first_try_successes: 5,
                lp_successes: 5,
                solve_time_ms: 100.0,
                ..SolverStatsDelta::default()
            },
        );

        assert_eq!(
            aggregate_solver_stats_log(&[low], None),
            aggregate_solver_stats_log(&[high], None)
        );
    }

    /// per-stage forward `stage_stats` summed element-wise across workers.
    ///
    /// Simulates 2 workers × 3 stages, verifying that the post-parallel reduction
    /// produces the correct element-wise sum without hot-path allocations.
    #[test]
    fn test_forward_stage_stats_summed_across_workers() {
        // Worker 0 processed some scenarios at each stage.
        let worker0: Vec<SolverStatsDelta> = vec![
            make_delta(10), // stage 0: 10 lp_solves
            make_delta(20), // stage 1: 20 lp_solves
            make_delta(30), // stage 2: 30 lp_solves
        ];

        // Worker 1 processed the remaining scenarios.
        let worker1: Vec<SolverStatsDelta> = vec![
            make_delta(5),  // stage 0: 5 lp_solves
            make_delta(15), // stage 1: 15 lp_solves
            make_delta(25), // stage 2: 25 lp_solves
        ];

        // Simulate the post-parallel reduction (mirrors run_forward_pass merge code).
        let n_stages = 3;
        let mut stage_stats: Vec<SolverStatsDelta> =
            (0..n_stages).map(|_| SolverStatsDelta::default()).collect();

        for worker_stage_stats in [&worker0, &worker1] {
            for (dst, src) in stage_stats.iter_mut().zip(worker_stage_stats) {
                SolverStatsDelta::accumulate_into(dst, src);
            }
        }

        // Element-wise sum must equal worker0[t] + worker1[t] for each stage.
        assert_eq!(
            stage_stats[0].lp_solves, 15,
            "stage 0: 10 + 5 = 15 lp_solves"
        );
        assert_eq!(
            stage_stats[1].lp_solves, 35,
            "stage 1: 20 + 15 = 35 lp_solves"
        );
        assert_eq!(
            stage_stats[2].lp_solves, 55,
            "stage 2: 30 + 25 = 55 lp_solves"
        );

        // Verify simplex_iterations also sum correctly (10× lp_solves in make_delta).
        assert_eq!(stage_stats[0].simplex_iterations, 150); // (10 + 5) * 10
        assert_eq!(stage_stats[1].simplex_iterations, 350); // (20 + 15) * 10
        assert_eq!(stage_stats[2].simplex_iterations, 550); // (30 + 25) * 10

        // Verify the log shape: one SolverStatsLogEntry per stage with stage
        // index 0..3. Forward rows use a real stage index, rank = local rank,
        // and the opening/worker_id sentinels (-1) decode to None.
        let log: Vec<SolverStatsLogEntry> = stage_stats
            .iter()
            .enumerate()
            .map(|(t, delta)| {
                SolverStatsLogEntry::from_raw(
                    1,
                    "forward",
                    Some(i32::try_from(t).expect("stage fits i32")),
                    -1,
                    0,  // rank
                    -1, // worker_id sentinel → None
                    delta.clone(),
                )
            })
            .collect();

        assert_eq!(log.len(), 3, "one entry per stage");
        for (t, entry) in log.iter().enumerate() {
            assert_eq!(entry.phase, "forward");
            assert_eq!(
                entry.stage_id,
                Some(i32::try_from(t).expect("stage fits i32")),
                "stage_id must match loop variable"
            );
            assert_eq!(
                entry.opening, None,
                "forward rows have no opening dimension"
            );
            assert_eq!(
                entry.worker_id, None,
                "forward rows have no per-worker dimension"
            );
        }
    }

    /// Verify `index(w, k) = w * n_slots + k` for several values.
    #[test]
    fn test_stage_worker_stats_buffer_index_layout() {
        let buf = StageWorkerStatsBuffer::new(3, 4);
        assert_eq!(buf.index(0, 0), 0);
        assert_eq!(buf.index(0, 3), 3);
        assert_eq!(buf.index(1, 0), 4);
        assert_eq!(buf.index(2, 3), 11);
        assert_eq!(buf.as_slice().len(), 12);
        assert_eq!(buf.n_workers(), 3);
        assert_eq!(buf.n_slots(), 4);
    }

    /// Verify `reset()` zeros every slot, even after non-default writes.
    #[test]
    fn test_stage_worker_stats_buffer_reset_zeroes_all_slots() {
        let mut buf = StageWorkerStatsBuffer::new(2, 3);
        for w in 0..2 {
            for k in 0..3 {
                let d = SolverStatsDelta {
                    lp_solves: 7,
                    ..SolverStatsDelta::default()
                };
                buf.set(w, k, d);
            }
        }
        for slot in buf.as_slice() {
            assert_eq!(slot.lp_solves, 7);
        }
        buf.reset();
        for slot in buf.as_slice() {
            assert_eq!(slot.lp_solves, 0);
        }
    }

    /// Round-trip pack→unpack must preserve every scalar field for every (w,k) pair.
    #[test]
    fn test_pack_worker_opening_stats_roundtrip() {
        let n_workers = 3;
        let n_slots = 4;
        let mut input: Vec<SolverStatsDelta> = Vec::with_capacity(n_workers * n_slots);
        for w in 0..n_workers {
            for k in 0..n_slots {
                input.push(SolverStatsDelta {
                    lp_solves: (w * 10 + k) as u64,
                    ..SolverStatsDelta::default()
                });
            }
        }
        let mut buf = vec![0.0_f64; worker_opening_stats_buffer_size(n_workers, n_slots)];
        assert_eq!(buf.len(), n_workers * n_slots * WORKER_STATS_ENTRY_STRIDE);
        pack_worker_opening_stats(&mut buf, &input, n_workers, n_slots);

        let mut recovered = vec![SolverStatsDelta::default(); n_workers * n_slots];
        unpack_worker_opening_stats(&buf, &mut recovered, n_workers, n_slots);

        for w in 0..n_workers {
            for k in 0..n_slots {
                assert_eq!(
                    recovered[w * n_slots + k].lp_solves,
                    (w * 10 + k) as u64,
                    "lp_solves mismatch at (w={w}, k={k})"
                );
            }
        }
    }

    /// Helper returns the precise `stride * n_workers * n_slots` size in `f64` units.
    #[test]
    fn test_pack_worker_opening_stats_buffer_size() {
        assert_eq!(worker_opening_stats_buffer_size(10, 20), 10 * 20 * 15);
        assert_eq!(worker_opening_stats_buffer_size(10, 20), 3000);
    }

    /// Layout invariants — `[w as f64, k as f64, ...]` per entry, row-major.
    #[test]
    fn test_pack_worker_opening_stats_layout_invariant() {
        let n_workers = 2;
        let n_slots = 4;
        let input = vec![SolverStatsDelta::default(); n_workers * n_slots];
        let mut buf = vec![0.0_f64; worker_opening_stats_buffer_size(n_workers, n_slots)];
        pack_worker_opening_stats(&mut buf, &input, n_workers, n_slots);

        // entry(w=0, k=0) at offset 0
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 0.0);
        // entry(w=0, k=1) at offset WORKER_STATS_ENTRY_STRIDE (= 15)
        assert_eq!(buf[WORKER_STATS_ENTRY_STRIDE], 0.0);
        assert_eq!(buf[WORKER_STATS_ENTRY_STRIDE + 1], 1.0);
        // entry(w=1, k=0) at offset WORKER_STATS_ENTRY_STRIDE * n_slots (= 15 * 4)
        let w1_k0 = WORKER_STATS_ENTRY_STRIDE * n_slots;
        assert_eq!(buf[w1_k0], 1.0);
        assert_eq!(buf[w1_k0 + 1], 0.0);
    }

    /// MPI wire-format pin: `SolverStatsDelta` pack/unpack uses a 13-element `f64` array.
    /// Uses distinct nonzero values for every field so that field-order swaps are caught
    /// at both pack and unpack.
    #[test]
    fn test_solver_stats_delta_mpi_wire_format_13_fields() {
        // Wire-format constant assertions.
        assert_eq!(SOLVER_STATS_DELTA_SCALAR_FIELDS, 13);
        assert_eq!(SCENARIO_STATS_STRIDE, 14);
        assert_eq!(WORKER_STATS_ENTRY_STRIDE, 15);

        // Construct a populated delta. Use distinct nonzero values
        // to catch field-order swaps at pack/unpack.
        let delta = SolverStatsDelta {
            lp_solves: 1,
            lp_successes: 2,
            first_try_successes: 3,
            lp_failures: 4,
            retry_attempts: 5,
            basis_offered: 6,
            basis_consistency_failures: 7,
            simplex_iterations: 8,
            load_model_count: 9,
            solve_time_ms: 10.5,
            load_model_time_ms: 11.25,
            set_bounds_time_ms: 12.125,
            basis_set_time_ms: 13.0625,
            retry_level_histogram: vec![1, 2, 3], // application-level; not in wire
        };

        let packed = pack_delta_scalars(&delta);
        assert_eq!(packed.len(), 13);

        // Cross-check packed field ordering via explicit index reads.
        // Indices 0..=8 are u64 fields cast to f64; indices 9..=12 are f64.
        assert_eq!(packed[0], 1.0); // lp_solves
        assert_eq!(packed[8], 9.0); // load_model_count
        assert!((packed[9] - 10.5).abs() < 1e-10); // solve_time_ms
        assert!((packed[12] - 13.0625).abs() < 1e-10); // basis_set_time_ms

        let unpacked = unpack_delta_scalars(&packed);

        // Wire-carried fields must match.
        assert_eq!(unpacked.lp_solves, 1);
        assert_eq!(unpacked.lp_successes, 2);
        assert_eq!(unpacked.first_try_successes, 3);
        assert_eq!(unpacked.lp_failures, 4);
        assert_eq!(unpacked.retry_attempts, 5);
        assert_eq!(unpacked.basis_offered, 6);
        assert_eq!(unpacked.basis_consistency_failures, 7);
        assert_eq!(unpacked.simplex_iterations, 8);
        assert_eq!(unpacked.load_model_count, 9);
        assert!((unpacked.solve_time_ms - 10.5).abs() < 1e-10);
        assert!((unpacked.load_model_time_ms - 11.25).abs() < 1e-10);
        assert!((unpacked.set_bounds_time_ms - 12.125).abs() < 1e-10);
        assert!((unpacked.basis_set_time_ms - 13.0625).abs() < 1e-10);

        // The histogram is NOT in the wire — unpack leaves it empty.
        assert!(unpacked.retry_level_histogram.is_empty());
    }

    /// Pins the wire-format array length at the type level. The function signature
    /// is `unpack_delta_scalars(&[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS])`, so the
    /// compiler enforces the length. This test confirms the constant equals 13 and
    /// that the `size_of` the array is exactly `13 * 8` bytes.
    #[test]
    fn test_unpack_delta_scalars_array_length_is_compile_time() {
        assert_eq!(
            std::mem::size_of::<[f64; SOLVER_STATS_DELTA_SCALAR_FIELDS]>(),
            13 * std::mem::size_of::<f64>()
        );
        // Also confirm that a valid 13-element buffer round-trips without panic.
        let buf: [f64; 13] = [0.0; 13];
        let _ = unpack_delta_scalars(&buf);
    }

    /// `clone_into_reuse` must copy all scalar fields and resize+overwrite
    /// the histogram without reallocating when the destination already has
    /// sufficient capacity (tested via distinct source and destination
    /// histogram lengths).
    #[test]
    fn solver_stats_delta_clone_into_reuse_preserves_values() {
        let src = SolverStatsDelta {
            lp_solves: 42,
            lp_successes: 40,
            first_try_successes: 38,
            lp_failures: 2,
            retry_attempts: 5,
            basis_offered: 35,
            basis_consistency_failures: 3,
            simplex_iterations: 1200,
            solve_time_ms: 99.5,
            load_model_count: 10,
            load_model_time_ms: 7.25,
            set_bounds_time_ms: 1.5,
            basis_set_time_ms: 0.75,
            retry_level_histogram: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        // dst starts with a different histogram length to verify resize.
        let mut dst = SolverStatsDelta {
            lp_solves: 0,
            retry_level_histogram: vec![0; 5], // different length from src
            ..SolverStatsDelta::default()
        };

        src.clone_into_reuse(&mut dst);

        assert_eq!(dst.lp_solves, 42);
        assert_eq!(dst.lp_successes, 40);
        assert_eq!(dst.first_try_successes, 38);
        assert_eq!(dst.lp_failures, 2);
        assert_eq!(dst.retry_attempts, 5);
        assert_eq!(dst.basis_offered, 35);
        assert_eq!(dst.basis_consistency_failures, 3);
        assert_eq!(dst.simplex_iterations, 1200);
        assert!((dst.solve_time_ms - 99.5).abs() < 1e-10);
        assert_eq!(dst.load_model_count, 10);
        assert!((dst.load_model_time_ms - 7.25).abs() < 1e-10);
        assert!((dst.set_bounds_time_ms - 1.5).abs() < 1e-10);
        assert!((dst.basis_set_time_ms - 0.75).abs() < 1e-10);
        assert_eq!(
            dst.retry_level_histogram,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        // Verify reuse: clone_into_reuse onto a dst that already has the
        // right histogram length must produce the same result.
        let src2 = SolverStatsDelta {
            lp_solves: 7,
            retry_level_histogram: vec![10; 12],
            ..SolverStatsDelta::default()
        };
        src2.clone_into_reuse(&mut dst);
        assert_eq!(dst.lp_solves, 7);
        assert_eq!(dst.retry_level_histogram, vec![10; 12]);
    }

    /// `reset_in_place` must zero all scalar fields and clear the histogram,
    /// leaving the existing `Vec` capacity intact.
    #[test]
    fn solver_stats_delta_reset_in_place_zeroes_all_fields() {
        let mut d = SolverStatsDelta {
            lp_solves: 99,
            lp_successes: 88,
            first_try_successes: 77,
            lp_failures: 11,
            retry_attempts: 6,
            basis_offered: 50,
            basis_consistency_failures: 4,
            simplex_iterations: 500,
            solve_time_ms: 42.0,
            load_model_count: 8,
            load_model_time_ms: 3.0,
            set_bounds_time_ms: 1.0,
            basis_set_time_ms: 0.5,
            retry_level_histogram: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        d.reset_in_place();

        assert_eq!(d.lp_solves, 0);
        assert_eq!(d.lp_successes, 0);
        assert_eq!(d.first_try_successes, 0);
        assert_eq!(d.lp_failures, 0);
        assert_eq!(d.retry_attempts, 0);
        assert_eq!(d.basis_offered, 0);
        assert_eq!(d.basis_consistency_failures, 0);
        assert_eq!(d.simplex_iterations, 0);
        assert_eq!(d.solve_time_ms, 0.0);
        assert_eq!(d.load_model_count, 0);
        assert_eq!(d.load_model_time_ms, 0.0);
        assert_eq!(d.set_bounds_time_ms, 0.0);
        assert_eq!(d.basis_set_time_ms, 0.0);
        assert!(d.retry_level_histogram.is_empty());
    }
}
