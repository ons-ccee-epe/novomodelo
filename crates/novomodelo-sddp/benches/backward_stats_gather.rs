//! Criterion micro-benchmark for the per-stage `StageWorkerStatsBuffer` gather.
//!
//! Guards the zero-allocation property of the gather loop; target < 100µs per
//! stage at production sizing (`n_workers = 10`, `n_openings = 20`).

#![allow(missing_docs)]

use std::sync::mpsc;

use cobre_core::{TrainingEvent, WorkerPhaseTimings, WorkerTimingPhase};
use cobre_solver::{
    SolverInterface,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use cobre_sddp::solver_stats::{SolverStatsDelta, StageWorkerStatsBuffer};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct BenchProfile;

/// `SolverInterface` mock holding a fixed `SolverStatistics` with a non-empty
/// `retry_level_histogram`, to exercise the `statistics_into` copy path.
struct BenchStatsMockSolver {
    stats: SolverStatistics,
}

impl SolverInterface for BenchStatsMockSolver {
    type Profile = BenchProfile;

    fn apply_profile(&mut self, _profile: &BenchProfile) {}
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _rows: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        Err(SolverError::InternalError {
            message: "bench mock".to_string(),
            error_code: None,
        })
    }
    fn get_basis(&mut self, _out: &mut Basis) {}
    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }
    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.stats);
    }
    fn name(&self) -> &'static str {
        "BenchStatsMock"
    }
    fn solver_name_version(&self) -> String {
        "BenchStatsMockSolver 0.0.0".to_string()
    }
}

fn bench_gather(c: &mut Criterion) {
    let n_workers = 10;
    let n_openings = 20;
    let mut buf = StageWorkerStatsBuffer::new(n_workers, n_openings);
    let deltas: Vec<SolverStatsDelta> = (0..n_workers * n_openings)
        .map(|_| SolverStatsDelta::default())
        .collect();
    c.bench_function("backward_stats_gather 10x20", |b| {
        b.iter(|| {
            buf.reset();
            for w in 0..n_workers {
                for k in 0..n_openings {
                    buf.set(w, k, black_box(deltas[w * n_openings + k].clone()));
                }
            }
            black_box(buf.as_slice().len());
        });
    });
}

/// Per-worker `WorkerTiming` event construction + send micro-benchmark.
///
/// Guards that the **event construction** path introduces no `Vec`/`Box`
/// allocation (the `[f64; 16]` payload is stack-resident); `mpsc::Sender::send`
/// itself does allocate a node, so the channel-internal allocations are expected
/// and are not what this bench measures.
#[allow(clippy::expect_used)]
fn bench_worker_timing_emit(c: &mut Criterion) {
    let n_workers: i32 = 10;
    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    c.bench_function("worker_timing_emit 10_workers", |b| {
        b.iter(|| {
            for w in 0..n_workers {
                let timings = black_box(WorkerPhaseTimings::default());
                let event = TrainingEvent::WorkerTiming {
                    rank: 0,
                    worker_id: w,
                    iteration: 1,
                    phase: WorkerTimingPhase::Backward,
                    timings,
                };
                tx.send(black_box(event)).expect("channel open");
            }
        });
    });
    // Drain so the channel does not grow unboundedly across bench runs.
    drop(tx);
    while rx.try_recv().is_ok() {}
}

/// Per-opening `statistics_into` snapshot micro-benchmark.
///
/// Guards the steady-state zero-allocation property of the backward hot loop's
/// stats snapshot: `statistics_into` resizes + copies the histogram into a
/// reused buffer rather than cloning a fresh `Vec`, so once its capacity is
/// stable no further heap allocation occurs.
fn bench_statistics_snapshot(c: &mut Criterion) {
    let solver = BenchStatsMockSolver {
        stats: SolverStatistics {
            solve_count: 100,
            success_count: 95,
            failure_count: 5,
            total_iterations: 4096,
            retry_count: 7,
            total_solve_time_seconds: 2.5,
            basis_consistency_failures: 1,
            first_try_successes: 88,
            basis_offered: 60,
            load_model_count: 3,
            total_load_model_time_seconds: 0.4,
            total_set_bounds_time_seconds: 0.2,
            total_basis_set_time_seconds: 0.1,
            basis_reconstructions: 12,
            retry_level_histogram: vec![1, 0, 2, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        },
    };

    // Pre-warm histogram capacity so the timed loop is the steady-state path.
    let mut buf = SolverStatistics::default();
    solver.statistics_into(&mut buf);

    c.bench_function("backward_statistics_snapshot", |b| {
        b.iter(|| {
            solver.statistics_into(black_box(&mut buf));
            black_box(buf.solve_count);
        });
    });
}

criterion_group!(
    benches,
    bench_gather,
    bench_worker_timing_emit,
    bench_statistics_snapshot
);
criterion_main!(benches);
