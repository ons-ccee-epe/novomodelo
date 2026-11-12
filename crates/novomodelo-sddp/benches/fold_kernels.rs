//! Kernels the PAR fold adds to every solve, one group per measurement. Each group names the
//! decision it informs:
//!
//! - `rhs_kernel`: the plain row loop of the cut-row bracket (`rhs_reference`) against the batched
//!   form (`cut_brackets_into`) and `dgemm`, at one thread.
//! - `rhs_kernel_all_cores`: the same variants with every core running the whole pass over one
//!   shared pool, timed as the slowest thread's pass. It is the gate of the batched form: at
//!   least 1.3x faster than the plain loop with every core busy.
//! - `noise_term_pass`: the per-stage noise-term pass as two steps against the fused RHS form.
//! - `chain_rule`: one chain rule per opening against the aggregate-first form, warm and cold.
//!   Its per-row cost sizes the out-of-LP estimate.
//! - `chain_rule_walk`: the production row-major walk against a 256-wide j-tile nest, at the
//!   touched-slot p99 and at the disjoint worst case; run under `taskset` on a performance core
//!   and on an efficiency core.
//! - `y_bar_formation`: ȳ formed with the touched bitmap against a dense scan of the pool.
//! - `by_node_finish`: the serial per-position finish of the backward against a loop over
//!   positions on scoped threads.
//! - `solver_push`: the set-based row-bound push against the lower-bounds range push.
//!
//! Every group with more than one variant checks, before any timing, that the variants agree.
//! The bench-local variants keep the association of the kernels they replace: one accumulator
//! per row, ascending slots, no fused multiply-add.

#![allow(
    missing_docs,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::array;
use std::hint::black_box;
use std::ops::Range;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use chrono::NaiveDate;
use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use novomodelo_core::scenario::InflowModel;
use novomodelo_core::{
    Block, BlockMode, EntityId, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig,
};
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::cut::chain_rule::{
    AggregateScratch, ChainRuleInputs, ChainRuleScratch, ChildOpenings, OpeningDuals,
    aggregate_first_cut_into, per_opening_cut_into,
};
use novomodelo_sddp::cut::rhs::{
    cut_brackets_into, cut_noise_term, cut_rhs_from_noise_terms_into, cut_rhs_into,
};
use novomodelo_sddp::cut::rhs_reference::cut_brackets_reference_into;
use novomodelo_sddp::indexer::{ChainRuleSlots, CutLagPoint, CutSlot, CutStateProjection};
use novomodelo_sddp::setup::NodeId;
use novomodelo_sddp::test_support::state_layout;
use novomodelo_solver::{ActiveSolver, RowBatch, SolverInterface, StageTemplate};
use novomodelo_stochastic::par::precompute::PrecomputedPar;

const N_HYDROS: usize = 168;
const LAG_DEPTH: usize = 12;
const OMEGA: usize = 20;
const B_T: usize = 169;
const TILE: usize = 256;
const WARM_ROWS: usize = 1_000;
const COLD_ROWS: usize = 25_000;
const K_Y_BAR: usize = 5_000;
const RHS_ROWS: [usize; 4] = [400, 1_000, 5_000, 20_000];
const CHAIN_CASES: [(usize, usize); 4] = [
    (5, WARM_ROWS),
    (30, WARM_ROWS),
    (B_T, WARM_ROWS),
    (500, K_Y_BAR),
];
const P99_TOUCHED_CLP: usize = 5;
const WORST_CASE_TOUCHED: usize = OMEGA * B_T;
const Y_BAR_NNZ: [usize; 2] = [30, B_T];
const FINISH_WORKERS: usize = 4;
const FINISH_POSITIONS: [usize; 2] = [8, 32];
const PUSH_ROWS: [usize; 3] = [600, 5_000, 20_000];
const SEED: u64 = 0x5EED_F01D;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f64(&mut self) -> f64 {
        loop {
            let bits = (self.next_u64() >> 12) & ((1_u64 << 52) - 1);
            let v = f64::from_bits((1023_u64 << 52) | bits) - 1.5;
            if v != 0.0 {
                return v;
            }
        }
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn vec(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.f64()).collect()
    }
}

fn selected(group: &str) -> bool {
    let mut filters = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .peekable();
    filters.peek().is_none()
        || filters.any(|f| {
            let anchored = f.starts_with('^') || f.ends_with('/');
            let f = f.trim_start_matches('^').trim_end_matches('/');
            f == group || f.starts_with(&format!("{group}/")) || (!anchored && group.contains(f))
        })
}

fn threads() -> usize {
    thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

fn assert_bits(group: &str, variant: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{group}: {variant} length differs");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{group}: {variant} differs at {i}: {:#018x} against {:#018x}",
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn assert_cut(group: &str, variant: &str, got: (&[f64], f64), want: (&[f64], f64)) {
    assert_bits(group, variant, got.0, want.0);
    assert!(
        got.1.to_bits() == want.1.to_bits(),
        "{group}: {variant} intercept differs: {:#018x} against {:#018x}",
        got.1.to_bits(),
        want.1.to_bits()
    );
}

fn assert_close(group: &str, variant: &str, got: (&[f64], f64), want: (&[f64], f64)) {
    let close = |g: f64, w: f64| (g - w).abs() <= 1e-9 * (1.0 + w.abs());
    for (i, (&g, &w)) in got.0.iter().zip(want.0).enumerate() {
        assert!(
            close(g, w),
            "{group}: {variant} differs at {i}: {g:e} against {w:e}"
        );
    }
    assert!(
        close(got.1, want.1),
        "{group}: {variant} intercept differs: {:e} against {:e}",
        got.1,
        want.1
    );
}

struct Layout {
    projection: CutStateProjection,
    slots: ChainRuleSlots,
    par: PrecomputedPar,
    stride: usize,
    n_state: usize,
    state_len: usize,
    lag_width: usize,
}

fn layout() -> Layout {
    let global = state_layout(N_HYDROS, LAG_DEPTH);
    let projection = CutStateProjection::new(
        &global,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let slots = ChainRuleSlots::compose(&global, &projection, &projection);
    let stride = projection.n_slots();
    assert_eq!(stride, N_HYDROS * (1 + LAG_DEPTH));
    let (lag0, shifted) = (slots.resident_lag0(), slots.resident_shifted());
    assert_eq!(lag0.end, shifted.start);

    let mut rng = Rng(SEED);
    let ids: Vec<EntityId> = (0..N_HYDROS).map(|h| EntityId::from(h as i32)).collect();
    let models: Vec<InflowModel> = ids
        .iter()
        .map(|&hydro_id| InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 1.0,
            std_m3s: 1.0,
            ar_coefficients: rng.vec(LAG_DEPTH).iter().map(|v| 0.1 * v).collect(),
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::default(),
        end_date: NaiveDate::default(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: String::new(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };
    let par = PrecomputedPar::build(&models, &[stage], &ids, None).expect("PAR models are valid");
    assert_eq!(par.max_order(), LAG_DEPTH);

    Layout {
        state_len: slots.non_lag()[1].1.end,
        lag_width: lag0.len() + shifted.len(),
        n_state: global.n_state,
        projection,
        slots,
        par,
        stride,
    }
}

fn build_pool(rows: usize, stride: usize, rng: &mut Rng) -> CutPool {
    let mut pool = CutPool::new(rows, stride, 1, 0);
    let mut row = vec![0.0; stride];
    for k in 0..rows {
        for v in &mut row {
            *v = rng.f64();
        }
        pool.add_cut(NodeId(0), k as u64, 0, rng.f64(), &row);
    }
    pool
}

fn ascending_dot(coefficients: &[f64], values: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(values)
        .fold(0.0, |acc, (&b, &v)| acc + b * v)
}

struct Case<'a> {
    pool: &'a CutPool,
    slots: &'a [usize],
    lag_point: &'a CutLagPoint,
    state: &'a [f64],
    mean: &'a [f64],
}

type RhsVariant = fn(&Case<'_>, &mut [f64], &mut [f64]);

const RHS_VARIANTS: [(&str, RhsVariant); 3] =
    [("plain", plain), ("batched", batched), ("dgemm", gemm)];

fn plain(case: &Case<'_>, out: &mut [f64], _scratch: &mut [f64]) {
    cut_brackets_reference_into(
        case.pool,
        case.slots,
        case.lag_point,
        case.state,
        case.mean,
        out,
    );
}

fn batched(case: &Case<'_>, out: &mut [f64], _scratch: &mut [f64]) {
    cut_brackets_into(
        case.pool,
        case.slots,
        case.lag_point,
        case.state,
        case.mean,
        out,
    );
}

fn shifted_dots_gemm(case: &Case<'_>, out: &mut [f64]) {
    let shifted = case.lag_point.shifted();
    let stride = case.pool.state_dimension;
    let rows = case.slots.len();
    assert!(case.slots.iter().enumerate().all(|(i, &slot)| slot == i));
    let block = &case.pool.coefficients_prefix()[shifted.start..];
    assert!(rows <= case.pool.populated() && block.len() >= (rows - 1) * stride + shifted.len());
    assert!(case.state.len() == shifted.len() && out.len() == rows);
    // SAFETY: `block` holds rows `0..rows` of the pool from the first shifted slot, so the
    // `rows x shifted.len()` operand with row stride `stride` stays inside it (asserted above).
    // `case.state` holds `shifted.len()` values and `out` holds `rows`, read and written with
    // unit strides, and the three buffers belong to distinct owners so they cannot alias.
    unsafe {
        matrixmultiply::dgemm(
            rows,
            shifted.len(),
            1,
            1.0,
            block.as_ptr(),
            stride as isize,
            1,
            case.state.as_ptr(),
            1,
            1,
            0.0,
            out.as_mut_ptr(),
            1,
            1,
        );
    }
}

fn gemm(case: &Case<'_>, out: &mut [f64], scratch: &mut [f64]) {
    shifted_dots_gemm(case, scratch);
    let lag0 = case.lag_point.lag0();
    for ((o, &s), &slot) in out.iter_mut().zip(&*scratch).zip(case.slots) {
        let mean_dot = ascending_dot(&case.pool.coefficient_row(slot)[lag0.clone()], case.mean);
        *o = (case.pool.intercept(slot) + s) + mean_dot;
    }
}

fn check_rhs_variants(group: &str, case: &Case<'_>) {
    let rows = case.slots.len();
    let (mut want, mut got, mut scratch) = (vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]);
    plain(case, &mut want, &mut scratch);
    batched(case, &mut got, &mut scratch);
    assert_bits(group, "batched", &got, &want);

    let shifted = case.lag_point.shifted();
    shifted_dots_gemm(case, &mut scratch);
    let n = shifted.len() as f64;
    let gamma = n * (f64::EPSILON / 2.0) / (1.0 - n * (f64::EPSILON / 2.0));
    for (i, &slot) in case.slots.iter().enumerate() {
        let row = &case.pool.coefficient_row(slot)[shifted.clone()];
        let reference = ascending_dot(row, case.state);
        let magnitude: f64 = row.iter().zip(case.state).map(|(b, v)| (b * v).abs()).sum();
        assert!(
            (scratch[i] - reference).abs() <= 2.0 * gamma * magnitude,
            "{group}: dgemm differs at {i}: {:#018x} against {:#018x}",
            scratch[i].to_bits(),
            reference.to_bits()
        );
    }
}

fn check_signed_zero_probe(group: &str, lay: &Layout) {
    let rows = 17;
    let mut pool = CutPool::new(rows, lay.stride, 1, 0);
    let row = vec![-1.0; lay.stride];
    for k in 0..rows {
        pool.add_cut(NodeId(0), k as u64, 0, -0.0, &row);
    }
    let lag_point = lay.projection.lag_point();
    let (state, mean) = (
        vec![0.0; lag_point.shifted().len()],
        vec![0.0; lag_point.lag0().len()],
    );
    let slots: Vec<usize> = (0..rows).collect();
    check_rhs_variants(
        group,
        &Case {
            pool: &pool,
            slots: &slots,
            lag_point,
            state: &state,
            mean: &mean,
        },
    );
}

fn bench_rhs_kernel(c: &mut Criterion) {
    if !selected("rhs_kernel") {
        return;
    }
    let lay = layout();
    let lag_point = lay.projection.lag_point();
    let mut rng = Rng(SEED);
    check_signed_zero_probe("rhs_kernel", &lay);
    let mut group = c.benchmark_group("rhs_kernel");
    for rows in RHS_ROWS {
        let pool = build_pool(rows, lay.stride, &mut rng);
        let slots: Vec<usize> = (0..rows).collect();
        let (state, mean) = (
            rng.vec(lag_point.shifted().len()),
            rng.vec(lag_point.lag0().len()),
        );
        let case = Case {
            pool: &pool,
            slots: &slots,
            lag_point,
            state: &state,
            mean: &mean,
        };
        check_rhs_variants("rhs_kernel", &case);

        group.throughput(Throughput::Elements(rows as u64));
        let (mut out, mut scratch) = (vec![0.0; rows], vec![0.0; rows]);
        for (name, run) in RHS_VARIANTS {
            group.bench_function(BenchmarkId::new(name, rows), |b| {
                b.iter(|| {
                    run(&case, &mut out, &mut scratch);
                    black_box(&out);
                });
            });
        }
    }
    group.finish();
}

fn slowest_pass(
    pool: &CutPool,
    slots: &[usize],
    lag_point: &CutLagPoint,
    states: &[(Vec<f64>, Vec<f64>)],
    run: RhsVariant,
    iters: u64,
) -> Duration {
    let barrier = Barrier::new(states.len());
    thread::scope(|s| {
        let workers: Vec<_> = states
            .iter()
            .map(|(state, mean)| {
                let barrier = &barrier;
                s.spawn(move || {
                    let case = Case {
                        pool,
                        slots,
                        lag_point,
                        state,
                        mean,
                    };
                    let (mut out, mut scratch) = (vec![0.0; slots.len()], vec![0.0; slots.len()]);
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        barrier.wait();
                        let start = Instant::now();
                        run(&case, &mut out, &mut scratch);
                        total += start.elapsed();
                        black_box(&out);
                    }
                    total
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("worker thread panicked"))
            .max()
            .expect("at least one thread")
    })
}

fn bench_rhs_kernel_all_cores(c: &mut Criterion) {
    if !selected("rhs_kernel_all_cores") {
        return;
    }
    let lay = layout();
    let lag_point = lay.projection.lag_point();
    let mut rng = Rng(SEED);
    let threads = threads();
    println!("rhs_kernel_all_cores: {threads} threads");
    check_signed_zero_probe("rhs_kernel_all_cores", &lay);
    let mut group = c.benchmark_group("rhs_kernel_all_cores");
    for rows in RHS_ROWS {
        let pool = build_pool(rows, lay.stride, &mut rng);
        let slots: Vec<usize> = (0..rows).collect();
        let states: Vec<(Vec<f64>, Vec<f64>)> = (0..threads)
            .map(|_| {
                (
                    rng.vec(lag_point.shifted().len()),
                    rng.vec(lag_point.lag0().len()),
                )
            })
            .collect();
        check_rhs_variants(
            "rhs_kernel_all_cores",
            &Case {
                pool: &pool,
                slots: &slots,
                lag_point,
                state: &states[0].0,
                mean: &states[0].1,
            },
        );

        group.throughput(Throughput::Elements(rows as u64));
        for (name, run) in RHS_VARIANTS {
            group.bench_function(BenchmarkId::new(name, rows), |b| {
                b.iter_custom(|iters| slowest_pass(&pool, &slots, lag_point, &states, run, iters));
            });
        }
    }
    group.finish();
}

fn bench_noise_term_pass(c: &mut Criterion) {
    if !selected("noise_term_pass") {
        return;
    }
    let lay = layout();
    let lag_point = lay.projection.lag_point();
    let mut rng = Rng(SEED);
    let mut group = c.benchmark_group("noise_term_pass");
    for rows in RHS_ROWS {
        let pool = build_pool(rows, lay.stride, &mut rng);
        let slots: Vec<usize> = (0..rows).collect();
        let (state, mean) = (
            rng.vec(lag_point.shifted().len()),
            rng.vec(lag_point.lag0().len()),
        );
        let noise = rng.vec(lag_point.lag0().len());
        let mut brackets = vec![0.0; rows];
        cut_brackets_into(&pool, &slots, lag_point, &state, &mean, &mut brackets);
        let (mut noise_terms, mut rhs, mut fused) =
            (vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]);

        let two_step = |noise_terms: &mut [f64], rhs: &mut [f64]| {
            for (w, &slot) in noise_terms.iter_mut().zip(&slots) {
                *w = cut_noise_term(pool.coefficient_row(slot), lag_point, &noise);
            }
            cut_rhs_from_noise_terms_into(&brackets, noise_terms, rhs);
        };
        two_step(&mut noise_terms, &mut rhs);
        cut_rhs_into(&pool, &slots, lag_point, &noise, &brackets, &mut fused);
        assert_bits("noise_term_pass", "two_step", &rhs, &fused);

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("two_step", rows), |b| {
            b.iter(|| {
                two_step(&mut noise_terms, &mut rhs);
                black_box(&rhs);
            });
        });
        group.bench_function(BenchmarkId::new("fused", rows), |b| {
            b.iter(|| {
                cut_rhs_into(&pool, &slots, lag_point, &noise, &brackets, &mut fused);
                black_box(&fused);
            });
        });
    }
    group.finish();
}

struct FixedDuals {
    objective: Vec<f64>,
    state: Vec<f64>,
    coupled: Vec<f64>,
    state_len: usize,
}

impl FixedDuals {
    fn new(lay: &Layout, rng: &mut Rng, openings: usize) -> Self {
        Self {
            objective: (0..openings).map(|_| 1.0 + rng.f64()).collect(),
            state: rng.vec(openings * lay.state_len),
            coupled: rng.vec(openings * N_HYDROS),
            state_len: lay.state_len,
        }
    }

    fn opening<'a>(&'a self, i: usize, cut_duals: &'a [(usize, f64)]) -> OpeningDuals<'a> {
        OpeningDuals {
            objective: self.objective[i],
            state_duals: &self.state[i * self.state_len..(i + 1) * self.state_len],
            coupled_duals: &self.coupled[i * N_HYDROS..(i + 1) * N_HYDROS],
            cut_duals,
        }
    }

    fn all<'a>(&'a self, cut_duals: &'a [Vec<(usize, f64)>]) -> [OpeningDuals<'a>; OMEGA] {
        array::from_fn(|i| self.opening(i, &cut_duals[i]))
    }
}

fn draw_cut_duals(rng: &mut Rng, nnz: usize, rows: usize) -> Vec<(usize, f64)> {
    assert!(nnz <= rows);
    let mut slots: Vec<usize> = Vec::with_capacity(nnz);
    if nnz * 4 > rows {
        let mut needed = nnz;
        for slot in 0..rows {
            if rng.below(rows - slot) < needed {
                slots.push(slot);
                needed -= 1;
            }
        }
    } else {
        while slots.len() < nnz {
            let missing = nnz - slots.len();
            slots.extend((0..missing).map(|_| rng.below(rows)));
            slots.sort_unstable();
            slots.dedup();
        }
    }
    slots.into_iter().map(|slot| (slot, rng.f64())).collect()
}

fn draw_weights(rng: &mut Rng) -> Vec<f64> {
    (0..OMEGA)
        .map(|_| (1.0 + rng.f64()) / OMEGA as f64)
        .collect()
}

#[derive(Clone, Copy)]
struct DualsShape {
    sets: usize,
    nnz: usize,
    rows: usize,
    cold: bool,
}

impl DualsShape {
    fn rng(self) -> Rng {
        Rng(SEED ^ ((self.nnz as u64) << 32) ^ self.rows as u64)
    }

    fn draw(self, rng: &mut Rng) -> Vec<Vec<(usize, f64)>> {
        (0..self.sets)
            .map(|_| draw_cut_duals(rng, self.nnz, self.rows))
            .collect()
    }

    fn describe(self, label: &str, span_bytes: usize) {
        let mut slots: Vec<usize> = self
            .draw(&mut self.rng())
            .iter()
            .flatten()
            .map(|&(slot, _)| slot)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        println!(
            "{label}: {} distinct rows of {} walked, {:.1} MiB of lag span{}",
            slots.len(),
            self.rows,
            (slots.len() * span_bytes) as f64 / f64::from(1_u32 << 20),
            if self.cold {
                " (drawn afresh per call)"
            } else {
                ""
            }
        );
    }
}

fn bench_duals(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: BenchmarkId,
    shape: DualsShape,
    mut run: impl FnMut(&[Vec<(usize, f64)>]) -> f64,
) {
    let mut rng = shape.rng();
    if shape.cold {
        group.bench_function(id, |b| {
            b.iter_batched_ref(
                || shape.draw(&mut rng),
                |duals| black_box(run(duals)),
                BatchSize::PerIteration,
            );
        });
    } else {
        let duals = shape.draw(&mut rng);
        group.bench_function(id, |b| b.iter(|| black_box(run(&duals))));
    }
}

fn chain_inputs<'a>(lay: &'a Layout, pool: &'a CutPool) -> ChainRuleInputs<'a> {
    ChainRuleInputs {
        slots: &lay.slots,
        resident_pool: pool,
        extracted: &lay.projection,
        par: &lay.par,
        stage: 0,
    }
}

fn scratch_for(pool_rows: usize, pair_hint: usize) -> AggregateScratch {
    let mut scratch = AggregateScratch::new(N_HYDROS, LAG_DEPTH);
    scratch.reserve(pool_rows, pair_hint);
    scratch
}

fn per_opening_weighted(
    inputs: &ChainRuleInputs<'_>,
    openings: &[OpeningDuals<'_>],
    weights: &[f64],
    x_hat: &[f64],
    scratch: &mut ChainRuleScratch,
    (tmp, out): (&mut [f64], &mut [f64]),
) -> f64 {
    out.fill(0.0);
    let mut alpha = 0.0;
    for (opening, &mu) in openings.iter().zip(weights) {
        alpha += mu * per_opening_cut_into(inputs, opening, x_hat, scratch, tmp);
        for (o, &t) in out.iter_mut().zip(&*tmp) {
            *o += mu * t;
        }
    }
    alpha
}

fn check_chain_variants(
    lay: &Layout,
    pool: &CutPool,
    (fixed, x_hat, weights): (&FixedDuals, &[f64], &[f64]),
    duals: &[Vec<(usize, f64)>],
) {
    let inputs = chain_inputs(lay, pool);
    let openings = fixed.all(duals);
    let mut per_scratch = ChainRuleScratch::new(N_HYDROS, LAG_DEPTH);
    let mut agg_scratch = scratch_for(pool.capacity, duals.iter().map(Vec::len).sum());
    let (mut want, mut got, mut tmp) = (
        vec![0.0; lay.stride],
        vec![0.0; lay.stride],
        vec![0.0; lay.stride],
    );

    let want_alpha =
        per_opening_cut_into(&inputs, &openings[0], x_hat, &mut per_scratch, &mut want);
    let alpha = aggregate_first_cut_into(
        &[ChildOpenings {
            inputs: chain_inputs(lay, pool),
            openings: &openings[..1],
        }],
        &[1.0],
        x_hat,
        &mut agg_scratch,
        &mut got,
    );
    assert_cut(
        "chain_rule",
        "aggregate_of_one_opening",
        (&got, alpha),
        (&want, want_alpha),
    );

    let want_alpha = per_opening_weighted(
        &inputs,
        &openings,
        weights,
        x_hat,
        &mut per_scratch,
        (&mut tmp, &mut want),
    );
    let alpha = aggregate_first_cut_into(
        &[ChildOpenings {
            inputs: chain_inputs(lay, pool),
            openings: &openings,
        }],
        weights,
        x_hat,
        &mut agg_scratch,
        &mut got,
    );
    assert_close(
        "chain_rule",
        "aggregate_of_openings",
        (&got, alpha),
        (&want, want_alpha),
    );
}

fn bench_chain_rule(c: &mut Criterion) {
    if !selected("chain_rule") {
        return;
    }
    let lay = layout();
    let mut rng = Rng(SEED);
    let fixed = FixedDuals::new(&lay, &mut rng, OMEGA);
    let x_hat = rng.vec(lay.n_state);
    let weights = draw_weights(&mut rng);
    let cold_pool = build_pool(COLD_ROWS, lay.stride, &mut rng);
    let mut group = c.benchmark_group("chain_rule");

    for (nnz, warm_rows) in CHAIN_CASES {
        let warm_pool = build_pool(warm_rows, lay.stride, &mut rng);
        group.throughput(Throughput::Elements((OMEGA * nnz) as u64));
        for (temperature, pool, cold) in [("warm", &warm_pool, false), ("cold", &cold_pool, true)] {
            let shape = DualsShape {
                sets: OMEGA,
                nnz,
                rows: pool.capacity,
                cold,
            };
            check_chain_variants(
                &lay,
                pool,
                (&fixed, &x_hat, &weights),
                &shape.draw(&mut shape.rng()),
            );
            shape.describe(
                &format!("chain_rule {temperature}/{nnz}"),
                lay.lag_width * 8,
            );

            let mut per_scratch = ChainRuleScratch::new(N_HYDROS, LAG_DEPTH);
            let (mut tmp, mut out) = (vec![0.0; lay.stride], vec![0.0; lay.stride]);
            bench_duals(
                &mut group,
                BenchmarkId::new(format!("per_opening_{temperature}"), nnz),
                shape,
                |duals| {
                    per_opening_weighted(
                        &chain_inputs(&lay, pool),
                        &fixed.all(duals),
                        &weights,
                        &x_hat,
                        &mut per_scratch,
                        (&mut tmp, &mut out),
                    )
                },
            );

            let mut scratch = scratch_for(pool.capacity, OMEGA * nnz);
            bench_duals(
                &mut group,
                BenchmarkId::new(format!("aggregate_{temperature}"), nnz),
                shape,
                |duals| {
                    aggregate_first_cut_into(
                        &[ChildOpenings {
                            inputs: chain_inputs(&lay, pool),
                            openings: &fixed.all(duals),
                        }],
                        &weights,
                        &x_hat,
                        &mut scratch,
                        &mut out,
                    )
                },
            );
        }
    }
    group.finish();
}

fn lag_span(slots: &ChainRuleSlots) -> Range<usize> {
    slots.resident_lag0().start..slots.resident_shifted().end
}

fn dot_trial_state(projection: &CutStateProjection, coefficients: &[f64], x_hat: &[f64]) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(j, &c)| c * x_hat[projection.global_state_index(CutSlot::new(j)).get()])
        .sum()
}

fn walk<const TILED: bool>(
    inputs: &ChainRuleInputs<'_>,
    cut_duals: &[(usize, f64)],
    acc: &mut [f64],
) {
    let span = lag_span(inputs.slots);
    if TILED {
        for (tile_at, tile) in acc.chunks_mut(TILE).enumerate() {
            let from = span.start + tile_at * TILE;
            for &(slot, y) in cut_duals {
                if y == 0.0 {
                    continue;
                }
                let row = &inputs.resident_pool.coefficient_row(slot)[from..from + tile.len()];
                for (a, &beta) in tile.iter_mut().zip(row) {
                    *a += y * beta;
                }
            }
        }
    } else {
        for &(slot, y) in cut_duals {
            if y == 0.0 {
                continue;
            }
            let row = &inputs.resident_pool.coefficient_row(slot)[span.clone()];
            for (a, &beta) in acc.iter_mut().zip(row) {
                *a += y * beta;
            }
        }
    }
}

fn psi_combine(inputs: &ChainRuleInputs<'_>, acc: &[f64], lag_out: &mut [f64]) {
    let (n, depth) = (inputs.slots.n_hydros(), inputs.slots.lag_depth());
    for (h, &g_h) in acc[..n].iter().enumerate() {
        let psi = if inputs.par.max_order() > 0 {
            inputs.par.psi_slice(inputs.stage, h)
        } else {
            &[]
        };
        for l in 0..depth {
            let psi_l = psi.get(l).copied().unwrap_or(0.0);
            let at = l * n + h;
            lag_out[at] = if l + 1 < depth {
                psi_l * g_h + acc[n + at]
            } else {
                psi_l * g_h
            };
        }
    }
}

fn replica_cut_into<const TILED: bool>(
    inputs: &ChainRuleInputs<'_>,
    opening: &OpeningDuals<'_>,
    x_hat: &[f64],
    acc: &mut [f64],
    coefficients: &mut [f64],
) -> f64 {
    for (slots, positions) in inputs.slots.non_lag() {
        coefficients[slots].copy_from_slice(&opening.state_duals[positions]);
    }
    let n = inputs.slots.n_hydros();
    for (g, &coupled) in acc[..n].iter_mut().zip(opening.coupled_duals) {
        *g = -coupled;
    }
    acc[n..].fill(-0.0);
    walk::<TILED>(inputs, opening.cut_duals, acc);
    psi_combine(inputs, acc, &mut coefficients[inputs.slots.extracted_lag()]);
    opening.objective - dot_trial_state(inputs.extracted, coefficients, x_hat)
}

fn check_walk_variants(
    lay: &Layout,
    pool: &CutPool,
    (fixed, x_hat): (&FixedDuals, &[f64]),
    pairs: &[(usize, f64)],
) {
    let inputs = chain_inputs(lay, pool);
    let opening = fixed.opening(0, pairs);
    let (mut want, mut got) = (vec![0.0; lay.stride], vec![0.0; lay.stride]);
    let want_alpha = per_opening_cut_into(
        &inputs,
        &opening,
        x_hat,
        &mut ChainRuleScratch::new(N_HYDROS, LAG_DEPTH),
        &mut want,
    );
    let mut acc = vec![0.0; lay.lag_width];
    let alpha = replica_cut_into::<false>(&inputs, &opening, x_hat, &mut acc, &mut got);
    assert_cut(
        "chain_rule_walk",
        "row_major",
        (&got, alpha),
        (&want, want_alpha),
    );
    let alpha = replica_cut_into::<true>(&inputs, &opening, x_hat, &mut acc, &mut got);
    assert_cut(
        "chain_rule_walk",
        "j_tile",
        (&got, alpha),
        (&want, want_alpha),
    );
}

fn bench_chain_rule_walk(c: &mut Criterion) {
    if !selected("chain_rule_walk") {
        return;
    }
    let lay = layout();
    let mut rng = Rng(SEED);
    let fixed = FixedDuals::new(&lay, &mut rng, 1);
    let x_hat = rng.vec(lay.n_state);
    let cold_pool = build_pool(COLD_ROWS, lay.stride, &mut rng);
    let mut group = c.benchmark_group("chain_rule_walk");

    for touched in [P99_TOUCHED_CLP, WORST_CASE_TOUCHED] {
        let warm_pool = build_pool(WARM_ROWS.max(touched), lay.stride, &mut rng);
        group.throughput(Throughput::Elements(touched as u64));
        let mut scratch = ChainRuleScratch::new(N_HYDROS, LAG_DEPTH);
        let mut acc = vec![0.0; lay.lag_width];
        let mut coefficients = vec![0.0; lay.stride];

        let no_pairs = DualsShape {
            sets: 1,
            nnz: 0,
            rows: warm_pool.capacity,
            cold: false,
        };
        let warm_inputs = chain_inputs(&lay, &warm_pool);
        bench_duals(
            &mut group,
            BenchmarkId::new("fixed_part", touched),
            no_pairs,
            |_| {
                per_opening_cut_into(
                    &warm_inputs,
                    &fixed.opening(0, &[]),
                    &x_hat,
                    &mut scratch,
                    &mut coefficients,
                )
            },
        );

        for (temperature, pool, cold) in [("warm", &warm_pool, false), ("cold", &cold_pool, true)] {
            let shape = DualsShape {
                sets: 1,
                nnz: touched,
                rows: pool.capacity,
                cold,
            };
            check_walk_variants(
                &lay,
                pool,
                (&fixed, &x_hat),
                &shape.draw(&mut shape.rng())[0],
            );
            shape.describe(
                &format!("chain_rule_walk {temperature}/{touched}"),
                lay.lag_width * 8,
            );

            let inputs = chain_inputs(&lay, pool);
            bench_duals(
                &mut group,
                BenchmarkId::new(format!("production_{temperature}"), touched),
                shape,
                |duals| {
                    per_opening_cut_into(
                        &inputs,
                        &fixed.opening(0, &duals[0]),
                        &x_hat,
                        &mut scratch,
                        &mut coefficients,
                    )
                },
            );
            bench_duals(
                &mut group,
                BenchmarkId::new(format!("row_major_{temperature}"), touched),
                shape,
                |duals| {
                    replica_cut_into::<false>(
                        &inputs,
                        &fixed.opening(0, &duals[0]),
                        &x_hat,
                        &mut acc,
                        &mut coefficients,
                    )
                },
            );
            bench_duals(
                &mut group,
                BenchmarkId::new(format!("j_tile_{temperature}"), touched),
                shape,
                |duals| {
                    replica_cut_into::<true>(
                        &inputs,
                        &fixed.opening(0, &duals[0]),
                        &x_hat,
                        &mut acc,
                        &mut coefficients,
                    )
                },
            );
        }
    }
    group.finish();
}

struct ReplicaScratch {
    y_bar: Vec<f64>,
    touched: Vec<u64>,
    pairs: Vec<(usize, f64)>,
    acc: Vec<f64>,
    child_lag: Vec<f64>,
}

impl ReplicaScratch {
    fn new(lay: &Layout, rows: usize, pair_hint: usize) -> Self {
        Self {
            y_bar: vec![0.0; rows],
            touched: vec![0; rows.div_ceil(64)],
            pairs: Vec::with_capacity(pair_hint),
            acc: vec![0.0; lay.lag_width],
            child_lag: vec![0.0; lay.lag_width],
        }
    }
}

fn aggregate_replica<const BITMAP: bool>(
    inputs: &ChainRuleInputs<'_>,
    openings: &[OpeningDuals<'_>],
    weights: &[f64],
    x_hat: &[f64],
    (scratch, coefficients): (&mut ReplicaScratch, &mut [f64]),
) -> f64 {
    let non_lag = inputs.slots.non_lag();
    let n = inputs.slots.n_hydros();
    coefficients.fill(-0.0);
    let mut q_bar = -0.0;
    for (opening, &mu) in openings.iter().zip(weights) {
        if mu == 0.0 {
            continue;
        }
        q_bar += mu * opening.objective;
        for (range, positions) in &non_lag {
            let dst = &mut coefficients[range.clone()];
            for (c, &d) in dst.iter_mut().zip(&opening.state_duals[positions.clone()]) {
                *c += mu * d;
            }
        }
    }

    let ReplicaScratch {
        y_bar,
        touched,
        pairs,
        acc,
        child_lag,
    } = scratch;
    acc[..n].fill(-0.0);
    let (mut first_word, mut last_word) = (usize::MAX, 0);
    for (opening, &mu) in openings.iter().zip(weights) {
        if mu == 0.0 {
            continue;
        }
        for (g, &coupled) in acc[..n].iter_mut().zip(opening.coupled_duals) {
            *g -= mu * coupled;
        }
        for &(slot, y) in opening.cut_duals {
            if y == 0.0 {
                continue;
            }
            y_bar[slot] += mu * y;
            if BITMAP {
                let word = slot / 64;
                touched[word] |= 1_u64 << (slot % 64);
                first_word = first_word.min(word);
                last_word = last_word.max(word);
            }
        }
    }
    if BITMAP {
        for (word_at, word) in touched
            .iter_mut()
            .enumerate()
            .take(last_word + 1)
            .skip(first_word)
        {
            let mut bits = std::mem::take(word);
            while bits != 0 {
                let slot = word_at * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                pairs.push((slot, std::mem::take(&mut y_bar[slot])));
            }
        }
    } else {
        for (slot, v) in y_bar.iter_mut().enumerate() {
            if *v != 0.0 {
                pairs.push((slot, std::mem::take(v)));
            }
        }
    }

    acc[n..].fill(-0.0);
    walk::<false>(inputs, pairs, acc);
    psi_combine(inputs, acc, child_lag);
    for (c, &v) in coefficients[inputs.slots.extracted_lag()]
        .iter_mut()
        .zip(child_lag.iter())
    {
        *c += v;
    }
    pairs.clear();
    q_bar - dot_trial_state(inputs.extracted, coefficients, x_hat)
}

fn bench_y_bar_formation(c: &mut Criterion) {
    if !selected("y_bar_formation") {
        return;
    }
    let lay = layout();
    let mut rng = Rng(SEED);
    let fixed = FixedDuals::new(&lay, &mut rng, OMEGA);
    let x_hat = rng.vec(lay.n_state);
    let weights = draw_weights(&mut rng);
    let pool = build_pool(K_Y_BAR, lay.stride, &mut rng);
    let mut group = c.benchmark_group("y_bar_formation");

    for nnz in Y_BAR_NNZ {
        let shape = DualsShape {
            sets: OMEGA,
            nnz,
            rows: K_Y_BAR,
            cold: false,
        };
        let mut scratch = scratch_for(K_Y_BAR, OMEGA * nnz);
        let mut replica = ReplicaScratch::new(&lay, K_Y_BAR, OMEGA * nnz);
        let (mut want, mut got) = (vec![0.0; lay.stride], vec![0.0; lay.stride]);

        let duals = shape.draw(&mut shape.rng());
        let openings = fixed.all(&duals);
        let want_alpha = aggregate_first_cut_into(
            &[ChildOpenings {
                inputs: chain_inputs(&lay, &pool),
                openings: &openings,
            }],
            &weights,
            &x_hat,
            &mut scratch,
            &mut want,
        );
        let alpha = aggregate_replica::<true>(
            &chain_inputs(&lay, &pool),
            &openings,
            &weights,
            &x_hat,
            (&mut replica, &mut got),
        );
        assert_cut(
            "y_bar_formation",
            "bitmap",
            (&got, alpha),
            (&want, want_alpha),
        );
        let alpha = aggregate_replica::<false>(
            &chain_inputs(&lay, &pool),
            &openings,
            &weights,
            &x_hat,
            (&mut replica, &mut got),
        );
        assert_cut(
            "y_bar_formation",
            "dense",
            (&got, alpha),
            (&want, want_alpha),
        );
        shape.describe(&format!("y_bar_formation/{nnz}"), lay.lag_width * 8);

        group.throughput(Throughput::Elements((OMEGA * nnz) as u64));
        bench_duals(
            &mut group,
            BenchmarkId::new("production", nnz),
            shape,
            |duals| {
                aggregate_first_cut_into(
                    &[ChildOpenings {
                        inputs: chain_inputs(&lay, &pool),
                        openings: &fixed.all(duals),
                    }],
                    &weights,
                    &x_hat,
                    &mut scratch,
                    &mut got,
                )
            },
        );
        bench_duals(
            &mut group,
            BenchmarkId::new("bitmap_replica", nnz),
            shape,
            |duals| {
                aggregate_replica::<true>(
                    &chain_inputs(&lay, &pool),
                    &fixed.all(duals),
                    &weights,
                    &x_hat,
                    (&mut replica, &mut got),
                )
            },
        );
        bench_duals(
            &mut group,
            BenchmarkId::new("dense_replica", nnz),
            shape,
            |duals| {
                aggregate_replica::<false>(
                    &chain_inputs(&lay, &pool),
                    &fixed.all(duals),
                    &weights,
                    &x_hat,
                    (&mut replica, &mut got),
                )
            },
        );
    }
    group.finish();
}

#[derive(Default)]
struct Records {
    keys: Vec<(usize, usize)>,
    objectives: Vec<f64>,
    state_duals: Vec<f64>,
    coupled_duals: Vec<f64>,
    pair_ends: Vec<usize>,
    pairs: Vec<(usize, f64)>,
}

impl Records {
    fn view(&self, i: usize, state_len: usize) -> OpeningDuals<'_> {
        let start = if i == 0 { 0 } else { self.pair_ends[i - 1] };
        OpeningDuals {
            objective: self.objectives[i],
            state_duals: &self.state_duals[i * state_len..(i + 1) * state_len],
            coupled_duals: &self.coupled_duals[i * N_HYDROS..(i + 1) * N_HYDROS],
            cut_duals: &self.pairs[start..self.pair_ends[i]],
        }
    }
}

fn build_records(
    lay: &Layout,
    rng: &mut Rng,
    positions: usize,
    nnz: usize,
) -> (Vec<Records>, Vec<(usize, usize)>) {
    let mut records: Vec<Records> = (0..FINISH_WORKERS).map(|_| Records::default()).collect();
    for pos in 0..positions {
        for omega in 0..OMEGA {
            let r = &mut records[(pos * OMEGA + omega) % FINISH_WORKERS];
            r.keys.push((pos, omega));
            r.objectives.push(1.0 + rng.f64());
            r.state_duals.extend(rng.vec(lay.state_len));
            r.coupled_duals.extend(rng.vec(N_HYDROS));
            r.pairs.extend(draw_cut_duals(rng, nnz, K_Y_BAR));
            r.pair_ends.push(r.pairs.len());
        }
    }
    let mut location = vec![(usize::MAX, usize::MAX); positions * OMEGA];
    for (worker, r) in records.iter().enumerate() {
        for (at, &(pos, omega)) in r.keys.iter().enumerate() {
            location[pos * OMEGA + omega] = (worker, at);
        }
    }
    (records, location)
}

struct Finish<'a> {
    lay: &'a Layout,
    pool: &'a CutPool,
    records: &'a [Records],
    location: &'a [(usize, usize)],
    weights: &'a [f64],
    x_hats: &'a [f64],
}

impl Finish<'_> {
    fn position(
        &self,
        pos: usize,
        scratch: &mut AggregateScratch,
        coefficients: &mut [f64],
    ) -> f64 {
        let located = &self.location[pos * OMEGA..(pos + 1) * OMEGA];
        let views: [OpeningDuals<'_>; OMEGA] = array::from_fn(|omega| {
            let (worker, at) = located[omega];
            self.records[worker].view(at, self.lay.state_len)
        });
        let n_state = self.lay.n_state;
        aggregate_first_cut_into(
            &[ChildOpenings {
                inputs: chain_inputs(self.lay, self.pool),
                openings: &views,
            }],
            self.weights,
            &self.x_hats[pos * n_state..(pos + 1) * n_state],
            scratch,
            coefficients,
        )
    }

    fn serial(&self, scratch: &mut AggregateScratch, arena: &mut [f64], intercepts: &mut [f64]) {
        for (pos, (out, icpt)) in arena
            .chunks_mut(self.lay.stride)
            .zip(intercepts)
            .enumerate()
        {
            *icpt = self.position(pos, scratch, out);
        }
    }

    fn parallel(
        &self,
        scratches: &mut [AggregateScratch],
        arena: &mut [f64],
        intercepts: &mut [f64],
    ) {
        let stride = self.lay.stride;
        let per_thread = intercepts.len().div_ceil(scratches.len());
        thread::scope(|s| {
            for (k, ((scratch, out), icpts)) in scratches
                .iter_mut()
                .zip(arena.chunks_mut(per_thread * stride))
                .zip(intercepts.chunks_mut(per_thread))
                .enumerate()
            {
                s.spawn(move || {
                    for (j, (chunk, icpt)) in out.chunks_mut(stride).zip(icpts).enumerate() {
                        *icpt = self.position(k * per_thread + j, scratch, chunk);
                    }
                });
            }
        });
    }
}

fn bench_by_node_finish(c: &mut Criterion) {
    if !selected("by_node_finish") {
        return;
    }
    let lay = layout();
    let mut rng = Rng(SEED);
    let weights = draw_weights(&mut rng);
    let pool = build_pool(K_Y_BAR, lay.stride, &mut rng);
    let threads = threads();
    println!("by_node_finish: {threads} threads");
    let mut group = c.benchmark_group("by_node_finish");

    for positions in FINISH_POSITIONS {
        let x_hats = rng.vec(positions * lay.n_state);
        for nnz in Y_BAR_NNZ {
            let (records, location) = build_records(&lay, &mut rng, positions, nnz);
            let finish = Finish {
                lay: &lay,
                pool: &pool,
                records: &records,
                location: &location,
                weights: &weights,
                x_hats: &x_hats,
            };
            let mut serial_scratch = scratch_for(K_Y_BAR, OMEGA * nnz);
            let mut scratches: Vec<AggregateScratch> = (0..threads.min(positions))
                .map(|_| scratch_for(K_Y_BAR, OMEGA * nnz))
                .collect();
            let (mut arena, mut intercepts) =
                (vec![0.0; positions * lay.stride], vec![0.0; positions]);
            let (mut arena_par, mut intercepts_par) = (arena.clone(), intercepts.clone());

            finish.serial(&mut serial_scratch, &mut arena, &mut intercepts);
            finish.parallel(&mut scratches, &mut arena_par, &mut intercepts_par);
            assert_bits("by_node_finish", "parallel_cuts", &arena_par, &arena);
            assert_bits(
                "by_node_finish",
                "parallel_intercepts",
                &intercepts_par,
                &intercepts,
            );

            group.throughput(Throughput::Elements(positions as u64));
            group.bench_function(
                BenchmarkId::new(format!("serial_{positions}_positions"), nnz),
                |b| {
                    b.iter(|| {
                        finish.serial(&mut serial_scratch, &mut arena, &mut intercepts);
                        black_box(&intercepts);
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(format!("parallel_{positions}_positions"), nnz),
                |b| {
                    b.iter(|| {
                        finish.parallel(&mut scratches, &mut arena_par, &mut intercepts_par);
                        black_box(&intercepts_par);
                    });
                },
            );
        }
    }
    group.finish();
}

fn push_template() -> StageTemplate {
    StageTemplate {
        num_cols: 2,
        num_rows: 1,
        num_nz: 2,
        col_starts: vec![0_i32, 1, 2],
        row_indices: vec![0_i32, 0],
        values: vec![1.0, 1.0],
        col_lower: vec![0.0, 0.0],
        col_upper: vec![f64::INFINITY, 10.0],
        objective: vec![1.0, 0.0],
        row_lower: vec![1.0],
        row_upper: vec![f64::INFINITY],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn push_cut_rows(rows: usize, rng: &mut Rng) -> RowBatch {
    let mut batch = RowBatch {
        num_rows: rows,
        row_starts: Vec::with_capacity(rows + 1),
        col_indices: Vec::with_capacity(2 * rows),
        values: Vec::with_capacity(2 * rows),
        row_lower: rng.vec(rows),
        row_upper: vec![f64::INFINITY; rows],
    };
    for i in 0..rows {
        batch.row_starts.push(2 * i as i32);
        batch.col_indices.extend([0_i32, 1]);
        batch.values.extend([1.0, rng.f64()]);
    }
    batch.row_starts.push(2 * rows as i32);
    batch
}

fn loaded_solver(batch: &RowBatch) -> ActiveSolver {
    let mut solver = ActiveSolver::new().expect("the solver builds");
    solver.load_model(&push_template());
    solver.add_rows(batch);
    solver
}

fn solved_objective(solver: &mut ActiveSolver) -> f64 {
    solver.solve(None).expect("the push LP solves").objective
}

fn bench_solver_push(c: &mut Criterion) {
    if !selected("solver_push") {
        return;
    }
    let mut rng = Rng(SEED);
    let mut group = c.benchmark_group("solver_push");
    for rows in PUSH_ROWS {
        let batch = push_cut_rows(rows, &mut rng);
        let indices: Vec<usize> = (1..=rows).collect();
        let upper = vec![f64::INFINITY; rows];
        let (lower_a, lower_b) = (rng.vec(rows), rng.vec(rows));

        let (mut by_set, mut by_range) = (loaded_solver(&batch), loaded_solver(&batch));
        by_set.set_row_bounds(&indices, &lower_b, &upper);
        by_range.set_row_lower_bounds_range(1, &lower_b);
        let (set_objective, range_objective) = (
            solved_objective(&mut by_set),
            solved_objective(&mut by_range),
        );
        assert!(
            (set_objective - range_objective).abs() <= 1e-9 * (1.0 + set_objective.abs()),
            "solver_push: the range push solves to {range_objective} against {set_objective} \
             for {rows} rows"
        );

        let mut solver = loaded_solver(&batch);
        solved_objective(&mut solver);
        group.throughput(Throughput::Elements(rows as u64));
        let mut flip = false;
        group.bench_function(BenchmarkId::new("set_row_bounds", rows), |b| {
            b.iter(|| {
                flip = !flip;
                let lower = if flip { &lower_a } else { &lower_b };
                solver.set_row_bounds(&indices, lower, &upper);
            });
        });
        group.bench_function(BenchmarkId::new("set_row_lower_bounds_range", rows), |b| {
            b.iter(|| {
                flip = !flip;
                let lower = if flip { &lower_a } else { &lower_b };
                solver.set_row_lower_bounds_range(1, lower);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30);
    targets = bench_rhs_kernel,
        bench_rhs_kernel_all_cores,
        bench_noise_term_pass,
        bench_chain_rule,
        bench_chain_rule_walk,
        bench_y_bar_formation,
        bench_by_node_finish,
        bench_solver_push
}
criterion_main!(benches);
