//! Microbenchmark for the cut-selection GEMM kernel.
//!
//! Synthetic fixtures at aggregated (K=945, M=384, D=155) and
//! disaggregated (K=1000, M=384, D=2080) sizes. Pinned at 1, 8, and
//! 96 rayon worker threads.
//!
//! The disaggregated case is unconditional (no `slow-` gate): Criterion runs
//! benches only on demand, so the user controls execution.

#![allow(missing_docs, clippy::expect_used, clippy::cast_possible_truncation)]

use cobre_sddp::cut::CutPool;
use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
use cobre_sddp::setup::NodeId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fill_f64(buf: &mut [f64], seed: u64) {
    let mut state = seed;
    for slot in buf.iter_mut() {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1u64 << 52) - 1);
        *slot = f64::from_bits((1023u64 << 52) | bits) - 1.5;
    }
}

fn make_pool(k: usize, d: usize, seed: u64) -> CutPool {
    let mut pool = CutPool::new(k, d, 1, 0);
    let mut state = seed;
    for slot in 0..k {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1u64 << 52) - 1);
        let intercept = f64::from_bits((1023u64 << 52) | bits) - 1.5;
        let mut coeffs = vec![0.0_f64; d];
        fill_f64(&mut coeffs, state.wrapping_add(slot as u64));
        pool.add_cut(NodeId(0), 0, slot as u32, intercept, &coeffs);
    }
    let metadata: Vec<CutMetadata> = (0..k)
        .map(|slot| {
            let mut meta = pool.metadata(slot).clone();
            meta.iteration_generated = 1;
            meta
        })
        .collect();
    let active: Vec<bool> = (0..k).map(|slot| pool.is_active(slot)).collect();
    pool.replace_selection(&metadata, &active);
    pool
}

fn make_states(m: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut buf = vec![0.0_f64; m * d];
    fill_f64(&mut buf, seed);
    buf
}

fn bench_one(c: &mut Criterion, name: &str, threads: usize, k: usize, d: usize, m: usize) {
    let rp = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let pool = make_pool(k, d, 0xDEAD_BEEF_DEAD_BEEF);
    let states = make_states(m, d, 0xFEED_FACE_FEED_FACE);
    let strategy = CutSelectionStrategy::Lml1 {
        check_frequency: 1,
        tie_tolerance: 1e-10,
    };
    c.bench_function(name, |b| {
        b.iter(|| {
            rp.install(|| {
                black_box(strategy.select(black_box(&pool), black_box(&states), 5));
            });
        });
    });
}

fn select_for_stage_aggregated_1thread(c: &mut Criterion) {
    bench_one(c, "select_for_stage/aggregated/1thread", 1, 945, 155, 384);
}

fn select_for_stage_aggregated_8threads(c: &mut Criterion) {
    bench_one(c, "select_for_stage/aggregated/8threads", 8, 945, 155, 384);
}

fn select_for_stage_aggregated_96threads(c: &mut Criterion) {
    // Host may have fewer than 96 cores; rayon caps internally.
    bench_one(
        c,
        "select_for_stage/aggregated/96threads",
        96,
        945,
        155,
        384,
    );
}

fn select_for_stage_disaggregated_8threads(c: &mut Criterion) {
    // TODO(measured-K-probe): update K=1000 to the measured
    // disaggregated value when the probe report lands.
    bench_one(
        c,
        "select_for_stage/disaggregated/8threads",
        8,
        1000,
        2080,
        384,
    );
}

criterion_group!(
    benches,
    select_for_stage_aggregated_1thread,
    select_for_stage_aggregated_8threads,
    select_for_stage_aggregated_96threads,
    select_for_stage_disaggregated_8threads,
);
criterion_main!(benches);
