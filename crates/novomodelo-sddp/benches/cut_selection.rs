//! Microbenchmark for `CutSelectionStrategy::Lml1` selection at aggregated scale
//! on 8 rayon worker threads.
//!
//! The rayon pool is built once outside the timed region so Criterion measures
//! only the `select` call.

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

fn select_for_stage_aggregated_8threads(c: &mut Criterion) {
    const K: usize = 945;
    const D: usize = 155;
    const M: usize = 384;
    const THREADS: usize = 8;

    let rp = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .expect("rayon pool");
    let pool = make_pool(K, D, 0xCAFE_BABE_CAFE_BABE);
    let states = make_states(M, D, 0xBADD_F00D_BADD_F00D);
    let strategy = CutSelectionStrategy::Lml1 {
        check_frequency: 1,
        tie_tolerance: 1e-10,
    };

    c.bench_function("cut_selection/aggregated/8threads", |b| {
        b.iter(|| {
            rp.install(|| {
                black_box(strategy.select(black_box(&pool), black_box(&states), 5));
            });
        });
    });
}

criterion_group!(benches, select_for_stage_aggregated_8threads);
criterion_main!(benches);
