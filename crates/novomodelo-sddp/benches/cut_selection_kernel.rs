//! Microbenchmark for the Level 1 ownership rebuild.
//!
//! Synthetic pools whose every cut holds a stored trial state, at aggregated
//! (K=945, D=155) and disaggregated (K=1000, D=2080) sizes: `rebuild_ownership`
//! scores every cut at every stored state. Pinned at 1, 8, and 96 rayon worker
//! threads.
//!
//! The disaggregated case is unconditional (no `slow-` gate): Criterion runs
//! benches only on demand, so the user controls execution.

#![allow(missing_docs, clippy::expect_used, clippy::cast_possible_truncation)]

use criterion::{Criterion, criterion_group, criterion_main};
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::setup::NodeId;

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
    pool.enable_trial_states();
    let mut state = seed;
    let mut coeffs = vec![0.0_f64; d];
    let mut trial_state = vec![0.0_f64; d];
    for slot in 0..k {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1u64 << 52) - 1);
        let intercept = f64::from_bits((1023u64 << 52) | bits) - 1.5;
        fill_f64(&mut coeffs, state.wrapping_add(slot as u64));
        pool.add_cut(NodeId(0), 0, slot as u32, intercept, &coeffs);
        fill_f64(&mut trial_state, state ^ (slot as u64));
        pool.set_trial_state_for_test(slot, &trial_state);
    }
    pool
}

fn bench_one(c: &mut Criterion, name: &str, threads: usize, k: usize, d: usize) {
    let rp = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let mut pool = make_pool(k, d, 0xDEAD_BEEF_DEAD_BEEF);
    c.bench_function(name, |b| {
        b.iter(|| rp.install(|| pool.rebuild_ownership()));
    });
}

fn rebuild_ownership_aggregated_1thread(c: &mut Criterion) {
    bench_one(c, "rebuild_ownership/aggregated/1thread", 1, 945, 155);
}

fn rebuild_ownership_aggregated_8threads(c: &mut Criterion) {
    bench_one(c, "rebuild_ownership/aggregated/8threads", 8, 945, 155);
}

fn rebuild_ownership_aggregated_96threads(c: &mut Criterion) {
    // Host may have fewer than 96 cores; rayon caps internally.
    bench_one(c, "rebuild_ownership/aggregated/96threads", 96, 945, 155);
}

fn rebuild_ownership_disaggregated_8threads(c: &mut Criterion) {
    // TODO(measured-K-probe): update K=1000 to the measured
    // disaggregated value when the probe report lands.
    bench_one(c, "rebuild_ownership/disaggregated/8threads", 8, 1000, 2080);
}

criterion_group!(
    benches,
    rebuild_ownership_aggregated_1thread,
    rebuild_ownership_aggregated_8threads,
    rebuild_ownership_aggregated_96threads,
    rebuild_ownership_disaggregated_8threads,
);
criterion_main!(benches);
