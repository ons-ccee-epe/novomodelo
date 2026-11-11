//! Criterion micro-benchmark for the per-solve `PatchBuffer::fill_load_patches`
//! and `fill_inflow_shift_patches` pair, at two shapes (see `SHAPES`). Bus
//! positions are in reverse (non-slot) order, matching the runtime caller's
//! `load_bus_indices`.

#![allow(missing_docs, clippy::unwrap_used, clippy::expect_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use novomodelo_sddp::indexer::{BlockGrid, BlockRowFamily, StateSpace};
use novomodelo_sddp::lead_time::AnticipatedResolution;
use novomodelo_sddp::lp::builder::PatchBuffer;
use novomodelo_sddp::test_support::{equipment_free_geometry, inflow_coupling_with_entries};
use std::hint::black_box;

const SHAPES: &[(&str, usize, usize, usize)] =
    &[("5x3x165", 5, 3, 165), ("64x24x400", 64, 24, 400)];

fn bench_patch_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_fill");
    for &(name, n_buses, n_blocks, n_hydros) in SHAPES {
        let state = StateSpace::new(
            n_hydros,
            0,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &vec![0; n_hydros],
        );
        let bus_positions: Vec<usize> = (0..n_buses).rev().collect();
        let geometry = equipment_free_geometry(&[n_blocks]);
        let first_inflow_row = n_buses * n_blocks;
        let entries: Vec<(usize, usize)> =
            (0..n_hydros).map(|h| (h, first_inflow_row + h)).collect();
        let coupling = inflow_coupling_with_entries(n_hydros, &entries);
        let mut buf = PatchBuffer::new(
            &state,
            &bus_positions,
            &geometry,
            std::slice::from_ref(&coupling),
        );
        let load_rhs = vec![100.0_f64; n_buses * n_blocks];
        let z_hat = vec![0.5_f64; n_hydros];
        let row_scale: [f64; 0] = [];
        let load_rows = BlockRowFamily::per_block(0..n_buses * n_blocks);

        group.bench_function(name, |b| {
            b.iter(|| {
                buf.fill_load_patches(
                    black_box(load_rows),
                    BlockGrid::new(n_blocks, 0),
                    black_box(&load_rhs),
                    black_box(&bus_positions),
                    black_box(&row_scale),
                );
                buf.fill_inflow_shift_patches(
                    black_box(&coupling),
                    black_box(&z_hat),
                    black_box(&row_scale),
                );
                black_box(buf.forward_patch_count());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_patch_fill);
criterion_main!(benches);
